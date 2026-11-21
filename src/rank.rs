//! The rank pass: one whole-vocabulary sweep per query-strand.
//!
//! SINTAX as published, and both reference implementations, score all N
//! references once per bootstrap replicate. That is 100 passes over the
//! database per query-strand, and it is the reason the algorithm is slow --
//! not the bootstrap, which is cheap, but the loop nest around it.
//!
//! This module swaps the nest. Each reference's exact match count against the
//! *whole* query vocabulary is computed once; the replicates then run against a
//! small candidate set drawn from the top of that ranking. The expensive pass
//! happens once instead of a hundred times, and it is the pass that vectorises
//! best: counters are bit-sliced across references, so one 64-bit operation
//! advances 64 counters at a time.
//!
//! Restricting the replicates to candidates is the one approximation, and it
//! carries a quantified bound rather than a hope -- see [`Bound`].

use crate::{
    K, SAMPLE_SIZE, WORDS, bits,
    index::{Index, Layout, Row, RowRef},
    scoring::Top,
    weight,
};
use anyhow::{Result, ensure};

/// Sized so everything the tile loop touches stays L1-resident: the counter
/// planes are 11 KiB at the widest (`PLANE_CAP * TILE / 64` u64s), the
/// carry-save accumulator another 18 KiB, and the tile's slice of `counts` is
/// 8 KiB. It is also the boundary the index's blocked rows are cut on, which is
/// why it lives in the crate root.
pub use crate::TILE;
const LANES: usize = TILE / 64;
/// A match count cannot exceed the query's vocabulary, which is capped just
/// below the whole word universe, so seventeen planes always suffice.
const PLANE_CAP: usize = 17;

/// Vocabularies must stay below this so every position, and every count, fits
/// a `u16`.
const VOCABULARY_CAP: usize = u16::MAX as usize;

/// Add `2^shift` into bit-sliced counters wherever `input` has a bit set.
///
/// Written for `depth` planes but it exits as soon as every lane's carry has
/// died, which for real data is after one or two -- incrementing a counter
/// ripples only as far as its first zero bit.
///
/// `shift` is how an integer weight is applied. Because the weight tiers are
/// powers of two, adding a weight of four is this same single pass started two
/// planes up, not four passes: weighting costs nothing per sampled word.
///
/// There is deliberately no per-lane `if carry == 0 { continue }` guard. It
/// only pays on sparse bitmaps, and nothing sparse reaches this function any
/// more: the rank sweep sends sparse rows down their own kernels, and the
/// candidate rows the replicates add are dense. The guard was a branch that
/// was rarely taken and often mispredicted, and LLVM will not speculate the
/// store behind it, so the guarded loop did not vectorise at any
/// instruction-set level. Dropping it from the rank kernel was worth -47% of
/// rank time (docs/RANK_results.md §4.3); dropping it from the replicates too
/// was -4.8% of all cycles (docs/RANK_round3.md §D).
///
/// The tile-wide early exit stays. It is not a per-lane branch and it is worth
/// keeping: the upper planes stop receiving carries long before `depth`.
fn ripple_add_dense(
    planes: &mut [u64],
    input: &[u64],
    depth: usize,
    carry: &mut Vec<u64>,
    shift: usize,
) {
    debug_assert!(shift < depth);
    let lanes = input.len();
    carry.clear();
    carry.extend_from_slice(input);
    for plane in shift..depth {
        let mut any = 0;
        for (slot, c) in planes[plane * lanes..][..lanes]
            .iter_mut()
            .zip(&mut carry[..lanes])
        {
            let next = *slot & *c;
            *slot ^= *c;
            *c = next;
            any |= next;
        }
        if any == 0 {
            return;
        }
    }
}

/// Number of weight levels the carry-save accumulator can hold.
///
/// A level `L` is only ever occupied once `2^L` rows have been folded into it,
/// so `PLANE_CAP` levels cover every vocabulary the planes themselves can.
const LEVELS: usize = PLANE_CAP;

/// Fold one dense row's tile bitmap into the carry-save accumulator.
///
/// The accumulator is a binary counter whose digits are whole bitmaps: level
/// `L` holds up to two bitmaps of weight `2^L`. Inserting a third at a level
/// compresses all three with a 3:2 carry-save adder,
///
/// ```text
/// sum   = a ^ b ^ c
/// carry = (a & b) | ((a ^ b) & c)
/// ```
///
/// leaving `sum` as the level's only occupant and promoting `carry`, which has
/// twice the weight, to the level above. That is five boolean operations per
/// lane and, amortised over a run of rows, about one compression per row --
/// against a multi-plane carry propagation per row for the direct approach.
/// Nothing here reads or writes the counter planes, so there is no loop-carried
/// dependency through them either.
///
/// This is *positional* population counting: each reference keeps its own sum.
/// A textbook Harley-Seal would finish by horizontally popcounting the
/// compressed words, which would destroy exactly the identity the rank pass
/// exists to produce; [`Workspace::rank`] ripples the survivors into the planes
/// at `shift = level` instead.
///
/// Only ever called with level zero already full: when it has room the caller
/// decodes the row straight into the free slot, which is the other half of the
/// copies this arrangement avoids.
fn csa_fold(
    acc: &mut [u64],
    occupied: &mut [u8],
    carries: &mut [u64],
    lanes: usize,
    input: &[u64],
) {
    debug_assert_eq!(occupied[0], 2);
    promote(acc, occupied, carries, lanes, 0, input);
}

/// The carry-save cascade of [`csa_fold`], from `level` up.
///
/// At level zero the incoming value is `input`; above it, it is the carry the
/// level below left in `carries`, and `input` is not read. `level` must be
/// full on entry.
fn promote(
    acc: &mut [u64],
    occupied: &mut [u8],
    carries: &mut [u64],
    lanes: usize,
    mut level: usize,
    input: &[u64],
) {
    debug_assert_eq!(occupied[level], 2);
    loop {
        debug_assert!(level + 1 < LEVELS);
        let up = level + 1;
        // The incoming value is the row itself at level zero, and above it the
        // carry the level below just promoted.
        let (below, here) = carries.split_at_mut(level * LANES);
        let c: &[u64] = if level == 0 {
            input
        } else {
            &below[(level - 1) * LANES..][..lanes]
        };
        // The carry weighs twice what this level's occupants do, so it belongs
        // one level up. When that level has room, write it *straight into* the
        // slot rather than into a scratch buffer that would then be copied
        // there. Those copies were the largest single source of `memcpy` in the
        // program; see docs/RANK_results.md.
        let (mine, above) = acc.split_at_mut(up * 2 * LANES);
        let free = occupied[up] < 2;
        let out: &mut [u64] = if free {
            &mut above[occupied[up] as usize * LANES..][..lanes]
        } else {
            &mut here[..lanes]
        };
        let (sum, other) = mine[level * 2 * LANES..].split_at_mut(LANES);
        for (((sum, &b), &c), out) in sum[..lanes]
            .iter_mut()
            .zip(&other[..lanes])
            .zip(c)
            .zip(out.iter_mut())
        {
            let ab = *sum ^ b;
            *out = (*sum & b) | (ab & c);
            *sum = ab ^ c;
        }
        occupied[level] = 1;
        if free {
            occupied[up] += 1;
            return;
        }
        level = up;
    }
}

/// Fold two dense rows' tiles into the accumulator straight out of the index:
/// one full adder against the ones digit, and no copy of either row.
///
/// The steady state of [`fold_dense`] used to decode each row before folding
/// it -- every other row into the free level-zero slot, the rest into a
/// scratch tile for [`csa_fold`] -- and those two copies were a sixth of all
/// cycles on a 15k-reference database. Taking the rows two at a time lets the
/// adder read both where they already are. The carry is promoted exactly as
/// `csa_fold` promotes it.
fn fold_pair(
    acc: &mut [u64],
    occupied: &mut [u8],
    carries: &mut [u64],
    lanes: usize,
    x: &[u8],
    y: &[u8],
) {
    if occupied[0] == 2 {
        // Level zero must hold at most one bitmap for the adder below. Two
        // can only arrive from a restored batch-core state; halve them first.
        let (ones, rest) = acc.split_at_mut(LANES);
        let free = occupied[1] < 2;
        let (second, above) = rest.split_at_mut(LANES);
        let out: &mut [u64] = if free {
            &mut above[occupied[1] as usize * LANES..][..lanes]
        } else {
            &mut carries[..lanes]
        };
        for ((a, &b), out) in ones[..lanes]
            .iter_mut()
            .zip(&second[..lanes])
            .zip(out.iter_mut())
        {
            *out = *a & b;
            *a ^= b;
        }
        occupied[0] = 1;
        if free {
            occupied[1] += 1;
        } else {
            promote(acc, occupied, carries, lanes, 1, &[]);
        }
    }
    let free = occupied[1] < 2;
    let (level0, above) = acc.split_at_mut(2 * LANES);
    let out: &mut [u64] = if free {
        &mut above[occupied[1] as usize * LANES..][..lanes]
    } else {
        &mut carries[..lanes]
    };
    let ones = &mut level0[..lanes];
    // Lanes the rows hold in whole: all of them, except in the database's
    // final tile, where a row ends inside its last lane. The lanes after
    // those are read zero-padded: the missing bytes stand for references
    // that do not exist, whose bits every row holds as zero.
    let whole = lanes.min(x.len() / 8).min(y.len() / 8);
    let padded = |row: &[u8], lane: usize| {
        let mut bytes = [0u8; 8];
        let tail = row.get(lane * 8..).unwrap_or_default();
        let n = tail.len().min(8);
        bytes[..n].copy_from_slice(&tail[..n]);
        u64::from_le_bytes(bytes)
    };
    let (xs, ys) = (&x[..whole * 8], &y[..whole * 8]);
    if occupied[0] == 0 {
        for ((o, (xw, yw)), out) in ones
            .iter_mut()
            .zip(xs.chunks_exact(8).zip(ys.chunks_exact(8)))
            .zip(out.iter_mut())
        {
            let a = u64::from_le_bytes(xw.try_into().expect("eight bytes"));
            let b = u64::from_le_bytes(yw.try_into().expect("eight bytes"));
            *o = a ^ b;
            *out = a & b;
        }
        for lane in whole..lanes {
            let (a, b) = (padded(x, lane), padded(y, lane));
            ones[lane] = a ^ b;
            out[lane] = a & b;
        }
    } else {
        for ((o, (xw, yw)), out) in ones
            .iter_mut()
            .zip(xs.chunks_exact(8).zip(ys.chunks_exact(8)))
            .zip(out.iter_mut())
        {
            let b = u64::from_le_bytes(xw.try_into().expect("eight bytes"));
            let c = u64::from_le_bytes(yw.try_into().expect("eight bytes"));
            let ab = *o ^ b;
            *out = (*o & b) | (ab & c);
            *o = ab ^ c;
        }
        for lane in whole..lanes {
            let (b, c) = (padded(x, lane), padded(y, lane));
            let o = ones[lane];
            let ab = o ^ b;
            out[lane] = (o & b) | (ab & c);
            ones[lane] = ab ^ c;
        }
    }
    occupied[0] = 1;
    if free {
        occupied[1] += 1;
    } else {
        promote(acc, occupied, carries, lanes, 1, &[]);
    }
}

/// Transpose an 8x8 bit matrix held in a `u64`: byte `p`, bit `j` becomes byte
/// `j`, bit `p`, counting bytes from the least significant.
///
/// Three delta swaps (Hacker's Delight 7-3). Eighteen instructions, no branches
/// and no memory traffic.
fn transpose8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x
}

/// Write one lane's 64 finished counts, reassembled from the counter planes.
///
/// The planes hold each reference's count transposed across bit positions; this
/// puts it back together. Byte `g` of each of the low eight planes forms an 8x8
/// bit matrix whose rows are planes and whose columns are the eight references
/// `8g..8g+8`, so transposing it turns every column into a byte -- that
/// reference's count, finished, in one write.
///
/// The previous implementation iterated the set bits and OR-ed each into
/// `counts`: with a mean dense count in the forties that is roughly 5M scattered
/// `u16` read-modify-writes per query-strand, and it required `counts` to be
/// zeroed first. Writing every count exactly once is both cheaper and what lets
/// the per-tile clear go away.
///
/// `out` must be exactly 64 wide. Planes at or above eight are not represented
/// here -- see the caller.
fn lane_counts(planes: &[u64], lanes: usize, lane: usize, depth: usize, out: &mut [u16]) {
    debug_assert_eq!(out.len(), 64);
    let mut rows = [[0u8; 8]; 8];
    for (plane, row) in rows.iter_mut().enumerate().take(depth) {
        *row = planes[plane * lanes + lane].to_le_bytes();
    }
    for (g, group) in out.chunks_exact_mut(8).enumerate() {
        let matrix = u64::from_le_bytes(std::array::from_fn(|plane| rows[plane][g]));
        for (slot, count) in group.iter_mut().zip(transpose8(matrix).to_le_bytes()) {
            *slot = u16::from(count);
        }
    }
}

/// Per-worker scratch, reused across queries.
///
/// Neither path is strictly allocation-free, and both allocations are recorded
/// here rather than removed because a mimalloc A/B moved nothing (0.01%):
///
/// - [`Workspace::rank`] and [`Workspace::rank_batch`] build each query's rows,
///   dense words and cursors per call, to partition the vocabulary by
///   representation. A handful of allocations per query-strand against tens
///   of milliseconds of sweeping.
/// - [`Workspace::replicate`] returns a fresh tie mask per call, which [`Top`]
///   then owns, and a strand keeps all `BOOTSTRAPS` of them alive at once --
///   roughly 200 allocations of `lanes * 8` bytes per query.
pub struct Workspace {
    scratch: Scratch,
    /// Count window for the final tile when it is short, so the blocked kernel
    /// always sees a whole tile (see [`Row::block_increment`]).
    short: Box<[u16; TILE]>,
    /// Rankings computed ahead by [`Workspace::rank_batch`], waiting to be
    /// [adopted](Workspace::adopt).
    batch: Vec<Ranking>,
    /// The query vocabulary as a bitmap over the whole word universe.
    ///
    /// Projection probes it once per k-mer of every candidate, so it has to
    /// answer out of L1: 8 KiB.
    present: Box<[u64; WORDS / 64]>,
    /// Words of the vocabulary below each 64-word block of `present`.
    ///
    /// Because the vocabulary is ascending, a word's position in it is its rank
    /// among the set bits of `present`: this prefix plus a popcount of the bits
    /// below it within its block. That replaces a 128 KiB word-to-position table
    /// which lived in L2 and had to be reset word by word on every call. 2 KiB,
    /// and nothing to reset; measured about 2% of cycles, within noise
    /// (docs/RANK_round2.md §3.3).
    prefix: Box<[u16; WORDS / 64]>,
    /// Word-major candidate membership: one row per vocabulary position,
    /// [`bits::stride`] lanes wide so the counting network can read whole
    /// chunks, padding lanes zero.
    matrix: Vec<u64>,
    candidate_planes: Vec<u64>,
    valid: Vec<u64>,
    /// Exact match count against the whole query vocabulary, per reference.
    pub counts: Vec<u16>,
    /// Scratch for [`Workspace::select`]'s partial histogram.
    histogram: Vec<u32>,
    /// Largest count in each run of 64 references, recorded while the tile is
    /// still in L1. [`Workspace::select`] uses it twice: to bound the
    /// threshold from below without a histogram of all `N` counts, and to skip
    /// every run that holds no candidate. Both uses need it to be the exact
    /// maximum, so [`Workspace::exclude`] recomputes the runs it lowers.
    lane_max: Vec<u16>,
    /// Size of the vocabulary the counts were taken against: the largest count
    /// possible, and the top of the histogram [`Workspace::select`] walks.
    vocabulary: usize,
    /// The rows of that vocabulary, in its order; see [`Workspace::row_refs`].
    refs: Vec<RowRef>,
    /// References selected for the bootstrap phase, ascending.
    pub candidates: Vec<u32>,
    /// [`Workspace::candidate_mask`]'s bitmap, one bit per reference, and the
    /// candidates below each of its lanes.
    mask: Vec<u64>,
    below: Vec<u32>,
    /// `log2(weight)` per query-vocabulary position, parallel to the words
    /// `project` was given. `replicate` samples positions, so this is indexed
    /// by exactly what that loop already holds: a small contiguous array, no
    /// new cache pressure on the hot path.
    shifts: Vec<u8>,
    /// Counter planes the heaviest possible replicate needs.
    depth: usize,
}

/// What the tile loop reuses for every query-tile it sweeps. Sized for one
/// tile of one query, so it stays L1-resident however many queries share a
/// sweep: queries take turns with it tile by tile.
struct Scratch {
    planes: Vec<u64>,
    tile: Vec<u64>,
    /// Carry-save accumulator: two bitmap slots per weight level.
    acc: Vec<u64>,
    /// How many of each level's two slots are in use.
    occupied: Vec<u8>,
    /// One carry bitmap per level, produced by [`csa_fold`].
    carries: Vec<u64>,
    carry: Vec<u64>,
    /// The accumulator as it stood after folding a batch's common dense rows,
    /// and its occupancy: the state every query of the batch starts its tile
    /// from.
    core_acc: Vec<u64>,
    core_occupied: Vec<u8>,
    /// The tile's blocked postings, gathered into one run so they are counted
    /// by one loop rather than one short loop per row; see [`count_blocks`].
    stream: Box<[u8; STREAM]>,
    /// Which blocks are gathered into `stream`.
    staging: Staging,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            planes: vec![0; PLANE_CAP * LANES],
            tile: vec![0; LANES],
            acc: vec![0; LEVELS * 2 * LANES],
            occupied: vec![0; LEVELS],
            carries: vec![0; LEVELS * LANES],
            carry: Vec::new(),
            core_acc: vec![0; LEVELS * 2 * LANES],
            core_occupied: vec![0; LEVELS],
            stream: Box::new([0; STREAM]),
            staging: Staging::default(),
        }
    }
}

/// Bytes of blocked postings gathered before one counting pass: 4,096
/// postings. Small enough to share L1 with the 8 KiB count window and the
/// accumulator even when two SMT threads share the core.
const STREAM: usize = 8 * 1024;

/// Which of a tile's blocks [`count_blocks`] gathers into one counting pass.
///
/// Both give the same counts; they trade differently. Gathering a block
/// saves its own loop -- a mispredicted exit and some bookkeeping -- at the
/// price of copying its postings once. With one worker per physical core
/// that pays for every block: on EUKARYOME it is 6.8% of all cycles at one
/// thread and 5.5% at four. When two workers share a core, the SMT sibling
/// already hides those exits, and copying the long blocks is then simply
/// more work: gathering everything costs 4.6% at eight threads on 4C/8T,
/// where gathering only the short blocks is neutral and still saves 4.6%
/// at one thread (docs/Optimization_round_6.md).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Staging {
    /// Only blocks of up to 32 postings, one fixed-size move each. Safe
    /// whatever shares the core.
    #[default]
    Short,
    /// Every block but the largest, in 128-byte moves. Best when each worker
    /// has a physical core to itself.
    All,
}

/// Add one to `window[local]` for every posting of this tile's blocked rows.
///
/// Counting each row's block in its own loop costs a mispredicted loop exit
/// and some bookkeeping per row-tile, and most blocks are short: on
/// EUKARYOME a mean of 46 postings, and 39% of a query's blocked rows
/// average fewer than 25 a tile (docs/Profiling_14_260926.md §3). This is the
/// short-row problem of sparse matrix-vector products, and the answer here is
/// theirs: decouple the loop from row boundaries. Blocks of up to `LIMIT`
/// bytes are copied back to back into `stream`, and one loop counts the lot;
/// a longer block amortises its own loop and is counted straight from the
/// row. See [`Staging`] for the two limits used.
///
/// Each block is copied in fixed `COPY`-byte moves, so a block of up to
/// `COPY` bytes costs one constant-size copy and no length-dependent branch.
/// A move may run past its block, into the rest of the same row, and those
/// bytes land where the next block's copy overwrites them, or past the end
/// of what is counted. Only `stream[..pos]`, the blocks' own postings, is
/// ever counted, so the increments are exactly the ones the rows hold, and
/// addition commutes. A move never reads past the row: where fewer than
/// `COPY` bytes of it remain, the rest of the block is copied exactly.
///
/// `spans` holds, per row, the row from its block onwards and the block's
/// length in bytes.
fn count_blocks<const COPY: usize, const LIMIT: usize>(
    spans: &[(&[u8], usize)],
    stream: &mut [u8; STREAM],
    window: &mut [u16; TILE],
) {
    const { assert!(LIMIT + COPY <= STREAM) };
    let mut pos = 0;
    for &(tail, bytes) in spans {
        if bytes > LIMIT {
            Row::block_increment(&tail[..bytes], window);
            continue;
        }
        if pos + bytes + COPY > STREAM {
            Row::block_increment(&stream[..pos], window);
            pos = 0;
        }
        let mut done = 0;
        loop {
            match tail.get(done..done + COPY) {
                Some(chunk) => {
                    let chunk: &[u8; COPY] = chunk.try_into().expect("a COPY-byte chunk");
                    let out: &mut [u8; COPY] = (&mut stream[pos + done..pos + done + COPY])
                        .try_into()
                        .expect("a COPY-byte chunk");
                    *out = *chunk;
                }
                None => {
                    stream[pos + done..pos + bytes].copy_from_slice(&tail[done..bytes]);
                    break;
                }
            }
            done += COPY;
            if done >= bytes {
                break;
            }
        }
        pos += bytes;
    }
    Row::block_increment(&stream[..pos], window);
}

/// Fold dense rows' tiles into the carry-save accumulator, continuing from
/// whatever it already holds.
///
/// Rows are taken two at a time straight from the index through
/// [`fold_pair`], the database's short final tile included: its last lane is
/// read zero-padded. Only an odd row out goes the old way: decoded into the
/// free level-zero slot, or into the scratch tile and then through
/// [`csa_fold`]. On a four-tile database the final tile used to send every
/// row that way: about a quarter of all folding cycles on ncbi_16s.
fn fold_dense(scratch: &mut Scratch, rows: &[Row<'_>], start: usize, lanes: usize) {
    let Scratch {
        tile,
        acc,
        occupied,
        carries,
        ..
    } = scratch;
    let mut single = |row: &Row<'_>, acc: &mut [u64], occupied: &mut [u8], carries: &mut [u64]| {
        if occupied[0] < 2 {
            // Room at the bottom: decode into the accumulator slot
            // itself and skip the scratch tile entirely.
            let slot = occupied[0] as usize * LANES;
            row.dense_tile(start, &mut acc[slot..slot + lanes]);
            occupied[0] += 1;
        } else {
            row.dense_tile(start, &mut tile[..lanes]);
            csa_fold(acc, occupied, carries, lanes, &tile[..lanes]);
        }
    };
    let mut pending: Option<(&Row<'_>, &[u8])> = None;
    for row in rows {
        let bytes = row.dense_bytes_upto(start, lanes);
        match pending.take() {
            None => pending = Some((row, bytes)),
            Some((_, first)) => fold_pair(acc, occupied, carries, lanes, first, bytes),
        }
    }
    if let Some((row, _)) = pending {
        single(row, acc, occupied, carries);
    }
}

/// Copy the occupied slots of one accumulator state into another.
fn copy_acc(
    from: &[u64],
    from_occupied: &[u8],
    to: &mut [u64],
    to_occupied: &mut [u8],
    lanes: usize,
) {
    to_occupied.copy_from_slice(from_occupied);
    for (level, &slots) in from_occupied.iter().enumerate() {
        for slot in 0..slots as usize {
            let at = (level * 2 + slot) * LANES;
            to[at..at + lanes].copy_from_slice(&from[at..at + lanes]);
        }
    }
}

/// One query's whole-vocabulary ranking, as [`Workspace::rank_batch`] leaves
/// it for [`Workspace::adopt`].
#[derive(Default)]
struct Ranking {
    counts: Vec<u16>,
    lane_max: Vec<u16>,
    refs: Vec<RowRef>,
    vocabulary: usize,
    /// Not yet adopted. Adopting swaps the arrays out, so a second adoption
    /// would hand over whatever the previous query left behind.
    fresh: bool,
}

/// One query's rows and cursors, carried from tile to tile of a sweep.
struct Sweep<'a> {
    /// Grouped by storage so each kernel sees one contiguous slice: bitmap
    /// rows, then blocked, then plain postings. Within a group the order is
    /// irrelevant: addition commutes, and every row outside the bitmap group
    /// carries its own cursor.
    rows: Vec<Row<'a>>,
    /// The words of the bitmap rows, parallel to them and ascending.
    dense_words: Vec<u16>,
    dense: usize,
    /// Leading bitmap rows the whole batch shares, folded once per tile for
    /// all of it rather than once per query.
    core: usize,
    blocked: usize,
    /// Counter planes the dense rows need.
    depth: usize,
    /// Byte offsets into the blocked rows, then posting offsets into the plain.
    cursors: Vec<usize>,
    /// Every word's row, in vocabulary order, for the replicates to reuse.
    refs: Vec<RowRef>,
}

impl<'a> Sweep<'a> {
    fn new(index: &'a Index, words: &[u16]) -> Result<Self> {
        ensure!(!words.is_empty(), "empty query vocabulary");
        ensure!(words.len() < VOCABULARY_CAP, "query vocabulary too large");
        let refs: Vec<RowRef> = words
            .iter()
            .map(|&word| index.locate(word))
            .collect::<Result<_>>()?;
        let mut rows: Vec<(u16, Row<'a>)> = words
            .iter()
            .zip(&refs)
            .map(|(&word, &at)| (word, index.row_at(at)))
            .collect();
        // Stable, so each group stays in word order.
        rows.sort_by_key(|(_, row)| match row.layout {
            Layout::Bitmap => 0u8,
            Layout::Blocked => 1,
            Layout::Postings => 2,
        });
        let group = |kind| rows.iter().filter(|(_, row)| row.layout == kind).count();
        let dense = group(Layout::Bitmap);
        let blocked = group(Layout::Blocked);
        let depth = depth_for(dense);
        debug_assert!(depth <= PLANE_CAP);
        let dense_words = rows[..dense].iter().map(|&(word, _)| word).collect();
        let rows: Vec<Row<'a>> = rows.into_iter().map(|(_, row)| row).collect();
        let cursors = vec![0; rows.len() - dense];
        Ok(Self {
            rows,
            dense_words,
            dense,
            core: 0,
            blocked,
            depth,
            cursors,
            refs,
        })
    }
}

/// Sweep every tile of the database for every query, tiles outermost.
///
/// `targets[i]` receives query `i`'s counts and run maxima.
fn sweep<'a>(
    scratch: &mut Scratch,
    short: &mut [u16; TILE],
    n: usize,
    queries: &mut [Sweep<'a>],
    targets: &mut [(&mut [u16], &mut [u16])],
) {
    let core = share_core(queries);
    let widest = queries.iter().map(|q| q.blocked).max().unwrap_or(0);
    let mut spans: Vec<(&'a [u8], usize)> = Vec::with_capacity(widest);
    let mut start = 0;
    while start < n {
        let width = (n - start).min(TILE);
        if core > 0 {
            // The rows every query holds, folded once for all of them.
            let lanes = width.div_ceil(64);
            scratch.occupied.fill(0);
            fold_dense(scratch, &queries[0].rows[..core], start, lanes);
            let Scratch {
                acc,
                occupied,
                core_acc,
                core_occupied,
                ..
            } = scratch;
            copy_acc(acc, occupied, core_acc, core_occupied, lanes);
        }
        for (i, (query, (all, lane_max))) in queries.iter_mut().zip(targets.iter_mut()).enumerate()
        {
            // Only the final tile can be short. It is assembled in a scratch
            // window and copied out at the end; what lies past `width` there
            // is never read.
            let window: &mut [u16; TILE] = if width == TILE {
                (&mut all[start..start + TILE])
                    .try_into()
                    .expect("a whole tile")
            } else {
                short
            };
            // The first query finds the core's state still in the
            // accumulator; the rest start from the copy.
            sweep_tile(scratch, query, &mut spans, start, width, window, i > 0);
            // The tile is complete and still in L1; its run maxima cost one
            // vectorised pass over 8 KiB here instead of a cold pass later.
            for (max, lane) in lane_max[start / 64..]
                .iter_mut()
                .zip(window[..width].chunks(64))
            {
                *max = lane.iter().copied().max().unwrap_or(0);
            }
            if width < TILE {
                all[start..start + width].copy_from_slice(&short[..width]);
            }
        }
        start += TILE;
    }
}

/// Queries of a batch with fewer common dense rows than this are swept
/// independently: a snapshot and a restore per query-tile cost more than a
/// handful of shared folds save.
const MIN_CORE: usize = 8;

/// Find the dense rows every query of a batch holds, and move them to the
/// front of each query's dense rows in the same order. Returns how many.
///
/// Counts are unaffected: the fold order of a sum does not matter, and each
/// query still folds every one of its own rows.
fn share_core(queries: &mut [Sweep<'_>]) -> usize {
    let [first, rest @ ..] = queries else {
        return 0;
    };
    if rest.is_empty() {
        return 0;
    }
    let mut core = first.dense_words.clone();
    for query in rest.iter() {
        core.retain(|word| query.dense_words.binary_search(word).is_ok());
    }
    if core.len() < MIN_CORE {
        return 0;
    }
    for query in queries.iter_mut() {
        let (mut shared, mut own) = (Vec::new(), Vec::new());
        for (&word, &row) in query.dense_words.iter().zip(&query.rows[..query.dense]) {
            if core.binary_search(&word).is_ok() {
                shared.push((word, row));
            } else {
                own.push((word, row));
            }
        }
        for (i, (word, row)) in shared.into_iter().chain(own).enumerate() {
            query.dense_words[i] = word;
            query.rows[i] = row;
        }
        query.core = core.len();
    }
    core.len()
}

/// Counts of one query for the references `start .. start + width`, into
/// `window[..width]`.
///
/// A query with a batch core (`query.core > 0`) starts from the core's
/// accumulator state instead of an empty one: from the copy in
/// `scratch.core_acc` when `restore` is set, and otherwise from `scratch.acc`
/// as [`sweep`] left it.
fn sweep_tile<'a>(
    scratch: &mut Scratch,
    query: &mut Sweep<'a>,
    spans: &mut Vec<(&'a [u8], usize)>,
    start: usize,
    width: usize,
    window: &mut [u16; TILE],
    restore: bool,
) {
    let (dense, rest) = query.rows.split_at(query.dense);
    let (blocked, plain) = rest.split_at(query.blocked);
    let depth = query.depth;
    let lanes = width.div_ceil(64);

    if depth > 0 {
        if query.core == 0 {
            scratch.occupied.fill(0);
        } else if restore {
            let Scratch {
                acc,
                occupied,
                core_acc,
                core_occupied,
                ..
            } = &mut *scratch;
            copy_acc(core_acc, core_occupied, acc, occupied, lanes);
        }
        fold_dense(scratch, &dense[query.core..], start, lanes);
    }
    let Scratch {
        planes,
        acc,
        occupied,
        carry,
        stream,
        staging,
        ..
    } = scratch;
    let counts = &mut window[..width];
    if depth > 0 {
        let planes = &mut planes[..depth * lanes];
        planes.fill(0);
        // Only the survivors reach the planes: at most two bitmaps per
        // weight level, so a couple of dozen ripples for a tile instead
        // of one per dense row.
        for (level, &slots) in occupied.iter().enumerate() {
            debug_assert!(slots == 0 || level < depth);
            for slot in 0..slots as usize {
                let at = (level * 2 + slot) * LANES;
                ripple_add_dense(planes, &acc[at..at + lanes], depth, carry, level);
            }
        }
        // Every count in the window is written here, which is why the
        // window is not cleared first.
        for lane in 0..lanes {
            let base = lane * 64;
            if base + 64 <= width {
                lane_counts(planes, lanes, lane, depth, &mut counts[base..base + 64]);
            } else {
                let mut tail = [0u16; 64];
                lane_counts(planes, lanes, lane, depth, &mut tail);
                counts[base..width].copy_from_slice(&tail[..width - base]);
            }
        }
        // A byte-wide transpose carries eight planes, so a count above
        // 255 has bits it cannot reach. That needs more than 255 dense
        // rows in one query's vocabulary -- around 190 is typical -- so
        // the planes above eight keep the old set-bit scatter. It stays
        // exact: the transpose has already written those bits as zero.
        if depth > 8 {
            for lane in 0..lanes {
                let base = lane * 64;
                let limit = width - base;
                for plane in 8..depth {
                    let mut bits = planes[plane * lanes + lane];
                    if limit < 64 {
                        bits &= (1u64 << limit) - 1;
                    }
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        counts[base + bit] |= 1u16 << plane;
                        bits &= bits - 1;
                    }
                }
            }
        }
    } else {
        // An all-sparse vocabulary assigns nothing, so it still needs a
        // zeroed window to increment into.
        counts.fill(0);
    }

    // Blocked rows are read strictly in tile order -- that is the
    // whole point of the layout -- so their cursors are byte offsets
    // that only ever move forward, and nothing is compared per posting.
    //
    // Every row's block length is read first, in a pass of its own.
    // Those loads are independent of each other, so they overlap
    // instead of each one waiting behind the previous row's postings;
    // they also pull each block's first line into L1 before it is
    // copied. The blocks are then counted together, see [`count_blocks`].
    // The pass used to touch each block's last byte as well, to put a
    // block's second line in flight early; the copies now read every
    // line straight after, and the touch measured 3.5% slower.
    let (blocked_cursors, plain_cursors) = query.cursors.split_at_mut(blocked.len());
    spans.clear();
    spans.extend(
        blocked
            .iter()
            .zip(blocked_cursors.iter_mut())
            .map(|(row, cursor)| {
                let data = row.bytes();
                let block = u16::from_le_bytes([data[*cursor], data[*cursor + 1]]) as usize;
                let at = *cursor + 2;
                *cursor = at + block * 2;
                (&data[at..], block * 2)
            }),
    );
    match staging {
        Staging::Short => count_blocks::<64, 64>(spans, stream, window),
        Staging::All => count_blocks::<128, { STREAM / 2 }>(spans, stream, window),
    }
    let counts = &mut window[..width];
    for (row, cursor) in plain.iter().zip(plain_cursors.iter_mut()) {
        row.tile_increment(start, counts, cursor);
    }
}

/// Size a count array and its run maxima for `n` references. Every tile
/// assigns its whole window, so zeroing is only needed on a resize.
fn size_for(counts: &mut Vec<u16>, lane_max: &mut Vec<u16>, n: usize) {
    if counts.len() != n {
        counts.clear();
        counts.resize(n, 0);
    }
    lane_max.resize(n.div_ceil(64), 0);
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            scratch: Scratch::default(),
            short: Box::new([0; TILE]),
            batch: Vec::new(),
            present: Box::new([0; WORDS / 64]),
            prefix: Box::new([0; WORDS / 64]),
            matrix: Vec::new(),
            candidate_planes: Vec::new(),
            valid: Vec::new(),
            counts: Vec::new(),
            histogram: Vec::new(),
            lane_max: Vec::new(),
            vocabulary: 0,
            refs: Vec::new(),
            candidates: Vec::new(),
            mask: Vec::new(),
            below: Vec::new(),
            shifts: Vec::new(),
            depth: depth_for(SAMPLE_SIZE),
        }
    }
}

fn depth_for(vocabulary: usize) -> usize {
    (usize::BITS - vocabulary.leading_zeros()) as usize
}

impl Workspace {
    /// Choose which blocked postings the sweep gathers before counting; see
    /// [`Staging`]. Counts are identical either way.
    pub fn set_staging(&mut self, staging: Staging) {
        self.scratch.staging = staging;
    }

    /// Exact `|W(query) intersect W(r)|` for every reference `r`.
    ///
    /// The sweep is tiled so the counter planes stay L1-resident, and each row
    /// representation gets the kernel it is actually suited to.
    ///
    /// **Dense rows** -- bitmaps, one bit per reference -- go through the
    /// bit-sliced adder, where one 64-bit AND/XOR advances 64 counters at once.
    /// They are only 35% of a typical query's vocabulary but 61% of its bytes.
    ///
    /// **Sparse rows** -- sorted posting lists -- increment `counts` directly.
    /// Building a 64-lane bitmap out of forty-odd postings and then rippling it
    /// through ten planes costs far more than forty-odd `+= 1`s into a window
    /// that is already in L1. This is the single largest structural change to
    /// the sweep: it removes roughly two thirds of all `ripple_add` calls.
    ///
    /// Because the planes now carry only the dense contribution, `depth` is set
    /// by the number of dense rows rather than by the whole vocabulary -- about
    /// eight planes instead of ten on a production query -- and the final `u16`
    /// count holds the combined total.
    ///
    /// The tile's count window is filled and its per-run maxima recorded while
    /// it is still hot, which is what lets the whole-array memset go away.
    /// There is deliberately no histogram of all `N` counts: [`Workspace::select`]
    /// needs only its upper tail, and builds that from the runs `lane_max`
    /// admits (docs/RANK_round3.md §A).
    pub fn rank(&mut self, index: &Index, words: &[u16]) -> Result<()> {
        let mut query = Sweep::new(index, words)?;
        let n = index.references;
        size_for(&mut self.counts, &mut self.lane_max, n);
        self.vocabulary = words.len();
        sweep(
            &mut self.scratch,
            &mut self.short,
            n,
            std::slice::from_mut(&mut query),
            &mut [(&mut self.counts[..], &mut self.lane_max[..])],
        );
        self.refs = query.refs;
        Ok(())
    }

    /// [`Workspace::rank`] for several queries in one sweep, each left waiting
    /// for [`Workspace::adopt`].
    ///
    /// The tile loop is turned inside out: tiles outside, queries inside, each
    /// query sweeping a tile with exactly the kernels a lone query would use.
    /// Every count is therefore the same. What changes is where the bytes come
    /// from. The dense rows are the most common words, so a batch's queries
    /// share most of them, and a tile of a shared row is fetched from DRAM by
    /// the first query that needs it and found in L2 by the rest
    /// (docs/RANK_round3.md §E).
    pub fn rank_batch(&mut self, index: &Index, queries: &[&[u16]]) -> Result<()> {
        let mut sweeps = queries
            .iter()
            .map(|words| Sweep::new(index, words))
            .collect::<Result<Vec<_>>>()?;
        let n = index.references;
        if self.batch.len() < queries.len() {
            self.batch.resize_with(queries.len(), Ranking::default);
        }
        let mut targets = Vec::with_capacity(queries.len());
        for (ranking, words) in self.batch.iter_mut().zip(queries) {
            size_for(&mut ranking.counts, &mut ranking.lane_max, n);
            ranking.vocabulary = words.len();
            ranking.fresh = true;
            targets.push((&mut ranking.counts[..], &mut ranking.lane_max[..]));
        }
        sweep(
            &mut self.scratch,
            &mut self.short,
            n,
            &mut sweeps,
            &mut targets,
        );
        for (ranking, sweep) in self.batch.iter_mut().zip(sweeps) {
            ranking.refs = sweep.refs;
        }
        Ok(())
    }

    /// Make the `slot`-th query of the last [`Workspace::rank_batch`] the one
    /// `select`, `exclude`, `project` and `counts` refer to, exactly as if
    /// [`Workspace::rank`] had just swept it. Constant time: the arrays are
    /// swapped, not copied, so each slot can be adopted once.
    pub fn adopt(&mut self, slot: usize) {
        let ranking = &mut self.batch[slot];
        assert!(ranking.fresh, "ranking {slot} was already adopted");
        ranking.fresh = false;
        std::mem::swap(&mut self.counts, &mut ranking.counts);
        std::mem::swap(&mut self.lane_max, &mut ranking.lane_max);
        std::mem::swap(&mut self.refs, &mut ranking.refs);
        self.vocabulary = ranking.vocabulary;
    }

    /// Where each word of the vocabulary last ranked has its row, in that
    /// vocabulary's order: the sweep's own directory reads, kept so the
    /// replicates that follow it need none (see [`crate::index::RowRef`]).
    pub fn row_refs(&self) -> &[RowRef] {
        &self.refs
    }

    /// Erase references the caller rejects, as if they were never indexed.
    ///
    /// Training needs the database to *not* contain the query's own sequence,
    /// or its whole genus, or its whole family, so that a fitted coefficient
    /// learns what a query with no close relative looks like. PROTAX does this
    /// with a per-query `seq_dont_use` list rather than by rebuilding the
    /// database, which is what lets one pass over the production index stand
    /// in for every state of database completeness.
    ///
    /// Zeroing the count is enough to erase a reference from everything
    /// downstream: `select` admits nothing below a threshold of one, so a
    /// zeroed reference cannot become a candidate, cannot be projected, and
    /// cannot win a replicate. `lane_max` is recomputed for every run that lost
    /// a count, because `select` reads it as an exact maximum.
    /// The predicate is handed the count as well as the id so it can be
    /// written without borrowing the workspace it is erasing from.
    pub fn exclude(&mut self, mut rejected: impl FnMut(usize, u16) -> bool) -> usize {
        let mut erased = 0;
        for (lane, (counts, max)) in self
            .counts
            .chunks_mut(64)
            .zip(self.lane_max.iter_mut())
            .enumerate()
        {
            let before = erased;
            for (offset, count) in counts.iter_mut().enumerate() {
                if *count > 0 && rejected(lane * 64 + offset, *count) {
                    *count = 0;
                    erased += 1;
                }
            }
            if erased != before {
                *max = counts.iter().copied().max().unwrap_or(0);
            }
        }
        erased
    }

    /// `histogram[v]` is the number of references whose count is `v`, over
    /// the whole vocabulary range.
    ///
    /// A full cold pass over all `N` counts. Only the opt-in `--risk` bound
    /// needs the lower part of the distribution; [`Workspace::select`] builds
    /// just the upper tail it needs.
    pub fn histogram(&self) -> Vec<u32> {
        let mut histogram = vec![0u32; self.vocabulary + 1];
        for &count in &self.counts {
            histogram[count as usize] += 1;
        }
        histogram
    }

    /// Smallest threshold whose `{r : count(r) >= threshold}` holds at least
    /// `target` references, and that set in ascending reference order.
    ///
    /// Ties at the threshold are never truncated: a candidate set that cuts
    /// through a group of equally-ranked references would bias which of them
    /// can win a replicate.
    ///
    /// The threshold is found without histogramming all `N` counts. A run of
    /// 64 references whose maximum is at least `t` holds at least one
    /// reference counting at least `t`, so the largest `t_low` at which
    /// `target` runs qualify is a lower bound on the threshold. Only the runs
    /// at or above `t_low` -- about `target` of them -- can hold a count that
    /// matters, and the exact histogram of those decides the threshold.
    pub fn select(&mut self, target: usize) -> u16 {
        let top = self.vocabulary;
        // Runs by their maximum. `lane_max` never exceeds the vocabulary.
        self.histogram.clear();
        self.histogram.resize(top + 1, 0);
        for &max in &self.lane_max {
            self.histogram[max as usize] += 1;
        }
        let mut low = top;
        let mut runs = 0usize;
        while low > 0 {
            runs += self.histogram[low] as usize;
            if runs >= target {
                break;
            }
            low -= 1;
        }
        // Count 0 means no shared words at all; such a reference can never win.
        let low = low.max(1);

        // Exact histogram of the counts at or above the bound, from the runs
        // that can hold one.
        self.histogram.clear();
        self.histogram.resize(top + 1 - low, 0);
        for (&max, counts) in self.lane_max.iter().zip(self.counts.chunks(64)) {
            if (max as usize) < low {
                continue;
            }
            for &count in counts {
                if count as usize >= low {
                    self.histogram[count as usize - low] += 1;
                }
            }
        }
        // The same descent a full histogram would make. If `target` runs
        // qualified at `low` it stops at or above it; if not, it would have
        // run down to zero and been lifted back to one, which is `low`.
        let mut threshold = top.max(low);
        let mut total = 0usize;
        while threshold > low {
            total += self.histogram[threshold - low] as usize;
            if total >= target {
                break;
            }
            threshold -= 1;
        }
        let threshold = threshold as u16;
        self.candidates.clear();
        // A few thousand candidates among 1.5M references occupy a small share
        // of the 64-reference runs, so most of the array is never read.
        for ((lane, &max), counts) in self.lane_max.iter().enumerate().zip(self.counts.chunks(64)) {
            if max < threshold {
                continue;
            }
            for (offset, &count) in counts.iter().enumerate() {
                if count >= threshold {
                    self.candidates.push((lane * 64 + offset) as u32);
                }
            }
        }
        threshold
    }

    /// Build the word-major membership matrix for the selected candidates.
    ///
    /// Word-major because the bootstrap phase consumes it one sampled word at a
    /// time: each row is a bitmap over candidates that can be added straight
    /// into the counter planes.
    ///
    /// `words` must be ascending and distinct, as [`vocabulary`] and
    /// [`revcomp_vocabulary`] produce it: a word's row is located by its rank in
    /// the vocabulary rather than through a lookup table.
    pub fn project(&mut self, index: &Index, words: &[u16], weights: &weight::Table) -> Result<()> {
        ensure!(words.len() < VOCABULARY_CAP, "query vocabulary too large");
        ensure!(
            words.windows(2).all(|pair| pair[0] < pair[1]),
            "query vocabulary must be ascending and distinct"
        );
        self.present.fill(0);
        self.shifts.clear();
        self.shifts.reserve(words.len());
        for &word in words {
            self.present[word as usize / 64] |= 1 << (word % 64);
            self.shifts.push(weights.get(word).trailing_zeros() as u8);
        }
        // Below the cap, so the running total cannot overflow.
        let mut below = 0u16;
        for (prefix, bits) in self.prefix.iter_mut().zip(self.present.iter()) {
            *prefix = below;
            below += bits.count_ones() as u16;
        }
        self.depth = depth_for(SAMPLE_SIZE * weights.max() as usize);
        debug_assert!(self.depth <= PLANE_CAP);

        let stride = bits::stride(self.candidates.len().div_ceil(64).max(1));
        self.matrix.clear();
        self.matrix.resize(words.len() * stride, 0);
        for (position, &reference) in self.candidates.iter().enumerate() {
            let (lane, bit) = (position / 64, position % 64);
            let (present, prefix, matrix) = (&self.present, &self.prefix, &mut self.matrix);
            index.visit_kmers(reference as usize, |word| {
                let (block, offset) = (word as usize / 64, word % 64);
                let bits = present[block];
                if bits >> offset & 1 == 0 {
                    return;
                }
                let position =
                    prefix[block] as usize + (bits & ((1u64 << offset) - 1)).count_ones() as usize;
                matrix[position * stride + lane] |= 1 << bit;
            })?;
        }
        Ok(())
    }

    /// Exact maximum and complete tie set for one replicate, over candidates.
    ///
    /// Scores cannot exceed `SAMPLE_SIZE * max weight`: six planes unweighted,
    /// eight at the widest tier. The 32 drawn rows go through the carry-save
    /// network in [`crate::bits`], which produces those planes directly rather
    /// than rippling each row through them in turn. The tie set is expressed
    /// over candidate positions, which are ascending in reference order, so
    /// the caller's uniform draw sees the same ordering the exact path would.
    pub fn replicate(&mut self, sample: &[u16; SAMPLE_SIZE]) -> Top {
        let lanes = self.candidates.len().div_ceil(64).max(1);
        let stride = bits::stride(lanes);
        let depth = self.depth;
        let offsets: [usize; SAMPLE_SIZE] = std::array::from_fn(|i| sample[i] as usize * stride);
        self.candidate_planes
            .resize(bits::WEIGHTED_PLANES * stride, 0);
        if depth > bits::PLANES {
            let shifts: [u8; SAMPLE_SIZE] =
                std::array::from_fn(|i| self.shifts[sample[i] as usize]);
            bits::count_weighted(
                &self.matrix,
                &offsets,
                &shifts,
                lanes,
                stride,
                &mut self.candidate_planes,
            );
        } else {
            bits::count32(
                &self.matrix,
                &offsets,
                lanes,
                stride,
                &mut self.candidate_planes,
            );
        }
        self.valid.clear();
        self.valid.resize(lanes, u64::MAX);
        let used = self.candidates.len();
        if used == 0 {
            self.valid[lanes - 1] = 0;
        } else if !used.is_multiple_of(64) {
            self.valid[lanes - 1] = (1u64 << (used % 64)) - 1;
        }
        let mut ties = vec![0u64; lanes];
        let max = bits::masked_max(
            &self.candidate_planes,
            stride,
            depth,
            lanes,
            &self.valid,
            &mut ties,
        );
        Top::from_bits(max as u8, ties)
    }

    /// The selected candidates as a bitmap over all `n` references, and how
    /// many candidates precede each 64-reference lane, for scoring them in
    /// place with [`crate::scoring::bitsliced_tops`] instead of through
    /// [`Workspace::project`].
    pub fn candidate_mask(&mut self, n: usize) -> (&[u64], &[u32]) {
        let lanes = n.div_ceil(64);
        self.mask.clear();
        self.mask.resize(lanes, 0);
        for &reference in &self.candidates {
            self.mask[reference as usize / 64] |= 1 << (reference % 64);
        }
        self.below.clear();
        self.below.reserve(lanes + 1);
        let mut total = 0u32;
        for &word in &self.mask {
            self.below.push(total);
            total += word.count_ones();
        }
        self.below.push(total);
        (&self.mask, &self.below)
    }

    /// The bitmap and per-lane prefix [`Workspace::candidate_mask`] last built,
    /// read back without borrowing the workspace mutably.
    pub fn candidate_view(&self) -> (&[u64], &[u32]) {
        (&self.mask, &self.below)
    }

    /// How many distinct [`TILE`]s of the database the candidates fall in.
    /// Candidates are ascending, so this is a count of tile changes.
    pub fn candidate_tiles(&self) -> usize {
        let mut tiles = 0;
        let mut last = usize::MAX;
        for &reference in &self.candidates {
            let tile = reference as usize / TILE;
            if tile != last {
                tiles += 1;
                last = tile;
            }
        }
        tiles
    }

    /// Reference id behind the `offset`-th member of a replicate's tie set.
    pub fn reference_at(&self, top: &Top, offset: usize) -> u32 {
        self.candidates[top.nth(offset) as usize]
    }
}

/// How safe it was to restrict a replicate to the candidate set.
///
/// The obvious deterministic bound, `score(b, r) <= min(SAMPLE_SIZE, count(r))`,
/// is useless: counts run into the hundreds while scores cap at 32, so it
/// certifies nothing. Measured on 90%-identity cross-validation pairs it held
/// for 0.0% of replicates at every candidate size tried.
///
/// What does work is the tail bound. Under the bootstrap's own randomness a
/// pruned reference's score is `Binomial(SAMPLE_SIZE, count(r) / vocabulary)`,
/// so the chance that any of them reaches the incumbent maximum is at most the
/// sum of their tail probabilities -- and the rank pass has already produced
/// the histogram of counts that sum needs.
pub struct Bound {
    tail: Vec<f64>,
}

impl Bound {
    /// Cache tail probabilities for one query-strand, indexed by count.
    pub fn new(histogram: &[u32], threshold: u16, vocabulary: usize, score: u8) -> Self {
        let mut tail = vec![0.0; histogram.len()];
        for (count, &references) in histogram.iter().enumerate().take(threshold as usize) {
            if references == 0 {
                continue;
            }
            tail[count] =
                references as f64 * binomial_tail(count as f64 / vocabulary as f64, score);
        }
        Self { tail }
    }

    /// Expected number of pruned references that would have reached `score`.
    pub fn risk(&self) -> f64 {
        self.tail.iter().sum()
    }
}

/// `P(Binomial(SAMPLE_SIZE, p) >= score)`.
fn binomial_tail(p: f64, score: u8) -> f64 {
    if score == 0 {
        return 1.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let n = SAMPLE_SIZE;
    let mut total = 0.0;
    // Terms fall off steeply above the mean, so accumulate from the top and
    // stop once further terms cannot matter.
    let mut term = p.powi(n as i32);
    for k in (score as usize..=n).rev() {
        if k < n {
            // term(k) from term(k+1): multiply by (k+1)/(n-k) * (1-p)/p
            term *= (k + 1) as f64 / (n - k) as f64 * (1.0 - p) / p;
        }
        total += term;
        if term == 0.0 {
            break;
        }
    }
    total.min(1.0)
}

/// Distinct k-mers of a normalized sequence, ascending, reusing `seen`.
pub fn vocabulary(sequence: &[u8], seen: &mut [u64; WORDS / 64], out: &mut Vec<u16>) {
    seen.fill(0);
    out.clear();
    let (mut word, mut run) = (0u16, 0usize);
    for &base in sequence {
        let code = match base {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => {
                run = 0;
                word = 0;
                continue;
            }
        };
        word = (word << 2) | code;
        run = (run + 1).min(K);
        if run == K {
            seen[word as usize / 64] |= 1 << (word % 64);
        }
    }
    for (i, bits) in seen.iter().enumerate() {
        let mut bits = *bits;
        while bits != 0 {
            out.push((i * 64 + bits.trailing_zeros() as usize) as u16);
            bits &= bits - 1;
        }
    }
}

/// Reverse-complement a word code in place of re-extracting the sequence.
///
/// The k-mer set of `revcomp(S)` is exactly the image of `S`'s under this map,
/// so the reverse strand costs a table lookup per word instead of a second
/// pass over the sequence.
pub fn revcomp_word(word: u16) -> u16 {
    let mut out = 0u16;
    for i in 0..K {
        let base = (word >> (2 * i)) & 3;
        out = (out << 2) | (3 - base);
    }
    out
}

pub fn revcomp_vocabulary(words: &[u16], seen: &mut [u64; WORDS / 64], out: &mut Vec<u16>) {
    seen.fill(0);
    out.clear();
    for &word in words {
        let flipped = revcomp_word(word);
        seen[flipped as usize / 64] |= 1 << (flipped % 64);
    }
    for (i, bits) in seen.iter().enumerate() {
        let mut bits = *bits;
        while bits != 0 {
            out.push((i * 64 + bits.trailing_zeros() as usize) as u16);
            bits &= bits - 1;
        }
    }
}
