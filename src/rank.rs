//! Rank pass: one sweep over the whole query vocabulary per query strand.
//!
//! Each reference's match count against the full query vocabulary is computed
//! once, and bootstrap replicates then run only against a small candidate set
//! from the top of that ranking. The pruning risk is bounded; see [`Bound`].

use crate::{
    K, SAMPLE_SIZE, WORDS,
    index::{Index, Layout, Row},
    scoring::Top,
    weight,
};
use anyhow::{Result, ensure};

/// Sized so the tile loop's working set (counter planes, carry-save accumulator,
/// and the tile's slice of `counts`) stays in L1. The index's blocked rows are
/// cut on the same boundary.
pub use crate::TILE;
const LANES: usize = TILE / 64;
/// A match count can't exceed the query's vocabulary size, which is capped
/// just below the word universe, so 17 planes is enough.
const PLANE_CAP: usize = 17;

/// Vocabularies must stay below this so every position and count fits a `u16`.
const VOCABULARY_CAP: usize = u16::MAX as usize;

/// Add `2^shift` into bit-sliced counters wherever `input` has a bit set.
///
/// Weights are powers of two, so a weight is applied by starting `shift` planes up.
/// There is no per-lane empty check (inputs are dense), which lets the loop
/// vectorise; the tile-wide early exit stays.
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

/// Number of weight levels in the carry-save accumulator.
///
/// Level `L` fills only after `2^L` rows, so `PLANE_CAP` levels are enough.
const LEVELS: usize = PLANE_CAP;

/// Fold one dense row's tile bitmap into the carry-save accumulator.
///
/// The accumulator is a binary counter whose digits are bitmaps: level `L` holds
/// up to two bitmaps of weight `2^L`. A third one triggers a 3:2 compression,
///
/// ```text
/// sum   = a ^ b ^ c
/// carry = (a & b) | ((a ^ b) & c)
/// ```
///
/// keeping `sum` at this level and promoting `carry` one level up. Counts stay
/// per reference (positional popcount); [`Workspace::rank`] ripples the
/// survivors into the planes at `shift = level`.
///
/// Only called when level zero is full; otherwise the caller decodes the row
/// straight into the free slot.
fn csa_fold(
    acc: &mut [u64],
    occupied: &mut [u8],
    carries: &mut [u64],
    lanes: usize,
    input: &[u64],
) {
    debug_assert_eq!(occupied[0], 2);
    let mut level = 0;
    loop {
        debug_assert!(level + 1 < LEVELS);
        let up = level + 1;
        // Level zero takes the row itself; higher levels take the carry from below.
        let (below, here) = carries.split_at_mut(level * LANES);
        let c: &[u64] = if level == 0 {
            input
        } else {
            &below[(level - 1) * LANES..][..lanes]
        };
        // The carry has twice this level's weight, so it moves up one level; write it
        // straight into a free slot there to avoid a copy.
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

/// Transpose an 8x8 bit matrix held in a `u64`: byte `p`, bit `j` becomes byte
/// `j`, bit `p`, counting bytes from the least significant.
///
/// Three delta swaps (Hacker's Delight 7-3), branch-free.
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
/// Byte `g` of each of the low eight planes forms an 8x8 bit matrix (rows are
/// planes, columns are references `8g..8g+8`); transposing it yields each
/// reference's count as a byte. `out` must be exactly 64 wide; planes at or
/// above eight are handled by the caller.
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

/// Largest value and the full set of positions holding it.
///
/// Walks planes from the most significant bit down, keeping only positions
/// that have the bit set whenever any do.
fn plane_max(planes: &[u64], lanes: usize, depth: usize, valid: &[u64]) -> (u32, Vec<u64>) {
    let mut candidates = valid.to_vec();
    let mut max = 0u32;
    for plane in (0..depth).rev() {
        let base = plane * lanes;
        let mut any = false;
        for lane in 0..lanes {
            if candidates[lane] & planes[base + lane] != 0 {
                any = true;
                break;
            }
        }
        if any {
            for lane in 0..lanes {
                candidates[lane] &= planes[base + lane];
            }
            max |= 1 << plane;
        }
    }
    (max, candidates)
}

/// Per-worker scratch buffers, reused across queries.
///
/// A few small allocations remain: [`Workspace::rank`] and [`Workspace::rank_batch`]
/// build per-query row lists, and each `plane_max` call returns a `Vec` owned by [`Top`].
pub struct Workspace {
    scratch: Scratch,
    /// Count window for a short final tile, so the blocked kernel always sees a
    /// whole tile (see [`Row::block_increment`]).
    short: Box<[u16; TILE]>,
    /// Rankings computed ahead by [`Workspace::rank_batch`], waiting to be
    /// [adopted](Workspace::adopt).
    batch: Vec<Ranking>,
    /// The query vocabulary as a bitmap over the whole word universe (8 KiB, fits L1).
    present: Box<[u64; WORDS / 64]>,
    /// Number of vocabulary words below each 64-word block of `present`.
    ///
    /// The vocabulary is ascending, so a word's position is this prefix plus the
    /// popcount of lower set bits in its block.
    prefix: Box<[u16; WORDS / 64]>,
    matrix: Vec<u64>,
    candidate_planes: Vec<u64>,
    carry: Vec<u64>,
    valid: Vec<u64>,
    /// Exact match count against the whole query vocabulary, per reference.
    pub counts: Vec<u16>,
    /// Scratch for [`Workspace::select`]'s partial histogram.
    histogram: Vec<u32>,
    /// Largest count in each run of 64 references. [`Workspace::select`] uses it to
    /// bound the threshold and to skip runs with no candidate, so it must be exact;
    /// [`Workspace::exclude`] recomputes any run it lowers.
    lane_max: Vec<u16>,
    /// Vocabulary size the counts were taken against, i.e. the largest possible count.
    vocabulary: usize,
    /// References selected for the bootstrap phase, ascending.
    pub candidates: Vec<u32>,
    /// `log2(weight)` per query-vocabulary position, parallel to the words given to `project`.
    shifts: Vec<u8>,
    /// Counter planes needed by the heaviest possible replicate.
    depth: usize,
}

/// Tile-loop scratch, sized for one tile of one query so it stays in L1;
/// queries in a batch take turns with it tile by tile.
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
    /// Accumulator state (and occupancy) after folding a batch's shared dense rows;
    /// every query in the batch starts its tile from here.
    core_acc: Vec<u64>,
    core_occupied: Vec<u8>,
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
        }
    }
}

/// Fold dense rows' tiles into the carry-save accumulator, continuing from
/// whatever it already holds.
fn fold_dense(scratch: &mut Scratch, rows: &[Row<'_>], start: usize, lanes: usize) {
    let Scratch {
        tile,
        acc,
        occupied,
        carries,
        ..
    } = scratch;
    for row in rows {
        if occupied[0] < 2 {
            // Room at level zero: decode straight into the accumulator slot.
            let slot = occupied[0] as usize * LANES;
            row.dense_tile(start, &mut acc[slot..slot + lanes]);
            occupied[0] += 1;
        } else {
            row.dense_tile(start, &mut tile[..lanes]);
            csa_fold(acc, occupied, carries, lanes, &tile[..lanes]);
        }
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

/// One query's whole-vocabulary ranking, left by [`Workspace::rank_batch`]
/// for [`Workspace::adopt`].
#[derive(Default)]
struct Ranking {
    counts: Vec<u16>,
    lane_max: Vec<u16>,
    vocabulary: usize,
    /// Not yet adopted. Adopting swaps the arrays out, so each slot is adopted once.
    fresh: bool,
}

/// One query's rows and cursors, carried from tile to tile of a sweep.
struct Sweep<'a> {
    /// Grouped by storage (bitmap, blocked, plain postings) so each kernel sees one
    /// contiguous slice; order within a group doesn't matter.
    rows: Vec<Row<'a>>,
    /// The words of the bitmap rows, parallel to them and ascending.
    dense_words: Vec<u16>,
    dense: usize,
    /// Leading bitmap rows shared by the whole batch, folded once per tile for all queries.
    core: usize,
    blocked: usize,
    /// Counter planes the dense rows need.
    depth: usize,
    /// Byte offsets into the blocked rows, then posting offsets into the plain ones.
    cursors: Vec<usize>,
}

impl<'a> Sweep<'a> {
    fn new(index: &'a Index, words: &[u16]) -> Result<Self> {
        ensure!(!words.is_empty(), "empty query vocabulary");
        ensure!(words.len() < VOCABULARY_CAP, "query vocabulary too large");
        let mut rows: Vec<(u16, Row<'a>)> = words
            .iter()
            .map(|&word| Ok((word, index.row(word)?)))
            .collect::<Result<_>>()?;
        // Stable sort, so each group stays in word order.
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
    let mut spans: Vec<&'a [u8]> = Vec::with_capacity(widest);
    let mut start = 0;
    while start < n {
        let width = (n - start).min(TILE);
        if core > 0 {
            // Fold the rows shared by every query once.
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
            // Only the final tile can be short; it is built in a scratch window and
            // copied out. Anything past `width` there is never read.
            let window: &mut [u16; TILE] = if width == TILE {
                (&mut all[start..start + TILE])
                    .try_into()
                    .expect("a whole tile")
            } else {
                short
            };
            // The first query finds the core state still in the accumulator; the rest
            // restore it from the copy.
            sweep_tile(scratch, query, &mut spans, start, width, window, i > 0);
            // Record run maxima while the tile is still in L1.
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

/// Batches sharing fewer dense rows than this are swept per query; the
/// snapshot/restore would cost more than the shared folds save.
const MIN_CORE: usize = 8;

/// Find the dense rows every query in a batch holds and move them to the front
/// of each query's dense rows, in the same order. Returns how many.
///
/// Counts are unaffected, since fold order doesn't change a sum.
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

/// Counts of one query for references `start .. start + width`, into
/// `window[..width]`.
///
/// With a batch core (`query.core > 0`) it starts from the core's accumulator
/// state: from `scratch.core_acc` if `restore` is set, else from `scratch.acc`.
fn sweep_tile<'a>(
    scratch: &mut Scratch,
    query: &mut Sweep<'a>,
    spans: &mut Vec<&'a [u8]>,
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
        ..
    } = scratch;
    let counts = &mut window[..width];
    if depth > 0 {
        let planes = &mut planes[..depth * lanes];
        planes.fill(0);
        // Only the survivors (at most two bitmaps per level) reach the planes.
        for (level, &slots) in occupied.iter().enumerate() {
            debug_assert!(slots == 0 || level < depth);
            for slot in 0..slots as usize {
                let at = (level * 2 + slot) * LANES;
                ripple_add_dense(planes, &acc[at..at + lanes], depth, carry, level);
            }
        }
        // Every count in the window is written here, so it isn't cleared first.
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
        // The byte transpose covers only eight planes; counts above 255 need the
        // higher planes added bit by bit. Still exact, since the transpose wrote
        // those bits as zero.
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
        // An all-sparse vocabulary writes nothing above, so zero the window first.
        counts.fill(0);
    }

    // Blocked rows are read in tile order, so their cursors are byte offsets that
    // only move forward. Block lengths are read first in a separate pass so the
    // loads overlap and prefetch each block's first line.
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
                // Touch the last byte too, so a block spanning two cache lines loads both.
                std::hint::black_box(data[*cursor - 1]);
                &data[at..*cursor]
            }),
    );
    for postings in spans.iter() {
        Row::block_increment(postings, window);
    }
    let counts = &mut window[..width];
    for (row, cursor) in plain.iter().zip(plain_cursors.iter_mut()) {
        row.tile_increment(start, counts, cursor);
    }
}

/// Size a count array and its run maxima for `n` references. Every tile
/// writes its whole window, so zeroing is only needed on resize.
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
            carry: Vec::new(),
            valid: Vec::new(),
            counts: Vec::new(),
            histogram: Vec::new(),
            lane_max: Vec::new(),
            vocabulary: 0,
            candidates: Vec::new(),
            shifts: Vec::new(),
            depth: depth_for(SAMPLE_SIZE),
        }
    }
}

fn depth_for(vocabulary: usize) -> usize {
    (usize::BITS - vocabulary.leading_zeros()) as usize
}

impl Workspace {
    /// Exact `|W(query) intersect W(r)|` for every reference `r`.
    ///
    /// The sweep is tiled so the counter planes stay in L1. Dense (bitmap) rows go
    /// through the bit-sliced adder; sparse (posting list) rows increment `counts`
    /// directly. Each tile's counts and run maxima are written while still hot;
    /// no full histogram is built, since [`Workspace::select`] only needs the upper tail.
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
        Ok(())
    }

    /// [`Workspace::rank`] for several queries in one sweep, each left waiting
    /// for [`Workspace::adopt`].
    ///
    /// Tiles are the outer loop and queries the inner one, using the same kernels,
    /// so counts are identical. Queries share most dense rows, so a shared tile
    /// is fetched from memory once and reused from cache.
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
        Ok(())
    }

    /// Make the `slot`-th query of the last [`Workspace::rank_batch`] current, as if
    /// [`Workspace::rank`] had just swept it. Arrays are swapped, not copied, so
    /// each slot can be adopted once.
    pub fn adopt(&mut self, slot: usize) {
        let ranking = &mut self.batch[slot];
        assert!(ranking.fresh, "ranking {slot} was already adopted");
        ranking.fresh = false;
        std::mem::swap(&mut self.counts, &mut ranking.counts);
        std::mem::swap(&mut self.lane_max, &mut ranking.lane_max);
        self.vocabulary = ranking.vocabulary;
    }

    /// Erase references the caller rejects, as if they were never indexed.
    ///
    /// Used in training to hide the query's own sequence, genus or family from the
    /// database without rebuilding it. A zeroed count can never be selected, so
    /// that is enough; `lane_max` is recomputed for affected runs.
    /// The predicate gets the count as well as the id to avoid borrowing `self`.
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

    /// `histogram[v]` is the number of references whose count is `v`, over the whole
    /// vocabulary range. A full pass over all counts; only `--risk` needs it.
    pub fn histogram(&self) -> Vec<u32> {
        let mut histogram = vec![0u32; self.vocabulary + 1];
        for &count in &self.counts {
            histogram[count as usize] += 1;
        }
        histogram
    }

    /// Smallest threshold whose `{r : count(r) >= threshold}` holds at least
    /// `target` references, and that set in ascending order.
    ///
    /// Ties at the threshold are never cut, so no tied reference is favoured.
    /// The largest `t_low` at which `target` runs have maximum >= `t_low` bounds
    /// the threshold from below, so only those runs need an exact histogram.
    pub fn select(&mut self, target: usize) -> u16 {
        let top = self.vocabulary;
        // Histogram runs by their maximum (never above the vocabulary size).
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
        // Count 0 means no shared words; such a reference can never win.
        let low = low.max(1);

        // Exact histogram of counts at or above the bound, from runs that can hold one.
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
        // Same descent as over a full histogram; it can't stop below `low`.
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
        // Candidates occupy few runs, so most of `counts` is never read.
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
    /// Each row is a bitmap over candidates that can be added straight into
    /// the counter planes during the bootstrap. `words` must be ascending and
    /// distinct, as from [`vocabulary`] and [`revcomp_vocabulary`].
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
        // Below the cap, so the running total can't overflow.
        let mut below = 0u16;
        for (prefix, bits) in self.prefix.iter_mut().zip(self.present.iter()) {
            *prefix = below;
            below += bits.count_ones() as u16;
        }
        self.depth = depth_for(SAMPLE_SIZE * weights.max() as usize);
        debug_assert!(self.depth <= PLANE_CAP);

        let lanes = self.candidates.len().div_ceil(64).max(1);
        self.matrix.clear();
        self.matrix.resize(words.len() * lanes, 0);
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
                matrix[position * lanes + lane] |= 1 << bit;
            })?;
        }
        Ok(())
    }

    /// Exact maximum and complete tie set for one replicate, over candidates.
    ///
    /// Scores are at most `SAMPLE_SIZE * max weight`. Candidate positions are
    /// in reference order, so tie draws match the exact path.
    pub fn replicate(&mut self, sample: &[u16; SAMPLE_SIZE]) -> Top {
        let lanes = self.candidates.len().div_ceil(64).max(1);
        let depth = self.depth;
        self.candidate_planes.clear();
        self.candidate_planes.resize(depth * lanes, 0);
        for &position in sample {
            let row = &self.matrix[position as usize * lanes..][..lanes];
            let shift = self.shifts[position as usize] as usize;
            ripple_add_dense(
                &mut self.candidate_planes,
                row,
                depth,
                &mut self.carry,
                shift,
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
        let (max, bits) = plane_max(&self.candidate_planes, lanes, depth, &self.valid);
        Top::from_bits(max as u8, bits)
    }

    /// Reference id of the `offset`-th member of a replicate's tie set.
    pub fn reference_at(&self, top: &Top, offset: usize) -> u32 {
        self.candidates[top.nth(offset) as usize]
    }
}

/// Risk bound for restricting replicates to the candidate set.
///
/// A pruned reference's score is `Binomial(SAMPLE_SIZE, count(r) / vocabulary)`,
/// so the chance any of them reaches the best score is at most the sum of
/// their tail probabilities, computed from the count histogram.
pub struct Bound {
    tail: Vec<f64>,
}

impl Bound {
    /// Tail probabilities for one query strand, indexed by count.
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
    // Sum from the top and stop once further terms are negligible.
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

/// Reverse complement of a word code.
///
/// Mapping the forward k-mer set gives the reverse strand's set without
/// re-reading the sequence.
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
