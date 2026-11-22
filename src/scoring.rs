//! The exact scan, kept as the oracle the rank pass is checked against.
//!
//! This is published SINTAX: every reference scored in every replicate. It is
//! what `--exact` runs, what the property tests compare against, and what the
//! candidate path falls back to when its selection cannot be trusted.
use crate::{
    SAMPLE_SIZE, TILE, bits,
    index::{Index, Layout, Row, RowRef},
    weight,
};
use anyhow::{Result, ensure};

pub type Sample = [u16; SAMPLE_SIZE];
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Top {
    pub score: u8,
    pub count: usize,
    pub bits: Vec<u64>,
}
impl Top {
    fn new(n: usize) -> Self {
        Self {
            score: 0,
            count: 0,
            bits: vec![0; n.div_ceil(64)],
        }
    }
    fn merge(&mut self, score: u8, block: usize, bits: u64) {
        if bits == 0 || score < self.score {
            return;
        }
        if score > self.score {
            self.bits.fill(0);
            self.count = 0;
            self.score = score;
        }
        self.bits[block] = bits;
        self.count += bits.count_ones() as usize;
    }
    pub fn nth(&self, mut offset: usize) -> u32 {
        assert!(offset < self.count);
        for (i, &word) in self.bits.iter().enumerate() {
            let n = word.count_ones() as usize;
            if offset >= n {
                offset -= n;
                continue;
            }
            let mut word = word;
            for _ in 0..offset {
                word &= word - 1;
            }
            return (i * 64 + word.trailing_zeros() as usize) as u32;
        }
        unreachable!()
    }
    /// Adopt a maximum and its tie-set mask computed elsewhere.
    pub fn from_bits(score: u8, bits: Vec<u64>) -> Self {
        let count = bits.iter().map(|w| w.count_ones() as usize).sum();
        Self { score, count, bits }
    }
    pub fn from_scores(scores: &[u8]) -> Self {
        let mut top = Self::new(scores.len());
        for (block, chunk) in scores.chunks(64).enumerate() {
            let max = *chunk.iter().max().unwrap();
            let bits = chunk
                .iter()
                .enumerate()
                .fold(0, |bits, (i, &s)| bits | (u64::from(s == max) << i));
            top.merge(max, block, bits);
        }
        top
    }
}

/// Independent brute-force oracle for tests and small diagnostics.
pub fn oracle_scores(
    reference_words: &[Vec<u16>],
    sample: &Sample,
    weights: &weight::Table,
) -> Vec<u8> {
    reference_words
        .iter()
        .map(|words| {
            sample
                .iter()
                .filter(|w| words.contains(w))
                .map(|&w| u32::from(weights.get(w)))
                .sum::<u32>() as u8
        })
        .collect()
}

/// Which references a replicate may be won by, and so what its tie mask is
/// written over.
#[derive(Clone, Copy)]
pub enum Eligible<'a> {
    /// Every reference: the published algorithm. Tie bits are reference ids.
    All,
    /// Only the references whose bit is set in `mask`, one bit per reference
    /// over the whole database: a candidate set. Tie bits are candidate
    /// *positions* -- bit `k` is the `k`-th set bit of `mask` -- which is
    /// exactly what [`crate::rank::Workspace::replicate`] reports, so the two
    /// are interchangeable. `below[l]` is how many candidates precede lane `l`.
    Candidates { mask: &'a [u64], below: &'a [u32] },
}

/// Scratch for [`bitsliced_tops`], reused across query-strands.
#[derive(Default)]
pub struct Engine {
    /// Per vocabulary position: its row of the tile matrix, or [`UNDRAWN`].
    slot: Vec<u32>,
    /// Vocabulary positions that were drawn, ascending: one matrix row each.
    drawn: Vec<u16>,
    /// Per drawn row, the position its sparse postings have been read to.
    cursors: Vec<usize>,
    /// Drawn rows' membership for the current tile, `stride` lanes each.
    matrix: Vec<u64>,
    planes: Vec<u64>,
    valid: Vec<u64>,
    best: Vec<u64>,
    /// Each replicate's draws as matrix rows.
    draws: Vec<[u32; SAMPLE_SIZE]>,
    /// Per matrix row, the last replicate that listed it among its distinct
    /// draws: a generation stamp, so deduplicating a replicate costs a pass
    /// over its 32 draws and no clearing.
    listed: Vec<u32>,
    /// Per replicate, its distinct rows with the rarest first, and how many:
    /// the order an intersection consumes them in, which is the same in every
    /// tile. Built the first time the replicate attempts one.
    distinct: Vec<([u32; SAMPLE_SIZE], u8)>,

    /// Per replicate, the range of its tie mask written so far, so that a new
    /// maximum clears only that and not the whole database's worth of bits.
    written: Vec<(usize, usize)>,
}

const UNDRAWN: u32 = u32::MAX;
/// A replicate whose distinct rows have not been listed yet.
const UNLISTED: u8 = u8::MAX;
/// Most draws per distinct row at which a saturated strand still reads its
/// bitmap rows in place rather than from a copy; see [`bitsliced_tops`].
const IN_PLACE_REUSE: usize = 4;

/// Fold one tile's maximum and tie mask into a replicate's running [`Top`].
///
/// The same rule [`Top::from_scores`] merges blocks by: a lower maximum is
/// ignored, a higher one replaces everything so far, an equal one adds its
/// ties. Tiles never overlap, so adding a tile's ties is an OR into bits no
/// earlier tile wrote.
fn merge_tile(
    top: &mut Top,
    written: &mut (usize, usize),
    score: u8,
    base: usize,
    ties: &[u64],
    eligible: Eligible<'_>,
) {
    let count: usize = ties.iter().map(|w| w.count_ones() as usize).sum();
    if count == 0 || score < top.score {
        return;
    }
    if score > top.score {
        let (lo, hi) = *written;
        if lo < hi {
            top.bits[lo..hi].fill(0);
        }
        *written = (usize::MAX, 0);
        top.count = 0;
        top.score = score;
    }
    match eligible {
        Eligible::All => {
            for (slot, &word) in top.bits[base..base + ties.len()].iter_mut().zip(ties) {
                *slot |= word;
            }
            *written = (written.0.min(base), written.1.max(base + ties.len()));
        }
        Eligible::Candidates { mask, below } => {
            // Ties are few -- a handful per replicate, all of the candidates
            // only when nothing scores at all -- so placing them one by one
            // costs less than any compaction of whole lanes would.
            for (offset, &word) in ties.iter().enumerate() {
                let lane = base + offset;
                let mut rest = word;
                while rest != 0 {
                    let bit = rest.trailing_zeros();
                    let position = below[lane] as usize
                        + (mask[lane] & ((1u64 << bit) - 1)).count_ones() as usize;
                    top.bits[position / 64] |= 1 << (position % 64);
                    *written = (
                        written.0.min(position / 64),
                        written.1.max(position / 64 + 1),
                    );
                    rest &= rest - 1;
                }
            }
        }
    }
    top.count += count;
}

/// Every replicate's maximum and complete tie set, bit-sliced: the published
/// all-reference scan, and [`scalar_scores`] + [`Top::from_scores`] exactly.
///
/// `samples` are positions into `words`, as the rest of the classifier
/// draws them. The database is walked in tiles of [`crate::TILE`]
/// references, the boundary the index's blocked rows are cut on. For each
/// tile the drawn words' rows are materialised once as bitmaps -- dense rows
/// copied, sparse ones set from their postings -- and then every replicate
/// counts its 32 rows through the carry-save network in [`crate::bits`], so
/// the database is read once per query-strand instead of once per
/// replicate. The tile's maximum and tie mask are merged into that
/// replicate's [`Top`] with the same rule `from_scores` merges blocks by, so
/// the result is identical to the scalar scan, bit for bit, including the
/// all-zero replicate whose tie set is every eligible reference.
///
/// A replicate some reference scores perfectly on -- every draw present --
/// has that reference among its winners, and the winners are then exactly the
/// intersection of the drawn rows. That intersection is far cheaper than
/// counting, so it is tried first while it keeps succeeding: on close
/// matches it nearly always does, and on novel queries a couple of misses
/// switch it off. Either way the answer is the same.
///
/// With [`Eligible::Candidates`] only the candidates can win or tie, tiles
/// without one are skipped outright, and ties are reported over candidate
/// positions: the same `Top`, bit for bit, that projecting the candidates and
/// running [`crate::rank::Workspace::replicate`] produces.
///
/// `saturated` says that some eligible reference holds every word of
/// `words`. That reference then holds every draw of every replicate, so each
/// replicate's maximum is its ceiling and its ties are exactly the
/// intersection of its rows: no replicate needs counting at all. The caller
/// knows this for free from the rank pass; passing `false` is always safe.
///
/// `known`, when given, is where each word of `words` has its row, from the
/// rank sweep that has just read the same directory entries
/// ([`crate::rank::Workspace::row_refs`]); without it they are looked up.
#[allow(clippy::too_many_arguments)]
pub fn bitsliced_tops(
    index: &Index,
    words: &[u16],
    samples: &[Sample],
    weights: &weight::Table,
    eligible: Eligible<'_>,
    saturated: bool,
    known: Option<&[RowRef]>,
    ws: &mut Engine,
) -> Result<Vec<Top>> {
    let n = index.references;
    let width = match eligible {
        Eligible::All => n,
        Eligible::Candidates { mask, below } => {
            ensure!(
                mask.len() == n.div_ceil(64) && below.len() == mask.len() + 1,
                "candidate mask does not cover the database"
            );
            // As wide as the candidate path's own tie masks, which are never
            // empty vectors.
            (below[mask.len()] as usize).max(1)
        }
    };
    let mut tops: Vec<Top> = samples.iter().map(|_| Top::new(width)).collect();
    ws.written.clear();
    ws.written.resize(samples.len(), (usize::MAX, 0));

    // One matrix row per distinct drawn position, in ascending position
    // order: which rows exist, and in what order, cannot affect a count.
    ws.slot.clear();
    ws.slot.resize(words.len(), UNDRAWN);
    for sample in samples {
        for &position in sample {
            ensure!(
                (position as usize) < words.len(),
                "draw outside the query vocabulary"
            );
            ws.slot[position as usize] = 0;
        }
    }
    ws.drawn.clear();
    for (position, slot) in ws.slot.iter_mut().enumerate() {
        if *slot != UNDRAWN {
            *slot = ws.drawn.len() as u32;
            ws.drawn.push(position as u16);
        }
    }
    if let Some(known) = known {
        ensure!(
            known.len() == words.len(),
            "row references do not cover the query vocabulary"
        );
    }
    let rows: Vec<Row<'_>> = ws
        .drawn
        .iter()
        .map(|&position| match known {
            Some(known) => Ok(index.row_at(known[position as usize])),
            None => index.row(words[position as usize]),
        })
        .collect::<Result<_>>()?;
    let shifts: Vec<u8> = ws
        .drawn
        .iter()
        .map(|&position| weights.get(words[position as usize]).trailing_zeros() as u8)
        .collect();
    let weighted = weights.max() > 1;
    let depth = if weighted {
        bits::WEIGHTED_PLANES
    } else {
        bits::PLANES
    };

    ws.draws.clear();
    let mut ceilings = Vec::with_capacity(samples.len());
    for sample in samples {
        ws.draws
            .push(std::array::from_fn(|i| ws.slot[sample[i] as usize]));
        // The highest score this replicate can produce: every draw present.
        ceilings.push(
            sample
                .iter()
                .map(|&p| u32::from(weights.get(words[p as usize])))
                .sum::<u32>() as u8,
        );
    }
    ws.listed.clear();
    ws.listed.resize(rows.len(), u32::MAX);
    ws.distinct.clear();
    ws.distinct
        .resize(samples.len(), ([0; SAMPLE_SIZE], UNLISTED));
    let mut perfect = vec![saturated; samples.len()];
    let (mut tried, mut hits) = (0usize, 0usize);
    // A saturated strand answers every replicate by intersection and never
    // counts, so its bitmap rows need not be copied into the matrix at all:
    // the intersections read them where they lie in the index. That saves a
    // copy of every bitmap row in every tile, which on a full-length 16S
    // self-test was an eighth of all cycles. It pays while a row is ANDed only
    // a few times; with many draws per row -- short amplicons, small
    // vocabularies -- the copy is the cheaper thing to read over and over.
    let in_place = saturated && rows.len() * IN_PLACE_REUSE >= samples.len() * SAMPLE_SIZE;

    ws.cursors.clear();
    ws.cursors.resize(rows.len(), 0);
    let mut start = 0;
    while start < n {
        let span = (n - start).min(TILE);
        let lanes = span.div_ceil(64);
        let stride = bits::stride(lanes);
        let base = start / 64;

        ws.valid.clear();
        ws.valid.resize(stride, 0);
        match eligible {
            Eligible::All => {
                ws.valid[..lanes].fill(u64::MAX);
                if !span.is_multiple_of(64) {
                    ws.valid[lanes - 1] = (1u64 << (span % 64)) - 1;
                }
            }
            Eligible::Candidates { mask, .. } => {
                ws.valid[..lanes].copy_from_slice(&mask[base..base + lanes]);
            }
        }
        if ws.valid.iter().all(|&w| w == 0) {
            for (row, cursor) in rows.iter().zip(&mut ws.cursors) {
                row.skip_tile(start, cursor);
            }
            start += TILE;
            continue;
        }

        // Every lane of every row it needs is written below, so the matrix is
        // only ever grown, never cleared.
        if ws.matrix.len() < rows.len() * stride {
            ws.matrix.resize(rows.len() * stride, 0);
        }
        for ((row, cursor), out) in rows
            .iter()
            .zip(&mut ws.cursors)
            .zip(ws.matrix.chunks_exact_mut(stride))
        {
            // Bitmap rows keep no cursor, so leaving one out of the matrix
            // costs nothing to catch up on in the next tile.
            if !(in_place
                && row.layout == Layout::Bitmap
                && row.dense_bytes(start, lanes).is_some())
            {
                row.tile_bits(start, cursor, out);
            }
        }
        ws.planes.resize(depth * stride, 0);
        ws.best.resize(stride, 0);

        for (b, top) in tops.iter_mut().enumerate() {
            let ceiling = ceilings[b];
            let written = &mut ws.written[b];
            if perfect[b] || tried < 4 || hits * 4 >= tried {
                // The distinct draws, the rarest row first: an intersection
                // that is going to come up empty usually does so on its
                // rarest row, and then the scan stops there. The list is the
                // same in every tile, so it is built once per replicate.
                if ws.distinct[b].1 == UNLISTED {
                    let mut distinct = [0u32; SAMPLE_SIZE];
                    let mut len = 0;
                    for &slot in &ws.draws[b] {
                        let stamp = &mut ws.listed[slot as usize];
                        if *stamp != b as u32 {
                            *stamp = b as u32;
                            distinct[len] = slot;
                            len += 1;
                        }
                    }
                    let rarest = (0..len)
                        .min_by_key(|&i| rows[distinct[i] as usize].count)
                        .unwrap_or(0);
                    distinct.swap(0, rarest);
                    ws.distinct[b] = (distinct, len as u8);
                }
                if !perfect[b] {
                    tried += 1;
                }
                let (distinct, len) = ws.distinct[b];
                let best = &mut ws.best[..lanes];
                best.copy_from_slice(&ws.valid[..lanes]);
                let mut whole = true;
                for &slot in &distinct[..len as usize] {
                    let row = &rows[slot as usize];
                    let bytes = if in_place && row.layout == Layout::Bitmap {
                        row.dense_bytes(start, lanes)
                    } else {
                        None
                    };
                    let any = match bytes {
                        Some(bytes) => bits::and_bytes(best, bytes),
                        None => {
                            bits::and_lanes(best, &ws.matrix[slot as usize * stride..][..lanes])
                        }
                    };
                    if !any {
                        whole = false;
                        break;
                    }
                }
                if whole {
                    if !perfect[b] {
                        hits += 1;
                    }
                    perfect[b] = true;
                    merge_tile(top, written, ceiling, base, &ws.best[..lanes], eligible);
                    continue;
                }
                // Some earlier tile already reached the ceiling, and nothing
                // here does, so nothing here can win or tie.
                if perfect[b] {
                    continue;
                }
            }
            debug_assert!(!in_place, "a saturated strand never counts");
            let offsets: [usize; SAMPLE_SIZE] =
                std::array::from_fn(|i| ws.draws[b][i] as usize * stride);
            if weighted {
                let draw_shifts: [u8; SAMPLE_SIZE] =
                    std::array::from_fn(|i| shifts[ws.draws[b][i] as usize]);
                bits::count_weighted(
                    &ws.matrix,
                    &offsets,
                    &draw_shifts,
                    lanes,
                    stride,
                    &mut ws.planes,
                );
            } else {
                bits::count32(&ws.matrix, &offsets, lanes, stride, &mut ws.planes);
            }
            let max = bits::masked_max(&ws.planes, stride, depth, lanes, &ws.valid, &mut ws.best);
            let max = max as u8;
            if max == ceiling {
                perfect[b] = true;
            }
            merge_tile(top, written, max, base, &ws.best[..lanes], eligible);
        }
        start += TILE;
    }
    Ok(tops)
}

pub fn scalar_scores(
    index: &Index,
    sample: &Sample,
    weights: &weight::Table,
    scores: &mut Vec<u8>,
) -> Result<()> {
    scores.resize(index.references, 0);
    scores.fill(0);
    for &word in sample {
        // `SAMPLE_SIZE * MAX_WEIGHT` is 128, so a u8 counter still cannot
        // overflow even when every draw lands on a maximally weighted word.
        let w = weights.get(word);
        index
            .row(word)?
            .visit(0, index.references, |id| scores[id] += w);
    }
    Ok(())
}
