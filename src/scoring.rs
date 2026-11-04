//! Exact scan: every reference scored in every replicate.
//!
//! Used by `--exact`, by tests, and as the fallback when candidate
//! selection can't be trusted.
use crate::{
    SAMPLE_SIZE, TILE, bits,
    index::{Index, Row},
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
    /// Build from a maximum and tie-set mask computed elsewhere.
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

/// Brute-force reference implementation for tests and diagnostics.
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

/// Which references may win a replicate, and what its tie mask is over.
#[derive(Clone, Copy)]
pub enum Eligible<'a> {
    /// Every reference: the published algorithm. Tie bits are reference ids.
    All,
    /// Only the candidates set in `mask`. Tie bits are candidate positions,
    /// as [`crate::rank::Workspace::replicate`] reports them. `below[l]` is how
    /// many candidates precede lane `l`.
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
    /// Per matrix row, the last replicate that listed it: dedups draws without clearing.
    listed: Vec<u32>,
    /// Per replicate, the range of its tie mask written so far.
    written: Vec<(usize, usize)>,
}

const UNDRAWN: u32 = u32::MAX;

/// Fold one tile's maximum and tie mask into a replicate's running [`Top`].
///
/// Same rule as [`Top::from_scores`]: a lower maximum is ignored, a higher
/// one replaces everything so far, an equal one adds its ties.
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
            // Ties are few, so place them one by one.
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

/// Every replicate's maximum and full tie set, bit-sliced. Same result as
/// [`scalar_scores`] + [`Top::from_scores`].
///
/// `samples` are positions into `words`. The database is walked in tiles of
/// [`crate::TILE`] references; per tile the drawn rows are materialised once
/// as bitmaps, then each replicate counts its 32 rows through
/// [`crate::bits`] and merges the tile result into its [`Top`].
///
/// If some reference has every draw, the winners are exactly the AND of the
/// drawn rows, which is much cheaper than counting. That is tried first
/// while it keeps succeeding.
///
/// With [`Eligible::Candidates`], tiles without a candidate are skipped and
/// ties are over candidate positions.
///
/// `saturated`: some eligible reference holds every word of `words`, so
/// every replicate is perfect and nothing needs counting. `false` is always safe.
pub fn bitsliced_tops(
    index: &Index,
    words: &[u16],
    samples: &[Sample],
    weights: &weight::Table,
    eligible: Eligible<'_>,
    saturated: bool,
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
            // Tie masks are never empty vectors.
            (below[mask.len()] as usize).max(1)
        }
    };
    let mut tops: Vec<Top> = samples.iter().map(|_| Top::new(width)).collect();
    ws.written.clear();
    ws.written.resize(samples.len(), (usize::MAX, 0));

    // One matrix row per distinct drawn position, ascending.
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
    let rows: Vec<Row<'_>> = ws
        .drawn
        .iter()
        .map(|&position| index.row(words[position as usize]))
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
    let mut perfect = vec![saturated; samples.len()];
    let (mut tried, mut hits) = (0usize, 0usize);

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

        // Every lane is written below, so the matrix is never cleared.
        if ws.matrix.len() < rows.len() * stride {
            ws.matrix.resize(rows.len() * stride, 0);
        }
        ws.listed.fill(u32::MAX);
        for ((row, cursor), out) in rows
            .iter()
            .zip(&mut ws.cursors)
            .zip(ws.matrix.chunks_exact_mut(stride))
        {
            row.tile_bits(start, cursor, out);
        }
        ws.planes.resize(depth * stride, 0);
        ws.best.resize(stride, 0);

        for (b, top) in tops.iter_mut().enumerate() {
            let ceiling = ceilings[b];
            let written = &mut ws.written[b];
            if perfect[b] || tried < 4 || hits * 4 >= tried {
                // Distinct draws, rarest row first so a miss stops early.
                let mut offsets = [0usize; SAMPLE_SIZE];
                let mut len = 0;
                for &slot in &ws.draws[b] {
                    let stamp = &mut ws.listed[slot as usize];
                    if *stamp != b as u32 {
                        *stamp = b as u32;
                        offsets[len] = slot as usize;
                        len += 1;
                    }
                }
                let rarest = (0..len)
                    .min_by_key(|&i| rows[offsets[i]].count)
                    .unwrap_or(0);
                offsets.swap(0, rarest);
                for offset in &mut offsets[..len] {
                    *offset *= stride;
                }
                let len = len as u8;
                if !perfect[b] {
                    tried += 1;
                }
                if bits::intersect(
                    &ws.matrix,
                    &offsets[..len as usize],
                    lanes,
                    &ws.valid,
                    &mut ws.best,
                ) {
                    if !perfect[b] {
                        hits += 1;
                    }
                    perfect[b] = true;
                    merge_tile(top, written, ceiling, base, &ws.best[..lanes], eligible);
                    continue;
                }
                // An earlier tile reached the ceiling and this one can't.
                if perfect[b] {
                    continue;
                }
            }
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
        // `SAMPLE_SIZE * MAX_WEIGHT` is 128, so a u8 counter can't overflow.
        let w = weights.get(word);
        index
            .row(word)?
            .visit(0, index.references, |id| scores[id] += w);
    }
    Ok(())
}
