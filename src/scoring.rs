//! Exact scoring kernels. All engines keep complete, ordered tie sets.
use crate::{
    SAMPLE_SIZE,
    index::{Index, Row},
};
use anyhow::{Result, ensure};
use clap::ValueEnum;
use std::collections::BTreeMap;

pub type Sample = [u16; SAMPLE_SIZE];
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Engine {
    Scalar,
    Batched,
    Bitslice,
    Avx2,
    Projected,
    Auto,
}

#[derive(Default)]
pub struct Workspace {
    pub scores: Vec<u8>,
    tile: Vec<u8>,
    projection: Vec<u8>,
}

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
pub fn oracle_scores(reference_words: &[Vec<u16>], sample: &Sample) -> Vec<u8> {
    reference_words
        .iter()
        .map(|words| sample.iter().filter(|w| words.contains(w)).count() as u8)
        .collect()
}

pub fn scalar_scores(index: &Index, sample: &Sample, scores: &mut Vec<u8>) -> Result<()> {
    scores.resize(index.references, 0);
    scores.fill(0);
    for &word in sample {
        index
            .row(word)?
            .visit(0, index.references, |id| scores[id] += 1);
    }
    Ok(())
}

pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Six bit planes, each holding four independent groups of 64 references.
pub fn planes(rows: &[Row<'_>], start: usize, avx2: bool) -> [[u64; 4]; 6] {
    assert!(rows.len() <= SAMPLE_SIZE && start % 64 == 0);
    #[cfg(target_arch = "x86_64")]
    if avx2 && avx2_available() {
        // SAFETY: AVX2 support was checked just above; all vector loads/stores
        // address local arrays of exactly four u64 values.
        return unsafe { planes_avx2(rows, start) };
    }
    let _ = avx2;
    let mut planes = [[0; 4]; 6];
    for row in rows {
        for lane in 0..4 {
            let mut carry = row.bits64(start + lane * 64);
            for plane in &mut planes {
                let next = plane[lane] & carry;
                plane[lane] ^= carry;
                carry = next;
            }
        }
    }
    planes
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn planes_avx2(rows: &[Row<'_>], start: usize) -> [[u64; 4]; 6] {
    use std::arch::x86_64::*;
    let mut accum = [_mm256_setzero_si256(); 6];
    for row in rows {
        let mut carry = if let Some(bytes) = row.dense256(start) {
            // SAFETY: dense256 returns exactly 32 readable bytes.
            unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) }
        } else {
            let words = [
                row.bits64(start),
                row.bits64(start + 64),
                row.bits64(start + 128),
                row.bits64(start + 192),
            ];
            // SAFETY: words has 32 readable bytes; the load is unaligned.
            unsafe { _mm256_loadu_si256(words.as_ptr().cast()) }
        };
        for plane in &mut accum {
            let next = _mm256_and_si256(*plane, carry);
            *plane = _mm256_xor_si256(*plane, carry);
            carry = next;
        }
    }
    let mut result = [[0u64; 4]; 6];
    for (dst, src) in result.iter_mut().zip(accum) {
        // SAFETY: each destination is a writable 32-byte array.
        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().cast(), src);
        }
    }
    result
}

fn bitslice_top(index: &Index, sample: &Sample, avx2: bool) -> Result<Top> {
    let rows: Vec<_> = sample
        .iter()
        .map(|&w| index.row(w))
        .collect::<Result<_>>()?;
    let mut top = Top::new(index.references);
    for start in (0..index.references).step_by(256) {
        let planes = planes(&rows, start, avx2);
        for (lane, _) in planes[0].iter().enumerate() {
            let pos = start + lane * 64;
            if pos >= index.references {
                break;
            }
            let n = (index.references - pos).min(64);
            let mut candidates = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            let mut max = 0;
            for bit in (0..6).rev() {
                let subset = candidates & planes[bit][lane];
                if subset != 0 {
                    candidates = subset;
                    max |= 1 << bit;
                }
            }
            top.merge(max, pos / 64, candidates);
        }
    }
    Ok(top)
}

fn batched(
    index: &Index,
    samples: &[Sample],
    tile_size: usize,
    ws: &mut Workspace,
) -> Result<Vec<Top>> {
    let b = samples.len();
    let mut weights = BTreeMap::<u16, Vec<u8>>::new();
    for (i, sample) in samples.iter().enumerate() {
        for &word in sample {
            weights.entry(word).or_insert_with(|| vec![0; b])[i] += 1;
        }
    }
    let rows: Vec<_> = weights
        .iter()
        .map(|(&word, weights)| Ok((index.row(word)?, weights)))
        .collect::<Result<_>>()?;
    let mut tops: Vec<_> = (0..b).map(|_| Top::new(index.references)).collect();
    ws.tile.resize(tile_size * b, 0);
    for start in (0..index.references).step_by(tile_size) {
        let end = (start + tile_size).min(index.references);
        ws.tile.fill(0);
        for (row, weights) in &rows {
            row.visit(start, end, |id| {
                let off = (id - start) * b;
                for (score, weight) in ws.tile[off..off + b].iter_mut().zip(weights.iter()) {
                    *score += weight;
                }
            });
        }
        for (i, top) in tops.iter_mut().enumerate() {
            for block in (start..end).step_by(64) {
                let mut max = 0;
                let mut bits = 0;
                for r in block..(block + 64).min(end) {
                    let score = ws.tile[(r - start) * b + i];
                    if score > max {
                        max = score;
                        bits = 0;
                    }
                    if score == max {
                        bits |= 1 << (r - block);
                    }
                }
                top.merge(max, block / 64, bits);
            }
        }
    }
    Ok(tops)
}

fn projected(
    index: &Index,
    samples: &[Sample],
    tile_size: usize,
    ws: &mut Workspace,
) -> Result<Vec<Top>> {
    let mut words: Vec<_> = samples.iter().flatten().copied().collect();
    words.sort_unstable();
    words.dedup();
    let source: Vec<_> = words.iter().map(|&w| index.row(w)).collect::<Result<_>>()?;
    let plans: Vec<[usize; SAMPLE_SIZE]> = samples
        .iter()
        .map(|sample| std::array::from_fn(|i| words.binary_search(&sample[i]).unwrap()))
        .collect();
    let bytes_per_row = tile_size / 8;
    ws.projection.resize(words.len() * bytes_per_row, 0);
    let mut tops: Vec<_> = samples.iter().map(|_| Top::new(index.references)).collect();
    let avx2 = avx2_available();
    for start in (0..index.references).step_by(tile_size) {
        let end = (start + tile_size).min(index.references);
        ws.projection.fill(0);
        for (i, row) in source.iter().enumerate() {
            let dst = &mut ws.projection[i * bytes_per_row..(i + 1) * bytes_per_row];
            if row.dense {
                for (block, chunk) in dst.chunks_exact_mut(8).enumerate() {
                    chunk.copy_from_slice(&row.bits64(start + block * 64).to_le_bytes());
                }
            } else {
                row.visit(start, end, |id| {
                    let local = id - start;
                    dst[local / 8] |= 1 << (local % 8);
                });
            }
        }
        for (plan, top) in plans.iter().zip(tops.iter_mut()) {
            let rows: [Row<'_>; SAMPLE_SIZE] = std::array::from_fn(|i| {
                let off = plan[i] * bytes_per_row;
                Row::bitmap(&ws.projection[off..off + bytes_per_row])
            });
            for local in (0..end - start).step_by(256) {
                let planes = planes(&rows, local, avx2);
                for (lane, _) in planes[0].iter().enumerate() {
                    let pos = start + local + lane * 64;
                    if pos >= end {
                        break;
                    }
                    let n = (end - pos).min(64);
                    let mut candidates = if n == 64 { u64::MAX } else { (1 << n) - 1 };
                    let mut max = 0;
                    for (bit, plane) in planes.iter().enumerate().rev() {
                        let selected = candidates & plane[lane];
                        if selected != 0 {
                            candidates = selected;
                            max |= 1 << bit;
                        }
                    }
                    top.merge(max, pos / 64, candidates);
                }
            }
        }
    }
    Ok(tops)
}

/// Callbacks run in bootstrap order. Tie storage per batch is bounded by
/// batch_size * ceil(N/64) * 8 bytes, never a full N*100 score matrix.
pub fn score_bootstraps(
    index: &Index,
    samples: &[Sample],
    engine: Engine,
    tile_size: usize,
    batch_size: usize,
    ws: &mut Workspace,
    mut consume: impl FnMut(usize, &Top),
) -> Result<()> {
    ensure!(
        [256, 1024, 4096].contains(&tile_size),
        "tile size must be 256, 1024, or 4096"
    );
    ensure!(
        [8, 16, 32].contains(&batch_size),
        "bootstrap batch must be 8, 16, or 32"
    );
    ensure!(
        engine != Engine::Avx2 || avx2_available(),
        "AVX2 is not available on this CPU"
    );
    if matches!(engine, Engine::Batched | Engine::Projected) {
        for (chunk, samples) in samples.chunks(batch_size).enumerate() {
            let tops = if engine == Engine::Projected {
                projected(index, samples, tile_size, ws)?
            } else {
                batched(index, samples, tile_size, ws)?
            };
            for (i, top) in tops.iter().enumerate() {
                consume(chunk * batch_size + i, top);
            }
        }
    } else {
        for (i, sample) in samples.iter().enumerate() {
            let actual = if engine == Engine::Auto {
                // Experimental cost rule; opt-in only.
                let dense = sample
                    .iter()
                    .map(|&w| index.row(w).map(|r| usize::from(r.dense)))
                    .sum::<Result<usize>>()?;
                if dense >= 16 {
                    if avx2_available() {
                        Engine::Avx2
                    } else {
                        Engine::Bitslice
                    }
                } else {
                    Engine::Scalar
                }
            } else {
                engine
            };
            let top = match actual {
                Engine::Bitslice | Engine::Avx2 => {
                    bitslice_top(index, sample, actual == Engine::Avx2)?
                }
                _ => {
                    scalar_scores(index, sample, &mut ws.scores)?;
                    Top::from_scores(&ws.scores)
                }
            };
            consume(i, &top);
        }
    }
    Ok(())
}
