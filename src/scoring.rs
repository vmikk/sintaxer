//! Exact scan: every reference scored in every replicate.
//!
//! Used by `--exact`, by tests, and as the fallback when candidate
//! selection can't be trusted.
use crate::{SAMPLE_SIZE, index::Index};
use anyhow::Result;

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
