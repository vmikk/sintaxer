//! Per-word integer weights for the scoring stage.
//!
//! Only scoring is weighted, never the rank pass. Weights are powers of two, so
//! adding one is a single `ripple_add` from plane `s`, and tiers are chosen by
//! integer comparison only, so results never depend on platform libm.
use crate::{WORDS, index::Index};
use anyhow::{Result, ensure};

/// Weight of a word the database does not consider informative.
pub const MIN_WEIGHT: u8 = 1;

/// Weight of an informative word. Also the largest a `u8` score allows:
/// `SAMPLE_SIZE * 4` is 128.
pub const MAX_WEIGHT: u8 = 4;

/// Percentage of occurring words, rarest first, promoted to [`MAX_WEIGHT`].
/// A quantile rather than a fixed df cut, because the df distribution at
/// k=8 is very skewed and its shape does not scale with database size.
pub const DEFAULT_SHARE: u8 = 25;

/// Minimum replicate score needed to vote. Same value as unweighted scoring,
/// so abstention behaves identically either way.
pub const FLOOR: u8 = 2 * MIN_WEIGHT;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum)]
pub enum Source {
    /// Every word weighs one (identical to unweighted scoring).
    #[default]
    Off,
    /// Inverse document frequency over references (sensitive to redundancy).
    Df,
    /// Inverse document frequency over distinct genera (robust to redundancy).
    Genus,
    /// Inverse document frequency over distinct families (coarser than genus).
    Family,
}

impl Source {
    fn label(self) -> &'static str {
        match self {
            Source::Off => "off",
            Source::Df => "df",
            Source::Genus => "genus",
            Source::Family => "family",
        }
    }
}

/// A 64 KiB word-indexed weight table, built once and shared by all workers.
pub struct Table {
    weights: Box<[u8; WORDS]>,
    source: Source,
    max: u8,
}

impl Table {
    /// Identity table: every word weighs 1.
    pub fn off() -> Self {
        Self {
            weights: Box::new([1; WORDS]),
            source: Source::Off,
            max: 1,
        }
    }

    pub fn build(index: &Index, source: Source, share: u8) -> Result<Self> {
        if source == Source::Off {
            return Ok(Self::off());
        }
        ensure!(
            (1..=99).contains(&share),
            "weight share must be a percentage in 1..=99"
        );
        ensure!(index.references > 0, "empty reference database");
        let counts: Vec<u32> = (0..WORDS)
            .map(|word| match source {
                Source::Off => unreachable!(),
                Source::Df => index.document_frequency(word as u16),
                Source::Genus => index.genus_frequency(word as u16),
                Source::Family => index.family_frequency(word as u16),
            })
            .collect();

        // Threshold is the `share`-th percentile over words that actually occur;
        // absent words would otherwise drag it to zero on small databases.
        let mut occurring: Vec<u32> = counts.iter().copied().filter(|&c| c > 0).collect();
        ensure!(!occurring.is_empty(), "no words occur in the database");
        occurring.sort_unstable();
        let rank = (occurring.len() * share as usize / 100).min(occurring.len() - 1);
        let threshold = occurring[rank];

        let mut weights = Box::new([MIN_WEIGHT; WORDS]);
        for (word, slot) in weights.iter_mut().enumerate() {
            // `<=` so ties at the quantile all go to the informative tier; otherwise a
            // database where most words occur once would promote nothing. Absent words
            // stay plain, as they score zero anyway.
            if counts[word] > 0 && counts[word] <= threshold {
                *slot = MAX_WEIGHT;
            }
        }
        Ok(Self {
            weights,
            source,
            max: MAX_WEIGHT,
        })
    }

    #[inline]
    pub fn get(&self, word: u16) -> u8 {
        self.weights[word as usize]
    }
    pub fn source(&self) -> Source {
        self.source
    }
    /// Largest weight any word can carry; sizes the counter planes and score ceiling.
    pub fn max(&self) -> u8 {
        self.max
    }
    /// Word counts per tier, for `--profile`.
    pub fn tiers(&self) -> (usize, usize) {
        let informative = self
            .weights
            .iter()
            .filter(|&&w| w == MAX_WEIGHT && self.max > 1)
            .count();
        (WORDS - informative, informative)
    }
    pub fn digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"sintaxer/weights\0");
        hasher.update(self.source.label().as_bytes());
        hasher.update(self.weights.as_slice());
        hasher.finalize().to_hex()[..16].to_string()
    }
    pub fn label(&self) -> &'static str {
        self.source.label()
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::off()
    }
}
