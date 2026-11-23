//! Versioned, deterministic SINTAX-style classification.
pub mod bits;
pub mod classify;
pub mod curate;
pub mod index;
pub mod input;
pub mod protax;
pub mod rank;
pub mod rng;
pub mod scoring;
pub mod sequence;
pub mod simd;
pub mod sketch;
pub mod taxonomy;
pub mod weight;

pub const ALGORITHM_VERSION: u32 = 1;
/// Scoring behaviour, for provenance and reports.
///
/// Deliberately separate from [`ALGORITHM_VERSION`], which is simultaneously
/// the index-compatibility tag and the BLAKE3 domain separator for every
/// bootstrap draw (`rng::query_key`). Bumping that to record a behaviour
/// change would re-key every draw even with weighting off, and destroy the one
/// check worth having here: that `--weights off` reproduces the previous
/// binary's output byte for byte. That check is a manual golden-output
/// comparison (docs/RANK_round2.md §6), not an automated test.
/// `sketch::CURATION_VERSION` exists for the same reason.
///
/// 2: per-word scoring weights (`--weights`).
/// 3: missing ranks abstain from the consensus vote instead of competing in
///    it, and `--support adjusted` can take them out of the denominator too.
/// 4: `--scorer protax` replaces the vote fraction with a calibrated
///    probability carrying an explicit novel-taxon outcome. `--scorer sintax`
///    is unchanged and remains the default.
pub const CLASSIFIER_VERSION: u32 = 4;
pub const K: usize = 8;
/// References per tile, shared by the rank sweep and the index layout.
///
/// The sweep sizes its counter planes by it, and format 5's blocked posting
/// rows are cut on the same boundary so that a posting fits in a `u16` and the
/// sweep's cursor never has to compare anything. Changing it changes the
/// on-disk format.
pub const TILE: usize = 4096;
pub const WORDS: usize = 1 << (2 * K);
pub const BOOTSTRAPS: usize = 100;
pub const SAMPLE_SIZE: usize = 32;
