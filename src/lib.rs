//! Deterministic, versioned SINTAX-style classification.
pub mod classify;
pub mod curate;
pub mod index;
pub mod input;
pub mod rank;
pub mod rng;
pub mod scoring;
pub mod sequence;
pub mod sketch;
pub mod taxonomy;
pub mod weight;

pub const ALGORITHM_VERSION: u32 = 1;
/// Scoring behaviour version, for provenance and reports.
///
/// Separate from [`ALGORITHM_VERSION`], which also keys every bootstrap draw,
/// so behaviour changes don't re-key draws with the new options off.
pub const CLASSIFIER_VERSION: u32 = 3;
pub const K: usize = 8;
pub const WORDS: usize = 1 << (2 * K);
pub const BOOTSTRAPS: usize = 100;
pub const SAMPLE_SIZE: usize = 32;
