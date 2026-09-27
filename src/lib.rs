//! Deterministic, versioned SINTAX-style classification.
pub mod classify;
pub mod index;
pub mod input;
pub mod rng;
pub mod scoring;
pub mod sequence;
pub mod taxonomy;

pub const ALGORITHM_VERSION: u32 = 1;
pub const K: usize = 8;
pub const WORDS: usize = 1 << (2 * K);
pub const BOOTSTRAPS: usize = 100;
pub const SAMPLE_SIZE: usize = 32;
