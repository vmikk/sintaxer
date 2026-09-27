//! Deterministic PRNG: BLAKE3 domain separation, SplitMix64 and rejection sampling.
use crate::ALGORITHM_VERSION;

pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn bounded(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return x % n;
            }
        }
    }
}

pub fn query_key(seed: u64, normalized: &[u8]) -> blake3::Hash {
    let mut h = blake3::Hasher::new();
    h.update(b"sintaxer/query\0");
    h.update(&ALGORITHM_VERSION.to_le_bytes());
    h.update(&seed.to_le_bytes());
    h.update(normalized);
    h.finalize()
}

pub fn stream(key: &blake3::Hash, strand: u8, bootstrap: usize, role: u8) -> Rng {
    let mut h = blake3::Hasher::new();
    h.update(b"sintaxer/stream\0");
    h.update(key.as_bytes());
    h.update(&[strand, role]); // role 0: sampling, role 1: tie selection
    h.update(&(bootstrap as u32).to_le_bytes());
    Rng(u64::from_le_bytes(
        h.finalize().as_bytes()[..8].try_into().unwrap(),
    ))
}
