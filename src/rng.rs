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

/// [`Rng::bounded`] for repeated draws from one `n`.
///
/// The rejection threshold is computed once and the remainder uses a
/// precomputed reciprocal. Every draw equals what `bounded(n)` returns.
pub struct Uniform {
    n: u64,
    threshold: u64,
    /// `floor(2^64 / n)`, or zero when `n` is one.
    reciprocal: u64,
}

impl Uniform {
    pub fn new(n: u64) -> Self {
        assert!(n > 0);
        Self {
            n,
            threshold: n.wrapping_neg() % n,
            reciprocal: if n == 1 {
                0
            } else {
                ((1u128 << 64) / u128::from(n)) as u64
            },
        }
    }

    /// `x % n`, exactly: the reciprocal quotient is off by at most one.
    #[inline]
    fn remainder(&self, x: u64) -> u64 {
        if self.n == 1 {
            return 0;
        }
        let q = ((u128::from(x) * u128::from(self.reciprocal)) >> 64) as u64;
        let r = x - q * self.n;
        if r >= self.n { r - self.n } else { r }
    }

    /// The next value of `rng.bounded(n)`.
    #[inline]
    pub fn sample(&self, rng: &mut Rng) -> u64 {
        loop {
            let x = rng.next_u64();
            if x >= self.threshold {
                return self.remainder(x);
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

const STREAM_DOMAIN: &[u8; 16] = b"sintaxer/stream\0";

/// One replicate's generator: role 0 samples its draws, role 1 breaks its tie.
///
/// Seeded from BLAKE3 of `STREAM_DOMAIN || key || strand || role || bootstrap`,
/// hashed in one call.
pub fn stream(key: &blake3::Hash, strand: u8, bootstrap: usize, role: u8) -> Rng {
    let mut input = [0u8; 16 + 32 + 2 + 4];
    input[..16].copy_from_slice(STREAM_DOMAIN);
    input[16..48].copy_from_slice(key.as_bytes());
    input[48] = strand;
    input[49] = role;
    input[50..].copy_from_slice(&(bootstrap as u32).to_le_bytes());
    let digest = blake3::hash(&input);
    let mut seed = [0u8; 8];
    seed.copy_from_slice(&digest.as_bytes()[..8]);
    Rng(u64::from_le_bytes(seed))
}
