//! Algorithm-v1 PRNG: BLAKE3 domain separation + SplitMix64 + rejection sampling.
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

/// [`Rng::bounded`] for one `n` drawn from many times, with its setup hoisted.
///
/// `bounded` spends two hardware divisions per draw: one for the rejection
/// threshold, which depends only on `n`, and one for the remainder. A
/// bootstrap strand makes 3,200 draws from the same `n`, so here the threshold
/// is computed once and the remainder comes from a precomputed reciprocal: a
/// 64x64-bit high multiply and one conditional subtraction.
///
/// It is not a different generator. The same values are consumed from the
/// stream and the same ones rejected, and the remainder is exact, so every
/// draw equals what `bounded(n)` returns -- which the tests check across `n`.
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

    /// `x % n`, exactly.
    ///
    /// `q = floor(x * reciprocal / 2^64)` undershoots `floor(x / n)` by at
    /// most one, because the reciprocal is short of `2^64 / n` by less than
    /// one and `x` is below `2^64`. So `x - q * n` is below `2n`, and one
    /// subtraction finishes the job.
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
/// The seed is the BLAKE3 hash of `STREAM_DOMAIN || key || strand || role ||
/// bootstrap`. It is hashed in one call over a stack buffer rather than fed to
/// an incremental `Hasher`, whose setup cost more than the single compression
/// the 54 bytes need; BLAKE3 hashes the byte string, not the way it was
/// split, so the value is the same.
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

/// [`stream`] for eight bootstraps of one strand and role at once.
///
/// A strand needs a hundred seeds, and each is one BLAKE3 compression of a
/// 54-byte input -- 60-70 ns through the crate's single-message API, which
/// has no way to run independent short messages side by side. Here eight run
/// in the lanes of AVX2 registers ([`crate::simd::blake3_first_words8`]). It
/// is the same function over the same bytes, not a different hash: the
/// inputs differ only in message words 12 and 13, which carry the bootstrap
/// number, and the test holds every lane to `blake3::hash`.
///
/// A one-block message is its own chunk and its own root, so the
/// compression runs once, from the IV, with counter zero, block length 54 and
/// the flags CHUNK_START | CHUNK_END | ROOT; a seed is the first eight bytes of
/// the root output, which are the first two output words. Without AVX2 the
/// seeds are hashed one at a time.
pub fn streams8(key: &blake3::Hash, strand: u8, role: u8, boots: [u32; 8]) -> [Rng; 8] {
    let word = |bytes: &[u8]| u32::from_le_bytes(bytes.try_into().expect("four bytes"));
    let mut common = [0u32; 16];
    for (i, chunk) in STREAM_DOMAIN.chunks_exact(4).enumerate() {
        common[i] = word(chunk);
    }
    for (i, chunk) in key.as_bytes().chunks_exact(4).enumerate() {
        common[4 + i] = word(chunk);
    }
    // Bytes 48..54: strand, role, then the bootstrap number little-endian;
    // bytes 54..64 are the block's zero padding.
    let low = u32::from(strand) | u32::from(role) << 8;
    let w12 = boots.map(|boot| low | (boot & 0xFFFF) << 16);
    let w13 = boots.map(|boot| boot >> 16);
    match crate::simd::blake3_first_words8(&common, w12, w13, SEED_INPUT as u32, ONE_BLOCK_ROOT) {
        Some(seeds) => seeds.map(Rng),
        None => boots.map(|boot| stream(key, strand, boot as usize, role)),
    }
}

/// Bytes hashed per seed: domain, key, strand, role and bootstrap.
const SEED_INPUT: usize = 16 + 32 + 2 + 4;
/// BLAKE3's CHUNK_START | CHUNK_END | ROOT: a message of one block.
const ONE_BLOCK_ROOT: u32 = 1 | 2 | 8;

