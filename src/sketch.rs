//! Bottom-k MinHash over the 16-bit word universe, used only for curation.
//! With `K = 8` a permutation of `u16` gives a collision-free min-hash.
//! Changes here affect which references curation keeps, never scoring.

/// Selection contract version. Independent of `ALGORITHM_VERSION`, since
/// curation changes the database, never the classifier.
pub const CURATION_VERSION: u32 = 1;

/// Values per LSH band. Four `u16` fill a `u64` key exactly.
pub const BAND: usize = 4;

/// A bijection on `u16`. Every step (xor-shift, odd multiply) is invertible
/// mod 2^16, so the bottom-k sample is uniform.
pub fn permute(word: u16) -> u16 {
    let mut x = word;
    x ^= x >> 8;
    x = x.wrapping_mul(0x88b5);
    x ^= x >> 7;
    x = x.wrapping_mul(0xe1b5);
    x ^= x >> 9;
    x
}

/// The `S` smallest permuted words, ascending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sketch<const S: usize> {
    values: [u16; S],
    len: u16,
}

impl<const S: usize> Default for Sketch<S> {
    fn default() -> Self {
        Self {
            values: [u16::MAX; S],
            len: 0,
        }
    }
}

impl<const S: usize> Sketch<S> {
    pub fn values(&self) -> &[u16] {
        &self.values[..self.len as usize]
    }
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// `u64` LSH key for band `band`, or `None` past the end of the sketch.
    /// Equal keys mean `BAND` consecutive min-hashes agree.
    pub fn band(&self, band: usize) -> Option<u64> {
        let start = band * BAND;
        let end = start + BAND;
        if end > self.len as usize {
            return None;
        }
        let mut key = 0u64;
        for &v in &self.values[start..end] {
            key = (key << 16) | u64::from(v);
        }
        Some(key)
    }
}

/// Bottom-`S` sketch of an ascending, distinct word list (as produced by
/// [`crate::sequence::unique_words`]).
pub fn sketch<const S: usize>(words: &[u16]) -> Sketch<S> {
    let mut out = Sketch::<S>::default();
    let mut len = 0usize;
    for &word in words {
        let value = permute(word);
        if len == S && value >= out.values[S - 1] {
            continue;
        }
        let mut i = len.min(S - 1);
        while i > 0 && out.values[i - 1] > value {
            out.values[i] = out.values[i - 1];
            i -= 1;
        }
        out.values[i] = value;
        len = (len + 1).min(S);
    }
    out.len = len as u16;
    out
}

/// Bottom-k Jaccard estimate: the fraction of the `S` smallest values of the
/// union that appear in both sketches.
pub fn jaccard<const S: usize>(a: &Sketch<S>, b: &Sketch<S>) -> f32 {
    let (x, y) = (a.values(), b.values());
    let (mut i, mut j) = (0usize, 0usize);
    let (mut seen, mut shared) = (0usize, 0usize);
    while seen < S && (i < x.len() || j < y.len()) {
        let take = match (x.get(i), y.get(j)) {
            (Some(&u), Some(&v)) => u.min(v),
            (Some(&u), None) => u,
            (None, Some(&v)) => v,
            (None, None) => break,
        };
        let in_a = x.get(i) == Some(&take);
        let in_b = y.get(j) == Some(&take);
        i += usize::from(in_a);
        j += usize::from(in_b);
        shared += usize::from(in_a && in_b);
        seen += 1;
    }
    if seen == 0 {
        return 0.0;
    }
    shared as f32 / seen as f32
}
