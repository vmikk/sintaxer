//! Rank pass: one sweep over the whole query vocabulary per query strand.
//!
//! Each reference's match count against the full query vocabulary is computed
//! once, and bootstrap replicates then run only against a small candidate set
//! from the top of that ranking. The pruning risk is bounded; see [`Bound`].

use crate::{
    K, SAMPLE_SIZE, WORDS,
    index::{Index, Row},
    scoring::Top,
};
use anyhow::{Result, ensure};

/// References per tile, sized so the counter planes stay in L1.
pub const TILE: usize = 4096;
const LANES: usize = TILE / 64;
/// A match count can't exceed the query's vocabulary size; 16 planes is enough.
const PLANE_CAP: usize = 16;

/// Marks a word absent from the query in the reverse lookup table.
const ABSENT: u16 = u16::MAX;

/// Add one membership bitmap into bit-sliced counters (carry-save).
///
/// Stops as soon as every lane's carry is zero, usually after one or two planes.
fn ripple_add(planes: &mut [u64], input: &[u64], depth: usize, carry: &mut Vec<u64>) {
    let lanes = input.len();
    carry.clear();
    carry.extend_from_slice(input);
    for plane in 0..depth {
        let mut any = 0;
        let base = plane * lanes;
        for lane in 0..lanes {
            let c = carry[lane];
            if c == 0 {
                continue;
            }
            let slot = &mut planes[base + lane];
            carry[lane] = *slot & c;
            *slot ^= c;
            any |= carry[lane];
        }
        if any == 0 {
            return;
        }
    }
}

/// Largest value and the full set of positions holding it.
///
/// Walks planes from the most significant bit down, keeping only positions
/// that have the bit set whenever any do.
fn plane_max(planes: &[u64], lanes: usize, depth: usize, valid: &[u64]) -> (u32, Vec<u64>) {
    let mut candidates = valid.to_vec();
    let mut max = 0u32;
    for plane in (0..depth).rev() {
        let base = plane * lanes;
        let mut any = false;
        for lane in 0..lanes {
            if candidates[lane] & planes[base + lane] != 0 {
                any = true;
                break;
            }
        }
        if any {
            for lane in 0..lanes {
                candidates[lane] &= planes[base + lane];
            }
            max |= 1 << plane;
        }
    }
    (max, candidates)
}

/// Per-worker scratch buffers, reused across queries.
pub struct Workspace {
    planes: Vec<u64>,
    tile: Vec<u64>,
    cursors: Vec<usize>,
    lookup: Box<[u16; WORDS]>,
    /// Small presence bitmap checked before `lookup`.
    ///
    /// Most probes miss, so this keeps the common case in L1.
    present: Box<[u64; WORDS / 64]>,
    touched: Vec<u16>,
    matrix: Vec<u64>,
    candidate_planes: Vec<u64>,
    carry: Vec<u64>,
    valid: Vec<u64>,
    /// Exact match count against the whole query vocabulary, per reference.
    pub counts: Vec<u16>,
    /// `histogram[v]` is the number of references whose count is `v`.
    pub histogram: Vec<u32>,
    /// References selected for the bootstrap phase, ascending.
    pub candidates: Vec<u32>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            planes: vec![0; PLANE_CAP * LANES],
            tile: vec![0; LANES],
            cursors: Vec::new(),
            lookup: Box::new([ABSENT; WORDS]),
            present: Box::new([0; WORDS / 64]),
            touched: Vec::new(),
            matrix: Vec::new(),
            candidate_planes: Vec::new(),
            carry: Vec::new(),
            valid: Vec::new(),
            counts: Vec::new(),
            histogram: Vec::new(),
            candidates: Vec::new(),
        }
    }
}

fn depth_for(vocabulary: usize) -> usize {
    (usize::BITS - vocabulary.leading_zeros()) as usize
}

impl Workspace {
    /// Exact `|W(query) intersect W(r)|` for every reference `r`.
    pub fn rank(&mut self, index: &Index, words: &[u16]) -> Result<()> {
        ensure!(!words.is_empty(), "empty query vocabulary");
        let n = index.references;
        let depth = depth_for(words.len()).min(PLANE_CAP);
        self.counts.clear();
        self.counts.resize(n, 0);
        self.cursors.clear();
        self.cursors.resize(words.len(), 0);

        let rows: Vec<Row<'_>> = words
            .iter()
            .map(|&word| index.row(word))
            .collect::<Result<_>>()?;

        let mut start = 0;
        while start < n {
            let lanes = (n - start).div_ceil(64).min(LANES);
            self.planes[..depth * lanes].fill(0);
            for (i, row) in rows.iter().enumerate() {
                row.tile_bits(start, &mut self.tile[..lanes], &mut self.cursors[i]);
                ripple_add(
                    &mut self.planes[..depth * lanes],
                    &self.tile[..lanes],
                    depth,
                    &mut self.carry,
                );
            }
            for lane in 0..lanes {
                // Skip lanes of 64 references that share no word with the query.
                if (0..depth).all(|p| self.planes[p * lanes + lane] == 0) {
                    continue;
                }
                for bit in 0..64 {
                    let reference = start + lane * 64 + bit;
                    if reference >= n {
                        break;
                    }
                    let mut value = 0u16;
                    for plane in 0..depth {
                        value |= (((self.planes[plane * lanes + lane] >> bit) & 1) as u16) << plane;
                    }
                    self.counts[reference] = value;
                }
            }
            start += TILE;
        }

        self.histogram.clear();
        self.histogram.resize(words.len() + 1, 0);
        for &count in &self.counts {
            self.histogram[count as usize] += 1;
        }
        Ok(())
    }

    /// Smallest threshold whose `{r : count(r) >= threshold}` holds at least
    /// `target` references, and that set in ascending order.
    ///
    /// Ties at the threshold are never cut, so no tied reference is favoured.
    pub fn select(&mut self, target: usize) -> u16 {
        let mut threshold = self.histogram.len() - 1;
        let mut total = 0usize;
        while threshold > 0 {
            total += self.histogram[threshold] as usize;
            if total >= target {
                break;
            }
            threshold -= 1;
        }
        // Count 0 means no shared words; such a reference can never win.
        let threshold = threshold.max(1) as u16;
        self.candidates.clear();
        for (reference, &count) in self.counts.iter().enumerate() {
            if count >= threshold {
                self.candidates.push(reference as u32);
            }
        }
        threshold
    }

    /// Every reference with a nonzero count, ascending. Used by `--exact` and
    /// when the bound rejects a smaller set.
    pub fn select_all(&mut self) -> u16 {
        self.select(usize::MAX)
    }

    /// Build the word-major membership matrix for the selected candidates.
    ///
    /// Each row is a bitmap over candidates that can be added straight into
    /// the counter planes during the bootstrap.
    pub fn project(&mut self, index: &Index, words: &[u16]) -> Result<()> {
        for &word in &self.touched {
            self.lookup[word as usize] = ABSENT;
            self.present[word as usize / 64] &= !(1 << (word % 64));
        }
        self.touched.clear();
        ensure!(words.len() < ABSENT as usize, "query vocabulary too large");
        for (i, &word) in words.iter().enumerate() {
            self.lookup[word as usize] = i as u16;
            self.present[word as usize / 64] |= 1 << (word % 64);
            self.touched.push(word);
        }

        let lanes = self.candidates.len().div_ceil(64).max(1);
        self.matrix.clear();
        self.matrix.resize(words.len() * lanes, 0);
        for (position, &reference) in self.candidates.iter().enumerate() {
            let (lane, bit) = (position / 64, position % 64);
            let (present, lookup, matrix) = (&self.present, &self.lookup, &mut self.matrix);
            index.visit_kmers(reference as usize, |word| {
                if present[word as usize / 64] >> (word % 64) & 1 == 0 {
                    return;
                }
                let index = lookup[word as usize];
                matrix[index as usize * lanes + lane] |= 1 << bit;
            })?;
        }
        Ok(())
    }

    /// Exact maximum and complete tie set for one replicate, over candidates.
    ///
    /// Scores are at most [`SAMPLE_SIZE`], so six planes suffice. Candidate
    /// positions are in reference order, so tie draws match the exact path.
    pub fn replicate(&mut self, sample: &[u16; SAMPLE_SIZE]) -> Top {
        let lanes = self.candidates.len().div_ceil(64).max(1);
        let depth = depth_for(SAMPLE_SIZE);
        self.candidate_planes.clear();
        self.candidate_planes.resize(depth * lanes, 0);
        for &word in sample {
            let row = &self.matrix[word as usize * lanes..][..lanes];
            ripple_add(&mut self.candidate_planes, row, depth, &mut self.carry);
        }
        self.valid.clear();
        self.valid.resize(lanes, u64::MAX);
        let used = self.candidates.len();
        if used == 0 {
            self.valid[lanes - 1] = 0;
        } else if used % 64 != 0 {
            self.valid[lanes - 1] = (1u64 << (used % 64)) - 1;
        }
        let (max, bits) = plane_max(&self.candidate_planes, lanes, depth, &self.valid);
        Top::from_bits(max as u8, bits)
    }

    /// Reference id of the `offset`-th member of a replicate's tie set.
    pub fn reference_at(&self, top: &Top, offset: usize) -> u32 {
        self.candidates[top.nth(offset) as usize]
    }
}

/// Risk bound for restricting replicates to the candidate set.
///
/// A pruned reference's score is `Binomial(SAMPLE_SIZE, count(r) / vocabulary)`,
/// so the chance any of them reaches the best score is at most the sum of
/// their tail probabilities, computed from the count histogram.
pub struct Bound {
    tail: Vec<f64>,
    vocabulary: usize,
}

impl Bound {
    /// Tail probabilities for one query strand, indexed by count.
    pub fn new(histogram: &[u32], threshold: u16, vocabulary: usize, score: u8) -> Self {
        let mut tail = vec![0.0; histogram.len()];
        for (count, &references) in histogram.iter().enumerate().take(threshold as usize) {
            if references == 0 {
                continue;
            }
            tail[count] =
                references as f64 * binomial_tail(count as f64 / vocabulary as f64, score);
        }
        Self { tail, vocabulary }
    }

    /// Expected number of pruned references that would have reached `score`.
    pub fn risk(&self) -> f64 {
        self.tail.iter().sum()
    }

    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }
}

/// `P(Binomial(SAMPLE_SIZE, p) >= score)`.
fn binomial_tail(p: f64, score: u8) -> f64 {
    if score == 0 {
        return 1.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let n = SAMPLE_SIZE;
    let mut total = 0.0;
    // Sum from the top and stop once further terms are negligible.
    let mut term = p.powi(n as i32);
    for k in (score as usize..=n).rev() {
        if k < n {
            // term(k) from term(k+1): multiply by (k+1)/(n-k) * (1-p)/p
            term *= (k + 1) as f64 / (n - k) as f64 * (1.0 - p) / p;
        }
        total += term;
        if term == 0.0 {
            break;
        }
    }
    total.min(1.0)
}

/// Distinct k-mers of a normalized sequence, ascending, reusing `seen`.
pub fn vocabulary(sequence: &[u8], seen: &mut [u64; WORDS / 64], out: &mut Vec<u16>) {
    seen.fill(0);
    out.clear();
    let (mut word, mut run) = (0u16, 0usize);
    for &base in sequence {
        let code = match base {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => {
                run = 0;
                word = 0;
                continue;
            }
        };
        word = (word << 2) | code;
        run = (run + 1).min(K);
        if run == K {
            seen[word as usize / 64] |= 1 << (word % 64);
        }
    }
    for (i, bits) in seen.iter().enumerate() {
        let mut bits = *bits;
        while bits != 0 {
            out.push((i * 64 + bits.trailing_zeros() as usize) as u16);
            bits &= bits - 1;
        }
    }
}

/// Reverse complement of a word code.
///
/// Mapping the forward k-mer set gives the reverse strand's set without
/// re-reading the sequence.
pub fn revcomp_word(word: u16) -> u16 {
    let mut out = 0u16;
    for i in 0..K {
        let base = (word >> (2 * i)) & 3;
        out = (out << 2) | (3 - base);
    }
    out
}

pub fn revcomp_vocabulary(words: &[u16], seen: &mut [u64; WORDS / 64], out: &mut Vec<u16>) {
    seen.fill(0);
    out.clear();
    for &word in words {
        let flipped = revcomp_word(word);
        seen[flipped as usize / 64] |= 1 << (flipped % 64);
    }
    for (i, bits) in seen.iter().enumerate() {
        let mut bits = *bits;
        while bits != 0 {
            out.push((i * 64 + bits.trailing_zeros() as usize) as u16);
            bits &= bits - 1;
        }
    }
}
