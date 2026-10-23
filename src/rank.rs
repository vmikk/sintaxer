//! Rank pass: one sweep over the whole query vocabulary per query strand.
//!
//! Each reference's match count against the full query vocabulary is computed
//! once, and bootstrap replicates then run only against a small candidate set
//! from the top of that ranking. The pruning risk is bounded; see [`Bound`].

use crate::{
    K, SAMPLE_SIZE, WORDS,
    index::{Index, Layout, Row},
    scoring::Top,
    weight,
};
use anyhow::{Result, ensure};

/// Sized so the tile loop's working set (counter planes, carry-save accumulator,
/// and the tile's slice of `counts`) stays in L1. The index's blocked rows are
/// cut on the same boundary.
pub use crate::TILE;
const LANES: usize = TILE / 64;
/// A match count can't exceed the query's vocabulary size, which is capped
/// just below the word universe, so 17 planes is enough.
const PLANE_CAP: usize = 17;

/// Marks a word absent from the query in the reverse lookup table.
const ABSENT: u16 = u16::MAX;

/// Add `2^shift` into bit-sliced counters wherever `input` has a bit set.
///
/// Stops as soon as every lane's carry is zero, usually after one or two planes.
/// Weights are powers of two, so a weight is applied by starting `shift` planes up.
fn ripple_add(planes: &mut [u64], input: &[u64], depth: usize, carry: &mut Vec<u64>, shift: usize) {
    let lanes = input.len();
    carry.clear();
    carry.extend_from_slice(input);
    for plane in shift..depth {
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

/// [`ripple_add`] without the per-lane empty check, for dense inputs.
///
/// Dropping the branch lets the loop vectorise. The tile-wide early exit stays.
fn ripple_add_dense(
    planes: &mut [u64],
    input: &[u64],
    depth: usize,
    carry: &mut Vec<u64>,
    shift: usize,
) {
    debug_assert!(shift < depth);
    let lanes = input.len();
    carry.clear();
    carry.extend_from_slice(input);
    for plane in shift..depth {
        let mut any = 0;
        for (slot, c) in planes[plane * lanes..][..lanes]
            .iter_mut()
            .zip(&mut carry[..lanes])
        {
            let next = *slot & *c;
            *slot ^= *c;
            *c = next;
            any |= next;
        }
        if any == 0 {
            return;
        }
    }
}

/// Number of weight levels in the carry-save accumulator.
///
/// Level `L` fills only after `2^L` rows, so `PLANE_CAP` levels are enough.
const LEVELS: usize = PLANE_CAP;

/// Fold one dense row's tile bitmap into the carry-save accumulator.
///
/// The accumulator is a binary counter whose digits are bitmaps: level `L` holds
/// up to two bitmaps of weight `2^L`. A third one triggers a 3:2 compression,
///
/// ```text
/// sum   = a ^ b ^ c
/// carry = (a & b) | ((a ^ b) & c)
/// ```
///
/// keeping `sum` at this level and promoting `carry` one level up. Counts stay
/// per reference (positional popcount); [`Workspace::rank`] ripples the
/// survivors into the planes at `shift = level`.
///
/// Only called when level zero is full; otherwise the caller decodes the row
/// straight into the free slot.
fn csa_fold(
    acc: &mut [u64],
    occupied: &mut [u8],
    carries: &mut [u64],
    lanes: usize,
    input: &[u64],
) {
    debug_assert_eq!(occupied[0], 2);
    let mut level = 0;
    loop {
        debug_assert!(level + 1 < LEVELS);
        let up = level + 1;
        // Level zero takes the row itself; higher levels take the carry from below.
        let (below, here) = carries.split_at_mut(level * LANES);
        let c: &[u64] = if level == 0 {
            input
        } else {
            &below[(level - 1) * LANES..][..lanes]
        };
        // The carry has twice this level's weight, so it moves up one level; write it
        // straight into a free slot there to avoid a copy.
        let (mine, above) = acc.split_at_mut(up * 2 * LANES);
        let free = occupied[up] < 2;
        let out: &mut [u64] = if free {
            &mut above[occupied[up] as usize * LANES..][..lanes]
        } else {
            &mut here[..lanes]
        };
        let (sum, other) = mine[level * 2 * LANES..].split_at_mut(LANES);
        for (((sum, &b), &c), out) in sum[..lanes]
            .iter_mut()
            .zip(&other[..lanes])
            .zip(c)
            .zip(out.iter_mut())
        {
            let ab = *sum ^ b;
            *out = (*sum & b) | (ab & c);
            *sum = ab ^ c;
        }
        occupied[level] = 1;
        if free {
            occupied[up] += 1;
            return;
        }
        level = up;
    }
}

/// Transpose an 8x8 bit matrix held in a `u64`: byte `p`, bit `j` becomes byte
/// `j`, bit `p`, counting bytes from the least significant.
///
/// Three delta swaps (Hacker's Delight 7-3), branch-free.
fn transpose8(mut x: u64) -> u64 {
    let mut t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x
}

/// Write one lane's 64 finished counts, reassembled from the counter planes.
///
/// Byte `g` of each of the low eight planes forms an 8x8 bit matrix (rows are
/// planes, columns are references `8g..8g+8`); transposing it yields each
/// reference's count as a byte. `out` must be exactly 64 wide; planes at or
/// above eight are handled by the caller.
fn lane_counts(planes: &[u64], lanes: usize, lane: usize, depth: usize, out: &mut [u16]) {
    debug_assert_eq!(out.len(), 64);
    let mut rows = [[0u8; 8]; 8];
    for (plane, row) in rows.iter_mut().enumerate().take(depth) {
        *row = planes[plane * lanes + lane].to_le_bytes();
    }
    for (g, group) in out.chunks_exact_mut(8).enumerate() {
        let matrix = u64::from_le_bytes(std::array::from_fn(|plane| rows[plane][g]));
        for (slot, count) in group.iter_mut().zip(transpose8(matrix).to_le_bytes()) {
            *slot = u16::from(count);
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
///
/// A few small allocations remain: [`Workspace::rank`] builds a `Vec<Row>` per
/// call, and each [`plane_max`] call returns a `Vec` owned by [`Top`].
pub struct Workspace {
    planes: Vec<u64>,
    tile: Vec<u64>,
    /// Carry-save accumulator: two bitmap slots per weight level.
    acc: Vec<u64>,
    /// How many of each level's two slots are in use.
    occupied: Vec<u8>,
    /// One carry bitmap per level, produced by [`csa_fold`].
    carries: Vec<u64>,
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
    /// `log2(weight)` per query-vocabulary position, parallel to the words given to `project`.
    shifts: Vec<u8>,
    /// Counter planes needed by the heaviest possible replicate.
    depth: usize,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            planes: vec![0; PLANE_CAP * LANES],
            tile: vec![0; LANES],
            acc: vec![0; LEVELS * 2 * LANES],
            occupied: vec![0; LEVELS],
            carries: vec![0; LEVELS * LANES],
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
            shifts: Vec::new(),
            depth: depth_for(SAMPLE_SIZE),
        }
    }
}

fn depth_for(vocabulary: usize) -> usize {
    (usize::BITS - vocabulary.leading_zeros()) as usize
}

impl Workspace {
    /// Exact `|W(query) intersect W(r)|` for every reference `r`.
    ///
    /// The sweep is tiled so the counter planes stay in L1. Dense (bitmap) rows go
    /// through the bit-sliced adder; sparse (posting list) rows increment `counts`
    /// directly. Each tile's counts are written and histogrammed while still hot.
    pub fn rank(&mut self, index: &Index, words: &[u16]) -> Result<()> {
        ensure!(!words.is_empty(), "empty query vocabulary");
        ensure!(words.len() < ABSENT as usize, "query vocabulary too large");
        let n = index.references;

        // Group rows by storage so each kernel gets one contiguous slice; order within
        // a group doesn't matter.
        let mut rows: Vec<Row<'_>> = words
            .iter()
            .map(|&word| index.row(word))
            .collect::<Result<_>>()?;
        rows.sort_unstable_by_key(|row| match row.layout {
            Layout::Bitmap => 0u8,
            Layout::Blocked => 1,
            Layout::Postings => 2,
        });
        let group = |kind| rows.iter().filter(|row| row.layout == kind).count();
        let (dense, rest) = rows.split_at(group(Layout::Bitmap));
        let (blocked, plain) = rest.split_at(group(Layout::Blocked));

        let depth = depth_for(dense.len());
        debug_assert!(depth <= PLANE_CAP);
        // Each tile clears its own window, so only zero the whole array on resize.
        if self.counts.len() != n {
            self.counts.clear();
            self.counts.resize(n, 0);
        }
        self.cursors.clear();
        self.cursors.resize(blocked.len() + plain.len(), 0);
        self.histogram.clear();
        self.histogram.resize(words.len() + 1, 0);

        let Self {
            planes,
            tile,
            acc,
            occupied,
            carries,
            cursors,
            counts,
            histogram,
            carry,
            ..
        } = self;

        let mut start = 0;
        while start < n {
            let width = (n - start).min(TILE);
            let lanes = width.div_ceil(64);
            let counts = &mut counts[start..start + width];

            if depth > 0 {
                let planes = &mut planes[..depth * lanes];
                planes.fill(0);
                occupied.fill(0);
                for row in dense {
                    if occupied[0] < 2 {
                        // Room at level zero: decode straight into the accumulator slot.
                        let slot = occupied[0] as usize * LANES;
                        row.dense_tile(start, &mut acc[slot..slot + lanes]);
                        occupied[0] += 1;
                    } else {
                        row.dense_tile(start, &mut tile[..lanes]);
                        csa_fold(acc, occupied, carries, lanes, &tile[..lanes]);
                    }
                }
                // Only the survivors (at most two bitmaps per level) reach the planes.
                for (level, &slots) in occupied.iter().enumerate() {
                    debug_assert!(slots == 0 || level < depth);
                    for slot in 0..slots as usize {
                        let at = (level * 2 + slot) * LANES;
                        ripple_add_dense(planes, &acc[at..at + lanes], depth, carry, level);
                    }
                }
                // Every count in the window is written here, so it isn't cleared first.
                for lane in 0..lanes {
                    let base = lane * 64;
                    if base + 64 <= width {
                        lane_counts(planes, lanes, lane, depth, &mut counts[base..base + 64]);
                    } else {
                        let mut tail = [0u16; 64];
                        lane_counts(planes, lanes, lane, depth, &mut tail);
                        counts[base..width].copy_from_slice(&tail[..width - base]);
                    }
                }
                // The byte transpose covers only eight planes; counts above 255 need the
                // higher planes added bit by bit. Still exact, since the transpose wrote
                // those bits as zero.
                if depth > 8 {
                    for lane in 0..lanes {
                        let base = lane * 64;
                        let limit = width - base;
                        for plane in 8..depth {
                            let mut bits = planes[plane * lanes + lane];
                            if limit < 64 {
                                bits &= (1u64 << limit) - 1;
                            }
                            while bits != 0 {
                                let bit = bits.trailing_zeros() as usize;
                                counts[base + bit] |= 1u16 << plane;
                                bits &= bits - 1;
                            }
                        }
                    }
                }
            } else {
                // An all-sparse vocabulary writes nothing above, so zero the window first.
                counts.fill(0);
            }

            // Blocked rows are read in tile order, so their cursors are byte offsets that
            // only move forward.
            let (blocked_cursors, plain_cursors) = cursors.split_at_mut(blocked.len());
            for (row, cursor) in blocked.iter().zip(blocked_cursors.iter_mut()) {
                row.blocked_tile(counts, cursor);
            }
            for (row, cursor) in plain.iter().zip(plain_cursors.iter_mut()) {
                row.tile_increment(start, counts, cursor);
            }

            // Histogram the tile while it is still in L1.
            for &count in counts.iter() {
                histogram[count as usize] += 1;
            }
            start += TILE;
        }
        Ok(())
    }

    /// Erase references the caller rejects, as if they were never indexed.
    ///
    /// Used in training to hide the query's own sequence, genus or family from the
    /// database without rebuilding it. A zeroed count can never be selected, so
    /// that is enough; the histogram is rebuilt for `select` and `Bound`.
    /// The predicate gets the count as well as the id to avoid borrowing `self`.
    pub fn exclude(&mut self, mut rejected: impl FnMut(usize, u16) -> bool) -> usize {
        let mut erased = 0;
        for (reference, count) in self.counts.iter_mut().enumerate() {
            if *count > 0 && rejected(reference, *count) {
                self.histogram[*count as usize] -= 1;
                self.histogram[0] += 1;
                *count = 0;
                erased += 1;
            }
        }
        erased
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

    /// Build the word-major membership matrix for the selected candidates.
    ///
    /// Each row is a bitmap over candidates that can be added straight into
    /// the counter planes during the bootstrap.
    pub fn project(&mut self, index: &Index, words: &[u16], weights: &weight::Table) -> Result<()> {
        for &word in &self.touched {
            self.lookup[word as usize] = ABSENT;
            self.present[word as usize / 64] &= !(1 << (word % 64));
        }
        self.touched.clear();
        ensure!(words.len() < ABSENT as usize, "query vocabulary too large");
        self.shifts.clear();
        self.shifts.reserve(words.len());
        for (i, &word) in words.iter().enumerate() {
            self.lookup[word as usize] = i as u16;
            self.present[word as usize / 64] |= 1 << (word % 64);
            self.touched.push(word);
            self.shifts.push(weights.get(word).trailing_zeros() as u8);
        }
        self.depth = depth_for(SAMPLE_SIZE * weights.max() as usize);
        debug_assert!(self.depth <= PLANE_CAP);

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
    /// Scores are at most `SAMPLE_SIZE * max weight`. Candidate positions are
    /// in reference order, so tie draws match the exact path.
    pub fn replicate(&mut self, sample: &[u16; SAMPLE_SIZE]) -> Top {
        let lanes = self.candidates.len().div_ceil(64).max(1);
        let depth = self.depth;
        self.candidate_planes.clear();
        self.candidate_planes.resize(depth * lanes, 0);
        for &position in sample {
            let row = &self.matrix[position as usize * lanes..][..lanes];
            let shift = self.shifts[position as usize] as usize;
            ripple_add(
                &mut self.candidate_planes,
                row,
                depth,
                &mut self.carry,
                shift,
            );
        }
        self.valid.clear();
        self.valid.resize(lanes, u64::MAX);
        let used = self.candidates.len();
        if used == 0 {
            self.valid[lanes - 1] = 0;
        } else if !used.is_multiple_of(64) {
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
        Self { tail }
    }

    /// Expected number of pruned references that would have reached `score`.
    pub fn risk(&self) -> f64 {
        self.tail.iter().sum()
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
