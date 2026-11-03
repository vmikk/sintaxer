//! Bit-sliced counting kernels for the bootstrap replicates.
//!
//! One bit per reference, 64 references per `u64` lane. A replicate's score
//! is the positional popcount of its 32 drawn rows, held as *planes*: plane
//! `p` holds bit `p` of every reference's score. The rows go through a fixed
//! Harley–Seal carry-save network that yields the planes directly.
//!
//! Lanes are processed four at a time as `[u64; 4]`; plain array arithmetic
//! is enough for LLVM to emit AVX2, so there is no `unsafe` here.
use crate::SAMPLE_SIZE;

/// Lanes processed together. Four `u64` make one AVX2 register.
pub const CHUNK: usize = 4;
/// Four lanes: 256 references, one bit each.
pub type Chunk = [u64; CHUNK];
const ZERO: Chunk = [0; CHUNK];
/// Planes an unweighted score needs: a count of at most 32 is six bits.
pub const PLANES: usize = 6;
/// Planes the widest weighted score needs: `SAMPLE_SIZE * 4` is 128.
pub const WEIGHTED_PLANES: usize = 8;

/// Lanes handed to the network at a time: 4,096 references, one index tile.
/// Fixed-size blocks let the optimiser drop the bounds checks.
pub const BLOCK_LANES: usize = 64;
const BLOCK_CHUNKS: usize = BLOCK_LANES / CHUNK;

/// `n` lanes rounded up to whole blocks: the stored width of every row the
/// kernels read. Padding lanes must be zero, or at least never eligible.
pub fn stride(lanes: usize) -> usize {
    lanes.max(1).div_ceil(BLOCK_LANES) * BLOCK_LANES
}

#[inline(always)]
fn xor(a: Chunk, b: Chunk) -> Chunk {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}
#[inline(always)]
fn and(a: Chunk, b: Chunk) -> Chunk {
    [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
}
#[inline(always)]
fn or(a: Chunk, b: Chunk) -> Chunk {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}

/// Full adder over bitmaps: `a + b + c` per bit as `(sum, carry)`.
#[inline(always)]
fn csa(a: Chunk, b: Chunk, c: Chunk) -> (Chunk, Chunk) {
    let u = xor(a, b);
    (xor(u, c), or(and(a, b), and(u, c)))
}

/// Running low digits of a Harley–Seal count: weights 1, 2, 4 and 8.
#[derive(Clone, Copy)]
struct Digits {
    ones: Chunk,
    twos: Chunk,
    fours: Chunk,
    eights: Chunk,
}

impl Digits {
    const ZERO: Self = Self {
        ones: ZERO,
        twos: ZERO,
        fours: ZERO,
        eights: ZERO,
    };

    /// Fold sixteen inputs in and return the sixteens they produced.
    ///
    /// Fifteen full adders per sixteen inputs. Every bit position is a
    /// reference, so the digits themselves are the per-reference counts.
    /// Inputs are fetched through `x` as consumed, to keep few chunks live.
    #[inline(always)]
    fn fold16(&mut self, x: impl Fn(usize) -> Chunk) -> Chunk {
        let (ones, twos_a) = csa(self.ones, x(0), x(1));
        let (ones, twos_b) = csa(ones, x(2), x(3));
        let (twos, fours_a) = csa(self.twos, twos_a, twos_b);
        let (ones, twos_a) = csa(ones, x(4), x(5));
        let (ones, twos_b) = csa(ones, x(6), x(7));
        let (twos, fours_b) = csa(twos, twos_a, twos_b);
        let (fours, eights_a) = csa(self.fours, fours_a, fours_b);
        let (ones, twos_a) = csa(ones, x(8), x(9));
        let (ones, twos_b) = csa(ones, x(10), x(11));
        let (twos, fours_a) = csa(twos, twos_a, twos_b);
        let (ones, twos_a) = csa(ones, x(12), x(13));
        let (ones, twos_b) = csa(ones, x(14), x(15));
        let (twos, fours_b) = csa(twos, twos_a, twos_b);
        let (fours, eights_b) = csa(fours, fours_a, fours_b);
        let (eights, sixteens) = csa(self.eights, eights_a, eights_b);
        *self = Self {
            ones,
            twos,
            fours,
            eights,
        };
        sixteens
    }
}

/// Six planes of the count of 32 inputs, fetched through `x`.
#[inline(always)]
fn count32_chunk(x: impl Fn(usize) -> Chunk) -> [Chunk; PLANES] {
    let mut digits = Digits::ZERO;
    let first = digits.fold16(&x);
    let second = digits.fold16(|i| x(16 + i));
    // Two sixteens make a thirty-two: a half adder on the last digit.
    [
        digits.ones,
        digits.twos,
        digits.fours,
        digits.eights,
        xor(first, second),
        and(first, second),
    ]
}

/// Six planes of the count of the first `n` of up to 32 inputs; inputs past
/// `n` read as zero.
#[inline(always)]
fn count_padded_chunk(x: impl Fn(usize) -> Chunk, n: usize) -> [Chunk; PLANES] {
    let mut digits = Digits::ZERO;
    let mut sixteens = [ZERO; 2];
    for (block, sixteen) in sixteens.iter_mut().enumerate().take(n.div_ceil(16)) {
        let from = block * 16;
        *sixteen = digits.fold16(|i| if from + i < n { x(from + i) } else { ZERO });
    }
    [
        digits.ones,
        digits.twos,
        digits.fours,
        digits.eights,
        xor(sixteens[0], sixteens[1]),
        and(sixteens[0], sixteens[1]),
    ]
}

/// Add `x << shift` into eight bit-sliced planes.
///
/// The index loop is deliberate: clippy's iterator form compiles to
/// noticeably slower code here.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
fn add_shifted(total: &mut [Chunk; WEIGHTED_PLANES], x: &[Chunk; PLANES], shift: usize) {
    let mut carry = ZERO;
    for plane in shift..WEIGHTED_PLANES {
        let digit = x.get(plane - shift).copied().unwrap_or(ZERO);
        let (sum, next) = csa(total[plane], digit, carry);
        total[plane] = sum;
        carry = next;
    }
    // Scores are at most `SAMPLE_SIZE * MAX_WEIGHT` = 128, so nothing carries out.
    debug_assert_eq!(carry, ZERO);
}

/// Block `block` of the row at `offset`, as a fixed-size array of chunks.
#[inline(always)]
fn block_of(matrix: &[u64], offset: usize, block: usize) -> &[Chunk; BLOCK_CHUNKS] {
    let at = offset + block * BLOCK_LANES;
    matrix[at..at + BLOCK_LANES]
        .as_chunks::<CHUNK>()
        .0
        .try_into()
        .expect("a block is BLOCK_CHUNKS chunks")
}

/// Block `block` of each of `DEPTH` planes, as fixed-size arrays.
#[inline(always)]
fn plane_blocks<const DEPTH: usize>(
    planes: &mut [u64],
    stride: usize,
    block: usize,
) -> [&mut [Chunk; BLOCK_CHUNKS]; DEPTH] {
    let mut each = planes[..DEPTH * stride].chunks_exact_mut(stride);
    std::array::from_fn(|_| {
        let plane = each.next().expect("DEPTH planes were sliced");
        let at = block * BLOCK_LANES;
        plane[at..at + BLOCK_LANES]
            .as_chunks_mut::<CHUNK>()
            .0
            .try_into()
            .expect("a block is BLOCK_CHUNKS chunks")
    })
}

/// Chunks of block `block` that hold any of the first `lanes` lanes.
fn chunks_in(lanes: usize, block: usize) -> usize {
    (lanes - block * BLOCK_LANES)
        .min(BLOCK_LANES)
        .div_ceil(CHUNK)
}

/// Unweighted replicate counts: plane `p` of the count of the 32 rows at
/// `offsets` into `planes[p * stride ..]`, for the first `lanes` lanes.
///
/// `stride` must be [`stride`]`(lanes)`. A row drawn twice counts twice,
/// as sampling with replacement requires.
pub fn count32(
    matrix: &[u64],
    offsets: &[usize; SAMPLE_SIZE],
    lanes: usize,
    stride: usize,
    planes: &mut [u64],
) {
    debug_assert_eq!(stride, self::stride(lanes));
    for block in 0..lanes.div_ceil(BLOCK_LANES) {
        let rows: [&[Chunk; BLOCK_CHUNKS]; SAMPLE_SIZE] =
            std::array::from_fn(|i| block_of(matrix, offsets[i], block));
        let mut out = plane_blocks::<PLANES>(planes, stride, block);
        let chunks = chunks_in(lanes, block).min(BLOCK_CHUNKS);
        for j in 0..chunks {
            for (plane, digit) in out.iter_mut().zip(count32_chunk(|i| rows[i][j])) {
                plane[j] = digit;
            }
        }
    }
}

/// Weighted replicate counts: like [`count32`], but draw `i` weighs
/// `1 << shifts[i]`, and eight planes are written.
///
/// Draws are grouped by weight tier, each tier counted with the same network,
/// and the tier counts added at their shifts.
pub fn count_weighted(
    matrix: &[u64],
    offsets: &[usize; SAMPLE_SIZE],
    shifts: &[u8; SAMPLE_SIZE],
    lanes: usize,
    stride: usize,
    planes: &mut [u64],
) {
    debug_assert_eq!(stride, self::stride(lanes));
    // Weights are powers of two up to `MAX_WEIGHT`: at most three shifts.
    const TIERS: usize = 3;
    let mut groups = [[0usize; SAMPLE_SIZE]; TIERS];
    let mut lens = [0usize; TIERS];
    for (&offset, &shift) in offsets.iter().zip(shifts) {
        let tier = shift as usize;
        debug_assert!(tier < TIERS, "weight tier {tier} out of range");
        groups[tier][lens[tier]] = offset;
        lens[tier] += 1;
    }
    for (group, &len) in groups.iter_mut().zip(&lens) {
        // Unused slots just need to be in bounds; they are never read.
        let first = group[0];
        group[len..].fill(first);
    }
    for block in 0..lanes.div_ceil(BLOCK_LANES) {
        let rows: [[&[Chunk; BLOCK_CHUNKS]; SAMPLE_SIZE]; TIERS] = std::array::from_fn(|tier| {
            std::array::from_fn(|i| block_of(matrix, groups[tier][i], block))
        });
        let mut out = plane_blocks::<WEIGHTED_PLANES>(planes, stride, block);
        let chunks = chunks_in(lanes, block).min(BLOCK_CHUNKS);
        for j in 0..chunks {
            let mut total = [ZERO; WEIGHTED_PLANES];
            for (tier, group) in rows.iter().enumerate() {
                if lens[tier] > 0 {
                    let counted = count_padded_chunk(|i| group[i][j], lens[tier]);
                    add_shifted(&mut total, &counted, tier);
                }
            }
            for (plane, digit) in out.iter_mut().zip(total) {
                plane[j] = digit;
            }
        }
    }
}

/// Largest count among the `valid` positions of `lanes` lanes; the positions
/// holding it are written into `out`.
///
/// Descends from the top plane, keeping positions that have each bit set.
/// With no valid position the answer is zero with an empty set.
pub fn masked_max(
    planes: &[u64],
    stride: usize,
    depth: usize,
    lanes: usize,
    valid: &[u64],
    out: &mut [u64],
) -> u32 {
    let out = &mut out[..lanes];
    out.copy_from_slice(&valid[..lanes]);
    let mut max = 0u32;
    for plane in (0..depth).rev() {
        let digit = &planes[plane * stride..][..lanes];
        if out.iter().zip(digit).any(|(&a, &b)| a & b != 0) {
            for (a, &b) in out.iter_mut().zip(digit) {
                *a &= b;
            }
            max |= 1 << plane;
        }
    }
    max
}

/// `out = valid & row_0 & row_1 & ...` over `lanes` lanes; whether it is
/// nonempty.
///
/// A nonempty result is the replicate's whole tie set at its highest possible
/// score. Stops as soon as the intersection is empty, so pass the rarest rows
/// first.
pub fn intersect(
    matrix: &[u64],
    offsets: &[usize],
    lanes: usize,
    valid: &[u64],
    out: &mut [u64],
) -> bool {
    let out = &mut out[..lanes];
    out.copy_from_slice(&valid[..lanes]);
    for &offset in offsets {
        let row = &matrix[offset..offset + lanes];
        let mut any = 0;
        for (a, &b) in out.iter_mut().zip(row) {
            *a &= b;
            any |= *a;
        }
        if any == 0 {
            return false;
        }
    }
    true
}
