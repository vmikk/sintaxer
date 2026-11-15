//! Bit-sliced counting kernels for the bootstrap replicates.
//!
//! A replicate scores a reference by how many of its [`SAMPLE_SIZE`] draws the
//! reference contains, each weighed by its word's tier. Held bit-sliced -- one
//! bit per reference, 64 references per `u64` lane -- that is the positional
//! population count of 32 membership bitmaps, and the result is a handful of
//! *planes*: plane `p` holds bit `p` of every reference's score.
//!
//! The replicates used to build those planes one draw at a time, rippling each
//! bitmap through planes kept in memory: about six plane passes of four
//! operations per lane for every one of the 32 draws. Here the 32 bitmaps go
//! through a fixed Harley–Seal carry-save network instead. Thirty full adders
//! and one half adder, five boolean operations each, turn them into the six
//! planes directly, with no pass through the planes in between. Measured over
//! a 64-lane tile it is 3.3 times faster than the ripple it replaced (37 ns
//! against 123 ns per four lanes). It is not free of memory traffic: the
//! network needs more live values than AVX2 has registers, so LLVM spills
//! some, and every input's row address is reloaded (docs/Optimization_round_5.md
//! §2.1 has the variants that were slower).
//!
//! Lanes are processed four at a time as `[u64; 4]` values. Plain array
//! arithmetic, written out element by element, is all LLVM needs to emit
//! 256-bit vector code for the crate's x86-64-v3 floor, so there is no
//! `unsafe` and no intrinsic here.
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

/// Lanes in one block: the widest run of lanes the network is handed at a
/// time, 4,096 references, the same as an index tile.
///
/// Rows reach the network as `&[Chunk; BLOCK_CHUNKS]`, fixed-size arrays, so
/// every index into them is checked against a constant the optimiser can
/// see, and the checks vanish. Handed plain slices instead, the network paid
/// a reload and a comparison on each of its 32 loads per chunk.
pub const BLOCK_LANES: usize = 64;
const BLOCK_CHUNKS: usize = BLOCK_LANES / CHUNK;

/// `n` lanes rounded up to whole blocks: how wide every row the kernels read
/// must be stored. Padding lanes must be zero, or at least never eligible;
/// the kernels compute only the chunks the real lanes need.
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

/// A full adder over bitmaps: per bit position, `a + b + c` as `(sum,
/// carry)`, the carry weighing twice what the inputs do.
///
/// Two forms of the same function, because which compiles better depends on
/// the instruction set; see [`Shared`] and [`Majority`].
pub trait Adder {
    fn csa(a: Chunk, b: Chunk, c: Chunk) -> (Chunk, Chunk);
}

/// `a ^ b` shared between sum and carry: five operations, the fewest AVX2
/// has. The portable network.
pub struct Shared;

/// Sum and carry as two separate three-input functions, a parity and a
/// majority, which AVX-512 computes in one `vpternlogq` each. Computed that
/// way the shared `a ^ b` of [`Shared`] is what stops LLVM forming the
/// carry's ternary op, so AVX-512 hosts get this form (docs/Optimization_round_6.md).
pub struct Majority;

impl Adder for Shared {
    #[inline(always)]
    fn csa(a: Chunk, b: Chunk, c: Chunk) -> (Chunk, Chunk) {
        let u = xor(a, b);
        (xor(u, c), or(and(a, b), and(u, c)))
    }
}

impl Adder for Majority {
    #[inline(always)]
    fn csa(a: Chunk, b: Chunk, c: Chunk) -> (Chunk, Chunk) {
        (xor(xor(a, b), c), or(and(a, b), and(c, or(a, b))))
    }
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
    /// Harley and Seal's arrangement: inputs pair into the ones digit, the
    /// carries pair into the twos, and so on up. Fifteen full adders per
    /// sixteen inputs, and afterwards `ones + 2 twos + 4 fours + 8 eights +
    /// 16 sixteens` has grown by exactly the number of inputs set, position by
    /// position. A textbook popcount would finish by counting the bits of each
    /// digit horizontally; here every position is a reference, so the digits
    /// themselves are the answer.
    ///
    /// Inputs are fetched through `x` as they are consumed rather than staged
    /// in an array first, so the network runs out of registers: at most a
    /// dozen chunks are live at once.
    #[inline(always)]
    fn fold16<A: Adder>(&mut self, x: impl Fn(usize) -> Chunk) -> Chunk {
        let csa = A::csa;
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
fn count32_chunk<A: Adder>(x: impl Fn(usize) -> Chunk) -> [Chunk; PLANES] {
    let mut digits = Digits::ZERO;
    let first = digits.fold16::<A>(&x);
    let second = digits.fold16::<A>(|i| x(16 + i));
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

/// Six planes of the count of the first `n` of up to 32 inputs fetched
/// through `x`; inputs past `n` read as zero.
///
/// A short group costs as much as a full block of sixteen but never more than
/// two.
#[inline(always)]
fn count_padded_chunk<A: Adder>(x: impl Fn(usize) -> Chunk, n: usize) -> [Chunk; PLANES] {
    let mut digits = Digits::ZERO;
    let mut sixteens = [ZERO; 2];
    for (block, sixteen) in sixteens.iter_mut().enumerate().take(n.div_ceil(16)) {
        let from = block * 16;
        *sixteen = digits.fold16::<A>(|i| if from + i < n { x(from + i) } else { ZERO });
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
#[inline(always)]
///
/// The index loop is deliberate. Clippy's iterator form, `iter_mut().enumerate()
/// .skip(shift)`, measured 23% more instructions and 9% more cycles over a
/// whole weighted classification run (docs/Optimization_round_5.md §2.1).
#[allow(clippy::needless_range_loop)]
fn add_shifted<A: Adder>(total: &mut [Chunk; WEIGHTED_PLANES], x: &[Chunk; PLANES], shift: usize) {
    let mut carry = ZERO;
    for plane in shift..WEIGHTED_PLANES {
        let digit = x.get(plane - shift).copied().unwrap_or(ZERO);
        let (sum, next) = A::csa(total[plane], digit, carry);
        total[plane] = sum;
        carry = next;
    }
    // A score cannot exceed `SAMPLE_SIZE * MAX_WEIGHT`, which is 128: nothing
    // is ever carried out of the top plane.
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
/// `stride` must be [`stride`]`(lanes)`, and every row that wide. Draws
/// repeat freely -- a row drawn twice is simply two inputs -- which is
/// exactly the published with-replacement score.
pub fn count32(
    matrix: &[u64],
    offsets: &[usize; SAMPLE_SIZE],
    lanes: usize,
    stride: usize,
    planes: &mut [u64],
) {
    if !crate::simd::count32(matrix, offsets, lanes, stride, planes) {
        count32_with::<Shared>(matrix, offsets, lanes, stride, planes);
    }
}

/// [`count32`] with a chosen full adder. Inlined into whatever calls it, so
/// that it is compiled for that caller's instruction set.
#[inline(always)]
pub fn count32_with<A: Adder>(
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
            for (plane, digit) in out.iter_mut().zip(count32_chunk::<A>(|i| rows[i][j])) {
                plane[j] = digit;
            }
        }
    }
}

/// Weighted replicate counts: like [`count32`], but draw `i` weighs
/// `1 << shifts[i]`, and eight planes are written.
///
/// The draws are split by weight tier and each tier counted with the same
/// network, zero-padded to whole blocks of sixteen; the tier counts are then
/// added at their shifts. Two tiers cost at most one extra block of sixteen
/// and a small adder over counting them all at weight one.
pub fn count_weighted(
    matrix: &[u64],
    offsets: &[usize; SAMPLE_SIZE],
    shifts: &[u8; SAMPLE_SIZE],
    lanes: usize,
    stride: usize,
    planes: &mut [u64],
) {
    if !crate::simd::count_weighted(matrix, offsets, shifts, lanes, stride, planes) {
        count_weighted_with::<Shared>(matrix, offsets, shifts, lanes, stride, planes);
    }
}

/// [`count_weighted`] with a chosen full adder; see [`count32_with`].
#[inline(always)]
pub fn count_weighted_with<A: Adder>(
    matrix: &[u64],
    offsets: &[usize; SAMPLE_SIZE],
    shifts: &[u8; SAMPLE_SIZE],
    lanes: usize,
    stride: usize,
    planes: &mut [u64],
) {
    debug_assert_eq!(stride, self::stride(lanes));
    // Group the draws by tier. Weights are powers of two no larger than
    // `MAX_WEIGHT`, so there are at most three shifts.
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
        // Unused slots repeat the group's first row. `count_padded_chunk`
        // never reads past `len`, so which row it is does not matter, only
        // that it is in bounds.
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
                    let counted = count_padded_chunk::<A>(|i| group[i][j], lens[tier]);
                    add_shifted::<A>(&mut total, &counted, tier);
                }
            }
            for (plane, digit) in out.iter_mut().zip(total) {
                plane[j] = digit;
            }
        }
    }
}

/// Largest count among the `valid` positions of `lanes` lanes, and the set of
/// positions holding it, written into `out`.
///
/// Descends from the most significant plane: at each bit, positions that have
/// it set dominate those that do not, so what survives every plane is exactly
/// the maximisers. With no valid position at all the answer is zero with an
/// empty set, which the callers treat as "nothing here".
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
/// A nonempty result is the complete set of positions holding *every* drawn
/// word, which is to say the replicate's whole tie set at the highest score
/// it can reach. Rows are consumed in the order given, and the scan stops as
/// soon as the running intersection is empty, so putting the rarest rows
/// first makes a miss cheap.
///
/// Every row is ANDed across every lane. Visiting only the lanes still alive
/// after the first row was measured and is slower -- 3% more cycles on
/// full-length 16S, 15% on V4 -- because the rarest drawn word is rarely rare
/// enough to leave few lanes, and the scalar walk loses the vector width.
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
        // Checking every row costs an OR per lane, which is nothing next to
        // the AND itself, and a miss usually shows within a few rows.
        if any == 0 {
            return false;
        }
    }
    true
}

/// `out &= row` over `out.len()` lanes; whether any bit is left.
///
/// One step of an intersection, for callers that feed it rows one at a time
/// and may stop early.
pub fn and_lanes(out: &mut [u64], row: &[u64]) -> bool {
    let mut any = 0;
    for (a, &b) in out.iter_mut().zip(row) {
        *a &= b;
        any |= *a;
    }
    any != 0
}

/// [`and_lanes`] with the row read in place, as the little-endian bytes of a
/// bitmap row in the index, eight per lane.
pub fn and_bytes(out: &mut [u64], row: &[u8]) -> bool {
    let mut any = 0;
    for (a, b) in out.iter_mut().zip(row.chunks_exact(8)) {
        *a &= u64::from_le_bytes(b.try_into().expect("eight bytes"));
        any |= *a;
    }
    any != 0
}
