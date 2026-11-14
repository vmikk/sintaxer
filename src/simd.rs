//! Hand-vectorised kernels: the one place the crate uses `core::arch`.
//!
//! Most of the crate's bit-parallel code is plain Rust that LLVM vectorises
//! for the x86-64-v3 floor. What lives here is what the floor cannot give:
//! code whose plain form stays scalar (the BLAKE3 lanes), and builds of
//! portable code for instructions beyond the floor (the AVX-512 counting
//! network). docs/Optimization_round_6.md §2.6-2.7 has the evidence for both.
//!
//! The rules every kernel here follows:
//!
//! - It has a portable twin elsewhere in the crate, which is both the
//!   fallback on a CPU without the instructions and the test oracle.
//! - It is a `#[target_feature]` function whose body is plain Rust or
//!   intrinsics that take no pointers, all of which are safe inside such a
//!   function.
//! - It is reached only through a wrapper that detects the feature at run
//!   time and then calls it. That call is the `unsafe` the kernel costs, and
//!   its reason sits next to it. Detection is cached by the standard library,
//!   so the check is a load and a branch.

/// The first two output words of BLAKE3's compression function, per lane,
/// for eight one-block messages that share every message word except 12 and
/// 13. `None` when the CPU lacks AVX2; the caller then hashes one message at
/// a time.
///
/// The compression runs from the IV with counter zero, as for any message
/// that is a single chunk. `common[12]` and `common[13]` are ignored.
pub fn blake3_first_words8(
    common: &[u32; 16],
    w12: [u32; 8],
    w13: [u32; 8],
    block_len: u32,
    flags: u32,
) -> Option<[u64; 8]> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: `blake3_avx2` requires AVX2 and nothing else, and the CPU
        // has just been found to support it.
        #[expect(unsafe_code, reason = "calling an AVX2 kernel after detecting AVX2")]
        return Some(unsafe { x86::blake3_avx2(common, w12, w13, block_len, flags) });
    }
    let _ = (common, w12, w13, block_len, flags);
    None
}

/// Whether the CPU runs the AVX-512 builds of the counting network: AVX-512F
/// for the registers and ternary logic, VL to use them on 256-bit chunks.
#[cfg(target_arch = "x86_64")]
fn avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

/// [`crate::bits::count32`] on an AVX-512 host: the same network with the
/// [`crate::bits::Majority`] adder, so that each full adder is two ternary
/// ops, and 32 vector registers for its live values. `false`, having done
/// nothing, anywhere else.
pub fn count32(
    matrix: &[u64],
    offsets: &[usize; crate::SAMPLE_SIZE],
    lanes: usize,
    stride: usize,
    planes: &mut [u64],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if avx512() {
        // SAFETY: `count32_avx512` requires AVX-512F and AVX-512VL and nothing
        // else, and the CPU has just been found to support both.
        #[expect(
            unsafe_code,
            reason = "calling an AVX-512 kernel after detecting AVX-512"
        )]
        unsafe {
            x86::count32_avx512(matrix, offsets, lanes, stride, planes)
        };
        return true;
    }
    let _ = (matrix, offsets, lanes, stride, planes);
    false
}

/// [`crate::bits::count_weighted`] on an AVX-512 host; see [`count32`].
pub fn count_weighted(
    matrix: &[u64],
    offsets: &[usize; crate::SAMPLE_SIZE],
    shifts: &[u8; crate::SAMPLE_SIZE],
    lanes: usize,
    stride: usize,
    planes: &mut [u64],
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if avx512() {
        // SAFETY: as for `count32`: AVX-512F and AVX-512VL, just detected.
        #[expect(
            unsafe_code,
            reason = "calling an AVX-512 kernel after detecting AVX-512"
        )]
        unsafe {
            x86::count_weighted_avx512(matrix, offsets, shifts, lanes, stride, planes)
        };
        return true;
    }
    let _ = (matrix, offsets, shifts, lanes, stride, planes);
    false
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use crate::{
        SAMPLE_SIZE,
        bits::{self, Majority},
    };
    use std::arch::x86_64::*;

    /// The portable network compiled for AVX-512: it is inlined here, so it
    /// is built with this function's instruction set.
    #[target_feature(enable = "avx512f,avx512vl")]
    pub(super) fn count32_avx512(
        matrix: &[u64],
        offsets: &[usize; SAMPLE_SIZE],
        lanes: usize,
        stride: usize,
        planes: &mut [u64],
    ) {
        bits::count32_with::<Majority>(matrix, offsets, lanes, stride, planes);
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    pub(super) fn count_weighted_avx512(
        matrix: &[u64],
        offsets: &[usize; SAMPLE_SIZE],
        shifts: &[u8; SAMPLE_SIZE],
        lanes: usize,
        stride: usize,
        planes: &mut [u64],
    ) {
        bits::count_weighted_with::<Majority>(matrix, offsets, shifts, lanes, stride, planes);
    }

    const IV: [u32; 8] = [
        0x6A09_E667,
        0xBB67_AE85,
        0x3C6E_F372,
        0xA54F_F53A,
        0x510E_527F,
        0x9B05_688C,
        0x1F83_D9AB,
        0x5BE0_CD19,
    ];

    /// BLAKE3's compression function on eight blocks, one per 32-bit lane:
    /// each addition, XOR and rotation of the reference implementation is one
    /// AVX2 instruction for all eight. The rotations by 16 and 8 are byte
    /// shuffles; those by 12 and 7 are two shifts and an OR.
    #[target_feature(enable = "avx2")]
    pub(super) fn blake3_avx2(
        common: &[u32; 16],
        w12: [u32; 8],
        w13: [u32; 8],
        block_len: u32,
        flags: u32,
    ) -> [u64; 8] {
        let splat = |x: u32| _mm256_set1_epi32(x as i32);
        let lanes = |w: [u32; 8]| {
            _mm256_setr_epi32(
                w[0] as i32,
                w[1] as i32,
                w[2] as i32,
                w[3] as i32,
                w[4] as i32,
                w[5] as i32,
                w[6] as i32,
                w[7] as i32,
            )
        };
        let rot16 = _mm256_setr_epi8(
            2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11,
            8, 9, 14, 15, 12, 13,
        );
        let rot8 = _mm256_setr_epi8(
            1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10,
            11, 8, 13, 14, 15, 12,
        );
        let mut m: [__m256i; 16] = std::array::from_fn(|i| splat(common[i]));
        m[12] = lanes(w12);
        m[13] = lanes(w13);
        let mut s: [__m256i; 16] = [
            splat(IV[0]),
            splat(IV[1]),
            splat(IV[2]),
            splat(IV[3]),
            splat(IV[4]),
            splat(IV[5]),
            splat(IV[6]),
            splat(IV[7]),
            splat(IV[0]),
            splat(IV[1]),
            splat(IV[2]),
            splat(IV[3]),
            splat(0),
            splat(0),
            splat(block_len),
            splat(flags),
        ];
        macro_rules! g {
            ($a:literal, $b:literal, $c:literal, $d:literal, $x:expr, $y:expr) => {
                s[$a] = _mm256_add_epi32(_mm256_add_epi32(s[$a], s[$b]), $x);
                s[$d] = _mm256_shuffle_epi8(_mm256_xor_si256(s[$d], s[$a]), rot16);
                s[$c] = _mm256_add_epi32(s[$c], s[$d]);
                let t = _mm256_xor_si256(s[$b], s[$c]);
                s[$b] = _mm256_or_si256(_mm256_srli_epi32::<12>(t), _mm256_slli_epi32::<20>(t));
                s[$a] = _mm256_add_epi32(_mm256_add_epi32(s[$a], s[$b]), $y);
                s[$d] = _mm256_shuffle_epi8(_mm256_xor_si256(s[$d], s[$a]), rot8);
                s[$c] = _mm256_add_epi32(s[$c], s[$d]);
                let t = _mm256_xor_si256(s[$b], s[$c]);
                s[$b] = _mm256_or_si256(_mm256_srli_epi32::<7>(t), _mm256_slli_epi32::<25>(t));
            };
        }
        macro_rules! round {
            () => {
                g!(0, 4, 8, 12, m[0], m[1]);
                g!(1, 5, 9, 13, m[2], m[3]);
                g!(2, 6, 10, 14, m[4], m[5]);
                g!(3, 7, 11, 15, m[6], m[7]);
                g!(0, 5, 10, 15, m[8], m[9]);
                g!(1, 6, 11, 12, m[10], m[11]);
                g!(2, 7, 8, 13, m[12], m[13]);
                g!(3, 4, 9, 14, m[14], m[15]);
            };
        }
        macro_rules! permute {
            () => {
                m = [
                    m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13], m[1], m[11], m[12], m[5],
                    m[9], m[14], m[15], m[8],
                ];
            };
        }
        round!();
        permute!();
        round!();
        permute!();
        round!();
        permute!();
        round!();
        permute!();
        round!();
        permute!();
        round!();
        permute!();
        round!();
        let low = _mm256_xor_si256(s[0], s[8]);
        let high = _mm256_xor_si256(s[1], s[9]);
        macro_rules! seed {
            ($lane:literal) => {
                u64::from(_mm256_extract_epi32::<$lane>(low) as u32)
                    | u64::from(_mm256_extract_epi32::<$lane>(high) as u32) << 32
            };
        }
        [
            seed!(0),
            seed!(1),
            seed!(2),
            seed!(3),
            seed!(4),
            seed!(5),
            seed!(6),
            seed!(7),
        ]
    }
}
