use crate::{K, WORDS};

/// Uppercase, convert RNA to DNA, and turn any other byte into `N`.
///
/// Non-ACGT bytes just break the k-mer window. Unknown bytes are counted
/// so the caller can report them instead of failing the build.
pub fn normalize(sequence: &[u8]) -> (Vec<u8>, usize) {
    let mut unknown = 0;
    let out = sequence
        .iter()
        .map(|&b| match b.to_ascii_uppercase() {
            b'A' | b'C' | b'G' | b'T' | b'R' | b'Y' | b'S' | b'W' | b'K' | b'M' | b'B' | b'D'
            | b'H' | b'V' | b'N' => b.to_ascii_uppercase(),
            b'U' => b'T',
            _ => {
                unknown += 1;
                b'N'
            }
        })
        .collect();
    (out, unknown)
}

/// Distinct two-bit-encoded words, ascending; ambiguous bases reset the window.
/// The ascending order matters for reproducible sampling.
pub fn unique_words(sequence: &[u8]) -> Vec<u16> {
    let mut seen = [0u64; WORDS / 64];
    let (mut word, mut run) = (0u16, 0usize);
    for &b in sequence {
        let code = match b {
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
    let mut words = Vec::with_capacity(sequence.len().min(WORDS));
    for (i, &bits) in seen.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            words.push((i * 64 + bits.trailing_zeros() as usize) as u16);
            bits &= bits - 1;
        }
    }
    words
}

pub fn reverse_complement(sequence: &[u8]) -> Vec<u8> {
    sequence
        .iter()
        .rev()
        .map(|b| match b {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' => b'A',
            b'R' => b'Y',
            b'Y' => b'R',
            b'K' => b'M',
            b'M' => b'K',
            b'B' => b'V',
            b'V' => b'B',
            b'D' => b'H',
            b'H' => b'D',
            _ => *b,
        })
        .collect()
}
