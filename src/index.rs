//! Immutable, memory-mapped `.stx` index (format 5). See docs/INDEX_FORMAT.md
//! for the byte-level contract.
use crate::{ALGORITHM_VERSION, K, TILE, WORDS, input, sequence, taxonomy::TreeBuilder};
use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapMut};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const HEADER: usize = 256;
/// Word-directory record: offset, length, df, representation, then the two
/// distinct-taxon counts added in format 3.
const ENTRY: usize = 32;
const NODE: usize = 24;
/// Per-node aggregates added in format 4: references, children, genus-rank
/// descendants. Three u32 in the order a `protax` design row reads them.
const NODE_STAT: usize = 12;
const MAGIC: &[u8; 8] = b"SINTAXER";
const FORMAT_VERSION: u32 = 5;

// Header slots for the v2 sequence sections, inside the range v1 reserved.
const H_SEQ_DIR: usize = 160;
const H_SEQ_PAYLOAD: usize = 168;
const H_SEQ_LEN: usize = 176;
// Header slots for the v4 sections, in the range v1 through v3 reserved.
const H_VOCAB: usize = 184;
const H_NODE_STATS: usize = 192;
/// Every k-mer of a packed record, in sequence order, duplicates included.
///
/// The single implementation behind both [`Index::visit_kmers`] and
/// [`Index::reference_words`], so the rule that an ambiguous base *breaks* the
/// window rather than being skipped over -- which is what makes these agree
/// with `sequence::unique_words` -- is written once.
///
/// The record is cut into runs of unambiguous bases, and each run's words are
/// read straight out of 64-bit loads rather than rolled in one base at a time.
/// Eight packed bytes hold 32 bases, low pair first; reversing the pairs of
/// the whole `u64` puts the first base in the top pair, which is the order a
/// word code keeps its bases in, so the word starting `p` bases into the load
/// is simply `y >> (48 - 2p)`. One load yields up to 25 words, each for a
/// shift and a truncation, where rolling cost a load-shift-mask-merge per base
/// plus the loop around it (docs/RANK_round3.md §C). The previous version,
/// kept as the test oracle, is `walk_kmers_scalar`.
fn walk_kmers(bases: usize, ambiguous: &[u8], packed: &[u8], mut visit: impl FnMut(u16)) {
    // Words per 64-bit load in the steady state. A load holds 32 bases, so up
    // to `32 - K + 1` words; stopping at a multiple of four keeps the next
    // load byte-aligned.
    const CHUNK: usize = (32 - K + 1) / 4 * 4;
    let mut breaks = ambiguous.chunks_exact(4).map(|b| u32_at(b, 0) as usize);
    let mut next_break = breaks.next().unwrap_or(usize::MAX);
    let mut at = 0;
    while at < bases {
        // This run of unambiguous bases is `at .. end`.
        let end = next_break.min(bases);
        if end >= at + K {
            // Start of the run's last word.
            let last = end - K;
            let mut start = at;
            while start <= last {
                let byte = start / 4;
                // Shift each word into the top sixteen bits in turn: the only
                // loop-carried work is one shift, where indexing by position
                // chained a counter through every iteration.
                let mut y = reversed_pairs(packed, byte);
                if start % 4 == 0 && start + CHUNK <= last + 1 {
                    // The common case, a whole chunk: a constant trip count,
                    // which unrolls and has no exit to mispredict.
                    for _ in 0..CHUNK {
                        visit((y >> (64 - 2 * K)) as u16);
                        y <<= 2;
                    }
                    start += CHUNK;
                } else {
                    // A run's first chunk, which also realigns it, and its
                    // last: words starting `first ..= stop` bases into the
                    // load, all inside its 32 bases and inside the run.
                    let first = start % 4;
                    let stop = (last - byte * 4).min(CHUNK - 1);
                    y <<= 2 * first;
                    for _ in 0..stop + 1 - first {
                        visit((y >> (64 - 2 * K)) as u16);
                        y <<= 2;
                    }
                    start = byte * 4 + stop + 1;
                }
            }
        }
        // Step over the ambiguous base itself and start a fresh window.
        at = end + 1;
        next_break = breaks.next().unwrap_or(usize::MAX);
    }
}

/// The 32 bases packed from `byte` on, first base in the top pair.
///
/// Bytes past the end of the record read as zero. They can only fill bases
/// after the run's last word, which no caller emits.
#[inline(always)]
fn reversed_pairs(packed: &[u8], byte: usize) -> u64 {
    let x = match packed.get(byte..byte + 8) {
        Some(bytes) => u64::from_le_bytes(bytes.try_into().unwrap()),
        None => {
            let mut bytes = [0u8; 8];
            let rest = packed.get(byte..).unwrap_or_default();
            bytes[..rest.len()].copy_from_slice(rest);
            u64::from_le_bytes(bytes)
        }
    };
    // Reverse the four pairs within each byte, then the bytes.
    let x = ((x >> 2) & 0x3333_3333_3333_3333) | ((x & 0x3333_3333_3333_3333) << 2);
    let x = ((x >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((x & 0x0F0F_0F0F_0F0F_0F0F) << 4);
    x.swap_bytes()
}

fn u16_at(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes(b[p..p + 2].try_into().unwrap())
}
fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
}
fn put16(b: &mut [u8], p: usize, v: u16) {
    b[p..p + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], p: usize, v: u32) {
    b[p..p + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], p: usize, v: u64) {
    b[p..p + 8].copy_from_slice(&v.to_le_bytes());
}
fn align64(n: u64) -> u64 {
    n.div_ceil(64) * 64
}

/// Distinct genera and families containing each word, swept from the postings.
///
/// The counts a taxonomic IDF weight needs. Word-major is what makes the
/// deduplication O(1) per incidence: within one word's posting list a stamp
/// array keyed by node can carry that word's own index, so a genus already
/// counted for this word is recognised without a set or a sort. Reference-major
/// -- the order `verify` otherwise walks in -- would need the references sorted
/// by genus first.
///
/// About one billion stamp comparisons on a 1.5-million-reference ITS
/// database, against a ninety-second build.
fn sweep_taxa(
    map: &[u8],
    references: usize,
    refs_offset: usize,
    nodes_offset: usize,
    nodes: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut genera = vec![0u32; WORDS];
    let mut families = vec![0u32; WORDS];
    // A missing rank is interned as its own empty-named node, so these count
    // taxon identities rather than annotated names: a reference with no genus
    // contributes its parent's sentinel. That is the right denominator for a
    // weight, which is asking how many distinct lineages a word spans.
    let mut stamp_g = vec![u32::MAX; nodes];
    let mut stamp_f = vec![u32::MAX; nodes];
    let leaf = |reference: usize| u32_at(map, refs_offset + reference * 4);
    let parent = |node: u32| u32_at(map, nodes_offset + node as usize * NODE);
    for word in 0..WORDS {
        let p = HEADER + word * ENTRY;
        let offset = u64_at(map, p) as usize;
        let length = u64_at(map, p + 8) as usize;
        let stamp = word as u32;
        let mut count = |reference: usize| {
            let g = leaf(reference);
            if stamp_g[g as usize] != stamp {
                stamp_g[g as usize] = stamp;
                genera[word] += 1;
                let f = parent(g);
                if stamp_f[f as usize] != stamp {
                    stamp_f[f as usize] = stamp;
                    families[word] += 1;
                }
            }
        };
        match Layout::from_code(map[p + 20]) {
            Some(Layout::Bitmap) => {
                for (i, byte) in map[offset..offset + length].iter().enumerate() {
                    let mut bits = *byte;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let reference = i * 8 + bit;
                        if reference < references {
                            count(reference);
                        }
                    }
                }
            }
            Some(Layout::Postings) => {
                for chunk in map[offset..offset + length].chunks_exact(4) {
                    count(u32::from_le_bytes(chunk.try_into().unwrap()) as usize);
                }
            }
            Some(Layout::Blocked) => {
                let row = &map[offset..offset + length];
                let (mut at, mut base) = (0usize, 0usize);
                while at + 2 <= row.len() {
                    let block = u16_at(row, at) as usize;
                    at += 2;
                    for local in row[at..at + block * 2].chunks_exact(2) {
                        count(base + u16::from_le_bytes(local.try_into().unwrap()) as usize);
                    }
                    at += block * 2;
                    base += TILE;
                }
            }
            None => {}
        }
    }
    (genera, families)
}

/// Per-node reference counts, child counts, and genus-rank descendant counts.
///
/// All three roll up in one descending pass because the format guarantees
/// `parent < node` for every non-root node, so a single reverse sweep sees each
/// child before its parent. That is O(nodes), not O(references x ranks).
///
/// `references` must count every reference under the node, hit or not. PROTAX's
/// `simMaxEml` counts non-hitting references deliberately -- it is what makes
/// "many references, few hits" read as mislabelling evidence -- so reading the
/// count off the candidate set instead would collapse the statistic. `genera`
/// stands in for PROTAX's `log(prior)`, which counts described species under a
/// node; genus is the deepest rank this format stores.
///
/// `children` counts only the **named** ones. The unnamed sentinel is not a
/// taxon and must not be counted among "the children nothing matched", which
/// is a claim about rival taxa. Counting it adds a phantom sibling to every
/// under-annotated clade -- precisely where the number is doing the most work.
struct NodeStats {
    references: Vec<u32>,
    children: Vec<u32>,
    genera: Vec<u32>,
}

impl NodeStats {
    fn compute(
        leaves: impl Iterator<Item = u32>,
        parents: &[u32],
        ranks: &[u8],
        named: &[bool],
    ) -> Self {
        let nodes = parents.len();
        let mut stats = Self {
            references: vec![0u32; nodes],
            children: vec![0u32; nodes],
            genera: ranks.iter().map(|&r| u32::from(r == 6)).collect(),
        };
        for leaf in leaves {
            stats.references[leaf as usize] += 1;
        }
        for node in (1..nodes).rev() {
            let parent = parents[node] as usize;
            stats.children[parent] += u32::from(named[node]);
            stats.references[parent] += stats.references[node];
            stats.genera[parent] += stats.genera[node];
        }
        stats
    }
}

pub struct Index {
    map: Mmap,
    pub references: usize,
    pub nodes: usize,
    refs_offset: usize,
    nodes_offset: usize,
    strings_offset: usize,
    seq_dir: usize,
    seq_payload: usize,
    seq_len: usize,
    vocab_offset: usize,
    node_stats_offset: usize,
    validated: Vec<AtomicBool>,
}

/// How a word's posting row is stored. The directory records it per word and
/// the builder picks whichever is smallest; see `docs/INDEX_FORMAT.md`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// Strictly increasing `u32` reference ids. Smallest for rare words.
    Postings,
    /// One bit per reference, `ceil(N/8)` bytes. Smallest for common words.
    Bitmap,
    /// One `u16` count per tile followed by that tile's ids as `u16` offsets
    /// from the tile start. Smallest in between, which on a production index is
    /// most of the payload and nearly all of a query's vocabulary.
    Blocked,
}

impl Layout {
    fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Postings),
            1 => Some(Self::Bitmap),
            2 => Some(Self::Blocked),
            _ => None,
        }
    }
    /// Bytes a row of `count` postings occupies over `references` references.
    fn len(self, references: usize, count: usize) -> u64 {
        match self {
            Self::Postings => count as u64 * 4,
            Self::Bitmap => (references as u64).div_ceil(8),
            Self::Blocked => (references.div_ceil(TILE) as u64 + count as u64) * 2,
        }
    }
    /// The smallest layout for a row, which is how the builder chooses.
    fn best(references: usize, count: usize) -> Self {
        // The bitmap test is kept exactly as format 4 had it so that the
        // dense/sparse split -- and with it the rank sweep's kernel mix and
        // counter depth -- does not move as a side effect of adding a third
        // layout. Blocked competes only with plain postings, and wins whenever
        // a row averages more than one posting per tile.
        if Self::Bitmap.len(references, count) < Self::Postings.len(references, count) {
            Self::Bitmap
        } else if references.div_ceil(TILE) < count {
            Self::Blocked
        } else {
            Self::Postings
        }
    }
    fn code(self) -> u8 {
        match self {
            Self::Postings => 0,
            Self::Bitmap => 1,
            Self::Blocked => 2,
        }
    }
}

/// Where a word's row lies in the index and how it is stored: what
/// [`Index::locate`] reads from the word directory and has validated. Not
/// tied to the index's lifetime, so a caller that will visit the same rows
/// again -- the replicates after the rank sweep -- can keep it and skip the
/// directory, whose 2 MiB cost a cache miss per word. [`Index::row_at`] turns
/// it back into a [`Row`]; handed a `RowRef` from another index it can only
/// panic on the bounds or read the wrong bytes, never read out of bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowRef {
    offset: usize,
    len: usize,
    count: usize,
    layout: Layout,
}

#[derive(Clone, Copy)]
pub struct Row<'a> {
    data: &'a [u8],
    pub layout: Layout,
    pub count: usize,
}
impl<'a> Row<'a> {
    /// Bytes of this row, for kernels that walk it with their own cursor.
    pub fn bytes(&self) -> &'a [u8] {
        self.data
    }
}
impl Row<'_> {
    fn id(&self, i: usize) -> usize {
        u32_at(self.data, i * 4) as usize
    }
    fn lower_bound(&self, target: usize) -> usize {
        let (mut lo, mut hi) = (0, self.count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.id(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
    /// Every reference id in `start .. end`, ascending.
    ///
    /// [`Layout::Blocked`] has no per-tile offset directory -- the rank sweep
    /// visits every tile in order and so never needs one -- which means a
    /// narrow range still walks the row from its start. Every caller asks for
    /// the full range.
    pub fn visit(&self, start: usize, end: usize, mut f: impl FnMut(usize)) {
        match self.layout {
            Layout::Bitmap => {
                let mut block = start / 64 * 64;
                while block < end {
                    let mut bits = self.bits64(block);
                    if block < start {
                        bits &= u64::MAX << (start - block);
                    }
                    if end - block < 64 {
                        bits &= (1u64 << (end - block)) - 1;
                    }
                    while bits != 0 {
                        f(block + bits.trailing_zeros() as usize);
                        bits &= bits - 1;
                    }
                    block += 64;
                }
            }
            Layout::Postings => {
                let mut i = self.lower_bound(start);
                while i < self.count {
                    let id = self.id(i);
                    if id >= end {
                        break;
                    }
                    f(id);
                    i += 1;
                }
            }
            Layout::Blocked => {
                let (mut at, mut base) = (0, 0);
                while at + 2 <= self.data.len() && base < end {
                    let block = u16_at(self.data, at) as usize;
                    at += 2;
                    for local in self.data[at..at + block * 2].chunks_exact(2) {
                        let id = base + u16::from_le_bytes(local.try_into().unwrap()) as usize;
                        if id >= start && id < end {
                            f(id);
                        }
                    }
                    at += block * 2;
                    base += TILE;
                }
            }
        }
    }
    /// Add one to `counts[id - start]` for every posting this **sparse** row
    /// holds inside the tile `start .. start + counts.len()`.
    ///
    /// The rank pass used to force sparse rows through the same bit-sliced
    /// kernel as dense ones: materialise a 64-lane bitmap, then ripple it into
    /// the counter planes. For a row contributing a few dozen postings to a
    /// 4096-reference tile that is almost all overhead -- a 512-byte clear and
    /// a multi-plane carry propagation to record forty-odd increments. Adding
    /// straight into the tile's count window is the same arithmetic with none
    /// of the machinery, and because postings are sorted the writes stay inside
    /// one 8 KiB L1-resident window.
    ///
    /// `cursor` counts postings consumed and carries across calls for ascending
    /// `start`. Rows stored this way are rare in a query's vocabulary: the
    /// builder blocks anything averaging more than one posting per tile.
    pub fn tile_increment(&self, start: usize, counts: &mut [u16], cursor: &mut usize) {
        debug_assert_eq!(self.layout, Layout::Postings);
        let end = start + counts.len();
        let mut i = *cursor;
        // A stale cursor can only point before the tile: postings are strictly
        // increasing and tiles are visited in order.
        while i < self.count && self.id(i) < start {
            i += 1;
        }
        // Slicing the remaining postings once lets the loop read them without
        // a bounds check per posting.
        for posting in self.data[i * 4..self.count * 4].chunks_exact(4) {
            let id = u32::from_le_bytes(posting.try_into().unwrap()) as usize;
            if id >= end {
                break;
            }
            counts[id - start] += 1;
            i += 1;
        }
        *cursor = i;
    }

    /// Add one to `counts[local]` for every posting of one **blocked** row's
    /// block.
    ///
    /// This is the tightest of the three kernels and the reason the layout
    /// exists. A block is a `u16` count followed by that many `u16` offsets from
    /// the tile's first reference, so the loop has a trip count known before it
    /// starts, no comparison per posting, no subtraction to localise an id, and
    /// half the bytes of a `u32` posting list. The `u32` version had to test
    /// every posting against the tile's end because it could not know where the
    /// tile stopped.
    ///
    /// `postings` is the block without its count. Blocks are emitted for every
    /// tile including empty ones, so the rank sweep walks each row with a byte
    /// cursor and no seeking; it reads the counts of all its rows in a pass of
    /// their own before any of their postings (see `rank::sweep_tile`).
    ///
    /// The window is a whole tile even when the final tile is short, and the
    /// offset is masked to it. [`Index::row`] has already checked every offset
    /// against its tile, so the mask changes nothing; it is what tells the
    /// compiler so, which removes a bounds check from every posting -- two of
    /// the loop's seven instructions (docs/RANK_round3.md §B).
    #[inline(always)]
    pub fn block_increment(postings: &[u8], counts: &mut [u16; TILE]) {
        const { assert!(TILE.is_power_of_two()) };
        for local in postings.chunks_exact(2) {
            let local = u16::from_le_bytes(local.try_into().unwrap()) as usize;
            counts[local & (TILE - 1)] += 1;
        }
    }

    /// Membership bits for a whole tile of a **dense** row, in one bulk read.
    ///
    /// A tile of a bitmap row is contiguous: `start` is always a multiple of
    /// the rank pass's tile width, so `start / 8` is a multiple of 512 and the
    /// whole tile is one aligned run of `out.len() * 8` bytes. Going through
    /// `bits64` per lane instead cost a bounds computation and a `copy_from_slice`
    /// -- which the release build emits as a real `memcpy` call -- 64 times per
    /// row per tile. Every lane is assigned, so no pre-zeroing is needed either.
    ///
    /// Only the final tile of the final row can be short, and only there does
    /// the tail path run.
    pub fn dense_tile(&self, start: usize, out: &mut [u64]) {
        debug_assert_eq!(self.layout, Layout::Bitmap);
        let off = start / 8;
        if let Some(src) = self.data.get(off..off + out.len() * 8) {
            for (slot, word) in out.iter_mut().zip(src.chunks_exact(8)) {
                *slot = u64::from_le_bytes(word.try_into().unwrap());
            }
            return;
        }
        let src = self.data.get(off..).unwrap_or_default();
        let mut whole = src.chunks_exact(8);
        for (slot, word) in out.iter_mut().zip(&mut whole) {
            *slot = u64::from_le_bytes(word.try_into().unwrap());
        }
        for slot in &mut out[src.len() / 8..] {
            *slot = 0;
        }
        let tail = whole.remainder();
        if !tail.is_empty()
            && let Some(slot) = out.get_mut(src.len() / 8)
        {
            let mut bytes = [0u8; 8];
            bytes[..tail.len()].copy_from_slice(tail);
            *slot = u64::from_le_bytes(bytes);
        }
    }

    /// The bytes of `lanes` lanes of a **dense** row from `start`, as far as
    /// the row goes: whole lanes, except that in the database's final tile
    /// the row can end inside its last lane. Kernels that read it zero-pad
    /// what is missing, which is exactly what [`Row::dense_tile`] writes.
    pub fn dense_bytes_upto(&self, start: usize, lanes: usize) -> &[u8] {
        debug_assert_eq!(self.layout, Layout::Bitmap);
        let data = self.data.get(start / 8..).unwrap_or_default();
        &data[..data.len().min(lanes * 8)]
    }

    /// The raw bytes of `lanes` whole lanes of a **dense** row from `start`,
    /// when the row holds them all -- every tile but a short final one -- for
    /// kernels that read the bitmap in place instead of copying it out.
    pub fn dense_bytes(&self, start: usize, lanes: usize) -> Option<&[u8]> {
        debug_assert_eq!(self.layout, Layout::Bitmap);
        let off = start / 8;
        self.data.get(off..off + lanes * 8)
    }

    /// Membership bits of one tile of this row, whatever its layout: bit `i`
    /// of `out` is reference `start + i`.
    ///
    /// Sparse rows carry `cursor` from tile to tile exactly as the rank
    /// sweep's cursors do, so tiles must be visited in ascending order, each
    /// once -- a tile not wanted goes through [`Row::skip_tile`] instead. `out`
    /// is overwritten in full, including any lanes past the end of the
    /// database, which come out zero. [`Index::row`] has already checked every
    /// posting against its tile and the database size, so nothing written can
    /// land past the tile's own lanes.
    pub fn tile_bits(&self, start: usize, cursor: &mut usize, out: &mut [u64]) {
        debug_assert!(start.is_multiple_of(TILE));
        match self.layout {
            Layout::Bitmap => self.dense_tile(start, out),
            Layout::Blocked => {
                out.fill(0);
                let block = u16_at(self.data, *cursor) as usize;
                let at = *cursor + 2;
                for local in self.data[at..at + block * 2].chunks_exact(2) {
                    let local = u16::from_le_bytes([local[0], local[1]]) as usize;
                    out[local / 64] |= 1 << (local % 64);
                }
                *cursor = at + block * 2;
            }
            Layout::Postings => {
                out.fill(0);
                let end = start + TILE;
                let mut i = *cursor;
                while i < self.count {
                    let id = self.id(i);
                    if id >= end {
                        break;
                    }
                    let local = id - start;
                    out[local / 64] |= 1 << (local % 64);
                    i += 1;
                }
                *cursor = i;
            }
        }
    }

    /// Advance a sparse row's cursor past the tile at `start` without reading
    /// its postings: the counterpart of [`Row::tile_bits`] for a tile no
    /// reference of which is wanted.
    pub fn skip_tile(&self, start: usize, cursor: &mut usize) {
        match self.layout {
            Layout::Bitmap => {}
            Layout::Blocked => {
                let block = u16_at(self.data, *cursor) as usize;
                *cursor += 2 + block * 2;
            }
            Layout::Postings => {
                let end = start + TILE;
                while *cursor < self.count && self.id(*cursor) < end {
                    *cursor += 1;
                }
            }
        }
    }

    /// Membership bits for 64 references of a **bitmap** row; `start` must be a
    /// multiple of 64. Only [`Row::visit`] needs this granularity -- the rank
    /// sweep reads a whole tile at once through [`Row::dense_tile`].
    fn bits64(&self, start: usize) -> u64 {
        debug_assert_eq!(start % 64, 0);
        debug_assert_eq!(self.layout, Layout::Bitmap);
        let off = start / 8;
        if off >= self.data.len() {
            return 0;
        }
        let mut bytes = [0; 8];
        let n = (self.data.len() - off).min(8);
        bytes[..n].copy_from_slice(&self.data[off..off + n]);
        u64::from_le_bytes(bytes)
    }
}

impl Index {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening index {}", path.display()))?;
        ensure!(
            file.metadata()?.len() >= HEADER as u64,
            "truncated index header"
        );
        // SAFETY: read-only mapping. Published indexes must never be truncated or
        // modified in place; our builder uses atomic replacement of a new inode.
        #[expect(unsafe_code, reason = "read-only mapping; see SAFETY above")]
        let map = unsafe { Mmap::map(&file)? };
        ensure!(&map[..8] == MAGIC, "not a Sintaxer index");
        ensure!(
            u32_at(&map, 8) == FORMAT_VERSION,
            "unsupported index format (rebuild required)"
        );
        ensure!(
            u32_at(&map, 12) == ALGORITHM_VERSION,
            "incompatible classifier algorithm"
        );
        ensure!(
            &map[224..256] == blake3::hash(&map[..224]).as_bytes(),
            "header checksum mismatch"
        );
        ensure!(
            u32_at(&map, 80) == 1 && u32_at(&map, 84) == 8,
            "unsupported normalization or k"
        );
        let references = u32_at(&map, 16) as usize;
        let nodes = u32_at(&map, 20) as usize;
        ensure!(references > 0 && nodes > 1, "empty index");
        let refs_offset = usize::try_from(u64_at(&map, 32))?;
        let nodes_offset = usize::try_from(u64_at(&map, 40))?;
        let strings_offset = usize::try_from(u64_at(&map, 48))?;
        let payload = usize::try_from(u64_at(&map, 56))?;
        let strings_len = usize::try_from(u64_at(&map, 72))?;
        ensure!(u64_at(&map, 64) == map.len() as u64, "index size mismatch");
        ensure!(
            u64_at(&map, 24) == HEADER as u64,
            "invalid directory offset"
        );
        ensure!(
            refs_offset == HEADER + WORDS * ENTRY,
            "invalid reference section"
        );
        ensure!(
            nodes_offset as u64 == align64(refs_offset as u64 + references as u64 * 4),
            "invalid node section"
        );
        ensure!(
            strings_offset as u64 == align64(nodes_offset as u64 + nodes as u64 * NODE as u64),
            "invalid strings section"
        );
        let strings_end = strings_offset
            .checked_add(strings_len)
            .context("string section overflow")?;
        ensure!(
            strings_end <= map.len(),
            "string section extends beyond file"
        );
        let seq_dir = usize::try_from(u64_at(&map, H_SEQ_DIR))?;
        let seq_payload = usize::try_from(u64_at(&map, H_SEQ_PAYLOAD))?;
        let seq_len = usize::try_from(u64_at(&map, H_SEQ_LEN))?;
        ensure!(
            seq_dir as u64 == align64(strings_end as u64),
            "invalid sequence directory section"
        );
        let seq_dir_end = seq_dir
            .checked_add((references + 1) * 8)
            .context("sequence directory overflow")?;
        ensure!(
            seq_payload as u64 == align64(seq_dir_end as u64),
            "invalid sequence payload section"
        );
        let seq_end = seq_payload
            .checked_add(seq_len)
            .context("sequence payload overflow")?;
        ensure!(seq_end <= map.len(), "sequence payload beyond file");
        let vocab_offset = usize::try_from(u64_at(&map, H_VOCAB))?;
        let node_stats_offset = usize::try_from(u64_at(&map, H_NODE_STATS))?;
        ensure!(
            vocab_offset as u64 == align64(seq_end as u64),
            "invalid vocabulary section"
        );
        let vocab_end = vocab_offset
            .checked_add(references * 2)
            .context("vocabulary section overflow")?;
        ensure!(
            node_stats_offset as u64 == align64(vocab_end as u64),
            "invalid node statistics section"
        );
        let node_stats_end = node_stats_offset
            .checked_add(nodes * NODE_STAT)
            .context("node statistics overflow")?;
        ensure!(node_stats_end <= map.len(), "node statistics beyond file");
        ensure!(
            payload as u64 == align64(node_stats_end as u64) && payload <= map.len(),
            "invalid payload section"
        );
        let mut next = payload as u64;
        for w in 0..WORDS {
            let p = HEADER + w * ENTRY;
            let (off, len, count, kind) = (
                u64_at(&map, p),
                u64_at(&map, p + 8),
                u32_at(&map, p + 16) as u64,
                map[p + 20],
            );
            let layout = Layout::from_code(kind);
            ensure!(
                count <= references as u64 && layout.is_some(),
                "invalid directory entry {w}"
            );
            let expected = layout
                .expect("checked above")
                .len(references, count as usize);
            ensure!(off == next && len == expected, "invalid row layout {w}");
            next = off.checked_add(len).context("row offset overflow")?;
            ensure!(next <= map.len() as u64, "row {w} extends beyond file");
        }
        ensure!(next == map.len() as u64, "trailing index data");
        let result = Self {
            map,
            references,
            nodes,
            refs_offset,
            nodes_offset,
            strings_offset,
            seq_dir,
            seq_payload,
            seq_len,
            vocab_offset,
            node_stats_offset,
            validated: (0..WORDS).map(|_| AtomicBool::new(false)).collect(),
        };
        for node in 0..nodes {
            let p = nodes_offset + node * NODE;
            let parent = u32_at(&result.map, p) as usize;
            let rank = result.map[p + 4];
            let off = u64_at(&result.map, p + 8);
            let len = u32_at(&result.map, p + 16) as u64;
            ensure!(
                off.checked_add(len)
                    .is_some_and(|end| end <= strings_len as u64),
                "invalid taxon string"
            );
            if node == 0 {
                ensure!(parent == 0 && rank == 255 && len == 0, "invalid root");
            } else {
                ensure!(parent < node && rank < 7, "invalid taxonomy node");
                ensure!(
                    if parent == 0 {
                        rank == 0
                    } else {
                        result.node_rank(parent as u32) + 1 == rank
                    },
                    "invalid taxonomy ancestry"
                );
            }
            let name = std::str::from_utf8(result.name_bytes(node as u32))
                .context("invalid taxon UTF-8")?;
            ensure!(
                !name.chars().any(char::is_control) && !name.contains([',', ';']),
                "invalid taxon name"
            );
        }
        for reference in 0..references {
            let leaf = result.leaf(reference);
            ensure!(
                (leaf as usize) < nodes && result.node_rank(leaf) == 6,
                "invalid reference lineage"
            );
        }
        Ok(result)
    }
    pub fn row(&self, word: u16) -> Result<Row<'_>> {
        Ok(self.row_at(self.locate(word)?))
    }

    /// The row a [`RowRef`] from [`Index::locate`] names: one bounds-checked
    /// slice, no directory read and nothing to validate again.
    pub fn row_at(&self, at: RowRef) -> Row<'_> {
        Row {
            data: &self.map[at.offset..at.offset + at.len],
            layout: at.layout,
            count: at.count,
        }
    }

    /// [`Index::row`]'s directory read and first-touch validation, returning
    /// where the row lies rather than the row itself; see [`RowRef`].
    pub fn locate(&self, word: u16) -> Result<RowRef> {
        let p = HEADER + word as usize * ENTRY;
        let off = u64_at(&self.map, p) as usize;
        let len = u64_at(&self.map, p + 8) as usize;
        let row = Row {
            data: &self.map[off..off + len],
            // `open` has already rejected any code outside the three layouts.
            layout: Layout::from_code(self.map[p + 20]).unwrap_or(Layout::Postings),
            count: u32_at(&self.map, p + 16) as usize,
        };
        // Validate payload structure once, lazily, before kernels use IDs.
        if !self.validated[word as usize].load(Ordering::Acquire) {
            match row.layout {
                Layout::Bitmap => {
                    let ones: usize = row.data.iter().map(|b| b.count_ones() as usize).sum();
                    ensure!(
                        ones == row.count,
                        "bitmap cardinality mismatch for word {word}"
                    );
                    if !self.references.is_multiple_of(8) {
                        ensure!(
                            row.data.last().unwrap() >> (self.references % 8) == 0,
                            "nonzero bitmap tail"
                        );
                    }
                }
                Layout::Postings => {
                    let mut prev = None;
                    for i in 0..row.count {
                        let id = row.id(i);
                        ensure!(
                            id < self.references && prev.is_none_or(|v| id > v),
                            "invalid posting for word {word}"
                        );
                        prev = Some(id);
                    }
                }
                // The blocked kernel indexes the tile's count window directly
                // with the stored offset and never compares it to anything, so
                // every offset has to be inside its tile before a kernel runs.
                // Checking the reconstructed id against `references` covers the
                // final short tile as well.
                //
                // Offsets strictly increasing, with the last inside the tile,
                // is the same condition as every offset being inside it. It is
                // checked as a branch-free fold over neighbouring pairs rather
                // than a comparison and a branch per posting: a row's first
                // touch validates every posting it holds, and on a short
                // EUKARYOME run the per-posting form was a seventh of all
                // cycles.
                Layout::Blocked => {
                    let tiles = self.references.div_ceil(TILE);
                    let (mut at, mut total) = (0usize, 0usize);
                    for tile in 0..tiles {
                        ensure!(at + 2 <= row.data.len(), "truncated block for word {word}");
                        let block = u16_at(row.data, at) as usize;
                        at += 2;
                        ensure!(
                            at + block * 2 <= row.data.len(),
                            "truncated block for word {word}"
                        );
                        let (postings, _) = row.data[at..at + block * 2].as_chunks::<2>();
                        if let Some(last) = postings.last() {
                            let increasing = postings
                                .iter()
                                .zip(&postings[1..])
                                .fold(true, |ok, (a, b)| {
                                    ok & (u16::from_le_bytes(*a) < u16::from_le_bytes(*b))
                                });
                            let limit = TILE.min(self.references - tile * TILE);
                            ensure!(
                                increasing && usize::from(u16::from_le_bytes(*last)) < limit,
                                "invalid posting for word {word}"
                            );
                        }
                        at += block * 2;
                        total += block;
                    }
                    ensure!(
                        at == row.data.len() && total == row.count,
                        "block cardinality mismatch for word {word}"
                    );
                }
            }
            self.validated[word as usize].store(true, Ordering::Release);
        }
        Ok(RowRef {
            offset: off,
            len,
            count: row.count,
            layout: row.layout,
        })
    }
    /// Bounds-checked view of one reference's packed-sequence record.
    ///
    /// Records are self-describing (base count, then ambiguous positions, then
    /// two-bit bases) so the directory is one offset per reference. Validation
    /// is per access rather than a startup sweep: the directory alone is 8 bytes
    /// per reference, and touching all of it would fault in pages an ordinary
    /// run never reads.
    fn record(&self, reference: usize) -> Result<(u32, &[u8], &[u8])> {
        ensure!(reference < self.references, "reference out of range");
        let dir = self.seq_dir + reference * 8;
        let start = usize::try_from(u64_at(&self.map, dir))?;
        let end = usize::try_from(u64_at(&self.map, dir + 8))?;
        ensure!(
            start <= end && end <= self.seq_len,
            "corrupt sequence directory"
        );
        let record = &self.map[self.seq_payload + start..self.seq_payload + end];
        ensure!(record.len() >= 8, "truncated sequence record");
        let bases = u32_at(record, 0);
        let ambiguous = u32_at(record, 4) as usize;
        let positions = 8usize
            .checked_add(ambiguous.checked_mul(4).context("ambiguity overflow")?)
            .context("ambiguity overflow")?;
        ensure!(positions <= record.len(), "truncated ambiguity list");
        let packed = &record[positions..];
        ensure!(
            packed.len() >= (bases as usize).div_ceil(4),
            "truncated packed sequence"
        );
        Ok((bases, &record[8..positions], packed))
    }

    /// Every k-mer of one reference, in sequence order, duplicates included.
    ///
    /// Deduplicating would cost an 8 KiB bitset clear per reference, which at
    /// a thousand candidates a query dominates everything else in the
    /// projection. Callers that only set membership bits do not care: setting
    /// the same bit twice is idempotent.
    pub fn visit_kmers(&self, reference: usize, visit: impl FnMut(u16)) -> Result<()> {
        let (bases, ambiguous, packed) = self.record(reference)?;
        walk_kmers(bases as usize, ambiguous, packed, visit);
        Ok(())
    }

    /// Distinct k-mers of one reference, in ascending order, via `visit`.
    ///
    /// This reproduces `sequence::unique_words` exactly, including the rule that
    /// an ambiguous base breaks the window rather than being skipped over. The
    /// scratch bitset is caller-owned so a scan over many candidates allocates
    /// nothing.
    pub fn reference_words(
        &self,
        reference: usize,
        seen: &mut [u64; WORDS / 64],
        mut visit: impl FnMut(u16),
    ) -> Result<()> {
        let (bases, ambiguous, packed) = self.record(reference)?;
        seen.fill(0);
        walk_kmers(bases as usize, ambiguous, packed, |word| {
            seen[word as usize / 64] |= 1 << (word % 64);
        });
        for (i, bits) in seen.iter().enumerate() {
            let mut bits = *bits;
            while bits != 0 {
                visit((i * 64 + bits.trailing_zeros() as usize) as u16);
                bits &= bits - 1;
            }
        }
        Ok(())
    }

    pub fn verify(&self) -> Result<()> {
        ensure!(
            &self.map[96..128] == blake3::hash(&self.map[HEADER..]).as_bytes(),
            "payload checksum mismatch"
        );
        for word in 0..WORDS {
            self.row(word as u16)?;
        }
        // Every sequence record, and the agreement between the two
        // representations of membership. The forward profile and the inverted
        // index are written from the same extraction, so a disagreement means
        // corruption rather than a logic error -- but it is exactly the kind of
        // corruption a payload checksum would not localise.
        let mut seen = [0u64; WORDS / 64];
        let mut counts = vec![0u32; WORDS];
        for reference in 0..self.references {
            let mut distinct = 0u32;
            self.reference_words(reference, &mut seen, |word| {
                counts[word as usize] += 1;
                distinct += 1;
            })?;
            ensure!(
                distinct.min(u32::from(u16::MAX)) == self.vocabulary(reference),
                "stored vocabulary size is wrong for reference {reference}"
            );
        }
        for (word, &count) in counts.iter().enumerate() {
            ensure!(
                count == self.row(word as u16)?.count as u32,
                "sequence and posting sections disagree on word {word}"
            );
        }
        // The taxon counts are recomputed from the postings, which the check
        // above has just tied back to the packed sequences.
        let (genera, families) = sweep_taxa(
            &self.map,
            self.references,
            self.refs_offset,
            self.nodes_offset,
            self.nodes,
        );
        for word in 0..WORDS {
            ensure!(
                genera[word] == self.genus_frequency(word as u16)
                    && families[word] == self.family_frequency(word as u16),
                "directory and postings disagree on taxon counts for word {word}"
            );
        }
        // The node aggregates roll up from the same leaf table and parent links
        // `open` has already checked, so a disagreement here is corruption of
        // the stored section rather than a tree that does not add up.
        let parents: Vec<u32> = (0..self.nodes).map(|n| self.parent(n as u32)).collect();
        let ranks: Vec<u8> = (0..self.nodes).map(|n| self.node_rank(n as u32)).collect();
        let named: Vec<bool> = (0..self.nodes)
            .map(|n| !self.name(n as u32).is_empty())
            .collect();
        let stats = NodeStats::compute(
            (0..self.references).map(|r| self.leaf(r)),
            &parents,
            &ranks,
            &named,
        );
        for node in 0..self.nodes {
            let n = node as u32;
            ensure!(
                stats.references[node] == self.node_references(n)
                    && stats.children[node] == self.node_children(n)
                    && stats.genera[node] == self.node_genera(n),
                "stored node statistics are wrong for node {node}"
            );
        }
        Ok(())
    }
    pub fn format_version(&self) -> u32 {
        FORMAT_VERSION
    }
    /// The algorithm version recorded in this file, which is not necessarily
    /// the one this binary was compiled with -- `open` rejects a mismatch, so
    /// they agree in practice, but reporting the binary's constant would hide a
    /// loader bug rather than expose it.
    pub fn algorithm_version(&self) -> u32 {
        u32_at(&self.map, 12)
    }
    /// References carrying a name at each stored rank, `d` through `g`.
    ///
    /// A reference database whose headers do not match the SINTAX `,` rank
    /// separator parses as its first rank and nothing else, which is a silent
    /// failure: the build succeeds and every call comes back at kingdom. This
    /// census turns that into something visible at `index` and `inspect` time.
    pub fn rank_census(&self) -> [u64; 7] {
        let mut census = [0u64; 7];
        for reference in 0..self.references {
            for (rank, node) in self.lineage(reference).into_iter().enumerate() {
                if !self.name_bytes(node).is_empty() {
                    census[rank] += 1;
                }
            }
        }
        census
    }
    /// Bytes held by the packed reference sequences, directory included.
    pub fn sequence_bytes(&self) -> usize {
        self.seq_len + (self.references + 1) * 8
    }
    /// Mean bases per reference, estimated from the packed payload alone: each
    /// record is an 8-byte header, its ambiguity list, and four bases a byte.
    /// Ambiguity lists are ignored, so this errs high on messy databases. A
    /// cost estimate for choosing between exact-equivalent strategies, never
    /// an input to scoring.
    pub fn mean_bases(&self) -> usize {
        let packed = self.seq_len.saturating_sub(self.references * 8);
        (packed * 4).checked_div(self.references).unwrap_or(0)
    }
    pub fn file_bytes(&self) -> usize {
        self.map.len()
    }
    pub fn source_hash(&self) -> String {
        self.map[128..160]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
    /// References containing `word`, read straight from the directory.
    ///
    /// Deliberately not `row(word)?.count`: `row` lazily validates a word's
    /// payload the first time it is touched, and sweeping all 65,536 words
    /// through it would fault in every posting row to build a 64 KiB table.
    pub fn document_frequency(&self, word: u16) -> u32 {
        u32_at(&self.map, HEADER + word as usize * ENTRY + 16)
    }
    /// Distinct genera containing `word`.
    ///
    /// The weight source to prefer on a database whose copy numbers track
    /// sequencing effort rather than biology: duplicating a genus's records
    /// cannot change how many genera contain a word, so this is exactly
    /// invariant to the redundancy `df` is fooled by.
    pub fn genus_frequency(&self, word: u16) -> u32 {
        u32_at(&self.map, HEADER + word as usize * ENTRY + 24)
    }
    /// Distinct families containing `word`.
    pub fn family_frequency(&self, word: u16) -> u32 {
        u32_at(&self.map, HEADER + word as usize * ENTRY + 28)
    }
    pub fn rows_with(&self, layout: Layout) -> usize {
        (0..WORDS)
            .filter(|w| self.map[HEADER + w * ENTRY + 20] == layout.code())
            .count()
    }
    pub fn dense_rows(&self) -> usize {
        self.rows_with(Layout::Bitmap)
    }
    pub fn leaf(&self, reference: usize) -> u32 {
        u32_at(&self.map, self.refs_offset + reference * 4)
    }
    pub fn parent(&self, node: u32) -> u32 {
        u32_at(&self.map, self.nodes_offset + node as usize * NODE)
    }
    pub fn node_rank(&self, node: u32) -> u8 {
        self.map[self.nodes_offset + node as usize * NODE + 4]
    }
    /// A taxon's name as stored, without the UTF-8 check [`Index::name`]
    /// repeats on every call. `open` has validated every name already, and
    /// byte order is `str` order, so this is what comparisons should use.
    pub fn name_bytes(&self, node: u32) -> &[u8] {
        let p = self.nodes_offset + node as usize * NODE;
        let off = self.strings_offset + u64_at(&self.map, p + 8) as usize;
        let len = u32_at(&self.map, p + 16) as usize;
        &self.map[off..off + len]
    }
    pub fn name(&self, node: u32) -> &str {
        std::str::from_utf8(self.name_bytes(node)).expect("validated taxonomy")
    }
    /// Distinct k-mers held by one reference, saturating at `u16::MAX`.
    ///
    /// The chance that a reference contains an arbitrary word is
    /// `vocabulary(r) / WORDS`, which is what the `protax` scorer subtracts
    /// before turning a containment into an identity estimate. A 12 kb protist
    /// operon holds 27.9% of the whole 8-mer space and matches anything by
    /// accident; this is the number that says so.
    ///
    /// Saturation is harmless because `WORDS` is 65,536 and a reference holding
    /// every word would be indistinguishable from one holding all but one.
    pub fn vocabulary(&self, reference: usize) -> u32 {
        let p = self.vocab_offset + reference * 2;
        u32::from(u16::from_le_bytes(self.map[p..p + 2].try_into().unwrap()))
    }
    /// References anywhere under `node`, including the node's own leaves.
    pub fn node_references(&self, node: u32) -> u32 {
        u32_at(
            &self.map,
            self.node_stats_offset + node as usize * NODE_STAT,
        )
    }
    /// Immediate **named** children of `node`. A node's unnamed sentinel is
    /// not counted: it records an absence of annotation, not a rival taxon.
    pub fn node_children(&self, node: u32) -> u32 {
        u32_at(
            &self.map,
            self.node_stats_offset + node as usize * NODE_STAT + 4,
        )
    }
    /// Genus-rank nodes under `node`; one if `node` is itself a genus.
    pub fn node_genera(&self, node: u32) -> u32 {
        u32_at(
            &self.map,
            self.node_stats_offset + node as usize * NODE_STAT + 8,
        )
    }
    pub fn lineage(&self, reference: usize) -> [u32; 7] {
        let mut lineage = [0; 7];
        let mut node = self.leaf(reference);
        for slot in lineage.iter_mut().rev() {
            *slot = node;
            node = self.parent(node);
        }
        lineage
    }
}

/// Resolves a written lineage back to the nodes that store it.
///
/// The index is an upward structure -- every node knows its parent, nothing
/// knows its children -- because classification only ever walks from a matched
/// reference toward the root. Training walks the other way: it needs the node
/// a query's *own* annotation names, so that the design matrix is built under
/// the true parent rather than under whichever parent the model preferred.
///
/// One pass over the node table, keyed the way `TreeBuilder` interns. Built
/// only when a training flag asks for it; an ordinary run never pays for it.
pub struct Locator {
    nodes: std::collections::HashMap<(u32, u8, Box<str>), u32>,
}

impl Locator {
    pub fn build(index: &Index) -> Self {
        let mut nodes = std::collections::HashMap::with_capacity(index.nodes);
        for node in 1..index.nodes as u32 {
            nodes.insert(
                (
                    index.parent(node),
                    index.node_rank(node),
                    index.name(node).into(),
                ),
                node,
            );
        }
        Self { nodes }
    }
    /// The node at each rank the lineage names, as far as the tree agrees.
    ///
    /// Stops at the first rank the index does not hold: a name that is not
    /// under the parent the ranks above resolved to is not the same taxon,
    /// whatever it is called elsewhere, and interning by `(parent, rank, name)`
    /// is what makes that distinction real.
    pub fn locate(&self, lineage: &crate::taxonomy::Lineage) -> [Option<u32>; 7] {
        let mut out = [None; 7];
        let mut parent = 0u32;
        for (rank, name) in lineage.iter().enumerate() {
            let name = name.as_deref().unwrap_or("");
            match self.nodes.get(&(parent, rank as u8, name.into())) {
                Some(&node) => {
                    out[rank] = Some(node);
                    parent = node;
                }
                None => break,
            }
        }
        out
    }
}

/// Disk-backed staging keeps incidence lists out of the builder's heap.
/// Temporary disk is about 6 bytes per (reference, distinct word) pair.
/// Two-bit pack one normalized sequence into a self-describing record.
///
/// Layout: base count, ambiguity count, the ambiguous positions ascending, then
/// the bases four to a byte. Ambiguous bases are stored as A and listed, because
/// what the kernel needs from them is only that they break the k-mer window --
/// which is exactly what `sequence::unique_words` does with them.
///
/// Storing sequences rather than word lists costs roughly a fifth of the space
/// (136 versus 700 bytes per reference on rdp_its) for comparable decode work,
/// since probing the query lookup table dominates either way.
fn pack_sequence(seq: &[u8]) -> Result<Vec<u8>> {
    let bases = u32::try_from(seq.len()).context("reference sequence too long")?;
    let mut ambiguous = Vec::new();
    let mut packed = vec![0u8; seq.len().div_ceil(4)];
    for (i, &base) in seq.iter().enumerate() {
        let code = match base {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => {
                ambiguous.push(i as u32);
                0
            }
        };
        packed[i / 4] |= code << ((i % 4) * 2);
    }
    let mut out = Vec::with_capacity(8 + ambiguous.len() * 4 + packed.len());
    out.extend_from_slice(&bases.to_le_bytes());
    out.extend_from_slice(&u32::try_from(ambiguous.len())?.to_le_bytes());
    for position in &ambiguous {
        out.extend_from_slice(&position.to_le_bytes());
    }
    out.extend_from_slice(&packed);
    Ok(out)
}

pub fn build(reference: &Path, output: &Path) -> Result<()> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let tempdir = tempfile::Builder::new()
        .prefix(".sintaxer-build-")
        .tempdir_in(parent)?;
    let mut buckets = Vec::new();
    for i in 0..256 {
        buckets.push(BufWriter::with_capacity(
            64 * 1024,
            File::create(tempdir.path().join(i.to_string()))?,
        ));
    }
    let mut counts = vec![0u32; WORDS];
    let (mut unknown_bases, mut unknown_records) = (0usize, 0usize);
    // Packed reference sequences are staged to disk alongside the incidence
    // buckets so the builder never holds the whole database in memory.
    let mut sequences =
        BufWriter::with_capacity(1024 * 1024, File::create(tempdir.path().join("sequences"))?);
    let mut seq_offsets: Vec<u64> = vec![0];
    let mut seq_total: u64 = 0;
    let mut vocabulary: Vec<u16> = Vec::new();
    let mut leaves = Vec::new();
    let mut tree = TreeBuilder::default();
    let mut source = blake3::Hasher::new();
    source.update(b"sintaxer/reference/v1\0");
    let mut reader = input::reader(reference)?.context("empty reference database")?;
    while let Some(record) = reader.next() {
        let record = record.context("reading reference FASTA")?;
        ensure!(
            record.format() == needletail::parser::Format::Fasta,
            "references must be FASTA"
        );
        let label = input::label(record.id())?;
        let tax = crate::taxonomy::parse(&label).with_context(|| format!("reference {label:?}"))?;
        let (seq, unknown) = sequence::normalize(&record.seq());
        unknown_bases += unknown;
        unknown_records += usize::from(unknown > 0);
        let words = sequence::unique_words(&seq);
        vocabulary.push(u16::try_from(words.len()).unwrap_or(u16::MAX));
        let id = u32::try_from(leaves.len()).context("too many reference sequences")?;
        ensure!(id < u32::MAX, "reference count exceeds format limit");
        leaves.push(tree.insert(tax)?);
        source.update(&(label.len() as u64).to_le_bytes());
        source.update(label.as_bytes());
        source.update(&(seq.len() as u64).to_le_bytes());
        source.update(&seq);
        for word in words {
            counts[word as usize] += 1;
            let bucket = &mut buckets[word as usize / 256];
            bucket.write_all(&word.to_le_bytes())?;
            bucket.write_all(&id.to_le_bytes())?;
        }
        let record = pack_sequence(&seq)?;
        sequences.write_all(&record)?;
        seq_total += record.len() as u64;
        seq_offsets.push(seq_total);
    }
    sequences.flush()?;
    drop(sequences);
    ensure!(!leaves.is_empty(), "empty reference database");
    if unknown_bases > 0 {
        eprintln!(
            "warning: {unknown_bases} unrecognised sequence bytes in {unknown_records} references \
             treated as ambiguous"
        );
    }
    for b in &mut buckets {
        b.flush()?;
    }
    drop(buckets);
    let n = leaves.len();
    let refs_offset = (HEADER + WORDS * ENTRY) as u64;
    let nodes_offset = align64(refs_offset + n as u64 * 4);
    let strings_offset = align64(nodes_offset + tree.nodes.len() as u64 * NODE as u64);
    let strings_len: u64 = tree.nodes.iter().map(|n| n.name.len() as u64).sum();
    let seq_dir = align64(strings_offset + strings_len);
    let seq_payload = align64(seq_dir + (n as u64 + 1) * 8);
    let vocab_offset = align64(seq_payload + seq_total);
    let node_stats_offset = align64(vocab_offset + n as u64 * 2);
    let payload = align64(node_stats_offset + tree.nodes.len() as u64 * NODE_STAT as u64);
    let mut offsets = vec![0u64; WORDS];
    let mut layouts = vec![Layout::Postings; WORDS];
    let mut total = payload;
    for w in 0..WORDS {
        offsets[w] = total;
        layouts[w] = Layout::best(n, counts[w] as usize);
        total = total
            .checked_add(layouts[w].len(n, counts[w] as usize))
            .context("index too large")?;
    }
    let mut out = tempfile::NamedTempFile::new_in(parent)?;
    out.as_file_mut().set_len(total)?;
    // SAFETY: this newly created private temporary file is exclusively owned.
    #[expect(unsafe_code, reason = "exclusively owned temporary; see SAFETY above")]
    let mut map = unsafe { MmapMut::map_mut(out.as_file())? };
    map[..8].copy_from_slice(MAGIC);
    put32(&mut map, 8, FORMAT_VERSION);
    put32(&mut map, 12, ALGORITHM_VERSION);
    put32(&mut map, 16, n as u32);
    put32(&mut map, 20, u32::try_from(tree.nodes.len())?);
    for (p, v) in [
        (24, HEADER as u64),
        (32, refs_offset),
        (40, nodes_offset),
        (48, strings_offset),
        (56, payload),
        (64, total),
        (72, strings_len),
        (H_SEQ_DIR as u64, seq_dir),
        (H_SEQ_PAYLOAD as u64, seq_payload),
        (H_SEQ_LEN as u64, seq_total),
        (H_VOCAB as u64, vocab_offset),
        (H_NODE_STATS as u64, node_stats_offset),
    ] {
        put64(&mut map, p as usize, v);
    }
    put32(&mut map, 80, 1);
    put32(&mut map, 84, 8);
    for w in 0..WORDS {
        let p = HEADER + w * ENTRY;
        put64(&mut map, p, offsets[w]);
        put64(&mut map, p + 8, layouts[w].len(n, counts[w] as usize));
        put32(&mut map, p + 16, counts[w]);
        map[p + 20] = layouts[w].code();
    }
    for (i, &leaf) in leaves.iter().enumerate() {
        put32(&mut map, refs_offset as usize + i * 4, leaf);
    }
    for (i, size) in vocabulary.iter().enumerate() {
        let p = vocab_offset as usize + i * 2;
        map[p..p + 2].copy_from_slice(&size.to_le_bytes());
    }
    {
        let parents: Vec<u32> = tree.nodes.iter().map(|n| n.parent).collect();
        let ranks: Vec<u8> = tree.nodes.iter().map(|n| n.rank).collect();
        let named: Vec<bool> = tree.nodes.iter().map(|n| !n.name.is_empty()).collect();
        let stats = NodeStats::compute(leaves.iter().copied(), &parents, &ranks, &named);
        for node in 0..tree.nodes.len() {
            let p = node_stats_offset as usize + node * NODE_STAT;
            put32(&mut map, p, stats.references[node]);
            put32(&mut map, p + 4, stats.children[node]);
            put32(&mut map, p + 8, stats.genera[node]);
        }
    }
    let mut name_offset = 0;
    for (i, node) in tree.nodes.iter().enumerate() {
        let p = nodes_offset as usize + i * NODE;
        put32(&mut map, p, node.parent);
        map[p + 4] = node.rank;
        put64(&mut map, p + 8, name_offset);
        put32(&mut map, p + 16, u32::try_from(node.name.len())?);
        let start = (strings_offset + name_offset) as usize;
        map[start..start + node.name.len()].copy_from_slice(node.name.as_bytes());
        name_offset += node.name.len() as u64;
    }
    drop(tree);
    for (i, offset) in seq_offsets.iter().enumerate() {
        put64(&mut map, seq_dir as usize + i * 8, *offset);
    }
    {
        let path = tempdir.path().join("sequences");
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(&path)?);
        let start = seq_payload as usize;
        reader.read_exact(&mut map[start..start + seq_total as usize])?;
        drop(reader);
        std::fs::remove_file(path)?;
    }
    // One streaming pass writes all three layouts. Blocked rows need no
    // counting pass first: the file is zero-filled, empty tiles therefore
    // already carry a zero count, and because a word's postings arrive in
    // ascending reference order the position of the next one is always known.
    // `cursors` is the byte offset of the next id, `blocks` the offset of the
    // count belonging to the tile being filled, and `tiles` that tile's index.
    let mut cursors = offsets.clone();
    let mut blocks = vec![0u64; WORDS];
    let mut tiles = vec![usize::MAX; WORDS];
    for bucket in 0..256 {
        let path = tempdir.path().join(bucket.to_string());
        let file = File::open(&path)?;
        let records = file.metadata()?.len() / 6;
        let mut reader = BufReader::with_capacity(1024 * 1024, file);
        let mut pair = [0; 6];
        for _ in 0..records {
            reader.read_exact(&mut pair)?;
            let w = u16::from_le_bytes(pair[..2].try_into().unwrap()) as usize;
            let id = u32::from_le_bytes(pair[2..].try_into().unwrap()) as usize;
            match layouts[w] {
                Layout::Bitmap => map[offsets[w] as usize + id / 8] |= 1 << (id % 8),
                Layout::Postings => {
                    put32(&mut map, cursors[w] as usize, id as u32);
                    cursors[w] += 4;
                }
                Layout::Blocked => {
                    let tile = id / TILE;
                    if tiles[w] != tile {
                        // Skip the count slots of the tiles in between, which
                        // are empty and already zero.
                        let skipped = tile.wrapping_sub(tiles[w].wrapping_add(1));
                        blocks[w] = cursors[w] + skipped as u64 * 2;
                        cursors[w] = blocks[w] + 2;
                        tiles[w] = tile;
                    }
                    put16(&mut map, cursors[w] as usize, (id % TILE) as u16);
                    cursors[w] += 2;
                    let at = blocks[w] as usize;
                    let filled = u16_at(&map, at) + 1;
                    put16(&mut map, at, filled);
                }
            }
        }
        drop(reader);
        std::fs::remove_file(path)?;
    }
    let nodes = u32_at(&map, 20) as usize;
    let (genera, families) =
        sweep_taxa(&map, n, refs_offset as usize, nodes_offset as usize, nodes);
    for w in 0..WORDS {
        let p = HEADER + w * ENTRY;
        put32(&mut map, p + 24, genera[w]);
        put32(&mut map, p + 28, families[w]);
    }
    let digest = blake3::hash(&map[HEADER..]);
    map[96..128].copy_from_slice(digest.as_bytes());
    map[128..160].copy_from_slice(source.finalize().as_bytes());
    let header_hash = blake3::hash(&map[..224]);
    map[224..256].copy_from_slice(header_hash.as_bytes());
    map.flush()?;
    drop(map);
    out.as_file().sync_all()?;
    out.persist(output)
        .map_err(|e| e.error)
        .context("publishing index")?;
    File::open(parent)?.sync_all()?;
    Ok(())
}
