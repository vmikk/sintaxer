//! Immutable, memory-mapped STX2 index.
use crate::{ALGORITHM_VERSION, K, WORDS, input, sequence, taxonomy::TreeBuilder};
use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapMut};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const HEADER: usize = 256;
const ENTRY: usize = 24;
const NODE: usize = 24;
const MAGIC: &[u8; 8] = b"SINTAXER";
const FORMAT_VERSION: u32 = 2;

// Header slots for the packed sequence sections.
const H_SEQ_DIR: usize = 160;
const H_SEQ_PAYLOAD: usize = 168;
const H_SEQ_LEN: usize = 176;
fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
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
    validated: Vec<AtomicBool>,
}

#[derive(Clone, Copy)]
pub struct Row<'a> {
    data: &'a [u8],
    pub dense: bool,
    pub count: usize,
}
impl Row<'_> {
    /// A scratch bitmap produced by the exact projection engine.
    pub fn bitmap(data: &[u8]) -> Row<'_> {
        Row {
            data,
            dense: true,
            count: 0,
        }
    }
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
    pub fn visit(&self, start: usize, end: usize, mut f: impl FnMut(usize)) {
        if self.dense {
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
        } else {
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
    }
    /// Membership bits for 64 references; `start` must be a multiple of 64.
    pub fn bits64(&self, start: usize) -> u64 {
        debug_assert_eq!(start % 64, 0);
        if self.dense {
            let off = start / 8;
            if off >= self.data.len() {
                return 0;
            }
            let mut bytes = [0; 8];
            let n = (self.data.len() - off).min(8);
            bytes[..n].copy_from_slice(&self.data[off..off + n]);
            u64::from_le_bytes(bytes)
        } else {
            let mut bits = 0;
            let mut i = self.lower_bound(start);
            while i < self.count {
                let id = self.id(i);
                if id >= start + 64 {
                    break;
                }
                bits |= 1 << (id - start);
                i += 1;
            }
            bits
        }
    }
    pub fn dense256(&self, start: usize) -> Option<&[u8]> {
        if self.dense {
            self.data.get(start / 8..start / 8 + 32)
        } else {
            None
        }
    }
}

impl Index {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening index {}", path.display()))?;
        ensure!(
            file.metadata()?.len() >= HEADER as u64,
            "truncated index header"
        );
        // SAFETY: read-only mapping. Published indexes are never modified in place;
        // the builder replaces them atomically with a new inode.
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
        ensure!(
            payload as u64 == align64(seq_end as u64) && payload <= map.len(),
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
            ensure!(
                count <= references as u64 && kind <= 1,
                "invalid directory entry {w}"
            );
            let expected = if kind == 1 {
                (references as u64).div_ceil(8)
            } else {
                count * 4
            };
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
        let p = HEADER + word as usize * ENTRY;
        let off = u64_at(&self.map, p) as usize;
        let len = u64_at(&self.map, p + 8) as usize;
        let row = Row {
            data: &self.map[off..off + len],
            dense: self.map[p + 20] == 1,
            count: u32_at(&self.map, p + 16) as usize,
        };
        // Validate the payload once, lazily, before kernels use its IDs.
        if !self.validated[word as usize].load(Ordering::Acquire) {
            if row.dense {
                let ones: usize = row.data.iter().map(|b| b.count_ones() as usize).sum();
                ensure!(
                    ones == row.count,
                    "bitmap cardinality mismatch for word {word}"
                );
                if self.references % 8 != 0 {
                    ensure!(
                        row.data.last().unwrap() >> (self.references % 8) == 0,
                        "nonzero bitmap tail"
                    );
                }
            } else {
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
            self.validated[word as usize].store(true, Ordering::Release);
        }
        Ok(row)
    }
    /// Bounds-checked view of one reference's packed-sequence record.
    ///
    /// Validated per access rather than at startup, so a run never faults in
    /// directory pages it does not use.
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

    /// Distinct k-mers of one reference, in ascending order, via `visit`.
    ///
    /// Matches `sequence::unique_words` exactly: an ambiguous base breaks the
    /// window. `seen` is caller-owned scratch so repeated calls don't allocate.
    pub fn reference_words(
        &self,
        reference: usize,
        seen: &mut [u64; WORDS / 64],
        mut visit: impl FnMut(u16),
    ) -> Result<()> {
        let (bases, ambiguous, packed) = self.record(reference)?;
        seen.fill(0);
        let mut next_break = usize::MAX;
        let mut breaks = ambiguous.chunks_exact(4).map(|b| u32_at(b, 0) as usize);
        if let Some(first) = breaks.next() {
            next_break = first;
        }
        let (mut word, mut run) = (0u16, 0usize);
        for position in 0..bases as usize {
            if position == next_break {
                word = 0;
                run = 0;
                next_break = breaks.next().unwrap_or(usize::MAX);
                continue;
            }
            let code = (packed[position / 4] >> ((position % 4) * 2)) & 3;
            word = (word << 2) | u16::from(code);
            run = (run + 1).min(K);
            if run == K {
                seen[word as usize / 64] |= 1 << (word % 64);
            }
        }
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
        // Check every sequence record, and that the forward profile agrees with the
        // inverted index. Both come from the same extraction, so a mismatch means corruption.
        let mut seen = [0u64; WORDS / 64];
        let mut counts = vec![0u32; WORDS];
        for reference in 0..self.references {
            self.reference_words(reference, &mut seen, |word| {
                counts[word as usize] += 1;
            })?;
        }
        for (word, &count) in counts.iter().enumerate() {
            ensure!(
                count == self.row(word as u16)?.count as u32,
                "sequence and posting sections disagree on word {word}"
            );
        }
        Ok(())
    }
    pub fn format_version(&self) -> u32 {
        FORMAT_VERSION
    }
    /// Bytes held by the packed reference sequences, directory included.
    pub fn sequence_bytes(&self) -> usize {
        self.seq_len + (self.references + 1) * 8
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
    pub fn dense_rows(&self) -> usize {
        (0..WORDS)
            .filter(|w| self.map[HEADER + w * ENTRY + 20] == 1)
            .count()
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
    fn name_bytes(&self, node: u32) -> &[u8] {
        let p = self.nodes_offset + node as usize * NODE;
        let off = self.strings_offset + u64_at(&self.map, p + 8) as usize;
        let len = u32_at(&self.map, p + 16) as usize;
        &self.map[off..off + len]
    }
    pub fn name(&self, node: u32) -> &str {
        std::str::from_utf8(self.name_bytes(node)).expect("validated taxonomy")
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

/// Two-bit packs one normalized sequence into a self-describing record.
///
/// Layout: base count, ambiguity count, ambiguous positions (ascending), then
/// bases four per byte. Ambiguous bases are stored as A; they only break the k-mer window.
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
    // Packed sequences are staged to disk alongside the incidence buckets.
    let mut sequences =
        BufWriter::with_capacity(1024 * 1024, File::create(tempdir.path().join("sequences"))?);
    let mut seq_offsets: Vec<u64> = vec![0];
    let mut seq_total: u64 = 0;
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
        let seq =
            sequence::normalize(&record.seq()).with_context(|| format!("reference {label:?}"))?;
        let words = sequence::unique_words(&seq);
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
    let payload = align64(seq_payload + seq_total);
    let mut offsets = vec![0u64; WORDS];
    let mut dense = vec![false; WORDS];
    let mut total = payload;
    for w in 0..WORDS {
        offsets[w] = total;
        dense[w] = (n as u64).div_ceil(8) < counts[w] as u64 * 4;
        total = total
            .checked_add(if dense[w] {
                (n as u64).div_ceil(8)
            } else {
                counts[w] as u64 * 4
            })
            .context("index too large")?;
    }
    let mut out = tempfile::NamedTempFile::new_in(parent)?;
    out.as_file_mut().set_len(total)?;
    // SAFETY: a freshly created private temporary file that we own exclusively.
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
    ] {
        put64(&mut map, p as usize, v);
    }
    put32(&mut map, 80, 1);
    put32(&mut map, 84, 8);
    for w in 0..WORDS {
        let p = HEADER + w * ENTRY;
        put64(&mut map, p, offsets[w]);
        put64(
            &mut map,
            p + 8,
            if dense[w] {
                (n as u64).div_ceil(8)
            } else {
                counts[w] as u64 * 4
            },
        );
        put32(&mut map, p + 16, counts[w]);
        map[p + 20] = u8::from(dense[w]);
    }
    for (i, leaf) in leaves.into_iter().enumerate() {
        put32(&mut map, refs_offset as usize + i * 4, leaf);
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
    let mut cursors = offsets.clone();
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
            if dense[w] {
                map[offsets[w] as usize + id / 8] |= 1 << (id % 8);
            } else {
                put32(&mut map, cursors[w] as usize, id as u32);
                cursors[w] += 4;
            }
        }
        drop(reader);
        std::fs::remove_file(path)?;
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
