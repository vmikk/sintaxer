//! Reference database curation: reduce each stored lineage to a few diverse
//! representatives so that copy counts stop acting as a prior. Two passes
//! (survey, then emit); query scoring is never affected.

use crate::{
    WORDS, input,
    sequence::{self},
    sketch::{self, Sketch},
    taxonomy::{self, TreeBuilder},
};
use anyhow::{Context, Result, ensure};
use std::{io::Write, path::Path};

/// Sketch width. Coarse for measuring identity, but enough for ranking
/// candidates by dissimilarity.
pub const SKETCH: usize = 32;

/// Properties of a record itself. Policy thresholds are applied later by
/// [`Policy`], so one survey can serve many policies (see [`save`], [`load`]).
pub mod flag {
    /// Neither family nor genus annotated.
    pub const SHALLOW: u8 = 1 << 0;
    /// Organellar or spike-in decoy (`d:_mitochondrion`, `Unispike*`, ...).
    pub const ORGANELLAR: u8 = 1 << 1;
}

#[derive(Clone, Debug)]
pub struct Policy {
    /// Representatives kept per distinct stored lineage.
    pub cap: usize,
    /// Soft length threshold: flags `DEMOTED`, never drops.
    pub demote_length: usize,
    /// Hard length ceiling. `None` (the default) keeps every length; on an
    /// untrimmed database a ceiling can delete whole genera.
    pub max_length: Option<usize>,
    /// Maximum non-ACGT fraction.
    pub max_ambiguity: f64,
    /// Minimum distinct 8-mers.
    pub min_words: usize,
    /// Sketch distance past which candidates are considered equally diverse.
    pub ceiling: f32,
    /// LSH bands consulted by [`Selector::Cover`]; see [`BANDS`].
    pub cover_bands: usize,
    /// Caps for records with neither family nor genus.
    pub shallow: ShallowCaps,
    /// Validation switches; not meant for production databases.
    pub drop_shallow: bool,
    pub drop_organellar: bool,
}

/// Caps for under-annotated records, by the guard role they play.
///
/// These guard against over-classification, so they are capped, not deleted.
/// They are capped harder because missing ranks collapse into one empty node
/// per parent and would otherwise vote as a bloc.
#[derive(Clone, Copy, Debug)]
pub struct ShallowCaps {
    /// Order already has genus-annotated members; keep only a token guard.
    pub covered: usize,
    /// Order has no genus coverage; these are its only representatives.
    pub orphan_order: usize,
    /// No order at all: deep-branching environmental diversity, most important
    /// at high ranks.
    pub unplaced: usize,
    /// Also cap `covered` groups relative to their annotated siblings, so a
    /// sparsely annotated order keeps proportionally fewer guards.
    pub ratio: f64,
}

impl Default for ShallowCaps {
    fn default() -> Self {
        Self {
            covered: 50,
            orphan_order: 50,
            unplaced: 50,
            ratio: 0.10,
        }
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            // Sensitivity drops sharply below a few hundred per lineage, genus faster
            // than family.
            cap: 1000,
            demote_length: 2_000,
            max_length: None,
            max_ambiguity: 0.10,
            min_words: 32,
            ceiling: DIVERSITY_CEILING,
            cover_bands: 4,
            shallow: ShallowCaps::default(),
            drop_shallow: false,
            drop_organellar: false,
        }
    }
}

impl Policy {
    /// Longer than the soft length threshold: sorts last among candidates but is
    /// never dropped for it.
    pub fn demoted(&self, summary: &Summary) -> bool {
        summary.length as usize > self.demote_length
    }
    /// Over the hard length ceiling, if one is set.
    pub fn oversize(&self, summary: &Summary) -> bool {
        self.max_length
            .is_some_and(|max| summary.length as usize > max)
    }
    pub fn low_complexity(&self, summary: &Summary) -> bool {
        (summary.words as usize) < self.min_words
    }
    pub fn ambiguous(&self, summary: &Summary) -> bool {
        summary.length > 0
            && f64::from(summary.ambiguous) / f64::from(summary.length) > self.max_ambiguity
    }
    /// Hard rejects, applied before anything takes a cap slot.
    pub fn rejects(&self, summary: &Summary) -> bool {
        self.low_complexity(summary) || self.ambiguous(summary) || self.oversize(summary)
    }
}

/// One record's pass-1 summary: everything selection needs, so that pass 2
/// is a plain byte copy. Policy-free, so it can be cached across runs.
#[derive(Clone, Debug)]
pub struct Summary {
    /// Position in the input, counting records that failed to parse. Pass 2
    /// matches on this.
    pub ordinal: u32,
    /// Node ID of the rank-6 leaf, i.e. the distinct stored lineage.
    pub group: u32,
    pub length: u32,
    pub words: u32,
    pub ambiguous: u32,
    pub digest: [u8; 16],
    pub sketch: Sketch<SKETCH>,
    pub flags: u8,
}

/// A distinct stored lineage and the node IDs of its seven ranks.
#[derive(Clone, Debug)]
pub struct Group {
    pub leaf: u32,
    pub path: [u32; 7],
    pub records: u32,
}

pub struct Survey {
    pub summaries: Vec<Summary>,
    pub tree: TreeBuilder,
    pub groups: Vec<Group>,
    /// Node ID -> index into `groups`, or `u32::MAX`.
    index_of: Vec<u32>,
    pub records: u64,
    pub bases: u64,
    pub unknown_bases: u64,
    pub unknown_records: u64,
    /// Records `taxonomy::parse` rejected. Skipped and counted, not fatal.
    pub parse_failures: Vec<(u32, String)>,
}

impl Survey {
    /// Records per order in genus-annotated lineages beneath it. Single source of
    /// truth shared by the report and the shallow caps.
    pub fn order_coverage(&self) -> std::collections::HashMap<u32, u64> {
        let mut coverage = std::collections::HashMap::new();
        for group in &self.groups {
            if !self.rank_name(group, GENUS).is_empty() {
                *coverage.entry(group.path[ORDER]).or_default() += u64::from(group.records);
            }
        }
        coverage
    }
    /// Guard role this group plays, or `None` if it names a family or genus.
    pub fn shallow_class(
        &self,
        group: &Group,
        coverage: &std::collections::HashMap<u32, u64>,
    ) -> Option<Shallow> {
        if !self.rank_name(group, FAMILY).is_empty() || !self.rank_name(group, GENUS).is_empty() {
            return None;
        }
        Some(if self.rank_name(group, ORDER).is_empty() {
            Shallow::Unplaced
        } else if coverage.contains_key(&group.path[ORDER]) {
            Shallow::Covered
        } else {
            Shallow::OrphanOrder
        })
    }
    pub fn group_of(&self, summary: &Summary) -> &Group {
        &self.groups[self.index_of[summary.group as usize] as usize]
    }
    pub fn name(&self, node: u32) -> &str {
        &self.tree.nodes[node as usize].name
    }
    /// Rank name for a group, `""` when that rank is unannotated.
    pub fn rank_name(&self, group: &Group, rank: usize) -> &str {
        self.name(group.path[rank])
    }
}

const DOMAIN: usize = 0;
const ORDER: usize = 4;
const FAMILY: usize = 5;
const GENUS: usize = 6;

/// Organellar and spike-in decoys are marked with a leading underscore on
/// the domain (`_mitochondrion`, `_plastid`, ...), plus `Unispike*` controls.
fn is_organellar(domain: &str) -> bool {
    domain.starts_with('_') || domain.starts_with("Unispike")
}

/// Pass 1: parse every record once and keep a fixed-size summary of each.
///
/// Policy-free, so one survey serves many policies ([`save`], [`load`]).
/// `oversize` receives labels of records past its length threshold as they
/// are seen, since summaries do not keep labels.
pub fn survey(reference: &Path, mut oversize: Option<(usize, &mut dyn Write)>) -> Result<Survey> {
    let mut tree = TreeBuilder::default();
    let mut summaries: Vec<Summary> = Vec::new();
    let mut parse_failures = Vec::new();
    let (mut records, mut bases) = (0u64, 0u64);
    let (mut unknown_bases, mut unknown_records) = (0u64, 0u64);

    let mut reader = input::reader(reference)?.context("empty reference database")?;
    while let Some(record) = reader.next() {
        let record = record.context("reading reference FASTA")?;
        ensure!(
            record.format() == needletail::parser::Format::Fasta,
            "references must be FASTA"
        );
        let ordinal = u32::try_from(records).context("too many reference sequences")?;
        records += 1;
        let label = input::label(record.id())?;
        let lineage = match taxonomy::parse(&label) {
            Ok(lineage) => lineage,
            Err(error) => {
                // Bound what we keep so a badly broken file can't exhaust memory.
                if parse_failures.len() < 64 {
                    parse_failures.push((ordinal, format!("{label:?}: {error}")));
                }
                continue;
            }
        };
        let (seq, unknown) = sequence::normalize(&record.seq());
        bases += seq.len() as u64;
        unknown_bases += unknown as u64;
        unknown_records += u64::from(unknown > 0);
        let words = sequence::unique_words(&seq);
        // Any base outside ACGT breaks the k-mer window, not only unrecognised bytes.
        let ambiguous = seq
            .iter()
            .filter(|b| !matches!(b, b'A' | b'C' | b'G' | b'T'))
            .count();

        let mut flags = 0u8;
        if lineage[FAMILY].is_none() && lineage[GENUS].is_none() {
            flags |= flag::SHALLOW;
        }
        if lineage[DOMAIN].as_deref().is_some_and(is_organellar) {
            flags |= flag::ORGANELLAR;
        }
        if let Some((threshold, list)) = oversize.as_mut()
            && seq.len() > *threshold
        {
            writeln!(list, "{label}\t{}", seq.len()).context("writing oversize list")?;
        }

        let mut digest = [0u8; 16];
        digest.copy_from_slice(&blake3::hash(&seq).as_bytes()[..16]);

        summaries.push(Summary {
            ordinal,
            group: tree.insert(lineage)?,
            length: u32::try_from(seq.len()).context("reference sequence too long")?,
            words: words.len() as u32,
            ambiguous: ambiguous as u32,
            digest,
            sketch: sketch::sketch::<SKETCH>(&words),
            flags,
        });
    }
    ensure!(!summaries.is_empty(), "empty reference database");
    let (index_of, groups) = index_groups(&tree, &summaries)?;

    Ok(Survey {
        summaries,
        tree,
        groups,
        index_of,
        records,
        bases,
        unknown_bases,
        unknown_records,
        parse_failures,
    })
}

/// Node ID -> group index, and the groups themselves. Rank-6 nodes are exactly
/// the distinct stored lineages. Rebuilt on cache load rather than stored.
fn index_groups(tree: &TreeBuilder, summaries: &[Summary]) -> Result<(Vec<u32>, Vec<Group>)> {
    let mut index_of = vec![u32::MAX; tree.nodes.len()];
    let mut groups = Vec::new();
    for (id, node) in tree.nodes.iter().enumerate() {
        if node.rank as usize != GENUS {
            continue;
        }
        let mut path = [0u32; 7];
        let mut cursor = id as u32;
        for rank in (0..7).rev() {
            path[rank] = cursor;
            cursor = tree.nodes[cursor as usize].parent;
        }
        index_of[id] = u32::try_from(groups.len())?;
        groups.push(Group {
            leaf: id as u32,
            path,
            records: 0,
        });
    }
    for summary in summaries {
        let slot = *index_of
            .get(summary.group as usize)
            .filter(|&&g| g != u32::MAX)
            .context("summary names a node that is not a stored lineage")?;
        groups[slot as usize].records += 1;
    }
    Ok((index_of, groups))
}

// --- Survey cache: pass 1 is policy-free, so it can be reused ---

const CACHE_MAGIC: &[u8; 8] = b"STXSURV\0";

/// Cache key: size and mtime of the input. Not a content hash, so an in-place
/// rewrite with identical size and mtime goes unnoticed; [`emit`] re-checks
/// the record count for that reason.
fn source_key(reference: &Path) -> Result<(u64, i64, u32)> {
    let meta =
        std::fs::metadata(reference).with_context(|| format!("reading {}", reference.display()))?;
    let modified = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    Ok(match modified {
        Some(d) => (meta.len(), d.as_secs() as i64, d.subsec_nanos()),
        None => (meta.len(), 0, 0),
    })
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.at.checked_add(n).context("truncated survey cache")?;
        ensure!(end <= self.bytes.len(), "truncated survey cache");
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    /// Length-prefixed UTF-8 string, bounded by the remaining input so a corrupt
    /// length can't trigger a huge allocation.
    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        ensure!(len <= self.bytes.len() - self.at, "truncated survey cache");
        String::from_utf8(self.take(len)?.to_vec()).context("invalid UTF-8 in survey cache")
    }
}

/// Write a survey cache next to the database, atomically so an interrupted
/// run can't leave a truncated cache behind.
pub fn save(survey: &Survey, reference: &Path, path: &Path) -> Result<()> {
    let (len, secs, nanos) = source_key(reference)?;
    let mut out = Vec::with_capacity(survey.summaries.len() * 110 + (1 << 16));
    out.extend_from_slice(CACHE_MAGIC);
    put_u32(&mut out, sketch::CURATION_VERSION);
    put_u64(&mut out, len);
    out.extend_from_slice(&secs.to_le_bytes());
    put_u32(&mut out, nanos);
    for value in [
        survey.records,
        survey.bases,
        survey.unknown_bases,
        survey.unknown_records,
    ] {
        put_u64(&mut out, value);
    }

    put_u64(&mut out, survey.summaries.len() as u64);
    for summary in &survey.summaries {
        for value in [
            summary.ordinal,
            summary.group,
            summary.length,
            summary.words,
            summary.ambiguous,
        ] {
            put_u32(&mut out, value);
        }
        out.extend_from_slice(&summary.digest);
        out.push(summary.flags);
        let values = summary.sketch.values();
        out.push(values.len() as u8);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    put_u64(&mut out, survey.tree.nodes.len() as u64);
    for node in &survey.tree.nodes {
        put_u32(&mut out, node.parent);
        out.push(node.rank);
        put_u32(&mut out, node.name.len() as u32);
        out.extend_from_slice(node.name.as_bytes());
    }

    put_u64(&mut out, survey.parse_failures.len() as u64);
    for (ordinal, message) in &survey.parse_failures {
        put_u32(&mut out, *ordinal);
        put_u32(&mut out, message.len() as u32);
        out.extend_from_slice(message.as_bytes());
    }

    out.extend_from_slice(blake3::hash(&out).as_bytes());

    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = tempfile::Builder::new()
        .prefix(".sintaxer-survey-")
        .tempfile_in(parent)?;
    temp.write_all(&out)?;
    temp.as_file().sync_all()?;
    temp.persist(path)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Read a survey cache, or `Ok(None)` if there is none at `path`. A cache for
/// a different database or contract, or a damaged one, is an error.
pub fn load(reference: &Path, path: &Path) -> Result<Option<Survey>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    ensure!(bytes.len() > 40, "{} is not a survey cache", path.display());
    let (body, digest) = bytes.split_at(bytes.len() - 32);
    ensure!(
        blake3::hash(body).as_bytes() == digest,
        "{} is damaged; delete it and re-survey",
        path.display()
    );

    let mut c = Cursor { bytes: body, at: 0 };
    ensure!(
        c.take(8)? == CACHE_MAGIC,
        "{} is not a survey cache",
        path.display()
    );
    let version = c.u32()?;
    ensure!(
        version == sketch::CURATION_VERSION,
        "{} was written by curation version {version}, this is {}; delete it and re-survey",
        path.display(),
        sketch::CURATION_VERSION
    );
    let (len, secs, nanos) = source_key(reference)?;
    let cached = (
        c.u64()?,
        i64::from_le_bytes(c.take(8)?.try_into().unwrap()),
        c.u32()?,
    );
    ensure!(
        cached == (len, secs, nanos),
        "{} describes a different {} (size/mtime {:?}, found {:?}); delete it and re-survey",
        path.display(),
        reference.display(),
        cached,
        (len, secs, nanos)
    );

    let (records, bases) = (c.u64()?, c.u64()?);
    let (unknown_bases, unknown_records) = (c.u64()?, c.u64()?);

    let count = c.u64()? as usize;
    ensure!(count as u64 <= records, "corrupt survey cache");
    let mut summaries = Vec::with_capacity(count);
    let mut values = [0u16; SKETCH];
    for _ in 0..count {
        let ordinal = c.u32()?;
        let group = c.u32()?;
        let length = c.u32()?;
        let words = c.u32()?;
        let ambiguous = c.u32()?;
        let mut digest = [0u8; 16];
        digest.copy_from_slice(c.take(16)?);
        let flags = c.u8()?;
        let width = c.u8()? as usize;
        ensure!(width <= SKETCH, "corrupt sketch in survey cache");
        for value in values.iter_mut().take(width) {
            *value = u16::from_le_bytes(c.take(2)?.try_into().unwrap());
        }
        summaries.push(Summary {
            ordinal,
            group,
            length,
            words,
            ambiguous,
            digest,
            sketch: Sketch::restore(&values[..width]),
            flags,
        });
    }
    ensure!(!summaries.is_empty(), "empty reference database");

    let nodes = c.u64()? as usize;
    let mut restored = Vec::with_capacity(nodes.min(1 << 20));
    for _ in 0..nodes {
        let parent = c.u32()?;
        let rank = c.u8()?;
        let name = c.string()?;
        restored.push(taxonomy::Node { parent, rank, name });
    }
    let tree = TreeBuilder::restore(restored);

    let failures = c.u64()? as usize;
    let mut parse_failures = Vec::with_capacity(failures.min(64));
    for _ in 0..failures {
        let ordinal = c.u32()?;
        parse_failures.push((ordinal, c.string()?));
    }
    ensure!(c.at == body.len(), "trailing data in survey cache");

    let (index_of, groups) = index_groups(&tree, &summaries)?;
    Ok(Some(Survey {
        summaries,
        tree,
        groups,
        index_of,
        records,
        bases,
        unknown_bases,
        unknown_records,
        parse_failures,
    }))
}

// --- Plan: what a policy would do, without writing anything ---

/// Guard role of an under-annotated record. Capped rather than deleted, but
/// over-weighted, since missing ranks share a single empty node per parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shallow {
    /// Order already has genus-annotated members: least informative.
    Covered,
    /// Order has no genus coverage at all: irreplaceable at order rank.
    OrphanOrder,
    /// No order annotation: deep-branching environmental diversity.
    Unplaced,
}

pub struct Occupancy {
    pub label: &'static str,
    pub floor: f64,
    pub records: u64,
    pub bases: u64,
    /// Most common domains in this class, descending.
    pub domains: Vec<(String, u64)>,
}

pub struct Plan {
    pub cap: usize,
    /// Hard length ceiling in force, so the hazard report can say whether the
    /// records it describes were kept or dropped.
    pub max_length: Option<usize>,
    /// Quantiles of within-lineage sketch distance over a bounded sample: all
    /// members, then only members within the length threshold.
    pub diversity: Option<[f32; 5]>,
    pub diversity_trimmed: Option<[f32; 5]>,
    pub demote_length: u64,
    pub min_words: u64,
    pub records: u64,
    pub parsed: u64,
    pub bases: u64,
    pub groups: usize,
    pub lengths: Percentiles,
    pub cap_curve: Vec<(usize, u64)>,
    pub shallow: [u64; 3],
    pub organellar: u64,
    pub organellar_bases: u64,
    pub occupancy: Vec<Occupancy>,
    pub demoted: u64,
    pub demoted_bases: u64,
    pub low_complexity: u64,
    pub ambiguous: u64,
    pub duplicates: u64,
    pub conflicting_duplicates: u64,
    pub truncation_artifact: Option<(u32, u64)>,
}

#[derive(Default)]
pub struct Percentiles {
    pub min: u32,
    pub p01: u32,
    pub p25: u32,
    pub median: u32,
    pub p75: u32,
    pub p95: u32,
    pub p99: u32,
    pub max: u32,
    pub mean: f64,
}

fn percentiles(lengths: &mut [u32]) -> Percentiles {
    lengths.sort_unstable();
    let n = lengths.len();
    let at = |q: f64| lengths[((n as f64 * q) as usize).min(n - 1)];
    Percentiles {
        min: lengths[0],
        p01: at(0.01),
        p25: at(0.25),
        median: at(0.50),
        p75: at(0.75),
        p95: at(0.95),
        p99: at(0.99),
        max: lengths[n - 1],
        mean: lengths.iter().map(|&l| l as f64).sum::<f64>() / n as f64,
    }
}

/// Cap values for the cap curve; extends past the default cap so the policy
/// marker is shown.
const CAPS: [usize; 13] = [1, 2, 3, 5, 10, 20, 25, 50, 100, 250, 500, 1000, 2500];

/// Sampled distribution of sketch distances within the same stored lineage,
/// used to calibrate [`DIVERSITY_CEILING`].
fn diversity_probe(survey: &Survey, members: &[Vec<u32>]) -> Option<[f32; 5]> {
    let mut distances: Vec<f32> = Vec::new();
    for group in members {
        if group.len() < 2 {
            continue;
        }
        // Pair each member with a partner halfway along the group so the sample
        // spans it rather than clustering on adjacent records.
        let stride = (group.len() / 2).max(1);
        for (offset, &i) in group.iter().enumerate().take(64) {
            let j = group[(offset + stride) % group.len()];
            if i == j {
                continue;
            }
            distances.push(
                1.0 - sketch::jaccard(
                    &survey.summaries[i as usize].sketch,
                    &survey.summaries[j as usize].sketch,
                ),
            );
        }
    }
    if distances.is_empty() {
        return None;
    }
    distances.sort_by(f32::total_cmp);
    let at = |q: f64| distances[((distances.len() as f64 * q) as usize).min(distances.len() - 1)];
    Some([at(0.05), at(0.25), at(0.50), at(0.75), at(0.95)])
}

/// Occupancy classes as a fraction of the 65,536-word universe. Expected
/// spurious score is `SAMPLE_SIZE x occupancy`, i.e. out of 32.
const CLASSES: [(&str, f64); 5] = [
    (">=20%", 0.20),
    ("10-20%", 0.10),
    ("5-10%", 0.05),
    ("2-5%", 0.02),
    ("<2%", 0.0),
];

pub fn plan(survey: &Survey, policy: &Policy) -> Plan {
    // Only the largest lineages matter here: those are the ones a cap bites into.
    let mut ranked: Vec<usize> = (0..survey.groups.len()).collect();
    ranked.sort_unstable_by_key(|&g| std::cmp::Reverse(survey.groups[g].records));
    ranked.truncate(500);
    let wanted: std::collections::HashSet<u32> =
        ranked.iter().map(|&g| survey.groups[g].leaf).collect();
    let mut probe: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    for (i, summary) in survey.summaries.iter().enumerate() {
        if wanted.contains(&summary.group) {
            let slot = probe.entry(summary.group).or_default();
            if slot.len() < 128 {
                slot.push(i as u32);
            }
        }
    }
    let mut probe: Vec<Vec<u32>> = probe.into_values().collect();
    probe.sort_unstable_by_key(|g| g.first().copied().unwrap_or(0));
    let diversity = diversity_probe(survey, &probe);
    // Jaccard collapses when lengths differ wildly, so repeat the probe over
    // length-typical members only; the gap shows the length hazard.
    let trimmed: Vec<Vec<u32>> = probe
        .iter()
        .map(|g| {
            g.iter()
                .copied()
                .filter(|&i| !policy.demoted(&survey.summaries[i as usize]))
                .collect()
        })
        .collect();
    let diversity_trimmed = diversity_probe(survey, &trimmed);

    let mut lengths: Vec<u32> = survey.summaries.iter().map(|s| s.length).collect();
    let lengths_stats = percentiles(&mut lengths);

    let cap_curve = CAPS
        .iter()
        .map(|&cap| {
            (
                cap,
                survey
                    .groups
                    .iter()
                    .map(|g| u64::from(g.records).min(cap as u64))
                    .sum(),
            )
        })
        .collect();

    let coverage = survey.order_coverage();

    let mut shallow = [0u64; 3];
    let (mut organellar, mut organellar_bases) = (0u64, 0u64);
    let (mut demoted, mut demoted_bases) = (0u64, 0u64);
    let (mut low_complexity, mut ambiguous) = (0u64, 0u64);
    let mut class_records = [0u64; 5];
    let mut class_bases = [0u64; 5];
    let mut class_domains: [std::collections::HashMap<u32, u64>; 5] = Default::default();
    let mut truncation: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();

    for summary in &survey.summaries {
        let group = survey.group_of(summary);
        if let Some(class) = survey.shallow_class(group, &coverage) {
            debug_assert!(summary.flags & flag::SHALLOW != 0);
            shallow[class as usize] += 1;
        }
        if summary.flags & flag::ORGANELLAR != 0 {
            organellar += 1;
            organellar_bases += u64::from(summary.length);
        }
        if policy.demoted(summary) {
            demoted += 1;
            demoted_bases += u64::from(summary.length);
        }
        low_complexity += u64::from(policy.low_complexity(summary));
        ambiguous += u64::from(policy.ambiguous(summary));

        let occupancy = f64::from(summary.words) / WORDS as f64;
        let class = CLASSES.iter().position(|&(_, f)| occupancy >= f).unwrap();
        class_records[class] += 1;
        class_bases[class] += u64::from(summary.length);
        *class_domains[class].entry(group.path[DOMAIN]).or_default() += 1;

        if summary.length > 4096 {
            *truncation.entry(summary.length).or_default() += 1;
        }
    }

    let occupancy = CLASSES
        .iter()
        .enumerate()
        .map(|(i, &(label, floor))| {
            let mut domains: Vec<_> = class_domains[i]
                .iter()
                .map(|(&node, &n)| (survey.name(node).to_owned(), n))
                .collect();
            domains.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            domains.truncate(3);
            Occupancy {
                label,
                floor,
                records: class_records[i],
                bases: class_bases[i],
                domains,
            }
        })
        .collect();

    // Exact duplicates, split by whether the copies agree on taxonomy. Agreeing
    // ones can collapse; disagreeing ones are reported and left alone.
    // Sorted runs instead of a hash map to keep allocations down.
    let mut keys: Vec<([u8; 16], u32, u32)> = survey
        .summaries
        .iter()
        .map(|s| (s.digest, s.length, s.group))
        .collect();
    keys.sort_unstable();
    let (mut duplicates, mut conflicting) = (0u64, 0u64);
    let mut run = 0usize;
    while run < keys.len() {
        let mut end = run + 1;
        while end < keys.len() && keys[end].0 == keys[run].0 && keys[end].1 == keys[run].1 {
            end += 1;
        }
        if end - run > 1 {
            // Group IDs within a run are already sorted by the tuple sort.
            let distinct = 1
                + (run + 1..end)
                    .filter(|&i| keys[i].2 != keys[i - 1].2)
                    .count();
            duplicates += (end - run - distinct) as u64;
            conflicting += u64::from(distinct > 1);
        }
        run = end;
    }
    drop(keys);

    // A length spike far above the rest points to upstream truncation, which
    // curation cannot fix.
    let truncation_artifact = truncation
        .into_iter()
        .max_by_key(|&(_, n)| n)
        .filter(|&(_, n)| n >= 100);

    Plan {
        cap: policy.cap,
        max_length: policy.max_length,
        diversity,
        diversity_trimmed,
        demote_length: policy.demote_length as u64,
        min_words: policy.min_words as u64,
        records: survey.records,
        parsed: survey.summaries.len() as u64,
        bases: survey.bases,
        groups: survey.groups.len(),
        lengths: lengths_stats,
        cap_curve,
        shallow,
        organellar,
        organellar_bases,
        occupancy,
        demoted,
        demoted_bases,
        low_complexity,
        ambiguous,
        duplicates,
        conflicting_duplicates: conflicting,
        truncation_artifact,
    }
}

// --- Reporting ---

/// Formats a number with thousands separators.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

impl Plan {
    /// Human-readable survey report, suitable for diffing between database
    /// revisions.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let p = &self.lengths;
        out.push_str(&format!(
            "input\n  records            {}\n  parsed             {}\n  \
             parse failures     {}\n  bases              {}\n  \
             length             min {} p01 {} p25 {} median {} p75 {} p95 {} p99 {} max {} (mean {:.0})\n",
            thousands(self.records),
            thousands(self.parsed),
            thousands(self.records - self.parsed),
            thousands(self.bases),
            p.min, p.p01, p.p25, p.median, p.p75, p.p95, p.p99, p.max, p.mean,
        ));

        out.push_str(&format!(
            "\ntaxonomy\n  distinct stored lineages (d..g)  {}\n  \
             mean records per lineage         {:.1}\n",
            thousands(self.groups as u64),
            self.parsed as f64 / self.groups.max(1) as f64,
        ));

        if let Some(q) = self.diversity {
            out.push_str("\nwithin-lineage sketch distance (500 largest lineages, sampled)\n");
            out.push_str(&format!(
                "  all members        p05 {:.3}  p25 {:.3}  median {:.3}  p75 {:.3}  p95 {:.3}\n",
                q[0], q[1], q[2], q[3], q[4]
            ));
            if let Some(t) = self.diversity_trimmed {
                out.push_str(&format!(
                    "  within length cap  p05 {:.3}  p25 {:.3}  median {:.3}  p75 {:.3}  p95 {:.3}\n",
                    t[0], t[1], t[2], t[3], t[4]
                ));
            }
        }

        out.push_str("\ncap curve (records retained by cap per stored lineage)\n");
        for &(cap, kept) in &self.cap_curve {
            let marker = if cap == self.cap { " <- policy" } else { "" };
            out.push_str(&format!(
                "  cap {:>3}  {:>10}  ({:5.1}% of parsed){}\n",
                cap,
                thousands(kept),
                percent(kept, self.parsed),
                marker
            ));
        }

        let shallow_total: u64 = self.shallow.iter().sum();
        out.push_str(&format!(
            "\nunder-annotated records (no family, no genus)   {}  ({:.1}% of parsed)\n  \
             order has genus coverage   {:>9}   (least informative)\n  \
             order lacks genus coverage {:>9}   (only representatives of their order)\n  \
             no order at all            {:>9}   (deep-branching environmental diversity)\n",
            thousands(shallow_total),
            percent(shallow_total, self.parsed),
            thousands(self.shallow[Shallow::Covered as usize]),
            thousands(self.shallow[Shallow::OrphanOrder as usize]),
            thousands(self.shallow[Shallow::Unplaced as usize]),
        ));

        out.push_str(&format!(
            "\nother\n  exact duplicate copies (same lineage)   {:>9}  ({:.1}% of parsed)\n  \
             identical sequences, conflicting lineage {:>8}  groups\n  \
             low complexity (< {} distinct words)     {:>8}\n  \
             above ambiguity limit                    {:>8}\n  \
             longer than {} bp                      {:>8}  ({:.2}% of records, {:.1}% of bases)\n",
            thousands(self.duplicates),
            percent(self.duplicates, self.parsed),
            thousands(self.conflicting_duplicates),
            self.min_words,
            thousands(self.low_complexity),
            thousands(self.ambiguous),
            thousands(self.demote_length),
            thousands(self.demoted),
            percent(self.demoted, self.parsed),
            percent(self.demoted_bases, self.bases),
        ));
        out
    }

    /// Word-space saturation warning, or `None` when nothing is saturated.
    ///
    /// With `K = 8` a long reference contains a large share of all possible
    /// words, so it scores well against any query and can win novel queries.
    pub fn hazard(&self) -> Option<String> {
        self.hazard_with(None)
    }

    /// The warning, optionally comparing the curated database against the input.
    pub fn hazard_with(&self, after: Option<&HazardAfter>) -> Option<String> {
        let notable: u64 = self
            .occupancy
            .iter()
            .filter(|c| c.floor >= 0.05)
            .map(|c| c.records)
            .sum();
        if notable == 0 {
            return None;
        }
        let mut out = format!(
            "warning: {} references saturate the 8-mer word space (k=8, {} words).\n  \
             Such references score highly against ANY query by chance, and can win\n  \
             the top hit for novel queries regardless of taxonomy.\n\n  \
             {:<10} {:>9}  {:<22}  {}\n",
            thousands(notable),
            thousands(WORDS as u64),
            "occupancy",
            "records",
            "expected spurious score",
            "most common domains"
        );
        for class in self.occupancy.iter().filter(|c| c.floor >= 0.02) {
            let domains = class
                .domains
                .iter()
                .map(|(name, n)| {
                    format!(
                        "{}({})",
                        if name.is_empty() { "-" } else { name },
                        thousands(*n)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let score = crate::SAMPLE_SIZE as f64 * class.floor;
            out.push_str(&format!(
                "  {:<10} {:>9}  {:<22}  {}\n",
                class.label,
                thousands(class.records),
                format!(">= {score:.1} / {}", crate::SAMPLE_SIZE),
                domains
            ));
        }
        if self.organellar > 0 {
            out.push_str(&format!(
                "\n  {} records are organellar or spike-in decoys (d:_mitochondrion,\n  \
                 _plastid, _Bacteria, _Archaea, _nucleomorph, _pseudogene, Unispike*):\n  \
                 {:.2}% of records but {:.2}% of all bases, mean length {} bp.\n  \
                 They are meant to catch off-target amplification, but at k=8 a\n  \
                 multi-kilobase decoy is not a specific decoy -- it is a universal magnet.\n",
                thousands(self.organellar),
                percent(self.organellar, self.parsed),
                percent(self.organellar_bases, self.bases),
                thousands(self.organellar_bases / self.organellar.max(1)),
            ));
        }
        if let Some((length, count)) = self.truncation_artifact {
            out.push_str(&format!(
                "\n  note: {} records are exactly {} bp. A spike that sharp is a\n  \
                 length-truncation artifact upstream, not biology -- worth fixing in the\n  \
                 database rather than here.\n",
                thousands(count),
                thousands(u64::from(length)),
            ));
        }
        if let Some(after) = after {
            let before_share = percent(notable, self.parsed);
            let after_share = percent(after.saturating, after.records);
            out.push_str(&format!(
                "\n  after curation: {} of {} records saturate ({:.3}% vs {:.3}% before),\n  \
                 organellar decoys {} of {} ({:.3}% vs {:.3}%).\n",
                thousands(after.saturating),
                thousands(after.records),
                after_share,
                before_share,
                thousands(after.organellar),
                thousands(self.organellar),
                percent(after.organellar, after.records),
                percent(self.organellar, self.parsed),
            ));
            // Trigger on the organellar share too: thinning the rest concentrates the
            // worst offenders even as the overall share drops.
            let organellar_before = percent(self.organellar, self.parsed);
            let organellar_after = percent(after.organellar, after.records);
            if organellar_after > organellar_before * 1.2 || after_share > before_share * 1.2 {
                out.push_str(
                    "  NOTE: curation CONCENTRATED these. Each multi-kilobase decoy sits in a\n  \
                     stored lineage of its own, so a per-lineage cap never trims it while\n  \
                     everything around it is thinned. The absolute number barely moves; their\n  \
                     share of the database -- and so their odds of reaching the candidate set\n  \
                     -- goes up. Trimming or --drop-organellar is the fix, not a larger cap.\n",
                );
            }
        }
        match self.max_length {
            None => out.push_str(
                "\n  These records are RETAINED: no --max-length is set. To act on this:\n    \
                 trim them   --oversize-list <path>   (worklist for ITSx/cmsearch)\n    \
                 or exclude  --max-length <bp>  /  --drop-organellar\n",
            ),
            Some(max) => out.push_str(&format!(
                "\n  Records over {} bp are DROPPED (--max-length). Trimming them and \
                 splicing\n  them back keeps their lineages; a ceiling deletes whatever \
                 only they represent.\n",
                thousands(max as u64)
            )),
        }
        Some(out)
    }
}

// --- Selection ---

/// How representatives are chosen. All but `Cover` act only on groups over
/// the cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Selector {
    /// Keep one representative per LSH neighbourhood, densest first; `--cap` is a ceiling.
    Cover,
    /// Uniform random sample, seeded per group (default).
    Random,
    /// Farthest-point selection over k-mer sketches (favours the lineage periphery).
    Maxmin,
    /// Keep records in input order, like a naive `head -n` cap.
    First,
}

/// Distances above this are treated as equal.
///
/// Stops farthest-point selection from favouring chimeras and misannotated
/// records. Set just above typical within-lineage distances; `--ceiling 1.0`
/// disables it and `--plan` prints the distribution to recalibrate.
pub const DIVERSITY_CEILING: f32 = 0.95;

/// LSH bands used for the density proxy, each covering `sketch::BAND` min-hashes.
const DENSITY_BANDS: usize = 4;

#[derive(Clone, Debug, Default)]
pub struct DropCounts {
    pub low_complexity: u64,
    pub ambiguous: u64,
    pub oversize: u64,
    pub duplicates: u64,
    pub over_cap: u64,
    /// Trimmed by a shallow-record cap rather than the lineage cap.
    pub over_shallow_cap: u64,
    /// Removed by a validation switch.
    pub deliberate: u64,
    /// Records reinstated because a filter would otherwise have emptied a stored
    /// lineage. Nonzero means the policy was too aggressive somewhere.
    pub restored: u64,
}

pub struct Selection {
    keep: Vec<u64>,
    pub kept: u64,
    pub counts: DropCounts,
    pub policy_digest: [u8; 32],
    /// Groups removed on purpose by a validation switch. Group preservation
    /// still applies to everything else.
    pub deliberate: Vec<bool>,
}

impl Selection {
    pub fn keeps(&self, ordinal: u32) -> bool {
        let (word, bit) = (ordinal as usize / 64, ordinal % 64);
        self.keep.get(word).is_some_and(|w| w >> bit & 1 == 1)
    }
    fn set(&mut self, ordinal: u32) {
        self.keep[ordinal as usize / 64] |= 1 << (ordinal % 64);
    }
}

/// Candidate order within a group: not oversize, fewest ambiguous bases,
/// length closest to the group median, then earliest.
fn preference(policy: &Policy, summary: &Summary, median_words: u32) -> (u8, u32, u32, u32) {
    (
        u8::from(policy.demoted(summary)),
        summary.ambiguous,
        summary.words.abs_diff(median_words),
        summary.ordinal,
    )
}

pub fn policy_digest(policy: &Policy, selector: Selector, seed: u64) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sintaxer/curation\0");
    hasher.update(&sketch::CURATION_VERSION.to_le_bytes());
    hasher.update(&(policy.cap as u64).to_le_bytes());
    hasher.update(&(policy.demote_length as u64).to_le_bytes());
    hasher.update(&(policy.max_length.unwrap_or(usize::MAX) as u64).to_le_bytes());
    hasher.update(&policy.max_ambiguity.to_bits().to_le_bytes());
    hasher.update(&(policy.min_words as u64).to_le_bytes());
    hasher.update(&policy.ceiling.to_bits().to_le_bytes());
    hasher.update(&(policy.cover_bands as u64).to_le_bytes());
    hasher.update(&(policy.shallow.covered as u64).to_le_bytes());
    hasher.update(&(policy.shallow.orphan_order as u64).to_le_bytes());
    hasher.update(&(policy.shallow.unplaced as u64).to_le_bytes());
    hasher.update(&policy.shallow.ratio.to_bits().to_le_bytes());
    hasher.update(&[
        u8::from(policy.drop_shallow),
        u8::from(policy.drop_organellar),
    ]);
    hasher.update(&[selector as u8]);
    hasher.update(&seed.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// Decide which records survive.
///
/// Order matters: hard rejects before any cap slot is used, dereplication
/// before selection, and group preservation last so no lineage vanishes.
pub fn select(survey: &Survey, policy: &Policy, selector: Selector, seed: u64) -> Selection {
    let n = survey.records as usize;
    let mut selection = Selection {
        keep: vec![0u64; n.div_ceil(64)],
        kept: 0,
        counts: DropCounts::default(),
        policy_digest: policy_digest(policy, selector, seed),
        deliberate: vec![false; survey.groups.len()],
    };

    // Group membership in CSR layout: one allocation instead of one Vec per group.
    // Counts come from `Group::records`.
    let groups = survey.groups.len();
    let mut offsets = vec![0u32; groups + 1];
    for (i, group) in survey.groups.iter().enumerate() {
        offsets[i + 1] = group.records;
    }
    for i in 0..groups {
        offsets[i + 1] += offsets[i];
    }
    let mut members = vec![0u32; survey.summaries.len()];
    let mut cursor = offsets.clone();
    for (i, summary) in survey.summaries.iter().enumerate() {
        let g = survey.index_of[summary.group as usize] as usize;
        members[cursor[g] as usize] = i as u32;
        cursor[g] += 1;
    }

    // Per-group caps. Under-annotated groups get a role-specific cap, so order
    // annotation coverage must be known up front.
    let coverage = survey.order_coverage();
    let caps: Vec<usize> = survey
        .groups
        .iter()
        .map(|group| match survey.shallow_class(group, &coverage) {
            None => policy.cap,
            Some(Shallow::Unplaced) => policy.shallow.unplaced,
            Some(Shallow::OrphanOrder) => policy.shallow.orphan_order,
            Some(Shallow::Covered) => {
                let annotated = coverage[&group.path[ORDER]] as f64;
                let proportional = (annotated * policy.shallow.ratio).ceil() as usize;
                policy.shallow.covered.min(proportional).max(1)
            }
        })
        .collect();

    let mut eligible: Vec<u32> = Vec::new();
    let mut dropped: Vec<u32> = Vec::new();
    for g in 0..groups {
        let span = &members[offsets[g] as usize..offsets[g + 1] as usize];
        let group = &survey.groups[g];
        let shallow = survey.shallow_class(group, &coverage).is_some();
        let organellar = is_organellar(survey.rank_name(group, DOMAIN));
        if (policy.drop_shallow && shallow) || (policy.drop_organellar && organellar) {
            selection.deliberate[g] = true;
            selection.counts.deliberate += span.len() as u64;
            continue;
        }
        let cap = caps[g];
        eligible.clear();
        dropped.clear();

        // (a) hard rejects
        for &i in span {
            let summary = &survey.summaries[i as usize];
            if policy.rejects(summary) {
                selection.counts.low_complexity += u64::from(policy.low_complexity(summary));
                selection.counts.ambiguous += u64::from(policy.ambiguous(summary));
                selection.counts.oversize += u64::from(policy.oversize(summary));
                dropped.push(i);
            } else {
                eligible.push(i);
            }
        }

        // (b) collapse exact duplicates, keeping the earliest copy
        eligible.sort_unstable_by_key(|&i| {
            let s = &survey.summaries[i as usize];
            (s.digest, s.length, s.ordinal)
        });
        let before = eligible.len();
        eligible.dedup_by(|a, b| {
            let (a, b) = (
                &survey.summaries[*a as usize],
                &survey.summaries[*b as usize],
            );
            a.digest == b.digest && a.length == b.length
        });
        selection.counts.duplicates += (before - eligible.len()) as u64;

        // (c) group preservation: every stored lineage in the input must appear in
        // the output.
        if eligible.is_empty() {
            let take = cap.min(dropped.len());
            let median = median_words(survey, &dropped);
            dropped.sort_unstable_by_key(|&i| {
                preference(policy, &survey.summaries[i as usize], median)
            });
            eligible.extend_from_slice(&dropped[..take]);
            selection.counts.restored += take as u64;
            // Undo the drop counters for every reason a restored record was rejected,
            // not just the first, so the manifest matches the written database.
            for &i in &dropped[..take] {
                let summary = &survey.summaries[i as usize];
                let c = &mut selection.counts;
                c.low_complexity = c
                    .low_complexity
                    .saturating_sub(u64::from(policy.low_complexity(summary)));
                c.ambiguous = c
                    .ambiguous
                    .saturating_sub(u64::from(policy.ambiguous(summary)));
                c.oversize = c
                    .oversize
                    .saturating_sub(u64::from(policy.oversize(summary)));
            }
        }

        // (d) reduce to representatives. `Cover` always runs, since how many a
        // group needs depends on the group; the others only run over the cap.
        let before = eligible.len();
        if selector == Selector::Cover {
            let median = median_words(survey, &eligible);
            select_cover(survey, policy, &mut eligible, cap, median);
        } else if before > cap {
            let median = median_words(survey, &eligible);
            match selector {
                Selector::Maxmin => {
                    select_maxmin(survey, policy, &mut eligible, cap, median);
                }
                Selector::First => {
                    eligible.sort_unstable_by_key(|&i| {
                        preference(policy, &survey.summaries[i as usize], median)
                    });
                }
                Selector::Random => {
                    // Seeded from the curation digest, not the clock or `rng::stream`, so the
                    // same group always yields the same draw.
                    let mut rng = group_rng(&selection.policy_digest, g as u32);
                    for i in (1..eligible.len()).rev() {
                        eligible.swap(i, rng.bounded(i as u64 + 1) as usize);
                    }
                }
                Selector::Cover => unreachable!("handled above"),
            }
            eligible.truncate(cap);
        }
        let trimmed = (before - eligible.len()) as u64;
        if shallow {
            selection.counts.over_shallow_cap += trimmed;
        } else {
            selection.counts.over_cap += trimmed;
        }

        for &i in &eligible {
            selection.set(survey.summaries[i as usize].ordinal);
            selection.kept += 1;
        }
    }
    selection
}

/// LSH bands a full-width sketch affords.
///
/// Two records share a band with probability `J^4`, so over `b` bands they
/// match with `1 - (1 - J^4)^b`. More bands means a looser radius and more
/// compression; four collapses pairs above about 0.8 Jaccard.
pub const BANDS: usize = SKETCH / sketch::BAND;

/// Keep one representative per LSH neighbourhood, densest neighbourhood first.
/// Runs regardless of cap; `cap` bounds it above, group preservation below.
fn select_cover(survey: &Survey, policy: &Policy, members: &mut Vec<u32>, cap: usize, median: u32) {
    let bands = policy.cover_bands.clamp(1, BANDS);
    let density = density(survey, members);

    let mut order: Vec<usize> = (0..members.len()).collect();
    order.sort_unstable_by_key(|&idx| {
        let summary = &survey.summaries[members[idx] as usize];
        (
            // Oversize records are kept only if nothing better covers their neighbourhood.
            u8::from(policy.demoted(summary)),
            std::cmp::Reverse(density[idx]),
            preference(policy, summary, median),
        )
    });

    let mut claimed: Vec<std::collections::HashSet<u64>> = vec![Default::default(); bands];
    let mut kept = Vec::new();
    for idx in order {
        let sketch = survey.summaries[members[idx] as usize].sketch;
        // Too few words to fill a band: keep it, since coverage can't be judged. Only
        // happens for records restored by group preservation.
        let judgeable = sketch.band(0).is_some();
        let covered = judgeable
            && (0..bands).any(|b| sketch.band(b).is_some_and(|key| claimed[b].contains(&key)));
        if covered {
            continue;
        }
        for (b, claimed) in claimed.iter_mut().enumerate() {
            if let Some(key) = sketch.band(b) {
                claimed.insert(key);
            }
        }
        kept.push(members[idx]);
        if kept.len() >= cap {
            break;
        }
    }
    kept.sort_unstable();
    *members = kept;
}

/// Approximate number of close relatives per member, via LSH banding.
/// Linear-time stand-in for neighbour counting.
fn density(survey: &Survey, members: &[u32]) -> Vec<u32> {
    let mut density = vec![0u32; members.len()];
    let mut keys: Vec<(u64, u32)> = Vec::with_capacity(members.len());
    for band in 0..DENSITY_BANDS {
        keys.clear();
        keys.extend(members.iter().enumerate().filter_map(|(idx, &i)| {
            survey.summaries[i as usize]
                .sketch
                .band(band)
                .map(|key| (key, idx as u32))
        }));
        keys.sort_unstable();
        let mut run = 0usize;
        while run < keys.len() {
            let mut end = run + 1;
            while end < keys.len() && keys[end].0 == keys[run].0 {
                end += 1;
            }
            let neighbours = (end - run - 1) as u32;
            for &(_, idx) in &keys[run..end] {
                density[idx as usize] += neighbours;
            }
            run = end;
        }
    }
    density
}

/// Truncated greedy farthest-point selection, maximising retained k-mer
/// coverage of the group.
fn select_maxmin(
    survey: &Survey,
    policy: &Policy,
    members: &mut Vec<u32>,
    cap: usize,
    median: u32,
) {
    let ceiling = policy.ceiling;
    let density = density(survey, members);
    let sketch_of = |i: u32| survey.summaries[i as usize].sketch;

    // Oversize records are picked only when nothing better is left; distance
    // alone can't tell a divergent congener from junk.
    let sound = |idx: usize| u8::from(!policy.demoted(&survey.summaries[members[idx] as usize]));

    // Seed with a typical member: the first pick shapes everything after it.
    let seed = (0..members.len())
        .max_by_key(|&idx| {
            let pref = preference(policy, &survey.summaries[members[idx] as usize], median);
            (sound(idx), density[idx], std::cmp::Reverse(pref))
        })
        .unwrap();

    let mut chosen = Vec::with_capacity(cap);
    chosen.push(seed);
    let mut mind: Vec<f32> = members
        .iter()
        .map(|&i| 1.0 - sketch::jaccard(&sketch_of(members[seed]), &sketch_of(i)))
        .collect();
    mind[seed] = f32::NEG_INFINITY;

    while chosen.len() < cap {
        let pick = (0..members.len())
            .filter(|&idx| mind[idx] > f32::NEG_INFINITY)
            .max_by(|&a, &b| {
                let key = |idx: usize| {
                    (
                        sound(idx),
                        mind[idx].min(ceiling),
                        density[idx],
                        std::cmp::Reverse(preference(
                            policy,
                            &survey.summaries[members[idx] as usize],
                            median,
                        )),
                    )
                };
                let (ka, kb) = (key(a), key(b));
                ka.0.cmp(&kb.0)
                    .then_with(|| ka.1.total_cmp(&kb.1))
                    .then_with(|| ka.2.cmp(&kb.2))
                    .then_with(|| ka.3.cmp(&kb.3))
            });
        let Some(pick) = pick else { break };
        chosen.push(pick);
        let picked = sketch_of(members[pick]);
        mind[pick] = f32::NEG_INFINITY;
        for (idx, &i) in members.iter().enumerate() {
            if mind[idx] > f32::NEG_INFINITY {
                mind[idx] = mind[idx].min(1.0 - sketch::jaccard(&picked, &sketch_of(i)));
            }
        }
    }

    // Emit in input order so the output reads as a subset of the original.
    let mut kept: Vec<u32> = chosen.into_iter().map(|idx| members[idx]).collect();
    kept.sort_unstable_by_key(|&i| survey.summaries[i as usize].ordinal);
    *members = kept;
}

fn median_words(survey: &Survey, members: &[u32]) -> u32 {
    if members.is_empty() {
        return 0;
    }
    let mut words: Vec<u32> = members
        .iter()
        .map(|&i| survey.summaries[i as usize].words)
        .collect();
    let mid = words.len() / 2;
    *words.select_nth_unstable(mid).1
}

fn group_rng(policy: &[u8; 32], group: u32) -> crate::rng::Rng {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"sintaxer/curate/group\0");
    hasher.update(policy);
    hasher.update(&group.to_le_bytes());
    crate::rng::Rng(u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8].try_into().unwrap(),
    ))
}

// --- Emit ---

/// Post-condition of curation: nothing the database named can disappear.
#[derive(Debug)]
pub struct Invariants {
    pub groups_preserved: bool,
    pub names_preserved: bool,
    pub missing: Vec<String>,
}

/// Check group and name preservation from the selection alone. The first
/// implies the second, but both are checked explicitly.
pub fn invariants(survey: &Survey, selection: &Selection) -> Invariants {
    let mut group_kept = vec![false; survey.groups.len()];
    let mut node_kept = vec![false; survey.tree.nodes.len()];
    for summary in &survey.summaries {
        if !selection.keeps(summary.ordinal) {
            continue;
        }
        let g = survey.index_of[summary.group as usize] as usize;
        group_kept[g] = true;
        for node in survey.groups[g].path {
            node_kept[node as usize] = true;
        }
    }
    let mut missing = Vec::new();
    for (g, kept) in group_kept.iter().enumerate() {
        if selection.deliberate[g] {
            continue;
        }
        if !kept && missing.len() < 32 {
            let group = &survey.groups[g];
            missing.push(
                (0..7)
                    .map(|r| survey.rank_name(group, r))
                    .filter(|n| !n.is_empty())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
    }
    // Every named node in the input must still be reachable in the output.
    let deliberate = selection.deliberate.iter().any(|&d| d);
    let mut names_preserved = true;
    for (id, node) in survey.tree.nodes.iter().enumerate() {
        if id != 0 && !node.name.is_empty() && !node_kept[id] {
            names_preserved = false;
            break;
        }
    }
    Invariants {
        groups_preserved: group_kept
            .iter()
            .enumerate()
            .all(|(g, &k)| k || selection.deliberate[g]),
        // Validation switches remove names on purpose.
        names_preserved: names_preserved || deliberate,
        missing,
    }
}

/// Saturation hazard in the curated database. Curation can raise the share
/// of saturating decoys, since each sits alone in a lineage the cap never touches.
pub struct HazardAfter {
    pub records: u64,
    pub saturating: u64,
    pub organellar: u64,
    pub over_demote: u64,
}

pub fn hazard_after(survey: &Survey, selection: &Selection, policy: &Policy) -> HazardAfter {
    let mut after = HazardAfter {
        records: 0,
        saturating: 0,
        organellar: 0,
        over_demote: 0,
    };
    for summary in &survey.summaries {
        if !selection.keeps(summary.ordinal) {
            continue;
        }
        after.records += 1;
        after.saturating += u64::from(f64::from(summary.words) / WORDS as f64 >= 0.05);
        after.organellar += u64::from(summary.flags & flag::ORGANELLAR != 0);
        after.over_demote += u64::from(summary.length as usize > policy.demote_length);
    }
    after
}

pub struct Emitted {
    pub records: u64,
    pub bases: u64,
    pub digest: blake3::Hash,
}

/// Pass 2: copy the selected records in input order. Sequences are written
/// unwrapped, one line each.
pub fn emit(
    reference: &Path,
    output: &Path,
    selection: &Selection,
    surveyed: u64,
) -> Result<Emitted> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = tempfile::Builder::new()
        .prefix(".sintaxer-curate-")
        .tempfile_in(parent)?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, temp.as_file_mut());
    let mut hasher = blake3::Hasher::new();
    let (mut records, mut bases) = (0u64, 0u64);

    let mut reader = input::reader(reference)?.context("empty reference database")?;
    let mut ordinal = 0u32;
    while let Some(record) = reader.next() {
        let record = record.context("re-reading reference FASTA")?;
        let current = ordinal;
        ordinal += 1;
        if !selection.keeps(current) {
            continue;
        }
        let id = record.id();
        let seq = record.seq();
        writer.write_all(b">")?;
        writer.write_all(id)?;
        writer.write_all(b"\n")?;
        writer.write_all(&seq)?;
        writer.write_all(b"\n")?;
        hasher.update(id);
        hasher.update(&seq);
        records += 1;
        bases += seq.len() as u64;
    }
    writer.flush()?;
    drop(writer);
    // Check the total as well as the kept count: a stale --survey-cache can shift
    // ordinals without changing how many records are kept.
    ensure!(
        u64::from(ordinal) == surveyed,
        "reference holds {ordinal} records but the survey saw {surveyed}; \
         the reference changed between passes, or --survey-cache is stale"
    );
    ensure!(
        records == selection.kept,
        "emitted {records} records but selected {}; the reference changed between passes",
        selection.kept
    );
    temp.as_file().sync_all()?;
    temp.persist(output)
        .with_context(|| format!("writing {}", output.display()))?;
    Ok(Emitted {
        records,
        bases,
        digest: hasher.finalize(),
    })
}

// --- Report ---

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Provenance manifest written next to the curated FASTA: enough to reproduce
/// it and to tell two curated databases apart.
pub struct Provenance<'a> {
    pub plan: &'a Plan,
    pub policy: &'a Policy,
    pub selector: Selector,
    pub seed: u64,
    pub selection: &'a Selection,
    pub emitted: Option<&'a Emitted>,
    pub invariants: &'a Invariants,
    pub reference: &'a Path,
    pub output: Option<&'a Path>,
    pub wall: f64,
}

pub fn report(p: &Provenance) -> String {
    let Provenance {
        plan,
        policy,
        selector,
        seed,
        selection,
        emitted,
        invariants,
        reference,
        output,
        wall,
    } = *p;
    let c = &selection.counts;
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"tool_version\": {},\n  \"curation_version\": {},\n  \"algorithm_version\": {},\n",
        json_string(env!("CARGO_PKG_VERSION")),
        sketch::CURATION_VERSION,
        crate::ALGORITHM_VERSION
    ));
    out.push_str(&format!(
        "  \"policy\": {{\n    \"cap\": {},\n    \"demote_length\": {},\n    \"max_length\": {},\n    \
         \"max_ambiguity\": {},\n    \"min_words\": {},\n    \"ceiling\": {},\n    \
         \"cover_bands\": {},\n    \"shallow_cap\": [{}, {}, {}],\n    \"shallow_ratio\": {},\n    \
         \"drop_shallow\": {},\n    \"drop_organellar\": {},\n    \
         \"select\": {},\n    \"seed\": {}\n  }},\n",
        policy.cap,
        policy.demote_length,
        policy
            .max_length
            .map_or("null".to_owned(), |m| m.to_string()),
        policy.max_ambiguity,
        policy.min_words,
        policy.ceiling,
        policy.cover_bands,
        policy.shallow.covered,
        policy.shallow.orphan_order,
        policy.shallow.unplaced,
        policy.shallow.ratio,
        policy.drop_shallow,
        policy.drop_organellar,
        json_string(&format!("{selector:?}").to_lowercase()),
        seed
    ));
    out.push_str(&format!(
        "  \"policy_digest\": {},\n",
        json_string(&hex(&selection.policy_digest))
    ));
    out.push_str(&format!(
        "  \"input\": {{\n    \"path\": {},\n    \"records\": {},\n    \"parsed\": {},\n    \
         \"parse_failures\": {},\n    \"bases\": {},\n    \"groups\": {},\n    \
         \"length\": {{\"min\": {}, \"p01\": {}, \"p25\": {}, \"median\": {}, \"p75\": {}, \
         \"p95\": {}, \"p99\": {}, \"max\": {}, \"mean\": {:.1}}}\n  }},\n",
        json_string(&reference.display().to_string()),
        plan.records,
        plan.parsed,
        plan.records - plan.parsed,
        plan.bases,
        plan.groups,
        plan.lengths.min,
        plan.lengths.p01,
        plan.lengths.p25,
        plan.lengths.median,
        plan.lengths.p75,
        plan.lengths.p95,
        plan.lengths.p99,
        plan.lengths.max,
        plan.lengths.mean
    ));
    match (output, emitted) {
        (Some(path), Some(e)) => out.push_str(&format!(
            "  \"output\": {{\n    \"path\": {},\n    \"records\": {},\n    \"bases\": {},\n    \
             \"blake3\": {}\n  }},\n",
            json_string(&path.display().to_string()),
            e.records,
            e.bases,
            json_string(&e.digest.to_hex())
        )),
        _ => out.push_str("  \"output\": null,\n"),
    }
    out.push_str(&format!(
        "  \"dropped\": {{\n    \"low_complexity\": {},\n    \"ambiguous\": {},\n    \
         \"oversize\": {},\n    \"duplicates_exact\": {},\n    \"over_cap\": {},\n    \
         \"over_shallow_cap\": {},\n    \"deliberate\": {},\n    \
         \"restored_for_group_preservation\": {}\n  }},\n",
        c.low_complexity,
        c.ambiguous,
        c.oversize,
        c.duplicates,
        c.over_cap,
        c.over_shallow_cap,
        c.deliberate,
        c.restored
    ));
    out.push_str(&format!(
        "  \"invariants\": {{\n    \"groups_preserved\": {},\n    \"names_preserved\": {},\n    \
         \"missing\": [{}]\n  }},\n",
        invariants.groups_preserved,
        invariants.names_preserved,
        invariants
            .missing
            .iter()
            .map(|m| json_string(m))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!(
        "  \"hazards\": {{\n    \"organellar_records\": {},\n    \"organellar_bases\": {},\n    \
         \"demoted_records\": {},\n    \"demoted_bases\": {},\n    \
         \"conflicting_duplicate_groups\": {},\n    \"occupancy\": [",
        plan.organellar,
        plan.organellar_bases,
        plan.demoted,
        plan.demoted_bases,
        plan.conflicting_duplicates
    ));
    out.push_str(
        &plan
            .occupancy
            .iter()
            .map(|c| {
                format!(
                    "\n      {{\"class\": {}, \"floor\": {}, \"records\": {}, \"expected_score\": {:.2}}}",
                    json_string(c.label),
                    c.floor,
                    c.records,
                    crate::SAMPLE_SIZE as f64 * c.floor
                )
            })
            .collect::<Vec<_>>()
            .join(","),
    );
    out.push_str("\n    ]\n  },\n");
    out.push_str("  \"shallow\": {");
    out.push_str(&format!(
        "\"order_has_genus\": {}, \"order_lacks_genus\": {}, \"no_order\": {}}},\n",
        plan.shallow[Shallow::Covered as usize],
        plan.shallow[Shallow::OrphanOrder as usize],
        plan.shallow[Shallow::Unplaced as usize]
    ));
    out.push_str("  \"cap_curve\": [");
    out.push_str(
        &plan
            .cap_curve
            .iter()
            .map(|(cap, kept)| format!("{{\"cap\": {cap}, \"records\": {kept}}}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("],\n");
    out.push_str(&format!(
        "  \"kept\": {},\n  \"wall_s\": {wall:.3}\n}}\n",
        selection.kept
    ));
    out
}
