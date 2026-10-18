//! PROTAX-style scoring: a calibrated probability with an explicit novel-taxon
//! outcome, built from the rank pass counts. Per node, a multinomial logistic over
//! the children plus an all-zero unknown branch (Somervuo et al. 2016).

use crate::{K, WORDS, index::Index};
use anyhow::{Context, Result, bail, ensure};
use std::path::Path;

/// Stored ranks, `d` through `g`, one coefficient block each.
pub const LEVELS: usize = 7;

/// Design-matrix columns. The all-zero unknown branch is never built.
pub const FEATURES: usize = 7;
/// Intercept for a named child with at least one scored reference.
pub const F_PRESENT: usize = 0;
/// Best PROTAX similarity to any reference under the child.
pub const F_S1: usize = 1;
/// Within-taxon mislabelling evidence; see [`gap`].
pub const F_GAP: usize = 2;
/// Share of bootstrap replicates whose winner fell under the child.
pub const F_VOTE: usize = 3;
/// `ln` of genus-rank descendants, standing in for PROTAX's `log(prior)`.
pub const F_GENERA: usize = 4;
/// Best similarity to the parent's unnamed references. Shared by every row.
pub const F_SENTINEL: usize = 5;
/// Intercept for a named child no scored reference fell under.
pub const F_ABSENT: usize = 6;

pub const COLUMNS: [&str; FEATURES] = [
    "present", "s1", "gap", "vote", "genera", "sentinel", "absent",
];

/// Identity below which a match carries no information, as in PROTAX
/// (`sim = identity - 0.7`, clamped at zero).
const FLOOR_IDENTITY: f32 = 0.7;

/// Fitted coefficients, one block per rank.
#[derive(Clone, Debug)]
pub struct Params {
    beta: [[f64; FEATURES]; LEVELS],
    digest: String,
}

impl Params {
    pub fn beta(&self, rank: usize) -> &[f64; FEATURES] {
        &self.beta[rank]
    }
    /// BLAKE3 of the parameter file, reported by `--profile`.
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Parse the whitespace-separated coefficient table, one line of floats per rank.
    pub fn parse(text: &str, digest: String) -> Result<Self> {
        let mut beta = [[0.0f64; FEATURES]; LEVELS];
        let mut seen = [false; LEVELS];
        for (number, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split_whitespace();
            let code = fields.next().unwrap();
            let rank = crate::taxonomy::RANKS
                .iter()
                .position(|r| code.as_bytes() == [*r])
                .with_context(|| format!("line {}: unknown rank {code:?}", number + 1))?;
            ensure!(!seen[rank], "line {}: rank {code} repeated", number + 1);
            seen[rank] = true;
            for (column, slot) in beta[rank].iter_mut().enumerate() {
                let value = fields.next().with_context(|| {
                    format!(
                        "line {}: rank {code} is missing column {}",
                        number + 1,
                        COLUMNS[column]
                    )
                })?;
                *slot = value
                    .parse::<f64>()
                    .with_context(|| format!("line {}: {value:?} is not a number", number + 1))?;
                ensure!(
                    slot.is_finite(),
                    "line {}: rank {code} column {} is not finite",
                    number + 1,
                    COLUMNS[column]
                );
            }
            ensure!(
                fields.next().is_none(),
                "line {}: rank {code} has more than {FEATURES} columns",
                number + 1
            );
        }
        if let Some(missing) = seen.iter().position(|s| !s) {
            bail!(
                "no coefficients for rank {}",
                crate::taxonomy::RANKS[missing] as char
            );
        }
        Ok(Self { beta, digest })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading protax parameters {}", path.display()))?;
        let digest = blake3::hash(&bytes).to_hex().to_string();
        let text = String::from_utf8(bytes).context("protax parameters must be UTF-8")?;
        Self::parse(&text, digest)
    }
}

/// One scored reference: a candidate, or a replicate winner from outside the candidate set.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub reference: u32,
    pub similarity: f32,
}

/// Estimated query-reference identity on PROTAX's scale.
///
/// Shared word count is corrected for chance matches (`v / WORDS`), then
/// `containment^(1/K)` estimates `1 - d` (Mash on containment). Short references are
/// penalised; the fitted coefficients absorb that.
pub fn similarity(index: &Index, reference: usize, count: u16, vocabulary: usize) -> f32 {
    if count == 0 || vocabulary == 0 {
        return 0.0;
    }
    let observed = f32::from(count) / vocabulary as f32;
    let chance = index.vocabulary(reference) as f32 / WORDS as f32;
    if chance >= 1.0 {
        return 0.0;
    }
    let adjusted = ((observed - chance) / (1.0 - chance)).clamp(0.0, 1.0);
    let identity = adjusted.powf(1.0 / K as f32);
    (identity - FLOOR_IDENTITY).max(0.0)
}

/// PROTAX's `eml`: how far the best reference in a taxon stands above the rest of it,
/// shrunk by `1 - n^-0.2` over all `n` references under the node. Uses the gap to
/// the second-best hit; `scored < 2` leaves `second` at zero, the strongest signal.
pub fn gap(first: f32, second: f32, references: u32, scored: u32) -> f64 {
    if references <= 1 {
        return 0.0;
    }
    let second = if scored >= 2 { second } else { 0.0 };
    let weight = 1.0 - f64::from(references).powf(-0.2);
    weight * f64::from(first - second).max(0.0)
}

/// One rank of a resolved lineage.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub rank: u8,
    pub node: u32,
    /// `P(query belongs to this node)`, the product down the path.
    pub probability: f64,
    /// `P(query belongs to no named child of this node's parent)`, on the same scale.
    /// Pools the unknown branch with the unmatched named children.
    pub novel: f64,
}

/// One design-matrix row: a named child with at least one scored reference.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub node: u32,
    pub s1: f32,
    pub gap: f64,
    pub vote: f64,
    pub genera: f64,
}

impl Row {
    fn score(&self, beta: &[f64; FEATURES], sentinel: f32) -> f64 {
        beta[F_PRESENT]
            + beta[F_S1] * f64::from(self.s1)
            + beta[F_GAP] * self.gap
            + beta[F_VOTE] * self.vote
            + beta[F_GENERA] * self.genera
            + beta[F_SENTINEL] * f64::from(sentinel)
    }
}

/// The multinomial's design matrix at one node: matched children as [`Design::rows`],
/// unmatched named children as a count, and an implicit all-zero unknown branch.
#[derive(Default, Debug)]
pub struct Design {
    pub rows: Vec<Row>,
    /// Named children of this node that no scored reference fell under.
    pub absent: u32,
    /// Best similarity among the node's unnamed references, shared by every row.
    /// Zero when the node has no unnamed child.
    pub sentinel: f32,
    /// The unnamed child, if any. Not a row; a lineage passes through it only when
    /// nothing is named at this rank.
    pub sentinel_node: Option<u32>,
}

impl Design {
    fn absent_score(&self, beta: &[f64; FEATURES]) -> f64 {
        beta[F_ABSENT] + beta[F_SENTINEL] * f64::from(self.sentinel)
    }
    /// Softmax over `rows`, the pooled absent children and the unknown branch.
    ///
    /// Returns the pooled probability of none of the visible named children.
    fn probabilities(&self, beta: &[f64; FEATURES], into: &mut Vec<f64>) -> f64 {
        into.clear();
        let absent_z = self.absent_score(beta);
        let mut peak = 0.0f64;
        for row in &self.rows {
            peak = peak.max(row.score(beta, self.sentinel));
        }
        if self.absent > 0 {
            peak = peak.max(absent_z);
        }
        // The unknown branch's row is all zeros, so its scaled score is exp(-peak).
        let mut unknown = (-peak).exp();
        let mut total = unknown;
        if self.absent > 0 {
            let pooled = f64::from(self.absent) * (absent_z - peak).exp();
            unknown += pooled;
            total += pooled;
        }
        for row in &self.rows {
            let term = (row.score(beta, self.sentinel) - peak).exp();
            total += term;
            into.push(term);
        }
        for p in into.iter_mut() {
            *p /= total;
        }
        unknown / total
    }
}

/// Per-node evidence accumulated over the scored references under one parent.
#[derive(Clone, Copy)]
struct Entry {
    stamp: u32,
    node: u32,
    first: f32,
    second: f32,
    scored: u32,
    votes: u32,
}

/// Open-addressed node tally, reused across ranks and queries. A generation stamp
/// replaces clearing.
pub struct Workspace {
    entries: Vec<Entry>,
    mask: usize,
    stamp: u32,
    occupied: Vec<u32>,
    lineages: Vec<[u32; LEVELS]>,
    winners: Vec<[u32; LEVELS]>,
    design: Design,
    probabilities: Vec<f64>,
}

const SLOTS: usize = 8192;

impl Default for Workspace {
    fn default() -> Self {
        Self {
            entries: vec![
                Entry {
                    stamp: 0,
                    node: 0,
                    first: 0.0,
                    second: 0.0,
                    scored: 0,
                    votes: 0,
                };
                SLOTS
            ],
            mask: SLOTS - 1,
            stamp: 0,
            occupied: Vec::new(),
            lineages: Vec::new(),
            winners: Vec::new(),
            design: Design::default(),
            probabilities: Vec::new(),
        }
    }
}

impl Workspace {
    /// Slot index for `node`, inserting an empty entry if it is new.
    fn slot(&mut self, node: u32) -> usize {
        // Fibonacci hashing: node ids are small and dense and would cluster under linear probing.
        let mut i = (node.wrapping_mul(2_654_435_769) as usize >> 13) & self.mask;
        loop {
            let entry = &mut self.entries[i];
            if entry.stamp == self.stamp && entry.node == node {
                return i;
            }
            if entry.stamp != self.stamp {
                *entry = Entry {
                    stamp: self.stamp,
                    node,
                    first: 0.0,
                    second: 0.0,
                    scored: 0,
                    votes: 0,
                };
                self.occupied.push(i as u32);
                return i;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Cache the lineage of every scored reference and replicate winner, once per descent.
    pub fn prepare(&mut self, index: &Index, hits: &[Hit], winners: &[u32]) {
        self.lineages.clear();
        self.lineages
            .extend(hits.iter().map(|h| index.lineage(h.reference as usize)));
        self.winners.clear();
        self.winners
            .extend(winners.iter().map(|&r| index.lineage(r as usize)));
    }

    /// Build the design matrix for the children of `parent` at `rank`.
    ///
    /// Call [`Workspace::prepare`] first. `None` when no scored reference lies under `parent`.
    pub fn design(
        &mut self,
        index: &Index,
        hits: &[Hit],
        winners: usize,
        rank: usize,
        parent: u32,
    ) -> Option<&Design> {
        // On wrap-around, clear the table so stale entries cannot look current.
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.entries.iter_mut().for_each(|e| e.stamp = u32::MAX);
            self.stamp = 1;
        }
        self.occupied.clear();
        for (i, hit) in hits.iter().enumerate() {
            let lineage = self.lineages[i];
            if rank > 0 && lineage[rank - 1] != parent {
                continue;
            }
            let slot = self.slot(lineage[rank]);
            let entry = &mut self.entries[slot];
            entry.scored += 1;
            if hit.similarity > entry.first {
                entry.second = entry.first;
                entry.first = hit.similarity;
            } else if hit.similarity > entry.second {
                entry.second = hit.similarity;
            }
        }
        for i in 0..self.winners.len() {
            let lineage = self.winners[i];
            if rank == 0 || lineage[rank - 1] == parent {
                let slot = self.slot(lineage[rank]);
                self.entries[slot].votes += 1;
            }
        }
        if self.occupied.is_empty() {
            return None;
        }

        // References with no name at this rank sit under an unnamed sentinel. A good unnamed
        // match suggests the query's taxon is absent, so its similarity enters every row.
        let design = &mut self.design;
        design.rows.clear();
        design.sentinel = 0.0;
        design.sentinel_node = None;
        let mut named = 0u32;
        let votes = winners.max(1) as f64;
        for &slot in &self.occupied {
            let entry = self.entries[slot as usize];
            if index.name(entry.node).is_empty() {
                design.sentinel = entry.first;
                design.sentinel_node = Some(entry.node);
                continue;
            }
            named += 1;
            design.rows.push(Row {
                node: entry.node,
                s1: entry.first,
                gap: gap(
                    entry.first,
                    entry.second,
                    index.node_references(entry.node),
                    entry.scored,
                ),
                vote: f64::from(entry.votes) / votes,
                genera: f64::from(index.node_genera(entry.node).max(1)).ln(),
            });
        }
        // Named children nothing matched; pooled with the unknown branch, as in PROTAX.
        // `node_children` counts named children only.
        design.absent = index.node_children(parent).saturating_sub(named);
        Some(design)
    }
}

/// Walk the taxonomy from the root, one multinomial logistic per rank.
///
/// `hits` are the scored references and `winners` the replicate winners (for the vote
/// predictor). Probabilities multiply down the path, so confidence is monotone.
pub fn resolve(
    index: &Index,
    params: &Params,
    hits: &[Hit],
    winners: &[u32],
    floor: f64,
    ws: &mut Workspace,
) -> Vec<Step> {
    ws.prepare(index, hits, winners);
    let mut steps = Vec::with_capacity(LEVELS);
    let mut parent = 0u32;
    let mut reached = 1.0f64;

    for rank in 0..LEVELS {
        if reached < floor {
            break;
        }
        let beta = *params.beta(rank);
        let mut probabilities = std::mem::take(&mut ws.probabilities);
        let Some(design) = ws.design(index, hits, winners.len(), rank, parent) else {
            ws.probabilities = probabilities;
            break;
        };
        let unknown = design.probabilities(&beta, &mut probabilities);
        let best = design
            .rows
            .iter()
            .zip(&probabilities)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(row, &p)| (row.node, p));
        let sentinel = design.sentinel_node;
        ws.probabilities = probabilities;

        let novel = reached * unknown;
        let Some((node, conditional)) = best else {
            // No named candidate: descend through the sentinel without spending probability or
            // emitting a rank, so annotation gaps can reach named lower ranks.
            match sentinel {
                Some(node) => {
                    parent = node;
                    continue;
                }
                None => break,
            }
        };
        reached *= conditional;
        steps.push(Step {
            rank: rank as u8,
            node,
            probability: reached,
            novel,
        });
        parent = node;
    }
    steps
}
