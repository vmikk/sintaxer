use crate::{
    BOOTSTRAPS, SAMPLE_SIZE, WORDS,
    index::Index,
    rank::{self, Bound},
    rng, scoring, sequence,
    taxonomy::RANKS,
    weight,
};
use anyhow::{Result, ensure};
use clap::ValueEnum;
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Strand {
    Both,
    Plus,
}
#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub cutoff: f64,
    pub strand: Strand,
    /// Score every reference in every replicate, as published SINTAX does. Slow; used to check the fast path.
    pub exact: bool,
    /// Size the candidate set is grown to before the replicates run.
    pub candidates: usize,
    /// Largest tolerated expected number of pruned references that could have won a
    /// replicate. Exceeding it widens the candidate set.
    pub risk: f64,
    /// Per-word weight table used in scoring. `Off` is the published algorithm.
    pub weights: weight::Source,
    /// Percentage of occurring words treated as informative under `weights`.
    pub weight_share: u8,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 1,
            cutoff: 0.8,
            strand: Strand::Both,
            exact: false,
            candidates: 2048,
            risk: 0.0,
            weights: weight::Source::Off,
            weight_share: weight::DEFAULT_SHARE,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.cutoff.is_finite() && (0.0..=1.0).contains(&self.cutoff),
            "cutoff must be finite and in [0,1]"
        );
        ensure!(self.candidates > 0, "candidate target must be positive");
        ensure!(
            self.risk.is_finite() && self.risk >= 0.0,
            "risk tolerance must be finite and non-negative"
        );
        // `Bound` assumes an unweighted binomial score. Weighting breaks that model in the
        // unsafe direction, so refuse to report a bound rather than report a wrong one.
        ensure!(
            self.risk == 0.0 || self.weights == weight::Source::Off,
            "--risk models unweighted scores and cannot be combined with --weights"
        );
        ensure!(
            (1..=99).contains(&self.weight_share),
            "weight share must be a percentage in 1..=99"
        );
        Ok(())
    }
}

#[derive(Default, Clone, Debug)]
pub struct Timings {
    pub extraction: Duration,
    pub sampling: Duration,
    pub scoring: Duration,
    pub taxonomy: Duration,
    /// Whether any strand needed a wider candidate set than the default.
    pub escalated: bool,
    /// Bytes that were neither ACGT nor a recognised ambiguity code.
    pub unknown_bases: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub strand: Option<char>,
    pub ranks: Vec<(u8, String, u8)>,
}
impl Prediction {
    pub fn unclassified() -> Self {
        Self {
            strand: None,
            ranks: Vec::new(),
        }
    }
    pub fn tsv(&self, label: &str, cutoff: f64) -> String {
        if self.strand.is_none() {
            return format!("{label}\t\t\t\n");
        }
        let full = self
            .ranks
            .iter()
            .map(|(r, n, c)| format!("{}:{n}({:.2})", *r as char, *c as f64 / BOOTSTRAPS as f64))
            .collect::<Vec<_>>()
            .join(",");
        let filtered = self
            .ranks
            .iter()
            .filter(|(_, _, c)| *c as f64 / BOOTSTRAPS as f64 >= cutoff)
            .map(|(r, n, _)| format!("{}:{n}", *r as char))
            .collect::<Vec<_>>()
            .join(",");
        format!("{label}\t{full}\t{}\t{filtered}\n", self.strand.unwrap())
    }
}

/// Per-worker scratch, reused across queries.
pub struct Workspace {
    rank: rank::Workspace,
    scores: Vec<u8>,
    forward: Vec<u16>,
    reverse: Vec<u16>,
    seen: Box<[u64; WORDS / 64]>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            rank: rank::Workspace::default(),
            scores: Vec::new(),
            forward: Vec::new(),
            reverse: Vec::new(),
            seen: Box::new([0; WORDS / 64]),
        }
    }
}

/// Positions drawn from the query vocabulary, with replacement.
///
/// Positions rather than word codes, so the exact and candidate paths consume the
/// RNG identically and stay comparable.
pub fn samples(vocabulary: usize, key: &blake3::Hash, strand: u8) -> Vec<[u16; SAMPLE_SIZE]> {
    (0..BOOTSTRAPS)
        .map(|boot| {
            let mut rng = rng::stream(key, strand, boot, 0);
            std::array::from_fn(|_| rng.bounded(vocabulary as u64) as u16)
        })
        .collect()
}

/// One strand's replicate outcomes.
struct Outcome {
    max: u8,
    winners: Vec<u32>,
    escalated: bool,
}

pub fn consensus(index: &Index, winners: &[u32], strand: char) -> Prediction {
    if winners.len() < BOOTSTRAPS / 2 {
        return Prediction::unclassified();
    }
    let lineages: Vec<_> = winners.iter().map(|r| index.lineage(*r as usize)).collect();
    let mut included = vec![true; lineages.len()];
    let mut ranks = Vec::new();
    for (rank, &code) in RANKS.iter().enumerate() {
        // Keyed by name. All candidates share the selected ancestor here, so names identify nodes.
        let mut counts = BTreeMap::<&str, (u32, u8)>::new();
        for (i, lineage) in lineages.iter().enumerate() {
            if included[i] {
                let node = lineage[rank];
                counts.entry(index.name(node)).or_insert((node, 0)).1 += 1;
            }
        }
        let mut best = None;
        for (name, (node, count)) in counts {
            if best.is_none_or(|(_, _, c)| count > c) {
                best = Some((name, node, count));
            }
        }
        let Some((name, node, count)) = best else {
            break;
        };
        for (i, lineage) in lineages.iter().enumerate() {
            included[i] &= lineage[rank] == node;
        }
        if !name.is_empty() {
            ranks.push((code, name.to_owned(), count));
        }
    }
    if ranks.is_empty() {
        Prediction::unclassified()
    } else {
        Prediction {
            strand: Some(strand),
            ranks,
        }
    }
}

/// Every replicate scored against the whole database, as published SINTAX does.
fn exact_tops(
    index: &Index,
    words: &[u16],
    samples: &[[u16; SAMPLE_SIZE]],
    weights: &weight::Table,
    ws: &mut Workspace,
) -> Result<Vec<scoring::Top>> {
    let mut tops = Vec::with_capacity(BOOTSTRAPS);
    for sample in samples {
        let codes: [u16; SAMPLE_SIZE] = std::array::from_fn(|i| words[sample[i] as usize]);
        scoring::scalar_scores(index, &codes, weights, &mut ws.scores)?;
        tops.push(scoring::Top::from_scores(&ws.scores));
    }
    Ok(tops)
}

/// Expected number of replicates whose winner could have differed had the pruned
/// references been scored.
fn strand_risk(tops: &[scoring::Top], histogram: &[u32], threshold: u16, vocabulary: usize) -> f64 {
    // Indexed by score, so it must span the weighted ceiling, not just SAMPLE_SIZE.
    let mut cache = [f64::NAN; SAMPLE_SIZE * weight::MAX_WEIGHT as usize + 1];
    let mut risk = 0.0;
    for top in tops {
        // Scores below two are never recorded, so that is the floor a pruned reference must reach.
        let score = top.score.max(2) as usize;
        if cache[score].is_nan() {
            cache[score] = Bound::new(histogram, threshold, vocabulary, score as u8).risk();
        }
        risk += cache[score];
    }
    risk
}

/// Score every replicate of one strand, exactly or against candidates.
#[allow(clippy::too_many_arguments)]
fn score_strand(
    index: &Index,
    words: &[u16],
    samples: &[[u16; SAMPLE_SIZE]],
    key: &blake3::Hash,
    strand: u8,
    config: &Config,
    weights: &weight::Table,
    ws: &mut Workspace,
) -> Result<Outcome> {
    let mut escalated = false;
    // With a candidate set that is a large share of the database, the rank pass and
    // projection cost more than scoring everything, so small databases go exact.
    let worth_it = !config.exact && index.references > config.candidates.saturating_mul(4);
    let mut from_candidates = worth_it;
    let mut tops = if !worth_it {
        exact_tops(index, words, samples, weights, ws)?
    } else {
        ws.rank.rank(index, words)?;
        // Widen the candidate set a couple of times at most, then fall back to the exact
        // path. Worst case is one rank pass plus one exact scan.
        let ceiling = config.candidates.saturating_mul(64).min(index.references);
        let mut target = config.candidates;
        loop {
            let threshold = ws.rank.select(target);
            ws.rank.project(index, words, weights)?;
            let mut tops = Vec::with_capacity(BOOTSTRAPS);
            for sample in samples {
                tops.push(ws.rank.replicate(sample));
            }
            // Ties can push the set far past its target, making projection dearer than an exact
            // scan; a threshold no candidate exceeds means the rank pass separated nothing.
            let oversized = ws.rank.candidates.len() * 2 >= index.references;
            let unseparated = ws
                .rank
                .candidates
                .iter()
                .all(|&r| ws.rank.counts[r as usize] <= threshold);
            let risky = config.risk > 0.0
                && strand_risk(&tops, &ws.rank.histogram, threshold, words.len()) > config.risk;
            if !oversized && !unseparated && !risky {
                break tops;
            }
            escalated = true;
            if oversized || unseparated || target >= ceiling || threshold <= 1 {
                from_candidates = false;
                break exact_tops(index, words, samples, weights, ws)?;
            }
            target = target.saturating_mul(8).min(ceiling);
        }
    };
    tops.truncate(BOOTSTRAPS);

    let mut max = 0;
    let mut winners = Vec::with_capacity(BOOTSTRAPS);
    for (boot, top) in tops.iter().enumerate() {
        max = max.max(top.score);
        if top.score >= weight::FLOOR {
            let mut ties = rng::stream(key, strand, boot, 1);
            let offset = ties.bounded(top.count as u64) as usize;
            winners.push(if from_candidates {
                ws.rank.reference_at(top, offset)
            } else {
                top.nth(offset)
            });
        }
    }
    Ok(Outcome {
        max,
        winners,
        escalated,
    })
}

pub fn classify(
    index: &Index,
    sequence: &[u8],
    config: &Config,
    weights: &weight::Table,
    ws: &mut Workspace,
) -> Result<(Prediction, Timings)> {
    config.validate()?;
    ensure!(
        weights.source() == config.weights,
        "weight table does not match the configured source"
    );
    let mut timings = Timings::default();
    let start = Instant::now();
    let (normalized, unknown) = sequence::normalize(sequence);
    rank::vocabulary(&normalized, &mut ws.seen, &mut ws.forward);
    timings.extraction += start.elapsed();
    timings.unknown_bases += unknown;
    if ws.forward.len() < SAMPLE_SIZE {
        return Ok((Prediction::unclassified(), timings));
    }
    let key = rng::query_key(config.seed, &normalized);
    let mut best: Option<(u8, Vec<u32>, char)> = None;
    for strand in 0..if config.strand == Strand::Both { 2 } else { 1 } {
        let start = Instant::now();
        if strand == 1 {
            let forward = std::mem::take(&mut ws.forward);
            rank::revcomp_vocabulary(&forward, &mut ws.seen, &mut ws.reverse);
            ws.forward = forward;
        }
        timings.extraction += start.elapsed();
        let words = if strand == 0 {
            std::mem::take(&mut ws.forward)
        } else {
            std::mem::take(&mut ws.reverse)
        };

        let start = Instant::now();
        let samples = samples(words.len(), &key, strand);
        timings.sampling += start.elapsed();

        let start = Instant::now();
        let outcome = score_strand(index, &words, &samples, &key, strand, config, weights, ws);
        timings.scoring += start.elapsed();
        if strand == 0 {
            ws.forward = words;
        } else {
            ws.reverse = words;
        }
        let outcome = outcome?;
        timings.escalated |= outcome.escalated;

        if best.as_ref().is_none_or(|(score, hits, _)| {
            (outcome.max, outcome.winners.len()) > (*score, hits.len())
        }) {
            best = Some((
                outcome.max,
                outcome.winners,
                if strand == 0 { '+' } else { '-' },
            ));
        }
    }
    let (_, winners, strand) = best.unwrap();
    let start = Instant::now();
    let prediction = consensus(index, &winners, strand);
    timings.taxonomy += start.elapsed();
    Ok((prediction, timings))
}
