use crate::{
    BOOTSTRAPS, SAMPLE_SIZE, WORDS,
    index::{Index, Locator},
    protax,
    rank::{self, Bound},
    rng, scoring, sequence, taxonomy,
    taxonomy::RANKS,
    weight,
};
use anyhow::{Context, Result, ensure};
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
/// Which decision rule turns the scored references into a lineage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Scorer {
    /// Published SINTAX: per-rank plurality vote over the replicate winners.
    Sintax,
    /// PROTAX-style per-rank logistic model with a novel-taxon outcome; needs `--protax-params`.
    Protax,
}
/// Part of the database hidden from each query, for training and validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Holdout {
    /// References with exactly the query's k-mer set (the query itself and duplicates).
    #[value(name = "self")]
    Identical,
    /// The query's whole domain.
    D,
    /// The query's whole kingdom.
    K,
    /// The query's whole phylum.
    P,
    /// The query's whole class.
    C,
    /// The query's whole order.
    O,
    /// The query's whole family.
    F,
    /// The query's whole genus.
    G,
}

impl Holdout {
    /// The rank whose node is hidden, or `None` for [`Holdout::Identical`].
    pub fn rank(self) -> Option<usize> {
        match self {
            Self::Identical => None,
            Self::D => Some(0),
            Self::K => Some(1),
            Self::P => Some(2),
            Self::C => Some(3),
            Self::O => Some(4),
            Self::F => Some(5),
            Self::G => Some(6),
        }
    }
}

/// What a per-rank confidence is a fraction of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Support {
    /// All replicates (the published denominator).
    Raw,
    /// Only replicates whose winning reference is named at that rank.
    Adjusted,
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
    /// Denominator for the per-rank confidence.
    pub support: Support,
    /// Under `Support::Adjusted`, the share of replicates that must be informative at a
    /// rank before it is renormalized; below it the raw support is used.
    pub min_informative: f64,
    /// Append a fifth column holding the informative replicate count per rank.
    pub emit_informative: bool,
    /// Decision rule. `Sintax` is the published one.
    pub scorer: Scorer,
    /// Smallest cumulative probability worth descending below (PROTAX `prob_th`).
    /// Ignored under `Scorer::Sintax`.
    pub floor: f64,
    /// Hide part of the database from each query, for training and validation.
    pub holdout: Option<Holdout>,
    /// Write design-matrix rows instead of calls.
    pub emit_features: bool,
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
            support: Support::Raw,
            min_informative: 0.5,
            emit_informative: false,
            scorer: Scorer::Sintax,
            floor: 0.001,
            holdout: None,
            emit_features: false,
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
        ensure!(
            self.min_informative.is_finite() && (0.0..=1.0).contains(&self.min_informative),
            "min-informative must be finite and in [0,1]"
        );
        ensure!(
            self.floor.is_finite() && (0.0..1.0).contains(&self.floor),
            "probability floor must be finite and in [0,1)"
        );
        // Hiding zeroes counts that the exact path never reads, so it would silently hide nothing.
        ensure!(
            self.holdout.is_none() || !self.exact,
            "--holdout cannot be combined with --exact: the exact path scores \
             from the posting lists and would not hide anything"
        );
        ensure!(
            !self.emit_features || self.holdout.is_some(),
            "--emit-features needs --holdout: without it every query can see \
             its own reference and every row would be fitted to a self-match"
        );
        // These describe the replicate vote, which the protax scorer does not have.
        ensure!(
            self.scorer == Scorer::Sintax
                || (self.support == Support::Raw && !self.emit_informative),
            "--support and --emit-informative describe the replicate vote and \
             do not apply to --scorer protax"
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
/// One rank of a prediction: the chosen taxon and the votes it drew.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankCall {
    /// Index into [`RANKS`].
    pub rank: u8,
    pub name: String,
    /// Replicates whose winning reference belonged to this node.
    pub votes: u8,
}
impl RankCall {
    pub fn code(&self) -> u8 {
        RANKS[self.rank as usize]
    }
}

// No `Eq`: protax columns are floats.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub strand: Option<char>,
    pub ranks: Vec<RankCall>,
    /// Replicates whose winning reference is named at each stored rank. Undefined at or
    /// past `voted`.
    pub informative: [u8; 7],
    /// How many ranks the vote reached before running out of candidates.
    pub voted: u8,
    /// Ranks where the unnamed bloc matched or beat the best named taxon. Diagnostic only.
    pub abstention_pluralities: u8,
    /// Probabilities under `Scorer::Protax`, parallel to `ranks`; empty under `Scorer::Sintax`.
    pub probabilities: Vec<f64>,
    /// Probability that the query belongs to no named child of this rank's parent,
    /// parallel to `ranks` and on the same scale as `probabilities`.
    pub novel: Vec<f64>,
}
impl Prediction {
    pub fn unclassified() -> Self {
        Self {
            strand: None,
            ranks: Vec::new(),
            informative: [0; 7],
            voted: 0,
            abstention_pluralities: 0,
            probabilities: Vec::new(),
            novel: Vec::new(),
        }
    }
    /// Per-rank confidence under `support`, made monotone down the lineage.
    ///
    /// `Adjusted` alone can put a genus above its family; a running minimum restores
    /// monotonicity so the cutoff still yields a prefix.
    fn confidences(&self, support: Support, min_informative: f64) -> Vec<f64> {
        // Protax probabilities multiply down the path, so they are already monotone.
        if !self.probabilities.is_empty() {
            return self.probabilities.clone();
        }
        let floor = min_informative * BOOTSTRAPS as f64;
        let mut running = f64::INFINITY;
        self.ranks
            .iter()
            .map(|call| {
                let informative = f64::from(self.informative[call.rank as usize]);
                // Too few informative replicates: keep the raw support rather than inflate a few
                // annotated hits inside a mostly unannotated clade.
                let value = if support == Support::Raw || informative < floor {
                    f64::from(call.votes) / BOOTSTRAPS as f64
                } else {
                    // The winning named taxon's own votes are in `informative`, so this is never zero.
                    f64::from(call.votes) / informative
                };
                running = running.min(value);
                running
            })
            .collect()
    }
    pub fn tsv(&self, label: &str, config: &Config) -> String {
        let extra =
            usize::from(config.emit_informative) + usize::from(config.scorer == Scorer::Protax);
        let columns = "\t".repeat(3 + extra);
        if self.strand.is_none() {
            return format!("{label}{columns}\n");
        }
        let confidence = self.confidences(config.support, config.min_informative);
        let full = self
            .ranks
            .iter()
            .zip(&confidence)
            .map(|(call, c)| format!("{}:{}({c:.2})", call.code() as char, call.name))
            .collect::<Vec<_>>()
            .join(",");
        let filtered = self
            .ranks
            .iter()
            .zip(&confidence)
            .filter(|(_, c)| **c >= config.cutoff)
            .map(|(call, _)| format!("{}:{}", call.code() as char, call.name))
            .collect::<Vec<_>>()
            .join(",");
        let mut line = format!("{label}\t{full}\t{}\t{filtered}", self.strand.unwrap());
        if config.scorer == Scorer::Protax {
            // Novel-taxon probability, reported separately from the confidence.
            line.push('\t');
            line.push_str(
                &self
                    .ranks
                    .iter()
                    .zip(&self.novel)
                    .map(|(call, p)| format!("{}:{p:.3}", call.code() as char))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        if config.emit_informative {
            line.push('\t');
            line.push_str(
                &(0..usize::from(self.voted))
                    .map(|rank| format!("{}:{}", RANKS[rank] as char, self.informative[rank]))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        line.push('\n');
        line
    }
}

/// Per-worker scratch, reused across queries.
pub struct Workspace {
    rank: rank::Workspace,
    protax: protax::Workspace,
    scores: Vec<u8>,
    forward: Vec<u16>,
    reverse: Vec<u16>,
    seen: Box<[u64; WORDS / 64]>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            rank: rank::Workspace::default(),
            protax: protax::Workspace::default(),
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
    /// Scored references for [`Scorer::Protax`]: the candidate set plus any replicate winner
    /// outside it. Empty under [`Scorer::Sintax`]. Collected per strand, because the rank
    /// pass overwrites its counts on the next strand.
    hits: Vec<protax::Hit>,
}

/// The candidate set as PROTAX similarities, with outside replicate winners folded in.
fn protax_hits(
    index: &Index,
    ws: &Workspace,
    vocabulary: usize,
    winners: &[u32],
) -> Vec<protax::Hit> {
    let similarity =
        |r: u32| protax::similarity(index, r as usize, ws.rank.counts[r as usize], vocabulary);
    let mut hits: Vec<protax::Hit> = ws
        .rank
        .candidates
        .iter()
        .map(|&reference| protax::Hit {
            reference,
            similarity: similarity(reference),
        })
        .collect();
    let mut extra: Vec<u32> = winners
        .iter()
        .copied()
        .filter(|w| ws.rank.candidates.binary_search(w).is_err())
        .collect();
    extra.sort_unstable();
    extra.dedup();
    hits.extend(extra.into_iter().map(|reference| protax::Hit {
        reference,
        similarity: similarity(reference),
    }));
    hits
}

pub fn consensus(index: &Index, winners: &[u32], strand: char) -> Prediction {
    if winners.len() < BOOTSTRAPS / 2 {
        return Prediction::unclassified();
    }
    let lineages: Vec<_> = winners.iter().map(|r| index.lineage(*r as usize)).collect();
    let mut included = vec![true; lineages.len()];
    let mut ranks = Vec::new();
    let mut informative = [0u8; 7];
    let mut voted = 0u8;
    let mut abstention_pluralities = 0u8;
    for rank in 0..RANKS.len() {
        // Keyed by name. All candidates share the selected ancestor here, so names identify
        // nodes, and interning leaves at most one unnamed sentinel among them.
        let mut counts = BTreeMap::<&str, (u32, u8)>::new();
        for (i, lineage) in lineages.iter().enumerate() {
            if included[i] {
                let node = lineage[rank];
                counts.entry(index.name(node)).or_insert((node, 0)).1 += 1;
            }
        }
        if counts.is_empty() {
            break;
        }
        voted += 1;
        // A missing rank is an abstention, not a competing taxon. Interning merges all
        // unannotated siblings into one node, which would otherwise outvote split named ones.
        let mut best: Option<(&str, u32, u8)> = None;
        let mut unnamed: Option<(&str, u32, u8)> = None;
        for (name, (node, count)) in counts {
            if name.is_empty() {
                unnamed = Some((name, node, count));
            } else {
                informative[rank] += count;
                if best.is_none_or(|(_, _, c)| count > c) {
                    best = Some((name, node, count));
                }
            }
        }
        if let (Some((_, _, abstained)), Some((_, _, named))) = (unnamed, best) {
            abstention_pluralities += u8::from(abstained >= named);
        }
        // No named candidate: keep the sentinel so an annotation gap can still reach a named
        // lower rank.
        let (name, node, count) = best.or(unnamed).expect("counts is not empty");
        for (i, lineage) in lineages.iter().enumerate() {
            included[i] &= lineage[rank] == node;
        }
        if !name.is_empty() {
            ranks.push(RankCall {
                rank: rank as u8,
                name: name.to_owned(),
                votes: count,
            });
        }
    }
    if ranks.is_empty() {
        Prediction::unclassified()
    } else {
        Prediction {
            strand: Some(strand),
            ranks,
            informative,
            voted,
            abstention_pluralities,
            probabilities: Vec::new(),
            novel: Vec::new(),
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

/// One query's hidden set, resolved against the index.
#[derive(Clone, Copy, Debug)]
pub enum Hide {
    /// References holding exactly the query's k-mers: all of its words and no others.
    Identical,
    /// Everything under one node.
    Clade { node: u32, rank: usize },
}

impl Hide {
    /// Whether one reference is hidden. `Clade` walks up from the leaf.
    fn rejects(&self, index: &Index, reference: usize, count: u16, vocabulary: usize) -> bool {
        match *self {
            Self::Identical => {
                usize::from(count) == vocabulary
                    && index.vocabulary(reference) as usize == vocabulary
            }
            Self::Clade { node, rank } => {
                let mut current = index.leaf(reference);
                for _ in rank..6 {
                    current = index.parent(current);
                }
                current == node
            }
        }
    }
}

/// Resolve `--holdout` against one query's own annotation.
///
/// `None` when there is nothing to hide: no holdout, or the query has no taxon (or an
/// unknown one) at that rank.
pub fn hide_for(config: &Config, locator: &Locator, label: &str) -> Result<Option<Hide>> {
    let Some(holdout) = config.holdout else {
        return Ok(None);
    };
    let Some(rank) = holdout.rank() else {
        return Ok(Some(Hide::Identical));
    };
    let lineage = taxonomy::parse(label)
        .with_context(|| format!("--holdout needs a tax= annotation on query {label:?}"))?;
    if lineage[rank].is_none() {
        return Ok(None);
    }
    Ok(locator.locate(&lineage)[rank].map(|node| Hide::Clade { node, rank }))
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
    hide: Option<Hide>,
    ws: &mut Workspace,
) -> Result<Outcome> {
    let mut escalated = false;
    // With a candidate set that is a large share of the database, the rank pass and
    // projection cost more than scoring everything, so small databases go exact.
    // Hiding only works on the candidate path, so it always takes it.
    let worth_it =
        (!config.exact && index.references > config.candidates.saturating_mul(4)) || hide.is_some();
    let mut from_candidates = worth_it;
    let mut tops = if !worth_it {
        exact_tops(index, words, samples, weights, ws)?
    } else {
        ws.rank.rank(index, words)?;
        if let Some(hide) = hide {
            let vocabulary = words.len();
            ws.rank
                .exclude(|reference, count| hide.rejects(index, reference, count, vocabulary));
        }
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
                if hide.is_some() {
                    // The exact fallback ignores hiding, so keep the candidate result.
                    break tops;
                }
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
    // Protax needs the whole-vocabulary counts, which the exact path never computes.
    // Re-selecting also keeps the feature window the same size after an escalation.
    let hits = if config.scorer == Scorer::Protax || config.emit_features {
        if !worth_it {
            ws.rank.rank(index, words)?;
        }
        ws.rank.select(config.candidates);
        protax_hits(index, ws, words.len(), &winners)
    } else {
        Vec::new()
    };
    Ok(Outcome {
        max,
        winners,
        escalated,
        hits,
    })
}

/// Score both strands and keep the better, with everything a scorer reads.
///
/// Shared by classification and training-row emission so both see the same evidence.
/// `Ok(None)` means the query has too few distinct words to sample from.
fn best_strand(
    index: &Index,
    sequence: &[u8],
    config: &Config,
    weights: &weight::Table,
    hide: Option<Hide>,
    ws: &mut Workspace,
    timings: &mut Timings,
) -> Result<Option<(Outcome, char)>> {
    let start = Instant::now();
    let (normalized, unknown) = sequence::normalize(sequence);
    rank::vocabulary(&normalized, &mut ws.seen, &mut ws.forward);
    timings.extraction += start.elapsed();
    timings.unknown_bases += unknown;
    if ws.forward.len() < SAMPLE_SIZE {
        return Ok(None);
    }
    let key = rng::query_key(config.seed, &normalized);
    let mut best: Option<(Outcome, char)> = None;
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
        let outcome = score_strand(
            index, &words, &samples, &key, strand, config, weights, hide, ws,
        );
        timings.scoring += start.elapsed();
        if strand == 0 {
            ws.forward = words;
        } else {
            ws.reverse = words;
        }
        let outcome = outcome?;
        timings.escalated |= outcome.escalated;

        if best.as_ref().is_none_or(|(incumbent, _)| {
            (outcome.max, outcome.winners.len()) > (incumbent.max, incumbent.winners.len())
        }) {
            best = Some((outcome, if strand == 0 { '+' } else { '-' }));
        }
    }
    Ok(best)
}

pub fn classify(
    index: &Index,
    sequence: &[u8],
    config: &Config,
    weights: &weight::Table,
    params: Option<&protax::Params>,
    hide: Option<Hide>,
    ws: &mut Workspace,
) -> Result<(Prediction, Timings)> {
    config.validate()?;
    ensure!(
        weights.source() == config.weights,
        "weight table does not match the configured source"
    );
    ensure!(
        config.scorer == Scorer::Sintax || params.is_some(),
        "--scorer protax requires --protax-params"
    );
    let mut timings = Timings::default();
    let Some((outcome, strand)) =
        best_strand(index, sequence, config, weights, hide, ws, &mut timings)?
    else {
        return Ok((Prediction::unclassified(), timings));
    };
    let start = Instant::now();
    let prediction = match config.scorer {
        Scorer::Sintax => consensus(index, &outcome.winners, strand),
        Scorer::Protax => protax_consensus(
            index,
            &outcome,
            strand,
            config,
            params.expect("checked"),
            ws,
        ),
    };
    timings.taxonomy += start.elapsed();
    Ok((prediction, timings))
}

/// Turn one strand's evidence into a lineage under [`Scorer::Protax`].
///
/// Uses the same unclassified guard as [`consensus`], so both scorers agree on queries
/// with no usable hit.
fn protax_consensus(
    index: &Index,
    outcome: &Outcome,
    strand: char,
    config: &Config,
    params: &protax::Params,
    ws: &mut Workspace,
) -> Prediction {
    if outcome.winners.len() < BOOTSTRAPS / 2 {
        return Prediction::unclassified();
    }
    let steps = protax::resolve(
        index,
        params,
        &outcome.hits,
        &outcome.winners,
        config.floor,
        &mut ws.protax,
    );
    if steps.is_empty() {
        return Prediction::unclassified();
    }
    let mut prediction = Prediction {
        strand: Some(strand),
        ranks: Vec::with_capacity(steps.len()),
        informative: [0; 7],
        voted: steps.last().map_or(0, |s| s.rank + 1),
        abstention_pluralities: 0,
        probabilities: Vec::with_capacity(steps.len()),
        novel: Vec::with_capacity(steps.len()),
    };
    for step in steps {
        prediction.ranks.push(RankCall {
            rank: step.rank,
            name: index.name(step.node).to_owned(),
            // Not the confidence here, but `--profile` and callers still read it.
            votes: u8::try_from(
                outcome
                    .winners
                    .iter()
                    .filter(|&&w| index.lineage(w as usize)[step.rank as usize] == step.node)
                    .count(),
            )
            .unwrap_or(u8::MAX),
        });
        prediction.probabilities.push(step.probability);
        prediction.novel.push(step.novel);
    }
    prediction
}

/// Column names of the design-matrix dump, in output order.
pub const FEATURE_HEADER: &str =
    "query\trank\tkind\tnode\ts1\tgap\tvote\tgenera\tsentinel\tmultiplicity\ttruth\n";

/// Emit the design matrix along the query's own lineage instead of a call.
///
/// Rows follow the query's annotation, not the model's choice, so they are trainable.
/// `kind=present` rows are named children with a scored reference; `kind=absent` pools
/// the rest with a `multiplicity`. The all-zero unknown branch is never written;
/// a group without `truth=1` means the unknown branch is the true outcome.
#[allow(clippy::too_many_arguments)]
pub fn training_rows(
    index: &Index,
    sequence: &[u8],
    label: &str,
    config: &Config,
    weights: &weight::Table,
    locator: &Locator,
    ws: &mut Workspace,
) -> Result<(String, Timings)> {
    config.validate()?;
    let mut timings = Timings::default();
    let lineage = taxonomy::parse(label)
        .with_context(|| format!("--emit-features needs a tax= annotation on query {label:?}"))?;
    let truth = locator.locate(&lineage);
    let hide = hide_for(config, locator, label)?;
    let Some((outcome, _)) = best_strand(index, sequence, config, weights, hide, ws, &mut timings)?
    else {
        return Ok((String::new(), timings));
    };
    if outcome.winners.len() < BOOTSTRAPS / 2 {
        return Ok((String::new(), timings));
    }

    let start = Instant::now();
    let mut out = String::new();
    ws.protax.prepare(index, &outcome.hits, &outcome.winners);
    let mut parent = 0u32;
    for rank in 0..protax::LEVELS {
        // Taxon not in this database: stop here.
        let Some(node) = truth[rank] else { break };
        // Blank rank in the query: descend through the sentinel without emitting rows,
        // since real databases skip ranks.
        if lineage[rank].is_none() {
            parent = node;
            continue;
        }
        let Some(design) =
            ws.protax
                .design(index, &outcome.hits, outcome.winners.len(), rank, parent)
        else {
            break;
        };
        let code = RANKS[rank] as char;
        let target = truth[rank];
        let mut resolved = false;
        for row in &design.rows {
            let hit = target == Some(row.node);
            resolved |= hit;
            out.push_str(&format!(
                "{label}\t{code}\tpresent\t{}\t{:.6}\t{:.6}\t{:.4}\t{:.6}\t{:.6}\t1\t{}\n",
                row.node,
                row.s1,
                row.gap,
                row.vote,
                row.genera,
                design.sentinel,
                u8::from(hit),
            ));
        }
        if design.absent > 0 {
            // True child present in the database but unseen by this query (PROTAX's `nseq0`).
            let hit = target.is_some() && !resolved;
            out.push_str(&format!(
                "{label}\t{code}\tabsent\t0\t0\t0\t0\t0\t{:.6}\t{}\t{}\n",
                design.sentinel,
                design.absent,
                u8::from(hit),
            ));
        }
        parent = node;
    }
    timings.taxonomy += start.elapsed();
    Ok((out, timings))
}
