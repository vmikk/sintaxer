use crate::{
    BOOTSTRAPS, SAMPLE_SIZE,
    index::Index,
    rng,
    scoring::{self, Engine, Sample, Workspace},
    sequence,
    taxonomy::RANKS,
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
    pub engine: Engine,
    pub tile_size: usize,
    pub bootstrap_batch: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 1,
            cutoff: 0.8,
            strand: Strand::Both,
            engine: Engine::Scalar,
            tile_size: 1024,
            bootstrap_batch: 16,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.cutoff.is_finite() && (0.0..=1.0).contains(&self.cutoff),
            "cutoff must be finite and in [0,1]"
        );
        ensure!(
            [256, 1024, 4096].contains(&self.tile_size),
            "invalid tile size"
        );
        ensure!(
            [8, 16, 32].contains(&self.bootstrap_batch),
            "invalid bootstrap batch"
        );
        ensure!(
            self.engine != Engine::Avx2 || scoring::avx2_available(),
            "AVX2 unavailable"
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

pub fn samples(words: &[u16], key: &blake3::Hash, strand: u8) -> Vec<Sample> {
    (0..BOOTSTRAPS)
        .map(|boot| {
            let mut rng = rng::stream(key, strand, boot, 0);
            std::array::from_fn(|_| words[rng.bounded(words.len() as u64) as usize])
        })
        .collect()
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

pub fn classify(
    index: &Index,
    sequence: &[u8],
    config: &Config,
    ws: &mut Workspace,
) -> Result<(Prediction, Timings)> {
    config.validate()?;
    let mut timings = Timings::default();
    let start = Instant::now();
    let normalized = sequence::normalize(sequence)?;
    let forward = sequence::unique_words(&normalized);
    timings.extraction += start.elapsed();
    if forward.len() < SAMPLE_SIZE {
        return Ok((Prediction::unclassified(), timings));
    }
    let key = rng::query_key(config.seed, &normalized);
    let mut best: Option<(u8, Vec<u32>, char)> = None;
    for strand in 0..if config.strand == Strand::Both { 2 } else { 1 } {
        let start = Instant::now();
        let words = if strand == 0 {
            forward.clone()
        } else {
            sequence::unique_words(&sequence::reverse_complement(&normalized))
        };
        timings.extraction += start.elapsed();
        let start = Instant::now();
        let samples = samples(&words, &key, strand);
        timings.sampling += start.elapsed();
        let (mut max, mut winners) = (0, Vec::with_capacity(BOOTSTRAPS));
        let start = Instant::now();
        scoring::score_bootstraps(
            index,
            &samples,
            config.engine,
            config.tile_size,
            config.bootstrap_batch,
            ws,
            |boot, top| {
                max = max.max(top.score);
                if top.score >= 2 {
                    let mut ties = rng::stream(&key, strand, boot, 1);
                    winners.push(top.nth(ties.bounded(top.count as u64) as usize));
                }
            },
        )?;
        timings.scoring += start.elapsed();
        if best
            .as_ref()
            .is_none_or(|(score, hits, _)| (max, winners.len()) > (*score, hits.len()))
        {
            best = Some((max, winners, if strand == 0 { '+' } else { '-' }));
        }
    }
    let (_, winners, strand) = best.unwrap();
    let start = Instant::now();
    let prediction = consensus(index, &winners, strand);
    timings.taxonomy += start.elapsed();
    Ok((prediction, timings))
}
