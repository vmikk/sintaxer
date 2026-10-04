use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use sintaxer::{
    classify::Workspace,
    classify::{self, Config, Strand, Timings},
    curate,
    index::{self, Index},
    input, taxonomy, weight,
};
use std::{
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(
    version,
    about = "Deterministic SINTAX-style amplicon classification using reusable indexes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Build an immutable index from tax=-annotated FASTA.
    Index {
        reference: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Classify FASTA/FASTQ (plain, gzip, zstd, or stdin).
    Classify {
        #[arg(default_value = "-")]
        reads: PathBuf,
        #[arg(long)]
        db: PathBuf,
        #[arg(short, long, default_value = "-")]
        output: PathBuf,
        #[arg(short, long, default_value_t = default_threads())]
        threads: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 0.8)]
        cutoff: f64,
        #[arg(long, value_enum, default_value_t = Strand::Both)]
        strand: Strand,
        /// Score every reference in every replicate, as published SINTAX does (slower).
        #[arg(long)]
        exact: bool,
        /// References carried into the replicates; larger is safer and slower.
        /// Small databases (under about four times this) skip ranking entirely.
        #[arg(long, default_value_t = 2048)]
        candidates: usize,
        /// Escalate to the exact path when the expected number of replicates a pruned
        /// reference could have tied exceeds this (0 disables). Escalates often on divergent queries.
        #[arg(long, default_value_t = 0.0)]
        risk: f64,
        /// Per-word scoring weights, down-weighting words common across the database.
        /// `off` is the published algorithm.
        #[arg(long, value_enum, default_value_t = weight::Source::Off)]
        weights: weight::Source,
        /// Percentage of the database's words treated as informative. Ignored when --weights is off.
        #[arg(long, default_value_t = weight::DEFAULT_SHARE)]
        weight_share: u8,
        /// Print summed worker-stage timings and first-result latency to stderr.
        #[arg(long)]
        profile: bool,
    },
    /// Survey a reference database and report what curation would do to it.
    Curate {
        reference: PathBuf,
        /// Survey and report only; write no curated FASTA.
        #[arg(long)]
        plan: bool,
        /// Curated FASTA to write. Required unless --plan.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Provenance manifest. Defaults to <output>.curation.json.
        #[arg(long)]
        report: Option<PathBuf>,
        /// How representatives are chosen once a group exceeds the cap.
        #[arg(long, value_enum, default_value_t = curate::Selector::Random)]
        select: curate::Selector,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Representatives to keep per distinct stored lineage. Lower is faster
        /// and less sensitive.
        #[arg(long, default_value_t = 1000)]
        cap: usize,
        /// Soft length threshold: flags a reference as a saturation hazard and
        /// sorts it last among candidates, but never drops it.
        #[arg(long, default_value_t = 2000)]
        demote_length: usize,
        /// Hard length ceiling. Off by default, since on an untrimmed database it
        /// can delete whole genera.
        #[arg(long)]
        max_length: Option<usize>,
        #[arg(long, default_value_t = 0.10)]
        max_ambiguity: f64,
        #[arg(long, default_value_t = 32)]
        min_words: usize,
        /// Sketch distance past which candidates count as equally diverse.
        #[arg(long, default_value_t = curate::DIVERSITY_CEILING)]
        ceiling: f32,
        /// Caps for records with neither family nor genus, as
        /// covered,orphan-order,unplaced. Capped rather than dropped to guard against over-classification.
        #[arg(long, default_value = "50,50,50", value_delimiter = ',')]
        shallow_cap: Vec<usize>,
        /// Also cap covered shallow groups relative to their annotated siblings.
        #[arg(long, default_value_t = 0.10)]
        shallow_ratio: f64,
        /// Drop every under-annotated record. For validation only, not production use.
        #[arg(long)]
        drop_shallow: bool,
        /// Drop organellar and spike-in decoys. For validation only.
        #[arg(long)]
        drop_organellar: bool,
        /// Write every label longer than --demote-length here, as a trimming worklist.
        #[arg(long)]
        oversize_list: Option<PathBuf>,
    },
    /// Read metadata, optionally verifying the entire index.
    Inspect {
        db: PathBuf,
        #[arg(long)]
        verify: bool,
    },
}
fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().min(8))
}

struct Record {
    label: String,
    sequence: Vec<u8>,
}
struct BatchResult {
    output: String,
    reads: usize,
    bases: usize,
    escalated: usize,
    timings: Timings,
}

fn process(
    index: &Index,
    config: &Config,
    weights: &weight::Table,
    batch: Vec<Record>,
    workspace: &mut Workspace,
) -> Result<BatchResult> {
    let mut result = BatchResult {
        output: String::new(),
        reads: 0,
        bases: 0,
        escalated: 0,
        timings: Timings::default(),
    };
    for record in batch {
        let (prediction, times) =
            classify::classify(index, &record.sequence, config, weights, workspace)
                .with_context(|| format!("query {:?}", record.label))?;
        result
            .output
            .push_str(&prediction.tsv(&record.label, config.cutoff));
        result.reads += 1;
        result.bases += record.sequence.len();
        result.escalated += usize::from(times.escalated);
        result.timings.extraction += times.extraction;
        result.timings.sampling += times.sampling;
        result.timings.scoring += times.scoring;
        result.timings.taxonomy += times.taxonomy;
    }
    Ok(result)
}

fn run_classify(
    reads: &Path,
    db: &Path,
    output: &Path,
    threads: usize,
    config: Config,
    profile: bool,
) -> Result<()> {
    ensure!((1..=1024).contains(&threads), "threads must be in 1..=1024");
    config.validate()?;
    let wall = Instant::now();
    let index = Index::open(db)?;
    let weights = weight::Table::build(&index, config.weights, config.weight_share)?;
    let open_time = wall.elapsed();
    let mut reader = input::reader(reads)?;
    // File outputs are written atomically; a failure leaves any old output intact.
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = if output.as_os_str() == "-" {
        None
    } else {
        for source in [reads, db] {
            if output.exists() && source.exists() {
                ensure!(
                    std::fs::canonicalize(output)? != std::fs::canonicalize(source)?,
                    "output must differ from input and database"
                );
            }
        }
        Some(tempfile::NamedTempFile::new_in(parent)?)
    };
    let sink: Box<dyn Write + '_> = match &mut temp {
        Some(file) => Box::new(file.as_file_mut()),
        None => Box::new(io::stdout()),
    };
    let mut writer = BufWriter::with_capacity(256 * 1024, sink);
    let (mut total_reads, mut total_bases, mut escalated) = (0usize, 0usize, 0usize);
    let mut totals = Timings::default();
    let (mut parsing, mut writing) = (Duration::ZERO, Duration::ZERO);
    let mut first_result = None;
    std::thread::scope(|scope| -> Result<()> {
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        let mut handles = Vec::new();
        for worker in 0..threads {
            let (tx, rx) = mpsc::sync_channel::<Vec<Record>>(1);
            let (result_tx, result_rx) = mpsc::sync_channel::<Result<BatchResult>>(1);
            let index = &index;
            let config = &config;
            let weights = &weights;
            handles.push(
                std::thread::Builder::new()
                    .name(format!("sintaxer-{worker}"))
                    .spawn_scoped(scope, move || {
                        let mut workspace = Workspace::default();
                        while let Ok(batch) = rx.recv() {
                            if result_tx
                                .send(process(index, config, weights, batch, &mut workspace))
                                .is_err()
                            {
                                break;
                            }
                        }
                    })?,
            );
            senders.push(tx);
            receivers.push(result_rx);
        }
        let result = (|| -> Result<()> {
            let mut eof = false;
            while !eof {
                let mut sent = 0;
                for sender in &senders {
                    let start = Instant::now();
                    let mut batch = Vec::new();
                    let mut bytes = 0;
                    while batch.len() < 8 && bytes < 2 * 1024 * 1024 {
                        match reader.as_mut().and_then(|r| r.next()) {
                            None => {
                                eof = true;
                                break;
                            }
                            Some(record) => {
                                let record = record.context("reading query FASTA/FASTQ")?;
                                let label = input::label(record.id())?;
                                ensure!(
                                    record.qual().is_none_or(|q| q
                                        .iter()
                                        .all(|b| (b'!'..=b'~').contains(b))),
                                    "invalid FASTQ quality byte in {label:?}"
                                );
                                let sequence = record.seq().into_owned();
                                bytes += label.len() + sequence.len();
                                batch.push(Record { label, sequence });
                            }
                        }
                    }
                    parsing += start.elapsed();
                    if batch.is_empty() {
                        break;
                    }
                    sender
                        .send(batch)
                        .context("classification worker stopped")?;
                    sent += 1;
                    if eof {
                        break;
                    }
                }
                for receiver in receivers.iter().take(sent) {
                    let result = receiver.recv().context("classification worker stopped")??;
                    let start = Instant::now();
                    writer.write_all(result.output.as_bytes())?;
                    if first_result.is_none() {
                        writer.flush()?;
                        first_result = Some(wall.elapsed());
                    }
                    writing += start.elapsed();
                    total_reads += result.reads;
                    total_bases += result.bases;
                    escalated += result.escalated;
                    totals.extraction += result.timings.extraction;
                    totals.sampling += result.timings.sampling;
                    totals.scoring += result.timings.scoring;
                    totals.taxonomy += result.timings.taxonomy;
                }
            }
            Ok(())
        })();
        drop(senders);
        drop(receivers);
        let mut panicked = false;
        for handle in handles {
            panicked |= handle.join().is_err();
        }
        ensure!(!panicked, "classification worker panicked");
        result
    })?;
    writer.flush()?;
    drop(writer);
    if let Some(temp) = temp {
        temp.as_file().sync_all()?;
        temp.persist(output).map_err(|e| e.error)?;
    }
    let elapsed = wall.elapsed().as_secs_f64();
    eprintln!(
        "reads={total_reads} bases={total_bases} wall_s={elapsed:.6} reads_per_s={:.2} bases_per_s={:.2}",
        total_reads as f64 / elapsed,
        total_bases as f64 / elapsed
    );
    if profile {
        eprintln!(
            "index_open_s={:.6} first_result_s={:.6} parse_decompress_s={:.6} write_s={:.6} extraction_worker_s={:.6} sampling_worker_s={:.6} scoring_worker_s={:.6} taxonomy_worker_s={:.6}",
            open_time.as_secs_f64(),
            first_result.unwrap_or_default().as_secs_f64(),
            parsing.as_secs_f64(),
            writing.as_secs_f64(),
            totals.extraction.as_secs_f64(),
            totals.sampling.as_secs_f64(),
            totals.scoring.as_secs_f64(),
            totals.taxonomy.as_secs_f64()
        );
        // Report escalations: when candidate restriction fails, it silently escalates
        // every query and loses the speedup.
        eprintln!(
            "escalated_queries={escalated} escalated_pct={:.3}",
            if total_reads > 0 {
                100.0 * escalated as f64 / total_reads as f64
            } else {
                0.0
            }
        );
        // Report the weights that reached the kernel. A single-tier table only scales
        // scores by a constant and cannot change any call.
        let (plain, informative) = weights.tiers();
        eprintln!(
            "classifier={} weights={} weights_blake3={} words_informative={informative} words_plain={plain}",
            sintaxer::CLASSIFIER_VERSION,
            weights.label(),
            weights.digest()
        );
    }
    Ok(())
}

/// `annotated=d:..,k:..,...`: references carrying a name at each stored rank.
///
/// A near-zero genus count usually means the FASTA's rank separator is not `,`.
fn census_line(index: &Index) -> String {
    let census = index.rank_census();
    let ranks = std::str::from_utf8(taxonomy::RANKS).expect("ascii");
    let fields: Vec<String> = ranks
        .chars()
        .zip(census)
        .map(|(rank, count)| format!("{rank}:{count}"))
        .collect();
    format!("annotated={}", fields.join(","))
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Index { reference, output } => {
            if reference.exists() && output.exists() {
                ensure!(
                    std::fs::canonicalize(&reference)? != std::fs::canonicalize(&output)?,
                    "index output must differ from reference input"
                );
            }
            let start = Instant::now();
            index::build(&reference, &output)?;
            let elapsed = start.elapsed().as_secs_f64();
            let opened = Index::open(&output)?;
            eprintln!(
                "index={} build_s={elapsed:.6} references={} {}",
                output.display(),
                opened.references,
                census_line(&opened)
            );
        }
        Command::Curate {
            reference,
            plan,
            output,
            report,
            select,
            seed,
            cap,
            demote_length,
            max_length,
            max_ambiguity,
            min_words,
            ceiling,
            shallow_cap,
            shallow_ratio,
            drop_shallow,
            drop_organellar,
            oversize_list,
        } => {
            // Curation reads the input twice, so it cannot take a stream.
            ensure!(
                reference.as_os_str() != "-",
                "curate needs a seekable reference file, not stdin"
            );
            ensure!(
                plan != output.is_some(),
                "pass either --output <fasta> or --plan, not both"
            );
            if let (Some(output), true) = (output.as_deref(), reference.exists()) {
                if output.exists() {
                    ensure!(
                        std::fs::canonicalize(&reference)? != std::fs::canonicalize(output)?,
                        "curated output must differ from reference input"
                    );
                }
            }
            ensure!(cap > 0, "cap must be at least 1");
            ensure!(
                shallow_cap.len() == 3 && shallow_cap.iter().all(|&c| c > 0),
                "shallow-cap takes three positive values: covered,orphan-order,unplaced"
            );
            ensure!(
                (0.0..=1.0).contains(&shallow_ratio),
                "shallow-ratio must be in 0..=1"
            );
            ensure!(
                (0.0..=1.0).contains(&max_ambiguity),
                "max-ambiguity must be in 0..=1"
            );
            let policy = curate::Policy {
                cap,
                demote_length,
                max_length,
                max_ambiguity,
                min_words,
                ceiling,
                shallow: curate::ShallowCaps {
                    covered: shallow_cap[0],
                    orphan_order: shallow_cap[1],
                    unplaced: shallow_cap[2],
                    ratio: shallow_ratio,
                },
                drop_shallow,
                drop_organellar,
            };
            let mut oversize = oversize_list
                .as_deref()
                .map(|p| -> Result<_> {
                    Ok(BufWriter::new(
                        std::fs::File::create(p)
                            .with_context(|| format!("creating {}", p.display()))?,
                    ))
                })
                .transpose()?;
            let start = Instant::now();
            let survey = curate::survey(
                &reference,
                &policy,
                oversize.as_mut().map(|w| w as &mut dyn Write),
            )?;
            if let Some(mut list) = oversize {
                list.flush()?;
            }
            let summary = curate::plan(&survey, &policy);
            for (ordinal, error) in survey.parse_failures.iter().take(5) {
                eprintln!("warning: skipped record {ordinal}: {error}");
            }

            if plan {
                print!("{}", summary.render());
                io::stdout().flush()?;
                if let Some(hazard) = summary.hazard() {
                    eprintln!();
                    eprint!("{hazard}");
                }
                eprintln!("curate_s={:.6}", start.elapsed().as_secs_f64());
            } else {
                let output = output.expect("checked above");
                let selection = curate::select(&survey, &policy, select, seed);
                let invariants = curate::invariants(&survey, &selection);
                // A vanished lineage is a bug, not a policy outcome.
                ensure!(
                    invariants.groups_preserved && invariants.names_preserved,
                    "group preservation failed; lineages lost: {:?}",
                    invariants.missing
                );
                let emitted = curate::emit(&reference, &output, &selection)?;
                let wall = start.elapsed().as_secs_f64();
                let manifest = report.unwrap_or_else(|| {
                    PathBuf::from(format!("{}.curation.json", output.display()))
                });
                std::fs::write(
                    &manifest,
                    curate::report(&curate::Provenance {
                        plan: &summary,
                        policy: &policy,
                        selector: select,
                        seed,
                        selection: &selection,
                        emitted: Some(&emitted),
                        invariants: &invariants,
                        reference: &reference,
                        output: Some(&output),
                        wall,
                    }),
                )
                .with_context(|| format!("writing {}", manifest.display()))?;
                let after = curate::hazard_after(&survey, &selection, &policy);
                if let Some(hazard) = summary.hazard_with(Some(&after)) {
                    eprintln!();
                    eprint!("{hazard}");
                    eprintln!();
                }
                let c = &selection.counts;
                eprintln!(
                    "curated={} in={} out={} groups={} dropped_dup={} dropped_cap={} \
                     dropped_shallow_cap={} dropped_filter={} restored={} curate_s={wall:.6}",
                    output.display(),
                    summary.parsed,
                    emitted.records,
                    summary.groups,
                    c.duplicates,
                    c.over_cap,
                    c.over_shallow_cap + c.deliberate,
                    c.low_complexity + c.ambiguous + c.oversize,
                    c.restored,
                );
            }
        }
        Command::Inspect { db, verify } => {
            let index = Index::open(&db)?;
            if verify {
                index.verify()?;
            }
            println!(
                "format={}\nalgorithm={}\nk=8\nreferences={}\n{}\ntaxonomy_nodes={}\ndense_rows={}\nbytes={}\nsequence_bytes={}\nsource_blake3={}\nverified={verify}",
                index.format_version(),
                index.algorithm_version(),
                index.references,
                census_line(&index),
                index.nodes,
                index.dense_rows(),
                index.file_bytes(),
                index.sequence_bytes(),
                index.source_hash()
            );
        }
        Command::Classify {
            reads,
            db,
            output,
            threads,
            seed,
            cutoff,
            strand,
            exact,
            candidates,
            risk,
            weights,
            weight_share,
            profile,
        } => {
            run_classify(
                &reads,
                &db,
                &output,
                threads,
                Config {
                    seed,
                    cutoff,
                    strand,
                    exact,
                    candidates,
                    risk,
                    weights,
                    weight_share,
                },
                profile,
            )?;
        }
    }
    Ok(())
}
