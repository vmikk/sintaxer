use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use sintaxer::{
    classify::{self, Config, Strand, Timings},
    index::{self, Index},
    input,
    scoring::{Engine, Workspace},
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
        /// Scoring engine; the optimized engines are experimental.
        #[arg(long, value_enum, default_value_t = Engine::Scalar)]
        engine: Engine,
        #[arg(long, default_value_t = 1024)]
        tile_size: usize,
        #[arg(long, default_value_t = 16)]
        bootstrap_batch: usize,
        /// Print summed worker-stage timings and first-result latency to stderr.
        #[arg(long)]
        profile: bool,
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
    timings: Timings,
}

fn process(
    index: &Index,
    config: &Config,
    batch: Vec<Record>,
    workspace: &mut Workspace,
) -> Result<BatchResult> {
    let mut result = BatchResult {
        output: String::new(),
        reads: 0,
        bases: 0,
        timings: Timings::default(),
    };
    for record in batch {
        let (prediction, times) = classify::classify(index, &record.sequence, config, workspace)
            .with_context(|| format!("query {:?}", record.label))?;
        result
            .output
            .push_str(&prediction.tsv(&record.label, config.cutoff));
        result.reads += 1;
        result.bases += record.sequence.len();
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
    let (mut total_reads, mut total_bases) = (0, 0);
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
            handles.push(
                std::thread::Builder::new()
                    .name(format!("sintaxer-{worker}"))
                    .spawn_scoped(scope, move || {
                        let mut workspace = Workspace::default();
                        while let Ok(batch) = rx.recv() {
                            if result_tx
                                .send(process(index, config, batch, &mut workspace))
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
    }
    Ok(())
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
            eprintln!(
                "index={} build_s={:.6}",
                output.display(),
                start.elapsed().as_secs_f64()
            );
        }
        Command::Inspect { db, verify } => {
            let index = Index::open(&db)?;
            if verify {
                index.verify()?;
            }
            println!(
                "format={}\nalgorithm={}\nk=8\nreferences={}\ntaxonomy_nodes={}\ndense_rows={}\nbytes={}\nsequence_bytes={}\nsource_blake3={}\nverified={verify}",
                index.format_version(),
                sintaxer::ALGORITHM_VERSION,
                index.references,
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
            engine,
            tile_size,
            bootstrap_batch,
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
                    engine,
                    tile_size,
                    bootstrap_batch,
                },
                profile,
            )?;
        }
    }
    Ok(())
}
