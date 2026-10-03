
## [0.5.0]

### Added
- New `curate` subcommand that subsamples a redundant reference FASTA to at most `--cap` representatives (default 1000) per distinct lineage while keeping every rank name.
- `curate --plan` surveys the database and reports saturation hazards and a cap curve without writing anything. Real runs write a JSON provenance manifest next to the output.
- Curation options include `--select random|maxmin|first`, length demotion (`--demote-length`, `--max-length`), shallow-lineage caps and `--oversize-list`.

## [0.4.0]

### Changed
- Default `--candidates` raised from 1024 to 2048.
- Unrecognised sequence bytes are now treated as `N` rather than rejected, and `U` is read as `T`. `index` warns with a count.

### Fixed
- Trailing or doubled commas in `tax=` annotations, and empty rank names such as `c:`, no longer abort index builds.
- Queries whose vocabulary covers almost the whole 8-mer space no longer overflow the rank counters.

## [0.3.0]

### Added
- `--exact` runs the published every-reference, every-replicate algorithm, for checking the fast path.
- `--risk`, an opt-in conservative mode that escalates to a wider candidate set. `--profile` reports how often escalation happened.

### Changed
- Classification ranks every reference against the whole query vocabulary once per strand, then runs the bootstrap replicates only on the top `--candidates` (default 1024). Small databases skip the ranking.

### Removed
- The experimental `--engine`, `--tile-size` and `--bootstrap-batch` options.

### Performance
- Much faster on large databases, and the advantage grows with database size.


### Changed
- Index format bumped to 2, which stores packed reference sequences alongside the postings. Rebuild existing indexes, since older ones are rejected with a clear error.
- `inspect` now prints the format version and the size of the sequence section.


## [0.1.0]

Initial release.

- `index`, `classify` and `inspect` subcommands for SINTAX-style 16S/ITS classification down to genus.
- Reusable, memory-mapped `.stx` index built from `tax=` annotated FASTA. `inspect --verify` checks it with BLAKE3.
- Accepts FASTA/FASTQ queries, plain, gzip or zstd, including stdin. Output is four-column TSV with bootstrap support and a `--cutoff` filter.
- Seeded, deterministic bootstrap sampling that gives identical output regardless of thread count or batching.
- Experimental scoring engines selectable with `--engine` (scalar default, plus batched, bitslice, avx2, projected and auto).
