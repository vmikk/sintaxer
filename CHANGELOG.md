# sintaxer changelog

## [0.2.0]

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
