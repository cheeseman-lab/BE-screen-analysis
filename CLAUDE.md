# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview
Analysis of Tyler's ABE base editing screen of start and stop codons. Code lives in
`src/be_screen_analysis/`; exploratory work goes in `notebooks/`.

## Data
`data/start_stop_ABE_screen_data/` (git-ignored). `summaries/` holds four CSVs: per-guide (`*_guide_summary.csv`)
and per-site (`*_codon_summary.csv`) results for start and stop codons. Guide files carry LFC,
p-values, FDR and sensor editing efficiency; site files carry MAGeCK RRA scores. The start site
file pairs each annotated start (`annot_*`) with a non-canonical start (`nc_*`); the stop site
file includes ClinVar annotations. `ABE_guide_counts/` holds raw counts (D0/D4/D6/D12/D21 x 3
reps) for the codon guides and for separate gene knockout (`*_KO_*`) guides from the same samples,
with shared AAVS1 and non-targeting controls; load them with `filtering.load_be_counts`.

## Development Setup
```bash
conda activate be-screen
uv pip install -e ".[dev]"
```

## Code Style
Reuse before reimplementing: check `src/be_screen_analysis/` (loaders, LFC/centring, nulls,
plotting helpers) for an existing function before writing a new one, and extend it with a
parameter rather than copying it. Shared logic belongs in `src/`; scripts in `scripts/` only wire
library functions together and write outputs.

```bash
ruff format src/
ruff check src/
```

## Running Tests
```bash
pytest tests/
```
