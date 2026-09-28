# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview
Analysis of Tyler's ABE base editing screen of start and stop codons. Code lives in
`src/be_screen_analysis/`; exploratory work goes in `notebooks/`.

## Data
`start_stop_ABE_screen_data/` (git-ignored) holds four CSVs: per-guide (`*_guide_summary.csv`)
and per-site (`*_codon_summary.csv`) results for start and stop codons. Guide files carry LFC,
p-values, FDR and sensor editing efficiency; site files carry MAGeCK RRA scores. The start site
file pairs each annotated start (`annot_*`) with a non-canonical start (`nc_*`); the stop site
file includes ClinVar annotations.

## Development Setup
```bash
conda activate be-screen
uv pip install -e ".[dev]"
```

## Code Style
```bash
ruff format src/
ruff check src/
```

## Running Tests
```bash
pytest tests/
```
