# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview
Analysis of Tyler's ABE base editing screen of start and stop codons. Code lives in
`src/be_screen_analysis/`; exploratory work goes in `notebooks/`.

`scripts/`, `figures/` and `results/` are organised by analysis axis, with matching folder names:
- `ABE_validation/correlation_k562/`: ABE annotated start and in-screen KO D12 vs K562
  untreated gene LFC, points coloured by DepMap % dependent lines, control histograms; the same
  two panels coloured by K562's own DepMap dependency probability
- `ABE_validation/correlation_edit_type/`: ATG->GTG vs ATG->ACG guide dropout at the same
  annotated starts (all guides; only guides with >= 20% editing; a grid of 20-80% editing cutoffs)
- `start_codon_analysis/`: start site trajectories; start site LFC vs distance to the next in-frame ATG;
  Tyler's annotated vs non-canonical start scatter with hits at FDR < 0.1 (his hit-category
  colours, and coloured by DepMap % dependent lines)
- `stop_codon_analysis/`: stop site LFC vs readthrough length (amino acids to the next in-frame stop)
- `hits/`: Tyler's start hits (FDR < 0.1) from efficient guides only (>= 20% editing at both D0
  and D12; site stats from `scripts/abe_mageck_analysis.py eff20`, summary rebuilt as
  `results/hits/start_codon_summary_eff20.csv`), filtered by DepMap % dependent lines
  (`scripts/hits/start_hit_lists.py` -> `results/hits/`): non-canonical-only hits in >= 90% genes,
  and every hit in < 30% (non-essential) genes; each highlighted on the annotated vs non-canonical
  scatter in `figures/hits/` (drawn by `scripts/start_codon_analysis/plot_start_depmap_scatter.py`,
  whose start_codon_analysis scatters use the same efficient-guide summary)
- `depmap/`: fraction of genes DepMap tags common essential per 1% bin of % dependent cell lines
- `old_analysis/`: superseded comparisons against the HeLa shake-off CRISPR KO screens (S1, S5;
  `data/shakeoff_data/`), kept for reference: `hits/` (BE x S1/S5 intersection),
  `ABE_validation/fitness_subset/` (gene LFC distributions with control-null cutoffs, 240-gene
  S1 / S5 / ABE start fitness subsets and their overlap), `ABE_validation/correlation_shakeoff/`
  (ABE start and in-screen KO vs S1/S5 at D4 and D6, control nulls, Tyler's start hit categories)

The current external comparison is the anti-mitotic K562 CRISPR KO screen (JCB 2024, Table S1;
`data/anti-mt_K562/`: `GW_counts.csv`, `GW_median_lfc.csv`, `Table1_legend_GW.csv`). Use only the
untreated arm endpoint (14 population doublings, ~ABE D12): `un.neg.tggtca` vs
`k562.initial.attggc`, gene level `un_init.median.noctrl.neg.lfc`.

Each script writes its figures to the matching `figures/` folder. `scripts/abe_mageck_analysis.py`
is shared upstream of all of them. `scripts/annotate_codon_summaries.py` writes copies of the
start/stop codon summaries with DepMap, protein-sequence and (start) initiation-efficiency columns
to `results/annot_summaries/` (references in `data/reference/`: GENCODE v49 genome + GTF, DepMap
CRISPRGeneDependency.csv, and CRISPRInferredCommonEssentials.csv (1,523 genes) from DepMap 24Q4
Public via Figshare (ndownloader.figshare.com/files/51064916), an older release than the
dependency file since newer ones are not on Figshare, and Model.csv (24Q4, ModelID -> cell line
name; K562 = ACH-000551, HeLa = ACH-001086 is not in the CRISPR dependency file); read-only external inputs: Ribo-TISH calls in
`/lab/barcheese01/mdiberna/ribotish/results/2026-10-09_reprocessed/`, input RNA HTSeq counts in
`/lab/barcheese01/aTIS_data/counts/`).
Never edit `data/start_stop_ABE_screen_data/` in place; add columns in new files.

`results/` (git-ignored, never commit or push it) holds only data: output tables (CSV) and the raw
outputs of tools such as MAGeCK (gene/sgRNA tables, counts, logs).

No summary or report files anywhere in the project (`results/`, `figures/`, `scripts/`,
`notebooks/` or elsewhere): no markdown/text write-ups describing results and no tool report
templates (MAGeCK's `.R`, `.report.Rmd`, `_summary.Rnw` are deleted after each run). Report
findings in the conversation instead.

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
