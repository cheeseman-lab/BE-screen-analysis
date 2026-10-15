# be-screen-analysis

Analysis of Tyler's adenine base editing (ABE) screen targeting start and stop codons.

## Data

Raw screen outputs live in `data/start_stop_ABE_screen_data/` (git-ignored). Summaries are in
`summaries/`:

| File | Level | Contents |
|---|---|---|
| `start_guide_summary.csv` | guide | Per-guide LFC, p-values, FDR, sensor editing efficiency (start codons) |
| `start_codon_summary.csv` | site | Annotated vs. non-canonical start site pairs, MAGeCK RRA scores |
| `stop_guide_summary.csv` | guide | Per-guide LFC, p-values, FDR, sensor editing efficiency (stop codons) |
| `stop_codon_summary.csv` | site | Stop site RRA scores with ClinVar annotations |

Raw counts are in `ABE_guide_counts/` (load with `filtering.load_be_counts`). D0/D4/D6/D12/D21 in
triplicate; every file carries the same AAVS1 and non-targeting control guides:

| File | Contents |
|---|---|
| `ABE_{start,stop}_count_matrix_perturbation_level.csv` | codon guides (`sgRNA` = summary `guide_id`) |
| `ABE_{start,stop}_KO_count_matrix_perturbation_level.csv` | gene knockout guides (~4 per gene), same samples |

## Setup

```bash
conda activate be-screen
uv pip install -e ".[dev]"
```

## Development

```bash
ruff format src/ && ruff check src/
pytest
```
