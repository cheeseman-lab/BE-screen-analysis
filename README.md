# be-screen-analysis

Analysis of Tyler's adenine base editing (ABE) screen targeting start and stop codons.

## Data

Raw screen outputs live in `start_stop_ABE_screen_data/` (git-ignored):

| File | Level | Contents |
|---|---|---|
| `start_guide_summary.csv` | guide | Per-guide LFC, p-values, FDR, sensor editing efficiency (start codons) |
| `start_codon_summary.csv` | site | Annotated vs. non-canonical start site pairs, MAGeCK RRA scores |
| `stop_guide_summary.csv` | guide | Per-guide LFC, p-values, FDR, sensor editing efficiency (stop codons) |
| `stop_codon_summary.csv` | site | Stop site RRA scores with ClinVar annotations |

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
