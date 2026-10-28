#!/bin/bash
#SBATCH --job-name=ating_isoform_mageck
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/lab/barcheese01/ating/slurm_logs/%x_%j.out
#SBATCH --error=/lab/barcheese01/ating/slurm_logs/%x_%j.err

# Isoform-aware MAGeCK on one library's abundance counts (s5, s1 or k562), then its annotated vs
# truncated figure; submit from the repo root:
#   sbatch scripts/isoform_aware_shakeoff/isoform_mageck.sbatch s5|s1|k562
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate be-screen
python scripts/isoform_aware_shakeoff/run_isoform_mageck.py "$1"
python scripts/isoform_aware_shakeoff/plot_annot_vs_truncated.py "$1"
