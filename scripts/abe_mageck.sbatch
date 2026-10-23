#!/bin/bash
#SBATCH --job-name=ating_abe_mageck
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/lab/barcheese01/ating/slurm_logs/%x_%j.out
#SBATCH --error=/lab/barcheese01/ating/slurm_logs/%x_%j.err

# Rerun MAGeCK on the raw ABE counts; submit from the repo root:
#   sbatch scripts/abe_mageck.sbatch [codon|ko|eff20|all]   (default all)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate be-screen
python scripts/abe_mageck_analysis.py "$@"
