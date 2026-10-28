#!/bin/bash
#SBATCH --job-name=ating_guide_classify
#SBATCH --partition=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/lab/barcheese01/ating/slurm_logs/%x_%j.out
#SBATCH --error=/lab/barcheese01/ating/slurm_logs/%x_%j.err

# Align the S1 / S5 / K562 spacers to hg38 and classify which start isoforms each cut knocks out;
# submit from the repo root:  sbatch guide_classify/classify.sbatch
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate be-screen
module load bowtie2/2.4.2

OUT=guide_classify/results
python guide_classify/run_classification.py fasta
# exact 20-nt end-to-end matches only (min score 0 = no mismatches or gaps), every hit, against
# the full hg38 (alt haplotypes and unplaced contigs included)
bowtie2 -p "$SLURM_CPUS_PER_TASK" -f -x /nfs/genomes/human_hg38_dec13/bowtie/hg38 \
    -U "$OUT/guides.fa" --end-to-end --score-min C,0,0 -N 0 -L 20 -a --no-unal --no-hd \
    -S "$OUT/guides.sam"
python guide_classify/run_classification.py classify
