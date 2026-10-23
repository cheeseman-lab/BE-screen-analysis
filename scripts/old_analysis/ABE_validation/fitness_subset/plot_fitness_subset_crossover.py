"""Where each screen's fitness subset lands on the other screens' gene LFC distributions.

fitness_subset_crossover.png  2x2, each screen's gene log2FC distribution drawn as in
                              lfc_distributions_cutoffs.png (number of genes, same bins; filled
                              and dashed: that screen's own 240-gene fitness subset), with the
                              genes of another screen's fitness subset marked as ticks under it:
    top row:    ABE annotated start D4, with the S5 (genome-wide) and the S1 (essential)
                fitness subsets
    bottom row: S5 78h and S1 D4, each with the ABE start fitness subset
  Values beyond the axis are pooled at the edges.

Gene LFCs as in the fitness subsets: S1 published MAGeCK D4 gene LFC, S5 intergenic-centred 78h,
ABE median annotated-start guide MAGeCK D4 LFC (>=20% editing). Run fitness_subset_analysis.py
and scripts/abe_mageck_analysis.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.correlations import abe_gene_lfc, s5_gene_dropout
from be_screen_analysis.filtering import s1_published_gene_lfc
from be_screen_analysis.plotting import INK, MUTED, style

ROOT = Path(__file__).resolve().parents[4]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
S1_MAGECK = ROOT / "data" / "shakeoff_data" / "table_s1" / "mageck.neg_HeLa.csv"
S5_FILE = ROOT / "data" / "shakeoff_data" / "table_s5" / "sgRNA_sequences_and_counts.csv"
OUT_DIR = ROOT / "figures" / "old_analysis" / "ABE_validation" / "fitness_subset"

LABELS = {
    "ABE": "ABE annotated start D4",
    "S5": "S5 genome-wide 78h",
    "S1": "S1 essential D4",
}
SUBSET_FILES = {
    "S1": "s1_d4_fitness_subset.csv",
    "S5": "s5_fitness_subset.csv",
    "ABE": "abe_start_fitness_subset.csv",
}
TIME_COLORS = {"S1": "#86b6ef", "S5": "#2a78d6", "ABE": "#86b6ef"}  # as lfc_distributions (D4, 78h)
PANELS = [("ABE", "S5"), ("ABE", "S1"), ("S5", "ABE"), ("S1", "ABE")]  # (distribution, subset)
LO, HI = -3.0, 1.5


def gene_lfc() -> dict[str, pd.Series]:
    """Gene LFC per screen at the timepoint its fitness subset was defined."""
    return {
        "S1": s1_published_gene_lfc(pd.read_csv(S1_MAGECK))["D4"].dropna(),
        "S5": s5_gene_dropout(pd.read_csv(S5_FILE))["S5 78h"].dropna(),
        "ABE": abe_gene_lfc(MAGECK_DIR, "start")["ABE D4"].dropna(),
    }


def subsets() -> dict[str, pd.Index]:
    return {
        s: pd.read_csv(OUT_DIR / f, comment="#", index_col=0).index for s, f in SUBSET_FILES.items()
    }


def crossover(lfc: dict[str, pd.Series], sets: dict[str, pd.Index]) -> Path:
    """2x2: each distribution as in lfc_distributions_cutoffs, other subset's genes as a rug."""
    bins = np.linspace(LO, HI, 91)  # same grid as lfc_distributions_cutoffs.png
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
    for ax, (dist, sub) in zip(axes.flat, PANELS):
        vals = lfc[dist]
        color = TIME_COLORS[dist]
        own_cut = vals[sets[dist]].max()  # where this screen's own fitness subset ends
        present = vals.reindex(sets[sub]).dropna()
        below = (present <= own_cut).sum()
        clipped = vals.clip(LO, HI)
        ax.hist(clipped[vals <= own_cut], bins=bins, color=color, alpha=0.45, linewidth=0)
        ax.hist(clipped, bins=bins, histtype="step", linewidth=1.6, color=color,
                label=f"all {dist} genes (n = {len(vals):,})")  # fmt: skip
        ax.axvline(own_cut, color=color, linestyle="--", linewidth=1.2,
                   label=f"{dist} fitness subset: LFC \u2264 {own_cut:+.2f} (filled)")  # fmt: skip
        # rug under the histogram: one tick per gene of the other screen's subset
        top = 1.5 * ax.get_ylim()[1]  # headroom so the legend sits above the histogram
        ax.vlines(present.clip(LO, HI), -0.09 * top, -0.02 * top, color=INK, linewidth=0.7,
                  label=f"{sub} fitness subset genes: {len(present)} of {len(sets[sub])} in "
                  f"{dist}, {below} in its fitness subset")  # fmt: skip
        ax.set_ylim(-0.1 * top, top)
        ax.axhline(0, color=MUTED, linewidth=0.6)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(LO, HI)
        ax.set_title(f"{sub} fitness subset on {LABELS[dist]}", loc="left", fontsize=10)
        ax.set_xlabel(f"{LABELS[dist]} gene log2FC", fontsize=8)
        ax.set_ylabel("Number of genes", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left", facecolor="white", edgecolor="none",
                  framealpha=0.9)  # fmt: skip
        style(ax)
        print(f"{sub} subset on {dist}: {len(present)} present, {below} at/below {own_cut:+.3f}")
    fig.suptitle(
        "Fitness subsets across screens: each screen's gene distribution as in "
        "lfc_distributions_cutoffs (values beyond the axis pooled at the edges)\n"
        "ticks below the axis: genes of another screen's fitness subset",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout()
    out = OUT_DIR / "fitness_subset_crossover.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    print(crossover(gene_lfc(), subsets()))


if __name__ == "__main__":
    main()
