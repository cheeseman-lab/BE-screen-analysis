"""DepMap common-essential label vs the percentage of cell lines dependent on a gene.

common_essential_by_pct_dependent.png
    For every gene in CRISPRGeneDependency.csv, binned by the percentage of cell lines dependent
    on it (P > 0.5; 1% bins, 100% on its own): the fraction of genes in the bin that DepMap tags
    common essential (CRISPRInferredCommonEssentials.csv, 24Q4). Dashed: the 90% (essential)
    cutoff.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.depmap import common_essentials, depmap_dependency
from be_screen_analysis.plotting import BLUE, INK, MUTED, style

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = ROOT / "data" / "reference"
OUT_DIR = ROOT / "figures" / "depmap"

CUTOFFS = {90: "≥ 90%: essential"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pct = depmap_dependency(REF_DIR / "CRISPRGeneDependency.csv")["depmap_pct_dependent"].dropna()
    ce = common_essentials(REF_DIR / "CRISPRInferredCommonEssentials.csv")
    tagged = pd.Series(pct.index.isin(ce), index=pct.index)
    bins = np.minimum(np.floor(pct), 99).astype(int)  # [k, k+1); 99 bin closed at 100
    bins[pct == 100] = 100  # every screened line dependent: own bin
    per = tagged.groupby(bins).agg(["mean", "size"]).reindex(range(101))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    ax.bar(per.index, per["mean"], width=0.9, color=BLUE, linewidth=0)
    for c in CUTOFFS:
        ax.axvline(c, color=MUTED, linestyle="--", linewidth=0.8)
    style(ax)
    for c, label in CUTOFFS.items():
        ax.text(c - 1, 0.5, label, rotation=90, fontsize=7, color=MUTED, va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))  # fmt: skip
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Fraction tagged common essential", fontsize=9)
    ax.set_title(
        f"DepMap common-essential tag by % of cell lines dependent ({len(pct):,} genes, "
        f"{tagged.sum():,} common essential)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax.set_xlabel("Cell lines dependent on the gene (%, 1% bins)", fontsize=9)
    ax.set_xlim(-1, 101)
    ax.set_xticks(range(0, 101, 5))  # labelled every 5%
    ax.set_xticks(range(101), minor=True)  # a tick under every bar
    ax.tick_params(axis="x", which="major", labelsize=7, length=4)
    ax.tick_params(axis="x", which="minor", length=2)
    fig.tight_layout()
    out = OUT_DIR / "common_essential_by_pct_dependent.png"
    fig.savefig(out)
    plt.close(fig)
    print(out)
    for lo, hi in [(0, 1), (1, 10), (10, 50), (50, 80), (80, 90), (90, 100)]:
        m = (pct >= lo) & (pct <= hi) if lo == 0 else (pct > lo) & (pct <= hi)
        print(f"pct ({lo}, {hi}]: {m.sum():,} genes, {tagged[m].mean():.1%} common essential")


if __name__ == "__main__":
    main()
