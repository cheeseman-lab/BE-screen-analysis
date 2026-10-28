"""Alternative starts that are a hit in several screens, shown in each screen's LFC space.

A start is a hit in a screen when Tyler's hit category (neg or pos FDR < 0.1, annotated or
alternative start) is not "neither", among the starts with their own construct set
(isoform.distinct_starts, isoform.shared_hits). Results in results/isoform_aware_shakeoff/,
figures in figures/isoform_aware_shakeoff/:

shared_hits.csv  one row per start that is a hit in all of S1, S5 and K562: gene,
    alt_start_id, alt_kind, then per screen {lib}_annot_lfc, {lib}_alt_lfc, {lib}_delta_lfc,
    {lib}_alt_num_constructs, {lib}_category
shared_hits_{a}_{b}.csv  the same for each pair of screens (hit in both a and b)
shared_hits_s1_s5_k562.png  one panel per screen: annotated (x) vs alternative (y) start
    LFC of every start with its own construct set in gray (truncated and internal together), the
    starts that are a hit in all three coloured by their hit category in that screen, labelled.
shared_hits_pairwise.png  one column per pair of screens (S1-S5, S1-K562, S5-K562), the
    pair's first screen on top and second below; each panel highlights the starts that are a hit
    in both, labelled by the largest |delta_lfc| up to MAX_LABELS per panel.

Run run_isoform_mageck.py for s1, s5 and k562 first; from the repo root:
  python scripts/isoform_aware_shakeoff/plot_shared_hits.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.isoform import (
    COMPARISON_LABELS,
    HIT_FDR,
    START_KEY,
    distinct_starts,
    shared_hits,
)
from be_screen_analysis.plotting import GRID, HIT_CATEGORIES, MUTED, label_all, style

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "isoform_aware_shakeoff"
FIG_DIR = ROOT / "figures" / "isoform_aware_shakeoff"
LIBS = ["s1", "s5", "k562"]
MAX_LABELS = 30  # per panel, the largest |delta_lfc| first
LEGEND_ROOM = 0.3  # axis room above the highest point, as a fraction of the data range


def panel(ax: plt.Axes, d: pd.DataFrame, shared: pd.DataFrame, lib: str, title: str) -> None:
    """``lib``'s starts in gray with the ``shared`` starts coloured by hit category, labelled."""
    h = d.merge(shared[START_KEY], on=START_KEY)
    ax.scatter(d["annot_lfc"], d["alt_lfc"], s=5, color=GRID, linewidths=0, zorder=1,
               label=f"other starts (n = {len(d) - len(h):,})")  # fmt: skip
    for c, (name, color) in HIT_CATEGORIES.items():
        m = h["category"] == c
        if m.any():
            ax.scatter(h.loc[m, "annot_lfc"], h.loc[m, "alt_lfc"], s=18, color=color,
                       linewidths=0, zorder=3, label=f"{name} (n = {m.sum()})")  # fmt: skip
    lo, hi = np.nanmin(d[["annot_lfc", "alt_lfc"]]), np.nanmax(d[["annot_lfc", "alt_lfc"]])
    # headroom above the data (equal scales, so also to the right) keeps the legend clear of it
    lim = (lo - 0.05 * (hi - lo), hi + LEGEND_ROOM * (hi - lo))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(f"annotated start LFC ({COMPARISON_LABELS[lib]})")
    ax.set_ylabel("alternative start LFC")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    # the largest |delta_lfc| first; one label per shared point
    rank = h.assign(_d=h["delta_lfc"].abs()).sort_values("_d", ascending=False)
    rank = rank.drop_duplicates(["gene", "annot_lfc", "alt_lfc"]).head(MAX_LABELS)
    label_all(ax, rank, "annot_lfc", "alt_lfc", "gene", fontsize=6)


def save(fig: plt.Figure, name: str) -> None:  # noqa: D103
    out = FIG_DIR / name
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


def main() -> None:  # noqa: D103
    starts = {
        lib: distinct_starts(pd.read_csv(RESULTS / lib / "gene_isoform_lfc.csv")) for lib in LIBS
    }
    hit = f"(neg or pos FDR < {HIT_FDR})"

    table = shared_hits(starts, LIBS)
    table.to_csv(RESULTS / "shared_hits.csv", index=False)
    print(f"{len(table)} starts are a hit in all of {', '.join(LIBS)} -> shared_hits.csv")
    fig, axes = plt.subplots(1, len(LIBS), figsize=(5.2 * len(LIBS), 5.4))
    # fixed layout: label_all places names on the final axes geometry
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.11, top=0.86, wspace=0.22)
    fig.suptitle(f"{len(table)} alternative starts that are a hit in S1, S5 and K562 {hit}",
                 x=0.01, ha="left", fontsize=10)  # fmt: skip
    for ax, lib in zip(axes, LIBS):
        panel(ax, starts[lib], table, lib, lib.upper())
    save(fig, "shared_hits_s1_s5_k562.png")

    pairs = list(combinations(LIBS, 2))
    fig, axes = plt.subplots(2, len(pairs), figsize=(5.2 * len(pairs), 10.4))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.06, top=0.92, wspace=0.22, hspace=0.25)
    fig.suptitle(f"Alternative starts that are a hit in both screens of each pair {hit}",
                 x=0.01, ha="left", fontsize=11)  # fmt: skip
    for col, (a, b) in zip(axes.T, pairs):
        shared = shared_hits(starts, [a, b])
        shared.to_csv(RESULTS / f"shared_hits_{a}_{b}.csv", index=False)
        print(f"{a} + {b}: {len(shared)} shared hits -> shared_hits_{a}_{b}.csv")
        pair = f"{a.upper()} + {b.upper()}: {len(shared)} shared hits"
        for ax, lib in zip(col, (a, b)):
            panel(ax, starts[lib], shared, lib, f"{pair}, in {lib.upper()}")
    save(fig, "shared_hits_pairwise.png")


if __name__ == "__main__":
    main()
