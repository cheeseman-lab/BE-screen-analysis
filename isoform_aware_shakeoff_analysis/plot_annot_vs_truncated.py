"""Annotated vs truncated / internal start LFC from the isoform-aware abundance MAGeCK.

figures/annot_vs_truncated_lfc_{library}.png  (library s5, s1 or k562)
    One point per (gene, non-annotated start) whose knocking-out construct set differs from the
    annotated start's (the others have the annotated start's LFC by construction and are left
    off); x = annot_lfc, y = alt_lfc (S5 78h_remaining vs 0h, S1 day5_remaining vs Day0, K562
    un.neg.tggtca vs k562.initial.attggc).
    Panels Truncated and Internal, each on equal axes with a least-squares fit; points coloured
    by alt_num_constructs, the guides (S5: constructs) knocking out the alternative start, as a
    confidence readout (viridis, 1 to the library maximum, shared by both panels); genes
    labelled (beside the point, else with a leader line, never overlapping) for hits in one start
    only, "both" hits with |delta_lfc| > 0.25 (Tyler's hit categories, neg or pos FDR < 0.1) and
    the largest |delta_lfc|; at most 30 per panel, keeping the largest |delta_lfc|; the panel's
    extreme points (lowest / highest annot_lfc and alt_lfc) are always labelled.
figures/annot_vs_truncated_lfc_{library}_hits.png  (colour mode "hits")
    The same panels with points coloured by Tyler's hit categories (start_codon_analysis
    colours), n per category in the legend.
figures/annot_vs_truncated_lfc_{library}_depmap.png  (colour mode "depmap")
    The same panels with points coloured by the gene's % of DepMap lines dependent
    (depmap.depmap_dependency on data/reference/CRISPRGeneDependency.csv; viridis 0-100, most
    dependent on top, genes without DepMap data in light gray).

Run run_isoform_mageck.py {library} first:
  python plot_annot_vs_truncated.py s5|s1|k562 [guides|hits|depmap]   (default guides)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.depmap import depmap_dependency
from be_screen_analysis.filtering import start_hit_category
from be_screen_analysis.plotting import label_all, scatter_fit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "isoform_aware_shakeoff_analysis" / "results"
FIG_DIR = ROOT / "isoform_aware_shakeoff_analysis" / "figures"
COMPARISONS = {
    "s5": "78h remaining vs 0h",
    "s1": "day 5 remaining vs day 0",
    "k562": "untreated endpoint vs initial",
}
MODES = ("guides", "hits", "depmap")
DEPMAP_FILE = ROOT / "data" / "reference" / "CRISPRGeneDependency.csv"
DEPMAP = "depmap_pct_dependent"
FDR_THRESH = 0.1
N_DELTA_LABELS = 8  # per panel, besides the hits
MAX_LABELS = 30  # per panel: past this, only the largest |delta_lfc| are labelled
BOTH_DELTA = 0.25  # label a "both" hit only if its starts differ by more than this
KINDS = ["Truncated", "Internal"]
CMAP = "viridis"
GUIDES = "alt_num_constructs"  # guides (S5: constructs) knocking out the alternative start
# start_codon_analysis hit-category colours (notebooks/start_scatter.ipynb)
CATS = {
    "neither": ("Neither", "#CCCCCC"),
    "annot_only": ("Annotated only", "#4477AA"),
    "nc_only": ("Alt start only", "#CCBB44"),
    "both": ("Both", "#EE6677"),
}


def panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    kind: str,
    comparison: str,
    mode: str,
    vrange: tuple[float, float],
) -> plt.cm.ScalarMappable:
    """One kind's scatter (coloured by ``mode``), fit and gene labels; returns the last points."""
    cat = start_hit_category(df, None, FDR_THRESH, other="alt")
    if mode == "hits":
        names = cat.map(lambda c: CATS[c][0])
        colors = {n: c for n, c in CATS.values() if (names == n).any()}
        scatter_fit(ax, df["annot_lfc"], df["alt_lfc"], groups=names, alpha=0.8,
                    group_colors=colors, stats_right=True, group_r=False)  # fmt: skip
    elif mode == "depmap":
        scatter_fit(ax, df["annot_lfc"], df["alt_lfc"], color_by=df[DEPMAP], cmap=CMAP,
                    vrange=(0, 100), stats_right=True,
                    missing_label="no DepMap data")  # fmt: skip
    else:
        scatter_fit(ax, df["annot_lfc"], df["alt_lfc"], color_by=df[GUIDES], cmap=CMAP,
                    vrange=vrange, stats_right=True)  # fmt: skip
    points = ax.collections[-1]
    # every point on the axes (scatter_fit clips to the central 99%)
    lo, hi = np.nanmin(df[["annot_lfc", "alt_lfc"]]), np.nanmax(df[["annot_lfc", "alt_lfc"]])
    lim = (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    big = df["delta_lfc"].abs().nlargest(N_DELTA_LABELS).index
    # hits in one start only, "both" hits that differ by > BOTH_DELTA, and the largest |delta|
    one_side = cat.isin(["annot_only", "nc_only"])
    both = (cat == "both") & (df["delta_lfc"].abs() > BOTH_DELTA)
    # the panel's extreme points (lowest / highest on either axis) are always labelled
    ends = [df[c].idxmin() for c in ("annot_lfc", "alt_lfc")]
    ends += [df[c].idxmax() for c in ("annot_lfc", "alt_lfc")]
    shown = df.index[one_side | both | df.index.isin(big) | df.index.isin(ends)]
    # one label per gene where its starts share a point (same construct set)
    shown = df.loc[shown].drop_duplicates(["gene", "annot_lfc", "alt_lfc"])
    capped = shown["delta_lfc"].abs().nlargest(MAX_LABELS).index
    shown = capped.union(shown.index.intersection(ends))
    ax.set_xlabel(f"annotated start LFC ({comparison})")
    ax.set_ylabel(f"{kind.lower()} start LFC")
    ax.set_title(
        f"{kind}: {len(df):,} starts with their own construct set",
        loc="left",
        fontsize=9,
    )
    # guide mode: the fit only (every start has a guide count, so no "no value" entry)
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, n) for h, n in zip(handles, labels) if mode != "guides" or n.startswith("fit")]
    keep = [(h, n) for h, n in keep if "(n = 0)" not in n]  # e.g. no gene lacks DepMap data
    ax.legend(*zip(*keep), frameon=False, fontsize=7, loc="upper left")
    label_all(ax, df.loc[shown], "annot_lfc", "alt_lfc", "gene", fontsize=6)
    return points


def main(lib: str, mode: str = "guides") -> None:  # noqa: D103
    if lib not in COMPARISONS or mode not in MODES:
        raise SystemExit("usage: plot_annot_vs_truncated.py s5|s1|k562 [guides|hits|depmap]")
    t = pd.read_csv(RESULTS / lib / "gene_isoform_lfc.csv")
    t = t.dropna(subset=["annot_lfc", "alt_lfc"])
    t = t[~t["same_as_annotated"].astype(bool)]
    if mode == "depmap":
        t = t.join(depmap_dependency(DEPMAP_FILE)[DEPMAP], on="gene")
    fig, axes = plt.subplots(1, len(KINDS), figsize=(6.5 * len(KINDS), 6.4))
    # fixed layout: label_all places names on the final axes geometry
    right = 0.98 if mode == "hits" else 0.9  # room for the colour bar
    fig.subplots_adjust(left=0.06, right=right, bottom=0.09, top=0.9, wspace=0.18)
    title = f"{lib.upper()} isoform-aware abundance"
    if mode == "hits":
        title += f" (hits: neg or pos FDR < {FDR_THRESH})"
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    vrange = (1, max(2, t[GUIDES].max()))
    for ax, kind in zip(np.atleast_1d(axes), KINDS):
        points = panel(ax, t[t["alt_kind"] == kind], kind, COMPARISONS[lib], mode, vrange)
    if mode != "hits":
        cax = fig.add_axes([0.925, 0.15, 0.012, 0.7])  # colour bar, outside the panels
        bar = fig.colorbar(points, cax=cax)
        if mode == "depmap":
            bar.set_label("% essential across DepMap lines (P(dependency) > 0.5)", fontsize=8)
        else:
            unit = "constructs" if lib == "s5" else "guides"
            bar.set_label(f"{unit} knocking out the alternative start", fontsize=8)
            bar.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        bar.ax.tick_params(labelsize=7)
    suffix = "" if mode == "guides" else f"_{mode}"
    out = FIG_DIR / f"annot_vs_truncated_lfc_{lib}{suffix}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3]) if len(sys.argv) > 1 else main("")
