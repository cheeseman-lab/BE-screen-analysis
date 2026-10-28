"""Tyler's annotated vs non-canonical start scatter (notebooks/start_scatter.ipynb), DepMap colour.

Every figure uses site statistics from efficient guides only (>= 20% sensor editing at both D0 and
D12): results/hits/start_codon_summary_eff20.csv (scripts/hits/start_hit_lists.py). Rows whose
annotated or non-canonical start has no efficient guide are left out.

start_annot_vs_nc_fdr.png     the notebook's figure exactly (hit categories in its colours, sizes
                              and alpha, its labels and counts box), with its FDR switch on and
                              LFC cutoff off: a start is a hit if neg or pos FDR < 0.1.

start_annot_vs_nc_depmap.png  the notebook's figure (one point per start codon summary row,
                              x = annot_lfc, y = nc_lfc, D12; one panel per non-canonical type,
                              shared limits, y = x, gene labels for |LFC| > 1
                              with overlaps dropped, hit category counts with FDR < 0.1, no
                              LFC cutoff) with the points coloured
                              by the percentage of DepMap cell lines dependent on the gene
                              (depmap_pct_dependent) instead of by hit category. Higher
                              percentages drawn on top; genes without DepMap data in gray.

figures/hits/start_annot_vs_nc_depmap90_nc_only.png
                              the same panels with only the start sites listed in
                              results/hits/nc_only_depmap90_starts.csv (non-canonical-only hits,
                              FDR < 0.1, in genes >= 90% of DepMap lines depend on) coloured and
                              labelled (every one: next to its point, or in a column with a
                              non-crossing leader line where points cluster); all other points gray.

figures/hits/start_annot_vs_nc_nonessential_depmap30_hits.png
                              the same for results/hits/nonessential_depmap30_hits.csv (every hit,
                              annotated only / non-canonical only / both, in genes < 30% of DepMap
                              lines depend on).

Run scripts/hits/start_hit_lists.py first.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from be_screen_analysis.filtering import start_hit_category
from be_screen_analysis.plotting import label_all

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "hits" / "start_codon_summary_eff20.csv"
HITS_DIR = ROOT / "results" / "hits"
OUT_DIR = ROOT / "figures" / "start_codon_analysis"
HITS_FIG_DIR = ROOT / "figures" / "hits"  # the highlighted figure, beside results/hits/

LFC_THRESH = None  # hits by FDR alone (the notebook with its LFC cutoff off) ...
FDR_THRESH = 0.1  # ... and neg or pos FDR < 0.1 (the notebook's FDR switch)
SIG_LABEL = f"FDR<{FDR_THRESH}"
EFF_LABEL = "guides \u226520% editing at D0 and D12"
LABEL_LFC = 1.0  # the notebook labels |LFC| > this when it has no LFC cutoff
DEPMAP = "depmap_pct_dependent"
CMAP = "viridis"
NO_DEPMAP = "#CCCCCC"
COUNT_LABELS = {"Annot only": "annot_only", "NC only": "nc_only", "Both": "both"}
LINE_COLORS = {"annot": "#4477AA", "nc": "#CCBB44"}  # notebook threshold-line colours
SITE_KEY = ["annot_start_site_target_id", "nc_start_site_target_id"]  # one summary row


def _guides(ax: plt.Axes, lim: list[float], title: str) -> None:
    """The notebook's reference lines, limits and axis labels."""
    ax.plot(lim, lim, color="black", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="gray", lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color="gray", lw=0.5, ls=":", zorder=0)
    for v in () if LFC_THRESH is None else (-LFC_THRESH, LFC_THRESH):
        ax.axhline(v, color=LINE_COLORS["nc"], lw=0.6, ls="--", zorder=0, alpha=0.6)
        ax.axvline(v, color=LINE_COLORS["annot"], lw=0.6, ls="--", zorder=0, alpha=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Annotated start LFC", fontsize=10)
    ax.set_ylabel("Non-canonical start LFC", fontsize=10)
    ax.set_title(title, fontsize=11)


def _by_extremity(df: pd.DataFrame) -> pd.DataFrame:
    """Rows ordered as the notebook labels them: largest |annot_lfc| + |nc_lfc| first."""
    return df.assign(_p=df["annot_lfc"].abs() + df["nc_lfc"].abs()).sort_values(
        "_p", ascending=False
    )


def _label_notebook(ax: plt.Axes, df: pd.DataFrame) -> None:
    """The notebook's labels: |LFC| > LABEL_LFC on either axis, overlapping labels dropped."""
    rows = _by_extremity(df[(df["annot_lfc"].abs() > LABEL_LFC) | (df["nc_lfc"].abs() > LABEL_LFC)])
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    placed = []
    for _, row in rows.iterrows():
        t = ax.text(row["annot_lfc"], row["nc_lfc"], row["Hugo_Symbol"], fontsize=4.5,
                    ha="center", va="bottom", color="black")  # fmt: skip
        bb = t.get_window_extent(renderer=renderer)
        if any(bb.overlaps(b) for b in placed):
            t.remove()
        else:
            placed.append(bb)


def plot_scatter(
    df: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    lim: list[float],
    highlight: pd.Series | None = None,
    highlight_label: str = "",
):
    """One notebook panel coloured by DepMap.

    ``highlight`` (bool per row): only those rows keep their colour and get labels (all of them);
    every other point is gray.
    """
    if highlight is not None:
        rest, hi = df[~highlight], df[highlight].sort_values(DEPMAP)
        ax.scatter(rest["annot_lfc"], rest["nc_lfc"], c=NO_DEPMAP, s=10, alpha=0.5,
                   linewidths=0, rasterized=True, zorder=1,
                   label=f"other start sites (n = {len(rest)})")  # fmt: skip
        pts = ax.scatter(hi["annot_lfc"], hi["nc_lfc"], c=hi[DEPMAP], cmap=CMAP, vmin=0,
                         vmax=100, s=12, alpha=0.85, linewidths=0, zorder=3,
                         label=f"{highlight_label} (n = {len(hi)})")  # fmt: skip
        _guides(ax, lim, title)
        label_all(ax, hi, "annot_lfc", "nc_lfc", "Hugo_Symbol")
        return pts

    missing = df[DEPMAP].isna()
    ax.scatter(df.loc[missing, "annot_lfc"], df.loc[missing, "nc_lfc"], c=NO_DEPMAP, s=12,
               alpha=0.6, linewidths=0, rasterized=True, zorder=1,
               label=f"no DepMap data (n = {missing.sum()})")  # fmt: skip
    d = df[~missing].sort_values(DEPMAP)  # most-dependent genes on top
    pts = ax.scatter(d["annot_lfc"], d["nc_lfc"], c=d[DEPMAP], cmap=CMAP, vmin=0, vmax=100,
                     s=12, alpha=0.85, linewidths=0, rasterized=True, zorder=2)  # fmt: skip
    _guides(ax, lim, title)
    _label_notebook(ax, df)
    cats = start_hit_category(df, LFC_THRESH, FDR_THRESH).value_counts()
    counts = "\n".join(f"{k}: {cats.get(c, 0)}" for k, c in COUNT_LABELS.items())
    ax.text(0.97, 0.03, f"n={len(df)}\n[{SIG_LABEL}]\n{counts}", transform=ax.transAxes,
            fontsize=7, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))  # fmt: skip
    return pts


# the notebook's category styling (start_scatter.ipynb)
SIG_COLORS = {
    "neither": "#CCCCCC",
    "annot_only": "#4477AA",
    "nc_only": "#CCBB44",
    "both": "#EE6677",
}
SIG_SIZES = {"neither": 8, "annot_only": 20, "nc_only": 20, "both": 25}
SIG_ALPHA = {"neither": 0.3, "annot_only": 0.8, "nc_only": 0.8, "both": 0.9}
SIG_ZORDER = {"neither": 1, "annot_only": 3, "nc_only": 3, "both": 4}
SIG_NAMES = {"neither": "Neither", "annot_only": "Annotated only", "nc_only": "NC only",
             "both": "Both"}  # fmt: skip


def plot_hit_categories(df: pd.DataFrame, ax: plt.Axes, title: str, lim: list[float]) -> None:
    """One panel of the notebook's figure: points coloured by hit category (FDR rule)."""
    cat = start_hit_category(df, LFC_THRESH, FDR_THRESH)
    for c in SIG_COLORS:
        m = cat == c
        ax.scatter(df.loc[m, "annot_lfc"], df.loc[m, "nc_lfc"], c=SIG_COLORS[c],
                   s=SIG_SIZES[c], alpha=SIG_ALPHA[c], zorder=SIG_ZORDER[c], linewidths=0,
                   rasterized=True)  # fmt: skip
    _guides(ax, lim, title)
    _label_notebook(ax, df)
    counts = cat.value_counts()
    box = "\n".join(f"{k}: {counts.get(c, 0)}" for k, c in COUNT_LABELS.items())
    ax.text(0.97, 0.03, f"n={len(df)}\n[{SIG_LABEL}]\n{box}", transform=ax.transAxes,
            fontsize=7, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))  # fmt: skip
    handles = [
        mpatches.Patch(
            color=SIG_COLORS[c],
            alpha=0.6 if c == "neither" else SIG_ALPHA[c],
            label=name if c == "neither" else f"{name} ({SIG_LABEL})",
        )  # fmt: skip
        for c, name in SIG_NAMES.items()
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HITS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY).dropna(subset=["annot_lfc", "nc_lfc"])
    nc_types = df["nc_start_site_type"].dropna().unique().tolist()
    all_lfc = pd.concat([df["annot_lfc"], df["nc_lfc"]])
    global_lim = [all_lfc.min() - 0.3, all_lfc.max() + 0.3]
    site = df.set_index(SITE_KEY).index

    def listed(name: str) -> pd.Series:
        rows = pd.read_csv(HITS_DIR / name).set_index(SITE_KEY).index
        return pd.Series(site.isin(rows), index=df.index)

    figures = {
        OUT_DIR / "start_annot_vs_nc_depmap.png": (None, "", "coloured by DepMap dependency"),
        HITS_FIG_DIR / "start_annot_vs_nc_depmap90_nc_only.png": (
            listed("nc_only_depmap90_starts.csv"), "NC only, ≥90% DepMap",
            "non-canonical-only hits in ≥90% DepMap-dependent genes"),
        HITS_FIG_DIR / "start_annot_vs_nc_nonessential_depmap30_hits.png": (
            listed("nonessential_depmap30_hits.csv"), "hit, <30% DepMap",
            "all hits in <30% DepMap-dependent (non-essential) genes"),
    }  # fmt: skip
    fig, axes = plt.subplots(1, len(nc_types), figsize=(7 * len(nc_types), 7), squeeze=False,
                             dpi=200)  # fmt: skip
    for ax, nc_type in zip(axes[0], nc_types):
        plot_hit_categories(df[df["nc_start_site_type"] == nc_type], ax, nc_type, global_lim)
    fig.suptitle(f"Annotated vs Non-canonical start — {SIG_LABEL}, {EFF_LABEL}", fontsize=13,
                 fontweight="bold")  # fmt: skip
    fig.tight_layout()
    out = OUT_DIR / "start_annot_vs_nc_fdr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)

    for out, (highlight, label, subtitle) in figures.items():
        fig, axes = plt.subplots(1, len(nc_types), figsize=(7 * len(nc_types) + 1, 7),
                                 squeeze=False, layout="constrained", dpi=200)  # fmt: skip
        for ax, nc_type in zip(axes[0], nc_types):
            m = df["nc_start_site_type"] == nc_type
            pts = plot_scatter(df[m], ax, nc_type, global_lim,
                               None if highlight is None else highlight[m], label)  # fmt: skip
            # highlighted figures: label columns fill the top, the lower right is empty
            ax.legend(fontsize=8, framealpha=0.8,
                      loc="upper left" if highlight is None else "lower right")  # fmt: skip
        fig.colorbar(pts, ax=axes[0].tolist(), shrink=0.6, pad=0.02,
                     label="DepMap cell lines dependent on the gene (%)")  # fmt: skip
        fig.suptitle(f"Annotated vs Non-canonical start — {SIG_LABEL}, {subtitle}\n({EFF_LABEL})",
                     fontsize=13, fontweight="bold")  # fmt: skip
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
