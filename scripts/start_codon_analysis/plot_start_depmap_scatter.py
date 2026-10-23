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
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox

from be_screen_analysis.filtering import start_hit_category

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
LABEL_STEP = 0.2  # vertical spacing of the label column (LFC units)
LABEL_GAP = 0.6  # label columns start this far outside their points ...
COLUMN_STEP, COLUMN_TRIES = 0.3, 20  # ... and step outward until clear of placed names
# a name next to its point: (dx, dy) offset in points, alignment; tried in this order
DIRECT = [(3, 0, "left", "center"), (-3, 0, "right", "center"), (0, 2.5, "center", "bottom"),
          (0, -2.5, "center", "top")]  # fmt: skip
POINT_RADIUS_PT = 2.5  # labels keep clear of other labelled points by this radius
CROWD_PT = 12  # neighbours within this distance count towards a point's crowding
# names that do not fit beside their point: leader-line offsets (points), nearest first
NEAR_RING = [
    (round(r * np.cos(a)), round(r * np.sin(a)))
    for r in (10, 16, 24, 32, 42, 54, 68)
    for a in np.deg2rad([0, 180, 90, 270, 45, 135, 225, 315])
]


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


def _leader(ax: plt.Axes, xy, xytext, textcoords: str = "data") -> None:
    """A thin leader line from a name at ``xytext`` to its point ``xy``.

    Drawn apart from the name, so the name's extent (used for overlap checks) excludes the line.
    """
    line = dict(arrowstyle="-", color="#777777", lw=0.4, shrinkA=1, shrinkB=2)
    ax.annotate("", xy, xytext=xytext, textcoords=textcoords, zorder=4, arrowprops=line)


def _crosses(p: tuple, q: tuple, r: tuple, s: tuple) -> bool:
    """Whether segment p-q crosses segment r-s (proper intersection)."""

    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    return side(p, q, r) * side(p, q, s) < 0 and side(r, s, p) * side(r, s, q) < 0


def _label_column(
    ax: plt.Axes, rows: pd.DataFrame, side: str, x_text: float, avoid: list[Bbox]
) -> None:
    """Label ``rows`` in a column ``side`` of them with leader lines that do not cross.

    Label slots are LABEL_STEP apart (closer if the column would not fit), the column centred on
    the points' heights (kept inside the axes); slots start in y order, then any two labels whose
    leader lines cross swap slots until none do. The column starts at ``x_text`` and steps outward
    by COLUMN_STEP until none of its names overlaps a box in ``avoid`` (names already placed).
    """
    rows = rows.sort_values("nc_lfc", ascending=False)
    lo, hi = ax.get_ylim()
    step = min(LABEL_STEP, (hi - lo - 4 * LABEL_STEP) / len(rows))  # a long column still fits
    slots = rows["nc_lfc"].to_list()
    for k in range(1, len(slots)):  # push labels down until they are ``step`` apart
        slots[k] = min(slots[k], slots[k - 1] - step)
    shift = rows["nc_lfc"].mean() - sum(slots) / len(slots)  # recentre the column on the points
    shift = min(shift, hi - 2 * LABEL_STEP - slots[0])  # top label inside the axes
    shift = max(shift, lo + 2 * LABEL_STEP - slots[-1])  # bottom label inside the axes
    slots = [y + shift for y in slots]
    pts = list(zip(rows["annot_lfc"], rows["nc_lfc"]))
    renderer = ax.figure.canvas.get_renderer()
    outward = -1 if side == "left" else 1
    for m in range(COLUMN_TRIES):
        x = x_text + outward * m * COLUMN_STEP
        ys = list(slots)
        for _ in range(len(ys) ** 2):  # each swap shortens the total leader length, so this ends
            swapped = False
            for i in range(len(ys)):
                for j in range(i + 1, len(ys)):
                    if _crosses((x, ys[i]), pts[i], (x, ys[j]), pts[j]):
                        ys[i], ys[j] = ys[j], ys[i]
                        swapped = True
            if not swapped:
                break
        texts = [
            ax.annotate(
                row["Hugo_Symbol"],
                (row["annot_lfc"], row["nc_lfc"]),
                xytext=(x, y),
                fontsize=4.5,
                va="center",
                zorder=5,
                ha="right" if side == "left" else "left",
            )  # fmt: skip
            for (_, row), y in zip(rows.iterrows(), ys)
        ]
        boxes = [t.get_window_extent(renderer) for t in texts]
        if m == COLUMN_TRIES - 1 or not any(b.overlaps(a) for b in boxes for a in avoid):
            avoid.extend(boxes)
            for (_, row), y in zip(rows.iterrows(), ys):
                _leader(ax, (row["annot_lfc"], row["nc_lfc"]), (x, y))
            return
        for t in texts:
            t.remove()


def _label_all(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Label every row: next to its point where it fits, else in a column with a leader line.

    Points are tried least crowded first; a name goes right, left, above or below its point if it
    then overlaps no placed name, no other labelled point and stays inside the axes; failing that,
    a short leader line to a nearby free spot. Names that still fit nowhere go to label columns just
    outside those points: those left of the leftovers' median x on the left, the rest on the
    right, so leader lines cannot cross.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xy = df[["annot_lfc", "nc_lfc"]].to_numpy()
    disp = ax.transData.transform(xy)
    r = POINT_RADIUS_PT * fig.dpi / 72
    point_boxes = [Bbox([[x - r, y - r], [x + r, y + r]]) for x, y in disp]
    near = ((disp[:, None, :] - disp[None, :, :]) ** 2).sum(-1) < (CROWD_PT * fig.dpi / 72) ** 2
    order = near.sum(axis=1).argsort(kind="stable")  # least crowded first
    axes_box = ax.get_window_extent(renderer)
    placed, leftover = [], []
    for k in order:
        row = df.iloc[k]
        for dx, dy, ha, va in DIRECT:
            t = ax.annotate(row["Hugo_Symbol"], xy[k], xytext=(dx, dy), textcoords="offset points",
                            fontsize=4.5, ha=ha, va=va, zorder=5)  # fmt: skip
            bb = t.get_window_extent(renderer)
            clash = (
                any(bb.overlaps(b) for b in placed)
                or any(bb.overlaps(b) for i, b in enumerate(point_boxes) if i != k)
                or not axes_box.contains(bb.x0, bb.y0)
                or not axes_box.contains(bb.x1, bb.y1)
            )
            if not clash:
                placed.append(bb)
                break
            t.remove()
        else:
            leftover.append(k)
    # second pass: a short leader line, the name at growing distance in 8 directions
    still = []
    for k in leftover:
        row = df.iloc[k]
        for dx, dy in NEAR_RING:
            t = ax.annotate(row["Hugo_Symbol"], xy[k], xytext=(dx, dy), textcoords="offset points",
                            fontsize=4.5, va="center", zorder=5,
                            ha="left" if dx > 0 else "right" if dx < 0 else "center")  # fmt: skip
            bb = t.get_window_extent(renderer)
            clash = (
                any(bb.overlaps(b) for b in placed)
                or any(bb.overlaps(b) for i, b in enumerate(point_boxes) if i != k)
                or not axes_box.contains(bb.x0, bb.y0)
                or not axes_box.contains(bb.x1, bb.y1)
            )
            if not clash:
                placed.append(bb)
                _leader(ax, xy[k], (dx, dy), "offset points")
                break
            t.remove()
        else:
            still.append(k)
    rest = df.iloc[sorted(still)]
    if rest.empty:
        return
    left = rest["annot_lfc"] <= rest["annot_lfc"].median()
    edges = {
        "left": rest["annot_lfc"].min() - LABEL_GAP,
        "right": rest["annot_lfc"].max() + LABEL_GAP,
    }
    for side, rows in (("left", rest[left]), ("right", rest[~left])):
        if len(rows):
            _label_column(ax, rows, side, edges[side], placed)


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
        _label_all(ax, hi)
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
