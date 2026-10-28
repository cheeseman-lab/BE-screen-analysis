"""Shared plotting for the BE x shake-off screen comparisons (S1, S5)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b3ad"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e2"
CONTROL_GRAY = "#4a4a4a"  # control (null) distributions
# Tyler's start hit categories (filtering.start_hit_category) -> legend name, colour; the
# start_codon_analysis colours (notebooks/start_scatter.ipynb)
HIT_CATEGORIES = {
    "neither": ("Neither", "#CCCCCC"),
    "annot_only": ("Annotated only", "#4477AA"),
    "nc_only": ("Alt start only", "#CCBB44"),
    "both": ("Both", "#EE6677"),
}
# label_all: label column spacing and offsets (data units), name offsets (points)
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


def style(ax: plt.Axes) -> None:
    """Recessive grid and no top/right spines."""
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _label(
    ax: plt.Axes, df: pd.DataFrame, names: pd.Series, dx: float = 0.35, dy: float = 0.2
) -> None:
    # a label within (dx, dy) data units of an already placed one goes below-right instead
    placed: list[tuple[float, float]] = []
    for (x, y), name in zip(df[["x", "be"]].to_numpy(), names):
        crowded = any(abs(x - px) < dx and abs(y - py) < dy for px, py in placed)
        ax.annotate(
            name,
            (x, y),
            xytext=(4, -9) if crowded else (4, 2),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
        placed.append((x, y))


def _finish(ax: plt.Axes, xlabel: str, ylabel: str, title: str, legend_loc: str) -> None:
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)


def _save(fig: plt.Figure, out: Path) -> Path:
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def be_vs_screen_start(
    be_dir: Path,
    screen_lfc: pd.Series,
    hits: pd.DataFrame,
    xlabel: str,
    out: Path,
    n_labels: int = 8,
) -> Path:
    """Annotated ATG LFC (BE) vs a shake-off screen LFC, one point per gene."""
    start = pd.read_csv(be_dir / "start_codon_summary.csv")
    be = start.drop_duplicates("annot_start_site_target_id").groupby("Hugo_Symbol")["annot_lfc"]
    df = pd.concat([be.first().rename("be"), screen_lfc.rename("x")], axis=1, join="inner")
    df["hit"] = df.index.isin(hits.loc[hits["site_type"] == "Annotated", "Hugo_Symbol"])
    rho = df[["be", "x"]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    rest, hit = df[~df["hit"]], df[df["hit"]]
    ax.scatter(
        rest["x"],
        rest["be"],
        s=10,
        color=GRAY,
        alpha=0.6,
        linewidths=0,
        label=f"Genes in both screens (n={len(df):,})",
    )
    ax.scatter(
        hit["x"],
        hit["be"],
        s=28,
        color=BLUE,
        edgecolors="white",
        linewidths=0.8,
        label=f"Intersection, annotated ATG (n={len(hit)})",
    )
    top = hit.nlargest(n_labels, "x")
    _label(ax, top, top.index)
    _finish(
        ax,
        xlabel,
        "ABE annotated start codon, log2FC D12 vs D0",
        f"Start codon loss vs shake-off enrichment (Spearman ρ = {rho:.2f})",
        "lower left",
    )
    return _save(fig, out)


def be_vs_screen_stop(
    be_dir: Path,
    screen_lfc: pd.Series,
    hits: pd.DataFrame,
    be_filtered: pd.DataFrame,
    screen_name: str,
    xlabel: str,
    out: Path,
) -> Path:
    """Stop codon LFC (BE) vs a shake-off screen LFC, one point per stop site."""
    stop = pd.read_csv(be_dir / "stop_codon_summary.csv")
    df = stop[["Hugo_Symbol", "stop_site_target_id", "lfc"]].rename(columns={"lfc": "be"})
    df = df.merge(screen_lfc.rename("x"), left_on="Hugo_Symbol", right_index=True)
    sig = be_filtered.loc[be_filtered["screen"] == "stop", "site_id"]
    df["hit"] = df["stop_site_target_id"].isin(hits.loc[hits["screen"] == "stop", "site_id"])
    df["be_only"] = df["stop_site_target_id"].isin(sig) & ~df["hit"]
    rho = df[["be", "x"]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    rest = df[~df["hit"] & ~df["be_only"]]
    ax.scatter(
        rest["x"],
        rest["be"],
        s=10,
        color=GRAY,
        alpha=0.6,
        linewidths=0,
        label=f"Stop sites, gene in both screens (n={len(df):,})",
    )
    for mask, color, label in [
        (df["be_only"], ORANGE, f"ABE-significant, fails {screen_name} filter"),
        (df["hit"], BLUE, "Intersection"),
    ]:
        sub = df[mask]
        ax.scatter(
            sub["x"],
            sub["be"],
            s=28,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=f"{label} (n={len(sub)})",
        )
        _label(ax, sub, sub["Hugo_Symbol"])
    _finish(
        ax,
        xlabel,
        "ABE stop codon, log2FC D12 vs D0",
        f"Stop codon loss vs shake-off enrichment (Spearman ρ = {rho:.2f})",
        "upper right",
    )
    return _save(fig, out)


def scatter_fit(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    stats_right: bool = False,
    groups: pd.Series | None = None,
    group_colors: dict[str, str] | None = None,
    alpha: float = 0.45,
    color_by: pd.Series | None = None,
    cmap: str = "viridis",
    vrange: tuple[float, float] = (0, 100),
    missing_label: str = "no value",
    group_r: bool = True,
) -> tuple[float, float]:
    """Scatter, least-squares fit and stats on equal axes (same log2FC scale on x and y).

    ``groups`` (aligned with x and y) colours the points by group, labelled with each group's
    n and (``group_r``) Pearson r; ``color_by`` instead colours them by a continuous value
    (``cmap`` over ``vrange``, highest drawn on top, points without a value in light gray,
    labelled ``missing_label``). The fit and stats always use all points. Returns the shared
    axis limits.
    """
    d = pd.concat([x, y], axis=1, join="inner").dropna()
    xs, ys = d.iloc[:, 0], d.iloc[:, 1]
    if color_by is not None:
        c = color_by.reindex(d.index)
        missing = c.isna()
        ax.scatter(xs[missing], ys[missing], s=6, color=GRID, linewidths=0,
                   label=f"{missing_label} (n = {missing.sum():,})")  # fmt: skip
        order = c[~missing].sort_values().index
        ax.scatter(xs[order], ys[order], c=c[order], cmap=cmap, vmin=vrange[0], vmax=vrange[1],
                   s=7, alpha=0.8, linewidths=0)  # fmt: skip
    elif groups is None:
        ax.scatter(xs, ys, s=6, color=GRAY, alpha=0.5, linewidths=0)
    else:
        g = groups.loc[d.index]
        for name, color in group_colors.items():
            m = g == name
            r_g = f", r = {xs[m].corr(ys[m]):+.2f}" if group_r else ""
            ax.scatter(xs[m], ys[m], s=6, color=color, alpha=alpha, linewidths=0,
                       label=f"{name} (n = {m.sum():,}{r_g})")  # fmt: skip
    slope, icept = np.polyfit(xs, ys, 1)  # least-squares fit of y on x
    fit_x = np.array([xs.min(), xs.max()])
    ax.plot(
        fit_x,
        slope * fit_x + icept,
        color=BLUE,
        linewidth=2,
        label=f"fit: y = {slope:.2f}x {icept:+.2f}",
    )
    r, rho = xs.corr(ys), xs.rank().corr(ys.rank())  # Spearman = Pearson on ranks
    ax.text(
        0.97 if stats_right else 0.03,
        0.04,
        f"n = {len(d):,}, slope = {slope:.2f}\nr = {r:+.2f}, ρ = {rho:+.2f}",
        transform=ax.transAxes,
        fontsize=7,
        color=INK,
        va="bottom",
        ha="right" if stats_right else "left",
    )
    # same limits and scale on both axes, covering the central 99% of x and y together
    lo, hi = np.nanpercentile(np.r_[xs, ys], [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    style(ax)
    return lim


def alluvial(
    ax: plt.Axes,
    data: pd.DataFrame,
    stages: dict[str, list[str]],
    group: str,
    group_colors: dict[str, str],
    gap: float = 0.03,
    width: float = 0.06,
    weight: str | None = None,
    break_min: int = 0,
) -> None:
    """Alluvial (river) plot: one column per stage, ribbons between consecutive stages.

    ``data`` has one row per item; ``stages`` maps each stage column to its category order (top
    to bottom) and ``group`` names the column whose value colours the ribbons. Heights are item
    counts; each stage's column is scaled to the full height with ``gap`` between categories.
    ``weight`` names a column of per-item display weights (< 1 compresses those items): boxes
    holding compressed items and at least ``break_min`` items get a break mark; labels always
    give the true counts.
    """
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath

    w = data[weight] if weight else pd.Series(1.0, index=data.index)
    names, groups = list(stages), list(group_colors)
    scale = (1 - gap * (max(len(c) for c in stages.values()) - 1)) / w.sum()
    top = {}  # (stage, category) -> y of the node's top edge
    for x, (stage, cats) in enumerate(stages.items()):
        y = 1.0
        for cat in cats:
            mask = data[stage] == cat
            h = w[mask].sum() * scale
            top[stage, cat] = y
            ax.add_patch(Rectangle((x - width / 2, y - h), width, h, color=INK, lw=0))
            if (w[mask] < 1).any() and mask.sum() >= break_min:  # break mark: drawn compressed
                for dy in (0.006, -0.006):
                    ax.plot(
                        [x - width * 0.75, x + width * 0.75],
                        [y - h / 2 + dy - 0.006, y - h / 2 + dy + 0.006],
                        color="white",
                        lw=2.5,
                        solid_capstyle="butt",
                        zorder=5,
                    )
            ax.text(
                x + width / 2 + 0.02,
                y - h / 2,
                f"{cat}\n{mask.sum():,} ({mask.mean():.1%})",
                va="center",
                ha="left",
                fontsize=8,
                color=INK,
                zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
            )
            y -= h + gap
    for x, (left, right) in enumerate(zip(names, names[1:])):
        out_y = {c: top[left, c] for c in stages[left]}
        in_y = {c: top[right, c] for c in stages[right]}
        flows = w.groupby([data[left], data[right], data[group]]).sum()
        flows = flows[flows > 0]
        # outgoing ribbons ordered by destination then group, incoming by source then group
        rank = lambda kv, a, b: (  # noqa: E731
            stages[a].index(kv[0][0 if a == left else 1]),
            stages[b].index(kv[0][0 if b == left else 1]),
            groups.index(kv[0][2]),
        )
        starts = {}
        for key, k in sorted(flows.items(), key=lambda kv: rank(kv, left, right)):
            starts[key] = out_y[key[0]]
            out_y[key[0]] -= k * scale
        for key, k in sorted(flows.items(), key=lambda kv: rank(kv, right, left)):
            h = k * scale
            y0, y1 = starts[key], in_y[key[1]]
            in_y[key[1]] -= h
            x0, x1 = x + width / 2, x + 1 - width / 2
            xm = (x0 + x1) / 2
            verts = [
                (x0, y0), (xm, y0), (xm, y1), (x1, y1),
                (x1, y1 - h), (xm, y1 - h), (xm, y0 - h), (x0, y0 - h), (x0, y0),
            ]  # fmt: skip
            codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO]
            codes += [MplPath.CURVE4] * 3 + [MplPath.CLOSEPOLY]
            ax.add_patch(
                PathPatch(MplPath(verts, codes), fc=group_colors[key[2]], ec="none", alpha=0.55)
            )
    for x, stage in enumerate(names):
        ax.text(x, 1.03, stage, ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xlim(-0.2, len(names) - 0.4)
    ax.set_ylim(-0.02, 1.08)
    ax.axis("off")


def lfc_vs_length(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    group_colors: dict[str, str],
    fit_per_group: bool = False,
) -> float:
    """Scatter of LFC ``y`` against a length ``x`` (linear axis from 0) with least-squares fits.

    Points are coloured by ``groups`` (labelled with n and Spearman ρ per group). One fit over
    all points (black), or with ``fit_per_group`` one per group in its colour. The
    returned Spearman ρ uses all points.
    """
    d = pd.concat([x.rename("x"), y.rename("y"), groups.rename("g")], axis=1).dropna()
    spearman = lambda a, b: a.rank().corr(b.rank())  # noqa: E731  Pearson on ranks (no scipy)
    for name, color in group_colors.items():
        m = d["g"] == name
        rho = spearman(d.x[m], d.y[m])
        ax.scatter(d.loc[m, "x"], d.loc[m, "y"], s=10, color=color, alpha=0.6, linewidths=0,
                   label=f"{name} (n = {m.sum():,}, ρ = {rho:+.2f})")  # fmt: skip
    fits = {name: (d["g"] == name, color) for name, color in group_colors.items()}
    if not fit_per_group:
        fits = {"": (d["g"].notna(), INK)}
    for name, (m, color) in fits.items():
        slope, icept = np.polyfit(d.x[m], d.y[m], 1)
        fit_x = np.array([d.x[m].min(), d.x[m].max()])
        ax.plot(fit_x, slope * fit_x + icept, color=color, linewidth=2.4 if name else 1.8,
                label=f"{name + ' ' if name else ''}fit: LFC = {slope:+.4f}·aa {icept:+.2f}, "
                      f"r = {d.x[m].corr(d.y[m]):+.2f}")  # fmt: skip
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(0, d["x"].max() + 5)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style(ax)
    return spearman(d["x"], d["y"])


def control_margins(ax: plt.Axes, x: pd.Series, y: pd.Series, x_name: str, y_name: str) -> plt.Axes:
    """Control histograms of ``x`` along the top and ``y`` along the right edge of ``ax``.

    Returns the top axes (for the panel title).
    """
    div = make_axes_locatable(ax)
    top = div.append_axes("top", size="14%", pad=0.08, sharex=ax)
    right = div.append_axes("right", size="14%", pad=0.08, sharey=ax)
    top.hist(x, bins=np.linspace(*ax.get_xlim(), 90), density=True, histtype="stepfilled",
             color=CONTROL_GRAY, alpha=0.6)  # fmt: skip
    right.hist(y, bins=np.linspace(*ax.get_ylim(), 90), density=True, histtype="stepfilled",
               color=CONTROL_GRAY, alpha=0.6, orientation="horizontal")  # fmt: skip
    for m in (top, right):
        m.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
        for side in ("top", "right", "left", "bottom"):
            m.spines[side].set_visible(False)
    top.axvline(0, color=MUTED, linewidth=0.8)
    right.axhline(0, color=MUTED, linewidth=0.8)
    top.text(0.01, 0.9, x_name, transform=top.transAxes, fontsize=7, va="top", color=INK)
    right.set_title(y_name, fontsize=7, color=INK)
    return top


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
    ax: plt.Axes,
    rows: pd.DataFrame,
    cols: tuple[str, str, str],
    side: str,
    x_text: float,
    avoid: list[Bbox],
    fontsize: float,
) -> None:
    """Label ``rows`` in a column ``side`` of them with leader lines that do not cross.

    Label slots are LABEL_STEP apart (closer if the column would not fit), the column centred on
    the points' heights (kept inside the axes); slots start in y order, then any two labels whose
    leader lines cross swap slots until none do. The column starts at ``x_text`` and steps outward
    by COLUMN_STEP until none of its names overlaps a box in ``avoid`` (names already placed).
    ``cols`` is the (x, y, name) columns as in ``label_all``.
    """
    xc, yc, name = cols
    rows = rows.sort_values(yc, ascending=False)
    lo, hi = ax.get_ylim()
    step = min(LABEL_STEP, (hi - lo - 4 * LABEL_STEP) / len(rows))  # a long column still fits
    slots = rows[yc].to_list()
    for k in range(1, len(slots)):  # push labels down until they are ``step`` apart
        slots[k] = min(slots[k], slots[k - 1] - step)
    shift = rows[yc].mean() - sum(slots) / len(slots)  # recentre the column on the points
    shift = min(shift, hi - 2 * LABEL_STEP - slots[0])  # top label inside the axes
    shift = max(shift, lo + 2 * LABEL_STEP - slots[-1])  # bottom label inside the axes
    slots = [y + shift for y in slots]
    pts = list(zip(rows[xc], rows[yc]))
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
                row[name],
                (row[xc], row[yc]),
                xytext=(x, y),
                fontsize=fontsize,
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
                _leader(ax, (row[xc], row[yc]), (x, y))
            return
        for t in texts:
            t.remove()


def label_all(
    ax: plt.Axes, df: pd.DataFrame, x: str, y: str, name: str, fontsize: float = 4.5
) -> None:
    """Label every row: next to its point where it fits, else in a column with a leader line.

    Points are tried least crowded first; a name goes right, left, above or below its point if it
    then overlaps no placed name, no other labelled point and stays inside the axes; failing that,
    a short leader line to a nearby free spot. Names that still fit nowhere go to label columns just
    outside those points: those left of the leftovers' median x on the left, the rest on the
    right, so leader lines cannot cross. ``x`` / ``y`` / ``name`` are columns of ``df``; slot
    spacing and column offsets (LABEL_STEP, LABEL_GAP, COLUMN_STEP) are in data units.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xy = df[[x, y]].to_numpy()
    disp = ax.transData.transform(xy)
    r = POINT_RADIUS_PT * fig.dpi / 72
    point_boxes = [Bbox([[px - r, py - r], [px + r, py + r]]) for px, py in disp]
    near = ((disp[:, None, :] - disp[None, :, :]) ** 2).sum(-1) < (CROWD_PT * fig.dpi / 72) ** 2
    order = near.sum(axis=1).argsort(kind="stable")  # least crowded first
    axes_box = ax.get_window_extent(renderer)
    placed, leftover = [], []
    for k in order:
        row = df.iloc[k]
        for dx, dy, ha, va in DIRECT:
            t = ax.annotate(row[name], xy[k], xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, ha=ha, va=va, zorder=5)  # fmt: skip
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
            t = ax.annotate(row[name], xy[k], xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, va="center", zorder=5,
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
    left = rest[x] <= rest[x].median()
    edges = {
        "left": rest[x].min() - LABEL_GAP,
        "right": rest[x].max() + LABEL_GAP,
    }
    for side, rows in (("left", rest[left]), ("right", rest[~left])):
        if len(rows):
            _label_column(ax, rows, (x, y, name), side, edges[side], placed, fontsize)
