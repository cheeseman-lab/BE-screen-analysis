"""ABE edit type: GTG vs ACG edits of the same annotated ATG start.

abe_gtg_vs_acg_lfc.png  one point per annotated ATG start that has guides for both edits
                        (ATG->GTG and ATG->ACG): x = median D12 log2FC of its GTG guides,
                        y = median D12 log2FC of its ACG guides (Tyler's guide summary). Two
                        panels, the same points coloured by the median D12 sensor editing of
                        the GTG guides (left) and of the ACG guides (right). Dark lines: x = 0,
                        y = 0; gray: least-squares fit.

abe_gtg_vs_acg_lfc_eff20.png  the same with only efficient guides (>= 20% sensor editing at D0 or
                        D12, the cutoff used elsewhere): each edit's median is over its efficient
                        guides, and a start needs an efficient guide for both edits.

abe_gtg_vs_acg_lfc_eff_thresholds.png  2x2, the same comparison at editing cutoffs of 20, 40,
                        60 and 80%, met at both D0 and D12 (so every guide in a panel is at or
                        above its cutoff in the D12 editing the colour shows); starts that lose
                        every guide of either edit at a cutoff are left out. Points coloured by
                        the lower of the two edits' median D12 editing; shared axes; gray:
                        least-squares fit per panel.
abe_gtg_vs_acg_lfc_eff_thresholds_{gtg,acg}_color.png  the same grid coloured by the median D12
                        editing of the GTG guides, or of the ACG guides.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.filtering import EFF_D0, EFF_D12
from be_screen_analysis.plotting import INK, style

ROOT = Path(__file__).resolve().parents[3]
GUIDES = ROOT / "data" / "start_stop_ABE_screen_data" / "summaries" / "start_guide_summary.csv"
OUT_DIR = ROOT / "figures" / "ABE_validation" / "correlation_edit_type"
EFF = "sensor_mean_D12_pct_successful_target_codon_mutation"
EDITS = ["GTG", "ACG"]  # edit type 1 (x), edit type 2 (y)
FIT_COLOR = "#9a9a9a"  # light gray best-fit line
MIN_EFF = 20  # % sensor editing at D0 or D12 for an efficient guide
THRESHOLDS = [20, 40, 60, 80]  # editing cutoffs for the threshold grid


def edit_type_table(
    guides: pd.DataFrame, min_eff: float | None = None, both_days: bool = False
) -> pd.DataFrame:
    """Per annotated ATG site with both edits: median LFC and editing per edit type.

    ``min_eff``: keep only guides with at least this % sensor editing first, at D0 or D12
    (default) or at both D0 and D12 (``both_days``).
    """
    if min_eff is not None:
        passed = guides[[EFF_D0, EFF_D12]] >= min_eff
        guides = guides[passed.all(axis=1) if both_days else passed.any(axis=1)]
    g = guides[(guides["reference_codon"] == "ATG")
               & guides["start_site_type"].str.contains("Annotated")]  # fmt: skip
    per = g.groupby(["start_site_target_id", "mutated_codon"])[["lfc", EFF]].median().unstack()
    per.columns = [f"{v}_{e}" for v, e in per.columns]
    return per.dropna(subset=[f"lfc_{e}" for e in EDITS])


def gtg_vs_acg(table: pd.DataFrame, name: str, subtitle: str = "") -> Path:
    x, y = table[f"lfc_{EDITS[0]}"], table[f"lfc_{EDITS[1]}"]
    lo, hi = np.nanpercentile(np.r_[x, y], [0.5, 99.5])
    lim = (lo - 0.2, hi + 0.2)
    slope, icept = np.polyfit(x, y, 1)  # least-squares fit of ACG on GTG LFC
    fit_x = np.array(lim)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=200)
    for ax, edit in zip(axes, EDITS):
        pts = ax.scatter(x, y, c=table[f"{EFF}_{edit}"], cmap="viridis", vmin=0, vmax=100, s=8,
                         linewidths=0)  # fmt: skip
        ax.axhline(0, color=INK, linewidth=1.2)  # dark x and y axes through the origin
        ax.axvline(0, color=INK, linewidth=1.2)
        ax.plot(fit_x, slope * fit_x + icept, color=FIT_COLOR, linewidth=1.8,
                label=f"fit: y = {slope:.2f}x {icept:+.2f}")  # fmt: skip
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"ATG→{EDITS[0]} guides, log2FC D12", fontsize=9)
        ax.set_ylabel(f"ATG→{EDITS[1]} guides, log2FC D12", fontsize=9)
        ax.set_title(f"Coloured by {edit} editing (n = {len(table):,}, r = {x.corr(y):+.2f})",
                     loc="left", fontsize=10)  # fmt: skip
        fig.colorbar(pts, ax=ax, shrink=0.8, label=f"ATG→{edit} sensor editing D12 (%)")
        style(ax)
        for side in ("left", "bottom"):  # darker axes
            ax.spines[side].set_color(INK)
            ax.spines[side].set_linewidth(1.5)
        ax.tick_params(colors=INK)
    fig.suptitle(f"Annotated ATG starts with both edit types: GTG vs ACG edit dropout{subtitle}",
                 x=0.01, ha="left", fontsize=10)  # fmt: skip
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


def threshold_grid(guides: pd.DataFrame, color_by: str | None = None) -> Path:
    """GTG vs ACG at increasing editing cutoffs, one panel per cutoff, shared axes.

    Points coloured by ``color_by``'s ("GTG" / "ACG") median D12 editing, or by the lower of the
    two edits' (None).
    """
    # a guide must reach the cutoff at both D0 and D12
    tables = {e: edit_type_table(guides, e, both_days=True) for e in THRESHOLDS}
    first = tables[THRESHOLDS[0]]
    lo, hi = np.nanpercentile(
        np.r_[first[f"lfc_{EDITS[0]}"], first[f"lfc_{EDITS[1]}"]], [0.5, 99.5]
    )
    lim = (lo - 0.2, hi + 0.2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
    for ax, (cut, table) in zip(axes.flat, tables.items()):
        x, y = table[f"lfc_{EDITS[0]}"], table[f"lfc_{EDITS[1]}"]
        if color_by is None:
            color = table[[f"{EFF}_{e}" for e in EDITS]].min(axis=1)  # weaker edit's editing
        else:
            color = table[f"{EFF}_{color_by}"]
        pts = ax.scatter(x, y, c=color, cmap="viridis", vmin=0, vmax=100, s=8, linewidths=0)
        ax.axhline(0, color=INK, linewidth=1.2)
        ax.axvline(0, color=INK, linewidth=1.2)
        if len(table) > 2:
            slope, icept = np.polyfit(x, y, 1)  # least-squares fit of ACG on GTG LFC
            ax.plot(np.array(lim), slope * np.array(lim) + icept, color=FIT_COLOR, linewidth=1.8,
                    label=f"fit: y = {slope:.2f}x {icept:+.2f}")  # fmt: skip
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"ATG→{EDITS[0]} guides, log2FC D12", fontsize=9)
        ax.set_ylabel(f"ATG→{EDITS[1]} guides, log2FC D12", fontsize=9)
        ax.set_title(f"\u2265 {cut}% at D0 and D12: n = {len(table):,} starts, "
                     f"r = {x.corr(y):+.2f}", loc="left", fontsize=10)  # fmt: skip
        style(ax)
        for side in ("left", "bottom"):  # darker axes
            ax.spines[side].set_color(INK)
            ax.spines[side].set_linewidth(1.5)
        ax.tick_params(colors=INK)
    fig.suptitle("Annotated ATG starts: GTG vs ACG edit dropout at increasing editing cutoffs "
                 "(met at both D0 and D12); starts losing either edit dropped", x=0.01, ha="left",
                 fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    cax = fig.add_axes((0.92, 0.3, 0.015, 0.4))
    label = ("Lower of the two edits' sensor editing D12 (%)" if color_by is None
             else f"ATG→{color_by} sensor editing D12 (%)")  # fmt: skip
    fig.colorbar(pts, cax=cax, label=label)
    suffix = "" if color_by is None else f"_{color_by.lower()}_color"
    out = OUT_DIR / f"abe_gtg_vs_acg_lfc_eff_thresholds{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guides = pd.read_csv(GUIDES)
    print(gtg_vs_acg(edit_type_table(guides), "abe_gtg_vs_acg_lfc.png"))
    print(gtg_vs_acg(edit_type_table(guides, MIN_EFF), f"abe_gtg_vs_acg_lfc_eff{MIN_EFF}.png",
                     f" (guides with \u2265{MIN_EFF}% editing only)"))  # fmt: skip
    for color_by in (None, *EDITS):
        print(threshold_grid(guides, color_by))


if __name__ == "__main__":
    main()
