"""ABE vs CRISPR knockout dropout LFC per gene at matched early and later timepoints.

abe_start_vs_s1_s5_d4_lfc.png  ABE annotated start D4 vs S1 D4 and vs S5 78h (S5's only
                      timepoint); one point per gene, least-squares fit. S1 here is the published
                      MAGeCK gene LFC (as in the fitness subsets). Side histograms: the control
                      distribution of each axis on its own at that timepoint (top: the knockout
                      screen's, right: ABE AAVS1), as in null_across_time.png.

inscreen_ko_vs_s1_s5_d4_lfc.png  the same for the ABE screen's in-screen KO guides (D4)

abe_start_vs_s1_s5_d6_lfc.png, inscreen_ko_vs_s1_s5_d6_lfc.png  the same at the later timepoint:
                      ABE D6 vs S1 D5 (nearest S1 day) and vs S5 78h, controls at those days.

null_across_time.png  the control (null) distributions behind these comparisons, one panel per
                      control set, every timepoint overlaid (light to dark = early to late):
                      ABE start and ABE in-screen KO, median of 4 safe-harbour AAVS1 guides from
                      the matching MAGeCK run (start / start_ko), D4-D21; S1, mean of 4
                      non-targeting guides with LFCs computed as MAGeCK does (published units,
                      not centred), D2-D5; S5, median of 2 intergenic constructs
                      (intergenic-centred), 78h. 100,000 pseudo-genes per timepoint.

Run scripts/abe_mageck_analysis.py first; this reads results/abe_mageck/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.correlations import (
    abe_gene_lfc,
    inscreen_ko_gene_lfc,
    s5_gene_dropout,
)
from be_screen_analysis.essential import abe_control_null, s1_nt_null, s5_intergenic_null
from be_screen_analysis.filtering import s1_published_gene_lfc
from be_screen_analysis.plotting import INK, MUTED, control_margins, scatter_fit, style

ROOT = Path(__file__).resolve().parents[4]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
OUT_DIR = ROOT / "figures" / "old_analysis" / "ABE_validation" / "correlation_shakeoff"


# (ABE day, matching knockout day per screen); S1 day nearest the ABE day, S5 has only 78h
TIMEPOINTS = [("D4", {"S1": "D4", "S5": "78h"}), ("D6", {"S1": "D5", "S5": "78h"})]
POINT_COLOR = "#86b6ef"
CONTROL_NAMES = {"S1": "non-targeting", "S5": "intergenic"}
N_CONTROL = 100_000  # control pseudo-genes per timepoint


def pooled(
    y: pd.DataFrame,
    prefix: str,
    label: str,
    screens: dict[str, pd.DataFrame],
    name: str,
    y_control: pd.Series,
    x_controls: dict[str, pd.Series],
    abe_day: str,
    screen_day: dict[str, str],
) -> Path:
    """ABE data ``y`` (column "{prefix} {abe_day}") vs each screen at ``screen_day``.

    One panel per screen; ``label`` names the ABE data on the axis and titles. ``y_control`` and
    ``x_controls`` (per screen) are control pseudo-genes at those timepoints, drawn on the edges.
    """
    fig, axes = plt.subplots(1, len(screens), figsize=(6.2 * len(screens), 6), dpi=200)
    for ax, (screen, ko) in zip(axes, screens.items()):
        ko_day = screen_day[screen]
        pair = pd.concat(
            [ko[ko_day].rename("ko"), y[f"{prefix} {abe_day}"].rename("abe")], axis=1
        ).dropna()
        tag = f"{prefix} {abe_day} vs {screen} {ko_day}"
        scatter_fit(ax, pair["ko"], pair["abe"], groups=pd.Series(tag, index=pair.index),
                    group_colors={tag: POINT_COLOR})  # fmt: skip
        ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=1.5)
        top = control_margins(
            ax,
            x_controls[screen],
            y_control,
            f"{screen} controls ({CONTROL_NAMES[screen]})",
            "ABE controls\n(AAVS1)",
        )
        ref = "published MAGeCK" if screen == "S1" else "intergenic-centred"
        ax.set_xlabel(f"{screen} CRISPR KO remaining vs t0, {ko_day}, log2FC ({ref})", fontsize=9)
        ax.set_ylabel(f"{label}, {abe_day} log2FC vs D0 (MAGeCK)", fontsize=9)
        top.set_title(f"{label} {abe_day} vs {screen} {ko_day}", fontsize=10)
        print(label, screen, len(pair), f"r = {pair.ko.corr(pair.abe):+.3f}")
    fig.suptitle(
        f"ABE {abe_day}: one point per gene; blue line: least-squares fit; edges: control "
        "distribution of each axis",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


def null_across_time(nulls: dict[str, pd.DataFrame]) -> Path:
    """One panel per control set: pseudo-gene distribution at every timepoint (columns)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=200)
    for ax, (name, null) in zip(axes.flat, nulls.items()):
        lo, hi = np.nanpercentile(null.to_numpy(), [0.1, 99.9])
        bins = np.linspace(lo, hi, 80)
        shades = plt.get_cmap("Blues")(np.linspace(0.4, 1, null.shape[1]))
        for color, t in zip(shades, null.columns):
            ax.hist(null[t], bins=bins, density=True, histtype="step", linewidth=1.5,
                    color=color, label=f"{t} (median {null[t].median():+.2f})")  # fmt: skip
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(name, loc="left", fontsize=9)
        ax.set_xlabel("Control pseudo-gene log2FC", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        style(ax)
    fig.suptitle("Null (control) distributions across time", x=0.01, ha="left", fontsize=10,
                 color=INK)  # fmt: skip
    fig.tight_layout()
    out = OUT_DIR / "null_across_time.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s5_counts = pd.read_csv(S5_DIR / "sgRNA_sequences_and_counts.csv")
    s1 = s1_published_gene_lfc(pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv"))
    s5 = s5_gene_dropout(s5_counts)
    screens = {"S1": s1, "S5": s5.rename(columns={"S5 78h": "78h"})}
    s1_counts = pd.read_csv(S1_DIR / "HeLa_sgRNA_sequences_and_counts.csv")
    nulls = {
        "ABE annotated start: AAVS1": abe_control_null(MAGECK_DIR / "start", N_CONTROL, 4),
        "ABE in-screen KO: AAVS1": abe_control_null(MAGECK_DIR / "start_ko", N_CONTROL, 4),
        "S1: non-targeting": s1_nt_null(s1_counts, N_CONTROL),
        "S5: intergenic": s5_intergenic_null(s5_counts, N_CONTROL),
    }
    runs = [
        ("ABE", "ABE annotated start", abe_gene_lfc(MAGECK_DIR, "start"), "abe_start",
         "ABE annotated start: AAVS1"),
        ("In-screen KO", "ABE in-screen KO guides", inscreen_ko_gene_lfc(MAGECK_DIR), "inscreen_ko",
         "ABE in-screen KO: AAVS1"),
    ]  # fmt: skip
    for abe_day, screen_day in TIMEPOINTS:
        x_controls = {
            "S1": nulls["S1: non-targeting"][screen_day["S1"]],
            "S5": nulls["S5: intergenic"][screen_day["S5"]],
        }
        for prefix, label, data, stem, null in runs:
            name = f"{stem}_vs_s1_s5_{abe_day.lower()}_lfc.png"
            print(pooled(data, prefix, label, screens, name, nulls[null][abe_day], x_controls,
                         abe_day, screen_day))  # fmt: skip
    print(null_across_time(nulls))


if __name__ == "__main__":
    main()
