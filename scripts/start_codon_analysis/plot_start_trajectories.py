"""Figures for the start codon trajectory analysis.

Run scripts/start_codon_analysis/start_trajectory_analysis.py first; this reads
results/start_trajectories/.
  1. start_site_trajectories.png  LFC vs day per efficient guide (>=20% editing), one panel per
                                  site type; guides at ATG starts coloured by their edit (ATG->GTG
                                  blue, ATG->ACG magenta), guides at other start codons gray;
                                  lines drawn in random order so neither colour hides the other;
                                  dashed = control null 5-95% (non-targeting, safe harbor)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from be_screen_analysis.mageck import MAGECK_DAYS
from be_screen_analysis.plotting import BLUE, GRAY, INK, MUTED, ORANGE, style
from be_screen_analysis.trajectories import DAYS

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / "results" / "start_trajectories"
OUT_DIR = ROOT / "figures" / "start_codon_analysis"

TYPES = ["Annotated", "Extended", "Truncated"]
NULLS = {"nt": ("Non-targeting", INK), "aavs1": ("Safe harbor (AAVS1)", ORANGE)}
EDIT_COLORS = {"GTG": BLUE, "ACG": "#c2185b"}  # edit of guides at ATG starts; others gray


def _null_bands(ax: plt.Axes, null: pd.DataFrame) -> None:
    days = list(DAYS.values())
    for name, (_, color) in NULLS.items():
        lfc = null.loc[null["control"] == name, MAGECK_DAYS]
        # dashed edges on top: a filled band is hidden by the grey bulk
        for q in (0.05, 0.95):
            ax.plot(days, [0.0, *lfc.quantile(q)], color=color, linestyle="--", linewidth=1.2)


def trajectories(guides: pd.DataFrame, null: pd.DataFrame) -> Path:
    days = list(DAYS.values())
    fig, axes = plt.subplots(1, len(TYPES), figsize=(13, 4.6), dpi=200, sharey=True)
    atg = guides["reference_codon"] == "ATG"
    color = guides["mutated_codon"].map(EDIT_COLORS).where(atg, GRAY)
    order = np.random.default_rng(0).permutation(len(guides))  # random draw order
    for ax, site_type in zip(axes, TYPES):
        idx = order[(guides["site_type"].to_numpy() == site_type)[order]]
        sub = guides.iloc[idx]
        segments = [np.column_stack([days, row]) for row in sub[list(DAYS)].to_numpy()]
        ax.add_collection(LineCollection(segments, colors=color.iloc[idx].tolist(),
                                         linewidths=0.4, alpha=0.35))  # fmt: skip
        _null_bands(ax, null)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xlim(-1, 22)
        ax.set_xticks(days)
        ax.set_xlabel("Day")
        n = {e: (sub["mutated_codon"].eq(e) & sub["reference_codon"].eq("ATG")).sum()
             for e in EDIT_COLORS}  # fmt: skip
        ax.set_title(
            f"{site_type} starts ({len(sub):,} guides)\n"
            f"ATG→GTG {n['GTG']:,}, ATG→ACG {n['ACG']:,}, "
            f"other codons {len(sub) - sum(n.values()):,}",
            loc="left",
            fontsize=10,
        )
        style(ax)
    axes[0].set_ylabel("Guide log2FC vs D0 (MAGeCK)")
    axes[0].set_ylim(-6, 2)
    handles = [
        plt.Line2D([], [], color=EDIT_COLORS["GTG"], linewidth=2, label="ATG→GTG guide"),
        plt.Line2D([], [], color=EDIT_COLORS["ACG"], linewidth=2, label="ATG→ACG guide"),
        plt.Line2D([], [], color=GRAY, linewidth=2, label="guide at another start codon"),
    ]
    handles += [
        plt.Line2D([], [], color=c, linestyle="--", label=f"{label} null 5–95%")
        for label, c in NULLS.values()
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(
        "Start codon ablation: per-guide dropout trajectories",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "start_site_trajectories.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guides = pd.read_csv(RES_DIR / "start_guide_trajectories.csv", index_col="guide_id")
    null = pd.read_csv(RES_DIR / "null_trajectories.csv")
    print(trajectories(guides, null))


if __name__ == "__main__":
    main()
