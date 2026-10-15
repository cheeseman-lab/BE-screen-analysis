"""Figures for the start codon trajectory analysis.

Run scripts/start_trajectory_analysis.py first; this reads results/start_trajectories/.
  1. start_site_trajectories.png  LFC vs day per start site, one panel per site type; dashed =
                                  control null 5-95% (non-targeting, safe harbor)
  2. start_rate_distribution.png  distribution of per-site dropout rates with both control nulls
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.mageck import MAGECK_DAYS
from be_screen_analysis.plotting import GRAY, INK, MUTED, ORANGE, style
from be_screen_analysis.trajectories import DAYS

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / "results" / "start_trajectories"
OUT_DIR = ROOT / "figures" / "start_trajectories"

TYPES = ["Annotated", "Extended", "Truncated"]
NULLS = {"nt": ("Non-targeting", INK), "aavs1": ("Safe harbor (AAVS1)", ORANGE)}


def _null_bands(ax: plt.Axes, null: pd.DataFrame) -> None:
    days = list(DAYS.values())
    for name, (_, color) in NULLS.items():
        lfc = null.loc[null["control"] == name, MAGECK_DAYS]
        # dashed edges on top: a filled band is hidden by the grey bulk
        for q in (0.05, 0.95):
            ax.plot(days, [0.0, *lfc.quantile(q)], color=color, linestyle="--", linewidth=1.2)


def trajectories(traj: pd.DataFrame, null: pd.DataFrame) -> Path:
    days = list(DAYS.values())
    fig, axes = plt.subplots(1, len(TYPES), figsize=(13, 4.6), dpi=200, sharey=True)
    for ax, site_type in zip(axes, TYPES):
        sub = traj[traj["site_type"] == site_type]
        ax.plot(days, sub[list(DAYS)].to_numpy().T, color=GRAY, linewidth=0.5, alpha=0.35)
        _null_bands(ax, null)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(days)
        ax.set_xlabel("Day")
        n_nt, n_aavs1 = (sub[f"faster_than_{n}_null"].sum() for n in NULLS)
        ax.set_title(
            f"{site_type} starts (n={len(sub):,})\n"
            f"faster than null: non-targeting {n_nt}, safe harbor {n_aavs1}",
            loc="left",
            fontsize=10,
        )
        style(ax)
    axes[0].set_ylabel("Site log2FC vs D0 (median efficient guide)")
    axes[0].set_ylim(-6, 2)
    handles = [
        plt.Line2D([], [], color=color, linestyle="--", label=f"{label} null 5–95%")
        for label, color in NULLS.values()
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
    fig.suptitle(
        "Start codon ablation: per-site dropout trajectories",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "start_site_trajectories.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def rate_distribution(traj: pd.DataFrame, null: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=200)
    lo, hi = traj["rate"].quantile([0.002, 0.998])
    bins = np.linspace(lo, hi, 80)
    ax.hist(traj["rate"], bins=bins, color=GRAY, label=f"Start sites (n={len(traj):,})")
    # nulls rescaled to the site count so the bulk can be compared with them directly
    for name, (label, color) in NULLS.items():
        rates = null.loc[null["control"] == name, "rate"]
        null_h, _ = np.histogram(rates, bins=bins)
        ax.step(
            bins[:-1],
            null_h * len(traj) / len(rates),
            where="post",
            color=color,
            linewidth=1.2,
            label=f"{label} null (scaled)",
        )
        ax.axvline(rates.quantile(0.05), color=color, linestyle=":", linewidth=1.2)
    ax.set_xlabel("Dropout rate (log2FC per day, slope through origin)")
    ax.set_ylabel("Start sites")
    ax.set_title("Per-site dropout rate (dotted: null 5th percentiles)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    style(ax)

    fig.tight_layout()
    out = OUT_DIR / "start_rate_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traj = pd.read_csv(RES_DIR / "start_site_trajectories.csv", index_col="site_id")
    null = pd.read_csv(RES_DIR / "null_trajectories.csv")
    print(trajectories(traj, null))
    print(rate_distribution(traj, null))


if __name__ == "__main__":
    main()
