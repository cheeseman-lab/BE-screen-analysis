"""Figures for the BE x S1 (ABE start/stop x S1 HeLa shake-off) analysis.

Run scripts/be_s1_intersect_analysis.py first; this reads its CSVs from results/be_s1/.
  1. be_vs_s1_shakeoff_start.png annotated ATG LFC (BE) vs D5 shake-off LFC (S1), per gene
  2. be_vs_s1_shakeoff_stop.png   stop codon LFC (BE) vs D5 shake-off LFC (S1), per stop site
  3. be_s1_efficiency_vs_lfc.png   guide LFC vs sensor editing efficiency
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.filtering import EFF_D0, EFF_D12
from be_screen_analysis.plotting import (
    BLUE,
    GRAY,
    MUTED,
    ORANGE,
    be_vs_screen_start,
    be_vs_screen_stop,
    style,
)

ROOT = Path(__file__).resolve().parents[2]
BE_DIR = ROOT / "data" / "start_stop_ABE_screen_data"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
RES_DIR = ROOT / "results" / "be_s1"
OUT_DIR = ROOT / "figures" / "be_s1"

S1_XLABEL = "S1 HeLa D5 shake-off vs remaining, log2FC\n(> 0: knockout cells detach)"


def efficiency_vs_lfc() -> Path:
    bins = np.arange(0, 105, 10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, sharey=True)
    panels = [
        ("start", "Annotated start codon guides", lambda g: g["start_site_type"] == "Annotated"),
        ("stop", "Stop codon guides", lambda g: g.index == g.index),
    ]
    for ax, (screen, title, keep) in zip(axes, panels):
        g = pd.read_csv(BE_DIR / f"{screen}_guide_summary.csv")
        g = g[keep(g)]
        eff = g[[EFF_D0, EFF_D12]].max(axis=1)
        ax.scatter(eff, g["lfc"], s=4, color=GRAY, alpha=0.35, linewidths=0)
        binned = g["lfc"].groupby(pd.cut(eff, bins, include_lowest=True))
        mid = bins[:-1] + 5
        ax.plot(
            mid,
            binned.median(),
            color=BLUE,
            linewidth=2,
            marker="o",
            markersize=5,
            label="Bin median",
        )
        ax.fill_between(
            mid,
            binned.quantile(0.25),
            binned.quantile(0.75),
            color=BLUE,
            alpha=0.15,
            linewidth=0,
            label="Bin IQR",
        )
        ax.axvline(20, color=ORANGE, linestyle="--", linewidth=1.2, label="20% cutoff")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        below = (eff < 20).sum()
        ax.set_title(f"{title} (n={len(g):,}; {below} below 20%)", loc="left", fontsize=10)
        ax.set_xlabel("Sensor editing efficiency, max of D0/D12 (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(-6, 2.5)
        style(ax)
    axes[0].set_ylabel("Guide log2FC D12 vs D0")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    out = OUT_DIR / "be_s1_efficiency_vs_lfc.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    hits = pd.read_csv(RES_DIR / "be_x_s1_intersection.csv")
    be_filtered = pd.read_csv(ROOT / "results" / "be_sites_filtered.csv")
    s1 = pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv").set_index("id")["d5s_d5r.neg.lfc"]
    print(be_vs_screen_start(BE_DIR, s1, hits, S1_XLABEL, OUT_DIR / "be_vs_s1_shakeoff_start.png"))
    print(
        be_vs_screen_stop(
            BE_DIR,
            s1,
            hits,
            be_filtered,
            "S1",
            S1_XLABEL,
            OUT_DIR / "be_vs_s1_shakeoff_stop.png",
        )
    )
    print(efficiency_vs_lfc())


if __name__ == "__main__":
    main()
