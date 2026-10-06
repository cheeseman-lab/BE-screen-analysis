"""Figures for the preliminary ABE start/stop x S1 HeLa shake-off analysis.

Run scripts/preliminary_filter.py first; this reads its CSVs from results/preliminary/.
  1. prelim_be_vs_s1_shakeoff.png   annotated ATG LFC (BE) vs D5 shake-off LFC (S1), per gene
  2. prelim_efficiency_vs_lfc.png   guide LFC vs sensor editing efficiency
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.filtering import EFF_D0, EFF_D12

ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "data" / "start_stop_ABE_screen_data"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
RES_DIR = ROOT / "results" / "preliminary"
OUT_DIR = ROOT / "figures"

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b3ad"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e2"


def style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def be_vs_s1(hits: pd.DataFrame) -> Path:
    start = pd.read_csv(BE_DIR / "start_codon_summary.csv")
    be = start.drop_duplicates("annot_start_site_target_id").groupby("Hugo_Symbol")["annot_lfc"]
    s1 = pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv").set_index("id")["d5s_d5r.neg.lfc"]
    df = pd.concat([be.first().rename("be"), s1.rename("s1")], axis=1, join="inner")
    df["hit"] = df.index.isin(hits.loc[hits["site_type"] == "Annotated", "Hugo_Symbol"])
    rho = df[["be", "s1"]].corr(method="spearman").iloc[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    rest = df[~df["hit"]]
    ax.scatter(
        rest["s1"],
        rest["be"],
        s=10,
        color=GRAY,
        alpha=0.6,
        linewidths=0,
        label=f"Genes in both screens (n={len(df):,})",
    )
    hit = df[df["hit"]]
    ax.scatter(
        hit["s1"],
        hit["be"],
        s=28,
        color=BLUE,
        edgecolors="white",
        linewidths=0.8,
        label=f"Intersection, annotated ATG (n={len(hit)})",
    )
    for gene, r in hit.nlargest(8, "s1").iterrows():
        ax.annotate(
            gene,
            (r["s1"], r["be"]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
            color=INK,
        )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("S1 HeLa D5 shake-off vs remaining, log2FC\n(> 0: knockout cells detach)")
    ax.set_ylabel("ABE annotated start codon, log2FC D12 vs D0")
    ax.set_title(
        f"Start codon loss vs shake-off enrichment (Spearman ρ = {rho:.2f})",
        loc="left",
        fontsize=10,
    )
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    out = OUT_DIR / "prelim_be_vs_s1_shakeoff.png"
    fig.savefig(out)
    plt.close(fig)
    return out


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
    out = OUT_DIR / "prelim_efficiency_vs_lfc.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    hits = pd.read_csv(RES_DIR / "be_x_s1_intersection.csv")
    for out in (be_vs_s1(hits), efficiency_vs_lfc()):
        print(out)


if __name__ == "__main__":
    main()
