"""ABE LFC vs CRISPR knockout dropout LFC, per gene, for each knockout timepoint.

abe_vs_ko_lfc.png   ABE annotated start codon (MAGeCK LFC) vs KO dropout; rows = ABE day
                    (D4, D6, D12, D21 vs D0), columns = KO timepoint (S1 D2-D5, S5 78h)

inscreen_ko_vs_crispr_ko_lfc.png  the ABE screen's own gene knockout guides (MAGeCK LFC, start and
                      stop KO sub-libraries pooled) vs CRISPR KO dropout; rows = day (D4-D21 vs
                      D0), columns = KO timepoint (S1 D2-D5, S5 78h)

abe_vs_inscreen_ko_lfc.png  ABE annotated start vs the in-screen KO guides of
                      the same gene on the same day (D4, D6, D12, D21; same samples and MAGeCK
                      normalisation), with the y = x line and a least-squares fit

Run scripts/abe_mageck_analysis.py first; this reads results/abe_mageck/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.correlations import (
    abe_gene_lfc,
    inscreen_ko_gene_lfc,
    s1_gene_dropout,
    s5_gene_dropout,
)
from be_screen_analysis.plotting import BLUE, GRAY, INK, MUTED, ORANGE, style

ROOT = Path(__file__).resolve().parents[2]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
OUT_DIR = ROOT / "figures" / "correlations"

N_BINS = 10


def _panel(ax: plt.Axes, x: pd.Series, y: pd.Series, stats_right: bool = False) -> None:
    d = pd.concat([x, y], axis=1, join="inner").dropna()
    xs, ys = d.iloc[:, 0], d.iloc[:, 1]
    ax.scatter(xs, ys, s=6, color=GRAY, alpha=0.5, linewidths=0)
    # binned medians (equal-count bins on x) show the shape of the relationship
    bins = pd.qcut(xs, N_BINS, duplicates="drop")
    med = d.groupby(bins, observed=True).median()
    ax.plot(med.iloc[:, 0], med.iloc[:, 1], color=BLUE, linewidth=2, marker="o", markersize=4)
    r, rho = xs.corr(ys), xs.rank().corr(ys.rank())  # Spearman = Pearson on ranks
    ax.text(
        0.97 if stats_right else 0.03,
        0.04,
        f"n = {len(d):,}\nr = {r:+.2f}, ρ = {rho:+.2f}",
        transform=ax.transAxes,
        fontsize=7,
        color=INK,
        va="bottom",
        ha="right" if stats_right else "left",
    )
    lo, hi = np.nanpercentile(xs, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    style(ax)


def grid(y: pd.DataFrame, ko: pd.DataFrame, ylabel: str, title: str, name: str) -> Path:
    """One panel per (y day, KO timepoint); ``ylabel`` is formatted with the day."""
    rows, cols = list(y.columns), list(ko.columns)
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(3 * len(cols), 2.8 * len(rows)),
        dpi=200,
        sharex="col",
        sharey="row",
    )
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            _panel(ax, ko[col], y[row])
            if i == 0:
                ax.set_title(col, fontsize=10, color=INK)
            if i == len(rows) - 1:
                ref = "NT-centred" if col.startswith("S1") else "intergenic-centred"
                ax.set_xlabel(f"KO remaining vs t0, log2FC\n({ref})", fontsize=8)
            if j == 0:
                ax.set_ylabel(ylabel.format(day=row.split()[-1]), fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"{title}; blue: binned medians", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    return out


def matched_days(abe: pd.DataFrame, ko: pd.DataFrame) -> Path:
    """ABE start vs in-screen KO per gene, one panel per shared day, on equal axes."""
    days = [c.split()[-1] for c in abe.columns]
    fig, axes = plt.subplots(1, len(days), figsize=(3.4 * len(days), 3.9), dpi=200)
    for ax, day in zip(axes, days):
        x, y = ko[f"In-screen KO {day}"], abe[f"ABE {day}"]
        _panel(ax, x, y, stats_right=True)  # lower left is crossed by y = x
        d = pd.concat([x, y], axis=1).dropna()
        xs, ys = d.iloc[:, 0], d.iloc[:, 1]
        # equal axes so the y = x line is a true 1:1 reference
        lo, hi = np.nanpercentile(np.r_[xs, ys], [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.plot(lim, lim, color=INK, linestyle="--", linewidth=1, label="y = x")
        slope, icept = np.polyfit(xs, ys, 1)
        ax.plot(
            lim,
            slope * np.array(lim) + icept,
            color=ORANGE,
            linewidth=1.5,
            label=f"fit: y = {slope:.2f}x {icept:+.2f}",
        )
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_title(f"{day} vs D0", fontsize=10, color=INK)
        ax.set_xlabel("In-screen KO guides, log2FC", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("ABE annotated start, log2FC", fontsize=8)
    fig.suptitle(
        "ABE annotated start codon loss vs in-screen knockout of the same gene, same day "
        "(MAGeCK, NT-normalised); blue: binned medians",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "abe_vs_inscreen_ko_lfc.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ko = s1_gene_dropout(pd.read_csv(S1_DIR / "HeLa_sgRNA_sequences_and_counts.csv")).join(
        s5_gene_dropout(pd.read_csv(S5_DIR / "sgRNA_sequences_and_counts.csv")), how="outer"
    )
    abe = abe_gene_lfc(MAGECK_DIR, "start")
    print(
        grid(
            abe,
            ko,
            "ABE annotated start\nlog2FC {day} vs D0",
            "ABE annotated start codon loss (MAGeCK, NT-normalised) vs CRISPR knockout dropout "
            "per gene (HeLa)",
            "abe_vs_ko_lfc.png",
        )
    )
    inscreen = inscreen_ko_gene_lfc(MAGECK_DIR)
    print(
        grid(
            inscreen,
            ko,
            "In-screen KO guides\nlog2FC {day} vs D0",
            "ABE screen's in-screen knockout guides (MAGeCK, NT-normalised) vs CRISPR knockout "
            "dropout per gene (HeLa)",
            "inscreen_ko_vs_crispr_ko_lfc.png",
        )
    )
    print(matched_days(abe, inscreen))


if __name__ == "__main__":
    main()
