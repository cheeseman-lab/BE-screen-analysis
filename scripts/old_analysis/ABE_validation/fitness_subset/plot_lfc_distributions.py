"""Gene-level LFC distributions of each screen at the timepoints near D4-D6.

lfc_distributions.png  number of genes vs log2FC (remaining/later timepoint vs t0), one panel per
                       dataset, one row per available timepoint among D4/D5/D6:
                         S1 CRISPR KO (focused library)   D4, D5   published MAGeCK gene LFC
                         S5 CRISPR KO (genome-wide)       78h      intergenic-centred (only time)
                         ABE in-screen KO guides          D4, D6   MAGeCK, NT-normalised
                         ABE annotated start codon        D4, D6   MAGeCK, NT-normalised
                         ABE stop codon                   D4, D6   MAGeCK, NT-normalised
                       S1 values are the published d4r_d0 / d5r_d0 gene LFC (mageck.neg_HeLa:
                       median-ratio normalised, uncentred, mean of the gene's guides). Other
                       datasets: median over the gene's guides (ABE codon guides: >=20% sensor
                       efficiency; start: annotated ATG only).

Dotted lines: control pseudo-genes, each the median of k random control guides (k = the dataset's
median guides per gene), with the same normalisation as the data: S5 intergenic (k=2; cut, no gene),
ABE safe-harbour AAVS1 guides from the matching MAGeCK run (edit, no gene; in-screen KO k=4,
start k=4, stop k=2); 1,000,000 per timepoint, scaled to the dataset's gene count. S1 has no
cutting control, only non-targeting guides: its null is the mean of 4 NT guide LFCs computed as
MAGeCK does (same units as the published values; NT guides lost to a zero count excluded),
centred so its median sits on the peak of the gene distribution (kernel density mode), as a
stand-in for a cut-no-gene control.

lfc_distributions_cutoffs.png  the same with a cutoff per dataset and timepoint at the alpha
                       (0.0001) quantile of the control pseudo-genes (dashed) and the genes below it
                       filled. S5 78h and ABE annotated start D4 use a rank cutoff instead: the
                       LFC of the 240th most depleted gene, where those fitness subsets end.

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
from be_screen_analysis.essential import (
    ALPHA,
    abe_control_null,
    s1_nt_null,
    s5_intergenic_null,
    shift_null,
)
from be_screen_analysis.filtering import s1_published_gene_lfc
from be_screen_analysis.plotting import INK, MUTED, style

ROOT = Path(__file__).resolve().parents[4]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
OUT_DIR = ROOT / "figures" / "old_analysis" / "ABE_validation" / "fitness_subset"

# light to dark: earlier to later timepoint
TIME_COLORS = {"D4": "#86b6ef", "D5": "#2a78d6", "D6": "#174a8c", "78h": "#2a78d6"}
N_NULL = 1_000_000  # pseudo-genes per timepoint
TOP_N = 240  # size of the fitness subsets; in these panels the cutoff is the 240th gene
TOP_N_PANELS = {("S5 CRISPR KO (genome-wide)", "78h"), ("ABE annotated start codon", "D4")}


def datasets() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Per dataset: gene x timepoint LFCs (columns = timepoints), control pseudo-genes, label."""
    s1_counts = pd.read_csv(S1_DIR / "HeLa_sgRNA_sequences_and_counts.csv")
    s5_counts = pd.read_csv(S5_DIR / "sgRNA_sequences_and_counts.csv")
    s1 = s1_published_gene_lfc(pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv"))
    s1_null = s1_nt_null(s1_counts, N_NULL)[["D4", "D5"]]
    s5 = s5_gene_dropout(s5_counts)
    pick = lambda df, cols: df[cols].set_axis([c.split()[-1] for c in cols], axis=1)  # noqa: E731

    abe = lambda run, k: abe_control_null(MAGECK_DIR / run, N_NULL, k)[["D4", "D6"]]  # noqa: E731
    return {
        "S1 CRISPR KO (focused)": (
            s1[["D4", "D5"]],
            s1_null.apply(lambda col: shift_null(col, s1[col.name])),
            "non-targeting, mean of 4 guides,\ncentred on gene peak",
        ),
        "S5 CRISPR KO (genome-wide)": (
            pick(s5, ["S5 78h"]),
            s5_intergenic_null(s5_counts, N_NULL),
            "intergenic, 2 constructs",
        ),
        "ABE in-screen KO": (
            pick(inscreen_ko_gene_lfc(MAGECK_DIR), ["In-screen KO D4", "In-screen KO D6"]),
            abe("start_ko", 4),
            "safe harbor (AAVS1), 4 guides",
        ),
        "ABE annotated start codon": (
            pick(abe_gene_lfc(MAGECK_DIR, "start"), ["ABE D4", "ABE D6"]),
            abe("start", 4),
            "safe harbor (AAVS1), 4 guides",
        ),
        "ABE stop codon": (
            pick(abe_gene_lfc(MAGECK_DIR, "stop"), ["ABE D4", "ABE D6"]),
            abe("stop", 2),
            "safe harbor (AAVS1), 2 guides",
        ),
    }


def lfc_distributions(
    data: dict[str, tuple[pd.DataFrame, pd.DataFrame, str]],
    cutoffs: bool = False,
) -> Path:
    """Histogram per dataset; ``cutoffs`` adds the control-percentile cutoff and genes below it."""
    # one bin grid for every panel so widths are comparable; tails beyond it are pooled at the edges
    lo, hi = -3.0, 1.5
    bins = np.linspace(lo, hi, 91)
    # one column per dataset, one row per timepoint (earlier on top)
    n_rows = max(df.shape[1] for df, *_ in data.values())
    fig, axes = plt.subplots(
        n_rows, len(data), figsize=(4 * len(data), 3.4 * n_rows), dpi=200, squeeze=False
    )
    for j, (name, (df, control, control_label)) in enumerate(data.items()):
        for i in range(df.shape[1], n_rows):
            axes[i, j].axis("off")  # dataset with fewer timepoints (S5)
        for i, t in enumerate(df.columns):
            ax = axes[i, j]
            vals = df[t].dropna().clip(lo, hi)
            null = control[t]
            label = f"median {df[t].median():+.2f}, SD {df[t].std():.2f}"
            if cutoffs:
                if (name, t) in TOP_N_PANELS:  # cutoff where the 240-gene fitness subset ends
                    cutoff = df[t].nsmallest(TOP_N).max()
                    called = df[t] <= cutoff
                    label = f"top {TOP_N}: LFC \u2264 {cutoff:+.2f}\n{called.mean():.0%} of genes"
                else:
                    cutoff = null.quantile(ALPHA)
                    called = df[t] < cutoff
                    label = (
                        f"α = {ALPHA:g}: LFC < {cutoff:+.2f}\n"
                        f"{called.sum():,} genes ({called.mean():.0%})"
                    )
                ax.hist(
                    vals[called[vals.index]],
                    bins=bins,
                    color=TIME_COLORS[t],
                    alpha=0.45,
                    linewidth=0,
                )
                ax.axvline(cutoff, color=TIME_COLORS[t], linestyle="--", linewidth=1.2)
            ax.hist(
                vals,
                bins=bins,
                histtype="step",
                linewidth=1.6,
                color=TIME_COLORS[t],
                label=label,
            )
            # control pseudo-genes, rescaled to the dataset's gene count
            ctrl_h, _ = np.histogram(null.clip(lo, hi), bins=bins)
            ax.step(
                bins[:-1],
                ctrl_h * len(vals) / len(null),
                where="post",
                color=TIME_COLORS[t],
                linestyle=":",
                linewidth=1.4,
            )
            ax.axvline(0, color=MUTED, linewidth=0.8)
            ax.set_title(
                f"{name}, {t} (n = {len(vals):,} genes)\ndotted: control, {control_label}",
                loc="left",
                fontsize=9,
            )
            ax.set_xlim(lo, hi)
            ax.set_xlabel("Gene log2FC vs t0", fontsize=8)
            ax.legend(
                fontsize=7, loc="upper left", facecolor="white", edgecolor="none", framealpha=0.9
            )
            ax.tick_params(labelsize=7)
            style(ax)
    for row in axes:
        row[0].set_ylabel("Number of genes")
    note = (
        f"; dashed: cutoff at alpha = {ALPHA:g} (control quantile; S5 and ABE start D4: "
        f"240th most depleted gene), filled: genes below it"
        if cutoffs
        else ""
    )
    fig.suptitle(
        "Gene-level log2FC distributions near D4-D6 (S5: 78h, its only timepoint); "
        f"values beyond the axis are pooled in the edge bins{note}",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    out = OUT_DIR / ("lfc_distributions_cutoffs.png" if cutoffs else "lfc_distributions.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = datasets()
    for name, (df, control, _) in data.items():
        print(name, df.describe().loc[["count", "50%", "std"]].round(3).to_dict())
        print("  control", control.describe().loc[["50%", "std"]].round(3).to_dict())
        print("  control 5th pct", control.quantile(0.05).round(3).to_dict())
    print(lfc_distributions(data))
    print(lfc_distributions(data, cutoffs=True))


if __name__ == "__main__":
    main()
