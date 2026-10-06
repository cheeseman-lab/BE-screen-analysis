"""Check essential-gene dropout over time in the S1 shake-off library (HeLa).

MAGeCK's median normalisation assumes most guides are neutral, which is false for an
essential-gene library, so guide LFCs here are centred on the non-targeting controls instead.
Each gene's median guide LFC is compared with a null built from random sets of 4
non-targeting guides.

Since S1 is a set of essential genes, CRIPSR knockout of genes should cause dropout. Heuristic
for editing efficiency.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.filtering import S1_NT, S1_REMAINING, nt_centred_guide_lfc, nt_null

ROOT = Path(__file__).resolve().parents[2]
COUNTS = ROOT / "data" / "shakeoff_data" / "table_s1" / "HeLa_sgRNA_sequences_and_counts.csv"
OUT_DIR = ROOT / "figures" / "be_s1"

DAYS = S1_REMAINING
NT = S1_NT
ALPHA = 0.05


def main() -> None:
    counts = pd.read_csv(COUNTS)
    lfc = nt_centred_guide_lfc(counts)
    is_nt = counts["gene"] == NT
    gene_lfc = lfc[~is_nt].groupby(counts.loc[~is_nt, "gene"]).median()
    null = nt_null(lfc, is_nt)

    print(f"{len(gene_lfc)} genes; one-sided depletion vs NT null (alpha={ALPHA})")
    for col, day in DAYS.items():
        cutoff = null[col].quantile(ALPHA)
        print(
            f"{day}: median gene LFC {gene_lfc[col].median():+.2f} | "
            f"depleted {100 * (gene_lfc[col] < cutoff).mean():.1f}% | "
            f"LFC<0 {100 * (gene_lfc[col] < 0).mean():.1f}%"
        )
    last = list(DAYS)[-1]
    never = gene_lfc[gene_lfc[last] >= null[last].quantile(ALPHA)]
    print(f"not depleted at D5: {len(never)} genes")
    never.sort_values(last, ascending=False).to_csv(OUT_DIR / "s1_not_depleted_D5.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    pos = np.arange(len(DAYS))
    for offset, data, color, label in [
        (-0.18, gene_lfc, "#2a78d6", f"Targeting genes (n={len(gene_lfc):,})"),
        (0.18, null, "#8a8984", "Non-targeting null (4-guide median)"),
    ]:
        parts = ax.violinplot(
            [data[c] for c in DAYS],
            positions=pos + offset,
            widths=0.32,
            showextrema=False,
            showmedians=True,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.35)
        parts["cmedians"].set_color(color)
        ax.plot([], [], color=color, linewidth=6, alpha=0.5, label=label)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(pos, list(DAYS.values()))
    ax.set_ylim(-4, 1.5)
    ax.set_xlabel("Day after induction (remaining population)")
    ax.set_ylabel("Log2 fold change vs D0\n(median of guides, NT-centred)")
    ax.set_title("HeLa S1 essential gene library: dropout over time", loc="left")
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    out = OUT_DIR / "s1_hela_dropout.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
