"""Figures for the BE x S5 (ABE start/stop x S5 genome-wide HeLa shake-off) analysis.

Run scripts/old_analysis/hits/be_s5_intersect_analysis.py first; this reads its CSVs
from results/old_analysis/hits/be_s5/ and the filtered BE sites from
results/old_analysis/hits/be_sites_filtered.csv.
  1. be_vs_s5_shakeoff_start.png  annotated ATG LFC (BE) vs 78h shake-off LFC (S5), per gene
  2. be_vs_s5_shakeoff_stop.png   stop codon LFC (BE) vs 78h shake-off LFC (S5), per stop site
  3. s5_dropout_essential.png     78h dropout: S1 essential subset vs other genes vs NT and
                                  intergenic (cutting) control nulls
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.filtering import (
    S5_DAY0,
    S5_NT,
    S5_REMAINING,
    nt_centred_guide_lfc,
    nt_null,
    s5_control_ids,
)
from be_screen_analysis.plotting import (
    BLUE,
    GRAY,
    INK,
    MUTED,
    ORANGE,
    be_vs_screen_start,
    be_vs_screen_stop,
    style,
)

ROOT = Path(__file__).resolve().parents[3]
BE_DIR = ROOT / "data" / "start_stop_ABE_screen_data" / "summaries"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
BE_FILTERED = ROOT / "results" / "old_analysis" / "hits" / "be_sites_filtered.csv"
RES_DIR = ROOT / "results" / "old_analysis" / "hits" / "be_s5"
OUT_DIR = ROOT / "figures" / "old_analysis" / "hits" / "be_s5"

S5_XLABEL = "S5 HeLa 78h shake-off vs remaining, log2FC\n(> 0: knockout cells detach)"
ALPHA = 0.05


def s5_dropout_essential(counts: pd.DataFrame, s1_genes: set[str]) -> Path:
    lfc = nt_centred_guide_lfc(
        counts, ref=S5_DAY0, cols=S5_REMAINING, gene_col="target_symbol", nt=S5_NT
    )
    col = list(S5_REMAINING)[0]
    is_nt = counts["target_symbol"] == S5_NT
    targeting = counts["category"] == "GENE_TARGETING"
    gene_lfc = lfc.loc[targeting, col].groupby(counts.loc[targeting, "target_symbol"]).median()
    gene_lfc = gene_lfc.drop("0", errors="ignore")  # constructs with no gene symbol
    essential = gene_lfc[gene_lfc.index.isin(s1_genes)]
    other = gene_lfc[~gene_lfc.index.isin(s1_genes)]
    null = nt_null(lfc, is_nt, guides_per_gene=2)[col]
    # intergenic constructs cut DNA but disrupt no gene: their shift from NT is cutting toxicity
    intergenic = nt_null(lfc, counts["category"] == "INTERGENIC_CONTROL", guides_per_gene=2)[col]
    cutoff = null.quantile(ALPHA)

    groups = [
        (null, MUTED, "Non-targeting\nnull (no cut)"),
        (intergenic, ORANGE, "Intergenic\nnull (cuts DNA)"),
        (other, GRAY, f"Other genes\n(n={len(other):,})"),
        (essential, BLUE, f"S1 essential\nsubset (n={len(essential):,})"),
    ]
    lo, hi = -1.5, 1.0
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=200)
    pos = np.arange(len(groups))
    parts = ax.violinplot(
        [g[0].clip(lo, hi) for g in groups],
        positions=pos,
        widths=0.75,
        showextrema=False,
        showmedians=True,
    )
    for body, (_, color, _) in zip(parts["bodies"], groups):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(1.5)
    for x, (data, _, _) in zip(pos, groups):
        below = 100 * (data < cutoff).mean()
        ax.annotate(
            f"median {data.median():+.2f}\n{below:.0f}% below",
            (x, lo),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.axhline(cutoff, color=INK, linestyle=":", linewidth=1.2)
    ax.text(
        0.99,
        0.98,
        f"dotted line: dropout_consistent cutoff\n(NT 5th percentile, {cutoff:+.2f})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=INK,
    )
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(pos, [g[2] for g in groups], fontsize=8)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("78h remaining vs 0h, log2FC\n(median of 2 constructs, NT-centred)")
    ax.set_title("S5 HeLa: dropout of essential vs other genes at 78h", loc="left", fontsize=10)
    style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = OUT_DIR / "s5_dropout_essential.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(RES_DIR / "be_x_s5_intersection.csv")
    be_filtered = pd.read_csv(BE_FILTERED)
    counts = pd.read_csv(S5_DIR / "sgRNA_sequences_and_counts.csv")
    s1_genes = set(pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv")["id"]) - {"nontargeting"}

    shake = pd.read_csv(S5_DIR / "MAGeCK_shake-off_screen.csv").set_index("id")["neg|lfc"]
    shake = shake[~s5_control_ids(shake.index, counts).to_numpy()]

    print(
        be_vs_screen_start(
            BE_DIR,
            shake,
            hits,
            S5_XLABEL,
            OUT_DIR / "be_vs_s5_shakeoff_start.png",
            n_labels=len(hits),
        )
    )
    print(
        be_vs_screen_stop(
            BE_DIR,
            shake,
            hits,
            be_filtered,
            "S5",
            S5_XLABEL,
            OUT_DIR / "be_vs_s5_shakeoff_stop.png",
        )
    )
    print(s5_dropout_essential(counts, s1_genes))


if __name__ == "__main__":
    main()
