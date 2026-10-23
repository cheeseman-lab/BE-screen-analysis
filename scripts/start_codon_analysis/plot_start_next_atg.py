"""Start codon site LFC vs distance to the next in-frame ATG.

start_lfc_vs_next_atg.png  one point per start site, one panel for annotated starts and one for
                           the non-canonical starts (extended / truncated): x = amino acids from
                           the start to the next in-frame ATG downstream (the N-terminus lost if
                           the start is edited away and translation begins at that ATG; linear
                           scale), y = site log2FC from the start codon summary (MAGeCK RRA, D12
                           vs D0). Sites with no in-frame ATG before the stop are left out (n in
                           the panel title). Least-squares fits: annotated, one over all sites
                           (black); non-canonical, one each for extended and truncated (lines in
                           their colours).

Run scripts/annotate_codon_summaries.py first; this reads results/annot_summaries/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from be_screen_analysis.plotting import BLUE, GRAY, ORANGE, lfc_vs_length

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "annot_summaries" / "annot_start_codon_summary.csv"
OUT_DIR = ROOT / "figures" / "start_codon_analysis"

PANELS = {
    "annot_": ("Annotated starts", {"Annotated": GRAY}),
    "nc_": ("Non-canonical starts", {"Extended": ORANGE, "Truncated": BLUE}),
}


def lfc_vs_next_atg(summary: pd.DataFrame) -> Path:
    """One panel per start class: site LFC against amino acids to the next in-frame ATG."""
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 5), dpi=200, sharey=True)
    for ax, (p, (title, colors)) in zip(axes, PANELS.items()):
        key = f"{p}start_site_target_id"
        sites = summary.drop_duplicates(key).dropna(subset=[key])
        dist = sites[f"{p}next_atg_dist_aa"]
        rho = lfc_vs_length(
            ax,
            dist,
            sites[f"{p}lfc"],
            sites[f"{p}start_site_type"],
            colors,
            fit_per_group=p == "nc_",
        )
        ax.set_xlabel("Amino acids to the next in-frame ATG (N-terminus lost)", fontsize=9)
        ax.set_title(
            f"{title}: dropout vs distance to next ATG (ρ = {rho:+.2f})\n"
            f"{dist.isna().sum():,} of {len(sites):,} sites with no in-frame ATG before the stop "
            "not shown",
            loc="left",
            fontsize=10,
        )
    axes[0].set_ylabel("Start site log2FC, D12 vs D0", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "start_lfc_vs_next_atg.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(lfc_vs_next_atg(pd.read_csv(SUMMARY)))


if __name__ == "__main__":
    main()
