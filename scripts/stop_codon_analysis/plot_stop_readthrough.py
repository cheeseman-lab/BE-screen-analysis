"""Stop codon site LFC vs readthrough length (distance to the next in-frame stop).

stop_lfc_vs_next_stop.png  one point per stop site: x = amino acids added by readthrough (the
                           edited stop's residue up to the next in-frame stop; linear scale; read
                           through the 3' UTR and past the transcript end if needed), y = site
                           log2FC from the stop codon summary (MAGeCK RRA, D12 vs D0). Coloured
                           by the site's edit (mutated_codon of its guides): stop-removing (gives
                           an amino acid), stop-keeping (TAA->TAG/TGA, still a stop), or both
                           when its guides differ; black line: least-squares fit of LFC on amino
                           acids added, over all sites.

Run scripts/annotate_codon_summaries.py first; this reads results/annot_summaries/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from be_screen_analysis.plotting import BLUE, GRAY, ORANGE, lfc_vs_length

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "annot_summaries" / "annot_stop_codon_summary.csv"
OUT_DIR = ROOT / "figures" / "stop_codon_analysis"

GROUPS = {
    "stop-removing edit": BLUE,
    "stop-keeping edit": GRAY,
    "both (guides differ)": ORANGE,
}


def lfc_vs_next_stop(sites: pd.DataFrame) -> Path:
    """Scatter of site LFC against readthrough length with a least-squares fit."""
    residues = sites["edited_stop_residue"].str.split(";")
    keeps = residues.map(lambda r: "*" in r)
    removes = residues.map(lambda r: any(x != "*" for x in r))
    removing, keeping, both = GROUPS
    group = pd.Series(
        np.select([keeps & removes, keeps], [both, keeping], removing), index=sites.index
    )
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    rho = lfc_vs_length(ax, sites["next_stop_dist_aa"], sites["lfc"], group, GROUPS)
    ax.set_xlabel("Amino acids added by readthrough (to the next in-frame stop)", fontsize=9)
    ax.set_ylabel("Stop site log2FC, D12 vs D0", fontsize=9)
    ax.set_title(f"Stop codon sites: dropout vs readthrough length (all sites ρ = {rho:+.2f})",
                 loc="left", fontsize=10)  # fmt: skip
    fig.tight_layout()
    out = OUT_DIR / "stop_lfc_vs_next_stop.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(lfc_vs_next_stop(pd.read_csv(SUMMARY)))


if __name__ == "__main__":
    main()
