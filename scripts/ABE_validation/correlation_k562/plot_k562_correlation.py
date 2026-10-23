"""ABE start and in-screen KO D12 vs the K562 anti-mitotic screen untreated arm (JCB 2024, S1).

abe_d12_vs_k562_untreated_lfc.png  two panels, one point per gene, x = K562 untreated endpoint
                                   vs initial median gene log2FC (14 population doublings):
    left:  y = ABE annotated start D12 log2FC vs D0 (median MAGeCK guide LFC, >=20% editing)
    right: y = ABE in-screen KO D12 log2FC vs D0 (median MAGeCK LFC of the KO guides of the
           start and stop sub-libraries)
  Points coloured by the percentage of DepMap cell lines dependent on the gene
  (CRISPRGeneDependency, P > 0.5; gray: not in DepMap). Blue: least-squares fit. Side
  histograms: the control distribution of each axis (top: K562, medians of 10 random CONTROL
  guides in the published units; right: ABE, medians of 4 random AAVS1 guides at D12 from the
  matching MAGeCK run, start / start_ko).

abe_d12_vs_k562_untreated_k562_dependency.png  the same two panels with genes coloured by
                                   K562's own DepMap dependency probability (ModelID ACH-000551;
                                   gray: not in DepMap).

Run scripts/abe_mageck_analysis.py first; this reads results/abe_mageck/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from be_screen_analysis.correlations import abe_gene_lfc, inscreen_ko_gene_lfc, k562_gene_lfc
from be_screen_analysis.depmap import depmap_dependency, line_dependency
from be_screen_analysis.essential import abe_control_null, k562_control_null
from be_screen_analysis.plotting import control_margins, scatter_fit

ROOT = Path(__file__).resolve().parents[3]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
K562_DIR = ROOT / "data" / "anti-mt_K562"
DEPMAP_FILE = ROOT / "data" / "reference" / "CRISPRGeneDependency.csv"
OUT_DIR = ROOT / "figures" / "ABE_validation" / "correlation_k562"

N_CONTROL = 100_000  # control pseudo-genes
K562_MODEL = "ACH-000551"  # K562 DepMap ModelID (data/reference/Model.csv)
CMAP = "viridis"


def two_panel(
    k562: pd.Series,
    panels: dict,
    k562_null: pd.Series,
    color_by: pd.Series,
    vrange: tuple[float, float],
    cbar_label: str,
    name: str,
) -> Path:
    """ABE start and in-screen KO D12 (panels) vs K562 untreated, coloured by ``color_by``."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4), dpi=200)
    for ax, (label, (abe, run)) in zip(axes, panels.items()):
        scatter_fit(ax, k562, abe, color_by=color_by, cmap=CMAP, vrange=vrange,
                    missing_label="not in DepMap")  # fmt: skip
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        abe_null = abe_control_null(MAGECK_DIR / run, N_CONTROL, 4)["D12"]
        top = control_margins(ax, k562_null, abe_null, "K562 controls", "ABE controls\n(AAVS1)")
        ax.set_xlabel("K562 CRISPR KO untreated (14 PD), log2FC vs initial", fontsize=9)
        ax.set_ylabel(f"{label} D12, log2FC vs D0 (MAGeCK)", fontsize=9)
        top.set_title(f"{label} D12 vs K562 untreated", fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes((0.945, 0.2, 0.012, 0.5))
    fig.colorbar(ScalarMappable(Normalize(*vrange), CMAP), cax=cax, label=cbar_label)
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    k562 = k562_gene_lfc(pd.read_csv(K562_DIR / "GW_median_lfc.csv"))
    k562_null = k562_control_null(pd.read_csv(K562_DIR / "GW_counts.csv"), N_CONTROL)["untreated"]
    panels = {
        "ABE annotated start": (abe_gene_lfc(MAGECK_DIR, "start")["ABE D12"], "start"),
        "ABE in-screen KO": (inscreen_ko_gene_lfc(MAGECK_DIR)["In-screen KO D12"], "start_ko"),
    }
    pct = depmap_dependency(DEPMAP_FILE)["depmap_pct_dependent"]
    print(two_panel(k562, panels, k562_null, pct, (0, 100),
                    "DepMap cell lines dependent on the gene (%)",
                    "abe_d12_vs_k562_untreated_lfc.png"))  # fmt: skip
    print(two_panel(k562, panels, k562_null, line_dependency(DEPMAP_FILE, K562_MODEL), (0, 1),
                    "K562 dependency probability (DepMap)",
                    "abe_d12_vs_k562_untreated_k562_dependency.png"))  # fmt: skip


if __name__ == "__main__":
    main()
