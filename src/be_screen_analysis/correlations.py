"""Gene-level LFCs for correlating the ABE screen with CRISPR knockout screens (S1/S5, K562)."""

from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import (
    EFF_D0,
    EFF_D12,
    S5_DAY0,
    S5_REMAINING,
    centred_guide_lfc,
)
from be_screen_analysis.mageck import MAGECK_DAYS


def _gene_median_by_day(guides: pd.DataFrame, label: str) -> pd.DataFrame:
    """Median MAGeCK guide LFC per gene, one column per day ("{label} D4" ... "{label} D21")."""
    cols = {f"mageck_{d}_lfc": f"{label} {d}" for d in MAGECK_DAYS}
    return guides.groupby("Hugo_Symbol")[list(cols)].median().rename(columns=cols)


def abe_gene_lfc(
    mageck_dir: Path, screen: str, min_eff: float = 20, site_type: str = "Annotated"
) -> pd.DataFrame:
    """Median MAGeCK guide LFC per gene over efficient guides, one column per day vs D0.

    Reads ``{screen}_guide_summary_mageck.csv`` (scripts/abe_mageck_analysis.py); columns
    "ABE D4" ... "ABE D21". Start uses the guides of ``site_type`` starts only (default annotated:
    start loss as a knockout proxy; "Extended" / "Truncated" for the non-canonical starts); stop
    uses all guides.
    """
    g = pd.read_csv(mageck_dir / f"{screen}_guide_summary_mageck.csv")
    g = g[(g[EFF_D0] >= min_eff) | (g[EFF_D12] >= min_eff)]
    if screen == "start":
        g = g[g["start_site_type"] == site_type]
    return _gene_median_by_day(g, "ABE")


def inscreen_ko_gene_lfc(mageck_dir: Path) -> pd.DataFrame:
    """Median MAGeCK LFC per gene over the in-screen KO guides, one column per day vs D0.

    Pools the KO guides of the start and stop sub-libraries (same samples, both NT-normalised);
    columns "In-screen KO D4" ... "In-screen KO D21". No sensor efficiency exists for KO guides.
    """
    g = pd.concat(
        [pd.read_csv(mageck_dir / f"{s}_ko_guides_mageck.csv") for s in ("start", "stop")]
    )
    return _gene_median_by_day(g, "In-screen KO")


def s5_gene_dropout(counts: pd.DataFrame) -> pd.DataFrame:
    """S5 HeLa 78h remaining vs 0h LFC per gene, centred on intergenic (cutting) controls."""
    lfc = centred_guide_lfc(
        counts, S5_DAY0, S5_REMAINING, counts["category"] == "INTERGENIC_CONTROL"
    )
    targeting = counts["category"] == "GENE_TARGETING"
    gene = lfc[targeting].groupby(counts.loc[targeting, "target_symbol"]).median()
    gene = gene.drop("0", errors="ignore")  # constructs with no gene symbol
    return gene.rename(columns={c: f"S5 {d}" for c, d in S5_REMAINING.items()})


K562_UNTREATED = "un_init.median.noctrl.neg.lfc"


def k562_gene_lfc(median_lfc: pd.DataFrame) -> pd.Series:
    """K562 anti-mitotic screen (JCB 2024) untreated endpoint vs initial gene median LFC.

    ``median_lfc`` is data/anti-mt_K562/GW_median_lfc.csv; the untreated arm ends after 14
    population doublings (~ABE D12).
    """
    return median_lfc.set_index("id")[K562_UNTREATED].rename("K562 untreated")
