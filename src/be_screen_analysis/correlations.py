"""Gene-level LFCs for correlating the ABE screen with the S1/S5 CRISPR knockout screens."""

from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import (
    EFF_D0,
    EFF_D12,
    S1_NT,
    S1_REMAINING,
    S5_DAY0,
    S5_REMAINING,
    centred_guide_lfc,
    nt_centred_guide_lfc,
)
from be_screen_analysis.mageck import MAGECK_DAYS


def _gene_median_by_day(guides: pd.DataFrame, label: str) -> pd.DataFrame:
    """Median MAGeCK guide LFC per gene, one column per day ("{label} D4" ... "{label} D21")."""
    cols = {f"mageck_{d}_lfc": f"{label} {d}" for d in MAGECK_DAYS}
    return guides.groupby("Hugo_Symbol")[list(cols)].median().rename(columns=cols)


def abe_gene_lfc(mageck_dir: Path, screen: str, min_eff: float = 20) -> pd.DataFrame:
    """Median MAGeCK guide LFC per gene over efficient guides, one column per day vs D0.

    Reads ``{screen}_guide_summary_mageck.csv`` (scripts/abe_mageck_analysis.py); columns
    "ABE D4" ... "ABE D21". Start uses annotated-ATG guides only (start loss as a knockout proxy);
    stop uses all guides.
    """
    g = pd.read_csv(mageck_dir / f"{screen}_guide_summary_mageck.csv")
    g = g[(g[EFF_D0] >= min_eff) | (g[EFF_D12] >= min_eff)]
    if screen == "start":
        g = g[g["start_site_type"] == "Annotated"]
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


def s1_gene_dropout(counts: pd.DataFrame) -> pd.DataFrame:
    """S1 HeLa remaining vs Day0 LFC per gene and timepoint (D2-D5), centred on NT guides."""
    lfc = nt_centred_guide_lfc(counts)
    is_nt = counts["gene"] == S1_NT
    gene = lfc[~is_nt].groupby(counts.loc[~is_nt, "gene"]).median()
    return gene.rename(columns={c: f"S1 {d}" for c, d in S1_REMAINING.items()})


def s5_gene_dropout(counts: pd.DataFrame) -> pd.DataFrame:
    """S5 HeLa 78h remaining vs 0h LFC per gene, centred on intergenic (cutting) controls."""
    lfc = centred_guide_lfc(
        counts, S5_DAY0, S5_REMAINING, counts["category"] == "INTERGENIC_CONTROL"
    )
    targeting = counts["category"] == "GENE_TARGETING"
    gene = lfc[targeting].groupby(counts.loc[targeting, "target_symbol"]).median()
    gene = gene.drop("0", errors="ignore")  # constructs with no gene symbol
    return gene.rename(columns={c: f"S5 {d}" for c, d in S5_REMAINING.items()})
