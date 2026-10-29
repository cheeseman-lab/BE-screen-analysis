"""OPS (aconcagua brieflow) screen: barcodes, guides and their feature tables.

Cells are assigned to guides by an in-situ read of the first PREFIX_LENGTH bases of the spacer
(brieflow ``barcode_type: simple``); ``cell_barcode_0`` in the feature tables is that prefix.
"""

from pathlib import Path

import pandas as pd

OPS_DIR = Path("/lab/barcheese01/ating/aconcagua-analysis/analysis")
OPS_CONFIG = OPS_DIR / "config"
BARCODE_LIBRARY = OPS_CONFIG / "barcode_library.tsv"
POOL_DESIGN = OPS_CONFIG / "pool10_design.tsv"
# brieflow aggregate outputs (read-only): single-cell parquets and per-guide / per-gene TSVs
AGGREGATE_DIR = Path(
    "/lab/ops_analysis_hdd/cheeseman/aconcagua-analysis/analysis/brieflow_output/aggregate"
)
FEATURE_PREFIX = "CeCl-all_ChCo-DAPI_TUBULIN_GH2AX_PHALLOIDIN__features"
CONSTRUCT_FEATURES = AGGREGATE_DIR / "tsvs" / f"{FEATURE_PREFIX}_constructs.tsv"
GENE_FEATURES = AGGREGATE_DIR / "tsvs" / f"{FEATURE_PREFIX}_genes.tsv"
SINGLECELL_FEATURES = AGGREGATE_DIR / "parquets" / f"{FEATURE_PREFIX}_singlecell.parquet"
PREFIX_LENGTH = 11  # SBS cycles read in situ
OPS_DIALOUTS = (0, 1)  # sub-libraries of the pool brieflow kept (create_dialout_filter([0, 1]))


def barcode_spacers(library: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """One row per library barcode with the full spacer it reads, as brieflow built the library.

    ``library`` is barcode_library.tsv (prefix, gene_symbol, gene_id); ``design`` is the pool
    design (sgRNA, gene_symbol, dialout, design, source). Design rows outside OPS_DIALOUTS are
    dropped, duplicate oligos collapsed, and each spacer's prefix is its first PREFIX_LENGTH
    bases. Columns: barcode, gene_symbol (the OPS symbol, ``gene_symbol_0`` in the feature
    tables), gene_id, design_symbol, sequence, design, source. Raises if a prefix reads more than
    one spacer or a barcode matches none.
    """
    d = design[design["dialout"].isin(OPS_DIALOUTS)].drop_duplicates("sgRNA")
    d = d.assign(barcode=d["sgRNA"].str[:PREFIX_LENGTH])
    shared = d["barcode"].duplicated(keep=False)
    if shared.any():
        raise ValueError(f"{d.loc[shared, 'barcode'].nunique()} prefixes read several spacers")
    d = d.rename(columns={"sgRNA": "sequence", "gene_symbol": "design_symbol"})
    out = library.rename(columns={"prefix": "barcode"})[["barcode", "gene_symbol", "gene_id"]]
    out = out.merge(d[["barcode", "design_symbol", "sequence", "design", "source"]], how="left")
    missing = out["sequence"].isna()
    if missing.any():
        raise ValueError(f"{missing.sum()} barcodes match no spacer in the design")
    return out
