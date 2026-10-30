"""OPS (aconcagua brieflow) screen: barcodes, guides and their feature tables.

Cells are assigned to guides by an in-situ read of the first PREFIX_LENGTH bases of the spacer
(brieflow ``barcode_type: simple``); ``cell_barcode_0`` in the feature tables is that prefix.
"""

from pathlib import Path

import pandas as pd

# copies of the brieflow config (barcode library, pool design) and aggregate feature TSVs; the
# single-cell parquet is a symlink to brieflow_output/aggregate/parquets/ on ops_analysis_hdd
OPS_DATA = Path(__file__).resolve().parents[2] / "data" / "OPS_aconcagua"
BARCODE_LIBRARY = OPS_DATA / "barcode_library.tsv"
POOL_DESIGN = OPS_DATA / "pool10_design.tsv"
FEATURE_PREFIX = "CeCl-all_ChCo-DAPI_TUBULIN_GH2AX_PHALLOIDIN__features"
CONSTRUCT_FEATURES = OPS_DATA / f"{FEATURE_PREFIX}_constructs.tsv"
GENE_FEATURES = OPS_DATA / f"{FEATURE_PREFIX}_genes.tsv"
SINGLECELL_FEATURES = OPS_DATA / f"{FEATURE_PREFIX}_singlecell.parquet"
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


def feature_columns(table: pd.DataFrame) -> list[str]:
    """The feature columns of an OPS table: every column after ``cell_count``."""
    return list(table.columns[table.columns.get_loc("cell_count") + 1 :])


def guide_features(constructs: pd.DataFrame, guides: pd.DataFrame) -> pd.DataFrame:
    """Guide labels joined to the per-guide feature medians, one row per featured barcode.

    ``constructs`` is the brieflow construct table (cell_barcode_0, gene_symbol_0, cell_count,
    features); ``guides`` is ops_guide_isoforms.csv. Columns: the guide columns, cell_count, then
    the features in construct-table order. Barcodes missing from either table are dropped; raises
    if a barcode's gene symbol differs between the two.
    """
    c = constructs.rename(columns={"cell_barcode_0": "barcode"})
    out = guides.merge(c, on="barcode")
    if (out["gene_symbol"] != out["gene_symbol_0"]).any():
        raise ValueError("gene symbols differ between the guide labels and the construct table")
    return out.drop(columns="gene_symbol_0").copy()  # copy consolidates the wide merge


def isoform_features(guide_table: pd.DataFrame, scaffold: pd.DataFrame) -> pd.DataFrame:
    """Per start isoform: the median feature value over the guides that knock it out.

    The isoform analogue of brieflow's gene table (median of construct medians). ``guide_table``
    is a ``guide_features`` table; ``scaffold`` is guide_classification's start_scaffold.csv. A
    targeting guide counts toward every start in its ``knocked_out`` on its gtf_gene (as in
    ``mageck.isoform_count_table``); guides knocking out nothing count toward no start.
    Nontargeting barcodes follow as one row each (gene = their gene_symbol, kind
    ``nontargeting``), as in brieflow. Columns: gene, start_id, kind (other start details: join
    start_scaffold.csv on gene, start_id), same_as_annotated (the start's guide set is its gene's
    annotated-start set), n_guides, barcodes (";"-joined), cell_count (summed), then the feature
    medians. Starts are in scaffold order.
    """
    feats = feature_columns(guide_table)
    t = guide_table[~guide_table["control"] & guide_table["gtf_gene"].notna()]
    ko = t["knocked_out"].fillna("").str.split(";")
    rows = [(i, f"{g}|{s}") for i, g, k in zip(t.index, t["gtf_gene"], ko) for s in k if s]
    idx, key = zip(*rows)
    expanded = t.loc[list(idx), ["barcode", "cell_count", *feats]].assign(key=key)
    grouped = expanded.groupby("key", sort=False)
    stats = pd.DataFrame(
        {
            "n_guides": grouped.size(),
            "barcodes": grouped["barcode"].agg(lambda b: ";".join(sorted(b))),
            "cell_count": grouped["cell_count"].sum(),
        }
    )
    meta_cols = ["gene", "start_id", "kind"]
    meta = scaffold[meta_cols].set_axis(scaffold["gene"] + "|" + scaffold["start_id"])
    meta = meta[meta.index.isin(stats.index)].copy()
    stats = stats.reindex(meta.index)
    annot = meta["kind"] == "Annotated"
    annot_sets = pd.Series(stats.loc[annot, "barcodes"].to_numpy(), index=meta.loc[annot, "gene"])
    meta["same_as_annotated"] = meta["gene"].map(annot_sets) == stats["barcodes"]
    starts = pd.concat([meta, stats, grouped[feats].median().reindex(meta.index)], axis=1)

    nt = guide_table[guide_table["control"]]
    controls = pd.DataFrame(
        {
            "gene": nt["gene_symbol"],
            "kind": "nontargeting",
            "n_guides": 1,
            "barcodes": nt["barcode"],
            "cell_count": nt["cell_count"],
        }
    )
    controls = pd.concat([controls, nt[feats]], axis=1)
    return pd.concat([starts, controls], ignore_index=True).reindex(columns=starts.columns)
