"""OPS features per start isoform: brieflow's guide -> gene median step, redone per start.

brieflow's gene table is the median of a gene's construct (guide) medians. Here each start
isoform instead takes the median over the guides that knock it out (cut downstream of it), from
the same construct table (ops.CONSTRUCT_FEATURES).

Outputs in results/isoform_aware_OPS/:
  guide_features.parquet    ops.guide_features: one row per featured barcode, the guide labels
                            (ops_guide_isoforms.csv) then cell_count and the feature medians
  isoform_features.parquet  ops.isoform_features: one row per start knocked out by >= 1 guide,
                            then one per nontargeting barcode; start metadata, n_guides,
                            barcodes, cell_count, then the feature medians (other start
                            details: join start_scaffold.csv on gene, start_id)

Run classify_ops_guides.py first; run from the repo root:
  python scripts/isoform_aware_OPS/aggregate_isoform_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from be_screen_analysis.ops import (
    CONSTRUCT_FEATURES,
    GENE_FEATURES,
    feature_columns,
    guide_features,
    isoform_features,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "isoform_aware_OPS"
SCAFFOLD = ROOT / "results" / "guide_classification" / "start_scaffold.csv"


def check_reproduction(starts: pd.DataFrame, guides: pd.DataFrame) -> None:
    """Annotated starts carrying all of their gene's guides should equal brieflow's gene row."""
    genes = pd.read_csv(GENE_FEATURES, sep="\t").set_index("gene_symbol_0")
    t = guides[~guides.control]
    all_guides = t.groupby("gene_symbol").barcode.agg(lambda b: ";".join(sorted(b)))
    a = starts[starts.kind == "Annotated"].set_index("gene")
    full = a[a.barcodes == all_guides.reindex(a.index)]
    feats = feature_columns(genes)
    diff = (full[feats] - genes.loc[full.index, feats]).abs().to_numpy()
    print(
        f"annotated starts with all their gene's guides: {len(full)} of {len(a)}; "
        f"max |diff| vs brieflow gene table: {np.nanmax(diff):.2g}"
    )


def main() -> None:  # noqa: D103
    guides = pd.read_csv(OUT_DIR / "ops_guide_isoforms.csv")
    table = guide_features(pd.read_csv(CONSTRUCT_FEATURES, sep="\t"), guides)
    table.to_parquet(OUT_DIR / "guide_features.parquet", index=False)
    print(f"guide features: {table.shape} -> {OUT_DIR / 'guide_features.parquet'}")

    starts = isoform_features(table, pd.read_csv(SCAFFOLD))
    starts.to_parquet(OUT_DIR / "isoform_features.parquet", index=False)
    print(f"isoform features: {starts.shape} -> {OUT_DIR / 'isoform_features.parquet'}")

    print("rows by kind:", starts.kind.value_counts().to_dict())
    alt = starts[starts.kind.isin(["Truncated", "Internal"])]
    own = alt[~alt.same_as_annotated.astype(bool)]
    print(f"alternative starts: {len(alt)}; with their own guide set: {len(own)}")
    print(
        "  guides per distinct alternative start:",
        own.n_guides.value_counts().sort_index().to_dict(),
    )
    check_reproduction(starts, table)


if __name__ == "__main__":
    main()
