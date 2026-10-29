"""Start isoforms knocked out / retained by each OPS (aconcagua) guide, keyed by its barcode.

The OPS library (pool10 dialout 0/1) is the S1 HeLa essential library: every one of its 20,445
spacers is an S1 guide. So each barcode is mapped to its 20-nt spacer (ops.barcode_spacers) and
takes that guide's labels from results/guide_classification/s1_guide_isoforms.csv.

Output: results/isoform_aware_OPS/ops_guide_isoforms.csv, one row per barcode: barcode (=
cell_barcode_0 in the OPS feature tables), gene_symbol (= gene_symbol_0), gene_id,
design_symbol, sequence, design, source, control (nontargeting), s1_sgrna, then the
guide_classification columns (gtf_gene, chrom, guide_strand, cut_pos, n_perfect_hits,
cut_region, in_maf, knocked_out, retained).

Run scripts/guide_classification first; run from the repo root:
  python scripts/isoform_aware_OPS/classify_ops_guides.py
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.ops import BARCODE_LIBRARY, CONSTRUCT_FEATURES, POOL_DESIGN, barcode_spacers

ROOT = Path(__file__).resolve().parents[2]
S1_LABELS = ROOT / "results" / "guide_classification" / "s1_guide_isoforms.csv"
OUT_DIR = ROOT / "results" / "isoform_aware_OPS"
LABEL_COLS = [
    "gtf_gene",
    "chrom",
    "guide_strand",
    "cut_pos",
    "n_perfect_hits",
    "cut_region",
    "in_maf",
    "knocked_out",
    "retained",
]


def main() -> None:  # noqa: D103
    if not S1_LABELS.exists():
        raise SystemExit(f"{S1_LABELS} missing: run scripts/guide_classification first")
    guides = barcode_spacers(
        pd.read_csv(BARCODE_LIBRARY, sep="\t"), pd.read_csv(POOL_DESIGN, sep="\t")
    )
    guides["control"] = guides["gene_symbol"].str.startswith("nontargeting")
    s1 = pd.read_csv(S1_LABELS).rename(columns={"sgrna": "s1_sgrna"})
    out = guides.merge(s1[["sequence", "s1_sgrna", *LABEL_COLS]], on="sequence", how="left")
    out = out.astype({"cut_pos": "Int64", "n_perfect_hits": "Int64"})

    print(f"barcodes: {len(out)}; matched to an S1 guide: {out.s1_sgrna.notna().sum()}")
    print("cut_region:", out.cut_region.value_counts(dropna=False).to_dict())
    ko = out.knocked_out.fillna("").str.split(";")
    alt = ko.map(lambda s: any(x and not x.startswith("Annotated") for x in s))
    print(f"guides knocking out a non-annotated start: {alt.sum()}")
    featured = set(pd.read_csv(CONSTRUCT_FEATURES, sep="\t", usecols=["cell_barcode_0"]).iloc[:, 0])
    absent = out.loc[~out.barcode.isin(featured), ["barcode", "gene_symbol"]]
    print(f"barcodes absent from the construct features: {len(absent)}", absent.values.tolist())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "ops_guide_isoforms.csv", index=False)
    print(f"-> {OUT_DIR / 'ops_guide_isoforms.csv'}")


if __name__ == "__main__":
    main()
