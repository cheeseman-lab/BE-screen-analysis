"""BE x S5 filtering of the ABE start/stop screen and the S5 genome-wide HeLa shake-off screen.

Reuses the filtered base editing sites from be_s1 (run scripts/be_s1_intersect_analysis.py first).

Writes two CSVs to results/be_s5/:
  1. s5_hela_filtered.csv       S5 genes with FDR < 0.05 in the shake-off or dropout comparison
  2. be_x_s5_intersection.csv   every filtered BE site whose gene is in (1)
"""

import sys
from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import S5_CMPS, filter_s5, intersect

ROOT = Path(__file__).resolve().parents[1]
BE_FILTERED = ROOT / "results" / "be_sites_filtered.csv"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
OUT_DIR = ROOT / "results" / "be_s5"

FDR = 0.05


def main() -> None:
    if not BE_FILTERED.exists():
        sys.exit(f"{BE_FILTERED} not found; run scripts/be_s1_intersect_analysis.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Same base editing sites as be_s1: >=1 guide >= 20% sensor efficiency at D0 or D12 and
    # MAGeCK site neg_fdr or pos_fdr < 0.05. The ABE screen was done in HeLa, as was S5.
    be = pd.read_csv(BE_FILTERED)

    # S5 is genome-wide (2 dual-guide constructs per gene). Keep genes with neg or pos FDR < FDR
    # in either comparison: shakeoff = 78h shake-off vs remaining, dropout = 78h remaining vs 0h.
    # The two attachment screens (unattached vs attached after trypsin) measure re-attachment,
    # a different phenotype, so they are not used. Non-targeting, intergenic and olfactory-
    # receptor control rows are removed before filtering.
    # in_s1_essential marks genes also in the S1 library (the essential subset of S5).
    # Dropout label (annotation only, not a filter), filled only for the essential subset:
    # construct CPM log2FC of 78h_remaining vs 0h, centred on the 200 non-targeting constructs,
    # median per gene (dropout_lfc_78h). dropout_consistent: below the 5th percentile of a null
    # of medians of 2 random NT constructs. NT-centring rather than MAGeCK's median
    # normalisation keeps the label consistent with be_s1. One timepoint, so no monotone column.
    s1_genes = set(pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv")["id"]) - {"nontargeting"}
    s5 = filter_s5(
        {c: pd.read_csv(S5_DIR / f) for c, f in S5_CMPS.items()},
        pd.read_csv(S5_DIR / "sgRNA_sequences_and_counts.csv"),
        s1_genes,
        fdr=FDR,
    )

    # S5 is gene-level, so every filtered BE site of a matching gene is kept. Symbols are matched
    # exactly; MAGeCK ids for the few S5 genes without a symbol are numeric and never match.
    both = intersect(be, s5)

    s5.to_csv(OUT_DIR / "s5_hela_filtered.csv", index=False)
    both.to_csv(OUT_DIR / "be_x_s5_intersection.csv", index=False)

    ess = s5["in_s1_essential"]
    print(
        f"S5 genes: {len(s5)} pass ({ess.sum()} in S1 essential subset, "
        f"{s5.loc[ess, 'dropout_consistent'].sum()} of those dropout_consistent)"
    )
    print(s5["comparisons_passed"].value_counts().to_string())
    print(
        f"Intersection: {len(both)} sites, {both['Hugo_Symbol'].nunique()} genes "
        f"({both.groupby('screen').size().to_dict()})"
    )
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
