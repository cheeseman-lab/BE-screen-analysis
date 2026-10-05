"""Preliminary filtering of the ABE start/stop screen and the S1 HeLa shake-off screen.

Writes three CSVs (plus a README for each) to results/preliminary/:
  1. be_sites_filtered.csv      sites with >=1 efficient guide and MAGeCK site FDR < 0.05
  2. s1_hela_filtered.csv       S1 HeLa genes with FDR < 0.05 in any shake-off/dropout comparison
  3. be_x_s1_intersection.csv   every BE site whose gene is in (2)
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import (
    S1_CMPS,
    filter_be_sites,
    filter_s1_hela,
    intersect,
    load_be_sites,
)

ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "data" / "start_stop_ABE_screen_data"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
OUT_DIR = ROOT / "results" / "preliminary"

MIN_EFF = 20
FDR = 0.05

HELA_NOTE = (
    "The ABE base editing screen was performed in HeLa, so only the HeLa S1 MAGeCK comparisons "
    "are used; the RPE1 comparisons are ignored."
)
BE_README = f"""\
be_sites_filtered.csv - ABE start/stop codon screen (HeLa), one row per unique codon site.

Inputs: data/start_stop_ABE_screen_data/{{start,stop}}_{{codon,guide}}_summary.csv
Filters:
  - efficiency: >=1 guide with sensor_mean_D0 or sensor_mean_D12
    pct_successful_target_codon_mutation >= {MIN_EFF}%
  - significance: MAGeCK RRA site-level neg_fdr < {FDR} OR pos_fdr < {FDR} (either direction)
Notes:
  - Start sites come from start_codon_summary.csv, unstacked from annotated/nc pairs and
    de-duplicated, giving one row per annotated or nc site. paired_annot_site_id gives the
    annotated start that an nc site was paired with.
  - 477 start sites present in start_guide_summary.csv have no row in start_codon_summary.csv,
    so they have no site FDR and are excluded.
  - n_efficient_guides, max_eff_D0/D12 and ensembl_transcript_id come from the guide summary.
  - direction: depleted (neg_fdr passes) or enriched (pos_fdr passes).
  - overlaps_other_site flags codons that share a base with another site (ZWILCH-1/ZWILCH-3).
"""
S1_README = f"""\
s1_hela_filtered.csv - S1 shake-off screen, HeLa only, one row per gene.

{HELA_NOTE}
Inputs: data/shakeoff_data/table_s1/mageck.{{neg,pos}}_HeLa.csv, HeLa_sgRNA_sequences_and_counts.csv
Filter: neg FDR < {FDR} OR pos FDR < {FDR} in any of: {", ".join(S1_CMPS)}
  (dNs_dNr = day N shake-off vs remaining; dNr_d0 = day N remaining vs day 0).
  comparisons_passed lists each comparison:direction that passed.
Dropout label (annotation only, NOT a filter): essential-gene knockouts are expected to deplete.
  Guide CPM log2FC of each remaining timepoint vs Day0, centred on the non-targeting guides; the
  gene value is the median of its guides (dropout_lfc_D2..D5).
  - dropout_consistent: D5 LFC below the 5th percentile of a null from medians of 4 random
    non-targeting guides, i.e. the gene drops out as expected.
  - dropout_monotone: LFC never increases from D2 to D5.
  MAGeCK's dNr_d0 LFCs are not used for the label: its median normalisation assumes most guides
  are neutral, which hides the dropout in this mostly-essential library.
"""
X_README = f"""\
be_x_s1_intersection.csv - rows of be_sites_filtered.csv whose Hugo_Symbol is in
s1_hela_filtered.csv, joined with the S1 gene columns. S1 is gene-level, so every passing site
(annotated, nc or stop) of a matching gene is kept.

{HELA_NOTE}
Gene symbols are matched exactly. S1 uses some older HGNC symbols (e.g. ADPRHL2 for ADPRS), so a
few true matches can be missed; HGNC previous-symbol mapping is not applied.
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    be_all = pd.concat(
        [load_be_sites(BE_DIR, s, min_eff=MIN_EFF) for s in ("start", "stop")], ignore_index=True
    )
    be = filter_be_sites(be_all, fdr=FDR)
    be.insert(1, "cell_line", "HeLa")

    s1 = filter_s1_hela(
        pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv"),
        pd.read_csv(S1_DIR / "mageck.pos_HeLa.csv"),
        pd.read_csv(S1_DIR / "HeLa_sgRNA_sequences_and_counts.csv"),
        fdr=FDR,
    )
    both = intersect(be, s1)

    for name, df, readme in [
        ("be_sites_filtered", be, BE_README),
        ("s1_hela_filtered", s1, S1_README),
        ("be_x_s1_intersection", both, X_README),
    ]:
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        (OUT_DIR / f"{name}_README.txt").write_text(readme)

    print("BE sites before -> after:")
    for screen, n in be_all.groupby("screen").size().items():
        kept = be[be["screen"] == screen]
        print(f"  {screen}: {n} -> {len(kept)} sites ({kept['Hugo_Symbol'].nunique()} genes)")
    print(f"S1 HeLa genes: {len(s1)} pass ({s1['dropout_consistent'].sum()} dropout_consistent)")
    print(
        f"Intersection: {len(both)} sites, {both['Hugo_Symbol'].nunique()} genes "
        f"({both.groupby('screen').size().to_dict()})"
    )
    unmatched = sorted(set(be["Hugo_Symbol"]) - set(s1["Hugo_Symbol"]))
    print(f"BE-filtered genes not in S1-filtered: {len(unmatched)}")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
