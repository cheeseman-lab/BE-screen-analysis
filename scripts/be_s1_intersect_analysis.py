"""BE x S1 filtering of the ABE start/stop screen and the S1 HeLa shake-off screen.

The ABE screen was done in HeLa, so only the HeLa S1 comparisons are used (RPE1 is ignored).

Writes be_sites_filtered.csv to results/ (shared with be_s5) and two CSVs to results/be_s1/:
  1. be_sites_filtered.csv      sites with >=1 efficient guide and MAGeCK site FDR < 0.05
  2. s1_hela_filtered.csv       S1 HeLa genes with FDR < 0.05 in any shake-off/dropout comparison
  3. be_x_s1_intersection.csv   every BE site whose gene is in (2)
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import (
    filter_be_sites,
    filter_s1_hela,
    intersect,
    load_be_sites,
)

ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "data" / "start_stop_ABE_screen_data"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
OUT_DIR = ROOT / "results" / "be_s1"

MIN_EFF = 20
FDR = 0.05


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    be_all = pd.concat(
        [load_be_sites(BE_DIR, s, min_eff=MIN_EFF) for s in ("start", "stop")], ignore_index=True
    )
    # One row per unique codon site. Start sites are unstacked from the annotated/nc pairs in
    # start_codon_summary.csv and de-duplicated; paired_annot_site_id is the annotated start an
    # nc site was paired with. 477 start sites in start_guide_summary.csv have no row in the
    # codon summary, so they have no site FDR and are excluded. overlaps_other_site flags codons
    # sharing a base with another site (ZWILCH-1/ZWILCH-3).
    # Filters: >=1 guide with sensor D0 or D12 pct_successful_target_codon_mutation >= MIN_EFF,
    # and MAGeCK RRA site neg_fdr < FDR or pos_fdr < FDR (direction column says which).
    be = filter_be_sites(be_all, fdr=FDR)
    be.insert(1, "cell_line", "HeLa")

    # The ABE screen was done in HeLa, so only the HeLa S1 MAGeCK comparisons are used; RPE1 is
    # ignored. Keep genes with neg or pos FDR < FDR in any of dNs_dNr (day N shake-off vs
    # remaining) or dNr_d0 (day N remaining vs day 0), N = 2-5; comparisons_passed lists each
    # comparison:direction that passed.
    # Dropout label (annotation only, not a filter): essential-gene knockouts should deplete.
    # Guide CPM log2FC of each remaining timepoint vs Day0, centred on the non-targeting guides,
    # median per gene (dropout_lfc_D2..D5). dropout_consistent: D5 LFC below the 5th percentile
    # of a null of medians of 4 random NT guides. dropout_monotone: LFC never rises D2 -> D5.
    # MAGeCK's dNr_d0 LFCs are not used for the label: its median normalisation assumes most
    # guides are neutral, which hides the dropout in this mostly-essential library.
    s1 = filter_s1_hela(
        pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv"),
        pd.read_csv(S1_DIR / "mageck.pos_HeLa.csv"),
        pd.read_csv(S1_DIR / "HeLa_sgRNA_sequences_and_counts.csv"),
        fdr=FDR,
    )
    # S1 is gene-level, so every passing BE site (annotated, nc or stop) of a matching gene is
    # kept. Symbols are matched exactly; S1 uses some older HGNC symbols (e.g. ADPRHL2 for
    # ADPRS), so a few true matches can be missed.
    both = intersect(be, s1)

    be.to_csv(OUT_DIR.parent / "be_sites_filtered.csv", index=False)
    s1.to_csv(OUT_DIR / "s1_hela_filtered.csv", index=False)
    both.to_csv(OUT_DIR / "be_x_s1_intersection.csv", index=False)

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
