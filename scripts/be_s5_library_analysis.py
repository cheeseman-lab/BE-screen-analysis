"""Guide-level candidate library for a follow-up screen: ABE start/stop x S5 (HeLa), no FDR.

Writes results/be_s5_library/be_s5_library_guides.csv, one row per ABE guide.
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import S5_CMPS, load_be_guides, s5_gene_table

ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "data" / "start_stop_ABE_screen_data" / "summaries"
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
OUT_DIR = ROOT / "results" / "be_s5_library"

MIN_EFF = 20


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Library candidates regardless of significance: no FDR filter on either screen. Each guide
    # is kept on its own if its sensor D0 or D12 pct_successful_target_codon_mutation >= MIN_EFF,
    # so a site keeps only its efficient guides. Sites come from the guide summaries, so the 474
    # efficient start sites missing from start_codon_summary.csv are included; their site_lfc /
    # site_neg_fdr / site_pos_fdr are blank, but guide_lfc / guide_fdr are present for every guide
    # and site_median_guide_lfc (median guide LFC over the site's kept guides) is filled for all.
    guides = pd.concat([load_be_guides(BE_DIR, s, min_eff=MIN_EFF) for s in ("start", "stop")])

    # S5 annotations for every non-control S5 gene (no FDR filter; S5 is genome-wide, so every BE
    # gene matches and the intersection removes nothing). Shake-off and dropout stats and
    # comparisons_passed (FDR < 0.05) are informational only. The ABE screen and S5 are both HeLa.
    # in_s1_essential marks genes in the S1 library; the dropout label is filled only for them.
    # Dropout label: construct CPM log2FC of 78h_remaining vs 0h, centred on the 600 intergenic
    # constructs, which cut DNA but disrupt no gene, so cutting toxicity cancels out (be_s5 uses
    # non-targeting controls instead). dropout_consistent: gene median below the 5th percentile
    # of a null of medians of 2 random intergenic constructs.
    s1_genes = set(pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv")["id"]) - {"nontargeting"}
    s5 = s5_gene_table(
        {c: pd.read_csv(S5_DIR / f) for c, f in S5_CMPS.items()},
        pd.read_csv(S5_DIR / "sgRNA_sequences_and_counts.csv"),
        s1_genes,
        control="intergenic",
    ).rename(columns={"dropout_lfc_78h": "dropout_lfc_78h_vs_intergenic"})
    # s5_dropout_lfc is MAGeCK's 78h-vs-0h LFC; s5_dropout_lfc_78h_vs_intergenic is the label's
    s5 = s5.rename(columns=lambda c: c if c in ("Hugo_Symbol", "cell_line") else f"s5_{c}")

    library = guides.merge(s5, on="Hugo_Symbol", how="inner")
    library.insert(1, "cell_line", library.pop("cell_line"))
    library.to_csv(OUT_DIR / "be_s5_library_guides.csv", index=False)

    unmatched = set(guides["Hugo_Symbol"]) - set(s5["Hugo_Symbol"])
    print(f"BE genes without an S5 match (dropped): {len(unmatched)}")
    for screen, g in library.groupby("screen"):
        print(
            f"{screen}: {len(g)} guides, {g['site_id'].nunique()} sites, "
            f"{g['Hugo_Symbol'].nunique()} genes "
            f"({g.loc[~g['in_codon_summary'], 'site_id'].nunique()} sites unscored)"
        )
    genes = library.drop_duplicates("Hugo_Symbol")
    ess = genes["s5_in_s1_essential"]
    print(
        f"Essential-subset genes: {ess.sum()} of {len(genes)}; "
        f"dropout_consistent (intergenic null): {genes.loc[ess, 's5_dropout_consistent'].sum()}"
    )
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
