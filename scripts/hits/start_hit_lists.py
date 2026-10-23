"""Start site hit lists filtered by DepMap dependency, from efficient guides only.

Site statistics come from the MAGeCK rerun on efficient guides (>= 20% sensor editing at both D0
and D12; scripts/abe_mageck_analysis.py eff20): the annotated start codon summary with every
start's num_guides, lfc, neg/pos rank and FDR replaced from that run (D12) is written to
results/hits/start_codon_summary_eff20.csv (starts with no efficient guide: NaN, never a hit).
Hits are Tyler's start categories (notebooks/start_scatter.ipynb) with his FDR switch on: a start
is significant if neg or pos FDR < 0.1 (D12; no LFC cutoff). Each list keeps the matching rows of
that summary with all its columns, sorted by nc_lfc. Writes to results/hits/:
  nc_only_depmap90_starts.csv        non-canonical-only hits in genes >= 90% of DepMap cell lines
                                     depend on (depmap_pct_dependent)
  nonessential_depmap30_hits.csv     every hit (annotated only, non-canonical only, both) in genes
                                     < 30% of DepMap cell lines depend on (genes without DepMap
                                     data left out); hit_category column added

Run scripts/annotate_codon_summaries.py and scripts/abe_mageck_analysis.py eff20 first.
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import start_hit_category
from be_screen_analysis.mageck import site_stats, with_site_stats

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "annot_summaries" / "annot_start_codon_summary.csv"
EFF_GUIDES = ROOT / "results" / "abe_mageck" / "start_guide_summary_mageck_eff20.csv"
OUT_DIR = ROOT / "results" / "hits"

LFC_THRESH = None  # no LFC cutoff: FDR only
FDR_THRESH = 0.1
ESSENTIAL_MIN = 90  # % of DepMap cell lines dependent on the gene
NONESSENTIAL_MAX = 30


def save(rows: pd.DataFrame, name: str) -> None:
    out = OUT_DIR / name
    rows.sort_values("nc_lfc").to_csv(out, index=False)
    print(out, f"{len(rows)} rows, {rows['Hugo_Symbol'].nunique()} genes")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = site_stats(pd.read_csv(EFF_GUIDES), "start")
    df = with_site_stats(pd.read_csv(SUMMARY), stats)
    df.to_csv(OUT_DIR / "start_codon_summary_eff20.csv", index=False)
    lost = df[["annot_lfc", "nc_lfc"]].isna().any(axis=1).sum()
    print(f"summary rows {len(df)}, with a start lacking efficient guides: {lost}")
    cat = start_hit_category(df, LFC_THRESH, FDR_THRESH)
    pct = df["depmap_pct_dependent"]
    save(
        df[(cat == "nc_only") & (pct >= ESSENTIAL_MIN)], f"nc_only_depmap{ESSENTIAL_MIN}_starts.csv"
    )
    hits = df.assign(hit_category=cat)[(cat != "neither") & (pct < NONESSENTIAL_MAX)]
    save(hits, f"nonessential_depmap{NONESSENTIAL_MAX}_hits.csv")
    print(pd.crosstab(hits["hit_category"], hits["nc_start_site_type"]))


if __name__ == "__main__":
    main()
