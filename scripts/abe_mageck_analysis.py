"""Rerun MAGeCK on the raw ABE counts for every timepoint and extend the guide summaries.

Writes results/abe_mageck/{start,stop}_guide_summary_mageck.csv (the original guide summary plus
mageck_{D4,D6,D12,D21}_* columns) and keeps the raw MAGeCK outputs in results/abe_mageck/{screen}/.
Also runs the in-screen gene knockout guides (ABE_{start,stop}_KO_* count files) grouped by gene:
results/abe_mageck/{start,stop}_ko_guides_mageck.csv (one row per KO guide, mageck_{day}_* guide
columns plus gene-level RRA columns).
Mode eff20 reruns the start screen with only efficient guides grouped by site (>= 20% sensor
editing at both D0 and D12; the others enter as {gene}_unassigned): results/abe_mageck/start_eff20/
and start_guide_summary_mageck_eff20.csv (mageck_site_member marks the guides in a site's test).
Run through Slurm: sbatch scripts/abe_mageck.sbatch [codon|ko|eff20|all]  (default all; all does
not include eff20)

Notes:
- Each day is tested against D0 (3 replicates each), normalised on the ~2,000 non-targeting
  guides (--norm-method control), so widespread dropout of targeting guides does not shift the
  scale. This reproduces the original guide summaries exactly at D12 (lfc, normalised counts;
  p/fdr to ~1e-3), so the summaries were made the same way.
- Guides are grouped by site (start/stop_site_target_id), so the site columns are MAGeCK RRA
  per site, like the codon summaries, and also cover sites missing from start_codon_summary.csv.
  Site lfc matches the codon summaries exactly; site FDRs are close but not identical (RRA
  permutation p-values and a different set of groups), and every site with codon-summary
  FDR < 0.05 is also < 0.05 here.
- ~100 count guides per library are not in the guide summary; they enter MAGeCK as
  {gene}_unassigned (so they still inform the variance model) but are not in the output tables.
- AAVS1 controls are in the run as one group (one test, like one site) but not used for
  normalisation. Tyler's run scored each AAVS1 guide as its own group and left out the unassigned
  guides (3,458 sites + 1,998 = 5,456 groups; set up that way, our ranks equal his exactly). Those
  1,998 extra tests are part of why his site FDRs are higher; the rest (his FDR ~1.36x ours with
  identical groups and ranks) is a MAGeCK version or setting difference in the FDR step.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from be_screen_analysis.filtering import EFF_D0, EFF_D12, load_be_counts
from be_screen_analysis.mageck import (
    MAGECK_DAYS,
    NT_GENE,
    augment_guide_summary,
    mageck_count_table,
    read_mageck,
    run_mageck_test,
)

ROOT = Path(__file__).resolve().parents[1]
BE_DIR = ROOT / "data" / "start_stop_ABE_screen_data"
OUT_DIR = ROOT / "results" / "abe_mageck"
MIN_EFF = 20  # eff20 mode: % sensor editing required at both D0 and D12


def _run_days(table: pd.DataFrame, work: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Write the count table and NT list to ``work``, run every day vs D0, return the results."""
    work.mkdir(parents=True, exist_ok=True)
    count_file, control_file = work / "counts.txt", work / "nt_sgrnas.txt"
    table.to_csv(count_file, sep="\t")
    nt_ids = table.index[table["Gene"] == NT_GENE]
    pd.Series(nt_ids).to_csv(control_file, index=False, header=False)
    per_guide, per_group, nt_medians = [], [], {}
    for day in MAGECK_DAYS:
        prefix = work / f"{day}_vs_D0"
        run_mageck_test(count_file, control_file, day, prefix)
        g, s = read_mageck(prefix, day)
        nt_medians[day] = round(g.loc[nt_ids, f"mageck_{day}_lfc"].median(), 3)
        per_guide.append(g)
        per_group.append(s)
    return pd.concat(per_guide, axis=1), pd.concat(per_group, axis=1), nt_medians


def run_screen(screen: str, min_eff: float | None = None) -> pd.DataFrame:
    guides = pd.read_csv(BE_DIR / "summaries" / f"{screen}_guide_summary.csv")
    grouped, tag = guides, ""
    if min_eff is not None:  # only efficient guides are grouped by site
        grouped = guides[(guides[EFF_D0] >= min_eff) & (guides[EFF_D12] >= min_eff)]
        tag = f"_eff{min_eff:g}"
    counts = load_be_counts(BE_DIR / "ABE_guide_counts", screen)
    table = mageck_count_table(counts, grouped, screen)
    per_guide, per_site, nt_medians = _run_days(table, OUT_DIR / f"{screen}{tag}")
    out = augment_guide_summary(guides, per_guide, per_site, screen)
    if min_eff is not None:
        out["mageck_site_member"] = out["guide_id"].isin(grouped["guide_id"])
        print(f"{screen}{tag}: {len(grouped):,} of {len(guides):,} guides grouped by site")
    out.to_csv(OUT_DIR / f"{screen}_guide_summary_mageck{tag}.csv", index=False)

    r = np.corrcoef(out["mageck_D12_lfc"], out["lfc"])[0, 1]
    offset = (out["mageck_D12_lfc"] - out["lfc"]).median()
    print(f"{screen}: {len(out):,} guides; D12 lfc vs original r = {r:.3f}, offset {offset:+.3f}")
    print(f"  NT guide median lfc per day: {nt_medians}")
    if screen == "start" and min_eff is None:
        scored = set(
            pd.read_csv(BE_DIR / "summaries" / "start_codon_summary.csv")
            .filter(like="start_site_target_id")
            .stack()
        )
        new = out.loc[out["mageck_D12_site_neg_fdr"].notna(), "start_site_target_id"]
        n_new = new[~new.isin(scored)].nunique()
        print(f"  sites scored here but absent from start_codon_summary: {n_new}")
    return out


def run_ko(screen: str) -> pd.DataFrame:
    # site columns from read_mageck are per gene here (guides are grouped by gene)
    counts = load_be_counts(BE_DIR / "ABE_guide_counts", screen, ko=True)
    per_guide, per_gene, nt_medians = _run_days(
        mageck_count_table(counts), OUT_DIR / f"{screen}_ko"
    )
    per_gene.columns = per_gene.columns.str.replace("_site_", "_gene_")
    ko = counts.loc[counts["category"] == "ko", ["Hugo_Symbol"]].rename_axis("guide_id")
    out = ko.join(per_guide).join(per_gene, on="Hugo_Symbol").reset_index()
    out.to_csv(OUT_DIR / f"{screen}_ko_guides_mageck.csv", index=False)
    print(f"{screen} KO: {len(out):,} guides, {out['Hugo_Symbol'].nunique():,} genes")
    print(f"  NT guide median lfc per day: {nt_medians}")
    return out


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "eff20":
        run_screen("start", MIN_EFF)
        return
    for screen in ("start", "stop"):
        if which in ("codon", "all"):
            run_screen(screen)
        if which in ("ko", "all"):
            run_ko(screen)


if __name__ == "__main__":
    main()
