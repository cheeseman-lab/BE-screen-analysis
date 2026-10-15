"""Rerun MAGeCK on the raw ABE counts for every timepoint and extend the guide summaries.

Writes results/abe_mageck/{start,stop}_guide_summary_mageck.csv (the original guide summary plus
mageck_{D4,D6,D12,D21}_* columns) and keeps the raw MAGeCK outputs in results/abe_mageck/{screen}/.
Run through Slurm: sbatch scripts/abe_mageck.sbatch

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
- AAVS1 controls are in the run but not used for normalisation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from be_screen_analysis.filtering import load_be_counts
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


def run_screen(screen: str) -> pd.DataFrame:
    work = OUT_DIR / screen
    work.mkdir(parents=True, exist_ok=True)
    guides = pd.read_csv(BE_DIR / "summaries" / f"{screen}_guide_summary.csv")
    table = mageck_count_table(load_be_counts(BE_DIR / "ABE_guide_counts", screen), guides, screen)
    count_file, control_file = work / "counts.txt", work / "nt_sgrnas.txt"
    table.to_csv(count_file, sep="\t")
    pd.Series(table.index[table["Gene"] == NT_GENE]).to_csv(control_file, index=False, header=False)

    per_guide, per_site, nt_medians = [], [], {}
    for day in MAGECK_DAYS:
        prefix = work / f"{day}_vs_D0"
        run_mageck_test(count_file, control_file, day, prefix)
        g, s = read_mageck(prefix, day)
        nt_medians[day] = g.loc[table.index[table["Gene"] == NT_GENE], f"mageck_{day}_lfc"].median()
        per_guide.append(g)
        per_site.append(s)
    out = augment_guide_summary(
        guides, pd.concat(per_guide, axis=1), pd.concat(per_site, axis=1), screen
    )
    out.to_csv(OUT_DIR / f"{screen}_guide_summary_mageck.csv", index=False)

    r = np.corrcoef(out["mageck_D12_lfc"], out["lfc"])[0, 1]
    offset = (out["mageck_D12_lfc"] - out["lfc"]).median()
    print(f"{screen}: {len(out):,} guides; D12 lfc vs original r = {r:.3f}, offset {offset:+.3f}")
    nt = {d: round(v, 3) for d, v in nt_medians.items()}
    print(f"  NT guide median lfc per day: {nt}")
    if screen == "start":
        scored = set(
            pd.read_csv(BE_DIR / "summaries" / "start_codon_summary.csv")
            .filter(like="start_site_target_id")
            .stack()
        )
        new = out.loc[out["mageck_D12_site_neg_fdr"].notna(), "start_site_target_id"]
        n_new = new[~new.isin(scored)].nunique()
        print(f"  sites scored here but absent from start_codon_summary: {n_new}")
    return out


def main() -> None:
    for screen in ("start", "stop"):
        run_screen(screen)


if __name__ == "__main__":
    main()
