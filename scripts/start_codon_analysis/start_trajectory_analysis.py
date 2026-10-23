"""Start codon ablation / downtuning: per-site dropout trajectories across the ABE timecourse.

Writes results/start_trajectories/:
  start_site_trajectories.csv  one row per start site: reference codon, edit types of its efficient
                               guides, median efficient-guide MAGeCK LFC at D0 (=0), D4, D6, D12,
                               D21, dropout rate, faster_than_{nt,aavs1}_null
  start_guide_trajectories.csv one row per efficient guide: site, site type, reference and edited
                               codon, MAGeCK LFC at D0 (=0), D4, D6, D12, D21
  null_trajectories.csv        control pseudo-site trajectories (D4-D21) and rates; control = nt
                               or aavs1

Run scripts/abe_mageck_analysis.py first (reads results/abe_mageck/).

Notes:
- Guides with >=20% sensor efficiency at D0 or D12 only; site value = median over its guides.
- Rate = least-squares slope of LFC vs day through the origin, so later days (D12, D21) weigh
  most; units log2FC per day.
- Two nulls, each from pseudo-sites of 2 random control guides (the median site size), same
  guides at every day, 20,000 draws: non-targeting guides (no binding, technical noise only) and
  AAVS1 guides (edit the AAVS1 safe-harbour locus, so they also carry any generic effect of
  editing genomic DNA). faster_than_{nt,aavs1}_null = rate below that null's 5th percentile.
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.mageck import control_guide_lfc
from be_screen_analysis.trajectories import (
    dropout_rate,
    guide_trajectories,
    null_trajectories,
    site_trajectories,
)

ROOT = Path(__file__).resolve().parents[2]
MAGECK_DIR = ROOT / "results" / "abe_mageck"
OUT_DIR = ROOT / "results" / "start_trajectories"
GUIDES_PER_SITE = 2
ALPHA = 0.05
NULLS = {"nt": "NT_CTRL", "aavs1": "AAVS1_CTRL"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guides = pd.read_csv(MAGECK_DIR / "start_guide_summary_mageck.csv")
    guide_trajectories(guides).to_csv(OUT_DIR / "start_guide_trajectories.csv")
    traj = site_trajectories(guides)
    traj["rate"] = dropout_rate(traj)
    nulls = []
    for name, gene in NULLS.items():
        null = null_trajectories(control_guide_lfc(MAGECK_DIR / "start", gene), GUIDES_PER_SITE)
        null["rate"] = dropout_rate(null)
        null.insert(0, "control", name)
        cutoff = null["rate"].quantile(ALPHA)
        traj[f"faster_than_{name}_null"] = traj["rate"] < cutoff
        print(f"{name} null 5th percentile rate {cutoff:+.4f} log2FC/day")
        nulls.append(null)
    null = pd.concat(nulls, ignore_index=True)
    traj.to_csv(OUT_DIR / "start_site_trajectories.csv")
    null.to_csv(OUT_DIR / "null_trajectories.csv", index=False)

    fast = [f"faster_than_{n}_null" for n in NULLS]
    print(f"{len(traj):,} start sites")
    print(traj.groupby("site_type")[fast].sum().assign(n=traj["site_type"].value_counts()))
    print(pd.crosstab(traj[fast[0]], traj[fast[1]]))


if __name__ == "__main__":
    main()
