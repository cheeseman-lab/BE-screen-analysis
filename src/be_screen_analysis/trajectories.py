"""Per-site dropout trajectories across the ABE timecourse (start codon ablation / downtuning)."""

import numpy as np
import pandas as pd

from be_screen_analysis.filtering import EFF_D0, EFF_D12, nt_null
from be_screen_analysis.mageck import MAGECK_DAYS

DAYS = {"D0": 0, **{d: int(d[1:]) for d in MAGECK_DAYS}}


def site_trajectories(guides: pd.DataFrame, min_eff: float = 20) -> pd.DataFrame:
    """Median MAGeCK guide LFC per start site and day over efficient guides; D0 = 0.

    ``guides`` is ``start_guide_summary_mageck.csv``. Mixed site types (e.g. "Extended;Truncated")
    are labelled "Mixed". ``edit_types`` lists the edited codons of the site's efficient guides
    (e.g. "ACG;GTG" for an ATG start with both ATG->ACG and ATG->GTG guides).
    """
    g = guides[(guides[EFF_D0] >= min_eff) | (guides[EFF_D12] >= min_eff)]
    by_site = g.groupby("start_site_target_id")
    traj = by_site[[f"mageck_{d}_lfc" for d in MAGECK_DAYS]].median()
    traj.columns = MAGECK_DAYS
    traj.insert(0, "D0", 0.0)
    site_type = by_site["start_site_type"].first()
    info = pd.DataFrame(
        {
            "Hugo_Symbol": by_site["Hugo_Symbol"].first(),
            "site_type": site_type.where(~site_type.str.contains(";"), "Mixed"),
            "n_guides": by_site.size(),
            "reference_codon": by_site["reference_codon"].first(),
            "edit_types": by_site["mutated_codon"].agg(lambda c: ";".join(sorted(set(c)))),
        }
    )
    return info.join(traj).rename_axis("site_id")


def guide_trajectories(guides: pd.DataFrame, min_eff: float = 20) -> pd.DataFrame:
    """MAGeCK LFC per efficient guide and day, D0 = 0, with its site type and codon edit.

    ``guides`` is ``start_guide_summary_mageck.csv``; mixed site types are labelled "Mixed".
    """
    g = guides[(guides[EFF_D0] >= min_eff) | (guides[EFF_D12] >= min_eff)]
    traj = g[[f"mageck_{d}_lfc" for d in MAGECK_DAYS]].set_axis(MAGECK_DAYS, axis=1)
    traj.insert(0, "D0", 0.0)
    site_type = g["start_site_type"].where(~g["start_site_type"].str.contains(";"), "Mixed")
    info = pd.DataFrame(
        {
            "site_id": g["start_site_target_id"],
            "site_type": site_type,
            "reference_codon": g["reference_codon"],
            "mutated_codon": g["mutated_codon"],
        }
    )
    return info.join(traj).set_index(g["guide_id"].rename("guide_id"))


def dropout_rate(traj: pd.DataFrame) -> pd.Series:
    """Least-squares slope through the origin of LFC vs day (log2FC per day; < 0 = depleting)."""
    t = np.array([DAYS[d] for d in MAGECK_DAYS], dtype=float)
    return (traj[MAGECK_DAYS] @ t / (t @ t)).rename("rate")


def null_trajectories(
    control_lfc: pd.DataFrame, guides_per_site: int, n: int = 20000
) -> pd.DataFrame:
    """Trajectories of pseudo-sites built from random sets of control guides.

    ``control_lfc`` has one column per day (MAGECK_DAYS); each draw uses the same guides at every
    day.
    """
    return nt_null(control_lfc, pd.Series(True, index=control_lfc.index), guides_per_site, n)
