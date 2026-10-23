import numpy as np
import pandas as pd

from be_screen_analysis.mageck import site_stats, with_site_stats


def _guides() -> pd.DataFrame:
    site = {
        "mageck_D12_site_lfc": -2.0,
        "mageck_D12_site_neg_rank": 1,
        "mageck_D12_site_neg_fdr": 0.01,
        "mageck_D12_site_pos_rank": 9,
        "mageck_D12_site_pos_fdr": 1.0,
    }
    rows = [
        {"guide_id": "g1", "start_site_target_id": "A-1", "mageck_site_member": True, **site},
        {"guide_id": "g2", "start_site_target_id": "A-1", "mageck_site_member": True, **site},
        {"guide_id": "g3", "start_site_target_id": "A-1", "mageck_site_member": False, **site},
        {"guide_id": "g4", "start_site_target_id": "B-1", "mageck_site_member": False, **site},
    ]
    return pd.DataFrame(rows)


def test_site_stats_counts_only_tested_guides():
    stats = site_stats(_guides(), "start")
    assert list(stats.index) == ["A-1"]  # B-1 has no tested guide
    assert stats.loc["A-1", "num_guides"] == 2
    assert stats.loc["A-1", "lfc"] == -2.0
    assert stats.loc["A-1", "neg_fdr"] == 0.01


def test_with_site_stats_replaces_and_drops_stale_columns():
    summary = pd.DataFrame(
        {
            "annot_start_site_target_id": ["A-1"],
            "annot_lfc": [-9.0],
            "annot_neg_rra_score": [0.5],
            "nc_start_site_target_id": ["B-1"],
            "nc_lfc": [-9.0],
            "nc_pass_fdr_0.1_cutoff": [True],
            "depmap_pct_dependent": [50.0],
        }
    )
    out = with_site_stats(summary, site_stats(_guides(), "start"))
    assert out.loc[0, "annot_lfc"] == -2.0
    assert out.loc[0, "annot_num_guides"] == 2
    assert np.isnan(out.loc[0, "nc_lfc"])  # no efficient guide at the non-canonical start
    assert "annot_neg_rra_score" not in out and "nc_pass_fdr_0.1_cutoff" not in out
    assert out.loc[0, "depmap_pct_dependent"] == 50.0
