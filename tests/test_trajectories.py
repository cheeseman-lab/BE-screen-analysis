import pandas as pd

from be_screen_analysis.filtering import EFF_D0, EFF_D12
from be_screen_analysis.trajectories import dropout_rate, site_trajectories


def test_site_trajectories_median_efficient_guides_and_rate():
    lfc = {"mageck_D4_lfc": [-0.4, -0.8, 9.0], "mageck_D6_lfc": [-0.6, -1.2, 9.0]}
    lfc |= {"mageck_D12_lfc": [-1.2, -2.4, 9.0], "mageck_D21_lfc": [-2.1, -4.2, 9.0]}
    guides = pd.DataFrame(
        {
            "start_site_target_id": ["A-1"] * 3,
            "start_site_type": ["Extended;Truncated"] * 3,
            "Hugo_Symbol": ["A"] * 3,
            EFF_D0: [50, 50, 5],  # third guide is inefficient and must be ignored
            EFF_D12: [50, 50, 5],
            **lfc,
        }
    )
    traj = site_trajectories(guides)
    assert traj.loc["A-1", "site_type"] == "Mixed" and traj.loc["A-1", "n_guides"] == 2
    assert traj.loc["A-1", "D0"] == 0 and abs(traj.loc["A-1", "D21"] + 3.15) < 1e-12
    # LFC = -0.15 * day exactly, so the slope through the origin is -0.15
    assert abs(dropout_rate(traj).loc["A-1"] + 0.15) < 1e-12
