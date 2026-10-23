import numpy as np
import pandas as pd

from be_screen_analysis.essential import bh_fdr, data_peak, null_pvalue, shift_null


def test_data_peak_finds_mode():
    rng = np.random.default_rng(0)
    x = pd.Series(np.concatenate([rng.normal(-0.3, 0.1, 5000), rng.normal(-1.5, 0.2, 300)]))
    assert abs(data_peak(x) - -0.3) < 0.03


def test_shift_null_moves_median_onto_data_peak():
    rng = np.random.default_rng(1)
    null = pd.Series(rng.normal(0, 0.1, 100_000))
    data = pd.Series(rng.normal(-0.5, 0.1, 5000))
    shifted = shift_null(null, data)
    assert np.isclose(shifted.median(), data_peak(data))
    assert np.allclose(shifted - null, data_peak(data) - null.median())


def test_null_pvalue_and_bh_fdr():
    null = pd.Series(np.arange(99, dtype=float))  # 0..98
    p = null_pvalue(pd.Series([-1.0, 0.0, 49.0, np.nan]), null)
    assert np.allclose(p[:3], [1 / 100, 2 / 100, 51 / 100]) and np.isnan(p[3])
    q = bh_fdr(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(q[:3], [0.03, 0.04, 0.04]) and np.isnan(q[3])
