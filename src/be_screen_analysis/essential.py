"""Essential genes from control-null cutoffs on gene-level LFC distributions."""

from pathlib import Path

import numpy as np
import pandas as pd

from be_screen_analysis.filtering import (
    S1_DAY0,
    S1_NT,
    S1_REMAINING,
    S5_DAY0,
    S5_REMAINING,
    centred_guide_lfc,
    mageck_guide_lfc,
    nt_null,
)
from be_screen_analysis.mageck import AAVS1_GENE, control_guide_lfc

ALPHA = 0.0001  # cutoff = this quantile of the control pseudo-genes (per-gene false-positive rate)


def data_peak(x: pd.Series, step: float = 0.001) -> float:
    """Mode of a distribution from a Gaussian kernel density (Silverman bandwidth).

    The evaluation grid is anchored on the data, so translating the data translates the peak by
    exactly the same amount (the called set does not depend on how the LFCs are centred).
    """
    x = x.dropna().to_numpy()
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    bw = 0.9 * min(x.std(), iqr / 1.34) * len(x) ** -0.2
    lo, hi = np.percentile(x, [1, 99])
    grid = lo + np.arange(0, hi - lo, step)
    density = np.exp(-0.5 * ((grid[:, None] - x) / bw) ** 2).sum(axis=1)
    return float(grid[density.argmax()])


def shift_null(null: pd.Series, data: pd.Series) -> pd.Series:
    """Move the null so its median sits on the data's peak (centre it on the genes)."""
    return null + data_peak(data) - null.median()


def s1_nt_null(counts: pd.DataFrame, n: int, guides_per_gene: int = 4) -> pd.DataFrame:
    """S1 pseudo-genes in published MAGeCK units: means of random NT guide sets; columns D2-D5.

    NT guides with a blank or zero count in any of the samples are left out: a lost guide is a
    sequencing dropout (nontargeting_62_0 is blank only in day4_remaining, present before and
    after) whose LFC of ~-7 would otherwise set the tail of a mean-of-k null.
    """
    lfc = mageck_guide_lfc(counts, S1_DAY0, S1_REMAINING).rename(columns=S1_REMAINING)
    detected = (counts[[S1_DAY0, *S1_REMAINING]].fillna(0) > 0).all(axis=1)
    is_nt = (counts["gene"] == S1_NT) & detected
    return nt_null(lfc, is_nt, guides_per_gene=guides_per_gene, n=n, agg="mean")


def s5_intergenic_null(counts: pd.DataFrame, n: int, guides_per_gene: int = 2) -> pd.DataFrame:
    """S5 pseudo-genes: medians of random intergenic constructs, intergenic-centred CPM; col 78h."""
    is_ig = counts["category"] == "INTERGENIC_CONTROL"
    lfc = centred_guide_lfc(counts, S5_DAY0, S5_REMAINING, is_ig).rename(columns=S5_REMAINING)
    return nt_null(lfc, is_ig, guides_per_gene=guides_per_gene, n=n)


K562_DAY0, K562_UNTREATED_END = "k562.initial.attggc", "un.neg.tggtca"


def k562_control_null(counts: pd.DataFrame, n: int, guides_per_gene: int = 10) -> pd.DataFrame:
    """K562 pseudo-genes: medians of random CONTROL guides; column "untreated".

    Guide LFC is the uncentred log2 CPM ratio, untreated endpoint vs initial (pseudocount 0.5),
    the units of the published ``un_init.median.noctrl.neg.lfc`` (reproduced at r = 0.99).
    """
    cpm = counts[[K562_DAY0, K562_UNTREATED_END]].add(0.5)
    cpm = cpm / counts[[K562_DAY0, K562_UNTREATED_END]].sum() * 1e6
    lfc = np.log2(cpm[K562_UNTREATED_END] / cpm[K562_DAY0]).to_frame("untreated")
    return nt_null(lfc, counts["gene"] == "CONTROL", guides_per_gene=guides_per_gene, n=n)


def abe_control_null(
    run_dir: Path, n: int, guides_per_gene: int, gene: str = AAVS1_GENE
) -> pd.DataFrame:
    """ABE pseudo-genes: medians of random control guides (default AAVS1) from a MAGeCK run.

    ``run_dir`` is e.g. results/abe_mageck/start; one column per day (D4-D21), NT-normalised.
    """
    ctrl = control_guide_lfc(run_dir, gene)
    return nt_null(ctrl, pd.Series(True, index=ctrl.index), guides_per_gene=guides_per_gene, n=n)


def null_pvalue(values: pd.Series, null: pd.Series) -> pd.Series:
    """One-sided empirical p: fraction of null pseudo-genes at or below each value (+1 smoothed)."""
    ranked = np.searchsorted(np.sort(null.to_numpy()), values.to_numpy(), side="right")
    return pd.Series((ranked + 1) / (len(null) + 1), index=values.index).where(values.notna())


def bh_fdr(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (NaN kept out of the count)."""
    x = p.dropna().sort_values()
    q = x * len(x) / np.arange(1, len(x) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(upper=1)
    return q.reindex(p.index)
