import numpy as np
import pandas as pd

from be_screen_analysis.initiation import initiation_efficiency, ribotish_tis_counts, site_keys


def test_tis_counts_and_efficiency(tmp_path):
    # + strand GenomePos start is 0-based, - strand end is the 5' base (1-based)
    f = tmp_path / "c.tsv"
    f.write_text(
        "Gid\tGenomePos\tTISCounts\n"
        "G1\tchr1:99-500:+\t30\n"
        "G1\tchr1:99-800:+\t30\n"  # same start, other transcript
        "G2\tchr2:100-2000:-\t8\n"
    )
    tis = ribotish_tis_counts(f)
    assert len(tis) == 2
    sites = pd.DataFrame(
        {"Chromosome": [1, 2, 3], "strand": [1, -1, 1],
         "target_codon_range": ["100;102", "1998;2000", "5;7"]}
    )  # fmt: skip
    rna = pd.Series({"G1": 60, "G2": 0})
    eff = initiation_efficiency(site_keys(sites), tis, rna).to_numpy()
    assert eff[0] == 0.5 and np.isnan(eff[1]) and np.isnan(eff[2])
