import pandas as pd

from be_screen_analysis.mageck import augment_guide_summary, mageck_count_table, read_mageck


def test_count_table_groups_by_site_unassigned_and_controls():
    counts = pd.DataFrame(
        {
            "Gene": ["NT_CTRL", "AAVS1_CTRL", "A_start", "A_start"],
            "Hugo_Symbol": [None, None, "A", "A"],
            "D0-rep1": [1, 2, 3, 4],
            "D0": [9, 9, 9, 9],
        },
        index=pd.Index(["n1", "a1", "g1", "g2"], name="sgRNA"),
    )
    guides = pd.DataFrame({"guide_id": ["g1"], "start_site_target_id": ["A-1_Annotated_ATG"]})
    out = mageck_count_table(counts, guides, "start")
    assert out["Gene"].tolist() == ["NT_CTRL", "AAVS1_CTRL", "A-1_Annotated_ATG", "A_unassigned"]
    assert list(out.columns) == ["Gene", "D0-rep1"]  # pooled day columns dropped
    ko = mageck_count_table(counts)  # no guide summary: grouped by gene
    assert ko["Gene"].tolist() == ["NT_CTRL", "AAVS1_CTRL", "A", "A"]


def test_read_mageck_and_augment(tmp_path):
    prefix = tmp_path / "D12_vs_D0"
    pd.DataFrame(
        {
            "sgrna": ["g1", "g2"],
            "LFC": [-2.0, 0.5],
            "p.low": [0.01, 0.9],
            "p.high": [0.99, 0.1],
            "p.twosided": [0.02, 0.2],
            "FDR": [0.04, 0.3],
        }
    ).to_csv(f"{prefix}.sgrna_summary.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "id": ["S1"],
            "neg|lfc": [-0.75],
            "neg|rank": [1],
            "neg|fdr": [0.05],
            "pos|rank": [1],
            "pos|fdr": [1.0],
        }
    ).to_csv(f"{prefix}.gene_summary.txt", sep="\t", index=False)
    g, s = read_mageck(prefix, "D12")
    assert g.loc["g1", "mageck_D12_rank"] == 1 and s.loc["S1", "mageck_D12_site_neg_fdr"] == 0.05

    guides = pd.DataFrame(
        {
            "guide_id": ["g1", "g2", "g3"],
            "stop_site_target_id": ["S1", "S1", "S2"],
            "lfc": [1, 2, 3],
        }
    )
    out = augment_guide_summary(guides, g, s, "stop")
    assert len(out) == 3 and out["lfc"].tolist() == [1, 2, 3]
    assert out.loc[0, "mageck_D12_lfc"] == -2.0 and pd.isna(out.loc[2, "mageck_D12_site_lfc"])
