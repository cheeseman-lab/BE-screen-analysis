import numpy as np
import pandas as pd

from be_screen_analysis.filtering import (
    _start_sites,
    filter_be_sites,
    filter_s5,
    flag_overlaps,
    intersect,
    load_be_counts,
    load_be_guides,
    s5_control_ids,
    s5_dropout_labels,
)


def _start_summary():
    def side(prefix, ids, types, ranges, neg, pos):
        return {
            f"{prefix}_start_site_target_id": ids,
            f"{prefix}_start_site_type": types,
            f"{prefix}_reference_codon": ["ATG"] * len(ids),
            f"{prefix}_target_codon_range": ranges,
            f"{prefix}_num_guides": [2] * len(ids),
            f"{prefix}_lfc": [-1.0] * len(ids),
            f"{prefix}_neg_fdr": neg,
            f"{prefix}_pos_fdr": pos,
        }

    return pd.DataFrame(
        {
            "Hugo_Symbol": ["A", "A", "B"],
            "Chromosome": ["1", "1", "2"],
            "strand": [1, 1, -1],
            **side(
                "annot",
                ["A-1", "A-1", "B-1"],
                ["Annotated"] * 3,
                ["10;12", "10;12", "50;52"],
                [0.01, 0.01, 0.5],
                [1, 1, 1],
            ),
            **side(
                "nc",
                ["A-2", "A-3", "B-2"],
                ["Extended", "Truncated", "Extended"],
                ["20;22", "30;32", "60;62"],
                [0.5, 0.5, 1],
                [1, 1, 0.01],
            ),
        }
    )


def test_start_sites_dedups_annotated_and_keeps_pairing():
    sites = _start_sites(_start_summary()).set_index("site_id")
    assert sorted(sites.index) == ["A-1", "A-2", "A-3", "B-1", "B-2"]
    assert sites.loc["A-3", "paired_annot_site_id"] == "A-1"
    assert pd.isna(sites.loc["A-1", "paired_annot_site_id"])


def test_filter_requires_efficiency_and_either_direction_fdr():
    sites = pd.DataFrame(
        {
            "site_id": ["neg", "pos", "ns", "ineff"],
            "n_efficient_guides": [1, 2, 3, 0],
            "neg_fdr": [0.01, 0.9, 0.2, 0.001],
            "pos_fdr": [0.9, 0.04, 0.3, 0.9],
        }
    )
    out = filter_be_sites(sites).set_index("site_id")
    assert list(out.index) == ["neg", "pos"]
    assert out.loc["neg", "direction"] == "depleted"
    assert out.loc["pos", "direction"] == "enriched"


def test_flag_overlaps_shared_base_only():
    sites = pd.DataFrame(
        {
            "Chromosome": ["1", "1", "1", "2"],
            "target_codon_range": ["10;12", "12;14", "20;22", "12;14"],
        }
    )
    assert flag_overlaps(sites).tolist() == [True, True, False, False]


def test_intersect_keeps_all_sites_of_matching_gene():
    be = pd.DataFrame({"Hugo_Symbol": ["A", "A", "B"], "site_id": ["A-1", "A-2", "B-1"]})
    s1 = pd.DataFrame(
        {"Hugo_Symbol": ["A", "C"], "cell_line": "HeLa", "n_comparisons_passed": [1, 2]}
    )
    out = intersect(be, s1)
    assert out["site_id"].tolist() == ["A-1", "A-2"]
    assert "cell_line" not in out


def _s5_counts():
    # 4 NT constructs, 2 constructs each for genes A (essential, drops out) and B (not essential)
    return pd.DataFrame(
        {
            "target_symbol": ["NONTARGETING_CONTROL"] * 4 + ["A", "A", "B", "B", "OR1A1"],
            "category": ["NONTARGETING_CONTROL"] * 4 + ["GENE_TARGETING"] * 4 + ["OR_CONTROL"],
            "0h": [100, 100, 100, 100, 100, 100, 100, 100, 100],
            "78h_remaining": [100, 101, 99, 100, 10, 12, 100, 100, 100],
        }
    )


def _s5_table(neg, pos):
    ids = ["A", "B", "C", "NONTARGETING_CONTROL", "INTERGENIC_CONTROL_chr1", "OR1A1"]
    return pd.DataFrame({"id": ids, "neg|lfc": 0.0, "neg|fdr": neg, "pos|fdr": pos})


def test_s5_control_ids():
    ids = pd.Index(["A", "NONTARGETING_CONTROL", "INTERGENIC_CONTROL_chr1", "OR1A1"])
    assert s5_control_ids(ids, _s5_counts()).tolist() == [False, True, True, True]


def test_filter_s5_either_comparison_controls_dropped_label_only_essential():
    tables = {
        "shakeoff": _s5_table([0.9, 0.9, 0.9, 0.01, 0.01, 0.01], [0.9, 0.01, 0.9, 1, 1, 1]),
        "dropout": _s5_table([0.01, 0.9, 0.9, 0.01, 0.01, 0.01], [0.9, 0.9, 0.9, 1, 1, 1]),
    }
    out = filter_s5(tables, _s5_counts(), s1_genes={"A"}).set_index("Hugo_Symbol")
    assert sorted(out.index) == ["A", "B"]
    assert out.loc["A", "comparisons_passed"] == "dropout:neg"
    assert out.loc["B", "comparisons_passed"] == "shakeoff:pos"
    assert out.loc["A", "in_s1_essential"] and not out.loc["B", "in_s1_essential"]
    assert out.loc["A", "dropout_consistent"]
    assert pd.isna(out.loc["B", "dropout_consistent"])
    assert pd.isna(out.loc["B", "dropout_lfc_78h"])


def test_load_be_guides_filters_per_guide_and_blanks_unscored_sites(tmp_path):
    summary = _start_summary()
    summary.to_csv(tmp_path / "start_codon_summary.csv", index=False)
    eff0 = "sensor_mean_D0_pct_successful_target_codon_mutation"
    eff12 = "sensor_mean_D12_pct_successful_target_codon_mutation"
    guides = pd.DataFrame(
        {
            "Hugo_Symbol": ["A", "A", "A", "C"],
            "ensembl_transcript_id": "ENST1",
            "start_site_target_id": ["A-1", "A-1", "A-1", "C-1"],
            "start_site_type": ["Annotated"] * 4,
            "Chromosome": "1",
            "strand": 1,
            "reference_codon": "ATG",
            "mutated_codon": "ACG",
            "target_codon_range": ["10;12", "10;12", "10;12", "90;92"],
            "guide_id": ["g1", "g2", "g3", "g4"],
            "Protospacer": "ACGT",
            eff0: [10, 50, 5, 90],
            eff12: [30, 10, 5, 90],
            "lfc": [-1.0, -3.0, 9.0, 0.5],
            "fdr": [0.01, 0.01, 0.9, 0.5],
        }
    )
    guides.to_csv(tmp_path / "start_guide_summary.csv", index=False)
    out = load_be_guides(tmp_path, "start").set_index("guide_id")
    assert sorted(out.index) == ["g1", "g2", "g4"]  # g3 fails both D0 and D12
    assert out.loc["g1", "site_median_guide_lfc"] == -2.0  # median of kept guides only
    assert out.loc["g1", "in_codon_summary"] and out.loc["g1", "site_neg_fdr"] == 0.01
    assert not out.loc["g4", "in_codon_summary"] and pd.isna(out.loc["g4", "site_lfc"])


def test_s5_dropout_label_intergenic_null_absorbs_cutting_effect():
    # intergenic constructs drop like gene A; NT does not -> A is dropout vs NT but not intergenic
    n_ctrl = 6
    counts = pd.DataFrame(
        {
            "target_symbol": ["NONTARGETING_CONTROL"] * n_ctrl
            + ["INTERGENIC_CONTROL_chr1"] * n_ctrl
            + ["A", "A"],
            "category": ["NONTARGETING_CONTROL"] * n_ctrl
            + ["INTERGENIC_CONTROL"] * n_ctrl
            + ["GENE_TARGETING"] * 2,
            "0h": [100] * (2 * n_ctrl + 2),
            "78h_remaining": [100, 101, 99, 100, 102, 98] + [50, 51, 49, 50, 52, 48] + [50, 50],
        }
    )
    nt = s5_dropout_labels(counts, {"A"}, control="nontargeting")
    ig = s5_dropout_labels(counts, {"A"}, control="intergenic")
    assert nt.loc["A", "dropout_consistent"]
    assert not ig.loc["A", "dropout_consistent"]


def test_load_be_counts_categories_symbols_and_pooled_days(tmp_path):
    reps = {f"{d}-rep{r}": [1, 2, 3] for d in ("D0", "D4", "D6", "D12", "D21") for r in (1, 2, 3)}
    raw = pd.DataFrame(
        {"sgRNA": ["c1", "c2", "g1"], "Gene": ["AAVS1_CTRL", "NT_CTRL", "AAAS_KO"], **reps}
    )
    raw.to_csv(tmp_path / "ABE_start_KO_count_matrix_perturbation_level.csv", index=False)
    out = load_be_counts(tmp_path, "start", ko=True)
    assert out["category"].tolist() == ["AAVS1", "NT", "ko"]
    assert out["Hugo_Symbol"].isna().tolist() == [True, True, False]
    assert out.loc["g1", "Hugo_Symbol"] == "AAAS"
    assert out.loc["g1", "D12"] == 9  # three replicates of 3 summed


def test_mageck_guide_lfc_median_ratio_and_blank_counts():
    from be_screen_analysis.filtering import mageck_guide_lfc

    # day1 is day0 sequenced twice as deep, except guide d which was lost (blank count)
    counts = pd.DataFrame({"day0": [100, 200, 300, 400], "day1": [200, 400, 600, None]})
    lfc = mageck_guide_lfc(counts, "day0", {"day1": "D1"})["day1"]
    assert np.allclose(lfc[:3], 0)
    assert np.isclose(lfc[3], np.log2(1 / (400 * 2**0.5 + 1)))  # day0 size factor 2**-0.5


def test_depmap_dependency_counts_lines_above_half(tmp_path):
    from be_screen_analysis.depmap import depmap_dependency

    f = tmp_path / "dep.csv"
    pd.DataFrame(
        {"A (1)": [0.9, 0.6, 0.2, None], "B (2)": [0.1, 0.5, 0.4, 0.3]},
        index=["ACH-1", "ACH-2", "ACH-3", "ACH-4"],
    ).to_csv(f)
    t = depmap_dependency(f)
    assert t.loc["A", "depmap_fraction"] == "2/3" and t.loc["B", "depmap_n_dependent"] == 0
    assert np.isclose(t.loc["A", "depmap_pct_dependent"], 200 / 3)
