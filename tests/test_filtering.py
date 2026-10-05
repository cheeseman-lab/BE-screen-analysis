import pandas as pd

from be_screen_analysis.filtering import (
    _start_sites,
    filter_be_sites,
    flag_overlaps,
    intersect,
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
