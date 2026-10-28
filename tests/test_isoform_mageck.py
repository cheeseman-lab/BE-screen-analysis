import pandas as pd

from be_screen_analysis.filtering import start_hit_category
from be_screen_analysis.mageck import isoform_count_table


def test_isoform_count_table_expands_by_knocked_out_start():
    ids = ["A_1", "A_2", "B_1", "NONTARGETING_CONTROL_1"]
    counts = pd.DataFrame({"0h": [10, 20, 30, 40], "78h": [1, 2, 3, 4]}, index=ids)
    labels = pd.DataFrame(
        {
            "category": ["GENE_TARGETING"] * 3 + ["NONTARGETING_CONTROL"],
            "target_symbol": ["A", "A", "B", "NONTARGETING_CONTROL"],
            "construct_gene": ["A", "A", "B", None],
            "knocked_out": ["Annotated_s1;Truncated_s2", "Annotated_s1", None, None],
        },
        index=ids,
    )
    out = isoform_count_table(counts, labels)
    assert out.index.tolist() == [
        "A_1|Annotated_s1",
        "A_1|Truncated_s2",
        "A_2|Annotated_s1",
        "B_1",
        "NONTARGETING_CONTROL_1",
    ]
    assert out["Gene"].tolist() == [
        "A|Annotated_s1",
        "A|Truncated_s2",
        "A|Annotated_s1",
        "B_no_start_ko",
        "NONTARGETING_CONTROL",
    ]
    assert out["0h"].tolist() == [10, 10, 20, 30, 40]  # counts copied unchanged
    assert list(out.columns) == ["Gene", "0h", "78h"]


def test_start_hit_category_other_prefix():
    df = pd.DataFrame(
        {
            "annot_lfc": [-2, 0],
            "annot_neg_fdr": [0.01, 0.9],
            "annot_pos_fdr": [1, 1],
            "alt_lfc": [0, -2],
            "alt_neg_fdr": [0.9, 0.01],
            "alt_pos_fdr": [1, 1],
        }
    )
    assert start_hit_category(df, None, 0.1, other="alt").tolist() == ["annot_only", "nc_only"]
