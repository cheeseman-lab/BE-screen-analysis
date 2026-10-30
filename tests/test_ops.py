import pandas as pd
import pytest

from be_screen_analysis.ops import barcode_spacers, guide_features, isoform_features

A = "AAAAAAAAAAACCCCCCCCC"
B = "GGGGGGGGGGGTTTTTTTTT"


def _design(rows):
    return pd.DataFrame(rows, columns=["sgRNA", "gene_symbol", "dialout", "design", "source"])


def test_barcode_spacers_collapses_oligos_and_drops_other_dialouts():
    design = _design(
        [
            (A, "X", 0, "essentials", "brunello"),
            (A, "X", 0, "essentials", "brunello"),  # duplicate oligo
            (B, "nontargeting", 1, "essentials_nontargeting", "gecko"),
            (A[:11] + "GGGGGGGGG", "Z", 2, "other", "tkov3"),  # same prefix, dropped sub-library
        ]
    )
    library = pd.DataFrame(
        {"prefix": [A[:11], B[:11]], "gene_symbol": ["X", "nontargeting"], "gene_id": [1, 2]}
    )
    out = barcode_spacers(library, design)
    assert out["barcode"].tolist() == [A[:11], B[:11]]
    assert out["sequence"].tolist() == [A, B]
    assert out["design_symbol"].tolist() == ["X", "nontargeting"]


def test_barcode_spacers_raises_on_shared_prefix():
    design = _design(
        [(A, "X", 0, "essentials", "s"), (A[:11] + "GGGGGGGGG", "Y", 0, "essentials", "s")]
    )
    library = pd.DataFrame({"prefix": [A[:11]], "gene_symbol": ["X"], "gene_id": [1]})
    with pytest.raises(ValueError):
        barcode_spacers(library, design)


def test_guide_and_isoform_features():
    guides = pd.DataFrame(
        {
            "barcode": ["b1", "b2", "b3", "n1"],
            "gene_symbol": ["X", "X", "X", "nontargeting_n1"],
            "gtf_gene": ["X", "X", "X", None],
            "control": [False, False, False, True],
            "knocked_out": ["X_annot;X_trunc", "X_annot", None, None],  # b3: no start
        }
    )
    constructs = pd.DataFrame(
        {
            "cell_barcode_0": ["b1", "b2", "b3", "n1", "missing"],
            "gene_symbol_0": ["X", "X", "X", "nontargeting_n1", "Y"],
            "cell_count": [10, 20, 30, 40, 50],
            "f1": [1.0, 3.0, 100.0, 0.5, 9.0],
        }
    )
    table = guide_features(constructs, guides)
    assert table["barcode"].tolist() == ["b1", "b2", "b3", "n1"]
    assert list(table.columns[-2:]) == ["cell_count", "f1"]

    scaffold = pd.DataFrame(
        {
            "gene": ["X", "X"],
            "start_id": ["X_annot", "X_trunc"],
            "kind": ["Annotated", "Truncated"],
        }
    )
    out = isoform_features(table, scaffold)
    assert out["start_id"].tolist()[:2] == ["X_annot", "X_trunc"]
    assert out["f1"].tolist() == [2.0, 1.0, 0.5]  # median of b1, b2; b1 alone; control as is
    assert out["cell_count"].tolist() == [30, 10, 40]
    assert out["barcodes"].tolist() == ["b1;b2", "b1", "n1"]
    assert out["same_as_annotated"].tolist()[:2] == [True, False]
    assert out["kind"].iloc[2] == "nontargeting"
