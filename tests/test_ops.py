import pandas as pd
import pytest

from be_screen_analysis.ops import barcode_spacers

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
