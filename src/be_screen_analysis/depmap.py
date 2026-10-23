"""DepMap CRISPR dependency calls per gene (number of dependent cell lines)."""

from pathlib import Path

import pandas as pd

DEPENDENT_P = 0.5  # DepMap portal: a line is dependent when P(dependency) > 0.5


def depmap_dependency(path: Path) -> pd.DataFrame:
    """Per gene: dependent cell lines, lines screened, percent and fraction text.

    ``path`` is DepMap's CRISPRGeneDependency.csv (rows = cell lines, columns "SYMBOL (Entrez)").
    Dependent = probability of dependency > DEPENDENT_P, as on the DepMap portal; screened =
    lines with a value for the gene.
    """
    prob = pd.read_csv(path, index_col=0)
    genes = prob.columns.str.replace(r"\s*\(\d+\)$", "", regex=True)
    n_dep = (prob > DEPENDENT_P).sum()
    n_scr = prob.notna().sum()
    table = pd.DataFrame(
        {
            "depmap_entrez": prob.columns.str.extract(r"\((\d+)\)$", expand=False).to_numpy(),
            "depmap_n_dependent": n_dep.to_numpy(),
            "depmap_n_screened": n_scr.to_numpy(),
        },
        index=pd.Index(genes, name="Hugo_Symbol"),
    )
    table["depmap_pct_dependent"] = 100 * table["depmap_n_dependent"] / table["depmap_n_screened"]
    table["depmap_fraction"] = (
        table["depmap_n_dependent"].astype(str) + "/" + table["depmap_n_screened"].astype(str)
    )
    # a symbol listed twice (rare): keep the entry screened in more lines
    table = table.sort_values("depmap_n_screened", ascending=False)
    return table[~table.index.duplicated()].sort_index()


def common_essentials(path: Path) -> pd.Index:
    """Gene symbols of DepMap's inferred common essentials (CRISPRInferredCommonEssentials.csv)."""
    genes = pd.read_csv(path).iloc[:, 0]
    return pd.Index(genes.str.replace(r"\s*\(\d+\)$", "", regex=True).unique(), name="Hugo_Symbol")


def line_dependency(path: Path, model_id: str) -> pd.Series:
    """One cell line's dependency probability per gene from CRISPRGeneDependency.csv.

    ``model_id`` is the DepMap ModelID (e.g. K562 = ACH-000551, from DepMap's Model.csv).
    """
    prob = pd.read_csv(path, index_col=0).loc[model_id]
    genes = prob.index.str.replace(r"\s*\(\d+\)$", "", regex=True)
    prob.index = pd.Index(genes, name="Hugo_Symbol")
    return prob[~prob.index.duplicated()].rename(model_id)
