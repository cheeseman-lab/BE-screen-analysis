"""Isoform-aware KO tables: guides grouped by the start isoforms they knock out.

Guide (or S5 construct) isoform labels come from ``guide_classify`` (scripts/guide_classification);
MAGeCK groups them per start with ``mageck.isoform_count_table``. These helpers build the
per-library construct labels, the gene-level count table and the per-start result table, and
select starts that are hits across screens.
"""

import pandas as pd

from be_screen_analysis.filtering import start_hit_category

HIT_FDR = 0.1  # a start is significant if its neg or pos FDR is below this (Tyler's rule)
START_KEY = ["gene", "alt_start_id"]
SHARED_COLS = ["annot_lfc", "alt_lfc", "delta_lfc", "alt_num_constructs", "category"]
# each library's abundance comparison, as axis-label text
COMPARISON_LABELS = {
    "s5": "78h remaining vs 0h",
    "s1": "day 5 remaining vs day 0",
    "k562": "untreated endpoint vs initial",
}


def single_guide_labels(guides: pd.DataFrame, control: pd.Series, nt: pd.Series) -> pd.DataFrame:
    """Guide isoform labels in the S5 construct columns: each guide is its own construct.

    ``control`` marks guides with no target gene (category = their library gene label), ``nt``
    the non-targeting ones (category NONTARGETING_CONTROL, the normalisation set); every other
    guide is GENE_TARGETING. target_symbol = library gene, construct_gene = its GENCODE gene.
    """
    category = guides["gene"].where(control, "GENE_TARGETING").mask(nt, "NONTARGETING_CONTROL")
    return pd.DataFrame(
        {
            "category": category,
            "target_symbol": guides["gene"],
            "construct_gene": guides["gtf_gene"],
            "knocked_out": guides["knocked_out"],
            "control": nt,
        }
    )


def gene_count_table(counts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Count table, one row per construct: Gene = construct_gene (controls: target_symbol)."""
    targeting = labels["category"] == "GENE_TARGETING"
    gene = labels["construct_gene"].where(targeting).fillna(labels["target_symbol"])
    return pd.concat([gene.rename("Gene"), counts], axis=1).rename_axis("sgRNA")


def isoform_lfc_table(
    labels: pd.DataFrame, scaffold: pd.DataFrame, gene_stats: pd.DataFrame, iso: pd.DataFrame
) -> pd.DataFrame:
    """One row per (gene, non-annotated start); genes with only an annotated start get one row.

    ``scaffold`` is guide_classify's start_scaffold.csv; ``gene_stats`` and ``iso`` are the gene
    and isoform MAGeCK runs (lfc, neg_fdr, pos_fdr; isoform groups keyed ``{gene}|{start_id}``).
    Columns: gene_lfc / gene_neg_fdr; annot_ and alt_ start_id, num_constructs (constructs
    knocking the start out), lfc, neg_fdr, pos_fdr (NaN when no construct knocks it out);
    alt_kind, alt_codon, alt_provenance, alt_in_frame_main; alt_ko_constructs; same_as_annotated
    (alt knocked out by exactly the annotated start's constructs, so its LFC is identical);
    delta_lfc (alt_lfc - annot_lfc).
    """
    t = labels[(labels["category"] == "GENE_TARGETING") & labels["construct_gene"].notna()]
    ko: dict[tuple[str, str], set[str]] = {}
    for cid, g, k in zip(t.index, t["construct_gene"], t["knocked_out"].fillna("")):
        for s in filter(None, k.split(";")):
            ko.setdefault((g, s), set()).add(cid)
    starts = scaffold[scaffold["gene"].isin(set(t["construct_gene"]))].copy()
    sets = [ko.get((g, s), set()) for g, s in zip(starts["gene"], starts["start_id"])]
    starts["num_constructs"] = [len(c) for c in sets]
    starts["ko_constructs"] = [";".join(sorted(c)) for c in sets]
    starts["ko_set"] = sets
    stats = iso.reindex(starts["gene"] + "|" + starts["start_id"])
    starts[list(stats)] = stats.to_numpy()

    keep = ["gene", "start_id", "num_constructs", "lfc", "neg_fdr", "pos_fdr", "ko_set"]
    annot = starts.loc[starts["kind"] == "Annotated", keep].add_prefix("annot_")
    alt_cols = ["kind", "codon", "provenance", "in_frame_main", "ko_constructs"]
    alt = starts.loc[starts["kind"] != "Annotated", keep + alt_cols].add_prefix("alt_")
    out = annot.merge(alt, left_on="annot_gene", right_on="alt_gene", how="left")
    out = out.rename(columns={"annot_gene": "gene"}).drop(columns="alt_gene")
    out.insert(1, "gene_lfc", out["gene"].map(gene_stats["lfc"]))
    out.insert(2, "gene_neg_fdr", out["gene"].map(gene_stats["neg_fdr"]))
    out["same_as_annotated"] = [
        (a == b) if isinstance(b, set) else pd.NA
        for a, b in zip(out["annot_ko_set"], out["alt_ko_set"])
    ]
    out["delta_lfc"] = out["alt_lfc"] - out["annot_lfc"]
    out = out.drop(columns=["annot_ko_set", "alt_ko_set"])
    return out.astype({"alt_num_constructs": "Int64"})


def distinct_starts(table: pd.DataFrame) -> pd.DataFrame:
    """Alternative starts with their own construct set (both starts tested) and a ``category``.

    ``table`` is an ``isoform_lfc_table``; category is Tyler's hit category of the annotated vs
    alternative start (``filtering.start_hit_category``, neg or pos FDR < HIT_FDR): annot_only,
    nc_only (alternative start only), both or neither.
    """
    t = table[~table["same_as_annotated"].astype(bool)].dropna(subset=["annot_lfc", "alt_lfc"])
    return t.assign(category=start_hit_category(t, None, HIT_FDR, other="alt"))


def shared_hits(starts: dict[str, pd.DataFrame], libs: list[str]) -> pd.DataFrame:
    """Starts that are a hit (category not neither) in every library of ``libs``.

    ``starts`` maps library -> ``distinct_starts``; each library's SHARED_COLS sit side by side
    as ``{lib}_{col}``, after gene, alt_start_id and alt_kind.
    """
    table = None
    for lib in libs:
        d = starts[lib]
        hit = d.loc[d["category"] != "neither", [*START_KEY, "alt_kind", *SHARED_COLS]]
        hit = hit.rename(columns={c: f"{lib}_{c}" for c in SHARED_COLS})
        table = hit if table is None else table.merge(hit.drop(columns="alt_kind"), on=START_KEY)
    return table.sort_values(START_KEY).reset_index(drop=True)
