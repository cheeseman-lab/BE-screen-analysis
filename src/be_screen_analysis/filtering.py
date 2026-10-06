"""Filtering of the ABE start/stop screen and the S1 HeLa shake-off screen.

The base editing screen was performed in HeLa, so only the HeLa S1 MAGeCK comparisons are used.
"""

from pathlib import Path

import numpy as np
import pandas as pd

EFF_D0 = "sensor_mean_D0_pct_successful_target_codon_mutation"
EFF_D12 = "sensor_mean_D12_pct_successful_target_codon_mutation"
STAT_COLS = ["num_guides", "lfc", "neg_fdr", "pos_fdr"]

S1_DAY0 = "Day0"
S1_REMAINING = {
    "day2_remaining": "D2",
    "day3_remaining": "D3",
    "day4_remaining": "D4",
    "day5_remaining": "D5",
}
S1_SHAKEOFF_CMPS = ["d2s_d2r", "d3s_d3r", "d4s_d4r", "d5s_d5r"]
S1_DROPOUT_CMPS = ["d2r_d0", "d3r_d0", "d4r_d0", "d5r_d0"]
S1_CMPS = S1_SHAKEOFF_CMPS + S1_DROPOUT_CMPS
S1_NT = "nontargeting"

S5_CMPS = {"shakeoff": "MAGeCK_shake-off_screen.csv", "dropout": "MAGeCK_abundance_78h_DOX.csv"}
S5_NT = "NONTARGETING_CONTROL"
S5_DAY0 = "0h"
S5_REMAINING = {"78h_remaining": "78h"}


def _start_sites(summary: pd.DataFrame) -> pd.DataFrame:
    """Unstack annotated/nc pairs into one row per unique start site."""
    parts = []
    for prefix in ("annot", "nc"):
        cols = {
            f"{prefix}_start_site_target_id": "site_id",
            f"{prefix}_start_site_type": "site_type",
            f"{prefix}_reference_codon": "reference_codon",
            f"{prefix}_target_codon_range": "target_codon_range",
            **{f"{prefix}_{c}": c for c in STAT_COLS},
        }
        part = summary[["Hugo_Symbol", "Chromosome", "strand", *cols]].rename(columns=cols)
        if prefix == "nc":
            part["paired_annot_site_id"] = summary["annot_start_site_target_id"]
        parts.append(part)
    sites = pd.concat(parts, ignore_index=True)
    # an nc site appears once; an annotated site repeats for every nc it is paired with
    paired = sites.groupby("site_id")["paired_annot_site_id"].first()
    sites = sites.drop_duplicates("site_id").drop(columns="paired_annot_site_id")
    return sites.merge(paired.rename("paired_annot_site_id"), on="site_id")


def _stop_sites(summary: pd.DataFrame) -> pd.DataFrame:
    sites = summary.rename(columns={"stop_site_target_id": "site_id"})
    sites["site_type"] = "Stop"
    keep = [
        "Hugo_Symbol",
        "Chromosome",
        "strand",
        "site_id",
        "site_type",
        "reference_codon",
        "target_codon_range",
        "ClinVar_ID",
        "Clinical_Significance",
        *STAT_COLS,
    ]
    return sites[keep]


def flag_overlaps(sites: pd.DataFrame) -> pd.Series:
    """Flag sites whose codon shares at least one base with another site."""
    ends = sites["target_codon_range"].str.split(";", expand=True).astype(int)
    lo, hi = ends.min(axis=1), ends.max(axis=1)
    flag = pd.Series(False, index=sites.index)
    for _, idx in sites.groupby("Chromosome").groups.items():
        order = lo[idx].sort_values().index
        a, b = lo[order].to_numpy(), hi[order].to_numpy()
        hit = a[1:] <= b[:-1]
        flag[order[1:][hit]] = True
        flag[order[:-1][hit]] = True
    return flag


def load_be_sites(data_dir: Path, screen: str, min_eff: float = 20) -> pd.DataFrame:
    """Load one row per unique site for the start or stop screen with guide-level summaries."""
    summary = pd.read_csv(data_dir / f"{screen}_codon_summary.csv")
    guides = pd.read_csv(data_dir / f"{screen}_guide_summary.csv")
    sites = _start_sites(summary) if screen == "start" else _stop_sites(summary)

    sid = f"{screen}_site_target_id"
    guides["efficient"] = (guides[EFF_D0] >= min_eff) | (guides[EFF_D12] >= min_eff)
    per_site = guides.groupby(sid).agg(
        ensembl_transcript_id=("ensembl_transcript_id", "first"),
        n_efficient_guides=("efficient", "sum"),
        max_eff_D0=(EFF_D0, "max"),
        max_eff_D12=(EFF_D12, "max"),
    )
    sites = sites.merge(per_site, left_on="site_id", right_index=True, how="left")
    sites["n_efficient_guides"] = sites["n_efficient_guides"].fillna(0).astype(int)
    sites["overlaps_other_site"] = flag_overlaps(sites)
    sites.insert(0, "screen", screen)
    return sites


def load_be_guides(data_dir: Path, screen: str, min_eff: float = 20) -> pd.DataFrame:
    """One row per guide with sensor efficiency >= ``min_eff`` at D0 or D12, plus site info.

    Site-level MAGeCK columns (``site_*``) come from the codon summary and are NA for sites that
    are not in it; ``site_median_guide_lfc`` is computed for every site from its kept guides.
    """
    guides = pd.read_csv(data_dir / f"{screen}_guide_summary.csv")
    guides = guides[(guides[EFF_D0] >= min_eff) | (guides[EFF_D12] >= min_eff)].rename(
        columns={
            f"{screen}_site_target_id": "site_id",
            f"{screen}_site_type": "site_type",
            EFF_D0: "eff_D0",
            EFF_D12: "eff_D12",
            "lfc": "guide_lfc",
            "fdr": "guide_fdr",
        }
    )
    if screen == "stop":
        guides["site_type"] = "Stop"
    keep = [
        "Hugo_Symbol",
        "ensembl_transcript_id",
        "site_id",
        "site_type",
        "Chromosome",
        "strand",
        "reference_codon",
        "mutated_codon",
        "target_codon_range",
        "guide_id",
        "Protospacer",
        "eff_D0",
        "eff_D12",
        "guide_lfc",
        "guide_fdr",
    ]
    if screen == "stop":
        keep += ["ClinVar_ID", "Clinical_Significance"]
    guides = guides[keep]

    sites = load_be_sites(data_dir, screen, min_eff=min_eff).set_index("site_id")
    site_cols = {"lfc": "site_lfc", "neg_fdr": "site_neg_fdr", "pos_fdr": "site_pos_fdr"}
    extra = ["overlaps_other_site"] + (["paired_annot_site_id"] if screen == "start" else [])
    guides = guides.join(sites[[*site_cols, *extra]].rename(columns=site_cols), on="site_id")
    guides["in_codon_summary"] = guides["site_id"].isin(sites.index)
    guides["site_median_guide_lfc"] = guides.groupby("site_id")["guide_lfc"].transform("median")
    guides.insert(0, "screen", screen)
    return guides.reset_index(drop=True)


def filter_be_sites(sites: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    """Keep sites with >=1 efficient guide and MAGeCK site FDR < ``fdr`` in either direction."""
    neg, pos = sites["neg_fdr"] < fdr, sites["pos_fdr"] < fdr
    out = sites[(sites["n_efficient_guides"] > 0) & (neg | pos)].copy()
    out["direction"] = np.where(neg[out.index], "depleted", "enriched")
    return out


def centred_guide_lfc(
    counts: pd.DataFrame, ref: str, cols: dict[str, str], is_control: pd.Series
) -> pd.DataFrame:
    """Guide log2 CPM fold change of each ``cols`` sample vs ``ref``, centred on control guides."""
    samples = [ref, *cols]
    cpm = counts[samples].div(counts[samples].sum()) * 1e6
    lfc = np.log2((cpm[list(cols)] + 1).div(cpm[ref] + 1, axis=0))
    return lfc.sub(lfc[is_control].median())


def nt_centred_guide_lfc(
    counts: pd.DataFrame,
    ref: str = S1_DAY0,
    cols: dict[str, str] = S1_REMAINING,
    gene_col: str = "gene",
    nt: str = S1_NT,
) -> pd.DataFrame:
    """Guide log2 CPM fold change of each ``cols`` sample vs ``ref``, centred on NT guides."""
    return centred_guide_lfc(counts, ref, cols, counts[gene_col] == nt)


def nt_null(
    lfc: pd.DataFrame, is_nt: pd.Series, guides_per_gene: int = 4, n: int = 20000, seed: int = 0
) -> pd.DataFrame:
    """Null distribution of gene LFCs from medians of random non-targeting guide sets."""
    rng = np.random.default_rng(seed)
    nt_lfc = lfc[is_nt].to_numpy()
    idx = np.array([rng.choice(len(nt_lfc), guides_per_gene, replace=False) for _ in range(n)])
    return pd.DataFrame(np.median(nt_lfc[idx], axis=1), columns=lfc.columns)


def s1_dropout_labels(counts: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per-gene essential-dropout labels from NT-centred guide LFCs (labels only, no filtering)."""
    lfc = nt_centred_guide_lfc(counts)
    is_nt = counts["gene"] == S1_NT
    gene_lfc = lfc[~is_nt].groupby(counts.loc[~is_nt, "gene"]).median()
    last = list(S1_REMAINING)[-1]
    cutoff = nt_null(lfc, is_nt)[last].quantile(alpha)
    return pd.DataFrame(
        {
            **{f"dropout_lfc_{d}": gene_lfc[c] for c, d in S1_REMAINING.items()},
            "dropout_consistent": gene_lfc[last] < cutoff,
            "dropout_monotone": (gene_lfc.diff(axis=1).iloc[:, 1:] <= 0).all(axis=1),
        }
    )


def _add_pass_columns(table: pd.DataFrame, cmps: list[str], fdr: float) -> pd.DataFrame:
    """Add n_comparisons_passed / comparisons_passed from ``{cmp}_{neg,pos}_fdr`` columns."""
    passed = pd.concat(
        [
            (table[f"{c}_{d}_fdr"] < fdr).map({True: f"{c}:{d}", False: ""})
            for c in cmps
            for d in ("neg", "pos")
        ],
        axis=1,
    )
    table["n_comparisons_passed"] = (passed != "").sum(axis=1)
    table["comparisons_passed"] = passed.apply(lambda r: ";".join(x for x in r if x), axis=1)
    return table


def filter_s1_hela(
    neg: pd.DataFrame, pos: pd.DataFrame, counts: pd.DataFrame, fdr: float = 0.05
) -> pd.DataFrame:
    """Keep S1 HeLa genes with FDR < ``fdr`` (neg or pos) in any shake-off/dropout comparison."""
    neg, pos = neg.set_index("id").drop(S1_NT), pos.set_index("id").drop(S1_NT)
    table = pd.DataFrame(index=neg.index)
    for c in S1_CMPS:
        table[f"{c}_lfc"] = neg[f"{c}.neg.lfc"]
        table[f"{c}_neg_fdr"] = neg[f"{c}.neg.fdr"]
        table[f"{c}_pos_fdr"] = pos[f"{c}.pos.fdr"].reindex(neg.index)
    table = _add_pass_columns(table, S1_CMPS, fdr).join(s1_dropout_labels(counts))
    out = table[table["n_comparisons_passed"] > 0]
    out.insert(0, "cell_line", "HeLa")
    return out.rename_axis("Hugo_Symbol").reset_index()


def s5_control_ids(ids: pd.Index, counts: pd.DataFrame) -> pd.Series:
    """Flag S5 MAGeCK ids that are controls: non-targeting, intergenic or olfactory receptor."""
    or_genes = set(counts.loc[counts["category"] == "OR_CONTROL", "target_symbol"])
    is_ctrl = ids.str.startswith((S5_NT, "INTERGENIC_CONTROL")) | ids.isin(or_genes)
    return pd.Series(is_ctrl, index=ids)


def s5_dropout_labels(
    counts: pd.DataFrame, genes: set[str], alpha: float = 0.05, control: str = "nontargeting"
) -> pd.DataFrame:
    """Essential-dropout labels from S5 counts (78h remaining vs 0h), filled only for ``genes``.

    ``control`` picks the constructs used to centre the LFCs and build the 2-construct null:
    "nontargeting" (no cut) or "intergenic" (cut DNA but disrupt no gene, so the label measures
    dropout beyond cutting toxicity).
    """
    if control == "nontargeting":
        is_ctrl = counts["target_symbol"] == S5_NT
    elif control == "intergenic":
        is_ctrl = counts["category"] == "INTERGENIC_CONTROL"
    else:
        raise ValueError(f"unknown control: {control!r}")
    lfc = centred_guide_lfc(counts, S5_DAY0, S5_REMAINING, is_ctrl)
    keep = counts["target_symbol"].isin(genes)
    gene_lfc = lfc[keep].groupby(counts.loc[keep, "target_symbol"]).median()
    col = list(S5_REMAINING)[0]
    cutoff = nt_null(lfc, is_ctrl, guides_per_gene=2)[col].quantile(alpha)
    return pd.DataFrame(
        {
            f"dropout_lfc_{S5_REMAINING[col]}": gene_lfc[col],
            "dropout_consistent": gene_lfc[col] < cutoff,
        }
    )


def s5_gene_table(
    tables: dict[str, pd.DataFrame],
    counts: pd.DataFrame,
    s1_genes: set[str],
    fdr: float = 0.05,
    control: str = "nontargeting",
) -> pd.DataFrame:
    """All S5 genes (controls removed) with per-comparison stats, pass columns and labels."""
    tables = {c: t.set_index("id") for c, t in tables.items()}
    ids = next(iter(tables.values())).index
    ids = ids[~s5_control_ids(ids, counts).to_numpy()]
    table = pd.DataFrame(index=ids)
    for c, t in tables.items():
        table[f"{c}_lfc"] = t["neg|lfc"].reindex(ids)
        table[f"{c}_neg_fdr"] = t["neg|fdr"].reindex(ids)
        table[f"{c}_pos_fdr"] = t["pos|fdr"].reindex(ids)
    table = _add_pass_columns(table, list(tables), fdr)
    table["in_s1_essential"] = table.index.isin(s1_genes)
    table = table.join(s5_dropout_labels(counts, s1_genes & set(ids), control=control))
    table["dropout_consistent"] = table["dropout_consistent"].astype("boolean")
    table.insert(0, "cell_line", "HeLa")
    return table.rename_axis("Hugo_Symbol").reset_index()


def filter_s5(
    tables: dict[str, pd.DataFrame],
    counts: pd.DataFrame,
    s1_genes: set[str],
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Keep S5 genes (controls removed) with FDR < ``fdr`` (neg or pos) in any comparison."""
    table = s5_gene_table(tables, counts, s1_genes, fdr)
    return table[table["n_comparisons_passed"] > 0].reset_index(drop=True)


def intersect(be: pd.DataFrame, screen: pd.DataFrame) -> pd.DataFrame:
    """Keep every BE site whose gene passes the screen filter, joined with its gene columns."""
    return be.merge(screen.drop(columns="cell_line"), on="Hugo_Symbol", how="inner")
