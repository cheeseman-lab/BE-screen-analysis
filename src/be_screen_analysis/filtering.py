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


def filter_be_sites(sites: pd.DataFrame, fdr: float = 0.05) -> pd.DataFrame:
    """Keep sites with >=1 efficient guide and MAGeCK site FDR < ``fdr`` in either direction."""
    neg, pos = sites["neg_fdr"] < fdr, sites["pos_fdr"] < fdr
    out = sites[(sites["n_efficient_guides"] > 0) & (neg | pos)].copy()
    out["direction"] = np.where(neg[out.index], "depleted", "enriched")
    return out


def nt_centred_guide_lfc(counts: pd.DataFrame) -> pd.DataFrame:
    """Guide log2 CPM fold change of each remaining timepoint vs Day0, centred on NT guides."""
    cols = [S1_DAY0, *S1_REMAINING]
    cpm = counts[cols].div(counts[cols].sum()) * 1e6
    lfc = np.log2((cpm[list(S1_REMAINING)] + 1).div(cpm[S1_DAY0] + 1, axis=0))
    return lfc.sub(lfc[counts["gene"] == S1_NT].median())


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


def filter_s1_hela(
    neg: pd.DataFrame, pos: pd.DataFrame, counts: pd.DataFrame, fdr: float = 0.05
) -> pd.DataFrame:
    """Keep S1 HeLa genes with FDR < ``fdr`` (neg or pos) in any shake-off/dropout comparison."""
    neg, pos = neg.set_index("id").drop(S1_NT), pos.set_index("id").drop(S1_NT)
    table = pd.DataFrame(index=neg.index)
    passed = []
    for c in S1_CMPS:
        table[f"{c}_lfc"] = neg[f"{c}.neg.lfc"]
        table[f"{c}_neg_fdr"] = neg[f"{c}.neg.fdr"]
        table[f"{c}_pos_fdr"] = pos[f"{c}.pos.fdr"].reindex(neg.index)
        for d in ("neg", "pos"):
            passed.append((table[f"{c}_{d}_fdr"] < fdr).map({True: f"{c}:{d}", False: ""}))
    passed = pd.concat(passed, axis=1)
    table["n_comparisons_passed"] = (passed != "").sum(axis=1)
    table["comparisons_passed"] = passed.apply(lambda r: ";".join(x for x in r if x), axis=1)
    table = table.join(s1_dropout_labels(counts))
    out = table[table["n_comparisons_passed"] > 0]
    out.insert(0, "cell_line", "HeLa")
    return out.rename_axis("Hugo_Symbol").reset_index()


def intersect(be: pd.DataFrame, s1: pd.DataFrame) -> pd.DataFrame:
    """Keep every BE site whose gene passes the S1 filter, joined with the S1 gene columns."""
    return be.merge(s1.drop(columns="cell_line"), on="Hugo_Symbol", how="inner")
