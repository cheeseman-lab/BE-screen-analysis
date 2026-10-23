"""S1 D4 fitness subset and how it fares in S1 D5, the S5 genome-wide screen and the ABE screen.

S5 fitness subset = the same number of S5 genes with the most negative 78h gene LFC, written to
figures/old_analysis/ABE_validation/fitness_subset/s5_fitness_subset.csv.

S1 D5 fitness subset = the 200 S1 genes with the most negative published D5 LFC, written to
figures/old_analysis/ABE_validation/fitness_subset/s1_d5_fitness_subset.csv.

ABE start codon fitness subset = the same number of genes with the most negative annotated-start
D4 LFC in Tyler's ABE screen, written to figures/old_analysis/ABE_validation/fitness_subset/
abe_start_fitness_subset.csv.

S1 fitness subset = S1 genes whose published D4 gene LFC (mageck.neg_HeLa d4r_d0) is below the
alpha = 0.0001 quantile of the S1 non-targeting control null centred on the gene peak (see
plot_lfc_distributions.py). Writes
figures/old_analysis/ABE_validation/fitness_subset/s1_d4_fitness_subset.csv
(provenance in its # header) and, one row per fitness-subset gene, to
results/old_analysis/ABE_validation/fitness_subset/:
  s1_d4_vs_d5.csv          published S1 gene LFC and percentile at D2-D5
  s5_78h.csv               S5 78h gene LFC and percentile, number of constructs
  abe_targeting.csv        guides per gene in Tyler's library (start by site type, stop, KO)
  abe_annotated_start.csv  annotated-start gene LFC and percentile at D4-D21

Descriptive only: no cutoff is applied in the follow-up tables. Percentile = rank among all genes
of that dataset and timepoint (low = most depleted). Genes are matched by exact HUGO symbol.

Run scripts/abe_mageck_analysis.py first; this reads results/abe_mageck/.
"""

import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from be_screen_analysis.correlations import abe_gene_lfc, s5_gene_dropout
from be_screen_analysis.essential import (
    ALPHA,
    abe_control_null,
    bh_fdr,
    null_pvalue,
    s1_nt_null,
    s5_intergenic_null,
    shift_null,
)
from be_screen_analysis.filtering import (
    EFF_D0,
    EFF_D12,
    S1_DAY0,
    S1_DROPOUT_CMPS,
    S1_NT,
    S1_REMAINING,
    load_be_counts,
    mageck_guide_lfc,
    s1_mageck_table,
    s1_published_gene_lfc,
)
from be_screen_analysis.mageck import MAGECK_DAYS

ROOT = Path(__file__).resolve().parents[4]
S1_DIR = ROOT / "data" / "shakeoff_data" / "table_s1"
S1_FILE = S1_DIR / "HeLa_sgRNA_sequences_and_counts.csv"
S5_DIR = ROOT / "data" / "shakeoff_data" / "table_s5"
S5_FILE = S5_DIR / "sgRNA_sequences_and_counts.csv"
S5_MAGECK = S5_DIR / "MAGeCK_abundance_78h_DOX.csv"
MAGECK_DIR = ROOT / "results" / "abe_mageck"
COUNTS_DIR = ROOT / "data" / "start_stop_ABE_screen_data" / "ABE_guide_counts"
LIST_DIR = ROOT / "figures" / "old_analysis" / "ABE_validation" / "fitness_subset"
OUT_DIR = ROOT / "results" / "old_analysis" / "ABE_validation" / "fitness_subset"

DAY = "D4"
N_NULL = 1_000_000  # matches plot_lfc_distributions.py
N_D5 = 200  # size of the S1 D5 fitness subset (most depleted genes at D5)
GUIDES_PER_GENE = 4
MIN_EFF = 20
N_EFF = f"n_start_annotated_eff{MIN_EFF}"


def _pct(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each value within its column over all genes (low = most depleted)."""
    return df.rank(pct=True).add_suffix(" pct")


def _with_pct(df: pd.DataFrame, genes: pd.Index) -> pd.DataFrame:
    """LFC and percentile columns side by side, restricted to ``genes`` (NaN if absent)."""
    pairs = [c for col in df for c in (col, f"{col} pct")]
    return df.join(_pct(df))[pairs].reindex(genes)


def _commit() -> str:
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa: E731
    return run("rev-parse", "--short", "HEAD").strip() + (
        " (uncommitted changes)" if run("status", "--porcelain") else ""
    )


def s1_fitness_subset(s1_counts: pd.DataFrame, s1: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """S1 genes below the control-null cutoff at DAY, most depleted first, with metrics.

    ``s1`` is the published gene LFC. Null p / FDR: empirical one-sided p of the gene LFC against
    the null, BH-adjusted over all S1 genes. Guide columns: the gene's guide LFCs at DAY
    (computed as MAGeCK does) and how many fall below the cutoff.
    MAGeCK columns: the published S1 HeLa remaining-vs-Day0 comparisons (d2r_d0 ... d5r_d0).
    """
    lfc = s1[f"S1 {DAY}"]
    null = shift_null(s1_nt_null(s1_counts, N_NULL, GUIDES_PER_GENE)[DAY], lfc)
    cutoff = null.quantile(ALPHA)
    p = null_pvalue(lfc, null)
    is_nt = s1_counts["gene"] == S1_NT
    day_col = {d: c for c, d in S1_REMAINING.items()}[DAY]
    guide_lfc = mageck_guide_lfc(s1_counts, S1_DAY0, S1_REMAINING)[day_col]
    guide = guide_lfc[~is_nt].groupby(s1_counts["gene"])
    mageck = s1_mageck_table(
        pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv"),
        pd.read_csv(S1_DIR / "mageck.pos_HeLa.csv"),
        S1_DROPOUT_CMPS,
        ranks=True,
    )
    table = pd.DataFrame(
        {
            f"s1_{DAY}_lfc": lfc,
            f"s1_{DAY}_pct": lfc.rank(pct=True),
            f"null_p_{DAY}": p,
            f"null_fdr_{DAY}": bh_fdr(p),
            "n_guides": guide.size(),
            f"n_guides_below_cutoff_{DAY}": guide.apply(lambda g: (g < cutoff).sum()),
            f"guide_lfc_min_{DAY}": guide.min(),
            f"guide_lfc_max_{DAY}": guide.max(),
        }
    ).join(mageck)
    called = table[table[f"s1_{DAY}_lfc"] < cutoff].sort_values(f"s1_{DAY}_lfc")
    return called.rename_axis("Hugo_Symbol"), cutoff


def write_list(genes: pd.DataFrame, header: list[str], name: str) -> Path:
    """Write ``genes`` to LIST_DIR/name with ``header`` as # comment lines above the table."""
    out = LIST_DIR / name
    with out.open("w") as f:
        f.writelines(f"# {line}\n" for line in header)
        genes.to_csv(f, float_format="%.4g")
    return out


def _generated() -> str:
    return f"generated {date.today()} by {Path(__file__).relative_to(ROOT)} at commit {_commit()}"


def write_s1_fitness_subset(genes: pd.DataFrame, cutoff: float) -> Path:
    header = [
        f"S1 {DAY} fitness subset (n = {len(genes)})",
        _generated(),
        f"input: {S1_DIR.relative_to(ROOT)}/ (HeLa, focused library)",
        f"LFC: published S1 HeLa MAGeCK gene LFC for {DAY} remaining vs Day0 "
        "(mageck.neg_HeLa.csv d4r_d0.neg.lfc: median-ratio normalised, pseudocount 1, uncentred, "
        "mean of the gene's guides)",
        f"null: {N_NULL:,} pseudo-genes, each the mean of {GUIDES_PER_GENE} random non-targeting "
        f"guide LFCs computed as MAGeCK does from {S1_FILE.name} (reproduces the published gene "
        f"LFC for 99.9% of genes within 0.01); NT guides with a zero count in any S1 sample "
        "excluded (nontargeting_62_0, a dropout in day4_remaining only); centred so its median "
        f"sits on the peak (Gaussian kernel density mode) of the gene {DAY} LFC distribution, "
        "as a stand-in for the cutting control S1 lacks (NT guides do not cut DNA)",
        f"cutoff: alpha = {ALPHA:g}, the {ALPHA * 100:g}th percentile of the null "
        f"= {cutoff:+.3f}; "
        f"genes with {DAY} LFC below it are listed",
        f"columns: s1_{DAY}_lfc / _pct = gene LFC and its percentile among all S1 genes; "
        f"null_p_{DAY} = (number of null pseudo-genes <= gene LFC + 1) / ({N_NULL:,} + 1); "
        f"null_fdr_{DAY} = Benjamini-Hochberg over all S1 genes; n_guides, "
        f"n_guides_below_cutoff_{DAY}, guide_lfc_min/max_{DAY} = the gene's guides at {DAY}",
        "columns d{2-5}r_d0_*: published S1 HeLa MAGeCK (mageck.neg_HeLa.csv, mageck.pos_HeLa.csv) "
        "for day N remaining vs day0: num = guides, lfc, neg_rank, neg_fdr (depletion), "
        "pos_fdr (enrichment)",
    ]
    return write_list(genes, header, "s1_d4_fitness_subset.csv")


def s5_fitness_subset(
    s5_counts: pd.DataFrame, s5: pd.Series, n: int, s1_set: pd.DataFrame
) -> pd.DataFrame:
    """The ``n`` S5 genes with the most negative 78h LFC, with null p / FDR and published MAGeCK."""
    p = null_pvalue(s5, s5_intergenic_null(s5_counts, N_NULL)["78h"])
    mageck = pd.read_csv(S5_MAGECK).set_index("id")
    cols = {"num": "num", "neg|lfc": "lfc", "neg|rank": "neg_rank", "neg|fdr": "neg_fdr"}
    cols["pos|fdr"] = "pos_fdr"
    table = pd.DataFrame(
        {
            "s5_78h_lfc": s5,
            "s5_78h_rank": s5.rank(method="first").astype(int),
            "s5_78h_pct": s5.rank(pct=True),
            "null_p_78h": p,
            "null_fdr_78h": bh_fdr(p),
            "n_constructs": s5_counts[s5_counts["category"] == "GENE_TARGETING"]
            .groupby("target_symbol")
            .size(),
        }
    ).join(mageck[list(cols)].rename(columns=lambda c: f"mageck_78h_{cols[c]}"))
    top = table.loc[s5.nsmallest(n).index].rename_axis("Hugo_Symbol")
    top["in_s1_fitness_subset"] = top.index.isin(s1_set.index)
    top[f"s1_{DAY}_lfc"] = s1_set[f"s1_{DAY}_lfc"].reindex(top.index)
    return top


def write_s5_fitness_subset(genes: pd.DataFrame, n_s5: int) -> Path:
    header = [
        f"S5 fitness subset (n = {len(genes)}): the {len(genes)} S5 genes with the most "
        f"negative 78h gene LFC, matching the size of the S1 {DAY} set",
        _generated(),
        f"input: {S5_DIR.relative_to(ROOT)}/ (HeLa, genome-wide, dual-guide constructs)",
        "LFC: log2((CPM 78h remaining + 1) / (CPM 0h + 1)) per construct, centred on the median "
        f"intergenic construct (cut, no gene); gene = median of its constructs; {n_s5:,} genes",
        f"selection: rank by LFC only, no cutoff; the 78h LFC of the last gene is "
        f"{genes['s5_78h_lfc'].max():+.3f}",
        f"columns: s5_78h_lfc / _rank / _pct = gene LFC, rank and percentile among all S5 genes; "
        f"null_p_78h = (number of pseudo-genes <= gene LFC + 1) / ({N_NULL:,} + 1), pseudo-genes "
        "= median of 2 random intergenic constructs; null_fdr_78h = Benjamini-Hochberg over all "
        "S5 genes; n_constructs",
        f"columns mageck_78h_*: published S5 MAGeCK ({S5_MAGECK.name}): num, lfc, neg_rank, "
        "neg_fdr (depletion), pos_fdr (enrichment)",
        f"columns in_s1_fitness_subset, s1_{DAY}_lfc: membership in and LFC from "
        "s1_d4_fitness_subset.csv",
    ]
    return write_list(genes, header, "s5_fitness_subset.csv")


def abe_fitness_subset(
    abe: pd.DataFrame, n: int, s1_set: pd.Index, s5_set: pd.Index
) -> pd.DataFrame:
    """The ``n`` ABE genes with the most negative annotated-start LFC at DAY, with null p / FDR.

    ``abe`` is abe_gene_lfc (median MAGeCK LFC of efficient annotated-start guides, D4-D21).
    """
    lfc = abe[f"ABE {DAY}"].dropna()
    p = null_pvalue(lfc, abe_control_null(MAGECK_DIR / "start", N_NULL, 4)[DAY])
    guides = pd.read_csv(MAGECK_DIR / "start_guide_summary_mageck.csv")
    ann = guides[guides["start_site_type"] == "Annotated"]
    eff = ann[(ann[EFF_D0] >= MIN_EFF) | (ann[EFF_D12] >= MIN_EFF)]
    table = pd.DataFrame(
        {
            f"abe_{DAY}_lfc": lfc,
            f"abe_{DAY}_rank": lfc.rank(method="first").astype(int),
            f"abe_{DAY}_pct": lfc.rank(pct=True),
            f"null_p_{DAY}": p,
            f"null_fdr_{DAY}": bh_fdr(p),
            N_EFF: eff.groupby("Hugo_Symbol").size(),
            **{f"abe_{d}_lfc": abe[f"ABE {d}"] for d in MAGECK_DAYS if d != DAY},
        }
    ).reindex(lfc.index)
    top = table.loc[lfc.nsmallest(n).index].rename_axis("Hugo_Symbol")
    top["in_s1_fitness_subset"] = top.index.isin(s1_set)
    top["in_s5_fitness_subset"] = top.index.isin(s5_set)
    return top


def write_abe_fitness_subset(genes: pd.DataFrame, n_abe: int) -> Path:
    header = [
        f"ABE start codon fitness subset (n = {len(genes)}): the {len(genes)} genes with the most "
        f"negative annotated-start {DAY} gene LFC, matching the size of the S1 {DAY} set",
        _generated(),
        f"input: {MAGECK_DIR.relative_to(ROOT)}/start_guide_summary_mageck.csv (Tyler's ABE "
        "start screen, HeLa; MAGeCK rerun on the raw counts, NT-normalised)",
        f"LFC: per guide, MAGeCK {DAY} vs D0; gene = median over its annotated-start (ATG) guides "
        f"with >={MIN_EFF}% sensor editing at D0 or D12; {n_abe:,} genes scored",
        f"selection: rank by {DAY} LFC only, no cutoff; the {DAY} LFC of the last gene is "
        f"{genes[f'abe_{DAY}_lfc'].max():+.3f}",
        f"columns: abe_{DAY}_lfc / _rank / _pct = gene LFC, rank and percentile among scored "
        f"genes; null_p_{DAY} = (number of pseudo-genes <= gene LFC + 1) / ({N_NULL:,} + 1), "
        "pseudo-genes = median of 4 random AAVS1 safe-harbour guides; null_fdr = Benjamini-"
        f"Hochberg over all scored genes; {N_EFF}; abe_D6/D12/D21_lfc = the gene's later days",
        "columns in_s1_fitness_subset, in_s5_fitness_subset: membership in "
        "s1_d4_fitness_subset.csv and s5_fitness_subset.csv",
    ]
    return write_list(genes, header, "abe_start_fitness_subset.csv")


def s1_d5_fitness_subset(s1: pd.DataFrame, d4_set: pd.Index) -> pd.DataFrame:
    """The N_D5 S1 genes with the most negative published D5 LFC, with their D4 values."""
    d5 = s1["S1 D5"].dropna()
    table = pd.DataFrame(
        {
            "s1_D5_lfc": d5,
            "s1_D5_rank": d5.rank(method="first").astype(int),
            "s1_D5_pct": d5.rank(pct=True),
            f"s1_{DAY}_lfc": s1[f"S1 {DAY}"],
            f"s1_{DAY}_pct": s1[f"S1 {DAY}"].rank(pct=True),
        }
    ).reindex(d5.index)
    top = table.loc[d5.nsmallest(N_D5).index].rename_axis("Hugo_Symbol")
    top[f"in_s1_{DAY.lower()}_fitness_subset"] = top.index.isin(d4_set)
    return top


def write_s1_d5_fitness_subset(genes: pd.DataFrame) -> Path:
    header = [
        f"S1 D5 fitness subset (n = {len(genes)}): the {len(genes)} S1 genes with the most "
        "negative D5 gene LFC",
        _generated(),
        f"input: {S1_DIR.relative_to(ROOT)}/mageck.neg_HeLa.csv (HeLa, focused library)",
        "LFC: published S1 HeLa MAGeCK gene LFC for D5 remaining vs Day0 (d5r_d0.neg.lfc: "
        "median-ratio normalised, pseudocount 1, uncentred, mean of the gene's guides)",
        "selection: rank by D5 LFC only, no cutoff; the D5 LFC of the last gene is "
        f"{genes['s1_D5_lfc'].max():+.3f}",
        f"columns: s1_D5_lfc / _rank / _pct = D5 gene LFC, rank and percentile among all S1 genes; "
        f"s1_{DAY}_lfc / _pct = the same gene at {DAY}; in_s1_{DAY.lower()}_fitness_subset = "
        "membership in s1_d4_fitness_subset.csv",
    ]
    return write_list(genes, header, "s1_d5_fitness_subset.csv")


def abe_targeting(genes: pd.Index) -> pd.DataFrame:
    """Guides per gene in Tyler's library: start by site type, efficient annotated, stop, KO."""
    start = pd.read_csv(MAGECK_DIR / "start_guide_summary_mageck.csv")
    stop = pd.read_csv(MAGECK_DIR / "stop_guide_summary_mageck.csv")
    ko = pd.concat([load_be_counts(COUNTS_DIR, s, ko=True) for s in ("start", "stop")])
    annotated = start[start["start_site_type"] == "Annotated"]
    efficient = annotated[(annotated[EFF_D0] >= MIN_EFF) | (annotated[EFF_D12] >= MIN_EFF)]
    by_type = start.groupby(["Hugo_Symbol", "start_site_type"]).size().unstack(fill_value=0)
    table = pd.DataFrame(
        {
            **{
                f"n_start_{t.lower()}": by_type.get(t)
                for t in ("Annotated", "Extended", "Truncated")
            },
            "n_start_mixed_type": by_type.loc[:, by_type.columns.str.contains(";")].sum(axis=1),
            N_EFF: efficient.groupby("Hugo_Symbol").size(),
            "n_stop": stop.groupby("Hugo_Symbol").size(),
            "n_ko": ko.groupby("Hugo_Symbol").size(),
        }
    )
    table = table.reindex(genes).fillna(0).astype(int)
    table["n_start"] = table.filter(like="n_start_").drop(columns=N_EFF).sum(axis=1)
    for kind in ("start", "stop", "ko"):
        table[f"targeted_{kind}"] = table[f"n_{kind}"] > 0
    return table


def main() -> None:
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s1_counts = pd.read_csv(S1_FILE)
    s5_counts = pd.read_csv(S5_FILE)
    s1 = s1_published_gene_lfc(pd.read_csv(S1_DIR / "mageck.neg_HeLa.csv")).add_prefix("S1 ")

    genes, cutoff = s1_fitness_subset(s1_counts, s1)
    print(write_s1_fitness_subset(genes, cutoff), f"{len(genes)} genes, cutoff {cutoff:+.3f}")
    idx = genes.index

    # 1. S1 D4 -> D5
    s1_table = _with_pct(s1, idx)
    s1_table.round(4).to_csv(OUT_DIR / "s1_d4_vs_d5.csv")
    bottom_d5 = s1["S1 D5"].nsmallest(len(idx)).index
    print(
        f"S1 D5: {idx.isin(bottom_d5).sum()}/{len(idx)} still among the {len(idx)} most depleted; "
        f"D5 percentile median {s1_table['S1 D5 pct'].median():.3f}, "
        f"max {s1_table['S1 D5 pct'].max():.3f}"
    )

    d5_set = s1_d5_fitness_subset(s1, idx)
    print(write_s1_d5_fitness_subset(d5_set))
    print(
        f"S1 D5 fitness subset: {len(d5_set)} genes, D5 LFC <= {d5_set['s1_D5_lfc'].max():+.3f}; "
        f"{d5_set[f'in_s1_{DAY.lower()}_fitness_subset'].sum()} also in the {DAY} subset"
    )

    # 2. S5 genome-wide
    s5 = s5_gene_dropout(s5_counts)
    s5_set = s5_fitness_subset(s5_counts, s5["S5 78h"], len(idx), genes)
    print(write_s5_fitness_subset(s5_set, s5["S5 78h"].notna().sum()))
    print(
        f"S5 set: {len(s5_set)} genes, LFC <= {s5_set['s5_78h_lfc'].max():+.3f}; "
        f"{s5_set['in_s1_fitness_subset'].sum()} also in the S1 set; "
        f"{s5_set.index.isin(s1.index).sum()} of them are in the S1 library"
    )
    targeting = s5_counts[s5_counts["category"] == "GENE_TARGETING"]
    s5_table = _with_pct(s5, idx)
    s5_table.insert(0, "n_s5_constructs", targeting.groupby("target_symbol").size().reindex(idx))
    s5_table.insert(0, "in_s5", idx.isin(s5.index))
    s5_table.round(4).to_csv(OUT_DIR / "s5_78h.csv")
    print(
        f"S5: {s5_table['in_s5'].sum()}/{len(idx)} present; 78h percentile median "
        f"{s5_table['S5 78h pct'].median():.3f}; absent: {list(idx[~s5_table['in_s5']])}"
    )

    # 3. ABE library coverage
    targeting_table = abe_targeting(idx)
    targeting_table.to_csv(OUT_DIR / "abe_targeting.csv")
    flags = targeting_table.filter(like="targeted_")
    print("ABE targeted:", flags.sum().to_dict(), f"any: {flags.any(axis=1).sum()}/{len(idx)}")
    print(f"  with >=1 efficient annotated start guide: {(targeting_table[N_EFF] > 0).sum()}")

    # 4. ABE annotated start LFCs
    abe = abe_gene_lfc(MAGECK_DIR, "start", min_eff=MIN_EFF)
    abe_table = _with_pct(abe, idx)
    abe_table.insert(0, N_EFF, targeting_table[N_EFF])
    abe_table.round(4).to_csv(OUT_DIR / "abe_annotated_start.csv")
    abe_set = abe_fitness_subset(abe, len(idx), idx, s5_set.index)
    print(write_abe_fitness_subset(abe_set, abe["ABE D4"].notna().sum()))
    print(
        f"ABE start fitness subset: {len(abe_set)} genes, {DAY} LFC <= "
        f"{abe_set[f'abe_{DAY}_lfc'].max():+.3f}; in S1 subset "
        f"{abe_set['in_s1_fitness_subset'].sum()}, in S5 subset "
        f"{abe_set['in_s5_fitness_subset'].sum()}"
    )
    pct = [f"ABE {d} pct" for d in MAGECK_DAYS]
    print(
        f"ABE annotated start: {abe_table['ABE D4'].notna().sum()} genes scored; median percentile",
        abe_table[pct].median().round(3).to_dict(),
    )


if __name__ == "__main__":
    main()
