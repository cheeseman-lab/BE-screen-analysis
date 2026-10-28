"""Rerun MAGeCK test on the raw ABE counts and add per-day metrics to the guide summaries.

Each later timepoint (D4, D6, D12, D21) is compared with D0, normalised on the non-targeting
guides. Guides are grouped by site so MAGeCK's RRA step gives site-level ranks.
"""

import subprocess
from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import BE_CONTROLS, BE_DAYS

MAGECK_DAYS = BE_DAYS[1:]  # every timepoint after D0
NT_GENE = "NT_CTRL"
AAVS1_GENE = "AAVS1_CTRL"
SGRNA_COLS = {
    "LFC": "lfc",
    "p.low": "p_low",
    "p.high": "p_high",
    "p.twosided": "p_twosided",
    "FDR": "fdr",
}
SITE_COLS = {
    "neg|lfc": "site_lfc",
    "neg|rank": "site_neg_rank",
    "neg|fdr": "site_neg_fdr",
    "pos|rank": "site_pos_rank",
    "pos|fdr": "site_pos_fdr",
}


def mageck_count_table(
    counts: pd.DataFrame, guides: pd.DataFrame | None = None, screen: str | None = None
) -> pd.DataFrame:
    """MAGeCK count table from ``load_be_counts`` output.

    With a guide summary, ``Gene`` is the guide's site (codon guides; count guides missing from
    ``guides`` get ``{Hugo_Symbol}_unassigned``, so passing a filtered summary leaves the other
    guides in the run, for the variance model, but out of every site). Without one, ``Gene`` is
    ``Hugo_Symbol`` (KO guides). Controls keep their label (``NT_CTRL`` / ``AAVS1_CTRL``).
    """
    is_control = counts["Gene"].isin(BE_CONTROLS)
    if guides is None:
        gene = counts["Hugo_Symbol"]
    else:
        site = guides.set_index("guide_id")[f"{screen}_site_target_id"]
        gene = counts.index.to_series().map(site).fillna(counts["Hugo_Symbol"] + "_unassigned")
    gene = gene.where(~is_control, counts["Gene"])
    reps = [c for c in counts.columns if "-rep" in c]
    return pd.concat([gene.rename("Gene"), counts[reps]], axis=1).rename_axis("sgRNA")


def mageck_test(
    count_file: Path, prefix: Path, treatment: list[str], control: list[str], control_file: Path
) -> None:
    """``mageck test`` of ``treatment`` vs ``control`` columns, normalised on the listed controls.

    MAGeCK's report templates (.R, .report.Rmd, _summary.Rnw) are deleted; tables and log stay.
    """
    cmd = [
        "mageck",
        "test",
        "-k",
        str(count_file),
        "-t",
        ",".join(treatment),
        "-c",
        ",".join(control),
        "--norm-method",
        "control",
        "--control-sgrna",
        str(control_file),
        "-n",
        str(prefix),
    ]
    subprocess.run(cmd, check=True)
    for report in (f"{prefix}.R", f"{prefix}.report.Rmd", f"{prefix}_summary.Rnw"):
        Path(report).unlink(missing_ok=True)


def run_mageck_test(count_file: Path, control_file: Path, day: str, prefix: Path) -> None:
    """``mageck test`` of ``day`` vs D0 (3 reps each), normalised on the listed control guides."""
    reps = lambda d: [f"{d}-rep{i}" for i in (1, 2, 3)]  # noqa: E731
    mageck_test(count_file, prefix, reps(day), reps("D0"), control_file)


def isoform_count_table(counts: pd.DataFrame, constructs: pd.DataFrame) -> pd.DataFrame:
    """MAGeCK count table grouping constructs (S5 guide pairs, S1 guides) by the starts they KO.

    ``counts`` and ``constructs`` (guide_classify s5_construct_isoforms.csv rows) share an index
    of construct ids. A construct gets one row per start in its ``knocked_out`` (Gene =
    ``{construct_gene}|{start_id}``, since readthrough genes share start ids with their parts;
    sgRNA = ``{construct}|{start_id}``), so each start's LFC is the median over every construct
    that cuts downstream of it. Targeting constructs knocking out no start stay once
    as ``{gene}_no_start_ko`` (in the variance model, in no start); controls (category other than
    GENE_TARGETING) stay once under their ``target_symbol``. Count columns are copied unchanged.
    """
    ko = constructs["knocked_out"].fillna("").str.split(";")
    ko = ko.map(lambda s: [x for x in s if x])
    targeting = constructs["category"] == "GENE_TARGETING"
    gene = constructs["construct_gene"].fillna(constructs["target_symbol"])
    rows = []
    for cid in counts.index:
        if not targeting[cid]:
            rows.append((cid, cid, constructs.at[cid, "target_symbol"]))
        elif ko[cid]:
            rows.extend((cid, f"{cid}|{s}", f"{gene[cid]}|{s}") for s in ko[cid])
        else:
            rows.append((cid, cid, f"{gene[cid]}_no_start_ko"))
    idx = pd.DataFrame(rows, columns=["construct", "sgRNA", "Gene"])
    out = counts.loc[idx["construct"]].set_axis(pd.Index(idx["sgRNA"], name="sgRNA"))
    out.insert(0, "Gene", idx["Gene"].to_numpy())
    return out


def read_mageck(prefix: Path, day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-guide and per-site MAGeCK results, columns prefixed ``mageck_{day}_``.

    Guide ``rank`` is the rank of LFC (1 = most depleted).
    """
    sg = pd.read_csv(f"{prefix}.sgrna_summary.txt", sep="\t").set_index("sgrna")
    per_guide = sg[list(SGRNA_COLS)].rename(columns=SGRNA_COLS)
    per_guide["rank"] = per_guide["lfc"].rank(method="min").astype(int)
    site = pd.read_csv(f"{prefix}.gene_summary.txt", sep="\t").set_index("id")
    per_site = site[list(SITE_COLS)].rename(columns=SITE_COLS)
    return per_guide.add_prefix(f"mageck_{day}_"), per_site.add_prefix(f"mageck_{day}_")


def control_guide_lfc(run_dir: Path, gene: str) -> pd.DataFrame:
    """MAGeCK LFC of every control guide labelled ``gene`` in one run, one column per day.

    ``run_dir`` holds that run's counts.txt and {day}_vs_D0 outputs (e.g. results/abe_mageck/start).
    """
    counts = pd.read_csv(run_dir / "counts.txt", sep="\t", index_col="sgRNA")
    ids = counts.index[counts["Gene"] == gene]
    cols = []
    for day in MAGECK_DAYS:
        g, _ = read_mageck(run_dir / f"{day}_vs_D0", day)
        cols.append(g.loc[ids, f"mageck_{day}_lfc"].rename(day))
    return pd.concat(cols, axis=1)


def site_stats(guides: pd.DataFrame, screen: str, day: str = "D12") -> pd.DataFrame:
    """Per site: MAGeCK site LFC, neg/pos rank and FDR at ``day`` and the number of guides tested.

    ``guides`` is a ``*_guide_summary_mageck*.csv`` table; only guides MAGeCK grouped into the
    site (``mageck_site_member``, all guides if the column is absent) are counted.
    """
    if "mageck_site_member" in guides:
        guides = guides[guides["mageck_site_member"]]
    cols = {f"mageck_{day}_{c}": c.removeprefix("site_") for c in SITE_COLS.values()}
    site = f"{screen}_site_target_id"
    stats = guides.groupby(site)[list(cols)].first().rename(columns=cols)
    stats.insert(0, "num_guides", guides.groupby(site).size())
    return stats


def augment_guide_summary(
    guides: pd.DataFrame, per_guide: pd.DataFrame, per_site: pd.DataFrame, screen: str
) -> pd.DataFrame:
    """Left-join MAGeCK guide and site columns onto the guide summary; its columns are unchanged."""
    out = guides.join(per_guide, on="guide_id")
    return out.join(per_site, on=f"{screen}_site_target_id")


# start codon summary columns that come from MAGeCK and cannot be rebuilt from site_stats
STALE_SITE_COLS = ["neg_lfc", "neg_rra_score", "neg_pval", "pos_lfc", "pos_rra_score", "pos_pval"]


def with_site_stats(summary: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """The start codon summary with each start's MAGeCK columns replaced from ``stats``.

    ``stats`` is ``site_stats`` output (e.g. a rerun on efficient guides only). For both the
    annotated and non-canonical start of every row, num_guides, lfc, neg/pos rank and FDR are
    looked up by site id (NaN: no guide of that site tested); the MAGeCK columns not in ``stats``
    and the nc_pass_* flags are dropped. All other columns are unchanged.
    """
    out = summary.drop(columns=[c for c in summary if c.startswith("nc_pass_")])
    for p in ("annot", "nc"):
        out = out.drop(columns=[f"{p}_{c}" for c in STALE_SITE_COLS if f"{p}_{c}" in out])
        site = out[f"{p}_start_site_target_id"]
        for c in stats.columns:
            out[f"{p}_{c}"] = site.map(stats[c]).to_numpy()
    return out
