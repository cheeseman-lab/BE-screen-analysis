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
    the summary get ``{Hugo_Symbol}_unassigned``). Without one, ``Gene`` is ``Hugo_Symbol`` (KO
    guides). Controls keep their label (``NT_CTRL`` / ``AAVS1_CTRL``).
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


def run_mageck_test(count_file: Path, control_file: Path, day: str, prefix: Path) -> None:
    """``mageck test`` of ``day`` vs D0 (3 reps each), normalised on the listed control guides."""
    reps = lambda d: ",".join(f"{d}-rep{i}" for i in (1, 2, 3))  # noqa: E731
    cmd = [
        "mageck",
        "test",
        "-k",
        str(count_file),
        "-t",
        reps(day),
        "-c",
        reps("D0"),
        "--norm-method",
        "control",
        "--control-sgrna",
        str(control_file),
        "-n",
        str(prefix),
    ]
    subprocess.run(cmd, check=True)


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


def augment_guide_summary(
    guides: pd.DataFrame, per_guide: pd.DataFrame, per_site: pd.DataFrame, screen: str
) -> pd.DataFrame:
    """Left-join MAGeCK guide and site columns onto the guide summary; its columns are unchanged."""
    out = guides.join(per_guide, on="guide_id")
    return out.join(per_site, on=f"{screen}_site_target_id")
