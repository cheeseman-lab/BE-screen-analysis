"""Isoform-aware MAGeCK on the shake-off library counts: abundance, last remaining vs time 0.

  s5  genome-wide dual-guide library, 78h_remaining vs 0h; a construct is target +
      target_version with both cuts made (guide_classify results/s5_construct_isoforms.csv)
  s1  HeLa essential library, day5_remaining vs Day0 (its last time point); a construct is one
      guide (guide_classify results/s1_guide_isoforms.csv)
  k562  K562 anti-mitotic genome-wide screen (data/anti-mt_K562), untreated endpoint
      un.neg.tggtca vs initial k562.initial.attggc; a construct is one guide
      (guide_classify results/k562_guide_isoforms.csv)

Every construct carries the starts it knocks out. Two MAGeCK runs per library, both normalised
on control constructs (constructs are duplicated in the isoform run, so median normalisation
would shift): non-targeting for S5 and S1, intergenic for K562 (see K562_NORM):
  gene     one row per construct, Gene = construct_gene (the published-style gene LFC)
  isoform  one row per (construct, knocked-out start), Gene = {gene}|{start_id}: a start's LFC is
           the median over every construct cutting downstream of it (mageck.isoform_count_table)

Outputs in isoform_aware_shakeoff_analysis/results/{library}/:
  gene/, isoform/        counts.txt, control_sgrnas.txt and MAGeCK tables + log
  gene_isoform_lfc.csv   one row per (gene, non-annotated start), genes with only an annotated
                         start one row with blank alt_ columns: gene_lfc / gene_neg_fdr (gene
                         run); annot_ and alt_ start_id, num_constructs (constructs knocking the
                         start out), lfc, neg_fdr, pos_fdr (isoform run; NaN when no construct
                         knocks it out); alt_kind, alt_codon, alt_provenance, alt_in_frame_main;
                         alt_ko_constructs; same_as_annotated (alt knocked out by exactly the
                         annotated start's constructs, so its LFC is identical); delta_lfc
                         (alt_lfc - annot_lfc)

Run guide_classify first; submit from the repo root with
  sbatch isoform_aware_shakeoff_analysis/isoform_mageck.sbatch s5|s1|k562
"""

import sys
from pathlib import Path

import pandas as pd

from be_screen_analysis.filtering import (
    S1_DAY0,
    S1_NT,
    S1_REMAINING,
    S5_DAY0,
    S5_NT,
    S5_REMAINING,
)
from be_screen_analysis.mageck import isoform_count_table, mageck_test, read_mageck

ROOT = Path(__file__).resolve().parents[1]
SHAKEOFF_DIR = ROOT / "data" / "shakeoff_data"
K562_FILE = ROOT / "data" / "anti-mt_K562" / "GW_counts.csv"
# K562 is normalised on its intergenic (cutting, no gene) guides: cutting alone depletes K562
# guides by ~0.6 log2 vs non-targeting, so a non-targeting baseline calls almost every gene a hit
K562_NORM = "INTERGENIC"
K562_NT = "CONTROL"  # non-targeting; stays as its own control group
K562_CONTROLS = {K562_NORM, K562_NT, "CTRL", "LIBCTRL0"}  # guides with no target gene
CLASSIFY_DIR = ROOT / "guide_classify" / "results"
OUT_DIR = ROOT / "isoform_aware_shakeoff_analysis" / "results"
# per library: time 0 and the last remaining time point
SAMPLES = {
    "s5": (S5_DAY0, list(S5_REMAINING)[-1]),  # 0h, 78h_remaining
    "s1": (S1_DAY0, list(S1_REMAINING)[-1]),  # Day0, day5_remaining
    "k562": ("k562.initial.attggc", "un.neg.tggtca"),  # initial, untreated endpoint
}
TAG = "abundance"
STAT_COLS = {
    f"mageck_{TAG}_site_lfc": "lfc",
    f"mageck_{TAG}_site_neg_fdr": "neg_fdr",
    f"mageck_{TAG}_site_pos_fdr": "pos_fdr",
}


def load_s5() -> tuple[pd.DataFrame, pd.DataFrame]:
    """S5 counts (0h, 78h_remaining) and construct isoform labels, indexed by construct id."""
    key = ["target", "target_version"]
    counts = pd.read_csv(SHAKEOFF_DIR / "table_s5" / "sgRNA_sequences_and_counts.csv",
                         usecols=[*key, *SAMPLES["s5"]])  # fmt: skip
    labels = pd.read_csv(CLASSIFY_DIR / "s5_construct_isoforms.csv")
    cid = lambda d: d["target"].astype(str) + "_" + d["target_version"].astype(str)  # noqa: E731
    counts = counts.set_index(cid(counts))[list(SAMPLES["s5"])]
    labels = labels.set_index(cid(labels)).loc[counts.index]
    labels["control"] = labels["target"] == S5_NT
    return counts, labels


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


def load_s1() -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1 HeLa counts (Day0, day5_remaining; blank = 0 reads) and guide isoform labels."""
    counts = pd.read_csv(SHAKEOFF_DIR / "table_s1" / "HeLa_sgRNA_sequences_and_counts.csv",
                         usecols=["sgrna", *SAMPLES["s1"]], index_col="sgrna")  # fmt: skip
    counts = counts.fillna(0)  # blank = no reads (32 guides), as in filtering.mageck_guide_lfc
    guides = pd.read_csv(CLASSIFY_DIR / "s1_guide_isoforms.csv", index_col="sgrna")
    guides = guides.loc[counts.index]
    nt = guides["gene"] == S1_NT
    return counts, single_guide_labels(guides, nt, nt)


def load_k562() -> tuple[pd.DataFrame, pd.DataFrame]:
    """K562 counts (initial, untreated endpoint) and guide isoform labels, intergenic-normalised."""
    counts = pd.read_csv(K562_FILE, usecols=["sgrna.name", *SAMPLES["k562"]],
                         index_col="sgrna.name").rename_axis("sgrna")  # fmt: skip
    guides = pd.read_csv(
        CLASSIFY_DIR / "k562_guide_isoforms.csv", index_col="sgrna", low_memory=False
    )
    guides = guides.loc[counts.index]
    control = guides["gene"].isin(K562_CONTROLS)
    labels = single_guide_labels(guides, control, guides["gene"] == K562_NT)
    labels["control"] = guides["gene"] == K562_NORM
    return counts, labels


def run(lib: str, name: str, table: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Write the count table and NT control list, run MAGeCK, return its per-group stats."""
    run_dir = OUT_DIR / lib / name
    run_dir.mkdir(parents=True, exist_ok=True)
    count_file, control_file = run_dir / "counts.txt", run_dir / "control_sgrnas.txt"
    table.to_csv(count_file, sep="\t")
    control_file.write_text("".join(f"{c}\n" for c in labels.index[labels["control"]]))
    day0, last = SAMPLES[lib]
    prefix = run_dir / f"{last}_vs_{day0}"
    mageck_test(count_file, prefix, [last], [day0], control_file)
    _, per_group = read_mageck(prefix, TAG)
    return per_group[list(STAT_COLS)].rename(columns=STAT_COLS)


def gene_table(counts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per construct, Gene = construct_gene (controls: their target_symbol)."""
    targeting = labels["category"] == "GENE_TARGETING"
    gene = labels["construct_gene"].where(targeting).fillna(labels["target_symbol"])
    return pd.concat([gene.rename("Gene"), counts], axis=1).rename_axis("sgRNA")


def isoform_lfc(labels: pd.DataFrame, gene_stats: pd.DataFrame, iso: pd.DataFrame) -> pd.DataFrame:
    """The gene_isoform_lfc.csv table (see module docstring)."""
    scaffold = pd.read_csv(CLASSIFY_DIR / "start_scaffold.csv")
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


def main(lib: str) -> None:  # noqa: D103
    loaders = {"s5": load_s5, "s1": load_s1, "k562": load_k562}
    if lib not in loaders:
        raise SystemExit("usage: run_isoform_mageck.py s5|s1|k562")
    counts, labels = loaders[lib]()
    gene_stats = run(lib, "gene", gene_table(counts, labels), labels)
    iso = run(lib, "isoform", isoform_count_table(counts, labels), labels)
    out = isoform_lfc(labels, gene_stats, iso)
    path = OUT_DIR / lib / "gene_isoform_lfc.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} rows, {out['gene'].nunique()} genes -> {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
