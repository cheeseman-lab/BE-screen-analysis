"""Translation initiation efficiency per start site: Ribo-TISH TIS counts over gene RNA counts."""

from pathlib import Path

import numpy as np
import pandas as pd

TIS_COLUMNS = ["Gid", "GenomePos", "TISCounts"]


def ribotish_tis_counts(path: Path) -> pd.DataFrame:
    """TIS read counts per start site from a Ribo-TISH table (e.g. *.candidates.tsv).

    Index (chrom, pos, strand) with ``pos`` the 5' base of the start codon, 1-based (GenomePos is
    0-based start / 1-based end); columns Gid (gene, versioned) and TISCounts. A site listed once
    per transcript keeps one row (its count is per position).
    """
    tis = pd.read_csv(path, sep="\t", usecols=TIS_COLUMNS)
    pos = tis["GenomePos"].str.extract(r"^(chr[^:]+):(\d+)-(\d+):([+-])$")
    tis = tis[pos[0].notna()]
    pos = pos[pos[0].notna()]
    start, end = pos[1].astype(int), pos[2].astype(int)
    tis = tis.assign(chrom=pos[0], pos=np.where(pos[3] == "+", start + 1, end), strand=pos[3])
    tis = tis.groupby(["chrom", "pos", "strand"]).agg(
        Gid=("Gid", "first"), TISCounts=("TISCounts", "max")
    )
    return tis


def htseq_gene_counts(paths: list[Path]) -> pd.Series:
    """Gene read counts summed over HTSeq count files (``__`` summary rows dropped)."""
    counts = [pd.read_csv(p, sep="\t", header=None, index_col=0).iloc[:, 0] for p in paths]
    total = pd.concat(counts, axis=1).sum(axis=1)
    return total[~total.index.str.startswith("__")]


def site_keys(sites: pd.DataFrame) -> pd.MultiIndex:
    """(chrom, pos, strand) keys from Chromosome, strand (+1/-1) and codon range "lo;hi"."""
    lo_hi = sites["target_codon_range"].str.split(";", expand=True).astype(int)
    plus = sites["strand"] > 0
    return pd.MultiIndex.from_arrays(
        [
            "chr" + sites["Chromosome"].astype(str),
            np.where(plus, lo_hi[0], lo_hi[1]),
            np.where(plus, "+", "-"),
        ],
        names=["chrom", "pos", "strand"],
    )


def initiation_efficiency(keys: pd.MultiIndex, tis: pd.DataFrame, rna: pd.Series) -> pd.Series:
    """TISCounts / gene RNA-seq counts per site (NaN: site not in ``tis`` or gene with 0 reads)."""
    hit = tis.reindex(keys)
    gene_rna = rna.reindex(hit["Gid"]).to_numpy()
    eff = hit["TISCounts"].to_numpy() / np.where(gene_rna > 0, gene_rna, np.nan)
    return pd.Series(eff, index=keys)
