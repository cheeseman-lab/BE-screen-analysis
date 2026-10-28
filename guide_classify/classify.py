"""Which start-codon isoforms a CRISPR KO guide knocks out, from where Cas9 cuts in the gene.

Each gene gets a scaffold of start codons: the annotated start (GENCODE v49 MANE Select, else
Ensembl canonical, else longest complete CDS), in-frame truncated starts and internal
out-of-frame ORFs. A cut disrupts every ORF that starts upstream of it and runs through it: all
annotated / truncated starts upstream of the cut (they share the annotated stop), and internal
ORFs only when the cut lies before their own stop. Starts downstream of the cut are retained.

Coordinates are 1-based inclusive. A cut is stored as ``cut_pos``: the break lies between genomic
bases ``cut_pos`` and ``cut_pos + 1``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from be_screen_analysis.sequence import STOPS, Genome, Transcript, revcomp

CODING_REGIONS = {"CDS", "CDS_splice_site"}  # cut regions where a frameshift disrupts ORFs


def strand_int(strand: str) -> int:
    """+1 / -1 from "+" / "-"."""
    return 1 if strand == "+" else -1


def load_gene_spans(gtf: Path) -> pd.DataFrame:
    """One row per GTF gene: gene_name, gene_id, gene_type, chrom, start, end, strand (+1/-1)."""
    rows = []
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = line.split("\t", 8)
            if feature != "gene" or "_PAR_Y" in attrs:
                continue
            rows.append(
                (
                    _attr(attrs, "gene_name"),
                    _attr(attrs, "gene_id"),
                    _attr(attrs, "gene_type"),
                    chrom,
                    int(start),
                    int(end),
                    strand_int(strand),
                )
            )
    cols = ["gene_name", "gene_id", "gene_type", "chrom", "start", "end", "strand"]
    return pd.DataFrame(rows, columns=cols)


def _attr(attrs: str, key: str) -> str:
    k = attrs.find(f'{key} "') + len(key) + 2
    return attrs[k : attrs.find('"', k)]


def main_transcript(txs: list[Transcript]) -> Transcript | None:
    """The gene's reference coding transcript: MANE Select, Ensembl canonical, longest CDS.

    Only transcripts with a CDS complete at the 5' end are considered.
    """
    coding = [t for t in txs if t.cds and "cds_start_NF" not in t.tags]
    if not coding:
        return None
    return min(
        coding,
        key=lambda t: (
            "MANE_Select" not in t.tags,
            "Ensembl_canonical" not in t.tags,
            -t.cds_length,
            t.transcript_id,
        ),
    )


def tx_genomic(tx: Transcript, idx: int) -> int | None:
    """Genomic position of 0-based spliced-transcript index ``idx`` (inverse of ``tx_index``)."""
    if idx < 0:
        return None
    for start, end in tx.exons:
        length = end - start + 1
        if idx < length:
            return start + idx if tx.strand > 0 else end - idx
        idx -= length
    return None


def codon_bounds(tx: Transcript, idx: int) -> tuple[int, int] | None:
    """(lo, hi) genomic span of the codon whose first base is transcript index ``idx``."""
    pos = [tx_genomic(tx, idx + k) for k in range(3)]
    if None in pos:
        return None
    return min(pos), max(pos)


def codon_seq(genome: Genome, tx: Transcript, idx: int) -> str:
    """Coding-strand codon at transcript index ``idx``."""
    bases = []
    for k in range(3):
        pos = tx_genomic(tx, idx + k)
        if pos is None:
            return ""
        base = genome.fetch(tx.chrom, pos, pos)
        bases.append(base if tx.strand > 0 else revcomp(base))
    return "".join(bases)


def first_base(lo: int, hi: int, strand: int) -> int:
    """5' (gene-orientation) base of a codon spanning lo..hi."""
    return lo if strand > 0 else hi


def start_id(kind: str, codon: str, chrom: str, lo: int, hi: int, strand: int) -> str:
    """Readable start label, e.g. ``Truncated_CTG_chr1:959217`` (5' base of the codon)."""
    return f"{kind}_{codon}_{chrom}:{first_base(lo, hi, strand)}"


def maf_starts(maf: pd.DataFrame, tx_gene: dict[str, str]) -> pd.DataFrame:
    """Annotated and truncated starts from the combined swissisoform MAF.

    ``gene`` is the GENCODE v49 gene of the row's transcript (bare ID) when known, else the MAF
    Hugo_Symbol. Extended and uORF rows are dropped.
    """
    m = maf[maf["Type"].isin(["Annotated", "Truncated"])]
    bare = m["Transcript_ID"].str.split(".").str[0]
    return pd.DataFrame(
        {
            "gene": [tx_gene.get(b, h) for b, h in zip(bare, m["Hugo_Symbol"])],
            "chrom": m["Chromosome"].to_numpy(),
            "strand": m["Strand"].map(strand_int).to_numpy(),
            "lo": m["Start_Position"].to_numpy(),
            "hi": m["End_Position"].to_numpy(),
            "codon": m["Reference_Allele"].to_numpy(),
            "source_type": m["Type"].to_numpy(),
            "transcript": m["Transcript_ID"].to_numpy(),
            "provenance": m["Provenance"].to_numpy(),
        }
    )


def ly_internal_starts(path: Path, cell_line: str) -> pd.DataFrame:
    """Internal (out-of-frame, inside the CDS) starts from a Ly Ribo-TISH TIS dictionary.

    The dictionary (``Alldata_uniqueTIS_filtered_dictionary.txt``) has one row per position
    ``chr:pos:strand`` with comma-separated transcripts, ORF DNA sequences and Ribo-TISH types.
    Position conversion follows swissisoform ``process_ly_hela.sh``: + strand codon pos+1..pos+3,
    - strand pos-2..pos.
    """
    cols = ["pos", "gene_id", "transcripts", "symbol", "codon", "tis_type", "dna", "protein"]
    d = pd.read_csv(path, sep="\t", header=None, names=cols)
    d = d[d["tis_type"].str.contains("Internal")]
    parts = d["pos"].str.split(":", expand=True)
    pos = parts[1].astype(int)
    plus = parts[2] == "+"
    return pd.DataFrame(
        {
            "chrom": parts[0],
            "strand": parts[2].map(strand_int),
            "lo": np.where(plus, pos + 1, pos - 2),
            "hi": np.where(plus, pos + 3, pos),
            "codon": d["codon"],
            "symbol": d["symbol"],
            "tis_type": d["tis_type"],
            "transcripts": d["transcripts"].str.split(","),
            "orf_lengths": d["dna"].str.split(",").map(lambda s: [len(x) for x in s]),
            "provenance": cell_line,
        }
    ).reset_index(drop=True)


def internal_orf_stop(
    row: pd.Series, txs_by_bare: dict[str, Transcript], genome: Genome
) -> tuple[Transcript, int] | None:
    """(transcript, genomic position of the ORF's last base) for an internal start, or None.

    The ORF (start through stop codon) is laid along the first listed transcript found in v49
    that carries the start codon at the site and a stop codon at the ORF's end.
    """
    for tid, length in zip(row["transcripts"], row["orf_lengths"]):
        tx = txs_by_bare.get(tid.split(".")[0])
        if tx is None:
            continue
        i = tx.tx_index(first_base(row["lo"], row["hi"], row["strand"]))
        if i is None or codon_seq(genome, tx, i) != row["codon"]:
            continue
        last = i + length - 1
        if codon_seq(genome, tx, last - 2) not in STOPS:
            continue
        end = tx_genomic(tx, last)
        if end is not None:
            return tx, end
    return None


def gene_scaffold(
    gene: str,
    main: Transcript | None,
    maf_rows: pd.DataFrame,
    internal_rows: pd.DataFrame,
    genome: Genome,
) -> pd.DataFrame:
    """Start scaffold of one gene: annotated, truncated and internal starts.

    Columns: gene, start_id, kind, chrom, strand, lo, hi, codon, first_base, orf_end (internal
    only: genomic last base of its stop codon), transcript, provenance, in_frame_main.

    MAF ``Annotated`` rows of other transcripts become truncated starts when they are downstream
    of and in frame with the main CDS; other annotated rows are dropped. Duplicate codons are
    merged (provenance and transcripts joined).
    """
    rows = []
    if main is not None:
        c = main.cds_index()
        lo, hi = codon_bounds(main, c)
        rows.append(
            dict(
                kind="Annotated",
                chrom=main.chrom,
                strand=main.strand,
                lo=lo,
                hi=hi,
                codon=codon_seq(genome, main, c),
                transcript=main.transcript_id,
                provenance="gencode",
                orf_end=pd.NA,
            )
        )
    for r in maf_rows.itertuples():
        pos = first_base(r.lo, r.hi, r.strand)
        frame = _in_frame_downstream(main, pos)
        if r.source_type == "Annotated":
            if main is not None and pos == main.cds_first_base:
                rows[0]["provenance"] = _join(rows[0]["provenance"], r.provenance)
                continue
            if not frame:
                continue
        rows.append(
            dict(
                kind="Truncated",
                chrom=r.chrom,
                strand=r.strand,
                lo=r.lo,
                hi=r.hi,
                codon=r.codon,
                transcript=r.transcript,
                provenance=r.provenance,
                orf_end=pd.NA,
            )
        )
    for r in internal_rows.itertuples():
        rows.append(
            dict(
                kind="Internal",
                chrom=r.chrom,
                strand=r.strand,
                lo=r.lo,
                hi=r.hi,
                codon=r.codon,
                transcript=r.transcript,
                provenance=r.provenance,
                orf_end=r.orf_end,
            )
        )
    cols = ["kind", "chrom", "strand", "lo", "hi", "codon", "transcript", "provenance", "orf_end"]
    if not rows:
        return pd.DataFrame(columns=["gene", "start_id", "first_base", "in_frame_main", *cols])
    s = pd.DataFrame(rows, columns=cols)
    s = (
        s.groupby(["kind", "chrom", "strand", "lo", "hi", "codon"], sort=False, dropna=False)
        .agg(
            transcript=("transcript", lambda x: ",".join(dict.fromkeys(x))),
            provenance=("provenance", lambda x: _join(*x)),
            orf_end=("orf_end", "first"),
        )
        .reset_index()
    )
    s.insert(0, "gene", gene)
    s["first_base"] = [first_base(lo, hi, st) for lo, hi, st in zip(s.lo, s.hi, s.strand)]
    ids = [start_id(*v) for v in zip(s.kind, s.codon, s.chrom, s.lo, s.hi, s.strand)]
    s.insert(1, "start_id", ids)
    s["in_frame_main"] = [
        k == "Annotated" or _in_frame_downstream(main, p) for k, p in zip(s.kind, s.first_base)
    ]
    order = s["first_base"] * s["strand"]
    return s.assign(_o=order).sort_values("_o").drop(columns="_o").reset_index(drop=True)


def _in_frame_downstream(main: Transcript | None, pos: int) -> bool:
    if main is None:
        return False
    i, c = main.tx_index(pos), main.cds_index()
    return i is not None and c < i < c + main.cds_length and (i - c) % 3 == 0


def _join(*values: str) -> str:
    seen = dict.fromkeys(v for value in values for v in str(value).split(",") if v)
    return ",".join(seen)


def parse_sam(path: Path) -> pd.DataFrame:
    """Mapped alignments from a bowtie2 SAM: guide, chrom, pos (leftmost), strand (+1/-1)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("@"):
                continue
            qname, flag, chrom, pos = line.split("\t", 4)[:4]
            flag = int(flag)
            if flag & 4:
                continue
            rows.append((qname, chrom, int(pos), -1 if flag & 16 else 1))
    return pd.DataFrame(rows, columns=["guide", "chrom", "pos", "guide_strand"])


def cut_position(genome: Genome, chrom: str, pos: int, guide_strand: int) -> int | None:
    """``cut_pos`` of a 20-nt protospacer aligned at ``pos`` (leftmost base), None without NGG.

    SpCas9 cuts 3 bp 5' of the PAM: + strand protospacer pos..pos+19 with PAM pos+20..pos+22
    breaks between pos+16 and pos+17; - strand (PAM CCN at pos-3..pos-1) between pos+2 and pos+3.
    """
    if pam_seq(genome, chrom, pos, guide_strand)[1:] != "GG":
        return None
    return pos + 16 if guide_strand > 0 else pos + 2


def pam_seq(genome: Genome, chrom: str, pos: int, guide_strand: int) -> str:
    """The 3 nt 3' of a 20-nt protospacer aligned at ``pos``, read on the guide strand."""
    if chrom not in genome.index:
        return ""
    if guide_strand > 0:
        return genome.fetch(chrom, pos + 20, pos + 22)
    return revcomp(genome.fetch(chrom, pos - 3, pos - 1))


def cut_region(main: Transcript | None, txs: list[Transcript], cut_pos: int) -> str:
    """Where a cut falls in the gene's main transcript.

    CDS / CDS_splice_site (break at a coding exon edge) / UTR_splice_site / 5'UTR / 3'UTR /
    intron; noncoding_exon / noncoding_intron (exon of any transcript or not) if the gene has no
    complete coding transcript.
    """
    if main is None:
        exonic = any(t.tx_index(cut_pos) is not None for t in txs)
        return "noncoding_exon" if exonic else "noncoding_intron"
    a, b = main.tx_index(cut_pos), main.tx_index(cut_pos + 1)
    c = main.cds_index()
    cds_end = c + main.cds_length + 3  # through the stop codon
    if a is None and b is None:
        return "intron"
    if a is None or b is None:
        i = a if a is not None else b
        return "CDS_splice_site" if c <= i < cds_end else "UTR_splice_site"
    i = min(a, b)  # 5' base of the break in transcript order
    if i < c:
        return "5'UTR"
    if i + 1 >= cds_end:
        return "3'UTR"
    return "CDS"


def knocked_out(starts: pd.DataFrame, cut_pos: int) -> pd.Series:
    """Per start: does a break between ``cut_pos`` and ``cut_pos + 1`` disrupt its ORF.

    True when the break lies downstream of the codon's 5' base (a cut inside the start codon
    counts); internal ORFs additionally need the break upstream of their last base.
    """
    plus = starts["strand"] > 0
    downstream = (cut_pos >= starts["first_base"]).where(plus, cut_pos < starts["first_base"])
    end = pd.to_numeric(starts["orf_end"], errors="coerce")
    before_end = (cut_pos < end).where(plus, cut_pos >= end)
    internal = starts["kind"] == "Internal"
    return downstream & (~internal | before_end.fillna(False).astype(bool))


def combine_cuts(
    spacers: list[tuple[list[str], list[str]]], order: dict[str, int]
) -> tuple[list[str], list[str]]:
    """Starts knocked out / retained when all of a construct's cuts are made.

    ``spacers`` holds each cut's (knocked_out, retained) start IDs (both empty for a cut with no
    gene); a start is lost if any cut knocks it out and kept only if every cut retains it.
    Both lists come back in ``order`` (start ID -> scaffold rank, 5' to 3').
    """
    ko = {s for lost, _ in spacers for s in lost}
    kept = {s for _, retained in spacers for s in retained} - ko
    return sorted(ko, key=order.__getitem__), sorted(kept, key=order.__getitem__)
