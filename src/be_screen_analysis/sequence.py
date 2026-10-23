"""Transcript sequences from a genome FASTA + GTF, and the proteins made from start/stop sites."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BASES = "TCAG"
AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODONS = {
    a + b + c: AMINO[16 * i + 4 * j + k]
    for i, a in enumerate(BASES)
    for j, b in enumerate(BASES)
    for k, c in enumerate(BASES)
}
STOPS = {c for c, aa in CODONS.items() if aa == "*"}
READTHROUGH_NT = 10_000  # genomic sequence read past a transcript's 3' end for stop readthrough


def revcomp(seq: str) -> str:
    """Reverse complement."""
    return seq[::-1].translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))


def translate(seq: str) -> str:
    """Codon-by-codon translation of ``seq`` (trailing partial codon dropped; unknown -> X)."""
    return "".join(CODONS.get(seq[i : i + 3], "X") for i in range(0, len(seq) - 2, 3))


class Genome:
    """Random access to an indexed FASTA (.fai alongside), 1-based inclusive coordinates."""

    def __init__(self, fasta: Path):  # noqa: D107
        self.handle = open(fasta, "rb")  # noqa: SIM115
        self.index = {}
        for line in Path(f"{fasta}.fai").read_text().splitlines():
            name, length, offset, line_bases, line_width = line.split("\t")[:5]
            self.index[name] = tuple(map(int, (length, offset, line_bases, line_width)))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Bases ``start``-``end`` of ``chrom`` (clipped to the chromosome), upper case."""
        length, offset, line_bases, line_width = self.index[chrom]
        start, end = max(start, 1), min(end, length)
        if end < start:
            return ""
        byte = lambda pos: offset + (pos - 1) // line_bases * line_width + (pos - 1) % line_bases  # noqa: E731
        self.handle.seek(byte(start))
        raw = self.handle.read(byte(end) - byte(start) + 1)
        return raw.replace(b"\n", b"").replace(b"\r", b"").decode().upper()


@dataclass
class Transcript:
    """One transcript: exons in transcript (5'->3') order, 1-based inclusive genomic coordinates."""

    transcript_id: str
    gene: str
    chrom: str
    strand: int  # +1 / -1
    tags: set[str] = field(default_factory=set)
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)

    def order(self) -> None:
        """Sort exons and CDS pieces 5' to 3' along the transcript."""
        self.exons.sort(key=lambda e: e[0] * self.strand)
        self.cds.sort(key=lambda e: e[0] * self.strand)

    @property
    def cds_first_base(self) -> int | None:
        """Genomic position of the first coding base (5' end of the annotated start codon)."""
        if not self.cds:
            return None
        return self.cds[0][0] if self.strand > 0 else self.cds[0][1]

    @property
    def cds_length(self) -> int:
        """Coding length in nt, stop codon excluded (GENCODE CDS rows leave it out)."""
        return sum(end - start + 1 for start, end in self.cds)

    def cds_index(self) -> int | None:
        """Transcript index of the first coding base."""
        return self.tx_index(self.cds_first_base) if self.cds else None

    def tx_index(self, pos: int) -> int | None:
        """0-based position of genomic base ``pos`` in the spliced transcript (None if intronic)."""
        before = 0
        for start, end in self.exons:
            if start <= pos <= end:
                return before + (pos - start if self.strand > 0 else end - pos)
            before += end - start + 1
        return None

    def sequence(self, genome: Genome, downstream: int = READTHROUGH_NT) -> tuple[str, int]:
        """Spliced mRNA plus ``downstream`` nt of genome past its 3' end; (sequence, mRNA length).

        Coding strand; the genomic tail lets a stop readthrough run off the transcript.
        """
        parts = [genome.fetch(self.chrom, s, e) for s, e in self.exons]
        if self.strand > 0:
            mrna = "".join(parts)
            last = self.exons[-1][1]
            tail = genome.fetch(self.chrom, last + 1, last + downstream)
        else:
            mrna = "".join(revcomp(p) for p in parts)
            last = self.exons[-1][0]
            tail = revcomp(genome.fetch(self.chrom, last - downstream, last - 1))
        return mrna + tail, len(mrna)


def load_transcripts(gtf: Path, ids: set[str], genes: set[str]) -> dict[str, Transcript]:
    """Exon and CDS structure of the GTF transcripts in ``ids`` (any version) or of ``genes``.

    Keyed by the GTF's versioned ID; ``tags`` holds the GTF tags (MANE_Select, cds_start_NF, ...).
    """
    bare = {i.split(".")[0] for i in ids}
    out: dict[str, Transcript] = {}
    with open(gtf) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = line.split("\t", 8)
            if feature not in ("exon", "CDS"):
                continue
            tid = _attr(attrs, "transcript_id")
            if "_PAR_Y" in tid:
                continue
            if tid not in out:
                gene = _attr(attrs, "gene_name")
                if tid.split(".")[0] not in bare and gene not in genes:
                    continue
                tags = set(re.findall(r'tag "([^"]+)"', attrs))
                out[tid] = Transcript(tid, gene, chrom, 1 if strand == "+" else -1, tags)
            tx = out[tid]
            (tx.exons if feature == "exon" else tx.cds).append((int(start), int(end)))
    for tx in out.values():
        tx.order()
    return out


def _attr(attrs: str, key: str) -> str:
    k = attrs.find(f'{key} "') + len(key) + 2
    return attrs[k : attrs.find('"', k)]


def pick_transcript(
    candidates: list[Transcript],
    codon_range: str,
    reference_codon: str,
    kind: str,
    library_id: str,
    sequence: Callable[[Transcript], tuple[str, int]],
) -> tuple[Transcript, int] | None:
    """Best transcript for a site: (transcript, site index), or None.

    ``kind`` is "annotated" (annotated start), "start" (other start) or "stop". Candidates must
    contain the site with ``reference_codon`` there. Preference, in order: a complete CDS on the
    side the protein needs (3' end for starts, 5' end for stops); the site placed as expected
    (annotated start = CDS start, other start in frame with the CDS, stop = CDS stop); the
    library's transcript (exact version first); MANE Select; Ensembl canonical; longest CDS.
    """
    ranked = []
    for tx in candidates:
        i = site_index(tx, codon_range)
        if i is None or sequence(tx)[0][i : i + 3] != reference_codon:
            continue
        c = tx.cds_index()
        if kind == "annotated":
            placed = c == i
        elif kind == "start":
            placed = c is not None and (i - c) % 3 == 0
        else:
            placed = c is not None and i == c + tx.cds_length
        complete = bool(tx.cds) and (
            "cds_end_NF" not in tx.tags if kind != "stop" else "cds_start_NF" not in tx.tags
        )
        same = tx.transcript_id.split(".")[0] == library_id.split(".")[0]
        key = (
            not complete,
            not placed,
            not same,
            tx.transcript_id != library_id,
            "MANE_Select" not in tx.tags,
            "Ensembl_canonical" not in tx.tags,
            -tx.cds_length,
        )
        ranked.append((key, tx.transcript_id, tx, i))
    if not ranked:
        return None
    _, _, tx, i = min(ranked, key=lambda r: r[:2])
    return tx, i


def site_index(tx: Transcript, codon_range: str) -> int | None:
    """Transcript index of a codon's first base; ``codon_range`` is "lo;hi" genomic."""
    lo, hi = map(int, codon_range.split(";"))
    return tx.tx_index(lo if tx.strand > 0 else hi)


def next_codon(seq: str, i: int, codons: set[str], stop_at_stop: bool = False) -> int | None:
    """Codons from ``i`` to the next in-frame codon in ``codons`` after it.

    None if there is none; with ``stop_at_stop``, none is looked for past the first in-frame stop.
    """
    for j in range(i + 3, len(seq) - 2, 3):
        codon = seq[j : j + 3]
        if codon in codons:
            return (j - i) // 3
        if stop_at_stop and codon in STOPS:
            return None
    return None


def start_site_protein(seq: str, i: int) -> dict:
    """Protein, next in-frame ATG and lost N-terminus for a start codon at ``i``.

    The protein runs to the first in-frame stop, with an initiator Met for near-cognate starts;
    the lost N-terminus runs from the start up to the next in-frame ATG.
    """
    stop = next_codon(seq, i, STOPS)
    if not stop:
        return dict.fromkeys(
            ["aa_sequence", "aa_length", "next_atg_dist_aa", "lost_aa_sequence"], np.nan
        )
    protein = "M" + translate(seq[i + 3 : i + 3 * stop])
    atg = next_codon(seq, i, {"ATG"}, stop_at_stop=True)
    lost = protein[:atg] if atg else protein
    return {
        "aa_sequence": protein,
        "aa_length": len(protein),
        "next_atg_dist_aa": atg if atg else np.nan,
        "lost_aa_sequence": lost,
    }


def stop_site_readthrough(seq: str, i: int, mrna_len: int, cds_start: int | None) -> dict:
    """Protein ending at the stop at ``i``, and the readthrough to the next in-frame stop.

    The protein runs from the annotated CDS start; the next stop is looked for past the mRNA end
    into the genome if needed, and the added residues are those between the two stops.
    """
    in_frame = cds_start is not None and (i - cds_start) % 3 == 0 and i >= cds_start
    protein = translate(seq[cds_start:i]) if in_frame else np.nan
    nxt = next_codon(seq, i, STOPS)
    return {
        "aa_sequence": protein,
        "aa_length": len(protein) if in_frame else np.nan,
        "next_stop_dist_aa": nxt if nxt else np.nan,
        "added_aa_sequence": translate(seq[i + 3 : i + 3 * nxt]) if nxt else np.nan,
        "readthrough_past_transcript": (i + 3 * nxt + 3 > mrna_len) if nxt else np.nan,
    }
