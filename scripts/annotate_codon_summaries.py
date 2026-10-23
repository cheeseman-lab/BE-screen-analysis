"""Tyler's start and stop codon summaries with DepMap and protein-sequence columns appended.

Writes results/annot_summaries/annot_start_codon_summary.csv and annot_stop_codon_summary.csv:
the original rows and columns unchanged (the inputs in data/ are only read), plus

  start sites (once for the annotated start, annot_*, and once for the paired non-canonical
  start, nc_*):
    in_frame_with_cds          the site is in frame with the annotated start codon
    aa_sequence, aa_length     protein from this start to the first in-frame stop; near-cognate
                               starts (CTG, GTG, ...) initiate with Met
    next_atg_dist_aa           amino acids from this start to the next in-frame ATG downstream
                               (the length of lost_aa_sequence; blank: none before the stop)
    lost_aa_sequence           residues from this start up to that ATG: the N-terminus lost if
                               this start is edited away (whole protein when there is no ATG)
  stop sites:
    aa_sequence, aa_length     canonical protein ending at this stop (annotated start to stop)
    edited_stop_residue        residue(s) the site's edits put in place of the stop
                               (mutated_codon of its guides; * = still a stop)
    next_stop_dist_aa          amino acids added by readthrough: the edited stop's residue up to
                               the next in-frame stop downstream, reading through the 3' UTR and
                               past the transcript end into the genome (up to 10 kb)
    added_aa_sequence          residues between the two stops (after the edited stop residue)
    readthrough_past_transcript  the next stop lies beyond the transcript's 3' end
  start sites, per Ribo-TISH condition (HeLa, K562, U2OS, RPE1_Async):
    initiation_efficiency_{condition}  TISCounts / gene RNA-seq counts: TIS-seq reads at the start
                               codon (Ribo-TISH all-tested table, pooled replicates) over the
                               gene's reads in the matched TIS input RNA libraries (HTSeq, both
                               replicates summed). Raw counts, not depth-normalised. Blank: no
                               Ribo-TISH entry at that position, or 0 RNA reads for the gene.
  per gene: depmap_pct_dependent, depmap_fraction (dependent / screened cell lines, "75/1361")
  (dependent = P(dependency) > 0.5 in data/reference/CRISPRGeneDependency.csv).

Sequences: data/reference/Gencode_v49_GRCh38.primary_assembly.genome.fa (+ .fai) and
gencode.v49.primary_assembly.annotation.gtf. The transcript for each site is chosen among the
gene's v49 transcripts (and the library transcript's other versions) that carry reference_codon
at the site, by sequence.pick_transcript: complete CDS on the side the protein needs, site where
expected (annotated start = CDS start, other start in frame, stop = CDS stop), then the library
transcript, MANE Select, Ensembl canonical, longest CDS. Sites with no such transcript are left
blank. Run through Slurm (the GTF is ~3 GB).
"""

from pathlib import Path

import pandas as pd

from be_screen_analysis.depmap import depmap_dependency
from be_screen_analysis.initiation import (
    htseq_gene_counts,
    initiation_efficiency,
    ribotish_tis_counts,
    site_keys,
)
from be_screen_analysis.sequence import (
    CODONS,
    Genome,
    load_transcripts,
    pick_transcript,
    start_site_protein,
    stop_site_readthrough,
)

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "data" / "start_stop_ABE_screen_data" / "summaries"
REF_DIR = ROOT / "data" / "reference"
GENOME = REF_DIR / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
GTF = REF_DIR / "gencode.v49.primary_assembly.annotation.gtf"
DEPMAP_FILE = REF_DIR / "CRISPRGeneDependency.csv"
OUT_DIR = ROOT / "results" / "annot_summaries"
INT_COLS = ["aa_length", "next_atg_dist_aa", "next_stop_dist_aa"]
RIBOTISH_DIR = Path("/lab/barcheese01/mdiberna/ribotish/results/2026-10-09_reprocessed")
RNA_COUNTS_DIR = Path("/lab/barcheese01/aTIS_data/counts")  # matched input RNA, HTSeq
CONDITIONS = ["HeLa", "K562", "U2OS", "RPE1_Async"]  # RPE1 quiescent / senescent left out


def initiation_tables() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Per condition: Ribo-TISH TIS counts per site and summed TIS input RNA counts per gene."""
    manifest = pd.read_csv(RNA_COUNTS_DIR / "manifest.csv")
    out = {}
    for cond in CONDITIONS:
        tis = ribotish_tis_counts(RIBOTISH_DIR / cond / "pooled" / f"{cond}_pooled.candidates.tsv")
        files = manifest.loc[
            (manifest["cell_line"] == cond) & (manifest["condition"] == "TIS"), "filename"
        ]
        out[cond] = (tis, htseq_gene_counts([RNA_COUNTS_DIR / f for f in files]))
    return out


def site_table(
    guides: pd.DataFrame, screen: str, transcripts: dict, genome: Genome
) -> pd.DataFrame:
    """One row per site of the guide summary with the sequence columns."""
    key = f"{screen}_site_target_id"
    sites = guides.drop_duplicates(key).set_index(key)
    edited = guides.groupby(key)["mutated_codon"].agg(
        lambda c: ";".join(sorted({CODONS[x] for x in c}))
    )
    by_gene, by_id = {}, {}
    for tx in transcripts.values():
        by_gene.setdefault(tx.gene, []).append(tx)
        by_id.setdefault(tx.transcript_id.split(".")[0], []).append(tx)
    seqs = {}

    def sequence(tx):
        if tx.transcript_id not in seqs:
            seqs[tx.transcript_id] = tx.sequence(genome)
        return seqs[tx.transcript_id]

    rows = {}
    for site, s in sites.iterrows():
        row = {}
        lib = s["ensembl_transcript_id"]
        candidates = {
            t.transcript_id: t
            for t in by_gene.get(s["Hugo_Symbol"], []) + by_id.get(lib.split(".")[0], [])
        }
        if screen == "stop":
            kind = "stop"
        else:
            kind = "annotated" if "Annotated" in s["start_site_type"] else "start"
        picked = pick_transcript(
            list(candidates.values()),
            s["target_codon_range"],
            s["reference_codon"],
            kind,
            lib,
            sequence,
        )
        if picked:
            tx, i = picked
            seq, mrna_len = sequence(tx)
            cds = tx.cds_index()
            if screen == "start":
                row["in_frame_with_cds"] = cds is not None and (i - cds) % 3 == 0
                row |= start_site_protein(seq, i)
            else:
                row["edited_stop_residue"] = edited[site]
                row |= stop_site_readthrough(seq, i, mrna_len, cds)
        rows[site] = row
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis(key)
    for col in table.columns.intersection(INT_COLS):
        table[col] = table[col].astype("Int64")
    return table


def annotate(
    screen: str,
    transcripts: dict,
    genome: Genome,
    depmap: pd.DataFrame,
    initiation: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_DIR / f"{screen}_codon_summary.csv")
    guides = pd.read_csv(SUMMARY_DIR / f"{screen}_guide_summary.csv")
    sites = site_table(guides, screen, transcripts, genome)
    prefixes = ["annot_", "nc_"] if screen == "start" else [""]
    out = summary
    for p in prefixes:
        out = out.merge(
            sites.add_prefix(p),
            left_on=f"{p}{screen}_site_target_id",
            right_index=True,
            how="left",
        )
    if screen == "start":
        for p in prefixes:
            cols = out[["Chromosome", "strand", f"{p}target_codon_range"]].dropna()
            keys = site_keys(cols.rename(columns={f"{p}target_codon_range": "target_codon_range"}))
            for cond, (tis, rna) in initiation.items():
                eff = initiation_efficiency(keys, tis, rna).to_numpy()
                out.loc[cols.index, f"{p}initiation_efficiency_{cond}"] = eff
    out = out.join(depmap, on="Hugo_Symbol")
    assert len(out) == len(summary)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    depmap = depmap_dependency(DEPMAP_FILE)[["depmap_pct_dependent", "depmap_fraction"]]
    depmap["depmap_pct_dependent"] = depmap["depmap_pct_dependent"].round(2)
    guides = pd.concat(
        [pd.read_csv(SUMMARY_DIR / f"{s}_guide_summary.csv") for s in ("start", "stop")]
    )
    ids = set(guides["ensembl_transcript_id"])
    transcripts = load_transcripts(GTF, ids, set(guides["Hugo_Symbol"]))
    genome = Genome(GENOME)
    exact = sum(t in transcripts for t in ids)
    print(f"{len(ids):,} library transcripts, {exact:,} in GENCODE v49 at the same version")
    print(f"{len(transcripts):,} v49 transcripts loaded")
    initiation = initiation_tables()
    for screen in ("start", "stop"):
        out = annotate(screen, transcripts, genome, depmap, initiation)
        path = OUT_DIR / f"annot_{screen}_codon_summary.csv"
        out.to_csv(path, index=False)
        print(path, out.shape)


if __name__ == "__main__":
    main()
