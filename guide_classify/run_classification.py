"""Start-codon isoforms knocked out / retained by each S1 / S5 shake-off and K562 KO guide.

Two steps (see classify.sbatch):
  python guide_classify/run_classification.py fasta     -> results/guides.fa (unique spacers)
  bowtie2 exact end-to-end alignment, every hit, to the
  full UCSC hg38 (alt haplotypes + unplaced contigs)    -> results/guides.sam
  python guide_classify/run_classification.py classify  -> the CSVs below
  (python guide_classify/run_classification.py summary     -> cut_region_summary.csv alone;
   python guide_classify/run_classification.py constructs  -> s5_construct_isoforms.csv alone)

Outputs in guide_classify/results/:
  start_scaffold.csv   one row per (gene, start) for every gene a guide was assigned to:
                       start_id, kind (Annotated / Truncated / Internal), chrom, strand, codon
                       lo/hi, codon, first_base (5' base), orf_end (internal ORFs: last base of
                       their stop codon), transcript(s), provenance (gencode and/or the cell lines
                       hela / rpe1 / hff / ipsc), in_frame_main (in frame with the main CDS)
  s1_guide_isoforms.csv  one row per S1 guide (essential library)
  s5_guide_isoforms.csv  one row per S5 spacer (genome-wide library; spacer = 1 or 2)
  k562_guide_isoforms.csv  one row per K562 anti-mitotic screen guide (genome-wide library,
                       data/anti-mt_K562/GW_counts.csv; sgrna = its sgrna.name)
  s5_construct_isoforms.csv  one row per S5 construct (target + target_version), both of its
                       cuts assumed made: spacer_1 / spacer_2 (sequences), gtf_gene_1/2 (each
                       spacer's gene), construct_gene (the gene the cuts are combined on; see
                       write_constructs), cut_region_1/2, cut_pos_1/2,
                       downstream_cut_pos (most downstream coding cut along the gene strand),
                       knocked_out (starts knocked out by either cut), retained (starts no cut
                       knocks out: downstream of the last coding cut, internal ORFs not spanning
                       a cut), any_retained
  cut_region_summary.csv targeting guides per cut_region group (REGION_GROUPS): n and % of the
                       library's targeting guides

Guide columns: gtf_gene (GENCODE v49 gene the cut was assigned to: the library symbol when it is a
v49 gene name and a hit lies in it, else the gene overlapping the hit, protein-coding first;
primary-chromosome hits are tried before alt / unplaced contigs), chrom, guide_strand, cut_pos
(break between cut_pos and cut_pos + 1), n_perfect_hits (genomic exact matches with an NGG PAM,
alt / unplaced contigs included), cut_region (classify.cut_region, or for guides with no gene:
no_alignment (no exact match), no_pam (exact matches, none with an NGG PAM), alt_contig_only
(PAM hits only on alt / unplaced contigs, which v49 does not annotate), no_hit_in_gene (PAM hits
on primary chromosomes, none in a gene usable for the symbol)), in_maf (gene has starts in the
swissisoform MAF), knocked_out
and retained (";"-separated start_ids, ordered 5' to 3'). Starts are knocked out only by cuts in
the CDS or at a coding splice site; any other region leaves every start retained. Control guides
keep blank isoform columns.

Starts: annotated = GENCODE v49 main transcript CDS start (classify.main_transcript); truncated
= swissisoform combined MAF Truncated rows plus its other-transcript Annotated rows in frame and
downstream of the main start; internal = Ribo-TISH Internal* rows of the Ly HeLa and RPE1 TIS
dictionaries, stop from the ORF sequence laid on its v49 transcript. Run through Slurm (the GTF
is ~3 GB).
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from classify import (
    CODING_REGIONS,
    combine_cuts,
    cut_position,
    cut_region,
    gene_scaffold,
    internal_orf_stop,
    knocked_out,
    load_gene_spans,
    ly_internal_starts,
    maf_starts,
    main_transcript,
    parse_sam,
)

from be_screen_analysis.sequence import Genome, load_transcripts

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "data" / "reference"
GENOME = REF_DIR / "Gencode_v49_GRCh38.primary_assembly.genome.fa"
GTF = REF_DIR / "gencode.v49.primary_assembly.annotation.gtf"
# the genome bowtie2 aligns to (UCSC contig names); PAMs are read from it
ALIGN_GENOME = Path("/nfs/genomes/human_hg38_dec13/fasta_whole_genome/hg38_all.fa")
S1_FILE = ROOT / "data" / "shakeoff_data" / "table_s1" / "HeLa_sgRNA_sequences_and_counts.csv"
S5_FILE = ROOT / "data" / "shakeoff_data" / "table_s5" / "sgRNA_sequences_and_counts.csv"
K562_FILE = ROOT / "data" / "anti-mt_K562" / "GW_counts.csv"
# K562 guides without a target gene: non-targeting, intergenic (cutting) and library controls
K562_CONTROLS = {"CONTROL", "INTERGENIC", "CTRL", "LIBCTRL0"}
RIBO_DIR = Path("/lab/barcheese01/mdiberna/swissisoform/data/ribosome_profiling")
MAF = RIBO_DIR / "combined_isoforms_with_transcripts_maf.tsv"
LY_DICTS = {
    "hela": RIBO_DIR / "ly_raw_hela" / "Alldata_uniqueTIS_filtered_dictionary.txt",
    "rpe1": RIBO_DIR / "ly_raw_rpe1" / "Alldata_uniqueTIS_filtered_dictionary.txt",
}
OUT_DIR = ROOT / "guide_classify" / "results"
FASTA = OUT_DIR / "guides.fa"
SAM = OUT_DIR / "guides.sam"
S1_ID_COLS = ["sgrna", "gene"]
S5_ID_COLS = ["target", "target_symbol", "target_source", "target_version", "category", "spacer"]
K562_ID_COLS = ["sgrna", "gene"]
LIBRARIES = ("s1", "s5", "k562")
# cut_region -> group for cut_region_summary.csv
REGION_GROUPS = {
    "5'UTR": "5'UTR",
    "CDS": "inside_gene",
    "CDS_splice_site": "inside_gene",
    "intron": "inside_gene",
    "noncoding_exon": "inside_gene",
    "noncoding_intron": "inside_gene",
    "UTR_splice_site": "UTR_splice_site",
    "3'UTR": "3'UTR",
    "no_hit_in_gene": "outside_target_gene",
    "alt_contig_only": "outside_target_gene",
    "no_pam": "unmapped",
    "no_alignment": "unmapped",
}
GUIDE_COLS = [
    "sequence",
    "gtf_gene",
    "chrom",
    "guide_strand",
    "cut_pos",
    "n_perfect_hits",
    "cut_region",
    "in_maf",
    "knocked_out",
    "retained",
]


def load_libraries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S1 guides, S5 spacers and K562 guides (one row each) with ``symbol`` and ``control``."""
    s1 = pd.read_csv(S1_FILE, usecols=["sequence", "sgrna", "gene"])
    s1["symbol"] = s1["gene"]
    s1["control"] = s1["gene"] == "nontargeting"
    cols = ["target", "target_symbol", "target_source", "target_version", "category"]
    s5 = pd.read_csv(S5_FILE, usecols=[*cols, "spacer_1", "spacer_2"])
    s5 = (
        s5.melt(id_vars=cols, value_vars=["spacer_1", "spacer_2"], var_name="spacer")
        .rename(columns={"value": "sequence"})
        .assign(spacer=lambda d: d["spacer"].str[-1].astype(int))
        .sort_values(["target", "target_version", "spacer"], kind="stable")
        .reset_index(drop=True)
    )
    s5["symbol"] = s5["target_symbol"]
    s5["control"] = s5["category"] != "GENE_TARGETING"
    k562 = pd.read_csv(K562_FILE, usecols=["sgrna.name", "gene", "sgrna.sequence"])
    k562 = k562.rename(columns={"sgrna.name": "sgrna", "sgrna.sequence": "sequence"})
    k562["symbol"] = k562["gene"]
    k562["control"] = k562["gene"].isin(K562_CONTROLS)
    return s1, s5, k562


def write_fasta(*libs: pd.DataFrame) -> None:
    """Unique targeting spacers of every library, each named by its own sequence."""
    seqs = pd.concat([lib.loc[~lib.control, "sequence"] for lib in libs]).unique()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FASTA.write_text("".join(f">{s}\n{s}\n" for s in sorted(seqs)))
    print(f"{len(seqs)} unique spacers -> {FASTA}")


def perfect_hits(genome: Genome) -> tuple[pd.DataFrame, set[str]]:
    """bowtie2 hits with an NGG PAM (cut_pos, n_perfect_hits, alt), and every aligned spacer.

    ``alt`` marks alt-haplotype / unplaced contigs (UCSC names contain "_"); within a spacer,
    primary-chromosome hits sort first.
    """
    hits = parse_sam(SAM)
    aligned = set(hits.guide)
    hits["cut_pos"] = [
        cut_position(genome, c, p, s) for c, p, s in zip(hits.chrom, hits.pos, hits.guide_strand)
    ]
    hits = hits.dropna(subset=["cut_pos"]).astype({"cut_pos": int})
    hits["n_perfect_hits"] = hits.groupby("guide")["guide"].transform("size")
    hits["alt"] = hits.chrom.str.contains("_")
    hits = hits.sort_values(["guide", "alt", "chrom", "cut_pos"]).reset_index(drop=True)
    return hits, aligned


def assign_genes(
    guides: pd.DataFrame, hits: pd.DataFrame, aligned: set[str], spans: pd.DataFrame
) -> pd.DataFrame:
    """One hit per targeting guide: chrom, guide_strand, cut_pos, n_perfect_hits, gtf_gene.

    The hit inside the library gene (its v49 gene name) wins; symbols that are not v49 names
    take the gene overlapping their first hit (protein-coding first). Guides left without a gene
    get ``cut_region`` no_alignment / no_pam / alt_contig_only / no_hit_in_gene.
    """
    by_name = {g: d for g, d in spans.groupby("gene_name")}
    by_chrom = {c: d for c, d in spans.groupby("chrom")}
    hits_by_seq = {s: d for s, d in hits.groupby("guide")}
    rows = []
    for seq, symbol, control in zip(guides.sequence, guides.symbol, guides.control):
        h = hits_by_seq.get(seq)
        if control or h is None:
            missing = "no_pam" if seq in aligned else "no_alignment"
            rows.append({"cut_region": None if control else missing})
            continue
        gene, hit = None, None
        if symbol in by_name:
            for r in h.itertuples():
                s = by_name[symbol]
                if ((s.chrom == r.chrom) & (s.start <= r.cut_pos) & (r.cut_pos < s.end)).any():
                    gene, hit = symbol, r
                    break
        else:
            for r in h.itertuples():
                s = by_chrom.get(r.chrom)
                if s is None:
                    continue
                s = s[(s.start <= r.cut_pos) & (r.cut_pos < s.end)]
                if len(s):
                    s = s.assign(pc=s.gene_type != "protein_coding").sort_values(
                        ["pc", "gene_name"]
                    )
                    gene, hit = s.gene_name.iloc[0], r
                    break
        if hit is None:
            hit = next(h.itertuples())
        rows.append(
            {
                "gtf_gene": gene,
                "chrom": hit.chrom,
                "guide_strand": "+" if hit.guide_strand > 0 else "-",
                "cut_pos": hit.cut_pos,
                "n_perfect_hits": hit.n_perfect_hits,
                "cut_region": None
                if gene
                else ("alt_contig_only" if h.alt.all() else "no_hit_in_gene"),
            }
        )
    out = pd.DataFrame(rows, index=guides.index)
    return out.astype({"cut_pos": "Int64", "n_perfect_hits": "Int64"})


def place_internal(raw: pd.DataFrame, transcripts: dict, genome: Genome) -> pd.DataFrame:
    """Internal starts with gene, transcript and orf_end from their v49 transcript.

    Starts whose ORF cannot be laid on a listed transcript (missing from v49, codon or stop
    mismatch) are dropped.
    """
    by_bare = {t.split(".")[0]: tx for t, tx in transcripts.items()}
    rows = []
    for _, r in raw.iterrows():
        found = internal_orf_stop(r, by_bare, genome)
        if found is not None:
            tx, end = found
            rows.append({**r, "gene": tx.gene, "transcript": tx.transcript_id, "orf_end": end})
    print(f"internal starts placed on v49 transcripts: {len(rows)} of {len(raw)}")
    return pd.DataFrame(rows, columns=[*raw.columns, "gene", "transcript", "orf_end"])


def classify(genome: Genome) -> None:
    """Assign cut sites, build the start scaffolds and write the guide CSVs."""
    s1, s5, k562 = load_libraries()
    spans = load_gene_spans(GTF)
    hits, aligned = perfect_hits(Genome(ALIGN_GENOME))
    for lib in (s1, s5, k562):
        assigned = assign_genes(lib, hits, aligned, spans)
        lib[list(assigned)] = assigned
    genes = set(s1.gtf_gene.dropna()) | set(s5.gtf_gene.dropna()) | set(k562.gtf_gene.dropna())

    maf = pd.read_csv(MAF, sep="\t")
    raw_internal = pd.concat(
        [ly_internal_starts(path, cell) for cell, path in LY_DICTS.items()], ignore_index=True
    )
    ids = set(maf.Transcript_ID) | {t for ts in raw_internal.transcripts for t in ts}
    transcripts = load_transcripts(GTF, ids, genes)
    tx_gene = {t.split(".")[0]: tx.gene for t, tx in transcripts.items()}
    by_gene: dict[str, list] = {}
    for tx in transcripts.values():
        by_gene.setdefault(tx.gene, []).append(tx)

    starts = maf_starts(maf, tx_gene)
    internal = place_internal(raw_internal, transcripts, genome)
    maf_genes = set(starts.gene) | set(maf.Hugo_Symbol)
    starts_by_gene = dict(list(starts.groupby("gene")))
    internal_by_gene = dict(list(internal.groupby("gene")))
    scaffolds, mains = {}, {}
    for g in sorted(genes):
        mains[g] = main_transcript(by_gene.get(g, []))
        scaffolds[g] = gene_scaffold(
            g,
            mains[g],
            starts_by_gene.get(g, starts.iloc[:0]),
            internal_by_gene.get(g, internal.iloc[:0]),
            genome,
        )
    pd.concat(scaffolds.values()).to_csv(OUT_DIR / "start_scaffold.csv", index=False)

    libs = ((s1, S1_ID_COLS, "s1"), (s5, S5_ID_COLS, "s5"), (k562, K562_ID_COLS, "k562"))
    for lib, id_cols, name in libs:
        ko, kept, regions = [], [], []
        for g, cut, region in zip(lib.gtf_gene, lib.cut_pos, lib.cut_region):
            if not isinstance(g, str):
                ko.append(None)
                kept.append(None)
                regions.append(region)
                continue
            region = cut_region(mains[g], by_gene.get(g, []), int(cut))
            s = scaffolds[g]
            hit = (
                knocked_out(s, int(cut)) if region in CODING_REGIONS else pd.Series(False, s.index)
            )
            ko.append(";".join(s.start_id[hit]))
            kept.append(";".join(s.start_id[~hit]))
            regions.append(region)
        lib["cut_region"], lib["knocked_out"], lib["retained"] = regions, ko, kept
        lib["in_maf"] = lib.gtf_gene.isin(maf_genes).where(~lib.control)
        out = OUT_DIR / f"{name}_guide_isoforms.csv"
        lib[id_cols + GUIDE_COLS].to_csv(out, index=False)
        print(f"{name}: {len(lib)} rows -> {out}")
    write_region_summary()
    write_constructs()


def write_constructs() -> None:
    """S5 constructs with both cuts made: starts lost to either cut, starts left after both.

    Each construct is anchored to one gene (``construct_gene``): the gene most of its spacers
    were assigned to, ties going to a gene with a coding cut, then to spacer 1. Only spacers on
    that gene are combined; a spacer assigned elsewhere or unplaced adds nothing.
    """
    s5 = pd.read_csv(OUT_DIR / "s5_guide_isoforms.csv")
    scaffold = pd.read_csv(OUT_DIR / "start_scaffold.csv")
    order = {s: i for i, s in enumerate(scaffold.start_id)}
    strand = scaffold.drop_duplicates("gene").set_index("gene").strand
    split = lambda v: v.split(";") if isinstance(v, str) else []  # noqa: E731
    constructs: dict[tuple, list[dict]] = {}
    for r in s5.sort_values("spacer", kind="stable").to_dict("records"):
        constructs.setdefault((r["target"], r["target_version"]), []).append(r)
    rows = []
    for spacers in constructs.values():
        first = spacers[0]
        row = {k: first[k] for k in ("target", "target_symbol", "target_source", "category")}
        row["target_version"] = first["target_version"]
        for r in spacers:
            i = r["spacer"]
            row |= {
                f"spacer_{i}": r["sequence"],
                f"gtf_gene_{i}": r["gtf_gene"],
                f"cut_region_{i}": r["cut_region"],
                f"cut_pos_{i}": r["cut_pos"],
            }
        placed = [r for r in spacers if isinstance(r["gtf_gene"], str)]
        if first["category"] != "GENE_TARGETING" or not placed:
            rows.append(row)
            continue
        votes = Counter(r["gtf_gene"] for r in placed)
        coding = {r["gtf_gene"] for r in placed if r["cut_region"] in CODING_REGIONS}
        gene = min(
            votes,
            key=lambda g: (
                -votes[g],
                g not in coding,
                min(r["spacer"] for r in placed if r["gtf_gene"] == g),
            ),
        )
        on = [r for r in placed if r["gtf_gene"] == gene]
        ko, kept = combine_cuts(
            [(split(r["knocked_out"]), split(r["retained"])) for r in on], order
        )
        cuts = [r["cut_pos"] for r in on if r["cut_region"] in CODING_REGIONS]
        down = None
        if cuts and gene in strand.index:
            down = max(cuts) if strand[gene] > 0 else min(cuts)
        row |= {
            "construct_gene": gene,
            "downstream_cut_pos": down,
            "knocked_out": ";".join(ko),
            "retained": ";".join(kept),
            "any_retained": bool(kept),
        }
        rows.append(row)
    cols = [
        "target",
        "target_symbol",
        "target_source",
        "target_version",
        "category",
        "spacer_1",
        "spacer_2",
        "gtf_gene_1",
        "gtf_gene_2",
        "construct_gene",
        "cut_region_1",
        "cut_region_2",
        "cut_pos_1",
        "cut_pos_2",
        "downstream_cut_pos",
        "knocked_out",
        "retained",
        "any_retained",
    ]
    out = pd.DataFrame(rows).reindex(columns=cols)
    out = out.astype({c: "Int64" for c in ["cut_pos_1", "cut_pos_2", "downstream_cut_pos"]})
    out.to_csv(OUT_DIR / "s5_construct_isoforms.csv", index=False)
    print(f"s5 constructs: {len(out)} rows -> {OUT_DIR / 's5_construct_isoforms.csv'}")


def write_region_summary() -> None:
    """Targeting guides per region group (REGION_GROUPS), n and % for each library."""
    groups = list(dict.fromkeys(REGION_GROUPS.values()))
    table = pd.DataFrame({"group": groups})
    for name in LIBRARIES:
        regions = pd.read_csv(OUT_DIR / f"{name}_guide_isoforms.csv").cut_region.dropna()
        n = regions.map(REGION_GROUPS).value_counts().reindex(groups, fill_value=0)
        table[f"{name}_n"] = n.values
        table[f"{name}_pct"] = (100 * n / len(regions)).round(2).values
    table.to_csv(OUT_DIR / "cut_region_summary.csv", index=False)
    print(f"cut region summary -> {OUT_DIR / 'cut_region_summary.csv'}")


def main(step: str) -> None:  # noqa: D103
    if step == "fasta":
        write_fasta(*load_libraries())
    elif step == "classify":
        classify(Genome(GENOME))
    elif step == "summary":
        write_region_summary()
    elif step == "constructs":
        write_constructs()
    else:
        raise SystemExit("usage: run_classification.py fasta|classify|summary|constructs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
