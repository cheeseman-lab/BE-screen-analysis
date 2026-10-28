import pandas as pd

from be_screen_analysis.guide_classify import (
    combine_cuts,
    cut_position,
    cut_region,
    gene_scaffold,
    knocked_out,
    main_transcript,
    tx_genomic,
)
from be_screen_analysis.sequence import Genome, Transcript, revcomp


def _genome(tmp_path, seq):
    fa = tmp_path / "g.fa"
    fa.write_text(f">chr1\n{seq}\n")
    (tmp_path / "g.fa.fai").write_text(f"chr1\t{len(seq)}\t6\t{len(seq)}\t{len(seq) + 1}\n")
    return Genome(fa)


def _plus_tx(tags=("MANE_Select",)):
    # exon1 1-30, intron 31-40, exon2 41-80; CDS 11-30 + 41-70 (annotated start at 11)
    tx = Transcript("T1.1", "G", "chr1", 1, set(tags), [(1, 30), (41, 80)], [(11, 30), (41, 70)])
    tx.order()
    return tx


def test_tx_genomic_inverts_tx_index():
    tx = Transcript("t", "G", "chr1", -1, exons=[(15, 20), (5, 10)])
    tx.order()
    assert tx_genomic(tx, 0) == 20 and tx_genomic(tx, 6) == 10
    assert tx_genomic(tx, 12) is None
    plus = _plus_tx()
    assert all(plus.tx_index(tx_genomic(plus, i)) == i for i in range(70))


def test_cut_position_plus_needs_ngg(tmp_path):
    proto = "ACGTACGTACGTACGTACGA"
    g = _genome(tmp_path, "CCCCC" + proto + "TGG" + "A" * 10)
    assert cut_position(g, "chr1", 6, 1) == 22  # break between 22 and 23 (3 bp 5' of PAM)
    assert cut_position(g, "chr1", 7, 1) is None


def test_cut_position_minus(tmp_path):
    proto = "ACGTACGTACGTACGTACGA"
    g = _genome(tmp_path, "CCA" + revcomp(proto) + "A" * 10)
    assert cut_position(g, "chr1", 4, -1) == 6  # break between 6 and 7


def test_cut_region_plus():
    tx = _plus_tx()
    assert cut_region(tx, [tx], 5) == "5'UTR"
    assert cut_region(tx, [tx], 45) == "CDS"
    assert cut_region(tx, [tx], 30) == "CDS_splice_site"
    assert cut_region(tx, [tx], 35) == "intron"
    assert cut_region(tx, [tx], 75) == "3'UTR"
    assert cut_region(None, [tx], 35) == "noncoding_intron"


def test_main_transcript_prefers_mane_then_canonical():
    mane, canon = _plus_tx(), _plus_tx(("Ensembl_canonical",))
    canon.transcript_id = "T2.1"
    frag = _plus_tx(("MANE_Select", "cds_start_NF"))
    assert main_transcript([canon, mane, frag]) is mane
    assert main_transcript([canon, frag]) is canon
    assert main_transcript([]) is None


def _scaffold(strand, starts):
    rows = [
        dict(kind=k, strand=strand, first_base=f, orf_end=e, start_id=k[0] + str(f))
        for k, f, e in starts
    ]
    return pd.DataFrame(rows)


def test_knocked_out_plus():
    s = _scaffold(1, [("Annotated", 11, pd.NA), ("Truncated", 23, pd.NA), ("Internal", 24, 42)])
    assert knocked_out(s, 10).tolist() == [False, False, False]
    assert knocked_out(s, 11).tolist() == [True, False, False]  # cut inside the start codon
    assert knocked_out(s, 30).tolist() == [True, True, True]
    assert knocked_out(s, 50).tolist() == [True, True, False]  # past the internal ORF's stop


def test_knocked_out_minus():
    s = _scaffold(-1, [("Annotated", 90, pd.NA), ("Internal", 80, 75), ("Truncated", 60, pd.NA)])
    assert knocked_out(s, 90).tolist() == [False, False, False]
    assert knocked_out(s, 89).tolist() == [True, False, False]
    assert knocked_out(s, 77).tolist() == [True, True, False]
    assert knocked_out(s, 74).tolist() == [True, False, False]
    assert knocked_out(s, 59).tolist() == [True, False, True]


def test_gene_scaffold_reclassifies_and_merges(tmp_path):
    seq = list("A" * 80)
    seq[10:13] = "ATG"  # annotated start 11-13
    seq[22:25] = "CTG"  # in-frame downstream start 23-25
    g = _genome(tmp_path, "".join(seq))
    tx = _plus_tx()
    maf = pd.DataFrame(
        [
            ("Annotated", 11, 13, "ATG", "T1.1", "hela"),  # main start: merged
            ("Annotated", 23, 25, "CTG", "T9.1", "ipsc"),  # in frame downstream: truncated
            ("Annotated", 24, 26, "TGA", "T8.1", "hff"),  # out of frame: dropped
            ("Truncated", 23, 25, "CTG", "T1.1", "rpe1"),  # same codon: merged
        ],
        columns=["source_type", "lo", "hi", "codon", "transcript", "provenance"],
    ).assign(chrom="chr1", strand=1)
    empty = pd.DataFrame(columns=["chrom", "strand", "lo", "hi", "codon", "transcript"])
    s = gene_scaffold("G", tx, maf, empty.assign(provenance=[], orf_end=[]), g)
    assert s.start_id.tolist() == ["Annotated_ATG_chr1:11", "Truncated_CTG_chr1:23"]
    assert s.provenance.tolist() == ["gencode,hela", "ipsc,rpe1"]
    assert s.in_frame_main.all()


ORDER = {"A": 0, "T1": 1, "I1": 2, "T2": 3}


def test_combine_cuts_keeps_only_starts_past_the_last_cut():
    early = (["A"], ["T1", "I1", "T2"])
    late = (["A", "T1", "I1"], ["T2"])
    assert combine_cuts([late, early], ORDER) == (["A", "T1", "I1"], ["T2"])


def test_combine_cuts_noncoding_spacer_adds_nothing():
    coding = (["A", "T1"], ["I1", "T2"])
    intronic = ([], ["A", "T1", "I1", "T2"])
    assert combine_cuts([intronic, coding], ORDER) == (["A", "T1"], ["I1", "T2"])


def test_combine_cuts_internal_orf_lost_to_either_cut():
    # first cut inside the internal ORF, second cut past its stop
    assert combine_cuts([(["A", "T1", "I1"], ["T2"]), (["A", "T1", "T2"], ["I1"])], ORDER) == (
        ["A", "T1", "I1", "T2"],
        [],
    )


def test_combine_cuts_unplaced_spacers():
    assert combine_cuts([([], []), ([], [])], ORDER) == ([], [])
