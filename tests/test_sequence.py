import numpy as np

from be_screen_analysis.sequence import (
    Genome,
    Transcript,
    pick_transcript,
    site_index,
    start_site_protein,
    stop_site_readthrough,
    translate,
)


def test_translate():
    assert translate("ATGGCTTAAG") == "MA*"


def test_genome_fetch_across_lines(tmp_path):
    fa = tmp_path / "g.fa"
    fa.write_text(">chr1\nACGTA\nCGTAC\nGT\n")
    (tmp_path / "g.fa.fai").write_text("chr1\t12\t6\t5\t6\n")
    g = Genome(fa)
    assert g.fetch("chr1", 4, 8) == "TACGT"
    assert g.fetch("chr1", 11, 20) == "GT"


def test_minus_strand_split_codon():
    # exons in transcript order on -: 20-15 then 10-5; codon spanning 16,15 | 10
    tx = Transcript("t", "G", "chr1", -1, exons=[(15, 20), (5, 10)])
    tx.order()
    assert tx.tx_index(20) == 0
    assert tx.tx_index(10) == 6
    assert tx.tx_index(12) is None
    assert site_index(tx, "10;16") == 4  # 5' base on - strand is the higher coordinate


def test_start_site_protein():
    # CTG start, in-frame ATG at codon 2, stop at codon 4
    out = start_site_protein("CCCTGAAAATGGGGTAG", 2)
    assert out["aa_sequence"] == "MKMG"
    assert out["next_atg_dist_aa"] == 2
    assert out["lost_aa_sequence"] == "MK"
    none = start_site_protein("CTGAAATAG", 0)
    assert np.isnan(none["next_atg_dist_aa"]) and none["lost_aa_sequence"] == "MK"


def test_stop_readthrough_past_transcript():
    seq = "ATGAAATGAGGGCCCTAA"  # mRNA ends after CCC (len 15); next stop in the genome tail
    out = stop_site_readthrough(seq, 6, mrna_len=15, cds_start=0)
    assert out["aa_sequence"] == "MK"
    assert out["next_stop_dist_aa"] == 3
    assert out["added_aa_sequence"] == "GP"
    assert out["readthrough_past_transcript"]


def test_pick_transcript_prefers_complete_placed():
    # same site (genomic 1-3, ATG); "lib" is a fragment (cds_end_NF), "mane" has its CDS start there
    seq = ("ATGAAATAG", 9)
    lib = Transcript("T1.2", "G", "chr1", 1, {"cds_end_NF"}, [(1, 9)], [(1, 6)])
    mane = Transcript("T2.1", "G", "chr1", 1, {"MANE_Select"}, [(1, 9)], [(1, 6)])
    tx, i = pick_transcript([lib, mane], "1;3", "ATG", "annotated", "T1.1", lambda t: seq)
    assert (tx.transcript_id, i) == ("T2.1", 0)
    assert pick_transcript([lib], "1;3", "CTG", "annotated", "T1.2", lambda t: seq) is None
