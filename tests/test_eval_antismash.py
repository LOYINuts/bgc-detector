import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.eval.eval_antismash import (
    binary_variants,
    map_by_coordinates,
    parse_antismash,
)

GENBANK = """LOCUS       TESTREC                  200 bp    DNA     linear   BCT 01-JAN-2020
FEATURES             Location/Qualifiers
     source          1..200
                     /organism="Testus testus"
     region          10..150
                     /region_number="1"
                     /product="other"
     CDS             20..40
                     /locus_tag="t1"
                     /gene_kind="biosynthetic"
                     /translation="MK"
     CDS             50..70
                     /locus_tag="t2"
                     /gene_kind="biosynthetic-additional"
                     /translation="MK"
     CDS             80..100
                     /locus_tag="t3"
                     /gene_kind="transport"
                     /translation="MK"
     CDS             120..140
                     /locus_tag="t4"
                     /translation="MK"
     CDS             160..180
                     /locus_tag="t5"
                     /gene_kind="biosynthetic"
                     /translation="MK"
ORIGIN
        1 {seq}
//
"""


def write_genbank(text: str) -> str:
    seq = "acgt" * 50
    lines = [seq[i : i + 60] for i in range(0, 200, 60)]
    body = "\n".join(f"{i * 60 + 1:9d} {line}" for i, line in enumerate(lines))
    handle = tempfile.NamedTemporaryFile("w", suffix=".gbk", delete=False)
    handle.write(text.replace("{seq}", body))
    handle.close()
    return handle.name


class AntiSMASHTierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = write_genbank(GENBANK)
        self.parsed = parse_antismash([self.path])
        self.addCleanup(Path(self.path).unlink)

    def test_tiers_follow_gene_kind_inside_regions_only(self) -> None:
        scores = self.parsed["scores"]
        self.assertEqual(scores["t1"], 0.9)
        self.assertEqual(scores["t2"], 0.8)
        self.assertEqual(scores["t3"], 0.7)
        self.assertEqual(scores["t4"], 0.6)
        self.assertNotIn(
            "t5", scores
        )  # gene_kind outside every region is no cluster call

    def test_region_members_stay_above_the_binary_cut(self) -> None:
        self.assertTrue(all(score > 0.5 for score in self.parsed["scores"].values()))

    def test_region_membership_is_coordinate_based(self) -> None:
        self.assertEqual(self.parsed["in_region"], {"t1", "t2", "t3", "t4"})

    def test_diagnostics_count(self) -> None:
        diagnostics = self.parsed["diagnostics"]
        self.assertEqual(diagnostics["n_regions"], 1)
        self.assertEqual(diagnostics["n_cds"], 5)
        self.assertEqual(diagnostics["gene_kind_counts"]["biosynthetic"], 2)


class AntiSMASHCoordinateMapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.genes = [
            {
                "gene_id": "A:g1",
                "accession": "A",
                "locus_tag": "g1",
                "start": 100,
                "end": 400,
            },
            {
                "gene_id": "A:g2",
                "accession": "A",
                "locus_tag": "g2",
                "start": 500,
                "end": 800,
            },
            {
                "gene_id": "A:g3",
                "accession": "A",
                "locus_tag": "g3",
                "start": 900,
                "end": 1200,
            },
        ]

    def test_overlapping_cds_transfers_tier(self) -> None:
        scores = map_by_coordinates(self.genes, [("A.1", 110, 395, 0.9)])
        self.assertEqual(scores, {"A:g1": 0.9})

    def test_short_cds_inside_long_gene_still_matches(self) -> None:
        scores = map_by_coordinates(self.genes, [("A.1", 600, 700, 0.7)])
        self.assertEqual(scores, {"A:g2": 0.7})

    def test_highest_tier_wins_per_gene(self) -> None:
        scores = map_by_coordinates(
            self.genes, [("A.1", 100, 250, 0.6), ("A.1", 260, 400, 0.9)]
        )
        self.assertEqual(scores, {"A:g1": 0.9})

    def test_uncovered_gene_and_other_accession_skipped(self) -> None:
        scores = map_by_coordinates(
            self.genes, [("B.1", 100, 400, 0.9), ("A.1", 1000, 1200, 0.0)]
        )
        self.assertEqual(scores, {})


class AntiSMASHBinaryVariantTest(unittest.TestCase):
    def test_variants_use_their_own_cut(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.6, 0.7, 0.0])
        in_region = np.array([1, 1, 1, 0])
        blocks = binary_variants(y_true, scores, in_region)
        self.assertEqual(blocks["in_region"]["n_predicted"], 3)
        self.assertEqual(blocks["in_protocluster"]["n_predicted"], 2)
        self.assertEqual(blocks["core_biosynthetic_only"]["n_predicted"], 1)
        self.assertEqual(blocks["core_biosynthetic_only"]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
