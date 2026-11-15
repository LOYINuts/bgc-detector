import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from bgc_gnn.dataset import _source_runs
from scripts.audit_antismash import audit_assembly
from scripts.audit_antismash import main as audit_main
from scripts.build_catalog import build_catalog, read_genes
from scripts.build_pretraining_data import main as build_pretraining_main


def record(identifier):
    r = SeqRecord(Seq("ATG" * 100), id=identifier, annotations={"molecule_type": "DNA"})
    r.features = [
        SeqFeature(
            FeatureLocation(i * 30, i * 30 + 27, strand=1),
            type="CDS",
            qualifiers={"translation": ["M" + "A" * (i + 2)]},
        )
        for i in range(4)
    ]
    return r


class CatalogTest(unittest.TestCase):
    def test_direct_genbank_exclusions_coordinates_and_contig_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mibig = root / "mibig"
            mibig.mkdir()
            sources = root / "sources"
            sources.mkdir()
            assembly = sources / "assembly"
            assembly.mkdir()
            SeqIO.write(record("BGC0001"), mibig / "BGC0001.gbk", "genbank")
            a, b = record("a"), record("b")
            a.features.append(SeqFeature(FeatureLocation(30, 57), type="region"))
            SeqIO.write([a, b], assembly / "whole.gbk", "genbank")
            rows, index = read_genes(assembly / "whole.gbk", "gtdb", "assembly", True)
            self.assertNotIn("gtdb:assembly:1", index["gene_ids"])
            self.assertEqual(
                [len(run) for run in _source_runs(index, "gtdb")], [1, 2, 4]
            )
            self.assertEqual(rows[-1]["contig"], "b")
            build_catalog(mibig, sources, root / "catalog")
            table = pq.read_table(root / "catalog/gene_catalog.parquet")
            self.assertEqual(len(table), 11)
            self.assertEqual(
                set(table.column_names),
                {"gene_id", "target_aa", "strand", "contig", "start", "end"},
            )
            self.assertNotIn(
                "class_vector",
                pq.read_table(root / "catalog/bgc_index.parquet").column_names,
            )
            self.assertEqual(
                json.loads((root / "catalog/metadata.json").read_text())["status"],
                "completed",
            )
            with self.assertRaisesRegex(ValueError, "exists"):
                build_catalog(mibig, sources, root / "catalog")


class AssemblyAuditTest(unittest.TestCase):
    def test_audit_report_feeds_pretraining_builder_without_external_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for i in range(2):
                folder = source / f"assembly{i}"
                folder.mkdir()
                r = record(f"contig{i}")
                SeqIO.write(r, folder / "whole.gbk", "genbank")
                (folder / "whole.json").write_text(
                    json.dumps(
                        {
                            "version": "8",
                            "records": [{"id": r.id, "seq": str(r.seq)}],
                        }
                    )
                )
            ng = root / "ng.gbk"
            SeqIO.write(
                SeqRecord(Seq("TTTT"), id="NG", annotations={"molecule_type": "DNA"}),
                ng,
                "genbank",
            )
            with (
                patch(
                    "sys.argv",
                    [
                        "audit",
                        "--source",
                        str(source),
                        "--ng-gbk",
                        str(ng),
                        "--output-dir",
                        str(root / "audit"),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                audit_main()
            args = [
                "build",
                "--antismash-dir",
                str(source),
                "--ng-gbk",
                str(ng),
                "--audit-dir",
                str(root / "audit"),
                "--output-dir",
                str(root / "prepared"),
            ]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                build_pretraining_main()
            report = json.loads((root / "prepared/audit.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["counts"]["assemblies"], 2)
            self.assertEqual(report["counts"]["train_windows"], 1)
            self.assertEqual(report["counts"]["val_windows"], 1)
            self.assertTrue(report["source_audit"]["exists"])

    def test_complete_vs_truncated_or_mismatched_outputs_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            folder = source / "assembly"
            folder.mkdir()
            r = record("contig")
            SeqIO.write(r, folder / "whole.gbk", "genbank")
            (folder / "whole.json").write_text(
                json.dumps(
                    {"version": "8", "records": [{"id": r.id, "seq": str(r.seq)}]}
                )
            )
            result = audit_assembly(folder, set(), set(), set())
            self.assertFalse(result["errors"])
            (folder / "whole.json").write_text("{")
            self.assertTrue(audit_assembly(folder, set(), set(), set())["errors"])
            (folder / "whole.json").write_text(
                json.dumps(
                    {"version": "8", "records": [{"id": r.id, "seq": str(r.seq)}]}
                )
            )
            ng = root / "ng.gbk"
            SeqIO.write(record("NG"), ng, "genbank")
            args = [
                "audit",
                "--source",
                str(source),
                "--ng-gbk",
                str(ng),
                "--output-dir",
                str(root / "audit"),
            ]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                audit_main()
            summary = json.loads((root / "audit/summary.json").read_text())
            self.assertEqual(summary["usable_assemblies"], ["assembly"])
            self.assertEqual(summary["status"], "completed")
            self.assertIsNone(summary["original_source"])


if __name__ == "__main__":
    unittest.main()
