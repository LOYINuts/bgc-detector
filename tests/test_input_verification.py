import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bgc_gnn.checkpoint import file_fingerprint
from bgc_gnn.verify_inputs import verify_inputs


class VerificationTest(unittest.TestCase):
    def fixture(self, root):
        data, cache = root / "data", root / "cache"
        data.mkdir()
        cache.mkdir()
        for name in [
            "windows_train.parquet",
            "windows_val.parquet",
            "proteins.parquet",
        ]:
            (data / name).write_bytes(name.encode())
        for name in ["embeddings.npy", "sequence_ids.json"]:
            (cache / name).write_bytes(name.encode())
        (data / "audit.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    **{
                        n: file_fingerprint(data / n)
                        for n in ["windows_train.parquet", "windows_val.parquet"]
                    },
                }
            )
        )
        (cache / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "model": "biohub/ESMC-300M",
                    "dim": 960,
                    "proteins": file_fingerprint(data / "proteins.parquet"),
                    "embeddings": file_fingerprint(cache / "embeddings.npy"),
                    "sequence_ids": file_fingerprint(cache / "sequence_ids.json"),
                }
            )
        )
        return data, cache

    def test_receipt_never_rehashes_large_inputs_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cache = self.fixture(root)
            receipt = root / "receipt.json"
            with patch(
                "bgc_gnn.verify_inputs.file_fingerprint", wraps=file_fingerprint
            ) as fp:
                original = verify_inputs(data, cache, write_to=receipt)
            self.assertEqual(fp.call_count, 7)
            self.assertEqual(original["provenance"]["mode"], "full_sha256")
            for _ in range(2):
                with patch(
                    "bgc_gnn.verify_inputs.file_fingerprint", wraps=file_fingerprint
                ) as fp:
                    actual = verify_inputs(data, cache, receipt)
                self.assertEqual(actual["inputs"], original["inputs"])
                self.assertEqual(actual["provenance"]["mode"], "batch_receipt")
                self.assertEqual(
                    {Path(c.args[0]).name for c in fp.call_args_list},
                    {"audit.json", "metadata.json", "receipt.json"},
                )

    def test_changed_identity_or_manifest_invalidates_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cache = self.fixture(root)
            receipt = root / "receipt.json"
            verify_inputs(data, cache, write_to=receipt)
            file = cache / "embeddings.npy"
            stat = file.stat()
            os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
            with self.assertRaisesRegex(ValueError, "input changed since batch"):
                verify_inputs(data, cache, receipt)
            (cache / "metadata.json").write_text(
                (cache / "metadata.json").read_text() + "\n"
            )
            with self.assertRaisesRegex(ValueError, "input manifests"):
                verify_inputs(data, cache, receipt)

    def test_equivalent_relative_paths_reuse_receipt(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            data, cache = self.fixture(root)
            receipt = root / "receipt.json"
            original = verify_inputs(data, cache, write_to=receipt)
            actual = verify_inputs(
                data.relative_to(Path.cwd()), cache.relative_to(Path.cwd()), receipt
            )
            self.assertEqual(original["inputs"], actual["inputs"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                verify_inputs(data, cache, write_to=receipt)

    def test_stale_receipt_cli_reports_error_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cache = self.fixture(root)
            receipt = root / "receipt.json"
            verify_inputs(data, cache, write_to=receipt)
            (cache / "embeddings.npy").write_bytes(b"changed")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bgc_gnn.pretrain",
                    "--data-dir",
                    str(data),
                    "--cache-dir",
                    str(cache),
                    "--input-verification",
                    str(receipt),
                    "--output-dir",
                    str(root / "output"),
                    "--device",
                    "cpu",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("error: input changed since batch", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "output").exists())

    def test_full_verification_detects_corrupt_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cache = self.fixture(root)
            (cache / "embeddings.npy").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                verify_inputs(data, cache)


if __name__ == "__main__":
    unittest.main()
