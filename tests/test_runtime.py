import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bgc_gnn.runtime import configure_cpu_threads, prefetch_file


class RuntimeTest(unittest.TestCase):
    def test_thread_count_uses_runner_environment_and_rejects_invalid_values(self):
        with (
            patch.dict(os.environ, {"OMP_NUM_THREADS": "3"}),
            patch("bgc_gnn.runtime.torch.set_num_threads") as setter,
        ):
            self.assertEqual(configure_cpu_threads(), 3)
            setter.assert_called_once_with(3)
        for value in ["0", "invalid", "-1"]:
            with patch.dict(os.environ, {"OMP_NUM_THREADS": value}):
                with self.assertRaises(ValueError):
                    configure_cpu_threads()

    def test_prefetch_closes_descriptor_and_handles_unsupported_filesystems(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npy"
            path.write_bytes(b"cache")
            with (
                patch.object(os, "posix_fadvise", create=True) as advise,
                patch.object(os, "POSIX_FADV_WILLNEED", 3, create=True),
            ):
                self.assertIn("requested", prefetch_file(path))
                descriptor = advise.call_args.args[0]
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
                advise.side_effect = OSError("unsupported filesystem")
                self.assertIn("unavailable", prefetch_file(path))


if __name__ == "__main__":
    unittest.main()
