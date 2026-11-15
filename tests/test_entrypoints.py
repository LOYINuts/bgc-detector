import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTest(unittest.TestCase):
    def test_documented_encoding_command_imports_without_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        p = subprocess.run(
            [sys.executable, "scripts/encode.py", "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--source", p.stdout)

    def test_pretraining_cli_has_no_recipe_switches(self):
        p = subprocess.run(
            [sys.executable, "-m", "bgc_gnn.pretrain", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        for switch in [
            "--epochs",
            "--patience",
            "--lr",
            "--batch-size",
            "--semantic-skip",
            "--context-readout",
            "--method",
        ]:
            self.assertNotIn(switch, p.stdout)

    @unittest.skipUnless(shutil.which("bash"), "runner requires bash")
    def test_runner_uses_fresh_outputs_and_fixed_stages(self):
        self._runner(fail=False)

    @unittest.skipUnless(shutil.which("bash"), "runner requires bash")
    def test_runner_stops_if_pretraining_fails(self):
        self._runner(fail=True)

    def _runner(self, fail):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/run_final.sh", t / "scripts/run_final.sh")
            ng = t / "data/derived/ng_embeddings"
            ng.mkdir(parents=True)
            (ng / "ng_gene_embeddings_960d.parquet").touch()
            (t / "placeholder").touch()
            for name, fn in [("data", "audit.json"), ("cache", "metadata.json")]:
                (t / name).mkdir(exist_ok=True)
                (t / name / fn).write_text("{}")
            worker = t / "worker.py"
            worker.write_text(
                """from pathlib import Path
import json,os,sys
args=sys.argv[1:]
def arg(name):return args[args.index(name)+1]
if 'bgc_gnn.pretrain' in args:
    stage='pretrain'; output=Path(arg('--output-dir'))
    assert not output.exists(), 'runner created output before pretraining'
    if os.environ.get('FAIL_PRETRAIN')=='1':sys.exit(1)
    output.mkdir(parents=True)
    (output/'pretrained_best.pt').touch()
    (output/'status.json').write_text(json.dumps({'status':'completed'}))
elif 'bgc_gnn.train' in args:
    stage='train'; output=Path(arg('--checkpoint-dir'))
    assert not output.exists(), 'runner created output before training'
    assert Path(arg('--pretrained-checkpoint')).is_file()
    assert not any(s in args for s in ['--epochs','--context-readout','--batch-size'])
    output.mkdir(parents=True)
    (output/'detector_960d_best.pt').touch()
else:
    stage='overlap'
    assert '--protocol' not in args
    Path(arg('--metrics-path')).write_text('{}')
with open(os.environ['STAGE_LOG'],'a') as f:f.write(stage+'\\n')
""",
                encoding="utf-8",
            )
            bindir = t / "bin"
            bindir.mkdir()
            fake_uv = bindir / "uv"
            fake_uv.write_text(
                '#!/usr/bin/env bash\nshift 2\nexec "$TEST_PYTHON" "$TEST_WORKER" "$@"\n',
                encoding="utf-8",
                newline="\n",
            )
            fake_uv.chmod(0o755)
            env = os.environ.copy()
            for name in ["PRETRAIN_REUSE", "TRAIN_DIR", "PRETRAIN_DIR"]:
                env.pop(name, None)
            env.update(
                PATH=str(bindir) + os.pathsep + env["PATH"],
                TEST_PYTHON=Path(sys.executable).as_posix(),
                TEST_WORKER=worker.as_posix(),
                STAGE_LOG=(t / "stages").as_posix(),
                SEEDS="1506",
                GPUS="0",
                TRAIN_DS=(t / "placeholder").as_posix(),
                VAL_DS=(t / "placeholder").as_posix(),
                EMB_PATH=(t / "placeholder").as_posix(),
                PT_DATA=(t / "data").as_posix(),
                PT_CACHE=(t / "cache").as_posix(),
                FAIL_PRETRAIN="1" if fail else "0",
            )
            p = subprocess.run(
                ["bash", (t / "scripts/run_final.sh").as_posix()],
                cwd=t,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if fail:
                self.assertNotEqual(p.returncode, 0)
                self.assertFalse((t / "runs/final_seed1506").exists())
                self.assertNotIn("All seeds completed", p.stdout)
            else:
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                self.assertEqual(
                    (t / "stages").read_text().splitlines(),
                    ["pretrain", "train", "overlap"],
                )


if __name__ == "__main__":
    unittest.main()
