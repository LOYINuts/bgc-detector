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

    @unittest.skipUnless(shutil.which("bash"), "runner requires bash")
    def test_runner_reads_external_historical_data_and_writes_separate_outputs(self):
        self._runner(fail=False, historical=True)

    @unittest.skipUnless(shutil.which("bash"), "runner requires bash")
    def test_runner_checks_ng_labels_before_launching_pretraining(self):
        self._runner(fail=False, missing_labels=True)

    @unittest.skipUnless(shutil.which("bash"), "runner requires bash")
    def test_two_seed_workers_share_one_batch_verification(self):
        self._runner(fail=False, two_seeds=True)

    def _runner(self, fail, historical=False, missing_labels=False, two_seeds=False):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "scripts").mkdir()
            shutil.copyfile(ROOT / "scripts/run_final.sh", t / "scripts/run_final.sh")
            data_root = t / "external data"
            ng = data_root / "derived/ng_embeddings"
            ng.mkdir(parents=True)
            (ng / "ng_gene_embeddings_960d.parquet").touch()
            raw = data_root / "raw/ng"
            raw.mkdir(parents=True)
            (raw / "ClusterFinder_Annotated_Contigs.full.gbk").touch()
            if not missing_labels:
                (raw / "291BGC.csv").touch()
            windows = data_root / "derived/train_test"
            windows.mkdir(parents=True)
            prefix = "dataset_legacy" if historical else "dataset"
            for split in ["train", "val"]:
                (windows / f"{prefix}_{split}.parquet").touch()
            catalog = data_root / "derived/catalog"
            catalog.mkdir(parents=True)
            (catalog / "gene_embeddings_960d.parquet").touch()
            pt = data_root / "derived/pretraining/antismash_v2_v1"
            cache = data_root / "derived/pretraining/antismash_v2_v1_restart_v2_cache"
            pt.mkdir(parents=True)
            cache.mkdir()
            for fn in [
                "audit.json",
                "windows_train.parquet",
                "windows_val.parquet",
                "proteins.parquet",
            ]:
                (pt / fn).write_text("{}")
            for fn in ["metadata.json", "embeddings.npy", "sequence_ids.json"]:
                (cache / fn).write_text("{}")
            before = {
                p.relative_to(data_root): p.read_bytes()
                for p in data_root.rglob("*")
                if p.is_file()
            }
            output_root = t / "separate outputs"
            worker = t / "worker.py"
            worker.write_text(
                """from pathlib import Path
import json,os,sys
args=sys.argv[1:]
assert all(os.environ[k]=='2' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
def arg(name):return args[args.index(name)+1]
if 'bgc_gnn.verify_inputs' in args:
    stage='verify'; output=Path(arg('--output'))
    assert not output.exists()
    output.write_text('{}')
elif 'bgc_gnn.pretrain' in args:
    assert Path(arg('--input-verification')).is_file()
    stage='pretrain'; output=Path(arg('--output-dir'))
    assert Path(arg('--data-dir')).parent.parent.parent == Path(os.environ['DATA_ROOT'])
    assert not output.exists(), 'runner created output before pretraining'
    if os.environ.get('FAIL_PRETRAIN')=='1':sys.exit(1)
    output.mkdir(parents=True)
    (output/'pretrained_best.pt').touch()
    (output/'status.json').write_text(json.dumps({'status':'completed'}))
elif 'bgc_gnn.train' in args:
    stage='train'; output=Path(arg('--checkpoint-dir'))
    assert Path(arg('--fixed-dataset')).name == os.environ['EXPECTED_TRAIN']
    assert Path(arg('--fixed-val-dataset')).is_file()
    assert not output.exists(), 'runner created output before training'
    assert Path(arg('--pretrained-checkpoint')).is_file()
    assert not any(s in args for s in ['--epochs','--context-readout','--batch-size'])
    output.mkdir(parents=True)
    (output/'detector_960d_best.pt').touch()
else:
    stage='overlap'
    assert '--protocol' not in args
    assert all(Path(arg(k)).is_file() for k in ['--ng-gbk','--labels','--emb-path'])
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
            for name in [
                "PRETRAIN_REUSE",
                "TRAIN_DIR",
                "PRETRAIN_DIR",
                "TRAIN_DS",
                "VAL_DS",
                "EMB_PATH",
                "NG_EMB",
                "NG_GBK",
                "NG_LABELS",
                "PT_DATA",
                "PT_CACHE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]:
                env.pop(name, None)
            env.update(
                PATH=str(bindir) + os.pathsep + env["PATH"],
                TEST_PYTHON=Path(sys.executable).as_posix(),
                TEST_WORKER=worker.as_posix(),
                STAGE_LOG=(t / "stages").as_posix(),
                SEEDS="1506 1507" if two_seeds else "1506",
                GPUS="0 1" if two_seeds else "0",
                DATA_ROOT=data_root.as_posix(),
                OUTPUT_ROOT=output_root.as_posix(),
                EXPECTED_TRAIN=f"{prefix}_train.parquet",
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
            self.assertEqual(
                before,
                {
                    p.relative_to(data_root): p.read_bytes()
                    for p in data_root.rglob("*")
                    if p.is_file()
                },
            )
            if missing_labels:
                self.assertNotEqual(p.returncode, 0)
                self.assertIn("291BGC.csv", p.stdout)
                self.assertFalse((t / "stages").exists())
                self.assertFalse(output_root.exists())
            elif fail:
                self.assertNotEqual(p.returncode, 0)
                self.assertFalse((output_root / "runs/final_seed1506").exists())
                self.assertNotIn("All seeds completed", p.stdout)
            else:
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                stages = (t / "stages").read_text().splitlines()
                if two_seeds:
                    self.assertEqual(stages[0], "verify")
                    self.assertEqual(stages.count("verify"), 1)
                    for stage in ["pretrain", "train", "overlap"]:
                        self.assertEqual(stages.count(stage), 2)
                else:
                    self.assertEqual(stages, ["verify", "pretrain", "train", "overlap"])


if __name__ == "__main__":
    unittest.main()
