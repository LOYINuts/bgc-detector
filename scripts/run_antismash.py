"""Run antiSMASH on GTDB genomes; write one output directory per assembly.

Requires an external antiSMASH installation and its databases. Audit outputs
with audit_antismash.py before building pretraining windows.

Usage:
    uv run python scripts/run_antismash.py --indir data/raw/gtdb/genomes
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run antiSMASH on all GTDB genomes")
    parser.add_argument(
        "--indir", required=True, help="Directory with genome .fna files (subdirs)"
    )
    parser.add_argument(
        "--outdir",
        default="data/raw/gtdb/antismash_results",
        help="Output directory for antiSMASH results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="antiSMASH timeout per genome (seconds, default 30 min)",
    )
    parser.add_argument(
        "--cpus", type=int, default=1, help="CPUs per antiSMASH run (default 1)"
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Max genomes (for testing)"
    )
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fna_files = sorted(indir.rglob("*.fna"))
    if args.max:
        fna_files = fna_files[: args.max]

    print(f"{'=' * 60}")
    print(f"Processing {len(fna_files)} genomes")
    print(f"  Input:  {indir}")
    print(f"  Output: {outdir}")
    print(f"{'=' * 60}\n")

    ok, skipped, failed, timed_out = 0, 0, 0, 0
    t_start = time.time()

    for i, fna_path in enumerate(fna_files):
        genome_id = fna_path.parent.name
        gdir = outdir / genome_id

        print(f"[{i + 1}/{len(fna_files)}] {genome_id}", end=" ", flush=True)

        if gdir.exists() and any(gdir.glob("*genomic.gbk")):
            print("skip (exists)")
            skipped += 1
            continue

        t0 = time.time()
        try:
            subprocess.run(
                [
                    "antismash",
                    "--minimal",
                    "--cpus",
                    str(args.cpus),
                    "--genefinding-tool",
                    "prodigal",
                    "--output-dir",
                    str(outdir / genome_id),
                    str(fna_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT ({time.time() - t0:.0f}s)")
            timed_out += 1
            continue
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode()[-150:] if e.stderr else str(e)
            print(f"FAIL: {err}")
            failed += 1
            continue
        except Exception as e:
            print(f"CRASH: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue

        elapsed = time.time() - t0
        print(f"OK ({elapsed:.0f}s)")
        ok += 1

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done in {total_time / 60:.0f} min")
    print(f"  OK:      {ok}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Timeout: {timed_out}")
    if failed or timed_out:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
