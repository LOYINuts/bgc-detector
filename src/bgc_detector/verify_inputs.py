"""Full input verification once per batch; unchanged-file checks for each seed."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from bgc_detector.checkpoint import file_fingerprint


def identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def verify_inputs(
    data_dir: Path,
    cache_dir: Path,
    receipt: Path | None = None,
    *,
    write_to: Path | None = None,
) -> dict:
    data_dir, cache_dir = data_dir.resolve(), cache_dir.resolve()
    receipt = receipt.resolve() if receipt else None
    if write_to is not None and write_to.exists():
        raise ValueError(
            "verification receipt already exists; use a fresh batch output"
        )
    manifests = {
        "audit": file_fingerprint(data_dir / "audit.json"),
        "cache": file_fingerprint(cache_dir / "metadata.json"),
    }
    audit = json.loads((data_dir / "audit.json").read_text())
    cache = json.loads((cache_dir / "metadata.json").read_text())
    if audit["status"] != "completed" or cache["status"] != "completed":
        raise ValueError("requires completed data and embedding cache")
    if cache["model"] != "biohub/ESMC-300M" or cache["dim"] != 960:
        raise ValueError("pretraining requires frozen ESMC-300M 960d")
    record = json.loads(receipt.read_text()) if receipt else None
    if record is not None and (
        record.get("version") != 1
        or record.get("status") != "verified"
        or record.get("manifests") != manifests
    ):
        raise ValueError("verification receipt does not match input manifests")
    inputs, identities = {}, {}
    for role, path, expected in [
        ("train", data_dir / "windows_train.parquet", audit["windows_train.parquet"]),
        ("val", data_dir / "windows_val.parquet", audit["windows_val.parquet"]),
        ("proteins", data_dir / "proteins.parquet", cache["proteins"]),
        ("embeddings", cache_dir / "embeddings.npy", cache["embeddings"]),
        ("sequence_ids", cache_dir / "sequence_ids.json", cache["sequence_ids"]),
    ]:
        before = identity(path)
        if record is None:
            print(f"Checking input SHA256: {role} ({path})", flush=True)
            fp = file_fingerprint(path)
        else:
            if record["identities"].get(role) != before:
                raise ValueError(f"input changed since batch verification: {role}")
            fp = record["inputs"][role]
            if Path(fp["path"]).resolve() != path.resolve():
                raise ValueError(f"receipt input path mismatch: {role}")
        if fp["sha256"] != expected["sha256"]:
            raise ValueError(f"input fingerprint mismatch: {role}")
        if before != identity(path) or fp["size"] != before["size"]:
            raise ValueError(f"input changed during verification: {role}")
        inputs[role], identities[role] = fp, before
    record = record or {
        "version": 1,
        "status": "verified",
        "manifests": manifests,
        "verified_at": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "identities": identities,
    }
    provenance = {
        "mode": "batch_receipt" if receipt else "full_sha256",
        "verified_at": record["verified_at"],
    }
    if receipt:
        provenance["receipt"] = file_fingerprint(receipt)
        print(
            "Reused batch SHA verification; input file identities unchanged", flush=True
        )
    if write_to is not None:
        write_to.parent.mkdir(parents=True, exist_ok=True)
        with write_to.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return {"inputs": inputs, "cache": cache, "audit": audit, "provenance": provenance}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        verify_inputs(a.data_dir, a.cache_dir, write_to=a.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
