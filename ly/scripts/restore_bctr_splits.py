"""Restore selected BCTR LMDB splits from the LFS part bundle, with SHA-256 checks.

Adapted from OpenOCR-main/tools/benchmark_bundle.py (which always restores all nine splits);
this version restores only the requested splits so the train splits do not fill the disk.
Usage:
    ly/.venv/bin/python ly/scripts/restore_bctr_splits.py --dest ly/data/bctr \
        scene/scene_val scene/scene_test web/web_val web/web_test document/document_val document/document_test
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

BLOCK = 8 * 1024 * 1024
BUNDLE = Path(__file__).resolve().parents[2] / "OpenOCR-main/datasets/benchmark_dataset_bundle"


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(BLOCK):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("splits", nargs="+", help="e.g. scene/scene_test")
    args = parser.parse_args()
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    items = {item["path"]: item for item in manifest["splits"]}
    for split in args.splits:
        item = items[f"{split}/data.mdb"]
        out = args.dest / item["path"]
        if out.is_file() and out.stat().st_size == item["bytes"] and digest(out) == item["sha256"]:
            print("already verified", split, flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".restoring")
        whole = hashlib.sha256()
        with open(tmp, "wb") as outgoing:
            for part in item["parts"]:
                with open(BUNDLE / part["path"], "rb") as incoming:
                    while block := incoming.read(BLOCK):
                        outgoing.write(block)
                        whole.update(block)
        if whole.hexdigest() != item["sha256"]:
            tmp.unlink()
            raise SystemExit(f"hash mismatch for {split}; run git lfs pull")
        os.replace(tmp, out)
        print("restored", split, item["bytes"] // 2**20, "MiB", flush=True)


if __name__ == "__main__":
    main()
