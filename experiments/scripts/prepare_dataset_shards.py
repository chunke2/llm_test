#!/usr/bin/env python3
"""Create identical data shards for deterministic multi-epoch stress tests."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=10)
    args = parser.parse_args()

    if args.copies < 2:
        raise SystemExit("--copies must be at least 2")
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)

    for index in range(args.copies):
        destination = output_dir / f"tiny_shakespeare_train_{index:03d}.bin"
        if not destination.exists() or sha256(destination) != source_hash:
            shutil.copyfile(source, destination)
        if sha256(destination) != source_hash:
            raise RuntimeError(f"copy verification failed: {destination}")

    print(
        f"dataset_shards_ready copies={args.copies} "
        f"source_sha256={source_hash} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
