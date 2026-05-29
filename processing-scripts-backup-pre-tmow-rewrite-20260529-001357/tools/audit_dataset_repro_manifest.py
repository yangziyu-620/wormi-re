#!/usr/bin/env python3
"""Write a reproducibility manifest for a generated JSONL dataset.

The manifest records file sizes, row counts, SHA256 hashes, and selected source
script hashes so a later rebuild can be compared byte-for-byte and code-for-code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_FILES = [
    "tools/build_virtualhome_dataset_tmow_compact.py",
    "tools/compact_virtualhome_observations.py",
    "tools/validate_virtualhome_dataset.py",
    "tools/audit_tmow_compact_alignment.py",
    "sh/wormi-build-vh-data-tmow-compact.sh",
    "sh/wormi-vh-paperlike-tmow-compact-full.sh",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def file_record(path: Path, *, base: Path | None = None, rows: bool = False) -> dict[str, Any]:
    rel = str(path.relative_to(base)) if base is not None else str(path)
    record = {
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows:
        record["rows"] = jsonl_rows(path)
    return record


def manifest(data_root: Path, repo_root: Path, extras: list[Path]) -> dict[str, Any]:
    data_files = sorted(data_root.glob("scene_*/train.jsonl")) + sorted(
        data_root.glob("test_*.jsonl")
    )
    metadata_files = [
        data_root / "virtualhome_manifest.json",
        data_root / "tmow_compact_summary.json",
    ]
    metadata_files = [path for path in metadata_files if path.exists()]

    source_files = []
    for rel in DEFAULT_SOURCE_FILES:
        path = repo_root / rel
        if path.exists():
            source_files.append(path)
    for path in extras:
        resolved = path if path.is_absolute() else repo_root / path
        if resolved.exists() and resolved not in source_files:
            source_files.append(resolved)

    data_records = [file_record(path, base=data_root, rows=True) for path in data_files]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "repo_root": str(repo_root),
        "data_files": data_records,
        "metadata_files": [file_record(path, base=data_root) for path in metadata_files],
        "source_files": [file_record(path, base=repo_root) for path in source_files],
        "totals": {
            "jsonl_files": len(data_records),
            "jsonl_rows": sum(int(item["rows"]) for item in data_records),
            "jsonl_bytes": sum(int(item["bytes"]) for item in data_records),
            "metadata_files": len(metadata_files),
            "source_files": len(source_files),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    summary = manifest(args.data_root, args.repo_root, args.extra)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["totals"], indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
