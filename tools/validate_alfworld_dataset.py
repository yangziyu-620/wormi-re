#!/usr/bin/env python3
"""Validate a generated ALFWorld protocol release.

This is the ALFWorld counterpart to ``tools/validate_virtualhome_dataset.py``.
It validates the canonical episode pool, split files, method views, leakage
constraints, and optional loader compatibility for releases produced by
``tools/build_alfworld_dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_alfworld_dataset import (  # noqa: E402
    PROTOCOLS,
    build_distribution_report,
    build_leakage_report,
    read_jsonl,
    row_key,
    validate_release,
    write_json,
)

SPLIT_NAMES = [
    "train",
    "monitor",
    "eval_col_1_seen_seen",
    "eval_col_2_seen_unseen",
    "eval_col_3_unseen_unseen",
    "unused_unseen_task_seen_scene",
]
EVAL_SPLITS = [
    "eval_col_1_seen_seen",
    "eval_col_2_seen_unseen",
    "eval_col_3_unseen_unseen",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _require(path: Path, errors: list[str]) -> bool:
    if not path.exists() and not path.is_symlink():
        errors.append(f"missing required file: {path}")
        return False
    return True


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row_key(row) for row in rows]


def _counter_equal(left: list[str], right: list[str]) -> bool:
    return Counter(left) == Counter(right)


def _load_release(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any], list[str]]:
    errors: list[str] = []
    manifest_path = root / "dataset_manifest.json"
    method_manifest_path = root / "method_views_manifest.json"
    canonical_path = root / "canonical" / "episodes.jsonl"

    manifest = _read_json(manifest_path) if _require(manifest_path, errors) else {}
    method_manifest = (
        _read_json(method_manifest_path)
        if _require(method_manifest_path, errors)
        else {}
    )
    canonical = read_jsonl(canonical_path) if _require(canonical_path, errors) else []

    splits: dict[str, list[dict[str, Any]]] = {}
    for name in SPLIT_NAMES:
        path = root / "splits" / f"{name}.jsonl"
        splits[name] = read_jsonl(path) if _require(path, errors) else []
    return manifest, canonical, splits, method_manifest, errors


def _validate_method_views(root: Path, splits: dict[str, list[dict[str, Any]]], method_manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}

    train_ids = _ids(splits["train"])
    monitor_ids = _ids(splits["monitor"])

    view_specs = {
        "llm_ft/train": (root / "views" / "llm_ft" / "train.jsonl", train_ids),
        "llm_ft/test": (root / "views" / "llm_ft" / "test.jsonl", monitor_ids),
        "wormi/adapter/train": (root / "views" / "wormi" / "adapter" / "train.jsonl", train_ids),
        "wormi/adapter/test": (root / "views" / "wormi" / "adapter" / "test.jsonl", monitor_ids),
        "planner_retrieval/index": (root / "views" / "planner_retrieval" / "index.jsonl", train_ids),
    }
    for label, (path, expected_ids) in view_specs.items():
        if not path.exists():
            errors.append(f"missing method view: {path}")
            continue
        rows = read_jsonl(path)
        counts[label] = len(rows)
        if not _counter_equal(_ids(rows), expected_ids):
            errors.append(f"{label} ids do not match expected split")

    for split in EVAL_SPLITS:
        path = root / "views" / "eval" / split / "test.jsonl"
        compat_path = root / split / "test.jsonl"
        expected_ids = _ids(splits[split])
        for label, p in [(f"views/eval/{split}", path), (f"compat/{split}", compat_path)]:
            if not p.exists() and not p.is_symlink():
                errors.append(f"missing eval view: {p}")
                continue
            rows = read_jsonl(p)
            counts[label] = len(rows)
            if not _counter_equal(_ids(rows), expected_ids):
                errors.append(f"{label} ids do not match {split}")

    world_root = root / "views" / "wormi" / "world_model"
    clusters_path = world_root / "world_model_clusters.json"
    if not clusters_path.exists():
        errors.append(f"missing world cluster manifest: {clusters_path}")
        clusters = {}
    else:
        clusters = _read_json(clusters_path)

    cluster_train_ids: list[str] = []
    cluster_monitor_ids: list[str] = []
    for name, info in sorted(clusters.items()):
        cluster_dir = world_root / name
        train_path = cluster_dir / "train.jsonl"
        test_path = cluster_dir / "test.jsonl"
        if not train_path.exists():
            errors.append(f"missing {name}/train.jsonl")
            continue
        if not test_path.exists():
            errors.append(f"missing {name}/test.jsonl")
            continue
        train_rows = read_jsonl(train_path)
        test_rows = read_jsonl(test_path)
        counts[f"wormi/world_model/{name}/train"] = len(train_rows)
        counts[f"wormi/world_model/{name}/test"] = len(test_rows)
        cluster_train_ids.extend(_ids(train_rows))
        cluster_monitor_ids.extend(_ids(test_rows))
        if len(train_rows) != int(info.get("train_episodes", -1)):
            errors.append(f"{name} train count differs from cluster manifest")
        if len(test_rows) != int(info.get("monitor_episodes", -1)):
            errors.append(f"{name} test count differs from cluster manifest")
        if not train_rows:
            errors.append(f"{name} has no train rows")
        if not test_rows:
            warnings.append(f"{name} has no monitor/test rows")

    if cluster_train_ids and not _counter_equal(cluster_train_ids, train_ids):
        errors.append("union of world-model cluster train ids does not equal train split")
    if cluster_monitor_ids and not _counter_equal(cluster_monitor_ids, monitor_ids):
        errors.append("union of world-model cluster test ids does not equal monitor split")

    manifest_clusters = method_manifest.get("wormi/world_model", {}).get("clusters", {})
    if manifest_clusters and set(manifest_clusters) != set(clusters):
        errors.append("method_views_manifest clusters differ from world_model_clusters.json")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": counts,
    }


def _check_loader(paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {"checked": [], "errors": []}
    try:
        from wormi.datasets.auto_jsonl import AutoJsonlDataset
    except Exception as exc:  # pragma: no cover - environment dependent
        result["errors"].append(f"could not import AutoJsonlDataset: {exc}")
        return result

    for path in paths:
        try:
            dataset = AutoJsonlDataset.load(
                path,
                cumulative=True,
                end_with_action=True,
            )
            result["checked"].append(
                {
                    "path": str(path),
                    "dataset_type": dataset.dataset_type,
                    "samples": len(dataset),
                    "columns": list(dataset.column_names),
                }
            )
        except Exception as exc:  # pragma: no cover - data-contract failure
            result["errors"].append(f"{path}: {exc}")
    return result


def validate_dataset(root: Path, check_loader: bool = False) -> dict[str, Any]:
    manifest, canonical, splits, method_manifest, load_errors = _load_release(root)
    protocol_name = manifest.get("protocol")
    if protocol_name not in PROTOCOLS:
        return {
            "valid": False,
            "errors": [*load_errors, f"unknown or missing protocol: {protocol_name!r}"],
            "warnings": [],
        }

    protocol = PROTOCOLS[protocol_name]
    leakage = build_leakage_report(protocol, splits, method_manifest)
    distribution = build_distribution_report(canonical, splits)
    release_validation = validate_release(protocol, canonical, splits, method_manifest, leakage)
    view_validation = _validate_method_views(root, splits, method_manifest)

    errors = [*load_errors, *release_validation["errors"], *view_validation["errors"]]
    warnings = [*release_validation["warnings"], *view_validation["warnings"]]
    loader_validation: dict[str, Any] | None = None
    if check_loader:
        loader_paths = [
            root / "views" / "llm_ft" / "train.jsonl",
            root / "views" / "wormi" / "world_model" / "cluster_00" / "train.jsonl",
            root / "views" / "eval" / "eval_col_3_unseen_unseen" / "test.jsonl",
        ]
        loader_validation = _check_loader(loader_paths)
        errors.extend(loader_validation["errors"])

    return {
        "valid": not errors,
        "protocol": protocol_name,
        "errors": errors,
        "warnings": warnings,
        "release_validation": release_validation,
        "view_validation": view_validation,
        "loader_validation": loader_validation,
        "leakage": leakage,
        "distribution": distribution,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=Path, help="Generated ALFWorld protocol root")
    p.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Validation report path. Defaults to <dataset_root>/validation_report.json.",
    )
    p.add_argument(
        "--check-loader",
        action="store_true",
        help="Also load representative JSONL files through AutoJsonlDataset.",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Validate only; do not rewrite validation/leakage/distribution reports.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    report = validate_dataset(root, check_loader=args.check_loader)
    if not args.no_write:
        output_path = args.output_path or root / "validation_report.json"
        write_json(output_path, report)
        if "leakage" in report:
            write_json(root / "leakage_report.json", report["leakage"])
        if "distribution" in report:
            write_json(root / "distribution_report.json", report["distribution"])
    print(json.dumps({k: report[k] for k in ["valid", "protocol", "errors", "warnings"] if k in report}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
