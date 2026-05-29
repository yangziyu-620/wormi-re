#!/usr/bin/env python3
"""Audit how the TMoW-style compact VirtualHome data aligns with WorMI needs.

This is a lightweight data-contract audit, not a training/eval script.  It
summarizes the generated JSONL metadata against the TMoW preprocessing pattern:
compact instruction-conditioned observations and compact action-conditioned
state-update targets, while preserving WorMI JSONL and split layout.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _mean(values: list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _split_files(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("scene_*/train.jsonl")) + sorted(
        data_root.glob("test_*.jsonl")
    )


def audit(data_root: Path, validation_json: Path | None) -> dict[str, Any]:
    files = _split_files(data_root)
    rows = []
    for path in files:
        for row in _load_jsonl(path):
            row["__file"] = str(path.relative_to(data_root))
            rows.append(row)

    mode_counts = Counter()
    next_mode_counts = Counter()
    fill_counts = Counter()
    source_counts = Counter()
    obs_before = []
    obs_after = []
    next_before = []
    next_after = []
    missing_task_args = 0
    missing_action_args = 0
    no_updates = 0
    malformed_preprocessing = 0

    for row in rows:
        prep = row.get("_meta", {}).get("observation_preprocessing", {})
        if not prep:
            malformed_preprocessing += 1
            continue
        mode_counts[str(prep.get("mode"))] += 1
        next_mode_counts[str(prep.get("next_mode"))] += 1
        fill_counts[bool(prep.get("fill_to_num_edges"))] += 1
        source_counts[str(prep.get("source"))] += 1
        obs_before.append(int(prep.get("source_observation_triples", 0)))
        obs_after.append(int(prep.get("compact_observation_triples", 0)))
        next_before.append(int(prep.get("source_next_observation_triples", 0)))
        next_after.append(int(prep.get("compact_next_observation_triples", 0)))
        obs_text = str(row.get("observation", "")).lower()
        action_text = str(row.get("action", "")).lower()
        task_args = [str(x).lower().replace(" ", "_") for x in row.get("_meta", {}).get("task_args", [])]
        if any(arg not in obs_text for arg in task_args):
            missing_task_args += 1
        action_args = action_text.split()[1:]
        if any(arg not in obs_text for arg in action_args):
            missing_action_args += 1
        if str(row.get("next_observation", "")) == "No updates":
            no_updates += 1

    validation = None
    if validation_json is not None and validation_json.exists():
        validation = json.loads(validation_json.read_text())

    def distribution(values: list[int]) -> dict[str, float | int]:
        return {
            "mean": _mean(values),
            "min": min(values) if values else 0,
            "p50": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "max": max(values) if values else 0,
        }

    verdicts = {
        "independent_graph_state_builder": all(
            mode == "tmow_compact_from_graph_state" for mode in mode_counts
        ),
        "tmow_num_edges_fill17": fill_counts == Counter({True: len(rows)}),
        "delta_next_observation": next_mode_counts == Counter({"delta": len(rows)}),
        "task_args_visible": missing_task_args == 0,
        "action_args_visible": missing_action_args == 0,
        "no_no_updates_targets": no_updates == 0,
    }
    if validation is not None:
        chat = validation.get("chat_template") or {}
        replay = validation.get("replay") or {}
        leakage = validation.get("leakage") or {}
        verdicts.update(
            {
                "validation_errors_zero": len(validation.get("errors", [])) == 0,
                "validation_warnings_zero": len(validation.get("warnings", [])) == 0,
                "replay_clean": all(int(v) == 0 for v in replay.values()),
                "chat_template_clean": int(chat.get("bad_count", -1)) == 0,
                "train_test_row_overlap_zero": int(leakage.get("exact_row_overlap", -1)) == 0,
                "trajectory_overlap_zero": int(leakage.get("trajectory_id_overlap", -1)) == 0,
            }
        )

    return {
        "data_root": str(data_root),
        "validation_json": str(validation_json) if validation_json else None,
        "files": [str(path.relative_to(data_root)) for path in files],
        "rows": len(rows),
        "mode_counts": dict(mode_counts),
        "source_counts": dict(source_counts),
        "next_mode_counts": dict(next_mode_counts),
        "fill_to_num_edges_counts": {str(k): v for k, v in fill_counts.items()},
        "source_observation_triples": distribution(obs_before),
        "compact_observation_triples": distribution(obs_after),
        "source_next_observation_triples": distribution(next_before),
        "compact_next_observation_triples": distribution(next_after),
        "missing_task_args_in_observation": missing_task_args,
        "missing_action_args_in_observation": missing_action_args,
        "no_updates_targets": no_updates,
        "malformed_preprocessing_rows": malformed_preprocessing,
        "validation_summary": {
            "errors": len(validation.get("errors", [])) if validation else None,
            "warnings": len(validation.get("warnings", [])) if validation else None,
            "replay": validation.get("replay") if validation else None,
            "leakage": validation.get("leakage") if validation else None,
            "chat_template": validation.get("chat_template") if validation else None,
        },
        "verdicts": verdicts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = audit(args.data_root, args.validation_json)
    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))

    failed = [key for key, value in summary["verdicts"].items() if not value]
    if failed:
        raise SystemExit(f"Failed alignment gates: {failed}")


if __name__ == "__main__":
    main()
