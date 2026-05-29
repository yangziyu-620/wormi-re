#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from pathlib import Path

RUN_TAG = "paperlike-tmow-compact-aa-fill17-20260528"
BASE = Path("/root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-tmow-compact-aa-fill17-20260528/wormi-vh-n6")
TABLE1 = BASE / f"table1-{RUN_TAG}" / "table1-summary.tsv"
ROLLOUT_DIR = BASE / f"vh-rollout-{RUN_TAG}"
ROLLOUT = ROLLOUT_DIR / "vh-rollout-summary.tsv"
STATUS = Path("/root/autodl-tmp/wormi-logs/vh-eval-paperlike-tmow-compact-aa-fill17-20260528/status.tsv")
REPORT = Path("reports/virtualhome/data-processing/tmow-data-preprocessing-for-wormi-reproduction-2026-05-28.md")
LOG = Path("/root/autodl-tmp/wormi-logs/vh-eval-paperlike-tmow-compact-aa-fill17-20260528/record-watcher.log")
MARKER = "<!-- aa-fill17-final-eval-recorded -->"


def write_log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}\n")


def rollout_process_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "eval-vh-rollout.*paperlike-tmow-compact-aa-fill17"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def read_text(path: Path) -> str:
    return path.read_text().strip() if path.exists() else "(missing)"


def summarize_episodes() -> str:
    lines: list[str] = []
    for path in sorted(ROLLOUT_DIR.glob("vh-rollout-*-episodes.jsonl")):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        if not rows:
            continue
        success_rate = sum(bool(row.get("success")) for row in rows) / len(rows)
        path_steps = sum(float(row.get("steps", 0)) for row in rows) / len(rows)
        invalid_actions = sum(float(row.get("invalid_actions", 0)) for row in rows) / len(rows)
        executed_actions = sum(float(row.get("executed_actions", 0)) for row in rows) / len(rows)
        lines.append(
            f"{path.name}: episodes={len(rows)}, SR={success_rate:.6f}, "
            f"PS={path_steps:.3f}, invalid={invalid_actions:.3f}, "
            f"executed={executed_actions:.3f}"
        )
    return "\n".join(lines) if lines else "(no episode files)"


def main() -> int:
    write_log("watcher started")
    while not ROLLOUT.exists():
        if not rollout_process_running():
            write_log("rollout process ended before summary appeared")
            break
        time.sleep(120)

    report_text = REPORT.read_text() if REPORT.exists() else ""
    if MARKER in report_text:
        write_log("marker already present; nothing to append")
        return 0

    now = dt.datetime.now().isoformat(timespec="seconds")
    section = f"""

{MARKER}

## 2026-05-28 Corrected AA Fill17 Eval Result

Recorded automatically at `{now}` by `sh/wormi-record-aa-fill17-eval.py`.

Run tag: `{RUN_TAG}`

Stage status:

```text
{read_text(STATUS)}
```

Offline Table1 exact-match result:

```text
{read_text(TABLE1)}
```

VirtualHome target-state rollout result:

```text
{read_text(ROLLOUT)}
```

Rollout episode progress/detail summary:

```text
{summarize_episodes()}
```

Artifacts:

```text
data_root: /root/autodl-tmp/wormi-data/virtualhome-paperlike-tmow-compact-aa-fill17-20260528
world_ckpt_root: /root/autodl-tmp/wormi-checkpoints/world-vh-paperlike-tmow-compact-aa-fill17-20260528
wormi_model: /root/autodl-tmp/wormi-checkpoints/wormi-vh-paperlike-tmow-compact-aa-fill17-20260528/wormi-vh-n6/last
table1_dir: {TABLE1.parent}
rollout_dir: {ROLLOUT_DIR}
logs: /root/autodl-tmp/wormi-logs/vh-eval-paperlike-tmow-compact-aa-fill17-20260528
```
"""
    with REPORT.open("a") as f:
        f.write(section)
    write_log("report appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
