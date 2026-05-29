#!/usr/bin/env bash
set -euo pipefail

DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
RUN_BASE="${RUN_BASE:-paperlike-tmow-compact-fill17-20260528}"
DATA_ROOT="${DATA_ROOT:-$DATA_DISK/wormi-data/virtualhome-paperlike-tmow-compact-fill17-20260528}"
WORLD_CKPT_ROOT="${WORLD_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/world-vh-paperlike-tmow-compact-fill17-20260528}"
WORMI_CKPT_ROOT="${WORMI_CKPT_ROOT:-$DATA_DISK/wormi-checkpoints/wormi-vh-paperlike-tmow-compact-fill17-20260528}"
LOG_ROOT="${LOG_ROOT:-$DATA_DISK/wormi-logs}"
PIPE_DIR="$LOG_ROOT/vh-pipeline-$RUN_BASE"
DETACHED_DIR="$LOG_ROOT/vh-pipeline-$RUN_BASE-detached"
SESSION_NAME="${SESSION_NAME:-wormi_tmow_compact_full}"
VALIDATION_JSON="${VALIDATION_JSON:-reports/virtualhome/validation/vh-paperlike-tmow-compact-fill17-validation-2026-05-28.json}"
ALIGNMENT_JSON="${ALIGNMENT_JSON:-reports/virtualhome/validation/vh-paperlike-tmow-compact-alignment-audit-2026-05-28.json}"
REPRO_MANIFEST_JSON="${REPRO_MANIFEST_JSON:-reports/virtualhome/validation/vh-paperlike-tmow-compact-repro-manifest-2026-05-28.json}"

echo "== WorMI TMoW-compact status =="
echo "time:        $(date -Is)"
echo "data root:   $DATA_ROOT"
echo "world ckpt:  $WORLD_CKPT_ROOT"
echo "wormi ckpt:  $WORMI_CKPT_ROOT"
echo

echo "-- tmux --"
tmux ls 2>/dev/null | grep -E "$SESSION_NAME|wormi_" || true
echo

echo "-- gpu --"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
else
  echo "nvidia-smi not found"
fi
echo

echo "-- compact status file --"
if [[ -f "$PIPE_DIR/status.tsv" ]]; then
  cat "$PIPE_DIR/status.tsv"
else
  echo "no status file yet: $PIPE_DIR/status.tsv"
fi
echo

echo "-- validation json --"
if [[ -f "$VALIDATION_JSON" ]]; then
  python3 -c 'import json, sys; p=sys.argv[1]; s=json.load(open(p)); chat=s.get("chat_template") or {}; replay=s.get("replay") or {}; print(f"file: {p}"); print(f"errors: {len(s.get("errors", []))}"); print(f"warnings: {len(s.get("warnings", []))}"); print(f"rows: {s.get("total_rows")}"); print(f"replay: {replay}"); print(f"chat_bad_count: {chat.get("bad_count")}"); print(f"chat_action_samples: {chat.get("action_samples")}"); print(f"chat_world_samples: {chat.get("world_samples")}"); print(f"chat_max_action_tokens: {chat.get("max_action_tokens")}"); print(f"chat_max_world_tokens: {chat.get("max_world_tokens")}"); print(f"loss_mask_samples: {chat.get("loss_mask_samples")}"); print(f"min_supervised_tokens: {chat.get("min_supervised_tokens")}"); print(f"max_supervised_tokens: {chat.get("max_supervised_tokens")}")' "$VALIDATION_JSON"
else
  echo "no validation json yet: $VALIDATION_JSON"
fi
echo

echo "-- alignment audit --"
if [[ -f "$ALIGNMENT_JSON" ]]; then
  python3 -c 'import json, sys; p=sys.argv[1]; s=json.load(open(p)); verdicts=s.get("verdicts") or {}; failed=[k for k,v in verdicts.items() if not v]; co=s.get("compact_observation_triples") or {}; cn=s.get("compact_next_observation_triples") or {}; print(f"file: {p}"); print(f"rows: {s.get("rows")}"); print(f"compact_obs_mean: {co.get("mean")}"); print(f"compact_next_mean: {cn.get("mean")}"); print(f"missing_task_args: {s.get("missing_task_args_in_observation")}"); print(f"missing_action_args: {s.get("missing_action_args_in_observation")}"); print(f"failed_verdicts: {failed}")' "$ALIGNMENT_JSON"
else
  echo "no alignment audit yet: $ALIGNMENT_JSON"
fi
echo

echo "-- repro manifest --"
if [[ -f "$REPRO_MANIFEST_JSON" ]]; then
  python3 -c 'import json, sys; p=sys.argv[1]; s=json.load(open(p)); totals=s.get("totals") or {}; print(f"file: {p}"); print(f"jsonl_files: {totals.get("jsonl_files")}"); print(f"jsonl_rows: {totals.get("jsonl_rows")}"); print(f"jsonl_bytes: {totals.get("jsonl_bytes")}"); print(f"metadata_files: {totals.get("metadata_files")}"); print(f"source_files: {totals.get("source_files")}")' "$REPRO_MANIFEST_JSON"
else
  echo "no repro manifest yet: $REPRO_MANIFEST_JSON"
fi
echo

echo "-- checkpoints --"
for i in 0 1 2 3 4 5; do
  if [[ -d "$WORLD_CKPT_ROOT/scene_${i}/last" ]]; then
    echo "stage1 scene_${i}: done"
  else
    echo "stage1 scene_${i}: missing"
  fi
done
if [[ -d "$WORMI_CKPT_ROOT/wormi-vh-n6/last" ]]; then
  echo "stage2 wormi-vh-n6: done"
else
  echo "stage2 wormi-vh-n6: missing"
fi
echo

echo "-- detached launch tail --"
if [[ -f "$DETACHED_DIR/launch.log" ]]; then
  tail -20 "$DETACHED_DIR/launch.log"
else
  echo "no detached launch log yet: $DETACHED_DIR/launch.log"
fi
echo

echo "-- pipeline log tail --"
if [[ -f "$PIPE_DIR/pipeline.log" ]]; then
  tail -40 "$PIPE_DIR/pipeline.log"
else
  echo "no pipeline log yet: $PIPE_DIR/pipeline.log"
fi
