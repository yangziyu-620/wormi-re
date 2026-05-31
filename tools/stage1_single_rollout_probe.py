"""Isolation experiment Part B: rollout a SINGLE stage1 world-model (plain
CausalLM, no meta, no retrieval, no implant) in EvolvingGraph, on one scene's
held-out test set. Reuses eval_vh_rollout._eval_episode verbatim so the env,
binding, goal contract are identical to the headline eval.

Tells us whether a lone SFT'd model also collapses to walk-loops at rollout
(=> behaviour-cloning compounding error, meta innocent) or rolls out fine
(=> meta-averaging is the culprit).

Usage:
  python tools/stage1_single_rollout_probe.py \
      --ckpt /root/autodl-tmp/wormi-checkpoints/world-vh/scene_0/last \
      --scene-test /root/autodl-tmp/.../scene_0/test.jsonl \
      --scene-inits /root/autodl-tmp/.../scene_inits.json \
      --max 60 --temperature 1.0
"""
from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wormi.scripts.eval_vh_rollout import (
    VirtualHomeRolloutArgs,
    _bootstrap_evolving_graph,
    _eval_episode,
    _load_eval_episodes,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene-test", required=True)
    ap.add_argument("--scene-inits", required=True)
    ap.add_argument(
        "--vh-src",
        default=str(VirtualHomeRolloutArgs.__dataclass_fields__["vh_src"].default),
    )
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    scene_inits = json.loads(Path(args.scene_inits).read_text())
    eg_modules = _bootstrap_evolving_graph(Path(args.vh_src))

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    tok.pad_token = "<|end_of_text|>"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    r_args = VirtualHomeRolloutArgs(
        curricula_path=Path("/dev/null"),
        scene_inits_json=Path(args.scene_inits),
        vh_src=Path(args.vh_src),
        max_steps=args.max_steps,
        temperature=args.temperature,
        observation_format="full",
    )

    episodes = _load_eval_episodes([Path(args.scene_test)])
    if args.max and len(episodes) > args.max:
        episodes = episodes[: args.max]

    sink = io.StringIO()  # swallow per-step detail; we summarise ourselves
    results = []
    rep_episodes = 0
    invalid_total = 0
    preds_all = collections.Counter()
    for i, ep in enumerate(episodes):
        # capture this episode's predictions by re-wrapping the detail sink
        local = io.StringIO()
        res = _eval_episode(
            model, tok, eg_modules, scene_inits, ep, r_args, local, i, [0]
        )
        results.append(res)
        invalid_total += res.invalid_actions
        preds = [
            json.loads(l)["prediction"] for l in local.getvalue().splitlines() if l.strip()
        ]
        preds_all.update(p.split()[0] if p else "" for p in preds)
        if preds:
            c = collections.Counter(preds)
            if c.most_common(1)[0][1] >= 5:
                rep_episodes += 1

    n = len(results)
    sr = sum(1 for r in results if r.success) / n
    ps = sum(r.steps for r in results) / n
    print(f"== ckpt={args.ckpt}")
    print(f"== scene-test={args.scene_test}  n={n}  temperature={args.temperature}")
    print(f"SINGLE-MODEL ROLLOUT  SR={sr*100:.1f}%  PS={ps:.2f}  "
          f"invalid/ep={invalid_total/n:.2f}")
    print(f"episodes with a prediction repeated >=5x: {rep_episodes}/{n}")
    print("predicted-verb distribution (rollout):")
    tot = sum(preds_all.values()) or 1
    for v, c in preds_all.most_common():
        print(f"   {v:10s} {c:5d}  {c/tot*100:5.1f}%")


if __name__ == "__main__":
    main()
