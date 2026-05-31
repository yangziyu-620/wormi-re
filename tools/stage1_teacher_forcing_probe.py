"""Isolation experiment Part A: teacher-forcing next-action accuracy of a single
stage1 world-model checkpoint (no meta, no rollout).

Splits "did SFT learn obs->action at all" from meta-averaging / compounding.
Greedy-decode the action for each (instruction, observation) using the SAME
prompt template the trainer/eval use, compare to gold, report exact-match
accuracy overall + per gold-verb. Runs on a checkpoint's own scene test/train.

Usage:
  python tools/stage1_teacher_forcing_probe.py \
      --ckpt /root/autodl-tmp/wormi-checkpoints/world-vh/scene_0/last \
      --data /root/autodl-tmp/wormi-data/virtualhome-realtasks-v3-20260530/scene_0/test.jsonl \
      --max 120
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wormi.datasets.virtualhome import BASE_PROMPT


def render_prompt(tokenizer, instruction: str, observation: str) -> str:
    chat = [
        {"role": "system", "content": BASE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Instruction: {instruction}\n\n"
                f"Observation: {observation}\n\n"
                f"Action: "
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    except TypeError:
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        return text + "<|start_header_id|>assistant<|end_header_id|>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    if args.max and len(rows) > args.max:
        rows = rows[: args.max]

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    exact = 0
    verb_total = collections.Counter()
    verb_hit = collections.Counter()
    pred_verb = collections.Counter()
    examples = []
    for i, r in enumerate(rows):
        prompt = render_prompt(tok, r["instruction"], r["observation"])
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )[0]
        gen = tok.decode(out[inp["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = gen.strip().splitlines()[0].strip() if gen.strip() else ""
        gold = r["action"].strip()
        gv = gold.split()[0] if gold else ""
        pv = pred.split()[0] if pred else ""
        verb_total[gv] += 1
        pred_verb[pv] += 1
        if pred == gold:
            exact += 1
            verb_hit[gv] += 1
        if i < 12:
            examples.append((gold, pred))

    n = len(rows)
    print(f"== ckpt={args.ckpt}")
    print(f"== data={args.data}  n={n}")
    print(f"EXACT-MATCH next-action accuracy: {exact}/{n} = {exact/n*100:.1f}%")
    print("per gold-verb accuracy:")
    for v, t in verb_total.most_common():
        print(f"   {v:10s} {verb_hit[v]:4d}/{t:4d} = {verb_hit[v]/t*100:5.1f}%")
    print("predicted-verb distribution:")
    for v, c in pred_verb.most_common():
        print(f"   {v:10s} {c:4d}  {c/n*100:5.1f}%")
    print("sample (gold | pred):")
    for g, p in examples:
        print(f"   {g!r:38s} | {p!r}")


if __name__ == "__main__":
    main()
