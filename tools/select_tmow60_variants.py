"""Path-(a) step 1: select 20 object-rich init-graph variants (6 seen / 14 unseen)
from the VirtualHome ActivityPrograms candidates, maximizing feasibility of the
tmow_const 78-task suite under equivalence mapping + loose filtering.

Writes scene_inits_tmow60.json:
  { "domains": [ {name, apartment, seen, feasible_task_ids:[...], graph:{...}} x20 ],
    "meta": {...} }
Decouples slow per-variant scoring from generation. Run once (background).
"""
from __future__ import annotations
import copy, json, glob, random, collections
from pathlib import Path
import sys
sys.path.insert(0, "/root/WorMI")
from tools import build_virtualhome_dataset as base
from tools import tmow_const as T

VH_SRC = Path("/root/autodl-tmp/wormi-data/virtualhome-src")
EQ = json.loads((VH_SRC / "virtualhome/resources/class_name_equivalence.json").read_text())
IFG = "/root/autodl-tmp/wormi-data/raw/programs_processed_precond_nograb_morepreconds/init_and_final_graphs"
OUT = Path("/root/autodl-tmp/wormi-data/scene-inits/scene_inits_tmow60.json")
CAND_PER_APT = 30
PER_BASE = [3, 3, 3, 3, 3, 3, 2]       # 20 domains across the 7 apartments
SEEN_PER_BASE = [1, 1, 1, 1, 1, 1, 0]  # 6 seen domains, spread across apartments
MAX_STEPS = 18
SEED = 42

eg = base._bootstrap_evolving_graph(VH_SRC)
EnvironmentGraph = eg["environment"].EnvironmentGraph
read_script = eg["scripts"].read_script_from_string
ScriptExecutor = eg["execution"].ScriptExecutor


def candidates(o):
    s = {o, o.replace("_", "")}
    if o in EQ:
        s |= set(EQ[o]) if isinstance(EQ[o], list) else {EQ[o]}
    for k, v in EQ.items():
        vs = v if isinstance(v, list) else [v]
        if o == k or o in vs:
            s.add(k); s |= set(vs)
    return s


def resolve(o, clsset):
    if o in clsset:
        return o
    for c in candidates(o):
        if c in clsset:
            return c
    return None


def feasible(family, gargs, g):
    try:
        sl, at, _ = base._paperlike_program(family, gargs, g)
    except Exception:
        return False
    if sl is None or at is None or len(at) > MAX_STEPS:
        return False
    try:
        ok, _f, gsl = ScriptExecutor(EnvironmentGraph(copy.deepcopy(g)),
                                     name_equivalence={}).execute(read_script(", ".join(sl)), w_graph_list=True)
    except Exception:
        return False
    if not ok or len(gsl) != len(at) + 1:
        return False
    return base._goal_satisfied(gsl[-1], family, gargs)  # loose: executed + goal reached


TASKS = [(i, *T.task_to_tuple(i)) for i in range(78)]


def feasible_tasks_for_graph(g):
    cls = {n["class_name"] for n in g["nodes"]}
    out = []
    for idx, fam, args in TASKS:
        gargs = tuple(resolve(o, cls) for o in args)
        if any(a is None for a in gargs):
            continue
        if feasible(fam, gargs, g):
            out.append(idx)
    return out


def main():
    rng = random.Random(SEED)
    apts = sorted({fp.split("/init_and_final_graphs/")[1].split("/")[0]
                   for fp in glob.glob(IFG + "/*/*/*.json")})
    print("apartments:", apts, flush=True)
    scored = {a: [] for a in apts}
    for ai, apt in enumerate(apts):
        files = glob.glob(f"{IFG}/{apt}/*/*.json")
        rng.shuffle(files)
        picked = 0
        for fp in files:
            if picked >= CAND_PER_APT:
                break
            try:
                g = json.load(open(fp))["init_graph"]
            except Exception:
                continue
            feas = feasible_tasks_for_graph(g)
            scored[apt].append((fp, g, feas))
            picked += 1
        best = max((len(f) for _, _, f in scored[apt]), default=0)
        print(f"  {apt}: scored {len(scored[apt])}, best feasible={best}", flush=True)

    domains = []
    for ai, apt in enumerate(apts):
        cand = sorted(scored[apt], key=lambda x: -len(x[2]))
        take = cand[:PER_BASE[ai]]
        for j, (fp, g, feas) in enumerate(take):
            seen = j < SEEN_PER_BASE[ai]
            domains.append({
                "name": f"{apt}__sel{j}",
                "apartment": apt,
                "source_file": fp,
                "seen": seen,
                "feasible_task_ids": feas,
                "graph": g,
            })
    n_seen = sum(d["seen"] for d in domains)
    total = sum(len(d["feasible_task_ids"]) for d in domains)
    cov = set().union(*[set(d["feasible_task_ids"]) for d in domains]) if domains else set()
    print(f"\nselected {len(domains)} domains, seen={n_seen}, unseen={len(domains)-n_seen}", flush=True)
    print(f"total feasible (task,domain): {total} | distinct tasks: {len(cov)}/78 "
          f"| seen-task: {len(cov & set(T.SEEN_TASKS))}/16", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump({
            "meta": {"seed": SEED, "cand_per_apt": CAND_PER_APT, "per_base": PER_BASE,
                     "seen_per_base": SEEN_PER_BASE, "total_feasible_pairs": total,
                     "distinct_tasks": len(cov), "seen_tasks_covered": len(cov & set(T.SEEN_TASKS))},
            "domains": domains,
        }, f)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
