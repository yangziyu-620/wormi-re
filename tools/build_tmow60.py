"""Path-(a) step 2: build the WorMI VirtualHome dataset from the 20 selected
object-rich variants (scene_inits_tmow60.json), tmow_const 78-task suite,
equivalence-mapped objects, full-graph instance-grounded observations, loose
filtering (executed + goal reached). WorMI JSONL layout, 3 eval columns.

instruction keeps the ORIGINAL tmow object name (task identity); the graph
program/observation use the equivalence-resolved class present in the graph.
"""
from __future__ import annotations
import copy, json, os, random, collections
from pathlib import Path
import sys
sys.path.insert(0, "/root/WorMI")
from tools import build_virtualhome_dataset as base
from tools import build_virtualhome_dataset_tmow_compact as cbuild
from tools import tmow_const as T
from tools import compact_virtualhome_observations as C

# OBS_MODE: "full" (instance-grounded full graph) or "compact" (TMoW retrieve-17 + delta)
OBS_MODE = os.environ.get("OBS_MODE", "compact")
COMPACT_NUM_EDGES = int(os.environ.get("COMPACT_NUM_EDGES", "17"))
VH_SRC = Path("/root/autodl-tmp/wormi-data/virtualhome-src")
EQ = json.loads((VH_SRC / "virtualhome/resources/class_name_equivalence.json").read_text())
SEL = Path("/root/autodl-tmp/wormi-data/scene-inits/scene_inits_tmow60.json")
_SUFFIX = "compact" if OBS_MODE == "compact" else "full"
OUT = Path(f"/root/autodl-tmp/wormi-data/virtualhome-tmow60-noUnity-{_SUFFIX}-20260529")
SEED = 42
SEEN_SEEN_EVAL_PER_TASK = 2
MAX_STEPS = 18
SEEN_TASKS = set(T.SEEN_TASKS)

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


def _write_jsonl(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def gen_rows(domain, task_idx):
    fam, orig_args = T.task_to_tuple(task_idx)
    g = domain["graph"]
    cls = {n["class_name"] for n in g["nodes"]}
    gargs = tuple(resolve(o, cls) for o in orig_args)
    if any(a is None for a in gargs):
        return None
    try:
        sl, at, dbg = base._paperlike_program(fam, gargs, g)
    except Exception:
        return None
    if sl is None or at is None or len(at) > MAX_STEPS:
        return None
    try:
        ok, _f, gsl = ScriptExecutor(EnvironmentGraph(copy.deepcopy(g)),
                                     name_equivalence={}).execute(read_script(", ".join(sl)), w_graph_list=True)
    except Exception:
        return None
    if not ok or len(gsl) != len(at) + 1:
        return None
    if not base._goal_satisfied(gsl[-1], fam, gargs):  # loose: executed + goal reached
        return None

    instruction = base.instruction_text(fam, orig_args)   # ORIGINAL object name
    sel = C.select_task_instances(g, fam, gargs)
    sel_ids = C.selected_instance_ids_from_selection(sel)
    tid = f"{domain['name']}:{fam}:{'|'.join(orig_args)}"
    rows = []
    for i, act in enumerate(at):
        if OBS_MODE == "compact":
            # reuse TMoW compact renderer; grounding uses RESOLVED args
            tmp = {"instruction": instruction, "observation": "", "action": act,
                   "next_observation": "", "_meta": {"task_args": list(gargs)}}
            crow = cbuild._compact_row_from_graph_states(
                tmp, gsl[i], gsl[i + 1], num_edges=COMPACT_NUM_EDGES, next_mode="delta")
            obs, nxt = crow["observation"], crow["next_observation"]
            prep = crow["_meta"].get("observation_preprocessing")
            obs_fmt = "tmow_compact_from_graph_state"
        else:
            obs = C.format_instance_grounded_observation(gsl[i], task_args=list(gargs), selected_node_ids=sel_ids)
            nxt = C.format_instance_grounded_observation(gsl[i + 1], task_args=list(gargs), selected_node_ids=sel_ids)
            prep = None
            obs_fmt = "instance_grounded_full"
        rows.append({
            "instruction": instruction,
            "observation": obs,
            "action": act,
            "next_observation": nxt,
            "_meta": {
                "scene": domain["name"],
                "apartment": domain["apartment"],
                "split": None,
                "task_id": f"{fam}:{'|'.join(orig_args)}",
                "task_family": fam,
                "task_args": list(orig_args),
                "resolved_args": list(gargs),
                "trajectory_id": tid,
                "step_index": i,
                "num_steps": len(at),
                "generator_mode": "paper_like_graph_planner_tmow60_noUnity",
                "observation_format": obs_fmt,
                "observation_preprocessing": prep,
                "protocol": "wormi_tmow60_noUnity_v1",
                "planner_debug": dbg,
            },
        })
    return rows


def main():
    sel = json.loads(SEL.read_text())
    domains = sel["domains"]
    seen_domains = [d for d in domains if d["seen"]]
    print(f"domains: {len(domains)} (seen {len(seen_domains)})", flush=True)

    # generate, tag split
    buckets = {"seen_seen": [], "seen_unseen": [], "unseen_seen": [], "unseen_unseen": []}
    traj_meta = []
    for d in domains:
        ds = d["seen"]
        for tidx in d["feasible_task_ids"]:
            rows = gen_rows(d, tidx)
            if not rows:
                continue
            ts = tidx in SEEN_TASKS
            tag = ("seen" if ts else "unseen") + "_" + ("seen" if ds else "unseen")
            for r in rows:
                r["_meta"]["split"] = tag
            buckets[tag].extend(rows)
            traj_meta.append({"trajectory_id": rows[0]["_meta"]["trajectory_id"],
                              "task_id": rows[0]["_meta"]["task_id"], "scene": d["name"],
                              "split": tag, "num_steps": rows[0]["_meta"]["num_steps"]})
    print("bucket trajectory counts:",
          {k: len({r["_meta"]["trajectory_id"] for r in v}) for k, v in buckets.items()}, flush=True)

    # per-seen-domain pools: seen-task and unseen-task (both seen-scene)
    seen_names = [d["name"] for d in seen_domains]
    name_to_idx = {n: i for i, n in enumerate(seen_names)}
    per_scene_seen = {i: collections.defaultdict(list) for i in range(len(seen_names))}
    per_scene_unseen = {i: collections.defaultdict(list) for i in range(len(seen_names))}
    for r in buckets["seen_seen"]:
        per_scene_seen[name_to_idx[r["_meta"]["scene"]]][r["_meta"]["trajectory_id"]].append(r)
    for r in buckets["unseen_seen"]:
        per_scene_unseen[name_to_idx[r["_meta"]["scene"]]][r["_meta"]["trajectory_id"]].append(r)

    # hold out eval_a (seen_seen column) from the SEEN-task pool only
    eval_ids = base._select_seen_seen_eval_ids(per_scene_seen, SEEN_SEEN_EVAL_PER_TASK, SEED)
    test_seen_seen = []
    for si, trajs in per_scene_seen.items():
        for tid in list(trajs):
            if tid in eval_ids:
                test_seen_seen.extend(trajs[tid])
                del trajs[tid]
    # train = remaining seen-task + all unseen-task (seen-scene world-model data)
    per_scene_train = {i: collections.defaultdict(list) for i in range(len(seen_names))}
    for si in range(len(seen_names)):
        for tid, rows in per_scene_seen[si].items():
            per_scene_train[si][tid] = rows
        for tid, rows in per_scene_unseen[si].items():
            per_scene_train[si][tid] = rows

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    columns = {
        "test_seen_task_seen_scene.jsonl": test_seen_seen,
        "test_seen_task_unseen_scene.jsonl": buckets["seen_unseen"],
        "test_unseen_task_unseen_scene.jsonl": buckets["unseen_unseen"],
    }
    all_rows = []
    print("\noutput:", flush=True)
    for fn, rows in columns.items():
        rng.shuffle(rows)
        _write_jsonl(OUT / fn, rows)
        all_rows += rows
        print(f"  {fn}: {len(rows)} rows, {len({r['_meta']['trajectory_id'] for r in rows})} traj", flush=True)

    for dirname, fn in [("eval_col_1_seen_seen", "test_seen_task_seen_scene.jsonl"),
                        ("eval_col_2_seen_unseen", "test_seen_task_unseen_scene.jsonl"),
                        ("eval_col_3_unseen_unseen", "test_unseen_task_unseen_scene.jsonl")]:
        ed = OUT / dirname; ed.mkdir(exist_ok=True)
        link = ed / "test.jsonl"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / fn)

    train_total = 0
    for si, name in enumerate(seen_names):
        sd = OUT / f"scene_{si}"; sd.mkdir(exist_ok=True)
        rows = [r for tid in sorted(per_scene_train[si])
                for r in sorted(per_scene_train[si][tid], key=lambda x: x["_meta"]["step_index"])]
        rng.shuffle(rows)
        _write_jsonl(sd / "train.jsonl", rows)
        all_rows += rows
        train_total += len({r["_meta"]["trajectory_id"] for r in rows})
        link = sd / "test.jsonl"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / "test_seen_task_seen_scene.jsonl")
        print(f"  scene_{si} ({name}): train {len(rows)} rows, "
              f"{len({r['_meta']['trajectory_id'] for r in rows})} traj", flush=True)

    all_traj = {r["_meta"]["trajectory_id"] for r in all_rows}
    all_tasks = {r["_meta"]["task_id"] for r in all_rows}
    summary = {
        "data_root": str(OUT),
        "dataset_mode": f"tmow60_noUnity_v1_{_SUFFIX}",
        "observation_format": ("tmow_compact_from_graph_state" if OBS_MODE == "compact" else "instance_grounded_full"),
        "filtering": "loose (executed + goal_reached only)",
        "total_rows": len(all_rows),
        "total_trajectories": len(all_traj),
        "train_trajectories": train_total,
        "distinct_tasks": len(all_tasks),
        "seen_tasks_present": len(all_tasks & {f"{T.task_to_tuple(i)[0]}:{'|'.join(T.task_to_tuple(i)[1])}" for i in SEEN_TASKS}),
        "seen_scenes": len(seen_names),
        "unseen_scenes": len(domains) - len(seen_names),
        "column_trajectories": {fn: len({r["_meta"]["trajectory_id"] for r in rows}) for fn, rows in columns.items()},
        "note": "Path (a): real graphs only, no Unity. 1023/78/16 not fully reached due to "
                "missing objects in available VirtualHome graphs.",
    }
    with (OUT / "tmow60_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (OUT / "trajectory_manifest.json").open("w") as f:
        json.dump({"trajectories": traj_meta}, f)
    print("\nSUMMARY:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
