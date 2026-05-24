# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

WorMI ("World Model Implanting for Test-time Adaptation of Embodied Agents", ICML 2025) is a research framework that implants one or more frozen "world model" CausalLMs into a frozen base CausalLM via trainable cross-attention adapters at chosen layer connections. Targets embodied-agent benchmarks (VirtualHome, ALFWorld).

Python ≥ 3.12 is required. Pinned versions matter — the model code calls into `transformers==4.45.2` / `trl==0.11.3` internals (e.g. `MllamaForConditionalGeneration`, `SFTTrainer.__init__` signature, `DataCollatorForCompletionOnlyLM`); upgrading those libraries will break things.

## Install & CLI

This repo uses **`uv`** for env/deps. Dependencies are inlined in `pyproject.toml` (no `requirements.txt`). From the repo root:

```bash
uv sync                       # create .venv and install pinned deps
uv run wormi --help           # or activate .venv first, then `wormi ...`
```

`uv sync` will pull `torch==2.5.1` and other heavy ML wheels — **don't run it on a login node**. Run it from a compute node, and consider pointing `UV_CACHE_DIR` at a larger filesystem.

The install registers the `wormi` console script (`wormi.scripts.main:main`). Subcommands are resolved from a nested dict in `wormi/scripts/main.py`:

- `wormi world train --curricula_path PATH` — SFT each world model defined in a `WorldModelCurricula`.
- `wormi world eval  --model-name NAME --dataset-path P [P ...] --output-path OUT [--num-samples N]` — accuracy eval of a single world model (uses `argparse`, not `HfArgumentParser`, and takes different flag names than the other commands).
- `wormi train       --curricula_path PATH [--test] [--interactive]` — train the WorMI integrated model from a `WorMICurricula`.
- `wormi eval        --curricula_path PATH [--model_name NAME]` — accuracy eval per test curriculum; defaults `model_name` to `<output_dir>/<run_name>/last`.

There is no test suite, lint config, or formatter pinned in the repo. `.editorconfig` sets 4-space indentation, LF, UTF-8.

## Curricula files (the user-supplied "config")

Both `train` commands take a `--curricula_path` pointing at a **Python file**, not JSON/YAML. The file is loaded via `ast.parse` + `exec` in `wormi/curricula.py` (`load_world_model_curricula`, `load_wormi_curricula`); the loader expects a top-level variable literally named `curricula` of type `WorldModelCurricula` or `WorMICurricula`. See `Readme.md` for full examples.

Path defaulting behavior worth knowing:
- If `output_dir` is left as the sentinel `/tmp/hf`, the loader rewrites it to the curricula file's parent directory (`load_wormi_curricula` uses `path.parent.parent`; `load_world_model_curricula` uses `path.parent`).
- If `name` matches the auto-generated timestamp pattern `YYYY-MM-DD_HH-MM-SS`, the loader replaces it with `path.parent.name` so checkpoints land under a meaningful folder.

## Architecture

### Model (`wormi/model.py`)

`WorMI(PreTrainedModel)` composes one frozen base LM with N frozen world LMs.

- `WorMIConfig.method` (`WorMIIntegrateMethod`) selects how world hidden states are mixed in: `CONCAT`, `ADD`, `CONCAT_WITH_ATTENTION`, `WORLD_WISE_ATTENTION`. Each picks a hook class from `wormi/modules/hooks.py`.
- `connections` is a list of base-model layer indices where adapters are attached. World models are added with `implant(world_model, connection_layer_indices)`; their connection list must have the same length as `connections` and is paired positionally.
- For each connection: an `ExtractHiddenStateHook` captures the world layer's output (forward hook), and a `BaseImplantHook` registered on the base layer reads those captured states and runs cross-attention over them. The hooks are the only trainable parameters — both base and world models are frozen via `freeze_model`.
- `forward()` runs all world models in a `torch.no_grad` block first (`_forward_world_mode`), then runs the base model. KV caches for base and world models are concatenated in `past_key_values` (base first, then world); `forward` splits them on entry by `num_base_layers`.
- `save_pretrained` is overridden to force `safe_serialization=False` (the world models' weight sharing trips safetensors).
- `state_dict` / `named_parameters` strip the name-mangled `_WorMI__base_model` / `_WorMI__world_models` keys so only the trainable hook params get serialized.
- The `vision: bool` config branch loads `MllamaForConditionalGeneration` instead of `AutoModelForCausalLM` and reaches into `model.get_decoder().layers` for the layer list.

### Known surface inconsistencies

The CLI scripts use names that don't match the current model API. If you're touching these paths, expect to either fix the scripts or update the model:

- `wormi/scripts/train.py` constructs `WorMIConfig(main_model=..., model_wise_positional_encoding=...)` but the dataclass fields are `base_model` and `world_wise_positional_encoding`. It also reads `config.main_model` from a saved config.
- `wormi/scripts/eval.py` calls `model.unplug_all()` and `model.plug(...)`; the model's actual methods are `remove_all()` and `implant(...)`. It also reads `config.main_model`.
- `WorMICurricula.merge` omits `vision` and `decaying_learning_rate` when reconstructing the merged object.

Do not "fix" these as drive-by changes when working on something else — confirm with the user, since the scripts may be intentionally pinned to a different model revision.

### Trainers (`wormi/trainer.py`)

Three layers:

1. `WorMISubTrainer(SFTTrainer)` — single SFT loop. Builds an `SFTConfig`, hard-codes `dataset_text_field="text"`, sets `pad_token = "<|end_of_text|>"`, and uses `DataCollatorForCompletionOnlyLM` with `response_template="<|start_header_id|>assistant<|end_header_id|>"`. The Llama-3 chat template is therefore baked in — datasets that don't tokenize to that template will silently produce no supervision signal.
2. `WorMITrainer` — drives multiple `WorMISubTrainer`s, one per `WorMICurriculum.train` entry, **concurrently in threads** with a ring of `Lock`s (`on_step_begin` releases the next trainer's lock and acquires its own). When `synchronize_trainers=True` (default), they share `lr_scheduler` and `global_step`. At each curriculum boundary, world models are swapped (`remove_all` → `implant`) according to the curriculum's `world_models` index list.
3. `WorMIMetaLearningTrainer` — Reptile-style: keeps a per-trainer snapshot of all model params and aggregates them across trainers at curriculum boundaries via `MetaLearningAggregationCallback`. Forces `synchronize_trainers=False` and centralizes `save_steps`.

The `WorMITrainerConfig` dataclass is the user-facing knob; it's converted to a `transformers.SchedulerType` and an `SFTConfig` inside `WorMISubTrainer.__init__`. `output_dir` defaults to `/tmp/hf` and run subfolders auto-numbered as `YYYY-MM-DD_NNN` if `run_name` is not set.

### Datasets (`wormi/datasets/`)

`JsonlDataset` extends `datasets.Dataset` + `ChatDataset`. Concrete subclasses (`AlfworldDataset`, `VirtualHomeDataset`) implement `is_valid(example)` for schema sniffing. `AutoJsonlDataset.load(path)` opens the file once, picks the first registered subclass whose `is_valid` accepts every row, and dispatches to its `load`. Subclasses must self-register on import — `wormi/datasets/__init__.py` imports them so registration happens when the package is imported.

`as_chat(tokenizer)` produces a `text` column shaped for the Llama-3 chat template, which is what the trainers expect. Common load options seen in scripts: `end_with_action=True`, `cumulative=True`.

Trainers expect dataset directories laid out as `{dataset_root}/{name}/{train,test}.jsonl` and optionally `unknown.jsonl` (merged into train for world model training).

### Module internals (`wormi/modules/`)

- `hooks.py` — the four implant-hook variants. They are `nn.Module`s (so their parameters get tracked) but are *registered* via `Module.register_forward_hook`. `WorldWiseAttentionImplantHook` is the variant used for the headline ICML method; the others are ablations.
- `layers.py` — custom `MultiheadAttention`, `MLP`, `world_wise_positional_encoding`, plus `freeze_model` (sets `requires_grad=False` and `eval()`).
- `utils.py` — `get_connections(num, total)` evenly spaces connection indices; `compose_causal_lm_output` merges per-world-model outputs into a single `CausalLMOutputWithPast`.

### Model selection (`wormi/model_store.py`)

`ModelStore` clusters dataset prototypes (sentence embeddings averaged per dataset) with KMeans (`sklearn.cluster.KMeans`) so that at test time the right world model can be retrieved by nearest cluster. Not wired into the CLI scripts; used programmatically.
