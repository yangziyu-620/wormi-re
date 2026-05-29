import copy
from pathlib import Path
from typing import Any, override

from wormi.datasets.auto_jsonl import AutoJsonlDataset
from wormi.datasets.jsonl import JsonlDataset

BASE_PROMPT = (
    "You are a home robot agent. You can use 10 skills, (go to [object], "
    "take [object] from [object], put [object] on [object], open [object], "
    "close [object], toggle [object], heat [object] with [object], "
    "cool [object] with [object], clean [object] with [object], look). "
    'You should return only a skill after "Action:". '
    "Room: livingroom, bathroom, kitchen, bedroom."
)


class AlfworldDataset(JsonlDataset):
    dataset_type = "alfworld"
    _keys = ["observation", "action", "reward", "dones", "next_observation"]

    @override
    @classmethod
    def _load_source(cls, dataset_path: str | Path, **options):
        end_with_action = options.get("end_with_action", False)
        cumulative = options.get("cumulative", False)

        source = super()._load_source(dataset_path)
        data = list[dict[str, Any]]()

        if cumulative:
            for elem in source:
                task = elem["task"]
                init_obs = elem["history"][0]["observation"]
                history = []
                for hist in elem["history"]:
                    history.append(
                        {
                            "action": hist["action"],
                            "observation": None,
                        }
                    )
                    if len(history) > 30:
                        break
                    data.append(
                        {
                            "task": task,
                            "initial_observation": init_obs,
                            "history": copy.deepcopy(history),
                        }
                    )
                    history[-1]["observation"] = hist["next_observation"]
                    if not end_with_action:
                        data.append(
                            {
                                "task": task,
                                "initial_observation": init_obs,
                                "history": copy.deepcopy(history),
                            }
                        )
            return data

        for elem in source:
            history = [
                {
                    "action": hist["action"],
                    "observation": hist["next_observation"],
                }
                for hist in elem["history"]
            ]

            init_obs = elem["history"][0]["observation"]
            if end_with_action:
                history[-1]["observation"] = None

            data.append(
                {
                    "task": elem["task"],
                    "initial_observation": init_obs,
                    "history": history,
                }
            )

        return data

    @classmethod
    def is_valid(cls, example: dict) -> bool:
        if "task" not in example:
            return False
        if "trial_name" not in example:
            return False
        if "history" not in example:
            return False
        if any(key not in e for e in example["history"] for key in cls._keys):
            return False
        return True

    def _convert_to_chat(self, example) -> list[dict[str, str]]:
        chat = [
            {"role": "system", "content": BASE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Instruction: {example['task']}\n\n"
                    f"Observation: {example['initial_observation']}\n\n"
                    f"Action: "
                ),
            },
        ]
        for i, elem in enumerate(example["history"]):
            if i != 0:
                chat.append({"role": "user", "content": "Action: "})
            chat.append({"role": "assistant", "content": elem["action"]})
            if elem["observation"] is not None:
                chat.append({"role": "user", "content": "Next observation: "})
                chat.append(
                    {"role": "assistant", "content": elem["observation"]},
                )
        return chat


AutoJsonlDataset.register("alfworld", AlfworldDataset)
