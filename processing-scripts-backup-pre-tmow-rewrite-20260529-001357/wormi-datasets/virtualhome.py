from pathlib import Path
from typing import override

from wormi.datasets.auto_jsonl import AutoJsonlDataset
from wormi.datasets.jsonl import JsonlDataset

BASE_PROMPT = (
    "You are a home robot agent. You can use 6 skills, (walk [object or room], "
    "grab [object], switchon [object], open [object], putin [object] "
    '[target object], put [object] [target object]). You should return only a '
    'skill after "Action:". Room: '
    "livingroom, bathroom, kitchen, bedroom."
)


class VirtualHomeDataset(JsonlDataset):
    dataset_type = "virtualhome"

    @override
    @classmethod
    def _load_source(cls, dataset_path: str | Path, **options):
        end_with_action = options.get("end_with_action", False)
        source = super()._load_source(dataset_path)
        data = []
        for idx, elem in enumerate(source):
            if elem["instruction"] != "No instruction" or not end_with_action:
                elem_action = elem.copy()
                elem_action["next_observation"] = None
                elem_action["auxiliary_task"] = "behavior_cloning"
                data.append(elem_action)
                if not end_with_action:
                    elem_dynamics = elem.copy()
                    elem_dynamics["auxiliary_task"] = "dynamics"
                    data.append(elem_dynamics)

                    elem_affordance = elem.copy()
                    elem_affordance["next_observation"] = None
                    elem_affordance["auxiliary_task"] = "affordance"
                    data.append(elem_affordance)
        return data

    @classmethod
    def is_valid(cls, example: dict) -> bool:
        return (
            "instruction" in example
            and "observation" in example
            and "action" in example
            and "next_observation" in example
        )

    @override
    def _convert_to_chat(cls, example) -> list[dict[str, str]]:
        auxiliary_task = example.get("auxiliary_task", "behavior_cloning")
        if auxiliary_task == "affordance":
            return [
                {"role": "system", "content": BASE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Auxiliary task: identify one feasible next action for "
                        "the current state.\n\n"
                        f"Observation: {example['observation']}\n\n"
                        f"Feasible action: "
                    ),
                },
                {"role": "assistant", "content": example["action"]},
            ]
        if auxiliary_task == "dynamics":
            return [
                {"role": "system", "content": BASE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Auxiliary task: predict the next observation after "
                        "executing the given action.\n\n"
                        f"Instruction: {example['instruction']}\n\n"
                        f"Observation: {example['observation']}\n\n"
                        f"Action: {example['action']}\n\n"
                        f"Next observation: "
                    ),
                },
                {"role": "assistant", "content": example["next_observation"]},
            ]

        chat = [
            {"role": "system", "content": BASE_PROMPT},
            {
                "role": "user",
                "content": (
                    "Auxiliary task: predict the next action conditioned on "
                    "the instruction and current state.\n\n"
                    f"Instruction: {example['instruction']}\n\n"
                    f"Observation: {example['observation']}\n\n"
                    f"Action: "
                ),
            },
            {"role": "assistant", "content": example["action"]},
        ]
        return chat


AutoJsonlDataset.register("virtualhome", VirtualHomeDataset)
