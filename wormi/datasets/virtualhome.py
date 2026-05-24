from pathlib import Path
from typing import override

from wormi.datasets.auto_jsonl import AutoJsonlDataset
from wormi.datasets.jsonl import JsonlDataset

BASE_PROMPT = (
    "You are a home robot agent. You can use 6 skills, (walk [object or room], "
    "grab [object], switch [object], open [object], putin [target object], put "
    '[target object]). You should return only a skill after "Action:". Room: '
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
                elem_no_obs = elem.copy()
                elem_no_obs["next_observation"] = None
                data.append(elem_no_obs)
                if not end_with_action:
                    data.append(elem)
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
        chat = [
            {"role": "system", "content": BASE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Instruction: {example['instruction']}\n\n"
                    f"Observation: {example['observation']}\n\n"
                    f"Action: "
                ),
            },
            {"role": "assistant", "content": example["action"]},
        ]
        if example["next_observation"] is not None:
            chat.append(
                {"role": "user", "content": "Next observation: "},
            )
            chat.append(
                {"role": "assistant", "content": example["next_observation"]},
            )
        return chat


AutoJsonlDataset.register("virtualhome", VirtualHomeDataset)
