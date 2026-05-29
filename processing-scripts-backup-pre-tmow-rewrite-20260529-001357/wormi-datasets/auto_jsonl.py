import json
from pathlib import Path

from wormi.datasets.jsonl import JsonlDataset


class AutoJsonlDataset:
    datasets: dict[str, type[JsonlDataset]] = {}

    @classmethod
    def register(
        cls, dataset_type: str, dataset_cls: type[JsonlDataset]
    ) -> None:
        cls.datasets[dataset_type] = dataset_cls

    @classmethod
    def load(cls, dataset_path: str | Path, **options) -> JsonlDataset:
        with open(dataset_path) as f:
            examples = [json.loads(line) for line in f.readlines()]

        for _, dataset_cls in cls.datasets.items():
            if all(dataset_cls.is_valid(example) for example in examples):
                return dataset_cls.load(dataset_path, **options)
        raise ValueError(
            "Unsupporrted dataset format. Please check the dataset format. "
            "If you made a new dataset type, please register it using "
            "AutoJsonlDataset.register()"
        )
