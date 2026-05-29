from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast, override

from datasets import DatasetDict
from numpy.random._generator import Generator as Generator
from typing_extensions import Self

from wormi.datasets.chat import ChatDataset


class JsonlDataset(ABC, ChatDataset):
    dataset_type = "base"

    @classmethod
    def _load_source(
        cls, dataset_path: str | Path, **options
    ) -> list[dict[str, Any]]:
        with open(dataset_path) as f:
            return [json.loads(line) for line in f]

    @classmethod
    def load(cls, dataset_path: str | Path, **options):
        source = cls._load_source(dataset_path, **options)
        keys = source[0].keys()
        data = {key: [elem[key] for elem in source] for key in keys}
        return cast(Self, cls.from_dict(data))

    def merge(self, other: JsonlDataset) -> JsonlDataset:
        data = {col: self[col] + other[col] for col in self.column_names}
        return cast(JsonlDataset, self.from_dict(data))

    @classmethod
    @abstractmethod
    def is_valid(cls, example: dict) -> bool:
        raise NotImplementedError

    @override
    def train_test_split(
        self,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        shuffle: bool = True,
        stratify_by_column: str | None = None,
        seed: int | None = None,
        generator: Generator | None = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool | None = None,
        train_indices_cache_file_name: str | None = None,
        test_indices_cache_file_name: str | None = None,
        writer_batch_size: int | None = 1000,
        train_new_fingerprint: str | None = None,
        test_new_fingerprint: str | None = None,
    ) -> DatasetDict:
        return super().train_test_split(
            test_size,
            train_size,
            shuffle,
            stratify_by_column,
            seed,
            generator,
            keep_in_memory,
            load_from_cache_file,
            train_indices_cache_file_name,
            test_indices_cache_file_name,
            writer_batch_size,
            train_new_fingerprint,
            test_new_fingerprint,
        )
