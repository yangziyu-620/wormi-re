from typing import Iterable, Sequence, overload, override

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from wormi.modules.utils import TensorSet

_Dataset = Iterable[str]


class ModelStore(Sequence[PreTrainedModel]):
    def __init__(
        self,
        sentence_embedding_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        n_clusters: int = 12,
    ):
        self.__sentence_embedding_model = sentence_embedding_model
        self.__tokenizer = tokenizer
        self.__n_clusters = n_clusters
        self.__models = list[tuple[PreTrainedModel, TensorSet]]()

        self.sentence_embedding_model.eval()

    @property
    def sentence_embedding_model(self) -> PreTrainedModel:
        return self.__sentence_embedding_model

    @property
    def tokenizer(self) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        return self.__tokenizer

    @property
    def n_clusters(self):
        return self.__n_clusters

    def _compute_prototype(self, dataset: _Dataset) -> TensorSet:
        texts = list(dataset)
        if not texts:
            raise ValueError("Cannot compute a prototype from an empty dataset.")

        with torch.no_grad():
            input_ids = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.sentence_embedding_model.device)
            outputs = self.sentence_embedding_model(**input_ids)

        mask = input_ids["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * mask).sum(dim=1)
        embeddings = embeddings / mask.sum(dim=1).clamp(min=1)
        embeddings = embeddings.detach().cpu()

        k = min(self.__n_clusters, embeddings.shape[0])
        center_indices = [0]
        min_dist = torch.cdist(embeddings, embeddings[[0]]).squeeze(1)
        for _ in range(1, k):
            idx = int(torch.argmax(min_dist).item())
            center_indices.append(idx)
            dist = torch.cdist(embeddings, embeddings[[idx]]).squeeze(1)
            min_dist = torch.minimum(min_dist, dist)

        return TensorSet(embeddings[center_indices])

    @overload
    def __getitem__(self, index: int) -> PreTrainedModel: ...

    @overload
    def __getitem__(self, index: slice) -> list[PreTrainedModel]: ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            return self.__models[index][0]
        else:
            return [x[0] for x in self.__models[index]]

    def __setitem__(
        self, index: int, value: tuple[PreTrainedModel, _Dataset]
    ) -> None:
        model, dataset = value
        self.__models[index] = (model, self._compute_prototype(dataset))

    def __delitem__(self, index: int) -> None:
        del self.__models[index]

    def __len__(self) -> int:
        return len(self.__models)

    def insert(
        self, index: int, model: PreTrainedModel, dataset: _Dataset, /
    ) -> None:
        return self.__models.insert(
            index, (model, self._compute_prototype(dataset))
        )

    def append(self, model: PreTrainedModel, dataset: _Dataset) -> None:
        return self.insert(len(self), model, dataset)

    def clear(self) -> None:
        self.__models.clear()

    def reverse(self) -> None:
        self.__models.reverse()

    def extend(
        self,
        models: Iterable[PreTrainedModel],
        datasets: Iterable[_Dataset],
    ) -> None:
        for model, dataset in zip(models, datasets):
            self.append(model, dataset)

    def pop(self, index: int = -1) -> PreTrainedModel:
        return self.__models.pop(index)[0]

    def remove(self, model: PreTrainedModel) -> None:
        self.__models = list(filter(lambda x: x[0] != model, self.__models))

    def __iadd__(
        self,
        other: "ModelStore" | Iterable[tuple[PreTrainedModel, _Dataset]],
    ) -> "ModelStore":
        if isinstance(other, ModelStore):
            self.__models.extend(other.__models)
        else:
            self.extend(*zip(*other))
        return self

    def retrieve(
        self, prototype: TensorSet, k: int = 4
    ) -> list[PreTrainedModel]:
        m = self.__models
        p = prototype
        top_k = list(sorted(m, key=lambda x: float(x[1].dist(p))))[:k]
        return [x[0] for x in top_k]
