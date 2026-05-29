from abc import abstractmethod

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


class ChatDataset(Dataset):
    def as_chat(self, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        def process(example):
            return {
                **example,
                "text": tokenizer.apply_chat_template(
                    self._convert_to_chat(example),
                    tokenize=False,
                ),
            }

        return self.map(process)

    @abstractmethod
    def _convert_to_chat(self, example) -> list[dict[str, str]]:
        raise NotImplementedError
