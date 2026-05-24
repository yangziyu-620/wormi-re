# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for CALM."""

from typing import Iterable, TypeGuard, TypeVar, cast, overload, override

import numpy as np
import scipy
import scipy.stats
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

T = TypeVar("T")


def check_connections(
    connections: list[tuple[int, int]],
    num_base_layers: int,
    num_world_model_layers: int,
) -> bool:
    """Checks if the connections are valid."""
    for connection in connections:
        if connection[0] < 0 or connection[0] >= num_base_layers:
            print(
                f"Please verify your connections again. Index {connection[0]} doesn't"
                f" exist as base model only has {num_base_layers} layers"
            )
            return False
        if connection[1] < 0 or connection[1] >= num_world_model_layers:
            print(
                f"Please verify your connections again. Index {connection[1]} doesn't"
                f" exist as world model only has {num_world_model_layers} layers"
            )
            return False
    return True


def get_connections(num_connections: int, num_layers: int) -> list[int]:
    """Gets the connections for PlugAndPlay."""
    conn = np.linspace(0, num_layers - 1, num_connections, dtype=int)

    return [int(x) for x in conn]


def get_hidden_dim(model: nn.Module) -> int:
    """Gets the hidden dimensions for the given layers."""
    if hasattr(model, "hidden_size") and isinstance(model.hidden_size, int):
        return model.hidden_size

    hidden_dim = model.model.layers[0].hidden_size
    for layer in model.model.layers[1:]:
        if hidden_dim != layer.hidden_size:
            raise ValueError("All layers should have the same hidden size.")
    return hidden_dim


def is_all_causal_lm_output_with_past(
    outputs,
) -> TypeGuard[list[CausalLMOutputWithPast]]:
    return all(isinstance(output, CausalLMOutputWithPast) for output in outputs)


def filter_none(lst: Iterable[T | None]) -> Iterable[T]:
    return (x for x in lst if x is not None)


def compose_causal_lm_output(
    outputs: list[CausalLMOutputWithPast],
) -> CausalLMOutputWithPast:
    if len(
        losses := list(filter_none(output.loss for output in outputs))
    ) == len(outputs):
        loss = cast(
            torch.FloatTensor,
            torch.stack(cast(list[torch.Tensor], losses)).mean(),
        )
    else:
        loss = None

    logits = cast(
        torch.FloatTensor,
        torch.concat([o.logits for o in outputs], dim=-2),
    )

    if len(
        pkvs := list(filter_none(o.past_key_values for o in outputs))
    ) == len(outputs):
        past_key_values = sumtpl(pkvs)
    else:
        past_key_values = None

    if len(
        h_states := list(filter_none(o.hidden_states for o in outputs))
    ) == len(outputs):
        hidden_states = sumtpl(h_states)
    else:
        hidden_states = None

    if len(attns := list(filter_none(o.attentions for o in outputs))) == len(
        outputs
    ):
        attentions = sumtpl(attns)
    else:
        attentions = None

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=cast(tuple[tuple[torch.FloatTensor]], past_key_values),
        hidden_states=hidden_states,
        attentions=attentions,
    )


def sumtpl(it: Iterable[tuple[T, ...]]) -> tuple[T, ...]:
    return sum(it, start=tuple())


class TensorSet(torch.Tensor):
    @override
    def dist(
        self, other: torch.Tensor, p: int | float | bool | complex = 2
    ) -> torch.Tensor:
        if isinstance(other, TensorSet):
            return torch.tensor(
                scipy.stats.wasserstein_distance_nd(self.cpu(), other.cpu())
            )
        return super(TensorSet).dist(other, p)
