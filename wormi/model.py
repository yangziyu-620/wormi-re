import os
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from typing import Callable, Iterable, TypeVar, cast

import torch
import torch.utils.hooks
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Cache,
    DynamicCache,
    MllamaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from wormi.modules.hooks import (
    AddImplantHook,
    BaseImplantHook,
    ConcatImplantHook,
    ConcatWithAttentionImplantHook,
    ExtractHiddenStateHook,
    WorldWiseAttentionImplantHook,
)
from wormi.modules.layers import freeze_model
from wormi.modules.utils import (
    compose_causal_lm_output,
    get_connections,
    get_hidden_dim,
)

T = TypeVar(
    "T",
    bound=tuple[DynamicCache | torch.Tensor, ...] | DynamicCache | torch.Tensor,
)


class WorMIIntegrateMethod(Enum):
    CONCAT = "concat"
    ADD = "add"
    CONCAT_WITH_ATTENTION = "concat-attn"
    WORLD_WISE_ATTENTION = "world-wise-attn"


WORLD_WISE_POS_ENC_METHODS = (
    WorMIIntegrateMethod.CONCAT,
    WorMIIntegrateMethod.CONCAT_WITH_ATTENTION,
)


class WorMIConfig(PretrainedConfig):
    model_type = "plug-and-play"

    def __init__(
        self,
        method: str = "concat",
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_model_config: AutoConfig | PretrainedConfig | None = None,
        connections: Iterable[int] | None = None,
        num_connections: int | None = None,
        num_heads: int = 1,
        self_attention: bool = True,
        world_wise_positional_encoding: bool = False,
        base_hidden_dim: int | None = None,
        world_model_hidden_dim: int | None = None,
        latent_variable_length: int = 128,
        vision: bool = False,
        **kwargs,
    ):
        self.method = method
        self.base_model = base_model
        self.base_model_config = base_model_config
        self.connections = (
            list(connections) if connections is not None else None
        )
        self.num_connections = num_connections
        self.num_heads = num_heads
        self.self_attention = self_attention
        self.world_wise_positional_encoding = world_wise_positional_encoding
        self.base_hidden_dim = base_hidden_dim
        self.world_model_hidden_dim = world_model_hidden_dim
        self.latent_variable_length = latent_variable_length
        self.vision = vision
        super().__init__(**kwargs)


class WorMI(PreTrainedModel, GenerationMixin):
    config_class = WorMIConfig

    @property
    def lm_head(self):
        return self.__base_model.lm_head

    @property
    def base_model(self):
        return self.__base_model

    @property
    def base_model_hidden_dim(self):
        return self.__base_model_hidden_dim

    @property
    def world_models(self):
        return self.__world_models

    @property
    def world_model_hidden_dim(self):
        if self.__world_model_hidden_dim is None:
            raise ValueError(
                "World model hidden dimension is not set. "
                "Please call `build` or `maybe_build` before accessing."
            )
        return self.__world_model_hidden_dim

    @property
    def num_world_models(self):
        return len(self.__world_models)

    @property
    def vocab_size(self):
        return self.__vocab_size

    @property
    def rms_norm_eps(self):
        return self.__rms_norm_eps

    @property
    def num_connections(self):
        return len(self.__connections)

    @property
    def connections(self):
        return self.__connections

    @property
    def extract_hidden_state_hooks(self):
        return self.__extract_hidden_state_hooks

    @property
    def device(self):
        return self.base_model.device

    @property
    def has_model_been_built(self):
        return self.__has_model_been_built

    def __init__(self, config: WorMIConfig):
        if config.base_model_config is None:
            config.base_model_config = AutoConfig.from_pretrained(
                config.base_model
            )

        super().__init__(config)  # pylint: disable=too-many-function-args
        self.config = config
        try:
            self.method = WorMIIntegrateMethod(config.method)
        except ValueError:
            raise ValueError(
                f"Invalid method: {config.method}. Valid methods are: "
                f"{', '.join(m.value for m in WorMIIntegrateMethod)}."
            )

        if self.config.vision:
            self.__base_model = MllamaForConditionalGeneration.from_pretrained(
                config.base_model,
                config=config.base_model_config,  # type: ignore
                torch_dtype=torch.bfloat16,
            )

            self.__base_config = self.__base_model.config.get_text_config()
            self.__base_layers = self.__base_model.get_decoder().layers
        else:
            self.__base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                config=config.base_model_config,
                torch_dtype=torch.bfloat16,
            )
            self.__base_config = self.__base_model.config
            self.__base_layers = self.__base_model.model.layers
        if config.base_hidden_dim is None:
            self.__base_model_hidden_dim = get_hidden_dim(self.__base_model)
            self.config.base_hidden_dim = self.__base_model_hidden_dim
        else:
            self.__base_model_hidden_dim = config.base_hidden_dim
        freeze_model(self.__base_model)

        self.__world_models = OrderedDict[int, PreTrainedModel]()
        self.__world_model_next_key = 0
        self.__world_model_hidden_dim = None

        self.__vocab_size = self.__base_config.vocab_size
        self.__num_base_layers = len(self.__base_layers)
        self.__rms_norm_eps = self.__base_config.rms_norm_eps

        if config.connections is not None:
            self.__connections = config.connections
        elif config.num_connections is not None:
            self.__connections = get_connections(
                config.num_connections, self.__num_base_layers
            )
        else:
            raise ValueError(
                "Either connections or num_connections must be provided."
            )

        self.__extract_hidden_state_hooks = {
            conn: dict[int, ExtractHiddenStateHook]()
            for conn in self.__connections
        }
        self.cross_attention_hooks = cast(
            list[BaseImplantHook], torch.nn.ModuleList()
        )

        self.__has_model_been_built = False

        if (
            self.config.base_hidden_dim is not None
            and self.config.world_model_hidden_dim is not None
        ):
            self.maybe_build(
                world_model_hidden_dim=self.config.world_model_hidden_dim
            )

    def build(self, world_model_hidden_dim: int):
        self.__world_model_hidden_dim = world_model_hidden_dim
        self.config.world_model_hidden_dim = world_model_hidden_dim

        match self.method:
            case WorMIIntegrateMethod.CONCAT:
                hook_cls = ConcatImplantHook
                hook_options = {
                    "world_wise_positional_encoding": self.config.world_wise_positional_encoding,
                }
            case WorMIIntegrateMethod.ADD:
                hook_cls = AddImplantHook
                hook_options = {}
            case WorMIIntegrateMethod.CONCAT_WITH_ATTENTION:
                hook_cls = ConcatWithAttentionImplantHook
                hook_options = {
                    "latent_variable_length": self.config.latent_variable_length,
                    "world_wise_positional_encoding": self.config.world_wise_positional_encoding,
                    "device": self.device,
                }
            case WorMIIntegrateMethod.WORLD_WISE_ATTENTION:
                hook_cls = WorldWiseAttentionImplantHook
                hook_options = {}

        self.cross_attention_hooks.extend(
            [
                hook_cls(
                    base_hidden_dim=self.__base_model_hidden_dim,
                    world_model_hidden_dim=self.__world_model_hidden_dim,
                    num_heads=self.config.num_heads,
                    rms_norm_eps=self.rms_norm_eps,
                    self_attention=self.config.self_attention,
                    **hook_options,  # type: ignore
                )
                for _ in range(self.num_connections)
            ]
        )

        for i, conn in enumerate(self.__connections):
            layer = self.__base_layers[conn]
            layer.register_forward_hook(self.cross_attention_hooks[i])

        self.__has_model_been_built = True

    def freeze_main_model(self):
        freeze_model(self.__base_model)

    def maybe_build(self, world_model_hidden_dim: int):
        if self.__has_model_been_built:
            if world_model_hidden_dim != self.__world_model_hidden_dim:
                raise ValueError(
                    "Auxiliary model hidden dimension does not match the "
                    "previously set hidden dimension. Please call `build` "
                    "with the correct hidden dimension."
                )
        else:
            self.build(world_model_hidden_dim)

    def __get_new_world_model_key(self) -> int:
        key = self.__world_model_next_key
        new_key = key + 1
        while new_key in self.__world_models:
            new_key += 1
        self.__world_model_next_key = new_key
        return key

    def __remove_world_model_key(self, key: int) -> None:
        del self.__world_models[key]
        if key < self.__world_model_next_key:
            self.__world_model_next_key = key

    def implant(
        self,
        world_model: PreTrainedModel | str,
        connection_layer_indices: Iterable[int] | None = None,
    ):
        if isinstance(world_model, str):
            world_model = cast(
                PreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    world_model, torch_dtype=torch.bfloat16
                ),
            )

        hidden_dim = get_hidden_dim(world_model)
        self.maybe_build(world_model_hidden_dim=hidden_dim)

        world_model_key = self.__get_new_world_model_key()
        self.world_models[world_model_key] = world_model
        num_word_model_layers = len(
            self.world_models[world_model_key].model.layers
        )

        if connection_layer_indices is None:
            connection_layer_indices = get_connections(
                self.num_connections, num_word_model_layers
            )
        connection_layer_indices = list(connection_layer_indices)
        if len(connection_layer_indices) != self.num_connections:
            raise ValueError(
                "The number of connection_layer_indices must be equal to the "
                "number of connections."
            )

        for main_conn, world_conn in zip(
            self.__connections, connection_layer_indices
        ):
            hook = ExtractHiddenStateHook()

            self.extract_hidden_state_hooks[main_conn][world_model_key] = hook

            layer = self.world_models[world_model_key].model.layers[world_conn]
            layer.register_forward_hook(hook)

        freeze_model(self.world_models[world_model_key])
        self.world_models[world_model_key].to(self.device)  # type: ignore

    def remove(self, world_model_key: int):
        del self.world_models[world_model_key]
        for hooks in self.extract_hidden_state_hooks.values():
            hooks.pop(world_model_key)
        self.__remove_world_model_key(world_model_key)

    def implant_all(
        self,
        world_models: Iterable[PreTrainedModel | str],
        connection_layer_indices: (
            Iterable[Iterable[int] | None] | Iterable[int] | None
        ) = None,
    ):
        world_models = list(world_models)
        if connection_layer_indices is None:
            connection_layer_indices = [None] * len(world_models)
        else:
            if isinstance(next(iter(connection_layer_indices)), int):
                connection_layer_indices = cast(
                    list[Iterable[int]],
                    [connection_layer_indices] * len(world_models),
                )
            else:
                connection_layer_indices = cast(
                    Iterable[Iterable[int]],
                    connection_layer_indices,
                )
        for world_model, connections in zip(
            world_models, connection_layer_indices
        ):
            self.implant(world_model, connections)

    def remove_all(self):
        del self.__world_models
        self.__world_models = OrderedDict[int, PreTrainedModel]()
        self.__world_model_next_key = 0
        for hooks in self.extract_hidden_state_hooks.values():
            hooks.clear()

    def release_memory(self):
        for hook in self.cross_attention_hooks:
            hook.world_hidden_states = None
            hook.world_mask = None
            hook.attn_weights = None
        for hooks in self.extract_hidden_state_hooks.values():
            for hook in hooks.values():
                hook.hidden_state = None

    def _forward_world_mode(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
    ):
        outputs = list[CausalLMOutputWithPast]()
        with torch.no_grad():
            for model in self.world_models.values():
                model.eval()
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )
                outputs.append(output)

        for i, conn in enumerate(self.__connections):
            aux_hidden_states = [
                hidden_state
                for aux_key in self.world_models.keys()
                if (
                    hidden_state := (
                        self.extract_hidden_state_hooks[conn][aux_key]
                    ).hidden_state
                )
                is not None
            ]

            self.cross_attention_hooks[i].world_hidden_states = (
                aux_hidden_states
            )
            if self.method == WorMIIntegrateMethod.CONCAT_WITH_ATTENTION:
                attention_mask = None
            if attention_mask is not None:
                self.cross_attention_hooks[i].world_mask = attention_mask

        return compose_causal_lm_output(outputs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = True,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> CausalLMOutputWithPast:
        if isinstance(past_key_values, Cache):
            pkv = past_key_values.past_key_values
        else:
            pkv = past_key_values

        if pkv is not None:
            base_past_key_values = pkv[: self.__num_base_layers]
            world_past_key_values = pkv[self.__num_base_layers :]
        else:
            base_past_key_values = None
            world_past_key_values = None

        world_output = self._forward_world_mode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=world_past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        world_past_key_values = deepcopy(world_output.past_key_values)
        del world_output

        output = self.__base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=base_past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **model_kwargs,
        )

        if return_dict:
            if (
                output.past_key_values is not None
                and world_past_key_values is not None
            ):
                output.past_key_values = (
                    *output.past_key_values,
                    *world_past_key_values,
                )
        return output

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        is_main_process: bool = True,
        state_dict: dict[str, dict[str, dict[str, torch.Tensor]]] | None = None,
        save_function: Callable[..., None] = torch.save,
        push_to_hub: bool = False,
        max_shard_size: int | str = "10GB",
        safe_serialization: bool = True,
        variant: str | None = None,
        token: bool | str | None = None,
        save_peft_format: bool = False,
        **kwargs,
    ):
        super().save_pretrained(  # pytype: disable=attribute-error
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_mopdel_name_or_path: str,
        config: WorMIConfig | None = None,
        **kwargs,
    ):
        if config is None:
            wormi_config = WorMIConfig.from_pretrained(
                pretrained_mopdel_name_or_path
            )
        else:
            wormi_config = config
        model = super().from_pretrained(
            pretrained_mopdel_name_or_path, config=wormi_config, **kwargs
        )
        return cast(WorMI, model)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = (
                    cache_position[0]
                    if cache_position is not None
                    else past_key_values.get_seq_length()
                )
                max_cache_length = (
                    torch.tensor(
                        past_key_values.get_max_length(),
                        device=input_ids.device,
                    )
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = (
                    past_length
                    if max_cache_length is None
                    else torch.min(max_cache_length, past_length)
                )
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of
            # input_ids, then we are in a setting where some of the inputs are
            # exclusively passed as part of the cache (e.g. when passing
            # input_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[
                    :, -(attention_mask.shape[1] - past_length) :
                ]
            # 2 - If the past_length is smaller than input_ids.shape[1], then
            # input_ids holds all input tokens. We can discard input_ids based on
            # the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume
            # input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop
            # the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st
        # generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds.contiguous()}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = (
            position_ids.shape[-1]
            if position_ids is not None
            else input_ids.shape[-1]
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def to(  # type: ignore
        self,
        device: torch.device | str | int | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ):
        self.base_model.to(device)  # type: ignore
        for world_model in self.world_models.values():
            world_model.to(device, dtype, non_blocking)  # type: ignore
        super().to(device, dtype, non_blocking)  # type: ignore
        return self

    def named_parameters(self, *args, **kwargs):
        for name, param in super().named_parameters(*args, **kwargs):
            if name.startswith(f"_{self.__class__.__name__}"):
                continue
            yield name, param

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        removal_keys = set[str]()
        for key in state_dict.keys():
            if key.startswith(f"_{self.__class__.__name__}"):
                removal_keys.add(key)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in removal_keys
        }
        return state_dict
