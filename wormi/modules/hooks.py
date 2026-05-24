"""Hooks for the plug-and-play models."""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama import modeling_llama

from wormi.modules.layers import (
    MLP,
    MultiheadAttention,
    world_wise_positional_encoding,
)


def process_hook_args(
    model: torch.nn.Module,  # pylint: disable=unused-argument
    inp: Union[
        torch.Tensor, tuple[torch.Tensor, ...]
    ],  # pylint: disable=unused-argument
    out: Union[torch.Tensor, tuple[torch.Tensor, ...]],
):
    """Extracts the main output tensor from a PyTorch hook output.

    Args:
        model: The nn.Module object to which the hook is attached.
        inp: Input tensor to the layer (ignored).
        out: Output from the layer. This can be a tensor or a tuple containing the
          tensor.
    Reference:
      register_forward_hook in
      https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    Returns:
        The main output tensor from the hooked block.
    """
    base_hidden_state = out[0] if isinstance(out, tuple) else out
    query = base_hidden_state
    return query, out


class BaseImplantHook(ABC, torch.nn.Module):
    """cross attention hook for CALM."""

    def __init__(
        self,
        base_hidden_dim: int,
        world_model_hidden_dim: int,
        num_heads: int,
        rms_norm_eps: float = 1e-6,
        self_attention: bool = False,
    ):
        """Initializes the cross attention hook.

        Args:
          base_hidden_dim: The hidden dimension of the base model.
          world_model_hidden_dim: The hidden dimension of the world model.
          num_heads: The number of attention heads in the hook
          rms_norm_eps: The epsilon value for the post-attention RMS norm layer
          self_attention: Whether to use self attention in

        Attributes:
          proj: The projection layer to project the augmented hidden state to the
            anchor hidden dimension.
          embed_dim: The hidden dimension of the anchor model.
          num_heads: The number of attention heads in the hook.
          cross_attention: The cross attention layer.
          aux_hidden_state: The auxiliary hidden state tensor. This is set by
            forward_aug in CALM.
          aug_mask: The augmented mask tensor. This is set by forward_aug in CALM.
          attn_weights: The attention weights tensor. This is set by the forward
            pass of the cross attention hook.
        Example:
          hook = CrossAttentionHook(anchor_hidden_dim, aug_hidden_dim, num_heads)
          model.register_forward_hook(hook)
          model(input)
          print(hook.attn_weights)
        """
        super().__init__()
        self.proj = MLP(world_model_hidden_dim, base_hidden_dim)
        self.out_proj = MLP(base_hidden_dim, base_hidden_dim)
        self.embed_dim = base_hidden_dim
        self.num_heads = num_heads
        self.do_self_attention = self_attention
        self.cross_attention_layernorm = modeling_llama.LlamaRMSNorm(
            self.embed_dim, eps=rms_norm_eps
        )
        self.cross_attention = MultiheadAttention(
            self.embed_dim,
            num_heads,
            kdim=self.embed_dim,
            vdim=self.embed_dim,
            batch_first=True,
        )
        if self.do_self_attention:
            self.self_attention_layernorm = modeling_llama.LlamaRMSNorm(
                self.embed_dim, eps=rms_norm_eps
            )
            self.self_attention = MultiheadAttention(
                self.embed_dim,
                num_heads,
                kdim=self.embed_dim,
                vdim=self.embed_dim,
                use_bias=False,
                batch_first=True,
            )
        self.world_hidden_states: list[torch.Tensor] | None = None
        self.world_mask: torch.Tensor | None = None
        self.attn_weights: torch.Tensor | None = None

    @abstractmethod
    def preprocess_params(
        self,
        query: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ):
        """Preprocesses the input tensors for the cross attention.

        Args:
          query: The query tensor.
          key: The key tensor.
          value: The value tensor.

        Returns:
          The preprocessed query, key, and value tensors.
        """
        raise NotImplementedError

    def postprocess_params(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Postprocesses the output tensors from the cross attention.

        Args:
          query: The query tensor.
          key: The key tensor.
          value: The value tensor.

        Returns:
          The postprocessed query, key, and value tensors.
        """
        return query, key, value

    def forward(self, *hook_args):
        """Forward pass of the cross attention hook.

        Args:
          *hook_args: The arguments passed to the hook.

        Raises:
          ValueError: If aug_hidden_state or aug_mask is None.

        The cross attention hook is registered to the anchor model. The hook
        extracts the hidden state from the anchor model and uses it as the query
        for the cross attention. The key and value for the cross attention are
        computed by projecting the hidden state from the augmented model. The
        augmented hidden state and mask are set by forward_aug in CALM.

        Returns:
          The modified output of the cross attention hook.
        """
        assert (
            self.world_hidden_states is not None and self.world_mask is not None
        )
        query, output = process_hook_args(*hook_args)
        keys = values = self.world_hidden_states
        query, key, value = self.preprocess_params(query, keys, values)

        key = self.proj(key)
        value = self.proj(value)

        query, key, value = self.postprocess_params(query, key, value)

        self.world_mask = self.world_mask.float()

        attn_output, attn_weights = self.cross_attention(
            query, key, value, need_weights=True
        )
        attn_output = self.cross_attention_layernorm(attn_output)
        attn_output = self.out_proj(attn_output)
        output_fin = attn_output + query

        if self.do_self_attention:
            attn_output, attn_weights = self.self_attention(
                output_fin, output_fin, output_fin, need_weights=True
            )
            attn_output = self.self_attention_layernorm(attn_output)
            output_fin = output_fin + attn_output

        self.attn_weights = attn_weights
        new_output = (output_fin,) + output[1:]  # type: ignore
        return new_output


class AddImplantHook(BaseImplantHook):
    @override
    def preprocess_params(
        self,
        query: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ):
        """Preprocesses the input tensors for the cross attention.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.

        Returns:
            The preprocessed query, key, and value tensors.
        """
        key = reduce(torch.add, keys)
        value = reduce(torch.add, values)
        return query, key, value


class ConcatImplantHook(BaseImplantHook):
    def __init__(
        self,
        base_hidden_dim: int,
        world_model_hidden_dim: int,
        num_heads: int,
        rms_norm_eps: float = 1e-6,
        world_wise_positional_encoding: bool = True,
        self_attention: bool = False,
    ):
        super().__init__(
            base_hidden_dim,
            world_model_hidden_dim,
            num_heads,
            rms_norm_eps,
            self_attention,
        )
        self.world_hidden_dim = world_model_hidden_dim
        self.world_wise_positional_encoding = world_wise_positional_encoding

    def __preprocess(self, param):
        param = torch.cat(param, dim=-1)
        if self.world_wise_positional_encoding:
            param = world_wise_positional_encoding(
                param,
                num_aux_models=len(param),
                aux_hidden_dim=self.world_hidden_dim,
            )
        return param

    @override
    def preprocess_params(
        self,
        query: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ):
        """Preprocesses the input tensors for the cross attention.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.

        Returns:
            The preprocessed query, key, and value tensors.
        """
        key = self.__preprocess(keys)
        value = self.__preprocess(values)
        return query, key, value


class ConcatWithAttentionImplantHook(BaseImplantHook):
    def __init__(
        self,
        base_hidden_dim: int,
        world_model_hidden_dim: int,
        num_heads: int,
        latent_variable_length: int = 512,
        rms_norm_eps: float = 1e-6,
        self_attention: bool = False,
        world_wise_positional_encoding: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__(
            base_hidden_dim,
            world_model_hidden_dim,
            num_heads,
            rms_norm_eps,
            self_attention,
        )
        self.concat_query = torch.nn.Parameter(
            torch.randn(
                1, latent_variable_length, world_model_hidden_dim, device=device
            )
        )
        self.concat_attn = torch.nn.MultiheadAttention(
            world_model_hidden_dim,
            num_heads,
            kdim=world_model_hidden_dim,
            vdim=world_model_hidden_dim,
            batch_first=True,
        )
        self.concat_attn_layernorm = modeling_llama.LlamaRMSNorm(
            world_model_hidden_dim, eps=rms_norm_eps
        )
        self.world_hidden_dim = world_model_hidden_dim
        self.world_wise_positional_encoding = world_wise_positional_encoding

    def __attn(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        out, _ = self.concat_attn(
            self.concat_query,
            x,
            x,
            key_padding_mask=attn_mask,
            need_weights=True,
        )
        out = self.concat_attn_layernorm(out)
        out = out + self.concat_query
        return out

    def __preprocess(self, param):
        param = [self.__attn(a) for a in param]
        param = torch.cat(param, dim=-2)
        if self.world_wise_positional_encoding:
            param = world_wise_positional_encoding(
                param,
                num_aux_models=len(param),
                aux_hidden_dim=self.world_hidden_dim,
            )
        return param

    @override
    def preprocess_params(
        self,
        query: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ):
        """Preprocesses the input tensors for the cross attention.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.

        Returns:
            The preprocessed query, key, and value tensors.
        """
        key = self.__preprocess(keys)
        value = self.__preprocess(values)
        return query, key, value


class WorldWiseAttentionImplantHook(BaseImplantHook):
    def __init__(
        self,
        base_hidden_dim: int,
        world_model_hidden_dim: int,
        num_heads: int,
        rms_norm_eps: float = 0.000001,
        self_attention: bool = False,
    ):
        super().__init__(
            base_hidden_dim,
            world_model_hidden_dim,
            num_heads,
            rms_norm_eps,
            self_attention,
        )
        self.world_wise_attn = MultiheadAttention(
            base_hidden_dim,
            num_heads,
            kdim=base_hidden_dim,
            vdim=base_hidden_dim,
            batch_first=True,
        )

    @override
    def preprocess_params(
        self: BaseImplantHook,
        query: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ):
        """Preprocesses the input tensors for the cross attention.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.

        Returns:
            The preprocessed query, key, and value tensors.
        """
        key = torch.stack(keys, dim=-3)
        value = torch.stack(values, dim=-3)
        return query, key, value

    @override
    def postprocess_params(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Postprocesses the output tensors from the cross attention.

        Args:
            query: The query tensor.
            key: The key tensor.
            value: The value tensor.

        Returns:
            The postprocessed query, key, and value tensors.
        """

        BLD = query.shape  # (batch, seq_len, hidden_dim)

        # Add a dimension for the num_aux_models
        query = query.unsqueeze(-2)

        # Transpose to (batch, seq_len, num_aux_models, hidden_dim)
        key = key.transpose(-3, -2)
        value = value.transpose(-3, -2)

        BLND = key.shape  # (batch, seq_len, num_aux_models, hidden_dim)

        # Reshape to (batch * seq_len, num_aux_models, hidden_dim)
        query = query.reshape(-1, *query.shape[-2:])
        key = key.reshape(-1, *key.shape[-2:])
        value = value.reshape(-1, *value.shape[-2:])

        # Perform world-wise attention
        _, attn_weight = self.world_wise_attn(
            query, key, value, need_weights=True
        )

        # Reshape back to (batch, seq_len, hidden_dim)
        query = query.reshape(*BLD)

        # Reshape back to (batch, seq_len, num_aux_models, hidden_dim)
        key = key.reshape(*BLND)

        # Sum over the num_aux_models dimension
        key = key.sum(dim=-2)

        # Multiply the attention weights with the value
        value = attn_weight @ value

        # Reshape back to (batch, seq_len, hidden_dim)
        value = value.reshape(*BLD)

        return query, key, value


class ExtractHiddenStateHook(torch.nn.Module):
    """Extract hidden state hook for CALM."""

    def __init__(self):
        """Initializes the extract hidden state hook.

        Attributes:
          hidden_state: The hidden state tensor. This is set by the forward pass of
            the extract hidden state hook.
        Example:
        ```python
          hook = ExtractHiddenStateHook()
          model.register_forward_hook(hook)
          model(input)
          print(hook.hidden_state)
        ```
        """
        super().__init__()
        self.hidden_state = None

    def forward(self, *hook_args):
        hidden_state, out = process_hook_args(*hook_args)
        self.hidden_state = hidden_state
        return out
