# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Torch custom-op bridges for Kimi KDA and its short convolution."""

import functools

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.kimi_k3 import ragged_kda


def kimi_short_conv_scan(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Correctness-first Kimi short convolution using an XLA scan.

    The shared Pallas ragged-convolution kernel can halt the TPU when many
    donated recurrent buffers are embedded in Kimi's compiled hybrid graph.
    Keep this model-local fallback until that kernel interaction is fixed.
    """
    num_tokens = x.shape[0]
    kernel_size = conv_weight.shape[-1]
    weights = jnp.swapaxes(conv_weight[:, 0, :], 0, 1)
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    # Dummy AOT runs may describe more one-token requests than fit in the
    # current token bucket; normal runtime metadata instead repeats the true
    # token total in its padded request tail.
    total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
    sequence_ids = jnp.searchsorted(query_start_loc[1:],
                                    token_ids,
                                    side="right")
    sequence_ids = jnp.minimum(sequence_ids, state_indices.shape[0] - 1)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial_state = seq_lens > query_lens

    def step(state, inputs):
        token_id, token, sequence_id = inputs
        valid = token_id < total_tokens
        state_idx = state_indices[sequence_id]
        old_state = state[state_idx]
        is_first = token_id == query_start_loc[sequence_id]
        old_state = jnp.where(is_first & ~has_initial_state[sequence_id],
                              jnp.zeros_like(old_state), old_state)
        window = jnp.concatenate((old_state, token[None, :]), axis=0)
        output = jnp.sum(window.astype(jnp.float32) *
                         weights.astype(jnp.float32),
                         axis=0).astype(x.dtype)
        new_state = window[-(kernel_size - 1):].astype(conv_state.dtype)
        state = jax.lax.cond(
            valid,
            lambda s: s.at[state_idx].set(new_state),
            lambda s: s,
            state,
        )
        return state, jnp.where(valid, output, jnp.zeros_like(output))

    new_state, output = jax.lax.scan(step, conv_state,
                                     (token_ids, x, sequence_ids))
    return output, new_state


def build_kimi_sconv_op(
    prefix: str,
    *,
    kernel_size: int,
    state_dim_first: bool,
):
    """Build the layer-specific Pallas short-convolution custom op."""

    def sconv_core(
        mixed_qkv: jax.Array,
        conv_state: jax.Array,
        q_weight: jax.Array,
        k_weight: jax.Array,
        v_weight: jax.Array,
        query_start_loc: jax.Array,
        state_indices: jax.Array,
        seq_lens: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        num_tokens, mixed_dim = mixed_qkv.shape
        if mixed_dim % 3:
            raise ValueError(
                "Kimi short-convolution input must contain Q, K, V")
        projection_size = mixed_dim // 3
        expected_weight_shape = (projection_size, 1, kernel_size)
        for name, weight in (("q", q_weight), ("k", k_weight), ("v",
                                                                v_weight)):
            if weight.shape != expected_weight_shape:
                raise ValueError(
                    f"{name}_weight must have shape {expected_weight_shape}, "
                    f"got {weight.shape}")
        num_sequences = state_indices.shape[0]
        if query_start_loc.shape != (num_sequences + 1, ):
            raise ValueError("query_start_loc and state_indices disagree")
        if seq_lens.shape != (num_sequences, ):
            raise ValueError("seq_lens and state_indices disagree")
        expected_state_tail = ((mixed_dim,
                                kernel_size - 1) if state_dim_first else
                               (kernel_size - 1, mixed_dim))
        if conv_state.shape[1:] != expected_state_tail:
            raise ValueError(
                f"conv_state must end in {expected_state_tail}, got "
                f"{conv_state.shape}")
        if (query_start_loc.dtype != jnp.int32
                or state_indices.dtype != jnp.int32
                or seq_lens.dtype != jnp.int32):
            raise TypeError("Short-convolution metadata tensors must be int32")

        state = (jnp.swapaxes(conv_state, 1, 2)
                 if state_dim_first else conv_state)
        weight = jnp.concatenate((q_weight, k_weight, v_weight), axis=0)
        output, new_state = kimi_short_conv_scan(
            mixed_qkv,
            state,
            weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        if state_dim_first:
            new_state = jnp.swapaxes(new_state, 1, 2)
        if output.shape != (num_tokens, mixed_dim):
            raise ValueError("Short-convolution output shape changed")
        return new_state, jax.nn.silu(output)

    op_name = f"pallas::kimi_sconv_{prefix.replace('.', '_')}"
    sconv_op = pallas.jax_op(op_name, sconv_core, donate_argnums=(1, ))

    def _fake_sconv(mixed_qkv, conv_state, *args, **kwargs):
        return torch.empty_like(conv_state), torch.empty_like(mixed_qkv)

    sconv_op.register_fake(_fake_sconv)

    def sconv_impl(
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        new_state, output = sconv_op(
            mixed_qkv,
            conv_state,
            q_weight,
            k_weight,
            v_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        conv_state.copy_(new_state)
        return output

    return sconv_impl


def build_kimi_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
):
    """Build the layer-specific Pallas KDA recurrence custom op."""
    wrapped = functools.partial(ragged_kda, lower_bound=lower_bound, eps=eps)
    op_name = f"pallas::kimi_kda_{prefix.replace('.', '_')}"
    kda_op = pallas.jax_op(op_name, wrapped, donate_argnums=(4, ))

    def _fake_kda(mixed_qkv, _raw_gate, _beta, _output_gate, recurrent_state,
                  *args, **kwargs):
        num_tokens = mixed_qkv.size(0)
        num_heads, head_dim = recurrent_state.shape[1:3]
        output = torch.empty((num_tokens, num_heads, head_dim),
                             dtype=mixed_qkv.dtype,
                             device=mixed_qkv.device)
        return output, torch.empty_like(recurrent_state)

    kda_op.register_fake(_fake_kda)

    def kda_impl(
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        recurrent_state: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        output, new_state = kda_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            recurrent_state,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        recurrent_state.copy_(new_state)
        return output

    return kda_impl
