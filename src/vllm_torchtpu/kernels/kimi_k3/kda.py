# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Ragged Kimi Delta Attention recurrence."""

import functools

import jax
import jax.numpy as jnp


def kda_step(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    raw_gate: jax.Array,
    beta: jax.Array,
    output_gate: jax.Array,
    state: jax.Array,
    decay: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array,
    *,
    lower_bound: float | None,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """One fp32 KDA recurrence step, shared by the kernel and tests."""
    activation_dtype = query.dtype
    query = query.astype(jnp.float32)
    query *= jax.lax.rsqrt(
        jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    query *= query.shape[-1]**-0.5
    key = key.astype(jnp.float32)
    key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    value = value.astype(jnp.float32)

    gate_input = raw_gate.astype(jnp.float32) + dt_bias
    if lower_bound is None:
        gate = -decay * jax.nn.softplus(gate_input)
    else:
        gate = lower_bound * jax.nn.sigmoid(decay * gate_input)

    state = state.astype(jnp.float32) * jnp.exp(gate)[..., None]
    prediction = jnp.einsum("hk,hkv->hv",
                            key,
                            state,
                            preferred_element_type=jnp.float32)
    delta = jax.nn.sigmoid(beta.astype(
        jnp.float32))[:, None] * (value - prediction)
    state += key[..., None] * delta[:, None, :]

    output = jnp.einsum("hk,hkv->hv",
                        query,
                        state,
                        preferred_element_type=jnp.float32)
    output = output.astype(activation_dtype).astype(jnp.float32)
    output *= jax.lax.rsqrt(
        jnp.mean(output * output, axis=-1, keepdims=True) + eps)
    output = (output *
              norm_weight.astype(jnp.float32)).astype(activation_dtype)
    output *= jax.nn.sigmoid(output_gate)
    return output, state


@functools.partial(
    jax.jit,
    static_argnames=("lower_bound", "eps"),
    donate_argnames=("recurrent_state", ),
)
def ragged_kda(
    mixed_qkv: jax.Array,
    raw_gate: jax.Array,
    beta: jax.Array,
    output_gate: jax.Array,
    recurrent_state: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    *,
    lower_bound: float | None,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Run KDA over flattened ragged requests and update their recurrent state."""
    num_tokens, mixed_dim = mixed_qkv.shape
    num_heads, head_dim, state_v_dim = recurrent_state.shape[1:]
    if state_v_dim != head_dim or mixed_dim != 3 * num_heads * head_dim:
        raise ValueError("Incompatible KDA activation and state shapes")
    if raw_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA gate shape")
    if beta.shape != (num_tokens, num_heads):
        raise ValueError("Incompatible KDA beta shape")
    if output_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA output-gate shape")
    num_sequences = state_indices.shape[0]
    if a_log.shape != (num_heads, ):
        raise ValueError("KDA A_log shape does not match the head count")
    if dt_bias.shape != (num_heads * head_dim, ):
        raise ValueError("KDA dt_bias shape does not match the projection")
    if norm_weight.shape != (head_dim, ):
        raise ValueError("KDA norm weight shape does not match the head size")
    if query_start_loc.shape != (num_sequences + 1, ):
        raise ValueError("KDA query starts and state indices disagree")
    if seq_lens.shape != (num_sequences, ):
        raise ValueError("KDA sequence lengths and state indices disagree")
    if (raw_gate.dtype != mixed_qkv.dtype or beta.dtype != mixed_qkv.dtype
            or output_gate.dtype != mixed_qkv.dtype):
        raise TypeError("KDA activations must have one common dtype")
    if recurrent_state.dtype != jnp.float32:
        raise TypeError("KDA recurrent state must be float32")
    if a_log.dtype != jnp.float32 or dt_bias.dtype != jnp.float32:
        raise TypeError("KDA A_log and dt_bias must be float32")
    if (query_start_loc.dtype != jnp.int32 or state_indices.dtype != jnp.int32
            or seq_lens.dtype != jnp.int32):
        raise TypeError("KDA metadata tensors must be int32")

    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
    sequence_ids = jnp.searchsorted(query_start_loc[1:],
                                    token_ids,
                                    side="right")
    sequence_ids = jnp.minimum(sequence_ids, num_sequences - 1)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial_state = seq_lens > query_lens
    decay = jnp.exp(a_log)[:, None]
    dt_bias_heads = dt_bias.reshape(num_heads, head_dim)

    def step(state, inputs):
        (token_id, qkv_token, gate_token, beta_token, output_gate_token,
         sequence_id) = inputs
        valid = token_id < total_tokens
        state_idx = state_indices[sequence_id]
        safe_state_idx = jnp.maximum(state_idx, 0)
        token_state = state[safe_state_idx]
        is_first = token_id == query_start_loc[sequence_id]
        token_state = jnp.where(is_first & ~has_initial_state[sequence_id],
                                jnp.zeros_like(token_state), token_state)
        qkv = qkv_token.reshape(3, num_heads, head_dim)
        output, next_token_state = kda_step(
            qkv[0],
            qkv[1],
            qkv[2],
            gate_token.reshape(num_heads, head_dim),
            beta_token,
            output_gate_token.reshape(num_heads, head_dim),
            token_state,
            decay,
            dt_bias_heads,
            norm_weight,
            lower_bound=lower_bound,
            eps=eps,
        )
        should_store = valid & (state_idx >= 0)
        state = jax.lax.cond(
            should_store,
            lambda s: s.at[safe_state_idx].set(next_token_state),
            lambda s: s,
            state,
        )
        output = jnp.where(valid, output, jnp.zeros_like(output))
        return state, output

    new_state, output = jax.lax.scan(
        step,
        recurrent_state,
        (token_ids, mixed_qkv, raw_gate, beta, output_gate, sequence_ids),
    )
    return output, new_state
