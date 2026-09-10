# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the local-routing fused MoE GMM wrapper."""

import types
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn, interleave_lane
from vllm_torchtpu.layers.common import fused_moe_gmm
from vllm_torchtpu.layers.common.fused_moe_gmm import (
    fused_moe_func, moe_gmm, prepare_routed_gmm_inputs)


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_select_ragged_gather_reduce(version):
    selected = fused_moe_gmm._select_ragged_gather_reduce(version)
    expected = getattr(fused_moe_gmm, f"ragged_gather_reduce_{version}")
    assert selected is expected


def test_ragged_gather_reduce_version_defaults_to_v2(monkeypatch):
    monkeypatch.delenv("RAGGED_GATHER_REDUCE_VERSION", raising=False)
    getter = envs.environment_variables["RAGGED_GATHER_REDUCE_VERSION"]
    assert getter() == "v2"


def test_ragged_gather_reduce_version_accepts_v3(monkeypatch):
    monkeypatch.setenv("RAGGED_GATHER_REDUCE_VERSION", "v3")
    getter = envs.environment_variables["RAGGED_GATHER_REDUCE_VERSION"]
    assert getter() == "v3"


def test_ragged_gather_reduce_uses_configured_version():
    expected = fused_moe_gmm._select_ragged_gather_reduce(
        envs.RAGGED_GATHER_REDUCE_VERSION)
    assert fused_moe_gmm.ragged_gather_reduce is expected


def test_owner_output_mode_defaults_to_off(monkeypatch):
    monkeypatch.delenv("TPU_MOE_OWNER_OUTPUT_MODE", raising=False)
    getter = envs.environment_variables["TPU_MOE_OWNER_OUTPUT_MODE"]
    assert getter() == "off"


@pytest.mark.parametrize("value", ["off", "on", "OFF", "ON"])
def test_owner_output_mode_accepts_off_and_on(monkeypatch, value):
    monkeypatch.setenv("TPU_MOE_OWNER_OUTPUT_MODE", value)
    getter = envs.environment_variables["TPU_MOE_OWNER_OUTPUT_MODE"]
    assert getter().lower() == value.lower()


def test_owner_output_mode_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("TPU_MOE_OWNER_OUTPUT_MODE", "auto")
    getter = envs.environment_variables["TPU_MOE_OWNER_OUTPUT_MODE"]
    with pytest.raises(ValueError, match="TPU_MOE_OWNER_OUTPUT_MODE"):
        getter()


def test_moe_gmm_uses_selected_ragged_gather_reduce(monkeypatch):
    gmm_outputs = iter([
        jnp.ones((2, 4), dtype=jnp.bfloat16),
        jnp.ones((2, 4), dtype=jnp.bfloat16),
    ])
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper",
                        lambda *args, **kwargs: next(gmm_outputs))
    monkeypatch.setattr(fused_moe_gmm, "get_packing_factor",
                        lambda *args, **kwargs: 1)

    expected = jnp.full((1, 4), 7, dtype=jnp.bfloat16)
    calls = []

    def fake_ragged_gather_reduce(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(fused_moe_gmm, "ragged_gather_reduce",
                        fake_ragged_gather_reduce)

    actual = fused_moe_gmm.moe_gmm(
        x=jnp.ones((2, 4), dtype=jnp.bfloat16),
        w1=jnp.ones((1, 4, 4), dtype=jnp.bfloat16),
        w1_scale=None,
        w1_bias=None,
        w2=jnp.ones((1, 4, 4), dtype=jnp.bfloat16),
        w2_scale=None,
        w2_bias=None,
        group_sizes=jnp.array([2], dtype=jnp.int32),
        argsort_revert_indices=jnp.array([0, 1], dtype=jnp.int32),
        topk_weights_flat=jnp.array([0.25, 0.75], dtype=jnp.float32),
        valid_mask_flat=jnp.array([True, True]),
        sorted_indices=jnp.array([0, 1], dtype=jnp.int32),
        activation="silu",
        num_tokens=1,
        topk=2,
        use_ep=True,
        use_sparse_core=True,
    )

    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == 1
    assert calls[0][1]["reduce_group_size"] == 2


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _reference_fused_moe(hidden_states, w1, w2, w1_bias, w2_bias, topk_weights,
                         topk_ids, activation):
    num_tokens, hidden_size = hidden_states.shape
    padded_hidden_size = w1.shape[1]

    hidden_states = jnp.pad(hidden_states,
                            ((0, 0), (0, padded_hidden_size - hidden_size)))
    out = jnp.zeros((num_tokens, padded_hidden_size), dtype=jnp.float32)

    for token_id in range(num_tokens):
        token_out = jnp.zeros((padded_hidden_size, ), dtype=jnp.float32)
        for expert_slot in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_id, expert_slot])
            if expert_id < 0:
                continue

            gate_up = jnp.matmul(hidden_states[token_id].astype(jnp.float32),
                                 w1[expert_id].astype(jnp.float32))
            if w1_bias is not None:
                gate_up = gate_up + w1_bias[expert_id, 0].astype(jnp.float32)

            gate, up = jnp.split(gate_up, 2, axis=-1)
            interleaved = interleave_lane(gate, up)
            activated = apply_act_fn(interleaved, activation)
            proj = jnp.matmul(activated.astype(jnp.float32),
                              w2[expert_id].astype(jnp.float32))
            if w2_bias is not None:
                proj = proj + w2_bias[expert_id, 0].astype(jnp.float32)

            token_out = token_out + topk_weights[token_id, expert_slot].astype(
                jnp.float32) * proj
        out = out.at[token_id].set(token_out)

    return out[:, :hidden_size].astype(hidden_states.dtype)


def test_prepare_routed_gmm_inputs_keeps_original_combine_metadata():
    hidden_states = jnp.arange(6, dtype=jnp.bfloat16).reshape(3, 2)
    topk_indices = jnp.array([[1, 0], [0, -1], [1, 0]], dtype=jnp.int32)
    topk_weights = jnp.array([[0.1, 0.2], [0.3, 0.0], [0.4, 0.5]],
                             dtype=jnp.bfloat16)
    flat_indices = np.asarray(topk_indices).reshape(-1)
    valid = flat_indices >= 0
    sorted_indices_expected = np.argsort(np.where(valid, flat_indices, 2),
                                         kind="stable")

    (_, group_sizes, argsort_revert_indices, topk_weights_flat, valid_mask,
     sorted_indices) = prepare_routed_gmm_inputs(
         hidden_states,
         topk_indices,
         topk_weights,
         local_num_experts=2,
         topk=2,
         use_ep=True,
         use_sparse_core=True,
         onehot_moe_permute_threshold=6,
     )

    np.testing.assert_array_equal(
        argsort_revert_indices,
        np.argsort(sorted_indices_expected, kind="stable"),
    )
    np.testing.assert_array_equal(topk_weights_flat,
                                  np.asarray(topk_weights).reshape(-1))
    np.testing.assert_array_equal(valid_mask, valid)
    np.testing.assert_array_equal(sorted_indices, sorted_indices_expected)
    np.testing.assert_array_equal(group_sizes, np.array([3, 2]))


@pytest.mark.parametrize(("mode", "owner_called"), [("on", True),
                                                    ("off", False)])
def test_output_onehot_owner_kernel_respects_mode(mode, owner_called):
    x = jnp.zeros((4, 2), dtype=jnp.bfloat16)
    w1 = jnp.zeros((2, 2, 4), dtype=jnp.bfloat16)
    w2 = jnp.zeros((2, 2, 2), dtype=jnp.bfloat16)
    group_sizes = jnp.array([2, 2], dtype=jnp.int32)
    argsort_revert_indices = jnp.array([2, 0, 1, 3], dtype=jnp.int32)
    topk_weights_flat = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.bfloat16)
    valid_mask = jnp.ones((4, ), dtype=jnp.bool_)
    sorted_indices = jnp.array([1, 2, 0, 3], dtype=jnp.int32)
    gmm1_res = jnp.zeros((4, 2), dtype=jnp.bfloat16)
    gmm2_res = jnp.zeros((4, 2), dtype=jnp.bfloat16)
    expected = jnp.ones((2, 2), dtype=jnp.bfloat16)

    with patch.object(
            envs,
            "TPU_MOE_OWNER_OUTPUT_MODE",
            mode,
    ), patch.object(
            fused_moe_gmm,
            "gmm_wrapper",
            side_effect=[gmm1_res, gmm2_res],
    ), patch.object(
            fused_moe_gmm,
            "can_use_blockwise_onehot_unpermute",
            return_value=True,
    ) as support_check, patch.object(
            fused_moe_gmm,
            "blockwise_onehot_unpermute",
            return_value=expected,
    ) as owner_kernel:
        actual = moe_gmm(
            x,
            w1,
            None,
            None,
            w2,
            None,
            None,
            group_sizes,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask,
            sorted_indices,
            activation="silu",
            num_tokens=2,
            topk=2,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=4,
        )

    if not owner_called:
        np.testing.assert_array_equal(actual, jnp.zeros_like(expected))
        support_check.assert_not_called()
        owner_kernel.assert_not_called()
        return

    np.testing.assert_array_equal(actual, expected)
    support_check.assert_called_once()
    owner_kernel.assert_called_once()
    call = owner_kernel.call_args
    np.testing.assert_array_equal(call.args[0], gmm2_res)
    np.testing.assert_array_equal(call.args[1], sorted_indices // 2)
    np.testing.assert_array_equal(call.args[2],
                                  topk_weights_flat[sorted_indices])
    np.testing.assert_array_equal(call.args[3], np.array([4], dtype=np.int32))
    assert call.kwargs == {"num_tokens": 2}


def test_fused_moe_local_routing_matches_reference():
    _require_tpu()

    # Match the local EP shard contract used by Qwen3-Coder-30B-A3B:
    # hidden_size=2048, moe_intermediate_size=768, hidden_act=silu,
    # num_experts_per_tok=8. This wrapper consumes already-local expert ids, so
    # use the per-shard expert count (128 global experts / 8 EP shards = 16).
    num_tokens = 4
    hidden_size = 2048
    padded_hidden_size = 2048
    intermediate_size = 768
    num_experts = 16
    topk = 8
    key = jax.random.key(0)
    hidden_key, w1_key, w2_key, w1_bias_key, w2_bias_key = jax.random.split(
        key, 5)

    hidden_states = (jax.random.normal(hidden_key, (num_tokens, hidden_size),
                                       dtype=jnp.float32) / 10).astype(
                                           jnp.bfloat16)
    w1 = (jax.random.normal(
        w1_key, (num_experts, padded_hidden_size, intermediate_size * 2),
        dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(w2_key,
                          (num_experts, intermediate_size, padded_hidden_size),
                          dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w1_bias = (jax.random.normal(w1_bias_key,
                                 (num_experts, 1, intermediate_size * 2),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2_bias = (jax.random.normal(w2_bias_key,
                                 (num_experts, 1, padded_hidden_size),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)

    topk_weights = jnp.array(
        [
            [0.30, 0.18, 0.14, 0.12, 0.10, 0.07, 0.05, 0.04],
            [0.22, 0.18, 0.16, 0.14, 0.11, 0.08, 0.06, 0.05],
            [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
            [0.28, 0.19, 0.15, 0.11, 0.09, 0.07, 0.06, 0.05],
        ],
        dtype=jnp.float32,
    )
    topk_ids = jnp.array(
        [
            [0, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 15],
            [15, 13, 11, 9, 7, 5, 3, 1],
            [14, 12, 10, 8, 6, 4, 2, 0],
        ],
        dtype=jnp.int32,
    )

    expected = _reference_fused_moe(hidden_states, w1, w2, w1_bias, w2_bias,
                                    topk_weights, topk_ids, "silu")
    actual = fused_moe_func(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        w1_scale=None,
        w2_scale=None,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=topk,
        activation="silu",
    )

    assert actual.dtype == hidden_states.dtype
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=1e-1,
                               rtol=1e-1)


def _make_ep_inputs(num_local_experts, seed=0):
    """Synthesize one local shard's weights + per-token local topk_ids.

    Matches the megablox-friendly shapes used by
    test_fused_moe_local_routing_matches_reference so the kernel does not
    return NaN on tile-misaligned inputs.
    """
    num_tokens = 4
    hidden_size = 2048
    intermediate_size = 768
    topk = 8
    key = jax.random.key(seed)
    hk, w1k, w2k, b1k, b2k = jax.random.split(key, 5)
    hidden_states = (
        jax.random.normal(hk, (num_tokens, hidden_size), dtype=jnp.float32) /
        10).astype(jnp.bfloat16)
    w1 = (jax.random.normal(
        w1k, (num_local_experts, hidden_size, intermediate_size * 2),
        dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2 = (
        jax.random.normal(w2k,
                          (num_local_experts, intermediate_size, hidden_size),
                          dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w1_bias = (jax.random.normal(b1k,
                                 (num_local_experts, 1, intermediate_size * 2),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    w2_bias = (jax.random.normal(b2k, (num_local_experts, 1, hidden_size),
                                 dtype=jnp.float32) / 10).astype(jnp.bfloat16)
    topk_weights = jnp.array(
        [
            [0.30, 0.18, 0.14, 0.12, 0.10, 0.07, 0.05, 0.04],
            [0.22, 0.18, 0.16, 0.14, 0.11, 0.08, 0.06, 0.05],
            [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
            [0.28, 0.19, 0.15, 0.11, 0.09, 0.07, 0.06, 0.05],
        ],
        dtype=jnp.float32,
    )
    # Local ids stay in [0, num_local_experts); distinct per slot to exercise
    # the full local shard.
    base = jnp.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=jnp.int32)
    local_ids = jnp.stack([
        base,
        base + 1,
        (base + 1)[::-1],
        base[::-1],
    ]) % num_local_experts
    return (hidden_states, w1, w2, w1_bias, w2_bias, topk_weights, local_ids,
            topk)


@pytest.mark.parametrize(
    "num_local_experts,experts_start",
    [
        (16, 0),  # rank 0: experts_start == 0 must behave like the legacy path
        (16, 32),  # rank 2 of 8-way EP (Qwen3-30B shape: 128 / 8 = 16 local)
        (20,
         60),  # rank 3 of 8-way EP (Qwen3-480B-FP8 shape: 160 / 8 = 20 local)
    ])
def test_fused_moe_global_id_remap_matches_local(num_local_experts,
                                                 experts_start):
    """Global ids + experts_start must produce the same output as local ids."""
    _require_tpu()
    (hidden_states, w1, w2, w1_bias, w2_bias, topk_weights, local_ids,
     topk) = _make_ep_inputs(num_local_experts)
    global_ids = local_ids + experts_start
    expected = fused_moe_func(hidden_states=hidden_states,
                              w1=w1,
                              w2=w2,
                              w1_scale=None,
                              w2_scale=None,
                              w1_bias=w1_bias,
                              w2_bias=w2_bias,
                              topk_weights=topk_weights,
                              topk_ids=local_ids,
                              topk=topk,
                              activation="silu",
                              use_ep=True)
    actual = fused_moe_func(hidden_states=hidden_states,
                            w1=w1,
                            w2=w2,
                            w1_scale=None,
                            w2_scale=None,
                            w1_bias=w1_bias,
                            w2_bias=w2_bias,
                            topk_weights=topk_weights,
                            topk_ids=global_ids,
                            experts_start=experts_start,
                            topk=topk,
                            activation="silu",
                            use_ep=True)
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=1e-1,
                               rtol=1e-1)


def test_fused_moe_global_id_remap_masks_out_of_range():
    """Non-local global ids must contribute zero (range check + jnp.where)."""
    _require_tpu()
    num_local_experts, experts_start = 16, 32
    (hidden_states, w1, w2, w1_bias, w2_bias, topk_weights, local_ids,
     topk) = _make_ep_inputs(num_local_experts)
    global_ids = local_ids + experts_start
    # Token 0 slot 0: below shard (-> masked). Token 1 slot 1: above shard.
    global_ids = global_ids.at[0, 0].set(experts_start - 1)
    global_ids = global_ids.at[1, 1].set(experts_start + num_local_experts)
    # Reference: same call but with the offending weights pre-zeroed.
    masked_weights = topk_weights.at[0, 0].set(0.0).at[1, 1].set(0.0)
    expected = fused_moe_func(hidden_states=hidden_states,
                              w1=w1,
                              w2=w2,
                              w1_scale=None,
                              w2_scale=None,
                              w1_bias=w1_bias,
                              w2_bias=w2_bias,
                              topk_weights=masked_weights,
                              topk_ids=local_ids,
                              topk=topk,
                              activation="silu",
                              use_ep=True)
    actual = fused_moe_func(hidden_states=hidden_states,
                            w1=w1,
                            w2=w2,
                            w1_scale=None,
                            w2_scale=None,
                            w1_bias=w1_bias,
                            w2_bias=w2_bias,
                            topk_weights=topk_weights,
                            topk_ids=global_ids,
                            experts_start=experts_start,
                            topk=topk,
                            activation="silu",
                            use_ep=True)
    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=1e-1,
                               rtol=1e-1)


def test_fused_moe_func_int4_packed_matches_reference():
    _require_tpu()

    num_tokens = 4
    hidden_size = 512
    logical_k = 512
    storage_k = logical_k // 8
    intermediate_size = 256
    intermediate_size_storage = intermediate_size // 8
    num_experts = 4
    topk = 2
    key = jax.random.key(0)

    hidden_states = jax.random.uniform(key, (num_tokens, hidden_size),
                                       dtype=jnp.bfloat16,
                                       minval=-1,
                                       maxval=1)
    topk_weights = jax.random.uniform(key, (num_tokens, topk),
                                      dtype=jnp.float32,
                                      minval=0.1,
                                      maxval=0.9)
    topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_ids = jax.random.randint(key, (num_tokens, topk),
                                  minval=0,
                                  maxval=num_experts)

    w1_key, w2_key = jax.random.split(key)
    w1_raw = jax.random.uniform(
        w1_key,
        (num_experts, logical_k, intermediate_size * 2),
        dtype=jnp.bfloat16,
        minval=-1,
        maxval=1,
    )
    w2_raw = jax.random.uniform(
        w2_key,
        (num_experts, intermediate_size, hidden_size),
        dtype=jnp.bfloat16,
        minval=-1,
        maxval=1,
    )

    def quantize_to_int4(x, axis, block_size):
        max_val = 7
        min_val = -8
        orig_shape = x.shape
        blocked_shape = (orig_shape[:axis] + (-1, block_size) +
                         orig_shape[axis + 1:])
        x_blocked = x.reshape(blocked_shape)
        x_blocked_abs_max = jnp.max(jnp.abs(x_blocked),
                                    axis=axis + 1,
                                    keepdims=True)
        scale = x_blocked_abs_max / max_val
        x_blocked_q = jnp.clip(x_blocked / scale, min_val,
                               max_val).astype(jnp.int32)
        x_q = x_blocked_q.reshape(orig_shape)
        scale = scale.squeeze(axis=axis + 1).astype(jnp.bfloat16)
        return x_q, scale

    w1_int4, w1_scale = quantize_to_int4(w1_raw, axis=1, block_size=64)
    w2_int4, w2_scale = quantize_to_int4(w2_raw, axis=1, block_size=64)

    w1_scale = jnp.expand_dims(w1_scale, axis=2)
    w2_scale = jnp.expand_dims(w2_scale, axis=2)

    w1_scale_tiled = jnp.repeat(w1_scale, 64, axis=1).squeeze(axis=2)
    w1_dequantized = w1_int4.astype(jnp.bfloat16) * w1_scale_tiled

    w2_scale_tiled = jnp.repeat(w2_scale, 64, axis=1).squeeze(axis=2)
    w2_dequantized = w2_int4.astype(jnp.bfloat16) * w2_scale_tiled

    expected = _reference_fused_moe(
        hidden_states,
        w1_dequantized,
        w2_dequantized,
        w1_bias=None,
        w2_bias=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
    )

    def pack_int4(x_int4, num_exp, storage_dim, out_dim):
        x_uint4 = (x_int4 + 8) & 0x0F
        x_reshaped = x_uint4.reshape(num_exp, storage_dim, 8, out_dim)
        shifts = jnp.arange(8, dtype=jnp.int32) * 4
        shifted = x_reshaped << shifts[None, None, :, None]
        x_packed = jnp.sum(shifted, axis=2).astype(jnp.int32)
        INT4_SIGN_XOR = -2004318072
        return x_packed ^ INT4_SIGN_XOR

    w1_packed = pack_int4(w1_int4, num_experts, storage_k,
                          intermediate_size * 2)
    w2_packed = pack_int4(w2_int4, num_experts, intermediate_size_storage,
                          hidden_size)

    actual = fused_moe_func(
        hidden_states=hidden_states,
        w1=w1_packed,
        w2=w2_packed,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=None,
        w2_bias=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        topk=topk,
        activation="silu",
        rhs_quant_dtype=jnp.int4,
    )

    np.testing.assert_allclose(np.asarray(actual),
                               np.asarray(expected),
                               atol=8.0,
                               rtol=3e-1)


def _fake_tpu_info(sparse_core):
    return types.SimpleNamespace(sparse_core=sparse_core)


def _fake_sc_info(num_lanes, num_cores, num_subcores):
    return types.SimpleNamespace(num_lanes=num_lanes,
                                 num_cores=num_cores,
                                 num_subcores=num_subcores)


def test_onehot_threshold_env_parsing(monkeypatch):
    getter = envs.environment_variables["ONEHOT_MOE_PERMUTE_THRESHOLD"]
    monkeypatch.delenv("ONEHOT_MOE_PERMUTE_THRESHOLD", raising=False)
    assert getter() is None
    monkeypatch.setenv("ONEHOT_MOE_PERMUTE_THRESHOLD", "")
    assert getter() is None
    monkeypatch.setenv("ONEHOT_MOE_PERMUTE_THRESHOLD", "-1")
    assert getter() == -1


def test_onehot_threshold_explicit_env_skips_tpu_info(monkeypatch):

    def _fail():
        raise AssertionError("get_tpu_info must not be called")

    monkeypatch.setattr(fused_moe_gmm.pltpu, "get_tpu_info", _fail)
    monkeypatch.setattr(envs, "ONEHOT_MOE_PERMUTE_THRESHOLD", 0, raising=False)
    assert fused_moe_gmm.resolve_onehot_permute_threshold() == 0


@pytest.mark.parametrize("explicit", [None, -1])
def test_onehot_threshold_auto_is_one_below_real_blocks(monkeypatch, explicit):
    monkeypatch.setattr(envs,
                        "ONEHOT_MOE_PERMUTE_THRESHOLD",
                        explicit,
                        raising=False)
    # 16 lanes x 2 cores x 16 subcores = 512-row block (v7x geometry) and a
    # 256-row block (v6e) both sit at or below the cap, so block - 1 wins:
    # a full block stays on the SparseCore path. None and -1 both select
    # auto.
    for sc_info, expected in ((_fake_sc_info(16, 2, 16), 511),
                              (_fake_sc_info(8, 2, 16), 255)):
        monkeypatch.setattr(fused_moe_gmm.pltpu,
                            "get_tpu_info",
                            lambda info=sc_info: _fake_tpu_info(info))
        assert fused_moe_gmm.resolve_onehot_permute_threshold() == expected


def test_onehot_threshold_auto_caps_oversized_block(monkeypatch):
    monkeypatch.setattr(envs,
                        "ONEHOT_MOE_PERMUTE_THRESHOLD",
                        None,
                        raising=False)
    # A hypothetical 1024-row block exceeds the cap, so the cap wins.
    monkeypatch.setattr(fused_moe_gmm.pltpu, "get_tpu_info",
                        lambda: _fake_tpu_info(_fake_sc_info(32, 2, 16)))
    assert fused_moe_gmm.resolve_onehot_permute_threshold() == 512


def test_onehot_threshold_auto_zero_without_sparse_core(monkeypatch):
    monkeypatch.setattr(envs,
                        "ONEHOT_MOE_PERMUTE_THRESHOLD",
                        None,
                        raising=False)
    monkeypatch.setattr(fused_moe_gmm.pltpu, "get_tpu_info",
                        lambda: _fake_tpu_info(None))
    assert fused_moe_gmm.resolve_onehot_permute_threshold() == 0


def test_onehot_threshold_auto_zero_on_non_tpu_host(monkeypatch):
    monkeypatch.setattr(envs,
                        "ONEHOT_MOE_PERMUTE_THRESHOLD",
                        None,
                        raising=False)

    def _raise():
        raise ValueError("Unsupported TPU device kind: cpu")

    monkeypatch.setattr(fused_moe_gmm.pltpu, "get_tpu_info", _raise)
    assert fused_moe_gmm.resolve_onehot_permute_threshold() == 0


def test_prepare_routed_onehot_permute_matches_plain_gather():
    num_tokens, topk, hidden = 4, 2, 8
    hidden_states = (jnp.arange(num_tokens * hidden, dtype=jnp.float32) /
                     7).reshape(num_tokens, hidden).astype(jnp.bfloat16)
    # -1 marks non-local experts (EP mask path).
    topk_indices = jnp.array([[0, 1], [1, -1], [2, 0], [-1, 3]],
                             dtype=jnp.int32)
    topk_weights = jnp.full((num_tokens, topk), 0.5, dtype=jnp.float32)

    common = dict(local_num_experts=4, topk=topk, use_ep=True)
    onehot_out = fused_moe_gmm.prepare_routed_gmm_inputs(
        hidden_states,
        topk_indices,
        topk_weights,
        use_sparse_core=True,
        onehot_moe_permute_threshold=num_tokens * topk,
        **common)
    plain_out = fused_moe_gmm.prepare_routed_gmm_inputs(
        hidden_states,
        topk_indices,
        topk_weights,
        use_sparse_core=False,
        onehot_moe_permute_threshold=0,
        **common)

    # Each one-hot row selects exactly one token, so the permuted activations
    # match the plain gather bit-for-bit and the routing metadata is shared.
    for got, want in zip(onehot_out, plain_out):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_moe_gmm_onehot_combine_matches_plain_reduce(monkeypatch):
    num_tokens, topk, hidden = 4, 2, 8
    rows = num_tokens * topk
    k1, k2 = jax.random.split(jax.random.key(1))
    gmm1 = (jax.random.normal(k1, (rows, hidden), dtype=jnp.float32) /
            10).astype(jnp.bfloat16)
    gmm2 = (jax.random.normal(k2, (rows, hidden), dtype=jnp.float32) /
            10).astype(jnp.bfloat16)
    monkeypatch.setattr(fused_moe_gmm, "get_packing_factor",
                        lambda *args, **kwargs: 1)

    def run(**overrides):
        outputs = iter([gmm1, gmm2])
        monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper",
                            lambda *args, **kwargs: next(outputs))
        kwargs = dict(
            x=jnp.ones((rows, hidden), dtype=jnp.bfloat16),
            w1=jnp.ones((1, hidden, hidden), dtype=jnp.bfloat16),
            w1_scale=None,
            w1_bias=None,
            w2=jnp.ones((1, hidden, hidden), dtype=jnp.bfloat16),
            w2_scale=None,
            w2_bias=None,
            group_sizes=jnp.array([rows], dtype=jnp.int32),
            argsort_revert_indices=jnp.array([3, 6, 1, 7, 0, 5, 2, 4],
                                             dtype=jnp.int32),
            sorted_indices=jnp.array([4, 2, 6, 0, 7, 5, 1, 3],
                                     dtype=jnp.int32),
            topk_weights_flat=jnp.linspace(0.1, 0.8, rows, dtype=jnp.float32),
            valid_mask_flat=jnp.array(
                [True, True, False, True, True, False, True, True]),
            activation="silu",
            num_tokens=num_tokens,
            topk=topk,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=rows,
        )
        kwargs.update(overrides)
        return fused_moe_gmm.moe_gmm(**kwargs)

    onehot_res = run()
    plain_res = run(use_ep=False)

    assert onehot_res.dtype == jnp.bfloat16
    # Path equivalence only: with these inputs both paths compute the same
    # weighted sums up to reduction order, so the bf16 results agree to
    # final-rounding tolerance. The dot's output dtype is asserted
    # separately in test_moe_gmm_onehot_combine_keeps_operand_dtype.
    np.testing.assert_allclose(np.asarray(onehot_res, dtype=np.float32),
                               np.asarray(plain_res, dtype=np.float32),
                               rtol=2e-2,
                               atol=1e-3)


def _small_moe_gmm_kwargs(num_tokens=4, topk=2, hidden=8):
    """Kwargs for a moe_gmm call whose gmm stages are monkeypatched away."""
    rows = num_tokens * topk
    return dict(
        x=jnp.ones((rows, hidden), dtype=jnp.bfloat16),
        w1=jnp.ones((1, hidden, hidden), dtype=jnp.bfloat16),
        w1_scale=None,
        w1_bias=None,
        w2=jnp.ones((1, hidden, hidden), dtype=jnp.bfloat16),
        w2_scale=None,
        w2_bias=None,
        group_sizes=jnp.array([rows], dtype=jnp.int32),
        argsort_revert_indices=jnp.array([3, 6, 1, 7, 0, 5, 2, 4],
                                         dtype=jnp.int32),
        sorted_indices=jnp.array([4, 2, 6, 0, 7, 5, 1, 3], dtype=jnp.int32),
        topk_weights_flat=jnp.full((rows, ), 0.5, dtype=jnp.bfloat16),
        valid_mask_flat=jnp.ones((rows, ), dtype=bool),
        activation="silu",
        num_tokens=num_tokens,
        topk=topk,
        use_ep=True,
        use_sparse_core=True,
    )


def test_moe_gmm_onehot_combine_keeps_operand_dtype(monkeypatch):
    # With production dtypes (bf16 weights, bf16 gmm output) the combine
    # matmul must NOT request f32 output: the MXU accumulates bf16 dots in
    # f32 either way, and a widened f32 output costs bandwidth at large row
    # counts (#485). Assert at the jaxpr level because backends that
    # accumulate bf16 matmuls in f32 make the widening invisible to output
    # comparison.
    kwargs = _small_moe_gmm_kwargs()
    rows = kwargs["argsort_revert_indices"].size
    monkeypatch.setattr(fused_moe_gmm, "get_packing_factor",
                        lambda *args, **kwargs: 1)
    # Identity gmm keeps the combine matmul a function of the traced input,
    # so it cannot constant-fold out of the jaxpr.
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper",
                        lambda lhs, *args, **kwargs: lhs)

    def run(x):
        return fused_moe_gmm.moe_gmm(**{
            **kwargs, "x": x,
            "onehot_moe_permute_threshold": rows
        })

    jaxpr = jax.make_jaxpr(run)(kwargs["x"])
    dots = [
        eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == "dot_general"
    ]
    assert dots, "one-hot combine must lower to a dot_general"
    assert all(
        eqn.params.get("preferred_element_type") != jnp.float32
        for eqn in dots)


def test_moe_gmm_onehot_combine_skip_padded_tokens_zeroes_nan(monkeypatch):
    # With skip_padded_tokens, gmm output rows past the valid prefix are
    # uninitialized (NaN here); the one-hot combine must zero them before the
    # matmul or they would spread to every token in the batch.
    kwargs = _small_moe_gmm_kwargs()
    rows = kwargs["argsort_revert_indices"].size
    valid_rows = rows - 2
    monkeypatch.setattr(fused_moe_gmm, "get_packing_factor",
                        lambda *args, **kwargs: 1)
    gmm2 = jnp.ones((rows, 8), dtype=jnp.bfloat16)
    gmm2 = gmm2.at[valid_rows:].set(jnp.nan)
    outputs = iter([jnp.ones((rows, 8), dtype=jnp.bfloat16), gmm2])
    monkeypatch.setattr(fused_moe_gmm, "gmm_wrapper",
                        lambda *args, **kwargs: next(outputs))
    # Mark the slots whose routed rows fall in the NaN tail as invalid, as
    # prepare_routed_gmm_inputs does for padded tokens.
    revert = kwargs["argsort_revert_indices"]
    kwargs["valid_mask_flat"] = revert < valid_rows
    kwargs["group_sizes"] = jnp.array([valid_rows], dtype=jnp.int32)

    out = fused_moe_gmm.moe_gmm(
        **{
            **kwargs,
            "onehot_moe_permute_threshold": rows,
            "skip_padded_tokens": True,
        })

    assert bool(jnp.all(jnp.isfinite(out.astype(jnp.float32))))
