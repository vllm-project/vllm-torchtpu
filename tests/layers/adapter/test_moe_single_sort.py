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
"""The single-sort MoE permutation matches the three-sort reference."""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core import fused_moe_gmm

LOCAL_EXPERTS, TOTAL_EXPERTS, TOPK = 64, 512, 10


def _expert_parallel_ids(num_tokens, seed):
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.choice(TOTAL_EXPERTS, TOPK, replace=False) for _ in range(num_tokens)]
    )


def _three_sort_reference(sort_keys):
    token_bits = max(1, (sort_keys.shape[0] // TOPK - 1).bit_length())
    token_ids = jnp.arange(sort_keys.shape[0], dtype=jnp.int32) // TOPK
    sorted_indices = jnp.argsort(sort_keys)
    packed = jax.lax.sort(
        jnp.left_shift(sort_keys, token_bits) | token_ids, is_stable=False
    )
    token_indices = jnp.bitwise_and(packed, (1 << token_bits) - 1)
    return sorted_indices, jnp.argsort(sorted_indices), token_indices


def _assert_single_sort_matches(num_tokens):
    ids = _expert_parallel_ids(num_tokens, seed=num_tokens)
    keys = jnp.asarray(
        np.where(ids < LOCAL_EXPERTS, ids, LOCAL_EXPERTS).astype(np.int32).reshape(-1)
    )
    flat_bits = max(1, (keys.shape[0] - 1).bit_length())
    actual = fused_moe_gmm._single_sort_permutation(keys, TOPK, flat_bits)
    for got, want in zip(actual, _three_sort_reference(keys)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_single_sort_permutation_matches_three_sorts_at_one_token():
    _assert_single_sort_matches(1)


def test_single_sort_permutation_matches_three_sorts_at_decode_batch():
    _assert_single_sort_matches(4)


def test_single_sort_permutation_matches_three_sorts_below_ragged_threshold():
    # 204 tokens * top-10 = 2,040 routed rows, just under the 2,048 threshold.
    _assert_single_sort_matches(204)


def test_prepare_routed_gmm_inputs_single_sort_matches_three_sorts_with_padded_tokens(
    monkeypatch,
):
    num_tokens, hidden = 8, 16
    rng = np.random.default_rng(0)
    ids = _expert_parallel_ids(num_tokens, seed=1)
    # EP marks experts owned by other ranks with a negative id.
    local_ids = jnp.asarray(np.where(ids < LOCAL_EXPERTS, ids, -1).astype(np.int32))
    weights = rng.random((num_tokens, TOPK)).astype(np.float32)
    weights[-2:] = 0.0  # Two padded tokens carry zero routing weight.
    hidden_states = jnp.asarray(
        rng.standard_normal((num_tokens, hidden)).astype(np.float32)
    )
    outputs = {}
    for single_sort in (False, True):
        monkeypatch.setattr(fused_moe_gmm, "_SINGLE_SORT", single_sort)
        outputs[single_sort] = fused_moe_gmm.prepare_routed_gmm_inputs(
            hidden_states,
            local_ids,
            jnp.asarray(weights),
            local_num_experts=LOCAL_EXPERTS,
            topk=TOPK,
            use_ep=True,
            use_sparse_core=False,
            skip_padded_tokens=True,
        )
    for three_sorts, one_sort in zip(outputs[False], outputs[True]):
        np.testing.assert_array_equal(np.asarray(three_sorts), np.asarray(one_sort))


def test_single_sort_permutation_topk_one():
    num_tokens = 8
    keys = jnp.arange(num_tokens, dtype=jnp.int32) % LOCAL_EXPERTS
    flat_bits = max(1, (num_tokens - 1).bit_length())
    actual = fused_moe_gmm._single_sort_permutation(keys, 1, flat_bits)
    # Reference for topk=1
    sorted_indices = jnp.argsort(keys)
    expected = (sorted_indices, jnp.argsort(sorted_indices), sorted_indices)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_single_sort_permutation_all_equal_keys():
    # All tokens routed to expert 0; verify stable tie-ordering
    n = 16
    keys = jnp.zeros(n, dtype=jnp.int32)
    flat_bits = max(1, (n - 1).bit_length())
    actual = fused_moe_gmm._single_sort_permutation(keys, TOPK, flat_bits)
    for got, want in zip(actual, _three_sort_reference(keys)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_single_sort_fallback_when_bits_exceed_int32(monkeypatch):
    # When flat_bits + expert_bits >= 31, single_sort falls back to 3-sort path
    num_tokens = 4
    hidden = 16
    rng = np.random.default_rng(2)
    local_ids = jnp.zeros((num_tokens, TOPK), dtype=jnp.int32)
    weights = jnp.ones((num_tokens, TOPK), dtype=jnp.float32)
    hidden_states = jnp.asarray(
        rng.standard_normal((num_tokens, hidden)).astype(np.float32)
    )
    monkeypatch.setattr(fused_moe_gmm, "_SINGLE_SORT", True)
    # Using local_num_experts with 30 bits forces bits >= 31
    large_num_experts = 1 << 29
    # Traced, not run: the three-sort path counts rows per expert with a
    # (rows, local_num_experts) int32 one-hot, which at 2**29 experts is 86 GB.
    res = jax.eval_shape(
        functools.partial(
            fused_moe_gmm.prepare_routed_gmm_inputs,
            local_num_experts=large_num_experts,
            topk=TOPK,
            use_ep=False,
            use_sparse_core=False,
        ),
        hidden_states,
        local_ids,
        weights,
    )
    assert res[0].shape[0] == num_tokens * TOPK


def test_single_sort_fallback_when_rows_exceed_max_cap(monkeypatch):
    # When routed rows > _SINGLE_SORT_MAX_ROWS (2,048), single_sort falls back
    # to the 3-sort path even if _SINGLE_SORT is True and ragged_permute is False.
    num_tokens = 205  # 205 * 10 = 2,050 routed rows > 2,048
    hidden = 16
    rng = np.random.default_rng(3)
    local_ids = jnp.zeros((num_tokens, TOPK), dtype=jnp.int32)
    weights = jnp.ones((num_tokens, TOPK), dtype=jnp.float32)
    hidden_states = jnp.asarray(
        rng.standard_normal((num_tokens, hidden)).astype(np.float32)
    )
    monkeypatch.setattr(fused_moe_gmm, "_SINGLE_SORT", True)
    res = fused_moe_gmm.prepare_routed_gmm_inputs(
        hidden_states,
        local_ids,
        weights,
        local_num_experts=LOCAL_EXPERTS,
        topk=TOPK,
        use_ep=False,
        use_sparse_core=False,
    )
    assert res[0].shape[0] == num_tokens * TOPK


def test_single_sort_env_default_enabled(monkeypatch):
    getter = envs.environment_variables["TPU_MOE_SINGLE_SORT"]
    monkeypatch.delenv("TPU_MOE_SINGLE_SORT", raising=False)
    assert getter() is True
    monkeypatch.setenv("TPU_MOE_SINGLE_SORT", "0")
    assert getter() is False
