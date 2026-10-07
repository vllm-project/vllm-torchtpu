# SPDX-License-Identifier: Apache-2.0
"""100% Full-module line and branch coverage test suite for .

Covers all lines and branches in:
src/vllm_torchtpu/layers/core/fused_moe_gmm.py (204 stmts, 48 branches, 100%)
under CPU (JAX_PLATFORMS=cpu, TPU_SKIP_MDS_QUERY=true).
"""

from __future__ import annotations

import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

# Ensure repository root is in sys.path so 'tests' and package modules can be imported
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force CPU platform for JAX
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import vllm_torchtpu.envs as envs  # noqa: E402
from vllm_torchtpu.layers.core import fused_moe_gmm  # noqa: E402


def test_select_ragged_gather_reduce():
    """Verify SparseCore ragged_gather_reduce kernel version dispatch."""
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import (
        ragged_gather_reduce as rgr_v1,
    )
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import (
        ragged_gather_reduce_v2 as rgr_v2,
    )
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
        ragged_gather_reduce as rgr_v3,
    )

    assert fused_moe_gmm._select_ragged_gather_reduce("v1") is rgr_v1
    assert fused_moe_gmm._select_ragged_gather_reduce("v2") is rgr_v2
    assert fused_moe_gmm._select_ragged_gather_reduce("v3") is rgr_v3
    with pytest.raises(KeyError):
        fused_moe_gmm._select_ragged_gather_reduce("invalid")


def test_resolve_onehot_permute_threshold(monkeypatch):
    """Verify resolve_onehot_permute_threshold across env and TPU info branches."""
    # 1. Explicit positive threshold
    monkeypatch.setattr(envs, "ONEHOT_MOE_PERMUTE_THRESHOLD", 64)
    assert fused_moe_gmm.resolve_onehot_permute_threshold() == 64

    # 2. Explicit negative threshold falls through
    monkeypatch.setattr(envs, "ONEHOT_MOE_PERMUTE_THRESHOLD", -1)
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        side_effect=ValueError("Unsupported"),
    ):
        assert fused_moe_gmm.resolve_onehot_permute_threshold() == 0

    # 3. Env unset, unsupported device kind (ValueError on CPU)
    monkeypatch.setattr(envs, "ONEHOT_MOE_PERMUTE_THRESHOLD", None)
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        side_effect=ValueError("Unsupported"),
    ):
        assert fused_moe_gmm.resolve_onehot_permute_threshold() == 0

    # 4. Env unset, device info has no sparse_core
    mock_info_no_sc = MagicMock()
    mock_info_no_sc.sparse_core = None
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=mock_info_no_sc,
    ):
        assert fused_moe_gmm.resolve_onehot_permute_threshold() == 0

    # 5. Env unset, device info has sparse_core
    mock_info_sc = MagicMock()
    mock_sc = MagicMock()
    mock_sc.num_lanes = 16
    mock_sc.num_cores = 4
    mock_sc.num_subcores = 2
    mock_info_sc.sparse_core = mock_sc
    with patch(
        "jax.experimental.pallas.tpu.get_tpu_info",
        return_value=mock_info_sc,
    ):
        # min(2 * (16 * 4 * 2) - 1, 768) = min(255, 768) = 255
        assert fused_moe_gmm.resolve_onehot_permute_threshold() == 255


def test_unpack_fp4_to_e2m1():
    """Verify unpacking packed uint8 weights to native float4_e2m1fn in K-major."""
    w_packed = jnp.zeros((2, 4, 8), dtype=jnp.uint8)
    out = fused_moe_gmm.unpack_fp4_to_e2m1(w_packed)
    assert out.shape == (2, 16, 4)
    assert out.dtype == jnp.float4_e2m1fn


def test_requant_unpack_kmajor():
    """Verify W4A8 requantization from block-16 fp4 checkpoint to kernel layout."""
    w_packed = jnp.zeros((4, 8), dtype=jnp.uint8)
    scale_f = jnp.ones((4, 1), dtype=jnp.float32)
    w_out, scale_out = fused_moe_gmm.requant_unpack_kmajor(w_packed, scale_f, block=16)
    assert w_out.shape == (16, 4)
    assert scale_out.shape == (1, 1, 4)


def test_quantize_to_native_fp4_kmajor():
    """Verify quantize_to_native_fp4_kmajor with pack=True and pack=False."""
    w = jnp.ones((16, 4), dtype=jnp.float32)
    p1, s1 = fused_moe_gmm.quantize_to_native_fp4_kmajor(w, block=16, pack=True)
    assert p1.shape == (8, 4)
    assert p1.dtype == jnp.uint8

    p2, s2 = fused_moe_gmm.quantize_to_native_fp4_kmajor(w, block=16, pack=False)
    assert p2.shape == (16, 4)
    assert p2.dtype == jnp.float4_e2m1fn


def test_gmm_wrapper():
    """Verify gmm_wrapper with fp4 weight and non-fp4 weight."""
    lhs = jnp.ones((4, 8), dtype=jnp.bfloat16)

    # 1. is_fp4_weight = True (float4_e2m1fn, maybe_quantize_lhs=False)
    rhs_fp4 = jnp.zeros((1, 8, 16), dtype=jnp.float4_e2m1fn)

    def mock_gmm_v2_call(lhs, rhs, maybe_quantize_lhs, **kwargs):
        assert maybe_quantize_lhs is False
        return jnp.zeros((lhs.shape[0], rhs.shape[-1]), dtype=lhs.dtype)

    with patch.object(fused_moe_gmm, "gmm_v2", side_effect=mock_gmm_v2_call):
        res = fused_moe_gmm.gmm_wrapper(
            lhs, rhs_fp4, None, None, jnp.array([4]), jnp.array([0])
        )
        assert res.shape == (4, 16)

    # 2. is_fp4_weight = False (bfloat16, maybe_quantize_lhs=True)
    rhs_bf16 = jnp.zeros((1, 8, 16), dtype=jnp.bfloat16)

    def mock_gmm_v2_call2(lhs, rhs, maybe_quantize_lhs, **kwargs):
        assert maybe_quantize_lhs is True
        return jnp.zeros((lhs.shape[0], rhs.shape[-1]), dtype=lhs.dtype)

    with patch.object(fused_moe_gmm, "gmm_v2", side_effect=mock_gmm_v2_call2):
        res2 = fused_moe_gmm.gmm_wrapper(
            lhs, rhs_bf16, None, None, jnp.array([4]), jnp.array([0])
        )
        assert res2.shape == (4, 16)


def test_counting_sort_positions(monkeypatch):
    """Verify _counting_sort_positions with num_buckets > block and padding branches."""
    # 1. num_buckets > block: returns None, None
    sort_keys = jnp.zeros(4, dtype=jnp.int32)
    pos, counts = fused_moe_gmm._counting_sort_positions(sort_keys, num_buckets=8)
    assert pos is None and counts is None

    # 2. pad > 0
    monkeypatch.setattr(fused_moe_gmm, "_COUNTING_SORT_BLOCK", 4)
    sort_keys_10 = jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    pos_10, counts_10 = fused_moe_gmm._counting_sort_positions(
        sort_keys_10, num_buckets=3
    )
    assert pos_10 is not None and counts_10 is not None
    assert pos_10.shape == (10,)

    # 3. pad == 0 (m is exact multiple of block)
    sort_keys_8 = jnp.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    pos_8, counts_8 = fused_moe_gmm._counting_sort_positions(sort_keys_8, num_buckets=3)
    assert pos_8 is not None and counts_8 is not None
    assert pos_8.shape == (8,)


def test_pairwise_sort_positions():
    """Verify _pairwise_sort_positions implementation (Cut a)."""
    sort_keys = jnp.array([2, 0, 1, 0], dtype=jnp.int32)
    tokens = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
    pos, sorted_tokens = fused_moe_gmm._pairwise_sort_positions(sort_keys, tokens)
    assert pos.shape == (4,)
    assert sorted_tokens.shape == (4,)
    np.testing.assert_array_equal(np.asarray(pos), np.array([3, 0, 2, 1]))
    np.testing.assert_array_equal(np.asarray(sorted_tokens), np.array([20, 40, 30, 10]))


def test_prepare_routed_gmm_inputs():
    """Verify prepare_routed_gmm_inputs across all dispatch paths."""
    hidden = jnp.ones((4, 16), dtype=jnp.bfloat16)
    topk_ids = jnp.array([[0, 1], [1, 2], [-1, 0], [1, -1]], dtype=jnp.int32)
    topk_weights = jnp.array(
        [[0.5, 0.5], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]], dtype=jnp.float32
    )

    # 1. skip_padded_tokens = True
    out1 = fused_moe_gmm.prepare_routed_gmm_inputs(
        hidden,
        topk_ids,
        topk_weights,
        local_num_experts=4,
        topk=2,
        use_ep=False,
        use_sparse_core=False,
        skip_padded_tokens=True,
    )
    assert out1[0].shape == (8, 16)

    # 2. non-EP, non-SC, packable
    out2a = fused_moe_gmm.prepare_routed_gmm_inputs(
        hidden,
        topk_ids,
        topk_weights,
        local_num_experts=4,
        topk=2,
        use_ep=False,
        use_sparse_core=False,
        skip_padded_tokens=False,
    )
    assert out2a[0].shape == (8, 16)

    # 3. non-EP, non-SC, not packable (large local_num_experts)
    big_experts = 1 << 30
    out2b = fused_moe_gmm.prepare_routed_gmm_inputs(
        hidden,
        topk_ids,
        topk_weights,
        local_num_experts=big_experts,
        topk=2,
        use_ep=False,
        use_sparse_core=False,
    )
    assert out2b[0].shape == (8, 16)

    # 4. use_ep=True, use_sparse_core=True, tokens <= threshold (pairwise + onehot)
    out3 = fused_moe_gmm.prepare_routed_gmm_inputs(
        hidden,
        topk_ids,
        topk_weights,
        local_num_experts=4,
        topk=2,
        use_ep=True,
        use_sparse_core=True,
        onehot_moe_permute_threshold=128,
    )
    assert out3[0].shape == (8, 16)

    # 5. use_ep=True, use_sparse_core=True, tokens > threshold (ragged permute)
    def dummy_gather(x, indices, starts, ends):
        return x[indices]

    with patch.object(fused_moe_gmm, "ragged_gather_v2", side_effect=dummy_gather):
        out4a = fused_moe_gmm.prepare_routed_gmm_inputs(
            hidden,
            topk_ids,
            topk_weights,
            local_num_experts=4,
            topk=2,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=0,
        )
        assert out4a[0].shape == (8, 16)

        # 6. use_ep=True, use_sparse_core=True, tokens > threshold, not packable
        out4b = fused_moe_gmm.prepare_routed_gmm_inputs(
            hidden,
            topk_ids,
            topk_weights,
            local_num_experts=big_experts,
            topk=2,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=0,
        )
        assert out4b[0].shape == (8, 16)


def test_moe_gmm(monkeypatch):
    """Verify moe_gmm across all routing, permutation, and combine branches."""
    num_tokens = 4
    topk = 2
    hidden_size = 16
    x = jnp.ones((num_tokens * topk, hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.ones((4, hidden_size, 32), dtype=jnp.bfloat16)
    w2 = jnp.ones((4, 32, hidden_size), dtype=jnp.bfloat16)
    group_sizes = jnp.array([2, 2, 2, 2], dtype=jnp.int32)
    argsort_revert = jnp.arange(num_tokens * topk, dtype=jnp.int32)
    topk_weights_flat = jnp.ones(num_tokens * topk, dtype=jnp.float32) * 0.5
    valid_mask = jnp.ones(num_tokens * topk, dtype=bool)
    token_indices_sorted = jnp.arange(num_tokens * topk, dtype=jnp.int32)

    def mock_gmm(lhs, rhs, *args, **kwargs):
        return jnp.zeros((lhs.shape[0], rhs.shape[-1]), dtype=lhs.dtype)

    with patch.object(fused_moe_gmm, "gmm_v2", side_effect=mock_gmm):
        # Branch 1: non-EP non-SC fallback
        out1 = fused_moe_gmm.moe_gmm(
            x,
            w1,
            None,
            None,
            w2,
            None,
            None,
            group_sizes,
            argsort_revert,
            topk_weights_flat,
            valid_mask,
            token_indices_sorted,
            activation="silu",
            num_tokens=num_tokens,
            topk=topk,
            use_ep=False,
            use_sparse_core=False,
        )
        assert out1.shape == (num_tokens, hidden_size)

        # Branch 2: EP + SC, size <= threshold, TPU_MOE_OWNER_OUTPUT_MODE == "off"
        monkeypatch.setattr(envs, "TPU_MOE_OWNER_OUTPUT_MODE", "off")
        out2 = fused_moe_gmm.moe_gmm(
            x,
            w1,
            None,
            None,
            w2,
            None,
            None,
            group_sizes,
            argsort_revert,
            topk_weights_flat,
            valid_mask,
            token_indices_sorted,
            activation="silu",
            num_tokens=num_tokens,
            topk=topk,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=128,
        )
        assert out2.shape == (num_tokens, hidden_size)

        # Branch 3: EP + SC, size <= threshold, owner_output == "on", blockwise
        monkeypatch.setattr(envs, "TPU_MOE_OWNER_OUTPUT_MODE", "on")
        with (
            patch.object(
                fused_moe_gmm,
                "can_use_blockwise_onehot_unpermute",
                return_value=True,
            ),
            patch.object(
                fused_moe_gmm,
                "blockwise_onehot_unpermute",
                return_value=jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16),
            ),
        ):
            out3 = fused_moe_gmm.moe_gmm(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                argsort_revert,
                topk_weights_flat,
                valid_mask,
                token_indices_sorted,
                activation="silu",
                num_tokens=num_tokens,
                topk=topk,
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=128,
            )
            assert out3.shape == (num_tokens, hidden_size)

        # Branch 4: EP + SC, size > threshold -> ragged_gather_reduce
        with patch.object(
            fused_moe_gmm,
            "ragged_gather_reduce",
            return_value=jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16),
        ):
            out4 = fused_moe_gmm.moe_gmm(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                argsort_revert,
                topk_weights_flat,
                valid_mask,
                token_indices_sorted,
                activation="silu",
                num_tokens=num_tokens,
                topk=topk,
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=0,
            )
            assert out4.shape == (num_tokens, hidden_size)

        # Branch 5: use_local_ragged_combine (non-EP, SC, topk=3)
        mock_tpu_info = MagicMock()
        mock_tpu_info.num_lanes = 16
        mock_tpu_info.vmem_capacity_bytes = 1000
        mock_sc_info = MagicMock()
        mock_sc_info.num_lanes = 16
        mock_tpu_info.sparse_core = mock_sc_info
        topk5 = 3
        x5 = jnp.ones((num_tokens * topk5, hidden_size), dtype=jnp.bfloat16)
        argsort_revert5 = jnp.arange(num_tokens * topk5, dtype=jnp.int32)
        topk_weights_flat5 = jnp.ones(num_tokens * topk5, dtype=jnp.float32) / topk5
        valid_mask5 = jnp.ones(num_tokens * topk5, dtype=bool)
        token_indices_sorted5 = jnp.arange(num_tokens * topk5, dtype=jnp.int32)

        with (
            patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=mock_tpu_info,
            ),
            patch.object(fused_moe_gmm, "token_block_alignment", return_value=4),
            patch.object(
                fused_moe_gmm,
                "ragged_gather_reduce_v3",
                return_value=jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16),
            ),
        ):
            out5 = fused_moe_gmm.moe_gmm(
                x5,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                argsort_revert5,
                topk_weights_flat5,
                valid_mask5,
                token_indices_sorted5,
                activation="silu",
                num_tokens=num_tokens,
                topk=topk5,
                use_ep=False,
                use_sparse_core=True,
                onehot_moe_permute_threshold=0,
            )
            assert out5.shape == (num_tokens, hidden_size)

        # Branch 5b: use_local_ragged_combine raises ValueError from get_tpu_info
        with patch(
            "jax.experimental.pallas.tpu.get_tpu_info",
            side_effect=ValueError("no TPU"),
        ):
            out5b = fused_moe_gmm.moe_gmm(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                argsort_revert,
                topk_weights_flat,
                valid_mask,
                token_indices_sorted,
                activation="silu",
                num_tokens=num_tokens,
                topk=topk,
                use_ep=False,
                use_sparse_core=True,
                onehot_moe_permute_threshold=0,
            )
            assert out5b.shape == (num_tokens, hidden_size)

        # Branch 5c: use_local_ragged_combine num_tokens < alignment -> falls through
        with (
            patch(
                "jax.experimental.pallas.tpu.get_tpu_info",
                return_value=mock_tpu_info,
            ),
            patch.object(fused_moe_gmm, "token_block_alignment", return_value=8),
        ):
            out5c = fused_moe_gmm.moe_gmm(
                x5,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                argsort_revert5,
                topk_weights_flat5,
                valid_mask5,
                token_indices_sorted5,
                activation="silu",
                num_tokens=num_tokens,
                topk=topk5,
                use_ep=False,
                use_sparse_core=True,
                onehot_moe_permute_threshold=0,
            )
            assert out5c.shape == (num_tokens, hidden_size)


def test_fused_moe_func(monkeypatch):
    """Verify fused_moe_func across scale expansion, experts_start, and dense expert."""
    num_tokens = 4
    hidden_size = 16
    intermediate_size = 32
    num_experts = 4
    topk = 2

    hidden_states = jnp.ones((num_tokens, hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.ones((num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16)
    w2 = jnp.ones((num_experts, intermediate_size, hidden_size), dtype=jnp.bfloat16)
    w1_scale = jnp.ones((num_experts, 1, intermediate_size), dtype=jnp.float32)
    w2_scale = jnp.ones((num_experts, 1, hidden_size), dtype=jnp.float32)
    topk_weights = jnp.ones((num_tokens, topk), dtype=jnp.float32) / topk
    topk_ids = jnp.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=jnp.int32)
    experts_start = jnp.array(1, dtype=jnp.int32)

    def mock_gmm(lhs, rhs, *args, **kwargs):
        return jnp.zeros((lhs.shape[0], rhs.shape[-1]), dtype=lhs.dtype)

    # 1. routed GMM with 3D scale expansion + experts_start remapping
    monkeypatch.setattr(fused_moe_gmm, "_DENSE_EXPERT_THRESHOLD", 0)
    with patch.object(fused_moe_gmm, "gmm_v2", side_effect=mock_gmm):
        out1 = fused_moe_gmm.fused_moe_func(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,
            None,
            topk_weights,
            topk_ids,
            experts_start=experts_start,
            topk=topk,
            use_ep=True,
            use_sparse_core=True,
            onehot_moe_permute_threshold=128,
        )
        assert out1.shape == (num_tokens, hidden_size)

    # 2. dense_expert_moe branch
    monkeypatch.setattr(fused_moe_gmm, "_DENSE_EXPERT_THRESHOLD", 64)
    with (
        patch.object(fused_moe_gmm, "can_use_dense_expert", return_value=True),
        patch.object(
            fused_moe_gmm,
            "dense_expert_moe",
            return_value=jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16),
        ),
    ):
        out2 = fused_moe_gmm.fused_moe_func(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,
            None,
            topk_weights,
            topk_ids,
            topk=topk,
        )
        assert out2.shape == (num_tokens, hidden_size)

    # 3. routed GMM with w1_scale=None and w2_scale=None
    monkeypatch.setattr(fused_moe_gmm, "_DENSE_EXPERT_THRESHOLD", 0)
    with patch.object(fused_moe_gmm, "gmm_v2", side_effect=mock_gmm):
        out3 = fused_moe_gmm.fused_moe_func(
            hidden_states,
            w1,
            w2,
            None,
            None,
            None,
            None,
            topk_weights,
            topk_ids,
            experts_start=None,
            topk=topk,
            use_ep=False,
            use_sparse_core=False,
            onehot_moe_permute_threshold=0,
        )
        assert out3.shape == (num_tokens, hidden_size)
