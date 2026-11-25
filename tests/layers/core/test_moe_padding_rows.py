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
"""Row-count padding filter in prepare_routed_gmm_inputs (TP, TP+EP, DP+EP).

When every real row carries non-zero routing weights, passing ``row_info``
builds the same routing layout as zeroing padded weights while leaving padded
row weights untouched. When a real row carries a zero routing weight, the
row-count filter keeps stock dispatch for that real route while excluding
padded rows by row index. Under EP / DP+EP, ``row_info`` holds one valid-row
count per equal-sized rank segment and composes with EP non-local expert
masking (``topk_indices_local < 0``).
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import fused_moe as fused_moe_adapter
from vllm_torchtpu.layers.core.fused_moe_gmm import prepare_routed_gmm_inputs

E, TOPK, H = 64, 8, 32


def _inputs(t, seed):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(t)]).astype(
        np.int32
    )
    w = rng.random((t, TOPK)).astype(np.float32) + 0.1
    x = rng.standard_normal((t, H)).astype(np.float32)
    return jnp.asarray(x), jnp.asarray(ids), jnp.asarray(w)


def _prep(x, ids, w, **kw):
    out = prepare_routed_gmm_inputs(
        x,
        ids,
        w,
        local_num_experts=E,
        topk=TOPK,
        use_ep=False,
        use_sparse_core=False,
        **kw,
    )
    x_sorted, group_sizes, revert, w_flat, valid, tok_sorted = (
        np.asarray(o) for o in out
    )
    return dict(
        x=x_sorted,
        group_sizes=group_sizes,
        revert=revert,
        valid=valid,
        tok_sorted=tok_sorted,
        weight_used=w_flat * valid,
    )


@pytest.mark.parametrize(
    "t, n", [(16, 16), (16, 15), (16, 12), (16, 1), (16, 0), (32, 17)]
)
@pytest.mark.parametrize("seed", [0, 1])
def test_count_filter_matches_zeroed_weights_with_nonzero_padded_gates(t, n, seed):
    x, ids, w = _inputs(t, seed)
    pad = jnp.arange(t) >= n
    zeroed = jnp.where(pad[:, None], 0.0, w)
    ref = _prep(x, ids, zeroed, skip_padded_tokens=True)
    # Deliberately pass non-zero weights on padded rows to verify exclusion is
    # driven by row_info count rather than weight zeroing.
    count = _prep(
        x,
        ids,
        w,
        skip_padded_tokens=True,
        row_info=jnp.asarray([n], jnp.int32),
    )
    for k in ref:
        np.testing.assert_array_equal(count[k], ref[k], err_msg=k)
    assert int(count["group_sizes"].sum()) == n * TOPK


def test_real_row_zero_gate_keeps_stock_dispatch():
    """Mixed zero-gate and positive-gate real rows: count filter keeps stock's
    valid_mask on real rows (differs from weight-zeroing) and drops padded rows.
    """
    t, n = 16, 12
    x, ids, w = _inputs(t, seed=7)
    w = w.at[0, 0].set(0.0).at[5, 3].set(0.0)
    pad = jnp.arange(t) >= n

    stock_real_only = _prep(x[:n], ids[:n], w[:n], skip_padded_tokens=False)
    zeroed = _prep(x, ids, jnp.where(pad[:, None], 0.0, w), skip_padded_tokens=True)
    count = _prep(
        x,
        ids,
        w,
        skip_padded_tokens=True,
        row_info=jnp.asarray([n], jnp.int32),
    )

    # Zeroing drops the two zero-gate real routes; count keeps them like stock.
    assert int(zeroed["group_sizes"].sum()) == n * TOPK - 2
    assert int(count["group_sizes"].sum()) == n * TOPK
    np.testing.assert_array_equal(count["group_sizes"], stock_real_only["group_sizes"])
    np.testing.assert_array_equal(
        count["valid"].reshape(t, TOPK)[:n], np.ones((n, TOPK), dtype=bool)
    )
    np.testing.assert_array_equal(
        count["valid"].reshape(t, TOPK)[n:], np.zeros((t - n, TOPK), dtype=bool)
    )
    np.testing.assert_array_equal(count["x"][: n * TOPK], stock_real_only["x"])
    np.testing.assert_array_equal(
        count["tok_sorted"][: n * TOPK], stock_real_only["tok_sorted"]
    )
    np.testing.assert_array_equal(
        count["weight_used"].reshape(t, TOPK)[:n], np.asarray(w[:n])
    )


def test_row_info_without_skipping_keeps_stock_dispatch():
    """row_info only takes effect with skip_padded_tokens=True."""
    t, n = 16, 4
    x, ids, w = _inputs(t, seed=3)
    stock = _prep(x, ids, w, skip_padded_tokens=False)
    with_count = _prep(
        x, ids, w, skip_padded_tokens=False, row_info=jnp.asarray([n], jnp.int32)
    )
    for k in stock:
        np.testing.assert_array_equal(with_count[k], stock[k], err_msg=k)
    assert int(with_count["group_sizes"].sum()) == t * TOPK


def _ep_inputs(t, local_experts, global_experts, experts_start, seed):
    rng = np.random.default_rng(seed)
    global_ids = np.stack(
        [rng.choice(global_experts, TOPK, replace=False) for _ in range(t)]
    ).astype(np.int32)
    local_ids = global_ids - experts_start
    is_local = (local_ids >= 0) & (local_ids < local_experts)
    remapped = np.where(is_local, local_ids, -1).astype(np.int32)
    w = rng.random((t, TOPK)).astype(np.float32) + 0.1
    w = np.where(is_local, w, 0.0).astype(np.float32)
    x = rng.standard_normal((t, H)).astype(np.float32)
    return jnp.asarray(x), jnp.asarray(remapped), jnp.asarray(w)


def _prep_ep(
    x,
    ids,
    w,
    local_experts,
    *,
    use_sparse_core=False,
    onehot_moe_permute_threshold=0,
    **kw,
):
    out = prepare_routed_gmm_inputs(
        x,
        ids,
        w,
        local_num_experts=local_experts,
        topk=TOPK,
        use_ep=True,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        **kw,
    )
    x_sorted, group_sizes, revert, w_flat, valid, tok_sorted = (
        np.asarray(o) for o in out
    )
    return dict(
        x=x_sorted,
        group_sizes=group_sizes,
        revert=revert,
        valid=valid,
        tok_sorted=tok_sorted,
        weight_used=w_flat * valid,
    )


_EP_PREP_MODES = [
    pytest.param(False, 0, id="plain_jax"),
    pytest.param(True, 10**9, id="ep_onehot"),
    pytest.param(True, 0, id="ep_ragged"),
]


@pytest.mark.parametrize(
    "use_sparse_core, onehot_threshold",
    _EP_PREP_MODES,
)
@pytest.mark.parametrize(
    "t_per_rank, counts",
    [
        # TP+EP (single segment): partial, full, and all-padded warmup
        (16, [10]),
        (16, [16]),
        (16, [0]),
        # DP+EP (multi-segment): asymmetric ranks including an empty rank (0)
        (8, [6, 0, 8, 3]),
        # DP+EP all-padded warmup (_dummy_run)
        (8, [0, 0, 0, 0]),
        # DP+EP full bucket on every rank
        (8, [8, 8, 8, 8]),
    ],
)
@pytest.mark.parametrize("experts_start", [0, 16])
def test_ep_segmented_row_count_matches_zeroed_weights(
    use_sparse_core, onehot_threshold, t_per_rank, counts, experts_start
):
    """Multi-segment row_info drops each rank's padded tail while keeping EP
    non-local expert filtering (topk_indices_local < 0) intact across plain,
    one-hot, and ragged EP routing-prep modes."""
    local_experts = 16
    global_experts = 64
    num_ranks = len(counts)
    t_total = t_per_rank * num_ranks
    x, ids, w = _ep_inputs(
        t_total, local_experts, global_experts, experts_start, seed=5
    )

    pad = np.zeros(t_total, dtype=bool)
    for r, n in enumerate(counts):
        pad[r * t_per_rank + n : (r + 1) * t_per_rank] = True
    zeroed_w = jnp.where(jnp.asarray(pad)[:, None], 0.0, w)

    mode_kw = dict(
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_threshold,
    )
    ref = _prep_ep(x, ids, zeroed_w, local_experts, skip_padded_tokens=True, **mode_kw)
    # Deliberately keep non-zero weights on padded rows to verify exclusion is
    # driven by row_info rather than weight zeroing.
    seg = _prep_ep(
        x,
        ids,
        w,
        local_experts,
        skip_padded_tokens=True,
        row_info=jnp.asarray(counts, jnp.int32),
        **mode_kw,
    )
    valid_routes = int(seg["group_sizes"].sum())
    expected_valid = (np.asarray(ids) >= 0) & (~pad[:, None])
    assert valid_routes == int(expected_valid.sum())
    for k in ref:
        if k == "x" and use_sparse_core and onehot_threshold == 0:
            # ragged_gather_v2 only materializes the valid prefix [0, valid_routes).
            np.testing.assert_array_equal(
                seg[k][:valid_routes], ref[k][:valid_routes], err_msg=k
            )
        else:
            np.testing.assert_array_equal(seg[k], ref[k], err_msg=k)


@pytest.mark.parametrize(
    "use_sparse_core, onehot_threshold",
    _EP_PREP_MODES,
)
def test_ep_segmented_row_count_preserves_real_zero_gate(
    use_sparse_core, onehot_threshold
):
    """Real rows with exact zero routing weight keep stock EP dispatch under
    multi-segment row_info, while padded rows in every rank are dropped."""
    local_experts, global_experts = 16, 32
    t_per_rank = 8
    counts = [5, 0, 8, 2]
    t_total = t_per_rank * len(counts)
    x, ids, w = _ep_inputs(t_total, local_experts, global_experts, 0, seed=11)

    # Set a real local route on rank 0 (row 1) and rank 2 (row 17) to weight 0.
    ids_np = np.asarray(ids)
    w_mod = w
    zeroed_real_routes = 0
    for row in (1, 2 * t_per_rank + 1):
        local_slots = np.flatnonzero(ids_np[row] >= 0)
        if local_slots.size > 0:
            w_mod = w_mod.at[row, int(local_slots[0])].set(0.0)
            zeroed_real_routes += 1
    assert zeroed_real_routes > 0

    pad = np.zeros(t_total, dtype=bool)
    for r, n in enumerate(counts):
        pad[r * t_per_rank + n : (r + 1) * t_per_rank] = True

    mode_kw = dict(
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_threshold,
    )
    seg = _prep_ep(
        x,
        ids,
        w_mod,
        local_experts,
        skip_padded_tokens=True,
        row_info=jnp.asarray(counts, jnp.int32),
        **mode_kw,
    )
    zero_wt = _prep_ep(
        x,
        ids,
        jnp.where(jnp.asarray(pad)[:, None], 0.0, w_mod),
        local_experts,
        skip_padded_tokens=True,
        **mode_kw,
    )
    expected_valid = (ids_np >= 0) & (~pad[:, None])
    assert int(seg["group_sizes"].sum()) == int(expected_valid.sum())
    assert (
        int(zero_wt["group_sizes"].sum())
        == int(expected_valid.sum()) - zeroed_real_routes
    )
    np.testing.assert_array_equal(seg["valid"].reshape(t_total, TOPK), expected_valid)


def test_segmented_row_info_rejects_non_divisible_tokens():
    x, ids, w = _inputs(10, seed=0)
    with pytest.raises(AssertionError, match="divisible"):
        _prep(
            x,
            ids,
            w,
            skip_padded_tokens=True,
            row_info=jnp.asarray([3, 2, 1], jnp.int32),
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_tokens,num_valid", [(16, 0), (16, 9), (16, 16)])
def test_adapter_row_count_through_custom_op_matches_stock(num_tokens, num_valid):
    """fused_moe_gmm hands row_info to the compiled custom op as a positional
    operand: real rows match the stock op bit for bit, padded rows (non-zero
    gates, perturbed hidden states) are dropped."""
    experts, hidden, inter, topk = 16, 2048, 768, 8
    dev = torch.device("tpu")
    gen = torch.Generator().manual_seed(0)
    w1 = (torch.randn(experts, hidden, 2 * inter, generator=gen) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(experts, inter, hidden, generator=gen) * 0.02).to(torch.bfloat16)
    x = torch.randn(num_tokens, hidden, generator=gen)
    x[num_valid:] *= 50
    rows = torch.arange(num_tokens)[:, None]
    ids = ((rows + 2 * torch.arange(topk)[None, :]) % experts).to(torch.int32)
    weights = torch.rand(num_tokens, topk, generator=gen) * 0.25 + 0.05
    if num_valid > 0:
        weights[0, 0] = 0.0
    count = torch.tensor([num_valid], dtype=torch.int32)
    w1, w2, x, ids, weights, count = (
        t.to(dev)
        for t in (w1, w2, x.to(torch.bfloat16), ids, weights.to(torch.bfloat16), count)
    )

    fused_moe_adapter.prebuild_fused_moe_kernel(
        topk=topk, activation="silu", use_ep=False, skip_padded_tokens=False
    )
    fused_moe_adapter.prebuild_fused_moe_kernel(
        topk=topk,
        activation="silu",
        use_ep=False,
        skip_padded_tokens=True,
        has_row_info=True,
    )

    def stock(x, weights, ids):
        return fused_moe_adapter.fused_moe_gmm(
            x,
            w1,
            w2,
            None,
            None,
            None,
            None,
            weights,
            ids,
            None,
            topk,
            "silu",
            skip_padded_tokens=False,
        )

    def row_count(x, weights, ids, count):
        return fused_moe_adapter.fused_moe_gmm(
            x,
            w1,
            w2,
            None,
            None,
            None,
            None,
            weights,
            ids,
            None,
            topk,
            "silu",
            skip_padded_tokens=True,
            row_info=count,
        )

    compile_opts = dict(backend="tpu", fullgraph=True, dynamic=False)
    ref = torch.compile(stock, **compile_opts)(x, weights, ids).cpu()
    out = torch.compile(row_count, **compile_opts)(x, weights, ids, count).cpu()
    assert torch.equal(out[:num_valid], ref[:num_valid])
    assert torch.equal(out[num_valid:], torch.zeros_like(out[num_valid:]))
    if num_valid < num_tokens:
        assert ref[num_valid:].float().abs().max() > 0


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "t_per_rank, counts, experts_start_val",
    [
        # One-hot permute / unpermute path (<= onehot_moe_permute_threshold)
        (16, [0], 0),
        (16, [9], 0),
        (16, [16], 0),
        (8, [6, 0, 8, 3], 0),
        (8, [6, 0, 8, 3], 8),
        (8, [0, 0, 0, 0], 0),
        # SparseCore ragged_gather_v2 permute gate (> 511 routed rows on v5p/7x)
        (64, [50, 0, 64, 7], 0),
        (64, [50, 0, 64, 7], 8),
        # SparseCore ragged_gather_reduce_v2 VMEM gate (~4,900 rows at H=2048 bf16)
        (320, [300, 0, 320, 7], 0),
        (320, [0, 0, 0, 0], 0),
    ],
)
def test_adapter_ep_segmented_row_count_through_custom_op(
    t_per_rank, counts, experts_start_val
):
    """fused_moe_gmm with use_ep=True and single- or multi-segment row_info:
    active rows match the stock EP op bit for bit and padded rows across every
    rank segment are exact zero (while non-zero in the stock reference)."""
    local_experts, global_experts, hidden, inter, topk = 8, 16, 2048, 768, 4
    num_tokens = t_per_rank * len(counts)
    dev = torch.device("tpu")
    gen = torch.Generator().manual_seed(0)
    w1 = (torch.randn(local_experts, hidden, 2 * inter, generator=gen) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(local_experts, inter, hidden, generator=gen) * 0.02).to(
        torch.bfloat16
    )
    x = torch.randn(num_tokens, hidden, generator=gen)
    pad = torch.zeros(num_tokens, dtype=torch.bool)
    for r, n in enumerate(counts):
        pad[r * t_per_rank + n : (r + 1) * t_per_rank] = True
    x[pad] *= 50

    rows = torch.arange(num_tokens)[:, None]
    ids = ((rows + 2 * torch.arange(topk)[None, :]) % global_experts).to(torch.int32)
    weights = torch.rand(num_tokens, topk, generator=gen) * 0.25 + 0.05
    experts_start = torch.tensor(experts_start_val, dtype=torch.int32)
    row_info = torch.tensor(counts, dtype=torch.int32)

    w1, w2, x, ids, weights, experts_start, row_info = (
        t.to(dev)
        for t in (
            w1,
            w2,
            x.to(torch.bfloat16),
            ids,
            weights.to(torch.bfloat16),
            experts_start,
            row_info,
        )
    )

    fused_moe_adapter.prebuild_fused_moe_kernel(
        topk=topk, activation="silu", use_ep=True, skip_padded_tokens=False
    )
    fused_moe_adapter.prebuild_fused_moe_kernel(
        topk=topk,
        activation="silu",
        use_ep=True,
        skip_padded_tokens=True,
        has_row_info=True,
    )

    def stock(x, weights, ids, experts_start):
        return fused_moe_adapter.fused_moe_gmm(
            x,
            w1,
            w2,
            None,
            None,
            None,
            None,
            weights,
            ids,
            experts_start,
            topk,
            "silu",
            skip_padded_tokens=False,
        )

    def row_count(x, weights, ids, experts_start, row_info):
        return fused_moe_adapter.fused_moe_gmm(
            x,
            w1,
            w2,
            None,
            None,
            None,
            None,
            weights,
            ids,
            experts_start,
            topk,
            "silu",
            skip_padded_tokens=True,
            row_info=row_info,
        )

    compile_opts = dict(backend="tpu", fullgraph=True, dynamic=False)
    ref = torch.compile(stock, **compile_opts)(x, weights, ids, experts_start).cpu()
    out = torch.compile(row_count, **compile_opts)(
        x, weights, ids, experts_start, row_info
    ).cpu()
    assert torch.isfinite(out).all()
    if (~pad).any():
        assert torch.equal(out[~pad], ref[~pad])
    if pad.any():
        assert torch.equal(out[pad], torch.zeros_like(out[pad]))
        assert ref[pad].float().abs().max() > 0
