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
"""Row-count padding filter in prepare_routed_gmm_inputs (non-EP).

When every real row carries non-zero routing weights, passing ``row_info``
builds the same routing layout as zeroing padded weights while leaving padded
row weights untouched. When a real row carries a zero routing weight, the
row-count filter keeps stock dispatch for that real route while excluding
padded rows by row index.
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
