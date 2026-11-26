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
"""TensorCore MoE combine against a NumPy reference (Pallas interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox import tc_combine

TOPK, EXPERTS, EP = 4, 128, 8


def _routes(num_tokens, hidden, skew, padded, seed=0, first_expert=None):
    """GMM output laid out as fused_moe_gmm's ragged EP path does: rows of
    this rank's 16 experts, grouped by expert, tokens ascending within a
    group; rows past the valid prefix stale (NaN)."""
    rng = np.random.default_rng(seed)
    p = rng.pareto(skew, EXPERTS) + 1e-3 if skew else np.ones(EXPERTS)
    ids = np.stack(
        [
            rng.choice(EXPERTS, TOPK, replace=False, p=p / p.sum())
            for _ in range(num_tokens)
        ]
    )
    if first_expert is not None:  # make it every token's first choice
        absent = ~(ids == first_expert).any(axis=1)
        ids[absent, 0] = first_expert
    local_experts = EXPERTS // EP
    valid = ids < local_experts
    valid[num_tokens - padded :] = False
    valid = valid.reshape(-1)
    keys = np.where(valid, ids.reshape(-1), local_experts)
    order = np.argsort(keys, kind="stable")
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    group_sizes = np.bincount(keys, minlength=local_experts + 1)[:local_experts]
    n_valid = int(group_sizes.sum())
    x = np.full((num_tokens * TOPK, hidden), np.nan, np.float32)
    x[:n_valid] = rng.standard_normal((n_valid, hidden))
    w = rng.random(num_tokens * TOPK).astype(np.float32)
    return x, pos.astype(np.int32), w, valid, group_sizes.astype(np.int32)


@pytest.mark.parametrize(
    "num_tokens, skew, padded, w_dtype",
    [
        (512, 0.0, 0, jnp.bfloat16),
        (1024, 0.5, 100, jnp.bfloat16),
        (768, 2.0, 7, jnp.float32),
    ],
)
def test_matches_reference(num_tokens, skew, padded, w_dtype):
    x, pos, w, valid, group_sizes = _routes(num_tokens, 256, skew, padded)
    xb = jnp.asarray(x, jnp.bfloat16)
    wj = jnp.asarray(w, w_dtype)
    out = tc_combine.tc_combine(
        xb,
        jnp.asarray(pos),
        wj,
        jnp.asarray(valid),
        jnp.asarray(group_sizes),
        topk=TOPK,
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    xf = np.asarray(xb.astype(jnp.float32))
    wf = np.asarray(wj.astype(jnp.float32))
    rows = np.where(valid[:, None], xf[pos] * wf[:, None], 0.0)
    ref = rows.reshape(num_tokens, TOPK, -1).sum(axis=1)
    got = np.asarray(out.astype(jnp.float32))
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, ref, atol=2e-2, rtol=1e-2)


def _check(x, pos, w, valid, group_sizes, num_tokens):
    xb = jnp.asarray(x, jnp.bfloat16)
    out = tc_combine.tc_combine(
        xb,
        jnp.asarray(pos),
        jnp.asarray(w, jnp.bfloat16),
        jnp.asarray(valid),
        jnp.asarray(group_sizes),
        topk=TOPK,
        interpret=pltpu.InterpretParams(),
    )
    xf = np.asarray(xb.astype(jnp.float32))
    wf = np.asarray(jnp.asarray(w, jnp.bfloat16).astype(jnp.float32))
    rows = np.where(valid[:, None], xf[pos] * wf[:, None], 0.0)
    ref = rows.reshape(num_tokens, TOPK, -1).sum(axis=1)
    got = np.asarray(out.astype(jnp.float32))
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, ref, atol=2e-2, rtol=1e-2)


def test_block_without_local_routes():
    x, pos, w, valid, group_sizes = _routes(768, 256, 0.0, 0, seed=1)
    valid = valid.reshape(768, TOPK)
    valid[256:512] = False  # the middle block routes entirely off-rank
    valid = valid.reshape(-1)
    keys = np.where(valid, np.searchsorted(np.cumsum(group_sizes), pos, "right"), 16)
    # Rebuild the expert layout for the reduced route set.
    order = np.argsort(keys, kind="stable")
    new_pos = np.empty_like(order)
    new_pos[order] = np.arange(order.size)
    sizes = np.bincount(keys, minlength=17)[:16].astype(np.int32)
    n_valid = int(sizes.sum())
    xs = np.full_like(x, np.nan)
    xs[new_pos[valid]] = x[pos[valid]]
    assert n_valid == int(valid.sum())
    _check(xs, new_pos.astype(np.int32), w, valid, sizes, 768)


def test_rows_out_of_token_order_fall_back():
    x, pos, w, valid, group_sizes = _routes(2048, 256, 0.0, 0, seed=2, first_expert=0)
    # Shuffle the rows of each expert group: token order is broken, so a
    # block's ranges span most of a large group, more tiles than the buffer
    # holds, and the gather path takes over.
    rng = np.random.default_rng(3)
    perm = np.arange(x.shape[0])
    start = 0
    for size in group_sizes:
        perm[start : start + size] = rng.permutation(perm[start : start + size])
        start += size
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.size)
    pos = inverse[pos].astype(np.int32)
    _, counts, _ = tc_combine._plan(
        jnp.asarray(pos),
        jnp.asarray(valid),
        jnp.asarray(group_sizes),
        2048,
        TOPK,
        tc_combine.TOKEN_BLOCK,
    )
    assert int(counts.max()) > tc_combine.max_tiles(tc_combine.TOKEN_BLOCK, TOPK, 16)
    _check(x[perm], pos, w, valid, group_sizes, 2048)


def test_supported_shapes():
    assert tc_combine.supported(8192, 6144, TOPK, 16, jnp.bfloat16)
    assert tc_combine.supported(4096, 6144, TOPK, 16, jnp.bfloat16)
    assert not tc_combine.supported(4100, 6144, TOPK, 16, jnp.bfloat16)
    assert not tc_combine.supported(8192, 6144, TOPK, 16, jnp.float32)


@pytest.mark.parametrize("hidden, inter", [(512, 128), (1024, 256)])
def test_shared_down_matches_unfused(hidden, inter):
    """The fused shared-expert down projection equals the XLA path's
    bf16(bf16(act @ w2) * s2) + routed (shared_expert_mlp's rounding)."""
    num_tokens = 512
    x, pos, w, valid, group_sizes = _routes(num_tokens, hidden, 0.5, 9, seed=3)
    rng = np.random.default_rng(4)
    act = jnp.asarray(rng.standard_normal((num_tokens, inter)), jnp.bfloat16)
    w2 = jnp.asarray(rng.standard_normal((inter, hidden)) * 0.1, jnp.float8_e4m3fn)
    s2 = jnp.asarray(rng.random((1, hidden)) + 0.5, jnp.float32)
    args = (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(pos),
        jnp.asarray(w, jnp.bfloat16),
        jnp.asarray(valid),
        jnp.asarray(group_sizes),
    )
    routed = tc_combine.tc_combine(*args, topk=TOPK, interpret=pltpu.InterpretParams())
    fused = tc_combine.tc_combine(
        *args,
        topk=TOPK,
        shared_down=(act, w2, s2),
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    y = jnp.matmul(act, w2.astype(jnp.bfloat16)).astype(jnp.float32) * s2
    ref = y.astype(jnp.bfloat16) + routed
    got = np.asarray(fused.astype(jnp.float32))
    assert not np.isnan(got).any()
    np.testing.assert_allclose(
        got, np.asarray(ref.astype(jnp.float32)), rtol=1e-2, atol=2e-2
    )
    assert tc_combine.supported(8192, 6144, 4, 16, jnp.bfloat16, shared_inter=384)


def test_token_pieces_equal_the_whole():
    """fused_moe_gmm's TPU_MOE_TC_COMBINE_PIECES: consecutive token pieces of
    the routes, combined one call each, give the whole call's rows."""
    x, pos, w, valid, group_sizes = _routes(1024, 256, 0.5, 9)
    args = (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(pos),
        jnp.asarray(w, jnp.bfloat16),
    )
    whole = tc_combine.tc_combine(
        *args, jnp.asarray(valid), jnp.asarray(group_sizes), topk=TOPK, interpret=True
    )
    half = 512 * TOPK
    pieces = [
        tc_combine.tc_combine(
            args[0],
            args[1][sl],
            args[2][sl],
            jnp.asarray(valid)[sl],
            jnp.asarray(group_sizes),
            topk=TOPK,
            fallback=False,
            interpret=True,
        )
        for sl in (slice(0, half), slice(half, 2 * half))
    ]
    np.testing.assert_array_equal(
        np.asarray(jnp.concatenate(pieces), np.float32), np.asarray(whole, np.float32)
    )


@pytest.mark.parametrize(
    "num_tokens, skew, padded, w_dtype",
    [(512, 0.0, 0, jnp.bfloat16), (768, 2.0, 7, jnp.float32)],
)
def test_slot_major_routes_match_token_major(num_tokens, skew, padded, w_dtype):
    """[topk, T] route arrays (the routing prep's layout) give bit-identical
    output to the flat token-major ones."""
    x, pos, w, valid, group_sizes = _routes(num_tokens, 256, skew, padded, seed=4)
    xb = jnp.asarray(x, jnp.bfloat16)
    wj = jnp.asarray(w, w_dtype)

    def run(*routes):
        return tc_combine.tc_combine(
            xb,
            *routes,
            jnp.asarray(group_sizes),
            topk=TOPK,
            interpret=pltpu.InterpretParams(),
        )

    def slot_major(a):
        return jnp.asarray(a).reshape(num_tokens, TOPK).T

    flat = run(jnp.asarray(pos), wj, jnp.asarray(valid))
    slots = run(slot_major(pos), slot_major(wj), slot_major(valid))
    np.testing.assert_array_equal(
        np.asarray(flat.astype(jnp.float32)), np.asarray(slots.astype(jnp.float32))
    )
