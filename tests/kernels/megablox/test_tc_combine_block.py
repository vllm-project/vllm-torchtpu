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
"""tc_combine: the token block, the per-block plan and the combine."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import tc_combine


@pytest.mark.parametrize(
    "tokens, experts, block",
    [
        (8192, 16, 256),  # EP8 on 128 experts
        (8192, 32, 128),  # 4 x 2 hierarchical EP: 256 tokens' tiles do not fit
        (8192, 64, 0),  # 2 x 4 hierarchical EP: neither block fits
        (8064, 16, 0),  # not a multiple of 128 tokens
    ],
)
def test_token_block(tokens, experts, block):
    assert tc_combine.pick_token_block(tokens, 6144, 4, experts) == block
    assert tc_combine.supported(tokens, 6144, 4, experts, jnp.bfloat16) == (block > 0)


def _routing(tokens, experts, topk, real, seed):
    """Stable expert sort of a uniform top-k routing; routes of padded tokens
    and of other EP ranks' experts (ids >= `experts`) sort after the rest."""
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((tokens, 2 * experts)), axis=1)[:, :topk]
    local = (ids < experts) & (np.arange(tokens)[:, None] < real)
    keys = np.where(local, ids, experts).reshape(-1)
    order = np.argsort(keys, kind="stable")
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    sizes = np.bincount(keys[keys < experts], minlength=experts)
    return pos.astype(np.int32), local.reshape(-1), sizes.astype(np.int32)


def _reference_plan(pos, local, sizes, token_block, topk):
    """Per block: the tiles covering each expert's rows, in expert order, and
    each route's row in the concatenation of those tiles (-1: not local)."""
    ends = np.cumsum(sizes)
    expert = np.searchsorted(ends, pos, side="right")
    routes = token_block * topk
    tiles, rows = [], np.full(pos.shape, -1)
    for b in range(pos.size // routes):
        blk = slice(b * routes, (b + 1) * routes)
        block_tiles = []
        for e in range(sizes.size):
            hit = local[blk] & (expert[blk] == e)
            if not hit.any():
                continue
            lo, hi = pos[blk][hit].min(), pos[blk][hit].max() + 1
            first = lo // tc_combine.TILE_ROWS
            shift = (len(block_tiles) - first) * tc_combine.TILE_ROWS
            rows[blk][hit] = pos[blk][hit] + shift
            block_tiles += range(first, -(-hi // tc_combine.TILE_ROWS))
        tiles.append(block_tiles)
    return tiles, rows


@pytest.mark.parametrize(
    "tokens, experts, real",
    [(2048, 36, 1507), (2048, 36, 2048), (1024, 16, 700), (512, 36, 0)],
)
def test_plan_matches_reference(tokens, experts, real):
    topk, block = 4, 128
    pos, local, sizes = _routing(tokens, experts, topk, real, seed=tokens + real)
    plan, rows = tc_combine._plan(
        jnp.asarray(pos), jnp.asarray(sizes), tokens, topk, block, interpret=True
    )
    tiles, ref_rows = _reference_plan(pos, local, sizes, block, topk)
    stride = tc_combine.plan_stride(tc_combine.max_tiles(block, topk, experts))
    plan = np.asarray(plan).reshape(-1, stride)
    for b, block_tiles in enumerate(tiles):
        assert plan[b, -1] == len(block_tiles)
        np.testing.assert_array_equal(plan[b, : len(block_tiles)], block_tiles)
    np.testing.assert_array_equal(np.asarray(rows).reshape(-1), ref_rows)


def test_combine_matches_gather():
    tokens, experts, topk, hidden = 512, 36, 4, 256
    pos, local, sizes = _routing(tokens, experts, topk, 400, seed=0)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((tokens * topk, hidden)).astype(np.float32)
    x[sizes.sum() :] = np.nan  # rows the GMM never wrote
    w = rng.random(tokens * topk).astype(np.float32)
    out = tc_combine.tc_combine(
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(pos),
        jnp.asarray(w),
        jnp.asarray(sizes),
        topk=topk,
        interpret=True,
    )
    xb = np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32), np.float64)
    routed = np.where(local[:, None], xb[np.where(local, pos, 0)] * w[:, None], 0)
    ref = routed.reshape(tokens, topk, hidden).sum(axis=1)
    np.testing.assert_allclose(np.asarray(out, np.float64), ref, rtol=1e-2, atol=1e-2)


def test_combine_residual_is_the_bf16_sum():
    tokens, experts, topk, hidden = 512, 36, 4, 256
    pos, _, sizes = _routing(tokens, experts, topk, 400, seed=2)
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal((tokens * topk, hidden)), jnp.bfloat16)
    w = jnp.asarray(rng.random(tokens * topk), jnp.float32)
    residual = jnp.asarray(
        rng.standard_normal((tokens, hidden))
        * np.exp(4 * rng.standard_normal((tokens, 1))),
        jnp.bfloat16,
    )
    args = (x, jnp.asarray(pos), w, jnp.asarray(sizes))
    routed = tc_combine.tc_combine(*args, topk=topk, interpret=True)
    fused = tc_combine.tc_combine(*args, topk=topk, residual=residual, interpret=True)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(fused, jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(routed + residual, jnp.uint16)),
    )
