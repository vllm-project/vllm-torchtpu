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
"""Config.compact_message (the routing message's 64-token slots) and Config.reorder_chain.

Why no token can be dropped: a top-k slot of a shard's message holds exactly one pair per token
of the shard (slot k = every token's k-th choice), so a slot's entry count is the shard's row
count, a static shape, never a routing-dependent count. pallas_geometry compacts only when the
shard's (padded) rows are <= COMPACT_MESSAGE_TOKENS, so every token keeps its pair in every slot.
Expert skew changes how many pairs land on a shard (the routed-rows tables, sized to the no-drop
bound, untouched by the flag), not the message layout.

test_geometry runs anywhere jax imports. The device tests (P3_BENCH=1, a v7x host) compare the layer with and
without the flag on the same input: the outputs must be bit-identical, under standard-normal
routing and under the adversarial "hot" routing where every token picks the same top_k experts
(all on shard 0), from 384 to 2048 global rows (the flag is inactive above 64 rows per shard).
P3_FEP_BRIDGE=1 runs the serving binding's settings, as test_fep_v3_sweep does.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import layer as L
from vllm_torchtpu.kernels.fused_moe.v3.config import COMPACT_MESSAGE_TOKENS

TOP_K = 10
SHARDS = 8


@pytest.mark.parametrize("per_shard", [8, 48, 56, 64, 72, 112, 128, 256, 1024])
def test_geometry(per_shard):
    bps = 8
    rows, tokens, pairs = L.pallas_geometry(per_shard, TOP_K, SHARDS, bps, compact=True)
    rows0, tokens0, pairs0 = L.pallas_geometry(per_shard, TOP_K, SHARDS, bps)
    assert rows == rows0
    # the bound that makes the flag lossless: every row of the shard has its pair in every slot
    assert rows <= tokens
    if rows <= COMPACT_MESSAGE_TOKENS:
        assert tokens == COMPACT_MESSAGE_TOKENS
        assert pairs == 1024 and pairs0 == 2048  # DP4xTP2 decode at 48-64 rows per core
    else:
        assert (tokens, pairs) == (tokens0, pairs0)  # inactive: the uncompacted geometry


DEVICE = bool(os.environ.get("P3_BENCH"))
device_only = pytest.mark.skipif(not DEVICE, reason="device tests: set P3_BENCH=1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe  # noqa: E402

if DEVICE:  # the device world (mesh, weights) of the fused EP scale test; it skips itself without P3_BENCH
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_p3_fused_ep_scale as S

BRIDGE = os.environ.get("P3_FEP_BRIDGE") == "1"
if BRIDGE:
    from vllm_torchtpu.kernels.fused_moe.v3 import layout as _LO
    L.WRAP_IN_JIT = False
    _LO.SIDE_EFFECTS = False
    _LO.SHARD_INDEX_OPERAND = True
TOKENS = [int(t) for t in os.environ.get("P3_FEP_COMPACT_TOKENS", "384,448,512,1024,2048").split(",") if t]


def _fn(mesh, cfg, n):
    def run(x, gating, w1, w2, s1, s2):
        extra = {}
        if BRIDGE:
            extra = dict(shard_index=jnp.arange(n, dtype=jnp.int32).reshape(n, 1),
                         valid_tokens=jnp.full((n,), x.shape[0] // n, jnp.int32),
                         drop_nan_rows=True)
        return fused_ep_moe(x, w1, w2, s1, s2, gating, top_k=S.TOPK, renormalize=True, mesh=mesh,
                            tile_rows=S.TILE_ROWS, weight_format="fp8", activation="silu",
                            config=cfg, experts_per_shard=S.E // n, **extra)
    return jax.jit(run)


def _gating(kind, tokens):
    g = S._host_gating(3, tokens, S.E)
    if kind == "hot":
        # every token's top_k is experts 0..top_k-1 (all on shard 0), in a fixed order
        g = 0.1 * g
        g[:, :S.TOPK] += np.arange(30.0, 30.0 - S.TOPK, -1.0, dtype=np.float32)
    return g


FLAGS = {"cm": {"compact_message": True}, "ro": {"reorder_chain": True},
         "cmro": {"compact_message": True, "reorder_chain": True}}
FLAG_SETS = [f for f in os.environ.get("P3_FEP_COMPACT_FLAGS", "cm,ro,cmro").split(",") if f]


@device_only
@pytest.mark.parametrize("flags", FLAG_SETS)
@pytest.mark.parametrize("routing", ["normal", "hot"])
@pytest.mark.parametrize("tokens", TOKENS)
def test_compact_is_bit_identical(tokens, routing, flags):
    mesh = S._mesh()
    n = S._shards(mesh)
    base = dataclasses.replace(
        Config.for_shape(hidden=S.H, inter=S.INTER, experts_per_shard=S.E // n, top_k=S.TOPK),
        combine_real_tokens=True, sort_index_compare=True)
    w1, w2, s1, s2 = S._prod_weights(mesh)
    x = S._make_x(mesh, 2, tokens, S.H)
    g = _gating(routing, tokens)
    if routing == "hot":
        assert (np.argsort(-g, axis=1)[:, :S.TOPK] == np.arange(S.TOPK)).all()
    gating = S._put(mesh, g, P(MESH_AXIS))
    want = _fn(mesh, base, n)(x, gating, w1, w2, s1, s2)
    got_fn = _fn(mesh, dataclasses.replace(base, **FLAGS[flags]), n)
    got = got_fn(x, gating, w1, w2, s1, s2)
    same = bool(S._scalar(jnp.all(got == want)))
    num = S._check(mesh, got_fn, x, w1, w2, s1, s2, gating, S.TOPK)
    print(f"FEP_COMPACT flags={flags} tokens={tokens} per_shard={tokens // n} routing={routing} "
          f"bit_identical={same} {num}", flush=True)
    assert same, f"{flags} changed the layer's output"
    assert num["numerics_pass"], num
