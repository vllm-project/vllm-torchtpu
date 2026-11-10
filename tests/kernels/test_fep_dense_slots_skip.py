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
"""Dense slots with the unrouted-expert skip (Config.dense_slots_skip_unrouted) and the weight
slot count (Config.dense_slots_weight_slots), on the device, at the DP4xTP2 band A shapes.

Same world as test_fep_dense_slots.py (Qwen3.5-397B geometry, fp8 weights with per-channel
scales, BRIDGE serving binding with P3_FEP_BRIDGE=1), plus a routing knob, because standard-normal
logits hit every expert and leave nothing to skip:
  P3_DS_ROUTING: comma list of routings; "uniform" (standard normal), "skew<s>" (each expert gets a
    fixed popularity bias s * N(0, 1), a lognormal-like load), "cold<f>" (a fixed fraction f of
    every shard's experts never routed; expert 0 of each shard is always cold, so a prefetched
    expert without rows is covered).
For each routing the host computes the live-expert fraction per shard (mean and max over shards;
the slowest shard sets the layer time).

test_identity: the served dense config (fused + merged + unpadded at multiples of 8 per shard,
fused + merged otherwise) with the skip and/or other weight slots against the same config
without them: every output value equal (bit identity), at elementwise float32 and bfloat16, with
partial valid rows when P3_DS_NV is set.
test_time: per-layer time (chained median, the bench's convention), base vs variants, interleaved
over P3_DS_REPEATS rounds in one process.
P3_DS_MUTANT=slot / drop breaks the skip path on purpose (identity must fail; drop may hang, so
run it under a timeout). One DS_SKIP json line per result.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import numpy as np
import pytest

if not os.environ.get("P3_BENCH"):
    pytest.skip("set P3_BENCH=1", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config  # noqa: E402
from vllm_torchtpu.kernels.fused_moe.v3 import dense_slots as DS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_p3_fused_ep_scale as S  # noqa: E402

TOKENS = [int(t) for t in os.environ.get("P3_FEP_TOKENS", "288,320,384").split(",") if t]
REPEATS = int(os.environ.get("P3_DS_REPEATS", "2"))
ROUTINGS = [r for r in os.environ.get("P3_DS_ROUTING", "uniform,skew1.0,cold0.25").split(",") if r]
MUTANT = os.environ.get("P3_DS_MUTANT", "")
NV_SHORT = int(os.environ.get("P3_DS_NV", "0"))
# Variants: name -> (skip_unrouted, weight_slots); "base" is the served config.
_ALL = {"base": (False, 0), "skip": (True, 0), "w4": (False, 4), "skip_w4": (True, 4),
        "w2": (False, 2), "skip_w2": (True, 2)}
VARIANTS = [v for v in os.environ.get("P3_DS_VARIANTS", "skip,w4,skip_w4").split(",") if v]

BRIDGE = os.environ.get("P3_FEP_BRIDGE") == "1"
if BRIDGE:
    from vllm_torchtpu.kernels.fused_moe.v3 import layer as _L, layout as _LO
    _L.WRAP_IN_JIT = False
    _LO.SIDE_EFFECTS = False
    _LO.SHARD_INDEX_OPERAND = True

if MUTANT:
    DS.SKIP_MUTANT = MUTANT


def _extra(n, x):
    if not BRIDGE:
        return {}
    per = x.shape[0] // n
    valid = jnp.asarray([per - (NV_SHORT if d % 2 else 0) for d in range(n)], jnp.int32)
    return dict(shard_index=jnp.arange(n, dtype=jnp.int32).reshape(n, 1),
                valid_tokens=valid, drop_nan_rows=True)


def _fn(mesh, cfg, n, tokens):
    up = (tokens // n) % 8 == 0  # the adapter asks for unpadded; the layer takes it at x8 only

    def run(x, gating, w1, w2, s1, s2):
        return DS.dense_slots_moe(x, w1, w2, s1, s2, gating, top_k=S.TOPK, renormalize=True,
                                  mesh=mesh, tile_rows=S.TILE_ROWS, config=cfg,
                                  experts_per_shard=S.E // n, fused=True, pallas_tables=False,
                                  merged=True, unpadded=up, **_extra(n, x))
    return jax.jit(run)


def _gating_host(routing, tokens, n):
    g = S._host_gating(2, tokens, S.E)
    G = S.E // n
    rng = np.random.default_rng(7)
    if routing == "uniform":
        pass
    elif routing.startswith("skew"):
        s = float(routing[4:])
        g = g + (s * rng.standard_normal(S.E)).astype(np.float32)[None, :]
    elif routing.startswith("cold"):
        f = float(routing[4:])
        n_cold = max(1, int(round(f * G)))
        bias = np.zeros(S.E, np.float32)
        for d in range(n):
            cold = [0] + list(rng.choice(np.arange(1, G), n_cold - 1, replace=False))
            bias[[d * G + c for c in cold]] = -30.0
        g = g + bias[None, :]
    else:
        raise ValueError(routing)
    return g


def _live_stats(g, tokens, n):
    """Live-expert fraction per shard under top-k of the logits (the select's choice; padding
    rows excluded via NV_SHORT the same way the kernel masks them)."""
    G = S.E // n
    per = tokens // n
    keep = np.ones(tokens, bool)
    if NV_SHORT:
        for d in range(1, n, 2):
            keep[d * per + per - NV_SHORT:(d + 1) * per] = False
    top = np.argsort(-g[keep], axis=1)[:, :S.TOPK]
    hit = np.zeros(S.E, bool)
    hit[np.unique(top)] = True
    frac = hit.reshape(n, G).mean(axis=1)
    return {"live_mean": round(float(frac.mean()), 4), "live_max": round(float(frac.max()), 4),
            "live_min": round(float(frac.min()), 4)}


def _checksum(out):
    h = jax.jit(lambda a: jnp.sum(a.astype(jnp.float32) *
                                  jnp.arange(1, a.shape[1] + 1, dtype=jnp.float32)))(out)
    return float(S._scalar(h))


def _emit(row):
    print("DS_SKIP " + json.dumps(row), flush=True)
    S._record(row)


_W = {}


def _world(mesh):
    if not _W:
        _W["w"] = S._prod_weights(mesh)
    return _W["w"]


def _cfg(base, name):
    skip, ws = _ALL[name]
    return dataclasses.replace(base, dense_slots_skip_unrouted=skip, dense_slots_weight_slots=ws)


@pytest.mark.parametrize("tokens", TOKENS)
def test_identity(tokens):
    mesh = S._mesh()
    n = S._shards(mesh)
    w1, w2, s1, s2 = _world(mesh)
    x = S._make_x(mesh, 2, tokens, S.H)
    base = Config.for_shape(hidden=S.H, inter=S.INTER, experts_per_shard=S.E // n, top_k=S.TOPK)
    bad = []
    for routing in ROUTINGS:
        gh = _gating_host(routing, tokens, n)
        gating = S._put(mesh, gh, P(MESH_AXIS))
        stats = _live_stats(gh, tokens, n)
        for ew in ("float32", "bfloat16"):
            b0 = dataclasses.replace(base, elementwise_dtype=ew)
            a = _fn(mesh, _cfg(b0, "base"), n, tokens)(x, gating, w1, w2, s1, s2)
            for name in VARIANTS:
                b = _fn(mesh, _cfg(b0, name), n, tokens)(x, gating, w1, w2, s1, s2)
                d = jax.jit(lambda u, v: (
                    jnp.max(jnp.abs(u.astype(jnp.float32) - v.astype(jnp.float32))),
                    jnp.sum((u != v).astype(jnp.int32))))(a, b)
                row = {"label": "ds_skip_identity", "tokens": tokens, "routing": routing,
                       "elementwise": ew, "variant": name, "mutant": MUTANT, "nv_short": NV_SHORT,
                       **stats, "max_abs_diff": float(S._scalar(d[0])),
                       "n_diff": int(S._scalar(d[1])), "n_values": int(a.size),
                       "checksum_base": _checksum(a), "checksum_variant": _checksum(b)}
                row["bit_identical"] = row["n_diff"] == 0
                _emit(row)
                if not row["bit_identical"]:
                    bad.append((routing, ew, name, row["n_diff"]))
    assert not bad, bad  # a mutant (P3_DS_MUTANT) must fail here


@pytest.mark.skipif(bool(MUTANT), reason="timing only on the real layer")
@pytest.mark.parametrize("tokens", TOKENS)
def test_time(tokens):
    mesh = S._mesh()
    n = S._shards(mesh)
    w1, w2, s1, s2 = _world(mesh)
    x = S._make_x(mesh, 2, tokens, S.H)
    base = Config.for_shape(hidden=S.H, inter=S.INTER, experts_per_shard=S.E // n, top_k=S.TOPK)
    fns = {name: _fn(mesh, _cfg(base, name), n, tokens) for name in ["base"] + VARIANTS}
    for routing in ROUTINGS:
        gh = _gating_host(routing, tokens, n)
        gating = S._put(mesh, gh, P(MESH_AXIS))
        stats = _live_stats(gh, tokens, n)
        args = (gating, w1, w2, s1, s2)
        for r in range(REPEATS):
            for name, fn in fns.items():
                row = {"label": "ds_skip_time", "variant": name, "tokens": tokens,
                       "routing": routing, "round": r, "per_shard": tokens // n,
                       "bridge": BRIDGE, **stats}
                try:
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(x, *args))
                    row["first_call_s"] = round(time.perf_counter() - t0, 1)
                    med, best, _ = S._loop_time(S._chained(fn), x, args, S.REPS, S.ITERS,
                                                chain=True)
                    row["us_per_layer_median"] = round(med * 1e6, 1)
                    row["us_per_layer_min"] = round(best * 1e6, 1)
                    b2b, b2b_min, out = S._loop_time(fn, x, args, S.REPS, S.ITERS, chain=False)
                    row["us_b2b_median"] = round(b2b * 1e6, 1)
                    row["out_checksum"] = _checksum(out)
                except Exception as e:  # a refusal (e.g. VMEM) is a result
                    row["error"] = f"{type(e).__name__}: {str(e)[:600]}"
                _emit(row)
