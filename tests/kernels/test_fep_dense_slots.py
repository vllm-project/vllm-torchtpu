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
"""Dense-slots fused EP layer (v3/dense_slots.py) against fused EP v4 at the DP4xTP2 decode
shapes: numerics vs the dense reference, bit identity vs v3 with bf16 result rows (the same
per-row math), and per-layer time, interleaved with v4 in the same process.

Reuses test_p3_fused_ep_scale.py's world (Qwen3.5-397B geometry, fp8 weights with per-channel
scales, standard-normal router logits, chained per-layer timing) and test_fep_v3_sweep.py's
serving-binding switch (P3_FEP_BRIDGE=1).

Env: P3_BENCH=1; P3_FEP_TOKENS (default 384,448,512); P3_DS_REPEATS (interleaved timing
rounds, default 2); P3_DS_TRACE=1 for an xprof trace of each variant at 448; P3_DS_MUTANT
(pos / onehot) breaks the layer on purpose. One DS_BENCH json line per result.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import pytest

if not os.environ.get("P3_BENCH"):
    pytest.skip("set P3_BENCH=1", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe  # noqa: E402
from vllm_torchtpu.kernels.fused_moe.v3 import dense_slots as DS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_p3_fused_ep_scale as S  # noqa: E402

TOKENS = [int(t) for t in os.environ.get("P3_FEP_TOKENS", "384,448,512").split(",") if t]
REPEATS = int(os.environ.get("P3_DS_REPEATS", "2"))
MUTANT = os.environ.get("P3_DS_MUTANT", "")
V4 = {"combine_real_tokens": True, "sort_index_compare": True, "compact_message": True,
      "reorder_chain": True}

BRIDGE = os.environ.get("P3_FEP_BRIDGE") == "1"
if BRIDGE:
    from vllm_torchtpu.kernels.fused_moe.v3 import layer as _L, layout as _LO
    _L.WRAP_IN_JIT = False
    _LO.SIDE_EFFECTS = False
    _LO.SHARD_INDEX_OPERAND = True

if MUTANT == "pos":  # the destination reads its neighbour pair's arrival row
    DS.TABLE_HOOK = lambda t: dict(t, pos=jnp.roll(t["pos"].reshape(-1), 1).reshape(t["pos"].shape))
elif MUTANT == "onehot":  # the gather picks each expert's rows one token off
    DS.TABLE_HOOK = lambda t: dict(t, mp=jnp.roll(t["mp"], 1, axis=1))
elif MUTANT == "mergedpos":  # the merged program's destination reads the next arrival row
    DS.MERGED_POS_SHIFT = 1
PT_ARMS = [f == "1" for f in os.environ.get("P3_DS_PT", "0").split(",")]
MG_ARMS = [f == "1" for f in os.environ.get("P3_DS_MG", "0").split(",")]
TILES = [int(t) for t in os.environ.get("P3_DS_TILES", str(S.TILE_ROWS)).split(",")]


NV_SHORT = int(os.environ.get("P3_DS_NV", "0"))  # odd shards get this many fewer valid rows
UP_ARMS = [f == "1" for f in os.environ.get("P3_DS_UP", "0").split(",")]


def _extra(n, x):
    if not BRIDGE:
        return {}
    per = x.shape[0] // n
    valid = jnp.asarray([per - (NV_SHORT if d % 2 else 0) for d in range(n)], jnp.int32)
    return dict(shard_index=jnp.arange(n, dtype=jnp.int32).reshape(n, 1),
                valid_tokens=valid, drop_nan_rows=True)


def _v3_fn(mesh, cfg, n):
    def run(x, gating, w1, w2, s1, s2):
        return fused_ep_moe(x, w1, w2, s1, s2, gating, top_k=S.TOPK, renormalize=True, mesh=mesh,
                            tile_rows=S.TILE_ROWS, weight_format="fp8", activation="silu",
                            config=cfg, experts_per_shard=S.E // n, **_extra(n, x))
    return jax.jit(run)


FUSED_ARMS = [f == "1" for f in os.environ.get("P3_DS_FUSED", "0,1").split(",")]


def _ds_fn(mesh, cfg, n, fused=False, pt=False, tile=None, mg=False, up=False):
    def run(x, gating, w1, w2, s1, s2):
        return DS.dense_slots_moe(x, w1, w2, s1, s2, gating, top_k=S.TOPK, renormalize=True,
                                  mesh=mesh, tile_rows=tile or S.TILE_ROWS, config=cfg,
                                  experts_per_shard=S.E // n, fused=fused, pallas_tables=pt,
                                  merged=mg, unpadded=up,
                                  **_extra(n, x))
    return jax.jit(run)


def _checksum(out):
    h = jax.jit(lambda a: jnp.sum(a.astype(jnp.float32) *
                                  jnp.arange(1, a.shape[1] + 1, dtype=jnp.float32)))(out)
    return float(S._scalar(h))


def _emit(row):
    print("DS_BENCH " + json.dumps(row), flush=True)
    S._record(row)


_W = {}


def _world(mesh):
    if not _W:
        _W["w"] = S._prod_weights(mesh)
    return _W["w"]


@pytest.mark.parametrize("tokens", TOKENS)
def test_identity_and_numerics(tokens):
    """dense slots vs v3 with bf16 result rows, at elementwise float32 (every compilation
    agrees to the bit) and at the served bfloat16; numerics vs the dense reference."""
    mesh = S._mesh()
    n = S._shards(mesh)
    w1, w2, s1, s2 = _world(mesh)
    x = S._make_x(mesh, 2, tokens, S.H)
    gating = S._put(mesh, S._host_gating(2, tokens, S.E), P(MESH_AXIS))
    base = Config.for_shape(hidden=S.H, inter=S.INTER, experts_per_shard=S.E // n, top_k=S.TOPK)
    for ew, fused, pt, tile, mg, up in [(e, f, p, t, m, u) for e in ("float32", "bfloat16")
                                        for f in FUSED_ARMS for p in PT_ARMS for t in TILES
                                        for m in MG_ARMS for u in UP_ARMS]:
        cfg_v3 = dataclasses.replace(base, **V4, result_rows="bf16", elementwise_dtype=ew)
        cfg_ds = dataclasses.replace(base, elementwise_dtype=ew)
        a = _v3_fn(mesh, cfg_v3, n)(x, gating, w1, w2, s1, s2)
        b = _ds_fn(mesh, cfg_ds, n, fused, pt, tile, mg, up)(x, gating, w1, w2, s1, s2)
        d = jax.jit(lambda u, v: (jnp.max(jnp.abs(u.astype(jnp.float32) - v.astype(jnp.float32))),
                                  jnp.sum((u != v).astype(jnp.int32))))(a, b)
        row = {"label": "ds_identity", "tokens": tokens, "elementwise": ew, "mutant": MUTANT,
               "fused": fused, "pallas_tables": pt, "tile": tile, "merged": mg, "unpadded": up,
               "nv_short": NV_SHORT,
               "max_abs_diff": float(S._scalar(d[0])), "n_diff": int(S._scalar(d[1])),
               "n_values": int(a.size), "checksum_v3_bf16res": _checksum(a),
               "checksum_ds": _checksum(b)}
        row["bit_identical"] = row["n_diff"] == 0
        _emit(row)
    if NV_SHORT:  # the dense reference has no valid-row notion; identity covers this case
        return
    for fused, pt, mg, up in [(f, p, m, u) for f in FUSED_ARMS for p in PT_ARMS for m in MG_ARMS
                              for u in UP_ARMS]:
        num = S._check(mesh, _ds_fn(mesh, dataclasses.replace(base), n, fused, pt, None, mg, up),
                       x, w1, w2, s1, s2, gating, S.TOPK)
        _emit({"label": "ds_numerics", "tokens": tokens, "mutant": MUTANT, "fused": fused,
               "pallas_tables": pt, "merged": mg, "unpadded": up, **num})
        assert num["numerics_pass"], num  # a mutant (P3_DS_MUTANT) must fail here


@pytest.mark.skipif(bool(MUTANT), reason="timing only on the real layer")
@pytest.mark.parametrize("tokens", TOKENS)
def test_time(tokens):
    """Per-layer time, interleaved: v4 (fp8 result rows, served), v3+bf16 result rows, dense
    slots; REPEATS rounds in that order."""
    mesh = S._mesh()
    n = S._shards(mesh)
    w1, w2, s1, s2 = _world(mesh)
    x = S._make_x(mesh, 2, tokens, S.H)
    gating = S._put(mesh, S._host_gating(2, tokens, S.E), P(MESH_AXIS))
    base = Config.for_shape(hidden=S.H, inter=S.INTER, experts_per_shard=S.E // n, top_k=S.TOPK)
    variants = {
        "v4": _v3_fn(mesh, dataclasses.replace(base, **V4), n),
        "v4_bf16res": _v3_fn(mesh, dataclasses.replace(base, **V4, result_rows="bf16"), n),
    }
    for fused in FUSED_ARMS:
        for pt in PT_ARMS:
            for tile in TILES:
                for mg in MG_ARMS:
                    for up in UP_ARMS:
                        name = ("dense" + ("_fused" if fused else "") + ("_pt" if pt else "") +
                                ("_mg" if mg else "") + ("_up" if up else "") +
                                ("" if tile == S.TILE_ROWS else f"_t{tile}"))
                        variants[name] = _ds_fn(mesh, base, n, fused, pt, tile, mg, up)
    args = (gating, w1, w2, s1, s2)
    for r in range(REPEATS):
        for name, fn in variants.items():
            row = {"label": "ds_time", "variant": name, "tokens": tokens, "round": r,
                   "per_shard": tokens // n, "bridge": BRIDGE}
            try:
                t0 = time.perf_counter()
                jax.block_until_ready(fn(x, *args))
                row["first_call_s"] = round(time.perf_counter() - t0, 1)
                med, best, _ = S._loop_time(S._chained(fn), x, args, S.REPS, S.ITERS, chain=True)
                row["us_per_layer_median"] = round(med * 1e6, 1)
                row["us_per_layer_min"] = round(best * 1e6, 1)
                b2b, b2b_min, out = S._loop_time(fn, x, args, S.REPS, S.ITERS, chain=False)
                row["us_b2b_median"] = round(b2b * 1e6, 1)
                row["us_b2b_min"] = round(b2b_min * 1e6, 1)
                row["out_checksum"] = _checksum(out)
                if os.environ.get("P3_DS_TRACE") == "1" and r == 0 and tokens == 448:
                    tdir = os.path.join(S.OUT, "trace", f"{name}_t{tokens}")
                    step = S._chained(fn)
                    jax.profiler.start_trace(tdir)
                    y = x
                    for _ in range(5):
                        y = step(y, *args)
                    jax.block_until_ready(y)
                    jax.profiler.stop_trace()
                    row["trace"] = tdir
            except Exception as e:  # a refusal is a result
                row["error"] = f"{type(e).__name__}: {str(e)[:600]}"
            _emit(row)
