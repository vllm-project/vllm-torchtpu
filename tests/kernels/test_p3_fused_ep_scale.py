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
"""Numerics and per-layer time of the fused EP v3 MoE layer at 8, 16 and 32 shards (p3 harness).

One mesh axis over every TensorCore of the slice (8 on a tpu7x 2x2x1 host, 16 on 2x2x2, 32 on
2x2x4), devices in ascending id order as `distributed.ep_mesh.build_ep_mesh` orders them. A
multi-host slice runs as one SPMD JAX program: every host runs the same pytest and JAX
distributed starts from JAX_COORDINATOR_ADDRESS / JAX_NUM_PROCESSES / JAX_PROCESS_ID.

The layer is called the way `layers/adapter/fused_moe_ep_v3.run_moe_v3` calls it: fp8 weights
with per-output-channel scales, `tile_rows=128`, silu, renormalized top-k on the router logits,
`Config.for_shape` (what `served_config` returns). Qwen3.5-397B geometry: hidden 4096,
intermediate 1024, 512 experts, top-k 10. Router logits are standard normal, as in the kernel's
own end-to-end test (tests/kernels/fused_moe/test_fused_ep_v3.py, e1dbd90f6 on
hning/fused-ep-v3-fulladapter), whose dense f32 reference, relative-L2 bands (batch 0.06, worst
token 0.075) and rotated-expert control (10x) are copied here with the shard count as a
parameter.

Per token count (P3_FEP_TOKENS, global tokens per call; P-leg steps are 8192 rows a host):

  * numerics against the dense reference and the rotated control;
  * a chained loop: each call's input is the RMS-normed residual sum of the previous call's
    input and output (as between two decoder layers; the router logits stay fixed, so every call
    routes the same), REPS calls per iteration from the same start, the median over ITERS
    iterations of wall time per call; the norm is one fused elementwise op per call, which the
    xprof trace separates from the layer's own programs. A second loop issues REPS calls on the
    same input back to back (phase 2 D2's protocol, test_fused_ep_dma_bench.py);
  * the exchange bytes of that routing: the dispatch is an all-gather of every shard's fp8 token
    rows (transport.start_transport), so a shard receives (N - 1) * tokens_per_shard rows; the
    combine sends one fp8 result row per (token, expert) pair whose expert is on another shard;
    GB/s is those bytes over the per-call time;
  * an xprof trace of a short chain for the kernels' device time
    (scripts/vllm/benchmark/p3_xplane_device_time.py, or fused_ep_trace_ops.py of phase 2 D2).

Collected only with P3_BENCH=1. Results: <P3_FEP_OUT or run_dir>/results.jsonl and
FUSED_EP_SCALE lines on stdout (host 0).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import pytest

if not os.environ.get("P3_BENCH"):
    pytest.skip("phase 3 harness bench; set P3_BENCH=1 to run", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v3 import (  # noqa: E402
    MESH_AXIS,
    Config,
    WeightFormat,
    fused_ep_moe,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "vllm", "benchmark"))
import p3_bench_common as common  # noqa: E402

H, INTER, E, TOPK = 4096, 1024, 512, 10
TILE_ROWS = 128  # layers/adapter/fused_moe_ep_v3.TILE_ROWS
FP8 = jnp.float8_e4m3fn
# fp8 rows plus one f32 scale per row, on the dispatch and on the combine (Config defaults
# token_rows="fp8", result_rows="fp8")
ROW_BYTES = H + 4

FP8_RELATIVE_BOUND = 0.06
FP8_TOKEN_BOUND = 0.075
ROTATED_CONTROL_FACTOR = 10

TOKENS = [int(t) for t in os.environ.get("P3_FEP_TOKENS", "8192,16384,32768").split(",") if t]
MAX_PER_SHARD = int(os.environ.get("P3_FEP_MAX_PER_SHARD", "2048"))
REPS = int(os.environ.get("P3_FEP_REPS", "20"))
ITERS = int(os.environ.get("P3_FEP_ITERS", "5"))
TRACE = os.environ.get("P3_FEP_TRACE", "1") == "1"
OUT = os.environ.get("P3_FEP_OUT") or common.run_dir("fused_ep_scale")

if os.path.isdir("/workspace/tmp"):
    jax.config.update("jax_compilation_cache_dir", "/workspace/tmp/p3-harness/jax-cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _init_distributed():
    if jax.distributed.is_initialized():
        return
    addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
    nproc = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
    if addr and nproc > 1:
        jax.distributed.initialize(
            coordinator_address=addr,
            num_processes=nproc,
            process_id=int(os.environ.get("JAX_PROCESS_ID", "0")),
        )


_MESH = []


def _mesh() -> Mesh:
    if not _MESH:
        _init_distributed()
        devs = sorted((d for d in jax.devices() if d.platform == "tpu"), key=lambda d: d.id)
        n = int(os.environ.get("P3_FEP_SHARDS", "0")) or len(devs)
        if len(devs) < n or n < 8:
            pytest.skip(f"needs {max(n, 8)} TPU devices, have {len(devs)}")
        _MESH.append(Mesh(np.asarray(devs[:n]), axis_names=(MESH_AXIS,)))
    return _MESH[0]


def _shards(mesh) -> int:
    return mesh.shape[MESH_AXIS]


_ENV: dict = {}


def _record(row: dict) -> None:
    lead = jax.process_index() == 0
    if not _ENV:
        os.makedirs(OUT, exist_ok=True)
        if lead:
            _ENV.update(common.write_env(OUT, {"bench": "fused_ep_scale"}, devices=True))
        else:
            _ENV["id"] = f"process{jax.process_index()}"
    row = {"environment_id": _ENV["id"], "process_count": jax.process_count(), **row}
    if lead:
        with open(os.path.join(OUT, "results.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    print("FUSED_EP_SCALE " + json.dumps(row), flush=True)


def _make_weights(mesh, seed, *, hidden, inter, e_total, weight_format):
    """(w1, w2, w1_scale, w2_scale) sharded on the experts, drawn on their own shards.

    fp8 scales are per output channel over each matmul's contraction axis: w1_scale
    [E, 2 * inter], w2_scale [E, hidden], the layout the kernel takes."""
    quantized = weight_format == WeightFormat.FP8
    sh = NamedSharding(mesh, P(MESH_AXIS))

    def build():
        k1, k2 = jax.random.split(jax.random.key(seed))
        out = []
        for key, shape in ((k1, (e_total, hidden, 2 * inter)), (k2, (e_total, inter, hidden))):
            w = jax.random.normal(key, shape, jnp.float32) / 10
            w = lax.with_sharding_constraint(w, sh)
            if not quantized:
                out.append((w.astype(jnp.bfloat16), None))
                continue
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
            out.append(((w / scale).astype(FP8), scale.reshape(shape[0], shape[2])))
        (w1, s1), (w2, s2) = out
        return (w1, w2, s1, s2) if quantized else (w1, w2)

    n_out = 4 if quantized else 2
    built = jax.jit(build, out_shardings=(sh,) * n_out)()
    jax.block_until_ready(built)
    return built if quantized else (built[0], built[1], None, None)


def _make_x(mesh, seed, tokens, hidden):
    sh = NamedSharding(mesh, P(MESH_AXIS))

    def build():
        x = jax.random.normal(jax.random.key(seed), (tokens, hidden), jnp.float32) / 10
        return x.astype(jnp.bfloat16)

    return jax.jit(build, out_shardings=sh)()


def _host_gating(seed, tokens, e_total):
    """Router logits drawn on the host (the same on every host), for the routing accounting."""
    return np.random.default_rng(seed).standard_normal((tokens, e_total), dtype=np.float32)


def _put(mesh, x, spec):
    return jax.make_array_from_callback(x.shape, NamedSharding(mesh, spec), lambda i: x[i])


def _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, gating, *, topk):
    """The plain MoE in f32: softmax, top-k, every expert on every token weighted by its
    renormalized router weight (zero where not selected), summed over the expert shards."""
    g_local = w1.shape[0] // _shards(mesh)
    inter = w1.shape[2] // 2
    scaled = w1_scale is not None

    def local(x_g, gating_g, w1_l, w2_l, *scales):
        s1, s2 = scales if scaled else (None, None)
        scores = jax.nn.softmax(gating_g.astype(jnp.float32), axis=-1)
        topk_weights, topk_idx = lax.top_k(scores, topk)
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
        first = lax.axis_index(MESH_AXIS) * g_local
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0, keepdims=False).astype(jnp.float32)
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0, keepdims=False).astype(jnp.float32)
            if scaled:
                w1e = w1e * lax.dynamic_index_in_dim(s1, e, 0, keepdims=True)
                w2e = w2e * lax.dynamic_index_in_dim(s2, e, 0, keepdims=True)
            weight = jnp.sum(jnp.where(topk_idx == first + e, topk_weights, 0.0), axis=-1)[:, None]
            acc1 = x32 @ w1e
            row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
            return acc + weight * row

        mine = lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32))
        return lax.psum(mine, MESH_AXIS)

    ax = P(MESH_AXIS)
    in_specs = (P(), P(), ax, ax) + ((ax, ax) if scaled else ())
    args = (x, gating, w1, w2) + ((w1_scale, w2_scale) if scaled else ())
    return jax.jit(
        jax.shard_map(local, mesh=mesh, in_specs=in_specs, out_specs=P(), check_vma=False)
    )(*args)


@jax.jit
def _errors(out, want, rotated):
    """(batch relative L2, worst-token relative L2, relative L2 vs the rotated reference, finite)."""
    a = out.astype(jnp.float32)
    d = jnp.sqrt(jnp.sum((a - want) ** 2, axis=-1))
    n = jnp.sqrt(jnp.sum(want**2, axis=-1))
    batch = jnp.sqrt(jnp.sum(d**2)) / jnp.sqrt(jnp.sum(n**2))
    worst = jnp.max(d / jnp.where(n == 0, 1.0, n))
    rot = jnp.sqrt(jnp.sum((a - rotated) ** 2)) / jnp.sqrt(jnp.sum(rotated**2))
    return batch, worst, rot, jnp.all(jnp.isfinite(a))


def _scalar(x):
    return x.addressable_data(0).item()


def _layer_fn(mesh, *, topk, fmt, config=None, experts_per_shard=None):
    def run(x, gating, w1, w2, *scales):
        s1, s2 = scales if scales else (None, None)
        return fused_ep_moe(
            x, w1, w2, s1, s2, gating,
            top_k=topk,
            renormalize=True,
            mesh=mesh,
            tile_rows=TILE_ROWS,
            weight_format=fmt,
            activation="silu",
            config=config,
            experts_per_shard=experts_per_shard,
        )  # fmt: skip

    return jax.jit(run)


def _check(mesh, fn, x, w1, w2, s1, s2, gating, topk):
    scales = () if s1 is None else (s1, s2)
    out = fn(x, gating, w1, w2, *scales)
    want = _dense_reference(mesh, x, w1, w2, s1, s2, gating, topk=topk)

    def roll(a):
        return None if a is None else jnp.roll(a, 1, axis=0)

    rotated = _dense_reference(mesh, x, roll(w1), roll(w2), roll(s1), roll(s2), gating, topk=topk)
    batch, worst, rot, finite = (_scalar(v) for v in _errors(out, want, rotated))
    ok = (bool(finite) and out.shape == x.shape and out.dtype == x.dtype
          and batch < FP8_RELATIVE_BOUND and worst < FP8_TOKEN_BOUND
          and rot > ROTATED_CONTROL_FACTOR * batch)
    return {"rel_l2": round(batch, 5), "worst_token_rel_l2": round(worst, 5),
            "rotated_rel_l2": round(rot, 4), "finite": bool(finite), "numerics_pass": ok}


def _routing_bytes(gating, n_shards, topk):
    """Exchange bytes per shard of this routing: dispatch rows received (all-gather) and
    combine rows sent (pairs whose expert is on another shard), mean and max over shards."""
    tokens, e_total = gating.shape
    t_local, e_local = tokens // n_shards, e_total // n_shards
    idx = np.argpartition(-gating, topk - 1, axis=1)[:, :topk]
    exp_shard = idx // e_local
    tok_shard = (np.arange(tokens) // t_local)[:, None]
    cross = exp_shard != tok_shard
    sent = np.bincount(exp_shard[cross], minlength=n_shards)
    pairs = np.bincount(exp_shard.ravel(), minlength=n_shards)
    dispatch = (n_shards - 1) * t_local * ROW_BYTES
    return {
        "tokens_per_shard": t_local,
        "experts_per_shard": e_local,
        "pairs_per_shard_mean": float(pairs.mean()),
        "pairs_per_shard_max": int(pairs.max()),
        "cross_pairs_frac": float(cross.mean()),
        "dispatch_recv_MB_per_shard": round(dispatch / 1e6, 3),
        "combine_send_MB_per_shard_mean": round(float(sent.mean()) * ROW_BYTES / 1e6, 3),
        "combine_send_MB_per_shard_max": round(float(sent.max()) * ROW_BYTES / 1e6, 3),
    }


def _chained(fn):
    """One layer of the chain: the layer, then the RMS-normed residual sum (std 0.1, the
    inputs' scale) as the next layer's input."""

    @jax.jit
    def step(x, *args):
        y = x.astype(jnp.float32) + fn(x, *args).astype(jnp.float32)
        y = y * lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + 1e-6) * 0.1
        return y.astype(x.dtype)

    return step


def _loop_time(fn, x0, args, reps, iters, chain):
    """Median and min wall seconds per call over iters loops of reps calls; chain feeds each
    call the previous call's result, otherwise every call takes x0."""
    out = fn(x0, *args)
    jax.block_until_ready(out)
    times = []
    for _ in range(iters):
        out = x0
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn(out if chain else x0, *args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) / reps)
    return statistics.median(times), min(times), out


def test_devices():
    mesh = _mesh()
    devs = [{"id": d.id, "coords": list(getattr(d, "coords", ())),
             "core": getattr(d, "core_on_chip", None), "process": d.process_index}
            for d in mesh.devices.flat]
    _record({"label": "devices", "shards": _shards(mesh), "mesh": devs, "jax": jax.__version__})


@pytest.mark.parametrize("fmt", ["fp8", "bf16"])
def test_small_shape(fmt):
    """The kernel test's small structurally complete shape, scaled to the shard count: 4 experts
    and 32 tokens a shard, hidden 1024, intermediate 256, top-k 4."""
    mesh = _mesh()
    n = _shards(mesh)
    wf = WeightFormat.FP8 if fmt == "fp8" else WeightFormat.BF16
    e_total, tokens, hidden, inter, topk = 4 * n, 32 * n, 1024, 256, 4
    w1, w2, s1, s2 = _make_weights(mesh, 0, hidden=hidden, inter=inter, e_total=e_total,
                                   weight_format=wf)
    x = _make_x(mesh, 1, tokens, hidden)
    gating = _put(mesh, _host_gating(1, tokens, e_total), P(MESH_AXIS))
    fn = _layer_fn(mesh, topk=topk, fmt=fmt)
    row = {"label": "small", "fmt": fmt, "shards": n, "tokens": tokens, "e_total": e_total,
           **_check(mesh, fn, x, w1, w2, s1, s2, gating, topk)}
    _record(row)
    assert row["numerics_pass"], row


_PROD = []


def _prod_weights(mesh):
    if not _PROD:
        _PROD.append(_make_weights(mesh, 7, hidden=H, inter=INTER, e_total=E,
                                   weight_format=WeightFormat.FP8))
    return _PROD[0]


@pytest.mark.parametrize("tokens", TOKENS)
def test_prod(tokens):
    """Qwen3.5-397B's MoE layer at `tokens` global rows per call: numerics, chained per-call
    time, exchange bytes and an xprof trace."""
    mesh = _mesh()
    n = _shards(mesh)
    if tokens % n or tokens // n > MAX_PER_SHARD or (tokens // n) % 8:
        pytest.skip(f"{tokens} tokens over {n} shards is outside the sweep")
    w1, w2, s1, s2 = _prod_weights(mesh)
    cfg = Config.for_shape(hidden=H, inter=INTER, experts_per_shard=E // n, top_k=TOPK)
    fn = _layer_fn(mesh, topk=TOPK, fmt="fp8", config=cfg, experts_per_shard=E // n)
    x = _make_x(mesh, 2, tokens, H)
    g_host = _host_gating(2, tokens, E)
    gating = _put(mesh, g_host, P(MESH_AXIS))
    row = {"label": "prod", "shards": n, "hosts": jax.process_count(), "tokens": tokens}
    t0 = time.perf_counter()
    row.update(_check(mesh, fn, x, w1, w2, s1, s2, gating, TOPK))
    row["first_call_and_reference_s"] = round(time.perf_counter() - t0, 1)
    row.update(_routing_bytes(g_host, n, TOPK))
    args = (gating, w1, w2, s1, s2)
    step = _chained(fn)
    med, best, out = _loop_time(step, x, args, REPS, ITERS, chain=True)
    row["chain_finite"] = bool(_scalar(jax.jit(lambda a: jnp.all(jnp.isfinite(
        a.astype(jnp.float32))))(out)))
    row["us_per_layer_median"] = round(med * 1e6, 1)
    row["us_per_layer_min"] = round(best * 1e6, 1)
    b2b, b2b_min, _ = _loop_time(fn, x, args, REPS, ITERS, chain=False)
    row["us_per_call_back_to_back_median"] = round(b2b * 1e6, 1)
    row["us_per_call_back_to_back_min"] = round(b2b_min * 1e6, 1)
    xbytes = (row["dispatch_recv_MB_per_shard"] + row["combine_send_MB_per_shard_mean"]) * 1e6
    row["exchange_GBps_per_shard"] = round(xbytes / med / 1e9, 1)
    row["weights_MB_per_shard"] = round((E // n) * (H * 2 * INTER + INTER * H) / 1e6, 1)
    if TRACE:
        proc = f"_p{jax.process_index()}" if jax.process_count() > 1 else ""
        tdir = os.path.join(OUT, "trace", f"n{n}_t{tokens}{proc}")
        jax.profiler.start_trace(tdir)
        out = x
        for _ in range(5):
            out = step(out, *args)
        jax.block_until_ready(out)
        jax.profiler.stop_trace()
        row["trace"] = tdir
    _record(row)
    assert row["numerics_pass"] and row["chain_finite"], row
