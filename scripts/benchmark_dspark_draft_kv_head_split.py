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
"""Microbenchmark: MiniMax-M3.1 DSpark draft KV under DCP, current sharded
layout (`cp_draft_attention`) vs the head-split layout
(`cp_draft_head_split`, additional_config draft_kv_head_split).

Served setup: TP8 on the 8 cores of a 2x2x1 slice, 4 draft KV heads (2 TP
replicas each), 8 query heads per rank, 5 draft layers, fp8 draft KV,
4096-token window, block of 7. DCP 2 (4 groups of 2 ranks, one KV head
each) or DCP 8 (one group, 4 KV heads). One `shard_map` over a
``(lane, outer, head) = (8 / dcp, dcp / 2, 2)`` mesh: TP collectives span
all three axes, the DCP group is ``(outer, head)``, a KV head's ranks are
``head``.

Per decode step, two parts that the layout changes (the rest of the
drafter, its matmuls / LM head / Markov loop, is the same in both):

  write   the target-context K/V write of all 5 draft layers, ``T_ctx``
          tokens (decode: R x 8 accepted+bonus rows; ``--mixed`` adds a
          prefill-chunk-sized write).
          old = served rank-rows path (`store_context_kv_rank_rows`): fc +
          norm on T/dcp rows, per layer one K/V matmul for all the group's
          KV heads, K norm + RoPE, `pack_rows`, one all-gather of every
          layer's packed rows over the DCP group, per layer
          `write_owned_rows`.
          new = column-parallel fc (+ TP all-gather) on all rows, per layer
          a K/V matmul of the own KV head, K norm + RoPE, one
          `write_layers` trace (`write_kv` per layer, no collective).
  attn    the 5 draft layers' context attention: query all-gather, window
          attention over the cache, LSE merge (`cp_lse_ag_out_rs`), block
          attention + merge. old = gather/merge over the DCP group,
          `attend_cached_flash` with the group's KV heads; new = over the
          KV head's 2 ranks, `attend_cached` (`attend_pages`).

Per-iteration time is the slope ``(t(2L) - t(L)) / L`` of a `fori_loop`
whose iterations feed each other (cache carried, outputs folded into the
next query), compiled once per (cell, part, layout).

Usage (TPU, 8 cores):
  PYTHONPATH=<worktree>/src python3 benchmark_dspark_draft_kv_head_split.py \\
      --dcp 2,8 --reqs 1,16,64 --mixed 2048 --json /cdk-outputs/outputs/hs.json
CPU (correctness of the script only; interpreted kernels, tiny shapes):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
      PYTHONPATH=<worktree>/src python3 benchmark_dspark_draft_kv_head_split.py --smoke
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu  # noqa: F401  (served libtpu defaults before jax init)
from vllm_torchtpu.layers.adapter import cp_draft_attention as cda
from vllm_torchtpu.layers.adapter import cp_draft_head_split as hs

TP, KV_HEADS, QH, HD = 8, 4, 8, 128
REPLICAS = TP // KV_HEADS
HID, NUM_AUX, LAYERS, K, PAGE, WINDOW = 6144, 5, 5, 7, 128, 4096
K_SCALE, V_SCALE = 0.7, 0.3
SM_SCALE = HD**-0.5
AXES = ("l", "o", "h")
SMOKE = False


def _dcp_rank():
    return lax.axis_index("o") * 2 + lax.axis_index("h")


def _ag(x, axes, dim):
    """vLLM GroupCoordinator.all_gather(x, dim) over `axes` (rank-major)."""
    g = lax.all_gather(x, axes, axis=0, tiled=False)
    g = jnp.moveaxis(g, 0, dim)
    return g.reshape(x.shape[:dim] + (-1,) + x.shape[dim + 1 :])


def _merge(out, lse, axes, n):
    """`dcp_comm.cp_lse_ag_out_rs` over `axes` (n ranks), return_lse."""
    if n == 1:
        return out, lse
    lse = lse.astype(jnp.float32)
    lses = lax.all_gather(lse, axes, axis=0, tiled=False)
    merged_lse = jax.nn.logsumexp(lses, axis=0)
    weight = jnp.exp(lse - merged_lse)[..., None]
    weighted = jnp.where(weight > 0, out.astype(jnp.float32) * weight, 0.0)
    merged = lax.psum_scatter(weighted, axes, scatter_dimension=1, tiled=True)
    heads = merged.shape[1]
    idx = _dcp_rank() if axes == ("o", "h") else lax.axis_index("h")
    return merged.astype(out.dtype), lax.dynamic_slice_in_dim(
        merged_lse, idx * heads, heads, 1
    )


def _rms(x, w):
    xf = x.astype(jnp.float32)
    return (
        xf * lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6) * (1 + w)
    ).astype(jnp.bfloat16)


def _rope(k, pos):
    """Partial NeoX RoPE on the first half of each head (elementwise cost)."""
    rot = HD // 2
    freq = 1.0 / (10000.0 ** (jnp.arange(rot // 2) / (rot // 2)))
    ang = pos[:, None].astype(jnp.float32) * freq[None, :]
    cos, sin = jnp.cos(ang)[:, None, :], jnp.sin(ang)[:, None, :]
    kf = k.astype(jnp.float32)
    a, b, rest = kf[..., : rot // 2], kf[..., rot // 2 : rot], kf[..., rot:]
    return jnp.concatenate([a * cos - b * sin, b * cos + a * sin, rest], -1).astype(
        k.dtype
    )


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def make_meta(rng, num_reqs, q_len, dcp, num_pages):
    """Per-request lengths (long contexts) and shuffled page tables."""
    lo, hi = (300, 3000) if SMOKE else (8192, 120000)
    ctx = rng.integers(lo, hi, num_reqs).astype(np.int32)
    q = np.full(num_reqs, q_len, np.int32)
    logical = int(-(-(ctx.max() + q_len) // (PAGE * dcp))) + 2
    perm = rng.permutation(num_pages - 1)
    tables = np.stack([np.roll(perm, -i * logical)[:logical] for i in range(num_reqs)])
    cu = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
    return ctx + q, tables.astype(np.int32).reshape(-1), cu


def _gen(mesh, shape, dtype, seed, scale=0.02, fp8=False):
    sh = NamedSharding(mesh, P(AXES))

    def gen(key):
        if fp8:  # random e4m3 bits without NaN codes
            bits = jax.random.bits(key, (TP,) + shape, jnp.uint8)
            return lax.bitcast_convert_type(bits & jnp.uint8(0x77), jnp.float8_e4m3fn)
        return (jax.random.normal(key, (TP,) + shape, jnp.float32) * scale).astype(
            dtype
        )

    return jax.jit(gen, out_shardings=sh)(jax.random.PRNGKey(seed))


def _rep(mesh, arr):
    return jax.device_put(
        np.broadcast_to(arr, (TP,) + arr.shape).copy(), NamedSharding(mesh, P(AXES))
    )


# ----------------------------------------------------------------------------
# Parts (per device, inside shard_map; leading device axis of size 1 removed)
# ----------------------------------------------------------------------------
def write_old(p, caches, aux, pos, meta, dcp):
    """Served rank-rows context K/V write (`store_context_kv_rank_rows`)."""
    seq_lens, tables, cu = meta
    T = aux.shape[0]
    n = -(-T // dcp)
    rows = jnp.minimum(_dcp_rank() * n + jnp.arange(n), T - 1)
    context = _rms(
        jnp.dot(aux[rows], p["fc"], preferred_element_type=jnp.float32), p["hn"]
    )
    pr = pos[rows]
    g = dcp // REPLICAS
    packed = []
    for i in range(LAYERS):
        kv = jnp.dot(context, p["wkv_group"][i], preferred_element_type=jnp.float32)
        kv = kv.astype(jnp.bfloat16).reshape(n, 2, g, HD)
        k = _rope(_rms(kv[:, 0], p["kn"][i]), pr)
        packed.append(
            cda.pack_rows(
                k,
                kv[:, 1],
                caches[i].shape[2:],
                caches[i].dtype,
                k_scale=K_SCALE,
                v_scale=V_SCALE,
            )
        )
    stacked = jnp.stack(packed, axis=1)  # [n, L, ...]
    gathered = _ag(stacked, ("o", "h"), 0)[:T]  # [T, L, ...] rank-major
    return tuple(
        cda.write_owned_rows(
            caches[i],
            gathered[:, i],
            seq_lens,
            tables,
            cu,
            dcp=dcp,
            rank=_dcp_rank(),
            interpret=SMOKE,
        )
        for i in range(LAYERS)
    )


def write_new(p, caches, aux, pos, meta, dcp):
    """Head-split context K/V write: column-parallel fc, own-head K/V of all
    rows, one `write_layers` trace."""
    seq_lens, tables, cu = meta
    tp_rank = lax.axis_index(AXES)
    cols = HID // TP
    w = lax.dynamic_slice_in_dim(p["fc"], tp_rank * cols, cols, axis=1)
    part = jnp.dot(aux, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    context = _rms(_ag(part, AXES, 1), p["hn"])
    keys, values = [], []
    for i in range(LAYERS):
        kv = jnp.dot(context, p["wkv_own"][i], preferred_element_type=jnp.float32)
        kv = kv.astype(jnp.bfloat16).reshape(-1, 2, 1, HD)
        keys.append(_rope(_rms(kv[:, 0], p["kn"][i]), pos))
        values.append(kv[:, 1])
    return hs.write_layers(
        caches,
        jnp.stack(keys),
        jnp.stack(values),
        seq_lens,
        tables,
        cu,
        dcp=dcp,
        head_ranks=min(dcp, REPLICAS),
        rank=_dcp_rank(),
        k_scale=K_SCALE,
        v_scale=V_SCALE,
        interpret=SMOKE,
    )


def attn_layers(p, caches, q, kv_blk, meta, dcp, layout):
    """The draft layers' context + block attention (per layer: q gather,
    window attention, merge, block attention + merge)."""
    seq_lens, tables, cu = meta
    T = q.shape[0]
    if layout == "old":
        axes, n = ("o", "h"), dcp
    else:
        axes, n = ("h",), min(dcp, REPLICAS)
    outs = []
    x = q
    for i in range(LAYERS):
        q_all = _ag(x, axes, 1) if n > 1 else x
        if layout == "old":
            o, l = cda.attend_cached_flash(
                q_all,
                caches[i],
                seq_lens,
                tables,
                cu,
                kv_heads=dcp // REPLICAS,
                max_query_len=K,
                sliding_window=WINDOW,
                sm_scale=SM_SCALE,
                dcp=dcp,
                rank=_dcp_rank(),
                k_scale=K_SCALE,
                v_scale=V_SCALE,
                interpret=SMOKE,
            )
        else:
            o, l = hs.attend_cached(
                q_all,
                caches[i],
                seq_lens,
                tables,
                cu,
                head_ranks=n,
                max_query_len=K,
                sliding_window=WINDOW,
                sm_scale=SM_SCALE,
                dcp=dcp,
                rank=_dcp_rank(),
                k_scale=K_SCALE,
                v_scale=V_SCALE,
                interpret=SMOKE,
            )
        mo, ml = _merge(o, l, axes, n)
        bo, bl = cda.attend_block(
            x,
            kv_blk[0],
            kv_blk[1],
            seq_lens,
            cu,
            max_query_len=K,
            sm_scale=SM_SCALE,
            causal=False,
        )
        out = cda.merge(mo.astype(bo.dtype), ml, bo, bl)[0]
        outs.append(out)
        x = (x.astype(jnp.float32) + 1e-3 * out.astype(jnp.float32)).astype(x.dtype)
    return x.reshape(T, QH, HD)


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def _time(fn, args, reps):
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e6


def _loop_fn(mesh, body, iters):
    def per_device(*args):
        args = [a[0] for a in args]  # drop the device axis

        def step(_, carry):
            return body(carry, args)

        out = lax.fori_loop(0, iters, step, body.init(args))
        return jax.tree.map(lambda a: a[None], out)

    spec = P(AXES)
    return jax.jit(
        jax.shard_map(
            per_device, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
        )
    )


def slope(mesh, body, args, loops, reps):
    t1 = _time(_loop_fn(mesh, body, loops), args, reps)
    t2 = _time(_loop_fn(mesh, body, 2 * loops), args, reps)
    return (t2 - t1) / loops


class _Body:
    """fori_loop body with an `init(args)` for the carry."""

    def __init__(self, fn, init):
        self.fn, self.init = fn, init

    def __call__(self, carry, args):
        return self.fn(carry, args)


def run_cell(mesh, dcp, num_reqs, t_ctx, label, loops, reps, rng):
    g = dcp // REPLICAS
    words = -(-2 * g // 4)
    num_pages = 2048 if SMOKE else 16384
    t_draft = num_reqs * K
    p = dict(
        fc=_gen(mesh, (NUM_AUX * HID if not SMOKE else HID, HID), jnp.bfloat16, 1),
        hn=_gen(mesh, (HID,), jnp.float32, 2, 0.1),
        kn=_gen(mesh, (LAYERS, HD), jnp.float32, 3, 0.1),
        wkv_group=_gen(mesh, (LAYERS, HID, 2 * g * HD), jnp.bfloat16, 4),
        wkv_own=_gen(mesh, (LAYERS, HID, 2 * HD), jnp.bfloat16, 5),
    )
    caches = tuple(
        _gen(mesh, (num_pages, PAGE, words, 4, HD), jnp.float8_e4m3fn, 10 + i, fp8=True)
        for i in range(LAYERS)
    )
    aux = _gen(mesh, (t_ctx, p["fc"].shape[1]), jnp.bfloat16, 20, 1.0)
    # Context write metadata: R requests sharing t_ctx tokens.
    q_ctx = max(1, t_ctx // num_reqs)
    sl_c, bt_c, cu_c = make_meta(rng, num_reqs, q_ctx, dcp, num_pages)
    cu_c = np.minimum(cu_c, t_ctx).astype(np.int32)
    pos = _rep(mesh, np.arange(t_ctx, dtype=np.int32) + 9000)
    meta_c = tuple(_rep(mesh, a) for a in (sl_c, bt_c, cu_c))
    # Draft metadata: R requests of 7 rows.
    sl_d, bt_d, cu_d = make_meta(rng, num_reqs, K, dcp, num_pages)
    meta_d = tuple(_rep(mesh, a) for a in (sl_d, bt_d, cu_d))
    q = _gen(mesh, (t_draft, QH, HD), jnp.bfloat16, 30, 1.0)
    kb = _gen(mesh, (t_draft, 1, HD), jnp.bfloat16, 31, 1.0)
    vb = _gen(mesh, (t_draft, 1, HD), jnp.bfloat16, 32, 1.0)
    pc = dict(fc=p["fc"], hn=p["hn"], kn=p["kn"])
    res = []
    for layout in ("old", "new"):
        wkv = p["wkv_group"] if layout == "old" else p["wkv_own"]
        write = write_old if layout == "old" else write_new

        def wbody(carry, a, write=write):
            fc, hn, kn, wk, ax, ps, sl, bt, cu, *cs = a
            prm = dict(fc=fc, hn=hn, kn=kn, wkv_group=wk, wkv_own=wk)
            return write(prm, carry, ax, ps, (sl, bt, cu), dcp)

        wargs = (pc["fc"], pc["hn"], pc["kn"], wkv, aux, pos, *meta_c, *caches)
        body = _Body(wbody, lambda a: tuple(a[9:]))
        t_write = slope(mesh, body, wargs, loops, reps)

        def abody(carry, a, layout=layout):
            qq, k_, v_, sl, bt, cu, *cs = a
            return attn_layers(None, cs, carry, (k_, v_), (sl, bt, cu), dcp, layout)

        aargs = (q, kb, vb, *meta_d, *caches)
        body = _Body(abody, lambda a: a[0])
        t_attn = slope(mesh, body, aargs, loops, reps)
        rec = dict(
            cell=label,
            dcp=dcp,
            reqs=num_reqs,
            t_ctx=t_ctx,
            t_draft=t_draft,
            layout=layout,
            write_us=round(t_write, 1),
            attn_us=round(t_attn, 1),
            total_us=round(t_write + t_attn, 1),
        )
        print("BENCH " + json.dumps(rec), flush=True)
        res.append(rec)
    old, new = res
    print(
        f"{label} DCP{dcp} R={num_reqs} T_ctx={t_ctx}: write {old['write_us']} -> "
        f"{new['write_us']} us, attn(5L) {old['attn_us']} -> {new['attn_us']} us, "
        f"total {old['total_us']} -> {new['total_us']} us "
        f"({new['total_us'] - old['total_us']:+.1f})",
        flush=True,
    )
    return res


def main():
    global SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", default="2,8")
    ap.add_argument("--reqs", default="1,16,64")
    ap.add_argument(
        "--ctx-per-req", type=int, default=8, help="decode context rows per request"
    )
    ap.add_argument(
        "--mixed", default="", help="extra write sizes, e.g. 2048,8192 (at R=64)"
    )
    ap.add_argument("--loops", type=int, default=20)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--json", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    SMOKE = a.smoke
    if SMOKE:
        global HID, WINDOW, NUM_AUX
        HID, WINDOW, NUM_AUX = 256, 512, 1
        a.loops, a.reps = 1, 1
        a.reqs = "1"
    devices = np.asarray(jax.devices()[:TP])
    rng = np.random.default_rng(0)
    out = []
    for dcp in [int(x) for x in a.dcp.split(",")]:
        mesh = Mesh(devices.reshape(TP // dcp, dcp // 2, 2), AXES)
        cells = [(int(r), int(r) * a.ctx_per_req, "decode") for r in a.reqs.split(",")]
        cells += [
            (64 if not SMOKE else 4, int(t), "mixed") for t in a.mixed.split(",") if t
        ]
        for num_reqs, t_ctx, label in cells:
            t_ctx = -(-t_ctx // 16) * 16
            out += run_cell(mesh, dcp, num_reqs, t_ctx, label, a.loops, a.reps, rng)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
