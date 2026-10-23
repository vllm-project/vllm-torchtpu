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
"""On-TPU cost of the GDN pool gather / scatter around a Qwen3.8 segment call.

Times, per rows bucket, (a) the composed call (pad, block ids, pool gather,
kernel, scatter, unpad) and (b) the bare segment kernel with the same operands,
as device-side chained loops whose carry is the residual (M-sweep slope).
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import pack_segment, random_segment_inputs
from vllm_torchtpu.megakernel.models.qwen38_step import make_flat_segment_step
from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import (
    T_VERIFY,
    build_segment_call,
    segment_tiles,
)


def _slope(step, carry, args, chains):
    """Per-call seconds of `carry = step(carry, *args)` chained `n` times on device.

    Operands are jit arguments, never closure constants (constants get copied).
    """
    best = {}
    for n in chains:
        f = jax.jit(
            lambda c, a, n=n: jax.lax.fori_loop(0, n, lambda _, x: step(x, *a), c)
        )
        jax.block_until_ready(f(carry, args))
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            jax.block_until_ready(f(carry, args))
            ts.append(time.perf_counter() - t)
        best[n] = min(ts)
    k, _ = np.polyfit(np.array(list(best), float), np.array(list(best.values()), float), 1)
    return round(k * 1e6, 1)


def run(
    B: int, L: int, chains: list[int], rings: tuple[int, int] | None = None, kernel_only: bool = False
) -> dict:
    g, R = qwen38.FULL, 8
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    NS, NL = B // T_VERIFY, L - 1
    ex = ExecSpec(R=R, B=B, T=T_VERIFY, ctx=128, tiles=segment_tiles(B, rings=rings))
    spec = qwen38.segment_spec(g, R, L=L)

    def gen(key):
        return random_segment_inputs(key, g, ex, L, gain=0.25)

    shapes = jax.eval_shape(gen, jax.random.PRNGKey(0))
    shard = jax.tree.map(lambda s: tp if len(s.shape) and s.shape[0] == R else rep, shapes)
    inp = jax.jit(gen, out_shardings=shard)(jax.random.PRNGKey(1))
    w_mk, states, attn = jax.jit(lambda i: pack_segment(spec, g, ex, i))(inp)
    flat = []
    for name in kernel.build(spec, ex)[1]["plan"].w_names:
        v = w_mk[name]
        flat.extend(v if isinstance(v, tuple) else (v,))
    flat = [jax.device_put(kernel.fold(x), tp) for x in flat]
    hidden = jax.device_put(jnp.tile(inp.x, (R, 1)), tp)
    attn_b = jax.device_put(inp.attn.reshape(R * B, -1), tp)

    nmb = 1 + max(NL, 1) * NS * T_VERIFY + 64
    pool_shape = (nmb * 4, 128, 1, 4, 256)
    pool = jax.device_put(jnp.zeros((R * pool_shape[0],) + pool_shape[1:], jnp.float8_e4m3fn), tp)
    layout = gs.Qwen38GdnPoolLayout(pool_shape=pool_shape, pool_dtype="float8_e4m3fn")
    ck = 1 + np.arange(max(NL, 1) * NS * T_VERIFY, dtype=np.int32).reshape(max(NL, 1), NS, T_VERIFY)
    meta = []
    for l in range(NL):
        meta += [
            jnp.asarray(ck[l, :, 0]),
            jnp.zeros((nmb,), jnp.int32),
            jnp.asarray(ck[l]),
            jnp.arange(NS + 1, dtype=jnp.int32) * T_VERIFY,
            jnp.full((NS,), 100, jnp.int32),
        ]
    meta = [jax.device_put(m, rep) for m in meta]

    nf = len(flat)
    out: dict = {"rows": B, "L": L, "rings": list(rings) if rings else None}
    if kernel_only:
        pass
    elif NL:
        # Seed every read block with valid random f32 states through the scatter.
        seed = jax.shard_map(
            lambda p_, c, r, wb: gs.scatter_gdn_ckpts(p_, c, r, wb, layout=layout),
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P("tp"), P()),
            out_specs=P("tp"),
            check_vma=False,
        )
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))
        conv_r = jax.random.normal(k1, (R * NL, NS, 8, 1280), jnp.float32) * 0.05
        rec_r = (jax.random.normal(k2, (R * NL, NS * T_VERIFY * 6, 128, 128), jnp.float32) * 0.05).astype(jnp.bfloat16)
        wb_seed = jnp.broadcast_to(jnp.asarray(ck[:NL, :, :1]), (NL, NS, T_VERIFY))
        pool = jax.jit(seed)(pool, jax.device_put(conv_r, tp), jax.device_put(rec_r, tp), jax.device_put(wb_seed, rep))
        results = {}
        for pg, fs in ((False, False), (True, False), (True, True)):
            full, _, _, _ = build_segment_call(mesh, g, L, B, layout, pool_gdn=pg, fast_scatter=fs, tiles=ex.tiles)
            res_, pool_ = jax.jit(full)(hidden, attn_b, *flat, pool, *meta)
            results[(pg, fs)] = (np.asarray(res_, np.float32), np.asarray(jax.lax.bitcast_convert_type(pool_, jnp.uint8)))
            # The pool is part of the carry, so the scatter is live work.
            out["composed_us" + ("_poolgdn" if pg else "_gather") + ("_fast" if fs else "")] = _slope(
                lambda c, at, *a, full=full: tuple(full(c[0], at, *a[:nf], c[1], *a[nf:])),
                (hidden, pool),
                (attn_b, *flat, *meta),
                chains,
            )
        base_ = results[(False, False)]
        for key, name in (((True, False), "poolgdn"), ((True, True), "poolgdn_fast")):
            out[name + "_res_maxdiff"] = float(np.abs(results[key][0] - base_[0]).max())
            # block 0 is the null block (padded requests); compare the rest
            nb = pool_shape[0]
            diff = (results[key][1] != base_[1]).reshape(R, nb, -1)[:, 4:]
            out[name + "_pool_bytes_differ"] = int(diff.sum())
    else:
        full, _, _, _ = build_segment_call(mesh, g, L, B, layout, tiles=ex.tiles)
        out["composed_us"] = _slope(lambda h, at, *a: full(h, at, *a)[0], hidden, (attn_b, *flat), chains)

    fn, _, _, _ = make_flat_segment_step(mesh, spec, ex)
    pos = jnp.zeros((B,), jnp.int32)
    hid_m = jax.device_put(jnp.tile(jnp.pad(inp.x, ((0, ex.M - B), (0, 0))), (R, 1)), tp)
    att_m = jax.device_put(jnp.pad(inp.attn, ((0, 0), (0, ex.M - B), (0, 0))).reshape(R * ex.M, -1), tp)
    st = [jax.device_put(kernel.fold(states[k]), tp) for k in ("conv", "rec")] if NL else []
    t_kernel = _slope(
        lambda h, p_, at, *a: fn(p_, h, at, *a)[0], hid_m, (pos, att_m, *flat, *st), chains
    )
    out["kernel_us"] = t_kernel
    if QSA_ATTN:
        # Same segment with the QSA sparse attention in-kernel: every live row
        # attends the full 2051-token budget from a shuffled paged FP8 pool.
        ex_q = ExecSpec(R=R, B=B, T=T_VERIFY, ctx=128, tiles=segment_tiles(B, rings=rings))
        nh, npages = g.NH // R, 1024
        rng = np.random.default_rng(B)
        pool_q = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(3), (R * npages, 128, 1, 4, 256), jnp.float32).astype(
                jnp.float8_e4m3fn
            ),
            tp,
        )
        q_q = jax.device_put(jax.random.normal(jax.random.PRNGKey(4), (R * ex.M, nh, g.HD), jnp.bfloat16), tp)
        g_q = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), (R * ex.M, nh, g.HD), jnp.bfloat16), tp)
        pages_q = jax.device_put(
            jnp.asarray(rng.permutation(npages * 32)[: ex.M * 513].reshape(ex.M, 513), jnp.int32), rep
        )
        kvlen_q = jax.device_put(jnp.asarray([2051] * B + [0] * (ex.M - B), jnp.int32), rep)
        off_q = jax.device_put(jnp.asarray([0], jnp.int32), rep)
        scl_q = jax.device_put(jnp.asarray([1.0, 1.0], jnp.float32), rep)
        arms = (("s8", dict(slots=8)), ("s16", dict(slots=16)), ("s8_no_dma", dict(slots=8, ablate="dma")))
        for label, opts in arms:
            spec_q = qwen38.segment_spec(
                g, R, L=L, qsa_attn=dict(budget=512, **opts)
            )
            fn_q, _, _, _ = make_flat_segment_step(mesh, spec_q, ex_q)
            us = _slope(
                lambda h, p_, *a, fn_q=fn_q: fn_q(p_, h, *a)[0],
                hid_m,
                (pos, pool_q, q_q, g_q, pages_q, kvlen_q, off_q, scl_q, *flat, *st),
                chains,
            )
            out["attn_us" + ("_" + label if label else "")] = round(us - t_kernel, 1)
    return out


QSA_ATTN = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 1])
    ap.add_argument("--chains", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--rings", nargs="*", default=[], help="nbe:nbd overrides to sweep")
    ap.add_argument("--kernel-only", action="store_true", help="time the bare kernel only")
    ap.add_argument("--qsa-attn", action="store_true", help="also time the in-kernel QSA attention variant")
    a = ap.parse_args()
    global QSA_ATTN
    QSA_ATTN = a.qsa_attn
    sweep = [tuple(int(v) for v in r.split(":")) for r in a.rings] or [None]
    for L, B, rings in ((L, B, r) for L in a.layers for B in a.rows for r in sweep):
        if True:
            try:
                r = run(B, L, a.chains, rings, a.kernel_only)
            except Exception as e:  # noqa: BLE001
                r = {"rows": B, "L": L, "error": f"{type(e).__name__}: {e}"[:1500]}
            print("RESULT", json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
