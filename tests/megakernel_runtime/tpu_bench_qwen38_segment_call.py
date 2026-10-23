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


def run(B: int, L: int, chains: list[int]) -> dict:
    g, R = qwen38.FULL, 8
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    NS, NL = B // T_VERIFY, L - 1
    ex = ExecSpec(R=R, B=B, T=T_VERIFY, ctx=128, tiles=segment_tiles(B))
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

    full, _, _, _ = build_segment_call(mesh, g, L, B, layout)
    nf = len(flat)
    if NL:
        # The pool is part of the carry, so the scatter is live work.
        t_full = _slope(
            lambda c, at, *a: tuple(full(c[0], at, *a[:nf], c[1], *a[nf:])),
            (hidden, pool),
            (attn_b, *flat, *meta),
            chains,
        )
    else:
        t_full = _slope(lambda h, at, *a: full(h, at, *a)[0], hidden, (attn_b, *flat), chains)

    fn, _, _, _ = make_flat_segment_step(mesh, spec, ex)
    pos = jnp.zeros((B,), jnp.int32)
    hid_m = jax.device_put(jnp.tile(jnp.pad(inp.x, ((0, ex.M - B), (0, 0))), (R, 1)), tp)
    att_m = jax.device_put(jnp.pad(inp.attn, ((0, 0), (0, ex.M - B), (0, 0))).reshape(R * ex.M, -1), tp)
    st = [jax.device_put(kernel.fold(states[k]), tp) for k in ("conv", "rec")] if NL else []
    t_kernel = _slope(
        lambda h, p_, at, *a: fn(p_, h, at, *a)[0], hid_m, (pos, att_m, *flat, *st), chains
    )
    return {"rows": B, "L": L, "composed_us": t_full, "kernel_us": t_kernel,
            "overhead_us": round(t_full - t_kernel, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 1])
    ap.add_argument("--chains", type=int, nargs="+", default=[8, 32])
    a = ap.parse_args()
    for L in a.layers:
        for B in a.rows:
            try:
                r = run(B, L, a.chains)
            except Exception as e:  # noqa: BLE001
                r = {"rows": B, "L": L, "error": f"{type(e).__name__}: {e}"[:1500]}
            print("RESULT", json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
