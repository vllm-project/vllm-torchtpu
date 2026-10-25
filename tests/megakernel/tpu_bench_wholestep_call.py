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
"""TPU: compile and time the served whole-step call (`build_wholestep_call`, L=44) per bucket.

Random weights from the kernel plan's shapes, a served-geometry unified pool (small block
count), runner-shaped metadata: `nreq` requests with contexts near `--ctx`, distinct blocks per
request and layer, uniform T=4 verify windows. Prints compile seconds and the median call
time per bucket. Usage (8 cores, one process):
  python tests/megakernel/tpu_bench_wholestep_call.py --buckets 4,8,16 --ctx 3000
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel_runtime.qwen38_gdn_state import Qwen38GdnPoolLayout
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import segment_tiles
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import build_wholestep_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="4,8,16")
    ap.add_argument("--ctx", type=int, default=3000)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--max-groups", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    from torch_tpu._internal.pallas import get_pallas_mesh  # the served device order

    mesh = get_pallas_mesh(axis_names=("tp",))
    R = mesh.size
    g = qwen38.FULL
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    L, n_att = 44, 11
    layers = [QSALayerCache(compressed_slot=o % 2, main_slot=(o + 1) % 2) for o in range(n_att)]
    pps = a.max_groups * 4 // 128
    for B in [int(b) for b in a.buckets.split(",")]:
        nreq = B // 4
        n_cols = pps
        # Blocks: GDN 33 layers x 4 ckpts per request, QSA main + compressed per request & layer.
        NB = 1 + nreq * (33 * 4 + n_cols * 2 + n_att) + 8
        layout = Qwen38GdnPoolLayout(pool_shape=(NB * 4, 128, 1, 4, 256), pool_dtype="float8_e4m3fn")
        call, in_specs, out_specs, donate = build_wholestep_call(
            mesh, g, B, layout, layers, budget=a.budget, max_groups=a.max_groups,
            norm_eps=1e-6, rotary_dim=64,
        )
        spec = wholestep_spec(
            g, R, L=L, first="qsa", pool_gdn=True, qsa=dict(budget=a.budget), step=True,
            indexed=dict(max_groups=a.max_groups, pages_per_seq=pps, carrier="fp8", n_layers=n_att),
        )
        ex = ExecSpec(R=R, B=B, T=4, ctx=128, tiles=segment_tiles(B))
        _, info = kernel.make_step(mesh, spec, ex)
        p, wsh = info["plan"], info["wshape"]
        key = jax.random.PRNGKey(0)

        def rnd(shape, dtype):
            nonlocal key
            key, k = jax.random.split(key)
            if jnp.issubdtype(dtype, jnp.floating):
                return jax.device_put((jax.random.normal(k, shape, jnp.float32) * 0.02).astype(dtype), tp)
            return jax.device_put(jnp.zeros(shape, dtype), tp)

        ws = []
        for n in p.w_names:
            s = wsh[n]
            if kernel.is_tupled_weight(s):
                ws += [rnd((R * x.shape[0],) + tuple(x.shape[1:]), x.dtype) for x in s]
            else:
                ws.append(rnd((R * s[0][0],) + tuple(s[0][1:]), s[1]))
        rng = np.random.default_rng(0)
        free = iter(rng.permutation(np.arange(1, NB)))
        ctx = np.array([a.ctx + q for q in range(nreq)] + [0] * (16 - nreq), np.int32)
        qsl = np.array([min(q, nreq) * 4 for q in range(17)], np.int32)
        seq = ctx + np.where(np.arange(16) < nreq, 4, 0)
        meta = []
        for _ in range(33):
            st = np.zeros(16, np.int32)
            ck = np.zeros((16, 4), np.int32)
            for q in range(nreq):
                ck[q] = [next(free) for _ in range(4)]
                st[q] = ck[q, 0]
            meta += [st, np.zeros(NB, np.int32), ck, qsl, seq]
        tabs = []
        for _ in range(n_att):
            ring = np.zeros((16, 8), np.int32)
            main = np.zeros((16, n_cols), np.int32)
            comp = np.zeros((16, n_cols), np.int32)
            for q in range(nreq):
                ring[q, 0] = next(free)
                main[q] = [next(free) for _ in range(n_cols)]
                comp[q] = main[q]  # timing only: keys overlap the main pages
            tabs += [ring.reshape(-1), main.reshape(-1), comp.reshape(-1)]
        positions = np.concatenate([ctx[q] + np.arange(4) for q in range(nreq)]).astype(np.int32)
        trig = jnp.zeros((a.ctx + 4096, 128), jnp.bfloat16)
        norms = [jnp.zeros((256,), jnp.bfloat16), jnp.zeros((128,), jnp.bfloat16)] * n_att
        hidden = rnd((R * B, g.NHC * g.H), jnp.bfloat16)
        pool = jax.device_put(jnp.zeros((R * NB * 4, 128, 1, 4, 256), jnp.float8_e4m3fn), tp)
        put = lambda x: jax.device_put(jnp.asarray(x), rep)  # noqa: E731
        args = [hidden, *ws, pool, *map(put, meta), *map(put, tabs), put(positions), put(qsl),
                put(ctx), put(trig), *map(put, norms)]
        f = jax.jit(call, donate_argnums=donate)
        t0 = time.time()
        res, pool = f(*args)
        jax.block_until_ready(res)
        print(f"B={B} first call (compile+run) {time.time() - t0:.1f} s", flush=True)
        ts = []
        for _ in range(a.iters):
            args[1 + len(ws)] = pool
            t0 = time.perf_counter()
            res, pool = f(*args)
            jax.block_until_ready((res, pool))
            ts.append(time.perf_counter() - t0)
        print(f"B={B} call median {1e3 * np.median(ts):.3f} ms p10 {1e3 * np.percentile(ts, 10):.3f} "
              f"finite {bool(jnp.isfinite(res.astype(jnp.float32)).all())}", flush=True)


if __name__ == "__main__":
    main()
