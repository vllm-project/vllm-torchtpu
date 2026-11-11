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
"""TPU: compile and time the MK draft-step call (`qwen38_draft.build_draft_call`).

One draft step (draft QSA layer at the served stage-C packing + MoE + final mixer +
lm_head + greedy pick + the layer's cache writes) per call, random weights from the
plan's shapes, `nreq` requests with contexts near `--ctx`. `--T 1` is a draft loop step
(one row per request), `--T 4` draft step 0 over the verify rows. Usage (8 cores):
  python tests/megakernel/tpu_bench_draft_call.py --reqs 1,2,4 --T 1 4
`--trace-only` / `--interpret` for CPU checks.
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
from vllm_torchtpu.megakernel_runtime.qwen38_draft import build_draft_call, draft_spec
from vllm_torchtpu.megakernel_runtime.qwen38_gdn_state import Qwen38GdnPoolLayout
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import wholestep_tiles


def device_report(trace_dir: str, tag: str) -> None:
    """Device time per call (TPU:0 XLA modules >= 50 us) and the top ops inside them."""
    import collections
    import glob
    import re

    paths = sorted(glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True))
    if not paths:
        print(f"{tag} profile: no xplane", flush=True)
        return
    prof = jax.profiler.ProfileData.from_file(paths[-1])
    for plane in prof.planes:
        if plane.name != "/device:TPU:0":
            continue
        mods, ops = [], []
        for line in plane.lines:
            if line.name == "XLA Modules":
                mods = [(e.start_ns, e.duration_ns, e.name) for e in line.events]
            elif line.name == "XLA Ops":
                ops = [(e.start_ns, e.duration_ns, e.name) for e in line.events]
        big = [m for m in mods if m[1] >= 50_000]
        if not big:
            print(f"{tag} profile: no module >= 50 us ({len(mods)} modules)", flush=True)
            return
        print(f"{tag} device per call median {np.median([m[1] for m in big]) / 1e3:.1f} us "
              f"({len(big)} calls; module {big[0][2][:60]})", flush=True)
        tot = collections.Counter()
        for st, du, _ in big:
            for o in ops:
                if st <= o[0] < st + du:
                    tot[re.sub(r"\.\d+$", "", o[2].split(" = ")[0])[:48]] += o[1]
        for k, v in tot.most_common(40):
            print(f"{tag}   {v / len(big) / 1e3:7.1f} us  {k}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", default="1,2,4", help="requests per call (rows = reqs * T, "
                    "rounded up to 4)")
    ap.add_argument("--T", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--ctx", type=int, default=3000)
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--max-groups", type=int, default=2048)
    ap.add_argument("--vocab-shard", type=int, default=31040)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--wstd", type=float, default=0.02)
    ap.add_argument("--served-flags", action="store_true",
                    help="shard / union / loop indexer modes (the served whole-step's)")
    ap.add_argument("--profile", default="", help="trace dir: device time per call + top ops")
    ap.add_argument("--trace-only", action="store_true")
    ap.add_argument("--interpret", action="store_true")
    a = ap.parse_args()
    if a.served_flags:
        from vllm_torchtpu.megakernel.wholestep import qsa_full_c as wsc

        wsc.SHARD_INDEX = wsc.UNION_ATTN = wsc.LOOP_REQUESTS = True
    if a.interpret or a.trace_only:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        if a.interpret:
            original = pl.pallas_call
            pl.pallas_call = lambda *x, **k: original(*x, **{**k, "interpret": True})
        chip = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        pltpu.get_tpu_info = lambda: chip
    devs = jax.devices()
    by_id = {d.id: d for d in devs}
    order = [i for i in (0, 1, 4, 5, 6, 7, 2, 3) if i in by_id] if len(devs) >= 8 else [
        d.id for d in devs[:8]
    ]
    mesh = jax.sharding.Mesh(np.array([by_id[i] for i in order]), ("tp",))
    R = mesh.size
    g = qwen38.FULL
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    layer = QSALayerCache(compressed_slot=0, main_slot=0)
    pps = a.max_groups * 4 // 128
    for T in a.T:
        for nreq in [int(x) for x in a.reqs.split(",")]:
            B = -(-(nreq * T) // 4) * 4
            NRQ = B // T  # requests the kernel sees (padding requests idle)
            NR = max(16, NRQ)
            n_used = -(-(a.ctx + 4 + nreq) // 128)
            NB = 1 + nreq * (1 + 2 * n_used) + 8
            layout = Qwen38GdnPoolLayout(
                pool_shape=(NB * 4, 128, 1, 4, 256), pool_dtype="float8_e4m3fn"
            )
            kw = dict(budget=a.budget, max_groups=a.max_groups, norm_eps=1e-6,
                      rotary_dim=64, vocab_shard=a.vocab_shard)
            call, in_specs, out_specs, donate = build_draft_call(
                mesh, g, B, T, layout, layer, interpret=a.interpret, **kw
            )
            ex = ExecSpec(R=R, B=B, T=T, ctx=128, tiles=wholestep_tiles(B, a.max_groups))
            spec = draft_spec(g, R, budget=a.budget, max_groups=a.max_groups,
                              vocab_shard=a.vocab_shard)
            _, info = kernel.make_step(mesh, spec, ex)
            p, wsh = info["plan"], info["wshape"]
            key = jax.random.PRNGKey(0)

            def rnd(shape, dtype):
                nonlocal key
                if a.trace_only:
                    return jax.ShapeDtypeStruct(shape, dtype, sharding=tp)
                key, k = jax.random.split(key)
                if jnp.issubdtype(dtype, jnp.floating):
                    return jax.device_put(
                        (jax.random.normal(k, shape, jnp.float32) * a.wstd).astype(dtype), tp
                    )
                return jax.device_put(jnp.zeros(shape, dtype), tp)

            ws = []
            for n in p.w_names:
                s = wsh[n]
                if kernel.is_tupled_weight(s):
                    ws += [rnd((R * x.shape[0],) + tuple(x.shape[1:]), x.dtype) for x in s]
                else:
                    ws.append(rnd((R * s[0][0],) + tuple(s[0][1:]), s[1]))
            rng = np.random.default_rng(0)
            free = iter(rng.permutation(np.arange(4, 4 * NB)))
            ctx = np.array([a.ctx + q for q in range(nreq)] + [0] * (NR - nreq), np.int32)
            qsl = np.array([min(q, nreq) * T for q in range(NR + 1)], np.int32)
            main = np.zeros((NR, pps), np.int32)
            comp = np.zeros((NR, pps), np.int32)
            ring = np.zeros((NR, 8), np.int32)
            for q in range(nreq):
                main[q, :n_used] = [next(free) for _ in range(n_used)]
                comp[q, :n_used] = [next(free) for _ in range(n_used)]
                ring[q, 0] = next(free)
            positions = np.zeros(B, np.int32)
            positions[: nreq * T] = np.concatenate(
                [ctx[q] + np.arange(T) for q in range(nreq)]
            )
            inv = 1.0 / (1e7 ** (np.arange(0, 64, 2) / 64))
            ang = np.arange(a.ctx + 4096)[:, None] * inv[None]
            trig = jnp.asarray(
                np.concatenate([np.cos(ang), np.sin(ang), np.zeros((len(ang), 64))], 1),
                jnp.bfloat16,
            )
            hidden = rnd((R * B, g.NHC * g.H), jnp.bfloat16)
            pool = jax.jit(
                lambda nb=NB: jnp.zeros((R * nb * 4, 128, 1, 4, 256), jnp.float8_e4m3fn),
                out_shardings=tp,
            )() if not a.trace_only else rnd((R * NB * 4, 128, 1, 4, 256),
                                             jnp.float8_e4m3fn)
            put = lambda x: jax.device_put(jnp.asarray(x), rep)  # noqa: E731
            args = [hidden, *ws, pool, put(ring.reshape(-1)), put(main.reshape(-1)),
                    put(comp.reshape(-1)), put(positions), put(qsl), put(ctx), put(trig),
                    put(jnp.zeros((256,), jnp.bfloat16)), put(jnp.zeros((128,), jnp.bfloat16))]
            f = jax.jit(call, donate_argnums=donate)
            if a.trace_only:
                out = jax.eval_shape(f, *args)
                print(f"T={T} reqs={nreq} B={B} traced: {[o.shape for o in out]}", flush=True)
                continue
            t0 = time.time()
            res, tok, pool = f(*args)
            jax.block_until_ready(res)
            print(f"T={T} reqs={nreq} B={B} first call {time.time() - t0:.1f} s finite "
                  f"{float(jnp.isfinite(res.astype(jnp.float32)).mean()):.4f} tokens "
                  f"{np.asarray(tok)[:nreq * T].tolist()[:8]}", flush=True)
            pi = 1 + len(ws)
            ts = []
            for _ in range(a.iters):
                args[pi] = pool
                t0 = time.perf_counter()
                res, tok, pool = f(*args)
                jax.block_until_ready((res, tok, pool))
                ts.append(time.perf_counter() - t0)
            if a.profile:
                d = f"{a.profile}/t{T}_r{nreq}"
                with jax.profiler.trace(d):
                    for _ in range(5):
                        args[pi] = pool
                        res, tok, pool = f(*args)
                    jax.block_until_ready((res, tok, pool))
                device_report(d, f"T={T} reqs={nreq} B={B}")
            ts = np.sort(np.asarray(ts) * 1e3)
            print(f"T={T} reqs={nreq} B={B} draft-step call median {np.median(ts):.3f} ms "
                  f"p10 {ts[len(ts) // 10]:.3f}", flush=True)


if __name__ == "__main__":
    main()
