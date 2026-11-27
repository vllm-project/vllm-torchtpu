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
"""TPU: compile and time the served whole-step call (`build_wholestep_call`, L=44) per
bucket.

Random weights from the kernel plan's shapes, a served-geometry unified pool (small
block
count), runner-shaped metadata: `nreq` requests with contexts near `--ctx`, distinct
blocks per
request and layer, uniform T=4 verify windows. Prints compile seconds and the median
call
time per bucket. Usage (8 cores, one process):
  python tests/megakernel/tpu_bench_wholestep_call.py --buckets 4,8,16 --ctx 3000
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc
from vllm_torchtpu.megakernel_runtime.qwen38_gdn_state import Qwen38GdnPoolLayout
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import segment_tiles
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_call import (
    build_wholestep_call,
    indexed_options,
)


def _install_ledger() -> None:
    """Wrap `kernel.build_scratch`: print the planned VMEM scratch of every kernel built."""
    import math

    from vllm_torchtpu.megakernel import kernel as _k

    orig = _k.build_scratch

    def wrapped(model_spec, ex, *args, **kw):
        cp, scratch = orig(model_spec, ex, *args, **kw)
        rows = sorted(
            ((math.prod(sh) * jnp.dtype(dt).itemsize, k, sh, jnp.dtype(dt).name)
             for k, (sh, dt) in cp.vmem_shapes.items()), reverse=True)
        t = ex.tiles
        print(f"LEDGER B={ex.B} MB={ex.MB} nbe={t.nbe} nbd={t.nbd} total="
              f"{cp.vmem_bytes / 2**20:.3f} MiB ({len(rows)} buffers)", flush=True)
        for b in model_spec.all_blocks():
            print(f"LEDGER   widths {b.name} {type(b).__name__} {b.shared_widths(model_spec.config, ex)}",
                  flush=True)
        for sz, k, sh, dt in rows:
            print(f"LEDGER   {sz / 2**20:8.3f} {k} {sh} {dt} {cp.phase_map.get(k, '')}", flush=True)
        return cp, scratch

    _k.build_scratch = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", default="4,8,16")
    ap.add_argument("--ctx", type=int, default=3000)
    ap.add_argument("--ctx-min", type=int, default=0,
                    help="mixed contexts: request q gets ctx - q * (ctx - ctx_min) // (nreq - 1) "
                    "(0: ctx + q, near-uniform)")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--max-groups", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--wstd", type=float, default=0.02)
    ap.add_argument("--L", type=int, default=44)
    ap.add_argument("--remove", default="",
                    help="timing-only removal arms (wrong results): comma list of "
                         "attn_dma, attn_matmul, keys, select, uattn")
    ap.add_argument("--shard", action="store_true",
                    help="request-sharded indexer (VLLM_TPU_MEGAKERNEL wholeshard=1)")
    ap.add_argument("--union", action="store_true",
                    help="per-request union attention (VLLM_TPU_MEGAKERNEL wholeunion=1)")
    ap.add_argument("--loop", action="store_true",
                    help="per-request work in a fori_loop (VLLM_TPU_MEGAKERNEL wholeloop=1)")
    ap.add_argument("--phys", action="store_true",
                    help="union ids -> physical page ids per request (VLLM_TPU_MEGAKERNEL wholephys=1)")
    ap.add_argument("--phys-mxu", type=int, default=0, choices=(0, 1, 2),
                    help="main call: prefetched table rows, MXU physical ids, micro-page rows "
                    "(VLLM_TPU_MEGAKERNEL wholephysmxu=N; 2: one matmul per 128 ids)")
    ap.add_argument("--slots", type=int, default=0, help="union attention ring slots (0: default)")
    ap.add_argument("--key-mp", type=int, default=0,
                    help="key-stream kernel pages per DMA (VLLM_TPU_MEGAKERNEL wholekeymp=N; needs "
                    "--kb 512 for N > 1: a 512-token block column holds 4 consecutive pages)")
    ap.add_argument("--key-slots", type=int, default=2,
                    help="key-stream ring slots (VLLM_TPU_MEGAKERNEL wholekeyslots=N; -1 = auto)")
    ap.add_argument("--key-lanes", type=int, default=0, choices=(0, 128, 256),
                    help="key-stream lanes per carrier row (VLLM_TPU_MEGAKERNEL wholekeylanes=N)")
    ap.add_argument("--kb", type=int, default=128, choices=(128, 512),
                    help="pool block tokens (the served tables address 512-token blocks)")
    ap.add_argument("--score-arm", default="", choices=("", "dma", "compute"),
                    help="timing-only key-stream arm (wrong scores): dma = copies + waits, zero "
                    "scores; compute = no copies, scores of the stale ring")
    ap.add_argument("--rand-pool", action="store_true",
                    help="random FP8 in the whole pool (made before the weights, so its program "
                    "loads into unfragmented HBM): distinct keys and KV, as a prefilled server")
    ap.add_argument("--row-mix", type=float, default=-1.0,
                    help="served-like union: each request's 4 rows get hidden = (1 - m) * row0 + "
                    "m * own row (m = 0: identical inputs, rows differ by position only; m = 1 "
                    "or < 0: independent rows, a near-disjoint union)")
    ap.add_argument("--fill-keys", action="store_true",
                    help="random FP8 in the requests' compressed-key and main-KV blocks (else "
                    "zeros: equal scores, attention of zeros), so a hash sees the selection")
    ap.add_argument("--dhash", action="store_true",
                    help="print a device-side digest of the first call's outputs (residual "
                    "bytes + pool sum / xor / weighted sum), for bitwise checks at long contexts")
    ap.add_argument("--profile", default="", help="jax.profiler trace dir for --profile-iters calls")
    ap.add_argument("--profile-iters", type=int, default=5)
    ap.add_argument("--stacked-meta-check", action="store_true",
                    help="VLLM_TPU_MK_STACKED_MAMBA_OPERANDS: compare the stacked-metadata call "
                    "with the per-layer one (bitwise) after the first call")
    ap.add_argument("--hash", action="store_true",
                    help="print a hash of the first call's outputs (bitwise comparison across "
                    "VLLM_TPU_MK_TILES / VLLM_TPU_MK_BUCKET_TUNE configs)")
    ap.add_argument("--trace-only", action="store_true",
                    help="CPU check: interpret Pallas, lower the call once, no run")
    ap.add_argument("--aot", action="store_true",
                    help="no TPU: lower + compile for a tpu7x:2x2x1 topology (CPU host with libtpu) "
                    "and print OK or the compile error (e.g. the scoped VMEM OOM)")
    ap.add_argument("--tail", action="store_true",
                    help="VLLM_TPU_MK_WHOLE_L47: + the last QSA layer as a one-layer kernel")
    ap.add_argument("--export", action="store_true",
                    help="VLLM_TPU_MK_WHOLE_EXPORT_SELECTION: the call also returns layer 47's "
                    "selection in the sink layout; DHASH stays over the target outputs only, and "
                    "SELCHK checks the rows against the stock selection's invariants")
    ap.add_argument("--ledger", action="store_true",
                    help="print each kernel's planned VMEM scratch (name, shape, dtype, MiB, phase)")
    a = ap.parse_args()
    _apply_removals(a.remove)
    if a.ledger:
        _install_ledger()
    if a.trace_only:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        original = pl.pallas_call
        pl.pallas_call = lambda *x, **k: original(*x, **{**k, "interpret": True})
        chip = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        pltpu.get_tpu_info = lambda: chip
    wsc.SHARD_INDEX, wsc.UNION_ATTN, wsc.LOOP_REQUESTS = a.shard, a.union, a.loop
    wsc.UNION_PHYS = a.phys
    wsc.UNION_PHYS_MXU = a.phys_mxu
    wsc.KEY_MP, wsc.KEY_SLOTS, wsc.KEY_LANES = a.key_mp, a.key_slots, a.key_lanes
    if a.score_arm:
        from vllm_torchtpu.megakernel.wholestep import qsa_full_b

        qsa_full_b.SCORE_ARM = a.score_arm
        print("score arm:", a.score_arm, flush=True)
    if a.slots:
        from vllm_torchtpu.megakernel.wholestep import qsa_full_b

        qsa_full_b.UNION_SLOTS = a.slots
    # The served Pallas mesh order (get_pallas_mesh in the server: [0, 1, 4, 5, 6, 7, 2,
    # 3]).
    if a.aot:
        from jax.experimental import topologies

        devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices
    else:
        devs = jax.devices()
    by_id = {d.id: d for d in devs}
    mesh = jax.sharding.Mesh(
        np.array([by_id[i] for i in (0, 1, 4, 5, 6, 7, 2, 3)]), ("tp",)
    )
    R = mesh.size
    g = qwen38.FULL
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    L = a.L
    n_att = max(1, -(-L // 4))  # --L 45: layers 3..47 (VLLM_TPU_MK_WHOLE_L45)
    n_gdn = L - n_att
    n_qsa = n_att + int(a.tail)
    layers = [
        QSALayerCache(compressed_slot=o % 2, main_slot=(o + 1) % 2)
        for o in range(n_qsa)
    ]
    pps = a.max_groups * 4 // 128
    for B in [int(b) for b in a.buckets.split(",")]:
        nreq = B // 4
        NR = max(16, nreq)  # runner request rows (max_num_reqs)
        n_cols = pps  # table width (static); only the pages the contexts reach are allocated
        kb = a.kb  # pool block tokens: 128 (QSA kernel blocks) or 512 (served blocks)
        per = 512 // kb  # pool blocks per 512-token GDN manager block
        n_used = -(-(a.ctx + 4 + nreq) // kb)
        # Blocks: GDN 33 layers x 4 ckpts per request, QSA main + compressed per request
        # & layer.
        # Timing only: every QSA layer reads the same main / compressed pages of a
        # request
        # (a 90K-token context would otherwise need 11 x 1.4K blocks per request).
        NB = 1 + nreq * (n_gdn * 4 + n_qsa + 2 * n_used) + 8
        if a.rand_pool and not (a.trace_only or a.aot):
            pool = args = res = None  # the previous bucket's pool: freed before the new one

            def rand_pool(nb=NB, per=per, kb=kb):
                k = jax.random.fold_in(jax.random.PRNGKey(5), jax.lax.axis_index("tp"))
                v = jax.random.normal(k, (nb * per, kb, 1, 4, 256), jnp.bfloat16) * 0.5
                return v.astype(jnp.float8_e4m3fn)

            pool_rand = jax.jit(jax.shard_map(rand_pool, mesh=mesh, in_specs=(), out_specs=P("tp"),
                                              check_vma=False))()
        layout = Qwen38GdnPoolLayout(
            pool_shape=(NB * per, kb, 1, 4, 256), pool_dtype="float8_e4m3fn"
        )
        call, in_specs, out_specs, donate = build_wholestep_call(
            mesh,
            g,
            B,
            layout,
            layers,
            L=L,
            budget=a.budget,
            max_groups=a.max_groups,
            norm_eps=1e-6,
            rotary_dim=64,
            tail=a.tail,
            export=a.export,
        )
        if a.stacked_meta_check:
            call_s, _, _, _ = build_wholestep_call(
                mesh, g, B, layout, layers, L=L, budget=a.budget, max_groups=a.max_groups,
                norm_eps=1e-6, rotary_dim=64, tail=a.tail, stacked_meta=True, export=a.export,
            )
        spec = wholestep_spec(
            g,
            R,
            L=L,
            first="qsa",
            pool_gdn=True,
            qsa=dict(budget=a.budget),
            step=True,
            indexed=dict(
                max_groups=a.max_groups,
                pages_per_seq=pps,
                carrier="fp8",
                n_layers=n_att,
                **indexed_options(B),
            ),
            hc_fp8=envs.VLLM_TPU_MK_HC_FP8,
        )
        ex = ExecSpec(R=R, B=B, T=4, ctx=128, tiles=segment_tiles(B))
        specs = [spec]
        if a.tail:
            specs.append(wholestep_spec(
                g, R, L=1, first="qsa", pool_gdn=True, qsa=dict(budget=a.budget), step=True,
                indexed=dict(max_groups=a.max_groups, pages_per_seq=pps, carrier="fp8",
                             n_layers=1, **indexed_options(B)),
                hc_fp8=envs.VLLM_TPU_MK_HC_FP8,
            ))
        key = jax.random.PRNGKey(0)

        def rnd(shape, dtype):
            nonlocal key
            if a.trace_only or a.aot:
                return jax.ShapeDtypeStruct(shape, dtype, sharding=tp)
            key, k = jax.random.split(key)
            if jnp.issubdtype(dtype, jnp.floating):
                return jax.device_put(
                    (jax.random.normal(k, shape, jnp.float32) * a.wstd).astype(dtype),
                    tp,
                )
            return jax.device_put(jnp.zeros(shape, dtype), tp)

        ws = []
        for sp in specs:
            _, info = kernel.make_step(mesh, sp, ex)
            p, wsh = info["plan"], info["wshape"]
            for n in p.w_names:
                s = wsh[n]
                if kernel.is_tupled_weight(s):
                    ws += [rnd((R * x.shape[0],) + tuple(x.shape[1:]), x.dtype) for x in s]
                else:
                    ws.append(rnd((R * s[0][0],) + tuple(s[0][1:]), s[1]))
        rng = np.random.default_rng(0)
        # GDN ids are manager blocks (4 kernel blocks each), QSA ids are kernel blocks
        # (128 tokens): draw them from disjoint ranges of the kernel-granularity pool.
        NG = 1 + nreq * n_gdn * 4
        free_gdn = iter(rng.permutation(np.arange(1, NG)))
        free = iter(rng.permutation(np.arange(per * NG, per * NB)))
        if a.ctx_min:
            ctx_q = [a.ctx - q * (a.ctx - a.ctx_min) // max(nreq - 1, 1) for q in range(nreq)]
        else:
            ctx_q = [a.ctx + q for q in range(nreq)]
        print(f"B={B} ctx per request {ctx_q}", flush=True)
        ctx = np.array(ctx_q + [0] * (NR - nreq), np.int32)
        qsl = np.array([min(q, nreq) * 4 for q in range(NR + 1)], np.int32)
        seq = ctx + np.where(np.arange(NR) < nreq, 4, 0)
        meta = []
        for _ in range(n_gdn):
            st = np.zeros(NR, np.int32)
            ck = np.zeros((NR, 4), np.int32)
            for q in range(nreq):
                ck[q] = [next(free_gdn) for _ in range(4)]
                st[q] = ck[q, 0]
            meta += [st, np.zeros(NB * per, np.int32), ck, qsl, seq]
        tabs = []
        main = np.zeros((NR, n_cols), np.int32)
        comp = np.zeros((NR, n_cols), np.int32)
        for q in range(nreq):
            main[q, :n_used] = [next(free) for _ in range(n_used)]
            comp[q, :n_used] = [next(free) for _ in range(n_used)]
        for _ in range(n_qsa):
            ring = np.zeros((NR, 8), np.int32)
            for q in range(nreq):
                ring[q, 0] = next(free)
            tabs += [ring.reshape(-1), main.reshape(-1), comp.reshape(-1)]
        positions = np.concatenate([ctx[q] + np.arange(4) for q in range(nreq)]).astype(
            np.int32
        )
        inv = 1.0 / (1e7 ** (np.arange(0, 64, 2) / 64))
        ang = np.arange(a.ctx + 4096)[:, None] * inv[None]
        trig = jnp.asarray(
            np.concatenate([np.cos(ang), np.sin(ang), np.zeros((len(ang), 64))], 1),
            jnp.bfloat16,
        )
        norms = [
            jnp.zeros((256,), jnp.bfloat16),
            jnp.zeros((128,), jnp.bfloat16),
        ] * n_qsa
        hidden = rnd((R * B, g.NHC * g.H), jnp.bfloat16)
        if 0 <= a.row_mix < 1 and not (a.trace_only or a.aot):
            h4 = hidden.astype(jnp.float32).reshape(R, nreq, 4, -1)
            hidden = jax.device_put(
                ((1 - a.row_mix) * h4[:, :, :1] + a.row_mix * h4).reshape(R * B, -1)
                .astype(jnp.bfloat16), tp)
            print(f"B={B} ROW_MIX {a.row_mix}", flush=True)
        # Allocated sharded: a host-side zeros of the global pool does not fit.
        if a.trace_only or a.aot:
            pool = rnd((R * NB * per, kb, 1, 4, 256), jnp.float8_e4m3fn)
        elif a.rand_pool:
            pool = pool_rand
        else:
            pool = jax.jit(
                lambda nb=NB: jnp.zeros((R * nb * per, kb, 1, 4, 256), jnp.float8_e4m3fn),
                out_shardings=tp,
            )()
        if a.fill_keys and not (a.trace_only or a.aot):
            ids = jnp.asarray(np.concatenate([comp[:nreq, :n_used], main[:nreq, :n_used]]).ravel())

            def fill(p, ids):
                k = jax.random.fold_in(jax.random.PRNGKey(5), jax.lax.axis_index("tp"))
                v = jax.random.normal(k, (ids.shape[0],) + p.shape[1:], jnp.float32) * 0.5
                return p.at[ids].set(v.astype(p.dtype))

            pool = jax.jit(jax.shard_map(fill, mesh=mesh, in_specs=(P("tp"), P()),
                                         out_specs=P("tp"), check_vma=False),
                           donate_argnums=0)(pool, ids)
        put = lambda x: jax.device_put(jnp.asarray(x), rep)  # noqa: E731
        if a.aot:
            put = lambda x: jax.ShapeDtypeStruct(  # noqa: E731
                np.shape(x), jnp.asarray(x).dtype, sharding=rep
            )
        args = [hidden, *ws, pool, *map(put, meta), *map(put, tabs), put(positions), put(qsl),
                put(ctx), put(trig), *map(put, norms)]
        from vllm_torchtpu.models.vllm.qwen4_exp.pool_layout import main_kv_dcp

        if main_kv_dcp() > 1:  # the QSA layers' dcp_rank buffer: tp rank % 2, one per rank
            rank = jnp.arange(R, dtype=jnp.int32) % main_kv_dcp()
            args.append(jax.ShapeDtypeStruct(rank.shape, rank.dtype, sharding=tp) if a.aot
                        else jax.device_put(rank, tp))
        f = jax.jit(call, donate_argnums=donate)
        if a.aot:
            t0 = time.time()
            try:
                f.lower(*args).compile()
                print(f"B={B} L={L} AOT OK ({time.time() - t0:.0f} s)", flush=True)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                lines = [x for x in msg.splitlines() if "mem" in x.lower() or "oom" in x.lower()]
                print(f"B={B} L={L} AOT FAIL ({time.time() - t0:.0f} s): "
                      + " | ".join((lines or msg.splitlines())[:6])[:1500], flush=True)
                if lines:  # the full allocation list (VMEM accounting)
                    print("AOT DETAIL\n" + msg[:20000], flush=True)
                else:
                    import traceback

                    print("AOT TRACE " + traceback.format_exc()[-3000:].replace("\n", " | "))
            continue
        if a.trace_only:
            out = jax.eval_shape(f, *args)
            print(f"B={B} L={L} tail={a.tail} traced: {[o.shape for o in out]}", flush=True)
            continue
        if a.stacked_meta_check:
            # The walk's stacks: rows in a shuffled order plus 4 unused rows (PLE etc.).
            i0 = 2 + len(ws)
            per = [np.asarray(m) for m in meta]
            G = n_gdn + 4
            order = rng.permutation(G)[:n_gdn]
            st_stack = np.full((G, NR), 7, np.int32)
            ck_stack = np.full((G, NR, 4), 7, np.int32)
            for o in range(n_gdn):
                st_stack[order[o]] = per[5 * o]
                ck_stack[order[o]] = per[5 * o + 2]
            meta_s = [st_stack, ck_stack, order.astype(np.int32), per[1], per[3], per[4]]
            args_s = args[:i0] + list(map(put, meta_s)) + args[i0 + len(meta):]
            zero_pool = jax.jit(lambda p: jnp.zeros_like(p), out_shardings=tp)
            fs = jax.jit(call_s, donate_argnums=donate)
            args[1 + len(ws)] = zero_pool(pool)
            r0, p0, *_ = f(*args)
            args_s[1 + len(ws)] = zero_pool(pool)
            r1, p1, *_ = fs(*args_s)
            same = bool(jnp.array_equal(r0, r1)) and bool(
                jnp.array_equal(jax.lax.bitcast_convert_type(p0, jnp.uint8),
                                jax.lax.bitcast_convert_type(p1, jnp.uint8)))
            print(f"B={B} STACKED_META bitwise={same}", flush=True)
            pool = p0
            args[1 + len(ws)] = pool
        t0 = time.time()
        res, pool, *sel = f(*args)
        jax.block_until_ready(res)
        if sel:
            _check_selection(np.asarray(sel[0]), R, B, a.budget, ctx_q)
        if a.hash:
            import hashlib

            hsh = hashlib.sha256(np.asarray(res).tobytes())
            hsh.update(np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8)).tobytes())
            print(f"B={B} HASH {hsh.hexdigest()[:16]}", flush=True)
        if a.dhash:
            import hashlib

            def digest(p):
                u = jax.lax.bitcast_convert_type(p.reshape(p.shape[0], -1, 4), jnp.uint32)
                w = jax.lax.broadcasted_iota(jnp.uint32, u.shape, 1) * jnp.uint32(2654435761) + 1
                return (jnp.sum(u, axis=1),
                        jax.lax.reduce(u, jnp.uint32(0), jax.lax.bitwise_xor, (1,)),
                        jnp.sum(u * w, axis=1))

            dig = jax.jit(digest)(pool)
            hsh = hashlib.sha256(np.asarray(res).tobytes())
            for x in dig:
                hsh.update(np.asarray(x).tobytes())
            print(f"B={B} DHASH {hsh.hexdigest()[:16]}", flush=True)
        print(
            f"B={B} first call (compile+run) {time.time() - t0:.1f} s, finite "
            f"{float(jnp.isfinite(res.astype(jnp.float32)).mean()):.4f}",
            flush=True,
        )
        ts = []
        for _ in range(a.iters):
            args[1 + len(ws)] = pool
            t0 = time.perf_counter()
            res, pool, *_ = f(*args)
            jax.block_until_ready((res, pool))
            ts.append(time.perf_counter() - t0)
        if a.profile:
            with jax.profiler.trace(f"{a.profile}/b{B}"):
                for _ in range(a.profile_iters):
                    args[1 + len(ws)] = pool
                    res, pool, *_ = f(*args)
                jax.block_until_ready((res, pool))
        r32 = res.astype(jnp.float32)
        fin_rows = np.asarray(jnp.isfinite(r32).all(axis=1)).reshape(R, B)
        print(
            f"B={B} L={L} finite rows per rank:",
            fin_rows.astype(int).tolist(),
            flush=True,
        )
        pool_fin = float(jnp.isfinite(pool.astype(jnp.float32)).mean())
        print(f"B={B} pool finite {pool_fin:.4f}", flush=True)
        in_mean = float(jnp.abs(hidden.astype(jnp.float32)).mean())
        print(
            f"B={B} call median {1e3 * np.median(ts):.3f} ms "
            f"p10 {1e3 * np.percentile(ts, 10):.3f} "
            f"finite {float(jnp.isfinite(r32).mean()):.4f} absmean "
            f"{float(jnp.nanmean(jnp.abs(r32))):.4g} (input {in_mean:.4g})",
            flush=True,
        )


def _check_selection(sel: np.ndarray, R: int, B: int, K: int, ctx_q: list) -> None:
    """`--export`: every rank returns the same rows, and each live row is a stock-layout
    selection (`qsa_rpa.selected_tokens`): min(n_vis, K) distinct ascending groups below
    n_vis (each as its 4 tokens), then the row's partial group's tokens, then -1."""
    import hashlib

    W = 4 * K + 3
    sel = sel.reshape(R, B, W)
    same = all(np.array_equal(sel[0], sel[r]) for r in range(R))
    bad, n_full, n_ga = [], 0, 0
    for row in range(B):
        q, t = divmod(row, 4)
        s = sel[0, row]
        if q >= len(ctx_q):
            continue
        n = ctx_q[q] + t + 1
        nv, ga = n // 4, ctx_q[q] // 4
        cnt = min(nv, K)
        g = np.where(s[: 4 * K : 4] >= 0, s[: 4 * K : 4] // 4, -1)  # block j: tokens 4 g + 0..3
        ok = (g[:cnt] >= 0).all() and (g[cnt + 1 :] == -1).all() and (g[:cnt] < nv).all()  # slot cnt: tail
        ok &= bool((np.diff(g[:cnt]) > 0).all())
        blk = np.where(g[:, None] >= 0, g[:, None] * 4 + np.arange(4), -1).reshape(-1)[: 4 * cnt]
        ok &= bool(np.array_equal(s[: 4 * cnt], blk))
        tail = [p for p in range(nv * 4, n)]
        # The served rows leave the partial group's tokens -1 (`export_selection`).
        ok &= bool(np.array_equal(s[4 * cnt : 4 * cnt + len(tail)], tail)
                   and (s[4 * cnt + len(tail) :] == -1).all()) or bool((s[4 * cnt :] == -1).all())
        n_full += int(nv > K)
        n_ga += int(nv > K and cnt and g[cnt - 1] == ga)
        if not ok:
            bad.append(row)
    out = os.environ.get("CDK_OUTPUT_DIR", "")
    if out:  # for an offline comparison (draft rows vs the target's selection)
        np.save(f"{out}/sel_b{B}_ctx{ctx_q[0]}.npy", sel[0])
    print(f"B={B} SELCHK ranks_equal={same} rows_bad={len(bad)} {bad[:8]} over_budget_rows={n_full} "
          f"g_a_selected={n_ga} digest {hashlib.sha256(sel[0].tobytes()).hexdigest()[:16]}",
          flush=True)


def _apply_removals(spec: str) -> None:
    """Timing-only removal arms: patch pieces of the QSA step out of the kernel (wrong
    results)."""
    import functools

    from vllm_torchtpu.megakernel.wholestep import indexer as ix
    from vllm_torchtpu.megakernel.wholestep import qsa_full_b, qsa_full_c

    arms = [x for x in spec.split(",") if x]
    for arm in arms:
        if arm in ("attn_dma", "attn_matmul"):  # attend_row's own ablation switch
            qsa_full_c.attend_row = functools.partial(
                qsa_full_c.attend_row, ablate=arm.split("_")[1]
            )
        elif arm == "keys":  # no compressed-key stream or scoring
            qsa_full_b.QSAFullIndexedBlock._score_request = lambda self, *a, **k: None
        elif arm == "uattn":  # union attention (DMAs + compute) replaced by zeros
            from vllm_torchtpu.megakernel.wholestep import union_attn as ua

            ua.attend_request = lambda *, T, qs, **k: [jnp.zeros_like(q) for q in qs]
        elif arm == "select":  # no thresholds / compaction arithmetic
            ix.thresholds = lambda bits, ids, nv, K, G: (
                jnp.zeros(bits.shape[:1] + (1, 1), jnp.int32),
                jnp.zeros(bits.shape[:1] + (1, 1), jnp.int32),
            )
            ix.compact_row = lambda b, th, tm, nv, K, consts: jnp.zeros(
                (1, K), jnp.int32
            )
        else:
            raise ValueError(arm)
    if arms:
        print("removal arms:", arms, flush=True)


if __name__ == "__main__":
    main()
