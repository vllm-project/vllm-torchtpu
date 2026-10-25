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
"""Two-host (R=16) device timing of the whole-step kernel, synthetic weights.

Run on both hosts of a tpu7x 2x2x2 slice (blocks/run_tpu2.sh) with COORD,
NPROC, PID set:

    python -m tests.kernels.deepseek_v4_megakernel.step_device2 --T 16 --owners 1

Full DSv4-Pro shapes: dense ring packs (S=16 slices), 24 local fp4 experts
per rank per layer, embedding; stub attention phase. Prints RESULT lines
(process 0).
"""

import argparse
import functools
import json
import os
import statistics
import time

import jax

jax.distributed.initialize(
    os.environ["COORD"], int(os.environ["NPROC"]), int(os.environ["PID"])
)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK  # noqa: E402

PID = jax.process_index()


def result(**kw):
    if PID == 0:
        print("RESULT " + json.dumps(kw), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--B", type=int, default=2)
    p.add_argument("--q", type=int, default=7)
    p.add_argument("--owners", type=int, default=1)
    p.add_argument("--banks", type=int, default=4)
    p.add_argument("--bank-mb", type=float, default=5.0)
    p.add_argument("--layers", type=int, default=0)
    p.add_argument("--moe", choices=("stub", "real"), default="real")
    p.add_argument("--iters", type=int, default=10)
    # extra variants on the same weights, comma separated: moe=stub,T=32/owners=4
    p.add_argument("--variants", default="")
    # every synthetic weight identical on every rank (each rank's slices and
    # experts hold the same values), so all ranks carry the same hidden states
    # (default: drawn per rank)
    p.add_argument("--replicated", action="store_true")
    a = p.parse_args()
    result(phase="start", args=vars(a))
    devs = sorted(jax.devices(), key=lambda d: d.id)
    R = len(devs)
    mesh = Mesh(np.array(devs), ("d",))
    EP = NamedSharding(mesh, P("d"))
    kw = dict(tp=R, rows=a.T, n_banks=a.banks, bank_bytes=int(a.bank_mb * 2**20))
    if a.layers:
        kw["ratios"] = dc.DSV4_PRO_RATIOS[: a.layers]
    cfg = dc.DenseChainConfig(**kw)
    L, H = cfg.n_layers, cfg.hidden
    keys = iter(jax.random.split(jax.random.key(0), 1000))

    def gen(shape, dtype, kind):
        """A random EP-sharded array built on device (leading axis split);
        --replicated: one rank's draw, the same on every rank."""
        one = (shape[0] // R,) + tuple(shape[1:]) if a.replicated else shape

        def draw(k):
            if kind == "fp8":
                z = jax.random.normal(k, one, jnp.bfloat16) * 64
                return jnp.clip(z, -448, 448).astype(dtype)
            if kind == "bits":
                return jax.random.bits(k, one, jnp.uint32)
            if kind == "int":
                return jax.random.randint(k, one, 0, cfg.n_experts, jnp.int32)
            if kind == "scale":
                return jax.random.uniform(k, one, jnp.float32, 0.5, 1.5) * 1e-2
            return (jax.random.normal(k, one, jnp.float32) * 0.02).astype(dtype)

        def f(k):
            x = draw(k)
            return jnp.tile(x, (R,) + (1,) * (x.ndim - 1)) if a.replicated else x

        return jax.jit(f, out_shardings=EP)(next(keys))

    t0 = time.time()
    tiles = {}
    for n, s in cfg.streams.items():
        nl = max(1, len(cfg.stream_layers(n)))
        kind = "fp8" if s.dtype == dc.FP8 else "bf16"
        shape = (R * nl, s.n_tiles, s.tile_rows, s.tile_cols)
        tiles[n] = gen(shape, jnp.dtype(s.dtype), kind)
    smalls = gen((R * L, cfg.smalls_rows, 128), jnp.float32, "scale")
    embed = gen((R, 129280, H), jnp.bfloat16, "bf16")
    pos_max = 1 << 17
    ang = jnp.arange(pos_max)[:, None] * 160000.0 ** (-jnp.arange(32) / 32.0)[None]
    cs = jnp.concatenate([jnp.cos(ang), jnp.sin(ang)], 1).astype(jnp.float32)
    cs = SK.SP.pack_cos_sin(cs)
    # rank shards of an axis-0 concatenation, not a [R, ...] stack: a per-rank
    # leading dim of 1 gets another entry layout and a copy every step
    cs = jax.device_put(jnp.tile(cs, (R, 1)), EP)
    moe_args = []
    if a.moe == "real":
        E, inter = cfg.n_experts, cfg.moe_inter
        # axis-0 rank shards (a per-rank leading 1 gets a relayout each step)
        moe_args += [gen((R * 129280, 6), jnp.int32, "int") for _ in range(3)]
        moe_args += [gen((R, L, E), jnp.float32, "scale")]
        for _ in range(L):
            moe_args += [
                gen((E, H // 8, 2 * inter), jnp.uint32, "bits"),
                gen((E, 16, 2 * inter), jnp.float32, "scale"),
                gen((E, inter // 8, H), jnp.uint32, "bits"),
                gen((E, 8, H), jnp.float32, "scale"),
            ]
    jax.block_until_ready((tiles, smalls, embed, moe_args))
    result(phase="weights", seconds=time.time() - t0)

    variants = [dict()] + [
        dict(kv.split("=") for kv in v.split("/")) for v in a.variants.split(",") if v
    ]
    for var in variants:
        run_variant(a, var, cfg, tiles, smalls, embed, cs, moe_args, mesh, EP, R)


def run_variant(a, var, cfg, tiles, smalls, embed, cs, moe_args, mesh, EP, R):
    import dataclasses

    T = int(var.get("T", a.T))
    owners = int(var.get("owners", a.owners))
    moe_impl = var.get("moe", a.moe)
    banks = int(var.get("banks", a.banks))
    nb = int(var.get("B", a.B))
    cfg = dataclasses.replace(cfg, rows=T, n_banks=banks)
    L = cfg.n_layers
    scfg = SK.StepKernelConfig(
        dense=cfg, q=a.q, B=nb, moe_impl=moe_impl, moe_leased=var.get("lease", "1") == "1",
        aux_layers=tuple(x for x in (L - 3, L - 2, L - 1) if x >= 0),
        attn_tp=int(var.get("attn_tp", 1)), attn_split=var.get("split") == "1",
        union_stats=var.get("ustats") == "1",
    )  # fmt: skip
    if var.get("pipeline") == "1":  # as VLLM_TPU_DSV4_MK_ATTN_PIPELINE=1 (step_impl.kernel_config)
        split = scfg.attn_tp > 1 and var.get("nosplit") != "1"  # nosplit: attribution only
        scfg = dataclasses.replace(scfg, attn_pipeline=True, attn_split=split)
    if moe_impl != "real":
        moe_args = []
    elif "moe_distinct" in var:  # per-layer expert arrays reuse k distinct sets
        k, head, per = int(var["moe_distinct"]), moe_args[:4], moe_args[4:]
        moe_args = head + [per[4 * (li % k) + j] for li in range(L) for j in range(4)]
    attn_impl = var.get("attn", "stub")
    # attention-phase attribution knobs (attn_phase.ABLATE; read at trace time)
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP

    attn_ablate = var.get("attn_ablate", "")
    AP.ABLATE = frozenset(x for x in attn_ablate.split("+") if x)
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import moe_phase as MP

    MP.ABLATE = frozenset(x for x in var.get("moe_ablate", "").split("+") if x)
    skip = tuple(x for x in var.get("skip", "").split("+") if x)
    if skip:
        scfg = dataclasses.replace(scfg, skip=skip)
    max_ctx = int(var.get("max_ctx", 131072))
    if attn_impl == "real":
        scfg = dataclasses.replace(scfg, attn_impl="real", max_ctx=max_ctx)
    M = scfg.M
    ids = np.zeros((R, M), np.int32)
    pos = np.zeros((R, M), np.int32)
    seq = np.zeros((R, nb), np.int32)
    rng = np.random.default_rng(0)
    GS = scfg.attn_tp  # attn_tp > 1: request i in group i % G, slot i // G
    for i in range(owners):
        r, k = (i % scfg.G) * GS, i // scfg.G
        p0 = int(var.get("p0", 120000)) + 37 * i
        ids[r, k * a.q : (k + 1) * a.q] = rng.integers(0, 129280, a.q)
        pos[r, k * a.q : (k + 1) * a.q] = np.arange(p0, p0 + a.q)
        seq[r, k] = p0 + a.q
        ids[r : r + GS], pos[r : r + GS], seq[r : r + GS] = ids[r], pos[r], seq[r]
    put = lambda x: jax.make_array_from_callback(x.shape, EP, lambda i: x[i])  # noqa: E731
    ids_d, pos_d, seq_d = put(ids), put(pos), put(seq)
    names = cfg.stream_names
    inplace = var.get("inplace") == "1"
    at = (
        attn_operands(
            mesh, EP, R, L, nb, owners, a.q, seq, int(var.get("synth_ctx", max_ctx)),
            replica_of=(lambda r: r // GS) if var.get("rep") == "group" else None,
        )  # cache arrays sized for this
        if attn_impl == "real"
        else None
    )
    n_arr = len(at["arrays"]) if at else 0
    n_moe = len(moe_args)

    def per_rank(step, ids, pos, seq, embed, cs, smalls, *rest):
        tl = rest[: len(names)]
        mo = rest[len(names) : len(names) + n_moe]
        ex = rest[len(names) + n_moe :]
        attn = None
        if at is not None:
            arrs, bt, wts = ex[:n_arr], ex[n_arr], ex[n_arr + 1 :]
            attn = dict(
                arrays=list(arrs), cache_index=at["cache_index"],
                cache_layers=at["cache_layers"], bt_flat=bt, bt_off=at["bt_off"],
                bt_w=at["bt_w"], weights=dict(zip(at["wnames"], wts)),
            )  # fmt: skip
        moe_w = None
        if moe_impl == "real":
            moe_w = dict(tid2eid=[mo[0], mo[1], mo[2]], bias=mo[3][0],
                         w13=list(mo[4::4]), w13_scale=list(mo[5::4]),
                         w2=list(mo[6::4]), w2_scale=list(mo[7::4]))  # fmt: skip
        outs = step(ids[0], pos[0], seq[0], embed[0], cs, smalls,
                    dict(zip(names, tl)), moe_w, attn,
                    eblk=jax.lax.axis_index("d"))  # fmt: skip
        res, aux = outs[0], outs[1]
        arrs_out = list(outs[5]) if at is not None else []  # [] when in place
        st = (outs[-1][None],) if scfg.union_stats else ()
        return (res[None], aux[None], *arrs_out, *st)

    args = [
        ids_d,
        pos_d,
        seq_d,
        embed,
        cs,
        smalls,
        *[tiles[n] for n in names],
        *moe_args,
    ]
    c0 = len(args)  # aliased: the cache arrays are donated and fed back each step
    if at is not None:
        args += [*at["arrays"], at["bt_flat"], *at["weights"]]

    def make(inplace, sc=None):
        sc = scfg if sc is None else sc
        step = SK.build(dataclasses.replace(sc, cache_inplace=inplace), axis="d")
        n_out = 0 if inplace else n_arr
        return jax.jit(
            jax.shard_map(functools.partial(per_rank, step), mesh=mesh,
                          in_specs=(P("d"),) * len(args),
                          out_specs=(P("d"),) * (2 + n_out + scfg.union_stats),
                          check_vma=False),
            donate_argnums=() if inplace else tuple(range(c0, c0 + n_out)),
        )  # fmt: skip

    f = make(inplace)

    if var.get("inplace_check") == "1" and at is not None:
        # one step each from identical caches: in place == aliased, bitwise
        diff = jax.jit(lambda x, y: jnp.sum(x != y))
        cp = jax.jit(lambda x: x + jnp.zeros((), x.dtype))
        orig = [cp(x) for x in args[c0 : c0 + n_arr]]
        ca, cb = [cp(x) for x in orig], [cp(x) for x in orig]
        oa = make(False)(*args[:c0], *ca, *args[c0 + n_arr :])
        ob = make(True)(*args[:c0], *cb, *args[c0 + n_arr :])
        jax.block_until_ready((oa, ob))
        na = oa[2:]
        mism = [int(diff(x, y)) for x, y in zip(na, cb)]
        changed = [int(diff(x, y)) for x, y in zip(na, orig)]
        changed_b = [int(diff(x, y)) for x, y in zip(cb, orig)]
        result(
            check="inplace_vs_aliased",
            res_mismatch=int(diff(oa[0], ob[0])),
            aux_mismatch=int(diff(oa[1], ob[1])),
            cache_mismatch=sum(mism),
            words_written_aliased=sum(changed),
            words_written_inplace=sum(changed_b),
            ok=sum(mism) == 0 and sum(changed) > 0 and int(diff(oa[0], ob[0])) == 0,
        )
        del orig, ca, cb, oa, ob, na

    if var.get("pipeline_check") == "1" and at is not None:
        # one in-place step each from identical caches and inputs: the program with
        # VLLM_TPU_DSV4_MK_ATTN_PIPELINE off vs on (pf_fault: the prefetch control)
        diff = jax.jit(lambda x, y: jnp.sum(x != y))
        cp = jax.jit(lambda x: x + jnp.zeros((), x.dtype))
        grp = scfg.attn_tp > 1
        modes = {"off": (False, False), "pipe": (True, grp), "pipe_nosplit": (True, False),
                 "split": (False, grp)}  # (attn_pipeline, attn_split); controls: pc_ref / pc_new
        off, on = (dataclasses.replace(scfg, attn_pipeline=modes[m][0], attn_split=modes[m][1])
                   for m in (var.get("pc_ref", "off"), var.get("pc_new", "pipe")))
        orig = [cp(x) for x in args[c0 : c0 + n_arr]]
        ca, cb = [cp(x) for x in orig], [cp(x) for x in orig]
        oa = make(True, off)(*args[:c0], *ca, *args[c0 + n_arr :])
        AP.PF_FAULT = var.get("pf_fault", "")
        try:
            ob = make(True, on)(*args[:c0], *cb, *args[c0 + n_arr :])
            jax.block_until_ready((oa, ob))
        finally:
            AP.PF_FAULT = ""
        mism = [int(diff(x, y)) for x, y in zip(ca, cb)]
        written = sum(int(diff(x, y)) for x, y in zip(ca, orig))
        res_m, aux_m = int(diff(oa[0], ob[0])), int(diff(oa[1], ob[1]))
        ust = {}
        if scfg.union_stats:  # [R, L, B] CSA union sizes: ref vs new, group ranks vs their first
            gdiv = jax.jit(lambda x: jnp.sum(x.reshape(R // GS, GS, -1)
                                             != x.reshape(R // GS, GS, -1)[:, :1]))
            ust = dict(union_sizes_mismatch=int(diff(oa[-1], ob[-1])),
                       ref_group_rank_union_diff=int(gdiv(oa[-1])),
                       new_group_rank_union_diff=int(gdiv(ob[-1])),
                       union_entries=int(jax.jit(lambda x: jnp.sum(x >= 0))(oa[-1])))
        rel = jax.jit(lambda x, y: jnp.max(jnp.abs(x.astype(jnp.float32) - y))
                      / jnp.maximum(jnp.max(jnp.abs(y.astype(jnp.float32))), 1e-30))
        # ranks of one attention group whose residual differs from the group's first rank
        # (the split mixes the group's selections: equal to the unsplit one only if equal)
        div = jax.jit(lambda x: jnp.sum(
            x.reshape(R // GS, GS, -1) != x.reshape(R // GS, GS, -1)[:, :1]))
        result(
            check="pipeline_off_vs_on", var=var, owners=owners, T=T,
            res_mismatch=res_m, aux_mismatch=aux_m, cache_mismatch=sum(mism),
            cache_arrays_differing=sum(m > 0 for m in mism), words_written=written,
            res_max_rel=float(rel(ob[0], oa[0])), aux_max_rel=float(rel(ob[1], oa[1])),
            ref_group_rank_res_diff=int(div(oa[0])), new_group_rank_res_diff=int(div(ob[0])),
            **ust,
            finite=bool(np.isfinite(np.asarray(oa[0].addressable_shards[0].data,
                                               np.float32)).all()),
            bitexact=sum(mism) == 0 and res_m == 0 and aux_m == 0 and written > 0,
        )  # fmt: skip
        del orig, ca, cb, oa, ob

    if var.get("hlocheck") == "1":  # large copies XLA inserted around the kernel
        import re

        txt = f.lower(*args).compile().as_text()
        big = []
        for line in txt.splitlines():
            m = re.search(
                r"= (\w+)\[([0-9,]+)\][^ ]* (copy|copy-start|transpose|bitcast-convert|fusion)\(",
                line,
            )
            if m:
                n = int(np.prod([int(x) for x in m.group(2).split(",")]))
                nbytes = n * {"s32": 4, "f32": 4, "u8": 1, "bf16": 2, "s8": 1}.get(
                    m.group(1), 4
                )
                if nbytes > (8 << 20):
                    big.append((m.group(3), m.group(1), m.group(2), nbytes))
        result(
            check="hlo_big_ops",
            attn=attn_impl,
            n=len(big),
            total_mb=sum(b[3] for b in big) / 2**20,
            top=big[:12],
        )

    def run():
        out = f(*args)
        if not inplace:
            args[c0 : c0 + n_arr] = out[2 : 2 + n_arr]
        return out

    t0 = time.time()
    out = jax.block_until_ready(run())
    result(phase="compile+first", seconds=time.time() - t0)
    for _ in range(2):
        jax.block_until_ready(run())
    ts = []
    for _ in range(a.iters):
        t1 = time.perf_counter()
        jax.block_until_ready(run())
        ts.append((time.perf_counter() - t1) * 1e3)
    ts.sort()
    # back-to-back steps, one sync: hides the host dispatch of the ~200 cache
    # operands behind the previous step (the serving loop dispatches async too)
    K = max(1, a.iters)
    jax.block_until_ready(run())
    t1 = time.perf_counter()
    for _ in range(K):
        o = run()
    jax.block_until_ready(o)
    pipe_ms = (time.perf_counter() - t1) * 1e3 / K
    t1 = time.perf_counter()
    o = run()
    disp_ms = (time.perf_counter() - t1) * 1e3  # host time to enqueue one step
    jax.block_until_ready(o)
    if var.get("profile") == "1":
        result(check="scopes", var=var, planes=profile_scopes(run))
    res0 = np.asarray(out[0].addressable_shards[0].data, np.float32)
    if scfg.union_stats:  # [R, L, B] CSA union sizes (-1: none); process 0's ranks
        us = np.concatenate([np.asarray(x.data) for x in out[-1].addressable_shards])
        live = us[us >= 0]
        result(
            check="union_stats",
            var=var,
            n=int(live.size),
            mean=float(live.mean()) if live.size else -1,
            max=int(live.max()) if live.size else -1,
            min=int(live.min()) if live.size else -1,
        )
    result(
        check="step_time", var=var, T=T, B=nb, lease=var.get("lease", "1"), owners=owners,
        layers=L, moe=moe_impl, attn=attn_impl, inplace=inplace, max_ctx=max_ctx,
        synth_ctx=int(var.get("synth_ctx", max_ctx)),
        p0=int(var.get("p0", 120000)), skip="+".join(skip), attn_ablate=attn_ablate,
        banks=banks, median_ms=statistics.median(ts), p10_ms=ts[len(ts) // 10],
        p90_ms=ts[(9 * len(ts)) // 10], n=a.iters, pipe_ms=pipe_ms, dispatch_ms=disp_ms,
        finite=bool(np.isfinite(res0).all()),
    )  # fmt: skip


def profile_scopes(run, n_steps=3, devices=("TPU:0", "TPU:3")):
    """In-kernel named_scope regions (needs LIBTPU_INIT_ARGS with
    --xla_enable_custom_call_region_trace=true): union time per region name
    per step, on the given local devices."""
    import glob
    import tempfile

    d = tempfile.mkdtemp()
    jax.block_until_ready(run())
    jax.profiler.start_trace(d)
    for _ in range(n_steps):
        out = run()
    jax.block_until_ready(out)
    jax.profiler.stop_trace()
    files = glob.glob(f"{d}/**/*.xplane.pb", recursive=True)
    pd = jax.profiler.ProfileData.from_file(files[0])
    res = {}
    for plane in pd.planes:
        dev = next((x for x in devices if plane.name.endswith(x)), None)
        if dev is None:
            continue
        spans: dict[str, list] = {}
        ops: dict[str, float] = {}
        for line in plane.lines:
            if line.name == "XLA Ops":  # every device op of the step, by name
                for ev in line.events:
                    ops[ev.name] = ops.get(ev.name, 0.0) + ev.duration_ns
            for ev in line.events:
                nm = ev.name
                if nm.startswith(("mk_", "moe_")) or "dsv4_step" in nm:
                    key = nm if "dsv4_step" not in nm else "step_op"
                    spans.setdefault(key, []).append((ev.start_ns, ev.end_ns))
        tot = {}
        for k, iv in spans.items():
            iv.sort()
            acc, cur = 0.0, None
            for a, b in iv:
                if cur is None or a > cur[1]:
                    if cur is not None:
                        acc += cur[1] - cur[0]
                    cur = [a, b]
                else:
                    cur[1] = max(cur[1], b)
            acc += cur[1] - cur[0]
            tot[k] = dict(ms_per_step=acc / n_steps / 1e6, n=len(iv) // n_steps)
        top = sorted(ops.items(), key=lambda kv: -kv[1])[:25]
        tot["_xla_ops"] = {k: round(v / n_steps / 1e3, 1) for k, v in top}  # us/step
        tot["_line_names"] = sorted({ln.name for ln in plane.lines})
        res[plane.name] = tot
    return res


def attn_operands(mesh, EP, R, L, nb, owners, q, seq, max_ctx, replica_of=None):
    """Per-rank synthetic served caches (attn_phase_synth), stacked on a
    leading rank axis; built on each process's own devices. replica_of(rank):
    ranks mapped to the same value get identical caches (an attention group's
    KV replicas, as serving keeps them); default: every rank its own."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import attn_phase_synth as S

    kinds = S.DSV4_KINDS[:L]
    devs = list(mesh.devices.flat)
    per = {}
    for d in jax.local_devices():
        r = devs.index(d)
        seqs = [int(seq[r, j]) if seq[r, j] > 0 else max_ctx - 72 for j in range(nb)]
        rk = r if replica_of is None else replica_of(r)
        out = S.synth_rank(jax.random.key(1000 + rk), kinds, seqs, max_ctx,
                           served_dtypes=True)  # fmt: skip
        per[d] = jax.tree.map(  # synth builds on the default device: move it
            lambda x, d=d: jax.device_put(x, d) if isinstance(x, jax.Array) else x, out
        )
        jax.block_until_ready(per[d])
        del out
    tmpl = per[jax.local_devices()[0]]
    wnames = list(tmpl["weights"])

    def stack(get, expand=False):
        loc = {d: (get(o)[None] if expand else get(o)) for d, o in per.items()}
        a0 = next(iter(loc.values()))
        shape = (R * a0.shape[0],) + a0.shape[1:]
        idx = EP.addressable_devices_indices_map(shape)
        return jax.make_array_from_single_device_arrays(
            shape, EP, [loc[d] for d in idx]
        )

    return dict(
        arrays=[stack(lambda o, i=i: o["arrays"][i]) for i in range(len(tmpl["arrays"]))],
        bt_flat=stack(lambda o: o["bt_flat"]),
        weights=[stack(lambda o, k=k: o["weights"][k]) for k in wnames],
        wnames=wnames, cache_index=tmpl["cache_index"], cache_layers=tmpl["cache_layers"],
        bt_off={k: jnp.asarray(v) for k, v in tmpl["bt_off"].items()},
        bt_w={k: jnp.asarray(v) for k, v in tmpl["bt_w"].items()},
    )  # fmt: skip


if __name__ == "__main__":
    main()
