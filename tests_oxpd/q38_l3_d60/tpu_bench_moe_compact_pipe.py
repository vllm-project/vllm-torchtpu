# SPDX-License-Identifier: Apache-2.0
"""q38 L3 D60 (lever 5c): Qwen3.8-Flash's served decode MoE kernel (fused_moe_decode, folded) at 160-256 rows,
dense vs lever 5b (compact_rows 32) vs lever 5c (compact_rows 32 + compact_pipe), on one 4-chip v7x host (2x2x1, 8
TensorCores), or a deviceless compile of every program (--aot), or a CPU harness check (--interpret).

Ported from DEC-FB-2's MDC1 bench (tests/decfb/tpu_bench_moe_decode_compact.py on q38-decfb2 45c51d2ca; credit; the
bench MDC1 / MDC2 ran) with the L3 script-bench rules of L3-H8Q8's bench (launch-barrier fix and assert, src assert,
engagement from the compiled text, selftest + mutants). Geometry: TP8 attention with EP8 experts (64 local experts per
rank), hidden 2560, routed intermediate 640, a BF16 shared expert whose 80-wide slice is padded to 128, gated by
sigmoid(x . w_g) in XLA (FUSED_MOE_DECODE_KERNEL_FOLD=xla), fp8 weights with per-channel scales, the in-kernel TP
all-reduce, fold=True. Top-10 of 512 routing from random gating (near uniform), and skewed (case suffix "s": a
gating bias, hot experts, some past the compact capacity of 32 rows so they take the dense body).

Arms: dense (compact_rows None; FINAL at 160-224), cr32 (compact_rows 32; FINAL at 256), cr32p (compact_rows 32,
compact_pipe; D60). Timing (MDC1's): one jitted program chains --layers calls (call i's output, renormalised per row to
unit RMS, is call i+1's input; routing fixed), divided by the layer count; --iters samples interleave the arms with a
rotating start; a split-half figure (|median of even samples - median of odd samples|) per arm is the noise figure.
Exactness, inline: every arm's output after 1 call and after the chain is compared bit for bit with dense's; a positive
control moves one route weight and must change dense's and cr32p's 1-call outputs. Engagement: the kernel name in the
compiled text (dense: no "-cr_"; cr32: "-cr_32" and not "-cr_32p"; cr32p: "-cr_32p").

The process strips any inherited --xla_tpu_use_enhanced_launch_barrier and region-trace flag from LIBTPU_INIT_ARGS,
appends the barrier =false and asserts exactly one entry, before the imports and again at client creation.
Writes --out JSON for q38_l3_d60/d60_read.py.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time

import numpy as np

N, E_LOCAL, H, INTER, INTER_S, INTER_S_PAD, TOPK = 8, 64, 2560, 640, 80, 128, 10
ARMS = {"dense": (None, False), "cr32": (32, False), "cr32p": (32, True)}
CASES = "160,192,208,224,256,224s,256s"
LB = "--xla_tpu_use_enhanced_launch_barrier"
LB_OFF = LB + "=false"
RT = "--xla_enable_custom_call_region_trace"
_NAME = re.compile(r"fused_moe_decode_fold-e_\d+-m_(\d+)-[\w-]*")


# ---------------------------------------------------------------- libtpu barrier rule
def check_barrier(final: str) -> None:
    toks = [t for t in final.split() if t.split("=")[0] == LB]
    if toks != [LB_OFF]:
        raise SystemExit(f"Q38_BARRIER_ASSERT FAIL: launch-barrier tokens {toks} (want exactly ['{LB_OFF}'])")


def apply_cfg() -> str:
    base = os.environ.get("LIBTPU_INIT_ARGS", "")
    kept = [t for t in base.split() if t.split("=")[0] not in (LB, RT)]
    final = " ".join(kept + [LB_OFF]).strip()
    check_barrier(final)
    os.environ["LIBTPU_INIT_ARGS"] = final
    return final


def barrier_selftest() -> int:
    bad = 0
    saved = os.environ.get("LIBTPU_INIT_ARGS")
    for base in ("--a=1 " + LB + "=true --b=2", LB + "=true " + LB + "=true", "--a=1", LB_OFF, "", LB, RT + "=true"):
        os.environ["LIBTPU_INIT_ARGS"] = base
        try:
            final = apply_cfg()
            ok = [t for t in final.split() if t.split("=")[0] == LB] == [LB_OFF]
            ok &= not any(t.split("=")[0] == RT for t in final.split())
            ok &= ("--a=1" in final.split()) == ("--a=1" in base.split())
        except SystemExit:
            ok = False
        bad += not ok
    for final, want_ok in ((LB + "=true", False), (LB_OFF + " " + LB_OFF, False), ("--a=1", False),
                           (LB_OFF + " " + LB + "=true", False), (LB, False), ("--a=1 " + LB_OFF, True)):
        try:
            check_barrier(final)
            got = True
        except SystemExit:
            got = False
        print(f"Q38_BARRIER_SELFTEST check_barrier({final!r}) -> {'accept' if got else 'refuse'} "
              f"(want {'accept' if want_ok else 'refuse'})")
        bad += got != want_ok
    if saved is None:
        os.environ.pop("LIBTPU_INIT_ARGS", None)
    else:
        os.environ["LIBTPU_INIT_ARGS"] = saved
    print("Q38_BARRIER_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


# ---------------------------------------------------------------- bench rules (pure functions, selftested)
def parse_cases(spec: str):
    out = []
    for v in spec.split(","):
        tokens, skew = int(v.rstrip("s")), v.endswith("s")
        if tokens % 16 or not 16 <= tokens <= 256:
            raise ValueError(f"case {v}: tokens must be a multiple of 16 in [16, 256]")
        out.append((v, tokens, skew))
    return out


def kernel_names(txt: str):
    return sorted(set(m.group(0) for m in _NAME.finditer(txt)))


def engaged(arm: str, tokens: int, names) -> bool:
    """Exactly one fused MoE kernel name, for this bucket, in the arm's mode."""
    if len(names) != 1 or f"-m_{tokens}-" not in names[0]:
        return False
    n = names[0]
    if arm == "dense":
        return "-cr_" not in n
    if arm == "cr32":
        return n.endswith("-cr_32")
    if arm == "cr32p":
        return n.endswith("-cr_32p")
    return False


def split_half(samples):
    if len(samples) < 4:
        return None
    return abs(statistics.median(samples[0::2]) - statistics.median(samples[1::2]))


def mismatches(a: np.ndarray, b: np.ndarray) -> int:
    return int((a.view(np.uint16) != b.view(np.uint16)).sum())


def selftest() -> int:
    bad = 0

    def ck(cond, what):
        nonlocal bad
        bad += not cond
        if not cond:
            print(f"D60_BENCH_SELFTEST MISMATCH {what}")

    ck(parse_cases("160,224s") == [("160", 160, False), ("224s", 224, True)], "parse_cases")
    for v in ("100", "272", "0"):
        try:
            parse_cases(v)
            ck(False, f"parse_cases accepted {v}")
        except ValueError:
            pass
    d = "fused_moe_decode_fold-e_64-m_224-h_2560-i_640-is_128_bf16-n_8-buffers_6"
    txt = f'custom_call_target="tpu_custom_call", name="{d}" ... "{d}-cr_32" "{d}-cr_32p"'
    ck(kernel_names(txt) == sorted([d, d + "-cr_32", d + "-cr_32p"]), "kernel_names")
    ck(engaged("dense", 224, [d]) and not engaged("dense", 224, [d + "-cr_32"]), "engaged dense")
    ck(engaged("cr32", 224, [d + "-cr_32"]) and not engaged("cr32", 224, [d + "-cr_32p"]), "engaged cr32")
    ck(engaged("cr32p", 224, [d + "-cr_32p"]) and not engaged("cr32p", 224, [d + "-cr_32"]), "engaged cr32p")
    ck(not engaged("cr32p", 256, [d + "-cr_32p"]), "engaged bucket")
    ck(not engaged("dense", 224, [d, d + "-cr_32"]), "engaged two names")
    ck(not engaged("dense", 224, []), "engaged no name")
    ck(split_half([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]) == 1.0 and split_half([1.0]) is None, "split_half")
    a = np.zeros((4, 4), np.float16)
    b = a.copy()
    b[1, 2] = 1
    ck(mismatches(a, a) == 0 and mismatches(a, b) == 1, "mismatches")
    ck(mismatches(np.array([0.0], np.float16), np.array([-0.0], np.float16)) == 1, "mismatches -0")
    print("D60_BENCH_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


def mutants() -> int:
    """Each mutant weakens one bench rule; the selftest must fail under every one of them."""
    g = globals()
    orig = {k: g[k] for k in ("engaged", "mismatches", "kernel_names", "parse_cases")}
    muts = {
        "cr32p_accepts_cr32": ("engaged", lambda arm, tokens, names: (
            len(names) == 1 and ("-cr_32" in names[0] if arm != "dense" else "-cr_" not in names[0]))),
        "no_bucket_check": ("engaged", lambda arm, tokens, names: orig["engaged"](
            arm, int(re.search(r"-m_(\d+)-", names[0]).group(1)) if names else tokens, names)),
        "value_compare": ("mismatches", lambda a, b: int((a.astype(np.float32) != b.astype(np.float32)).sum())),
        "names_first_only": ("kernel_names", lambda txt: [m.group(0) for m in _NAME.finditer(txt)][:1]),
        "cases_no_range": ("parse_cases", lambda spec: [(v, int(v.rstrip("s")), v.endswith("s"))
                                                         for v in spec.split(",")]),
    }
    killed = 0
    for name, (fn, repl) in muts.items():
        g[fn] = repl
        try:
            rc = selftest()
        except Exception as exc:  # noqa: BLE001 - a selftest that crashes under a mutant did not pass
            print(f"D60_BENCH_SELFTEST raised {type(exc).__name__}: {exc}")
            rc = 1
        finally:
            g.update(orig)
        killed += rc != 0
        print(f"D60_BENCH_MUTANT {name}: {'killed' if rc else 'SURVIVED'}")
    print(f"D60_BENCH_MUTANTS {killed}/{len(muts)} killed")
    return 0 if killed == len(muts) else 1


# ---------------------------------------------------------------- programs
def _stub_vllm_logger():
    import importlib.util
    import logging
    import types
    if importlib.util.find_spec("vllm") is not None:
        return
    vllm = types.ModuleType("vllm")
    vlog = types.ModuleType("vllm.logger")
    vlog.init_logger = logging.getLogger
    vllm.logger = vlog
    sys.modules["vllm"], sys.modules["vllm.logger"] = vllm, vlog


def build_program(mesh, layers, arm, interpret=False):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod
    compact, pipe = ARMS[arm]

    def body(rank, x, w1, w2, w1s, w2s, sw1, sw2, gate_w, tw, ti):
        def one(xc):
            kw = {"compact_pipe": True} if pipe else {}
            return kmod.fused_moe_decode(
                rank, xc, w1, w2, w1s, w2s, sw1, jnp.ones((2 * INTER_S_PAD,), jnp.float32), sw2,
                jnp.ones((H,), jnp.float32), gate_w, tw, ti, n_ranks=N, expert_start="rank", fold=True,
                compact_rows=compact, interpret=interpret, **kw)

        def step(_, xc):
            y = one(xc).astype(jnp.float32)
            rms = jnp.sqrt(jnp.mean(y * y, axis=-1, keepdims=True) + 1e-12)
            return jnp.clip(y / rms, -4.0, 4.0).astype(jnp.bfloat16)

        out = lax.fori_loop(0, layers, step, x) if layers > 1 else one(x)
        return out[None]

    specs = (P("d"), P(), P("d"), P("d"), P("d"), P("d"), P("d"), P("d"), P(), P(), P())
    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False))


def abstract_args(mesh, tokens):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    S = jax.ShapeDtypeStruct
    sh, rp = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    f8 = jnp.float8_e4m3fn
    return (S((N, 2), jnp.int32, sharding=sh), S((tokens, H), jnp.bfloat16, sharding=rp),
            S((N * E_LOCAL, H, 2 * INTER), f8, sharding=sh), S((N * E_LOCAL, INTER, H), f8, sharding=sh),
            S((N * E_LOCAL, 1, 1, 2 * INTER), jnp.float32, sharding=sh), S((N * E_LOCAL, 1, 1, H), jnp.float32, sharding=sh),
            S((N * H, 2 * INTER_S_PAD), jnp.bfloat16, sharding=sh), S((N * INTER_S_PAD, H), jnp.bfloat16, sharding=sh),
            S((1, H), jnp.bfloat16, sharding=rp), S((tokens, TOPK), jnp.bfloat16, sharding=rp),
            S((tokens, TOPK), jnp.int32, sharding=rp))


def make_inputs(mesh, tokens, seed, skew):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    sh, rp = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    rng = np.random.default_rng(seed)
    total = N * E_LOCAL
    logits = rng.standard_normal((tokens, total)) * 2.0
    if skew:
        logits += -4.0 * np.arange(total)[None, :] / total
    ids = np.argsort(-logits, axis=1)[:, :TOPK]
    sel = np.take_along_axis(logits, ids, axis=1)
    tw = np.exp(sel - sel.max(axis=1, keepdims=True))
    tw /= tw.sum(axis=1, keepdims=True)
    x = jax.device_put(jnp.asarray(rng.standard_normal((tokens, H)), jnp.bfloat16), rp)

    def weights(key):
        k = jax.random.split(key, 6)
        w1 = jax.random.normal(k[0], (total, H, 2 * INTER), jnp.float32).astype(jnp.float8_e4m3fn)
        w2 = jax.random.normal(k[1], (total, INTER, H), jnp.float32).astype(jnp.float8_e4m3fn)
        s1 = ((1.0 + 0.1 * jax.random.uniform(k[2], (total, 1, 1, 2 * INTER))) / np.sqrt(H)).astype(jnp.float32)
        s2 = ((1.0 + 0.1 * jax.random.uniform(k[3], (total, 1, 1, H))) / np.sqrt(INTER)).astype(jnp.float32)
        sw1 = (jax.random.normal(k[4], (N * H, 2 * INTER_S_PAD), jnp.float32) * 0.02).astype(jnp.bfloat16)
        sw2 = (jax.random.normal(k[5], (N * INTER_S_PAD, H), jnp.float32) * 0.02).astype(jnp.bfloat16)
        return w1, w2, s1, s2, sw1, sw2

    w = jax.jit(weights, out_shardings=(sh,) * 6)(jax.random.PRNGKey(seed))
    rank = jax.device_put(np.stack([np.arange(N), np.arange(N) * E_LOCAL], 1).astype(np.int32), sh)
    gate = jax.device_put(jnp.asarray(rng.standard_normal((1, H)) * 0.02, jnp.bfloat16), rp)
    per_rank = np.bincount(ids.ravel() // E_LOCAL, minlength=N)
    counts = np.bincount(ids.ravel(), minlength=total)
    stats = {"active_per_rank": [int((counts[r * E_LOCAL:(r + 1) * E_LOCAL] > 0).sum()) for r in range(N)],
             "max_rows_per_expert": int(counts.max()), "experts_over_32": int((counts > 32).sum()),
             "routes_per_rank": [int(v) for v in per_rank]}
    return (rank, x, *w, gate, jax.device_put(jnp.asarray(tw, jnp.bfloat16), rp),
            jax.device_put(jnp.asarray(ids, jnp.int32), rp)), stats


def _commit():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    try:
        return open(os.path.join(root, "COMMIT")).read().strip()
    except OSError:
        return None


def aot_check(cases, arms, layers, out):
    import jax
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod
    devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices
    try:
        vmem = kmod._vmem_capacity()
        how = "pltpu.get_tpu_info()"
    except Exception as e:  # noqa: BLE001 - deviceless: no local TPU to ask
        vmem = 64 << 20
        how = f"fallback 64 MiB (v7x TensorCore VMEM; get_tpu_info: {type(e).__name__})"
        kmod._vmem_capacity = lambda: vmem
    print(f"D60 AOT vmem_capacity {vmem} via {how}; pltpu {pltpu.__name__}", flush=True)
    mesh = Mesh(np.array(devs[:N]), ("d",))
    bad, res = 0, {}
    for case, tokens, _skew in cases:
        if case.endswith("s"):
            continue  # same programs as the unskewed case
        for arm in arms:
            for nl in (layers, 1):
                t0 = time.time()
                key = f"{tokens} {arm} layers {nl}"
                try:
                    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                        low = build_program(mesh, nl, arm).lower(*abstract_args(mesh, tokens))
                    comp = low.compile()
                    m = comp.memory_analysis()
                    names = kernel_names(comp.as_text())
                    eng = engaged(arm, tokens, names)
                    bad += not eng
                    res[key] = {"compiled": True, "names": names, "engaged": eng,
                                "arg_mib": round(m.argument_size_in_bytes / 2**20, 1),
                                "temp_mib": round(m.temp_size_in_bytes / 2**20, 2)}
                    print(f"D60 AOT m {tokens} {arm:5s} layers {nl:3d}: COMPILED arg {res[key]['arg_mib']} MiB temp "
                          f"{res[key]['temp_mib']} MiB [{time.time() - t0:.1f} s] kernel {names} engaged {eng}",
                          flush=True)
                except Exception as e:  # noqa: BLE001 - every failure is reported and counted
                    bad += 1
                    res[key] = {"compiled": False, "error": str(e)[:3000]}
                    print(f"D60 AOT m {tokens} {arm} layers {nl}: FAILED {str(e)[:3000]}", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump({"aot": res, "bad": bad}, f, indent=1)
    print("D60 AOT PASS" if not bad else f"D60 AOT FAIL ({bad})", flush=True)
    return 1 if bad else 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--barrier-selftest"]:
        return barrier_selftest()
    if argv[:1] == ["--selftest"]:
        rc = selftest()
        return rc | (mutants() if "--mutants" in argv else 0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--arms", default="dense,cr32,cr32p")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--aot", action="store_true", help="deviceless tpu7x:2x2x1 compile of every program, no timing")
    ap.add_argument("--interpret", action="store_true", help="CPU harness check only (8 host devices)")
    ap.add_argument("--small", action="store_true", help="with --interpret: 16 local experts, hidden 1024")
    a = ap.parse_args(argv)
    if a.small:
        if not a.interpret:
            raise SystemExit("--small is a CPU harness geometry (--interpret only)")
        global E_LOCAL, H
        E_LOCAL, H = 16, 1024
    arms = a.arms.split(",")
    assert arms[0] == "dense" and set(arms) <= set(ARMS), f"arms must start with dense, from {sorted(ARMS)}"
    cases = parse_cases(a.cases)
    inherited = os.environ.get("LIBTPU_INIT_ARGS", "")
    final = apply_cfg()
    _stub_vllm_logger()
    import vllm_torchtpu
    import jax
    from jax.sharding import Mesh
    at_client = os.environ.get("LIBTPU_INIT_ARGS", "")
    check_barrier(at_client)
    want_src = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src",
                                             "vllm_torchtpu"))
    got_src = os.path.realpath(os.path.dirname(vllm_torchtpu.__file__))
    if got_src != want_src:
        raise SystemExit(f"Q38_SRC_ASSERT FAIL: vllm_torchtpu from {got_src}, want this tree's {want_src}")
    print(f"Q38_LIBTPU_FINAL {final}", flush=True)
    if a.aot:
        return aot_check(cases, arms, a.layers, a.out)
    assert a.out
    if a.interpret:
        from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod
        kmod._vmem_capacity = lambda: 64 << 20
    devs = jax.devices()  # the client is created here, after apply_cfg and the vllm_torchtpu import
    at_client2 = os.environ.get("LIBTPU_INIT_ARGS", "")
    check_barrier(at_client2)
    print(f"Q38_LIBTPU_AT_CLIENT {at_client2}", flush=True)
    host = {"node": platform.node(), "jax": jax.__version__, "devices": len(devs), "kind": devs[0].device_kind,
            "seed": a.seed, "layers": a.layers, "iters": a.iters, "argv": sys.argv, "interpret": a.interpret,
            "libtpu_inherited": inherited, "libtpu_final": final, "libtpu_at_client": at_client2,
            "commit": _commit(), "vllm_torchtpu_file": vllm_torchtpu.__file__}
    print(f"D60 host {host}", flush=True)
    mesh = Mesh(np.array(devs[:N]), ("d",))
    res = {"host": host, "arms": arms, "cases": {}, "ok": False}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def dump():
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(a.out + ".tmp", a.out)

    for case_name, tokens, skew in cases:
        args, stats = make_inputs(mesh, tokens, a.seed, skew)
        progs, out1, outN, comp_s, names, posctl = {}, {}, {}, {}, {}, {}
        tw2 = args[9].at[0, 0].set(args[9][0, 0] * 2)
        for arm in arms:
            t0 = time.time()
            cN = build_program(mesh, a.layers, arm, a.interpret).lower(*args).compile()
            c1 = build_program(mesh, 1, arm, a.interpret).lower(*args).compile()
            comp_s[arm] = round(time.time() - t0, 1)
            names[arm] = kernel_names(c1.as_text()) if not a.interpret else []
            progs[arm] = cN
            out1[arm] = np.asarray(jax.device_get(c1(*args)))
            outN[arm] = np.asarray(jax.device_get(cN(*args)))
            if arm in ("dense", "cr32p"):
                moved = np.asarray(jax.device_get(c1(*args[:9], tw2, args[10])))
                posctl[arm] = mismatches(moved, out1[arm])
        for arm in arms:
            jax.block_until_ready(progs[arm](*args))
        samples = {arm: [] for arm in arms}
        for it in range(a.iters):
            for arm in arms[it % len(arms):] + arms[:it % len(arms)]:
                t0 = time.perf_counter()
                jax.block_until_ready(progs[arm](*args))
                samples[arm].append((time.perf_counter() - t0) * 1e6 / a.layers)
        case = {"stats": stats}
        for arm in arms:
            s = sorted(samples[arm])
            u = outN[arm].view(np.uint16)
            case[arm] = {"us_per_layer_median": round(statistics.median(s), 3), "us_per_layer_min": round(s[0], 3),
                         "us_per_layer_p90": round(s[int(0.9 * (len(s) - 1))], 3),
                         "split_half": None if split_half(samples[arm]) is None else round(split_half(samples[arm]), 3),
                         "samples": [round(v, 3) for v in samples[arm]],
                         "mismatch_vs_dense_1call": mismatches(out1[arm], out1["dense"]),
                         "mismatch_vs_dense_chain": mismatches(outN[arm], outN["dense"]),
                         "mismatch_vs_cr32_1call": (mismatches(out1[arm], out1["cr32"]) if "cr32" in out1 else None),
                         "mismatch_vs_cr32_chain": (mismatches(outN[arm], outN["cr32"]) if "cr32" in outN else None),
                         "elements": int(u.size), "nonzero_out": int((u != 0).sum()),
                         "finite": not bool(((u & 0x7F80) == 0x7F80).any()), "compile_s": comp_s[arm],
                         "kernel_names": names[arm],
                         "engaged": (engaged(arm, tokens, names[arm]) if not a.interpret else None),
                         "posctl_mismatch": posctl.get(arm)}
            print(f"D60 case {case_name} {arm:5s}: median {case[arm]['us_per_layer_median']:8.2f} us/layer (min "
                  f"{case[arm]['us_per_layer_min']:.2f}, p90 {case[arm]['us_per_layer_p90']:.2f}, split-half "
                  f"{case[arm]['split_half']}) mismatch 1 {case[arm]['mismatch_vs_dense_1call']} chain "
                  f"{case[arm]['mismatch_vs_dense_chain']} of {u.size}; nonzero {case[arm]['nonzero_out']}; "
                  f"finite {case[arm]['finite']}; posctl {posctl.get(arm)}; kernel {names[arm]} engaged "
                  f"{case[arm]['engaged']}", flush=True)
        print(f"D60 case {case_name} routing {stats}", flush=True)
        res["cases"][case_name] = case
        dump()
    res["ok"] = len(res["cases"]) == len(cases)
    dump()
    print(f"D60 DONE -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
