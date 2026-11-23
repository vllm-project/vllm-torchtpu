# SPDX-License-Identifier: Apache-2.0
"""q38 DECODE-L3b-RPLEW (2026-11-19): the R form's PLE op (`pool.pooled_short_conv_speculative`, dilation 3, state_len 9,
C 10240, window 4) timed per call with arm A (RM4096's PLE flags), W (+ VLLM_TPU_PLE_WRDEFER=4, layer W) and WU (W +
VLLM_TPU_PLE_UNIFORM_COND=1, layer U), base / base2 = A alone, on one 4-chip v7x host (all local TensorCores at once, each
running the op on its own pool replica, as every decode rank runs the replicated PLE layer).

Kit pattern from DECODE-L3-INJMXU-2's tests/q38_decl3/tpu_bench_injmxu.py (credit DECODE-L3, DECODE-L3-INJMXU(-2), HCDEC-XLAHUNT
K3DX for the all-TensorCore method and the launch-barrier rule) with a new body: the PLE op, its chain, its health and its
engagement. The op and the ring / cond come from L1's PLE-WRDEFER and D07-b2 (credit L1).

Rows (T tokens) 128 / 160 / 192 / 256, count (max-num-seqs rows) 40 / 40 / 48 / 64: 128 / 160 at count 40 are RM4096's geometry
(decode MAX_NUM_SEQS 40); 192 / 48 and 256 / 64 are T90m-style geometry, not RM4096 (the smallest counts that hold uniform windows;
on RM4096 those buckets take the request path, where U falls back statically), descriptive for WU in the reader; every live
request a 4-token verify window (uniform k3), plus one
non-uniform row at 160 (one request of 1 token: pred False, the cond takes the stock branch). Pool pages of 512 tokens x
1 x 4 x 256 f8e4m3fn (the served page geometry), 2048 pages per device.

Chain: one call of the op per iteration, L calls per program (pool carried and donated, each call's output is the next call's
x), L_LONG 24 / L_SHORT 4. Each dispatch records every TensorCore's completion time (one waiter thread per TensorCore blocking
in block_until_ready, the GIL released while it waits; a TC-count check) and the wall time. A per-row watchdog
(--row-budget-s, default 1800 s) turns a stalled row into a FAIL for that row and the run moves on. Per TensorCore: per-call us = (median long - median short) / (L_LONG - L_SHORT). The figure of record per
case is the MAX over TensorCores (the served step waits for the slowest rank); the median over TensorCores and the wall-time
figure are descriptive; split halves (max over TensorCores of each half) as a noise figure (DECODE-L3b ruling 2026-11-19).
Order per row: base (A, R reps), then A / W / WU interleaved with a rotating order (R rounds), then base2 (A, R reps).
Health (before timing, every row): one short-chain call per arm from the same pool: pool bytes and output bits of W and WU
equal A's (computed on device). Pred witness: `pool.uniform_window_pred` on the row's metadata (True on uniform rows, False on
the non-uniform row). Engagement (platform independent, from each chain's jaxpr outside Pallas kernel bodies): A no
q38_ple_wrdefer call and no cond; W one q38_ple_wrdefer call and no cond; WU one q38_ple_wrdefer call and one cond. On TPU the
compiled HLO's tpu_custom_call / conditional counts are recorded as well.

Every program is lowered from CONCRETE arrays with default layouts and only the compiled executable is called (`--cpu-real`
runs this same path on CPU devices: Pallas in TPU interpret mode with a value-level typed load / store stand-in, SMALL
geometry; the CDK CPU proof's P8 leg). Modes: (default) TPU timing; --cpu-real; --selftest [--mutants]; --barrier-selftest.
Writes --out JSON for q38_decl3b/rple_read.py.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time

try:
    import numpy as np
except ImportError:  # selftests run without numpy
    np = None

LB = "--xla_tpu_use_enhanced_launch_barrier"
LB_OFF = LB + "=false"
RT = "--xla_enable_custom_call_region_trace"
VMEM_64MIB = 67108864
CFG_ADD = [f"--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes={VMEM_64MIB}",
           f"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes={VMEM_64MIB}",
           "--xla_tpu_use_dynamic_smem_negotiation=true"]
ROWS = (128, 160, 192, 256)
COUNTS = {128: 40, 160: 40, 192: 48, 256: 64}
NONUNIFORM_ROW = 160
ARMS = ("A", "W", "WU")
CASES = ("base", "A", "W", "WU", "base2")
FLAG_W, FLAG_U = "VLLM_TPU_PLE_WRDEFER", "VLLM_TPU_PLE_UNIFORM_COND"
HRK, CHUNK, SKIP = "VLLM_TPU_PLE_HISTORY_ROW_KERNEL", "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS", "VLLM_TPU_PLE_SKIP_IDLE_ROWS"
COMMON_ENV = {HRK: "1", CHUNK: "512"}  # RM4096's decode-leg PLE flags (render f814bbb0); nothing else
ARM_ENV = {"A": {}, "W": {FLAG_W: "4"}, "WU": {FLAG_W: "4", FLAG_U: "1"}}
WINDOW, STATE_LEN, DILATION, TAPS = 4, 9, 3, 4
L_LONG, L_SHORT, REPS = 24, 4, 7
FLASH = dict(C=10240, BLOCK=512, PAGES=2048)
SMALL = dict(C=256, BLOCK=18, PAGES=0)  # --cpu-real; PAGES 0 = count * window + 8
RING, COND = "q38_ple_wrdefer", "cond"
ENGAGE = {"A": {RING: 0, COND: 0}, "W": {RING: 1, COND: 0}, "WU": {RING: 1, COND: 1}}


# ---------------------------------------------------------------- libtpu (ARB1 / K3DX rule, copied from the INJMXU bench)
def check_barrier(final: str) -> None:
    toks = [t for t in final.split() if t.split("=")[0] == LB]
    if toks != [LB_OFF]:
        raise SystemExit(f"Q38_BARRIER_ASSERT FAIL: launch-barrier tokens {toks} (want exactly ['{LB_OFF}'])")


def apply_cfg() -> str:
    base = os.environ.get("LIBTPU_INIT_ARGS", "")
    names = {a.split("=")[0] for a in CFG_ADD} | {LB, RT}
    kept = [t for t in base.split() if t.split("=")[0] not in names]
    final = " ".join(kept + CFG_ADD + [LB_OFF]).strip()
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
            ok &= all(a in final.split() for a in CFG_ADD)
        except SystemExit:
            ok = False
        bad += not ok
    for final, want in ((LB + "=true", False), (LB_OFF + " " + LB_OFF, False), ("--a=1", False),
                        (LB_OFF + " " + LB + "=true", False), (LB, False), ("--a=1 " + LB_OFF, True)):
        try:
            check_barrier(final)
            got = True
        except SystemExit:
            got = False
        print(f"Q38_BARRIER_SELFTEST check_barrier({final!r}) -> {'accept' if got else 'refuse'} "
              f"(want {'accept' if want else 'refuse'})")
        bad += got != want
    if saved is None:
        os.environ.pop("LIBTPU_INIT_ARGS", None)
    else:
        os.environ["LIBTPU_INIT_ARGS"] = saved
    print("Q38_BARRIER_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


# ---------------------------------------------------------------- pure helpers (selftested)
def per_call_us(long_s, short_s, l_long=L_LONG, l_short=L_SHORT):
    """(median long - median short) / (l_long - l_short), in us."""
    if not long_s or not short_s:
        return None
    return (statistics.median(long_s) - statistics.median(short_s)) / (l_long - l_short) * 1e6


def halves(long_s, short_s, l_long=L_LONG, l_short=L_SHORT):
    h = len(long_s) // 2
    if h < 1 or len(short_s) // 2 < 1:
        return None
    hs = len(short_s) // 2
    return [per_call_us(long_s[:h], short_s[:hs], l_long, l_short), per_call_us(long_s[h:], short_s[hs:], l_long, l_short)]


def combine_tc(long_tc, short_tc, l_long=L_LONG, l_short=L_SHORT):
    """Per-TensorCore per-call us from per-rep, per-device completion times (seconds after the dispatch):
    -> (max over TensorCores = the figure of record, since the served step waits for the slowest rank; median over
    TensorCores, descriptive; the per-TensorCore list). None when a device has no samples."""
    if not long_tc or not short_tc:
        return None, None, None
    n = len(long_tc[0])
    per = []
    for i in range(n):
        v = per_call_us([r[i] for r in long_tc], [r[i] for r in short_tc], l_long, l_short)
        if v is None:
            return None, None, None
        per.append(v)
    return max(per), statistics.median(per), per


class RowWatchdog(Exception):
    """A row's work outlived its budget (--row-budget-s): the row reads FAIL and the run moves on."""


def wait_all(p_shards, x_shards, t0, deadline, n_expected):
    """One waiter thread per TensorCore: each blocks in block_until_ready() (the GIL is released while it waits) on that
    TensorCore's output shards and records its completion time (seconds after the dispatch at t0). Raises RowWatchdog if
    any waiter is still blocked at ``deadline`` (time.perf_counter() clock) and ValueError if fewer than n_expected
    TensorCores reported (the TC-count check)."""
    done = [None] * len(p_shards)

    def waiter(i):
        p_shards[i].data.block_until_ready()
        x_shards[i].data.block_until_ready()
        done[i] = time.perf_counter() - t0

    threads = [threading.Thread(target=waiter, args=(i,), daemon=True) for i in range(len(p_shards))]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=max(0.0, deadline - time.perf_counter()))
    if any(th.is_alive() for th in threads):
        raise RowWatchdog(f"RPLE ROW WATCHDOG: {sum(th.is_alive() for th in threads)} TensorCore waiter(s) still blocked "
                          f"at the row deadline")
    got = [d for d in done if isinstance(d, float)]
    if len(done) != n_expected or len(got) != n_expected:
        raise ValueError(f"RPLE TC COUNT: {len(got)} of {n_expected} TensorCores reported a completion time")
    return done


def rotation(rnd: int, arms=ARMS):
    k = rnd % len(arms)
    return arms[k:] + arms[:k]


def engaged(arm: str, counts: dict) -> bool:
    want = ENGAGE[arm]
    return all(counts.get(k) == v for k, v in want.items())


def metadata(T: int, count: int, uniform: bool):
    """Per-request token counts of a row: every live request a 4-token window (uniform), or the same with request 1 holding
    1 token (non-uniform: pred False)."""
    live = min(T // WINDOW, count)
    lengths = [WINDOW] * live + [0] * (count - live)
    if not uniform:
        lengths[1] = 1
    return lengths


def plan(rows=ROWS):
    return {"rows": list(rows), "counts": {str(r): COUNTS[r] for r in rows}, "arms": list(ARMS), "cases": list(CASES),
            "nonuniform_row": NONUNIFORM_ROW, "chain": [L_LONG, L_SHORT], "reps": REPS,
            "common_env": COMMON_ENV, "arm_env": ARM_ENV}


def _checks(verbose=True) -> int:
    bad = 0

    def chk(name, cond):
        nonlocal bad
        if verbose:
            print(f"Q38_RPLE_SELFTEST {name}: {'ok' if cond else 'BAD'}")
        bad += not cond

    g = globals()
    chk("per_call_us", abs(g["per_call_us"]([2.4e-3] * 3, [0.4e-3] * 3, 24, 4) - 100.0) < 1e-6)
    chk("per_call_us empty", g["per_call_us"]([], [1.0]) is None)
    hv = g["halves"]([2.4e-3, 2.4e-3, 4.4e-3, 4.4e-3], [0.4e-3] * 4, 24, 4)
    chk("halves", hv is not None and [round(x, 6) for x in hv] == [100.0, 200.0])
    mx, md, per = g["combine_tc"]([[2.4e-3, 4.4e-3], [2.4e-3, 4.4e-3], [2.4e-3, 4.4e-3]],
                                  [[0.4e-3, 0.4e-3]] * 3, 24, 4)
    chk("combine_tc max / median", per is not None and [round(x, 6) for x in per] == [100.0, 200.0]
        and round(mx, 6) == 200.0 and round(md, 6) == 150.0)
    chk("combine_tc empty", g["combine_tc"]([], [[1.0]]) == (None, None, None))

    class _D:
        def __init__(self, delay):
            self.delay = delay

        def block_until_ready(self):
            time.sleep(self.delay)

    class _S:
        def __init__(self, delay):
            self.data = _D(delay)

    try:
        done = g["wait_all"]([_S(0.001 * i) for i in range(8)], [_S(0.0)] * 8, time.perf_counter(), time.perf_counter() + 5, 8)
        chk("wait_all 8 TensorCores", len(done) == 8 and all(isinstance(d, float) for d in done) and done[7] >= done[0])
    except Exception as e:  # noqa: BLE001
        chk(f"wait_all 8 TensorCores ({type(e).__name__})", False)
    try:
        g["wait_all"]([_S(5.0)] + [_S(0.0)] * 7, [_S(0.0)] * 8, time.perf_counter(), time.perf_counter() + 0.2, 8)
        chk("wait_all watchdog", False)
    except RowWatchdog:
        chk("wait_all watchdog", True)
    except Exception:  # noqa: BLE001
        chk("wait_all watchdog", False)
    chk("rotation", [g["rotation"](i) for i in range(4)] == [("A", "W", "WU"), ("W", "WU", "A"), ("WU", "A", "W"),
                                                             ("A", "W", "WU")])
    eg = g["engaged"]
    chk("engaged A", eg("A", {RING: 0, COND: 0}) and not eg("A", {RING: 1, COND: 0}) and not eg("A", {RING: 0, COND: 1}))
    chk("engaged W", eg("W", {RING: 1, COND: 0}) and not eg("W", {RING: 1, COND: 1}) and not eg("W", {RING: 0, COND: 0}))
    chk("engaged WU", eg("WU", {RING: 1, COND: 1}) and not eg("WU", {RING: 1, COND: 0}) and not eg("WU", {RING: 0, COND: 1}))
    md = g["metadata"]
    chk("metadata uniform", md(160, 40, True) == [4] * 40 and md(128, 40, True) == [4] * 32 + [0] * 8
        and md(256, 64, True) == [4] * 64)
    chk("metadata nonuniform", md(160, 40, False)[:3] == [4, 1, 4] and sum(md(160, 40, False)) == 157)
    chk("counts hold uniform windows", all(COUNTS[r] * WINDOW >= r for r in ROWS) and COUNTS[128] == COUNTS[160] == 40)
    chk("arm env", ARM_ENV["A"] == {} and ARM_ENV["W"] == {FLAG_W: "4"} and ARM_ENV["WU"] == {FLAG_W: "4", FLAG_U: "1"}
        and COMMON_ENV == {HRK: "1", CHUNK: "512"})
    return bad


_MUTANTS = {
    "per_call ignores the short chain": ("per_call_us", lambda l, s, a=L_LONG, b=L_SHORT:
                                         (statistics.median(l) / a * 1e6) if l and s else None),
    "per_call uses the mean of the two chains": ("per_call_us", lambda l, s, a=L_LONG, b=L_SHORT:
                                                 ((statistics.median(l) + statistics.median(s)) / (a + b) * 1e6)
                                                 if l and s else None),
    "rotation never moves": ("rotation", lambda rnd, arms=ARMS: arms),
    "engagement ignores the cond": ("engaged", lambda arm, counts: counts.get(RING) == ENGAGE[arm][RING]),
    "engagement ignores the ring": ("engaged", lambda arm, counts: counts.get(COND) == ENGAGE[arm][COND]),
    "nonuniform row is uniform": ("metadata", lambda T, count, uniform: [WINDOW] * min(T // WINDOW, count)
                                  + [0] * (count - min(T // WINDOW, count))),
    "halves swapped": ("halves", lambda l, s, a=L_LONG, b=L_SHORT: None),
    "one TensorCore waiter dropped": ("wait_all", lambda ps, xs, t0, dl, n: [time.perf_counter() - t0
                                                                             for i in range(len(ps) - 1)]),
    "figure of record is the median over TensorCores": ("combine_tc", lambda lt, st, a=L_LONG, b=L_SHORT:
        (lambda per: (statistics.median(per), statistics.median(per), per) if per else (None, None, None))(
            [per_call_us([r[i] for r in lt], [r[i] for r in st], a, b) for i in range(len(lt[0]))] if lt and st else [])),
}


def selftest(mutants: bool) -> int:
    bad = _checks()
    if mutants:
        killed = 0
        for name, (fn, broken) in _MUTANTS.items():
            real = globals()[fn]
            globals()[fn] = broken
            try:
                caught = _checks(verbose=False) > 0
            finally:
                globals()[fn] = real
            killed += caught
            print(f"Q38_RPLE_SELFTEST MUTANT {name}: {'killed' if caught else 'SURVIVED'}")
        print(f"Q38_RPLE_SELFTEST MUTANTS {killed}/{len(_MUTANTS)} killed")
        bad += killed != len(_MUTANTS)
    print("Q38_RPLE_SELFTEST", "PASS" if not bad else f"FAIL ({bad})")
    return 0 if not bad else 1


# ---------------------------------------------------------------- device harness
def _value_store(block_ref, values, *, lane_split=1):
    import jax
    import jax.numpy as jnp

    del lane_split
    u = jnp.swapaxes(jax.lax.bitcast_convert_type(values, jnp.uint8), -1, -2)
    block_ref[...] = jax.lax.bitcast_convert_type(u, block_ref.dtype).reshape(block_ref.shape)


def _value_load(block_ref, *, view_dtype, lane_split=1):
    import jax
    import jax.numpy as jnp

    raw = block_ref[...]
    lanes = raw.shape[-1]
    if jnp.dtype(raw.dtype) == jnp.dtype(view_dtype):
        out = raw.reshape(-1, lanes)
    else:
        ratio = jnp.dtype(view_dtype).itemsize // jnp.dtype(raw.dtype).itemsize
        u = jax.lax.bitcast_convert_type(raw, jnp.uint8).reshape(-1, ratio, lanes)
        out = jax.lax.bitcast_convert_type(jnp.swapaxes(u, -1, -2), view_dtype)
    return out.reshape(-1, lanes // lane_split)


def _cpu_interpret():
    """--cpu-real: Pallas in TPU interpret mode (on_wait DMAs) and a value-level typed load / store (the interpreter cannot
    run Mosaic's ref bitcast); the stand-in is used by every arm alike."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels import typed_ldst

    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    real = pl.pallas_call
    pl.pallas_call = lambda *a, **k: real(*a, **{**k, "interpret": params})
    typed_ldst.store_typed = _value_store
    typed_ldst.load_typed = _value_load


def _walk(jaxpr):
    for e in jaxpr.eqns:
        yield e
        if e.primitive.name == "pallas_call":
            continue
        for p in e.params.values():
            for sub in (p if isinstance(p, (list, tuple)) else [p]):
                if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    yield from _walk(sub.jaxpr)
                elif hasattr(sub, "eqns"):
                    yield from _walk(sub)


def jaxpr_counts(closed) -> dict:
    eqns = list(_walk(closed.jaxpr))
    ring = sum(1 for e in eqns if e.primitive.name == "pallas_call"
               and RING in str(e.params.get("name_and_src_info", e.params.get("name", ""))))
    conds = sum(1 for e in eqns if e.primitive.name == "cond")
    return {RING: ring, COND: conds}


def hlo_counts(text: str) -> dict:
    lines = text.splitlines()
    return {"tpu_custom_calls": sum('custom_call_target="tpu_custom_call"' in ln for ln in lines),
            "ring_custom_calls": sum('custom_call_target="tpu_custom_call"' in ln and RING in ln for ln in lines),
            "conditionals": sum(" conditional(" in ln for ln in lines)}


def set_arm(arm: str) -> None:
    for k in (FLAG_W, FLAG_U, SKIP):
        os.environ.pop(k, None)
    os.environ.update(COMMON_ENV)
    os.environ.update(ARM_ENV[arm])


def build_inputs(T, count, geo, n_dev, uniform, seed):
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    C, block = geo["C"], geo["BLOCK"]
    pages = geo["PAGES"] or count * WINDOW + count + 8
    lengths = metadata(T, count, uniform)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    perm = rng.permutation(np.arange(1, pages))
    ckpt = perm[: count * WINDOW].reshape(count, WINDOW).astype(np.int32)
    ids = perm[count * WINDOW: count * WINDOW + count].astype(np.int32)
    read_ids = ckpt[np.arange(count), rng.integers(0, WINDOW, size=count)].astype(np.int32)
    one = dict(
        x=rng.normal(size=(T, C)).astype(np.float32),
        weight=(0.5 * rng.normal(size=(C, TAPS))).astype(np.float32),
        starts=starts, ids=ids, initial=np.ones(count, bool), read_ids=read_ids, ckpt_ids=ckpt,
        windowed=np.ones(count, bool),
    )
    def rep(a, dtype=None):
        a = np.broadcast_to(a, (n_dev,) + a.shape)
        return jnp.asarray(a, dtype) if dtype is not None else jnp.asarray(a)

    host = [rep(one["x"], jnp.bfloat16), rep(one["weight"], jnp.bfloat16), None, rep(one["starts"]), rep(one["ids"]),
            rep(one["initial"]), rep(one["read_ids"]), rep(one["ckpt_ids"]), rep(one["windowed"])]
    shape = (n_dev, pages, block, 1, 4, 256)
    return host, shape, lengths


def make_pool(shape, shard, seed):
    """A fresh pool on the devices (never on the host): bytes in [0, 64), so every bf16 the history read forms from two
    of them is finite and below 2 in magnitude (no NaN payloads for the health compare to trip on)."""
    import jax
    import jax.numpy as jnp

    def f(key):
        raw = jax.random.bits(key, shape, jnp.uint8) & jnp.uint8(0x3F)
        return jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)

    return jax.jit(f, out_shardings=shard)(jax.random.key(seed))


def run(args) -> int:
    t_start = time.time()
    cpu = args.cpu_real
    if not cpu:
        final = apply_cfg()
    else:
        final = os.environ.get("LIBTPU_INIT_ARGS", "")
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    if cpu:
        _cpu_interpret()
    from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod

    geo = SMALL if cpu else FLASH
    l_long, l_short, reps = (args.l_long, args.l_short, args.reps)
    devs = jax.devices()
    n_dev = len(devs) if args.devices <= 0 else args.devices
    devs = devs[:n_dev]
    mesh = Mesh(np.array(devs), ("d",))
    shard = NamedSharding(mesh, P("d"))
    rows = tuple(int(r) for r in args.rows.split(","))
    out = {"plan": plan(rows), "host": {"platform": platform.node(), "backend": jax.default_backend(), "devices": n_dev,
                                        "device_kind": devs[0].device_kind, "chain": [l_long, l_short], "reps": reps,
                                        "warmup": args.warmup, "row_budget_s": args.row_budget_s,
                                        "geometry": geo, "cpu_real": cpu, "libtpu_final": final,
                                        "pool_file": os.path.realpath(pool_mod.__file__)},
           "cases": {}, "health": {}, "engagement": {}, "pred": {}, "errors": []}
    print(f"RPLE host {out['host']}", flush=True)

    def op(x, w, pool, starts, ids, initial, read_ids, ckpt, windowed):
        return pool_mod.pooled_short_conv_speculative(x, w, pool, starts, ids, initial, read_ids, ckpt, windowed,
                                                      dilation=DILATION, state_len=STATE_LEN)

    def chain_fn(length):
        def per_dev(x, w, pool, starts, ids, initial, read_ids, ckpt, windowed):
            sq = [a[0] for a in (x, w, starts, ids, initial, read_ids, ckpt, windowed)]
            x0, w0, s0, i0, n0, r0, c0, wd0 = sq

            def body(_, carry):
                pool_c, xc = carry
                pool_n, out_n = op(xc, w0, pool_c[0], s0, i0, n0, r0, c0, wd0)
                return pool_n[None], out_n

            pool_o, x_o = jax.lax.fori_loop(0, length, body, (pool, x0))
            return pool_o, x_o[None]

        f = jax.shard_map(per_dev, mesh=mesh, in_specs=(P("d"),) * 9, out_specs=(P("d"), P("d")), check_vma=False)
        return jax.jit(f, donate_argnums=(2,))

    def put(a):
        return jax.device_put(a, shard)

    def same_fn():
        def f(pa, pb, xa, xb):
            ua = jax.lax.bitcast_convert_type(pa, jnp.uint8)
            ub = jax.lax.bitcast_convert_type(pb, jnp.uint8)
            return jnp.sum(ua != ub), jnp.sum(jax.lax.bitcast_convert_type(xa, jnp.uint16)
                                                != jax.lax.bitcast_convert_type(xb, jnp.uint16))
        return jax.jit(f)

    same = same_fn()
    row_list = [(r, True) for r in rows] + ([(NONUNIFORM_ROW, False)] if NONUNIFORM_ROW in rows else [])
    for T, uniform in row_list:
        row_deadline = time.perf_counter() + args.row_budget_s  # the per-row watchdog (compile is not interruptible)
        key = str(T) if uniform else f"{T}nu"
        count = COUNTS[T]
        seed = T + (0 if uniform else 1)
        host, pshape, lengths = build_inputs(T, count, geo, n_dev, uniform, seed=seed)
        args_d = [put(a) if a is not None else None for a in host]

        def pool0():
            return make_pool(pshape, shard, seed)

        compiled, eng = {}, {}
        for arm in ARMS:
            set_arm(arm)
            for length in (l_long, l_short):
                fn = chain_fn(length)  # a fresh jit per arm: the flags are read at trace time
                pool_in = pool0()
                a_in = list(args_d)
                a_in[2] = pool_in
                try:
                    if length == l_short:
                        eng[arm] = jaxpr_counts(jax.make_jaxpr(lambda *z: fn(*z))(*a_in))
                    lowered = fn.lower(*a_in)
                    exe = lowered.compile()
                    compiled[(arm, length)] = exe
                    if not cpu and length == l_long:
                        eng.setdefault(arm + "_hlo", hlo_counts(exe.as_text()))
                except Exception as e:  # noqa: BLE001
                    out["errors"].append(f"{key} {arm} L{length}: {type(e).__name__}: {str(e)[:800]}")
                    print(f"RPLE COMPILE ERROR {key} {arm} L{length}: {type(e).__name__}: {str(e)[:800]}", flush=True)
        out["engagement"][key] = {arm: {"compiled": (arm, l_long) in compiled and (arm, l_short) in compiled,
                                        "counts": eng.get(arm), "hlo": eng.get(arm + "_hlo"),
                                        "engaged": bool(eng.get(arm)) and engaged(arm, eng[arm])} for arm in ARMS}
        set_arm("A")
        cut = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32))
        out["pred"][key] = bool(jax.jit(lambda s: pool_mod.uniform_window_pred(s, T // WINDOW, WINDOW))(cut)) \
            if hasattr(pool_mod, "uniform_window_pred") else None
        print(f"RPLE row {key}: count {count} engagement {json.dumps(out['engagement'][key])} pred {out['pred'][key]}",
              flush=True)
        if not all(out["engagement"][key][a]["compiled"] for a in ARMS):
            out["health"][key] = {"ok": False, "why": "not compiled"}
            continue
        try:
            if time.perf_counter() > row_deadline:
                raise RowWatchdog("RPLE ROW WATCHDOG: the row's budget ran out before health")
            # health: one short chain per arm from the same pool
            res = {}
            for arm in ARMS:
                a_in = list(args_d)
                a_in[2] = pool0()
                res[arm] = compiled[(arm, l_short)](*a_in)
                jax.block_until_ready(res[arm])
            diffs = {arm: [int(v) for v in same(res["A"][0], res[arm][0], res["A"][1], res[arm][1])] for arm in ("W", "WU")}
            ok = all(d == [0, 0] for d in diffs.values())
            out["health"][key] = {"ok": ok, "pool_and_output_diff_vs_A": diffs}
            print(f"RPLE health {key}: {out['health'][key]}", flush=True)
            del res
            # timing
            state = {arm: pool0() for arm in ARMS}
            samples = {c: {"long": [], "short": [], "long_tc": [], "short_tc": []} for c in CASES}

            def timed(arm, case):
                """One dispatch per chain length: the wall time (until every TensorCore is done) and each TensorCore's
                completion time, read by polling its output shards' readiness."""
                for length, tag in ((l_long, "long"), (l_short, "short")):
                    a_in = list(args_d)
                    a_in[2] = state[arm]
                    t0 = time.perf_counter()
                    p_out, x_out = compiled[(arm, length)](*a_in)
                    ps = sorted(p_out.addressable_shards, key=lambda s: s.device.id)
                    xs = sorted(x_out.addressable_shards, key=lambda s: s.device.id)
                    done = wait_all(ps, xs, t0, row_deadline, n_dev)
                    jax.block_until_ready((p_out, x_out))
                    samples[case][tag].append(time.perf_counter() - t0)
                    samples[case][tag + "_tc"].append(done)
                    state[arm] = p_out

            for arm in ARMS:  # warm-up
                for _ in range(args.warmup):
                    for length in (l_long, l_short):
                        a_in = list(args_d)
                        a_in[2] = state[arm]
                        p_out, _x = compiled[(arm, length)](*a_in)
                        jax.block_until_ready(p_out)
                        state[arm] = p_out
            for _ in range(reps):
                timed("A", "base")
            for rnd in range(reps):
                for arm in rotation(rnd):
                    timed(arm, arm)
            for _ in range(reps):
                timed("A", "base2")
            cases = {}
            for c in CASES:
                lt, st = samples[c]["long_tc"], samples[c]["short_tc"]
                mx, md, per = combine_tc(lt, st, l_long, l_short)
                h = len(lt) // 2
                halves_tc = ([combine_tc(lt[:h], st[:h], l_long, l_short)[0], combine_tc(lt[h:], st[h:], l_long, l_short)[0]]
                             if h >= 1 else None)
                cases[c] = {"us_per_call": mx, "us_per_call_median_tc": md, "us_per_tc": per, "halves": halves_tc,
                            "us_per_call_wall": per_call_us(samples[c]["long"], samples[c]["short"], l_long, l_short),
                            "long_s": samples[c]["long"], "short_s": samples[c]["short"],
                            "long_tc_s": lt, "short_tc_s": st}
            out["cases"][key] = cases
            print("RPLE timing " + key + " max-over-TC " + json.dumps({c: cases[c]["us_per_call"] for c in CASES})
                  + " median-TC " + json.dumps({c: cases[c]["us_per_call_median_tc"] for c in CASES}), flush=True)
            del state, compiled
        except Exception as e:  # noqa: BLE001  (one row failing must not lose the other rows; the reader VOIDs it)
            out["errors"].append(f"{key} run: {type(e).__name__}: {str(e)[:800]}")
            out["health"].setdefault(key, {"ok": False, "why": "run error"})
            print(f"RPLE RUN ERROR {key}: {type(e).__name__}: {str(e)[:800]}", flush=True)
    out["host"]["wall_s"] = round(time.time() - t_start, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    missing = [f"{c}@{k}" for k in [str(r) for r in rows] + ([f"{NONUNIFORM_ROW}nu"] if NONUNIFORM_ROW in rows else [])
               for c in CASES if not isinstance(((out["cases"].get(k) or {}).get(c) or {}).get("us_per_call"), float)]
    print(f"RPLE done wall {out['host']['wall_s']}s errors {len(out['errors'])} missing timing cases {missing}", flush=True)
    return 0 if not out["errors"] and not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS))
    ap.add_argument("--out", default="rple.json")
    ap.add_argument("--devices", type=int, default=0, help="0 = every local device")
    ap.add_argument("--l-long", type=int, default=L_LONG)
    ap.add_argument("--l-short", type=int, default=L_SHORT)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--row-budget-s", type=float, default=1800.0,
                    help="per-row watchdog: a row (health + timing; compile is not interruptible) still running this many "
                         "seconds after its start reads FAIL (error 'RPLE ROW WATCHDOG') and the run moves on")
    ap.add_argument("--cpu-real", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--barrier-selftest", action="store_true")
    a = ap.parse_args()
    if a.barrier_selftest:
        return barrier_selftest()
    if a.selftest:
        return selftest(a.mutants)
    for r in a.rows.split(","):
        if int(r) not in COUNTS:
            raise SystemExit(f"row {r} not planned")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
