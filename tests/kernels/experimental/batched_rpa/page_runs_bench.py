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
"""Verify-window attention with one DMA per KV page vs one per page run, one core.

    PYTHONPATH=src python3 page_runs_bench.py check|time|trace [arm ...] [--out DIR]

Shape: 26 sequences of 131072-132095 cached tokens plus 4 new ones, 16 query
heads per KV head, head_dim 256, FP8 KV in 128-token SEQ_ALONG_LANE pages,
a 1M-token page table (8194 columns, so the kernel's compact table), block
(bq 4, bkv 3072, 8 lanes, 3 buffers). Pages come from 2176-token manager
blocks (17 pages) of a 1744-block pool, in shuffled order.

Arms: `base` (kv_run_pages unset), `runs` (17), `runs1` (1: the page-major
window with one DMA per page). `s:<arm>` adds named scopes around the stitch,
QK-softmax and PV for a body breakdown.

- check: outputs and KV cache of each arm vs `base`, bitwise.
- time: device time per call (a loop of N calls minus one call), alternating
  arm order over rounds. Includes the schedule kernel.
- trace: one profile per arm (run with
  LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true). Per decode
  kernel: time, pipeline scopes, and "outside" = kernel time minus the
  outermost ep_* scopes (where the next blocks' copies are issued). Prints the
  verdict line for `runs` vs `base`.
"""

import argparse
import bisect
import collections
import functools
import glob
import json
import os
import re
import shutil
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    flash_attention,
    stitch_utils,
    wrapper,
)

SEQS, NEW, PAGE, BLOCK_PAGES, POOL_BLOCKS, WIDTH = 26, 4, 128, 17, 1744, 8194
HQ, HKV, HD = 16, 1, 256
FP8 = jnp.dtype("float8_e4m3fn")
BLOCK = configs.BlockSizes(bq_sz=4, bq_c_sz=4, bkv_sz=3072, batch_size=8, n_buffer=3)
ARMS = {"base": None, "runs": 17, "runs1": 1}
GO_OUTSIDE_DROP_US, GO_BODY_GROWTH_US = 50.0, 10.0


def log(msg):
    print(msg, flush=True)


def _cdiv(a, b):
    return -(-a // b)


def make_case(seed=0):
    rng = np.random.default_rng(seed)
    cached = rng.integers(131072, 132096, SEQS)
    kv_lens = cached + NEW
    blocks = [_cdiv(_cdiv(int(n), PAGE), BLOCK_PAGES) for n in kv_lens]
    order = rng.permutation(np.arange(1, POOL_BLOCKS, dtype=np.int32))
    table = np.zeros((SEQS, WIDTH), np.int32)
    start = 0
    for s, n in enumerate(blocks):
        ids = order[start : start + n]
        start += n
        pages = (ids[:, None] * BLOCK_PAGES + np.arange(BLOCK_PAGES)).ravel()
        table[s, : len(pages)] = pages
    shape = wrapper.get_kv_cache_shape(
        POOL_BLOCKS * BLOCK_PAGES,
        PAGE,
        HKV,
        HD,
        FP8,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    keys = jax.random.split(jax.random.key(seed), 4)
    tokens = SEQS * NEW
    cache = jax.random.normal(keys[0], shape, jnp.float32).astype(FP8)
    inputs = (
        (jax.random.normal(keys[1], (tokens, HQ, HD)) * 2).astype(jnp.bfloat16),
        jax.random.normal(keys[2], (tokens, HKV, HD)).astype(FP8),
        jax.random.normal(keys[3], (tokens, HKV, HD)).astype(FP8),
        jnp.asarray(kv_lens, jnp.int32),
        jnp.asarray(table.ravel()),
        jnp.arange(0, tokens + 1, NEW, dtype=jnp.int32),
        jnp.array([SEQS] * 3, jnp.int32),
    )
    return cache, inputs, cached


def schedule_stats(cached):
    """Blocks, steps, and steps holding a lane that is not a full cache block
    (these copy pages with dynamic sizes and fetch new tokens)."""
    full = []
    for c in cached:
        n = _cdiv(int(c) + NEW, BLOCK.bkv_sz)
        full += [(k + 1) * BLOCK.bkv_sz <= c for k in range(n)]
    cells = len(full)
    steps = _cdiv(cells, BLOCK.batch_size)
    full += [False] * (steps * BLOCK.batch_size - cells)
    lanes = BLOCK.batch_size
    masked = sum(not all(full[i : i + lanes]) for i in range(0, len(full), lanes))
    return {"blocks": cells, "steps": steps, "masked_steps": masked}


def _scoped(fn, name):
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        with jax.named_scope(name):
            return fn(*args, **kwargs)

    return inner


_ORIG = {
    (stitch_utils, "stitch_new_kv_lane"): stitch_utils.stitch_new_kv_lane,
    (stitch_utils, "store_new_kv_lane"): stitch_utils.store_new_kv_lane,
    (flash_attention, "flash_attention_qk_softmax"): (
        flash_attention.flash_attention_qk_softmax
    ),
    (flash_attention, "flash_attention_pv"): flash_attention.flash_attention_pv,
}
_SCOPES = {
    "stitch_new_kv_lane": "pr_stitch",
    "store_new_kv_lane": "pr_store",
    "flash_attention_qk_softmax": "pr_qk_softmax",
    "flash_attention_pv": "pr_pv",
}


def set_scopes(on):
    for (mod, attr), fn in _ORIG.items():
        setattr(mod, attr, _scoped(fn, _SCOPES[attr]) if on else fn)


def arm_runs(arm):
    return ARMS[arm.removeprefix("s:")]


def call(arm, cache, inputs):
    q, k, v, kv_lens, table, cu, dist = inputs
    return wrapper.ragged_paged_attention(
        q,
        k,
        v,
        cache,
        kv_lens,
        table,
        cu,
        dist,
        sm_scale=1 / 16,
        k_scale=1.0,
        v_scale=1.0,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=NEW,
        decode_block_sizes=BLOCK,
        kv_run_pages=arm_runs(arm),
    )


PROGS = {}


def prog(arm, cache, inputs):
    """AOT program: a loop of `count` calls (traced count), donating the cache."""
    if arm not in PROGS:
        jax.clear_caches()
        set_scopes(arm.startswith("s:"))

        def run(cache, count, inputs):
            body = lambda i, c: call(arm, c, inputs)[1]  # noqa: E731
            return jax.lax.fori_loop(0, count, body, cache)

        t0 = time.perf_counter()
        PROGS[arm] = (
            jax.jit(run, donate_argnums=(0,))
            .lower(cache, jnp.int32(1), inputs)
            .compile()
        )
        set_scopes(False)
        log(f"COMPILED {arm} s={time.perf_counter() - t0:.1f}")
    return PROGS[arm]


def timed(arm, state, n):
    p = prog(arm, state["cache"], state["inputs"])
    one, many = jnp.int32(1), jnp.int32(n + 1)
    cache = p(state["cache"], many, state["inputs"])
    jax.block_until_ready(cache)
    t = []
    for c in (one, many, one, many):
        t0 = time.perf_counter()
        cache = p(cache, c, state["inputs"])
        jax.block_until_ready(cache)
        t.append(time.perf_counter() - t0)
    state["cache"] = cache
    return ((t[1] + t[3]) - (t[0] + t[2])) / 2 * 1e6 / n


def do_check(arms, out):
    cache, inputs, _ = make_case()
    rows = []

    def bits(x):
        x = np.asarray(jax.device_get(x))
        return x.view(np.uint8 if x.itemsize == 1 else np.uint16)

    results = {}
    for arm in ["base"] + [a for a in arms if a != "base"]:
        jax.clear_caches()
        o, c = jax.jit(functools.partial(call, arm))(jnp.copy(cache), inputs)
        results[arm] = (bits(o), bits(c))
    ro, rc = results["base"]
    for arm, (o, c) in results.items():
        if arm == "base":
            continue
        row = {
            "arm": arm,
            "out_equal": bool(np.array_equal(o, ro)),
            "cache_equal": bool(np.array_equal(c, rc)),
            "out_diff": int(np.sum(o != ro)),
            "cache_diff": int(np.sum(c != rc)),
        }
        rows.append(row)
        log("CHECK " + " ".join(f"{k}={v}" for k, v in row.items()))
    ok = all(r["out_equal"] and r["cache_equal"] for r in rows)
    log(f"CHECK_SUMMARY arms={len(rows)} bitwise={'PASS' if ok else 'FAIL'}")
    return {"check": rows, "check_pass": ok}


def log_schedule(cached):
    stats = schedule_stats(cached)
    log("SCHEDULE " + " ".join(f"{k}={v}" for k, v in stats.items()))
    return stats


def do_time(arms, out, n, rounds):
    cache, inputs, cached = make_case()
    stats = log_schedule(cached)
    state = {"cache": cache, "inputs": inputs}
    rows = collections.defaultdict(list)
    for r in range(rounds):
        for arm in arms if r % 2 == 0 else arms[::-1]:
            us = timed(arm, state, n)
            rows[arm].append(us)
            log(f"TIME {arm} round={r} us_per_call={us:.1f}")
    med = {a: statistics.median(v) for a, v in rows.items()}
    for a in arms:
        log(f"TIME_MEDIAN {a} us_per_call={med[a]:.1f} rounds={len(rows[a])}")
    return {"time": dict(rows), "time_median": med, "schedule": stats}


SCOPE_RX = re.compile(r"^(ep_\w+|pr_\w+|accum_\d+)$")


def _load_plane(path):
    from vllm_torchtpu.tools import xplane_pb2

    space = xplane_pb2.XSpace()
    with open(path, "rb") as f:
        space.ParseFromString(f.read())
    return next(p for p in space.planes if p.name == "/device:TPU:0")


def _events(plane, line):
    out = []
    for ev in line.events:
        t0 = line.timestamp_ns * 1000 + ev.offset_ps
        out.append((t0, t0 + ev.duration_ps, plane.event_metadata[ev.metadata_id].name))
    out.sort(key=lambda e: (e[0], -e[1]))
    return out


def analyze(path):
    """Per decode-kernel instance: time, steps, outermost scope sums (us)."""
    plane = _load_plane(path)
    lines = {ln.name: ln for ln in plane.lines}
    kernels = [
        (a, b)
        for a, b, name in _events(plane, lines["XLA Ops"])
        if re.match(r"%?RPAd-", name)
    ]
    best, best_n = None, 0
    for ln in plane.lines:
        if ln.name in ("XLA Ops", "XLA Modules", "Async XLA Ops"):
            continue
        n = sum(
            1
            for e in ln.events
            if SCOPE_RX.match(plane.event_metadata[e.metadata_id].name)
        )
        if n > best_n:
            best, best_n = ln.name, n
    scopes = (
        [e for e in _events(plane, lines[best]) if SCOPE_RX.match(e[2])] if best else []
    )
    starts = [s[0] for s in scopes]
    insts = []
    for k0, k1 in kernels:
        i0, i1 = bisect.bisect_left(starts, k0), bisect.bisect_right(starts, k1)
        sums, ends, steps = (
            collections.defaultdict(float),
            collections.defaultdict(int),
            0,
        )
        for s0, s1, name in scopes[i0:i1]:
            if s0 < ends[name]:
                continue
            ends[name] = s1
            sums[name] += (s1 - s0) / 1e6
            steps += name == "ep_run_kernel"
        us = (k1 - k0) / 1e6
        ep = sum(v for k, v in sums.items() if k.startswith("ep_"))
        insts.append(
            {"us": us, "steps": steps, "outside": us - ep, "scopes": dict(sums)}
        )
    if not insts:
        return None
    keys = sorted({k for i in insts for k in i["scopes"]})
    return {
        "n": len(insts),
        "kernel_us": statistics.median(i["us"] for i in insts),
        "steps": statistics.median(i["steps"] for i in insts),
        "outside_us": statistics.median(i["outside"] for i in insts),
        "scopes_us": {
            k: statistics.median(i["scopes"].get(k, 0.0) for i in insts) for k in keys
        },
    }


def do_trace(arms, out):
    cache, inputs, cached = make_case()
    stats = log_schedule(cached)
    state = {"cache": cache, "inputs": inputs}
    res = {}
    for arm in arms:
        timed(arm, state, 4)
        p = prog(arm, state["cache"], inputs)
        tmp = os.path.join("/tmp/page-runs-trace", arm.replace(":", "_"))
        shutil.rmtree(tmp, ignore_errors=True)
        jax.profiler.start_trace(tmp)
        c = state["cache"]
        for _ in range(2):
            c = p(c, jnp.int32(4), inputs)
            jax.block_until_ready(c)
        jax.profiler.stop_trace()
        state["cache"] = c
        files = sorted(
            glob.glob(f"{tmp}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime
        )
        if not files:
            log(f"TRACE {arm} no xplane")
            continue
        dst = os.path.join(out, "xplane", arm.replace(":", "_") + ".xplane.pb")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(files[-1], dst)
        a = analyze(files[-1])
        res[arm] = a
        if a is None:
            log(f"TRACE {arm} no decode kernel in trace")
            continue
        sc = " ".join(f"{k}={v:.1f}" for k, v in a["scopes_us"].items() if v >= 0.05)
        log(
            f"TRACE {arm} n={a['n']} kernel_us={a['kernel_us']:.1f} steps={a['steps']} "
            f"outside_us={a['outside_us']:.1f} {sc}"
        )
    verdict = None
    if res.get("base") and res.get("runs"):
        b, r = res["base"], res["runs"]
        drop = b["outside_us"] - r["outside_us"]
        growth = r["scopes_us"].get("ep_run_kernel", 0.0) - b["scopes_us"].get(
            "ep_run_kernel", 0.0
        )
        go = drop >= GO_OUTSIDE_DROP_US and growth < GO_BODY_GROWTH_US
        verdict = {
            "outside_drop_us": drop,
            "body_growth_us": growth,
            "kernel_delta_us": r["kernel_us"] - b["kernel_us"],
            "go": go,
        }
        log(
            f"VERDICT outside_drop_us={drop:.1f} (>= {GO_OUTSIDE_DROP_US:.0f}) "
            f"body_growth_us={growth:.1f} (< {GO_BODY_GROWTH_US:.0f}) "
            f"kernel_delta_us={verdict['kernel_delta_us']:.1f} "
            f"-> {'GO' if go else 'NO-GO'}"
        )
    return {"trace": res, "verdict": verdict, "schedule": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "time", "trace"])
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--out", default="/tmp/page-runs-bench")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")
    log(f"DEVICE {jax.devices()[0]} jax={jax.__version__} libtpu_args={libtpu_args}")
    if a.mode == "check":
        res = do_check(a.arms or ["runs", "runs1"], a.out)
    elif a.mode == "time":
        res = do_time(a.arms or ["base", "runs", "runs1"], a.out, a.n, a.rounds)
    else:
        res = do_trace(a.arms or ["base", "runs", "runs1", "s:base", "s:runs"], a.out)
    with open(os.path.join(a.out, f"{a.mode}.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    log(f"DONE {a.mode}")
    return 0 if a.mode != "check" or res["check_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
