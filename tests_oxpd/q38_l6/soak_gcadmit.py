#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q38 L6 GC-ADMIT-FREEZE leak soak (CDK CPU pod only; never the Mac or the VM).

Usage (from a tree root, PYTHONPATH=src):  python3 tests/q38_l6/soak_gcadmit.py
Last line: SOAK_DONE rc=0 (every expectation below held) | SOAK_DONE rc=1.

Each arm runs in its own subprocess (a fresh heap: freezes cannot be undone) and repeats CYCLES cycles of the
worker's admission / decode / finish life of long-context requests:

* admission step: one request's NewRequestData-like payload (a 100,000-token prompt id list, pickled as the
  engine core's broadcast does) is unpickled and its state built as vLLM's CachedRequestState (the worker's own
  class, with SamplingParams and two KV groups' block ids); then the arm's freeze action;
* STEPS decode steps: one output token appended to every live request, a few step objects, and GARB small cyclic
  structures (2 nodes + a 1 KiB payload) that become garbage within the step (cyclic garbage created after
  freezes);
* finish: once LIVE requests are live, the oldest is dropped from the request dict.

The worker's warm-up is vLLM's own freeze_gc_heap() after importing vLLM / torch and building a static heap.

Arms:
  base   no freeze after warm-up.
  naive  gc.freeze() after each admission step, the brief's literal form (no pre-freeze collection, no cap):
         the positive control. At gen-0 threshold 50000 it must FAIL the bound (it pins each cycle's garbage),
         or the soak could not see a leak.
  lever  the real lever: core/l6_gcadmit.after_update_states with VLLM_TPU_L6_GC_ADMIT_FREEZE=1 (default cap
         and check), called right after the state is built.
  adv    the real lever, plus G_B live cyclic objects that are alive at each admission's freeze and die right
         after it (frozen cyclic garbage, the residual leak the cap bounds), with CAP_ADV (and CHECK_ADV, which
         since the cumulative cap only sets the GCADMIT line cadence).
Each arm at gen-0 threshold 700 (CPython default) and 50000 (served D1e VLLM_TPU_HOST_GC=threshold).

PRE-STATED BOUNDS (fixed in this file before any run; printed first as SOAK_BOUND):
  B-RSS  over cycles [WARM, CYCLES): the least-squares RSS slope <= 1 % of RSS(WARM) per 100 cycles, and
         median(RSS of the last 20 cycles) - median(RSS of cycles WARM..WARM+19) <= 1 % of RSS(WARM) per 100
         cycles of the span between them.
  B-FC   for every cycle c >= WARM: gc.get_freeze_count()(c) - gc.get_freeze_count() at the arm's first cycle
         <= LIVE * P_MAX + S  (live requests x tracked objects per request state + step slack). P_MAX is a
         stated constant; the measured objects per request state must not exceed it.
  adv    (v1, superseded for adv by v2 below; still computed and printed as SOAK_V1) B-FC with
         CAP_ADV + CHECK_ADV * (G_B + P_MAX + S) in place of LIVE * P_MAX + S; the cap must trip (capped=1);
         B-RSS over the cycles from 10 after the cap tripped to the end (at least 60 cycles).
Expectations: base, lever and adv PASS at both thresholds; naive FAILS at 50000 (positive control); naive at
700 is reported, not required.

BOUND v2 (2026-11-14, adv arm only; stated before the v2 run):
  Reason. v1 started the adv RSS window 10 cycles after the cap tripped. Once capped, the lever stops freezing
  and the arm behaves like base, and at gen-0 threshold 50000 base's RSS climbs while uncollected garbage builds
  up to the gen-0 trigger. Evidence, j-698978c0 (output.log md5 928e3106, M): base th=50000 RSS 1001.6 -> 1047.5
  MB over cycles 0-100, flat after; adv th=50000 capped at cycle 199, RSS 1028.7 -> 1058.7 MB over cycles
  200-280, flat after (1058.7 at 300, 1057.0 at 319), freeze count flat at 20,018 after the cap. v1's window
  caught the climb; the freeze count shows no leak after the cap.
  v2. The adv RSS window is [capped_at + 1 + WARM, capped_at + 1 + WARM + ADV_WINDOW) with ADV_WINDOW =
  CYCLES - WARM = 220: the same warm-up and length as the other arms' [WARM, CYCLES). The adv arm runs until
  max(CYCLES, capped_at + 1 + WARM + ADV_WINDOW) cycles (at most ADV_MAX_CYCLES; a cap that never trips FAILS).
  The lever's cap is cumulative (it stops once its freezes moved more than CAP objects in total), so
  B-FC(adv): gc.get_freeze_count() growth <= CAP_ADV + G_B + P_MAX + S at every cycle. Justification: growth <=
  cum (frozen objects leave only by dying); cum <= CAP_ADV before the freeze that trips the cap; one freeze moves
  at most the G_B held cyclic objects, one new request state (<= P_MAX) and the step objects (<= S). The
  measured per-freeze m (max and mean, from the lever's frozen_cum) is logged and its max must also be
  <= G_B + P_MAX + S.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import random
import statistics
import subprocess
import sys
import time

CYCLES = 320
WARM = 100
LIVE = 4
TOKENS = 100_000
STEPS = 16
GARB = 16
P_MAX = 64
S = 1000
G_B = 100
CAP_ADV = 20000
CHECK_ADV = 8
ADV_WINDOW = 220  # CYCLES - WARM
ADV_MAX_CYCLES = 2000
RSS_PCT_PER_100 = 1.0
THRESHOLDS = (700, 50000)
ARMS = ("base", "naive", "lever", "adv")
BLOCK = 256
VOCAB = 151_000


def bound_line() -> str:
    return (f"SOAK_BOUND cycles={CYCLES} warm={WARM} live={LIVE} tokens={TOKENS} steps={STEPS} garb={GARB} "
            f"P_MAX={P_MAX} S={S} G_B={G_B} cap_adv={CAP_ADV} check_adv={CHECK_ADV} "
            f"B-RSS='slope over [{WARM},{CYCLES}) <= {RSS_PCT_PER_100}% of RSS({WARM}) per 100 cycles; "
            f"median(last 20) - median({WARM}..{WARM + 19}) <= {RSS_PCT_PER_100}% per 100 cycles of span' "
            f"B-FC='fc(c) - fc(arm start) <= LIVE*P_MAX+S = {LIVE * P_MAX + S} for c >= {WARM}' "
            f"adv_v1='fc growth <= CAP_ADV + CHECK_ADV*(G_B+P_MAX+S) = {CAP_ADV + CHECK_ADV * (G_B + P_MAX + S)}; "
            f"capped=1; B-RSS from cap+10 (>= 60 cycles)' "
            f"adv_v2='fc growth <= CAP_ADV+G_B+P_MAX+S = {CAP_ADV + G_B + P_MAX + S}; max m <= G_B+P_MAX+S = "
            f"{G_B + P_MAX + S}; capped=1; B-RSS over [cap+1+{WARM}, cap+1+{WARM}+{ADV_WINDOW})' "
            f"expect='base/lever/adv PASS at {THRESHOLDS}; naive FAIL at 50000'")


def rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class _Node:
    __slots__ = ("peer", "buf", "__weakref__")


def _cyclic(nbytes: int) -> _Node:
    a, b = _Node(), _Node()
    a.peer, b.peer = b, a
    a.buf, b.buf = bytes(nbytes), None
    return a


def _make_state(CachedRequestState, SamplingParams, req_id, ids):
    nblk = -(-len(ids) // BLOCK)
    return CachedRequestState(
        req_id=req_id, prompt_token_ids=ids, mm_features=[],
        sampling_params=SamplingParams(temperature=1.0, top_k=20, top_p=0.95, max_tokens=4096),
        generator=None, block_ids=(list(range(nblk)), list(range(nblk, 2 * nblk))),
        num_computed_tokens=len(ids), output_token_ids=[])


def measure_p() -> int:
    """Tracked objects per request state (16-token prompt; the prompt list is one object at any length)."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    _make_state(CachedRequestState, SamplingParams, "w", [1, 2])  # first-use caches outside the count
    gc.collect()
    gc.freeze()
    before = len(gc.get_objects())
    states = [_make_state(CachedRequestState, SamplingParams, f"p{i}", list(range(16))) for i in range(100)]
    gc.collect()
    after = len(gc.get_objects())
    gc.unfreeze()
    assert len(states) == 100
    return -(-(after - before - 1) // 100)


def child(arm: str, th: int) -> int:
    os.environ.pop("VLLM_TPU_HOST_GC", None)
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            del os.environ[k]
    if arm in ("lever", "adv"):
        os.environ["VLLM_TPU_L6_GC_ADMIT_FREEZE"] = "1"
    if arm == "adv":
        os.environ["VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP"] = str(CAP_ADV)
        os.environ["VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK"] = str(CHECK_ADV)
    from types import SimpleNamespace as NS

    from vllm.sampling_params import SamplingParams
    from vllm.utils.gc_utils import freeze_gc_heap
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_torchtpu.core import l6_ab, l6_gcadmit

    l6_ab._resolve()
    l6_gcadmit._resolve()
    assert l6_gcadmit.ON == (arm in ("lever", "adv"))
    l6_ab._rank_world = lambda: (0, 1)

    rng = random.Random(1234)
    payloads = []
    for i in range(8):  # engine-side broadcast payloads (bytes: untracked); each unpickle builds a new list
        ids = [rng.randrange(VOCAB) for _ in range(TOKENS)]
        payloads.append(pickle.dumps({"req_id": f"tmpl{i}", "prompt_token_ids": ids}, protocol=5))
        del ids
    static = [{"k": [i, i + 1], "t": (i, str(i))} for i in range(100_000)]  # the warm-up heap
    gc.set_threshold(th, 10, 10)
    freeze_gc_heap()  # vLLM's warm-up freeze (collect 0 / 1 / 2, then freeze)

    g2 = {"n": 0, "us": 0, "max": 0, "t0": 0}

    def timer(phase, info):
        if phase == "start":
            g2["t0"] = time.perf_counter_ns()
        elif g2["t0"]:
            us = (time.perf_counter_ns() - g2["t0"]) // 1000
            g2["t0"] = 0
            if info.get("generation") == 2:
                g2["n"] += 1
                g2["us"] += us
                g2["max"] = max(g2["max"], us)

    gc.callbacks.append(timer)
    runner = NS(warmup_done=True)
    requests: dict = {}
    live: list = []
    rss, fc, admit_us, ms, capped_at = [], [], [], [], -1
    fc0 = gc.get_freeze_count()
    tok = 0
    limit = ADV_MAX_CYCLES if arm == "adv" else CYCLES
    c = -1
    while c + 1 < limit:
        c += 1
        rid = f"req{c}"
        nrd = pickle.loads(payloads[c % len(payloads)])
        held = [_cyclic(1024) for _ in range(G_B // 2)] if arm == "adv" else None  # alive at the freeze
        t0 = time.perf_counter_ns()
        requests[rid] = _make_state(CachedRequestState, SamplingParams, rid, nrd["prompt_token_ids"])
        so = NS(scheduled_new_reqs=[NS(req_id=rid)], total_num_scheduled_tokens=1, num_scheduled_tokens={rid: 1})
        if arm == "naive":
            gc.freeze()
        elif arm in ("lever", "adv"):
            cum0 = l6_gcadmit._st["cum"]
            l6_gcadmit.after_update_states(runner, so)
            if l6_gcadmit._st["cum"] != cum0:
                ms.append(l6_gcadmit._st["cum"] - cum0)
        admit_us.append((time.perf_counter_ns() - t0) // 1000)
        del held, so, nrd
        live.append(rid)
        if len(live) > LIVE:
            del requests[live.pop(0)]
        for _ in range(STEPS):
            step = NS(num_scheduled_tokens={r: 1 for r in live}, req_ids=list(live))
            for r in live:
                tok = (tok + 7919) % VOCAB
                requests[r].output_token_ids.append(tok)
            for _ in range(GARB):
                _cyclic(1024)  # garbage within the step
            del step
        rss.append(rss_bytes())
        fc.append(gc.get_freeze_count() - fc0)
        if capped_at < 0 and l6_gcadmit._st.get("off") and arm == "adv":
            capped_at = c
            limit = max(CYCLES, capped_at + 1 + WARM + ADV_WINDOW)
    gc.callbacks.remove(timer)
    counts = dict(l6_ab._state.get("counts", {}))
    out = {"arm": arm, "th": th, "rss": rss, "fc": fc, "admit_us": admit_us, "capped_at": capped_at, "ms": ms,
           "counts": counts, "g2": {k: g2[k] for k in ("n", "us", "max")}, "static": len(static)}
    print("SOAK_CHILD_JSON " + json.dumps(out), flush=True)
    return 0


def _slope(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def rss_check(rss, lo, hi=None):
    """(pass, slope % per 100 cycles, end growth % per 100 cycles of span) over cycles [lo, hi)."""
    hi = len(rss) if hi is None else hi
    seg = rss[lo:hi]
    ref = rss[lo]
    slope_pct = _slope(list(range(lo, hi)), seg) * 100 / ref * 100
    first, last = statistics.median(seg[:20]), statistics.median(seg[-20:])
    span = (hi - 10) - (lo + 10)
    end_pct = (last - first) / ref * 100 / (span / 100)
    return slope_pct <= RSS_PCT_PER_100 and end_pct <= RSS_PCT_PER_100, slope_pct, end_pct


def _tail(c):
    return (f"freezes={c.get('gcadmit.freeze', 0)} reclaimed={c.get('gcadmit.reclaimed', 0)} "
            f"capped={c.get('gcadmit.capped', 0)}")


def verdict(d, p_meas):
    """(ok, text) under the bound in force: v1 for base / naive / lever, v2 for adv."""
    arm, fc, rss, c = d["arm"], d["fc"], d["rss"], d["counts"]
    if arm == "adv":
        cap, ms = d["capped_at"], d["ms"]
        lo, hi = cap + 1 + WARM, cap + 1 + WARM + ADV_WINDOW
        m_max = max(ms) if ms else 0
        if cap < 0 or len(rss) < hi:
            return False, f"v2: cap tripped at {cap}, cycles run {len(rss)} (needs a trip and {hi} cycles)"
        fc_lim = CAP_ADV + G_B + P_MAX + S
        fc_ok = max(fc) <= fc_lim and m_max <= G_B + P_MAX + S
        ok_r, sl, en = rss_check(rss, lo, hi)
        ok = fc_ok and ok_r and p_meas <= P_MAX
        return ok, (f"v2: fc_max={max(fc)} fc_lim={fc_lim} m_max={m_max} m_lim={G_B + P_MAX + S} fc_ok={int(fc_ok)} "
                    f"rss_window=[{lo},{hi}) slope_pct_per100={sl:.3f} end_pct_per100={en:.3f} rss_ok={int(ok_r)} "
                    + _tail(c))
    fc_lim = LIVE * P_MAX + S
    fc_ok = max(fc[WARM:]) <= fc_lim
    ok_r, sl, en = rss_check(rss, WARM)
    ok = fc_ok and ok_r and p_meas <= P_MAX
    return ok, (f"fc_max={max(fc[WARM:])} fc_lim={fc_lim} fc_ok={int(fc_ok)} rss_window=[{WARM},{len(rss)}) "
                f"slope_pct_per100={sl:.3f} end_pct_per100={en:.3f} rss_ok={int(ok_r)} " + _tail(c))


def verdict_v1_adv(d, p_meas):
    """The adv arm under bound v1, as stated before j-698978c0 (reported, not used for the expectation)."""
    fc, rss, cap = d["fc"], d["rss"], d["capped_at"]
    fc_lim = CAP_ADV + CHECK_ADV * (G_B + P_MAX + S)
    if cap < 0 or len(rss) - (cap + 10) < 60:
        return False, f"v1: cap tripped at {cap}"
    ok_r, sl, en = rss_check(rss, cap + 10)
    ok = max(fc) <= fc_lim and ok_r and p_meas <= P_MAX
    return ok, (f"v1: fc_max={max(fc)} fc_lim={fc_lim} rss_window=[{cap + 10},{len(rss)}) slope_pct_per100={sl:.3f} "
                f"end_pct_per100={en:.3f} rss_ok={int(ok_r)}")


def parent() -> int:
    print(bound_line(), flush=True)
    p_meas = measure_p()
    print(f"SOAK_P measured tracked objects per request state = {p_meas} (P_MAX {P_MAX})", flush=True)
    results, rc = {}, 0
    for th in THRESHOLDS:
        for arm in ARMS:
            env = dict(os.environ)
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", arm, str(th)], env=env,
                               capture_output=True, text=True, timeout=3000)
            js = [ln for ln in p.stdout.splitlines() if ln.startswith("SOAK_CHILD_JSON ")]
            if p.returncode != 0 or not js:
                print(f"SOAK arm={arm} th={th} CHILD FAILED rc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
                rc = 1
                continue
            d = json.loads(js[-1][len("SOAK_CHILD_JSON "):])
            ok, why = verdict(d, p_meas)
            results[(arm, th)] = ok
            au = sorted(d["admit_us"][WARM:])
            mb = [r / 2**20 for r in d["rss"]]
            ms = d["ms"]
            mstat = f"m_freezes={len(ms)} m_max={max(ms)} m_mean={statistics.fmean(ms):.1f}" if ms else "m_freezes=0"
            print(f"SOAK arm={arm} th={th} {why} rss_mb start={mb[0]:.1f} warm={mb[WARM]:.1f} end={mb[-1]:.1f} "
                  f"cycles={len(mb)} fc_end={d['fc'][-1]} {mstat} admit_us p50={au[len(au) // 2]} max={au[-1]} "
                  f"g2_all n={d['g2']['n']} us={d['g2']['us']} max_us={d['g2']['max']} "
                  f"capped_at={d['capped_at']} VERDICT={'PASS' if ok else 'FAIL'}", flush=True)
            if arm == "adv":
                ok1, why1 = verdict_v1_adv(d, p_meas)
                print(f"SOAK_V1 arm=adv th={th} {why1} VERDICT_V1={'PASS' if ok1 else 'FAIL'}", flush=True)
            for cyc in list(range(0, len(mb), 20)) + [len(mb) - 1]:
                print(f"SOAK_TAB arm={arm} th={th} cycle={cyc} rss_mb={mb[cyc]:.1f} fc_growth={d['fc'][cyc]}")
    want = {(a, t): True for a in ("base", "lever", "adv") for t in THRESHOLDS}
    want[("naive", 50000)] = False
    for k, v in want.items():
        got = results.get(k)
        good = got is v
        print(f"SOAK_EXPECT arm={k[0]} th={k[1]} want={'PASS' if v else 'FAIL'} "
              f"got={'missing' if got is None else ('PASS' if got else 'FAIL')} {'ok' if good else 'BAD'}")
        rc |= 0 if good else 1
    n7 = results.get(("naive", 700))
    print(f"SOAK_INFO naive th=700 (reported, not required): {'missing' if n7 is None else ('PASS' if n7 else 'FAIL')}")
    print(f"SOAK_DONE rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "child":
        sys.exit(child(sys.argv[2], int(sys.argv[3])))
    sys.exit(parent())
