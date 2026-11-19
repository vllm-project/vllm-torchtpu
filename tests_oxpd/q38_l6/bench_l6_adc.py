# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever adcarry: CPU microbench of host us per decode step saved.

CDK CPU pod only (never the Mac or the VM). Usage, from the tree root:
  python3 tests/q38_l6/bench_l6_adc.py [--n 4,9,11,25,52] [--steps 300] [--warm 20] [--reps 3]

Drives the real ``TPUModelRunner._prepare_inputs`` and the real Qwen3.8
adapter (``begin_step`` / ``prepare_inputs``) through l6_adc_harness.py on a
served-shaped plan: async scheduling, MTP K = 3 (every request schedules
1 + 3 tokens), VLLM_TPU_POSTFWD_FUSED, the served adapter flags (FUSED_PREP,
NGRAM_WINDOW=1, FUSED_PREP_FUNCTIONAL), the pooled cache, the combined RPA /
GDN request distribution, max_num_seqs 64 (request bucket 64), token buckets
16..512, steady decode (the carried case).

Arms (each a fresh runner and the same seeded stream; reps interleave the arms):
  sc   STEADY + CARRY (STACK-D v4's L6 state);
  sca  sc + VLLM_TPU_L6_ADAPT_CARRY=1;
  ab   sc + VLLM_TPU_L6_ADAPT_CARRY=ab:1 with step windows of 50 (2 classes): one runner, one
       stream, steps split by the lever's state (the first 2 steps after each switch dropped);
       the paired in-process delta, robust to run-to-run drift.

Per step: the wall time of adapter.begin_step + _prepare_inputs +
adapter.prepare_inputs, minus the time inside the step-input programs (on CPU
they run as the eager functions they wrap; on TPU each is a compiled-program
dispatch) and inside uploads (on CPU a ``.to(device)`` is a clone), so the
``host`` column is Python / numpy work only. Programs per step and uploads
per step are reported separately (``prog/step``, ``h2d/step``). GC is off in
the timed loop. The sc and sca digests must be equal (identity). Last line:
``BENCH_DONE identity=PASS|FAIL``.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l6_adc_harness as A  # noqa: E402
import l6_carry_harness as H  # noqa: E402

import torch  # noqa: E402

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.models.vllm.qwen4_exp import inputs as qinputs  # noqa: E402
from vllm_torchtpu.runner import adapt_carry, decode_carry, tpu_runner  # noqa: E402

PERF = time.perf_counter_ns
ARMS = {
    "sc": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1"},
    "sca": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1", "VLLM_TPU_L6_ADAPT_CARRY": "1"},
    "ab": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1", "VLLM_TPU_L6_ADAPT_CARRY": "ab:1",
           "VLLM_TPU_L6_AB_STEPS": "50", "VLLM_TPU_L6_AB_CLASSES": "2"},
}
PADS = (16, 32, 64, 128, 256, 512)


class Timers:
    def __init__(self):
        self.prog_ns = 0
        self.prog_n = 0
        self.h2d_ns = 0
        self.h2d_n = 0


T = Timers()


def timed_prog(fn):
    def wrapped(*a, **k):
        t0 = PERF()
        try:
            return fn(*a, **k)
        finally:
            T.prog_ns += PERF() - t0
            T.prog_n += 1
    return wrapped


class TimedUploads(H.Uploads):
    def __enter__(self):
        super().__enter__()
        inner = torch.Tensor.to

        def to(t, *args, **kwargs):
            dev = args[0] if args else kwargs.get("device")
            if not isinstance(dev, torch.device):
                return inner(t, *args, **kwargs)
            t0 = PERF()
            try:
                return inner(t, *args, **kwargs)
            finally:
                T.h2d_ns += PERF() - t0
                T.h2d_n += 1

        torch.Tensor.to = to
        return self


def set_arm(arm):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            del os.environ[k]
    os.environ.update(ARMS[arm])
    l6_ab._resolve()


def run_arm(arm, n, steps, warm):
    set_arm(arm)
    shape = dict(nreq=n, max_num_reqs=64, paddings=PADS, mamba=False, comb=True, pooled=True, max_len=4096)
    sim = A._sim(shape, 11, n)  # steady stream: the warm-up steps warm its one shape
    host, progs, h2ds, digests, cls = [], [], [], [], []
    last, since = None, 0
    gc.disable()
    try:
        for s in range(steps + warm):
            if s == 2:
                A.warm(sim)
            T.prog_ns = T.prog_n = T.h2d_ns = T.h2d_n = 0
            d = sim.step()
            state = l6_ab.on("adcarry")
            since = since + 1 if state == last else 0
            last = state
            if s >= warm:
                host.append((sim.region_ns[-1] - T.prog_ns - T.h2d_ns) / 1000.0)
                progs.append(T.prog_n)
                h2ds.append(T.h2d_n)
                cls.append(None if since < 2 else state)
            digests.append(d)
    finally:
        gc.enable()
    return host, progs, h2ds, digests, sim, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="4,9,11,25,52")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.environ["VLLM_TPU_POSTFWD_FUSED"] = "1"
    os.environ.update(A.adapter_env())
    undo = A.install_eager_programs()
    # time every step-input program (eager here) and every upload
    tpu_runner._postfwd_subst_fused = timed_prog(tpu_runner._postfwd_subst_fused)
    decode_carry.carry_fused = timed_prog(decode_carry.carry_fused)
    adapt_carry.program_postfwd = timed_prog(adapt_carry.program_postfwd)
    adapt_carry.program_carry = timed_prog(adapt_carry.program_carry)
    real_fsi = qinputs.fused_step_inputs
    H.Uploads = TimedUploads
    ok = True
    rows = []
    try:
        for n in [int(x) for x in a.n.split(",")]:
            meds = {arm: [] for arm in ARMS}
            pm, hm, dg = {}, {}, {}
            ab_on, ab_off, ab_pon, ab_poff = [], [], [], []
            for _ in range(a.reps):
                for arm in ARMS:
                    host, progs, h2ds, digests, sim, cls = run_arm_with_adapter_timer(arm, n, a.steps, a.warm,
                                                                                      real_fsi)
                    dg[arm] = H.digest_all(digests)
                    if arm == "ab":
                        on = [h for h, c in zip(host, cls) if c is True]
                        off = [h for h, c in zip(host, cls) if c is False]
                        ab_on.append(statistics.median(on))
                        ab_off.append(statistics.median(off))
                        ab_pon.append(statistics.mean(p for p, c in zip(progs, cls) if c is True))
                        ab_poff.append(statistics.mean(p for p, c in zip(progs, cls) if c is False))
                        continue
                    meds[arm].append(statistics.median(host))
                    pm[arm], hm[arm] = statistics.mean(progs), statistics.mean(h2ds)
            same = dg["sc"] == dg["sca"] == dg["ab"]
            ok &= same
            sc = (statistics.median(meds["sc"]), min(meds["sc"]), max(meds["sc"]), pm["sc"], hm["sc"])
            sca = (statistics.median(meds["sca"]), min(meds["sca"]), max(meds["sca"]), pm["sca"], hm["sca"])
            ab = (statistics.median(ab_off), statistics.median(ab_on), min(o - x for o, x in zip(ab_off, ab_on)),
                  max(o - x for o, x in zip(ab_off, ab_on)), statistics.mean(ab_poff), statistics.mean(ab_pon))
            rows.append((n, sc, sca, ab, same))
            print(f"BENCH n={n} sc host_us={sc[0]:.1f} [{sc[1]:.1f},{sc[2]:.1f}] prog/step={sc[3]:.2f} "
                  f"h2d/step={sc[4]:.2f} | sca host_us={sca[0]:.1f} [{sca[1]:.1f},{sca[2]:.1f}] "
                  f"prog/step={sca[3]:.2f} h2d/step={sca[4]:.2f} | saved_us={sc[0] - sca[0]:.1f} | "
                  f"ab off={ab[0]:.1f} on={ab[1]:.1f} paired_saved_us={ab[0] - ab[1]:.1f} "
                  f"[{ab[2]:.1f},{ab[3]:.1f}] prog/step off={ab[4]:.2f} on={ab[5]:.2f} | identity={same}", flush=True)
    finally:
        undo()
    print("BENCH_TABLE n | sc host us | sca host us | saved us (arms) | ab paired saved us [min,max over reps] | "
          "programs/step sc->sca | uploads/step sc->sca")
    for n, sc, sca, ab, same in rows:
        print(f"BENCH_TABLE {n} | {sc[0]:.1f} | {sca[0]:.1f} | {sc[0] - sca[0]:.1f} | {ab[0] - ab[1]:.1f} "
              f"[{ab[2]:.1f},{ab[3]:.1f}] | {sc[3]:.2f}->{sca[3]:.2f} | {sc[4]:.2f}->{sca[4]:.2f}")
    print(f"BENCH_DONE identity={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def run_arm_with_adapter_timer(arm, n, steps, warm, real_fsi):
    """run_arm with the adapter's own program (eager fused_step_inputs) timed too."""
    orig = A.make_adapter

    def make_adapter(r, **kw):
        ad = orig(r, **kw)
        ad._fused = timed_prog(real_fsi)
        return ad

    A.make_adapter = make_adapter
    try:
        return run_arm(arm, n, steps, warm)
    finally:
        A.make_adapter = orig


if __name__ == "__main__":
    sys.exit(main())
