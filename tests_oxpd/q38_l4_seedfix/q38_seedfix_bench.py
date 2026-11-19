# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SEEDFIX host-cost microbench (CDK CPU pod only; run from the tree root with PYTHONPATH=src).

The served VPS route (fallback program, full draws, TP1 head of width 10240, K = 3) through the runner's real
sample_tokens on the sf_fakes runner, one chunk of R rows (R = 4 / 8 / 16) with 0 / 1 / R seeded requests, flag on vs
flag off, interleaved in blocks; median per-step wall time (one torch thread). Columns:
  wall      the whole step incl. the piece draws' RNG compute (on CPU the "device" work is host work too);
  dispatch  the same with every seedfix piece draw replaced by torch.empty of the same shapes, i.e. the overlay's host
            dispatch cost without the RNG arithmetic (on TPU the draws run on the device);
  check0    the 0-seeded cost in isolation: the step's added statements when no request is seeded (one dict truthiness
            check in sample_tokens plus the per-site None checks), timed with timeit (ns per step).
Lines: BENCH R=<r> seeded=<s> on_ms=<..> off_ms=<..> delta_us=<..> | dispatch on_ms=<..> off_ms=<..> delta_us=<..>
(whole-step deltas: noise-dominated, the step is 30-100 ms of fake-verify CPU work), and
       BENCH_ISO V=<vocab> R=<r> seeded=<s> host_dispatch_us=<..> wall_incl_cpu_rng_us=<..> eager_ops_per_step=<..>
(SEEDFIX's own work, isolated: build_plan + the VPS site's overlay; dispatch = piece draws replaced by torch.empty).
"""

import os
import statistics
import sys
import time
import timeit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sf_fakes as F  # noqa: E402
from vllm_torchtpu.runner import seedfix as sf  # noqa: E402

torch.set_num_threads(1)
STEPS, BLOCK, WARM = 120, 10, 20


def bench(R, S, dispatch_only):
    for k in list(os.environ):
        if k.startswith(("VLLM_TPU_L6_", "VLLM_TPU_L4_", "VLLM_TPU_VPS_")):
            os.environ.pop(k)
    real = sf.draw_pieces
    if dispatch_only:
        def fake(gen, vocab, k, cw, device):
            e = torch.empty
            return (e((vocab,), device=device), e((k,), device=device) if k else None,
                    e((k, vocab), device=device) if k else None, e((cw,), device=device) if cw and k else None,
                    e((k, cw), device=device) if cw and k else None)
        sf.draw_pieces = fake
    try:
        times = {True: [], False: []}
        runners = {}
        chunks = {}
        for on in (True, False):
            runners[on] = F.Runner(vps=True, fb_max=8)
            chunks[on] = [F.Req(f"b{i}", 500 + i if i < S else None, logit_seed=i, temp=[0.8, 0.6, 1.0][i % 3])
                          for i in range(R)]
        step = 0
        for blk in range(-(WARM // BLOCK), STEPS // BLOCK):
            for on in (True, False):
                os.environ["VLLM_TPU_SEEDED_SAMPLING"] = "1" if on else "0"
                for _ in range(BLOCK):
                    t0 = time.perf_counter_ns()
                    F.run_step(runners[on], [chunks[on]], step=step % 7)
                    dt = time.perf_counter_ns() - t0
                    if blk >= 0:
                        times[on].append(dt)
                    step += 1
        return statistics.median(times[True]) / 1e6, statistics.median(times[False]) / 1e6
    finally:
        sf.draw_pieces = real
        os.environ.pop("VLLM_TPU_SEEDED_SAMPLING", None)


def check0():
    """The 0-seeded added statements, as sample_tokens and the sites run them."""
    class Rn:
        pass

    r = Rn()
    r.input_batch = F.NS(generators={})
    r._seed_plan = None

    def added():
        r._seed_plan = (sf.build_plan(r, None) if r.input_batch.generators else None)
        for _ in range(3):  # sites reached per chunk on the VPS route (verify draws, fb draw, after-loop clear)
            if getattr(r, "_seed_plan", None) is not None:
                pass
        r._seed_plan = None

    n = 200000
    return min(timeit.repeat(added, number=n, repeat=5)) / n * 1e9


def isolated(R, S, vocab, dispatch_only, n=200):
    """SEEDFIX's own per-step host work, isolated: the sample_tokens check, build_plan (S requests' piece draws) and
    the VPS site's overlay (bonus rows + draft slots) on pre-drawn chunk tensors [B_pad, V] / [T_pad] / [T_pad, V];
    median us over n steps (index tensors cached after the first step, as in steady decode). Also the eager ops per
    step issued from runner/seedfix.py (a dispatch census)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    os.environ["VLLM_TPU_SEEDED_SAMPLING"] = "1"
    real = sf.draw_pieces
    if dispatch_only:
        def fake(gen, vocab_, k, cw, device):
            e = torch.empty
            return (e((vocab_,), device=device), e((k,), device=device) if k else None,
                    e((k, vocab_), device=device) if k else None, None, None)
        sf.draw_pieces = fake
    try:
        r = F.Runner(vps=True, fb_max=8, vocab=vocab)
        reqs = [F.Req(f"i{i}", 700 + i if i < S else None, logit_seed=i, vocab=vocab) for i in range(R)]
        sched = r.set_batch(reqs)
        b_pad, t_pad = max(R + 1, 8), max(16, 1 << (R * (F.K + 1) - 1).bit_length())
        md = F.compose_spec(reqs, 0, b_pad=b_pad, t_pad=t_pad)[1] if vocab == F.TV else None
        if md is None:
            dl = np.array([F.K] * R, dtype=np.int32)
            md = F.NS(draft_lengths_cpu=dl)
        bu = torch.rand(b_pad, vocab)
        au = torch.rand(t_pad)
        ru = torch.rand(t_pad, vocab)

        def one():
            plan = sf.build_plan(r, sched) if r.input_batch.generators else None
            if plan is not None:
                sf.overlay_verify(r, plan, bu, au, ru, 0, R, md)

        for _ in range(5):
            one()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            one()
            ts.append(time.perf_counter_ns() - t0)

        class Ops(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.n = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                f = sys._getframe(1)
                while f is not None and not f.f_code.co_filename.endswith(os.path.join("runner", "seedfix.py")):
                    f = f.f_back
                if f is not None:
                    self.n += 1
                return func(*args, **(kwargs or {}))

        with Ops() as ops:
            one()
        return statistics.median(ts) / 1e3, ops.n
    finally:
        sf.draw_pieces = real
        os.environ.pop("VLLM_TPU_SEEDED_SAMPLING", None)


def main():
    F.patch_all(F.Patcher())
    print(f"BENCH_CFG torch_threads={torch.get_num_threads()} steps={STEPS} block={BLOCK} vocab={F.TV} K={F.K} route=vps_fb_full")
    print(f"BENCH_CHECK0 ns_per_step={check0():.1f}")
    for vocab in (F.TV, 248320):
        for R in (4, 8, 16):
            for S in sorted({0, 1, R}):
                if vocab != F.TV and S == 0:
                    continue
                d_us, d_ops = isolated(R, S, vocab, True)
                w_us, _ = isolated(R, S, vocab, False, n=40 if vocab != F.TV else 200)
                print(f"BENCH_ISO V={vocab} R={R} seeded={S} host_dispatch_us={d_us:.1f} wall_incl_cpu_rng_us={w_us:.1f} "
                      f"eager_ops_per_step={d_ops} per_seeded_row={d_ops / S if S else 0:.1f}")
    for R in (4, 8, 16):
        for S in sorted({0, 1, R}):
            on, off = bench(R, S, False)
            don, doff = bench(R, S, True)
            print(f"BENCH R={R} seeded={S} on_ms={on:.3f} off_ms={off:.3f} delta_us={(on - off) * 1e3:.1f} | "
                  f"dispatch on_ms={don:.3f} off_ms={doff:.3f} delta_us={(don - doff) * 1e3:.1f}")
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
