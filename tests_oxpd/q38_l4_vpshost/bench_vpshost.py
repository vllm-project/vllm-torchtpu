# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VPSHOST host microbench (CDK CPU pod, served image; not the TPU host class). Usage: bench_vpshost.py

Times the real runner code on the fake InputBatch of test_l4_vpshost.py, one thread, median of 7 repeats of 2000
calls, at 3 / 16 / 52 / 64 active rows (padded as the runner pads):
  A_topk      TPUModelRunner._sampler_topk_candidates (the scan stage 0 skips)
  DEC off/on  execute_model's decision block (exec of the real statements) with sampfb off / on, fallback path
  TODAY       what stage 1 replaces, per VPS chunk: _vps_fallback_rows (B) + _padded_sampling_params with
              SAMPLING_PARAMS_REUSE (C; all three uploads hit) + the num_active tensor and its upload (D)
  SNAP        the same three through sampsnap on a snapshot hit
  STATS       _vps_verify_fb's host time per fallback chunk over 1000 chunks with VLLM_TPU_VPS_STATS_EVERY=200,
              statsasync off / on (CPU tensors; the event is ready at the first query). Host cost only: the
              device-side stall a blocking .tolist() causes on a TPU cannot be measured here.
Prints BENCH lines and BENCH_DONE.
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TPU_SAMPLER_TOPK_CANDIDATES"] = "64"

import torch  # noqa: E402

import test_l4_vpshost as t  # noqa: E402
from vllm_torchtpu.core import l6_ab  # noqa: E402

REPS, N = 7, 2000


def bench(fn):
    out = []
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        for _ in range(N):
            fn()
        out.append((time.perf_counter_ns() - t0) / N / 1e3)
    return statistics.median(out)


def arm(**env):
    for k in list(os.environ):
        if k.startswith(("VLLM_TPU_L6_", "VLLM_TPU_L4_")):
            del os.environ[k]
    os.environ.update(env)
    l6_ab._resolve()
    l6_ab.begin_step()


def batch_of(n, range_reqs=0):
    b = t.FakeBatch()
    for i in range(n):
        b.add_request(f"r{i}", (0.6, 0 if i < range_reqs else 20, 0.95))
    return b


def main():
    torch.set_num_threads(1)
    print("BENCH host", os.cpu_count(), "cpus; torch", torch.__version__, "python", sys.version.split()[0])
    md = object()
    for n in (3, 16, 52, 64):
        for rr in (0, 1):
            r = t.DecRunner(batch_of(n, rr), fb_max=8)
            arm()
            a = bench(lambda: r._real_topk(0, n, None))
            off = bench(lambda: t.decide(r, True, md, 0, n))
            arm(VLLM_TPU_L4_SAMP_FB_ORDER="1")
            on = bench(lambda: t.decide(r, True, md, 0, n))
            assert t.decide(r, True, md, 0, n) == t.oracle(r, True, md, 0, n)
            print(f"BENCH sampfb rows={n} range_reqs={rr} F={t.oracle(r, True, md, 0, n)[1]} A_topk={a:.2f}us "
                  f"DEC off={off:.2f}us on={on:.2f}us delta={on - off:+.2f}us")
    for reuse in ("1", "0"):
        for n in (3, 16, 52, 64):
            P = t.pad_of(n)
            r = t.SnapRunner(batch_of(n), fb_max=8)
            bf = torch.bfloat16
            arm(VLLM_TPU_SAMPLING_PARAMS_REUSE=reuse)

            def today():
                r._vps_fallback_rows(0, n)
                r._real_padded(0, n, P, bf, r.device)
                r._real_upload("vps_num_active", torch.tensor([n], dtype=torch.int32))

            b = bench(lambda: r._vps_fallback_rows(0, n))
            c = bench(lambda: r._real_padded(0, n, P, bf, r.device))
            d = bench(lambda: r._real_upload("vps_num_active", torch.tensor([n], dtype=torch.int32)))
            tt = bench(today)
            arm(VLLM_TPU_SAMPLING_PARAMS_REUSE=reuse, VLLM_TPU_L4_SAMP_SNAP="1")
            snap = r._samp_snap

            def hit():
                r._vps_fallback_rows(0, n)
                snap.params(r, 0, n, P, bf, r.device)
                snap.num_active(r, n)

            hit()
            g0 = snap.gen
            ts = bench(hit)
            assert snap.gen == g0                                  # every timed call was a hit
            print(f"BENCH sampsnap reuse={reuse} rows={n} pad={P} B_fbrows={b:.2f}us C_padded={c:.2f}us "
                  f"D_nactive={d:.2f}us TODAY={tt:.2f}us SNAP_hit={ts:.2f}us saving={tt - ts:.2f}us")
    import types as _types

    real_fb, real_ev, real_log = t.tr._vps_verify_fb_compiled, t.l4_vpshost._event, t.tr.logger
    t.tr._vps_verify_fb_compiled = lambda *a: (torch.zeros(4, dtype=torch.int32), torch.tensor([3, 2, 1], dtype=torch.int32))
    t.l4_vpshost._event = lambda: _types.SimpleNamespace(query=lambda: True, synchronize=lambda: None)
    t.tr.logger = _types.SimpleNamespace(info=lambda *a, **k: None)
    try:
        for name, lev in (("off", {}), ("on", {"VLLM_TPU_L4_VPS_STATS_ASYNC": "1"})):
            arm(VLLM_TPU_VPS_STATS_EVERY="200", **lev)
            r = t.StatsRunner()
            x = torch.zeros(4, t.LOCALV)
            args = (None,) * 15
            meds = []
            for _ in range(REPS):
                t0 = time.perf_counter_ns()
                for _ in range(1000):
                    r._vps_verify_fb(x, args, 8, 0, None)
                meds.append((time.perf_counter_ns() - t0) / 1000 / 1e3)
            print(f"BENCH statsasync {name} per_chunk={statistics.median(meds):.2f}us (1000 chunks, every 200)")
    finally:
        t.tr._vps_verify_fb_compiled, t.l4_vpshost._event, t.tr.logger = real_fb, real_ev, real_log
    arm()
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
