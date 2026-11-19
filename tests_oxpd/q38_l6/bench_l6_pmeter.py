# SPDX-License-Identifier: Apache-2.0
"""q38 L6 P-METER CPU microbench (CDK CPU pod only): the per-step host cost of the
instrument at N = 4 / 9 / 11 / 25 / 52 requests.

Per N, the real TPUModelRunner._modify_prev_results on a served-shaped stub (N
requests, K = 3 drafts, 2 tokens accepted each) timed in three modes:
  off      every L6 lever unset, HOST_PHASES off (the stock path);
  pm       VLLM_TPU_L6_PMETER=1 (wait stamps), HOST_PHASES off;
  pm_subs  PMETER=1 and the HOSTSUBS census on (the ~mpr_* sub-marks record);
plus the N-independent per-step hook cost (deq + on_step + before_forward +
wait_begin / wait_end + loop_enqueue, lever on vs every lever off) and the window
emission (PMETER + THREADSTAT lines, W = 200) amortised per step.
Median of 7 repeats of 2,000 calls, microseconds per call.
Lines: PMBENCH N=<n> off_us= pm_us= pm_subs_us= d_pm_us= d_subs_us=
       PMBENCH_HOOKS on_us= off_us= d_us= emit_us_per_step=
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import torch

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import host_phases, l6_pmeter

ITERS, REPS = 2000, 7


def _clear_env():
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            del os.environ[k]


def _mode(name):
    _clear_env()
    if name in ("pm", "pm_subs"):
        os.environ["VLLM_TPU_L6_PMETER"] = "1"
    l6_ab._resolve()
    l6_pmeter._resolve()
    host_phases._EVERY = 200 if name == "pm_subs" else 0
    host_phases._SUBS = name == "pm_subs"
    host_phases._SLOW_NS = 0
    host_phases._GC = False
    host_phases._last = None
    host_phases.logger = NS(info=lambda *a, **k: None)
    l6_pmeter.logger = NS(info=lambda *a, **k: None)
    l6_ab.logger.info = lambda *a, **k: None
    if l6_ab.ACTIVE:
        l6_ab.begin_step()
        l6_pmeter.on_step()


def _stub(n):
    from vllm_torchtpu.runner.tpu_runner import INVALID_TOKEN_ID

    toks = torch.full((n, 4), INVALID_TOKEN_ID, dtype=torch.int64)
    toks[:, :2] = 7
    states = [NS(output_token_ids=[0] * (4 * ITERS * REPS * 3 + 16)) for _ in range(n)]
    pre = NS(req_ids=[f"r{i}" for i in range(n)],
             request_seq_lens=[(i, states[i], 100, f"r{i}") for i in range(n)],
             discard_sampled_tokens_req_indices=[], num_draft_per_req={i: 3 for i in range(n)},
             wait_for_copy=lambda: toks)
    width = 2 * ITERS * REPS * 3 + 64
    batch = NS(req_id_to_index={f"r{i}": i for i in range(n)},
               num_tokens_no_spec=np.full(n, width - 8, dtype=np.int64),
               token_ids_cpu=np.zeros((n, width), dtype=np.int64))
    return NS(_pre_async_results=pre, input_batch=batch), batch, width


def bench_mpr(n):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    out = {}
    for name in ("off", "pm", "pm_subs"):
        _mode(name)
        fake, batch, width = _stub(n)
        fn = TPUModelRunner._modify_prev_results
        reps = []
        for _ in range(REPS):
            batch.num_tokens_no_spec[:] = width - 8
            t = time.perf_counter_ns()
            for _ in range(ITERS):
                fn(fake)
                # each call rolls num_tokens_no_spec back by 2 (4 placeholders, 2 accepted); restore it
                batch.num_tokens_no_spec += 2
            reps.append((time.perf_counter_ns() - t) / ITERS / 1000)
        out[name] = statistics.median(reps)
    return out


def bench_hooks():
    res = {}
    for name in ("off", "pm"):
        _mode(name)
        reps = []
        for _ in range(REPS):
            t = time.perf_counter_ns()
            for _ in range(ITERS):
                l6_pmeter.deq()
                if l6_ab.ACTIVE:
                    l6_ab.begin_step()
                    l6_pmeter.on_step()
                l6_pmeter.before_forward()
                l6_pmeter.wait_begin()
                l6_pmeter.wait_end()
                l6_pmeter.loop_enqueue()
            reps.append((time.perf_counter_ns() - t) / ITERS / 1000)
        res[name] = statistics.median(reps)
    _mode("pm")  # seq 1 open
    for _ in range(l6_pmeter.WINDOW - 2):  # seqs 1 .. W-2 committed, W-1 open: a nearly full ring, no emission yet
        l6_pmeter.deq()
        l6_ab.begin_step()
        l6_pmeter.on_step()
        l6_pmeter.before_forward()
        l6_pmeter.wait_begin()
        l6_pmeter.wait_end()
        l6_pmeter.loop_enqueue()
    seq_end = l6_pmeter.WINDOW
    assert l6_pmeter.window_line(seq_end).count(",") >= 4 * (l6_pmeter.WINDOW - 2)
    t = time.perf_counter_ns()
    for _ in range(20):
        l6_pmeter.window_line(seq_end)
        l6_pmeter.thread_line(seq_end)
    emit = (time.perf_counter_ns() - t) / 20 / 1000 / l6_pmeter.WINDOW
    return res, emit


def main():
    print(f"PMBENCH_ENV python {sys.version.split()[0]} torch {torch.__version__} nproc {os.cpu_count()} "
          f"iters {ITERS} reps {REPS}")
    for n in (4, 9, 11, 25, 52):
        r = bench_mpr(n)
        print(f"PMBENCH N={n} off_us={r['off']:.2f} pm_us={r['pm']:.2f} pm_subs_us={r['pm_subs']:.2f} "
              f"d_pm_us={r['pm'] - r['off']:+.2f} d_subs_us={r['pm_subs'] - r['off']:+.2f}")
    h, emit = bench_hooks()
    print(f"PMBENCH_HOOKS on_us={h['pm']:.2f} off_us={h['off']:.2f} d_us={h['pm'] - h['off']:+.2f} "
          f"emit_us_per_step={emit:.2f}")
    _clear_env()
    print("PMBENCH_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
