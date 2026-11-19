# SPDX-License-Identifier: Apache-2.0
"""CPU timing of the HOSTPHASES report: the per-phase reference vs
host_phases._report (T-1(ii) HP-REPORT), on the same served-like ragged
windows (45 phases, N=200), interleaved, the logger formatting included.
Run with PYTHONPATH=src from the tree root:  python3 tests/q38_l6hprep/hp_report_bench.py [reps]
Prints HPREPORT_BENCH lines; exit 1 if any reference/new line differs."""

import copy
import os
import statistics as st
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hp_report_ref as ref  # noqa: E402
from vllm_torchtpu.runner import host_phases  # noqa: E402


class _Fmt:
    def __init__(self):
        self.last = ""

    def info(self, fmt, *args):
        self.last = fmt % args


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(38)
    windows = [ref.window(rng, every=200) for _ in range(8)]
    log_old, log_new = _Fmt(), _Fmt()
    host_phases.logger = log_new
    host_phases._EVERY = 200
    host_phases._rank = 0
    host_phases._steps = 200
    old_us, new_us, bad = [], [], 0
    for i in range(reps):
        w = windows[i % len(windows)]
        a, b = copy.deepcopy(w), copy.deepcopy(w)
        order = (0, 1) if i % 2 == 0 else (1, 0)
        for which in order:
            if which == 0:
                t0 = time.perf_counter_ns()
                ref.report_ref(a, 200, 0, 200, log_old)
                old_us.append((time.perf_counter_ns() - t0) / 1e3)
            else:
                host_phases._samples = b
                t0 = time.perf_counter_ns()
                host_phases._report()
                new_us.append((time.perf_counter_ns() - t0) / 1e3)
        bad += log_old.last != log_new.last
    po, pn = st.median(old_us), st.median(new_us)
    print(f"HPREPORT_BENCH python {sys.version.split()[0]} numpy {np.__version__} reps {reps} phases "
          f"{len(windows[0])} N 200 groups {len({len(v) for v in windows[0].values()})}")
    print(f"HPREPORT_BENCH old_us p50={po:.0f} p10={np.percentile(old_us, 10):.0f} "
          f"p90={np.percentile(old_us, 90):.0f}")
    print(f"HPREPORT_BENCH new_us p50={pn:.0f} p10={np.percentile(new_us, 10):.0f} "
          f"p90={np.percentile(new_us, 90):.0f}")
    print(f"HPREPORT_BENCH speedup_p50={po / pn:.2f}x saved_us_p50={po - pn:.0f} line_mismatches={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
