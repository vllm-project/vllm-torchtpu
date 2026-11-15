# SPDX-License-Identifier: Apache-2.0
"""CPU microbench: `_prepare_inputs` up to the metadata walk, full rebuild vs
VLLM_TPU_STEADY_DECODE_METADATA, in steady decode at ~100K context.

    python tests/runner/bench_steady_decode_metadata.py [--steps 400]

Geometry follows the served Qwen3.8 decode (K=3 drafts, 4 verify rows per
request, attention page 512) with one attention group and a configurable
number of unified-pool mamba groups; the rest of the model's 59 groups only
add per-group walk cost, which the steady path does not change. Device ops
run eagerly on CPU, so absolute numbers are host-side only: on TPU the
skipped walk is one compiled-program dispatch.
"""

import argparse
import statistics
import time

from tests.runner.steady_decode_fakes import Workload, env, make_runner, run_prepare


def bench(
    num_reqs: int,
    steady: bool,
    steps: int,
    mamba_groups: int,
    extra: dict,
    seed: int = 0,
):
    max_model_len = 131072
    wl = Workload(
        seed,
        num_reqs,
        context=100_000,
        churn=0.0,
        prefill=0.0,
        max_model_len=max_model_len,
        attn_block=512,
        mamba_block=512,
    )
    runner = make_runner(
        max_model_len=max_model_len,
        attn_block=512,
        mamba_block=512,
        num_mamba_groups=mamba_groups,
    )
    times = []
    with env(steady=steady, extra=extra):
        for step in range(steps):
            if step:
                wl.step()
            num_decode, num_windowed = wl.segments()
            wl.apply(runner)
            so = wl.scheduler_output()
            t0 = time.perf_counter_ns()
            run_prepare(runner, so, num_decode, num_windowed)
            times.append((time.perf_counter_ns() - t0) / 1e3)
    times = times[20:]
    cache = runner._steady_decode_cache
    return statistics.median(times), statistics.mean(times), cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--mamba-groups", type=int, default=8)
    parser.add_argument(
        "--fast-build",
        action="store_true",
        help="VLLM_TPU_FAST_METADATA_BUILD=1 on both sides (isolates the rest)",
    )
    args = parser.parse_args()
    extra = {"VLLM_TPU_FAST_METADATA_BUILD": "1" if args.fast_build else "0"}
    for num_reqs in (1, 4):
        old = bench(num_reqs, False, args.steps, args.mamba_groups, extra)
        new = bench(num_reqs, True, args.steps, args.mamba_groups, extra)
        c = new[2]
        print(
            f"c{num_reqs}: full p50 {old[0]:.0f} us (mean {old[1]:.0f}) | "
            f"steady p50 {new[0]:.0f} us (mean {new[1]:.0f}) | "
            f"delta p50 {new[0] - old[0]:+.0f} us | hits {c.hits}/{c.hits + c.misses}"
            f" walk skips {c.walk_skips}/{c.walks}"
        )


if __name__ == "__main__":
    main()
