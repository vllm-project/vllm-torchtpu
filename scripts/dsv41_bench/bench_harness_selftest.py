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
"""Harness self-test: vLLM TP collectives on the residual stream shape.

Exercises every piece of ``common.py`` (launch, vLLM TP group, timing,
``--verify``, golden write/read, XProf, LLO dump, roofline, HBM) on the
collectives that ``v2`` (``all_reduce``) and ``TP8+SP`` (``reduce_scatter`` /
``all_gather``) put on ``bf16[T, hidden]``.

Inputs are small integers so every reduction is exact in bf16 and the
analytic reference is bitwise comparable.

  python scripts/dsv41_bench/bench_harness_selftest.py --subop all_reduce
  python scripts/dsv41_bench/bench_harness_selftest.py --subop reduce_scatter \
      --profile /tmp/prof --dump-llo /tmp/llo
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common  # noqa: E402

SUBOPS = ("all_reduce", "reduce_scatter", "all_gather")
CALLS_PER_STEP = 41


def _rank_input(torch, rank: int, tokens: int, hidden: int):
    """Row ``t`` col ``h`` holds ``(t + h + rank) % 7``: exact in bf16."""
    t = torch.arange(tokens, dtype=torch.int32)[:, None]
    h = torch.arange(hidden, dtype=torch.int32)[None, :]
    return ((t + h + rank) % 7).to(torch.bfloat16)


def _reference(torch, subop: str, rank: int, world: int, tokens: int, hidden: int):
    full = sum(_rank_input(torch, r, tokens, hidden).float() for r in range(world))
    if subop == "all_reduce":
        return full.to(torch.bfloat16)
    if subop == "reduce_scatter":
        shard = tokens // world
        return full[rank * shard : (rank + 1) * shard].to(torch.bfloat16)
    return torch.cat(
        [_rank_input(torch, r, tokens // world, hidden) for r in range(world)]
    )


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict:
    import torch
    from vllm.distributed import (
        get_tp_group,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )

    geo = common.Dsv41Geometry()
    tokens, hidden = args.chunk_size, geo.hidden_size
    if tokens % ctx.tp_size:
        raise ValueError(f"--chunk-size {tokens} must divide by tp {ctx.tp_size}")
    tp_rank = get_tp_group().rank_in_group

    if args.subop == "all_gather":
        x = _rank_input(torch, tp_rank, tokens // ctx.tp_size, hidden).to("tpu")

        def step():
            return tensor_model_parallel_all_gather(x, dim=0)

    elif args.subop == "reduce_scatter":
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        x = _rank_input(torch, tp_rank, tokens, hidden).to("tpu")

        def step():
            return tensor_model_parallel_reduce_scatter(x, dim=0)

    else:
        x = _rank_input(torch, tp_rank, tokens, hidden).to("tpu")

        def step():
            return tensor_model_parallel_all_reduce(x)

    # Eager DeviceCommunicatorBase.all_reduce is dist.all_reduce(input_) in
    # place, so repeated steps compound into x; verify the first call only.
    common.synchronize(x)
    x_fresh = x.clone()
    out = step()
    common.synchronize(out)
    first_out = out.cpu()
    x.copy_(x_fresh)
    del x_fresh
    common.synchronize(x)

    hbm_before = common.hbm_stats()
    samples, _ = common.time_steps(ctx, step, warmup=args.warmup, iters=args.iters)
    profile_dir = common.profile_steps(ctx, args, step, label=f"selftest_{args.subop}")

    expected = _reference(torch, args.subop, tp_rank, ctx.tp_size, tokens, hidden)
    report = {}
    if args.verify:
        report = common.verify_outputs(
            {"out": first_out}, {"out": expected}, common.EXACT, label=args.subop
        )
    golden = common.check_against_golden(
        ctx, args, f"selftest_{args.subop}_T{tokens}", {"out": first_out}, common.EXACT
    )

    hw = common.Tpu7xRoofline()
    nbytes = tokens * hidden * 2
    comm_s = {
        "all_reduce": hw.all_reduce_s,
        "reduce_scatter": hw.reduce_scatter_s,
        "all_gather": hw.all_gather_s,
    }[args.subop](nbytes, ctx.tp_size)
    roof = common.RooflineEstimate(
        f"{args.subop} bf16[{tokens},{hidden}] x{ctx.tp_size}", comm_s=comm_s
    ).report(hw, statistics.median(samples))
    roof["x41_per_step_lower_bound_ms"] = roof["lower_bound_ms"] * CALLS_PER_STEP
    return {
        "wall_samples_ms": samples,
        "verify": report,
        "golden": golden,
        "roofline_vs_wall": roof,
        "hbm": {"before": hbm_before, "after": common.hbm_stats()},
        "profile_dir": profile_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, SUBOPS)
    common.run_main("harness_selftest", parser, worker)


if __name__ == "__main__":
    main()
