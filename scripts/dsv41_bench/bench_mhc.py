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
"""Replay ``deepseek_v41/v2`` mHC (``shifted_post_pre``) and its collective.

``--mode tp`` replays today's TP8 layout: every rank runs mHC on all ``T``
tokens and the sublayer output is all-reduced. ``--mode sp`` replays
TP8+SP: each rank runs mHC on its ``T/8`` token slice and the sublayer
output is reduce-scattered.

``--verify`` (default on): in ``sp`` mode the rank's slice must match the
``tp`` result for the same rows computed in-run; in ``tp`` mode outputs are
checked against the recorded golden (``--update-golden``).

Subops:
  all              post + delayed pre (Pallas flat sums + gates) + rms_norm
  mhc_post_only    pallas::deepseek_v4_mhc_post_v1
  delayed_pre_only pallas::dsv41v2_delayed_mhc_pre_flat (incl. Sinkhorn)
  sinkhorn_only    gate/softmax/Sinkhorn on [T, 24] mix logits
  rms_norm_only    v2 ops.rms_norm on [T, 5120]
  collective_only  all_reduce (tp) or reduce_scatter (sp) of bf16[T, 5120]

  python scripts/dsv41_bench/bench_mhc.py --mode tp --subop all
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import common  # noqa: E402

SUBOPS = (
    "all",
    "mhc_post_only",
    "delayed_pre_only",
    "sinkhorn_only",
    "rms_norm_only",
    "collective_only",
)
HC_EPS = 1e-6
HC_POST_ALPHA = 2.0
RMS_EPS = 1e-20
NORM_EPS = 1e-20
# Collectives are checked against an fp32-accumulated truth rounded once to
# bf16. AR and RS round partial sums to bf16 in implementation-defined order,
# so only the coordination-doc gate (cos >= 0.9999) is enforced; rel_l2 and
# max_abs are reported per path in ``diagnostics_vs_fp32_truth``.
COLLECTIVE_MATCH = common.MatchCriteria("lossless", min_cos=0.9999, max_rel_l2=1.0)


def _full_inputs(torch, tokens: int, geo: common.Dsv41Geometry, seed: int):
    """Rank-identical inputs for all ``tokens`` rows, built on CPU."""
    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import pack_mix_weight

    g = torch.Generator().manual_seed(seed)
    m, d = geo.hc_mult, geo.hidden_size
    mixes = m * (m + 2)

    def randn(*shape, scale=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    post_mix = torch.sigmoid(randn(tokens, m, 1)) * HC_POST_ALPHA
    comb = torch.softmax(randn(tokens, m, m), dim=-1)
    pre_mix = torch.sigmoid(randn(tokens, m)) + HC_EPS
    return {
        "x": randn(tokens, d, dtype=torch.bfloat16),
        # Flat [T, M*D] like v2's in-graph residual: a [T, M, D] parameter gets
        # T(4,128) tiling and adds a SparseCore relayout that production lacks.
        "residual": randn(tokens, m, d, dtype=torch.bfloat16).flatten(1),
        "post_mix": post_mix,
        "comb": comb,
        "pre_mix": pre_mix,
        "fn": pack_mix_weight(randn(mixes, m * d, scale=1e-2)),
        "hc_scale": randn(3, scale=0.1) + 1.0,
        "hc_base": randn(mixes, scale=0.1),
        "norm_weight": (randn(d, scale=0.1) + 1.0).to(torch.bfloat16),
        "mixes": randn(tokens, mixes),
        "sqrsum": randn(tokens, 1).abs() * (m * d),
    }


def _gates_op(geo: common.Dsv41Geometry):
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    # jax_op inspects annotations with eval_str=True against module globals;
    # this module uses postponed annotations and imports jax lazily, so set
    # concrete annotation objects instead of using functools.partial.
    def gates(mixes, sqrsum, scale, base):
        return _gates_jax(
            mhc._pre_gates,
            geo.hc_mult,
            geo.hidden_size,
            geo.hc_sinkhorn_iters,
            mixes,
            sqrsum,
            scale,
            base,
        )

    gates.__annotations__ = {
        **dict.fromkeys(("mixes", "sqrsum", "scale", "base"), jax.Array),
        "return": tuple[jax.Array, jax.Array, jax.Array],
    }

    op = pallas.jax_op(
        "bench::dsv41v2_mhc_pre_gates",
        gates,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(), P(), P(), P()),
    )

    def fake(mixes, sqrsum, scale, base):
        t, mult = mixes.shape[0], geo.hc_mult
        return (
            mixes.new_empty((t, mult)),
            mixes.new_empty((t, mult)),
            mixes.new_empty((t, mult, mult)),
        )

    op.register_fake(fake)
    return op


def _gates_jax(pre_gates, mult, dim, repeats, mixes, sqrsum, scale, base):
    return pre_gates(
        mixes,
        sqrsum,
        mult,
        dim,
        scale,
        base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        HC_POST_ALPHA,
        repeats,
    )


def _make_fn(subop: str, geo: common.Dsv41Geometry):
    from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.ops import rms_norm

    consts = (RMS_EPS, HC_EPS, HC_EPS, HC_POST_ALPHA, geo.hc_sinkhorn_iters)

    def unflat(residual):
        return residual.unflatten(1, (geo.hc_mult, geo.hidden_size))

    if subop == "all":

        def fn(i):
            residual, post, comb, value, next_pre, _ = mhc.shifted_post_pre(
                i["x"],
                unflat(i["residual"]),
                i["post_mix"],
                i["comb"],
                i["fn"],
                i["hc_scale"],
                i["hc_base"],
                *consts,
                pre_mix=i["pre_mix"],
                norm_weight=i["norm_weight"],
                norm_eps=NORM_EPS,
            )
            # Flat [T, M*D] is what v2's next delayed pre consumes; returning
            # [T, M, D] adds an output-only relayout to the timing.
            return {
                "residual": residual.flatten(1),
                "post": post,
                "comb": comb,
                "value": value,
                "next_pre": next_pre,
            }

    elif subop == "mhc_post_only":

        def fn(i):
            out = mhc.post(i["x"], unflat(i["residual"]), i["post_mix"], i["comb"])
            return {"residual": out.flatten(1)}

    elif subop == "delayed_pre_only":

        def fn(i):
            post, comb, value, next_pre = mhc.delayed_pre(
                unflat(i["residual"]),
                None,
                i["fn"],
                i["hc_scale"],
                i["hc_base"],
                i["pre_mix"],
                *consts,
            )
            return {"post": post, "comb": comb, "value": value, "next_pre": next_pre}

    elif subop == "sinkhorn_only":
        op = _gates_op(geo)

        def fn(i):
            pre, post, comb = op(i["mixes"], i["sqrsum"], i["hc_scale"], i["hc_base"])
            return {"pre": pre, "post": post, "comb": comb}

    elif subop == "rms_norm_only":

        def fn(i):
            return {"value": rms_norm(i["x"], i["norm_weight"], NORM_EPS)}

    else:
        raise ValueError(subop)
    return fn


_ROW_SHARDED = ("x", "residual", "post_mix", "comb", "pre_mix", "mixes", "sqrsum")


def _slice_rows(inputs: dict, start: int, stop: int) -> dict:
    return {k: (v[start:stop] if k in _ROW_SHARDED else v) for k, v in inputs.items()}


def _to_tpu(inputs: dict) -> dict:
    return {k: v.to("tpu") for k, v in inputs.items()}


def _cpu(outputs: dict) -> dict:
    return {k: v.cpu() for k, v in outputs.items()}


def _maybe_compile(fn, enabled: bool):
    if not enabled:
        return fn
    import torch
    from torch_tpu._internal import compile as tpu_compile

    return torch.compile(
        fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
    )


def _roofline(subop: str, tokens: int, geo: common.Dsv41Geometry, tp: int, mode: str):
    """HBM bytes / FLOPs of the dominant streams; small gate tensors ignored."""
    m, d = geo.hc_mult, geo.hidden_size
    bf16 = 2
    stream = tokens * m * d * bf16
    row = tokens * d * bf16
    hw = common.Tpu7xRoofline()
    if subop == "collective_only":
        nbytes = tokens * tp * d * bf16 if mode == "sp" else row
        comm = (
            hw.reduce_scatter_s(nbytes, tp)
            if mode == "sp"
            else hw.all_reduce_s(nbytes, tp)
        )
        return common.RooflineEstimate(f"{mode} collective", comm_s=comm)
    post = common.RooflineEstimate("post", hbm_bytes=row + 2 * stream)
    pre = common.RooflineEstimate(
        "delayed_pre",
        flops=2 * tokens * m * d * 128,
        hbm_bytes=stream + row + m * d * 128 * bf16,
    )
    norm = common.RooflineEstimate("rms_norm", hbm_bytes=2 * row)
    sink = common.RooflineEstimate("sinkhorn", hbm_bytes=tokens * (24 + 1 + 24) * 4)
    if subop == "all":
        return common.RooflineEstimate(
            "all",
            flops=pre.flops,
            hbm_bytes=post.hbm_bytes + pre.hbm_bytes + norm.hbm_bytes,
        )
    return {
        "mhc_post_only": post,
        "delayed_pre_only": pre,
        "rms_norm_only": norm,
        "sinkhorn_only": sink,
    }[subop]


def _collective(ctx, args, tokens_full: int, geo: common.Dsv41Geometry) -> dict:
    import torch
    from vllm.distributed import (
        get_tp_group,
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_reduce_scatter,
    )

    tp_rank = get_tp_group().rank_in_group

    def rank_input(r: int):
        g = torch.Generator().manual_seed(args.seed + 1000 + r)
        return torch.randn(tokens_full, geo.hidden_size, generator=g).to(torch.bfloat16)

    x_cpu = rank_input(tp_rank)
    x = x_cpu.to("tpu")
    shard = tokens_full // ctx.tp_size
    rows = slice(tp_rank * shard, (tp_rank + 1) * shard)
    # Ground truth: fp32 sum of every rank's bf16 input, rounded once to bf16.
    truth = sum(rank_input(r)[rows].float() for r in range(ctx.tp_size))
    truth = truth.to(torch.bfloat16)

    def ar():
        return tensor_model_parallel_all_reduce(x)

    def rs():
        return tensor_model_parallel_reduce_scatter(x, dim=0)

    step_fn = rs if args.mode == "sp" else ar
    step = _maybe_compile(step_fn, args.compile)
    # Eager all_reduce is in place (FINDINGS §3): verify fresh calls only.
    ar_eager = ar().cpu()[rows]
    x.copy_(x_cpu.to("tpu"))
    first = step()
    common.synchronize(first)
    first_cpu = first.cpu()
    if args.mode == "tp":
        first_cpu = first_cpu[rows]
    x.copy_(x_cpu.to("tpu"))
    common.synchronize(x)
    diag = {
        name: common.compare_tensors(val, truth, COLLECTIVE_MATCH)
        for name, val in (("eager_all_reduce", ar_eager), (args.mode, first_cpu))
    }
    return {
        "step": step,
        "outputs": {"out": first_cpu},
        "reference": {"out": truth},
        "diagnostics_vs_fp32_truth": diag,
        "tokens_local": tokens_full if args.mode == "tp" else shard,
    }


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict:
    import torch
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import (
        initialize_delayed_mhc,
        initialize_mhc_post,
    )

    geo = common.Dsv41Geometry()
    tokens_full = args.chunk_size
    if tokens_full % (ctx.tp_size * 64):
        raise ValueError(
            f"--chunk-size {tokens_full} must be a multiple of "
            f"64*tp={64 * ctx.tp_size} so each SP slice stays on the Pallas "
            "flat-pre path"
        )
    tp_rank = get_tp_group().rank_in_group
    shard = tokens_full // ctx.tp_size
    lo, hi = tp_rank * shard, (tp_rank + 1) * shard

    with common.model_wrapper_context():
        initialize_delayed_mhc()
        initialize_mhc_post()
        if args.subop == "collective_only":
            run = _collective(ctx, args, tokens_full, geo)
            step, outputs, reference = run["step"], run["outputs"], run["reference"]
            tokens_local = run["tokens_local"]
            coll_diag = run["diagnostics_vs_fp32_truth"]
        else:
            coll_diag = {}
            full_cpu = _full_inputs(torch, tokens_full, geo, args.seed)
            fn = _make_fn(args.subop, geo)
            compiled = _maybe_compile(fn, args.compile)
            full_out = _cpu(compiled(_to_tpu(full_cpu)))
            reference = {k: v[lo:hi] for k, v in full_out.items()}
            if args.mode == "sp":
                inputs = _to_tpu(_slice_rows(full_cpu, lo, hi))
                tokens_local = shard
                outputs = _cpu(compiled(inputs))
            else:
                inputs = _to_tpu(full_cpu)
                tokens_local = tokens_full
                outputs = full_out
            common.synchronize(inputs)

            def step():
                return compiled(inputs)

        hbm_before = common.hbm_stats()
        samples, _ = common.time_steps(ctx, step, warmup=args.warmup, iters=args.iters)
        profile_dir = common.profile_steps(
            ctx, args, step, label=f"mhc_{args.mode}_{args.subop}"
        )

    report: dict = {}
    is_coll = args.subop == "collective_only"
    sp_match = COLLECTIVE_MATCH if is_coll else common.LOSSLESS
    if args.verify and (args.mode == "sp" or is_coll):
        ref_label = "fp32-truth" if is_coll else "tp"
        report = common.verify_outputs(
            outputs,
            reference,
            sp_match,
            label=f"{args.mode}-vs-{ref_label} {args.subop}",
        )
    suffix = "compiled" if args.compile else "eager"
    tp_case = f"mhc_{args.subop}_T{tokens_full}_tp_{suffix}"
    if args.mode == "tp":
        golden = common.check_against_golden(ctx, args, tp_case, outputs, common.EXACT)
    elif args.update_golden:
        raise ValueError("record goldens with --mode tp; sp checks the tp golden rows")
    elif args.verify:
        tp_golden, meta = common.load_golden(
            common.golden_path(args.golden_dir, tp_case, ctx.rank)
        )
        # collective_only already stores the rank's [T/tp] shard in tp mode.
        rows = tp_golden if is_coll else {k: v[lo:hi] for k, v in tp_golden.items()}
        golden = {
            "golden": "verified_sp_rows_vs_tp_golden",
            "golden_meta": meta,
            "report": common.verify_outputs(
                outputs, rows, sp_match, label=f"sp-vs-tp-golden {args.subop}"
            ),
        }
    else:
        golden = {"golden": "skipped"}

    est = _roofline(args.subop, tokens_local, geo, ctx.tp_size, args.mode)
    return {
        "mode": args.mode,
        "tokens_local": tokens_local,
        "wall_samples_ms": samples,
        "verify": report,
        "diagnostics_vs_fp32_truth": coll_diag,
        "golden": golden,
        "roofline_vs_wall": est.report(
            common.Tpu7xRoofline(), statistics.median(samples)
        ),
        "hbm": {"before": hbm_before, "after": common.hbm_stats()},
        "profile_dir": profile_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common.add_common_args(parser, SUBOPS)
    parser.add_argument("--mode", choices=["tp", "sp"], default="tp")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="wrap the step in torch.compile(TpuBackend) as in serving",
    )
    common.run_main("mhc", parser, worker)


if __name__ == "__main__":
    main()
