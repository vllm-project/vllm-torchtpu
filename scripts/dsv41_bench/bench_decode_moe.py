#!/usr/bin/env python3
"""Single-launch TPU7x-8 micro-benchmark for Decode mHC & Decode MoE."""

import argparse
import statistics
from typing import Any

from scripts.dsv41_bench import bench_mhc, common

SUBOPS = ("all", "mhc_only", "moe_only")


def _bench_ms(
    ctx: common.WorkerContext, fn, *, warmup: int = 3, iters: int = 10
) -> float:
    samples, _ = common.time_steps(ctx, fn, warmup=warmup, iters=iters)
    return round(float(statistics.median(samples)), 4)


def _run_mhc_sweep(
    ctx: common.WorkerContext, args: argparse.Namespace
) -> dict[str, Any]:
    import torch
    from vllm.distributed import (
        get_tp_group,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_reduce_scatter,
    )

    from vllm_torchtpu.models.vllm.deepseek_v41.common.mhc import (
        initialize_delayed_mhc,
        initialize_mhc_post,
    )

    initialize_delayed_mhc()
    initialize_mhc_post()
    torch._dynamo.config.recompile_limit = 64
    geo = common.Dsv41Geometry()
    tp_rank = get_tp_group().rank_in_group
    mhc_results: dict[str, Any] = {}

    for t_full in (8, 16, 32, 64, 128, 256, 512):
        shard = t_full // ctx.tp_size
        lo, hi = tp_rank * shard, (tp_rank + 1) * shard
        full_cpu = bench_mhc._full_inputs(torch, t_full, geo, args.seed)
        fn_all = bench_mhc._maybe_compile(bench_mhc._make_fn("all", geo), True)
        fn_pre = bench_mhc._maybe_compile(
            bench_mhc._make_fn("delayed_pre_only", geo), True
        )

        inp_tp = bench_mhc._to_tpu(full_cpu)
        inp_sp = bench_mhc._to_tpu(bench_mhc._slice_rows(full_cpu, lo, hi))
        common.synchronize((inp_tp, inp_sp))

        out_tp = bench_mhc._cpu(fn_all(inp_tp))
        out_sp = bench_mhc._cpu(fn_all(inp_sp))
        ref_sp = {k: v[lo:hi] for k, v in out_tp.items()}
        cmp_val = common.compare_tensors(
            out_sp["value"], ref_sp["value"], common.LOSSLESS
        )

        tp_all_ms = _bench_ms(ctx, lambda f=fn_all, x=inp_tp: f(x))
        sp_all_ms = _bench_ms(ctx, lambda f=fn_all, x=inp_sp: f(x))
        tp_pre_ms = _bench_ms(ctx, lambda f=fn_pre, x=inp_tp: f(x))
        sp_pre_ms = _bench_ms(ctx, lambda f=fn_pre, x=inp_sp: f(x))

        # Collective comparison: TP all_reduce([T, D]) vs SP reduce_scatter + all_gather
        x_tp = inp_tp["x"]
        fn_ar = bench_mhc._maybe_compile(
            lambda z: tensor_model_parallel_all_reduce(z.clone()), True
        )
        fn_rs_ag = bench_mhc._maybe_compile(
            lambda z: tensor_model_parallel_all_gather(
                tensor_model_parallel_reduce_scatter(z, 0), 0
            ),
            True,
        )
        tp_coll_ms = _bench_ms(ctx, lambda f=fn_ar, x=x_tp: f(x))
        sp_coll_ms = _bench_ms(ctx, lambda f=fn_rs_ag, x=x_tp: f(x))

        mhc_results[f"T={t_full}"] = {
            "T_full": t_full,
            "T_sp_local": shard,
            "mhc_all_tp_ms": tp_all_ms,
            "mhc_all_sp_ms": sp_all_ms,
            "mhc_pre_tp_ms": tp_pre_ms,
            "mhc_pre_sp_ms": sp_pre_ms,
            "coll_tp_ar_ms": tp_coll_ms,
            "coll_sp_rs_ag_ms": sp_coll_ms,
            "total_2x_per_layer_tp_ms": round(2 * (tp_all_ms + tp_coll_ms), 4),
            "total_2x_per_layer_sp_ms": round(2 * (sp_all_ms + sp_coll_ms), 4),
            "sp_vs_tp_delta_per_layer_ms": round(
                2 * (sp_all_ms + sp_coll_ms) - 2 * (tp_all_ms + tp_coll_ms), 4
            ),
            "cos_sim": round(float(cmp_val["cos_sim"]), 7),
            "max_abs_diff": round(float(cmp_val["max_abs_diff"]), 6),
        }
    return mhc_results


def _run_moe_sweep(
    ctx: common.WorkerContext, args: argparse.Namespace
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas
    from vllm.distributed import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_reduce_scatter,
    )

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME, build_ep_mesh
    from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2
    from vllm_torchtpu.layers.core.fused_moe_gmm import fused_moe_func

    geo = common.Dsv41Geometry()
    d = geo.hidden_size  # 5120
    inter_orig = geo.moe_intermediate_size  # 2304
    inter_padded = 2560  # ceil(2304 / 512) * 512
    e_global = geo.n_routed_experts  # 384
    k_top = geo.num_experts_per_tok  # 6
    ep_size = ctx.tp_size  # 8
    e_local = e_global // ep_size  # 48
    block_size = 512

    g = torch.Generator(device="cpu").manual_seed(args.seed + ctx.rank)
    # K-major uint8 packed FP4 for gmm_v2: w1 [E, K//2, 2*I_pad], w2 [E, I_pad//2, K]
    w1_gmm = torch.randint(
        0, 256, (e_local, d // 2, 2 * inter_padded), generator=g, dtype=torch.uint8
    ).to("tpu")
    w1_gmm[:, :, inter_orig:inter_padded] = 0
    w1_gmm[:, :, inter_padded + inter_orig :] = 0
    w2_gmm = torch.randint(
        0, 256, (e_local, inter_padded // 2, d), generator=g, dtype=torch.uint8
    ).to("tpu")
    w2_gmm[:, inter_orig // 2 :, :] = 0

    w1_scale_gmm = torch.full(
        (e_local, d // block_size, 1, 2 * inter_padded),
        0.01,
        dtype=torch.float32,
        device="tpu",
    )
    w2_scale_gmm = torch.full(
        (e_local, inter_padded // block_size, 1, d),
        0.01,
        dtype=torch.float32,
        device="tpu",
    )

    # For fused_ep_moe_v2: float4_e2m1fn_x2 [E, K, 2*I_pad//2] and [E, I_pad, K//2]
    w1_v2 = (
        torch.randint(
            0, 256, (e_local, d, inter_padded), generator=g, dtype=torch.uint8
        )
        .to("tpu")
        .view(torch.float4_e2m1fn_x2)
    )
    w2_v2 = (
        torch.randint(
            0, 256, (e_local, inter_padded, d // 2), generator=g, dtype=torch.uint8
        )
        .to("tpu")
        .view(torch.float4_e2m1fn_x2)
    )
    w1_scale_v2 = w1_scale_gmm.squeeze(2).contiguous()
    w2_scale_v2 = w2_scale_gmm.squeeze(2).contiguous()
    experts_start_t = torch.tensor(ctx.rank * e_local, dtype=torch.int32, device="tpu")
    rank_buf = torch.tensor([[ctx.rank]], dtype=torch.int32, device="tpu")
    common.synchronize(
        (
            w1_gmm,
            w2_gmm,
            w1_scale_gmm,
            w2_scale_gmm,
            w1_v2,
            w2_v2,
            w1_scale_v2,
            w2_scale_v2,
            experts_start_t,
            rank_buf,
        )
    )

    ep_mesh = build_ep_mesh()
    assert ep_mesh is not None

    def _make_gmm_op(use_sc: bool, onehot_thr: int = 512):
        def _fn(
            hs: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w1s: jax.Array,
            w2s: jax.Array,
            tk_w: jax.Array,
            tk_id: jax.Array,
            exp_start: jax.Array,
        ) -> jax.Array:
            return jax.shard_map(
                lambda h, a1, a2, s1, s2, tw, ti, es: fused_moe_func(
                    h,
                    a1,
                    a2,
                    s1,
                    s2,
                    None,
                    None,
                    tw,
                    ti,
                    experts_start=es,
                    topk=k_top,
                    activation="silu_and_mul_with_clamp",
                    use_ep=True,
                    use_sparse_core=use_sc,
                    onehot_moe_permute_threshold=onehot_thr,
                    rhs_quant_dtype=jnp.float4_e2m1fn,
                ),
                mesh=ep_mesh,
                in_specs=(P(),) * 8,
                out_specs=P(),
                check_vma=False,
            )(hs, w1, w2, w1s, w2s, tk_w, tk_id, exp_start)

        op = pallas.jax_op(
            f"pallas::bench_decode_moe_gmm_sc{int(use_sc)}_thr{onehot_thr}",
            _fn,
            mesh=ep_mesh,
            input_partition_specs=(P(),) * 8,
        )
        op.register_fake(lambda hs, *_: torch.empty_like(hs))
        return op

    from vllm_torchtpu.kernels.experimental.moe_dense_expert import dense_expert_moe

    def _make_dense_expert_op():
        def _fn(
            hs: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w1s: jax.Array,
            w2s: jax.Array,
            tk_w: jax.Array,
            tk_id: jax.Array,
            exp_start: jax.Array,
        ) -> jax.Array:
            def _inner(h, a1, a2, s1, s2, tw, ti, es):
                tw = tw.astype(h.dtype)
                local_ids = ti - es
                valid = (local_ids >= 0) & (local_ids < a1.shape[0])
                tw = jnp.where(valid, tw, jnp.zeros_like(tw))
                ti_loc = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))
                return dense_expert_moe(
                    h,
                    a1,
                    a2,
                    s1,
                    s2,
                    tw,
                    ti_loc,
                    activation="silu_and_mul_with_clamp",
                    rhs_quant_dtype=jnp.float4_e2m1fn,
                )

            return jax.shard_map(
                _inner,
                mesh=ep_mesh,
                in_specs=(P(),) * 8,
                out_specs=P(),
                check_vma=False,
            )(hs, w1, w2, w1s, w2s, tk_w, tk_id, exp_start)

        op = pallas.jax_op(
            "pallas::bench_decode_moe_dense_expert",
            _fn,
            mesh=ep_mesh,
            input_partition_specs=(P(),) * 8,
        )
        op.register_fake(lambda hs, *_: torch.empty_like(hs))
        return op

    gmm_sc1_op = _make_gmm_op(True, 512)
    gmm_sc1_raw_op = _make_gmm_op(True, 0)
    gmm_sc0_op = _make_gmm_op(False, 0)
    dense_exp_op = _make_dense_expert_op()

    all_ranks_group = (tuple(range(ep_size)),)
    sp_ranks_group = tuple((r,) for r in range(ep_size))

    def _make_v2_op(tag: str, token_groups):
        def _fn(
            x: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w1s: jax.Array,
            w2s: jax.Array,
            gating: jax.Array,
            rank: jax.Array,
        ) -> jax.Array:
            return fused_ep_moe_v2(
                x,
                w1,
                w2,
                w1s,
                w2s,
                gating.astype(jnp.float32),
                rank=rank,
                topk=k_top,
                renormalize=True,
                mesh=ep_mesh,
                capacity=128,
                block=None,
                ragged_stride=None,
                weight_format=WeightFormat("fp4"),
                rhs_qb=block_size,
                act_fn="silu_and_mul_with_clamp",
                mesh_ep_ranks=None,
                sharded_plan=True,
                token_replica_groups=token_groups,
                scoring_fn="sigmoid",
                routed_scaling_factor=2.5,
            )

        spec = P(EP_AXIS_NAME)
        op = pallas.jax_op(
            f"pallas::bench_decode_moe_v2_{tag}",
            _fn,
            mesh=ep_mesh,
            input_partition_specs=(spec,) * 7,
            output_partition_specs=spec,
        )
        op.register_fake(lambda x, *_: torch.empty_like(x))
        return op

    v2_tp_op = _make_v2_op("tp", all_ranks_group)
    v2_sp_op = _make_v2_op("sp", sp_ranks_group)

    moe_results: dict[str, Any] = {}
    for t_full in (4, 8, 16, 32, 64):
        g_tok = torch.Generator(device="cpu").manual_seed(args.seed + t_full)
        x_full = (
            torch.randn((t_full, d), generator=g_tok, dtype=torch.bfloat16) * 0.1
        ).to("tpu")
        logits_full = (
            torch.randn((t_full, e_global), generator=g_tok, dtype=torch.float32) * 0.1
        ).to("tpu")
        scores = torch.sigmoid(logits_full)
        topk_vals, topk_ids = torch.topk(scores, k=k_top, dim=-1)
        topk_weights = (
            (topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-20)) * 2.5
        ).to(torch.bfloat16)
        topk_ids = topk_ids.to(torch.int32)
        common.synchronize((x_full, logits_full, topk_weights, topk_ids))

        def _gmm_sc1_tp(x=x_full, tw=topk_weights, ti=topk_ids):
            return tensor_model_parallel_all_reduce(
                gmm_sc1_op(
                    x,
                    w1_gmm,
                    w2_gmm,
                    w1_scale_gmm,
                    w2_scale_gmm,
                    tw,
                    ti,
                    experts_start_t,
                )
            )

        def _gmm_sc1_raw_tp(x=x_full, tw=topk_weights, ti=topk_ids):
            return tensor_model_parallel_all_reduce(
                gmm_sc1_raw_op(
                    x,
                    w1_gmm,
                    w2_gmm,
                    w1_scale_gmm,
                    w2_scale_gmm,
                    tw,
                    ti,
                    experts_start_t,
                )
            )

        def _gmm_sc0_tp(x=x_full, tw=topk_weights, ti=topk_ids):
            return tensor_model_parallel_all_reduce(
                gmm_sc0_op(
                    x,
                    w1_gmm,
                    w2_gmm,
                    w1_scale_gmm,
                    w2_scale_gmm,
                    tw,
                    ti,
                    experts_start_t,
                )
            )

        def _dense_exp_tp(x=x_full, tw=topk_weights, ti=topk_ids):
            return tensor_model_parallel_all_reduce(
                dense_exp_op(
                    x,
                    w1_gmm,
                    w2_gmm,
                    w1_scale_gmm,
                    w2_scale_gmm,
                    tw,
                    ti,
                    experts_start_t,
                )
            )

        out_sc1 = _gmm_sc1_tp()
        out_sc0 = _gmm_sc0_tp()
        out_dense = _dense_exp_tp()
        common.synchronize((out_sc1, out_sc0, out_dense))
        cmp_sc = common.compare_tensors(out_sc1.cpu(), out_sc0.cpu(), common.LOSSLESS)
        cmp_dense = common.compare_tensors(
            out_dense.cpu(), out_sc1.cpu(), common.LOSSLESS
        )

        row: dict[str, Any] = {
            "T": t_full,
            "gmm_sc1_onehot_tp_ms": _bench_ms(ctx, _gmm_sc1_tp),
            "gmm_sc1_ragged_tp_ms": _bench_ms(ctx, _gmm_sc1_raw_tp),
            "gmm_sc0_argsort_tp_ms": _bench_ms(ctx, _gmm_sc0_tp),
            "dense_expert_tp_ms": _bench_ms(ctx, _dense_exp_tp),
            "sc1_vs_sc0_cos": round(float(cmp_sc["cos_sim"]), 6),
            "dense_vs_sc1_cos": round(float(cmp_dense["cos_sim"]), 7),
            "dense_vs_sc1_max_abs": round(float(cmp_dense["max_abs_diff"]), 6),
        }

        if t_full >= ep_size and t_full % ep_size == 0:

            def _v2_w4a8_tp(x=x_full, lg=logits_full):
                return v2_tp_op(
                    x,
                    w1_v2,
                    w2_v2,
                    w1_scale_v2,
                    w2_scale_v2,
                    lg,
                    rank_buf,
                )

            out_v2 = _v2_w4a8_tp()
            common.synchronize(out_v2)
            row["fused_ep_v2_w4a8_tp_ms"] = _bench_ms(ctx, _v2_w4a8_tp)

            t_local = t_full // ep_size
            x_sp = x_full[ctx.rank * t_local : (ctx.rank + 1) * t_local].contiguous()
            logits_sp = logits_full[
                ctx.rank * t_local : (ctx.rank + 1) * t_local
            ].contiguous()
            common.synchronize((x_sp, logits_sp))

            def _gmm_sc1_sp(xs=x_sp, tw=topk_weights, ti=topk_ids):
                x_ag = tensor_model_parallel_all_gather(xs, dim=0)
                out = gmm_sc1_op(
                    x_ag,
                    w1_gmm,
                    w2_gmm,
                    w1_scale_gmm,
                    w2_scale_gmm,
                    tw,
                    ti,
                    experts_start_t,
                )
                return tensor_model_parallel_reduce_scatter(out, dim=0)

            def _dense_exp_sp(xs=x_sp, tw=topk_weights, ti=topk_ids):
                x_ag = tensor_model_parallel_all_gather(xs, dim=0)
                out = dense_exp_op(
                    x_ag,
                    w1_gmm,
                    w2_gmm,
                    w1_scale_gmm,
                    w2_scale_gmm,
                    tw,
                    ti,
                    experts_start_t,
                )
                return tensor_model_parallel_reduce_scatter(out, dim=0)

            def _v2_w4a8_sp(xs=x_sp, lgs=logits_sp):
                return v2_sp_op(
                    xs,
                    w1_v2,
                    w2_v2,
                    w1_scale_v2,
                    w2_scale_v2,
                    lgs,
                    rank_buf,
                )

            out_v2_sp = _v2_w4a8_sp()
            out_dense_sp = _dense_exp_sp()
            common.synchronize((out_v2_sp, out_dense_sp))
            row["gmm_sc1_onehot_sp_ms"] = _bench_ms(ctx, _gmm_sc1_sp)
            row["dense_expert_sp_ms"] = _bench_ms(ctx, _dense_exp_sp)
            row["fused_ep_v2_w4a8_sp_ms"] = _bench_ms(ctx, _v2_w4a8_sp)

        moe_results[f"T={t_full}"] = row
    return moe_results


def worker(ctx: common.WorkerContext, args: argparse.Namespace) -> dict[str, Any]:
    with common.model_wrapper_context():
        mhc_res = _run_mhc_sweep(ctx, args) if args.subop in ("all", "mhc_only") else {}
        moe_res = _run_moe_sweep(ctx, args) if args.subop in ("all", "moe_only") else {}
    if ctx.rank == 0:
        print(
            "DECODE_MOE_BENCH_SUMMARY:",
            common.json.dumps({"mhc": mhc_res, "moe": moe_res}, indent=2),
        )
    return {
        "wall_samples_ms": [1.0],
        "mhc_decode": mhc_res,
        "moe_decode": moe_res,
        "hbm": common.hbm_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_args(parser, SUBOPS)
    common.run_main("decode_moe_mhc", parser, worker)


if __name__ == "__main__":
    main()
