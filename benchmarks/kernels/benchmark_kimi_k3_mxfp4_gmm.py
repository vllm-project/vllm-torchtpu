#!/usr/bin/env python3
"""Benchmark Kimi K3's batch-1 MXFP4 grouped GEMMs on TPU.

This models a single TEP32 rank: 28 local experts and a static routed-row
capacity of 16, with a configurable 1--16 active local experts. The
authoritative timing puts a dependent chain inside one compiled loop to
amortize host dispatch.
"""

import argparse
import importlib.util
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.megablox.gmm_v2 import TileSizes, gmm_v2

E = 28
M = 16
H = 3584
I = 3072
BLOCK = 32
ACTIVATION = "situ:4.0:25.0"


def _parse_tile(value: str) -> TileSizes:
    try:
        tm, tk, tn = (int(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tile must be TM,TK,TN") from exc
    return TileSizes(tm, tk, tn, tm)


def _inputs(compact_row: bool, active_experts: int, weight_experts: int,
            scale_block: int):
    # Runtime arguments (rather than closed-over constants) prevent XLA from
    # folding away weight reads.  Zero-valued FP4 is sufficient for timing.
    x = jnp.zeros((1 if compact_row else M, H), jnp.bfloat16)
    w1 = jnp.zeros((weight_experts, H, 2 * I), jnp.float4_e2m1fn)
    s1 = jnp.ones((weight_experts, H // scale_block, 1, 2 * I), jnp.float32)
    w2 = jnp.zeros((weight_experts, I, H), jnp.float4_e2m1fn)
    s2 = jnp.ones((weight_experts, I // scale_block, 1, H), jnp.float32)
    group_sizes = jnp.zeros((weight_experts, ),
                            jnp.int32).at[:active_experts].set(1)
    return x, w1, s1, w2, s2, group_sizes


def _single_group(w, scale, group_sizes):
    expert = jnp.argmax(group_sizes)
    w = jax.lax.dynamic_slice_in_dim(w, expert, 1)
    if scale is not None:
        scale = jax.lax.dynamic_slice_in_dim(scale, expert, 1)
    return w, scale, group_sizes.sum(keepdims=True)


def _gmm1(x, w, scale, group_sizes, tile, single_group_slice=False):
    if single_group_slice:
        w, scale, group_sizes = _single_group(w, scale, group_sizes)
    return gmm_v2(x,
                  w,
                  group_sizes,
                  rhs_scale=scale,
                  zero_initialize=False,
                  maybe_quantize_lhs=False,
                  fuse_act=ACTIVATION,
                  tile_info=tile,
                  group_offset=jnp.array([0], jnp.int32))


def _gmm2(x, w, scale, group_sizes, tile, single_group_slice=False):
    if single_group_slice:
        w, scale, group_sizes = _single_group(w, scale, group_sizes)
    return gmm_v2(x,
                  w,
                  group_sizes,
                  rhs_scale=scale,
                  zero_initialize=False,
                  maybe_quantize_lhs=False,
                  preferred_element_type=jnp.bfloat16,
                  tile_info=tile,
                  group_offset=jnp.array([0], jnp.int32))


def _time_queued(fn: Callable, args: tuple, iterations: int) -> float:
    """Return device microseconds/call for queued compiled executions."""
    compiled = jax.jit(fn).lower(*args).compile()
    value = compiled(*args)
    jax.block_until_ready(value)
    start = time.perf_counter()
    for _ in range(iterations):
        value = compiled(*args)
    jax.block_until_ready(value)
    return (time.perf_counter() - start) * 1e6 / iterations


def _time_dependent(fn: Callable, args: tuple, iterations: int) -> float:
    """Return device microseconds/call for a shape-preserving chain."""
    compiled = jax.jit(fn).lower(*args).compile()
    value = compiled(*args)
    jax.block_until_ready(value)
    start = time.perf_counter()
    for _ in range(iterations):
        value = compiled(value, *args[1:])
    jax.block_until_ready(value)
    return (time.perf_counter() - start) * 1e6 / iterations


def _time_device_loop(fn: Callable, args: tuple,
                      iterations: int) -> list[float]:
    """Time a dependent loop inside one executable to amortize dispatch."""

    def loop_fn(value, *loop_args):
        return jax.lax.fori_loop(0, iterations,
                                 lambda _, carry: fn(carry, *loop_args), value)

    compiled = jax.jit(loop_fn).lower(*args).compile()
    value = compiled(*args)
    jax.block_until_ready(value)
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        value = compiled(value, *args[1:])
        jax.block_until_ready(value)
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return samples


def _profile_dependent(fn: Callable, args: tuple, iterations: int,
                       profile_dir: Path) -> None:
    compiled = jax.jit(fn).lower(*args).compile()
    value = compiled(*args)
    jax.block_until_ready(value)
    profile_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(profile_dir):
        for _ in range(iterations):
            value = compiled(value, *args[1:])
        jax.block_until_ready(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmm1-tile", type=_parse_tile, default=None)
    parser.add_argument("--gmm2-tile", type=_parse_tile, default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--single-group-slice", action="store_true")
    parser.add_argument("--compact-row", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--active-experts", type=int, default=1)
    parser.add_argument("--use-production-wrapper", action="store_true")
    parser.add_argument("--full-fused", action="store_true")
    parser.add_argument("--full-fused-auto-tiles", action="store_true")
    parser.add_argument("--weight-experts", type=int, default=E)
    parser.add_argument("--no-scales", action="store_true")
    parser.add_argument("--scale-block", type=int, default=BLOCK)
    args = parser.parse_args()

    if args.full_fused_auto_tiles and not args.full_fused:
        parser.error("--full-fused-auto-tiles requires --full-fused")
    if not 0 < args.active_experts <= min(M, args.weight_experts):
        parser.error("--active-experts exceeds routed rows or weight experts")
    x, w1, s1, w2, s2, group_sizes = _inputs(args.compact_row,
                                             args.active_experts,
                                             args.weight_experts,
                                             args.scale_block)
    if args.no_scales:
        s1 = s2 = None

    if args.full_fused:
        if x.shape[0] != 1 or args.weight_experts != E:
            parser.error("--full-fused requires --compact-row and 28 experts")
        wrapper_path = (Path(__file__).parents[2] / "src" / "vllm_torchtpu" /
                        "layers" / "common" / "fused_moe_gmm.py")
        spec = importlib.util.spec_from_file_location("_fused_moe_gmm_bench",
                                                      wrapper_path)
        assert spec is not None and spec.loader is not None
        wrapper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper_module)
        if args.full_fused_auto_tiles:

            def auto_gmm_wrapper(lhs,
                                 rhs,
                                 rhs_scale,
                                 rhs_bias,
                                 groups,
                                 group_offset,
                                 zero_initialize=False,
                                 fuse_act=None,
                                 preferred_element_type=None,
                                 rhs_quant_dtype=None):
                is_fp4 = (jnp.issubdtype(rhs.dtype, jnp.floating)
                          and jax.dtypes.itemsize_bits(rhs.dtype) == 4)
                return gmm_v2(
                    lhs,
                    rhs,
                    groups,
                    rhs_scale=rhs_scale,
                    rhs_bias=rhs_bias,
                    group_offset=group_offset[0],
                    zero_initialize=zero_initialize,
                    fuse_act=fuse_act,
                    preferred_element_type=preferred_element_type,
                    maybe_quantize_lhs=not is_fp4,
                    rhs_quant_dtype=rhs_quant_dtype,
                )

            wrapper_module.gmm_wrapper = auto_gmm_wrapper
        topk_ids = jnp.array([[0] + [E] * 15], dtype=jnp.int32)
        topk_weights = jnp.ones((1, 16), dtype=jnp.bfloat16)
        experts_start = jnp.array(0, dtype=jnp.int32)

        def fused_fn(lhs, rhs1, scale1, rhs2, scale2, weights, ids, start):
            return wrapper_module.fused_moe_func(
                lhs,
                rhs1,
                rhs2,
                scale1,
                scale2,
                None,
                None,
                weights,
                ids,
                start,
                topk=16,
                activation=ACTIVATION,
                use_ep=True,
                use_sparse_core=True,
                onehot_moe_permute_threshold=64,
            )

        samples = _time_device_loop(
            fused_fn,
            (x, w1, s1, w2, s2, topk_weights, topk_ids, experts_start),
            args.iterations)
        print(f"device={jax.devices()[0].device_kind} full_fused=True "
              f"auto_tiles={args.full_fused_auto_tiles}")
        print("full_fused_device_loop_us=" + ",".join(f"{sample:.3f}"
                                                      for sample in samples))
        return
    # Use gmm_v2's normal heuristic unless the caller pins a sweep candidate.
    if args.gmm1_tile is None:
        from vllm_torchtpu.kernels.megablox.gmm_v2 import calculate_tiling
        tile1 = calculate_tiling
    else:
        tile1 = args.gmm1_tile
    if args.gmm2_tile is None:
        from vllm_torchtpu.kernels.megablox.gmm_v2 import calculate_tiling
        tile2 = calculate_tiling
    else:
        tile2 = args.gmm2_tile

    if args.use_production_wrapper:
        wrapper_path = (Path(__file__).parents[2] / "src" / "vllm_torchtpu" /
                        "layers" / "common" / "fused_moe_gmm.py")
        spec = importlib.util.spec_from_file_location("_fused_moe_gmm_bench",
                                                      wrapper_path)
        assert spec is not None and spec.loader is not None
        wrapper_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper_module)

        def gmm1_fn(lhs, rhs, scale, groups):
            return wrapper_module.gmm_wrapper(lhs,
                                              rhs,
                                              scale,
                                              None,
                                              groups,
                                              jnp.array([0], jnp.int32),
                                              fuse_act=ACTIVATION)

        def gmm2_fn(lhs, rhs, scale, groups):
            return wrapper_module.gmm_wrapper(
                lhs,
                rhs,
                scale,
                None,
                groups,
                jnp.array([0], jnp.int32),
                preferred_element_type=jnp.bfloat16)
    else:
        gmm1_fn = lambda lhs, rhs, scale, groups: _gmm1(
            lhs, rhs, scale, groups, tile1, args.single_group_slice)
        gmm2_fn = lambda lhs, rhs, scale, groups: _gmm2(
            lhs, rhs, scale, groups, tile2, args.single_group_slice)
    gmm2_input = jnp.zeros((x.shape[0], I), jnp.bfloat16)

    def pair_fn(lhs, rhs1, scale1, rhs2, scale2, groups):
        intermediate = gmm1_fn(lhs, rhs1, scale1, groups)
        return gmm2_fn(intermediate, rhs2, scale2, groups)

    if args.profile_dir is not None:
        _profile_dependent(pair_fn, (x, w1, s1, w2, s2, group_sizes),
                           args.iterations, args.profile_dir)

    gmm1_us = _time_queued(gmm1_fn, (x, w1, s1, group_sizes), args.iterations)
    gmm2_us = _time_queued(gmm2_fn, (gmm2_input, w2, s2, group_sizes),
                           args.iterations)
    pair_us = _time_dependent(pair_fn, (x, w1, s1, w2, s2, group_sizes),
                              args.iterations)
    device_loop_samples = _time_device_loop(pair_fn,
                                            (x, w1, s1, w2, s2, group_sizes),
                                            args.iterations)

    print(f"device={jax.devices()[0].device_kind} E={args.weight_experts} "
          f"M={x.shape[0]} "
          f"active={args.active_experts} "
          f"H={H} I={I} single_group_slice={args.single_group_slice}")
    print(f"gmm1_tile={args.gmm1_tile or 'auto'} gmm1={gmm1_us:.3f} us")
    print(f"gmm2_tile={args.gmm2_tile or 'auto'} gmm2={gmm2_us:.3f} us")
    print(f"pair_same_module={pair_us:.3f} us")
    print("pair_device_loop_us=" + ",".join(f"{sample:.3f}"
                                            for sample in device_loop_samples))
    print(f"separate_sum={gmm1_us + gmm2_us:.3f} us")


if __name__ == "__main__":
    main()
