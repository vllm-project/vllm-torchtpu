# SPDX-License-Identifier: Apache-2.0
"""Compose dense HCA compression and attention, preserving cache writes."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    config,
    hca_decode_compressor,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    proj_and_save_state as proj,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention import swa_hca_decode as att


def entry(
    pos,
    dist,
    st,
    kt,
    h,
    w,
    ape,
    norm,
    cs,
    state,
    main,
    sl,
    sp,
    cu,
    ml,
    mp,
    attend,
    sinks,
    q,
    kv,
    sw,
    output,
    state_out,
    main_out,
    sw_out,
    *,
    pcfg,
    ccfg,
    state_stride,
    kv_stride,
    swa_stride,
    main_stride,
    eps,
    sm_scale,
):
    hca_decode_compressor._entry(
        pos,
        st,
        kt,
        h,
        w,
        ape,
        norm,
        cs,
        state,
        main,
        state_out,
        main_out,
        pcfg=pcfg,
        ccfg=ccfg,
        state_stride=state_stride,
        kv_stride=kv_stride,
    )

    @pl.with_scoped(
        swa_kv=pltpu.VMEM((2, 1, 256, 8, 128), jnp.uint8),
        query=pltpu.VMEM((2, 1, 64, 512), jnp.bfloat16),
        output_buf=pltpu.VMEM((2, 1, 64, 512), jnp.bfloat16),
        swa_sems=pltpu.SemaphoreType.DMA((2, 5)),
        main_kv=pltpu.VMEM((2, 256, 8, 128), jnp.uint8),
        main_sems=pltpu.SemaphoreType.DMA((2,)),
    )
    def attention(swa_kv, query, output_buf, swa_sems, main_kv, main_sems):
        att._input_kernel(
            dist,
            sl,
            sp,
            cu,
            ml,
            mp,
            attend,
            sinks,
            q,
            kv,
            sw,
            main,
            output,
            sw_out,
            swa_kv,
            query,
            output_buf,
            swa_sems,
            main_kv,
            main_sems,
            sm_scale=sm_scale,
            logical_page_size=128,
            sliding_window=128,
            block_pages=32,
            swa_pages_per_seq=swa_stride,
            main_pages_per_seq=main_stride,
            positions=pos,
            cos_sin_cache=cs,
            eps=eps,
        )

    attention()


def shared_entry(
    pos,
    dist,
    st,
    kt,
    h,
    w,
    ape,
    norm,
    cs,
    main,
    sl,
    sp,
    cu,
    ml,
    mp,
    attend,
    sinks,
    q,
    kv,
    shared,
    output,
    main_out,
    shared_out,
    **kwargs,
):
    # State and SWA use distinct logical pages of the same physical allocation.
    # One writable operand preserves both updates without cloning that allocation.
    entry(
        pos,
        dist,
        st,
        kt,
        h,
        w,
        ape,
        norm,
        cs,
        shared,
        main,
        sl,
        sp,
        cu,
        ml,
        mp,
        attend,
        sinks,
        q,
        kv,
        shared,
        output,
        shared_out,
        main_out,
        shared_out,
        **kwargs,
    )


@functools.partial(
    jax.jit, static_argnames=("eps", "rms_eps", "sm_scale", "shared_state")
)
def run(
    b,
    h,
    w,
    ape,
    norm,
    state,
    st,
    sw,
    comp_cu,
    comp_pages,
    *,
    eps,
    rms_eps,
    sm_scale,
    shared_state=False,
):
    q, kv, sl, sp, cu, dist, main, ml, attend, mp, sinks, pos, cs = b
    ccfg = config.Configs.make(
        config.Mode.HCA,
        size_n=4,
        physical_page_size=16,
        state_physical_page_size=256,
        state_block_size=32,
        rms_eps=rms_eps,
        tile_n=4,
        head_dim=512,
        rope_head_dim=64,
        compress_ratio=128,
        quant_block=0,
    )
    pcfg = proj.Configs(
        proj.TileSizes(
            proj.PROJECTION_OUTPUT_TILES.get((1, 4096, 1024, 128), 1024),
            proj._select_tile_k(4096),
            8,
        ),
        proj.Dimensions(1024, 4096, 8, 128),
        256,
    )
    hb = pl.BlockSpec(memory_space=pltpu.HBM)
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    # 2 scalar metadata + 19 arrays; state/main/swa donation stays explicit.
    operands = (
        pos,
        dist,
        st,
        comp_pages,
        h,
        w,
        ape,
        norm.reshape(4, 128),
        cs,
        state,
        main,
        sl,
        sp,
        cu,
        ml,
        mp,
        attend,
        sinks,
        q,
        kv,
        sw,
    )
    specs = [hb, hb, hb, hb, vm, vm, hb, hb, hb, hb, hb, hb, hb, hb, hb, vm, vm, vm, hb]
    if shared_state:
        operands = operands[:9] + operands[10:]
        specs = specs[:7] + specs[8:]
    result = pl.pallas_call(
        functools.partial(
            shared_entry if shared_state else entry,
            pcfg=pcfg,
            ccfg=ccfg,
            state_stride=st.size // (comp_cu.size - 1),
            kv_stride=comp_pages.size // (comp_cu.size - 1),
            swa_stride=sp.size // sl.size,
            main_stride=mp.size // ml.size,
            eps=eps,
            sm_scale=sm_scale,
        ),
        out_shape=tuple(
            jax.ShapeDtypeStruct(a.shape, a.dtype)
            for a in ((q, main, sw) if shared_state else (q, state, main, sw))
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(),
            in_specs=specs,
            out_specs=[hb] * (3 if shared_state else 4),
        ),
        input_output_aliases=(
            {17: 0, 9: 1, 19: 2} if shared_state else {18: 0, 9: 1, 10: 2, 20: 3}
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, vmem_limit_bytes=48 * 1024 * 1024
        ),
        name="HCA_COMPRESS_ATTENTION_COMPOSE",
    )(*operands)
    if shared_state:
        output, updated_main, updated_shared = result
        return output, updated_shared, updated_main, updated_shared
    return result
