# SPDX-License-Identifier: Apache-2.0
"""CSA decode projection, boundary metadata and compression in one launch."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    compressor_v1 as ref,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    config,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    kernel as comp,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    proj_and_save_state as proj,
)


def _project_compress(
    slots,
    pages,
    bpos,
    kv,
    first,
    first_rope,
    grid,
    positions,
    hidden,
    weights,
    ape,
    norm,
    cs,
    cache,
    rope,
    out,
    out_rope,
    *,
    pcfg,
    ccfg,
):
    tile_n = pcfg.tile_sizes.tile_n
    tile_m = pcfg.tile_sizes.tile_m
    last_dim = cache.shape[-1]

    @pl.with_scoped(
        acc=pltpu.VMEM((2, tile_n, tile_m // last_dim, 1, last_dim), jnp.float32),
        sem=pltpu.SemaphoreType.REGULAR((2,)),
        counts=pltpu.SMEM((2,), jnp.int32),
    )
    def project(acc, sem, counts):
        # kernel waits all state DMAs before returning: compression may read.
        proj.kernel(
            slots,
            positions,
            hidden,
            weights,
            ape,
            cache,
            out,
            acc,
            sem,
            counts,
            cfgs=pcfg,
        )

    project()

    @pl.with_scoped(window=pltpu.VMEM(ccfg.window_shape(), jnp.float32))
    def compress(window):
        comp.kernel_fn(
            pages,
            bpos,
            kv,
            first,
            first_rope,
            grid,
            norm,
            cs,
            cache,
            rope,
            None,
            out,
            out_rope,
            window,
            cfgs=ccfg,
        )

    compress()


def entry(
    slots,
    pages,
    bpos,
    kv,
    first,
    first_rope,
    grid,
    breq,
    positions,
    hidden,
    weights,
    ape,
    norm,
    cs,
    cache,
    rope,
    out,
    out_rope,
    *,
    pcfg,
    ccfg,
):
    cap = ccfg.dims.size_n

    @pl.with_scoped(
        ps=pltpu.SMEM((cap * 2,), jnp.int32),
        bp=pltpu.SMEM((cap,), jnp.int32),
        ks=pltpu.SMEM((cap,), jnp.int32),
        fm=pltpu.SMEM((cap,), jnp.int32),
        fr=pltpu.SMEM((cap,), jnp.int32),
        gs=pltpu.SMEM((), jnp.int32),
        rq=pltpu.SMEM((cap,), jnp.int32),
        count=pltpu.SMEM((), jnp.int32),
    )
    def build(ps, bp, ks, fm, fr, gs, rq, count):
        count[...] = 0

        @pl.loop(0, cap)
        def initialize(i):
            bp[i] = 0
            ks[i] = -1
            fm[i] = 0
            fr[i] = 0
            rq[i] = 0
            ps[2 * i] = 0
            ps[2 * i + 1] = 0

        @pl.loop(0, cap)
        def compact(i):
            @pl.when(kv[i] >= 0)
            def valid():
                dst = count[...]
                ks[dst] = kv[i]
                bp[dst] = bpos[i]
                rq[dst] = breq[i]
                ps[2 * dst] = pages[2 * i]
                ps[2 * dst + 1] = pages[2 * i + 1]
                fm[dst] = 1
                fr[dst] = 1

                # Match ownership within the existing 4-token compression tile.
                @pl.loop(0, dst % 4)
                def earlier(j):
                    prev = dst - dst % 4 + j

                    @pl.when(ks[prev] // 4 == ks[dst] // 4)
                    def same():
                        fr[dst] = 0

                count[...] = dst + 1

        gs[...] = (count[...] + 3) // 4
        _project_compress(
            slots,
            ps,
            bp,
            ks,
            fm,
            fr,
            gs,
            positions,
            hidden,
            weights,
            ape,
            norm,
            cs,
            cache,
            rope,
            out,
            out_rope,
            pcfg=pcfg,
            ccfg=ccfg,
        )

    build()


def compressor_forward(
    hidden_states,
    wkv_wgate,
    ape,
    norm_weight,
    cos_sin_cache,
    positions,
    block_table,
    query_start_loc,
    kv_block_table,
    cache,
    rope_cache,
    distribution,
    *,
    state_block_size,
    head_dim,
    compress_ratio,
    overlap,
    rms_eps,
    quant_block,
    state_cache=None,
):
    if (head_dim, compress_ratio, overlap, quant_block) != (512, 4, True, 64):
        raise NotImplementedError("Launch fusion currently supports CSA only")
    if state_cache is not None or rope_cache is None or cache.dtype != jnp.int32:
        raise NotImplementedError("Requires shared int32 state/NoPE and split RoPE")
    n, hidden_size = hidden_states.shape
    if not 1 <= n <= 128:
        raise NotImplementedError("Dedicated single projection tile supports B1..128")
    reqs = query_start_loc.size - 1
    # Dedicated one-token-per-request decode contract, checked by the host
    # adapter/validate_decode_metadata before entering a compiled graph.
    if reqs < n:
        raise ValueError("Decode requires at least one metadata row per token")
    if state_block_size != 8 or cache.shape[1:] != (128, 128):
        raise NotImplementedError("Decode specialization requires CSA page128/state8")
    req = jnp.arange(n, dtype=jnp.int32)
    real = req < distribution[0]
    state_stride = ref.block_table_row_stride(block_table, reqs)
    kv_stride = ref.block_table_row_stride(kv_block_table, reqs)
    slots = (
        block_table[req * state_stride + positions // 8] * 128 + (positions % 8) * 16
    )
    slots = jnp.where(real, slots, -1)
    compressed = positions // 4
    kv = kv_block_table[req * kv_stride + compressed // 128] * 128 + compressed % 128
    boundary = real & ((positions + 1) % 4 == 0)
    pad = pl.cdiv(n, 4) * 4 - n
    idx = jnp.arange(n, dtype=jnp.int32)
    valid = boundary
    bpos = jnp.pad(jnp.where(valid, positions[idx], 0), (0, pad))
    breq = jnp.pad(jnp.where(valid, req[idx], 0), (0, pad))
    kv = jnp.pad(jnp.where(valid, kv[idx], -1), (0, pad), constant_values=-1)
    ccfg = config.Configs.make(
        config.Mode.CSA,
        size_n=bpos.size,
        physical_page_size=cache.shape[1],
        state_physical_page_size=cache.shape[1],
        state_block_size=state_block_size,
        rms_eps=rms_eps,
        tile_n=4,
        head_dim=head_dim,
        rope_head_dim=cos_sin_cache.shape[1],
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )
    first = jnp.zeros_like(kv, dtype=jnp.bool_)
    first_rope = jnp.zeros_like(kv, dtype=jnp.bool_)
    grid = jnp.asarray(0, jnp.int32)
    offsets = jnp.arange(1 - ccfg.pages_to_buffer_per_token, 1, dtype=bpos.dtype)
    blocks = jnp.maximum(bpos[:, None] // state_block_size + offsets, 0)
    stride = ref.block_table_row_stride(block_table, reqs)
    pages = block_table[breq[:, None] * stride + blocks].reshape(-1)
    tile_n = min(128, pl.cdiv(n, 8) * 8)
    padded = pl.cdiv(n, tile_n) * tile_n
    if padded != n:
        hidden_states = jnp.pad(hidden_states, ((0, padded - n), (0, 0)))
        positions = jnp.pad(positions, (0, padded - n))
        slots = jnp.pad(slots, (0, padded - n), constant_values=-1)
    state_dim = wkv_wgate.shape[1]
    pcfg = proj.Configs(
        proj.TileSizes(
            state_dim if n == 1 else state_dim // 2,
            proj._select_tile_k(hidden_size),
            tile_n,
        ),
        proj.Dimensions(state_dim, hidden_size, padded, compress_ratio),
        cache.shape[1],
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out, out_rope = pl.pallas_call(
        functools.partial(
            entry,
            pcfg=pcfg,
            ccfg=ccfg,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=8,
            grid=(),
            in_specs=(hbm, hbm, hbm, vmem, vmem, hbm, hbm, hbm),
            out_specs=(hbm, hbm),
        ),
        input_output_aliases={14: 0, 15: 1},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="csa_decode_project_metadata_compress",
    )(
        slots,
        pages,
        bpos,
        kv,
        first,
        first_rope,
        grid,
        breq,
        positions,
        hidden_states,
        wkv_wgate,
        ape,
        norm_weight.reshape(ccfg.head_tiles, 128),
        cos_sin_cache,
        cache,
        rope_cache,
    )
    return out, out_rope, out
