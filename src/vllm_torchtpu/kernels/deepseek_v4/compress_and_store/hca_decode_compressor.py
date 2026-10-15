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
"""Single-query HCA metadata, projection and compression launch fusion."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import config
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import kernel as comp
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    proj_and_save_state as proj,
)
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store.tuned_block_sizes import (
    HCA_DECODE_METADATA_PAGES,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import get_device_name

logger = init_logger(__name__)


def is_supported(
    hidden: jax.Array,
    weights: jax.Array,
    cache: jax.Array,
    state: jax.Array | None,
    cu: jax.Array,
    st: jax.Array,
    kt: jax.Array,
    *,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    state_block_size: int,
) -> bool:
    """Select only the measured dense-query geometry and metadata capacity."""
    if not (
        head_dim == 512
        and compress_ratio == 128
        and not overlap
        and state_block_size == 32
        and hidden.shape == (1, 4096)
        and weights.shape == (4096, 1024)
        and hidden.dtype == weights.dtype == jnp.bfloat16
        and state is not None
        and state.shape[1:] in ((256, 128), (256, 4, 128))
        and (
            (state.ndim == 3 and state.dtype == jnp.int32)
            or (state.ndim == 4 and state.dtype == jnp.uint8)
        )
        and cache.shape[1:] == (16, 4, 128)
        and cache.dtype == jnp.uint8
        and cache.shape[0] > 0
        and state.shape[0] > 0
        and cu.size >= 2
        and st.ndim == kt.ndim == cu.ndim == 1
        and st.dtype == kt.dtype == cu.dtype == jnp.int32
        and st.size % (cu.size - 1) == 0
        and kt.size % (cu.size - 1) == 0
    ):
        return False
    assert state is not None
    key = (
        get_device_name(),
        hidden.shape[1],
        weights.shape[1],
        compress_ratio,
        state_block_size,
        state.shape[1],
    )
    pages = HCA_DECODE_METADATA_PAGES.get(key)
    if pages is None:
        logger.warning_once("HCA compressor fusion tuning unavailable for key=%s", key)
        return False
    return (
        0 < st.size // (cu.size - 1) <= pages[0]
        and 0 < kt.size // (cu.size - 1) <= pages[1]
    )


def _entry(
    pos,
    state_table,
    kv_table,
    hidden,
    weights,
    ape,
    norm,
    cs,
    state,
    cache,
    state_out,
    cache_out,
    *,
    pcfg,
    ccfg,
    state_stride,
    kv_stride,
):
    n = pcfg.tile_sizes.tile_n
    pages_per_token = ccfg.pages_to_buffer_per_token
    # HBM slices must cover complete physical lane tiles. A single-request
    # table is copied whole; for larger tables the extra next-row entries
    # are staged but never indexed.
    state_copy_size = min(pl.cdiv(state_stride, 128) * 128, state_table.shape[0])
    kv_copy_size = min(pl.cdiv(kv_stride, 128) * 128, kv_table.shape[0])

    @pl.with_scoped(
        slots=pltpu.SMEM((n,), jnp.int32),
        pages=pltpu.SMEM((4 * pages_per_token,), jnp.int32),
        bpos=pltpu.SMEM((4,), jnp.int32),
        kv=pltpu.SMEM((4,), jnp.int32),
        first=pltpu.SMEM((4,), jnp.bool_),
        grid=pltpu.SMEM((), jnp.int32),
        st=pltpu.SMEM((state_copy_size,), jnp.int32),
        kt=pltpu.SMEM((kv_copy_size,), jnp.int32),
        st_sem=pltpu.SemaphoreType.DMA,
        kt_sem=pltpu.SemaphoreType.DMA,
    )
    def run(slots, pages, bpos, kv, first, grid, st, kt, st_sem, kt_sem):
        state_dma = pltpu.make_async_copy(
            state_table.at[pl.ds(0, state_copy_size)], st, st_sem
        )
        kv_dma = pltpu.make_async_copy(kv_table.at[pl.ds(0, kv_copy_size)], kt, kt_sem)
        state_dma.start()
        kv_dma.start()
        p = pos[0]
        boundary = (p + 1) % ccfg.dims.compress_ratio == 0
        for i in range(n):
            slots[i] = -1
        state_dma.wait()
        kv_dma.wait()
        slots[0] = (
            st[p // ccfg.state_block_size] * ccfg.state_physical_page_size
            + (p % ccfg.state_block_size) * ccfg.state_rows_per_token
        )
        for i in range(4):
            bp = jnp.where((i == 0) & boundary, p, 0)
            bpos[i] = bp
            first[i] = (i == 0) & boundary
            compressed = p // ccfg.dims.compress_ratio
            page = kt[compressed // ccfg.kv_block_size]
            kv[i] = jnp.where(
                (i == 0) & boundary,
                page * ccfg.kv_block_size * ccfg.kv_stride
                + compressed % ccfg.kv_block_size * ccfg.kv_stride,
                -1,
            )
            for j in range(pages_per_token):
                index = jnp.maximum(
                    bp // ccfg.state_block_size + j + 1 - pages_per_token, 0
                )
                pages[i * pages_per_token + j] = st[index]
        grid[...] = boundary.astype(jnp.int32)

        @pl.with_scoped(
            acc=pltpu.VMEM((2, n, pcfg.tile_sizes.tile_m // 128, 1, 128), jnp.float32),
            sem=pltpu.SemaphoreType.REGULAR((2,)),
            counts=pltpu.SMEM((2,), jnp.int32),
        )
        def project(acc, sem, counts):
            proj.kernel(
                slots,
                pos,
                hidden,
                weights,
                ape,
                state,
                state_out,
                acc,
                sem,
                counts,
                cfgs=pcfg,
            )

        project()

        # Projection drains its state DMAs before compression reads the same ref.
        @pl.with_scoped(window=pltpu.VMEM(ccfg.window_shape(), jnp.float32))
        def compress(window):
            comp.kernel_fn(
                pages,
                bpos,
                kv,
                first,
                first,
                grid,
                norm,
                cs,
                cache,
                None,
                state,
                cache_out,
                None,
                window,
                cfgs=ccfg,
            )

        compress()

    run()


def compressor_forward(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    norm_weight: jax.Array,
    cos_sin_cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,
    query_start_loc: jax.Array,
    kv_block_table: jax.Array,
    cache: jax.Array,
    rope_cache: jax.Array | None,
    distribution: jax.Array,
    *,
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    state_cache: jax.Array | None = None,
) -> tuple[jax.Array, None, jax.Array]:
    """Fuse a validated one-token HCA decode while publishing both caches."""
    assert state_cache is not None
    reqs = query_start_loc.size - 1
    ccfg = config.Configs.make(
        config.Mode.HCA,
        size_n=4,
        physical_page_size=cache.shape[1],
        state_physical_page_size=state_cache.shape[1],
        state_block_size=state_block_size,
        rms_eps=rms_eps,
        tile_n=4,
        head_dim=head_dim,
        rope_head_dim=cos_sin_cache.shape[1],
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )
    # Keep the baseline's joint-output tile and K accumulation order.
    m = wkv_wgate.shape[1]
    tm = proj.PROJECTION_OUTPUT_TILES.get(
        (1, hidden_states.shape[1], m, compress_ratio), m
    )
    pcfg = proj.Configs(
        proj.TileSizes(tm, proj._select_tile_k(hidden_states.shape[1]), 8),
        proj.Dimensions(m, hidden_states.shape[1], 8, compress_ratio),
        state_cache.shape[1],
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    state, kv = pl.pallas_call(
        functools.partial(
            _entry,
            pcfg=pcfg,
            ccfg=ccfg,
            state_stride=block_table.size // reqs,
            kv_stride=kv_block_table.size // reqs,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(state_cache.shape, state_cache.dtype),
            jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=(hbm, hbm, hbm, hbm, vmem, vmem, hbm, hbm, hbm),
            out_specs=(hbm, hbm),
        ),
        input_output_aliases={8: 0, 9: 1},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="hca_decode_project_metadata_compress",
    )(
        positions,
        block_table,
        kv_block_table,
        hidden_states,
        wkv_wgate,
        ape,
        norm_weight.reshape(ccfg.head_tiles, 128),
        cos_sin_cache,
        state_cache,
        cache,
    )
    return kv, None, state
