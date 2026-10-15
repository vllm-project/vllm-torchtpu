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
"""Single-token SWA/HCA composition with the original BF16 rounding boundary."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout, qkv_preprocess
from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla_swa
from vllm_torchtpu.kernels.deepseek_v4.core_attention.tuned_block_sizes import (
    FUSED_HCA_DECODE_KV_PAGES,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import get_device_name

logger = init_logger(__name__)


def get_block_pages(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    main_cache: jax.Array,
    *,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    two_caches_same_buffer: bool,
    non_causal_block: bool,
) -> int | None:
    """Return a measured tile only within the fused kernel's static envelope."""
    if (
        swa_only
        or two_caches_same_buffer
        or non_causal_block
        or q.shape != (1, 64, 512)
        or q.dtype != jnp.bfloat16
        or new_kv.shape != (1, 512)
        or new_kv.dtype != jnp.bfloat16
        or not sw_cache.shape[0]
        or not main_cache.shape[0]
        or main_cache.dtype != jnp.uint8
        or main_cache.shape[1:] != (16, 4, 128)
        or sw_cache.dtype not in (jnp.uint8, jnp.int32)
        or logical_page_size != 128
        or sliding_window != 128
    ):
        return None
    slab_shape = csa_cache_layout.u8_slab_shape(sw_cache.shape, sw_cache.dtype)
    if slab_shape[1] < 2 * logical_page_size or slab_shape[2:] != (4, 128):
        return None
    key = (
        get_device_name(),
        q.shape[1],
        q.shape[2],
        logical_page_size,
        main_cache.shape[1] // 2,
    )
    pages = FUSED_HCA_DECODE_KV_PAGES.get(key)
    if pages is None:
        logger.warning_once("HCA fused decode tuning unavailable for key=%s", key)
    return pages


def _kernel(
    swa_lengths,
    swa_pages,
    cu,
    bounds,
    main_lengths,
    main_pages,
    attend,
    sinks,
    q,
    new_kv,
    sw_cache,
    main_cache,
    output,
    updated_cache,
    swa_kv,
    query_buffer,
    output_buffer,
    swa_sems,
    main_kv,
    main_sems,
    *,
    sm_scale,
    logical_page_size,
    sliding_window,
    block_pages,
    resident_inputs=False,
    swa_page_count=None,
    main_page_count=None,
):
    page_size = main_cache.shape[1] // 2
    slots_per_token = 2 * main_cache.shape[2]
    lanes = main_cache.shape[3]
    block_tokens = block_pages * page_size
    head_dim = q.shape[-1]

    def transfer(tile, slot, wait):
        length = jnp.clip(main_lengths[0] - tile * block_tokens, 0, block_tokens)
        destination = main_kv.at[slot]
        if wait:
            view = destination.at[pl.ds(0, length)]
            pltpu.make_async_copy(view, view, main_sems.at[slot]).wait()
        else:
            cache_rows = main_cache.reshape(
                main_cache.shape[0] * page_size, slots_per_token, lanes
            )
            for page in range(block_pages):
                size = jnp.clip(length - page * page_size, 0, page_size)
                index = jnp.minimum(
                    tile * block_pages + page,
                    (
                        main_pages.shape[0]
                        if main_page_count is None
                        else main_page_count
                    )
                    - 1,
                )
                pltpu.make_async_copy(
                    cache_rows.at[pl.ds(main_pages[index] * page_size, size)],
                    destination.at[pl.ds(page * page_size, size)],
                    main_sems.at[slot],
                ).start()

    def fetch(_batch, _slot, wait):
        # Overlap the first compressed-KV tile with the existing SWA fetch.
        transfer(jnp.int32(0), jnp.int32(0), wait)

    def finalize(_batch, _slot, queries, swa_acc, swa_l, swa_m, destination):
        blocks = jnp.maximum(1, (main_lengths[0] + block_tokens - 1) // block_tokens)

        def body(index, carry):
            acc, normalizer, row_max = carry
            slot = index % 2

            @pl.when(index + 1 < blocks)
            def prefetch():
                transfer(index + 1, (index + 1) % 2, False)

            # The SWA fetch hook has already waited for tile zero.
            @pl.when(index > 0)
            def wait():
                transfer(index, slot, True)

            kv = pltpu.bitcast(main_kv.at[slot][...], jnp.bfloat16).reshape(
                1, block_tokens, head_dim
            )
            positions = jnp.arange(block_tokens) + index * block_tokens
            # DMA leaves the tail unwritten; avoid propagating stale NaNs via PV.
            kv = jnp.where(positions[None, :, None] < main_lengths[0], kv, 0)
            scores = (
                jnp.einsum(
                    "bnd,bmd->bnm", queries, kv, preferred_element_type=jnp.float32
                )
                * sm_scale
            )
            scores = jnp.where(
                positions[None, None, :] < attend[0], scores, jnp.finfo(jnp.float32).min
            )
            next_max = jnp.maximum(row_max, jnp.max(scores, axis=2))
            probabilities = jnp.exp(scores - next_max[:, :, None])
            correction = jnp.exp(row_max - next_max)
            pv = jnp.einsum(
                "bnm,bmd->bnd", probabilities, kv, preferred_element_type=jnp.float32
            )
            return (
                correction[:, :, None] * acc + pv,
                correction * normalizer + jnp.sum(probabilities, axis=2),
                next_max,
            )

        # The SWA callback passes BF16 acc, exactly as the unfused HBM boundary.
        acc, normalizer, row_max = lax.fori_loop(
            0, blocks, body, (swa_acc.astype(jnp.float32), swa_l, swa_m)
        )
        denominator = normalizer + jnp.exp(sinks[...][None, :] - row_max)
        destination[...] = (
            acc * pl.reciprocal(denominator[:, :, None], approx=True)
        ).astype(q.dtype)

    mla_swa._mla_batched_decode_kernel(
        swa_lengths,
        swa_pages,
        cu,
        bounds,
        sinks,
        q,
        new_kv,
        sw_cache,
        None,
        None,
        None,
        output,
        updated_cache,
        None,
        None,
        swa_kv,
        query_buffer,
        output_buffer,
        None,
        None,
        swa_sems,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        unnormalized_output=True,
        bkv_p=2,
        seq_batch_sz=1,
        fetch_extra=fetch,
        finalize=finalize,
        resident_inputs=resident_inputs,
        page_indices_count=swa_page_count,
    )


def _input_kernel(
    distribution,
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
    updated,
    swa_kv,
    query_buffer,
    output_buffer,
    swa_sems,
    main_kv,
    main_sems,
    *,
    sm_scale,
    logical_page_size,
    sliding_window,
    block_pages,
    swa_pages_per_seq,
    main_pages_per_seq,
    positions=None,
    cos_sin_cache=None,
    eps=None,
):
    refs = (sl, sp, cu, ml, mp, attend)
    sizes = (1, swa_pages_per_seq, 2, 1, main_pages_per_seq, 1)
    copy_sizes = tuple(
        min(pl.cdiv(size, 128) * 128, ref.shape[0]) for ref, size in zip(refs, sizes)
    )
    specs = {
        f"meta_{i}": pltpu.SMEM((size,), jnp.int32) for i, size in enumerate(copy_sizes)
    }
    specs.update({f"sem_{i}": pltpu.SemaphoreType.DMA for i in range(6)})
    specs.update(
        bounds=pltpu.SMEM((2,), jnp.int32), packed=pltpu.VMEM((1, 8, 128), jnp.uint8)
    )

    if positions is not None:
        specs.update(
            q_norm=pltpu.VMEM(q.shape, q.dtype),
            kv_rope=pltpu.VMEM(kv.shape, kv.dtype),
            raw_cs=pltpu.VMEM((1, 64), jnp.float32),
            cs_sem=pltpu.SemaphoreType.DMA,
        )

    @pl.with_scoped(**specs)
    def stage(**scratch):
        copies = [
            pltpu.make_async_copy(
                ref.at[pl.ds(0, size)], scratch[f"meta_{i}"], scratch[f"sem_{i}"]
            )
            for i, (ref, size) in enumerate(zip(refs, copy_sizes))
        ]
        for copy in copies:
            copy.start()
        query, value = q, kv
        if positions is not None:
            qkv_preprocess._kernel(
                positions,
                q,
                kv,
                cos_sin_cache,
                scratch["q_norm"],
                scratch["kv_rope"],
                scratch["raw_cs"],
                scratch["cs_sem"],
                eps=eps,
            )
            query, value = scratch["q_norm"], scratch["kv_rope"]
        scratch["packed"][...] = pltpu.bitcast(value[...].reshape(1, 4, 128), jnp.uint8)
        scratch["bounds"][0] = 0
        scratch["bounds"][1] = distribution[0]
        for copy in copies:
            copy.wait()
        meta = [
            (
                scratch[f"meta_{i}"]
                if i in (1, 4)
                else scratch[f"meta_{i}"].at[pl.ds(0, size)]
            )
            for i, size in enumerate(sizes)
        ]
        _kernel(
            meta[0],
            meta[1],
            meta[2],
            scratch["bounds"],
            meta[3],
            meta[4],
            meta[5],
            sinks,
            query,
            scratch["packed"],
            sw,
            main,
            output,
            updated,
            swa_kv,
            query_buffer,
            output_buffer,
            swa_sems,
            main_kv,
            main_sems,
            sm_scale=sm_scale,
            logical_page_size=logical_page_size,
            sliding_window=sliding_window,
            block_pages=block_pages,
            resident_inputs=True,
            swa_page_count=swa_pages_per_seq,
            main_page_count=main_pages_per_seq,
        )

    stage()


def _qkv_input_kernel(
    positions,
    distribution,
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
    cs,
    output,
    updated,
    swa_kv,
    query_buffer,
    output_buffer,
    swa_sems,
    main_kv,
    main_sems,
    *,
    eps,
    **options,
):
    _input_kernel(
        distribution,
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
        updated,
        swa_kv,
        query_buffer,
        output_buffer,
        swa_sems,
        main_kv,
        main_sems,
        positions=positions,
        cos_sin_cache=cs,
        eps=eps,
        **options,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "eps",
        "sm_scale",
        "logical_page_size",
        "sliding_window",
        "num_kv_pages_per_block",
    ),
)
def fused_single_decode(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_lengths: jax.Array,
    swa_pages: jax.Array,
    cu: jax.Array,
    distribution: jax.Array,
    main_cache: jax.Array,
    main_lengths: jax.Array,
    attend: jax.Array,
    main_pages: jax.Array,
    sinks: jax.Array,
    *,
    sm_scale: float,
    logical_page_size: int,
    sliding_window: int,
    num_kv_pages_per_block: int,
    positions: jax.Array | None = None,
    cos_sin_cache: jax.Array | None = None,
    eps: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Run a validated dense one-query decode; caller owns metadata eligibility."""
    tokens, heads, dim = q.shape
    page_size = main_cache.shape[1] // 2
    slots_per_token = 2 * main_cache.shape[2]
    lanes = main_cache.shape[3]
    swa_pages_per_seq = swa_pages.shape[0] // swa_lengths.shape[0]
    main_pages_per_seq = main_pages.shape[0] // main_lengths.shape[0]
    scratch = [
        pltpu.VMEM(
            (2, tokens, 2 * logical_page_size, slots_per_token, lanes), jnp.uint8
        ),
        pltpu.VMEM((2, tokens, heads, dim), q.dtype),
        pltpu.VMEM((2, tokens, heads, dim), q.dtype),
        pltpu.SemaphoreType.DMA((2, tokens + 4)),
        pltpu.VMEM(
            (2, num_kv_pages_per_block * page_size, slots_per_token, lanes), jnp.uint8
        ),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    preprocess = positions is not None
    inputs = (
        distribution,
        swa_lengths,
        swa_pages,
        cu,
        main_lengths,
        main_pages,
        attend,
        sinks,
        q,
        new_kv,
        sw_cache,
        main_cache,
    )
    if preprocess:
        assert q.shape == (1, 64, 512) and new_kv.shape == (1, 512)
        assert positions.shape == (1,) and positions.dtype == jnp.int32
        assert cos_sin_cache is not None and eps is not None
        inputs = (positions, *inputs, cos_sin_cache)
    return pl.pallas_call(
        functools.partial(
            _qkv_input_kernel if preprocess else _input_kernel,
            **(dict(eps=eps) if preprocess else {}),
            sm_scale=sm_scale,
            logical_page_size=logical_page_size,
            sliding_window=sliding_window,
            block_pages=num_kv_pages_per_block,
            swa_pages_per_seq=swa_pages_per_seq,
            main_pages_per_seq=main_pages_per_seq,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2 if preprocess else 1,
            in_specs=[hbm] * 6
            + [pl.BlockSpec(memory_space=pltpu.VMEM)] * 3
            + [hbm] * (3 if preprocess else 2),
            out_specs=[hbm, hbm],
            grid=(jnp.where(distribution[0] > 0, 1, 0),),
            scratch_shapes=scratch,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct(sw_cache.shape, sw_cache.dtype),
        ],
        input_output_aliases={9: 0, 11: 1} if preprocess else {8: 0, 10: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=mla_swa.DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        name="QKV_SWA_HCA_INPUTS_DECODE" if preprocess else "SWA_HCA_INPUTS_DECODE",
    )(*inputs)
