# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools

import jax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.utils import TPU_HEAD_SIZE_ALIGNMENT, get_dtype_packing


def _ceil_div(a, b):
    assert b != 0
    return (a + b - 1) // b


def _kv_cache_update_kernel(
    # Prefetch
    slices_ref,  # [3, padded_num_slices], list of (kv_cache_start, new_kv_start,
    # slice_len)
    num_slices_ref,  # [1]
    # Input
    new_kv_hbm_ref,  # [num_tokens, num_combined_kv_heads, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages * page_size, num_combined_kv_heads,
    # head_dim]
    # Output
    _,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    # Scratch
    scratch,  # [num_slices_per_block, page_size, num_combined_kv_heads,
    # head_dim]
    sem,
    *,
    pipelined_dma: bool = False,
):
    block_idx = pl.program_id(0)
    num_slices_per_block = scratch.shape[0]

    if not pipelined_dma:
        async_copies = []
        # Copy from new_kv_hbm_ref to scratch
        for i in range(num_slices_per_block):
            offset_i = i + block_idx * num_slices_per_block
            new_kv_start = jax.lax.select(
                offset_i < num_slices_ref[0], slices_ref[1, offset_i], 0
            )
            length = jax.lax.select(
                offset_i < num_slices_ref[0], slices_ref[2, offset_i], 0
            )
            async_copy = pltpu.make_async_copy(
                new_kv_hbm_ref.at[pl.ds(new_kv_start, length), ...],
                scratch.at[i, pl.ds(0, length), ...],
                sem,
            )
            async_copy.start()
            async_copies.append(async_copy)

        for async_copy in async_copies:
            async_copy.wait()

        # Copy from scratch to kv_cache_hbm_ref
        async_copies.clear()
        for i in range(num_slices_per_block):
            offset_i = i + block_idx * num_slices_per_block
            kv_cache_start = jax.lax.select(
                offset_i < num_slices_ref[0], slices_ref[0, offset_i], 0
            )
            length = jax.lax.select(
                offset_i < num_slices_ref[0], slices_ref[2, offset_i], 0
            )
            async_copy = pltpu.make_async_copy(
                scratch.at[i, pl.ds(0, length), ...],
                kv_cache_hbm_ref.at[pl.ds(kv_cache_start, length), ...],
                sem,
            )
            async_copy.start()
            async_copies.append(async_copy)
        for async_copy in async_copies:
            async_copy.wait()
    else:
        # Pipelined double-buffering: overlap reading slice i+1 with writing slice i to HBM
        offset_0 = block_idx * num_slices_per_block
        new_kv_start_0 = jax.lax.select(
            offset_0 < num_slices_ref[0], slices_ref[1, offset_0], 0
        )
        len_0 = jax.lax.select(offset_0 < num_slices_ref[0], slices_ref[2, offset_0], 0)
        read_copy = pltpu.make_async_copy(
            new_kv_hbm_ref.at[pl.ds(new_kv_start_0, len_0), ...],
            scratch.at[0, pl.ds(0, len_0), ...],
            sem,
        )
        read_copy.start()

        write_copy = None
        for i in range(num_slices_per_block):
            read_copy.wait()
            curr_buf_idx = i % 2
            next_buf_idx = (i + 1) % 2

            if write_copy is not None:
                write_copy.wait()

            offset_i = i + block_idx * num_slices_per_block
            kv_cache_start_i = jax.lax.select(
                offset_i < num_slices_ref[0], slices_ref[0, offset_i], 0
            )
            len_i = jax.lax.select(
                offset_i < num_slices_ref[0], slices_ref[2, offset_i], 0
            )
            write_copy = pltpu.make_async_copy(
                scratch.at[curr_buf_idx, pl.ds(0, len_i), ...],
                kv_cache_hbm_ref.at[pl.ds(kv_cache_start_i, len_i), ...],
                sem,
            )
            write_copy.start()

            if i + 1 < num_slices_per_block:
                offset_next = (i + 1) + block_idx * num_slices_per_block
                new_kv_start_next = jax.lax.select(
                    offset_next < num_slices_ref[0], slices_ref[1, offset_next], 0
                )
                len_next = jax.lax.select(
                    offset_next < num_slices_ref[0], slices_ref[2, offset_next], 0
                )
                read_copy = pltpu.make_async_copy(
                    new_kv_hbm_ref.at[pl.ds(new_kv_start_next, len_next), ...],
                    scratch.at[next_buf_idx, pl.ds(0, len_next), ...],
                    sem,
                )
                read_copy.start()

        if write_copy is not None:
            write_copy.wait()


def _dynamic_validate_inputs(
    slices, new_token_num, kv_cache_token_num, page_size, num_slices
):
    slices = slices.tolist()
    # NOTE: The padding part is unnecessary to check because kv_cache_start, new_kv_start,
    # slice_len will be set to 0 in the kernel implementation.
    for i in range(num_slices[0]):
        kv_cache_start = slices[0][i]
        new_kv_start = slices[1][i]
        slice_len = slices[2][i]
        if new_kv_start < 0:
            raise ValueError(f"{new_kv_start=} must be greater than or equal to 0")
        if kv_cache_start < 0:
            raise ValueError(f"{kv_cache_start=} must be greater than or equal to 0")
        if not 0 < slice_len <= page_size:
            raise ValueError(
                f"{slice_len=} must be less or equal to {page_size=} and greater than 0"
            )
        if new_kv_start + slice_len > new_token_num:
            raise ValueError(
                f"{new_kv_start=} + {slice_len=} must be less or equal to {new_token_num=}"
            )
        if kv_cache_start + slice_len > kv_cache_token_num:
            raise ValueError(
                f"{kv_cache_start=} + {slice_len=} must be less or equal to {kv_cache_token_num=}"
            )
        if kv_cache_start // page_size != (kv_cache_start + slice_len - 1) // page_size:
            raise ValueError(
                f"Each slice must reside in the same page, but got {kv_cache_start=} and {slice_len=}"
            )

    new_kv_intervals = []
    kv_cache_intervals = []
    for i in range(num_slices[0]):
        new_kv_intervals.append((slices[1][i], slices[1][i] + slices[2][i]))
        kv_cache_intervals.append((slices[0][i], slices[0][i] + slices[2][i]))

    new_kv_intervals.sort()
    kv_cache_intervals.sort()

    # The new_kv slices should be continuous
    for i in range(len(new_kv_intervals) - 1):
        if new_kv_intervals[i][1] != new_kv_intervals[i + 1][0]:
            raise ValueError(
                f"{new_kv_intervals[i][1]=} is expeced to equal to {new_kv_intervals[i + 1][0]}"
            )

    # There should be no overlap among the kv cache slices
    for i in range(len(kv_cache_intervals) - 1):
        if kv_cache_intervals[i][1] > kv_cache_intervals[i + 1][0]:
            raise ValueError(
                f"Overlap detected in kv_cache intervals: {kv_cache_intervals[i]} and {kv_cache_intervals[i + 1]}"
            )


def _kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_combined_kv_heads, head_dim]
    slices: jax.Array,  # [3, slices], list of (kv_cache_start, new_kv_start,
    # slice_len)
    kv_cache: jax.Array,  # [total_num_pages * page_size, num_combined_kv_heads,
    # head_dim]
    num_slices: jax.Array,  # [1]
    page_size: int,
    num_slices_per_block: int,
    dynamic_validate_inputs: bool,
    vmem_limit_bytes: int = 40 * 1024 * 1024,
    pipelined_dma: bool = False,
):
    new_token_num, num_combined_kv_heads, head_dim = new_kv.shape
    assert kv_cache.shape[1] == num_combined_kv_heads
    assert kv_cache.shape[2] == head_dim
    assert head_dim % 128 == 0
    if dynamic_validate_inputs is True:
        _dynamic_validate_inputs(
            slices, new_token_num, kv_cache.shape[0], page_size, num_slices
        )

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
    ]

    out_specs = [pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY)]
    out_shape = [jax.ShapeDtypeStruct(kv_cache.shape, dtype=kv_cache.dtype)]

    scalar_prefetches = [slices, num_slices]
    scratch = pltpu.VMEM(
        (num_slices_per_block, page_size, num_combined_kv_heads, head_dim),
        new_kv.dtype,
    )

    scratch_shapes = [
        scratch,
        pltpu.SemaphoreType.DMA,
    ]

    kernel_fn = (
        functools.partial(_kv_cache_update_kernel, pipelined_dma=True)
        if pipelined_dma
        else _kv_cache_update_kernel
    )

    kernel = pl.pallas_call(
        kernel_fn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(_ceil_div(num_slices[0], num_slices_per_block),),
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        input_output_aliases={len(scalar_prefetches) + 1: 0},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
        ),
    )

    return kernel(*scalar_prefetches, new_kv, kv_cache)[0]


def _prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def _get_page_size_bytes(
    block_size: int, num_combined_kv_heads: int, head_size: int, kv_cache_dtype
) -> int:
    """Returns the size in bytes of one page of the KV cache."""
    kv_cache_dtype_bit_size = dtypes.itemsize_bits(kv_cache_dtype)
    padded_head_size = (
        _ceil_div(head_size, TPU_HEAD_SIZE_ALIGNMENT) * TPU_HEAD_SIZE_ALIGNMENT
    )

    # NOTE: for the implicit padding in XLA
    packing = get_dtype_packing(kv_cache_dtype)
    num_combined_kv_heads = _ceil_div(num_combined_kv_heads, packing) * packing

    return (
        block_size
        * num_combined_kv_heads
        * padded_head_size
        * kv_cache_dtype_bit_size
        // 8
    )


def _get_num_slices_per_kv_cache_update_block(
    page_size_bytes: int, vmem_limit_bytes: int
) -> int:
    """Find the optimum number of slices to copy per Pallas program instance.
    Increasing the number of slices copied in one instance of the kernel program
    will increase HBM bandwidth utilization via more in-flight DMAs.
    However, it will also use more VMEM, and experimentally, we observed
    performance regression at 128 slices on v6e, likely due to running
    out of scalar registers. Thus this function will limit the number of
    slices to 64.
    """
    # NOTE: We assume 1MB vmem is used for register spill and others
    assert vmem_limit_bytes >= 1024 * 1024, "vmem_limit_bytes must be at least 1MB"
    num_slices_per_block = (vmem_limit_bytes - 1024 * 1024) // page_size_bytes
    assert num_slices_per_block > 0, "Number of slices should be positive"
    num_slices_per_block = _prev_power_of_2(num_slices_per_block)
    return min(num_slices_per_block, 64)


@functools.partial(
    jax.jit,
    static_argnames=[
        "page_size",
        "num_slices_per_block",
        "mesh",
        "kv_cache_pspec",
        "pipelined_dma",
    ],
    donate_argnames="kv_cache",
)
def kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_combined_kv_heads, head_dim]
    slices: jax.Array,  # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    kv_cache: jax.Array,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    num_slices: jax.Array,  # [1]
    *,
    page_size: int = 32,
    num_slices_per_block: int | None = None,
    mesh: Mesh | None = None,
    kv_cache_pspec: P | None = None,  # Only sharding along head_dim is supported
    dynamic_validate_inputs: bool = False,
    vmem_limit_bytes: int = 40 * 1024 * 1024,
    pipelined_dma: bool | None = None,
):
    if pipelined_dma is None:
        pipelined_dma = envs.USE_RPA_PIPELINED_DMA_STAGING

    if num_slices_per_block is None:
        _, num_combined_kv_heads, head_dim = new_kv.shape
        page_size_bytes = _get_page_size_bytes(
            page_size, num_combined_kv_heads, head_dim, kv_cache.dtype
        )
        num_slices_per_block = _get_num_slices_per_kv_cache_update_block(
            page_size_bytes, vmem_limit_bytes
        )

    if mesh is None:
        return _kv_cache_update(
            new_kv,
            slices,
            kv_cache,
            num_slices,
            page_size,
            num_slices_per_block,
            dynamic_validate_inputs,
            vmem_limit_bytes=vmem_limit_bytes,
            pipelined_dma=pipelined_dma,
        )

    if kv_cache_pspec is None:
        raise ValueError("kv_cache_pspec must be provided when mesh is specified")

    in_specs = (kv_cache_pspec, P(), kv_cache_pspec, P())
    out_specs = kv_cache_pspec
    shard_map_wrapped = jax.shard_map(
        functools.partial(
            _kv_cache_update,
            page_size=page_size,
            num_slices_per_block=num_slices_per_block,
            dynamic_validate_inputs=dynamic_validate_inputs,
            vmem_limit_bytes=vmem_limit_bytes,
            pipelined_dma=pipelined_dma,
        ),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    return shard_map_wrapped(new_kv, slices, kv_cache, num_slices)
