# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""In-place Pallas partial slot update for KV cache pools (b/546309078).

``index_copy_`` lowers to ``stablehlo.scatter``. XLA is functional, so scatter
means "produce a whole new tensor"; on an eager single-op graph there is no
``input_output_alias``, so buffer assignment hands out two pool-sized
allocations and the whole cache crosses HBM to update a few slots. The cost
tracks pool capacity instead of update size.

The invariant the fix rests on:

    Every shape or dtype adaptation happens on a *ref inside the kernel*.
    Nothing reshapes or bitcasts a pool-sized tensor at the XLA level.

An XLA-level reshape is not free -- ``bf16[N,1024]`` and ``bf16[N,8,128]`` are
different tiled layouts, so XLA materialises the conversion, and a pool-sized
copy per call puts the cost straight back. The same trap applies to dtype: see
``pool_adapters`` -- "an XLA-level bitcast of a pool slice forces a pool-sized
layout copy per step".

Rank decides how the kernel is written, never whether the fix is needed:

* ``ndim >= 3`` -- the slot index sits outside the last two dims, so a block
  spec covers one slot natively and Pallas emits the DMA. Mirrors
  ``pool_adapters.scatter_region``.
* ``ndim == 2`` -- the slot index lands *on* the second-minor dim, where a
  one-slot block pins it to 1 and fails the tiling rule. Use a sublane-tall
  block and read-modify-write it, applying every update that targets the
  block so the step is idempotent. Costs one sublane group of bytes per
  update instead of one slot -- still O(update), still capacity-independent.

Layout caveat, which correctness testing cannot see. Pallas needs the pool in
the standard descending layout; when the trailing dims are awkward XLA picks a
permuted one for the array and inserts a pool-sized conversion, leaving the
result correct but no longer O(update). The Kimi MLA ``(N, 16, 1, 640)`` pool
is one known case and is dispatched to the pinned-upstream Torch fill by the
connector. For an unfamiliar shape, check the buffer assignment: the
parameter's layout and the kernel's should match, and a ``copy`` at pool size
means they did not.
"""

import math

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas as torch_pallas

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import typed_ldst


def _blockspec_kernel(_sidx_ref, vals_ref, _pool_in_ref, pool_out_ref):
    """One slot per grid step.

    The output block covers exactly the target slot and is fully written, so
    the untouched complement is preserved through the HBM alias with no
    read-modify-write. ``pool_in_ref`` is never read -- it exists only to give
    ``input_output_aliases`` something to alias.
    """
    pool_out_ref[...] = vals_ref[...]


def _scatter_slots_nd(pool, indices, vals):
    """Rank >= 3: native block spec, no manual DMA, no scratch."""
    na = indices.shape[0]
    pad = (0, ) * (pool.ndim - 1)
    block = (1, ) + pool.shape[1:]
    assert vals.shape == (na, ) + pool.shape[1:], (vals.shape, pool.shape)

    return pl.pallas_call(
        _blockspec_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec(block, lambda i, s: (i, ) + pad),
                # Aliased with the output; never read, so leave it in HBM.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(block, lambda i, s: (s[i], ) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(indices, vals, pool)


def _make_sublane_block_kernel(width: int, sublanes: int, num_updates: int):
    """Rank 2 on TensorCore: a sublane-tall block, read-modify-write.

    The block-shape rule wants the second-minor dim divisible by 8 *or* equal
    to the array dim. A one-slot block pins it to 1 and fails. A
    ``(8, width)`` block satisfies it directly -- so instead of fighting the
    rule, take the smallest block that obeys it and pick our slot out of it.

    Cost is one sublane group of bytes per update rather than one slot, plus
    an unrolled pass over the update list per step. Both are O(update) and
    independent of pool capacity, which is what the bug is about.
    """

    def _kernel(sidx_ref, vals_ref, pool_in_ref, pool_out_ref):
        step = pl.program_id(0)
        my_block = sidx_ref[step] // sublanes
        rows = jax.lax.broadcasted_iota(jnp.int32, (sublanes, width), 0)

        # Apply *every* update that lands in this block, not just this step's.
        # That makes the step idempotent, which is what correctness rests on:
        # several steps can target the same block, and Pallas does not refetch
        # an input block whose index has not changed, so a step that applied
        # only its own row would read the original pool and revert its
        # predecessors.
        out = pool_in_ref[...]
        for j in range(num_updates):
            slot_j = sidx_ref[j]
            in_block = (slot_j // sublanes) == my_block
            dst = slot_j - (slot_j // sublanes) * sublanes
            # Static j, so this is an ordinary unrolled index, not a dynamic
            # one; vals is resident whole so any row is reachable.
            row_j = vals_ref[j][None, :]
            out = jnp.where(jnp.logical_and(rows == dst, in_block), row_j, out)
        pool_out_ref[...] = out

    return _kernel


def _scatter_slots_2d(pool, indices, vals):
    """Rank 2 via an 8-row block. See ``_make_sublane_block_kernel``."""
    na = indices.shape[0]
    num_slots, width = pool.shape
    # Ask the hardware rather than assuming 8x128, the way the rest of the
    # repo does (batched_rpa/wrapper.py, ragged_gather.py). The tile is a
    # property of the generation, not a constant.
    sublanes = pltpu.get_tpu_info().num_sublanes
    if num_slots % sublanes:
        raise ValueError(
            f"2D num_slots {num_slots} must be a multiple of {sublanes}: the "
            "block covers whole groups of sublanes, and a partial trailing "
            "block would write past the end of the aliased output")
    return pl.pallas_call(
        _make_sublane_block_kernel(width, sublanes, na),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                # The whole update array, resident: the kernel needs any row
                # of it, not just this step's. Block dims equal the array
                # dims, so the tiling escape clause covers it and no padding
                # is needed. vals is kilobytes.
                pl.BlockSpec((na, width), lambda i, s: (0, 0)),
                # Read, unlike the rank >= 3 path: the block holds the slot's
                # neighbours and they have to survive the write-back.
                pl.BlockSpec((sublanes, width), lambda i, s:
                             (s[i] // sublanes, 0)),
            ],
            out_specs=pl.BlockSpec((sublanes, width), lambda i, s:
                                   (s[i] // sublanes, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(indices, vals, pool)


def pallas_scatter_slots(pool: jax.Array, indices: jax.Array,
                         vals: jax.Array) -> jax.Array:
    """``pool[indices[i]] = vals[i]``, in place, cost O(len(indices)).

    pool:    (num_slots, ...) -- aliased with the output, never reshaped here.
    indices: (num_updates,)   -- target slot indices.
    vals:    (num_updates, ...) matching ``pool.shape[1:]``.
    """
    if pool.dtype != vals.dtype:
        raise ValueError(
            f"pool dtype {pool.dtype} != vals dtype {vals.dtype}; a cross-dtype "
            "view must be taken on the ref inside the kernel, never in XLA")
    indices = indices.astype(jnp.int32)
    if pool.ndim >= 3:
        return _scatter_slots_nd(pool, indices, vals)
    if pool.ndim == 2:
        return _scatter_slots_2d(pool, indices, vals)
    raise ValueError(f"unsupported pool ndim: {pool.ndim}")


def _fill_slots_kernel(sidx_ref, fill_ref, _pool_in_ref, pool_out_ref):
    """Write one constant-valued slot per grid step.

    The pool input exists solely to donate its HBM buffer to the output.  In
    particular, do not build a slot-shaped value tensor outside this kernel:
    doing so is both unnecessary and prohibitively large for long-context KV
    cache fills.
    """
    del sidx_ref, _pool_in_ref
    # Ref stores require an exact-shape value (they do not scalar-broadcast).
    # This broadcast is local to the selected one-slot VMEM tile, never the
    # full HBM pool.
    values = jnp.broadcast_to(fill_ref[0], pool_out_ref.shape)
    pool_out_ref[...] = values.astype(pool_out_ref.dtype)


def pallas_fill_slots(pool: jax.Array, indices: jax.Array,
                      fill_value: jax.Array) -> jax.Array:
    """Set ``pool[indices]`` to one scalar value in place.

    This is the constant-fill counterpart of :func:`pallas_scatter_slots`.
    Its work and temporary storage scale with the number of selected slots,
    not with pool capacity.  Rank-2 pools are deliberately excluded: all TPU
    attention and recurrent-state cache layouts supported by the decode bench
    connector have a leading slot dimension plus at least two tiled trailing
    dimensions.
    """
    if pool.ndim < 3:
        raise ValueError(
            f"constant slot fill requires pool ndim >= 3, got {pool.ndim}")
    if indices.ndim != 1 or indices.shape[0] == 0:
        raise ValueError("indices must be a non-empty rank-1 array")
    if fill_value.shape != (1, ):
        raise ValueError(
            f"fill_value must have shape (1,), got {fill_value.shape}")
    if not jnp.issubdtype(fill_value.dtype, jnp.floating):
        raise ValueError(
            f"fill_value must have floating dtype, got {fill_value.dtype}")

    indices = indices.astype(jnp.int32)
    num_updates = indices.shape[0]
    padding = (0, ) * (pool.ndim - 1)
    block_shape = (1, ) + pool.shape[1:]
    return pl.pallas_call(
        _fill_slots_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(num_updates, ),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                block_shape,
                lambda i, slots, _fill: (slots[i], ) + padding,
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(indices, fill_value, pool)


def _normal_fill_slots_kernel(sidx_ref, mean_std_ref, seed_ref, _pool_in_ref,
                              pool_out_ref):
    """Write one normally-distributed slot per grid step.

    Random values are generated in the selected one-slot VMEM tile.  As with
    the constant kernel, no slot-shaped update tensor exists in HBM.
    """
    del _pool_in_ref
    step = pl.program_id(0)
    # Include the physical slot in the seed so output is stable under index
    # reordering and duplicate indices remain idempotent.
    pltpu.prng_seed(seed_ref[0], sidx_ref[step])
    mean = mean_std_ref[0].astype(jnp.float32)
    std = mean_std_ref[1].astype(jnp.float32)
    # Mosaic TPU only exposes 32-bit PRNG words. Asking JAX for a bf16 normal
    # distribution selects its 8-bit path and fails lowering, so sample in
    # fp32 and cast only the one-slot VMEM result at the store boundary.
    samples = pltpu.stateful_normal(pool_out_ref.shape, dtype=jnp.float32)
    pool_out_ref[...] = (samples * std + mean).astype(pool_out_ref.dtype)


def pallas_normal_fill_slots(pool: jax.Array, indices: jax.Array,
                             mean_std: jax.Array,
                             seed: jax.Array) -> jax.Array:
    """Set ``pool[indices]`` to samples from ``Normal(mean, std)`` in place.

    ``mean_std`` contains ``[mean, std]`` as float32 values and ``seed`` is a
    one-element integer array. Work and temporary storage scale with selected
    slots, while the pool remains donated to the output.
    """
    if pool.ndim < 3:
        raise ValueError(
            f"normal slot fill requires pool ndim >= 3, got {pool.ndim}")
    if indices.ndim != 1 or indices.shape[0] == 0:
        raise ValueError("indices must be a non-empty rank-1 array")
    if mean_std.shape != (2, ):
        raise ValueError(
            f"mean_std must have shape (2,), got {mean_std.shape}")
    if not jnp.issubdtype(mean_std.dtype, jnp.floating):
        raise ValueError(
            f"mean_std must have floating dtype, got {mean_std.dtype}")
    if seed.shape != (1, ):
        raise ValueError(f"seed must have shape (1,), got {seed.shape}")
    if not jnp.issubdtype(seed.dtype, jnp.integer):
        raise ValueError(f"seed must have integer dtype, got {seed.dtype}")

    indices = indices.astype(jnp.int32)
    seed = seed.astype(jnp.int32)
    num_updates = indices.shape[0]
    padding = (0, ) * (pool.ndim - 1)
    block_shape = (1, ) + pool.shape[1:]
    return pl.pallas_call(
        _normal_fill_slots_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(num_updates, ),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(
                block_shape,
                lambda i, slots, _mean_std, _seed: (slots[i], ) + padding,
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3: 0},
    )(indices, mean_std, seed, pool)


# A unified hybrid page can be selected by its attention group, its GDN state
# group, or both. The worker ORs these bits for each manager block so one Pallas
# program owns every write to that donated page.
_HYBRID_FILL_ATTENTION = 1
_HYBRID_FILL_STATE = 2


def _typed_values(shape: tuple[int, ...], dtype, params_ref, *, normal: bool):
    mean = params_ref[0].astype(jnp.float32)
    if normal:
        std = params_ref[1].astype(jnp.float32)
        values = pltpu.stateful_normal(shape, dtype=jnp.float32) * std + mean
    else:
        values = jnp.full(shape, mean, dtype=jnp.float32)
    return values.astype(dtype)


def _store_typed_live_rows(block_ref, num_rows: int, dtype, params_ref, *,
                           normal: bool) -> None:
    typed_ref = typed_ldst.view_typed(block_ref, dtype)
    if num_rows <= 0 or num_rows > typed_ref.shape[0]:
        raise ValueError(
            f"invalid typed live-row count {num_rows} for {typed_ref.shape}")
    values = _typed_values(typed_ref.shape, dtype, params_ref, normal=normal)
    rows = jax.lax.broadcasted_iota(jnp.int32, typed_ref.shape, 0)
    typed_ref[...] = jnp.where(rows < num_rows, values, typed_ref[...])


def _hybrid_state_segments(pool: jax.Array, *, split: int, ssm_bytes: int,
                           conv_bytes: int):
    """Static per-kernel-block live segments for the pooled GDN ABI."""
    if ssm_bytes == 0 and conv_bytes == 0:
        return (), ()
    if min(ssm_bytes, conv_bytes) <= 0:
        raise ValueError("ssm_bytes and conv_bytes must both be positive")

    token_bytes = math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize
    manager_tokens = split * pool.shape[1]
    layout = derive_pooled_gdn_state_layout(ssm_bytes=ssm_bytes,
                                            conv_bytes=conv_bytes,
                                            token_bytes=token_bytes)
    if layout.required_tokens > manager_tokens:
        raise ValueError(
            "pooled GDN state exceeds a unified manager page: "
            f"required={layout.required_tokens}, available={manager_tokens}")

    kernel_block_bytes = pool.shape[1] * token_bytes
    lanes = pool.shape[-1]

    def segments(start_bytes: int, live_bytes: int, itemsize: int):
        result = []
        end_bytes = start_bytes + live_bytes
        for block in range(split):
            block_start = block * kernel_block_bytes
            block_end = block_start + kernel_block_bytes
            live_start = max(start_bytes, block_start)
            live_end = min(end_bytes, block_end)
            if live_start >= live_end:
                continue
            local_start = live_start - block_start
            segment_bytes = live_end - live_start
            if local_start % token_bytes:
                raise ValueError(
                    "pooled state segment must start on a token row")
            row_bytes = itemsize * lanes
            if segment_bytes % row_bytes:
                raise ValueError(
                    "pooled state live bytes must contain complete typed rows: "
                    f"bytes={segment_bytes}, row_bytes={row_bytes}")
            token_row = local_start // token_bytes
            token_rows = (segment_bytes + token_bytes - 1) // token_bytes
            result.append(
                (block, token_row, token_rows, segment_bytes // row_bytes))
        return tuple(result)

    ssm_segments = segments(0, ssm_bytes, jnp.dtype(jnp.float32).itemsize)
    conv_start = layout.ssm_tokens * token_bytes
    conv_segments = segments(conv_start, conv_bytes,
                             jnp.dtype(jnp.bfloat16).itemsize)
    return ssm_segments, conv_segments


def _make_hybrid_fill_kernel(*, normal: bool, ssm_segments, conv_segments):

    def _kernel(indices_ref, kinds_ref, params_ref, *refs):
        if normal:
            seed_ref, pool_in_ref, pool_out_ref = refs
        else:
            pool_in_ref, pool_out_ref = refs
            seed_ref = None
        step = pl.program_id(0)
        block_id = indices_ref[step]
        fill_kind = kinds_ref[step]
        fill_attention = (fill_kind & _HYBRID_FILL_ATTENTION) != 0
        fill_state = (fill_kind & _HYBRID_FILL_STATE) != 0
        if normal:
            assert seed_ref is not None
            pltpu.prng_seed(seed_ref[0], block_id)

        # Satisfy the full-write rule first. State-only pages copy through;
        # attention pages receive a scalar/normal value in the pool dtype.
        for block in range(pool_out_ref.shape[0]):
            pool_out_ref.at[block][...] = pool_in_ref.at[block][...]

            @pl.when(fill_attention)
            def _fill_attention_block():
                shape = pool_out_ref.at[block].shape
                if normal:
                    values = _typed_values(shape,
                                           pool_out_ref.dtype,
                                           params_ref,
                                           normal=True)
                else:
                    values = _typed_values(shape,
                                           pool_out_ref.dtype,
                                           params_ref,
                                           normal=False)
                pool_out_ref.at[block][...] = values

        # Typed stores run after the raw attention store, so an overlapping
        # attention/state block ends with valid FP32 SSM and BF16 conv values.
        @pl.when(fill_state)
        def _fill_state_regions():
            for block, token_row, token_rows, live_rows in ssm_segments:
                region = pool_out_ref.at[block, pl.ds(token_row, token_rows)]
                _store_typed_live_rows(region,
                                       live_rows,
                                       jnp.float32,
                                       params_ref,
                                       normal=normal)
            for block, token_row, token_rows, live_rows in conv_segments:
                region = pool_out_ref.at[block, pl.ds(token_row, token_rows)]
                _store_typed_live_rows(region,
                                       live_rows,
                                       jnp.bfloat16,
                                       params_ref,
                                       normal=normal)

    return _kernel


def _pallas_fill_hybrid_slots(pool: jax.Array, indices: jax.Array,
                              fill_kinds: jax.Array, params: jax.Array,
                              seed: jax.Array | None, *, split: int,
                              ssm_bytes: int, conv_bytes: int) -> jax.Array:
    if pool.ndim < 3:
        raise ValueError(
            f"hybrid slot fill requires pool ndim >= 3, got {pool.ndim}")
    if split <= 0 or pool.shape[0] % split:
        raise ValueError(
            f"invalid manager split {split} for {pool.shape[0]} pool rows")
    if indices.ndim != 1 or indices.shape[0] == 0:
        raise ValueError("indices must be a non-empty rank-1 array")
    if fill_kinds.shape != indices.shape:
        raise ValueError(
            f"fill_kinds shape {fill_kinds.shape} != indices {indices.shape}")
    normal = seed is not None
    expected_params = (2, ) if normal else (1, )
    if params.shape != expected_params:
        raise ValueError(
            f"fill parameters must have shape {expected_params}, got "
            f"{params.shape}")
    if normal and seed.shape != (1, ):
        raise ValueError(f"seed must have shape (1,), got {seed.shape}")

    indices = indices.astype(jnp.int32)
    fill_kinds = fill_kinds.astype(jnp.int32)
    ssm_segments, conv_segments = _hybrid_state_segments(pool,
                                                         split=split,
                                                         ssm_bytes=ssm_bytes,
                                                         conv_bytes=conv_bytes)
    kernel = _make_hybrid_fill_kernel(normal=normal,
                                      ssm_segments=ssm_segments,
                                      conv_segments=conv_segments)
    padding = (0, ) * (pool.ndim - 1)
    block_shape = (split, ) + pool.shape[1:]
    num_updates = indices.shape[0]
    if normal:
        assert seed is not None
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                grid=(num_updates, ),
                in_specs=[
                    pl.BlockSpec(
                        block_shape, lambda i, ids, _kinds, _params, _seed:
                        (ids[i], ) + padding)
                ],
                out_specs=pl.BlockSpec(
                    block_shape, lambda i, ids, _kinds, _params, _seed:
                    (ids[i], ) + padding),
            ),
            out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
            input_output_aliases={4: 0},
        )(indices, fill_kinds, params, seed.astype(jnp.int32), pool)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(num_updates, ),
            in_specs=[
                pl.BlockSpec(
                    block_shape, lambda i, ids, _kinds, _params:
                    (ids[i], ) + padding)
            ],
            out_specs=pl.BlockSpec(
                block_shape, lambda i, ids, _kinds, _params:
                (ids[i], ) + padding),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3: 0},
    )(indices, fill_kinds, params, pool)


def pallas_fill_hybrid_slots(pool: jax.Array, indices: jax.Array,
                             fill_kinds: jax.Array, fill_value: jax.Array, *,
                             split: int, ssm_bytes: int,
                             conv_bytes: int) -> jax.Array:
    """Fill attention and typed GDN regions with one donated pool update."""
    return _pallas_fill_hybrid_slots(pool,
                                     indices,
                                     fill_kinds,
                                     fill_value,
                                     None,
                                     split=split,
                                     ssm_bytes=ssm_bytes,
                                     conv_bytes=conv_bytes)


def pallas_normal_fill_hybrid_slots(pool: jax.Array, indices: jax.Array,
                                    fill_kinds: jax.Array, mean_std: jax.Array,
                                    seed: jax.Array, *, split: int,
                                    ssm_bytes: int,
                                    conv_bytes: int) -> jax.Array:
    """Normal-distribution variant of :func:`pallas_fill_hybrid_slots`."""
    return _pallas_fill_hybrid_slots(pool,
                                     indices,
                                     fill_kinds,
                                     mean_std,
                                     seed,
                                     split=split,
                                     ssm_bytes=ssm_bytes,
                                     conv_bytes=conv_bytes)


# --- PyTorch bridge -------------------------------------------------------


def _jax_scatter_op(pool: jax.Array, indices: jax.Array,
                    vals: jax.Array) -> tuple[jax.Array, jax.Array]:
    return pallas_scatter_slots(pool, indices, vals), indices[0]


_pallas_scatter_torch_op = torch_pallas.jax_op(
    "pallas::kv_pool_partial_update", _jax_scatter_op, donate_argnums=(0, ))


def _fake_scatter_op(pool: torch.Tensor, indices: torch.Tensor,
                     vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(pool), torch.empty((),
                                               dtype=indices.dtype,
                                               device=indices.device)


_pallas_scatter_torch_op.register_fake(_fake_scatter_op)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def pallas_index_copy_(cache: torch.Tensor, indices: torch.Tensor,
                       updates: torch.Tensor) -> torch.Tensor:
    """Drop-in for ``cache.index_copy_(0, indices, updates)`` on TPU.

    ``torch.compile`` is load-bearing, not a speed-up. In strict eager the op
    and the ``copy_`` writeback are two separate executions, so the writeback
    lowers to a real pool-sized ``tt_jit_to_copy_copy_``. Inside one compiled
    graph XLA folds it into the donation and it becomes a pointer alias --
    the same pattern as ``mamba_state_copy_op.copy_mamba_state_blocks``.
    """
    new_cache, _ = _pallas_scatter_torch_op(cache, indices.to(torch.int32),
                                            updates)
    cache.copy_(new_cache)
    return cache


def _jax_fill_op(pool: jax.Array, indices: jax.Array,
                 fill_value: jax.Array) -> tuple[jax.Array, jax.Array]:
    return pallas_fill_slots(pool, indices, fill_value), indices[0]


_pallas_fill_torch_op = torch_pallas.jax_op("pallas::kv_pool_constant_fill",
                                            _jax_fill_op,
                                            donate_argnums=(0, ))


def _fake_fill_op(
        pool: torch.Tensor, indices: torch.Tensor,
        fill_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    del fill_value
    return torch.empty_like(pool), torch.empty((),
                                               dtype=indices.dtype,
                                               device=indices.device)


_pallas_fill_torch_op.register_fake(_fake_fill_op)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def pallas_fill_slots_(cache: torch.Tensor, indices: torch.Tensor,
                       fill_value: torch.Tensor) -> torch.Tensor:
    """Fill selected leading-dimension slots without copying the KV pool."""
    new_cache, _ = _pallas_fill_torch_op(cache, indices.to(torch.int32),
                                         fill_value)
    cache.copy_(new_cache)
    return cache


def _jax_normal_fill_op(pool: jax.Array, indices: jax.Array,
                        mean_std: jax.Array,
                        seed: jax.Array) -> tuple[jax.Array, jax.Array]:
    return pallas_normal_fill_slots(pool, indices, mean_std, seed), indices[0]


_pallas_normal_fill_torch_op = torch_pallas.jax_op(
    "pallas::kv_pool_normal_fill", _jax_normal_fill_op, donate_argnums=(0, ))


def _fake_normal_fill_op(
        pool: torch.Tensor, indices: torch.Tensor, mean_std: torch.Tensor,
        seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    del mean_std, seed
    return torch.empty_like(pool), torch.empty((),
                                               dtype=indices.dtype,
                                               device=indices.device)


_pallas_normal_fill_torch_op.register_fake(_fake_normal_fill_op)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def pallas_normal_fill_slots_(cache: torch.Tensor, indices: torch.Tensor,
                              mean_std: torch.Tensor,
                              seed: torch.Tensor) -> torch.Tensor:
    """Normally fill selected leading slots without copying the KV pool."""
    new_cache, _ = _pallas_normal_fill_torch_op(cache, indices.to(torch.int32),
                                                mean_std, seed.to(torch.int32))
    cache.copy_(new_cache)
    return cache


def _jax_hybrid_fill_op(pool: jax.Array, indices: jax.Array,
                        fill_kinds: jax.Array, fill_value: jax.Array,
                        split: int, ssm_bytes: int,
                        conv_bytes: int) -> tuple[jax.Array, jax.Array]:
    updated = pallas_fill_hybrid_slots(pool,
                                       indices,
                                       fill_kinds,
                                       fill_value,
                                       split=split,
                                       ssm_bytes=ssm_bytes,
                                       conv_bytes=conv_bytes)
    return updated, indices[0]


_pallas_hybrid_fill_torch_op = torch_pallas.jax_op(
    "pallas::kv_pool_hybrid_constant_fill",
    _jax_hybrid_fill_op,
    donate_argnums=(0, ))


def _fake_hybrid_fill_op(pool: torch.Tensor, indices: torch.Tensor,
                         fill_kinds: torch.Tensor, fill_value: torch.Tensor,
                         split: int, ssm_bytes: int,
                         conv_bytes: int) -> tuple[torch.Tensor, torch.Tensor]:
    del fill_kinds, fill_value, split, ssm_bytes, conv_bytes
    return torch.empty_like(pool), torch.empty((),
                                               dtype=indices.dtype,
                                               device=indices.device)


_pallas_hybrid_fill_torch_op.register_fake(_fake_hybrid_fill_op)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def pallas_fill_hybrid_slots_(cache: torch.Tensor, indices: torch.Tensor,
                              fill_kinds: torch.Tensor,
                              fill_value: torch.Tensor, *, split: int,
                              ssm_bytes: int, conv_bytes: int) -> torch.Tensor:
    """Fill selected unified attention/GDN manager pages in one donation."""
    new_cache, _ = _pallas_hybrid_fill_torch_op(cache, indices.to(torch.int32),
                                                fill_kinds.to(torch.int32),
                                                fill_value, split, ssm_bytes,
                                                conv_bytes)
    cache.copy_(new_cache)
    return cache


def _jax_normal_hybrid_fill_op(pool: jax.Array, indices: jax.Array,
                               fill_kinds: jax.Array, mean_std: jax.Array,
                               seed: jax.Array, split: int, ssm_bytes: int,
                               conv_bytes: int) -> tuple[jax.Array, jax.Array]:
    updated = pallas_normal_fill_hybrid_slots(pool,
                                              indices,
                                              fill_kinds,
                                              mean_std,
                                              seed,
                                              split=split,
                                              ssm_bytes=ssm_bytes,
                                              conv_bytes=conv_bytes)
    return updated, indices[0]


_pallas_normal_hybrid_fill_torch_op = torch_pallas.jax_op(
    "pallas::kv_pool_hybrid_normal_fill",
    _jax_normal_hybrid_fill_op,
    donate_argnums=(0, ))


def _fake_normal_hybrid_fill_op(
        pool: torch.Tensor, indices: torch.Tensor, fill_kinds: torch.Tensor,
        mean_std: torch.Tensor, seed: torch.Tensor, split: int, ssm_bytes: int,
        conv_bytes: int) -> tuple[torch.Tensor, torch.Tensor]:
    del fill_kinds, mean_std, seed, split, ssm_bytes, conv_bytes
    return torch.empty_like(pool), torch.empty((),
                                               dtype=indices.dtype,
                                               device=indices.device)


_pallas_normal_hybrid_fill_torch_op.register_fake(_fake_normal_hybrid_fill_op)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def pallas_normal_fill_hybrid_slots_(cache: torch.Tensor,
                                     indices: torch.Tensor,
                                     fill_kinds: torch.Tensor,
                                     mean_std: torch.Tensor,
                                     seed: torch.Tensor, *, split: int,
                                     ssm_bytes: int,
                                     conv_bytes: int) -> torch.Tensor:
    """Normally fill unified attention/GDN pages in one donated update."""
    new_cache, _ = _pallas_normal_hybrid_fill_torch_op(
        cache, indices.to(torch.int32), fill_kinds.to(torch.int32), mean_std,
        seed.to(torch.int32), split, ssm_bytes, conv_bytes)
    cache.copy_(new_cache)
    return cache
