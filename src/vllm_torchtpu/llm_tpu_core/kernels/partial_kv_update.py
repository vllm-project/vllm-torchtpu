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
result correct but no longer O(update). Empirically, across 14 measured
shapes, the cost stays O(update) when the minormost dim is a multiple of 128
and the second-minor is a power of two. That describes when XLA happens to
choose the standard layout rather than a documented contract, so for an
unfamiliar shape check the buffer assignment: the parameter's layout and the
kernel's should match, and a ``copy`` at pool size means they did not.
"""

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas as torch_pallas


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
    pad = (0,) * (pool.ndim - 1)
    block = (1,) + pool.shape[1:]
    assert vals.shape == (na,) + pool.shape[1:], (vals.shape, pool.shape)

    return pl.pallas_call(
        _blockspec_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na,),
            in_specs=[
                pl.BlockSpec(block, lambda i, s: (i,) + pad),
                # Aliased with the output; never read, so leave it in HBM.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(block, lambda i, s: (s[i],) + pad),
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
            "block would write past the end of the aliased output"
        )
    return pl.pallas_call(
        _make_sublane_block_kernel(width, sublanes, na),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na,),
            in_specs=[
                # The whole update array, resident: the kernel needs any row
                # of it, not just this step's. Block dims equal the array
                # dims, so the tiling escape clause covers it and no padding
                # is needed. vals is kilobytes.
                pl.BlockSpec((na, width), lambda i, s: (0, 0)),
                # Read, unlike the rank >= 3 path: the block holds the slot's
                # neighbours and they have to survive the write-back.
                pl.BlockSpec((sublanes, width), lambda i, s: (s[i] // sublanes, 0)),
            ],
            out_specs=pl.BlockSpec(
                (sublanes, width), lambda i, s: (s[i] // sublanes, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(indices, vals, pool)


def pallas_scatter_slots(
    pool: jax.Array, indices: jax.Array, vals: jax.Array
) -> jax.Array:
    """``pool[indices[i]] = vals[i]``, in place, cost O(len(indices)).

    pool:    (num_slots, ...) -- aliased with the output, never reshaped here.
    indices: (num_updates,)   -- target slot indices.
    vals:    (num_updates, ...) matching ``pool.shape[1:]``.
    """
    if pool.dtype != vals.dtype:
        raise ValueError(
            f"pool dtype {pool.dtype} != vals dtype {vals.dtype}; a cross-dtype "
            "view must be taken on the ref inside the kernel, never in XLA"
        )
    indices = indices.astype(jnp.int32)
    if pool.ndim >= 3:
        return _scatter_slots_nd(pool, indices, vals)
    if pool.ndim == 2:
        return _scatter_slots_2d(pool, indices, vals)
    raise ValueError(f"unsupported pool ndim: {pool.ndim}")


# --- PyTorch bridge -------------------------------------------------------


def _jax_scatter_op(
    pool: jax.Array, indices: jax.Array, vals: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return pallas_scatter_slots(pool, indices, vals), indices[0]


_pallas_scatter_torch_op = torch_pallas.jax_op(
    "pallas::kv_pool_partial_update", _jax_scatter_op, donate_argnums=(0,)
)


def _fake_scatter_op(
    pool: torch.Tensor, indices: torch.Tensor, vals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(pool), torch.empty(
        (), dtype=indices.dtype, device=indices.device
    )


_pallas_scatter_torch_op.register_fake(_fake_scatter_op)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def pallas_index_copy_(
    cache: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor
) -> torch.Tensor:
    """Drop-in for ``cache.index_copy_(0, indices, updates)`` on TPU.

    ``torch.compile`` is load-bearing, not a speed-up. In strict eager the op
    and the ``copy_`` writeback are two separate executions, so the writeback
    lowers to a real pool-sized ``tt_jit_to_copy_copy_``. Inside one compiled
    graph XLA folds it into the donation and it becomes a pointer alias --
    the same pattern as ``mamba_state_copy_op.copy_mamba_state_blocks``.
    """
    new_cache, _ = _pallas_scatter_torch_op(cache, indices.to(torch.int32), updates)
    cache.copy_(new_cache)
    return cache
