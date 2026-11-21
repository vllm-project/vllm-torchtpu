# Copyright (c) 2024, Albert Gu and Tri Dao.
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
"""Pallas TPU kernels for Mamba-2 and Mamba-3 Selective State Space (SSD) scan.

Implements the two head-batched Mosaic TPU kernels used by chunked SSD:
  - ``chunk_state_pallas``: Step 2 of 4 (intra-chunk state generation).
  - ``chunk_scan_pallas``: Step 4 of 4 (state-to-output conversion).

Both kernels support VMEM-derived ``head_block`` tiling, grouped ``B_c``/``C_c``
projections (including ``blocks_per_group > 1``), sublane-aligned
``decay_states``, and ``Precision.HIGHEST`` ``float32`` state accumulation so
Mamba-2 and Mamba-3 (SISO and MIMO) share the same Pallas kernels.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.math_utils import largest_divisor

# Fallback TPU VMEM parameters used only when `interpret=True` on a CPU host.
_INTERPRET_DEFAULT_VMEM_BYTES = 16 * 1024 * 1024
_INTERPRET_DEFAULT_SUBLANES = 8
_INTERPRET_DEFAULT_LANES = 128
_F32_BYTES = 4
_SCOPED_VMEM_FRACTION = 0.80


def _aligned_vmem_tile_bytes(
    lead_dim: int,
    rows: int,
    cols: int,
    elem_bytes: int,
    num_sublanes: int,
    num_lanes: int,
) -> int:
    """Byte footprint of a 3D VMEM block padded to `(num_sublanes, num_lanes)`."""
    aligned_rows = ((rows + num_sublanes - 1) // num_sublanes) * num_sublanes
    aligned_cols = ((cols + num_lanes - 1) // num_lanes) * num_lanes
    return lead_dim * aligned_rows * aligned_cols * elem_bytes


def _select_head_block_size(
    n_heads: int,
    n_groups: int,
    block_len: int,
    d_head: int,
    d_state: int,
    *,
    include_decay_tile: bool,
    interpret: bool,
) -> tuple[int, int | None]:
    """Selects ``(head_block, vmem_limit_bytes)`` that fit in TPU VMEM.

    Double-buffered input/output tiles are budgeted against half of physical
    VMEM (``vmem_capacity_bytes // 2``), while ``vmem_limit_bytes`` raises
    Mosaic's scoped limit to ``int(0.80 * vmem_capacity_bytes)`` so large-tile
    configs keep 30% of VMEM for ``float32`` accumulators, broadcasts, and
    compiler scratch instead of hitting Mosaic's smaller chip-default cap.

    When `n_groups < n_heads`, `head_block` divides `n_heads // n_groups` so
    each grid step loads a single group slice of `B_c` or `C_c`. When
    `n_groups == n_heads`, `head_block` divides `n_heads`.
    """
    if not interpret:
        tpu_info = pltpu.get_tpu_info()
        vmem_capacity_bytes = int(tpu_info.vmem_capacity_bytes)
        num_sublanes = int(tpu_info.num_sublanes)
        num_lanes = int(tpu_info.num_lanes)
        vmem_limit_bytes: int | None = int(_SCOPED_VMEM_FRACTION * vmem_capacity_bytes)
    else:
        vmem_capacity_bytes = _INTERPRET_DEFAULT_VMEM_BYTES
        num_sublanes = _INTERPRET_DEFAULT_SUBLANES
        num_lanes = _INTERPRET_DEFAULT_LANES
        vmem_limit_bytes = None

    # Reserve half of VMEM for double-buffered input/output blocks, leaving the
    # remaining scoped VMEM budget for compiler scratch and register spills.
    vmem_budget_bytes = vmem_capacity_bytes // 2

    bc_single_bytes = 2 * _aligned_vmem_tile_bytes(
        1, block_len, d_state, _F32_BYTES, num_sublanes, num_lanes
    )
    states_per_head_bytes = 2 * _aligned_vmem_tile_bytes(
        1, d_head, d_state, _F32_BYTES, num_sublanes, num_lanes
    )
    x_per_head_bytes = 2 * _aligned_vmem_tile_bytes(
        1, block_len, d_head, _F32_BYTES, num_sublanes, num_lanes
    )
    decay_per_head_bytes = (
        2
        * _aligned_vmem_tile_bytes(1, 1, block_len, _F32_BYTES, num_sublanes, num_lanes)
        if include_decay_tile
        else 0
    )

    if n_groups < n_heads:
        target_heads = n_heads // n_groups
        fixed_bytes = bc_single_bytes
        per_head_bytes = states_per_head_bytes + x_per_head_bytes + decay_per_head_bytes
    else:
        target_heads = n_heads
        fixed_bytes = 0
        per_head_bytes = (
            bc_single_bytes
            + states_per_head_bytes
            + x_per_head_bytes
            + decay_per_head_bytes
        )

    max_head_block = max(1, (vmem_budget_bytes - fixed_bytes) // per_head_bytes)
    return largest_divisor(target_heads, max_head_block), vmem_limit_bytes


def chunk_state_pallas(
    B_c: jax.Array,
    decay_states: jax.Array,
    X_c: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Pallas kernel for Step 2 of 4 of SSD scan (intra-chunk state generation).

    Computes in float32:
      states = jnp.einsum("cgln,chlp,chl->chpn", B_c, X_c, decay_states[:, :, 0, :])
    where each group `g` in `B_c` spans `n_heads // n_groups` consecutive heads.

    Args:
      B_c: `(n_chunks, n_groups, block_len, d_state)` where `n_groups` divides
        `n_heads`.
      decay_states: `(n_chunks, n_heads, 1, block_len)` with the size-1 axis in
        the sublane dimension so TPU VMEM pads `1` to `8` sublanes rather than
        `128` lanes.
      X_c: `(n_chunks, n_heads, block_len, d_head)`.
      interpret: Run Pallas in host interpret mode (for CPU unit tests only).

    Returns:
      states: `(n_chunks, n_heads, d_head, d_state)` in float32.
    """
    if X_c.ndim != 4 or min(X_c.shape) <= 0:
        raise ValueError(
            "Expected positive 4D X_c (n_chunks, n_heads, block_len, d_head), "
            f"got {X_c.shape}."
        )
    n_chunks, n_heads, block_len, d_head = X_c.shape
    if (
        B_c.ndim != 4
        or B_c.shape[0] != n_chunks
        or B_c.shape[2] != block_len
        or B_c.shape[1] <= 0
        or B_c.shape[3] <= 0
        or n_heads % B_c.shape[1] != 0
    ):
        raise ValueError(
            "Expected positive 4D B_c (n_chunks, n_groups, block_len, d_state) "
            f"with leading/chunk dims ({n_chunks}, n_groups, {block_len}) and "
            f"n_groups dividing n_heads={n_heads}, got {B_c.shape}."
        )
    if decay_states.shape != (n_chunks, n_heads, 1, block_len):
        raise ValueError(
            f"Expected decay_states shape {(n_chunks, n_heads, 1, block_len)}, "
            f"got {decay_states.shape}."
        )
    n_groups = B_c.shape[1]
    d_state = B_c.shape[3]
    head_block, vmem_limit_bytes = _select_head_block_size(
        n_heads,
        n_groups,
        block_len,
        d_head,
        d_state,
        include_decay_tile=True,
        interpret=interpret,
    )
    b_head_block = 1 if n_groups < n_heads else head_block
    blocks_per_group = (n_heads // n_groups) // head_block if n_groups < n_heads else 1

    def kernel(B_ref, decay_ref, X_ref, out_ref):
        B_val = B_ref[...].astype(jnp.float32)
        if b_head_block != head_block:
            B_val = jnp.broadcast_to(B_val, (head_block, block_len, d_state))
        decay_val = jnp.swapaxes(decay_ref[...].astype(jnp.float32), -1, -2)
        X_val = X_ref[...].astype(jnp.float32)

        X_scaled = X_val * decay_val  # (head_block, block_len, d_head)
        # Keep Precision.HIGHEST so TPU Mosaic does not round the float32
        # `X_scaled` operand to bfloat16 before MXU contraction (which degrades
        # `final_states` max abs error by 409x from 4.10e-5 to 1.68e-2 on bf16
        # inputs and 825x from 6.06e-5 to 5.00e-2 on f32 inputs).
        out_ref[...] = jax.lax.dot_general(
            X_scaled,
            B_val,
            (((1,), (1,)), ((0,), (0,))),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    grid = (n_chunks, n_heads // head_block)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk // blocks_per_group, 0, 0),
            block_shape=(None, b_head_block, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(None, head_block, 1, block_len),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(None, head_block, block_len, d_head),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h_blk: (c, h_blk, 0, 0),
        block_shape=(None, head_block, d_head, d_state),
    )

    out_shape = (n_chunks, n_heads, d_head, d_state)

    return cast(
        jax.Array,
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_spec,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name="ssd_chunk_state_pallas",
        )(B_c, decay_states, X_c),
    )


def chunk_scan_pallas(
    C_c: jax.Array,
    states: jax.Array,
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Pallas kernel for Step 4 of 4 of SSD scan (state-to-output conversion).

    Computes:
      Y_off = jnp.einsum("cgln,chpn->chlp", C_c, states)
    where each group `g` in `C_c` spans `n_heads // n_groups` consecutive heads.
    The caller applies the per-position decay `exp(A_cumsum)` to the result.

    Args:
      C_c: `(n_chunks, n_groups, block_len, d_state)` where `n_groups` divides
        `n_heads`.
      states: `(n_chunks, n_heads, d_head, d_state)`.
      out_dtype: Output dtype; defaults to `C_c.dtype` when omitted.
      interpret: Run Pallas in host interpret mode (for CPU unit tests only).

    Returns:
      Y_off: `(n_chunks, n_heads, block_len, d_head)` in `out_dtype` (or `C_c.dtype`).
    """
    if C_c.ndim != 4 or min(C_c.shape) <= 0:
        raise ValueError(
            "Expected positive 4D C_c (n_chunks, n_groups, block_len, d_state), "
            f"got {C_c.shape}."
        )
    n_chunks, n_groups, block_len, d_state = C_c.shape
    if (
        states.ndim != 4
        or min(states.shape) <= 0
        or states.shape[0] != n_chunks
        or states.shape[3] != d_state
        or states.shape[1] % n_groups != 0
    ):
        raise ValueError(
            "Expected positive 4D states (n_chunks, n_heads, d_head, d_state) "
            f"with n_chunks={n_chunks}, d_state={d_state}, and n_heads "
            f"divisible by n_groups={n_groups}, got {states.shape}."
        )
    n_heads = states.shape[1]
    d_head = states.shape[2]
    head_block, vmem_limit_bytes = _select_head_block_size(
        n_heads,
        n_groups,
        block_len,
        d_head,
        d_state,
        include_decay_tile=False,
        interpret=interpret,
    )
    c_head_block = 1 if n_groups < n_heads else head_block
    blocks_per_group = (n_heads // n_groups) // head_block if n_groups < n_heads else 1
    resolved_out_dtype = jnp.dtype(C_c.dtype if out_dtype is None else out_dtype)

    def kernel(C_ref, states_ref, out_ref):
        C_val = C_ref[...].astype(jnp.float32)
        if c_head_block != head_block:
            C_val = jnp.broadcast_to(C_val, (head_block, block_len, d_state))
        states_val = states_ref[...].astype(jnp.float32)

        # Keep Precision.HIGHEST so TPU Mosaic preserves the float32 inter-chunk
        # carry `states_val` during MXU contraction instead of truncating it to
        # bfloat16 (which degrades output `Y` max abs error by 4,368x from
        # 1.45e-4 to 6.33e-1).
        prod = jax.lax.dot_general(
            C_val,
            states_val,
            (((2,), (2,)), ((0,), (0,))),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        out_ref[...] = prod.astype(resolved_out_dtype)

    grid = (n_chunks, n_heads // head_block)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk // blocks_per_group, 0, 0),
            block_shape=(None, c_head_block, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(None, head_block, d_head, d_state),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h_blk: (c, h_blk, 0, 0),
        block_shape=(None, head_block, block_len, d_head),
    )

    out_shape = (n_chunks, n_heads, block_len, d_head)

    return cast(
        jax.Array,
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(out_shape, resolved_out_dtype),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_spec,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name="ssd_chunk_scan_pallas",
        )(C_c, states),
    )
