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
"""Pallas TPU kernels for intra-chunk state and output steps of Mamba-2 SSD."""

from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.utils import largest_divisor

# Maximum number of SSM heads blocked per Pallas grid step. At `head_block=8`,
# `block_len=64`, `d_head=64`, and `d_state=128` (or even `block_len=256`),
# double-buffered VMEM blocks for `B_c`/`C_c`, `X_c`/`out`, `states`, and the
# 128-lane padded `(block_len, 1)` decay vector consume ~1.6 MiB (~5.5 MiB at
# `block_len=256` in float32), fitting comfortably within TPU scoped VMEM.
_MAX_HEAD_BLOCK_SIZE = 8


def chunk_state_pallas(
    B_c: jax.Array,
    decay_states: jax.Array,
    X_c: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Pallas kernel for Step 2 of 4 of SSD scan (intra-chunk state generation).

    Computes in float32:
      states = jnp.einsum("chln,chlp,chl->chpn", B_c, X_c, decay_states[..., 0])

    Several heads are blocked per grid step (`block_shape=(1, H, ...)` with `H`
    dividing `n_heads`) to amortize per-step Pallas launch overhead while
    keeping the chunk axis `parallel` for megacore.

    Args:
      B_c: (n_chunks, n_heads, block_len, d_state)
      decay_states: (n_chunks, n_heads, block_len, 1)
      X_c: (n_chunks, n_heads, block_len, d_head)
      interpret: Run Pallas in host interpret mode (for CPU unit tests only).

    Returns:
      states: (n_chunks, n_heads, d_head, d_state) in float32.
    """
    if X_c.ndim != 4:
        raise ValueError(
            f"Expected 4D X_c (n_chunks, n_heads, block_len, d_head), got {X_c.shape}."
        )
    n_chunks, n_heads, block_len, d_head = X_c.shape
    if B_c.ndim != 4 or B_c.shape[:3] != (n_chunks, n_heads, block_len):
        raise ValueError(
            f"Expected 4D B_c with leading shape {(n_chunks, n_heads, block_len)}, "
            f"got {B_c.shape}."
        )
    if decay_states.shape != (n_chunks, n_heads, block_len, 1):
        raise ValueError(
            f"Expected decay_states shape {(n_chunks, n_heads, block_len, 1)}, "
            f"got {decay_states.shape}."
        )
    d_state = B_c.shape[-1]
    head_block = largest_divisor(n_heads, _MAX_HEAD_BLOCK_SIZE)

    def kernel(B_ref, decay_ref, X_ref, out_ref):
        B_val = jnp.squeeze(B_ref[...], axis=0).astype(jnp.float32)
        decay_val = jnp.squeeze(decay_ref[...], axis=0).astype(jnp.float32)
        X_val = jnp.squeeze(X_ref[...], axis=0).astype(jnp.float32)

        X_scaled = X_val * decay_val  # (H, block_len, d_head)
        # Keep Precision.HIGHEST so TPU Mosaic does not round the float32
        # `X_scaled` operand to bfloat16 before MXU contraction (which degrades
        # `final_states` max abs error by 409x from 4.10e-5 to 1.68e-2 on bf16
        # inputs and 825x from 6.06e-5 to 5.00e-2 on f32 inputs).
        out = jax.lax.dot_general(
            X_scaled,
            B_val,
            (((1,), (1,)), ((0,), (0,))),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        out_ref[...] = jnp.expand_dims(out, axis=0)

    grid = (n_chunks, n_heads // head_block)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, 1),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, d_head),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h_blk: (c, h_blk, 0, 0),
        block_shape=(1, head_block, d_head, d_state),
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
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name="ssd_chunk_state_pallas",
        )(B_c, decay_states, X_c),
    )


def chunk_scan_pallas(
    C_c: jax.Array,
    states: jax.Array,
    decay_states: jax.Array,
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Pallas kernel for Step 4 of 4 of SSD scan (state-to-output conversion).

    Computes:
      Y_off = jnp.einsum("chln,chpn,chl->chlp", C_c, states, decay_states[..., 0])

    Args:
      C_c: (n_chunks, n_heads, block_len, d_state)
      states: (n_chunks, n_heads, d_head, d_state)
      decay_states: (n_chunks, n_heads, block_len, 1)
      out_dtype: Output dtype; defaults to `C_c.dtype` when omitted.
      interpret: Run Pallas in host interpret mode (for CPU unit tests only).

    Returns:
      Y_off: (n_chunks, n_heads, block_len, d_head) in `out_dtype` (or `C_c.dtype`).
    """
    if C_c.ndim != 4:
        raise ValueError(
            f"Expected 4D C_c (n_chunks, n_heads, block_len, d_state), got {C_c.shape}."
        )
    n_chunks, n_heads, block_len, d_state = C_c.shape
    if (
        states.ndim != 4
        or states.shape[:2] != (n_chunks, n_heads)
        or states.shape[3] != d_state
    ):
        raise ValueError(
            "Expected 4D states with shape "
            f"({n_chunks}, {n_heads}, d_head, {d_state}), "
            f"got {states.shape}."
        )
    if decay_states.shape != (n_chunks, n_heads, block_len, 1):
        raise ValueError(
            f"Expected decay_states shape {(n_chunks, n_heads, block_len, 1)}, "
            f"got {decay_states.shape}."
        )
    d_head = states.shape[-2]
    head_block = largest_divisor(n_heads, _MAX_HEAD_BLOCK_SIZE)
    resolved_out_dtype = jnp.dtype(C_c.dtype if out_dtype is None else out_dtype)

    def kernel(C_ref, states_ref, decay_ref, out_ref):
        C_val = jnp.squeeze(C_ref[...], axis=0).astype(jnp.float32)
        states_val = jnp.squeeze(states_ref[...], axis=0).astype(jnp.float32)
        decay_val = jnp.squeeze(decay_ref[...], axis=0).astype(jnp.float32)

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
        out = (prod * decay_val).astype(resolved_out_dtype)
        out_ref[...] = jnp.expand_dims(out, axis=0)

    grid = (n_chunks, n_heads // head_block)

    in_specs = [
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, d_head, d_state),
        ),
        pl.BlockSpec(
            memory_space=pltpu.VMEM,
            index_map=lambda c, h_blk: (c, h_blk, 0, 0),
            block_shape=(1, head_block, block_len, 1),
        ),
    ]

    out_spec = pl.BlockSpec(
        memory_space=pltpu.VMEM,
        index_map=lambda c, h_blk: (c, h_blk, 0, 0),
        block_shape=(1, head_block, block_len, d_head),
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
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name="ssd_chunk_scan_pallas",
        )(C_c, states, decay_states),
    )


def _segsum_jax(x: jax.Array) -> jax.Array:
    """Computes lower-triangular segment sums of `x` along its last axis.

    For `x` of shape `(..., T)`, returns `out` of shape `(..., T, T)` where:
      - `out[..., i, j] = sum(x[..., j + 1 : i + 1])` for `i > j`
      - `out[..., i, j] = 0.0` for `i == j`
      - `out[..., i, j] = -inf` for `i < j`
    """
    T = x.shape[-1]
    x_expanded = jnp.expand_dims(x, axis=-1)
    x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))

    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_masked = jnp.where(mask, x_repeated, 0.0)

    x_segsum = jnp.cumsum(x_masked, axis=-2)

    mask_diag = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum_masked = jnp.where(mask_diag, x_segsum, -jnp.inf)
    return x_segsum_masked


def ssd_minimal_discrete_hybrid(
    X_c: jax.Array,
    A_c: jax.Array,
    B_c: jax.Array,
    C_c: jax.Array,
    last_chunk_indices: jax.Array,
    initial_states: jax.Array | None = None,
    reset_mask: jax.Array | None = None,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Hybrid JAX/Pallas implementation of the Selective State Space (SSD) scan.

    Implements the discrete SSD scan using Pallas TPU kernels (`kernels/ssd_scan/`)
    for intra-chunk state generation and state-to-output conversion, and native
    JAX for cumulative sums and f32 sequential inter-chunk recurrence.

    Assumes packed layout: `(n_chunks, n_heads, block_len, dim)`.

    Args:
      X_c: (n_chunks, n_heads, block_len, d_head) - Discretized input tensor
      A_c: (n_chunks, n_heads, block_len) - Log decay parameter (dt * A)
      B_c: (n_chunks, n_heads, block_len, d_state) - Input projection
      C_c: (n_chunks, n_heads, block_len, d_state) - Output projection
      last_chunk_indices: (num_seqs,) - Maps each sequence to its last chunk
      initial_states: (n_chunks, n_heads, d_head, d_state) - Optional initial
        states (read only where `reset_mask` is True)
      reset_mask: (n_chunks,) - Boolean mask indicating sequence starts
      interpret: Run Pallas kernels in host interpret mode (CPU tests only)

    Returns:
      Y: (n_chunks, n_heads, block_len, d_head) in `X_c.dtype`
      final_states: (num_seqs, n_heads, d_head, d_state) in `float32`
    """
    if X_c.ndim != 4:
        raise ValueError(
            f"Expected 4D X_c (n_chunks, n_heads, block_len, d_head), got {X_c.shape}."
        )
    n_chunks, n_heads, block_len, d_head = X_c.shape
    if A_c.shape != (n_chunks, n_heads, block_len):
        raise ValueError(
            f"Expected A_c shape {(n_chunks, n_heads, block_len)}, got {A_c.shape}."
        )
    if B_c.ndim != 4 or B_c.shape[:3] != (n_chunks, n_heads, block_len):
        raise ValueError(
            f"Expected 4D B_c with leading shape {(n_chunks, n_heads, block_len)}, "
            f"got {B_c.shape}."
        )
    d_state = B_c.shape[-1]
    if C_c.shape != (n_chunks, n_heads, block_len, d_state):
        raise ValueError(
            f"Expected C_c shape {(n_chunks, n_heads, block_len, d_state)}, "
            f"got {C_c.shape}."
        )
    if last_chunk_indices.ndim != 1:
        raise ValueError(
            "Expected 1D last_chunk_indices (num_seqs,), "
            f"got {last_chunk_indices.shape}."
        )

    if initial_states is None:
        initial_states_f32 = jnp.zeros(
            (n_chunks, n_heads, d_head, d_state), dtype=jnp.float32
        )
    else:
        if initial_states.shape != (n_chunks, n_heads, d_head, d_state):
            raise ValueError(
                "Expected initial_states shape "
                f"{(n_chunks, n_heads, d_head, d_state)}, "
                f"got {initial_states.shape}."
            )
        initial_states_f32 = initial_states.astype(jnp.float32)

    if reset_mask is None:
        reset_mask = jnp.zeros((n_chunks,), dtype=bool).at[0].set(True)
    elif reset_mask.shape != (n_chunks,):
        raise ValueError(
            f"Expected reset_mask shape {(n_chunks,)}, got {reset_mask.shape}."
        )

    A_c_f32 = A_c.astype(jnp.float32)
    A_cumsum = jnp.cumsum(A_c_f32, axis=-1)

    # 1. Intra-chunk diagonal output (Y_diag)
    L = jnp.exp(_segsum_jax(A_c_f32))
    Y_diag = jnp.einsum(
        "chln,chsn,chls,chsp->chlp",
        C_c,
        B_c,
        L,
        X_c,
        precision=jax.lax.Precision.HIGHEST,
    )

    # 2. Intra-chunk state generation in float32 (Pallas). Append a size-1
    # trailing axis `(..., block_len, 1)` so the tensor satisfies Pallas TPU's
    # 2D minimum tiled `block_shape` requirement and broadcasts directly against
    # `(head_block, block_len, d_head)` in registers.
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)[..., None]
    states_c = chunk_state_pallas(B_c, decay_states, X_c, interpret=interpret)

    # 3. Inter-chunk recurrence in float32 (JAX Scan)
    decay_chunk = jnp.exp(A_cumsum[..., -1])

    def scan_body(carry, args):
        states_c_c, decay_chunk_c, init_state_c, reset_c = args
        incoming_state = jnp.where(reset_c, init_state_c, carry)
        decay_exp = decay_chunk_c[:, None, None]
        next_state = incoming_state * decay_exp + states_c_c
        return next_state, (incoming_state, next_state)

    scan_args = (states_c, decay_chunk, initial_states_f32, reset_mask)
    init_carry = jnp.zeros((n_heads, d_head, d_state), dtype=jnp.float32)
    _, (states_incoming, states_outgoing) = jax.lax.scan(
        scan_body, init_carry, scan_args
    )

    # 4. State to output conversion (Pallas). Keep Y_off in float32 until it is
    # summed with Y_diag so opposite-sign terms do not suffer pre-cancellation
    # bfloat16 rounding before the single output cast.
    state_decay_out = jnp.exp(A_cumsum)[..., None]
    Y_off = chunk_scan_pallas(
        C_c,
        states_incoming,
        state_decay_out,
        out_dtype=jnp.float32,
        interpret=interpret,
    )

    Y = (Y_diag.astype(jnp.float32) + Y_off).astype(X_c.dtype)

    # Extract final states for each sequence using `last_chunk_indices`. In
    # static-shape ragged batches (`mamba_mixer2.py`), inactive/padded sequence
    # slots beyond `distribution[2]` carry `seq_chunk_starts[1:] - 1 == -1`
    # sentinels whose outputs are masked out by the caller; clipping to
    # `[0, n_chunks - 1]` avoids out-of-bounds gather indexing.
    safe_last_chunk_indices = jnp.clip(last_chunk_indices, 0, n_chunks - 1)
    final_states = states_outgoing[safe_last_chunk_indices]
    return Y, final_states
