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
"""Pallas TPU kernels and hybrid scan for Mamba-2 and Mamba-3 SSD.

Adapted from the four-step chunked Selective State Space Dual (SSD) formulation
in ``state-spaces/mamba`` (``mamba_ssm.modules.ssd_minimal.ssd_minimal_discrete``,
Apache-2.0) and Mamba-3 SSD-3, with TPU serving extensions:
  - Head-batched Pallas Mosaic kernels (``chunk_state_pallas``,
    ``chunk_scan_pallas``) preserving ``float32`` state accumulation.
  - Ragged multi-sequence resets (``reset_mask``, ``initial_states``,
    ``last_chunk_indices``) in a single ``jax.lax.scan``.
  - Strict-causal intra-chunk masking (``strict_causal=True``) and MIMO channel
    rank ``R`` (``mimo_rank >= 1``) so Mamba-3 SISO and MIMO share the same
    Pallas ``chunk_state_pallas`` and ``chunk_scan_pallas`` kernels as Mamba-2.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.utils import largest_divisor

# Optional module-top-level Tokamax experimental seam (never imported inside a
# traced or compiled hot path). Disabled by default while vllm-torchtpu builds
# without a tokamax pin.
USE_TOKAMAX: bool = False

try:
    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.ssd_scan import (
        ssd_minimal_discrete_hybrid as _tokamax_ssd_scan,
    )
except ImportError:
    _tokamax_ssd_scan = None

# Maximum number of SSM heads blocked per Pallas grid step. At `head_block=8`,
# `block_len=64`, `d_head=64`, and `d_state=128` (or even `block_len=256`),
# double-buffered VMEM blocks for `B_c`/`C_c`, `X_c`/`out`, `states`, and
# `chunk_state_pallas`'s 128-lane padded `(block_len, 1)` decay vector consume
# ~1.6 MiB (~5.5 MiB at `block_len=256` in float32), fitting within scoped VMEM.
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
    *,
    out_dtype: jax.typing.DTypeLike | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Pallas kernel for Step 4 of 4 of SSD scan (state-to-output conversion).

    Computes:
      Y_off = jnp.einsum("chln,chpn->chlp", C_c, states)

    The caller applies the per-position decay `exp(A_cumsum)` to the result.

    Args:
      C_c: (n_chunks, n_heads, block_len, d_state)
      states: (n_chunks, n_heads, d_head, d_state)
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
    d_head = states.shape[-2]
    head_block = largest_divisor(n_heads, _MAX_HEAD_BLOCK_SIZE)
    resolved_out_dtype = jnp.dtype(C_c.dtype if out_dtype is None else out_dtype)

    def kernel(C_ref, states_ref, out_ref):
        C_val = jnp.squeeze(C_ref[...], axis=0).astype(jnp.float32)
        states_val = jnp.squeeze(states_ref[...], axis=0).astype(jnp.float32)

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
        out_ref[...] = jnp.expand_dims(prod.astype(resolved_out_dtype), axis=0)

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
        )(C_c, states),
    )


def _segsum_jax(
    x: jax.Array,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
) -> jax.Array:
    """Computes lower-triangular segment sums of ``x`` along its last axis.

    For ``x`` of shape ``(..., T)`` with ``mimo_rank == 1``, returns ``out`` of
    shape ``(..., T, T)`` where:
      - ``out[..., i, j] = sum(x[..., j + 1 : i + 1])`` for ``i > j``
      - ``out[..., i, i] = 0.0`` if ``not strict_causal`` else ``-inf``
      - ``out[..., i, j] = -inf`` for ``i < j``

    When ``mimo_rank > 1`` (``T = n_tok * mimo_rank``), the causal mask is
    applied at the token block level ``(n_tok, n_tok)`` and repeated by
    ``mimo_rank`` along both axes so all ``mimo_rank`` sub-channels of token
    ``s`` are included (or excluded under ``strict_causal=True`` when
    ``s >= l``) together.
    """
    if mimo_rank < 1:
        raise ValueError(f"Expected mimo_rank >= 1, got {mimo_rank}.")
    T = x.shape[-1]
    if T % mimo_rank != 0:
        raise ValueError(
            f"Expected last dim ({T}) divisible by mimo_rank ({mimo_rank})."
        )

    x_expanded = jnp.expand_dims(x, axis=-1)
    x_repeated = jnp.broadcast_to(x_expanded, x.shape + (T,))

    mask_subdiag = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x_masked = jnp.where(mask_subdiag, x_repeated, 0.0)
    x_segsum = jnp.cumsum(x_masked, axis=-2)

    k_diag = -1 if strict_causal else 0
    if mimo_rank > 1:
        n_tok = T // mimo_rank
        tok_mask = jnp.tril(jnp.ones((n_tok, n_tok), dtype=bool), k=k_diag)
        mask_keep = jnp.repeat(
            jnp.repeat(tok_mask, mimo_rank, axis=0), mimo_rank, axis=1
        )
    else:
        mask_keep = (
            mask_subdiag if strict_causal else jnp.tril(jnp.ones((T, T), dtype=bool))
        )

    return jnp.where(mask_keep, x_segsum, -jnp.inf)


def ssd_minimal_discrete_hybrid(
    X_c: jax.Array,
    A_c: jax.Array,
    B_c: jax.Array,
    C_c: jax.Array,
    last_chunk_indices: jax.Array,
    initial_states: jax.Array | None = None,
    reset_mask: jax.Array | None = None,
    *,
    strict_causal: bool = False,
    mimo_rank: int = 1,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Hybrid JAX/Pallas Selective State Space (SSD) scan for Mamba-2 and Mamba-3.

    Implements the discrete SSD scan using Pallas TPU kernels
    (``chunk_state_pallas`` and ``chunk_scan_pallas``) for intra-chunk state
    generation and state-to-output conversion, and native JAX for cumulative
    sums and ``float32`` sequential inter-chunk recurrence.

    Assumes packed layout: ``(n_chunks, n_heads, block_len, dim)``, where
    ``block_len == n_tok * mimo_rank`` for MIMO models (``mimo_rank > 1``).

    Args:
      X_c: ``(n_chunks, n_heads, block_len, d_head)`` discretized input tensor.
      A_c: Log decay ``(n_chunks, n_heads, block_len)``; when ``mimo_rank > 1``,
        either per-token ``(n_chunks, n_heads, block_len // mimo_rank)`` or
        rank-expanded ``(n_chunks, n_heads, block_len)`` is accepted.
      B_c: ``(n_chunks, n_heads, block_len, d_state)`` input/key projection.
      C_c: ``(n_chunks, n_heads, block_len, d_state)`` output/query projection.
      last_chunk_indices: ``(num_seqs,)`` final chunk index per sequence.
      initial_states: Optional ``(n_chunks, n_heads, d_head, d_state)`` initial
        states (read only where ``reset_mask`` is True).
      reset_mask: Optional ``(n_chunks,)`` boolean sequence-start mask.
      strict_causal: Exclude intra-token ``s == l`` contributions when True
        (used by Mamba-3, which adds the trapezoidal diagonal analytically).
      mimo_rank: MIMO channel rank ``R >= 1`` flattened into ``block_len``.
      interpret: Run Pallas kernels in host interpret mode (CPU tests only).

    Returns:
      ``(Y, final_states)`` where ``Y`` has shape
      ``(n_chunks, n_heads, block_len, d_head)`` in ``X_c.dtype`` and
      ``final_states`` has shape ``(num_seqs, n_heads, d_head, d_state)`` in
      ``float32``.
    """
    if X_c.ndim != 4:
        raise ValueError(
            f"Expected 4D X_c (n_chunks, n_heads, block_len, d_head), got {X_c.shape}."
        )
    n_chunks, n_heads, block_len, d_head = X_c.shape
    if mimo_rank < 1:
        raise ValueError(f"Expected mimo_rank >= 1, got {mimo_rank}.")
    if block_len % mimo_rank != 0:
        raise ValueError(
            f"Expected block_len ({block_len}) divisible by mimo_rank ({mimo_rank})."
        )
    if mimo_rank == 1:
        if A_c.shape != (n_chunks, n_heads, block_len):
            raise ValueError(
                f"Expected A_c shape {(n_chunks, n_heads, block_len)}, got {A_c.shape}."
            )
    else:
        n_tok = block_len // mimo_rank
        valid_a_shapes = (
            (n_chunks, n_heads, n_tok),
            (n_chunks, n_heads, block_len),
        )
        if A_c.shape not in valid_a_shapes:
            raise ValueError(
                f"Expected A_c shape in {valid_a_shapes} for mimo_rank={mimo_rank}, "
                f"got {A_c.shape}."
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

    if initial_states is not None and initial_states.shape != (
        n_chunks,
        n_heads,
        d_head,
        d_state,
    ):
        raise ValueError(
            "Expected initial_states shape "
            f"{(n_chunks, n_heads, d_head, d_state)}, "
            f"got {initial_states.shape}."
        )
    if reset_mask is not None and reset_mask.shape != (n_chunks,):
        raise ValueError(
            f"Expected reset_mask shape {(n_chunks,)}, got {reset_mask.shape}."
        )

    if USE_TOKAMAX and _tokamax_ssd_scan is not None:
        return _tokamax_ssd_scan(
            X_c,
            A_c,
            B_c,
            C_c,
            last_chunk_indices,
            initial_states,
            reset_mask,
            strict_causal=strict_causal,
            mimo_rank=mimo_rank,
            interpret=interpret,
        )

    if initial_states is None:
        initial_states_f32 = jnp.zeros(
            (n_chunks, n_heads, d_head, d_state), dtype=jnp.float32
        )
    else:
        initial_states_f32 = initial_states.astype(jnp.float32)

    if reset_mask is None:
        reset_mask = jnp.zeros((n_chunks,), dtype=bool).at[0].set(True)

    if mimo_rank > 1:
        n_tok = block_len // mimo_rank
        A_raw_f32 = A_c.astype(jnp.float32)
        if A_raw_f32.shape[-1] == block_len:
            A_tok = jnp.sum(
                A_raw_f32.reshape(n_chunks, n_heads, n_tok, mimo_rank), axis=-1
            )
        else:
            A_tok = A_raw_f32
        # Concentrate each token's decay at sub-channel 0 and 0.0 on 1..R-1 so
        # cumulative decay `A_cumsum` applies uniformly across all R sub-channels
        # of a token before readout and state accumulation in Pallas.
        A_zeros = jnp.zeros(
            (n_chunks, n_heads, n_tok, mimo_rank - 1), dtype=jnp.float32
        )
        A_c_f32 = jnp.concatenate([A_tok[..., None], A_zeros], axis=-1).reshape(
            n_chunks, n_heads, block_len
        )
    else:
        A_c_f32 = A_c.astype(jnp.float32)

    A_cumsum = jnp.cumsum(A_c_f32, axis=-1)

    # 1. Intra-chunk diagonal output (Y_diag). Keep Precision.HIGHEST so XLA TPU
    # does not truncate the float32 segment-decay matrix `L` to bfloat16 before
    # contracting across `block_len` (which degrades float32 Y_diag max abs error
    # from <1e-5 to >3e-3 on 64-token chunks).
    L = jnp.exp(_segsum_jax(A_c_f32, strict_causal=strict_causal, mimo_rank=mimo_rank))
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

    # 3. Inter-chunk recurrence in float32, resetting carry at `reset_mask` starts.
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
    # bfloat16 rounding before the single output cast. The decay from the chunk
    # start is applied here, where XLA fuses it into the sum.
    Y_off = chunk_scan_pallas(
        C_c, states_incoming, out_dtype=jnp.float32, interpret=interpret
    )
    state_decay_out = jnp.exp(A_cumsum)[..., None]
    Y = (Y_diag.astype(jnp.float32) + Y_off * state_decay_out).astype(X_c.dtype)

    # Extract final states for each sequence using `last_chunk_indices`. In
    # static-shape ragged batches (`mamba_mixer2.py` / `mamba_mixer3.py`),
    # inactive/padded sequence slots beyond `distribution[2]` carry `-1`
    # sentinels whose outputs are masked out by the caller; clipping to
    # `[0, n_chunks - 1]` avoids out-of-bounds gather indexing.
    safe_last_chunk_indices = jnp.clip(last_chunk_indices, 0, n_chunks - 1)
    final_states = states_outgoing[safe_last_chunk_indices]
    return Y, final_states
