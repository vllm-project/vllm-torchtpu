# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""KDA decode kernel: GDN v3's recurrent path with per-channel (KDA) gating.

Migrated from the former ``kernels/kda/decode.py``. The kernel body is unchanged; only the
imports differ (``l2norm_fwd`` is now inlined as ``_l2_norm``, so this file does
not depend on the package it came from).

Why a decode kernel at all
--------------------------
The chunked KDA kernel (``chunk_kda.py``'s ``kda_forward``) is correct for
pads every 1-token segment to ``chunk_size=64``, so it does ~64x the useful work.
At 128 sequences that measured 2.49 ms/layer against 0.045 ms on a GB200 --- 55x.
Profiling the chunked path at decode shapes showed the Pallas kernel is only
14.3% of device time; 42.9% is four HBM-bound ``gather_fusion`` ops doing the
per-segment state gather in XLA. So decode needs a genuine single-token
recurrence *with the state gather pulled inside the kernel*, which is exactly
what GDN v3's ``GDNMode.BATCHED`` path already is.

What is borrowed from GDN v3
----------------------------
The token-step algebra and layout tricks come from
``gdn/v3/compute_gdn.py:recurrent_gdn_per_seq``:

* ``fused_transpose_broadcast`` is imported directly rather than
  reimplemented. (L2 normalisation uses a local fp32-accumulating helper --
  see ``_l2_norm`` for why GDN's is not reused.)
* The ordering that defers the ``k_t`` outer product until after ``v_new`` is
  formed --- multiplying earlier would expand the ``d_v`` axis by ``d_k``.
* The paged-state contract: a state pool whose slot 0 is a reserved null block,
  addressed through ``state_indices``, updated in place.

What differs from GDN, and why it matters
-----------------------------------------
GDN's gate is one scalar per (token, head): ``a: [batch, n_v]``, decaying the
whole state matrix uniformly (``state * gating_curr`` with ``gating_curr``
shaped ``[n_v, 1, 1]``). KDA's gate is **per channel** --- ``[n_v, d_k]``, one
decay per key dimension --- so the decay becomes ``[n_v, d_k, 1]`` and scales
each *row* of the ``[d_k, d_v]`` state independently.

For the recurrent path that is nearly free: the multiply just broadcasts over
lanes instead of over both axes. This is why decode is the cheap place to add
KDA gating. (The chunked path is where per-channel gating actually hurts: the
``[chunk, chunk]`` scalar decay matrix that lets GDN reweight one matmul cannot
be formed, so it degenerates into a three-index contraction. See
``README.md``.)

Two layout consequences, both handled with GDN's own transpose helper:

* the activated gate needs ``d_k`` on sublanes to scale state rows, so it takes
  the same lane->sublane transpose GDN already applies to ``k``;
* ``A_log`` is per head and broadcasts over channels, while ``dt_bias`` is per
  channel, so gate activation happens in the ``[n_v, d_k]`` layout before that
  transpose.

Relationship to naive_kda.py
----------------------------
``naive_kda.py``'s ``naive_ragged_kda`` computes the same recurrence in plain XLA
and is the natural reference for this kernel, but the two do **not** share an ABI
yet, so they are not drop-in interchangeable:

* activations -- this kernel takes ``q``/``k``/``v``/``g`` split and shaped
  ``[num_seqs, n_v, d]``; ``naive_ragged_kda`` takes them packed and flat
  (``mixed_qkv`` ``[num_tokens, 3 * n_v * d]``, gate ``[num_tokens, n_v * d]``).
* ``beta`` -- taken here **already activated** (post-sigmoid), matching
  ``chunk_kda.py``'s ``kda_forward`` so both validate against the same
  ``naive_ragged_kda`` sigmoids it internally.
* fusion -- ``naive_ragged_kda`` also folds in the gated output RMSNorm
  (``norm_weight``, ``eps``) and the output gate. This kernel stops at the
  recurrence, matching the chunked kernel's boundary and that of the GPU kernel
  it is benchmarked against (``fused_recurrent_kda_packed_decode`` consumes an
  already-convolved ``mixed_qkv`` and norms outside), so measurements stay
  comparable.
* sequence validity -- ``slot > 0`` here (slot 0 reserved); ``state_idx >= 0``
  there.
* ``has_initial_state`` -- passed in explicitly here; derived from
  ``seq_lens > query_lens`` there.

Reconciling those is a separate step.

Scope
-----
* One token per sequence (``window_size == 1``). GDN v3's speculative verify
  windows, which checkpoint one state per window position, are a natural
  extension but are not implemented here.
* conv1d is **not** fused in, unlike ``fused_conv1d_gdn``. This matches the
  boundary of the chunked KDA kernel and of the GPU kernel it is benchmarked
  against, so measurements stay comparable.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Reused verbatim from the GDN v3 recurrent path rather than reimplemented.
from vllm_torchtpu.kernels.gdn.v3.compute_gdn import fused_transpose_broadcast

__all__ = ["kda_decode"]

# Matches ``chunk_kda.py``'s ``l2norm_fwd`` and the inline normalisation in
# ``naive_kda.py:kda_step``.
_L2_EPS = 1e-6


def _l2_norm(x: jax.Array) -> jax.Array:
    """Row-wise L2 normalisation, accumulating in float32.

    ``x * rsqrt(sum(x^2) + eps)`` -- not ``F.normalize``'s clamped form, and not
    GDN's ``l2_norm``: GDN sums the squares in the *input* dtype, which is exact
    enough at this kernel's default ``compute_dtype=float32`` but silently loses
    precision if a caller lowers it to bfloat16 (128 squares accumulated with ~3
    decimal digits). Accumulating in fp32 also keeps prefill and decode
    consistent, since the chunked KDA path always upcasts.

    Inlined from ``chunk_kda.py``'s ``l2norm_fwd`` (which returns
    ``(normalised, rstd)``; only the first is wanted) so this file carries no
    dependency on the package it was migrated out of. The cast back to
    ``x.dtype`` is deliberate and matches that helper: a no-op at float32, a
    truncation at bfloat16.
    """
    x_f = x.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + _L2_EPS)
    return (x_f * rstd[..., None]).astype(x.dtype)


def _activate_gate(g, a_log, dt_bias, lower_bound):
    """KDA gate activation in the ``[n_v, d_k]`` layout.

    ``g`` and ``dt_bias`` are per channel; ``a_log`` is per head and broadcasts
    over the channel (lane) axis. Mirrors tokamax's ``base._activate_gate``,
    and is the same formula as ``naive_kda.py:kda_step``'s gate.
    """
    g_f = g.astype(jnp.float32)
    if dt_bias is not None:
        g_f = g_f + dt_bias.astype(jnp.float32)
    a = jnp.exp(a_log.astype(jnp.float32))  # [n_v, 1]
    if lower_bound is None:
        return -a * jax.nn.softplus(g_f)
    return lower_bound * jax.nn.sigmoid(a * g_f)


def _decode_step(q, k, k_t, v, decay, beta, state, compute_dtype):
    """One KDA token step for one sequence.

    Shapes: ``q``/``k`` ``[n_v, 1, d_k]``, ``k_t`` ``[n_v, d_k, 1]``, ``v``
    ``[n_v, 1, d_v]``, ``decay`` ``[n_v, d_k, 1]``, ``beta`` ``[n_v, 1, 1]``,
    ``state`` ``[n_v, d_k, d_v]``.

    Structure follows ``recurrent_gdn_per_seq``; the only change is that
    ``decay`` carries a ``d_k`` axis instead of being a scalar per head.
    """
    contract_dk = (((2, ), (1, )), ((0, ), (0, )))

    # Per-channel decay: scales row k of the state, broadcasting over d_v.
    state_decayed = state * decay

    # v_updated = k @ (decayed state)   -> [n_v, 1, d_v]
    v_updated = jax.lax.dot(
        k,
        state_decayed,
        dimension_numbers=contract_dk,
        preferred_element_type=jnp.float32,
    ).astype(compute_dtype)

    v_new = beta * (v - v_updated)

    # Deferred outer product: forming this before v_new would expand d_v by d_k.
    state = state_decayed + k_t * v_new

    # Output uses the *updated* state, matching the naive reference.
    out = jax.lax.dot(
        q,
        state,
        dimension_numbers=contract_dk,
        preferred_element_type=jnp.float32,
    )
    return out[:, 0, :], state


def _kernel(
    # Scalar prefetch.
    state_indices_ref,  # [num_seqs] int32
    has_initial_state_ref,  # [num_seqs] int32
    # Inputs.
    q_ref,  # [1, n_v, d_k]
    k_ref,  # [1, n_v, d_k]
    v_ref,  # [1, n_v, d_v]
    g_ref,  # [1, n_v, d_k]  raw gate
    beta_ref,  # [1, n_v, 1]  already activated
    a_log_ref,  # [n_v, 1]
    dt_bias_ref,  # [n_v, d_k]
    state_ref,  # [1, n_v, d_k, d_v]  paged slot
    # Outputs.
    out_ref,  # [1, n_v, d_v]
    state_out_ref,  # [1, n_v, d_k, d_v]  aliased to state_ref
    *,
    n_v: int,
    d_k: int,
    lower_bound: float | None,
    compute_dtype,
    scale: float,
    state_transposed: bool,
):
    s = pl.program_id(0)
    slot = state_indices_ref[s]
    # Two independent flags, easy to conflate:
    #   `slot > 0`          -- the sequence is real. Slot 0 is the reserved null
    #                          block (GDN v3's contract, and the reference GPU
    #                          kernel's `state_idx <= 0` check), used for padded
    #                          or invalid entries: emit nothing, write nothing.
    #   `has_initial_state` -- whether a *real* sequence carries state in from a
    #                          previous step. On its first step it does not, and
    #                          its slot may hold garbage, so start from zero --
    #                          but it still computes and still produces output.
    valid = slot > 0
    use_carried_state = jnp.logical_and(valid, has_initial_state_ref[s] != 0)

    q = q_ref[0].astype(compute_dtype)  # [n_v, d_k]
    k = k_ref[0].astype(compute_dtype)
    v = v_ref[0].astype(compute_dtype)

    q = _l2_norm(q) * scale
    k = _l2_norm(k)

    decay_2d = jnp.exp(
        _activate_gate(g_ref[0], a_log_ref[...], dt_bias_ref[...],
                       lower_bound)).astype(compute_dtype)

    # Promote to the 3-D per-head layout the step works in. Adding leading or
    # trailing size-1 axes is free; moving d_k from lanes to sublanes is not,
    # hence GDN's transpose helper.
    q3 = q.reshape(n_v, 1, d_k)
    k3 = k.reshape(n_v, 1, d_k)
    v3 = v.reshape(n_v, 1, -1)
    k_t = fused_transpose_broadcast(k3, src_dim=2, dst_dim=1)  # [n_v, d_k, 1]
    decay = fused_transpose_broadcast(decay_2d.reshape(n_v, 1, d_k),
                                      src_dim=2,
                                      dst_dim=1)  # [n_v, d_k, 1]
    # beta arrives with n_v already on sublanes, so this needs no transpose --
    # unlike GDN, which receives it with n_v on lanes and must move it.
    beta = beta_ref[0].reshape(n_v, 1, 1).astype(compute_dtype)

    # The pool may store the state as [d_v, d_k] rather than the [d_k, d_v] this
    # kernel works in -- vLLM's KDA cache does. Convert on load and back on
    # store, inside the kernel, so the paged DMA still fetches only this slot.
    stored = state_ref[0].astype(jnp.float32)
    state_in = jnp.swapaxes(stored, -1, -2) if state_transposed else stored
    state = jnp.where(use_carried_state, state_in, 0.0)

    out, new_state = _decode_step(q3, k3, k_t, v3, decay, beta, state,
                                  compute_dtype)

    new_stored = (jnp.swapaxes(new_state, -1, -2)
                  if state_transposed else new_state)
    # Invalid entries emit zeros and leave their slot as found. Their writes land
    # on slot 0, which is reserved and never handed to a request.
    out = jnp.where(valid, out, 0.0)
    new_stored = jnp.where(valid, new_stored, stored)

    out_ref[0] = out.astype(out_ref.dtype)
    state_out_ref[0] = new_stored.astype(state_out_ref.dtype)


@functools.partial(
    jax.jit,
    donate_argnames=("state", ),
    static_argnames=("lower_bound", "compute_dtype", "scale",
                     "state_transposed"),
)
def kda_decode(
    q: jax.Array,  # [num_seqs, n_v, d_k]
    k: jax.Array,  # [num_seqs, n_v, d_k]
    v: jax.Array,  # [num_seqs, n_v, d_v]
    g: jax.Array,  # [num_seqs, n_v, d_k]  raw gate
    beta: jax.Array,  # [num_seqs, n_v]    already activated
    state: jax.Array,  # [num_slots, n_v, d_k, d_v]  slot 0 reserved
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array | None,  # [n_v, d_k]
    state_indices: jax.Array,  # [num_seqs] int32, <=0 means null
    has_initial_state: jax.Array,  # [num_seqs] int32/bool
    *,
    lower_bound: float | None = None,
    compute_dtype: jnp.dtype = jnp.float32,
    scale: float | None = None,
    state_transposed: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """One KDA decode step for ``num_seqs`` sequences, one token each.

    The grid runs one sequence per step, so each step's state slot is fetched by
    ``index_map`` from ``state_indices``. That is the point of the kernel: the
    gather Pallas pipelines as a DMA (overlapping the next slot's fetch with
    this step's compute) instead of XLA materialising it as a separate
    HBM-bound ``gather_fusion``.

    ``state_transposed`` says the pool stores each slot as ``[d_v, d_k]`` instead
    of this kernel's ``[d_k, d_v]``. vLLM's KDA recurrent cache does: validating
    against a GPU dump showed our state was exactly the transpose of vLLM's
    (rmse_rel 1.41 raw vs 0.0015 transposed, uniformly across heads). Set it when
    reading or writing a vLLM-owned pool; leave it off for a pool this kernel
    owns end to end.

    Returns ``(out, state)`` where ``out`` is ``[num_seqs, n_v, d_v]`` and
    ``state`` is the updated pool. ``state`` is donated, so the pool is updated
    in place.
    """
    num_seqs, n_v, d_k = q.shape
    d_v = v.shape[-1]
    if scale is None:
        scale = d_k**-0.5
    if dt_bias is None:
        dt_bias = jnp.zeros((n_v, d_k), dtype=jnp.float32)
    if dt_bias.shape != (n_v, d_k):
        raise ValueError(
            f"dt_bias must be [n_v, d_k] = {(n_v, d_k)} for the decode kernel "
            f"(kda_forward takes it flat as [n_v * d_k]); got {dt_bias.shape}."
        )

    a_log_2d = a_log.reshape(n_v, 1)

    kernel = functools.partial(
        _kernel,
        n_v=n_v,
        d_k=d_k,
        lower_bound=lower_bound,
        compute_dtype=jnp.dtype(compute_dtype),
        scale=float(scale),
        state_transposed=bool(state_transposed),
    )

    # Per-sequence activations are indexed by grid step; the state pool is
    # indexed by that step's slot. Weights are resident (index_map -> 0).
    def seq_map(s, si, hi):
        return (s, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(num_seqs, ),
        in_specs=[
            pl.BlockSpec((1, n_v, d_k), seq_map),  # q
            pl.BlockSpec((1, n_v, d_k), seq_map),  # k
            pl.BlockSpec((1, n_v, d_v), seq_map),  # v
            pl.BlockSpec((1, n_v, d_k), seq_map),  # g
            # beta is carried as [num_seqs, n_v, 1]: Pallas requires a block's
            # last two dims to be divisible by (8, 128) or match the array, and
            # a 2-D [num_seqs, n_v] array blocked as [1, n_v] satisfies neither.
            pl.BlockSpec((1, n_v, 1), seq_map),  # beta
            pl.BlockSpec((n_v, 1), lambda s, si, hi: (0, 0)),  # a_log
            pl.BlockSpec((n_v, d_k), lambda s, si, hi: (0, 0)),  # dt_bias
            # Paged state: this is the gather, expressed as a DMA.
            pl.BlockSpec((1, n_v, d_k, d_v), lambda s, si, hi:
                         (si[s], 0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, n_v, d_v), seq_map),  # out
            pl.BlockSpec((1, n_v, d_k, d_v), lambda s, si, hi:
                         (si[s], 0, 0, 0)),  # state
        ],
    )

    out_shapes = [
        jax.ShapeDtypeStruct((num_seqs, n_v, d_v), v.dtype),
        jax.ShapeDtypeStruct(state.shape, state.dtype),
    ]

    # Alias the state pool in->out so the update is in place. Operand order is
    # (scalar prefetch..., q, k, v, g, beta, a_log, dt_bias, state); the state
    # is operand 9 and maps to output 1.
    return pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=out_shapes,
        input_output_aliases={9: 1},
        name="kda_decode_recurrent",
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
            vmem_limit_bytes=int(0.9 *
                                 pltpu.get_tpu_info().vmem_capacity_bytes),
        ),
    )(
        state_indices.astype(jnp.int32),
        has_initial_state.astype(jnp.int32),
        q,
        k,
        v,
        g,
        beta.reshape(num_seqs, n_v, 1),
        a_log_2d,
        dt_bias,
        state,
    )
