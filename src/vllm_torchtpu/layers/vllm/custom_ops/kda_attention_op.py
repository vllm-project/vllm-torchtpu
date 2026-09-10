# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Torch custom-op bridge for Kimi KDA on the fused conv1d + GDN v3 kernel."""

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _token_sequence_ids(
    query_start_loc: jax.Array,
    num_tokens: int,
    num_seqs: int,
) -> jax.Array:
    """Map each token row to the 0-indexed sequence that owns it.

    A token's sequence index is the number of interior sequence starts
    (``query_start_loc[1:num_seqs]``) lying at or below its row, so marking each
    start and taking an inclusive cumsum computes the whole mapping in one pass
    over the token axis.

    ``jnp.searchsorted`` expresses the same thing more directly, but XLA lowers
    it to a vectorised O(num_tokens * num_seqs) comparison below a request
    bucket of about 64 and to a binary search above, which puts the bucket sizes
    production actually uses on the expensive side of the threshold: at 2048
    tokens it costs 0.74ms per layer at a bucket of 64 against 0.23ms at 68.
    This form is flat in the bucket at 0.14ms, which is the dispatch floor.

    ``mode="drop"`` discards starts at or past the token bucket instead of
    clamping them onto the last row -- dummy AOT runs describe more requests
    than fit. Scatter-add accumulates duplicates, so empty sequences correctly
    advance the index by more than one. Total marks cannot exceed
    ``num_seqs - 1``, so the clamp only restates that bound.
    """
    starts = query_start_loc[1:num_seqs]
    marks = jnp.zeros((num_tokens, ), jnp.int32).at[starts].add(1, mode="drop")
    return jnp.minimum(jnp.cumsum(marks), num_seqs - 1)


def kimi_short_conv_scan(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    start_seq: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Kimi short convolution over a ragged batch, without a token loop.

    The shared Pallas ragged-convolution kernel can halt the TPU when many
    donated recurrent buffers are embedded in Kimi's compiled hybrid graph.
    Keep this model-local fallback until that kernel interaction is fixed.

    ``start_seq`` restricts the work to the tokens of segments at or after that
    index; earlier tokens keep a zero output row and their convolution state is
    left untouched. That is how a caller can hand the decode part of a batch to
    a decode kernel that does its own convolution.

    The kernel width is static, so the whole convolution is ``kernel_size``
    shifted multiply-accumulates over the token axis with no loop at all. Only
    the taps reaching back past a sequence's first token need care, and those
    come from that sequence's carried window. Selecting per tap between ``x`` and
    a gather from the window table means there is no special-cased path for the
    first few tokens of a sequence.

    This replaced a per-token loop. Two things made that loop expensive and
    neither was arithmetic: each token gathered from and scattered into the whole
    state pool, which XLA lowers to a dynamic-slice / dynamic-update-slice pair
    over a buffer more than a megabyte wide, and the loop machinery plus the
    per-token output scatter cost about 2us per token on their own. Together they
    came to 13.3us per token, or 27ms for a 2048-token prefill; this form does
    the same 2048 tokens in 0.36ms and is bitwise identical on every live slot.

    The ``lax.cond`` matters for decode. Where the old loop's device-valued trip
    count made a fully restricted batch free, fixed work over the token axis
    would pay full price to produce a result that is then entirely masked away --
    which is exactly what every pure-decode step does. This is a branch on a
    device value inside one traced graph, not a host branch on batch
    composition, so it is safe under the no-guards compile wrapper.
    """
    num_tokens = x.shape[0]
    num_seqs = state_indices.shape[0]
    # Dummy AOT runs may describe more one-token requests than fit in the
    # current token bucket; normal runtime metadata instead repeats the true
    # token total in its padded request tail.
    total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
    if start_seq is None:
        first_seq = jnp.zeros((), jnp.int32)
    else:
        first_seq = jnp.minimum(jnp.asarray(start_seq, jnp.int32), num_seqs)
    first_token = jnp.minimum(query_start_loc[first_seq], total_tokens)

    return jax.lax.cond(
        first_token < total_tokens,
        lambda: _short_conv_tokens(x, conv_state, conv_weight, query_start_loc,
                                   state_indices, seq_lens, first_seq,
                                   first_token, total_tokens),
        lambda: (jnp.zeros_like(x), conv_state),
    )


def _short_conv_tokens(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    first_seq: jax.Array,
    first_token: jax.Array,
    total_tokens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """The loop-free convolution body. See ``kimi_short_conv_scan``."""
    num_tokens = x.shape[0]
    num_seqs = state_indices.shape[0]
    kernel_size = conv_weight.shape[0]
    history = kernel_size - 1

    sequence_ids = _token_sequence_ids(query_start_loc, num_tokens, num_seqs)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    has_initial_state = seq_lens > query_lens

    # Each sequence's carried window, zeroed where there is no history to carry.
    state_broadcast = has_initial_state.reshape((num_seqs, ) + (1, ) *
                                                (conv_state.ndim - 1))
    windows = jnp.where(state_broadcast, conv_state[state_indices], 0)
    windows_2d = windows.reshape(num_seqs, history, -1)
    windows_flat = windows_2d.reshape(num_seqs * history, -1)

    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    position = token_ids - query_start_loc[sequence_ids]

    # Tap j reads the row `history - j` places behind the current token. That row
    # lies inside the sequence once the position is far enough in, and is window
    # row `position + j` otherwise -- which is in range exactly when it is
    # needed, since the fallback only applies while position < history - j. The
    # clip keeps the masked-away lanes in bounds; padded token rows can compute a
    # negative position and are dropped by `walked` below.
    taps = []
    for j in range(kernel_size):
        shift = history - j
        from_x = jnp.pad(x, ((shift, 0), (0, 0)))[:num_tokens]
        flat = sequence_ids * history + position + j
        from_window = windows_flat[jnp.clip(flat, 0, num_seqs * history - 1)]
        taps.append(
            jnp.where((position >= shift)[:, None], from_x, from_window))

    # Reduced over the tap axis the way the per-token loop reduced over its
    # stacked window, so the two agree bit for bit.
    output = jnp.sum(jnp.stack(taps, axis=0).astype(jnp.float32) *
                     conv_weight[:, None, :].astype(jnp.float32),
                     axis=0).astype(x.dtype)
    walked = (token_ids >= first_token) & (token_ids < total_tokens)
    output = jnp.where(walked[:, None], output, 0)

    # The outgoing window is a sequence's last `history` source rows. Those come
    # from `x`, except for a sequence shorter than the window, whose older rows
    # are still the incoming window's.
    rows = jnp.arange(history, dtype=jnp.int32)
    local = query_lens[:, None] - history + rows
    from_x = x[jnp.clip(query_start_loc[:-1][:, None] + local, 0,
                        num_tokens - 1)]
    from_window = jnp.take_along_axis(windows_2d,
                                      jnp.clip(query_lens[:, None] + rows, 0,
                                               history - 1)[..., None],
                                      axis=1)
    new_window = jnp.where((local >= 0)[..., None], from_x, from_window)
    if conv_state.ndim > 3:
        new_window = new_window.reshape(num_seqs, history,
                                        *conv_state.shape[2:])

    # Only sequences that were walked and hold tokens may advance their slot.
    # Everything else would wipe a live slot, so point it at the null block --
    # and hand it what slot 0 already holds, so that write is a no-op. Routing
    # the index alone would leave scratch in slot 0, which `decode_kda` and the
    # per-token loop this replaced both leave strictly alone.
    sequence_all = jnp.arange(num_seqs, dtype=jnp.int32)
    keep = ((query_lens > 0) & (sequence_all >= first_seq)
            & (query_start_loc[:-1] < total_tokens))
    write_indices = jnp.where(keep, state_indices, 0)
    keep_broadcast = keep.reshape((num_seqs, ) + (1, ) * (conv_state.ndim - 1))
    written = jnp.where(keep_broadcast, new_window.astype(conv_state.dtype),
                        conv_state[0])
    new_state = conv_state.at[write_indices].set(written)
    return output, new_state


# ===========================================================================
# Shared KDA helpers
# ===========================================================================


def _gated_output_norm(
    output: jax.Array,  # [num_rows, H, D]
    output_gate: jax.Array,  # [num_rows, H * D] raw gate
    norm_weight: jax.Array,  # [D]
    eps: float,
    activation_dtype,
) -> jax.Array:
    """Per-head ``o_norm`` followed by the sigmoid output gate."""
    output = output.astype(activation_dtype).astype(jnp.float32)
    output *= jax.lax.rsqrt(
        jnp.mean(output * output, axis=-1, keepdims=True) + eps)
    output = (output *
              norm_weight.astype(jnp.float32)).astype(activation_dtype)
    return output * jax.nn.sigmoid(output_gate.reshape(output.shape))


def _check_kda_abi(
    mixed_qkv: jax.Array,
    raw_gate: jax.Array,
    beta: jax.Array,
    output_gate: jax.Array,
    recurrent_state: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
) -> tuple[int, int, int, int]:
    """Validate the shared KDA operand contract; return the shape tuple.

    Deliberately the same contract `ragged_kda` enforces, so swapping ops cannot
    silently change what a caller is allowed to pass.
    """
    num_tokens, mixed_dim = mixed_qkv.shape
    num_heads, head_dim, state_v_dim = recurrent_state.shape[1:]
    if state_v_dim != head_dim or mixed_dim != 3 * num_heads * head_dim:
        raise ValueError("Incompatible KDA activation and state shapes")
    if raw_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA gate shape")
    if beta.shape != (num_tokens, num_heads):
        raise ValueError("Incompatible KDA beta shape")
    if output_gate.shape != (num_tokens, num_heads * head_dim):
        raise ValueError("Incompatible KDA output-gate shape")
    num_sequences = state_indices.shape[0]
    if a_log.shape != (num_heads, ):
        raise ValueError("KDA A_log shape does not match the head count")
    if dt_bias.shape != (num_heads * head_dim, ):
        raise ValueError("KDA dt_bias shape does not match the projection")
    if norm_weight.shape != (head_dim, ):
        raise ValueError("KDA norm weight shape does not match the head size")
    if query_start_loc.shape != (num_sequences + 1, ):
        raise ValueError("KDA query starts and state indices disagree")
    if seq_lens.shape != (num_sequences, ):
        raise ValueError("KDA sequence lengths and state indices disagree")
    if (raw_gate.dtype != mixed_qkv.dtype or beta.dtype != mixed_qkv.dtype
            or output_gate.dtype != mixed_qkv.dtype):
        raise TypeError("KDA activations must have one common dtype")
    if recurrent_state.dtype != jnp.float32:
        raise TypeError("KDA recurrent state must be float32")
    if a_log.dtype != jnp.float32 or dt_bias.dtype != jnp.float32:
        raise TypeError("KDA A_log and dt_bias must be float32")
    if (query_start_loc.dtype != jnp.int32 or state_indices.dtype != jnp.int32
            or seq_lens.dtype != jnp.int32):
        raise TypeError("KDA metadata tensors must be int32")
    return num_tokens, num_sequences, num_heads, head_dim


def _build_fused_core(lower_bound: float | None, eps: float):
    """The same op, on the fused conv1d + GDN v3 kernel.

    One Pallas call replaces the four-stage manual pipeline: the fused kernel
    owns the convolution, silu, q/k L2-norm, the gate activation, beta's
    sigmoid, both recurrences and the state write-back, and it dispatches
    decode against prefill/mixed off `distribution` inside the kernel rather
    than by emitting both paths and selecting per token.

    Only `_gated_output_norm` stays outside, since the fused kernel has no
    notion of the output gate.
    """

    def fused_core(
        mixed_qkv: jax.Array,  # [T, 3 * H * D], pre-convolution
        raw_gate: jax.Array,  # [T, H * D]
        beta: jax.Array,  # [T, H]
        output_gate: jax.Array,  # [T, H * D]
        conv_state: jax.Array,  # [num_slots, K - 1, 3, H, D]
        recurrent_state: jax.Array,  # [num_slots, H, K, V]
        conv_weight: jax.Array,  # [kernel_size, 3, H, D], fused at load time
        a_log: jax.Array,  # [H]
        dt_bias: jax.Array,  # [H * D]
        norm_weight: jax.Array,  # [D]
        query_start_loc: jax.Array,  # [N + 1]
        state_indices: jax.Array,  # [N]
        seq_lens: jax.Array,  # [N]
        distribution: jax.Array,  # [3] int32: [decode_end, _, mixed_end]
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        num_tokens, _, num_heads, head_dim = _check_kda_abi(
            mixed_qkv, raw_gate, beta, output_gate, recurrent_state, a_log,
            dt_bias, norm_weight, query_start_loc, state_indices, seq_lens)
        kernel_size = conv_weight.shape[0]

        # `mixed_qkv`, `conv_state` and `conv_weight` all lay their channels
        # out as (3, H, D) in row-major order, which is exactly the flat
        # `dim` axis the fused kernel indexes -- the reshapes are free. The
        # kernel wants the weight transposed to [dim, 1, kernel_size].
        conv_weight_flat = conv_weight.reshape(kernel_size, -1)
        conv_weight_flat = jnp.transpose(conv_weight_flat, (1, 0))[:, None]

        (new_conv_state, new_pool), out = gdn_wrapper.fused_conv1d_gdn(
            qkv=mixed_qkv,
            # Raw: the kernel applies sigmoid to `b` and the gate activation
            # to `a` itself.
            b=beta,
            a=raw_gate,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            conv_weight=conv_weight_flat,
            # Kimi's short convolution is bias-free.
            conv_bias=None,
            a_log=a_log,
            # Per-channel under KDA, and Mosaic rejects the flat form.
            dt_bias=dt_bias.reshape(num_heads, head_dim),
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            seq_lens=seq_lens,
            n_kq=num_heads,
            n_v=num_heads,
            d_k=head_dim,
            d_v=head_dim,
            kernel_size=kernel_size,
            attention_mode=gdn_config.AttentionMode.KDA,
            gate_lower_bound=lower_bound,
        )

        output = out.reshape(num_tokens, num_heads, head_dim)
        output = _gated_output_norm(output, output_gate, norm_weight, eps,
                                    mixed_qkv.dtype)
        return output, new_conv_state, new_pool

    return fused_core


def build_kimi_dispatched_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
    state_dim_first: bool,
):
    """Build the KDA op on the fused conv1d + GDN v3 kernel.

    Takes the *pre-convolution* projection output and owns both caches: the
    fused kernel owns the convolution, silu, q/k L2-norm, the gate activations,
    both recurrences and the state write-back, and dispatches decode against
    prefill/mixed off ``distribution`` inside the kernel.
    """
    core = _build_fused_core(lower_bound, eps)

    # vLLM's compile cache is keyed on the model config, so the variant goes in
    # the op name. If this line disagrees with the kernels in a profile, the
    # run came from cache.
    logger.info("KDA op %s: using the fused conv1d + GDN v3 path.", prefix)

    op_name = f"pallas::kimi_dispatched_kda_fused_{prefix.replace('.', '_')}"
    dispatched_op = pallas.jax_op(op_name, core, donate_argnums=(4, 5))

    def _fake_dispatched(mixed_qkv, _raw_gate, _beta, _output_gate, conv_state,
                         recurrent_state, *args, **kwargs):
        num_heads, head_dim = recurrent_state.shape[1:3]
        output = torch.empty((mixed_qkv.size(0), num_heads, head_dim),
                             dtype=mixed_qkv.dtype,
                             device=mixed_qkv.device)
        return (output, torch.empty_like(conv_state),
                torch.empty_like(recurrent_state))

    dispatched_op.register_fake(_fake_dispatched)

    def dispatched_impl(
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        conv_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        distribution: torch.Tensor,
    ) -> torch.Tensor:
        output, new_conv_state, new_pool = dispatched_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            conv_state,
            recurrent_state,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
            distribution,
        )
        conv_state.copy_(new_conv_state)
        recurrent_state.copy_(new_pool)
        return output

    return dispatched_impl
