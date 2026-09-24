# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Torch custom-op bridge for Kimi KDA on the fused conv1d + GDN v3 kernel."""

import math
from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.config import VllmConfig

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_VerifyWindowResult = TypeVar("_VerifyWindowResult")


def _guard_kda_verify_window(
    num_window_reqs: jax.Array,
    run_verify_window: Callable[[], _VerifyWindowResult],
    skip_verify_window: Callable[[], _VerifyWindowResult],
) -> _VerifyWindowResult:
    """Skip KDA verification when the device-side window is empty."""
    return jax.lax.cond(num_window_reqs > 0, run_verify_window,
                        skip_verify_window)


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


def _build_fused_core(lower_bound: float | None,
                      eps: float,
                      num_spec_tokens: int = 0):
    """The same op, on the fused conv1d + GDN v3 kernel.

    The fused kernel replaces the four-stage manual pipeline and
    owns the convolution, silu, q/k L2-norm, the gate activation, beta's
    sigmoid, both recurrences and the state write-back. Speculative decoding
    invokes its batched and per-sequence programs separately so empty verify
    windows and empty prefill segments do not touch state.

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
        window_distribution: jax.Array | None = None,
        slot_read_offsets: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        num_tokens, _, num_heads, head_dim = _check_kda_abi(
            mixed_qkv, raw_gate, beta, output_gate, recurrent_state, a_log,
            dt_bias, norm_weight, query_start_loc, state_indices, seq_lens)
        kernel_size = conv_weight.shape[0]
        if window_distribution is None:
            window_distribution = distribution

        # `mixed_qkv`, `conv_state` and `conv_weight` all lay their channels
        # out as (3, H, D) in row-major order, which is exactly the flat
        # `dim` axis the fused kernel indexes -- the reshapes are free. The
        # kernel wants the weight transposed to [dim, 1, kernel_size].
        conv_weight_flat = conv_weight.reshape(kernel_size, -1)
        conv_weight_flat = jnp.transpose(conv_weight_flat, (1, 0))[:, None]
        if num_spec_tokens > 0 and slot_read_offsets is None:
            raise ValueError("slot_read_offsets are required for KDA verify")
        read_offsets = (slot_read_offsets[state_indices]
                        if num_spec_tokens > 0 else None)

        def run_kernel(conv_in,
                       rec_in,
                       *,
                       batched_only=False,
                       prefill_only=False):
            return gdn_wrapper.fused_conv1d_gdn(
                qkv=mixed_qkv,
                # Raw: the kernel applies sigmoid to `b` and the gate activation
                # to `a` itself.
                b=beta,
                a=raw_gate,
                conv_state=conv_in,
                recurrent_state=rec_in,
                conv_weight=conv_weight_flat,
                # Kimi's short convolution is bias-free.
                conv_bias=None,
                a_log=a_log,
                # Per-channel under KDA, and Mosaic rejects the flat form.
                dt_bias=dt_bias.reshape(num_heads, head_dim),
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                # The batched segment contains both ordinary one-token decodes and
                # K+1 target-verify windows.  Ragged query lengths distinguish the
                # two while the recurrent kernel uses one static window shape.
                distribution=window_distribution,
                seq_lens=seq_lens,
                read_offsets=read_offsets,
                n_kq=num_heads,
                n_v=num_heads,
                d_k=head_dim,
                d_v=head_dim,
                kernel_size=kernel_size,
                attention_mode=gdn_config.AttentionMode.KDA,
                gate_lower_bound=lower_bound,
                num_spec_tokens=num_spec_tokens,
                batched_only=batched_only,
                prefill_only=prefill_only,
            )

        if num_spec_tokens > 0:
            window_end = window_distribution[0]
            empty_out = jnp.zeros((num_tokens, num_heads * head_dim),
                                  mixed_qkv.dtype)
            (verify_states, verify_out) = _guard_kda_verify_window(
                window_end,
                lambda: run_kernel(
                    conv_state, recurrent_state, batched_only=True),
                lambda: ((conv_state, recurrent_state), empty_out),
            )
            first_prefill_token = query_start_loc[window_end]
            total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
            ((new_conv_state, new_pool), prefill_out) = jax.lax.cond(
                first_prefill_token < total_tokens,
                lambda: run_kernel(*verify_states, prefill_only=True),
                lambda: (verify_states, empty_out),
            )
            out = jnp.where((jnp.arange(num_tokens)
                             < first_prefill_token)[:, None], verify_out,
                            prefill_out)
        else:
            (new_conv_state,
             new_pool), out = run_kernel(conv_state, recurrent_state)

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
    num_spec_tokens: int = 0,
):
    """Build fused KDA with separately guarded speculative and prefill segments."""
    core = _build_fused_core(lower_bound, eps, num_spec_tokens)
    variant = "fused_window" if num_spec_tokens else "fused"
    logger.info("KDA op %s: using the %s GDN v3 path.", prefix, variant)
    op_name = f"pallas::kimi_dispatched_kda_{variant}_{prefix.replace('.', '_')}"
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
        window_distribution: torch.Tensor,
        slot_read_offsets: torch.Tensor,
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
            window_distribution,
            slot_read_offsets,
        )
        conv_state.copy_(new_conv_state)
        recurrent_state.copy_(new_pool)
        return output

    return dispatched_impl


def _pooled_kda_core(
    mixed_qkv: jax.Array,
    raw_gate: jax.Array,
    beta: jax.Array,
    output_gate: jax.Array,
    pool: jax.Array,  # (num_blocks, block_size, *payload, lanes)
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    norm_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    distribution: jax.Array,
    *,
    lower_bound: float | None,
    eps: float,
    pool_block_tokens: int,
    conv_shard_heads: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    num_reqs = state_indices.shape[0]
    num_heads = a_log.shape[0]
    head_dim = norm_weight.shape[0]
    kernel_size = conv_weight.shape[0]
    elem_bytes = jnp.dtype(pool.dtype).itemsize
    tok_bytes = math.prod(pool.shape[2:]) * elem_bytes
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=num_heads * head_dim * head_dim * 4,
        conv_bytes=(kernel_size - 1) * 3 * num_heads * head_dim * elem_bytes,
        token_bytes=tok_bytes,
    )
    # state_indices are MANAGER block ids, and the region offsets are
    # manager-block-relative. The pool's shape[1] unit is a packed token
    # row (an MLA pool packs `pool.shape[2]` tokens per row), and a backend
    # with a fixed kernel block size splits each manager block into `split`
    # consecutive pool blocks — so measure the regions against the manager
    # block and let gather_region/scatter_region remap the ids
    # (`state_indices * split + kb`) through their split branch.
    tokens_per_row = pool.shape[2] if pool.ndim > 3 else 1
    if pool_block_tokens % tokens_per_row != 0:
        raise ValueError("Manager block size is not a whole number of pool "
                         f"rows: {pool_block_tokens} tokens at "
                         f"{tokens_per_row} tokens/row")
    manager_rows = pool_block_tokens // tokens_per_row
    if manager_rows % pool.shape[1] != 0:
        raise ValueError("Manager block does not split into whole pool "
                         f"blocks: {manager_rows} rows vs pool block of "
                         f"{pool.shape[1]}")
    split = manager_rows // pool.shape[1]
    if layout.required_tokens > manager_rows:
        raise ValueError("KDA state regions exceed the pool block: "
                         f"{layout.required_tokens} > {manager_rows}")

    # SSM region [0, ssm_tokens): f32 view with one head_dim-wide lane group
    # per typed row, so the leading H * D rows are exactly the state.
    ssm_gathered = pool_adapters.gather_region(pool,
                                               state_indices,
                                               tok0=0,
                                               ntok=layout.ssm_tokens,
                                               out_dtype=jnp.float32,
                                               out_lanes=head_dim,
                                               split=split)
    ssm_rows = num_heads * head_dim
    ssm_local = ssm_gathered[:, :ssm_rows, :].reshape(num_reqs, num_heads,
                                                      head_dim, head_dim)

    # Stage-3 stores whole, padded prefill-head shards in rank order. Each
    # shard keeps the source's BF16 lane packing, so Raiden can concatenate
    # aligned byte ranges without half-word scatter/gather. Restore logical
    # (taps, qkv, heads, dim) order in the existing pooled adapter, just as
    # Qwen's pair-blocked layout does. Ordinary serving has one shard.
    shard_heads = num_heads if conv_shard_heads is None else conv_shard_heads
    if shard_heads <= 0 or num_heads % shard_heads:
        raise ValueError("KDA conv layout requires whole source head shards")
    conv_shards = num_heads // shard_heads
    if layout.conv_tokens % conv_shards:
        raise ValueError("KDA conv shards must occupy whole packed pool rows")
    shard_elems = (kernel_size - 1) * 3 * shard_heads * head_dim
    padded_shard_elems = (layout.conv_tokens // conv_shards * tok_bytes //
                          elem_bytes)
    if shard_elems > padded_shard_elems:
        raise ValueError("KDA conv shard exceeds its padded pool region")
    conv_gathered = pool_adapters.gather_region(pool,
                                                state_indices,
                                                tok0=layout.ssm_tokens,
                                                ntok=layout.conv_tokens,
                                                out_dtype=pool.dtype,
                                                split=split)
    conv_local = conv_gathered.reshape(num_reqs, conv_shards,
                                       padded_shard_elems)[:, :, :shard_elems]
    conv_local = conv_local.reshape(num_reqs, conv_shards, kernel_size - 1, 3,
                                    shard_heads, head_dim)
    conv_local = conv_local.transpose(0, 2, 3, 1, 4,
                                      5).reshape(num_reqs, kernel_size - 1, 3,
                                                 num_heads, head_dim)

    # Dense slot 0 is scratch for the kernel's idempotent null-block writes.
    conv_buf = jnp.concatenate([jnp.zeros_like(conv_local[:1]), conv_local],
                               axis=0)
    ssm_buf = jnp.concatenate([jnp.zeros_like(ssm_local[:1]), ssm_local],
                              axis=0)
    identity = jnp.arange(1, num_reqs + 1, dtype=jnp.int32)

    # The fused conv1d + GDN v3 kernel takes the conv state flat
    # [slots, K - 1, 3*H*D]; the buffers here lay their channels out as
    # (K - 1, 3, H, D) in row-major order, so both reshapes are free. Padded
    # and zero-length sequences are safe under identity indices because the
    # kernel gates their state DMAs to zero size rather than redirecting them
    # to the null slot.
    fused_core = _build_fused_core(lower_bound, eps)
    output, new_conv_buf, new_ssm_buf = fused_core(
        mixed_qkv,
        raw_gate,
        beta,
        output_gate,
        conv_buf.reshape(num_reqs + 1, kernel_size - 1, -1),
        ssm_buf,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        query_start_loc,
        identity,
        seq_lens,
        distribution,
    )
    new_conv_buf = new_conv_buf.reshape(num_reqs + 1, kernel_size - 1, 3,
                                        num_heads, head_dim)

    # Scatter the real slots back; the padding rows/elems of each region are
    # zero-filled so the pool bytes stay deterministic.
    ssm_region_rows = layout.ssm_tokens * tok_bytes // (4 * head_dim)
    new_ssm = new_ssm_buf[1:].reshape(num_reqs, ssm_rows, head_dim)
    new_ssm = jnp.pad(new_ssm,
                      ((0, 0), (0, ssm_region_rows - ssm_rows), (0, 0)))
    pool = pool_adapters.scatter_region(pool,
                                        new_ssm,
                                        state_indices,
                                        tok0=0,
                                        ntok=layout.ssm_tokens,
                                        split=split)

    new_conv = new_conv_buf[1:].reshape(num_reqs, kernel_size - 1, 3,
                                        conv_shards, shard_heads, head_dim)
    new_conv = new_conv.transpose(0, 3, 1, 2, 4,
                                  5).reshape(num_reqs, conv_shards,
                                             shard_elems)
    new_conv = jnp.pad(new_conv,
                       ((0, 0), (0, 0), (0, padded_shard_elems - shard_elems)))
    new_conv = new_conv.reshape(num_reqs, -1, pool.shape[-1])
    pool = pool_adapters.scatter_region(pool,
                                        new_conv,
                                        state_indices,
                                        tok0=layout.ssm_tokens,
                                        ntok=layout.conv_tokens,
                                        split=split)

    return output, pool


def build_kimi_pooled_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
    vllm_config: VllmConfig,
):
    """Build the KDA op that reads and writes its state through the pool.

    Same torch-level contract as the dense dispatched op, except the two
    state caches are replaced by the single attention-shaped pool buffer.
    The state regions are gathered out of the pool, run through the fused
    conv1d + GDN v3 kernel (the only kernel the pooled path supports — the
    unfused core is not available here), and scattered back.

    The manager block size (``cache_config.block_size``) lets the op size
    the state regions against the manager block and derive the
    manager->pool-block split for the state-index remap. It is read PER
    CALL, not at build time: the platform's block-size derivation for the
    pool runs after model construction (hybrid layers are built before the
    adjusted value exists), so a value captured at build time is the
    pre-adjustment one (16) and the op would size its regions against the
    wrong manager block.
    """

    conv_shard_heads = None
    if (tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER
            and tpu_envs.TPU_RAIDEN_KIMIK3_ADMISSION):
        total_heads = int(vllm_config.model_config.hf_text_config.
                          linear_attn_config["num_heads"])
        source_tp = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        if source_tp <= 0 or total_heads % source_tp:
            raise ValueError("KDA conv source TP must divide num_heads")
        conv_shard_heads = total_heads // source_tp

    def pooled_core(
        mixed_qkv: jax.Array,
        raw_gate: jax.Array,
        beta: jax.Array,
        output_gate: jax.Array,
        pool: jax.Array,
        conv_weight: jax.Array,
        a_log: jax.Array,
        dt_bias: jax.Array,
        norm_weight: jax.Array,
        query_start_loc: jax.Array,
        state_indices: jax.Array,
        seq_lens: jax.Array,
        distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # Explicit annotated signature: pallas.jax_op's _verify_signature
        # rejects unannotated *args.
        return _pooled_kda_core(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            pool,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
            distribution,
            lower_bound=lower_bound,
            eps=eps,
            pool_block_tokens=vllm_config.cache_config.block_size,
            conv_shard_heads=conv_shard_heads)

    # vLLM's compile cache is keyed on the model config and not on the op
    # body, so the kernel variant goes in the op name (see
    # build_kimi_dispatched_kda_op). A pooled program cached from the unfused
    # era would otherwise be served stale.
    op_name = f"pallas::kimi_pooled_kda_fused_{prefix.replace('.', '_')}"
    if conv_shard_heads is not None:
        # Stored state order is part of the compiled program, including for
        # decode-local requests and prefix-cache seeds, not just PD loads.
        op_name += f"_conv_rank_blocks_v1_h{conv_shard_heads}"
    pooled_op = pallas.jax_op(op_name, pooled_core, donate_argnums=(4, ))

    def _fake_pooled(mixed_qkv, _raw_gate, _beta, _output_gate, pool,
                     _conv_weight, a_log, _dt_bias, norm_weight, *args,
                     **kwargs):
        output = torch.empty(
            (mixed_qkv.size(0), a_log.shape[0], norm_weight.shape[0]),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device)
        return output, torch.empty_like(pool)

    pooled_op.register_fake(_fake_pooled)

    def pooled_impl(
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        pool: torch.Tensor,
        conv_weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        distribution: torch.Tensor,
    ) -> torch.Tensor:
        output, new_pool = pooled_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            pool,
            conv_weight,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
            distribution,
        )
        pool.copy_(new_pool)
        return output

    return pooled_impl
