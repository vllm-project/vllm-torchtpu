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
# ==============================================================================

"""Dynamic tiling heuristics and VMEM memory estimation for Fused Conv1D-GDN."""

import math

import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.gdn.v3 import config

# Cap decode tiles at 16 sequences: larger tiles increase per-slot state DMA
# descriptors and register spills without reducing per-slot compute.
DECODE_TILE_SIZES = (16, 8, 4, 2, 1)
MIXED_TILE_SIZES = (128, 64, 32, 16, 8, 4, 2, 1)

# Decode step-vs-transfer balance. Each grid step pays a fixed pipeline
# overhead across (num_seqs / T) steps, while the unoverlapped first state
# fetch and last write-back scale with T * n_v. Balancing those terms gives
# T^2 * n_v <= K * num_seqs, where K is the step overhead in per-head state
# transfers.
_DECODE_STEP_COST_IN_HEAD_TRANSFERS = 16


def decode_tile_target(num_seqs: int, n_v: int) -> int:
    """Returns the largest candidate T with T^2 * n_v <= K * num_seqs, else 1."""
    budget = _DECODE_STEP_COST_IN_HEAD_TRANSFERS * num_seqs
    return next((t for t in DECODE_TILE_SIZES if t * t * n_v <= budget), 1)


# Guard 1: at >= 64 value heads per core (Qwen 3.8 at TP=2, Qwen 3.5 at DP=8),
# the per-head 128x128 float32 chunk tiles spill vector registers to VMEM, so a
# 128-token chunk costs nearly as much as two 64-token chunks. C = 128 only wins
# once a batch holds at least 2,048 tokens (16 chunks of 128).
_HIGH_V_HEAD_THRESHOLD = 64
_MIN_HIGH_V_HEAD_BATCH_TOKENS_FOR_C128 = 2048

# Guard 2: in a mixed batch, use C = 128 only if at least 256 prefill tokens
# (two 128-token chunks) can remain once every other active slot is assumed to
# be a decode or verify window. Bigger chunks save a fixed setup cost per chunk,
# but each prefill sequence is padded up to a whole chunk, and on a short,
# ragged prefill part that padding costs more than the setup it saves. The
# bound is a fixed token count rather than one that grows with the head count:
# a per-head bound would also cap a lone prefill of a few hundred tokens in the
# smallest request bucket, which runs 10% to 20% slower at C = 64.
_MIN_MIXED_PREFILL_TOKENS = 256

# The chunk size both guards fall back to (64).
_CAPPED_MIXED_TILE_SIZE = MIXED_TILE_SIZES[1]


def has_prefill_tokens(batch_size: int, active_num_seqs: int, window_size: int) -> bool:
    """Whether a batch holds more tokens than `active_num_seqs` decode windows.

    A pure decode or speculative-verify bucket gives every sequence at most
    `window_size` tokens (1 for decode, `num_spec_tokens + 1` for verify), so
    its token count is at most `active_num_seqs * window_size`. Any token
    beyond that must belong to a sequence being prefilled, which is the only
    kind of sequence the chunked `PER_SEQ` path pads up to a whole chunk.
    """
    return batch_size > active_num_seqs * window_size


def min_prefill_tokens_in_batch(
    batch_size: int, active_num_seqs: int, window_size: int
) -> int:
    """Lower bound on the prefill tokens of a mixed batch.

    The slot table says how many sequences are active, not which are prefills.
    In the worst case for padding, all but one of them are decode or verify
    windows, so at least `batch_size - (active_num_seqs - 1) * window_size`
    tokens belong to prefills.
    """
    return batch_size - (active_num_seqs - 1) * window_size


def mixed_tile_target(
    batch_size: int,
    active_num_seqs: int,
    n_v: int,
    window_size: int = 1,
) -> int:
    """Returns the speed-optimal upper bound (64 or 128) on the mixed chunk size.

    `calculate_mixed_tile_size` only checks whether a chunk fits in VMEM. Even
    when C = 128 fits, C = 64 is faster in two cases:

    1. `n_v >= 64` and `batch_size < 2048`: the 128x128 per-head tiles spill
       vector registers, and the batch is too short to amortize the spills.
    2. Fragmented mixed steps: up to `active_num_seqs - 1` slots hold
       `window_size`-token decodes, leaving
       `batch_size - (active_num_seqs - 1) * window_size` prefill tokens. Below
       256 of them, 128-token ragged-tail padding costs more than the per-chunk
       overhead it saves.

    Pure decode/verify buckets (no prefill tokens, see `has_prefill_tokens`)
    are left uncapped at 128.

    Args:
        batch_size: Token count of the compiled batch bucket (`qkv.shape[0]`),
            which the caller has already padded; not the number of live tokens.
        active_num_seqs: Upper bound on active sequences in the batch.
        n_v: Number of value heads per device.
        window_size: Tokens per decode or speculative-verify sequence.

    Returns:
        Upper bound on the prefill/mixed chunk size (64 or 128).
    """
    max_chunk = MIXED_TILE_SIZES[0]

    # Guard 1: register spills at high value-head counts.
    if (
        n_v >= _HIGH_V_HEAD_THRESHOLD
        and batch_size < _MIN_HIGH_V_HEAD_BATCH_TOKENS_FOR_C128
    ):
        return _CAPPED_MIXED_TILE_SIZE

    # Guard 2: ragged-tail padding on fragmented mixed steps. Only a batch
    # with more than one active sequence and at least one prefill token can
    # be fragmented; a single prefill or a pure decode/verify bucket is not.
    if active_num_seqs > 1 and has_prefill_tokens(
        batch_size, active_num_seqs, window_size
    ):
        effective_prefill_tokens = min_prefill_tokens_in_batch(
            batch_size, active_num_seqs, window_size
        )
        if effective_prefill_tokens < _MIN_MIXED_PREFILL_TOKENS:
            return _CAPPED_MIXED_TILE_SIZE

    return max_chunk


def align_to(x: int, alignment: int) -> int:
    """Aligns an integer upward to the nearest multiple of alignment."""
    return pl.cdiv(x, alignment) * alignment


def get_vmem_estimate_bytes(
    tile_b: int,
    chunk_sz: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    act_out_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    conv_state_dim_size: int,
    is_decode: bool = False,
    carry_scratch_bytes: int = 4,
    window_size: int = 1,
) -> int:
    """Estimates total on-chip VMEM footprint in bytes for a GDN tile."""
    aligned_num_v_heads = align_to(n_v, num_lanes)
    aligned_d_k = align_to(d_k, num_lanes)
    aligned_d_v = align_to(d_v, num_lanes)
    dim_size = align_to(2 * n_kq * d_k + n_v * d_v, num_lanes)
    aligned_out_dim = align_to(n_v * d_v, num_lanes)

    # 1. Double-buffered input activation buffers (QKV, A, B), upcast to float32.
    qkv_bytes = 2 * (tile_b * chunk_sz * dim_size * carry_scratch_bytes)
    b_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * carry_scratch_bytes)
    a_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * carry_scratch_bytes)

    # 2. Double-buffered state cache buffers (convolution and recurrent states).
    # Input state windows hold 1 state per sequence; output state windows hold
    # `window_size` checkpoints per sequence, giving `1 + window_size` state
    # slots per sequence per buffer.
    state_slots_per_seq = 1 + window_size
    conv_state_buffer_bytes = 2 * (
        tile_b
        * state_slots_per_seq
        * max(0, kernel_size - 1)
        * conv_state_dim_size
        * carry_scratch_bytes
    )
    recurrent_state_buffer_bytes = 2 * (
        tile_b * state_slots_per_seq * n_v * d_v * d_k * rec_state_bytes
    )

    # 3. Double-buffered output activation buffer.
    out_bytes = 2 * (tile_b * chunk_sz * aligned_out_dim * act_out_bytes)

    # 4. Working state. Both modes keep one f32 copy of the recurrent state
    # alongside the double-buffered state windows; prefill also keeps the
    # convolution carry.
    scratch_recurrent_bytes = tile_b * n_v * d_v * d_k * carry_scratch_bytes
    if is_decode:
        scratch_conv_bytes = 0
    else:
        scratch_conv_bytes = (
            tile_b * max(0, kernel_size - 1) * dim_size * carry_scratch_bytes
        )

    # 5. Static weight cache references in on-chip memory.
    # Conv weight [K, dim] plus conv bias [dim] in f32; a_log and dt_bias.
    weights_bytes = ((kernel_size + 1) * dim_size * carry_scratch_bytes) + (
        aligned_num_v_heads * 8
    )

    # 6. Working memory for intra-chunk recurrence and per-head projections
    # (chunk_sz is the verify window in decode) plus per-slot vector
    # temporaries.
    intermediate_bytes = (
        tile_b
        * (
            n_v * (5 * chunk_sz * chunk_sz + 3 * chunk_sz * (aligned_d_v + aligned_d_k))
            + 8 * 8 * num_lanes
        )
        * carry_scratch_bytes
    )

    return (
        qkv_bytes
        + b_bytes
        + a_bytes
        + conv_state_buffer_bytes
        + recurrent_state_buffer_bytes
        + out_bytes
        + scratch_conv_bytes
        + scratch_recurrent_bytes
        + weights_bytes
        + intermediate_bytes
    )


def fits_qkv_pad_scratch(
    cfg: config.GDNConfig,
    qpad_shape: tuple[int, int, int, int],
    vmem_limit_bytes: int,
    num_lanes: int,
) -> bool:
    """Returns whether chunked GDN prefill has VMEM headroom for padding.

    Args:
        cfg: Chunked prefill GDN configuration.
        qpad_shape: Candidate float32 odd-column scratch shape from
            `GDNConfig.qkv_pad_scratch_shape`.
        vmem_limit_bytes: Target per-core VMEM budget in bytes.
        num_lanes: TPU vector lane count.
    """
    qpad_bytes = math.prod(qpad_shape) * 4
    vmem_est = get_vmem_estimate_bytes(
        tile_b=cfg.seq_tile_size,
        chunk_sz=cfg.chunk_size,
        n_kq=cfg.num_kq_heads,
        n_v=cfg.num_v_heads,
        d_k=cfg.kq_head_dim,
        d_v=cfg.v_head_dim,
        kernel_size=cfg.kernel_size,
        act_out_bytes=jnp.dtype(cfg.dtypes.act_out).itemsize,
        rec_state_bytes=jnp.dtype(cfg.dtypes.recurrent_state).itemsize,
        num_lanes=num_lanes,
        conv_state_dim_size=cfg.dim_size,
        is_decode=False,
        window_size=cfg.window_size,
    )
    # Sequential chunked prefill keeps one f32 working state live alongside
    # `carry_recurrent_scratch_ref`; block doubling streams heads instead of
    # holding a full extra working state, so only sequential needs that margin.
    working_state_bytes = (
        0
        if cfg.use_block_doubling
        else cfg.seq_tile_size * cfg.num_v_heads * cfg.kq_head_dim * cfg.v_head_dim * 4
    )
    return vmem_est + working_state_bytes + qpad_bytes <= vmem_limit_bytes


def calculate_decode_tile_size(
    batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_out_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
    window_size: int = 1,
) -> int:
    """Largest decode tile whose VMEM estimate fits the limit.

    The speed optimum is decode_tile_target; this is the feasibility ceiling.

    Args:
      batch_size: Total batch size of the active decode sequence.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key head dimension.
      d_v: Value head dimension.
      conv_state_dim_size: Feature dimension size for conv state.
      act_out_dtype: Data type for output activations.
      recurrent_state_dtype: Data type for recurrent state matrix.
      num_lanes: Number of lanes for TPU vector layout alignment.
      vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
      kernel_size: 1D convolution kernel window size.
      window_size: Number of state checkpoints per sequence in decode.

    Returns:
      Derived batch tile size fitting within VMEM capacity limits.
    """
    # Return a minimum valid tile size of 1 for empty or zero-length batches.
    if batch_size <= 0:
        return 1

    act_out_bytes = jnp.dtype(act_out_dtype).itemsize
    rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

    # Search descending power-of-2 tile candidates (up to 16) bounded by
    # batch size.
    decode_candidates = [c for c in DECODE_TILE_SIZES if c <= batch_size]
    decode_tile_size = decode_candidates[-1]

    for cand in decode_candidates:
        vmem_est = get_vmem_estimate_bytes(
            tile_b=cand,
            chunk_sz=window_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            act_out_bytes=act_out_bytes,
            rec_state_bytes=rec_state_bytes,
            num_lanes=num_lanes,
            conv_state_dim_size=conv_state_dim_size,
            is_decode=True,
            window_size=window_size,
        )
        if vmem_est <= vmem_capacity_limit_bytes:
            return cand

    return decode_tile_size


def calculate_mixed_tile_size(
    seq_len: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_out_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
) -> int:
    """Largest prefill chunk whose VMEM estimate fits the limit.

    The speed optimum is mixed_tile_target; this is the feasibility ceiling.

    Args:
      seq_len: Upper bound on the longest prefill sequence, in tokens.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key head dimension.
      d_v: Value head dimension.
      conv_state_dim_size: Feature dimension size for conv state.
      act_out_dtype: Data type for output activations.
      recurrent_state_dtype: Data type for recurrent state matrix.
      num_lanes: Number of lanes for TPU vector layout alignment.
      vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
      kernel_size: 1D convolution kernel window size.

    Returns:
      Derived chunk tile size fitting within VMEM capacity limits.
    """
    # Return a minimum valid chunk size of 1 for empty or zero-length sequences.
    if seq_len <= 0:
        return 1

    act_out_bytes = jnp.dtype(act_out_dtype).itemsize
    rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

    # Select the largest candidate chunk (up to 128 tokens) that fits in VMEM.
    prefill_candidates = [c for c in MIXED_TILE_SIZES if c <= seq_len]
    mixed_tile_size = prefill_candidates[-1]
    for candidate in prefill_candidates:
        vmem_est = get_vmem_estimate_bytes(
            tile_b=1,
            chunk_sz=candidate,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            act_out_bytes=act_out_bytes,
            rec_state_bytes=rec_state_bytes,
            num_lanes=num_lanes,
            conv_state_dim_size=conv_state_dim_size,
            is_decode=False,
        )
        if vmem_est <= vmem_capacity_limit_bytes:
            return candidate

    return mixed_tile_size


def clip_mixed_tile_size(mixed_tile_size: int, batch_size: int) -> int:
    """Clips `mixed_tile_size` to `batch_size` and rounds down to a power of two.

    Guarantees a strictly positive (>= 1) power-of-two chunk size so
    `GDNConfig.triangular_block_size` divides the chunk even for small
    non-power-of-two token buckets or explicit tile overrides.
    """
    clipped = max(1, min(mixed_tile_size, batch_size))
    return 1 << (clipped.bit_length() - 1)


def get_tile_sizes(
    batch_size: int,
    num_seqs: int,
    padded_batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    conv_state_dim_size: int,
    act_out_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    decode_vmem_limit_bytes: int,
    mixed_vmem_limit_bytes: int,
    window_size: int = 1,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
) -> tuple[int, int]:
    """Derives decode and mixed tile sizes within each kernel's VMEM limit.

    The decode kernel holds window_size (num_spec_tokens + 1) state checkpoints
    per slot and compiles with decode_vmem_limit_bytes; the prefill/mixed kernel
    compiles with window_size = 1 under mixed_vmem_limit_bytes.

    Args:
        batch_size: Total unpadded token count in the batch.
        num_seqs: Total number of sequence slots.
        padded_batch_size: Padded token count in the batch.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key head dimension.
        d_v: Value head dimension.
        kernel_size: 1D convolution kernel window size.
        conv_state_dim_size: Feature dimension size for conv state.
        act_out_dtype: Data type for output activations.
        recurrent_state_dtype: Data type for recurrent state matrix.
        num_lanes: Number of lanes for TPU vector layout alignment.
        decode_vmem_limit_bytes: Maximum VMEM capacity in bytes for decode.
        mixed_vmem_limit_bytes: Maximum VMEM capacity in bytes for prefill/mixed.
        window_size: Number of state checkpoints per sequence in decode.
        decode_tile_size: Explicit decode sequence tile size override, if provided.
        mixed_tile_size: Explicit prefill/mixed chunk size override, if provided.

    Returns:
        Tuple of (decode_tile_size, mixed_tile_size).
    """
    # A batch of padded_batch_size tokens spans at most padded_batch_size
    # active sequences even when num_seqs reflects a larger static slot table.
    active_num_seqs = max(1, min(padded_batch_size, num_seqs))
    if decode_tile_size is None or decode_tile_size <= 0:
        vmem_fit = calculate_decode_tile_size(
            batch_size=active_num_seqs,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            conv_state_dim_size=conv_state_dim_size,
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=recurrent_state_dtype,
            num_lanes=num_lanes,
            vmem_capacity_limit_bytes=decode_vmem_limit_bytes,
            kernel_size=kernel_size,
            window_size=window_size,
        )
        decode_tile_size = min(decode_tile_target(active_num_seqs, n_v), vmem_fit)

    if mixed_tile_size is None or mixed_tile_size <= 0:
        # Take the smaller of the largest chunk that fits in VMEM
        # (`calculate_mixed_tile_size`) and the speed-optimal chunk bound
        # (`mixed_tile_target`), mirroring `decode_tile_size` above.
        vmem_fit = calculate_mixed_tile_size(
            seq_len=batch_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            conv_state_dim_size=conv_state_dim_size,
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=recurrent_state_dtype,
            num_lanes=num_lanes,
            vmem_capacity_limit_bytes=mixed_vmem_limit_bytes,
            kernel_size=kernel_size,
        )
        mixed_tile_size = min(
            mixed_tile_target(
                batch_size=batch_size,
                active_num_seqs=active_num_seqs,
                n_v=n_v,
                window_size=window_size,
            ),
            vmem_fit,
        )

    # Guarantee strictly positive tile sizes (>= 1) for Pallas grid compilation,
    # keeping mixed_tile_size a power of two so triangular_block_size divides it.
    decode_tile_size = max(1, min(decode_tile_size, padded_batch_size))
    mixed_tile_size = clip_mixed_tile_size(mixed_tile_size, batch_size)
    return decode_tile_size, mixed_tile_size
