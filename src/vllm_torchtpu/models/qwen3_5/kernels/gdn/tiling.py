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
# Copied from src/vllm_torchtpu/kernels/gdn/v3/tiling.py at vllm-torchtpu main
# c350823af74ba9cdf728daad64df7dd5bcf84e8e. This copy carries the Qwen3.5 GDN v3
# changes and serves the Qwen3.5 MoE architectures only (gdn_kernel_wrapper in
# layers/core/gdn_attention.py picks it); every other model keeps kernels/gdn/v3.
# Upstream v3 changes are not picked up automatically.

"""Tile-size selection and VMEM estimation for the fused Conv1D-GDN v3 kernel."""

import math
from collections.abc import Sequence

import jax.numpy as jnp

# Decode tiles stop at 16 sequences. At n_v=8, T=32 measured +59.6% (N=64) and
# +15.8% (N=256) slower than T=16: the per-slot f32 state copy and vector
# temporaries spill (18.25 MiB at T=32 vs 9.55 MiB at T=16 in the post-RA LLO
# ledger), and each extra slot adds state DMA descriptors without reducing
# per-slot compute.
DECODE_TILE_SIZES = (16, 8, 4, 2, 1)
MIXED_TILE_SIZES = (128, 64, 32, 16, 8, 4, 2, 1)

# Decode tile target balances fixed per-grid-step overhead against per-slot
# work; the VMEM estimate acts as a feasibility ceiling. Constants are
# device-time fits from gdn_attention_benchmark_test on the Qwen3.5 shapes.
_STEP_OVERHEAD_US = 0.31
_DECODE_BASE_US_PER_SEQ = 0.26
_DECODE_US_PER_SEQ_PER_HEAD = 0.0168

_F32_BYTES = 4

# Extra VMEM per qkv column of a PER_SEQ tile when qkv streams as 2D bf16 rows
# (GDNConfig.qkv_2d) rather than f32 rows. Measured for TPU7x (AOT compile) as
# the smallest vmem_limit_bytes that compiles, dense and pooled states, dim
# 1536 to 12288, chunks 32 to 128: 68 to 136 bytes per column, the maximum at
# dim 12288. 64 of them are the two 16-row DMA tails of the double-buffered
# bf16 tile.
_QKV_2D_PER_SEQ_EXTRA_BYTES_PER_COLUMN = 136


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def align_to(x: int, alignment: int) -> int:
    """Aligns an integer upward to the nearest multiple of alignment."""
    return _cdiv(x, alignment) * alignment


def _nearest_candidate(value: float, candidates: Sequence[int]) -> int:
    """Returns the candidate closest to value in log2 space."""
    return min(
        candidates,
        key=lambda c: abs(math.log2(c) - math.log2(max(value, 1.0))),
    )


def decode_tile_target(num_seqs: int, n_v: int) -> int:
    """Returns the smallest T within 2% of the measured decode optimum.

    Model: t(T) = O_step * ceil(N / T) + w_seq(n_v) * ceil(N / T) * T. The
    square-root balance point reproduces the measured knee for N <= 256 and
    n_v in {8, 32, 64}; DECODE_TILE_SIZES bounds it where the model does not
    encode spill growth.
    """
    if num_seqs <= 0 or n_v <= 0:
        return 1
    w_seq = _DECODE_BASE_US_PER_SEQ + _DECODE_US_PER_SEQ_PER_HEAD * n_v
    return _nearest_candidate(
        math.sqrt(2 * num_seqs * _STEP_OVERHEAD_US / w_seq), DECODE_TILE_SIZES
    )


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
    carry_scratch_bytes: int = _F32_BYTES,
    window_size: int = 1,
    qkv_2d: bool = False,
) -> int:
    """Estimates the on-chip VMEM footprint in bytes of one GDN grid step.

    Args:
      tile_b: Sequences per tile (decode T; 1 for PER_SEQ).
      chunk_sz: Tokens per sequence in the tile (decode: window_size).
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key head dimension.
      d_v: Value head dimension.
      kernel_size: 1D convolution kernel window size.
      act_out_bytes: Itemsize of the output activation.
      rec_state_bytes: Itemsize of a dense recurrent state buffer.
      num_lanes: Number of lanes for TPU vector layout alignment.
      conv_state_dim_size: Feature dimension size for conv state.
      is_decode: BATCHED tile (no inter-tile carry) rather than PER_SEQ.
      carry_scratch_bytes: Itemsize of the f32 activations, dense conv state,
        carries and working states.
      window_size: State checkpoints per sequence (num_spec_tokens + 1).
      qkv_2d: qkv streams as 2D bf16 rows (wrapper bf16_qkv_mode=2).
    """
    aligned_num_v_heads = align_to(n_v, num_lanes)
    aligned_d_k = align_to(d_k, num_lanes)
    aligned_d_v = align_to(d_v, num_lanes)
    dim_size = align_to(2 * n_kq * d_k + n_v * d_v, num_lanes)
    aligned_out_dim = align_to(n_v * d_v, num_lanes)
    state_elems = n_v * d_v * d_k

    # 1. Double-buffered input activations: wrapper.py casts b and a to f32
    # before the pipeline, and qkv too unless it streams as bf16.
    qkv_bytes = 2 * tile_b * chunk_sz * dim_size * carry_scratch_bytes
    # BATCHED tiles take no 2D term: at the measured decode shapes the estimate
    # already exceeds the need with 2D bf16 qkv (n_v=8, T=16: 50.3 MiB
    # estimated, 39.3 MiB needed; n_v=64, T=2: 50.1 and 43.8 MiB).
    if qkv_2d and not is_decode:
        qkv_bytes += tile_b * dim_size * _QKV_2D_PER_SEQ_EXTRA_BYTES_PER_COLUMN
    b_bytes = 2 * tile_b * chunk_sz * aligned_num_v_heads * carry_scratch_bytes
    a_bytes = 2 * tile_b * chunk_sz * aligned_num_v_heads * carry_scratch_bytes

    # 2. Double-buffered state tiles. The input window holds one incoming
    # state per sequence and the output window one checkpoint per window
    # position (memory_ref.StateInBufferedRef / StateOutBufferedRef), so each
    # sequence holds 1 + window_size state tiles per buffer.
    # The conv state is f32; the recurrent state tile holds the state dtype
    # (main's granule-addressed state windows, pooled or dense).
    conv_region_bytes = (
        max(0, kernel_size - 1) * conv_state_dim_size * carry_scratch_bytes
    )
    rec_region_bytes = state_elems * rec_state_bytes
    state_slots_per_seq = 1 + window_size
    conv_state_buffer_bytes = 2 * tile_b * state_slots_per_seq * conv_region_bytes
    recurrent_state_buffer_bytes = 2 * tile_b * state_slots_per_seq * rec_region_bytes

    # 3. Double-buffered output activation.
    out_bytes = 2 * tile_b * chunk_sz * aligned_out_dim * act_out_bytes

    # 4. f32 working recurrent state. BATCHED keeps the previous state plus
    # one per emitted checkpoint (1 + window_size per slot). PER_SEQ keeps
    # one copy, its inter-tile recurrent carry, plus the conv carry
    # (GDNConfig.get_scratch_shape_dict).
    if is_decode:
        live_state_copies = 1 + window_size
        scratch_conv_bytes = 0
    else:
        live_state_copies = 1
        scratch_conv_bytes = (
            tile_b * max(0, kernel_size - 1) * dim_size * carry_scratch_bytes
        )
    scratch_recurrent_bytes = (
        tile_b * live_state_copies * state_elems * carry_scratch_bytes
    )

    # 5. Weights resident in VMEM: conv weight [K, dim] + bias [dim] in f32;
    # a_log and dt_bias.
    weights_bytes = (kernel_size + 1) * dim_size * carry_scratch_bytes + (
        aligned_num_v_heads * 8
    )

    # 6. Intra-chunk recurrence, per-head projections and per-sequence
    # vector temporaries (8 arrays of shape [8, num_lanes] per slot).
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
      batch_size: Number of active decode sequences.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key head dimension.
      d_v: Value head dimension.
      conv_state_dim_size: Feature dimension size for conv state.
      act_out_dtype: Data type for output activations.
      recurrent_state_dtype: Data type for a dense recurrent state matrix.
      num_lanes: Number of lanes for TPU vector layout alignment.
      vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
      kernel_size: 1D convolution kernel window size.
      window_size: Number of state checkpoints per sequence in decode.

    Returns:
      The largest DECODE_TILE_SIZES entry <= batch_size that fits, 1 for an
      empty batch, or the smallest candidate when nothing fits.
    """
    if batch_size <= 0:
        return 1

    candidates = [c for c in DECODE_TILE_SIZES if c <= batch_size]
    for cand in candidates:
        vmem_est = get_vmem_estimate_bytes(
            tile_b=cand,
            chunk_sz=window_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            act_out_bytes=jnp.dtype(act_out_dtype).itemsize,
            rec_state_bytes=jnp.dtype(recurrent_state_dtype).itemsize,
            num_lanes=num_lanes,
            conv_state_dim_size=conv_state_dim_size,
            is_decode=True,
            window_size=window_size,
        )
        if vmem_est <= vmem_capacity_limit_bytes:
            return cand
    return candidates[-1]


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
    qkv_2d: bool = False,
) -> int:
    """Largest prefill chunk whose VMEM estimate fits the limit.

    MIXED_TILE_SIZES stops at C=128, the largest chunk measured: C=128 beat
    C=64 on every swept shape where both fit (n_v=8 and 32), and the VMEM
    estimate rejects C=128 at n_v=64. C > 128 is unmeasured.

    Args:
      seq_len: Upper bound on the chunk, in tokens.
      n_kq: Number of key/query heads.
      n_v: Number of value heads.
      d_k: Key head dimension.
      d_v: Value head dimension.
      conv_state_dim_size: Feature dimension size for conv state.
      act_out_dtype: Data type for output activations.
      recurrent_state_dtype: Data type for a dense recurrent state matrix.
      num_lanes: Number of lanes for TPU vector layout alignment.
      vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
      kernel_size: 1D convolution kernel window size.
      qkv_2d: qkv streams as 2D bf16 rows, see get_vmem_estimate_bytes.

    Returns:
      The largest MIXED_TILE_SIZES entry <= seq_len that fits, 1 for an empty
      sequence, or the smallest candidate when nothing fits.
    """
    if seq_len <= 0:
        return 1

    candidates = [c for c in MIXED_TILE_SIZES if c <= seq_len]
    for cand in candidates:
        vmem_est = get_vmem_estimate_bytes(
            tile_b=1,
            chunk_sz=cand,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            act_out_bytes=jnp.dtype(act_out_dtype).itemsize,
            rec_state_bytes=jnp.dtype(recurrent_state_dtype).itemsize,
            num_lanes=num_lanes,
            conv_state_dim_size=conv_state_dim_size,
            is_decode=False,
            window_size=1,
            qkv_2d=qkv_2d,
        )
        if vmem_est <= vmem_capacity_limit_bytes:
            return cand
    return candidates[-1]


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
    qkv_2d: bool = False,
) -> tuple[int, int]:
    """Derives decode and mixed tile sizes within each kernel's VMEM limit.

    The decode kernel holds window_size (num_spec_tokens + 1) state checkpoints
    per slot and compiles with decode_vmem_limit_bytes; the prefill/mixed kernel
    compiles with window_size = 1 under mixed_vmem_limit_bytes. Each limit is
    the vmem_limit_bytes its pallas_call compiles with
    (GDNConfig.get_vmem_limit_bytes), so sizing and compilation share one
    number.

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
        recurrent_state_dtype: Data type for a dense recurrent state matrix.
        num_lanes: Number of lanes for TPU vector layout alignment.
        decode_vmem_limit_bytes: Maximum VMEM capacity in bytes for decode.
        mixed_vmem_limit_bytes: Maximum VMEM capacity in bytes for prefill/mixed.
        window_size: Number of state checkpoints per sequence in decode.
        decode_tile_size: Caller's decode tile, used in place of
            decode_tile_target and still capped by the VMEM fit.
        mixed_tile_size: Caller's prefill/mixed chunk, capped to the largest
            VMEM-feasible MIXED_TILE_SIZES chunk. None derives the chunk from
            batch_size.
        qkv_2d: qkv streams as 2D bf16 rows; sizes the prefill/mixed chunk
            only, see get_vmem_estimate_bytes.

    Returns:
        Tuple of (decode_tile_size, mixed_tile_size).
    """
    shape = dict(
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        conv_state_dim_size=conv_state_dim_size,
        act_out_dtype=act_out_dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        num_lanes=num_lanes,
        kernel_size=kernel_size,
    )

    # A batch of padded_batch_size tokens spans at most padded_batch_size
    # active sequences even when num_seqs reflects a larger static slot table.
    active_num_seqs = max(1, min(padded_batch_size, num_seqs))
    vmem_fit = calculate_decode_tile_size(
        batch_size=active_num_seqs,
        vmem_capacity_limit_bytes=decode_vmem_limit_bytes,
        window_size=window_size,
        **shape,
    )
    if decode_tile_size is None or decode_tile_size <= 0:
        decode_tile_size = decode_tile_target(active_num_seqs, n_v)
    decode_tile_size = min(decode_tile_size, vmem_fit)

    if mixed_tile_size is None or mixed_tile_size <= 0:
        # Size the prefill/mixed chunk from batch_size rather than comparing
        # against num_seqs * window_size, because num_seqs can be a static slot
        # capacity (such as under DCP=4) when a single prefill occupies the bucket.
        mixed_tile_size = calculate_mixed_tile_size(
            seq_len=batch_size,
            vmem_capacity_limit_bytes=mixed_vmem_limit_bytes,
            qkv_2d=qkv_2d,
            **shape,
        )
    else:
        mixed_tile_size = min(
            mixed_tile_size,
            calculate_mixed_tile_size(
                seq_len=MIXED_TILE_SIZES[0],
                vmem_capacity_limit_bytes=mixed_vmem_limit_bytes,
                qkv_2d=qkv_2d,
                **shape,
            ),
        )

    # Guarantee strictly positive tile sizes (>= 1) for Pallas grid compilation.
    decode_tile_size = max(1, min(decode_tile_size, padded_batch_size))
    mixed_tile_size = max(1, min(mixed_tile_size, batch_size))
    return decode_tile_size, mixed_tile_size


def spec_window_tile_cap(
    *,
    window: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    dim: int,
    aligned_num_v_heads: int,
    num_buffers: int,
    vmem_budget: int,
    rec_state_bytes: int = 4,
) -> int:
    """Sequences per BATCHED tile whose state windows fit half the scoped VMEM.

    Each sequence holds one incoming state plus one checkpoint per window
    position in VMEM, which multiplies the per-sequence footprint by
    1 + window (2 at window = 1). Shrink the tile so the double-buffered
    state tiles fit in roughly half the scoped-VMEM budget (the rest goes to
    weights, activations scratch and compiler temporaries). The recurrent
    state tile holds `rec_state_bytes` per element (the state dtype); the
    conv state tile is f32.
    """
    state_positions = 1 + window
    rec_bytes = state_positions * n_v * d_k * d_v * rec_state_bytes
    conv_bytes = state_positions * (kernel_size - 1) * dim * 4
    bytes_per_seq = (
        rec_bytes
        + conv_bytes
        # qkv (fp32), b/a (fp32), out (act_out).
        + window * (dim * 4 + 2 * aligned_num_v_heads * 4 + n_v * d_v * 2)
    )
    return ((vmem_budget // 2) // num_buffers) // bytes_per_seq


def legacy_spec_batch_tile(batch_size: int) -> int:
    """Select token-count decode tile size for speculative verify windows."""
    return 1 if batch_size <= 8 else (2 if batch_size <= 32 else 4)


def select_decode_tile_size(
    *,
    default_tile: int,
    dyn_tile: int | None,
    batch_size: int,
    num_spec_tokens: int,
    spec_tile_cap: int,
    is_kda: bool,
    override: int = 0,
    num_seqs: int | None = None,
) -> int:
    """Final BATCHED-mode tile T for fused_conv1d_gdn.

    Args:
      default_tile: The caller's decode_tile_size (static path).
      dyn_tile: get_tile_sizes' decode tile, or None when dynamic tiling is
        off. Replaces both default_tile and, for verify windows, the legacy
        token-count heuristic.
      batch_size: Tokens in the call (qkv rows).
      num_spec_tokens: Speculative draft tokens per verify window.
      spec_tile_cap: spec_window_tile_cap(...) at window num_spec_tokens + 1.
        Caps the static tile at every window size, since the static tile has
        no VMEM fit of its own, and the dynamic tile on verify windows.
      is_kda: Whether the call runs KDA rather than GDN.
      override: The caller's explicit decode tile (fused_conv1d_gdn
        decode_tile_override); > 0 forces T, except for KDA verify windows,
        which must stay at one sequence.
      num_seqs: Optional upper bound on the number of sequences. Because
        compute_batched_seq_metadata pads its SMEM arrays to a multiple of T,
        T does not need to divide num_seqs.
    """
    tile = default_tile if dyn_tile is None else dyn_tile
    tile = min(tile, batch_size)
    if num_seqs is not None and num_seqs > 0:
        tile = min(tile, num_seqs)
    if dyn_tile is None or num_spec_tokens > 0:
        tile = min(tile, spec_tile_cap)
    if num_spec_tokens > 0:
        heuristic = tile if dyn_tile is not None else legacy_spec_batch_tile(batch_size)
        tile = max(1, min(tile, heuristic))
        if is_kda:
            # KDA verify keeps one sequence per tile until multi-sequence KDA
            # verify tiles are sized by a KDA-aware VMEM estimate.
            tile = 1
    if override > 0 and not (is_kda and num_spec_tokens > 0):
        cap = (
            batch_size
            if (num_seqs is None or num_seqs <= 0)
            else min(batch_size, num_seqs)
        )
        tile = max(1, min(override, cap))
    return max(1, tile)
