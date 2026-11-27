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

import dataclasses

import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.gdn.v3 import config

GDNConfig = config.GDNConfig
DECODE_TILE_SIZES = GDNConfig.DECODE_TILE_SIZES
MIXED_TILE_SIZES = GDNConfig.MIXED_TILE_SIZES

# Decode step-vs-transfer balance. Each grid step pays a fixed pipeline
# overhead across (num_seqs / T) steps, while the unoverlapped first state
# fetch and last write-back scale with T * n_v. Balancing those terms gives
# T^2 * n_v <= K * num_seqs, where K is the step overhead in per-head state
# transfers.
_DECODE_STEP_COST_IN_HEAD_TRANSFERS = 16
# Default TPU sublane count (8 on TPU v6e/v7x) for callers that omit num_sublanes.
_DEFAULT_NUM_SUBLANES = 8
# TPU lane count for callers that omit num_lanes.
_DEFAULT_NUM_LANES = 128
# KDA chunked prefill keeps 3 live [chunk_sz, d_k] decay-scaled buffers in
# compute_kda.chunked_kda_per_seq: k_beta_gating, q_gated, and k_gating_last.
_KDA_LIVE_DECAY_TENSORS = 3


@dataclasses.dataclass(frozen=True)
class VmemTileSpec:
    """Tile geometry and dtypes used to estimate declared and peak VMEM."""

    tile_b: int
    chunk_sz: int
    n_kq: int
    n_v: int
    d_k: int
    d_v: int
    kernel_size: int
    act_out_bytes: int
    rec_state_bytes: int
    num_lanes: int
    conv_state_dim_size: int
    is_decode: bool = False
    carry_scratch_bytes: int = 4
    window_size: int = 1
    attention_mode: config.AttentionMode = config.AttentionMode.GDN
    num_sublanes: int = _DEFAULT_NUM_SUBLANES

    @classmethod
    def from_config(
        cls,
        cfg: config.GDNConfig,
        conv_state_dim_size: int,
        num_lanes: int,
        num_sublanes: int = _DEFAULT_NUM_SUBLANES,
    ) -> "VmemTileSpec":
        """Build a VmemTileSpec directly from a GDNConfig."""
        return cls(
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
            conv_state_dim_size=conv_state_dim_size,
            is_decode=cfg.mode == config.GDNMode.BATCHED,
            window_size=cfg.window_size,
            attention_mode=cfg.attention_mode,
            num_sublanes=num_sublanes,
        )


def get_declared_vmem_bytes(spec: VmemTileSpec) -> int:
    """Estimate declared VMEM buffer, carry scratch, and weight bytes."""
    aligned_num_v_heads = align_to(spec.n_v, spec.num_lanes)
    aligned_d_k = align_to(spec.d_k, spec.num_lanes)
    dim_size = align_to(2 * spec.n_kq * spec.d_k + spec.n_v * spec.d_v, spec.num_lanes)
    aligned_out_dim = align_to(spec.n_v * spec.d_v, spec.num_lanes)
    is_kda = spec.attention_mode == config.AttentionMode.KDA
    aligned_gate_dim = align_to(
        spec.n_v * spec.d_k if is_kda else spec.n_v, spec.num_lanes
    )

    # 1. Double-buffered input activation buffers (QKV, A, B), upcast to float32.
    inputs_bytes = (
        2
        * spec.tile_b
        * spec.chunk_sz
        * (dim_size + aligned_num_v_heads + aligned_gate_dim)
        * spec.carry_scratch_bytes
    )

    # 2. Double-buffered state cache buffers (convolution and recurrent states).
    # Input state windows hold 1 state per sequence; output state windows hold
    # `window_size` checkpoints per sequence (`1 + window_size` slots/buffer).
    state_slots_per_seq = 1 + spec.window_size
    conv_state_buffer_bytes = (
        2
        * spec.tile_b
        * state_slots_per_seq
        * max(0, spec.kernel_size - 1)
        * spec.conv_state_dim_size
        * spec.carry_scratch_bytes
    )
    recurrent_state_buffer_bytes = (
        2
        * spec.tile_b
        * state_slots_per_seq
        * spec.n_v
        * spec.d_v
        * spec.d_k
        * spec.rec_state_bytes
    )

    # 3. Double-buffered output activation buffer.
    out_bytes = 2 * spec.tile_b * spec.chunk_sz * aligned_out_dim * spec.act_out_bytes

    # 4. Working carry scratch. Prefill keeps f32 copies of convolution and
    # recurrent carries for inter-tile chunking; decode mode has no carries.
    if spec.is_decode:
        scratch_bytes = 0
    else:
        scratch_conv_bytes = (
            spec.tile_b
            * max(0, spec.kernel_size - 1)
            * dim_size
            * spec.carry_scratch_bytes
        )
        scratch_recurrent_bytes = (
            spec.tile_b * spec.n_v * spec.d_v * spec.d_k * spec.carry_scratch_bytes
        )
        scratch_bytes = scratch_conv_bytes + scratch_recurrent_bytes

    # 5. Static weight cache references in on-chip memory.
    # Conv weight [K, dim] plus conv bias [dim] in f32; a_log [n_v] (1D
    # lane-aligned) and dt_bias ([n_v, d_k] 2D sublane-x-lane tiled for KDA,
    # or [n_v] 1D lane-aligned for GDN).
    dt_bias_elems = (
        align_to(spec.n_v, spec.num_sublanes) * aligned_d_k
        if is_kda
        else aligned_num_v_heads
    )
    weights_bytes = (spec.kernel_size + 1) * dim_size * spec.carry_scratch_bytes + (
        aligned_num_v_heads + dt_bias_elems
    ) * spec.carry_scratch_bytes

    return (
        inputs_bytes
        + conv_state_buffer_bytes
        + recurrent_state_buffer_bytes
        + out_bytes
        + scratch_bytes
        + weights_bytes
    )


def decode_tile_target(num_seqs: int, n_v: int) -> int:
    """Returns the largest candidate T with T^2 * n_v <= K * num_seqs, else 1."""
    budget = _DECODE_STEP_COST_IN_HEAD_TRANSFERS * num_seqs
    return next((t for t in DECODE_TILE_SIZES if t * t * n_v <= budget), 1)


def align_to(x: int, alignment: int) -> int:
    """Aligns an integer upward to the nearest multiple of alignment."""
    return pl.cdiv(x, alignment) * alignment


_MIB = 1024 * 1024
# Headroom above the VMEM tile estimate for Mosaic internal scratch and VREG
# spills, matching collectives/all_gather_matmul.py. Empirical: on TPU v5e
# (JAX 0.11.2, libtpu 0.0.50) the bare estimate fell at least 3.6 MiB short
# for some KDA decode and prefill launches, and every such launch swept
# compiled with this margin. The sweep covered GDN up to 64 value heads, KDA
# up to 64 heads, FP32 and BF16 dense and pooled state, 128- and 512-token
# buckets, and static tiling plus dynamic tiling for KDA.
SCOPED_VMEM_MARGIN_BYTES = 8 * _MIB


def _estimate_vmem_from_spec(spec: VmemTileSpec) -> int:
    """Estimate total on-chip VMEM footprint in bytes from a VmemTileSpec."""
    declared_bytes = get_declared_vmem_bytes(spec)

    # 6. Working memory for intra-chunk recurrence and per-head projections
    # (chunk_sz is the verify window in decode) plus per-slot vector
    # temporaries and the working f32 recurrent state during decode.
    aligned_d_k = align_to(spec.d_k, spec.num_lanes)
    aligned_d_v = align_to(spec.d_v, spec.num_lanes)
    is_kda = spec.attention_mode == config.AttentionMode.KDA
    decode_working_recurrent_bytes = (
        spec.tile_b * spec.n_v * spec.d_v * spec.d_k * spec.carry_scratch_bytes
        if spec.is_decode
        else 0
    )
    kda_decay_elems = (
        _KDA_LIVE_DECAY_TENSORS * spec.chunk_sz * aligned_d_k if is_kda else 0
    )
    intermediate_bytes = (
        spec.tile_b
        * (
            spec.n_v
            * (
                5 * spec.chunk_sz * spec.chunk_sz
                + 3 * spec.chunk_sz * (aligned_d_v + aligned_d_k)
                + kda_decay_elems
            )
            + spec.num_sublanes * spec.num_sublanes * spec.num_lanes
        )
        * spec.carry_scratch_bytes
    )

    return declared_bytes + decode_working_recurrent_bytes + intermediate_bytes


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
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    num_sublanes: int = _DEFAULT_NUM_SUBLANES,
) -> int:
    """Estimate total on-chip VMEM footprint in bytes for a GDN or KDA tile."""
    spec = VmemTileSpec(
        tile_b=tile_b,
        chunk_sz=chunk_sz,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_out_bytes=act_out_bytes,
        rec_state_bytes=rec_state_bytes,
        num_lanes=num_lanes,
        conv_state_dim_size=conv_state_dim_size,
        is_decode=is_decode,
        carry_scratch_bytes=carry_scratch_bytes,
        window_size=window_size,
        attention_mode=attention_mode,
        num_sublanes=num_sublanes,
    )
    return _estimate_vmem_from_spec(spec)


def get_scoped_vmem_limit_bytes(
    cfg: config.GDNConfig,
    conv_state_dim_size: int,
    num_lanes: int,
    num_sublanes: int = _DEFAULT_NUM_SUBLANES,
    default_limit_bytes: int | None = None,
    margin_bytes: int | None = None,
) -> int:
    """Size per-launch scoped VMEM from the tile estimate plus compile margin.

    Rounded up to 1 MiB and never above ``default_limit_bytes`` so the launch
    leaves unused VMEM available for XLA weight prefetching without changing
    tile selection. Speculative verify windows keep ``default_limit_bytes``.
    """
    cap_bytes = (
        GDNConfig.get_vmem_limit_bytes(cfg.window_size)
        if default_limit_bytes is None
        else default_limit_bytes
    )
    if cfg.window_size > 1:
        # Mosaic keeps every retained window checkpoint live, so its scratch
        # grows faster than this estimate: KDA with 32 heads and a 4-token
        # window needs 36.71 MiB on v5e against an estimated 33 MiB.
        return cap_bytes
    margin = SCOPED_VMEM_MARGIN_BYTES if margin_bytes is None else margin_bytes
    spec = VmemTileSpec.from_config(
        cfg=cfg,
        conv_state_dim_size=conv_state_dim_size,
        num_lanes=num_lanes,
        num_sublanes=num_sublanes,
    )
    estimate_bytes = _estimate_vmem_from_spec(spec)
    return min(cap_bytes, align_to(estimate_bytes + margin, _MIB))


def get_cost_estimate(
    cfg: config.GDNConfig,
    num_seqs: int,
    conv_state_dim_size: int,
    has_conv_bias: bool = False,
    has_prefill_checkpoint: bool = False,
    num_lanes: int = _DEFAULT_NUM_LANES,
    conv_weight_dtype: jnp.dtype = jnp.float32,
    a_log_dtype: jnp.dtype = jnp.float32,
    dt_bias_dtype: jnp.dtype = jnp.float32,
) -> pl.CostEstimate:
    """Analytical ``pl.CostEstimate`` for a GDN/KDA v3 launch from bucket shapes.

    Gives XLA's scheduler the launch's compute and HBM traffic so it can
    overlap next-layer weight prefetches with the custom call. Compute is
    counted for every tile the bucket can launch, so it bounds launches whose
    sequences fill fewer tiles. Per tile it follows the traced conv, SiLU and
    recurrence math: ``flops`` counts 2 per dot_general multiply-add, 1 per
    output element of a floating-point add, sub, mul or div, and 1 per element
    a reduce_sum folds away (masked lane transposes included); max, min, neg,
    select and transcendentals add no flops. ``transcendentals`` counts each
    exp, log1p, logistic, rsqrt and sqrt output element. Weight bytes use the
    dtypes the kernel receives.
    """
    tokens = int(cfg.batch_size)
    dim = int(cfg.dim_size)
    n_kq = int(cfg.num_kq_heads)
    n_v = int(cfg.num_v_heads)
    d_k = int(cfg.kq_head_dim)
    d_v = int(cfg.v_head_dim)
    k_sz = int(cfg.kernel_size)
    tile_size = int(cfg.tile_size)
    window_size = int(cfg.window_size)
    num_seqs = int(num_seqs)
    conv_state_dim_size = int(conv_state_dim_size)
    num_lanes = int(num_lanes)
    gate_dim = n_v * d_k if cfg.is_kda else n_v
    # The wrapper pads b and a to whole lanes, and the kernel DMAs and
    # activates every padded lane.
    aligned_v_heads = align_to(n_v, num_lanes)
    aligned_gate_dim = align_to(gate_dim, num_lanes)

    active_seqs = min(tokens, num_seqs)
    if cfg.mode == config.GDNMode.BATCHED:
        # Ordinary decodes and full verify windows share the static tile shape.
        checkpoints = min(tokens, active_seqs * window_size)
        num_tiles = pl.cdiv(active_seqs, tile_size)
        chunk = window_size
        rows_per_tile = tile_size * window_size
    else:
        # Each nonempty segment consumes one tile for its first token and one
        # more for every full chunk after that. Checkpoints can split a sequence.
        segments = min(tokens, active_seqs * (1 + has_prefill_checkpoint))
        num_tiles = segments + (tokens - segments) // tile_size if segments else 0
        chunk = rows_per_tile = tile_size
        checkpoints = active_seqs
        if has_prefill_checkpoint:
            checkpoints += min(active_seqs, tokens // 2)
    compute_rows = num_tiles * rows_per_tile

    act_in_bytes = jnp.dtype(cfg.dtypes.act_in).itemsize
    act_out_bytes = jnp.dtype(cfg.dtypes.act_out).itemsize
    rec_bytes = jnp.dtype(cfg.dtypes.recurrent_state).itemsize
    conv_bytes = jnp.dtype(cfg.dtypes.conv_state).itemsize

    # Activations (qkv, lane-padded b and a in; out out) + static weights read
    # once from HBM.
    act_bytes = tokens * (
        (dim + aligned_v_heads + aligned_gate_dim) * act_in_bytes
        + n_v * d_v * act_out_bytes
    )
    weight_bytes = (
        (k_sz + (1 if has_conv_bias else 0))
        * dim
        * jnp.dtype(conv_weight_dtype).itemsize
        + n_v * jnp.dtype(a_log_dtype).itemsize
        + gate_dim * jnp.dtype(dt_bias_dtype).itemsize
    )

    state_slot_bytes = (
        max(0, k_sz - 1) * conv_state_dim_size * conv_bytes
        + n_v * d_k * d_v * rec_bytes
    )

    # Every row runs the causal conv taps (and bias) and SiLU over dim, the
    # q/k L2 norms and q scale, sigmoid(b) (scaled when beta_scale != 1), the
    # gate's softplus or bounded sigmoid, and lane transposes of beta (and of
    # GDN's gate) as masked sums over the padded lanes.
    bounded_gate = cfg.is_kda and cfg.gate_lower_bound is not None
    gate_elems = n_v * d_k if cfg.is_kda else aligned_v_heads
    row_flops = (
        dim * (2 * k_sz + (1 if has_conv_bias else 0) + 1)
        + 7 * n_kq * d_k
        + (3 if bounded_gate else 5) * gate_elems
        + (aligned_v_heads if cfg.beta_scale != 1.0 else 0)
        + (1 if cfg.is_kda else 2) * aligned_v_heads * (aligned_v_heads - 1)
    )
    row_transcendentals = (
        dim + 2 * n_kq + aligned_v_heads + (1 if bounded_gate else 2) * aligned_gate_dim
    )
    # exp(A_log): once per tile for KDA, per lane-padded head for GDN.
    tile_transcendentals = n_v if cfg.is_kda else aligned_v_heads
    tile_flops = 0
    if cfg.use_recurrent:
        # Per token: state decay, the [k, q] projection dot and the rank-1
        # update, the beta-scaled residual and output, q . k, and k's
        # transpose as a masked sum.
        row_flops += (
            7 * n_v * d_k * d_v
            + 4 * n_v * d_v
            + 2 * n_kq * d_k
            - n_kq
            + n_kq * d_k * (d_k - 1)
        )
        # Per-token decay exp; GDN also recomputes exp(A_log) per position.
        row_transcendentals += aligned_gate_dim
        if not cfg.is_kda:
            tile_transcendentals = chunk * aligned_v_heads
    else:
        block = min(cfg.triangular_block_size, chunk)
        num_blocks = chunk // block
        # Shared by both solves: T_inv @ [v, k], [w, q] @ state, the state
        # update, the attention @ (u - ws) output, the blocked forward
        # substitution of T_inv, and the beta, residual and output products.
        tile_flops = (
            2 * n_v * chunk * chunk * (d_v + d_k)
            + 4 * n_v * chunk * d_k * d_v
            + 2 * n_v * d_k * d_v * chunk
            + 2 * n_v * chunk * chunk * d_v
            + n_v * block * block * chunk * num_blocks * (num_blocks - 1)
            + (num_blocks - 1) * n_v * block * chunk
            + num_blocks * n_v * chunk * block * (block - 1)
            + 2 * n_v * d_k * d_v
            + 3 * n_v * chunk * d_v
        )
        if cfg.is_kda:
            # Sub-block solve of the per-channel decay: factored off-diagonal
            # exps, exact pairwise diagonal decays, the forward-substituted
            # transition rows, and the decay-scaled k/q/state terms.
            pairs = num_blocks * (num_blocks - 1) // 2
            block_rows = block * (block - 1) // 2
            transition_rows = num_blocks * block_rows
            diag_work = n_v * chunk * block * d_k
            tile_flops += (
                (chunk - 1) * n_v * d_k
                + pairs * n_v * block * d_k * (4 * block + 7)
                + num_blocks * n_v * block * block * (4 * d_k - 1)
                + num_blocks * n_v * d_k * (3 * block_rows + block - 1)
                + n_v * transition_rows * (d_k - 1)
                + 5 * n_v * chunk * d_k
            )
            tile_transcendentals += (
                3 * n_v * pairs * block * d_k
                + diag_work
                + n_v * transition_rows * d_k
                + 3 * n_v * chunk * d_k
                + n_v * d_k
            )
        else:
            # beta k k^T and q k^T, the pairwise decay map, the decay-scaled
            # q/k terms, and the decay map's transpose as a masked sum.
            tile_flops += (
                (chunk - 1) * aligned_v_heads
                + 2 * n_v * chunk * chunk * d_k
                + 2 * n_kq * chunk * chunk * d_k
                + 3 * n_v * chunk * chunk
                + 4 * n_v * chunk * d_k
                + n_v * chunk * (chunk - 1)
            )
            # Pairwise decay map plus forward and backward decay vectors.
            tile_transcendentals += n_v * (chunk * chunk + 2 * chunk)

    state_bytes = (active_seqs + checkpoints) * state_slot_bytes

    return pl.CostEstimate(
        flops=compute_rows * row_flops + num_tiles * tile_flops,
        transcendentals=(
            compute_rows * row_transcendentals + num_tiles * tile_transcendentals
        ),
        bytes_accessed=act_bytes + weight_bytes + state_bytes,
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
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    num_sublanes: int = _DEFAULT_NUM_SUBLANES,
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
      attention_mode: Attention mode (GDN or KDA).
      num_sublanes: Number of sublanes for 2D TPU vector layout alignment.

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
            attention_mode=attention_mode,
            num_sublanes=num_sublanes,
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
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    num_sublanes: int = _DEFAULT_NUM_SUBLANES,
) -> int:
    """Largest prefill chunk whose VMEM estimate fits the limit.

    Searches candidate tile sizes within maximum VMEM capacity limits.

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
      attention_mode: Attention mode (GDN or KDA).
      num_sublanes: Number of sublanes for 2D TPU vector layout alignment.

    Returns:
      Derived chunk tile size fitting within VMEM capacity limits.
    """
    # Return a minimum valid chunk size of 1 for empty or zero-length sequences.
    if seq_len <= 0:
        return 1

    act_out_bytes = jnp.dtype(act_out_dtype).itemsize
    rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

    # In KDA, each chunk solves O((C / B)^2) sub-blocks with per-channel [d_k]
    # gate decay. Once n_v >= HEADS_PER_GROUP fills a full sublane group,
    # C = MAX_KDA_FULL_GROUP_CHUNK saturates the intra-chunk pipeline and
    # C = 128 doubles per-token off-diagonal work.
    is_kda_full_group = (
        attention_mode == config.AttentionMode.KDA and n_v >= GDNConfig.HEADS_PER_GROUP
    )
    prefill_candidates = [
        c
        for c in MIXED_TILE_SIZES
        if c <= seq_len
        and (not is_kda_full_group or c <= GDNConfig.MAX_KDA_FULL_GROUP_CHUNK)
    ]
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
            attention_mode=attention_mode,
            num_sublanes=num_sublanes,
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
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    num_sublanes: int = _DEFAULT_NUM_SUBLANES,
) -> tuple[int, int]:
    """Derive decode and mixed tile sizes within each kernel's VMEM limit.

    The decode kernel holds window_size (num_spec_tokens + 1) state checkpoints
    per slot and compiles with decode_vmem_limit_bytes; the prefill/mixed kernel
    compiles with window_size = 1 under mixed_vmem_limit_bytes. The returned
    mixed tile size is clipped to batch_size and rounded down to a power of two.

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
      mixed_tile_size: Explicit prefill/mixed chunk size override, if provided
        (clipped to batch_size and rounded down to a power of two).
      attention_mode: Attention mode (GDN or KDA).
      num_sublanes: Number of sublanes for 2D TPU vector layout alignment.

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
            attention_mode=attention_mode,
            num_sublanes=num_sublanes,
        )
        decode_tile_size = min(decode_tile_target(active_num_seqs, n_v), vmem_fit)

    if mixed_tile_size is None or mixed_tile_size <= 0:
        # Size C from the token bucket: a static (B, N, W) shape cannot tell a
        # decode step from a prefill, and an empty PER_SEQ launch costs the
        # same at any C.
        mixed_tile_size = calculate_mixed_tile_size(
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
            attention_mode=attention_mode,
            num_sublanes=num_sublanes,
        )

    # Guarantee strictly positive tile sizes (>= 1) for Pallas grid compilation,
    # keeping mixed_tile_size a power of two so triangular_block_size divides it.
    decode_tile_size = max(1, min(decode_tile_size, padded_batch_size))
    mixed_tile_size = clip_mixed_tile_size(mixed_tile_size, batch_size)
    return decode_tile_size, mixed_tile_size
