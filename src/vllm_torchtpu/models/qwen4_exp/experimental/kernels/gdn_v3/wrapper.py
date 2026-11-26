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
# Provenance: model-owned copy of src/vllm_torchtpu/kernels/gdn/v3/wrapper.py
#   (reference at 33ee547ba947, Qwen3.8 changes through
#   acc624492a26eb9b83768fe2c75de1f09ac5454a).
#   Owned by models/qwen4_exp from this copy on.
#   Ported from main: #2438's power-of-two clip of the static (KDA) mixed tile.

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger

from ...kernels.gdn_v3 import (
    compute_conv1d,
    compute_gdn,
    compute_kda,
    config,
    memory_ref,
    metadata,
    tiling,
    vmem_ldst,
)
from ...kernels.gdn_v3.input_layout import (
    Conv1DInput,
)

logger = init_logger(__name__)

# Per-setup kernel options; the model plan sets them with `configure`.
# TPU_GDN_CHUNK_PAIR: PER_SEQ chunked GDN tiles hold two chunks of a sequence.
_OPTIONS = {"TPU_GDN_CHUNK_PAIR": False}


def configure(**options) -> None:
    """Sets this module's kernel options (keys of `_OPTIONS`)."""
    unknown = set(options) - set(_OPTIONS)
    if unknown:
        raise ValueError(f"unknown GDN kernel options: {sorted(unknown)}")
    _OPTIONS.update(options)


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size]
    b_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    conv_state_in_slot_ref: jax.Ref,  # [seq, 1, prev_kernel_size, 1, dim_size]
    recurrent_in_slot_ref: jax.Ref,  # [seq, 1, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: jax.Ref,  # [seq, chunk, num_v_heads, v_head]
    conv_state_out_slot_ref: jax.Ref,  # [seq, window, prev_kernel_size, 1, dim]
    recurrent_out_slot_ref: jax.Ref,  # [seq, window, num_v_heads, kq, v]
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array | None = None,
):
    """Orchestrates computation of Conv1D and GDN for a single tile.

    This kernel acts as a facade adhering to strict separation of concerns. It
    operates VMEM reference without knowledge on DMA logic. Furthermore, the
    kernel invokes vmem_ldst to pre-processes data needed for compute and
    invokes compute_conv1d and compute_gdn for actual compute.

    Args:
        qkv_slot_ref: qkv VMEM ref that stores data loaded from HBM.
        b_slot_ref: b VMEM ref that stores data loaded from HBM.
        a_slot_ref: a VMEM ref that stores data loaded from HBM.
        conv_state_in_slot_ref: Convolution state VMEM ref loaded from HBM.
        recurrent_in_slot_ref: Recurrent state VMEM ref loaded from HBM.
        out_slot_ref: Output VMEM ref that will be used for VMEM to HBM write.
        conv_state_out_slot_ref: Convolution state VMEM ref written to HBM.
        recurrent_out_slot_ref: Recurrent state VMEM ref written to HBM.
        metadata_ref: Metadata reference containing grid and sequence mappings.
        weights_ref: Weight references for Conv1D and GDN in VMEM.
        carry_conv_scratch_ref: Optional VMEM scratch reference for inter-tile
            convolution carry.
        carry_recurrent_scratch_ref: Optional VMEM scratch reference for inter-tile
            recurrent state carry.
        cfg: GDN configuration object.
        p_id: Optional explicit tile index. Communication-fused outer kernels
            pass their global compute-tile index because their surrounding
            pipeline owns the Pallas program id.
    """

    if p_id is None:
        p_id = pl.program_id(0)

    # Prepare states.
    real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=conv_state_in_slot_ref,
        recurrent_in_slot_ref=recurrent_in_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        cfg=cfg,
    )
    commit_sizes = (
        vmem_ldst.load_commit_sizes(metadata_ref, p_id, real_sizes, cfg)
        if cfg.single_checkpoint
        else None
    )

    # Step 1: Conv1D.
    # NOTE: Conv1D requires performing sliding window where inputs are slided
    # across rows. If typical 2D layout was used, multiple rows are stored in a
    # single register which necessitate costly shuffling for every sliding.
    # Therefore, it is extremely important to leverage compact layout that
    # ensures 1 register only stores data from 1 row.
    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

    # Prepare conv1d weights.
    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    conv_bias = None
    if weights_ref.conv.bias is not None:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

    qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
        real_sizes=real_sizes,
        lhs=qkv_in_compact,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        cfg=cfg,
        commit_sizes=commit_sizes,
    )

    if cfg.state_plan is None:
        conv_state_out_slot_ref[...] = new_conv_state
    else:
        # External source tiles: encode each window position's checkpoint
        # through the region's typed view; copy_out pushes the raw bytes
        # of the valid checkpoints back to the source.
        for idx in range(cfg.seq_tile_size):
            for t in range(cfg.num_state_checkpoints):
                args = (
                    conv_state_out_slot_ref.at[idx, t],
                    cfg.state_plan.conv,
                    new_conv_state[idx, t],
                )
                if carry_conv_scratch_ref is None:
                    vmem_ldst.store_state_region(*args)
                else:
                    # copy_out only DMAs the tile that ends a sequence.
                    pl.when(metadata_ref.p_id_is_last_tile[p_id, idx])(
                        functools.partial(vmem_ldst.store_state_region, *args)
                    )
    if carry_conv_scratch_ref is not None:
        # The next tile resumes from the state after this tile's last token,
        # which is the final checkpoint.
        carry_conv_scratch_ref[...] = new_conv_state[:, -1]

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    # Step 2: GDN.

    # Prepare gdn weights.
    assert weights_ref.gdn.dt_bias is not None
    if cfg.is_kda:
        # KDA reshapes these per (head, channel) rather than padding out to
        # a lane multiple over heads.
        a_log = weights_ref.gdn.a_log[...]
        dt_bias = weights_ref.gdn.dt_bias[...]
    else:
        padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
        a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
        dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

    # NOTE: Ideally, we want to move this branching logic into gdn.py. However,
    # load_activation_as_compact and load_activation_as_large leverages vmem ldst.
    # Passing refs into gdn.py breaks strict separation of concerns.
    if cfg.use_recurrent:
        recurrent_fn = (
            compute_kda.recurrent_kda if cfg.is_kda else compute_gdn.recurrent_gdn
        )
        q_compact, k_compact, v_compact, b_compact, a_compact = (
            vmem_ldst.load_activation_as_compact(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfg,
            )
        )

        out, new_recurrent_state = recurrent_fn(
            q_compact=q_compact,
            k_compact=k_compact,
            v_compact=v_compact,
            b_compact=b_compact,
            a_compact=a_compact,
            state_prev=prev_recurrent,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
            commit_sizes=commit_sizes,
        )

    else:
        chunked_fn = compute_kda.chunked_kda if cfg.is_kda else compute_gdn.chunked_gdn
        q_large, k_large, v_large, b_large, a_large = (
            vmem_ldst.load_activation_as_large(
                qkv_vreg=qkv_out_compact,
                qkv_vmem_ref=qkv_slot_ref,
                b_vmem_ref=b_slot_ref,
                a_vmem_ref=a_slot_ref,
                cfgs=cfg,
            )
        )

        out, new_recurrent_state = chunked_fn(
            q_large=q_large,
            k_large=k_large,
            v_large=v_large,
            b_large=b_large,
            a_large=a_large,
            state_prev=prev_recurrent,
            a_log=a_log,
            dt_bias=dt_bias,
            cfg=cfg,
            real_sizes=real_sizes,
        )

    # Store output and recurrent to vmem.
    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
    if cfg.state_plan is None:
        recurrent_out_slot_ref[...] = new_recurrent_state.astype(
            recurrent_out_slot_ref.dtype
        )
    else:
        for idx in range(cfg.seq_tile_size):
            for t in range(cfg.num_state_checkpoints):
                args = (
                    recurrent_out_slot_ref.at[idx, t],
                    cfg.state_plan.recurrent,
                    new_recurrent_state[idx, t],
                )
                if carry_recurrent_scratch_ref is None:
                    vmem_ldst.store_state_region(*args)
                else:
                    pl.when(metadata_ref.p_id_is_last_tile[p_id, idx])(
                        functools.partial(vmem_ldst.store_state_region, *args)
                    )

    if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
):
    """Setup memory allocations and emit pipeline for running inner_kernel."""

    if cfg.state_plan is not None:
        # Both state regions stream from/to the single external source ref,
        # which rides the conv-state input/output slots.
        recurrent_state_ref = conv_state_ref
        recurrent_state_out_ref = conv_state_out_ref

    (
        qkv_alloc,
        b_alloc,
        a_alloc,
        conv_in_alloc,
        recurrent_in_alloc,
        out_alloc,
        conv_out_alloc,
        recurrent_out_alloc,
    ) = memory_ref.create_allocs(
        metadata_ref=metadata_ref,
        qkv_ref=qkv_ref,
        b_ref=b_ref,
        a_ref=a_ref,
        out_ref=out_ref,
        conv_state_ref=conv_state_ref,
        recurrent_state_ref=recurrent_state_ref,
        cfg=cfg,
    )

    num_tiles = metadata_ref.num_tiles[...]

    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(
            inner_kernel,
            cfg=cfg,
        ),
        grid=(num_tiles,),
        in_specs=(
            qkv_alloc.spec,
            b_alloc.spec,
            a_alloc.spec,
            conv_in_alloc.spec,
            recurrent_in_alloc.spec,
        ),
        out_specs=(
            out_alloc.spec,
            conv_out_alloc.spec,
            recurrent_out_alloc.spec,
        ),
    )

    @pl.with_scoped(
        allocations=(
            qkv_alloc,
            b_alloc,
            a_alloc,
            conv_in_alloc,
            recurrent_in_alloc,
            out_alloc,
            conv_out_alloc,
            recurrent_out_alloc,
        ),
    )
    def _run(allocations):
        pipeline_func(
            qkv_ref,
            b_ref,
            a_ref,
            conv_state_ref,
            recurrent_state_ref,
            out_ref,
            conv_state_out_ref,
            recurrent_state_out_ref,
            scratches=(
                metadata_ref,
                weights_ref,
                carry_conv_scratch_ref,
                carry_recurrent_scratch_ref,
            ),
            allocations=allocations,
        )

    _run()


def kernel_metadata(
    cfg: config.GDNConfig,
    *,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    distribution: jax.Array,
    ckpt_indices: jax.Array | None,
    per_seq_inputs: tuple[jax.Array, ...] | None = None,
    prefill_ckpt: tuple[jax.Array, jax.Array] | None = None,
    commit_lens: jax.Array | None = None,
) -> memory_ref.MetadataRef:
    """The scalar-prefetch metadata of one `fused_conv1d_gdn` kernel call."""
    if cfg.mode != config.GDNMode.PER_SEQ:
        return metadata.compute_batched_seq_metadata(
            cfg=cfg,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            end_seq=distribution[0],
            ckpt_indices=ckpt_indices,
            commit_lens=commit_lens if cfg.single_checkpoint else None,
        )
    if per_seq_inputs is not None:
        ps_qsl, ps_seq_lens, ps_state, ps_read, ps_ckpt = per_seq_inputs
        return metadata.compute_per_seq_metadata(
            cfg=cfg,
            seq_lens=ps_seq_lens,
            query_start_loc=ps_qsl,
            state_indices=ps_state,
            read_offsets=ps_read,
            start_seq=distribution[0],
            end_seq=distribution[-1],
            ckpt_indices=ps_ckpt,
            prerolled=True,
            prefill_ckpt=prefill_ckpt,
        )
    return metadata.compute_per_seq_metadata(
        cfg=cfg,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        read_offsets=read_offsets,
        start_seq=distribution[0],
        end_seq=distribution[-1],
        ckpt_indices=ckpt_indices,
        prefill_ckpt=prefill_ckpt,
    )


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state", "state_source"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "conv_weight_kmajor",
        "decode_tile_size",
        "mixed_tile_size",
        "batched_only",
        "prefill_only",
        "zero_initialize_out",
        "compute_precision",
        "state_plan",
        "attention_mode",
        "gate_lower_bound",
        "prefill_ckpt_alignment",
        "single_checkpoint",
        "interpret",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array | Conv1DInput,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]; [batch_size, n_v * d_k] for KDA
    conv_state: jax.Array | None,  # [num_seqs + 1, kernel_size - 1, dim_size]
    recurrent_state: jax.Array | None,  # [num_seqs + 1, nv, dk, dv]
    conv_weight: jax.Array,  # [kernel_size - 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v] for GDN, [n_v, d_k] for KDA
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [num_seqs]
    read_offsets: jax.Array | None = None,  # [num_seqs]
    ckpt_indices: jax.Array | None = None,  # [num_seqs, num_spec_tokens + 1]
    commit_lens: jax.Array | None = None,  # [num_seqs]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    gate_lower_bound: float | None = None,
    state_source: jax.Array | None = None,
    state_plan: config.StateSourcePlan | None = None,
    num_spec_tokens: int = 0,
    conv_weight_kmajor: bool = False,
    single_checkpoint: bool = False,
    batched_only: bool = False,
    prefill_only: bool = False,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    # TODO(kyuyeunk): Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
    per_seq_inputs: tuple[jax.Array, ...] | None = None,
    prefill_ckpt: tuple[jax.Array, jax.Array] | None = None,
    prefill_ckpt_alignment: int = 0,
    interpret: Any = False,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array] | tuple[jax.Array, jax.Array]:
    """Perform conv1d and gdn in a single fused kernel.

    Args:
        qkv: Mixed query, key, value input tensor of shape [batch_size, dim_size],
            where `dim_size = n_kq * d_k * 2 + n_v * d_v`, or a prepared
            Conv1DInput with FP32 values and the original output dtype.
        b: b tensor (for beta) of shape [batch_size, n_v].
        a: a tensor (for g) of shape [batch_size, n_v].
        conv_state: Convolution state cache tensor of shape [num_seqs + 1,
            kernel_size - 1, dim_size] containing the last (kernel_size - 1) tokens
            from the last sequence invocation. The first slot is a null block used for
            padded or invalid tokens. It may contain garbage data if it is a first
            invocation of a sequence. None when `state_source` carries the states.
        recurrent_state: Recurrent state cache tensor of shape [num_seqs + 1, n_v,
            d_k, d_v]. The first slot is a null block used for padded or invalid
            tokens. It may contain garbage data if it is a first invocation of a
            sequence. None when `state_source` carries the states.
        conv_weight: Convolution weight tensor of shape [kernel_size - 1, dim_size].
        conv_bias: Optional convolution bias tensor of shape [dim_size].
        a_log: a_log tensor of shape [n_v].
        dt_bias: dt_bias tensor of shape [n_v]; [n_v, d_k] under KDA, whose
            gate is per-channel. Kept 2D rather than flattened because the
            KDA path reshapes it to broadcast over (head, channel), and
            Mosaic rejects that shape cast from a flat vector
            ("infer-vector-layout: unsupported shape cast").
        query_start_loc: Start locations of sequences of shape [num_seqs + 1].
        state_indices: Indices mapping sequences to state cache slots of shape
            [num_seqs]. With speculative decoding these are the *base* slots
            of per-request groups of `num_spec_tokens + 1` consecutive slots.
        distribution: Tensor of shape [3] int32 — [decode_end, prefill_end,
            mixed_end]. With `num_spec_tokens > 0`, the first segment holds
            speculative verify windows of up to `num_spec_tokens + 1` tokens
            rather than 1-token decodes.
        seq_lens: Sequence lengths for each sequence of shape [num_seqs].
        read_offsets: Optional [num_seqs] int32 — per-sequence state read
            offset (num_accepted - 1 from the last verify step). Windowed
            sequences read their initial state from
            `state_indices[s] + read_offsets[s]` and write one checkpoint
            per window position to `state_indices[s] + t`. Required when
            `num_spec_tokens > 0`.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key/query dimension.
        d_v: Value dimension.
        kernel_size: Convolution kernel size.
        attention_mode: `GDN` for the per-head scalar gate, `KDA` for the
            per-channel one. Selects which compute path runs.
        gate_lower_bound: KDA only, and must be negative. `None` gives the
            unbounded gate `-exp(A_log) * softplus(g + dt_bias)`
            (Kimi-Linear-48B); a float gives the bounded
            `gate_lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` (every K3
            variant, at -5.0). Different functions of the same weights, so a
            mismatch with the model's KDA config silently changes the decay.
        state_source: Optional indexed external state source (e.g. the unified
            KV pool) replacing the dense state tensors: both state regions are
            streamed between it and the kernel pipeline per `state_plan`, with
            `state_indices` addressing source block windows. Donated and
            returned in place of the dense states.
        state_plan: Static copy-plan describing the source's state regions.
            Must be set iff `state_source` is set.
        num_spec_tokens: Number of speculative draft tokens (0 gives the
            plain 1-token-per-sequence decode path).
        batched_only: Run only the leading decode/verify segment.
        prefill_only: Run only PER_SEQ, leaving decode/verify checkpoints
            untouched. Mutually exclusive with batched_only.
        zero_initialize_out: Whether to zero-initialize the output buffer before
            executing non-batched sequences.
        compute_precision: Computation precision dtype.
        decode_tile_size: Tile size along sequence dimension for decode sequences.
        per_seq_inputs: Optional `metadata.roll_per_seq_inputs(query_start_loc,
            seq_lens, state_indices, read_offsets, distribution[0],
            ckpt_indices)` (read offsets and checkpoint ids as int32), rolled
            once per step by the caller instead of once per layer here.
            Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.
        mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
            sequences. Ignored when TPU_ENABLE_GDN_DYNAMIC_TILING is set.
        prefill_ckpt: Optional `(offsets, blocks)`, int32 [num_seqs] each:
            internal prefill checkpoints (VLLM_TPU_MAMBA_TAIL_FOLD). A PER_SEQ
            sequence also writes its state after its first `offsets[s]` query
            tokens to checkpoint block `blocks[s]`; 0 means none. See
            `metadata.compute_per_seq_metadata`.
        prefill_ckpt_alignment: Every nonzero checkpoint offset is a multiple
            of this; the PER_SEQ tile must divide it.

    Returns:
        (new_conv_state, new_recurrent_state): Updated convolution state cache and
            recurrent state cache tensors. With a state source: the updated
            source (aliased in place) instead of the state pair.
        out: Fused output tensor.
    """
    # Conv1D consumes FP32 while outputs retain the projection activation dtype.
    prepared_qkv = isinstance(qkv, Conv1DInput)
    if isinstance(qkv, Conv1DInput):
        act_out_dtype = qkv.output_dtype
        qkv = qkv.values
        if qkv.ndim != 3 or qkv.shape[1] != 1 or qkv.dtype != jnp.float32:
            raise ValueError("Conv1DInput requires FP32 [tokens, 1, features]")
    else:
        act_out_dtype = qkv.dtype
    pooled = state_plan is not None
    if pooled:
        assert conv_state is None and recurrent_state is None
        assert state_source is not None
        # Checkpoints are individually addressable source blocks; without
        # the index array they would all alias onto the slot's own block.
        assert num_spec_tokens == 0 or ckpt_indices is not None, (
            "num_spec_tokens > 0 requires ckpt_indices"
        )
    else:
        assert state_source is None
        assert conv_state is not None and recurrent_state is not None
        conv_out_dtype = conv_state.dtype
        recurrent_out_dtype = recurrent_state.dtype
        conv_state = conv_state.astype(jnp.float32)

    # Step 1: Validate inputs.
    is_kda = attention_mode == config.AttentionMode.KDA

    if not prepared_qkv:
        qkv = qkv.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)
    if gate_lower_bound is not None:
        if not is_kda:
            raise ValueError(
                "gate_lower_bound only applies to KDA; got "
                f"attention_mode={attention_mode}."
            )
        # The chunked solve needs every causal exponent g_r - g_t to stay
        # non-positive, which `gate_lower_bound * sigmoid(.)` only satisfies
        # for a negative bound.
        if gate_lower_bound >= 0:
            raise ValueError(
                "gate_lower_bound must be negative (it bounds a log-decay); "
                f"got {gate_lower_bound}. A non-negative bound makes the gate "
                "increase along the chunk, which this kernel does not "
                "support."
            )

    num_seqs = state_indices.size
    batch_size, dim = qkv.shape[0], qkv.shape[-1]
    if conv_weight_kmajor:
        # Already float32 in the kernel's [kernel_size, 1, dim] order.
        assert conv_weight.shape == (kernel_size, 1, dim)
        assert conv_weight.dtype == jnp.float32
    else:
        assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim,)
    assert query_start_loc.shape == (num_seqs + 1,)
    assert state_indices.shape == (num_seqs,)
    assert distribution.shape == (3,)
    if num_spec_tokens > 0 and not single_checkpoint:
        assert read_offsets is not None, (
            "read_offsets is required when num_spec_tokens > 0"
        )
    if read_offsets is None:
        read_offsets = jnp.zeros((num_seqs,), dtype=jnp.int32)
    assert read_offsets.shape == (num_seqs,)
    read_offsets = read_offsets.astype(jnp.int32)
    if ckpt_indices is not None:
        # One source block per checkpoint (vLLM's ssm_state_indices scheme).
        # Under single_checkpoint=True, allow 1 (in-place single slot), 2
        # (2-slot ping-pong), or window_size columns.
        allowed_ckpts = (
            tuple(dict.fromkeys((1, 2, num_spec_tokens + 1)))
            if single_checkpoint
            else (num_spec_tokens + 1,)
        )
        if (
            ckpt_indices.ndim != 2
            or ckpt_indices.shape[0] != num_seqs
            or ckpt_indices.shape[1] not in allowed_ckpts
        ):
            raise ValueError(
                f"ckpt_indices must have shape ({num_seqs}, {allowed_ckpts}); got "
                f"{ckpt_indices.shape}."
            )
        ckpt_indices = ckpt_indices.astype(jnp.int32)
    if commit_lens is not None:
        if not single_checkpoint:
            raise ValueError("commit_lens requires single_checkpoint=True.")
        if prefill_only:
            raise ValueError("commit_lens is not supported with prefill_only=True.")
        if commit_lens.shape != (num_seqs,):
            raise ValueError(
                f"commit_lens must have shape ({num_seqs},); got {commit_lens.shape}."
            )
        commit_lens = commit_lens.astype(jnp.int32)
    act_in_dtype = jnp.dtype(jnp.float32)
    assert a.dtype == b.dtype == act_in_dtype

    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    if single_checkpoint:
        decode_tile_size = max(1, min(decode_tile_size, num_seqs, batch_size))
    else:
        decode_tile_size = min(decode_tile_size, batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes
    window_size = num_spec_tokens + 1
    decode_vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(
        window_size, single_checkpoint=single_checkpoint
    )

    # KDA stays on static tiles until the VMEM estimate models its gate.
    if not is_kda:
        if pooled and single_checkpoint:
            assert state_plan is not None
            rec_state_dtype = jnp.dtype(
                jnp.bfloat16
                if state_plan.recurrent.view_dtype == jnp.uint32
                else state_plan.recurrent.view_dtype
            )
        else:
            rec_state_dtype = jnp.dtype(
                jnp.float32 if recurrent_state is None else recurrent_state.dtype
            )
        decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
            batch_size=batch_size,
            num_seqs=num_seqs,
            padded_batch_size=padded_batch_size,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim if conv_state is None else conv_state.shape[-1],
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=rec_state_dtype,
            num_lanes=num_lanes,
            decode_vmem_limit_bytes=decode_vmem_limit_bytes,
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
            window_size=window_size,
            single_checkpoint=single_checkpoint,
        )
        logger.info_once(
            "GDN dynamic tiling: num_seqs=%d batch=%d window=%d s1=%s -> T=%d C=%d",
            num_seqs,
            batch_size,
            window_size,
            single_checkpoint,
            decode_tile_size,
            mixed_tile_size,
        )
    else:
        # Round the static clip down to a power of two so a small
        # non-power-of-two token bucket (e.g. TPU_TOKEN_BUCKET_EXTRA or tiny
        # max_num_batched_tokens) stays divisible by triangular_block_size.
        mixed_tile_size = tiling.clip_mixed_tile_size(mixed_tile_size, batch_size)
        # Each sequence holds 1 input state slot plus output checkpoint slots
        # per buffer in VMEM (`1` under `single_checkpoint`, else `window`).
        window = num_spec_tokens + 1
        num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
        state_positions = 1 + (1 if single_checkpoint else window)
        recurrent_bytes_per_seq = state_positions * n_v * d_k * d_v * 4
        bytes_per_seq = (
            # Recurrent checkpoints (fp32) — the dominant term.
            recurrent_bytes_per_seq
            # Conv checkpoints (fp32).
            + state_positions * (kernel_size - 1) * dim * 4
            # qkv (fp32), b/a (fp32), out (act_out).
            + window * (dim * 4 + 2 * aligned_num_v_heads * 4 + n_v * d_v * 2)
        )
        spec_tile_budget = (decode_vmem_limit_bytes // 2) // num_buffers
        decode_tile_cap = (
            min(decode_tile_size, num_seqs, spec_tile_budget // bytes_per_seq)
            if single_checkpoint
            else min(decode_tile_size, spec_tile_budget // bytes_per_seq)
        )
        decode_tile_size = max(1, decode_tile_cap)

    if num_spec_tokens > 0 and is_kda:
        # KDA verify keeps one sequence per tile until multi-sequence KDA
        # verify tiles are sized by a KDA-aware VMEM estimate.
        decode_tile_size = 1

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    # KDA's gate carries d_k channels per value head, so it pads out to a
    # lane multiple of n_v * d_k rather than of n_v.
    gate_dim = n_v * d_k if is_kda else n_v
    gate_padding_size = (pl.cdiv(gate_dim, num_lanes) * num_lanes) - gate_dim
    qkv_padding = ((0, batch_padding_size),) + ((0, 0),) * (qkv.ndim - 1)
    qkv = jnp.pad(qkv, qkv_padding)
    qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    # Step 3: States and weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
    if not pooled:
        conv_state_shape = cast(jax.Array, conv_state).shape
        conv_state = cast(jax.Array, conv_state).reshape(-1, kernel_size - 1, 1, dim)
    if not conv_weight_kmajor:
        conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
    conv_bias = conv_bias.astype(jnp.float32) if conv_bias is not None else None

    # Step 4: Wrap inputs for the kernel.
    conv_weights = memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias)
    gdn_weights = memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
    weights = memory_ref.WeightRefs(conv=conv_weights, gdn=gdn_weights)

    # Step 5: Create specs.
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    def call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array | None,
        in_act: jax.Array | None,
        mode: config.GDNMode,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        # With a state source, it rides the conv-state slot and the dense
        # recurrent-state input is absent.
        chunk_pair = False
        if mode == config.GDNMode.PER_SEQ:
            tile_size = mixed_tile_size
            # Prefill / mixed sequences keep a single state checkpoint.
            window_size = 1
            # Two chunks per grid step (chunked GDN only; chunk 1 is the
            # token-recurrent scan).
            if (
                _OPTIONS["TPU_GDN_CHUNK_PAIR"]
                and not is_kda
                and tile_size > 1
                and 2 * tile_size <= padded_batch_size
            ):
                tile_size, chunk_pair = 2 * tile_size, True
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1

        cfg = config.GDNConfig(
            mode=mode,
            batch_size=padded_batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            window_size=window_size,
            single_checkpoint=(single_checkpoint and mode != config.GDNMode.PER_SEQ),
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            attention_mode=attention_mode,
            gate_lower_bound=gate_lower_bound,
            chunk_pair=chunk_pair,
            dtypes=config.Dtypes(
                act_in=act_in_dtype,
                act_out=act_out_dtype,
                compute=compute_precision,
                recurrent_state=(
                    jnp.float32.dtype
                    if pooled
                    else cast(jax.Array, in_recurrent_state).dtype
                ),
                conv_state=(jnp.float32.dtype if pooled else in_conv_state.dtype),
            ),
            state_plan=state_plan,
        )

        # Step 6: Metadata preprocessing. Runs once per layer: each layer's
        # op is its own program, so XLA does not share it across layers
        # (pass `per_seq_inputs` to hoist the PER_SEQ roll).
        ckpt = prefill_ckpt if mode == config.GDNMode.PER_SEQ else None
        if ckpt is not None:
            if prefill_ckpt_alignment <= 0:
                raise ValueError("prefill_ckpt needs a prefill_ckpt_alignment")
            # A checkpoint needs a query longer than one alignment unit, so
            # smaller buckets (and their odd tile sizes) never carry one.
            if padded_batch_size <= prefill_ckpt_alignment:
                ckpt = None
            elif prefill_ckpt_alignment % cfg.chunk_size:
                raise ValueError(
                    f"GDN prefill tile {cfg.chunk_size} does not divide the "
                    f"checkpoint alignment {prefill_ckpt_alignment}"
                )
        metadata_obj = kernel_metadata(
            cfg,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            distribution=distribution,
            ckpt_indices=ckpt_indices,
            per_seq_inputs=per_seq_inputs,
            prefill_ckpt=ckpt,
            commit_lens=commit_lens if cfg.single_checkpoint else None,
        )

        metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)

        # Step 7: Handle case where write needs to be done in existing out.
        # Flat aliasing indices account for the absent recurrent-state
        # input in source mode (None contributes no leaves).
        in_out_spec = None
        num_state_inputs = 1 if pooled else 2
        input_output_aliases = {len(metadata_obj) + 3: 1}
        if not pooled:
            input_output_aliases[len(metadata_obj) + 4] = 2
        out_shape = cfg.get_out_shape()

        if in_act is None and zero_initialize_out:
            in_act = jnp.zeros_like(out_shape)
        if in_act is not None:
            out_shape = in_act
            in_out_spec = hbm_spec
            input_output_aliases[len(metadata_obj) + 3 + num_state_inputs] = 0

        vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(
            window_size, single_checkpoint=cfg.single_checkpoint
        )

        return pl.pallas_call(
            functools.partial(outer_kernel, cfg=cfg),
            out_shape=(out_shape, in_conv_state, in_recurrent_state),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                None if pooled else hbm_spec,
                in_out_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, hbm_spec, None if pooled else hbm_spec),
            scratch_shapes=cfg.get_scratch_shape_dict(),
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            interpret=interpret,
            name=cfg.get_kernel_name(),
            # pyrefly: ignore[bad-argument-type]
            metadata=cfg.get_metadata(),
        )(
            metadata_obj,
            qkv,
            b,
            a,
            in_conv_state,
            in_recurrent_state,
            in_act,
            weights,
        )

    if batched_only and prefill_only:
        raise ValueError("batched_only and prefill_only are mutually exclusive")

    if pooled:
        out_act, out_source = None, state_source
        if not prefill_only:
            out_act, out_source, _ = call_kernel(
                # pyrefly: ignore[bad-argument-type]
                state_source,
                None,
                None,
                config.GDNMode.BATCHED,
            )
        if not batched_only:
            out_act, out_source, _ = call_kernel(
                # pyrefly: ignore[bad-argument-type]
                out_source,
                None,
                out_act,
                config.GDNMode.PER_SEQ,
            )
        # pyrefly: ignore[missing-attribute]
        out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
        # pyrefly: ignore[bad-return]
        return out_source, out_act

    # The first segment holds verify windows of up to `num_spec_tokens + 1`
    # tokens, or plain 1-token decodes without speculative decoding.
    out_act, out_conv_state, out_recurrent_state = None, conv_state, recurrent_state
    if not prefill_only:
        out_act, out_conv_state, out_recurrent_state = call_kernel(
            # pyrefly: ignore[bad-argument-type]
            conv_state,
            recurrent_state,
            None,
            config.GDNMode.BATCHED,
        )
    if not batched_only:
        out_act, out_conv_state, out_recurrent_state = call_kernel(
            # pyrefly: ignore[bad-argument-type]
            out_conv_state,
            out_recurrent_state,
            out_act,
            config.GDNMode.PER_SEQ,
        )

    # pyrefly: ignore[missing-attribute]
    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    # pyrefly: ignore[missing-attribute, unbound-name]
    out_conv_state = out_conv_state.astype(conv_out_dtype)
    # pyrefly: ignore[unbound-name]
    out_conv_state = out_conv_state.reshape(conv_state_shape)
    # pyrefly: ignore[missing-attribute, unbound-name]
    out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

    return (out_conv_state, out_recurrent_state), out_act


def _multi_layer_replay_inner_kernel(
    qkv_slot_ref: jax.Ref,
    b_slot_ref: jax.Ref,
    a_slot_ref: jax.Ref,
    conv_state_in_slot_ref: jax.Ref,
    recurrent_in_slot_ref: jax.Ref,
    out_slot_ref: jax.Ref,
    conv_state_out_slot_ref: jax.Ref,
    recurrent_out_slot_ref: jax.Ref,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    *,
    cfg: config.GDNConfig,
) -> None:
    """Single-launch multi-layer 2-slot ping-pong batched replay inner body.

    Weights are packed with (8, num_lanes)-aligned inner dimensions in VMEM:
      - weights_ref.conv.weight: float32[L, k_pad, dim_pad]
      - weights_ref.conv.bias:   None or float32[L, 8, dim_pad]
      - weights_ref.gdn.a_log:   float32[L, 8, aligned_num_v_heads]
                                 (row 0 = padded a_log, row 1 = padded dt_bias)
    """
    del out_slot_ref
    p_id = pl.program_id(0)
    l_id = metadata_ref.p_id_to_layer[p_id]

    real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=conv_state_in_slot_ref,
        recurrent_in_slot_ref=recurrent_in_slot_ref,
        carry_conv_scratch_ref=None,
        carry_recurrent_scratch_ref=None,
        cfg=cfg,
    )
    commit_sizes = vmem_ldst.load_commit_sizes(metadata_ref, p_id, real_sizes, cfg)

    qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
    qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

    cw_tile = weights_ref.conv.weight[l_id]
    conv_weight = cw_tile[: cfg.kernel_size, : cfg.dim_size][:, None, :]
    conv_bias = (
        None
        if weights_ref.conv.bias is None
        else weights_ref.conv.bias[l_id][0, : cfg.dim_size]
    )

    qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
        real_sizes=real_sizes,
        lhs=qkv_in_compact,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        cfg=cfg,
        commit_sizes=commit_sizes,
    )
    assert cfg.state_plan is not None
    for idx in range(cfg.seq_tile_size):
        vmem_ldst.store_state_region(
            conv_state_out_slot_ref.at[idx, 0],
            cfg.state_plan.conv,
            new_conv_state[idx, 0],
        )

    qkv_out_compact = jax.nn.silu(qkv_out_compact)

    ab_tile = weights_ref.gdn.a_log[l_id]
    a_log = ab_tile[0]
    dt_bias = ab_tile[1]

    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )
    _, new_recurrent_state = compute_gdn.recurrent_gdn(
        q_compact=q_compact,
        k_compact=k_compact,
        v_compact=v_compact,
        b_compact=b_compact,
        a_compact=a_compact,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
        commit_sizes=commit_sizes,
    )
    for idx in range(cfg.seq_tile_size):
        vmem_ldst.store_state_region(
            recurrent_out_slot_ref.at[idx, 0],
            cfg.state_plan.recurrent,
            new_recurrent_state[idx, 0],
        )


def _multi_layer_replay_outer_kernel(
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    source_ref: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    out_ref: jax.Array,
    source_out_ref: jax.Array,
    *,
    cfg: config.GDNConfig,
) -> None:
    (
        qkv_alloc,
        b_alloc,
        a_alloc,
        conv_in_alloc,
        recurrent_in_alloc,
        out_alloc,
        conv_out_alloc,
        recurrent_out_alloc,
    ) = memory_ref.create_allocs(
        metadata_ref=metadata_ref,
        qkv_ref=qkv_ref,
        b_ref=b_ref,
        a_ref=a_ref,
        out_ref=out_ref,
        conv_state_ref=source_ref,
        recurrent_state_ref=source_ref,
        cfg=cfg,
    )
    num_tiles = metadata_ref.num_tiles[...]
    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(_multi_layer_replay_inner_kernel, cfg=cfg),
        grid=(num_tiles,),
        in_specs=(
            qkv_alloc.spec,
            b_alloc.spec,
            a_alloc.spec,
            conv_in_alloc.spec,
            recurrent_in_alloc.spec,
        ),
        out_specs=(
            out_alloc.spec,
            conv_out_alloc.spec,
            recurrent_out_alloc.spec,
        ),
    )

    @pl.with_scoped(
        allocations=(
            qkv_alloc,
            b_alloc,
            a_alloc,
            conv_in_alloc,
            recurrent_in_alloc,
            out_alloc,
            conv_out_alloc,
            recurrent_out_alloc,
        ),
    )
    def _run(allocations):
        pipeline_func(
            qkv_ref,
            b_ref,
            a_ref,
            source_ref,
            source_ref,
            out_ref,
            source_out_ref,
            source_out_ref,
            scratches=(metadata_ref, weights_ref),
            allocations=allocations,
        )

    _run()


def run_multi_layer_batched_replay_pooled(
    qkv_3d: jax.Array,  # [L, win_rows, dim]
    b_3d: jax.Array,  # [L, win_rows, n_v]
    a_3d: jax.Array,  # [L, win_rows, n_v]
    state_source: jax.Array,
    conv_weight_stack: jax.Array,  # [L, K, 1, dim] or [L, dim, 1, K]
    conv_bias_stack: jax.Array | None,  # [L, dim] or None
    a_log_stack: jax.Array,  # [L, n_v]
    dt_bias_stack: jax.Array,  # [L, n_v]
    state_indices_stack: jax.Array,  # [L, max_seqs]
    read_offsets_stack: jax.Array,  # [L, max_seqs]
    ckpt_indices_stack: jax.Array,  # [L, max_seqs, 2]
    commit_lens_stack: jax.Array,  # [1, max_seqs] or [L, max_seqs]
    query_start_loc: jax.Array,  # [max_seqs + 1]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [max_seqs]
    *,
    state_plan: config.StateSourcePlan,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int,
    conv_weight_kmajor: bool = False,
    enable_dynamic_tiling: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int = 16,
    interpret: Any = False,
) -> jax.Array:
    """Single-launch L-layer 2-slot ping-pong batched replay over `state_source`.

    Compacts active rejected requests (`commit_lens > 0`) into contiguous tiles
    across all `L` GDN layers (e.g. 36 or 69) using 1D `jnp.nonzero` when
    `commit_lens_stack.shape[0] == 1` and skipping all output-activation DMAs
    via `skip_out_write=True`. When no request requires replay, `num_tiles == 0`
    so the kernel returns `state_source` unmodified.
    """
    num_layers, win_rows, dim = qkv_3d.shape
    if num_layers <= 0:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}.")
    max_seqs = seq_lens.size
    if state_indices_stack.shape != (num_layers, max_seqs):
        raise ValueError(
            f"state_indices_stack must have shape ({num_layers}, {max_seqs}); "
            f"got {state_indices_stack.shape}."
        )
    if ckpt_indices_stack.shape != (num_layers, max_seqs, 2):
        raise ValueError(
            f"ckpt_indices_stack must have shape ({num_layers}, {max_seqs}, 2); "
            f"got {ckpt_indices_stack.shape}."
        )
    if (
        commit_lens_stack.ndim != 2
        or commit_lens_stack.shape[0] not in (1, num_layers)
        or commit_lens_stack.shape[1] != max_seqs
    ):
        raise ValueError(
            f"commit_lens_stack must have shape (1|{num_layers}, {max_seqs}); "
            f"got {commit_lens_stack.shape}."
        )
    act_out_dtype = qkv_3d.dtype
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes
    nv_pad = aligned_num_v_heads - n_v
    window_size = num_spec_tokens + 1
    padded_win_rows = max(8, pl.cdiv(win_rows, 8) * 8)
    win_row_pad = padded_win_rows - win_rows

    # Pack weights with (8, num_lanes) inner tile alignment before sizing tiles
    # so the VMEM tile budget accounts for resident stacked weights.
    if conv_weight_kmajor:
        cw_3d = conv_weight_stack.astype(jnp.float32).reshape(
            num_layers, kernel_size, dim
        )
    else:
        cw_3d = (
            jnp.swapaxes(conv_weight_stack, 1, 3)
            .astype(jnp.float32)
            .reshape(num_layers, kernel_size, dim)
        )
    k_pad = pl.cdiv(kernel_size, 8) * 8 - kernel_size
    dim_pad = pl.cdiv(dim, num_lanes) * num_lanes - dim
    if k_pad > 0 or dim_pad > 0:
        cw_3d = jnp.pad(cw_3d, ((0, 0), (0, k_pad), (0, dim_pad)))

    if conv_bias_stack is not None:
        cb_3d = conv_bias_stack.astype(jnp.float32)[:, None, :]
        cb_3d = jnp.pad(cb_3d, ((0, 0), (0, 7), (0, dim_pad)))
    else:
        cb_3d = None

    alog_p = a_log_stack.astype(jnp.float32)
    dtb_p = dt_bias_stack.astype(jnp.float32)
    if nv_pad > 0:
        alog_p = jnp.pad(alog_p, ((0, 0), (0, nv_pad)))
        dtb_p = jnp.pad(dtb_p, ((0, 0), (0, nv_pad)))
    ab_2 = jnp.stack([alog_p, dtb_p], axis=1)
    ab_tile_stack = jnp.pad(ab_2, ((0, 0), (0, 6), (0, 0)))

    # Keep `dt_bias=None` so a single packed [L, 8, aligned_num_v_heads] array
    # holds both `a_log` (row 0) and `dt_bias` (row 1) without double VMEM cost.
    weights = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(weight=cw_3d, bias=cb_3d),
        gdn=memory_ref.GDNWeightsRef(a_log=ab_tile_stack, dt_bias=None),
    )

    weight_vmem_bytes = sum(
        int(x.size) * jnp.dtype(x.dtype).itemsize
        for x in jax.tree_util.tree_leaves(weights)
    )
    max_cap_bytes = int(0.92 * tpu_info.vmem_capacity_bytes)
    decode_vmem_limit_bytes = config.GDNConfig.get_vmem_limit_bytes(
        window_size, single_checkpoint=True
    )
    tile_vmem_budget_bytes = max(
        1, min(decode_vmem_limit_bytes, max_cap_bytes - weight_vmem_bytes)
    )
    vmem_limit_bytes = min(max_cap_bytes, tile_vmem_budget_bytes + weight_vmem_bytes)

    tile_sz = max(1, min(decode_tile_size, max_seqs, padded_win_rows))
    if enable_dynamic_tiling:
        rec_dt = (
            jnp.bfloat16
            if state_plan.recurrent.view_dtype == jnp.uint32
            else state_plan.recurrent.view_dtype
        )
        tile_sz, _ = tiling.get_tile_sizes(
            batch_size=padded_win_rows,
            num_seqs=max_seqs,
            padded_batch_size=padded_win_rows,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=kernel_size,
            conv_state_dim_size=dim,
            act_out_dtype=act_out_dtype,
            recurrent_state_dtype=jnp.dtype(rec_dt),
            num_lanes=num_lanes,
            decode_vmem_limit_bytes=tile_vmem_budget_bytes,
            mixed_vmem_limit_bytes=config.GDNConfig.get_vmem_limit_bytes(),
            window_size=window_size,
            single_checkpoint=True,
        )
    else:
        num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
        state_positions = 2
        bytes_per_seq = (
            state_positions * n_v * d_k * d_v * 4
            + state_positions * (kernel_size - 1) * dim * 4
            + window_size * (dim * 4 + 2 * aligned_num_v_heads * 4 + n_v * d_v * 2)
        )
        spec_tile_budget = (tile_vmem_budget_bytes // 2) // num_buffers
        tile_sz = max(1, min(tile_sz, max(1, spec_tile_budget // bytes_per_seq)))

    cfg = config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        batch_size=padded_win_rows,
        kernel_size=kernel_size,
        tile_size=tile_sz,
        window_size=window_size,
        single_checkpoint=True,
        skip_out_write=True,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        attention_mode=config.AttentionMode.GDN,
        gate_lower_bound=None,
        chunk_pair=False,
        dtypes=config.Dtypes(
            act_in=jnp.dtype(jnp.float32),
            act_out=act_out_dtype,
            compute=compute_precision,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        state_plan=state_plan,
    )

    qkv_f32 = qkv_3d.astype(jnp.float32)
    b_f32 = b_3d.astype(jnp.float32)
    a_f32 = a_3d.astype(jnp.float32)
    if win_row_pad > 0:
        qkv_f32 = jnp.pad(qkv_f32, ((0, 0), (0, win_row_pad), (0, 0)))
        b_f32 = jnp.pad(b_f32, ((0, 0), (0, win_row_pad), (0, 0)))
        a_f32 = jnp.pad(a_f32, ((0, 0), (0, win_row_pad), (0, 0)))
    if nv_pad > 0:
        b_f32 = jnp.pad(b_f32, ((0, 0), (0, 0), (0, nv_pad)))
        a_f32 = jnp.pad(a_f32, ((0, 0), (0, 0), (0, nv_pad)))
    qkv_all = qkv_f32.reshape(num_layers * padded_win_rows, 1, dim)
    b_all = b_f32.reshape(num_layers * padded_win_rows, 1, aligned_num_v_heads)
    a_all = a_f32.reshape(num_layers * padded_win_rows, 1, aligned_num_v_heads)

    # Vectorized multi-layer active-sequence tile compaction in JAX.
    qsl_i32 = jnp.minimum(query_start_loc.astype(jnp.int32), win_rows)
    query_lens = qsl_i32[1:] - qsl_i32[:-1]
    all_seqs = jnp.arange(max_seqs, dtype=jnp.int32)
    end_seq = distribution[0].astype(jnp.int32)
    is_valid_seqs = all_seqs < end_seq
    has_init_1d = (seq_lens.astype(jnp.int32) - query_lens) > 0

    if commit_lens_stack.shape[0] == 1:
        cl_1d = jnp.where(
            is_valid_seqs,
            jnp.clip(commit_lens_stack[0].astype(jnp.int32), 0, query_lens),
            0,
        )
        needs_1d = cl_1d > 0
        num_act_0 = jnp.sum(needs_1d.astype(jnp.int32))
        act_seqs_0 = jnp.nonzero(needs_1d, size=max_seqs, fill_value=0)[0].astype(
            jnp.int32
        )
        clamped_commit = jnp.broadcast_to(cl_1d[None, :], (num_layers, max_seqs))
        num_active = jnp.broadcast_to(num_act_0, (num_layers,))
        active_seqs = jnp.broadcast_to(act_seqs_0[None, :], (num_layers, max_seqs))
        t_max = pl.cdiv(num_act_0, cfg.seq_tile_size)
    else:
        clamped_commit = jnp.where(
            is_valid_seqs[None, :],
            jnp.clip(commit_lens_stack.astype(jnp.int32), 0, query_lens[None, :]),
            0,
        )
        needs_commit = clamped_commit > 0
        num_active = jnp.sum(needs_commit.astype(jnp.int32), axis=1)
        active_seqs = jax.vmap(
            lambda m: jnp.nonzero(m, size=max_seqs, fill_value=0)[0].astype(jnp.int32)
        )(needs_commit)
        tiles_per_layer = pl.cdiv(num_active, cfg.seq_tile_size)
        t_max = jnp.max(tiles_per_layer)

    max_tiles_per_layer = pl.cdiv(max_seqs, cfg.seq_tile_size)
    total_active_tiles = num_layers * t_max
    t_safe = jnp.maximum(t_max, 1)

    max_total_tiles = num_layers * max_tiles_per_layer
    p_ids = jnp.arange(max_total_tiles, dtype=jnp.int32)
    p_layer = jnp.minimum(p_ids // t_safe, num_layers - 1)
    p_lt = p_ids % t_safe

    slot_offsets = (
        p_lt[:, None] * cfg.seq_tile_size
        + jnp.arange(cfg.seq_tile_size, dtype=jnp.int32)[None, :]
    )
    layer_num_active = num_active[p_layer][:, None]
    active_mask = (p_ids[:, None] < total_active_tiles) & (
        slot_offsets < layer_num_active
    )
    safe_slot = jnp.minimum(slot_offsets, max_seqs - 1)
    s_local = jnp.where(active_mask, active_seqs[p_layer[:, None], safe_slot], 0)
    s_global = p_layer[:, None] * max_seqs + s_local
    r_base = jnp.where(
        active_mask,
        p_layer[:, None] * padded_win_rows + qsl_i32[s_local],
        p_layer[:, None] * padded_win_rows,
    )
    r_size = jnp.where(active_mask, clamped_commit[p_layer[:, None], s_local], 0)

    metadata_obj = memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=total_active_tiles,
        p_id_to_s_idx=s_global.reshape(-1),
        p_id_to_r_base=r_base.reshape(-1),
        p_id_to_r_size=r_size.reshape(-1),
        p_id_is_first_tile=active_mask.reshape(-1),
        p_id_is_last_tile=active_mask.reshape(-1),
        s_idx_has_initial_state=jnp.broadcast_to(
            has_init_1d[None, :], (num_layers, max_seqs)
        ).reshape(-1),
        s_idx_to_state_indices=state_indices_stack.astype(jnp.int32).reshape(-1),
        s_idx_to_read_offset=read_offsets_stack.astype(jnp.int32).reshape(-1),
        s_idx_to_ckpt_indices=ckpt_indices_stack.astype(jnp.int32).reshape(
            num_layers * max_seqs, 2
        ),
        s_idx_to_commit_len=clamped_commit.reshape(-1),
        p_id_to_layer=p_layer,
    )

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    out_rows = max(padded_win_rows, cfg.seq_tile_size * cfg.chunk_size)
    out_shape = jax.ShapeDtypeStruct((out_rows, n_v, d_v), act_out_dtype)

    num_meta_leaves = len(metadata_obj)
    _, out_source = pl.pallas_call(
        functools.partial(_multi_layer_replay_outer_kernel, cfg=cfg),
        out_shape=(out_shape, state_source),
        in_specs=(
            metadata_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, hbm_spec),
        input_output_aliases={num_meta_leaves + 3: 1},
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
        name=f"{cfg.get_kernel_name()}_ml{num_layers}",
        metadata=cast(Any, cfg.get_metadata()),
    )(
        metadata_obj,
        qkv_all,
        b_all,
        a_all,
        state_source,
        weights,
    )
    return out_source
