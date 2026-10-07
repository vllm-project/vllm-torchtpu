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

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import (
    compute_conv1d,
    compute_gdn,
    compute_kda,
    config,
    legacy_unified,
    memory_ref,
    metadata,
    vmem_ldst,
)


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Array,  # [seq, chunk, 1, dim_size]
    b_slot_ref: jax.Array,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: jax.Array,  # [seq, chunk, 1, num_v_heads]
    conv_state_slot_ref: jax.Array,  # [seq, prev_kernel_size, 1, dim_size]
    recurrent_slot_ref: jax.Array,  # [seq, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: jax.Array,  # [seq * chunk, num_v_heads, v_head]
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
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
        conv_state_slot_ref: Convolution state VMEM ref that stores data loaded from
            HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
        recurrent_slot_ref: Recurrent state VMEM ref that stores data loaded from
            HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
        out_slot_ref: Output VMEM ref that will be used for VMEM to HBM write.
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
        conv_state_slot_ref=conv_state_slot_ref,
        recurrent_slot_ref=recurrent_slot_ref,
        carry_conv_scratch_ref=carry_conv_scratch_ref,
        carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
        cfg=cfg,
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
    )

    if cfg.state_plan is None:
        # Only the checkpoint positions; the extra incoming-state position
        # (`cfg.state_in_pos`) is never copied out.
        conv_state_slot_ref[:, : cfg.window_size] = new_conv_state
    else:
        # TODO: Delete this after completing unified pool redesign.
        legacy_unified.store_external_state(
            conv_state_slot_ref, cfg.state_plan.conv, new_conv_state, cfg
        )
    if carry_conv_scratch_ref is not None:
        # The next tile resumes from the state after this tile's last token,
        # which is the final checkpoint.
        carry_conv_scratch_ref[...] = new_conv_state[:, -1]

    # Apply activation function.
    qkv_out_compact = jax.nn.silu(qkv_out_compact)
    if cfg.is_kda and cfg.window_size > 1:
        # The one-token K3 decode kernel stores the post-SiLU activation in
        # the model activation dtype before promoting it for the recurrent
        # math. Preserve that numerical contract for a fused verify window;
        # otherwise its target logits can drift solely because the KDA window
        # keeps this intermediate in fp32.
        qkv_out_compact = qkv_out_compact.astype(cfg.dtypes.act_out).astype(jnp.float32)

    # Step 2: GDN.

    # Prepare gdn weights.
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
        # Round to the state dtype, then pack into the uint32 window. Only
        # the checkpoint positions; see `cfg.state_in_pos`.
        recurrent_slot_ref[:, : cfg.window_size] = pltpu.bitcast(
            new_recurrent_state.astype(cfg.dtypes.recurrent_state), jnp.uint32
        )
    else:
        # TODO: Delete this after unified pool migration completes.
        legacy_unified.store_external_state(
            recurrent_slot_ref, cfg.state_plan.recurrent, new_recurrent_state, cfg
        )

    if carry_recurrent_scratch_ref is not None:
        carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    states_ref: jax.Array | tuple[jax.Array, jax.Array],
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    states_out_ref: jax.Array | tuple[jax.Array, jax.Array],
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
):
    """Setup memory allocations and emit pipeline for running inner_kernel."""
    del states_out_ref

    if cfg.is_shared:
        # One pool row holds both regions; they are told apart by their
        # granule offset, which `create_allocs` derives from this ref.
        assert not isinstance(states_ref, tuple)
        conv_state_ref = recurrent_state_ref = states_ref
    else:
        assert len(states_ref) == 2
        conv_state_ref, recurrent_state_ref = states_ref
    qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
        memory_ref.create_allocs(
            metadata_ref=metadata_ref,
            qkv_ref=qkv_ref,
            b_ref=b_ref,
            a_ref=a_ref,
            out_ref=out_ref,
            conv_state_ref=conv_state_ref,
            recurrent_state_ref=recurrent_state_ref,
            cfg=cfg,
        )
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
            conv_alloc.spec,
            recurrent_alloc.spec,
        ),
        out_specs=(out_alloc.spec,),
    )

    @pl.with_scoped(
        allocations=(
            qkv_alloc,
            b_alloc,
            a_alloc,
            conv_alloc,
            recurrent_alloc,
            out_alloc,
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
            scratches=(
                metadata_ref,
                weights_ref,
                carry_conv_scratch_ref,
                carry_recurrent_scratch_ref,
            ),
            allocations=allocations,
        )

    _run()


@jax.jit(
    donate_argnames=("states",),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "decode_tile_size",
        "mixed_tile_size",
        "batched_only",
        "prefill_only",
        "zero_initialize_out",
        "compute_precision",
        "recurrent_dtype",
        "conv_dtype",
        "attention_mode",
        "gate_lower_bound",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]; [batch_size, n_v * d_k] for KDA
    states: jax.Array | tuple[jax.Array, jax.Array],
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
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    attention_mode: config.AttentionMode = config.AttentionMode.GDN,
    gate_lower_bound: float | None = None,
    recurrent_dtype: jnp.dtype | None = None,
    conv_dtype: jnp.dtype | None = None,
    num_spec_tokens: int = 0,
    batched_only: bool = False,
    prefill_only: bool = False,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    # TODO(kyuyeunk): Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
) -> tuple[jax.Array | tuple[jax.Array, jax.Array], jax.Array]:
    """Perform conv1d and gdn in a single fused kernel.

    Args:
        qkv: Mixed query, key, value input tensor of shape [batch_size, dim_size],
            where `dim_size = n_kq * d_k * 2 + n_v * d_v`.
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
        num_spec_tokens: Number of speculative draft tokens (0 gives the
            plain 1-token-per-sequence decode path).
        batched_only: Run only the leading decode/verify segment.
        prefill_only: Run only PER_SEQ, leaving decode/verify checkpoints
            untouched. Mutually exclusive with batched_only.
        zero_initialize_out: Whether to zero-initialize the output buffer before
            executing non-batched sequences.
        compute_precision: Computation precision dtype.
        decode_tile_size: Tile size along sequence dimension for decode sequences.
        mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
            sequences.

    Returns:
        new_states: The updated state cache, aliased in place, with the same
            structure as `states`.
        out: Fused output tensor.
    """
    # TODO(kyuyeunk): Support bf16
    act_out_dtype = qkv.dtype

    is_shared = isinstance(states, jax.Array)
    if is_shared:
        state_packing = 4 // states.dtype.itemsize
        # Ensure compact layout is used.
        assert states.shape[-2] == state_packing
        assert conv_dtype is not None
        assert recurrent_dtype is not None
    else:
        assert len(states) == 2
        recurrent_dtype = states[1].dtype
        conv_dtype = states[0].dtype
    # Conv1D reads the compact layout, which needs fp32. The recurrent state
    # may be bf16: it is upcast once on load and rounded once on store.
    if jnp.dtype(conv_dtype) != jnp.float32 or jnp.dtype(recurrent_dtype) not in (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.bfloat16),
    ):
        raise NotImplementedError(
            "GDN v3 needs an fp32 conv state and an fp32 or bf16 recurrent "
            f"state; got conv={conv_dtype} recurrent={recurrent_dtype}."
        )

    qkv = qkv.astype(jnp.float32)
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)

    # Step 1: Validate inputs.
    is_kda = attention_mode == config.AttentionMode.KDA
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
    batch_size, dim = qkv.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim,)
    assert query_start_loc.shape == (num_seqs + 1,)
    assert state_indices.shape == (num_seqs,)
    assert distribution.shape == (3,)
    if num_spec_tokens > 0:
        assert read_offsets is not None, (
            "read_offsets is required when num_spec_tokens > 0"
        )
    if read_offsets is None:
        read_offsets = jnp.zeros_like(state_indices)
    assert read_offsets.shape == (num_seqs,)
    if ckpt_indices is not None:
        # One source block per checkpoint (vLLM's ssm_state_indices scheme):
        # checkpoint t of sequence s lives at block ckpt_indices[s, t], so
        # the pool block only ever has to hold a single state.
        assert ckpt_indices.shape == (num_seqs, num_spec_tokens + 1)
    else:
        # Iniitalize the ckpt_indices to default value.
        spec_offset = jnp.arange(num_spec_tokens + 1).reshape(1, -1)
        ckpt_indices = state_indices.reshape(-1, 1) + spec_offset

    act_in_dtype = qkv.dtype
    assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 4 // act_in_dtype.itemsize
    padded_batch_size = pl.cdiv(batch_size, packing) * packing
    decode_tile_size = min(decode_tile_size, batch_size)
    mixed_tile_size = min(mixed_tile_size, batch_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

    if num_spec_tokens > 0:
        # A verify window holds one state checkpoint per window position per
        # sequence in VMEM, which multiplies the per-sequence footprint by
        # the window size. Shrink the tile so the double-buffered windows
        # fit in roughly half the scoped-VMEM budget (the rest goes to
        # weights, activations scratch and compiler temporaries).
        window = num_spec_tokens + 1
        num_buffers = config.GDNConfig.__dataclass_fields__["num_buffers"].default
        # State tiles hold `window` checkpoints plus one dedicated
        # incoming-state position (`GDNConfig.state_in_pos`) when window > 1.
        state_positions = window + (1 if window > 1 else 0)
        recurrent_bytes_per_seq = (
            state_positions * n_v * d_k * d_v * jnp.dtype(recurrent_dtype).itemsize
        )
        bytes_per_seq = (
            # Recurrent checkpoints — the dominant term.
            recurrent_bytes_per_seq
            # Conv checkpoints (fp32).
            + state_positions * (kernel_size - 1) * dim * 4
            # qkv (fp32), b/a (fp32), out (act_out).
            + window * (dim * 4 + 2 * aligned_num_v_heads * 4 + n_v * d_v * 2)
        )
        vmem_budget = int(
            config.GDNConfig.WINDOWED_VMEM_FRACTION
            * pltpu.get_tpu_info().vmem_capacity_bytes
        )
        spec_tile_budget = (vmem_budget // 2) // num_buffers
        decode_tile_size = max(
            1, min(decode_tile_size, spec_tile_budget // bytes_per_seq)
        )

        if is_kda:
            # Keep exactly one KDA sequence in each window tile.  The state
            # pipeline issues one async recurrent-state DMA per sequence but
            # shares a semaphore and drains the tile with one aggregate wait.
            # Bounding only the total bytes is insufficient: TP32's four
            # 3-head sequences have the same 6 MiB footprint as one TP8
            # 12-head sequence, but the former still has four independent DMA
            # starts and corrupts recurrent checkpoints in long-running
            # serving.  A one-sequence tile is the only schedule validated for
            # repeated rollback and slot reuse on v7, independent of TP size.
            decode_tile_size = 1

    batch_padding_size = padded_batch_size - batch_size
    num_v_padding_size = aligned_num_v_heads - n_v
    # KDA's gate carries d_k channels per value head, so it pads out to a
    # lane multiple of n_v * d_k rather than of n_v.
    gate_dim = n_v * d_k if is_kda else n_v
    gate_padding_size = (pl.cdiv(gate_dim, num_lanes) * num_lanes) - gate_dim
    qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
    b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
    a = jnp.pad(a, ((0, batch_padding_size), (0, gate_padding_size)))

    qkv = qkv.reshape(padded_batch_size, 1, -1)
    b = b.reshape(padded_batch_size, 1, -1)
    a = a.reshape(padded_batch_size, 1, -1)

    # TODO(kyuyeunk): Ensure no reshape is needed and convert to assertion.
    conv_state_shape = None
    recurrent_state_shape = None
    if not is_shared:
        conv_state_shape = states[0].shape
        rec_state = states[1]
        if jnp.dtype(recurrent_dtype) == jnp.bfloat16:
            recurrent_state_shape = rec_state.shape
            rec_state = rec_state.reshape(-1, n_v * (d_k // 2), 2, d_v)
        states = (states[0].reshape(-1, kernel_size - 1, 1, dim), rec_state)

    # Step 3: Weights pre-processing.
    # TODO(kyuyeunk): To eliminate runtime cost, move this logic into model
    # loading stage.
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
        in_states: jax.Array | tuple[jax.Array, jax.Array],
        in_act: jax.Array | None,
        mode: config.GDNMode,
    ) -> tuple[jax.Array, jax.Array | tuple[jax.Array, jax.Array]]:
        if mode == config.GDNMode.PER_SEQ:
            tile_size = mixed_tile_size
            # Prefill / mixed sequences keep a single state checkpoint.
            window_size = 1
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1

        cfg = config.GDNConfig(
            mode=mode,
            batch_size=padded_batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            window_size=window_size,
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            attention_mode=attention_mode,
            gate_lower_bound=gate_lower_bound,
            dtypes=config.Dtypes(
                act_in=act_in_dtype,
                act_out=act_out_dtype,
                compute=compute_precision,
                recurrent_state=recurrent_dtype,
                conv_state=conv_dtype,
            ),
            is_shared=is_shared,
        )

        # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
        # but will be CSEed by compiler.
        if mode != config.GDNMode.PER_SEQ:
            metadata_obj = metadata.compute_batched_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                read_offsets=read_offsets,
                end_seq=distribution[0],
                ckpt_indices=ckpt_indices,
            )
        else:
            metadata_obj = metadata.compute_per_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                read_offsets=read_offsets,
                start_seq=distribution[0],
                end_seq=distribution[-1],
                ckpt_indices=ckpt_indices,
            )

        metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)
        state_spec = jax.tree.map(lambda _: hbm_spec, in_states)

        # Step 7: Handle case where write needs to be done in existing out.
        input_output_aliases = {}
        state_pos = len(metadata_obj) + 3
        # This ensures the logic can support both unified and non-unified pool.
        num_states = len(jax.tree.leaves(in_states))
        for i in range(num_states):
            input_output_aliases[state_pos + i] = 1 + i
        act_pos = state_pos + num_states

        in_out_spec = None
        out_shape = cfg.get_out_shape()
        if in_act is None and zero_initialize_out:
            in_act = jnp.zeros_like(out_shape)
        if in_act is not None:
            out_shape = in_act
            in_out_spec = hbm_spec
            input_output_aliases[act_pos] = 0

        return pl.pallas_call(
            functools.partial(outer_kernel, cfg=cfg),
            out_shape=(out_shape, in_states),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                state_spec,
                in_out_spec,
                weights_spec,
            ),
            out_specs=(hbm_spec, state_spec),
            scratch_shapes=cfg.get_scratch_shape_dict(),
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=cfg.get_vmem_limit_bytes(),
            ),
            name=cfg.get_kernel_name(),
            metadata=cfg.get_metadata(),
        )(
            metadata_obj,
            qkv,
            b,
            a,
            in_states,
            in_act,
            weights,
        )

    # The first segment holds verify windows of up to `num_spec_tokens + 1`
    # tokens, or plain 1-token decodes without speculative decoding.
    if batched_only and prefill_only:
        raise ValueError("batched_only and prefill_only are mutually exclusive")

    out_act, out_states = None, states
    if not prefill_only:
        out_act, out_states = call_kernel(states, None, config.GDNMode.BATCHED)
    if not batched_only:
        out_act, out_states = call_kernel(out_states, out_act, config.GDNMode.PER_SEQ)

    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
    # TODO(kyuyeunk): Ensure no reshape is needed and convert to assertion.
    if conv_state_shape is not None:
        out_states = (out_states[0].reshape(conv_state_shape), out_states[1])
    if recurrent_state_shape is not None:
        out_states = (out_states[0], out_states[1].reshape(recurrent_state_shape))
    return out_states, out_act
