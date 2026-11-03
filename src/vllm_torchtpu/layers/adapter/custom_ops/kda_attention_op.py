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
from vllm_torchtpu.gdn_pool_layout import (
    DEFAULT_POOLED_GDN_CONV_STATE_DTYPE,
    derive_pooled_gdn_state_layout,
    pooled_gdn_conv_state_bytes,
    pooled_gdn_ssm_state_bytes,
    pooled_gdn_state_itemsize,
)
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_config
from vllm_torchtpu.kernels.gdn.v3 import wrapper as gdn_wrapper
from vllm_torchtpu.layers.core.causal_conv1d import ragged_causal_conv1d
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_VerifyWindowResult = TypeVar("_VerifyWindowResult")


def _guard_kda_verify_window(
    num_window_reqs: jax.Array,
    run_verify_window: Callable[[], _VerifyWindowResult],
    skip_verify_window: Callable[[], _VerifyWindowResult],
) -> _VerifyWindowResult:
    """Skip KDA verification when the device-side window is empty."""
    return jax.lax.cond(num_window_reqs > 0, run_verify_window, skip_verify_window)


def kimi_short_conv_scan(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    start_seq: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Kimi short convolution over a ragged batch, delegating to ``causal_conv1d``."""
    return ragged_causal_conv1d(
        x,
        conv_state,
        conv_weight,
        conv_bias=None,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        seq_lens=seq_lens,
        start_seq=start_seq,
    )


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
    output *= jax.lax.rsqrt(jnp.mean(output * output, axis=-1, keepdims=True) + eps)
    output = (output * norm_weight.astype(jnp.float32)).astype(activation_dtype)
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
    if a_log.shape != (num_heads,):
        raise ValueError("KDA A_log shape does not match the head count")
    if dt_bias.shape != (num_heads * head_dim,):
        raise ValueError("KDA dt_bias shape does not match the projection")
    if norm_weight.shape != (head_dim,):
        raise ValueError("KDA norm weight shape does not match the head size")
    if query_start_loc.shape != (num_sequences + 1,):
        raise ValueError("KDA query starts and state indices disagree")
    if seq_lens.shape != (num_sequences,):
        raise ValueError("KDA sequence lengths and state indices disagree")
    if (
        raw_gate.dtype != mixed_qkv.dtype
        or beta.dtype != mixed_qkv.dtype
        or output_gate.dtype != mixed_qkv.dtype
    ):
        raise TypeError("KDA activations must have one common dtype")
    if recurrent_state.dtype != jnp.float32:
        raise TypeError("KDA recurrent state must be float32")
    if a_log.dtype != jnp.float32 or dt_bias.dtype != jnp.float32:
        raise TypeError("KDA A_log and dt_bias must be float32")
    if (
        query_start_loc.dtype != jnp.int32
        or state_indices.dtype != jnp.int32
        or seq_lens.dtype != jnp.int32
    ):
        raise TypeError("KDA metadata tensors must be int32")
    return num_tokens, num_sequences, num_heads, head_dim


def _build_fused_core(lower_bound: float | None, eps: float, num_spec_tokens: int = 0):
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
        conv_state: jax.Array,  # [num_slots, K - 1, 1, 3 * H * D] or flat 3-D
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
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            recurrent_state,
            a_log,
            dt_bias,
            norm_weight,
            query_start_loc,
            state_indices,
            seq_lens,
        )
        kernel_size = conv_weight.shape[0]
        if window_distribution is None:
            window_distribution = distribution

        # `mixed_qkv` and `conv_weight` lay their channels out as (3, H, D) in
        # row-major order, which is exactly the flat `dim` axis the fused
        # kernel indexes -- the reshapes are free. The dense `conv_state`
        # cache is declared in the kernel's (K - 1, 1, dim) slot layout so it
        # is read in place; the pooled path hands in its gathered slots flat.
        # The kernel wants the weight transposed to [dim, 1, kernel_size].
        conv_weight_flat = conv_weight.reshape(kernel_size, -1)
        conv_weight_flat = jnp.transpose(conv_weight_flat, (1, 0))[:, None]
        read_offsets = None
        if num_spec_tokens > 0:
            if slot_read_offsets is None:
                raise ValueError("slot_read_offsets are required for KDA verify")
            read_offsets = slot_read_offsets[state_indices]

        def run_kernel(conv_in, rec_in, *, batched_only=False, prefill_only=False):
            return gdn_wrapper.fused_conv1d_gdn(
                qkv=mixed_qkv,
                # Raw: the kernel applies sigmoid to `b` and the gate activation
                # to `a` itself.
                b=beta,
                a=raw_gate,
                states=(conv_in, rec_in),
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
            empty_out = jnp.zeros((num_tokens, num_heads * head_dim), mixed_qkv.dtype)
            (verify_states, verify_out) = _guard_kda_verify_window(
                window_end,
                lambda: run_kernel(conv_state, recurrent_state, batched_only=True),
                lambda: ((conv_state, recurrent_state), empty_out),
            )
            first_prefill_token = query_start_loc[window_end]
            total_tokens = jnp.minimum(query_start_loc[-1], num_tokens)
            ((new_conv_state, new_pool), prefill_out) = jax.lax.cond(
                first_prefill_token < total_tokens,
                lambda: run_kernel(*verify_states, prefill_only=True),
                lambda: (verify_states, empty_out),
            )
            out = jnp.where(
                (jnp.arange(num_tokens) < first_prefill_token)[:, None],
                verify_out,
                prefill_out,
            )
        else:
            (new_conv_state, new_pool), out = run_kernel(conv_state, recurrent_state)

        output = out.reshape(num_tokens, num_heads, head_dim)
        output = _gated_output_norm(
            output, output_gate, norm_weight, eps, mixed_qkv.dtype
        )
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

    def _fake_dispatched(
        mixed_qkv,
        _raw_gate,
        _beta,
        _output_gate,
        conv_state,
        recurrent_state,
        *args,
        **kwargs,
    ):
        num_heads, head_dim = recurrent_state.shape[1:3]
        output = torch.empty(
            (mixed_qkv.size(0), num_heads, head_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        return (output, torch.empty_like(conv_state), torch.empty_like(recurrent_state))

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
    window_distribution: jax.Array | None = None,
    slot_read_offsets: jax.Array | None = None,
    ckpt_indices: jax.Array | None = None,
    *,
    lower_bound: float | None,
    eps: float,
    pool_block_tokens: int,
    conv_shard_heads: int | None = None,
    num_spec_tokens: int = 0,
) -> tuple[jax.Array, jax.Array]:
    num_reqs = state_indices.shape[0]
    window = num_spec_tokens + 1
    dense_offsets = None
    identity = 1 + jnp.arange(num_reqs, dtype=jnp.int32) * window
    if num_spec_tokens:
        if slot_read_offsets is None or ckpt_indices is None:
            raise ValueError(
                "Pooled KDA verify requires read offsets and checkpoint indices"
            )
        if ckpt_indices.shape != (num_reqs, window):
            raise ValueError(
                "Pooled KDA checkpoint indices must match the verify window"
            )
        # Manager blocks need not be contiguous. Keep every checkpoint in a
        # contiguous scratch window, including untouched ragged/prefill tails.
        # TODO: follow #889's is_shared=True path and access the unified pool
        # directly via ckpt_indices[s, offset], avoiding per-step gather/scatter
        # and dense checkpoint temporaries.
        num_reqs *= window
        dense_offsets = jnp.zeros(num_reqs + 1, dtype=jnp.int32)
        dense_offsets = dense_offsets.at[identity].set(slot_read_offsets[state_indices])
        state_indices = ckpt_indices.reshape(-1)
    num_heads = a_log.shape[0]
    head_dim = norm_weight.shape[0]
    kernel_size = conv_weight.shape[0]
    # The pool dtype is the attention KV dtype. The state regions keep the
    # dtypes gdn_pool_layout fixes for every reader of pool bytes (block
    # sizing, the Raiden manifest), so an fp8 pool still holds fp32 conv rows,
    # the only conv dtype the fused kernel accepts.
    conv_dtype = jnp.dtype(DEFAULT_POOLED_GDN_CONV_STATE_DTYPE.removeprefix("torch."))
    conv_elem_bytes = pooled_gdn_state_itemsize(DEFAULT_POOLED_GDN_CONV_STATE_DTYPE)
    tok_bytes = math.prod(pool.shape[2:]) * jnp.dtype(pool.dtype).itemsize
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=pooled_gdn_ssm_state_bytes(
            num_v_heads=num_heads, head_k_dim=head_dim, head_v_dim=head_dim
        ),
        conv_bytes=pooled_gdn_conv_state_bytes(
            kernel_size=kernel_size, conv_dim=3 * num_heads * head_dim
        ),
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
        raise ValueError(
            "Manager block size is not a whole number of pool "
            f"rows: {pool_block_tokens} tokens at "
            f"{tokens_per_row} tokens/row"
        )
    manager_rows = pool_block_tokens // tokens_per_row
    if manager_rows % pool.shape[1] != 0:
        raise ValueError(
            "Manager block does not split into whole pool "
            f"blocks: {manager_rows} rows vs pool block of "
            f"{pool.shape[1]}"
        )
    split = manager_rows // pool.shape[1]
    if layout.required_tokens > manager_rows:
        raise ValueError(
            "KDA state regions exceed the pool block: "
            f"{layout.required_tokens} > {manager_rows}"
        )

    # SSM region [0, ssm_tokens): f32 view with one head_dim-wide lane group
    # per typed row, so the leading H * D rows are exactly the state.
    ssm_gathered = pool_adapters.gather_region(
        pool,
        state_indices,
        tok0=0,
        ntok=layout.ssm_tokens,
        out_dtype=jnp.float32,
        out_lanes=head_dim,
        split=split,
    )
    ssm_rows = num_heads * head_dim
    ssm_local = ssm_gathered[:, :ssm_rows, :].reshape(
        num_reqs, num_heads, head_dim, head_dim
    )

    # Stage-3 stores whole, padded prefill-head shards in rank order. Each
    # shard occupies whole pool rows, so Raiden can concatenate
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
    padded_shard_elems = (
        layout.conv_tokens // conv_shards * tok_bytes // conv_elem_bytes
    )
    if shard_elems > padded_shard_elems:
        raise ValueError("KDA conv shard exceeds its padded pool region")
    conv_gathered = pool_adapters.gather_region(
        pool,
        state_indices,
        tok0=layout.ssm_tokens,
        ntok=layout.conv_tokens,
        out_dtype=conv_dtype,
        split=split,
    )
    conv_local = conv_gathered.reshape(num_reqs, conv_shards, padded_shard_elems)[
        :, :, :shard_elems
    ]
    conv_local = conv_local.reshape(
        num_reqs, conv_shards, kernel_size - 1, 3, shard_heads, head_dim
    )
    conv_local = conv_local.transpose(0, 2, 3, 1, 4, 5).reshape(
        num_reqs, kernel_size - 1, 3, num_heads, head_dim
    )

    # Dense slot 0 is scratch for the kernel's idempotent null-block writes.
    conv_buf = jnp.concatenate([jnp.zeros_like(conv_local[:1]), conv_local], axis=0)
    ssm_buf = jnp.concatenate([jnp.zeros_like(ssm_local[:1]), ssm_local], axis=0)

    # The fused conv1d + GDN v3 kernel takes the conv state flat
    # [slots, K - 1, 3*H*D]; the buffers here lay their channels out as
    # (K - 1, 3, H, D) in row-major order, so both reshapes are free. Padded
    # and zero-length sequences are safe under identity indices because the
    # kernel gates their state DMAs to zero size rather than redirecting them
    # to the null slot.
    fused_core = _build_fused_core(lower_bound, eps, num_spec_tokens)
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
        window_distribution,
        dense_offsets,
    )
    new_conv_buf = new_conv_buf.reshape(
        num_reqs + 1, kernel_size - 1, 3, num_heads, head_dim
    )

    # Scatter the real slots back; the padding rows/elems of each region are
    # zero-filled so the pool bytes stay deterministic.
    ssm_region_rows = layout.ssm_tokens * tok_bytes // (4 * head_dim)
    new_ssm = new_ssm_buf[1:].reshape(num_reqs, ssm_rows, head_dim)
    new_ssm = jnp.pad(new_ssm, ((0, 0), (0, ssm_region_rows - ssm_rows), (0, 0)))
    pool = pool_adapters.scatter_region(
        pool, new_ssm, state_indices, tok0=0, ntok=layout.ssm_tokens, split=split
    )

    new_conv = new_conv_buf[1:].reshape(
        num_reqs, kernel_size - 1, 3, conv_shards, shard_heads, head_dim
    )
    new_conv = new_conv.transpose(0, 3, 1, 2, 4, 5).reshape(
        num_reqs, conv_shards, shard_elems
    )
    new_conv = jnp.pad(
        new_conv, ((0, 0), (0, 0), (0, padded_shard_elems - shard_elems))
    )
    new_conv = new_conv.reshape(num_reqs, -1, pool.shape[-1])
    pool = pool_adapters.scatter_region(
        pool,
        new_conv,
        state_indices,
        tok0=layout.ssm_tokens,
        ntok=layout.conv_tokens,
        split=split,
    )

    return output, pool


def build_kimi_pooled_kda_op(
    prefix: str,
    *,
    lower_bound: float | None,
    eps: float,
    vllm_config: VllmConfig,
    num_spec_tokens: int = 0,
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
    if (
        tpu_envs.TPU_USE_RAIDEN_KV_CACHE_MANAGER
        and tpu_envs.TPU_RAIDEN_KIMIK3_ADMISSION
    ):
        total_heads = int(
            vllm_config.model_config.hf_text_config.linear_attn_config["num_heads"]
        )
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
        window_distribution: jax.Array | None = None,
        slot_read_offsets: jax.Array | None = None,
        ckpt_indices: jax.Array | None = None,
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
            window_distribution,
            slot_read_offsets,
            ckpt_indices,
            lower_bound=lower_bound,
            eps=eps,
            pool_block_tokens=vllm_config.cache_config.block_size,
            conv_shard_heads=conv_shard_heads,
            num_spec_tokens=num_spec_tokens,
        )

    # vLLM's compile cache is keyed on the model config and not on the op
    # body, so the kernel variant goes in the op name (see
    # build_kimi_dispatched_kda_op). A pooled program cached from the unfused
    # era would otherwise be served stale.
    op_name = f"pallas::kimi_pooled_kda_fused_{prefix.replace('.', '_')}"
    op_name += f"_conv_{DEFAULT_POOLED_GDN_CONV_STATE_DTYPE.removeprefix('torch.')}"
    if num_spec_tokens:
        op_name += f"_window{num_spec_tokens + 1}"
    if conv_shard_heads is not None:
        # Stored state order is part of the compiled program, including for
        # decode-local requests and prefix-cache seeds, not just PD loads.
        op_name += f"_conv_rank_blocks_v1_h{conv_shard_heads}"
    pooled_op = pallas.jax_op(op_name, pooled_core, donate_argnums=(4,))

    def _fake_pooled(
        mixed_qkv,
        _raw_gate,
        _beta,
        _output_gate,
        pool,
        _conv_weight,
        a_log,
        _dt_bias,
        norm_weight,
        *args,
        **kwargs,
    ):
        output = torch.empty(
            (mixed_qkv.size(0), a_log.shape[0], norm_weight.shape[0]),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
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
        window_distribution: torch.Tensor | None = None,
        slot_read_offsets: torch.Tensor | None = None,
        ckpt_indices: torch.Tensor | None = None,
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
            window_distribution,
            slot_read_offsets,
            ckpt_indices,
        )
        pool.copy_(new_pool)
        return output

    return pooled_impl
