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
"""Public facade and orchestration entry points for the SparseMLA TPU kernel."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import (
    dsa_gather as dsa_gather,
    gather_dispatch,
    native_sc_gather as native_sc_gather,
)
from vllm_torchtpu.kernels.mla.sparse.attention_body import (
    rope_q_block as rope_q_block,
)
from vllm_torchtpu.kernels.mla.sparse.attention_kernel import (
    _attention_kernel as _attention_kernel,
    _decode_attention_kernel as _decode_attention_kernel,
    _mixed_prefill_attention_kernel as _mixed_prefill_attention_kernel,
    select_attention_kernel as select_attention_kernel,
)
from vllm_torchtpu.kernels.mla.sparse.config import (
    DECODE_BODY_DEFAULT as DECODE_BODY_DEFAULT,
    DECODE_BODY_LEVERS as DECODE_BODY_LEVERS,
    DECODE_STEP8_TOKENS as DECODE_STEP8_TOKENS,
    DEFAULT_ATTENTION_KERNEL_BATCH_SIZE as DEFAULT_ATTENTION_KERNEL_BATCH_SIZE,
    DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE as DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE,
    DEFAULT_VMEM_LIMIT_BYTES as DEFAULT_VMEM_LIMIT_BYTES,
    FUSED_PREP_MAX_TOKENS as FUSED_PREP_MAX_TOKENS,
    FUSED_PREP_META_ROWS as FUSED_PREP_META_ROWS,
    GATHER_ATOM_ROWS as GATHER_ATOM_ROWS,
    MIXED_BODY_DEFAULT as MIXED_BODY_DEFAULT,
    MIXED_BODY_LEVERS as MIXED_BODY_LEVERS,
    MIXED_BODY_MAX_SKIP_TOKENS as MIXED_BODY_MAX_SKIP_TOKENS,
    MIXED_KERNEL_V2_LEVERS as MIXED_KERNEL_V2_LEVERS,
    Q_ROPE_DIM as Q_ROPE_DIM,
    Q_ROPE_HEAD_DIM as Q_ROPE_HEAD_DIM,
    SEND_EXCHANGE_COLLECTIVE_ID as SEND_EXCHANGE_COLLECTIVE_ID,
    SEND_EXCHANGE_PIN_ROLES as SEND_EXCHANGE_PIN_ROLES,
    SEND_EXCHANGE_VMEM_LIMIT_BYTES as SEND_EXCHANGE_VMEM_LIMIT_BYTES,
    SEND_PAD_ROWS as SEND_PAD_ROWS,
    SEND_SLAB_ROW0 as SEND_SLAB_ROW0,
    SPARSECORE_CACHE_LAYOUT as SPARSECORE_CACHE_LAYOUT,
    TENSORCORE_CACHE_LAYOUT as TENSORCORE_CACHE_LAYOUT,
    UNSET,
    AttentionKernelConfig as AttentionKernelConfig,
    BodyExecutionStrategy as BodyExecutionStrategy,
    BodyOptimizationConfig as BodyOptimizationConfig,
    CacheLayout as CacheLayout,
    CappedWindowConfig as CappedWindowConfig,
    DcpConfig as DcpConfig,
    GatherConfig as GatherConfig,
    GatherWaitMode as GatherWaitMode,
    HbmPinRole as HbmPinRole,
    LseConfig as LseConfig,
    SendExchangeConfig as SendExchangeConfig,
    SparseMlaConfig as SparseMlaConfig,
    _Unset,
    align_to as align_to,
    cdiv as cdiv,
    chunk_kernel_batch_size,
    get_dtype_bitwidth as get_dtype_bitwidth,
    get_dtype_packing as get_dtype_packing,
    largest_divisor,
    live_group_supported as live_group_supported,
    merge_pieces_supported as merge_pieces_supported,
    mixed_fused_prep_supported as mixed_fused_prep_supported,
    resolve_entry_config,
    send_layout_lanes as send_layout_lanes,
    send_layout_shape as send_layout_shape,
    send_layout_supported as send_layout_supported,
)
from vllm_torchtpu.kernels.mla.sparse.index_prep import (
    capped_live_windows as capped_live_windows,
    gather_page_ids as gather_page_ids,
    live_group_table as live_group_table,
    prepare_capped_window_fused as prepare_capped_window_fused,
    prepare_capped_window_indices as prepare_capped_window_indices,
    prepare_capped_window_live as prepare_capped_window_live,
    prepare_capped_window_live_fused as prepare_capped_window_live_fused,
    prepare_gather_indices as prepare_gather_indices,
    scatter_padding_slots as scatter_padding_slots,
)
from vllm_torchtpu.kernels.mla.sparse.ref_registry import (
    PallasRefRegistry as PallasRefRegistry,
    build_attention_ref_registry,
)
from vllm_torchtpu.kernels.mla.sparse.testing_faults import (
    NO_FAULT_HOOKS as NO_FAULT_HOOKS,
    PROLOGUE_FAULTS as PROLOGUE_FAULTS,
    SEND_EXCHANGE_FAULTS as SEND_EXCHANGE_FAULTS,
    SEND_FAULTS as SEND_FAULTS,
    FaultHooks as FaultHooks,
)

# Backward-compatible private aliases imported by adapter/tests.
_largest_divisor = largest_divisor
_chunk_kernel_batch_size = chunk_kernel_batch_size

_DEFAULT_BODY = BodyOptimizationConfig()

_ATTENTION_CHUNK_LEGACY_DEFAULTS: dict[str, Any] = {
    "small_chunk_tokens": 0,
    "small_chunk_batch_size": None,
    "small_chunk_live_gather": False,
    "live_gather_all_chunks": False,
}


def prepare_q_inputs(q: jax.Array) -> jax.Array:
    """Pads query heads to dtype packing and head dimension to 128 lanes."""
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    return jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )


def prepare_outputs(
    out: jax.Array,
    actual_num_q_heads: int,
    actual_head_dim: int,
) -> jax.Array:
    """Slices padded attention output back to the caller's head geometry."""
    return out[:, :actual_num_q_heads, :actual_head_dim]


def _resolve_entry_hooks(
    hooks: FaultHooks,
    *,
    allow_send_exchange: bool,
    prologue_fault: str | None = None,
    out_pieces_fault: str | None = None,
    kv_write_fault: str | None = None,
) -> FaultHooks:
    """Merges legacy fault-name kwargs into ``FaultHooks`` and validates context."""
    if (
        prologue_fault is not None
        or out_pieces_fault is not None
        or kv_write_fault is not None
    ):
        if hooks != NO_FAULT_HOOKS:
            raise ValueError("cannot pass both hooks and legacy fault kwargs")
        hooks = FaultHooks.from_names(
            prologue_fault=prologue_fault,
            out_pieces_fault=out_pieces_fault,
            kv_write_fault=kv_write_fault,
            allow_send_exchange=allow_send_exchange,
        )
    hooks.validate_for_context(allow_send_exchange=allow_send_exchange)
    return hooks


def _prepare_lse_and_pieces(
    q: jax.Array,
    actual_num_q_heads: int,
    *,
    num_chunks: int,
    has_q_pe: bool,
    has_kv_write: bool,
    config: SparseMlaConfig,
) -> tuple[bool, jax.Array | None, int]:
    """Validates send/piece settings and allocates the initial LSE buffer."""
    gather_cfg = config.gather
    lse_cfg = config.lse
    dcp_cfg = config.dcp
    num_tokens, num_q_heads, _ = q.shape
    chunk_size = gather_cfg.chunk_size
    uniform_chunk = min(chunk_size, num_tokens)

    lse_fill = lse_cfg.effective_lse_fill(num_chunks)
    if dcp_cfg.send_world:
        if not (lse_fill and has_q_pe and q.dtype == jnp.bfloat16):
            raise ValueError(
                "send_world needs lse_in_kernel on one chunk, split q and bf16 q"
            )
        _, send_batch = chunk_kernel_batch_size(
            uniform_chunk,
            config.attention_kernel_batch_size,
            gather_cfg.small_chunk_tokens,
            gather_cfg.small_chunk_batch_size,
        )
        if q.shape[1] != actual_num_q_heads or not send_layout_supported(
            num_tokens, q.shape[1], dcp_cfg.send_world, q.shape[2], send_batch
        ):
            raise ValueError(
                f"send layout unsupported: q {q.shape}, world {dcp_cfg.send_world}, "
                f"batch {send_batch}"
            )

    # Rows the grid never reaches (padding past `batch_end`) keep this sentinel
    # so downstream merges weight them at exp(min - max) == 0.
    lse = (
        jnp.full(
            (num_tokens, num_q_heads)
            if lse_cfg.lane_dense
            else (num_tokens, num_q_heads, 1),
            -jnp.inf if lse_cfg.mask_empty else jnp.finfo(jnp.float32).min,
            dtype=jnp.float32,
        )
        if lse_cfg.return_lse and not lse_fill
        else None
    )

    piece_rows = 0
    if dcp_cfg.out_pieces > 1 and merge_pieces_supported(
        num_tokens, chunk_size, dcp_cfg.out_pieces
    ):
        if not (
            lse_cfg.return_lse
            and lse_cfg.lane_dense
            and lse_cfg.mask_empty
            and not lse_fill
            and not has_kv_write
        ):
            raise ValueError(
                "out_pieces needs return_lse, lse_lane_dense and lse_mask_empty"
            )
        piece_rows = max(num_tokens // dcp_cfg.out_pieces, chunk_size)

    return lse_fill, lse, piece_rows


def _build_chunk_config(
    config: SparseMlaConfig,
    capped: CappedWindowConfig,
    prepared: gather_dispatch.PreparedIndices,
    *,
    num_tokens: int,
    lse_fill: bool,
) -> SparseMlaConfig:
    """Specializes ``config`` for per-chunk execution."""
    has_group = prepared.group_tab is not None
    chunk_capped = dataclasses.replace(
        capped,
        mixed_live_widths=capped.mixed_live_widths if has_group else (),
        decode_row_widths=capped.decode_row_widths if not has_group else (),
        mixed_body=(
            capped.mixed_body
            if has_group and num_tokens > MIXED_BODY_MAX_SKIP_TOKENS
            else _DEFAULT_BODY
        ),
        decode_body=(
            capped.decode_body
            if not has_group and prepared.fused_prep and capped.decode_row_widths
            else _DEFAULT_BODY
        ),
    )
    return dataclasses.replace(
        config,
        capped=chunk_capped,
        lse=dataclasses.replace(config.lse, in_kernel=lse_fill),
    )


def _invoke_chunk(
    chunk_idx: int,
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    prepared: gather_dispatch.PreparedIndices,
    chunk_lse: jax.Array | None,
    *,
    chunk_size: int,
    num_tokens: int,
    token_end: jax.Array,
    chunk_config: SparseMlaConfig,
    hooks: FaultHooks,
    topk: int,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    kv_write: gather_dispatch.KvWriteOperand | None,
    send_peer: jax.Array | None,
    send_recv_in: jax.Array | None,
    out_piece: tuple[int, int, jax.Array | None] | None = None,
    gather_only: bool = False,
    pre_gathered: gather_dispatch.PreGatheredChunk | None = None,
) -> Any:
    """Invokes ``sparse_attention_chunk`` for chunk ``chunk_idx``."""
    start_pos = chunk_idx * chunk_size
    end_pos = min(start_pos + chunk_size, num_tokens)
    return sparse_attention_chunk(
        q,
        cache_kv_nope,
        cache_kv_rope,
        prepared.kv_lens,
        prepared.physical_indices,
        chunk_lse,
        start_pos=start_pos,
        end_pos=end_pos,
        token_end=token_end,
        config=chunk_config,
        hooks=hooks,
        topk_override=topk,
        window_indices=prepared.window_indices_for_chunk(chunk_idx),
        gather_live_in_indices=prepared.fused_prep,
        q_pe=q_pe,
        group_tab=prepared.group_tab,
        group_num_live=prepared.group_num_live_for_chunk(chunk_idx),
        q_rope_table=q_rope_table,
        kv_write=kv_write,
        send_peer=send_peer,
        send_recv_in=send_recv_in,
        out_piece=out_piece,
        gather_only=gather_only,
        pre_gathered=pre_gathered,
    )


def _run_chunks_and_pieces(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    prepared: gather_dispatch.PreparedIndices,
    lse: jax.Array | None,
    *,
    num_chunks: int,
    chunk_size: int,
    piece_rows: int,
    token_end: jax.Array,
    chunk_config: SparseMlaConfig,
    hooks: FaultHooks,
    topk: int,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    kv_write: gather_dispatch.KvWriteOperand | None,
    send_peer: jax.Array | None,
    send_recv_in: jax.Array | None,
) -> tuple[jax.Array, jax.Array | None, jax.Array, jax.Array]:
    """Executes chunked gather + attention across chunks and optional pieces."""
    num_tokens, num_q_heads, _ = q.shape
    return_lse = chunk_config.lse.return_lse
    gather_first = bool(piece_rows) and chunk_config.dcp.out_pieces_gather_first
    chunk_kwargs: dict[str, Any] = {
        "chunk_size": chunk_size,
        "num_tokens": num_tokens,
        "token_end": token_end,
        "chunk_config": chunk_config,
        "hooks": hooks,
        "topk": topk,
        "q_pe": q_pe,
        "q_rope_table": q_rope_table,
        "kv_write": kv_write,
        "send_peer": send_peer,
        "send_recv_in": send_recv_in,
    }

    # With gather-first merge pieces, issue every chunk's SparseCore gather up
    # front and barrier after piece 0 so piece 0's reduce-scatter is not queued
    # behind gathers the schedule still holds open.
    pre: list[Any] = [None] * num_chunks
    if gather_first:
        pre = [
            _invoke_chunk(
                i,
                q,
                cache_kv_nope,
                cache_kv_rope,
                prepared,
                None,
                gather_only=True,
                **chunk_kwargs,
            )
            for i in range(num_chunks)
        ]

    chunks_per_piece = max(1, piece_rows // chunk_size)
    piece_outs: list[jax.Array] = []
    piece_lses: list[jax.Array] = []
    prev_out: Any = None
    for i in range(num_chunks):
        start_pos = i * chunk_size
        out_piece: tuple[int, int, jax.Array | None] | None = None
        piece_lse = lse
        if piece_rows:
            if gather_first and i == chunks_per_piece and i < num_chunks:
                rest, piece_outs, piece_lses = lax.optimization_barrier(
                    (pre[i:], piece_outs, piece_lses)
                )
                pre[i:] = rest
            offset = start_pos // piece_rows * piece_rows
            carried: jax.Array | None = None
            if start_pos == offset:
                # Piece's first chunk starts with fresh output and -inf LSE rows.
                piece_lse = jnp.full((piece_rows, num_q_heads), -jnp.inf, jnp.float32)
            else:
                piece_lse = piece_lses.pop()
                carried = piece_outs.pop()
            out_piece = (piece_rows, offset, carried)

        q_in = q
        if piece_rows and i:
            # Independent merge-piece calls may be reordered by XLA; hand call
            # `i` its read-only `q` through a barrier with call `i-1`'s output.
            q_in, _ = lax.optimization_barrier((q, prev_out))

        result = _invoke_chunk(
            i,
            q_in,
            cache_kv_nope,
            cache_kv_rope,
            prepared,
            piece_lse,
            out_piece=out_piece,
            pre_gathered=pre[i],
            **chunk_kwargs,
        )
        if kv_write is not None:
            result, (cache_kv_nope, cache_kv_rope) = result
        if piece_rows:
            prev_out = result[0]
            piece_outs.append(result[0])
            piece_lses.append(result[1])
        elif return_lse:
            q, lse = result
        else:
            q = result

    if piece_rows:
        piece_outs = hooks.transform_piece_outs(piece_outs)
        q = jnp.concatenate(piece_outs, axis=0)
        lse = jnp.concatenate(piece_lses, axis=0)

    return q, lse, cache_kv_nope, cache_kv_rope


def _validate_chunk_args(
    config: SparseMlaConfig,
    *,
    physical_indices: jax.Array | None,
    lse: jax.Array | None,
    topk_override: int | None,
    window_indices: jax.Array | None,
    gather_live_in_indices: bool,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    kv_write: gather_dispatch.KvWriteOperand | None,
    preserve_suffix: bool,
) -> tuple[bool, int, int | None]:
    """Validates chunk arguments and resolves ``(return_lse, topk, capped_windows)``."""
    gather_cfg = config.gather
    lse_fill = config.lse.in_kernel
    topk_cap = config.capped.topk_cap
    sparsecore_layout = gather_cfg.is_sparsecore

    if kv_write is not None and not gather_live_in_indices:
        raise ValueError("kv_write rides the fused-prep capped gather only")
    return_lse = lse is not None or lse_fill
    assert not (lse_fill and lse is not None), "lse_fill owns the lse buffer"
    assert q_rope_table is None or q_pe is not None, "q_rope_table needs split q"

    if topk_override is not None:
        topk = topk_override
    else:
        assert physical_indices is not None
        topk = physical_indices.shape[-1]

    if window_indices is not None and topk_cap is None:
        raise ValueError("window_indices requires topk_cap")
    if topk_cap is not None and not sparsecore_layout:
        raise ValueError("topk_cap requires the sparsecore cache layout")
    if gather_cfg.packed_rope and not sparsecore_layout:
        raise ValueError("packed_rope requires the sparsecore cache layout")
    if topk_cap is not None and (
        topk_cap <= 0 or topk_cap % GATHER_ATOM_ROWS or topk % topk_cap
    ):
        raise ValueError(
            f"topk_cap ({topk_cap}) must be a multiple of "
            f"{GATHER_ATOM_ROWS} dividing topk ({topk})"
        )
    if config.dcp.send_world and not (lse_fill and q_pe is not None):
        raise ValueError("send_world needs lse_fill and split q")
    if preserve_suffix and (return_lse or sparsecore_layout):
        raise ValueError("Prefix stores require TensorCore caches without LSE")

    capped_windows = None if topk_cap is None else topk // topk_cap
    return return_lse, topk, capped_windows


def _resolve_chunk_bounds(
    start_pos: int,
    end_pos: int,
    token_end: jax.Array,
    config: SparseMlaConfig,
) -> tuple[bool, int, jax.Array, int, jax.Array]:
    """Computes gather and attention batch sizes and aligned end bounds."""
    gather_cfg = config.gather
    chunk_size = end_pos - start_pos
    small_chunk, kernel_batch_size = chunk_kernel_batch_size(
        chunk_size,
        config.attention_kernel_batch_size,
        gather_cfg.small_chunk_tokens,
        gather_cfg.small_chunk_batch_size,
    )
    assert chunk_size % kernel_batch_size == 0
    # The kernel grid walks [start_pos, batch_end) in `kernel_batch_size` steps.
    valid_rows = jnp.maximum(0, jnp.minimum(token_end, end_pos) - start_pos)
    batch_end = start_pos + cdiv(valid_rows, kernel_batch_size) * kernel_batch_size
    # step8_attention_grid changes only the attention grid step; prep and
    # gather keep `kernel_batch_size`.
    attn_batch_size, attn_batch_end = kernel_batch_size, batch_end
    if config.capped.decode_body.step8_attention_grid and small_chunk:
        attn_batch_size = largest_divisor(chunk_size, DECODE_STEP8_TOKENS)
        attn_batch_end = start_pos + cdiv(valid_rows, attn_batch_size) * attn_batch_size
    return small_chunk, kernel_batch_size, batch_end, attn_batch_size, attn_batch_end


def _build_attention_kernel_config(
    config: SparseMlaConfig,
    *,
    batch_size: int,
    capped_windows: int | None,
    preserve_suffix: bool,
    return_lse: bool,
    split_q: bool,
    q_rope: bool,
    has_group_tab: bool,
    out_row_offset: int,
    out_in: bool,
    recv_in: bool,
) -> AttentionKernelConfig:
    """Assembles the static ``AttentionKernelConfig`` for one Pallas launch."""
    capped_cfg = config.capped
    if has_group_tab:
        active_body = capped_cfg.mixed_body
        active_live_widths = capped_cfg.mixed_live_widths
        active_row_widths: tuple[int, ...] = ()
    elif capped_cfg.decode_row_widths and capped_windows is not None:
        active_body = capped_cfg.decode_body
        active_live_widths = ()
        active_row_widths = capped_cfg.decode_row_widths
    else:
        active_body = _DEFAULT_BODY
        active_live_widths = ()
        active_row_widths = ()

    return AttentionKernelConfig(
        sm_scale=config.sm_scale,
        k_scale=config.k_scale,
        batch_size=batch_size,
        preserve_suffix=preserve_suffix,
        return_lse=return_lse,
        sparsecore_layout=config.gather.is_sparsecore,
        capped_windows=capped_windows,
        is_dcp_mode=config.dcp.is_dcp_mode,
        split_q=split_q,
        q_rope=q_rope,
        lse_lane_dense=config.lse.lane_dense,
        lse_mask_empty=config.lse.mask_empty,
        lse_fill=config.lse.in_kernel,
        has_group_tab=has_group_tab,
        live_widths=active_live_widths,
        row_widths=active_row_widths,
        body=active_body,
        send_world=config.dcp.send_world,
        out_row_offset=out_row_offset,
        out_in=out_in,
        send_exchange=config.dcp.send_exchange,
        recv_in=recv_in,
    )


def _launch_attention_kernel(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    kv_lens: jax.Array,
    *,
    start_seq_idx: int | jax.Array,
    end_seq_idx: jax.Array,
    token_end: jax.Array,
    kernel_batch_size: int,
    config: SparseMlaConfig,
    hooks: FaultHooks,
    capped_windows: int | None,
    preserve_suffix: bool,
    return_lse: bool,
    lse: jax.Array | None,
    q_pe: jax.Array | None,
    q_rope_table: jax.Array | None,
    group_tab: jax.Array | None,
    live_views: tuple[jax.Array, jax.Array] | None,
    out_piece: tuple[int, int, jax.Array | None] | None,
    send_peer: jax.Array | None,
    send_recv_in: jax.Array | None,
) -> Any:
    """Builds the attention ref registry and launches the specialized Pallas kernel."""
    lse_fill = config.lse.in_kernel
    send_exchange = config.dcp.send_exchange
    batch_size = kernel_batch_size
    out_offset, out_carried = (
        (0, None) if out_piece is None else (out_piece[1], out_piece[2])
    )
    end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
    grid: tuple[Any, ...] = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
    if lse_fill:
        # Never empty: step 0 lays down the LSE sentinel.
        grid = (jnp.maximum(grid[0], 1),)

    scope_name = (
        f"MLA-{config.gather.cache_layout.value}-p_{cache_kv_rope.shape[1]}-"
        f"bz_{batch_size}-gcz_{cache_kv_nope.shape[0]}"
    )
    if capped_windows is not None:
        scope_name += f"-capw_{capped_windows}"

    vmem_limit = config.vmem_limit_bytes
    extra_compiler_params: dict[str, Any] = {}
    pin_fn = None
    if send_exchange is not None:
        # Fused exchange runs on a one-step grid covering the whole bucket.
        assert config.dcp.send_world and lse_fill and batch_size == q.shape[0], (
            config.dcp.send_world, lse_fill, batch_size, q.shape,
        )
        grid = (1,)
        scope_name += "-xa2a"
        vmem_limit = min(vmem_limit, send_exchange.vmem_limit_bytes)
        extra_compiler_params["collective_id"] = SEND_EXCHANGE_COLLECTIVE_ID
        pin_fn = send_exchange.should_pin

    kernel_cfg = _build_attention_kernel_config(
        config, batch_size=batch_size, capped_windows=capped_windows,
        preserve_suffix=preserve_suffix, return_lse=return_lse,
        split_q=q_pe is not None, q_rope=q_rope_table is not None,
        has_group_tab=live_views is not None, out_row_offset=out_offset,
        out_in=out_carried is not None, recv_in=send_recv_in is not None,
    )
    registry, bounds_layout = build_attention_ref_registry(
        kernel_cfg, q=q, cache_kv_nope=cache_kv_nope, cache_kv_rope=cache_kv_rope,
        kv_lens=kv_lens, start_seq_idx=start_seq_idx, end_seq_idx=end_seq_idx,
        token_end=token_end, group_tab=group_tab, q_pe=q_pe,
        q_rope_table=q_rope_table, live_views=live_views, lse=lse,
        send_recv_in=send_recv_in, out_piece=out_piece, send_peer=send_peer,
    )
    kernel_body = functools.partial(
        select_attention_kernel(kernel_cfg),
        cfg=kernel_cfg, ref_layout=registry.layout,
        bounds_layout=bounds_layout, hooks=hooks,
    )
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            kernel_body,
            grid_spec=registry.build_grid_spec(grid),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit,
                disable_bounds_checks=True,
                # pyrefly: ignore[bad-argument-type]
                **extra_compiler_params,
            ),
            out_shape=registry.out_shape,
            input_output_aliases=registry.input_output_aliases,
            name=scope_name,
        )
    )
    return kernel(*registry.call_args(pin_fn=pin_fn))


# Main attention kernel for GLM-5.2 DSA (top-k gather and attention).
# The KV entries referenced by `topk_indices` (including the current batch's
# own tokens) must already be resident in the caches before this call;
# per-token kv_lens derive from the "-1" padding in `topk_indices`.
@functools.partial(
    jax.jit,
    static_argnames=(
        "config",
        "hooks",
        "sm_scale",
        "k_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
        "return_lse",
        "cache_layout",
        "sparsecore_atoms_per_batch",
        "sparsecore_gather_wait_mode",
        "sparsecore_gather_pipelined",
        "small_chunk_tokens",
        "small_chunk_batch_size",
        "small_chunk_live_gather",
        "small_chunk_gather_atoms",
        "live_gather_all_chunks",
        "topk_cap",
        "is_dcp_mode",
        "capped_live_rows",
        "lse_lane_dense",
        "lse_mask_empty",
        "fused_window_prep",
        "mixed_live_gather",
        "mixed_live_widths",
        "mixed_body_levers",
        "lse_in_kernel",
        "prologue_fault",
        "kv_write_fault",
        "mixed_fused_prep",
        "decode_row_widths",
        "decode_body_levers",
        "send_world",
        "send_exchange",
        "out_pieces",
        "out_pieces_fault",
        "out_pieces_gather_first",
        "packed_rope",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    # TensorCore: u8[pages, page_size, 4, 128].
    # SparseCore: u32[pages, page_size, 128].
    cache_kv_nope: jax.Array,
    # TensorCore: u8[pages, page_size/4, 4, 128].
    # SparseCore: u32[pages, page_size/4, 128] (page_size/8 with packed_rope).
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array | None,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    config: SparseMlaConfig | None = None,
    hooks: FaultHooks = NO_FAULT_HOOKS,
    sm_scale: float | _Unset = UNSET,
    k_scale: float | _Unset = UNSET,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int | _Unset = UNSET,
    attention_kernel_batch_size: int | _Unset = UNSET,
    vmem_limit_bytes: int | _Unset = UNSET,
    return_lse: bool | _Unset = UNSET,
    cache_layout: str | _Unset = UNSET,
    sparsecore_atoms_per_batch: int | _Unset = UNSET,
    sparsecore_gather_wait_mode: str | _Unset = UNSET,
    sparsecore_gather_pipelined: bool | _Unset = UNSET,
    small_chunk_tokens: int | _Unset = UNSET,
    small_chunk_batch_size: int | None | _Unset = UNSET,
    small_chunk_live_gather: bool | _Unset = UNSET,
    small_chunk_gather_atoms: int | _Unset = UNSET,
    live_gather_all_chunks: bool | _Unset = UNSET,
    topk_cap: int | None | _Unset = UNSET,
    is_dcp_mode: bool | _Unset = UNSET,
    capped_live_rows: bool | _Unset = UNSET,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool | _Unset = UNSET,
    lse_mask_empty: bool | _Unset = UNSET,
    fused_window_prep: bool | _Unset = UNSET,
    mixed_live_gather: bool | _Unset = UNSET,
    mixed_live_widths: tuple[int, ...] | _Unset = UNSET,
    mixed_body_levers: tuple[str, ...] | _Unset = UNSET,
    q_rope_table: jax.Array | None = None,
    lse_in_kernel: bool | _Unset = UNSET,
    prologue_fault: str | None = None,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    kv_write_fault: str | None = None,
    mixed_fused_prep: bool | _Unset = UNSET,
    decode_row_widths: tuple[int, ...] | _Unset = UNSET,
    send_world: int | _Unset = UNSET,
    out_pieces: int | _Unset = UNSET,
    decode_body_levers: tuple[str, ...] | _Unset = UNSET,
    out_pieces_fault: str | None = None,
    out_pieces_gather_first: bool | _Unset = UNSET,
    send_exchange: tuple[Any, ...] | None | _Unset = UNSET,
    send_peer: jax.Array | None = None,
    send_recv_in: jax.Array | None = None,
    packed_rope: bool | _Unset = UNSET,
) -> jax.Array | tuple[jax.Array, ...]:
    """MLA Ragged paged attention supporting mixed prefill and decode.

    Accepts either a typed ``SparseMlaConfig`` (and optional ``FaultHooks``)
    or the legacy keyword arguments for backward compatibility.

    Returns:
      The output of attention, or ``(output, lse)`` when ``return_lse`` is set.
      ``lse`` is ``f32[max_num_tokens, actual_num_q_heads]``. With ``kv_write``,
      the updated ``(cache_kv_nope, cache_kv_rope)`` follow.
    """
    config = resolve_entry_config(
        config,
        {
            "sm_scale": sm_scale,
            "k_scale": k_scale,
            "gather_and_attention_chunk_size": gather_and_attention_chunk_size,
            "attention_kernel_batch_size": attention_kernel_batch_size,
            "vmem_limit_bytes": vmem_limit_bytes,
            "return_lse": return_lse,
            "cache_layout": cache_layout,
            "sparsecore_atoms_per_batch": sparsecore_atoms_per_batch,
            "sparsecore_gather_wait_mode": sparsecore_gather_wait_mode,
            "sparsecore_gather_pipelined": sparsecore_gather_pipelined,
            "small_chunk_tokens": small_chunk_tokens,
            "small_chunk_batch_size": small_chunk_batch_size,
            "small_chunk_live_gather": small_chunk_live_gather,
            "small_chunk_gather_atoms": small_chunk_gather_atoms,
            "live_gather_all_chunks": live_gather_all_chunks,
            "topk_cap": topk_cap,
            "is_dcp_mode": is_dcp_mode,
            "capped_live_rows": capped_live_rows,
            "lse_lane_dense": lse_lane_dense,
            "lse_mask_empty": lse_mask_empty,
            "fused_window_prep": fused_window_prep,
            "mixed_live_gather": mixed_live_gather,
            "mixed_live_widths": mixed_live_widths,
            "mixed_body_levers": mixed_body_levers,
            "lse_in_kernel": lse_in_kernel,
            "mixed_fused_prep": mixed_fused_prep,
            "decode_row_widths": decode_row_widths,
            "decode_body_levers": decode_body_levers,
            "send_world": send_world,
            "send_exchange": send_exchange,
            "out_pieces": out_pieces,
            "out_pieces_gather_first": out_pieces_gather_first,
            "packed_rope": packed_rope,
        },
    )
    hooks = _resolve_entry_hooks(
        hooks,
        allow_send_exchange=config.dcp.send_exchange is not None,
        prologue_fault=prologue_fault,
        out_pieces_fault=out_pieces_fault,
        kv_write_fault=kv_write_fault,
    )
    if config.dcp.send_exchange is not None and send_peer is None:
        raise ValueError(
            "send_exchange = (axis, lane_axis) needs send_world and send_peer "
            "(int32 [2]: rank, lane)"
        )

    page_size = config.validate_caches(cache_kv_nope, cache_kv_rope)
    _, actual_num_q_heads, actual_head_dim = q.shape
    config.validate_queries(q, q_pe, q_rope_table, topk_indices)
    if q_pe is None:
        q = prepare_q_inputs(q)

    topk = topk_indices.shape[-1]
    capped = config.capped.resolve_for_inputs(
        config.gather.is_sparsecore, topk, q_rope_table is not None
    )
    token_end = cu_q_lens[distribution[2]]
    prepared = gather_dispatch.prepare_attention_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        token_end,
        num_tokens=q.shape[0],
        page_size=page_size,
        total_num_pages=cache_kv_nope.shape[0],
        config=config,
        capped=capped,
        hooks=hooks,
        has_kv_write=kv_write is not None,
        prepare_live_fused_fn=prepare_capped_window_live_fused,
    )

    chunk_size = config.gather.chunk_size
    num_chunks = cdiv(q.shape[0], chunk_size)
    lse_fill, lse, piece_rows = _prepare_lse_and_pieces(
        q,
        actual_num_q_heads,
        num_chunks=num_chunks,
        has_q_pe=q_pe is not None,
        has_kv_write=kv_write is not None,
        config=config,
    )
    chunk_config = _build_chunk_config(
        config,
        capped,
        prepared,
        num_tokens=q.shape[0],
        lse_fill=lse_fill,
    )
    q, lse, cache_kv_nope, cache_kv_rope = _run_chunks_and_pieces(
        q,
        cache_kv_nope,
        cache_kv_rope,
        prepared,
        lse,
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        piece_rows=piece_rows,
        token_end=token_end,
        chunk_config=chunk_config,
        hooks=hooks,
        topk=topk,
        q_pe=q_pe,
        q_rope_table=q_rope_table,
        kv_write=kv_write,
        send_peer=send_peer,
        send_recv_in=send_recv_in,
    )

    out = prepare_outputs(q, actual_num_q_heads, actual_head_dim)
    caches = () if kv_write is None else (cache_kv_nope, cache_kv_rope)
    if config.lse.return_lse:
        lse = cast(jax.Array, lse)
        if config.dcp.send_world:
            # `lse` is the send buffer; `out` was not written.
            return out, lse, *caches
        if config.lse.lane_dense:
            return out, lse[:, :actual_num_q_heads], *caches
        return out, lse[:, :actual_num_q_heads, 0], *caches
    return (out, *caches) if caches else out


def sparse_attention_chunk(
    q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim], padded
    # TensorCore: u8[pages, page_size, 4, 128].
    # SparseCore: u32[pages, page_size, 128].
    cache_kv_nope: jax.Array,
    # TensorCore: u8[pages, page_size/4, 4, 128].
    # SparseCore: u32[pages, page_size/4, 128] (page_size/8 with packed_rope).
    cache_kv_rope: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_tokens]
    physical_indices: jax.Array | None,  # i32[max_num_tokens, csa_topk]
    lse: jax.Array | None,  # f32[max_num_tokens, num_q_heads, 1]
    *,
    start_pos: int,
    end_pos: int,
    token_end: jax.Array,  # i32
    config: SparseMlaConfig | None = None,
    hooks: FaultHooks = NO_FAULT_HOOKS,
    sm_scale: float | _Unset = UNSET,
    k_scale: float | _Unset = UNSET,
    preserve_suffix: bool = False,
    attention_kernel_batch_size: int | _Unset = UNSET,
    vmem_limit_bytes: int | _Unset = UNSET,
    cache_layout: str | _Unset = UNSET,
    sparsecore_atoms_per_batch: int | _Unset = UNSET,
    sparsecore_gather_wait_mode: str | _Unset = UNSET,
    sparsecore_gather_pipelined: bool | _Unset = UNSET,
    small_chunk_tokens: int | _Unset = UNSET,
    small_chunk_batch_size: int | None | _Unset = UNSET,
    small_chunk_live_gather: bool | _Unset = UNSET,
    small_chunk_gather_atoms: int | _Unset = UNSET,
    live_gather_all_chunks: bool | _Unset = UNSET,
    topk_cap: int | None | _Unset = UNSET,
    capped_live_rows: bool | _Unset = UNSET,
    topk_override: int | None = None,
    is_dcp_mode: bool | _Unset = UNSET,
    window_indices: jax.Array | None = None,
    gather_live_in_indices: bool = False,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool | _Unset = UNSET,
    lse_mask_empty: bool | _Unset = UNSET,
    group_tab: jax.Array | None = None,
    group_num_live: jax.Array | None = None,
    group_widths: tuple[int, ...] | _Unset = UNSET,
    group_levers: tuple[str, ...] | _Unset = UNSET,
    row_levers: tuple[str, ...] | _Unset = UNSET,
    q_rope_table: jax.Array | None = None,
    lse_fill: bool | _Unset = UNSET,
    prologue_fault: str | None = None,
    kv_write: jax.Array | tuple[jax.Array, jax.Array, jax.Array] | None = None,
    kv_write_fault: str | None = None,
    row_widths: tuple[int, ...] | _Unset = UNSET,
    send_world: int | _Unset = UNSET,
    out_piece: tuple[int, int, jax.Array | None] | None = None,
    gather_only: bool = False,
    pre_gathered: tuple[Any, ...] | None = None,
    send_exchange: tuple[Any, ...] | None | _Unset = UNSET,
    send_peer: jax.Array | None = None,
    send_recv_in: jax.Array | None = None,
    packed_rope: bool | _Unset = UNSET,
) -> Any:
    """Gathers and attends query rows ``[start_pos, min(end_pos, token_end))``.

    The output aliases ``q``. Returns the updated ``q``, or ``(q, lse)`` when
    ``lse`` is given. The attention grid is rounded up to its kernel batch size,
    so up to ``attention_kernel_batch_size - 1`` rows at or after ``token_end``
    may also be overwritten; rows further on keep their input values. Private
    split routing sets ``preserve_suffix`` to restrict stores to owned rows.
    """
    config = resolve_entry_config(
        config,
        {
            "sm_scale": sm_scale,
            "k_scale": k_scale,
            "attention_kernel_batch_size": attention_kernel_batch_size,
            "vmem_limit_bytes": vmem_limit_bytes,
            "cache_layout": cache_layout,
            "packed_rope": packed_rope,
            "sparsecore_atoms_per_batch": sparsecore_atoms_per_batch,
            "sparsecore_gather_wait_mode": sparsecore_gather_wait_mode,
            "sparsecore_gather_pipelined": sparsecore_gather_pipelined,
            "small_chunk_tokens": small_chunk_tokens,
            "small_chunk_batch_size": small_chunk_batch_size,
            "small_chunk_live_gather": small_chunk_live_gather,
            "small_chunk_gather_atoms": small_chunk_gather_atoms,
            "live_gather_all_chunks": live_gather_all_chunks,
            "topk_cap": topk_cap,
            "capped_live_rows": capped_live_rows,
            "is_dcp_mode": is_dcp_mode,
            "lse_lane_dense": lse_lane_dense,
            "lse_mask_empty": lse_mask_empty,
            "group_widths": group_widths,
            "row_widths": row_widths,
            "group_levers": group_levers,
            "row_levers": row_levers,
            "lse_fill": lse_fill,
            "send_world": send_world,
            "send_exchange": send_exchange,
        },
        defaults=_ATTENTION_CHUNK_LEGACY_DEFAULTS,
        required=("sm_scale", "k_scale"),
    )
    hooks = _resolve_entry_hooks(
        hooks,
        allow_send_exchange=config.dcp.send_exchange is not None,
        prologue_fault=prologue_fault,
        kv_write_fault=kv_write_fault,
    )
    return_lse, topk, capped_windows = _validate_chunk_args(
        config,
        physical_indices=physical_indices,
        lse=lse,
        topk_override=topk_override,
        window_indices=window_indices,
        gather_live_in_indices=gather_live_in_indices,
        q_pe=q_pe,
        q_rope_table=q_rope_table,
        kv_write=kv_write,
        preserve_suffix=preserve_suffix,
    )
    small_chunk, kernel_batch_size, batch_end, attn_batch_size, attn_batch_end = (
        _resolve_chunk_bounds(start_pos, end_pos, token_end, config)
    )
    gathered = gather_dispatch.gather_chunk_kv(
        cache_kv_nope,
        cache_kv_rope,
        kv_lens,
        physical_indices,
        start_pos=start_pos,
        end_pos=end_pos,
        topk=topk,
        kernel_batch_size=kernel_batch_size,
        small_chunk=small_chunk,
        batch_end=batch_end,
        config=config,
        hooks=hooks,
        window_indices=window_indices,
        gather_live_in_indices=gather_live_in_indices,
        group_tab=group_tab,
        group_num_live=group_num_live,
        kv_write=kv_write,
        pre_gathered=cast(gather_dispatch.PreGatheredChunk | None, pre_gathered),
    )
    if gather_only:
        return gathered.nope, gathered.rope, gathered.live_views

    result = _launch_attention_kernel(
        q,
        gathered.nope,
        gathered.rope,
        kv_lens,
        start_seq_idx=start_pos,
        end_seq_idx=attn_batch_end,
        token_end=token_end,
        kernel_batch_size=attn_batch_size,
        config=config,
        hooks=hooks,
        capped_windows=capped_windows,
        preserve_suffix=preserve_suffix,
        return_lse=return_lse,
        lse=lse,
        q_pe=q_pe,
        q_rope_table=q_rope_table,
        group_tab=group_tab,
        live_views=gathered.live_views,
        out_piece=out_piece,
        send_peer=send_peer,
        send_recv_in=send_recv_in,
    )
    if kv_write is not None:
        return result, (gathered.cache_kv_nope, gathered.cache_kv_rope)
    return result
