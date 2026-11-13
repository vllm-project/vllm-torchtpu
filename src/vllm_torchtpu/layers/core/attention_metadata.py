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
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import torch
from vllm.distributed import get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder as BaseAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    DEFAULT_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
)


def attention_block_table_width(
    block_table: Any,
    spec: KVCacheSpec,
    vllm_config: Any,
    model_len: int | None = None,
) -> int:
    """Match allocated, rank-local kernel page ordinals, including block splits."""
    if model_len is None:
        return block_table.max_num_blocks_per_req
    pages = spec.max_num_blocks_per_req(vllm_config, model_len)
    return min(
        pages * getattr(block_table, "blocks_per_kv_block", 1),
        block_table.max_num_blocks_per_req,
    )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "mamba_slot_read_offsets",
        "mamba_ckpt_indices",
        "mamba_request_distribution",
        "mamba_prefill_ckpt_offsets",
        "mamba_prefill_ckpt_indices",
        "pcp_metadata",
    ],
    meta_fields=[
        "sequence_layout_kind",
        "sequence_layout_protocol",
        "sequence_layout_version",
    ],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu", "prompt_lens_cpu"],
)
@dataclass
class AttentionMetadata:
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array | torch.Tensor | None
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array | torch.Tensor | None = None
    # (max_num_seqs,)
    seq_lens: jax.Array | torch.Tensor | None = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array | torch.Tensor | None = None
    # (3,)
    request_distribution: jax.Array | torch.Tensor | None = None
    # (max_num_seqs,) int32 - physical slot id in the mamba kv-cache for the
    # request currently in each persistent-batch position. Compact-mamba uses
    # an explicit slot from its independent pool; unified align mode derives
    # the slot from the current entry in the device block table. None for
    # attention groups, non-mamba models, and Mamba modes whose native fallback
    # addresses block_tables[:, 0].
    mamba_state_indices: jax.Array | torch.Tensor | None = None
    # (mamba_num_blocks,) int32 — per-*slot* read offset for speculative
    # decoding with mamba layers. `mamba_slot_read_offsets[base_slot]` is
    # `num_accepted - 1` from the request's most recent verify step: the GDN
    # kernel reads the request's initial state from `base_slot + offset`
    # (the checkpoint of the last accepted token) and writes fresh
    # checkpoints starting at `base_slot`. Indexed by physical slot (not
    # batch position) so the value survives requests being rescheduled or
    # condensed. Updated on device after each sampling step; None unless the
    # model has mamba layers *and* speculative decoding is enabled.
    mamba_slot_read_offsets: jax.Array | torch.Tensor | None = None
    # `(num_reqs, num_spec + 1)` source block per checkpoint. Unified pool
    # with speculative decoding only; None elsewhere, where the sole
    # checkpoint is the request's own state block.
    mamba_ckpt_indices: jax.Array | torch.Tensor | None = None
    # (3,) int32 — GDN-specific request distribution, same format as
    # `request_distribution` but with the first segment covering all
    # *windowed* sequences (plain decodes and speculative verify windows of
    # up to num_spec + 1 tokens) instead of only 1-token decodes. The
    # persistent batch is ordered [decode][verify][prefill/mixed] so both
    # segmentations hold at once: ragged paged attention keeps its 1-token
    # decode front segment while the GDN kernel runs its windowed mode over
    # the first two groups. None unless the model has mamba layers and spec
    # decoding is enabled.
    mamba_request_distribution: jax.Array | torch.Tensor | None = None
    # (max_num_seqs,) int32: per-request internal prefill checkpoint offset in
    # logical query tokens (0 if none). None when prefill checkpoints are off.
    mamba_prefill_ckpt_offsets: jax.Array | torch.Tensor | None = None
    # (max_num_seqs,) int32: this mamba group's block for the internal prefill
    # checkpoint (-1 if none). Present iff the group declares prefill
    # checkpoint blocks and mamba_prefill_ckpt_offsets is set.
    mamba_prefill_ckpt_indices: jax.Array | torch.Tensor | None = None
    # PCP manager only (see `runner/pcp/manager.py`); None elsewhere.
    pcp_metadata: PcpMetadata | None = None
    sequence_layout_kind: str = SequenceLayoutKind.ALL.value
    sequence_layout_protocol: str = DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
    sequence_layout_version: int = 1

    # Immutable host scheduling snapshots for model-side input planning. They
    # never participate in device graphs and require no device-to-host reads.
    query_start_loc_cpu: tuple[int, ...] | None = field(default=None, init=False)
    seq_lens_cpu: tuple[int, ...] | None = field(default=None, init=False)
    prompt_lens_cpu: tuple[int, ...] | None = field(default=None, init=False)


@dataclass
class AttentionMetadataBuilderContext:
    """Per-call inputs the TPU metadata builder reads from the runner.

    Stashed on TPUModelRunner before invoking _build_attention_metadata so
    builder.build() can pick them up. Only fields TPU populates differently from
    the parent runner (seq_lens / query_start_loc) live in TPU's own _cpu
    staging tensors, not parent's CpuGpuBuffers) or that have no
    `CommonAttentionMetadata` equivalent (num_reqs, use_max_model_len,
    start_index, request_distribution, position_ids_override) live here.
    """

    num_reqs: int
    start_index: int
    use_max_model_len: bool
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    request_distribution: torch.Tensor
    position_ids_override: torch.Tensor | None = None
    query_start_loc_cpu: tuple[int, ...] | None = None
    seq_lens_cpu: tuple[int, ...] | None = None
    prompt_lens_cpu: tuple[int, ...] | None = None
    # Compact-mamba per-request recurrent-slot ids (device int32, length =
    # target_num_reqs). Only the mamba group's builder reads it; None when the
    # model has no mamba layers. See AttentionMetadata.mamba_state_indices.
    mamba_state_indices: torch.Tensor | None = None
    # Spec decode with mamba layers only; see the AttentionMetadata fields of
    # the same names. Only the mamba group's builder reads them.
    mamba_slot_read_offsets: torch.Tensor | None = None
    # State checkpoints per request on the unified pool (num_spec + 1, each
    # its own block). 1 means a single state per request, so the builder
    # emits no `mamba_ckpt_indices`.
    mamba_ckpt_window: int = 1
    mamba_request_distribution: torch.Tensor | None = None
    # Device int32 (target_num_reqs,); populated on main-model steps and dummy
    # runs when GDN layers declare prefill checkpoint blocks.
    mamba_prefill_ckpt_offsets: torch.Tensor | None = None
    # Unified pool + spec decode only: the runner seeds this with an empty
    # list and each mamba group's builder appends the per-request state
    # block ids it derived from its block table, so the post-sampling
    # read-offset scatter can address every group's state block (groups
    # allocate distinct blocks, unlike the compact pool's shared slots).
    unified_mamba_state_indices: list[torch.Tensor] | None = None
    sequence_layout_descriptor: SequenceLayoutDescriptor = (
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR
    )
    # Per-chunk cache of the shared mamba row-offset plan, keyed by
    # (target_block_size, target_num_blocks) -> (state_index, ckpt_index,
    # ckpt_in_row). Valid only because every context is constructed fresh
    # per chunk: the plan derives from this chunk's seq_lens/ckpt_window,
    # which the key deliberately omits. Do not reuse a context across
    # chunks.
    mamba_row_plans: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ] = field(default_factory=dict)
    # Per-chunk batched block-table upload: kv_cache_group_id -> flat device
    # view into one staged H2D transfer (`stage_block_table_uploads`); None
    # means each build() uploads its own table (standalone builds, tests).
    staged_block_tables: dict | None = None
    # Per-chunk batched unified-pool mamba indices: kv_cache_group_id ->
    # (state_indices, ckpt_indices | None, prefill_ckpt_indices | None) from
    # the single `_staged_walk_products_compiled` program; None -> per-group
    # fallback.
    staged_mamba_products: dict | None = None


def _gather_prefill_ckpt_ids(
    table: torch.Tensor,
    state_cols: torch.Tensor,
    offsets: torch.Tensor,
    num_speculative_blocks: int,
) -> torch.Tensor:
    """Gathers each row's prefill checkpoint block ID, or -1 if unreserved.

    When reserved, the checkpoint block sits at column state_col - 1 (where
    state_col = (seq_len - 1) // B). A row is masked to -1 when:
    - offsets == 0 (which also masks padded rows with stale seq_lens) or
      state_col == 0.
    - The column holds the null block (0), meaning MambaManager did not
      reserve a checkpoint on this step (e.g. non-final chunks under
      prefix_cache_retention_interval=0 in vLLM #59175).
    - The block ID at state_col - 1 also appears in
      [state_col, state_col + num_speculative_blocks]. Because the worker's
      block table is append-only, a speculative block that MambaManager
      relocated from state_col - 1 into the new state/speculative tail still
      lingers at its old column in the worker table.
    """
    num_cols = table.shape[1]
    cols = state_cols - 1
    has_ckpt = (offsets > 0) & (cols >= 0)
    safe = torch.where(has_ckpt, cols, torch.zeros_like(cols))
    ids = torch.gather(table, 1, safe.unsqueeze(1)).squeeze(1)
    tail_cols = state_cols.unsqueeze(1) + torch.arange(
        num_speculative_blocks + 1, device=table.device, dtype=state_cols.dtype
    ).unsqueeze(0)
    in_row = tail_cols < num_cols
    tail = torch.gather(table, 1, torch.where(in_row, tail_cols, num_cols - 1))
    moved = ((tail == ids.unsqueeze(1)) & in_row).any(dim=1)
    reserved = has_ckpt & (ids != 0) & ~moved
    return torch.where(reserved, ids, torch.full_like(ids, -1))


_STAGE_BT_ROW_BUCKETS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
# Shared column divisors corresponding to context buckets (8k, 16k, 32k, 64k,
# 128k, 256k, 512k at max_model_len=524288): cols_g = cdiv(tnb, col_div).
_STAGE_BT_COL_DIVS: tuple[int, ...] = (64, 32, 16, 8, 4, 2, 1)


def _select_stage_row_bucket(num_reqs: int, num_reqs_padded: int) -> int:
    for rb in _STAGE_BT_ROW_BUCKETS:
        if rb >= num_reqs and rb < num_reqs_padded:
            return rb
    return num_reqs_padded


def _select_stage_col_div(
    entries: list[tuple[int, Any, int]],
    ctx: AttentionMetadataBuilderContext,
) -> int:
    if ctx.position_ids_override is not None or ctx.num_reqs <= 0:
        return 1
    needs: list[tuple[int, int]] = []
    for _, block_table_obj, tnb in entries:
        nbpr = getattr(block_table_obj, "num_blocks_per_row", None)
        if nbpr is None:
            return 1
        row_slice = nbpr[ctx.start_index : ctx.start_index + ctx.num_reqs]
        need_g = min(
            tnb, max(1, int(row_slice.max()) if len(row_slice) > 0 else 1)
        )
        needs.append((need_g, tnb))
    for col_div in _STAGE_BT_COL_DIVS:
        if all(
            min(tnb, max(1, cdiv(tnb, col_div))) >= need_g
            for need_g, tnb in needs
        ):
            return col_div
    return 1


def _staged_walk_products(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
    prefill_ckpt_offsets: torch.Tensor | None = None,
) -> tuple:
    """Splits and zero-pads the staged flat block-table buffer into per-group
    tables `[num_reqs_padded * target_num_blocks]` and derives unified-pool
    mamba state and checkpoint indices."""
    outs = []
    tables_2d = []
    for geom in table_geom:
        if len(geom) == 3:
            off, n_pad, t_pad = geom
            r_up, c_up = n_pad, t_pad
        else:
            off, r_up, c_up, n_pad, t_pad = geom
        sub = staged_dev[off : off + r_up * c_up].view(r_up, c_up)
        if r_up == n_pad and c_up == t_pad:
            tbl_2d = sub
            outs.append(sub.reshape(n_pad * t_pad).clone())
        else:
            tbl_2d = (
                torch.cat(
                    (
                        sub,
                        torch.zeros(
                            (r_up, t_pad - c_up),
                            dtype=sub.dtype,
                            device=sub.device,
                        ),
                    ),
                    dim=1,
                )
                if t_pad > c_up
                else sub
            )
            if n_pad > r_up:
                tbl_2d = torch.cat(
                    (
                        tbl_2d,
                        torch.zeros(
                            (n_pad - r_up, t_pad),
                            dtype=tbl_2d.dtype,
                            device=tbl_2d.device,
                        ),
                    ),
                    dim=0,
                )
            outs.append(tbl_2d.reshape(n_pad * t_pad))
        tables_2d.append(tbl_2d)
    states = []
    ckpts = []
    prefill_ckpts = []
    if mamba_geom:
        active = seq_lens > 0
        for ti, t, bs, prefill_ckpt_spec_blocks in mamba_geom:
            tbl = tables_2d[ti]
            state_offsets = torch.clamp((seq_lens - 1) // bs, min=0, max=t - 1).to(
                torch.int64
            )
            gathered = torch.gather(tbl, 1, state_offsets.unsqueeze(1)).squeeze(1)
            states.append(torch.where(active, gathered, torch.zeros_like(gathered)))
            if ckpt_window > 1:
                ckpt_offsets = state_offsets.unsqueeze(1) + torch.arange(
                    ckpt_window, device=seq_lens.device, dtype=torch.int64
                ).unsqueeze(0)
                in_row = (ckpt_offsets < t) & active.unsqueeze(1)
                safe = torch.where(in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets))
                ckpts.append(torch.gather(tbl, 1, safe) * in_row)
            if (
                prefill_ckpt_offsets is not None
                and prefill_ckpt_spec_blocks is not None
            ):
                prefill_ckpts.append(
                    _gather_prefill_ckpt_ids(
                        tbl,
                        state_offsets,
                        prefill_ckpt_offsets,
                        prefill_ckpt_spec_blocks,
                    )
                )
    return tuple(outs) + tuple(states) + tuple(ckpts) + tuple(prefill_ckpts)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _staged_walk_products_compiled(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
    prefill_ckpt_offsets: torch.Tensor | None = None,
) -> tuple:
    """Fused device program for per-group block-table splits, 2D zero-padding
    (graph outputs are distinct buffers, so no per-group eager clone) and
    unified-pool mamba state/checkpoint gathers.

    - table_geom: ((offset, row_bucket, cols_g, num_reqs_padded,
      target_num_blocks), ...) in walk order (the legacy 3-tuple
      (offset, num_reqs, num_blocks) form is also accepted).
    - mamba_geom: ((table_geom_index, num_blocks, block_size,
      prefill_ckpt_spec_blocks), ...), where prefill_ckpt_spec_blocks is the
      group's speculative block count when it declares prefill checkpoints and
      None otherwise. Both geoms are static per (row_bucket, col_div) bucket.
    - Returns: tables + states + (ckpts if ckpt_window > 1) + (prefill_ckpts
      for declaring groups if prefill_ckpt_offsets is given).

    Issued for every staged walk, dummy runs included, so warmup and lockstep
    ranks dispatch the same program sequence.
    """
    return _staged_walk_products(
        staged_dev,
        seq_lens,
        table_geom,
        mamba_geom,
        ckpt_window,
        prefill_ckpt_offsets,
    )


# Default cap on the number of recycled H2D host buffers registered on a runner.
# Each metadata walk calls `zeros_h2d_source` at most twice (once for the
# batched block table in `stage_block_table_uploads` / `build()`, and once for
# compact Mamba slot indices in `_build_mamba_state_indices`). With MTP /
# speculative decoding (`num_speculative_tokens` up to ~5), one decode step runs
# 1 target walk + up to (K + 1) draft walks (+ 1 optional DP dummy walk) ~= 8
# walks (16 buffers). Because draft walks run after the step's input-staging
# fence is recorded, two consecutive decode steps can hold buffers in flight or
# in the free list at once (~32 buffers); 64 gives a 2x generous upper bound.
#
# TODO: Extract this non-blocking H2D host-buffer staging, fence-lifetime
# tracking, and recycling pool into a dedicated standalone class/module in a
# future refactor instead of spreading the state across `TPUModelRunner` and
# `attention_metadata`.
_DEFAULT_MAX_H2D_BUFFERS = 64


def _noop_release() -> None:
    pass


def stage_h2d_upload(runner: Any, host: torch.Tensor) -> torch.Tensor:
    """`host.to(runner.device, non_blocking=True)`, keeping `host` alive until
    the runner's next input-staging fence has been synchronized.

    TorchTPU reads the CPU source asynchronously without holding a reference
    to it, so a per-call buffer would otherwise be freed (and its memory
    reused) while the transfer is still in flight. The runner moves the
    `(host, release_cb)` entry from `_unfenced_h2d_sources` to
    `_fenced_h2d_sources` when it records the fence and invokes `release_cb`
    once that fence has been waited on. Only buffers registered on `runner`
    by `zeros_h2d_source` are recycled into `_free_h2d_sources`; one-off
    tensors use a no-op callback and are dropped after synchronization.
    """
    assert host.is_contiguous()
    recycle = id(host) in runner._registered_h2d_sources
    if recycle:
        key = (host.numel(), host.dtype)

        def release_cb() -> None:
            free = runner._free_h2d_sources.pop(key, None)
            if free is None:
                free = []
            free.append(host)
            runner._free_h2d_sources[key] = free
    else:
        release_cb = _noop_release
    runner._unfenced_h2d_sources.append((host, release_cb))
    return host.to(runner.device, non_blocking=True)


def zeros_h2d_source(
    runner: Any,
    numel: int,
    dtype: torch.dtype = torch.int32,
    max_buffers: int = _DEFAULT_MAX_H2D_BUFFERS,
) -> torch.Tensor:
    """Return a zeroed 1D CPU buffer of `(numel, dtype)`, recycling one
    released by `_wait_input_staging_fence` when available.

    Buffers up to `max_buffers` are registered in `runner._registered_h2d_sources`
    so `stage_h2d_upload` recycles them into `runner._free_h2d_sources` once the
    fence synchronizes. When `max_buffers` is reached, the least-recently-used
    buffer in `runner._free_h2d_sources` is popped and unregistered before
    registering the new buffer, bounding both `_registered_h2d_sources` and
    `_free_h2d_sources`.
    """
    free = runner._free_h2d_sources.get((numel, dtype))
    if free:
        buf = free.pop().zero_()
        if id(buf) in runner._registered_h2d_sources:
            return buf
    else:
        buf = torch.zeros((numel,), dtype=dtype, device="cpu", pin_memory=PIN_MEMORY)
    if len(runner._registered_h2d_sources) >= max_buffers:
        for old_key, old_free in list(runner._free_h2d_sources.items()):
            if old_free:
                evicted = old_free.pop(0)
                if not old_free:
                    del runner._free_h2d_sources[old_key]
                runner._registered_h2d_sources.discard(id(evicted))
                break
    if len(runner._registered_h2d_sources) < max_buffers:
        runner._registered_h2d_sources.add(id(buf))
    return buf


class AttentionMetadataBuilder(BaseAttentionMetadataBuilder):
    """Single shared metadata builder for every TPU attention group."""

    def __init__(
        self, kv_cache_spec, layer_names, vllm_config, device, runner, kv_cache_group_id
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.runner = runner
        self.kv_cache_group_id = kv_cache_group_id
        self.is_mamba_group = isinstance(kv_cache_spec, MambaSpec)
        self.target_block_size = self.kv_cache_spec.block_size
        if self.is_mamba_group:
            try:
                cp_world_size = get_pcp_group().world_size
            except Exception:
                cp_world_size = 1
            self.target_block_size *= cp_world_size
        # Store num_speculative_blocks for groups declaring prefill checkpoints
        # (None otherwise) so _gather_prefill_ckpt_ids knows how many trailing
        # speculative columns to scan for relocated blocks.
        self.prefill_ckpt_spec_blocks: int | None = (
            kv_cache_spec.num_speculative_blocks
            if self.is_mamba_group and kv_cache_spec.num_prefill_checkpoint_blocks > 0
            else None
        )
        self.has_prefill_ckpt_blocks = self.prefill_ckpt_spec_blocks is not None
        # Only mamba/GDN layers consume physical state slot ids; attention
        # groups leave AttentionMetadata.mamba_state_indices None.

    def _mamba_row_plan(
        self, ctx: AttentionMetadataBuilderContext, target_num_blocks: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Row offsets into a mamba block table, cached and shared across groups.

        Mamba groups share identical sequence lengths, block sizes, and checkpoint
        geometry. Deriving row offset indices once per (block_size, target_num_blocks)
        allows all mamba groups to share the plan, leaving only per-group gathers.
        """
        key = (self.target_block_size, target_num_blocks)
        plan = ctx.mamba_row_plans.get(key)
        if plan is not None:
            return plan

        is_active = (ctx.seq_lens > 0).unsqueeze(1)
        state_offsets = torch.clamp(
            (ctx.seq_lens - 1) // self.target_block_size,
            min=0,
            max=target_num_blocks - 1,
        ).to(torch.int64)
        state_index = state_offsets.unsqueeze(1)

        ckpt_index: torch.Tensor | None = None
        in_row: torch.Tensor | None = None
        if ctx.mamba_ckpt_window > 1:
            # Spec decoding: the manager allocates `window - 1` checkpoint
            # blocks right after the positional state block, so checkpoint t
            # is the row entry `state_offsets + t`.
            ckpt_offsets = state_index + torch.arange(
                ctx.mamba_ckpt_window,
                device=state_offsets.device,
                dtype=state_offsets.dtype,
            ).unsqueeze(0)
            # A row too short for the group would alias checkpoints onto one
            # block; clamp to the null block and let the caller zero those
            # columns via `in_row`.
            in_row = (ckpt_offsets < target_num_blocks) & is_active
            ckpt_index = torch.where(
                in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets)
            )

        plan = (state_index, ckpt_index, in_row)
        ctx.mamba_row_plans[key] = plan
        return plan

    def _block_table_geometry(
        self, ctx: AttentionMetadataBuilderContext
    ) -> tuple[Any, int]:
        """The (block_table_obj, target_num_blocks) this group's build() uses.

        Shared with `stage_block_table_uploads`, which must pack each group's
        rows with exactly the width the group's build() will slice back out.
        """
        runner = self.runner
        block_table_obj = runner.input_batch.block_table[self.kv_cache_group_id]
        if ctx.use_max_model_len:
            target_num_blocks = block_table_obj.max_num_blocks_per_req
        else:
            assert runner.most_model_len is not None
            spec = self.kv_cache_spec
            first_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            if isinstance(first_spec, AttentionSpec):
                target_num_blocks = attention_block_table_width(
                    block_table_obj,
                    self.kv_cache_spec,
                    self.vllm_config,
                    runner.most_model_len,
                )
            else:
                target_num_blocks = cdiv(runner.most_model_len, self.target_block_size)
            if self.is_mamba_group and ctx.mamba_ckpt_window > 1:
                # Speculative decoding: the manager appends
                # `num_speculative_blocks` (= window - 1) checkpoint blocks
                # after the request's positional state block, so a mamba row
                # is that much wider than the positional part alone. This
                # mirrors `MambaSpec.max_num_blocks_per_req`, which the
                # use_max_model_len branch above already gets for free.
                # Without it the row stops short of the checkpoint group and
                # `in_row` below silently redirects checkpoints to the null
                # block, costing rollback accuracy with no error.
                target_num_blocks = min(
                    target_num_blocks + ctx.mamba_ckpt_window - 1,
                    block_table_obj.max_num_blocks_per_req,
                )
        return block_table_obj, target_num_blocks

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        target_num_reqs = common_attn_metadata.num_reqs
        block_table_obj, target_num_blocks = self._block_table_geometry(ctx)

        # `position_ids` is only used for the dummy run in dummy runs, where we
        # want to use fixed position IDs instead of copying from the CPU tensor
        # that gets updated every step.
        staged = (
            ctx.staged_block_tables.get(self.kv_cache_group_id)
            if ctx.staged_block_tables is not None
            else None
        )
        if staged is not None:
            # The walk pre-uploaded every group's table in one batched
            # transfer; see `stage_block_table_uploads`.
            block_tables_dev = staged
            input_positions = (
                ctx.position_ids_override
                if ctx.position_ids_override is not None
                else runner.position_ids
            )
        elif ctx.position_ids_override is not None:
            block_tables_dev = torch.zeros(
                (target_num_reqs * target_num_blocks,), dtype=torch.int32
            ).to(runner.device)
            input_positions = ctx.position_ids_override
        else:
            # Per-call buffer: a later chunk or draft walk must not overwrite a
            # table whose upload is still in flight (see `stage_h2d_upload`).
            flat = zeros_h2d_source(runner, target_num_reqs * target_num_blocks)
            block_tables = flat.view(target_num_reqs, target_num_blocks)
            # PCP currently only read block table inside pcpmetadata
            # TODO(weiyulin): Evaluate if we want to derive split/replicated block
            # tables for pcpmanager from md.blocktable
            if runner.pcp_manager is None:
                source_block_tables = block_table_obj.get_cpu_tensor()
                block_tables[: ctx.num_reqs, :target_num_blocks] = source_block_tables[
                    ctx.start_index : ctx.start_index + ctx.num_reqs, :target_num_blocks
                ]
            # Flatten on CPU before H2D to avoid device-side
            # as_strided/reshape materialization on every decode step.
            block_tables_dev = stage_h2d_upload(runner, flat)
            input_positions = runner.position_ids

        mamba_ckpt_indices = None
        mamba_prefill_ckpt_indices = None
        if not self.is_mamba_group:
            mamba_state_indices = None
        elif ctx.mamba_state_indices is not None:
            # Default compact-mamba path: the runner provides a per-request
            # physical slot id from the compact slot pool.
            mamba_state_indices = ctx.mamba_state_indices
        elif runner._unified_kv_layout:
            # The pool keys mamba state by vLLM block ids in every cache
            # mode (it has no compact slot pool). Derive the current physical
            # state
            # slot on device from the zero-padded `block_tables_dev` (padded
            # tail rows resolve to the null block, never a stale id — the
            # GDN op scans the full length every step), avoiding a
            # D2H -> CPU gather -> H2D dependency.
            staged_products = (
                ctx.staged_mamba_products.get(self.kv_cache_group_id)
                if ctx.staged_mamba_products is not None
                else None
            )
            if staged_products is not None:
                # Derived in the single `_staged_walk_products_compiled`
                # program alongside the staged tables.
                (
                    mamba_state_indices,
                    mamba_ckpt_indices,
                    mamba_prefill_ckpt_indices,
                ) = staged_products
            else:
                block_tables_2d = block_tables_dev.reshape(
                    target_num_reqs, target_num_blocks
                )
                # Everything but the gathers is shared with the other mamba
                # groups; see `_mamba_row_plan`.
                state_index, ckpt_index, ckpt_in_row = self._mamba_row_plan(
                    ctx, target_num_blocks
                )
                gathered_state_indices = torch.gather(
                    block_tables_2d,
                    dim=1,
                    index=state_index,
                ).squeeze(1)
                mamba_state_indices = torch.where(
                    ctx.seq_lens > 0,
                    gathered_state_indices,
                    torch.zeros_like(gathered_state_indices),
                )
                if ckpt_index is not None:
                    mamba_ckpt_indices = torch.gather(
                        block_tables_2d,
                        dim=1,
                        index=ckpt_index,
                    ) * cast(torch.Tensor, ckpt_in_row)
                if (
                    ctx.mamba_prefill_ckpt_offsets is not None
                    and self.prefill_ckpt_spec_blocks is not None
                ):
                    mamba_prefill_ckpt_indices = _gather_prefill_ckpt_ids(
                        block_tables_2d,
                        state_index.squeeze(1),
                        ctx.mamba_prefill_ckpt_offsets,
                        self.prefill_ckpt_spec_blocks,
                    )
            if ctx.unified_mamba_state_indices is not None:
                # Spec decode: expose this group's state blocks for the
                # post-sampling read-offset scatter.
                ctx.unified_mamba_state_indices.append(mamba_state_indices)
        else:
            mamba_state_indices = None

        if self.is_mamba_group:
            mamba_slot_read_offsets = ctx.mamba_slot_read_offsets
            mamba_request_distribution = ctx.mamba_request_distribution
        else:
            mamba_slot_read_offsets = None
            mamba_request_distribution = None
        mamba_prefill_ckpt_offsets = (
            ctx.mamba_prefill_ckpt_offsets if self.has_prefill_ckpt_blocks else None
        )
        if mamba_prefill_ckpt_offsets is not None:
            assert mamba_prefill_ckpt_indices is not None, (
                "Prefill checkpoints need the unified KV pool."
            )

        pcp_metadata = (
            None
            if runner.pcp_manager is None
            else runner.pcp_manager.build_metadata(
                runner=runner,
                block_table_obj=block_table_obj,
                target_num_reqs=target_num_reqs,
                target_num_blocks=target_num_blocks,
            )
        )

        metadata = AttentionMetadata(
            input_positions=input_positions,
            block_tables=block_tables_dev,
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
            mamba_state_indices=mamba_state_indices,
            mamba_slot_read_offsets=mamba_slot_read_offsets,
            mamba_ckpt_indices=mamba_ckpt_indices,
            mamba_request_distribution=mamba_request_distribution,
            pcp_metadata=pcp_metadata,
            mamba_prefill_ckpt_offsets=mamba_prefill_ckpt_offsets,
            mamba_prefill_ckpt_indices=mamba_prefill_ckpt_indices,
            sequence_layout_kind=ctx.sequence_layout_descriptor.kind.value,
            sequence_layout_protocol=(ctx.sequence_layout_descriptor.protocol),
            sequence_layout_version=ctx.sequence_layout_descriptor.version,
        )
        metadata.query_start_loc_cpu = ctx.query_start_loc_cpu
        metadata.seq_lens_cpu = ctx.seq_lens_cpu
        metadata.prompt_lens_cpu = ctx.prompt_lens_cpu
        return metadata


def stage_block_table_uploads(
    runner: Any,
    ctx: AttentionMetadataBuilderContext,
    num_reqs_padded: int,
    gids: set[int] | None = None,
) -> None:
    """One batched H2D transfer for every block table a walk will build.

    Packs all participating groups' rows into one CPU buffer owned by this
    walk, transfers once, and hands each group's build() a flat view of its
    region via `ctx.staged_block_tables` (instead of one small per-group H2D
    per build). `gids` limits staging to a subset of groups; None stages every
    group. Dummy runs stage the same layout zero-filled so warmup dispatches
    the program sequence serving will. Views die with the per-chunk context.
    """
    if not runner.attn_groups:
        # Pre-initialization dummy runs build metadata before the KV cache
        # groups exist; each build() synthesizes its own table there.
        return

    # Walk in the same order the builds will, deduping shared group ids.
    entries: list[tuple[int, Any, int]] = []
    entry_builders: list[Any] = []
    seen: set[int] = set()
    for group_list in runner.attn_groups:
        for group in group_list:
            builder = group.metadata_builders[0]
            gid = builder.kv_cache_group_id
            if gid in seen or (gids is not None and gid not in gids):
                continue
            seen.add(gid)
            block_table_obj, target_num_blocks = builder._block_table_geometry(ctx)
            entries.append((gid, block_table_obj, target_num_blocks))
            entry_builders.append(builder)
    if not entries:
        return

    # Mirror of build()'s branch order: the compact path
    # (ctx.mamba_state_indices) wins, otherwise the unified layout derives
    # per-group indices — which is what this precompute batches.
    unified_mamba = ctx.mamba_state_indices is None and runner._unified_kv_layout
    mamba_geom: list[tuple[int, int, int, int | None]] = []
    mamba_entry_indices: list[int] = []
    if unified_mamba:
        for i, builder in enumerate(entry_builders):
            if builder.is_mamba_group:
                mamba_geom.append(
                    (
                        i,
                        entries[i][2],
                        builder.target_block_size,
                        builder.prefill_ckpt_spec_blocks,
                    )
                )
                mamba_entry_indices.append(i)

    # The PCP manager's layers read block tables inside pcp_metadata.
    fill_rows = ctx.position_ids_override is None and runner.pcp_manager is None

    if ctx.start_index > 0:
        runner._block_tables_dirty = True
        runner._block_tables_append_only = False
    bt_cache_key = (
        ctx.start_index,
        ctx.num_reqs,
        num_reqs_padded,
        ctx.use_max_model_len,
        tuple((gid, tnb) for gid, _, tnb in entries),
    )
    can_cache_bt = (
        ctx.position_ids_override is None
        and (not mamba_geom or ctx.mamba_ckpt_window <= 1)
        and ctx.mamba_prefill_ckpt_offsets is None
        and gids is None
        and ctx.start_index == 0
    )
    if (
        can_cache_bt
        and not getattr(runner, "_block_tables_dirty", True)
        and getattr(runner, "_cached_bt_key", None) == bt_cache_key
        and getattr(runner, "_cached_staged_block_tables", None) is not None
    ):
        ctx.staged_block_tables = runner._cached_staged_block_tables
        ctx.staged_mamba_products = getattr(
            runner, "_cached_staged_mamba_products", None
        )
        return

    if fill_rows and ctx.num_reqs > 0:
        row_bucket = _select_stage_row_bucket(ctx.num_reqs, num_reqs_padded)
        col_div = _select_stage_col_div(entries, ctx)
    else:
        row_bucket = num_reqs_padded
        col_div = 1

    upload_cols = [
        min(tnb, max(1, cdiv(tnb, col_div))) for _, _, tnb in entries
    ]
    total = sum(row_bucket * cols_g for cols_g in upload_cols)
    full_total = sum(num_reqs_padded * tnb for _, _, tnb in entries)

    # Per-walk upload buffer: TorchTPU reads the CPU source until the async
    # H2D drains, so a later chunk or draft walk must not refill a shared
    # scratch (see `stage_h2d_upload`). The persistent
    # `runner._block_table_stage_cpu` scratch is therefore never uploaded
    # directly; it is the host shadow of the last cacheable staging that the
    # append-only incremental path patches in place before copying it into
    # the walk's own buffer.
    flat = zeros_h2d_source(runner, total)

    stage_layout_key = (
        row_bucket,
        tuple(upload_cols),
        ctx.num_reqs,
        tuple(runner.input_batch.req_ids[: ctx.num_reqs])
        if getattr(runner, "input_batch", None) is not None
        else (),
        tuple(id(bt) for _, bt, _ in entries),
    )
    shadow: torch.Tensor | None = None
    if can_cache_bt and fill_rows:
        shadow = getattr(runner, "_block_table_stage_cpu", None)
        if shadow is None or shadow.numel() < full_total:
            shadow = torch.zeros(
                (full_total,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
            )
            runner._block_table_stage_cpu = shadow
            runner._block_tables_append_only = False
    prev_nbpr_map = getattr(runner, "_bt_prev_nbpr", None)
    use_incremental = (
        shadow is not None
        and getattr(runner, "_block_tables_append_only", False)
        and getattr(runner, "_cached_bt_stage_layout", None) == stage_layout_key
        and prev_nbpr_map is not None
        and all(
            getattr(bt, "num_blocks_per_row", None) is not None
            and gid in prev_nbpr_map
            for gid, bt, _ in entries
        )
    )
    if use_incremental:
        assert shadow is not None and prev_nbpr_map is not None
        shadow_flat = shadow[:total]
        shadow_np = shadow_flat.numpy()
        offset = 0
        for (gid, block_table_obj, _), cols_g in zip(entries, upload_cols):
            prev_nbpr = prev_nbpr_map[gid]
            cur_nbpr = block_table_obj.num_blocks_per_row[: ctx.num_reqs]
            changed = (cur_nbpr != prev_nbpr).nonzero()[0]
            if changed.size > 0:
                rows_np = shadow_np[offset : offset + row_bucket * cols_g].reshape(
                    row_bucket, cols_g
                )
                src_np = block_table_obj.get_numpy_array()
                for r in changed:
                    c0 = int(prev_nbpr[r])
                    c1 = min(cols_g, int(cur_nbpr[r]))
                    if c0 < c1:
                        rows_np[r, c0:c1] = src_np[r, c0:c1]
                prev_nbpr[:] = cur_nbpr
            offset += row_bucket * cols_g
        flat.copy_(shadow_flat)
    else:
        if not can_cache_bt:
            runner._block_tables_dirty = True
            runner._block_tables_append_only = False
            runner._cached_bt_stage_layout = None
        if shadow is not None:
            # Rebuild the shadow from scratch so later append-only steps can
            # patch it; `flat` is already zeroed.
            target = shadow[:total]
            target.zero_()
        else:
            target = flat
        if fill_rows:
            offset = 0
            for (_, block_table_obj, _), cols_g in zip(entries, upload_cols):
                rows = target[offset : offset + row_bucket * cols_g].view(
                    row_bucket, cols_g
                )
                source = block_table_obj.get_cpu_tensor()
                rows[: ctx.num_reqs] = source[
                    ctx.start_index : ctx.start_index + ctx.num_reqs, :cols_g
                ]
                offset += row_bucket * cols_g
        if target is not flat:
            flat.copy_(target)
    # Dummy runs keep the zeroed buffer: same staged layout, null blocks.

    staged_dev = stage_h2d_upload(runner, flat)

    # One compiled program splits and zero-pads the staged tensor into static
    # per-group `[num_reqs_padded * target_num_blocks]` tables and derives
    # every unified-pool mamba group's state/checkpoint indices.
    table_geom: list[tuple[int, int, int, int, int]] = []
    offset = 0
    for (_, _, target_num_blocks), cols_g in zip(entries, upload_cols):
        table_geom.append(
            (offset, row_bucket, cols_g, num_reqs_padded, target_num_blocks)
        )
        offset += row_bucket * cols_g

    if (
        len(entries) == 1
        and not mamba_geom
        and row_bucket == num_reqs_padded
        and upload_cols[0] == entries[0][2]
    ):
        gid, _, target_num_blocks = entries[0]
        length = num_reqs_padded * target_num_blocks
        ctx.staged_block_tables = {gid: staged_dev[:length]}
        if can_cache_bt:
            runner._cached_bt_key = bt_cache_key
            runner._cached_bt_stage_layout = stage_layout_key
            runner._cached_staged_block_tables = ctx.staged_block_tables
            runner._cached_staged_mamba_products = None
            runner._bt_prev_nbpr = {
                g: bt.num_blocks_per_row[: ctx.num_reqs].copy()
                for g, bt, _ in entries
                if getattr(bt, "num_blocks_per_row", None) is not None
            }
            runner._block_tables_dirty = False
            runner._block_tables_append_only = shadow is not None
        return

    walk = (
        _staged_walk_products_compiled
        if staged_dev.device.type == "tpu"
        else _staged_walk_products
    )
    products = walk(
        staged_dev,
        ctx.seq_lens,
        tuple(table_geom),
        tuple(mamba_geom),
        ctx.mamba_ckpt_window,
        ctx.mamba_prefill_ckpt_offsets,
    )

    n_tables = len(entries)
    n_mamba = len(mamba_geom)
    ctx.staged_block_tables = {
        gid: products[i] for i, (gid, _, _) in enumerate(entries)
    }
    if n_mamba:
        states = products[n_tables : n_tables + n_mamba]
        rest = products[n_tables + n_mamba :]
        if ctx.mamba_ckpt_window > 1:
            ckpts = rest[:n_mamba]
            rest = rest[n_mamba:]
        else:
            ckpts = (None,) * n_mamba
        # Prefill checkpoint ids follow, one per group that declares them.
        prefill_ckpts = iter(rest)
        ctx.staged_mamba_products = {}
        for j, ei in enumerate(mamba_entry_indices):
            prefill_ckpt = (
                next(prefill_ckpts)
                if ctx.mamba_prefill_ckpt_offsets is not None
                and mamba_geom[j][3] is not None
                else None
            )
            ctx.staged_mamba_products[entries[ei][0]] = (
                states[j],
                ckpts[j],
                prefill_ckpt,
            )
    if can_cache_bt:
        runner._cached_bt_key = bt_cache_key
        runner._cached_bt_stage_layout = stage_layout_key
        runner._cached_staged_block_tables = ctx.staged_block_tables
        runner._cached_staged_mamba_products = ctx.staged_mamba_products
        runner._bt_prev_nbpr = {
            g: bt.num_blocks_per_row[: ctx.num_reqs].copy()
            for g, bt, _ in entries
            if getattr(bt, "num_blocks_per_row", None) is not None
        }
        runner._block_tables_dirty = False
        runner._block_tables_append_only = shadow is not None


@torch.no_grad()
def precompile_staged_block_table_uploads(runner: Any) -> None:
    """Warm `_staged_walk_products_compiled` across all 2D `(row_bucket, col_div)`
    variants under `no_grad` so no serving-time recompile occurs inside
    `execute_model`."""
    if not runner.attn_groups or runner.device.type != "tpu":
        return
    from vllm_torchtpu.utils import synchronize_tensors

    modes = [True]
    if runner.most_model_len is not None:
        modes.append(False)

    for use_max_model_len in modes:
        num_reqs_padded = (
            runner.num_reqs_max_model_len
            if use_max_model_len
            else runner.num_reqs_most_model_len
        )
        assert num_reqs_padded is not None
        dummy_seq_lens = torch.zeros(
            (num_reqs_padded,), dtype=torch.int32, device=runner.device
        )
        dummy_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs_padded,
            start_index=0,
            use_max_model_len=use_max_model_len,
            seq_lens=dummy_seq_lens,
            query_start_loc=torch.zeros(
                (num_reqs_padded + 1,), dtype=torch.int32, device=runner.device
            ),
            request_distribution=torch.zeros(
                (3,), dtype=torch.int32, device=runner.device
            ),
            mamba_ckpt_window=runner._mamba_ckpt_window,
        )
        entries: list[tuple[int, Any, int]] = []
        entry_builders: list[Any] = []
        seen: set[int] = set()
        for group_list in runner.attn_groups:
            for group in group_list:
                builder = group.metadata_builders[0]
                gid = builder.kv_cache_group_id
                if gid in seen:
                    continue
                seen.add(gid)
                block_table_obj, tnb = builder._block_table_geometry(dummy_ctx)
                entries.append((gid, block_table_obj, tnb))
                entry_builders.append(builder)
        if not entries:
            continue

        unified_mamba = (
            not runner._has_mamba_state and runner._unified_kv_layout
        )
        mamba_geom: list[tuple[int, int, int, int | None]] = []
        if unified_mamba:
            for i, builder in enumerate(entry_builders):
                if builder.is_mamba_group:
                    mamba_geom.append(
                        (
                            i,
                            entries[i][2],
                            builder.target_block_size,
                            builder.prefill_ckpt_spec_blocks,
                        )
                    )
        # Groups declaring prefill checkpoints always receive per-request
        # offsets on main-model steps, which adds one tensor input to the
        # compiled walk; warm that variant too.
        dummy_prefill_ckpt_offsets = (
            torch.zeros((num_reqs_padded,), dtype=torch.int32, device=runner.device)
            if any(b.prefill_ckpt_spec_blocks is not None for b in entry_builders)
            else None
        )

        row_buckets = sorted(
            {
                *(rb for rb in _STAGE_BT_ROW_BUCKETS if rb < num_reqs_padded),
                num_reqs_padded,
            }
        )
        seen_shapes: set[tuple[int, tuple[int, ...]]] = set()
        for rb in row_buckets:
            for col_div in _STAGE_BT_COL_DIVS:
                upload_cols = tuple(
                    min(tnb, max(1, cdiv(tnb, col_div)))
                    for _, _, tnb in entries
                )
                shape_key = (rb, upload_cols)
                if shape_key in seen_shapes:
                    continue
                seen_shapes.add(shape_key)
                if (
                    len(entries) == 1
                    and not mamba_geom
                    and rb == num_reqs_padded
                    and upload_cols[0] == entries[0][2]
                ):
                    continue
                table_geom: list[tuple[int, int, int, int, int]] = []
                offset = 0
                for (_, _, tnb), cols_g in zip(entries, upload_cols):
                    table_geom.append(
                        (offset, rb, cols_g, num_reqs_padded, tnb)
                    )
                    offset += rb * cols_g
                dummy_staged = torch.zeros(
                    (offset,), dtype=torch.int32, device=runner.device
                )
                products = _staged_walk_products_compiled(
                    dummy_staged,
                    dummy_seq_lens,
                    tuple(table_geom),
                    tuple(mamba_geom),
                    runner._mamba_ckpt_window,
                    dummy_prefill_ckpt_offsets,
                )
                synchronize_tensors(list(products))
