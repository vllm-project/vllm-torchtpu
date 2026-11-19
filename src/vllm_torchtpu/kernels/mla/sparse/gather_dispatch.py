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
"""Index-preparation mode selection and chunk KV gather dispatch for SparseMLA."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import (
    dsa_gather,
    index_prep,
    native_sc_gather,
)
from vllm_torchtpu.kernels.mla.sparse.config import (
    FUSED_PREP_MAX_TOKENS,
    GATHER_ATOM_ROWS,
    CappedWindowConfig,
    GatherConfig,
    SparseMlaConfig,
    cdiv,
    chunk_kernel_batch_size,
    largest_divisor,
    live_group_supported,
    mixed_fused_prep_supported,
)
from vllm_torchtpu.kernels.mla.sparse.testing_faults import (
    NO_FAULT_HOOKS,
    FaultHooks,
)

KvWriteOperand = jax.Array | tuple[jax.Array, jax.Array, jax.Array]
PreGatheredChunk = tuple[
    jax.Array, jax.Array, tuple[jax.Array, jax.Array] | None
]


@dataclasses.dataclass(frozen=True)
class PreparedIndices:
    """Outputs of top-level SparseMLA index preparation across all chunks."""

    kv_lens: jax.Array
    physical_indices: jax.Array | None
    window_indices_all: Any
    chunk_window_rows: int
    fused_prep: bool
    group_tab: jax.Array | None
    group_num_live: jax.Array | None

    def window_indices_for_chunk(self, chunk_idx: int) -> jax.Array | None:
        """Returns the window-index buffer consumed by chunk ``chunk_idx``."""
        if self.fused_prep or self.window_indices_all is None:
            return self.window_indices_all
        if isinstance(self.window_indices_all, tuple):
            return self.window_indices_all[chunk_idx]
        start = chunk_idx * self.chunk_window_rows
        return self.window_indices_all[start : start + self.chunk_window_rows]

    def group_num_live_for_chunk(self, chunk_idx: int) -> jax.Array | None:
        """Returns the live-group index bound for chunk ``chunk_idx``."""
        if self.group_num_live is None:
            return None
        return self.group_num_live[chunk_idx]


@dataclasses.dataclass(frozen=True)
class GatheredChunk:
    """Gathered NOPE/ROPE KV buffers and (possibly updated) cache tensors."""

    nope: jax.Array
    rope: jax.Array
    live_views: tuple[jax.Array, jax.Array] | None
    cache_kv_nope: jax.Array
    cache_kv_rope: jax.Array


def _count_kv_lens(
    topk_indices: jax.Array,
    token_end: jax.Array,
    *,
    zero_dcp_padding: bool,
) -> jax.Array:
    """Counts non-padding top-k slots per token, optionally zeroing padding rows."""
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)
    if zero_dcp_padding:
        kv_lens = jnp.where(
            jnp.arange(kv_lens.shape[0], dtype=jnp.int32) < token_end,
            kv_lens,
            0,
        )
    return kv_lens


def _can_use_window_prep(
    resolved_topk_cap: int | None,
    page_indices: jax.Array | None,
    num_tokens: int,
    uniform_chunk: int,
) -> bool:
    """Whether every chunk shares one static size for capped window-major prep."""
    return (
        resolved_topk_cap is not None
        and page_indices is not None
        and num_tokens % uniform_chunk == 0
    )


def _can_use_fused_decode_prep(
    capped: CappedWindowConfig,
    window_prep: bool,
    num_chunks: int,
    num_tokens: int,
) -> bool:
    """Whether single-chunk decode fused prep applies."""
    return (
        capped.fused_window_prep
        and window_prep
        and num_chunks == 1
        and num_tokens <= FUSED_PREP_MAX_TOKENS
    )


def _can_use_mixed_live(
    capped: CappedWindowConfig,
    *,
    fused_prep: bool,
    window_prep: bool,
    num_tokens: int,
    uniform_chunk: int,
    prep_batch: int,
    resolved_topk_cap: int | None,
) -> bool:
    """Whether mixed/prefill live-group prep applies."""
    return (
        not fused_prep
        and window_prep
        and capped.mixed_live_gather
        and num_tokens > FUSED_PREP_MAX_TOKENS
        and resolved_topk_cap is not None
        and live_group_supported(uniform_chunk, prep_batch, resolved_topk_cap)
    )


def _can_use_mixed_fused_prep(
    capped: CappedWindowConfig,
    *,
    use_mixed_live: bool,
    num_tokens: int,
    topk: int,
    uniform_chunk: int,
    prep_batch: int,
    resolved_topk_cap: int | None,
) -> bool:
    """Whether fused live-group window prep applies to the mixed bucket."""
    return (
        use_mixed_live
        and capped.mixed_fused_prep
        and resolved_topk_cap is not None
        and mixed_fused_prep_supported(
            num_tokens, topk, uniform_chunk, prep_batch, resolved_topk_cap
        )
    )


def _prepare_fused_decode_indices(
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    token_end: jax.Array,
    *,
    topk_cap: int,
    page_size: int,
    total_num_pages: int,
    prep_small: bool,
    prep_batch: int,
    config: SparseMlaConfig,
) -> PreparedIndices:
    """Runs single-chunk fused decode window preparation."""
    gather_cfg = config.gather
    # Only the attention kernel reads kv_lens here: fused prep counts live
    # entries itself, so XLA may run this reduce in the gather's shadow.
    kv_lens = _count_kv_lens(
        topk_indices, token_end, zero_dcp_padding=config.dcp.is_dcp_mode
    )
    gather_atoms = (
        gather_cfg.small_chunk_gather_atoms
        if prep_small and gather_cfg.small_chunk_live_gather
        else gather_cfg.sparsecore_atoms_per_batch
    )
    window_indices_all = index_prep.prepare_capped_window_fused(
        topk_indices,
        page_indices,
        cu_q_lens,
        token_end,
        topk_cap=topk_cap,
        page_size=page_size,
        total_num_pages=total_num_pages,
        kernel_batch_size=prep_batch,
        gather_atoms=gather_atoms,
    )
    return PreparedIndices(
        kv_lens=kv_lens,
        physical_indices=None,
        window_indices_all=window_indices_all,
        chunk_window_rows=0,
        fused_prep=True,
        group_tab=None,
        group_num_live=None,
    )


def _prepare_mixed_live_indices(
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    token_end: jax.Array,
    *,
    topk: int,
    topk_cap: int,
    uniform_chunk: int,
    prep_batch: int,
    page_size: int,
    total_num_pages: int,
    use_mixed_fused_prep: bool,
    hooks: FaultHooks,
    prepare_live_fused_fn: Callable[..., Any] | None,
) -> PreparedIndices:
    """Runs fused or unfused mixed/prefill live-group window preparation."""
    chunk_window_rows = topk * uniform_chunk // 128
    # No is_dcp_mode padding zeroing here: live-group steps never take the
    # kv rungs it feeds, and mixed fused prep returns its own kv_lens, so
    # zeroing only the unfused branch would split them bit for bit.
    if use_mixed_fused_prep:
        live_fused_fn = (
            index_prep.prepare_capped_window_live_fused
            if prepare_live_fused_fn is None
            else prepare_live_fused_fn
        )
        window_indices_all, kv_lens, group_tab, group_num_live = live_fused_fn(
            topk_indices,
            page_indices,
            cu_q_lens,
            token_end,
            chunk_size=uniform_chunk,
            kernel_batch_size=prep_batch,
            topk_cap=topk_cap,
            page_size=page_size,
            total_num_pages=total_num_pages,
            **({"hooks": hooks} if hooks.live_prep is not None else {}),
        )
    else:
        kv_lens = jnp.sum(topk_indices != -1, axis=-1)
        live_windows = index_prep.capped_live_windows(
            kv_lens,
            token_end,
            chunk_size=uniform_chunk,
            kernel_batch_size=prep_batch,
            topk_cap=topk_cap,
            num_windows=topk // topk_cap,
        )
        group_tab, group_num_live = index_prep.live_group_table(
            kv_lens,
            token_end,
            chunk_size=uniform_chunk,
            kernel_batch_size=prep_batch,
            topk_cap=topk_cap,
            num_windows=topk // topk_cap,
        )
        window_indices_all = index_prep.prepare_capped_window_live(
            topk_indices,
            page_indices,
            cu_q_lens,
            live_windows,
            group_tab,
            chunk_size=uniform_chunk,
            kernel_batch_size=prep_batch,
            topk_cap=topk_cap,
            page_size=page_size,
            total_num_pages=total_num_pages,
        )
    return PreparedIndices(
        kv_lens=kv_lens,
        physical_indices=None,
        window_indices_all=window_indices_all,
        chunk_window_rows=chunk_window_rows,
        fused_prep=False,
        group_tab=group_tab,
        group_num_live=group_num_live,
    )


def _prepare_capped_window_indices(
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    token_end: jax.Array,
    *,
    topk: int,
    topk_cap: int,
    uniform_chunk: int,
    prep_batch: int,
    page_size: int,
    total_num_pages: int,
    is_dcp_mode: bool,
) -> PreparedIndices:
    """Runs plain capped window-major index preparation."""
    chunk_window_rows = topk // topk_cap * uniform_chunk
    kv_lens = _count_kv_lens(topk_indices, token_end, zero_dcp_padding=is_dcp_mode)
    live_windows = index_prep.capped_live_windows(
        kv_lens,
        token_end,
        chunk_size=uniform_chunk,
        kernel_batch_size=prep_batch,
        topk_cap=topk_cap,
        num_windows=topk // topk_cap,
    )
    window_indices_all = index_prep.prepare_capped_window_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        live_windows,
        chunk_size=uniform_chunk,
        topk_cap=topk_cap,
        page_size=page_size,
        total_num_pages=total_num_pages,
    )
    return PreparedIndices(
        kv_lens=kv_lens,
        physical_indices=None,
        window_indices_all=window_indices_all,
        chunk_window_rows=chunk_window_rows,
        fused_prep=False,
        group_tab=None,
        group_num_live=None,
    )


def _prepare_fallback_indices(
    topk_indices: jax.Array,
    page_indices: jax.Array | None,
    cu_q_lens: jax.Array,
    token_end: jax.Array,
    *,
    page_size: int,
    total_num_pages: int,
    is_dcp_mode: bool,
) -> PreparedIndices:
    """Runs full-width physical index preparation."""
    kv_lens, physical_indices = index_prep.prepare_gather_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        page_size=page_size,
        total_num_pages=total_num_pages,
        is_dcp_mode=is_dcp_mode,
        num_valid_tokens=token_end,
    )
    return PreparedIndices(
        kv_lens=kv_lens,
        physical_indices=physical_indices,
        window_indices_all=None,
        chunk_window_rows=0,
        fused_prep=False,
        group_tab=None,
        group_num_live=None,
    )


@dataclasses.dataclass(frozen=True)
class _PrepPlan:
    """Validated index-preparation flags for one ``prepare_attention_indices`` call."""

    window_prep: bool
    fused_prep: bool
    prep_small: bool
    prep_batch: int
    use_mixed_live: bool
    use_mixed_fused_prep: bool


def _plan_index_prep(
    *,
    num_tokens: int,
    topk: int,
    uniform_chunk: int,
    num_chunks: int,
    resolved_topk_cap: int | None,
    page_indices: jax.Array | None,
    config: SparseMlaConfig,
    capped: CappedWindowConfig,
    hooks: FaultHooks,
    has_kv_write: bool,
) -> _PrepPlan:
    """Resolves and validates which index-preparation strategy to execute."""
    gather_cfg = config.gather
    window_prep = _can_use_window_prep(
        resolved_topk_cap, page_indices, num_tokens, uniform_chunk
    )
    fused_prep = _can_use_fused_decode_prep(capped, window_prep, num_chunks, num_tokens)
    if fused_prep and not config.lse.mask_empty:
        raise ValueError("fused_window_prep requires lse_mask_empty")
    if has_kv_write and not fused_prep:
        raise ValueError("kv_write needs the fused-prep decode path (one gather)")

    prep_small, prep_batch = chunk_kernel_batch_size(
        uniform_chunk,
        config.attention_kernel_batch_size,
        gather_cfg.small_chunk_tokens,
        gather_cfg.small_chunk_batch_size,
    )
    use_mixed_live = _can_use_mixed_live(
        capped,
        fused_prep=fused_prep,
        window_prep=window_prep,
        num_tokens=num_tokens,
        uniform_chunk=uniform_chunk,
        prep_batch=prep_batch,
        resolved_topk_cap=resolved_topk_cap,
    )
    use_mixed_fused_prep = _can_use_mixed_fused_prep(
        capped,
        use_mixed_live=use_mixed_live,
        num_tokens=num_tokens,
        topk=topk,
        uniform_chunk=uniform_chunk,
        prep_batch=prep_batch,
        resolved_topk_cap=resolved_topk_cap,
    )
    if hooks.live_prep is not None and not use_mixed_fused_prep:
        raise ValueError("live_prep fault requires mixed_fused_prep")

    return _PrepPlan(
        window_prep=window_prep,
        fused_prep=fused_prep,
        prep_small=prep_small,
        prep_batch=prep_batch,
        use_mixed_live=use_mixed_live,
        use_mixed_fused_prep=use_mixed_fused_prep,
    )


def prepare_attention_indices(
    topk_indices: jax.Array,
    page_indices: jax.Array | None,
    cu_q_lens: jax.Array,
    token_end: jax.Array,
    *,
    num_tokens: int,
    page_size: int,
    total_num_pages: int,
    config: SparseMlaConfig,
    capped: CappedWindowConfig,
    hooks: FaultHooks = NO_FAULT_HOOKS,
    has_kv_write: bool = False,
    prepare_live_fused_fn: Callable[..., Any] | None = None,
) -> PreparedIndices:
    """Selects and executes the index-preparation strategy for an attention call."""
    topk = topk_indices.shape[-1]
    resolved_topk_cap = capped.topk_cap
    uniform_chunk = min(config.gather.chunk_size, num_tokens)
    plan = _plan_index_prep(
        num_tokens=num_tokens,
        topk=topk,
        uniform_chunk=uniform_chunk,
        num_chunks=cdiv(num_tokens, config.gather.chunk_size),
        resolved_topk_cap=resolved_topk_cap,
        page_indices=page_indices,
        config=config,
        capped=capped,
        hooks=hooks,
        has_kv_write=has_kv_write,
    )
    if plan.fused_prep:
        assert resolved_topk_cap is not None and page_indices is not None
        return _prepare_fused_decode_indices(
            topk_indices, page_indices, cu_q_lens, token_end,
            topk_cap=resolved_topk_cap, page_size=page_size,
            total_num_pages=total_num_pages, prep_small=plan.prep_small,
            prep_batch=plan.prep_batch, config=config,
        )
    if plan.use_mixed_live:
        assert resolved_topk_cap is not None and page_indices is not None
        return _prepare_mixed_live_indices(
            topk_indices, page_indices, cu_q_lens, token_end,
            topk=topk, topk_cap=resolved_topk_cap, uniform_chunk=uniform_chunk,
            prep_batch=plan.prep_batch, page_size=page_size,
            total_num_pages=total_num_pages,
            use_mixed_fused_prep=plan.use_mixed_fused_prep,
            hooks=hooks, prepare_live_fused_fn=prepare_live_fused_fn,
        )
    if plan.window_prep:
        assert resolved_topk_cap is not None and page_indices is not None
        return _prepare_capped_window_indices(
            topk_indices, page_indices, cu_q_lens, token_end,
            topk=topk, topk_cap=resolved_topk_cap, uniform_chunk=uniform_chunk,
            prep_batch=plan.prep_batch, page_size=page_size,
            total_num_pages=total_num_pages, is_dcp_mode=config.dcp.is_dcp_mode,
        )
    return _prepare_fallback_indices(
        topk_indices, page_indices, cu_q_lens, token_end,
        page_size=page_size, total_num_pages=total_num_pages,
        is_dcp_mode=config.dcp.is_dcp_mode,
    )


def apply_dcp_skip_windows(
    indices: jax.Array | None,
    kv_lens_slice: jax.Array,
    *,
    topk: int,
    sparsecore_layout: bool,
    sparsecore_atoms_per_batch: int,
) -> tuple[jax.Array | None, int, int]:
    """Rewrites dead DCP gather windows to ``SKIP_INDEX``.

    Points whole dead gather windows past each token's ``kv_len`` at
    ``SKIP_INDEX`` so the uncapped gather drops them instead of reading
    scattered pad rows.

    Returns:
        ``(indices, tc_num_row_subchunks, dcp_gather_atoms)``.
    """
    tc_num_row_subchunks = 32
    dcp_gather_atoms = sparsecore_atoms_per_batch
    gather_step = native_sc_gather.OCCURRENCES_PER_ATOM
    if topk >= gather_step and topk % gather_step == 0:
        tc_steps_per_window = largest_divisor(topk // gather_step, 4)
        tc_num_row_subchunks = tc_steps_per_window * 4
        dcp_gather_atoms = largest_divisor(
            topk // gather_step, sparsecore_atoms_per_batch
        )
        gather_window_size = (
            dcp_gather_atoms * gather_step
            if sparsecore_layout
            else tc_steps_per_window * gather_step
        )
        window_start = (
            jnp.arange(topk, dtype=jnp.int32) // gather_window_size
        ) * gather_window_size
        if indices is not None:
            indices = jnp.where(
                window_start[None, :] >= kv_lens_slice[:, None],
                dsa_gather.SKIP_INDEX,
                indices,
            )
    return indices, tc_num_row_subchunks, dcp_gather_atoms


def gather_capped_sparsecore(
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    indices: jax.Array | None,
    window_indices: jax.Array | None,
    kv_lens_slice: jax.Array,
    *,
    chunk_size: int,
    topk: int,
    topk_cap: int,
    capped_windows: int,
    kernel_batch_size: int,
    small_chunk: bool,
    live_tokens: jax.Array,
    gather_cfg: GatherConfig,
    capped_cfg: CappedWindowConfig,
    gather_live_in_indices: bool = False,
    group_tab: jax.Array | None = None,
    group_num_live: jax.Array | None = None,
    kv_write: KvWriteOperand | None = None,
    hooks: FaultHooks = NO_FAULT_HOOKS,
) -> GatheredChunk:
    """Gathers a chunk using the capped window-major native SparseCore path."""
    gather_atoms = gather_cfg.sparsecore_atoms_per_batch
    if small_chunk and gather_cfg.small_chunk_live_gather:
        gather_atoms = gather_cfg.small_chunk_gather_atoms

    live_kwargs: dict[str, Any]
    if gather_live_in_indices:
        # Fused prep 128-lane form: flat bitcast, live vector follows rows.
        live_kwargs = {
            "live_vec_offset": chunk_size * topk,
            "linear_indices": True,
        }
    elif group_tab is not None:
        # Live-group prep: 128-lane linear rows, live bound from table.
        assert window_indices is not None and group_num_live is not None
        live_kwargs = {
            "num_live_indices": group_num_live,
            "linear_indices": True,
        }
    else:
        token_windows = jnp.where(
            jnp.arange(chunk_size) < live_tokens,
            (jnp.maximum(kv_lens_slice, 1) + topk_cap - 1) // topk_cap,
            1,
        )
        live_windows = jnp.clip(jnp.max(token_windows), 1, capped_windows)
        num_live_capped = (
            live_windows - 1
        ) * chunk_size * topk_cap + jnp.maximum(0, live_tokens) * topk_cap
        live_kwargs = {"num_live_indices": num_live_capped}
        if capped_cfg.capped_live_rows:
            window_start = topk_cap * jnp.arange(capped_windows, dtype=jnp.int32)
            live_slots = jnp.clip(
                kv_lens_slice[None, :] - window_start[:, None],
                0,
                topk_cap,
            )
            live_slots = jnp.where(
                jnp.arange(chunk_size) < live_tokens, live_slots, 0
            )
            live_kwargs["row_live_atoms"] = (
                (live_slots + GATHER_ATOM_ROWS - 1) // GATHER_ATOM_ROWS
            ).reshape(-1)

    if window_indices is None:
        assert indices is not None
        window_indices = (
            indices.reshape(chunk_size, capped_windows, topk_cap)
            .transpose(1, 0, 2)
            .reshape(capped_windows * chunk_size, topk_cap)
        )

    gathered = native_sc_gather.dsa_gather_native_sc(
        cache_kv_nope,
        cache_kv_rope,
        window_indices,
        out_size=chunk_size * topk,
        atoms_per_batch=gather_atoms,
        wait_mode=gather_cfg.sparsecore_wait_mode.value,
        serve_pipelined=gather_cfg.sparsecore_pipelined,
        live_multiple=kernel_batch_size
        * (GATHER_ATOM_ROWS if group_tab is not None else topk_cap),
        kv_write=kv_write,
        write_fault=hooks.kv_write,
        packed_rope=gather_cfg.packed_rope,
        **live_kwargs,
    )
    gathered_nope, gathered_rope = gathered[:2]
    if kv_write is not None:
        cache_kv_nope, cache_kv_rope = gathered[2:]
    gathered_nope = gathered_nope.reshape(
        capped_windows * chunk_size, topk_cap, 128
    )
    gathered_rope = gathered_rope.reshape(
        capped_windows * chunk_size,
        topk_cap // native_sc_gather.ROPE_PACKING,
        128,
    )
    live_views: tuple[jax.Array, jax.Array] | None = None
    if group_tab is not None:
        block = kernel_batch_size * GATHER_ATOM_ROWS
        live_views = (
            gathered_nope.reshape(
                chunk_size * topk // block,
                kernel_batch_size,
                GATHER_ATOM_ROWS,
                128,
            ),
            gathered_rope.reshape(
                chunk_size * topk // block,
                kernel_batch_size,
                GATHER_ATOM_ROWS // native_sc_gather.ROPE_PACKING,
                128,
            ),
        )
    return GatheredChunk(
        nope=gathered_nope,
        rope=gathered_rope,
        live_views=live_views,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
    )


def gather_uncapped_sparsecore(
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    indices: jax.Array,
    *,
    chunk_size: int,
    topk: int,
    kernel_batch_size: int,
    small_chunk: bool,
    num_valid_indices: jax.Array,
    gather_cfg: GatherConfig,
    is_dcp_mode: bool,
    dcp_gather_atoms: int,
) -> GatheredChunk:
    """Gathers a chunk using the uncapped native SparseCore path."""
    live_kwargs: dict[str, Any] = {}
    gather_atoms = gather_cfg.sparsecore_atoms_per_batch
    if small_chunk and gather_cfg.small_chunk_live_gather:
        gather_atoms = gather_cfg.small_chunk_gather_atoms
        live_kwargs = {
            "num_live_indices": num_valid_indices,
            "live_multiple": kernel_batch_size * topk,
        }
    elif gather_cfg.live_gather_all_chunks:
        live_kwargs = {
            "num_live_indices": num_valid_indices,
            "live_multiple": kernel_batch_size * topk,
        }
    gathered_nope, gathered_rope = native_sc_gather.dsa_gather_native_sc(
        cache_kv_nope,
        cache_kv_rope,
        indices,
        out_size=chunk_size * topk,
        atoms_per_batch=dcp_gather_atoms if is_dcp_mode else gather_atoms,
        wait_mode=gather_cfg.sparsecore_wait_mode.value,
        serve_pipelined=gather_cfg.sparsecore_pipelined,
        **({"is_dcp_mode": True} if is_dcp_mode else {}),
        packed_rope=gather_cfg.packed_rope,
        **live_kwargs,
    )
    return GatheredChunk(
        nope=gathered_nope.reshape(chunk_size, topk, 128),
        rope=gathered_rope.reshape(
            chunk_size, topk // native_sc_gather.ROPE_PACKING, 128
        ),
        live_views=None,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
    )


def gather_tensorcore(
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    indices: jax.Array,
    *,
    chunk_size: int,
    topk: int,
    num_valid_indices: jax.Array,
    is_dcp_mode: bool,
    tc_num_row_subchunks: int,
) -> GatheredChunk:
    """Gathers a chunk using the TensorCore-layout ``dsa_gather`` kernel."""
    gathered_nope, gathered_rope = dsa_gather.dsa_gather(
        cache_kv_nope,
        cache_kv_rope,
        indices.reshape(-1),
        num_valid_indices=num_valid_indices,
        # Non-DCP calls keep the exact baseline kwargs signature.
        **(
            {"num_row_subchunks": tc_num_row_subchunks, "is_dcp_mode": True}
            if is_dcp_mode
            else {}
        ),
    )
    return GatheredChunk(
        nope=gathered_nope.reshape(
            chunk_size,
            topk * kv_cache_utils.WORD_BYTES,
            kv_cache_utils.TILE_LANE_BYTES,
        ),
        rope=gathered_rope.reshape(chunk_size, topk, -1),
        live_views=None,
        cache_kv_nope=cache_kv_nope,
        cache_kv_rope=cache_kv_rope,
    )


def gather_chunk_kv(
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    kv_lens: jax.Array,
    physical_indices: jax.Array | None,
    *,
    start_pos: int,
    end_pos: int,
    topk: int,
    kernel_batch_size: int,
    small_chunk: bool,
    batch_end: jax.Array,
    config: SparseMlaConfig,
    hooks: FaultHooks = NO_FAULT_HOOKS,
    window_indices: jax.Array | None = None,
    gather_live_in_indices: bool = False,
    group_tab: jax.Array | None = None,
    group_num_live: jax.Array | None = None,
    kv_write: KvWriteOperand | None = None,
    pre_gathered: PreGatheredChunk | None = None,
) -> GatheredChunk:
    """Dispatches KV gathering for one chunk across all supported strategies."""
    gather_cfg = config.gather
    capped_cfg = config.capped
    is_dcp_mode = config.dcp.is_dcp_mode
    sparsecore_layout = gather_cfg.is_sparsecore
    topk_cap = capped_cfg.topk_cap
    chunk_size = end_pos - start_pos
    capped_windows = None if topk_cap is None else topk // topk_cap

    if group_tab is not None and topk_cap is None:
        raise ValueError("group_tab requires topk_cap")
    if pre_gathered is not None:
        if kv_write is not None:
            raise ValueError("pre_gathered does not carry kv_write caches")
        nope, rope, live_views = pre_gathered
        return GatheredChunk(
            nope=nope,
            rope=rope,
            live_views=live_views,
            cache_kv_nope=cache_kv_nope,
            cache_kv_rope=cache_kv_rope,
        )

    indices = (
        None if physical_indices is None else physical_indices[start_pos:end_pos]
    )
    kv_lens_slice = kv_lens[start_pos:end_pos]
    if is_dcp_mode and capped_windows is None:
        indices, tc_num_row_subchunks, dcp_gather_atoms = apply_dcp_skip_windows(
            indices,
            kv_lens_slice,
            topk=topk,
            sparsecore_layout=sparsecore_layout,
            sparsecore_atoms_per_batch=gather_cfg.sparsecore_atoms_per_batch,
        )
    else:
        tc_num_row_subchunks = 32
        dcp_gather_atoms = gather_cfg.sparsecore_atoms_per_batch

    if topk_cap is not None:
        assert capped_windows is not None
        return gather_capped_sparsecore(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            window_indices,
            kv_lens_slice,
            chunk_size=chunk_size,
            topk=topk,
            topk_cap=topk_cap,
            capped_windows=capped_windows,
            kernel_batch_size=kernel_batch_size,
            small_chunk=small_chunk,
            live_tokens=batch_end - start_pos,
            gather_cfg=gather_cfg,
            capped_cfg=capped_cfg,
            gather_live_in_indices=gather_live_in_indices,
            group_tab=group_tab,
            group_num_live=group_num_live,
            kv_write=kv_write,
            hooks=hooks,
        )

    num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk
    assert indices is not None
    if sparsecore_layout:
        return gather_uncapped_sparsecore(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            chunk_size=chunk_size,
            topk=topk,
            kernel_batch_size=kernel_batch_size,
            small_chunk=small_chunk,
            num_valid_indices=num_valid_indices,
            gather_cfg=gather_cfg,
            is_dcp_mode=is_dcp_mode,
            dcp_gather_atoms=dcp_gather_atoms,
        )
    return gather_tensorcore(
        cache_kv_nope,
        cache_kv_rope,
        indices,
        chunk_size=chunk_size,
        topk=topk,
        num_valid_indices=num_valid_indices,
        is_dcp_mode=is_dcp_mode,
        tc_num_row_subchunks=tc_num_row_subchunks,
    )
