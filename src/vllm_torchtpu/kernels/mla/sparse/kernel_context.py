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
"""Trace-time kernel geometry and double-buffer state for SparseMLA."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse.config import (
    Q_ROPE_DIM,
    SEND_PAD_ROWS,
    SEND_SLAB_ROW0,
    AttentionKernelConfig,
    align_to,
    get_dtype_packing,
    send_layout_lanes,
    send_layout_supported,
)
from vllm_torchtpu.kernels.mla.sparse.ref_registry import AttentionRefs


def compute_kv_rungs(bkv_sz: int) -> tuple[int, ...]:
    """Returns ascending DCP KV slice rungs dividing ``bkv_sz``.

    Shared by the fetch and compute ladders so a compute rung never reads
    VMEM rows that its matching fetch rung did not populate.
    """
    for div in (8, 4, 2):
        step = bkv_sz // div
        if bkv_sz % div == 0 and step >= 64 and step % 64 == 0:
            return tuple(step * k for k in (1, 2, 4, 8) if k <= div)
    return (bkv_sz,)


def _derive_send_exchange_geometry(
    refs: AttentionRefs,
    cfg: AttentionKernelConfig,
    num_tokens: int,
    num_q_heads: int,
    o_dim: int,
) -> tuple[int, int, int, int, int]:
    """Validates send/exchange refs and returns ``(per, rows, lanes, xrows, xblk)``."""
    if not cfg.send_enabled:
        assert not cfg.xchg_enabled, "the fused exchange needs the send layout"
        return 0, 0, 0, 0, 0

    send_world = cfg.send_world
    batch_size = cfg.batch_size
    assert send_layout_supported(
        num_tokens, num_q_heads, send_world, o_dim, batch_size
    ), (num_tokens, num_q_heads, send_world, o_dim, batch_size)
    send_per = num_q_heads // send_world
    send_rows = num_tokens * send_per
    send_lanes = send_layout_lanes(num_tokens, num_q_heads, send_world)

    if not cfg.xchg_enabled:
        return send_per, send_rows, send_lanes, 0, 0

    assert cfg.send_exchange is not None
    assert not cfg.preserve_suffix and batch_size == num_tokens, (
        "the fused exchange runs on a one-step grid"
    )
    assert refs.bo_x2 is not None and refs.lse is not None
    # Rows each destination receives from row SEND_SLAB_ROW0 of its block.
    xchg_rows = SEND_SLAB_ROW0 + send_rows
    xchg_block = SEND_PAD_ROWS + send_rows
    assert refs.bo_x2.shape[2] == xchg_rows, refs.bo_x2.shape
    if cfg.xchg_merge:
        assert refs.recv_v is not None
        assert refs.recv_v.shape == (send_world, xchg_rows, o_dim), (
            refs.recv_v.shape
        )
        assert refs.lse.shape == (send_rows, o_dim), refs.lse.shape
        assert refs.lse.dtype == jnp.int16 and send_rows % 16 == 0
    else:
        assert refs.lse.shape == (send_world * xchg_block, o_dim), (
            refs.lse.shape
        )
    return send_per, send_rows, send_lanes, xchg_rows, xchg_block


@dataclasses.dataclass(frozen=True)
class KernelGeometry:
    """Validated static shapes and per-step scalar bounds for one kernel call."""

    num_tokens: int
    num_q_heads: int
    head_dim: int
    o_dim: int
    nope_dim: int
    bkv_sz: int
    q_dtype: Any
    nofill: bool
    send_per: int
    send_rows: int
    send_lanes: int
    xchg_rows: int
    xchg_block: int
    kv_rungs: tuple[int, ...]
    start_seq_idx: Any
    end_seq_idx: Any
    prefix_end: Any
    batch_start_seq_idx: Any
    batch_end_seq_idx: Any

    @classmethod
    def from_refs(
        cls,
        refs: AttentionRefs,
        cfg: AttentionKernelConfig,
    ) -> KernelGeometry:
        """Validates ref geometry and derives static/dynamic bounds at entry."""
        q_hbm_ref = refs.q
        o_hbm_ref = refs.out
        assert o_hbm_ref is not None
        assert q_hbm_ref.shape[1:] == o_hbm_ref.shape[1:]

        num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
        batch_size = cfg.batch_size

        # Skip sentinel fill DMA when a single grid step stores every row's lse.
        nofill = (
            cfg.body.skip_redundant_lse_fill
            and cfg.lse_fill
            and not cfg.send_enabled
            and num_tokens <= batch_size
        )
        # Width of the stored output (q_nope's in split-q mode, else head_dim).
        o_dim = head_dim
        if cfg.split_q:
            assert refs.q_pe is not None
            pe_dim = Q_ROPE_DIM if cfg.q_rope else refs.q_pe.shape[-1]
            head_dim = align_to(head_dim + pe_dim, 128)
        nope_dim = kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
        assert refs.kv_lens.shape[0] == num_tokens
        bkv_sz = (
            refs.cache_kv_nope.shape[1]
            if cfg.sparsecore_layout
            else refs.cache_kv_nope.shape[1] // kv_cache_utils.WORD_BYTES
        )

        q_dtype = q_hbm_ref.dtype
        q_packing = get_dtype_packing(q_dtype)
        assert o_hbm_ref.dtype == q_dtype
        assert head_dim % 128 == 0
        assert num_q_heads % q_packing == 0

        send_per, send_rows, send_lanes, xchg_rows, xchg_block = (
            _derive_send_exchange_geometry(refs, cfg, num_tokens, num_q_heads, o_dim)
        )

        start_seq_idx = refs.bounds.start_seq_idx
        end_seq_idx = refs.bounds.end_seq_idx
        prefix_end = refs.bounds.prefix_end if cfg.preserve_suffix else None
        batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
        batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

        return cls(
            num_tokens=num_tokens,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            o_dim=o_dim,
            nope_dim=nope_dim,
            bkv_sz=bkv_sz,
            q_dtype=q_dtype,
            nofill=nofill,
            send_per=send_per,
            send_rows=send_rows,
            send_lanes=send_lanes,
            xchg_rows=xchg_rows,
            xchg_block=xchg_block,
            kv_rungs=compute_kv_rungs(bkv_sz),
            start_seq_idx=start_seq_idx,
            end_seq_idx=end_seq_idx,
            prefix_end=prefix_end,
            batch_start_seq_idx=batch_start_seq_idx,
            batch_end_seq_idx=batch_end_seq_idx,
        )


@dataclasses.dataclass(frozen=True)
class DoubleBufferState:
    """Explicit ping-pong slot state backed by SMEM ``sem_ids[2]``."""

    sem_ids: Any

    @staticmethod
    def next_slot(slot: Any) -> Any:
        """Returns the alternate double-buffer slot index (``0 <-> 1``)."""
        return lax.select(slot == 0, 1, 0)

    def current_in_slot(self) -> Any:
        """Reads the active input double-buffer slot (``sem_ids[0]``)."""
        return self.sem_ids[0]

    def flip_in_slot(self, next_slot: Any) -> None:
        """Advances the active input double-buffer slot (``sem_ids[0]``)."""
        self.sem_ids[0] = next_slot

    def acquire_out_slot(self) -> Any:
        """Reads the active output slot (``sem_ids[1]``) and flips it."""
        slot = self.sem_ids[1]
        self.sem_ids[1] = self.next_slot(slot)
        return slot
