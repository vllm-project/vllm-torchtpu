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

import dataclasses
import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvWeightsRef:
    weight: Any
    bias: Any | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNWeightsRef:
    a_log: Any
    dt_bias: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightRefs:
    conv: ConvWeightsRef
    gdn: GDNWeightsRef


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Maps physical 1-D data into logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    def _get_pos(self, indices):
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)

        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    num_tiles: Any
    p_id_to_s_idx: SmemWrapper
    p_id_to_r_base: SmemWrapper
    p_id_to_r_size: SmemWrapper
    p_id_is_first_tile: SmemWrapper
    p_id_is_last_tile: SmemWrapper
    s_idx_has_initial_state: Any
    s_idx_to_state_indices: Any
    # Per-sequence state read offset for speculative decoding: the initial
    # state is read from `s_idx_to_state_indices[s] + s_idx_to_read_offset[s]`
    # (the checkpoint of the last accepted token). Zero everywhere without
    # speculative decoding.
    s_idx_to_read_offset: Any
    # Speculative decoding, pooled source only: `[max_seqs, window_size]`
    # source-block index per checkpoint, so checkpoint `t` of sequence `s`
    # lives at block `s_idx_to_ckpt_indices[s, t]`. Independent blocks mean
    # the group need not fit inside one manager block, which is what keeps
    # the pool's block size independent of `window_size`. None without
    # speculative decoding, where the only checkpoint is the slot's own
    # state block (`s_idx_to_state_indices`). Two-slot checkpoints use
    # `[max_seqs, 4]`: the pair's two blocks, then the block a mamba block
    # boundary strictly inside the window completes (0 if none) and the
    # window position whose state that boundary needs (-1 if the window
    # completes no block); see `two_slot_boundary`.
    s_idx_to_ckpt_indices: Any = None

    @classmethod
    def create(
        cls,
        cfgs: config.GDNConfig,
        num_tiles: jax.Array,
        p_id_to_s_idx: jax.Array,
        p_id_to_r_base: jax.Array,
        p_id_to_r_size: jax.Array,
        p_id_is_first_tile: jax.Array,
        p_id_is_last_tile: jax.Array,
        s_idx_has_initial_state: jax.Array,
        s_idx_to_state_indices: jax.Array,
        s_idx_to_read_offset: jax.Array,
        s_idx_to_ckpt_indices: jax.Array | None = None,
    ):
        # NOTE: First dim does not matter when it comes to calculating stride.
        shape = (1, cfgs.seq_tile_size)
        return cls(
            num_tiles=num_tiles,
            p_id_to_s_idx=SmemWrapper(p_id_to_s_idx, shape),
            p_id_to_r_base=SmemWrapper(p_id_to_r_base, shape),
            p_id_to_r_size=SmemWrapper(p_id_to_r_size, shape),
            p_id_is_first_tile=SmemWrapper(p_id_is_first_tile, shape),
            p_id_is_last_tile=SmemWrapper(p_id_is_last_tile, shape),
            s_idx_has_initial_state=s_idx_has_initial_state,
            s_idx_to_state_indices=s_idx_to_state_indices,
            s_idx_to_read_offset=s_idx_to_read_offset,
            s_idx_to_ckpt_indices=s_idx_to_ckpt_indices,
        )

    def __len__(self) -> int:
        return len(jax.tree_util.tree_leaves(self))


def two_slot_read(metadata_ref: MetadataRef, s_idx, r_size):
    """(start block, tokens to recompute) of a sequence under two-slot
    checkpoints (`GDNConfig.two_slot_ckpts`).

    The read offset packs both: its low bit names the block of the
    sequence's pair holding the start state, the rest counts the leading
    tokens of the previous window that were accepted but are not in that
    state yet. Sequences that read no state recompute nothing.
    """
    code = metadata_ref.s_idx_to_read_offset[s_idx]
    reads = jnp.logical_and(metadata_ref.s_idx_has_initial_state[s_idx], r_size > 0)
    return code & 1, jnp.where(reads, code >> 1, 0)


def two_slot_boundary(metadata_ref: MetadataRef, s_idx, r_size):
    """(snapshot position, ends on boundary) of a verify window under
    two-slot checkpoints.

    A window whose tokens complete a mamba block leaves the exact state at
    that boundary in the block it completes, for prefix-cache hits and KV
    connectors. When the boundary falls strictly inside the window, that
    block lies before the pair: the window stores the state after the
    snapshot position into it (-1 when there is nothing to store). When
    the window's last token completes the block, it is the pair's block 0,
    which then receives the last state. One-token windows commit to block 0
    anyway.
    """
    ckpt_indices = metadata_ref.s_idx_to_ckpt_indices
    pos = ckpt_indices[s_idx, 3]
    verify = r_size >= 2
    inside = jnp.logical_and(
        jnp.logical_and(verify, pos >= 0),
        jnp.logical_and(pos < r_size - 1, ckpt_indices[s_idx, 2] != 0),
    )
    ends = jnp.logical_and(verify, pos == r_size - 1)
    return jnp.where(inside, pos, -1), ends


@dataclasses.dataclass(frozen=True, kw_only=True)
class BaseBufferedRef(pltpu.BufferedRef):
    cfg: config.GDNConfig = dataclasses.field(metadata=dict(static=True))
    # NOTE: Despite being ref, metadata_ref should be set to static. This is
    # because the memory will be allocated outside of kernel and metadata_ref
    # merely points to the reference.
    metadata_ref: MetadataRef = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfg: config.GDNConfig,
        metadata_ref: MetadataRef,
        **fields,
    ):
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            cfg=cfg,
            metadata_ref=metadata_ref,
            **fields,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class InBufferedRef(BaseBufferedRef):
    def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        for idx in range(self.cfg.seq_tile_size):
            r_base = self.metadata_ref.p_id_to_r_base[p_id, idx]
            dma_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            pltpu.make_async_copy(
                src_ref.at[pl.ds(r_base, dma_size)],
                vmem_ref.at[idx, pl.ds(0, dma_size)],
                sem,
            ).start()

    def wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        dma_size = 0
        for idx in range(self.cfg.seq_tile_size):
            dma_size += self.metadata_ref.p_id_to_r_size[p_id, idx]

        pltpu.make_async_copy(
            vmem_ref.at[0, pl.ds(0, dma_size)],
            vmem_ref.at[0, pl.ds(0, dma_size)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutBufferedRef(BaseBufferedRef):
    def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        for idx in range(self.cfg.seq_tile_size):
            r_base = self.metadata_ref.p_id_to_r_base[p_id, idx]
            dma_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            pltpu.make_async_copy(
                vmem_ref.at[idx, pl.ds(0, dma_size)],
                dst_ref.at[pl.ds(r_base, dma_size)],
                sem,
            ).start()

    def wait_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        dma_size = 0
        for idx in range(self.cfg.seq_tile_size):
            dma_size += self.metadata_ref.p_id_to_r_size[p_id, idx]

        pltpu.make_async_copy(
            vmem_ref.at[0, pl.ds(0, dma_size)],
            vmem_ref.at[0, pl.ds(0, dma_size)],
            sem,
        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class StateBufferedRef(BaseBufferedRef):
    """Input/output buffered ref for per-sequence state (conv / recurrent).

    The VMEM window holds one state per window position,
    [seq_tile_size, window_size, *state_shape]. The initial state is read
    from `state_indices[s] + read_offset[s]` into position 0 of the
    sequence's window row, and after compute the first
    `min(r_size, window_size)` checkpoints are written back to
    `state_indices[s] .. + that many slots`.

    Without speculative decoding `window_size` is 1 and `read_offset` is 0,
    so this reduces to reading and writing the single state at
    `state_indices[s]`.
    """

    # --- source-layout hooks; overridden for an external (pooled) source ---

    def _wait_before_slot_reuse(
        self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]
    ) -> None:
        """Drain the previous output before overwriting its shared VMEM slot.

        emit_pipeline starts input prefetch before its normal output wait.
        For an INOUT ref, prefetch can therefore overwrite a slot while an
        earlier output DMA still reads it. Each grid tile uses a new window
        (no lookahead); tile p reuses the slot last used by p-buffer_count.
        """
        assert not self.use_lookahead
        if not self.is_output:
            # An input-only state ref shares no slot with a write-back.
            return
        p_id = grid_indices[0]

        @pl.when(p_id >= self.buffer_count)
        def _wait() -> None:
            previous = cast(
                StateBufferedRef,
                self.with_slot_index(wait_out_slot=self.cumulative_copy_in),
            )
            previous._wait_out_dma(src_ref, (p_id - self.buffer_count,))

    def wait_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        """Drain trailing outputs whose slots are not reused by another input.

        Earlier outputs were already consumed by _wait_before_slot_reuse.
        Waiting for them again would consume the same DMA semaphore twice.
        Keep emit_pipeline's slot advancement, but only wait for the tail.
        An output-only state ref waits for every output like other outputs.
        """
        if not self.is_input:
            self._wait_out_dma(dst_ref, grid_indices)
            return
        tail_start = self.metadata_ref.num_tiles[...] - self.buffer_count

        @pl.when(grid_indices[0] >= tail_start)
        def _wait() -> None:
            self._wait_out_dma(dst_ref, grid_indices)

    def _src_slice(self, ref: jax.Ref, state_idx, count):
        """One slot's source window: `count` states at the slot's index."""
        return ref.at[pl.ds(state_idx, count)]

    def _unit(self) -> int:
        """Source rows one state occupies (checkpoints for the dense path)."""
        return 1

    def _wait_slice(self, vmem_ref: jax.Ref, count):
        """Never-executed self-copy ref carrying the bytes to wait on.

        ``count`` is a number of participating SLOTS, not source rows: one
        slot's copy is `_unit()` rows, and this ref must cover exactly the
        bytes those copies moved.
        """
        return vmem_ref.at[0, pl.ds(0, count)]

    def _read_offset(self, s_idx):
        """Spec-decode checkpoint to resume from; 0 without a window."""
        return self.metadata_ref.s_idx_to_read_offset[s_idx]

    def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        self._wait_before_slot_reuse(src_ref, grid_indices)
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        for idx in range(self.cfg.seq_tile_size):
            is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            dma_size = jnp.where(should_read, self._unit(), 0)

            # Resume from the checkpoint of the last accepted token.
            state_idx += self._read_offset(s_idx)

            pltpu.make_async_copy(
                self._src_slice(src_ref, state_idx, dma_size),
                vmem_ref.at[idx, pl.ds(0, dma_size)],
                sem,
            ).start()

    def wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        dma_size = 0
        for idx in range(self.cfg.seq_tile_size):
            is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            dma_size += jnp.where(should_read, 1, 0)

        # NOTE: With bounds checks disabled, the self-copy descriptor may
        # nominally exceed the window row; it is never executed, only used
        # to wait for the same number of bytes `copy_in` issued.
        wait_ref = self._wait_slice(vmem_ref, dma_size)
        pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()

    def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        for idx in range(self.cfg.seq_tile_size):
            is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            state_idx = self.metadata_ref.s_idx_to_state_indices[s_idx]
            # Write one checkpoint per valid window position, starting at the
            # group's base slot. `r_size` never exceeds `window_size` for
            # windowed sequences; the clamp is for PER_SEQ tiles, which hold
            # many tokens but keep only the final state.
            r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)
            dma_size = jnp.where(is_last_tile, num_ckpts * self._unit(), 0)

            pltpu.make_async_copy(
                vmem_ref.at[idx, pl.ds(0, dma_size)],
                self._src_slice(dst_ref, state_idx, dma_size),
                sem,
            ).start()

    def _wait_out_dma(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        dma_size = 0
        for idx in range(self.cfg.seq_tile_size):
            is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
            r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)
            dma_size += jnp.where(is_last_tile, num_ckpts, 0)

        # NOTE: With bounds checks disabled, the self-copy descriptor may
        # nominally exceed the window row; it is never executed, only used
        # to wait for the same number of bytes `copy_out` issued.
        wait_ref = self._wait_slice(vmem_ref, dma_size)
        pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ExternalStateBufferedRef(StateBufferedRef):
    """State tiles streamed from/to an indexed external state source.

    Unlike ``StateBufferedRef``'s dense per-slot state tensor, the source
    stores each slot's state as raw bytes inside a window of ``stride``
    consecutive source blocks addressed by the slot's state index;
    ``region`` selects the block/row range of this state within that
    window. copy_in/copy_out move the whole region with one contiguous
    async copy per slot, gated by the same first/last-tile and
    has_initial_state metadata as the dense path, so padded or invalid
    slots move no bytes in either direction. vmem_ldst applies the
    region's typed view when the tile is loaded or stored.

    The VMEM tile holds one region per window position,
    [seq_tile_size, window_size, nblocks, nrows, *payload], mirroring the
    dense path: the initial state — the checkpoint selected by the
    sequence's read offset — is read into window position 0, and after
    compute the first `min(r_size, window_size)` checkpoints are written
    back to the slot's checkpoints `0..`. Without speculative decoding
    `window_size` is 1, reducing to a single region per slot.
    """

    region: config.StateRegion = dataclasses.field(metadata=dict(static=True))
    stride: int = dataclasses.field(metadata=dict(static=True))
    # The PCP fused kernel writes one compact update per sequence and applies
    # those updates to the donated pool outside Pallas.  Keeping the compact
    # output ref here lets the regular buffered pipeline retain its state
    # copy-in schedule without exposing the whole pool as a Pallas output.
    # Other V3 callers leave this unset and continue to write directly to the
    # aliased external source.
    compact_output_ref: Any | None = dataclasses.field(
        default=None, metadata=dict(static=True)
    )
    # Two-slot checkpoints: whether this region's start-state tile is written
    # back on every verify window (the conv region, which also carries the
    # window's tokens) or only when tokens were recomputed into it.
    start_always: bool = dataclasses.field(default=False, metadata=dict(static=True))

    def _ckpt_state_idx(self, s_idx, ckpt):
        """Source block holding checkpoint `ckpt` of sequence `s_idx`.

        With speculative decoding every checkpoint is an independent
        source block, so the index is looked up rather than derived from a
        base. Without it there is only checkpoint 0, in the slot's own
        state block.
        """
        ckpt_indices = self.metadata_ref.s_idx_to_ckpt_indices
        if ckpt_indices is None:
            return self.metadata_ref.s_idx_to_state_indices[s_idx]
        return ckpt_indices[s_idx, ckpt]

    def _region_slice(self, src_ref: jax.Ref, state_idx, nblocks):
        base = state_idx * self.stride + self.region.kb0
        if self.cfg.state_plan.whole_block_dma or (
            self.region.row0 == 0 and self.region.nrows == src_ref.shape[1]
        ):
            return src_ref.at[pl.ds(base, nblocks)]
        return src_ref.at[
            pl.ds(base, nblocks), pl.ds(self.region.row0, self.region.nrows)
        ]

    def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        self._wait_before_slot_reuse(src_ref, grid_indices)
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        for idx in range(self.cfg.seq_tile_size):
            is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            nblocks = jnp.where(should_read, self.region.nblocks, 0)

            # Resume from the checkpoint of the last accepted token.
            read_ckpt = self.metadata_ref.s_idx_to_read_offset[s_idx]
            if self.cfg.two_slot_ckpts:
                read_ckpt = read_ckpt & 1
            state_idx = self._ckpt_state_idx(s_idx, read_ckpt)

            pltpu.make_async_copy(
                self._region_slice(src_ref, state_idx, nblocks),
                vmem_ref.at[idx, 0, pl.ds(0, nblocks)],
                sem,
            ).start()

    def _unit(self) -> int:
        return self.region.nblocks

    def _copy_out_compact(self, grid_indices: tuple[int | jax.Array]):
        """PCP fused kernel: one compact update per sequence.

        The updates are applied to the donated pool outside Pallas, so the
        pool never has to be exposed as a Pallas output. This path predates
        per-checkpoint blocks and PCP rejects speculative decoding, so it
        writes the compact tile without checkpoint addressing.
        """
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        for idx in range(self.cfg.seq_tile_size):
            is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)
            dma_size = jnp.where(is_last_tile, num_ckpts * self._unit(), 0)

            pltpu.make_async_copy(
                vmem_ref.at[idx, 0, pl.ds(0, dma_size)],
                self.compact_output_ref.at[s_idx, pl.ds(0, dma_size)],
                sem,
            ).start()

    def _two_slot_writes(self, p_id, idx):
        """Two-slot write-back of one sequence: ((tile, block, write), ...).

        A verify window writes the state after its last real token to the
        pair's other block and, when the start tile changed, the start
        state back to the start block. A window whose last token completes
        the pair's block 0 swaps them: the last state goes to block 0, the
        block a prefix hit resumes from, and the start state to block 1
        (`two_slot_boundary`). A boundary strictly inside the window gets
        its snapshot tile. A one-token window commits: its state goes to
        block 0 and the start block is not written.
        """
        is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
        s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
        r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
        slot, replay = two_slot_read(self.metadata_ref, s_idx, r_size)
        snap_pos, ends = two_slot_boundary(self.metadata_ref, s_idx, r_size)
        verify = r_size >= 2
        live = jnp.logical_and(is_last_tile, r_size >= 1)
        start = jnp.where(ends, 1, slot)
        last_ckpt = jnp.where(verify, 1 - start, 0)
        write_start = jnp.logical_and(
            jnp.logical_and(live, verify),
            jnp.logical_or(
                jnp.logical_or(self.start_always, replay > 0), start != slot
            ),
        )
        return (
            (1, self._ckpt_state_idx(s_idx, last_ckpt), live),
            (0, self._ckpt_state_idx(s_idx, start), write_start),
            (2, self._ckpt_state_idx(s_idx, 2), jnp.logical_and(live, snap_pos >= 0)),
        )

    def _two_slot_tiles(self) -> bool:
        return self.cfg.two_slot_tiles

    def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        if self.compact_output_ref is not None:
            return self._copy_out_compact(grid_indices)
        if self._two_slot_tiles():
            assert self.sem_sends is not None
            assert self.window_ref is not None
            slot = self.current_copy_out_slot
            sem = self.sem_sends.at[slot]
            vmem_ref = self.window_ref.at[slot]
            p_id = grid_indices[0]
            for idx in range(self.cfg.seq_tile_size):
                for tile, state_idx, write in self._two_slot_writes(p_id, idx):
                    nblocks = jnp.where(write, self.region.nblocks, 0)
                    pltpu.make_async_copy(
                        vmem_ref.at[idx, tile, pl.ds(0, nblocks)],
                        self._region_slice(dst_ref, state_idx, nblocks),
                        sem,
                    ).start()
            return None
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        for idx in range(self.cfg.seq_tile_size):
            is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            # One checkpoint per valid window position, starting at the
            # slot's checkpoint 0. `r_size` never exceeds `window_size` for
            # windowed sequences; the clamp is for PER_SEQ tiles, which
            # hold many tokens but keep only the final state.
            r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)

            for t in range(self.cfg.window_size):
                should_write = jnp.logical_and(is_last_tile, t < num_ckpts)
                nblocks = jnp.where(should_write, self.region.nblocks, 0)
                state_idx = self._ckpt_state_idx(s_idx, t)
                pltpu.make_async_copy(
                    vmem_ref.at[idx, t, pl.ds(0, nblocks)],
                    self._region_slice(dst_ref, state_idx, nblocks),
                    sem,
                ).start()

    def wait_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        dma_size = 0
        for idx in range(self.cfg.seq_tile_size):
            is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            dma_size += jnp.where(should_read, 1, 0)

        # Each slot's copy covers one full region tile (all nblocks), read
        # into window position 0; wait per slot along the window dim of one
        # sequence row. NOTE: With bounds checks disabled, the descriptor
        # may nominally exceed the window row; it is never executed, only
        # used to wait for the same number of bytes `copy_in` issued.
        wait_ref = vmem_ref.at[0, pl.ds(0, dma_size)]
        pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()

    def _wait_out_dma(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        dma_size = 0
        for idx in range(self.cfg.seq_tile_size):
            if self._two_slot_tiles():
                for _, _, write in self._two_slot_writes(p_id, idx):
                    dma_size += jnp.where(write, 1, 0)
                continue
            is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
            r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)
            dma_size += jnp.where(is_last_tile, num_ckpts, 0)

        # Each checkpoint's copy covers one full region tile; wait for that
        # many region tiles along the window dim of one sequence row (the
        # descriptor may nominally exceed it, see wait_in's NOTE).
        wait_ref = vmem_ref.at[0, pl.ds(0, dma_size)]
        pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()


def _state_allocs(
    cls,
    split: bool,
    block_spec_partial,
    shape: tuple[int, ...],
    ref: jax.Array,
    kwargs: dict,
):
    """One state's buffered ref, or its (input, output) pair when split.

    `shape` is the [seq_tile_size, window_size, ...] checkpoint tile; the
    split input tile keeps only the window position the state is read into.
    """
    if not split:
        return cls.input_output(
            spec=block_spec_partial(block_shape=shape), dtype_or_type=ref, **kwargs
        )
    in_shape = (shape[0], 1, *shape[2:])
    return (
        cls.input(
            spec=block_spec_partial(block_shape=in_shape), dtype_or_type=ref, **kwargs
        ),
        cls.output(
            spec=block_spec_partial(block_shape=shape), dtype_or_type=ref, **kwargs
        ),
    )


def create_allocs(
    metadata_ref: MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    out_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    cfg: config.GDNConfig,
    conv_state_output_ref: jax.Array | None = None,
    recurrent_state_output_ref: jax.Array | None = None,
    split_state_io: bool = False,
) -> tuple[
    InBufferedRef,
    InBufferedRef,
    InBufferedRef,
    StateBufferedRef | tuple[StateBufferedRef, StateBufferedRef],
    StateBufferedRef | tuple[StateBufferedRef, StateBufferedRef],
    OutBufferedRef,
]:
    """Pipeline allocations of one fused kernel call.

    With `split_state_io` each state gets an input buffer holding the one
    state a tile reads and a separate output buffer holding its
    checkpoints, returned as an (input, output) pair. A tile's prefetch
    then never waits for the previous tile's write-back to leave the shared
    slot, so the write-back overlaps the next tile's compute.
    """
    compact_state_output = conv_state_output_ref is not None
    if compact_state_output != (recurrent_state_output_ref is not None):
        raise ValueError(
            "Conv and recurrent compact state outputs must be set together."
        )
    if compact_state_output and cfg.state_plan is None:
        raise ValueError("Compact state outputs require an external state source plan.")
    if compact_state_output and cfg.window_size != 1:
        raise ValueError(
            "Compact external state outputs do not support checkpoint "
            f"windows, got window_size={cfg.window_size}."
        )
    if compact_state_output and split_state_io:
        raise ValueError("Compact external state outputs are not split.")

    qkv_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.dim_size)
    if cfg.packed_tokens:
        num_lanes = pltpu.get_tpu_info().num_lanes
        qkv_shape = (
            cfg.seq_tile_size,
            cfg.chunk_size,
            pl.cdiv(cfg.dim_size, num_lanes) + 2,
            num_lanes,
        )
    b_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.aligned_num_v_heads)
    # Same shape as b under GDN; under KDA the gate is per-channel, so it is
    # d_k times wider (see GDNConfig.gate_dim).
    a_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.aligned_gate_dim)

    out_shape = (
        cfg.seq_tile_size,
        cfg.chunk_size,
        cfg.num_v_heads,
        cfg.v_head_dim,
    )

    pipeline_mode = pl.Buffered(buffer_count=cfg.num_buffers, use_lookahead=False)

    block_spec_partial = functools.partial(
        pl.BlockSpec,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pipeline_mode,
    )

    qkv_spec = block_spec_partial(block_shape=qkv_shape)
    b_spec = block_spec_partial(block_shape=b_shape)
    a_spec = block_spec_partial(block_shape=a_shape)
    in_buffered_partial = functools.partial(
        InBufferedRef.input,
        buffer_count=pipeline_mode.buffer_count,
        use_lookahead=pipeline_mode.use_lookahead,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )
    qkv_alloc = in_buffered_partial(spec=qkv_spec, dtype_or_type=qkv_ref)
    b_alloc = a_alloc = None
    if not cfg.packed_tokens:
        b_alloc = in_buffered_partial(spec=b_spec, dtype_or_type=b_ref)
        a_alloc = in_buffered_partial(spec=a_spec, dtype_or_type=a_ref)

    out_alloc = OutBufferedRef.output(
        spec=block_spec_partial(block_shape=out_shape),
        dtype_or_type=out_ref,
        buffer_count=pipeline_mode.buffer_count,
        use_lookahead=pipeline_mode.use_lookahead,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )

    if cfg.state_plan is None:
        # One state checkpoint per window position per sequence (a single one
        # without speculative decoding, where window_size is 1).
        conv_shape = (
            cfg.seq_tile_size,
            cfg.num_ckpt_tiles,
            cfg.prev_kernel_size,
            1,
            cfg.dim_size,
        )
        recurrent_shape = (
            cfg.seq_tile_size,
            cfg.num_ckpt_tiles,
            cfg.num_v_heads,
            cfg.kq_head_dim,
            cfg.v_head_dim,
        )
        state_kwargs = dict(
            buffer_count=pipeline_mode.buffer_count,
            use_lookahead=pipeline_mode.use_lookahead,
            cfg=cfg,
            metadata_ref=metadata_ref,
        )
        conv_alloc = _state_allocs(
            StateBufferedRef,
            split_state_io,
            block_spec_partial,
            conv_shape,
            conv_state_ref,
            state_kwargs,
        )
        recurrent_alloc = _state_allocs(
            StateBufferedRef,
            split_state_io,
            block_spec_partial,
            recurrent_shape,
            recurrent_state_ref,
            state_kwargs,
        )
    else:
        # Both state regions stream from the one external source ref
        # (passed as both state refs); the tiles keep the source's raw
        # block layout and vmem_ldst applies the typed region view. Like
        # the dense path, the tile holds one region per window position
        # (a single one without speculative decoding).
        plan = cfg.state_plan
        if plan.whole_block_dma:
            # Whole-block DMA transfers complete pages, retaining all inner dims.
            conv_shape = (
                cfg.seq_tile_size,
                cfg.num_ckpt_tiles,
                plan.conv.nblocks,
                *conv_state_ref.shape[1:],
            )
            recurrent_shape = (
                cfg.seq_tile_size,
                cfg.num_ckpt_tiles,
                plan.recurrent.nblocks,
                *recurrent_state_ref.shape[1:],
            )
        else:
            payload = conv_state_ref.shape[2:]
            conv_shape = (
                cfg.seq_tile_size,
                cfg.num_ckpt_tiles,
                plan.conv.nblocks,
                plan.conv.nrows,
                *payload,
            )
            recurrent_shape = (
                cfg.seq_tile_size,
                cfg.num_ckpt_tiles,
                plan.recurrent.nblocks,
                plan.recurrent.nrows,
                *payload,
            )
        state_kwargs = dict(
            buffer_count=pipeline_mode.buffer_count,
            use_lookahead=pipeline_mode.use_lookahead,
            cfg=cfg,
            metadata_ref=metadata_ref,
            stride=plan.stride,
        )
        conv_alloc = _state_allocs(
            ExternalStateBufferedRef,
            split_state_io,
            block_spec_partial,
            conv_shape,
            conv_state_ref,
            dict(
                state_kwargs,
                region=plan.conv,
                compact_output_ref=conv_state_output_ref,
                start_always=cfg.two_slot_ckpts,
            ),
        )
        recurrent_alloc = _state_allocs(
            ExternalStateBufferedRef,
            split_state_io,
            block_spec_partial,
            recurrent_shape,
            conv_state_ref,
            dict(
                state_kwargs,
                region=plan.recurrent,
                compact_output_ref=recurrent_state_output_ref,
            ),
        )

    return qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc
