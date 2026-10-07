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
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, legacy_unified


def granule_view(ref: jax.Ref, leading_shape: tuple[int, ...]) -> jax.Ref:
    num_lanes = pltpu.get_tpu_info().num_lanes
    return ref.bitcast(jnp.uint32).reshape(*leading_shape, -1, 1, num_lanes)


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
    # state block (`s_idx_to_state_indices`).
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
    [seq_tile_size, state_window_alloc, *state_shape]. The initial state is
    read from the checkpoint selected by the sequence's read offset into
    `cfg.state_in_pos`: position 0 without speculative decoding, and an
    extra position past the `window_size` checkpoints with it, so it never
    aliases a checkpoint that an in-flight copy_out of the same slot is
    still reading. After compute the first `min(r_size, window_size)`
    checkpoints are written back from positions `0..`.

    Without speculative decoding `window_size` is 1 and `read_offset` is 0,
    so this reduces to reading and writing the single state at
    `state_indices[s]`.
    """

    # HBM address offset.
    offset: int = dataclasses.field(default=0, metadata=dict(static=True))

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

    def copy_in(self, src_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        vmem_ref = granule_view(vmem_ref, vmem_ref.shape[:2])
        src_ref = granule_view(src_ref, src_ref.shape[:1])
        num_granules = vmem_ref.shape[2]

        for idx in range(self.cfg.seq_tile_size):
            is_first_tile = self.metadata_ref.p_id_is_first_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            dma_size = jnp.where(should_read, 1, 0)

            # Resume from the checkpoint of the last accepted token.
            read_ckpt = self.metadata_ref.s_idx_to_read_offset[s_idx]
            state_idx = self._ckpt_state_idx(s_idx, read_ckpt)

            # Into the dedicated incoming-state position (see
            # `GDNConfig.state_in_pos`), never a checkpoint position that an
            # in-flight copy_out of this slot may still be reading.
            pltpu.make_async_copy(
                src_ref.at[
                    pl.ds(state_idx, dma_size), pl.ds(self.offset, num_granules)
                ],
                vmem_ref.at[idx, pl.ds(self.cfg.state_in_pos, dma_size)],
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
        wait_ref = vmem_ref.at[0, pl.ds(0, dma_size)]
        pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()

    def copy_out(self, dst_ref: jax.Ref, grid_indices: tuple[int | jax.Array]):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        p_id = grid_indices[0]

        vmem_ref = granule_view(vmem_ref, vmem_ref.shape[:2])
        dst_ref = granule_view(dst_ref, dst_ref.shape[:1])
        num_granules = vmem_ref.shape[2]

        for idx in range(self.cfg.seq_tile_size):
            is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
            s_idx = self.metadata_ref.p_id_to_s_idx[p_id, idx]
            # Write one checkpoint per valid window position, starting at the
            # group's base slot. `r_size` never exceeds `window_size` for
            # windowed sequences; the clamp is for PER_SEQ tiles, which hold
            # many tokens but keep only the final state.
            r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)

            for t in range(self.cfg.window_size):
                should_write = jnp.logical_and(is_last_tile, t < num_ckpts)
                dma_size = jnp.where(should_write, 1, 0)
                state_idx = self._ckpt_state_idx(s_idx, t)
                pltpu.make_async_copy(
                    vmem_ref.at[pl.ds(idx, dma_size), t],
                    dst_ref.at[
                        pl.ds(state_idx, dma_size), pl.ds(self.offset, num_granules)
                    ],
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
            is_last_tile = self.metadata_ref.p_id_is_last_tile[p_id, idx]
            r_size = self.metadata_ref.p_id_to_r_size[p_id, idx]
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)
            dma_size += jnp.where(is_last_tile, num_ckpts, 0)

        # Each checkpoint moved one window row's worth of bytes; see wait_in
        # for why this counts them on the typed tile.
        wait_ref = vmem_ref.at[0, pl.ds(0, dma_size)]
        pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()


def create_allocs(
    metadata_ref: MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    out_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[
    InBufferedRef,
    InBufferedRef,
    InBufferedRef,
    StateBufferedRef,
    StateBufferedRef,
    OutBufferedRef,
]:
    del recurrent_state_ref

    qkv_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.dim_size)
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
        # without speculative decoding, where window_size is 1), plus the
        # dedicated incoming-state position with it (`cfg.state_in_pos`).
        conv_shape = (
            cfg.seq_tile_size,
            cfg.state_window_alloc,
            cfg.prev_kernel_size,
            1,
            cfg.dim_size,
        )
        # The window stores the state as uint32 words; `packing` state
        # values share one word, so it has kq_head_dim // packing rows.
        packing = 4 // jnp.dtype(cfg.dtypes.recurrent_state).itemsize
        recurrent_shape = (
            cfg.seq_tile_size,
            cfg.state_window_alloc,
            cfg.num_v_heads,
            cfg.kq_head_dim // packing,
            cfg.v_head_dim,
        )
        state_buffered_partial = functools.partial(
            StateBufferedRef.input_output,
            buffer_count=pipeline_mode.buffer_count,
            use_lookahead=pipeline_mode.use_lookahead,
            cfg=cfg,
            metadata_ref=metadata_ref,
        )
        conv_alloc = state_buffered_partial(
            spec=block_spec_partial(block_shape=conv_shape),
            dtype_or_type=cfg.dtypes.conv_state,
            offset=cfg.conv_hbm_offset,
        )
        recurrent_alloc = state_buffered_partial(
            spec=block_spec_partial(block_shape=recurrent_shape),
            dtype_or_type=jnp.uint32,
            offset=cfg.recurrent_hbm_offset,
        )
    else:
        # TODO: Delete this after unified pool migration completes.
        conv_alloc, recurrent_alloc = legacy_unified.external_state_allocs(
            cfg=cfg,
            metadata_ref=metadata_ref,
            state_source_ref=conv_state_ref,
            block_spec_partial=block_spec_partial,
            pipeline_mode=pipeline_mode,
        )

    return qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc
