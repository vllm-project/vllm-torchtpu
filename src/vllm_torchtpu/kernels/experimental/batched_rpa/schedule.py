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
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils


class FieldOffset:
    """A Python descriptor that generates the `.at[pos + offset]` lazy lookup.

    This is necessary because JAX does not support dynamically slicing a
    range (e.g. `data.at[pos:pos+4]`) using traced indices inside a loop,
    but it natively supports retrieving/updating single dynamically-indexed
    elements (e.g. `data.at[pos+1]`).
    """

    def __init__(self, offset: int | None = None, is_abstract: bool = False):
        if offset is None:
            assert is_abstract

        self.offset = offset
        self.__isabstractmethod__ = is_abstract

    def __get__(self, obj, objtype=None):
        return obj.data.at[obj.pos + self.offset]


@dataclasses.dataclass(frozen=True)
class DmaNew(ABC):
    data: Any
    pos: Any

    # New-KV fetch. The new tokens are contiguous in HBM and in VMEM, so a
    # single DMA covers them all: copy_in takes the addresses from entry 0 and
    # the length from the sum of fetch_val over all entries.
    fetch_hbm = FieldOffset(is_abstract=True)
    fetch_vmem = FieldOffset(is_abstract=True)
    # Writeback, which stays per-entry because page_indices scatters the
    # destinations. wb_hbm packs them as (page_idx << page_size_log2) | p_off.
    wb_hbm = FieldOffset(is_abstract=True)
    wb_vmem = FieldOffset(is_abstract=True)
    # Fetch/writeback token counts, packed per the subclass's field layout.
    _flags = FieldOffset(is_abstract=True)

    @staticmethod
    @abstractmethod
    def num_fields() -> int: ...

    @abstractmethod
    def set_flags(self, fetch_val, wb_val) -> None: ...

    @property
    @abstractmethod
    def fetch_val(self) -> jax.Array: ...

    @property
    @abstractmethod
    def wb_val(self) -> jax.Array: ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SeqAlongLaneDmaNew(DmaNew):
    fetch_hbm = FieldOffset(0)
    fetch_vmem = FieldOffset(1)
    wb_hbm = FieldOffset(2)
    wb_vmem = FieldOffset(3)
    _flags = FieldOffset(4)

    @staticmethod
    def num_fields() -> int:
        return 5

    @property
    def fetch_val(self):
        # First 16 bits are for fetch
        return self._flags[...] & 0xFFFF

    @property
    def wb_val(self):
        # Last 16 bits are for writeback
        return self._flags[...] >> 16

    def set_flags(self, fetch_val, wb_val):
        self._flags[...] = fetch_val | (wb_val << 16)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HeadAlongSublaneDmaNew(DmaNew):
    # Each page writes back exactly the tokens it fetched, from the offset it
    # fetched them into, so vmem offsets and token counts alias down to 4 slots.
    wb_hbm = FieldOffset(0)
    fetch_hbm = FieldOffset(1)
    fetch_vmem = FieldOffset(2)
    wb_vmem = FieldOffset(2)
    _flags = FieldOffset(3)

    @staticmethod
    def num_fields() -> int:
        return 4

    @property
    def fetch_val(self):
        return self._flags[...]

    @property
    def wb_val(self):
        return self._flags[...]

    def set_flags(self, fetch_val, wb_val):
        self._flags[...] = fetch_val


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemWrapper:
    """Maps physical 1-D data into logical N-D representation."""

    data: Any
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))
    # When set, the leading (step) index wraps modulo this power of two, so the
    # `shape[0] == ring_steps` steps held are a sliding window of a longer schedule.
    ring_steps: int | None = dataclasses.field(
        default=None, kw_only=True, metadata=dict(static=True)
    )

    @classmethod
    def create_shape_dtype(cls, shape, ring_steps=None):
        return cls(
            data=jax.ShapeDtypeStruct((np.prod(shape),), jnp.int32),
            shape=shape,
            ring_steps=ring_steps,
        )

    def _wrap_step(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if self.ring_steps is None:
            return indices
        return (indices[0] & (self.ring_steps - 1), *indices[1:])

    def _get_pos(self, indices):
        indices = self._wrap_step(indices)
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)
        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)]

    def __setitem__(self, indices, value):
        self.data[self._get_pos(indices)] = value


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SmemArrayOfStructs(SmemWrapper):
    """Maps physical 1-D data into logical Array of Structs."""

    struct_cls: type[DmaNew] = dataclasses.field(metadata=dict(static=True))
    struct_size: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(
        cls, shape, struct_cls, struct_size, ring_steps=None
    ):  # pytype: disable=bad-override
        assert struct_size == struct_cls.num_fields()
        return cls(
            data=jax.ShapeDtypeStruct((np.prod(shape) * struct_size,), jnp.int32),
            shape=shape,
            struct_cls=struct_cls,
            struct_size=struct_size,
            ring_steps=ring_steps,
        )

    def _get_pos(self, indices):
        indices = self._wrap_step(indices)
        strides = pl.strides_from_shape(self.shape)
        assert len(strides) == len(indices)

        pos = 0
        for stride, idx in zip(strides, indices):
            pos += stride * idx
        return pos * self.struct_size

    def __getitem__(self, indices):
        pos_start = self._get_pos(indices)
        return self.struct_cls(self.data, pos_start)


def _aligned_kv_cache_tokens(
    kv_left_frm_cache: jax.Array, *, cfgs: configs.RpaConfigs
) -> jax.Array:
    """Cached tokens a KV block fetches: its pages' aligned DMA sizes, summed."""
    align = cfgs.kv_token_align
    return jnp.minimum(
        utils.align_to(jnp.maximum(kv_left_frm_cache, 0), align),
        cfgs.bkv_p_cache * cfgs.serve.page_size,
    )


def _kv_cache_page_dmas(
    p_offset: jax.Array,
    kv_cache_tokens: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
    """(page-table index, token count) of each cached page of a KV block.

    For `kv_cache_tokens = _aligned_kv_cache_tokens(kv_left_frm_cache)` these are
    `min(p_offset + i, num_page_indices - 1)` and
    `align_to(clip(kv_left_frm_cache - i * page_size, 0, page_size), align)`:
    `align` divides `page_size`, so page i holds the part of the aligned count
    that falls in `[i * page_size, (i + 1) * page_size)`. Both operands are
    non-negative, so the mins run unsigned, one scalar op each.
    """
    page_size = cfgs.serve.page_size
    assert page_size % cfgs.kv_token_align == 0
    u32 = jnp.uint32
    p_offset = p_offset.astype(u32)
    tokens = kv_cache_tokens.astype(u32)
    last_page = u32(cfgs.serve.num_page_indices - 1)
    dmas = []
    covered = u32(0)
    for i in range(cfgs.bkv_p_cache):
        src_hbm = jnp.minimum(p_offset + u32(i), last_page)
        covered_next = jnp.minimum(tokens, u32((i + 1) * page_size))
        dmas.append(
            (src_hbm.astype(jnp.int32), (covered_next - covered).astype(jnp.int32))
        )
        covered = covered_next
    return dmas


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RpaSchedule:
    """Container for metadata arrays with integrated shape/spec logic."""

    s_idx: SmemWrapper  # [steps, batch]
    q_idx: SmemWrapper  # [steps, batch]
    k_idx: SmemWrapper  # [steps, batch]
    is_last_k: SmemWrapper  # [steps, batch]
    do_writeback: SmemWrapper  # [steps, batch]
    dma_q: SmemWrapper  # [steps, batch, 2]
    # Cached-KV page DMAs. Either the per-page table [steps, batch, bkv_p_cache, 3]
    # (dma_kv_cache), or only the block's aligned cached-token count
    # [steps, batch] (kv_cache_tokens), from which `kv_cache_page_dmas` derives
    # every page's DMA; the other field is None.
    dma_kv_cache: SmemWrapper | None
    kv_cache_tokens: SmemWrapper | None
    dma_kv_new: SmemArrayOfStructs  # [steps, batch, bkv_p_new]
    total_wait_kv_in: SmemWrapper  # [steps]
    total_wait_kv_out: SmemWrapper  # [steps]
    total_wait_q_in: SmemWrapper  # [steps]
    total_wait_o_out: SmemWrapper  # [steps]
    total_wait_lse_out: SmemWrapper  # [steps]
    actual_steps: Any  # [1]

    cfgs: configs.RpaConfigs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create_shape_dtype(
        cls,
        cfgs: configs.RpaConfigs,
        dma_kv_new_struct_cls: type[DmaNew] = SeqAlongLaneDmaNew,
        multiplier: int = 1,
        per_page_kv_cache_dma: bool = False,
        ring_steps: int | None = None,
    ):
        """`ring_steps` sizes an SMEM window of that many steps instead."""
        effective_max_steps = ring_steps or cfgs.max_steps_ub * multiplier
        ring = functools.partial(SmemWrapper.create_shape_dtype, ring_steps=ring_steps)

        idx_wrapper = ring((effective_max_steps, cfgs.batch_size))

        steps_wrapper = ring((effective_max_steps,))

        return cls(
            s_idx=idx_wrapper,
            q_idx=idx_wrapper,
            k_idx=idx_wrapper,
            is_last_k=idx_wrapper,
            do_writeback=idx_wrapper,
            dma_q=ring((effective_max_steps, cfgs.batch_size, 2)),
            dma_kv_cache=(
                ring((effective_max_steps, cfgs.batch_size, cfgs.bkv_p_cache, 3))
                if per_page_kv_cache_dma
                else None
            ),
            kv_cache_tokens=None if per_page_kv_cache_dma else idx_wrapper,
            dma_kv_new=SmemArrayOfStructs.create_shape_dtype(
                (
                    effective_max_steps,
                    cfgs.batch_size,
                    cfgs.bkv_p_new,
                ),
                struct_cls=dma_kv_new_struct_cls,
                struct_size=cfgs.dma_kv_new_size,
                ring_steps=ring_steps,
            ),
            total_wait_kv_in=steps_wrapper,
            total_wait_kv_out=steps_wrapper,
            total_wait_q_in=steps_wrapper,
            total_wait_o_out=steps_wrapper,
            total_wait_lse_out=steps_wrapper,
            actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),
            cfgs=cfgs,
        )

    def kv_cache_page_dmas(
        self,
        step: jax.typing.ArrayLike,
        batch_idx: jax.typing.ArrayLike,
    ) -> list[tuple[jax.Array, jax.Array | int, jax.Array]]:
        """(page-table index, VMEM token offset, token count) of each cached page."""
        cfgs = self.cfgs
        if self.dma_kv_cache is not None:
            return [
                (
                    self.dma_kv_cache[step, batch_idx, i, 0],
                    self.dma_kv_cache[step, batch_idx, i, 1],
                    self.dma_kv_cache[step, batch_idx, i, 2],
                )
                for i in range(cfgs.bkv_p_cache)
            ]
        assert self.kv_cache_tokens is not None
        p_offset = (
            self.s_idx[step, batch_idx] * cfgs.serve.pages_per_seq
            + self.k_idx[step, batch_idx] * cfgs.bkv_p
        )
        # Masked lanes (s_idx == -1) fetch nothing; keep their index in range.
        p_offset = jnp.maximum(p_offset, 0)
        return [
            (src_hbm, i << cfgs.serve.page_size_log2, sz)
            for i, (src_hbm, sz) in enumerate(
                _kv_cache_page_dmas(
                    p_offset, self.kv_cache_tokens[step, batch_idx], cfgs=cfgs
                )
            )
        ]

    def kv_cache_tokens_in(
        self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike
    ) -> jax.Array:
        """Cached tokens fetched for (step, batch_idx), summed over its pages."""
        if self.kv_cache_tokens is not None:
            return self.kv_cache_tokens[step, batch_idx]
        return sum(sz for _, _, sz in self.kv_cache_page_dmas(step, batch_idx))

    def get_dma_q(
        self, step: jax.typing.ArrayLike, batch_idx: jax.typing.ArrayLike
    ) -> tuple[jax.Array, jax.Array]:
        # 0: src_hbm, 1: size
        src_hbm = self.dma_q[step, batch_idx, 0]
        sz = self.dma_q[step, batch_idx, 1]
        return src_hbm, sz

    def scratch_shapes(self):
        """Returns a Pytree of SMEM scratch memory."""

        return jax.tree.map(
            lambda x: pltpu.SMEM(x.shape, x.dtype),
            self,
        )

    def in_specs(self):
        """Returns a Pytree of input BlockSpecs."""

        def wrapper(x):
            if x.size == 1:
                return pl.BlockSpec(memory_space=pltpu.SMEM)
            else:
                # Since we use maximum upper bound when allocating scheduler data,
                # it is not feasible to use scalar prefetch and fetch entire scheduler
                # data into the kernel. Instead, we stored it to HBM first and perform
                # dynamic sized DMA inside the kernel using actual number of steps.
                return pl.BlockSpec(memory_space=pltpu.HBM)

        return jax.tree.map(wrapper, self)

    def out_specs(self):
        """Returns a Pytree of output BlockSpecs."""

        return jax.tree.map(
            lambda x: pl.BlockSpec(memory_space=pltpu.HBM),
            self,
        )


def _mask_out_steps(
    step: jax.typing.ArrayLike,
    schedule_smem: RpaSchedule,
    b_idx: jax.typing.ArrayLike,
):
    """Mask out unvisited metadata schedule entries at (step, b_idx)."""
    schedule_smem.s_idx[step, b_idx] = -1
    schedule_smem.q_idx[step, b_idx] = 0
    schedule_smem.k_idx[step, b_idx] = 0
    schedule_smem.is_last_k[step, b_idx] = 0
    schedule_smem.do_writeback[step, b_idx] = 0

    schedule_smem.dma_q[step, b_idx, 0] = 0
    schedule_smem.dma_q[step, b_idx, 1] = 0

    if schedule_smem.dma_kv_cache is not None:
        for i in range(schedule_smem.cfgs.bkv_p_cache):
            schedule_smem.dma_kv_cache[step, b_idx, i, 0] = 0
            schedule_smem.dma_kv_cache[step, b_idx, i, 1] = 0
            schedule_smem.dma_kv_cache[step, b_idx, i, 2] = 0
    else:
        schedule_smem.kv_cache_tokens[step, b_idx] = 0

    for i in range(schedule_smem.cfgs.bkv_p_new):
        dma_entry = schedule_smem.dma_kv_new[step, b_idx, i]
        dma_entry.fetch_hbm[...] = 0
        dma_entry.fetch_vmem[...] = 0
        dma_entry.wb_hbm[...] = 0
        dma_entry.wb_vmem[...] = 0
        dma_entry.set_flags(0, 0)

    schedule_smem.total_wait_kv_in[step] = 0
    schedule_smem.total_wait_kv_out[step] = 0
    schedule_smem.total_wait_q_in[step] = 0
    schedule_smem.total_wait_o_out[step] = 0
    schedule_smem.total_wait_lse_out[step] = 0


def _write_schedule_to_hbm(
    schedule_smem: RpaSchedule,
    schedule_hbm: RpaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    num_steps: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    """Writes `num_steps` of metadata from `schedule_smem` to
    `schedule_hbm` at `hbm_offset`."""
    hbm_offset_aligned = pl.multiple_of(
        cast(jax.Array, hbm_offset), 128
    )  # pytype: disable=bad-argument-type
    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm)
    flat_smem = jax.tree_util.tree_leaves(schedule_smem)
    dma_list = []
    for h, s in zip(flat_hbm, flat_smem):
        element_size = s.shape[0] // cfgs.max_steps_ub
        if h.shape[0] > 1:
            write_size = num_steps * element_size
            write_size = utils.align_to(write_size, 128)
            output_offset = hbm_offset_aligned * element_size
        else:
            write_size = h.shape[0]
            output_offset = 0

        copy = pltpu.make_async_copy(
            s.at[pl.ds(0, write_size)],
            h.at[pl.ds(output_offset, write_size)],
            dma_sem.at[0],
        )
        dma_list.append(copy)

    jax.tree.map(lambda x: x.start(), dma_list)
    jax.tree.map(lambda x: x.wait(), dma_list)


def _lane_wait_tokens(
    schedule: RpaSchedule,
    step: jax.typing.ArrayLike,
    b: int,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Tokens lane b of `step` moves: (kv in, kv out, q in, o out)."""
    kv_in = schedule.kv_cache_tokens_in(step, b)
    for i in range(cfgs.bkv_p_new):
        kv_in += schedule.dma_kv_new[step, b, i].fetch_val

    kv_out = 0
    do_writeback = schedule.do_writeback[step, b] == 1
    for i in range(cfgs.bkv_p_new):
        dma_entry = schedule.dma_kv_new[step, b, i]
        kv_out += jnp.where(do_writeback, dma_entry.wb_val, 0)

    _, q_sz = schedule.get_dma_q(step, b)
    is_last_k = schedule.is_last_k[step, b] == 1
    return kv_in, kv_out, q_sz, jnp.where(is_last_k, q_sz, 0)


def _store_step_waits(
    schedule: RpaSchedule,
    step: jax.typing.ArrayLike,
    tokens: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    cfgs: configs.RpaConfigs,
):
    """Stores the wait row/lane counts of `step` from its summed lane tokens."""
    kv_in_tokens, kv_out_tokens, q_in_tokens, o_out_tokens = tokens
    # Waits count contiguous 512-byte rows: 128 u32 lanes.
    num_lanes = pltpu.get_tpu_info().num_lanes
    dma_chunk_size = num_lanes * 4

    schedule.total_wait_kv_in[step] = (
        kv_in_tokens * cfgs.kv_bytes_per_token
    ) // dma_chunk_size
    schedule.total_wait_kv_out[step] = (
        kv_out_tokens * cfgs.kv_bytes_per_token
    ) // dma_chunk_size
    schedule.total_wait_q_in[step] = (
        q_in_tokens * cfgs.q_bytes_per_token
    ) // dma_chunk_size
    schedule.total_wait_o_out[step] = (
        o_out_tokens * cfgs.o_bytes_per_token
    ) // dma_chunk_size
    lse_bytes_per_token = (
        cfgs.model.num_kv_heads
        * cfgs.aligned_num_q_heads_per_kv_head
        * 128
        * jnp.dtype(cfgs.serve.dtype_out).itemsize
    )
    schedule.total_wait_lse_out[step] = (
        o_out_tokens * lse_bytes_per_token
    ) // dma_chunk_size


@jax.named_scope("compute_waits")
def _compute_step_waits(
    schedule: RpaSchedule,
    step: jax.typing.ArrayLike,
    *,
    cfgs: configs.RpaConfigs,
):
    """Computes the total wait row/lane counts of one step from its entries."""
    lanes = [
        _lane_wait_tokens(schedule, step, b, cfgs=cfgs) for b in range(cfgs.batch_size)
    ]
    tokens = tuple(sum(lane[i] for lane in lanes) for i in range(4))
    _store_step_waits(schedule, step, tokens, cfgs=cfgs)


def _compute_waits(
    schedule: RpaSchedule,
    start_step: jax.typing.ArrayLike,
    end_step: jax.typing.ArrayLike,
    *,
    cfgs: configs.RpaConfigs,
):
    """Computes total wait row/lane counts for a range of steps."""

    def body(step, _):
        _compute_step_waits(schedule, step, cfgs=cfgs)

    jax.lax.fori_loop(start_step, end_step, body, None)


def flush_to_hbm(
    count: jax.typing.ArrayLike,
    schedule: RpaSchedule,
    schedule_hbm_ref: RpaSchedule,
    hbm_offset: jax.typing.ArrayLike,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
):
    last_step = utils.align_to(count, cfgs.batch_size)

    @pl.loop(count, last_step)
    @jax.named_scope("mask_out_steps")
    def body(idx):
        step, b_idx = divmod(idx, cfgs.batch_size)
        _mask_out_steps(step, b_idx=b_idx, schedule_smem=schedule)

    _compute_waits(schedule, 0, last_step // cfgs.batch_size, cfgs=cfgs)
    _write_schedule_to_hbm(
        schedule,
        schedule_hbm_ref,
        hbm_offset,
        cfgs.max_steps_ub,
        dma_sem,
        cfgs=cfgs,
    )

    return hbm_offset + cfgs.max_steps_ub, 0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class LoopCarry:
    hbm_offset: jax.Array | int
    count: jax.Array | int


class BaseMetadataComputer:
    """Base metadata scheduler computing seq -> q -> k loops."""

    @classmethod
    def get_rpa_schedule(
        cls,
        cfgs: configs.RpaConfigs,
        multiplier: int = 1,
        ring_steps: int | None = None,
    ) -> RpaSchedule:
        """Returns the RpaSchedule shape/dtype struct for this metadata computer."""
        dma_kv_new_struct_cls = (
            HeadAlongSublaneDmaNew
            if cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
            else SeqAlongLaneDmaNew
        )
        return RpaSchedule.create_shape_dtype(
            cfgs,
            dma_kv_new_struct_cls=dma_kv_new_struct_cls,
            multiplier=multiplier,
            ring_steps=ring_steps,
        )

    def __init__(
        self,
        schedule: RpaSchedule,
        schedule_hbm_ref: RpaSchedule,
        dma_sem: jax.Ref,
        *,
        cfgs: configs.RpaConfigs,
        extra_refs: Sequence[jax.Ref] = (),
        **kwargs,
    ):
        self.schedule = schedule
        self.schedule_hbm_ref = schedule_hbm_ref
        self.dma_sem = dma_sem
        self.cfgs = cfgs
        self.extra_refs = extra_refs

    @jax.named_scope("k_loop")
    def k_loop(
        self,
        k_idx,
        carry: LoopCarry,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        kv_cache_len,
        kv_new_len,
        end_k_idx,
    ) -> LoopCarry:
        step, target_lane = divmod(carry.count, self.cfgs.batch_size)
        self.write_entry(
            step,
            target_lane,
            k_idx,
            s_idx=s_idx,
            q_idx=q_idx,
            q_end=q_end,
            q_src=q_src,
            q_sz_task=q_sz_task,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            end_k_idx=end_k_idx,
        )
        return self.advance_carry(carry)

    def write_entry(
        self,
        step,
        target_lane,
        k_idx,
        *,
        s_idx,
        q_idx,
        q_end,
        q_src,
        q_sz_task,
        kv_cache_len,
        kv_new_len,
        end_k_idx,
    ) -> None:
        """Writes the schedule entry of KV block `k_idx` at (step, target_lane)."""
        cfgs = self.cfgs
        schedule = self.schedule

        schedule.s_idx[step, target_lane] = s_idx
        schedule.q_idx[step, target_lane] = q_idx
        schedule.k_idx[step, target_lane] = k_idx

        is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
        schedule.is_last_k[step, target_lane] = is_last_k

        schedule.dma_q[step, target_lane, 0] = q_src
        schedule.dma_q[step, target_lane, 1] = q_sz_task

        kv_len_start = k_idx * cfgs.bkv_sz
        k_len = kv_cache_len + kv_new_len
        kv_left = k_len - kv_len_start
        if cfgs.serve.skip_kv_update:
            kv_left_frm_cache = kv_left
        else:
            kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)
        # The pages' DMAs follow from this count and (s_idx, k_idx); the kernel
        # derives them per step rather than reading a per-page table.
        schedule.kv_cache_tokens[step, target_lane] = _aligned_kv_cache_tokens(
            kv_left_frm_cache, cfgs=cfgs
        )

        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

        # Writeback logic: each new k block is written back by the first q block
        # that attends to it.
        q_wb = jnp.maximum(0, (kv_len_start - kv_cache_len)) // cfgs.bq_sz
        do_writeback = jnp.where((new_sz > 0) & (q_idx == q_wb), 1, 0)
        schedule.do_writeback[step, target_lane] = do_writeback
        src_hbm = q_end - kv_left_frm_new

        def fill_dma_kv_new(i, dst_vmem, dma_sz):
            dma_entry = schedule.dma_kv_new[step, target_lane, i]
            if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
                align = cfgs.kv_token_align
                # Fetch: the new tokens are contiguous, so one DMA moves all of them
                # even though this loop runs once per page. Entry 0 carries that DMA
                # and the other entries stay zero.
                if i == 0:
                    src, _, fetch_val = utils.align_span(src_hbm, new_sz, align)
                    dma_entry.fetch_hbm[...] = src
                    dma_entry.fetch_vmem[...] = utils.align_to(bkv_sz_cache, align)
                    fetch_val = jnp.where(new_sz > 0, fetch_val, 0)
                else:
                    fetch_val = 0

                # Writeback: one DMA per page, since page_indices scatters them.
                tok_idx = kv_len_start + dst_vmem
                p_idx = jnp.minimum(
                    tok_idx >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq - 1,
                )
                p_off = tok_idx & cfgs.serve.page_size_mask
                dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)
                dma_entry.wb_hbm[...] = (
                    (s_idx * cfgs.serve.pages_per_seq + p_idx)
                    << cfgs.serve.page_size_log2
                ) | dst
                dma_entry.wb_vmem[...] = dst_vmem - lead
                dma_entry.set_flags(fetch_val, jnp.where(dma_sz > 0, wb_val, 0))
            else:
                p_idx = jnp.minimum(
                    (kv_len_start + dst_vmem) >> cfgs.serve.page_size_log2,
                    cfgs.serve.pages_per_seq - 1,
                )
                p_off = (kv_len_start + dst_vmem) & cfgs.serve.page_size_mask
                dst_hbm = (
                    (s_idx * cfgs.serve.pages_per_seq + p_idx)
                    << cfgs.serve.page_size_log2
                ) | p_off

                dma_entry.fetch_hbm[...] = src_hbm
                dma_entry.fetch_vmem[...] = dst_vmem
                dma_entry.wb_hbm[...] = dst_hbm
                dma_entry.set_flags(dma_sz, dma_sz)

        if cfgs.bkv_p_new < cfgs.bkv_p:
            # General decode path for any bkv_p_new
            curr_vmem = bkv_sz_cache
            curr_rem = new_sz
            for i in range(cfgs.bkv_p_new):
                slot_start = (curr_vmem // cfgs.serve.page_size) * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size
                dma_sz = jnp.minimum(curr_rem, slot_end - curr_vmem)
                dma_sz = jnp.where(curr_rem > 0, dma_sz, 0)
                fill_dma_kv_new(i, curr_vmem, dma_sz)
                curr_vmem += dma_sz
                curr_rem = jnp.maximum(0, curr_rem - dma_sz)
        else:
            iters = max(cfgs.bkv_p, cfgs.bkv_p_new)
            for i in range(iters):
                slot_start = i * cfgs.serve.page_size
                slot_end = slot_start + cfgs.serve.page_size

                dst_vmem = jnp.maximum(slot_start, bkv_sz_cache)
                end_in_slot = jnp.minimum(slot_end, bkv_sz_cache + new_sz)
                dma_sz = jnp.maximum(0, end_in_slot - dst_vmem)

                fill_dma_kv_new(i, dst_vmem, dma_sz)

    def advance_carry(self, carry: LoopCarry) -> LoopCarry:
        new_count = carry.count + 1
        return jax.lax.cond(
            new_count < self.cfgs.max_steps_ub * self.cfgs.batch_size,
            lambda c: c,
            self.flush_carry,
            LoopCarry(carry.hbm_offset, new_count),
        )

    def flush_carry(self, carry: LoopCarry) -> LoopCarry:
        new_hbm_offset, _ = flush_to_hbm(
            carry.count,
            self.schedule,
            self.schedule_hbm_ref,
            carry.hbm_offset,
            self.dma_sem,
            cfgs=self.cfgs,
        )
        return LoopCarry(new_hbm_offset, 0)

    def q_block(self, q_idx, *, q_start, q_end, q_offset, kv_cache_len, num_k):
        """(q_src, q_sz_task, start_k_idx, end_k_idx) of q block `q_idx`."""
        cfgs = self.cfgs
        q_src = q_start + q_idx * cfgs.bq_sz
        q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

        start_k_idx = 0
        if (sliding_window := cfgs.model.sliding_window) is not None:
            sw_start_idx = q_offset + q_idx * cfgs.bq_sz - sliding_window + 1
            start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

        end_k_idx_causal = (
            q_offset + q_idx * cfgs.bq_sz + q_sz_task - 1
        ) // cfgs.bkv_sz + 1
        end_k_idx = jnp.minimum(num_k, end_k_idx_causal)
        if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
            start_k_idx = jnp.maximum(start_k_idx, kv_cache_len // cfgs.bkv_sz)
        return q_src, q_sz_task, start_k_idx, end_k_idx

    @jax.named_scope("q_loop")
    def q_loop(
        self,
        q_idx,
        carry: LoopCarry,
        *,
        s_idx,
        q_start,
        q_end,
        q_offset,
        kv_cache_len,
        kv_new_len,
        num_k,
    ) -> LoopCarry:
        q_src, q_sz_task, start_k_idx, end_k_idx = self.q_block(
            q_idx,
            q_start=q_start,
            q_end=q_end,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            num_k=num_k,
        )
        k_loop_fn = functools.partial(
            self.k_loop,
            s_idx=s_idx,
            q_idx=q_idx,
            q_end=q_end,
            q_src=q_src,
            q_sz_task=q_sz_task,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            end_k_idx=end_k_idx,
        )
        return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

    @jax.named_scope("seq_loop")
    def seq_loop(
        self,
        s_idx,
        carry: LoopCarry,
        *,
        cu_q_lens_ref,
        q_offsets_ref,
        kv_cache_lens_ref,
        kv_new_lens_ref,
    ) -> LoopCarry:
        q_start = cu_q_lens_ref[s_idx]
        q_end = cu_q_lens_ref[s_idx + 1]
        q_len = q_end - q_start
        q_offset = q_offsets_ref[s_idx]
        kv_cache_len = kv_cache_lens_ref[s_idx]
        kv_new_len = kv_new_lens_ref[s_idx]
        k_len = kv_cache_len + kv_new_len
        num_q = pl.cdiv(q_len, self.cfgs.bq_sz)
        num_k = pl.cdiv(k_len, self.cfgs.bkv_sz)
        q_loop_fn = functools.partial(
            self.q_loop,
            s_idx=s_idx,
            q_start=q_start,
            q_end=q_end,
            q_offset=q_offset,
            kv_cache_len=kv_cache_len,
            kv_new_len=kv_new_len,
            num_k=num_k,
        )

        return jax.lax.fori_loop(0, num_q, q_loop_fn, carry)

    def compute(
        self,
        cu_q_lens_ref: jax.Ref,
        q_offsets_ref: jax.Ref,
        kv_cache_lens_ref: jax.Ref,
        kv_new_lens_ref: jax.Ref,
        distribution_ref: jax.Ref,
    ) -> LoopCarry:
        """Generates the metadata schedule across all sequences."""
        seq_loop_fn = functools.partial(
            self.seq_loop,
            cu_q_lens_ref=cu_q_lens_ref,
            q_offsets_ref=q_offsets_ref,
            kv_cache_lens_ref=kv_cache_lens_ref,
            kv_new_lens_ref=kv_new_lens_ref,
        )
        start_seq_idx, end_seq_idx = self.cfgs.seq_range(
            distribution_ref
        )  # pytype: disable=bad-argument-type
        init_carry = LoopCarry(0, 0)
        return jax.lax.fori_loop(start_seq_idx, end_seq_idx, seq_loop_fn, init_carry)


# RPAd can generate its decode schedule itself (`DecodeScheduleStream`): step
# t writes step t + STREAM_LOOKAHEAD, after step t's copy_in has read step t + 2,
# and the pipeline reads steps t - 2 .. t + 2, so the SMEM ring holds at least
# STREAM_LOOKAHEAD + 3 steps.
STREAM_LOOKAHEAD = 3
STREAM_RING_STEPS = 8
assert STREAM_LOOKAHEAD >= 3
assert STREAM_RING_STEPS >= STREAM_LOOKAHEAD + 3
assert STREAM_RING_STEPS & (STREAM_RING_STEPS - 1) == 0


def streams_decode_schedule(
    cfgs: configs.RpaConfigs, computer_cls: type[BaseMetadataComputer]
) -> bool:
    """Whether the RPA kernel generates this call's schedule itself."""
    return (
        cfgs.mode == configs.RpaCase.DECODE
        and computer_cls is BaseMetadataComputer
        # One q block per decode sequence.
        and cfgs.bq_sz >= cfgs.serve.decode_query_size
    )


class DecodeScheduleStream:
    """Decode schedule entries generated in-kernel, equal to the schedule kernel's.

    A decode sequence has one q block, so the schedule kernel's work items are
    the KV blocks start_k..end_k - 1 of each sequence in turn, item i landing on
    (step i // batch_size, lane i % batch_size). `init` tabulates the sequences
    that have items, with their per-sequence schedule values; `generate_step`
    then walks a cursor over that table, one item per lane, with no branches,
    so a step's generation interleaves with the RPA step's vector work.
    """

    # Columns of the per-sequence table.
    _S_IDX, _ITEM_END, _K_OFFSET, _Q_END, _Q_SRC, _Q_SZ, _KV_CACHE, _KV_NEW, _END_K = (
        range(9)
    )
    _NUM_COLS = 9

    @classmethod
    def scratch_shapes(cls, cfgs: configs.RpaConfigs):
        # One row per sequence with items, then a sentinel row that ends the walk.
        return (
            pltpu.SMEM(((cfgs.serve.num_seqs + 1) * cls._NUM_COLS,), jnp.int32),
            pltpu.SMEM((2,), jnp.int32),  # cursor row, total items
        )

    def __init__(
        self,
        schedule: RpaSchedule,
        scratch: tuple[jax.Ref, jax.Ref],
        *,
        cu_q_lens_ref: jax.Ref,
        q_offsets_ref: jax.Ref,
        kv_cache_lens_ref: jax.Ref,
        kv_new_lens_ref: jax.Ref,
        distribution_ref: jax.Ref,
        cfgs: configs.RpaConfigs,
    ):
        self.schedule = schedule
        self.table_ref, self.state_ref = scratch
        self.cu_q_lens_ref = cu_q_lens_ref
        self.q_offsets_ref = q_offsets_ref
        self.kv_cache_lens_ref = kv_cache_lens_ref
        self.kv_new_lens_ref = kv_new_lens_ref
        self.distribution_ref = distribution_ref
        self.cfgs = cfgs

    def _cell(self, row, col):
        return self.table_ref.at[row * self._NUM_COLS + col]

    def init(self) -> jax.Array:
        """Tabulates the sequences with items and returns the number of steps."""
        cfgs = self.cfgs
        computer = BaseMetadataComputer(self.schedule, None, None, cfgs=cfgs)
        start_seq, end_seq = cfgs.seq_range(self.distribution_ref)
        assert isinstance(start_seq, int) and start_seq == 0

        def body(s_idx, carry):
            row, total = carry
            q_start = self.cu_q_lens_ref[s_idx]
            q_end = self.cu_q_lens_ref[s_idx + 1]
            kv_cache_len = self.kv_cache_lens_ref[s_idx]
            kv_new_len = self.kv_new_lens_ref[s_idx]
            q_src, q_sz_task, start_k, end_k = computer.q_block(
                0,
                q_start=q_start,
                q_end=q_end,
                q_offset=self.q_offsets_ref[s_idx],
                kv_cache_len=kv_cache_len,
                num_k=pl.cdiv(kv_cache_len + kv_new_len, cfgs.bkv_sz),
            )
            has_items = (s_idx < end_seq) & (q_end > q_start) & (end_k > start_k)
            num_items = jnp.where(has_items, end_k - start_k, 0)
            # Written unconditionally: a sequence without items leaves `row` where
            # it is, and the next sequence (or the sentinel) overwrites it.
            values = {
                self._S_IDX: s_idx,
                self._ITEM_END: total + num_items,
                self._K_OFFSET: start_k - total,
                self._Q_END: q_end,
                self._Q_SRC: q_src,
                self._Q_SZ: q_sz_task,
                self._KV_CACHE: kv_cache_len,
                self._KV_NEW: kv_new_len,
                self._END_K: end_k,
            }
            for col, value in values.items():
                self._cell(row, col)[...] = value
            return row + has_items.astype(jnp.int32), total + num_items

        num_rows, total = jax.lax.fori_loop(
            0, cfgs.serve.num_seqs, body, (jnp.int32(0), jnp.int32(0))
        )
        # Sentinel: the cursor never passes it, and its zeros give the schedule
        # kernel's masked entry.
        for col in range(self._NUM_COLS):
            self._cell(num_rows, col)[...] = 0
        self._cell(num_rows, self._ITEM_END)[...] = jnp.iinfo(jnp.int32).max
        self.state_ref[0] = 0
        self.state_ref[1] = total
        return pl.cdiv(total, cfgs.batch_size)

    @jax.named_scope("stream_decode_schedule")
    def generate_step(self, step: jax.typing.ArrayLike) -> None:
        """Writes every lane's entry of `step` and the step's wait counts.

        Steps must be generated in order. Each lane's entry is built in
        registers, and the wait counts are summed from those values rather than
        read back from SMEM.
        """
        cfgs = self.cfgs
        recorder = _StepRecorder(cfgs)
        computer = BaseMetadataComputer(recorder.schedule, None, None, cfgs=cfgs)
        row = self.state_ref[0]
        total = self.state_ref[1]
        tokens = (0, 0, 0, 0)
        for lane in range(cfgs.batch_size):
            item = step * cfgs.batch_size + lane
            # Every tabulated sequence has an item, so one item moves at most a row.
            row = row + (item >= self._cell(row, self._ITEM_END)[...]).astype(jnp.int32)

            def col(c, row=row):
                return self._cell(row, c)[...]

            valid = item < total
            s_idx = col(self._S_IDX)
            computer.write_entry(
                0,
                lane,
                jnp.where(valid, item + col(self._K_OFFSET), 0),
                s_idx=s_idx,
                q_idx=0,
                q_end=col(self._Q_END),
                q_src=col(self._Q_SRC),
                q_sz_task=col(self._Q_SZ),
                kv_cache_len=col(self._KV_CACHE),
                kv_new_len=col(self._KV_NEW),
                end_k_idx=col(self._END_K),
            )
            recorder.schedule.s_idx[0, lane] = jnp.where(valid, s_idx, -1)
            lane_tokens = _lane_wait_tokens(recorder.schedule, 0, lane, cfgs=cfgs)
            tokens = tuple(t + n for t, n in zip(tokens, lane_tokens))
            recorder.store(self.schedule, step)
        self.state_ref[0] = row
        _store_step_waits(self.schedule, step, tokens, cfgs=cfgs)


class _RecordedRef:
    """Stands in for one schedule field's SMEM ref, keeping stores as values."""

    def __init__(self):
        self.values = {}

    def __setitem__(self, pos, value):
        self.values[pos] = value

    def __getitem__(self, pos):
        return self.values[pos]

    @property
    def at(self):
        return _RecordedAt(self)


@dataclasses.dataclass(frozen=True)
class _RecordedAt:
    ref: _RecordedRef

    def __getitem__(self, pos):
        return _RecordedCell(self.ref, pos)


@dataclasses.dataclass(frozen=True)
class _RecordedCell:
    ref: _RecordedRef
    pos: int

    def __setitem__(self, idx, value):
        assert idx is Ellipsis
        self.ref[self.pos] = value

    def __getitem__(self, idx):
        assert idx is Ellipsis
        return self.ref[self.pos]


class _StepRecorder:
    """A one-step RpaSchedule whose fields hold traced values, not SMEM.

    Indexing it at step 0 with static lanes gives static positions, which double
    as the field offsets within one step of the real schedule.
    """

    def __init__(self, cfgs: configs.RpaConfigs):
        shapes = BaseMetadataComputer.get_rpa_schedule(cfgs, ring_steps=1)
        self.step_sizes = [x.shape[0] for x in jax.tree_util.tree_leaves(shapes)]
        self.schedule = jax.tree.map(lambda _: _RecordedRef(), shapes)

    def store(self, schedule: RpaSchedule, step: jax.typing.ArrayLike) -> None:
        """Stores the values recorded since the last call into `step` of `schedule`."""
        ring_steps = schedule.s_idx.ring_steps
        slot = step if ring_steps is None else step & (ring_steps - 1)
        recorded = jax.tree_util.tree_leaves(
            self.schedule, is_leaf=lambda x: isinstance(x, _RecordedRef)
        )
        refs = jax.tree_util.tree_leaves(schedule)
        assert len(recorded) == len(refs) == len(self.step_sizes)
        for rec, ref, size in zip(recorded, refs, self.step_sizes):
            for pos, value in rec.values.items():
                ref[slot * size + pos] = value
            rec.values.clear()


def rpa_metadata_schedule_kernel(
    ## Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    distribution_ref: jax.Ref,
    extra_scalars_ref: Sequence[jax.Ref],
    # Outputs.
    schedule_hbm_ref: RpaSchedule,
    # Scratch.
    schedule_ref: RpaSchedule,
    dma_sem: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    computer_cls: type[BaseMetadataComputer] = BaseMetadataComputer,
):
    """Generates the HBM-to-VMEM DMA schedule.

    This kernel:
    1. Iterates through each (potentially ragged) sequence
    2. Breaks Queries (Q) and Key-Values (KV) into blocks (bq_sz, bkv_sz).
    3. Assigns tasks to 'lanes' (TPU batch items) based on current lane occupancy
        to ensure balanced execution across the batch dimension.
    4. Encodes DMA offsets:
        - dma_q: HBM start index and size for Query blocks.
        - dma_kv_cache: Paged indices for existing KV tokens.
        - dma_kv_new: offsets for new tokens being added to the cache.
        - do_writeback: boolean flag indicating if a block should be flushed to
        HBM (ie does this block contain new tokens to add to KV cache).

    Args:
        cu_q_lens_ref: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        q_offsets_ref: [max_num_seqs]. Starting Q index for each sequence.
        kv_cache_lens_ref: [max_num_seqs]. Existing kv cache length of each
            sequence.
        kv_new_lens_ref: [max_num_seqs]. New kv length of each sequence.
        distribution_ref: [3]. Cumulative sum of number of decode, prefill, and
            mixed
        extra_scalars_ref: Additional scalar refs for custom metadata computers.
        schedule_hbm_ref: HBM memory that will store output of the kernel.
        schedule_ref: Scratch memory where schedule results gets written.
        dma_sem: Semaphore used for writing scheduler output to HBM.
        cfgs: Configuration of the kernel.
        computer_cls: Metadata computer class to use for schedule generation.
    """

    # Step 1: Compute and fill scheduler metadata.
    computer = computer_cls(
        schedule=schedule_ref,
        schedule_hbm_ref=schedule_hbm_ref,
        dma_sem=dma_sem,
        cfgs=cfgs,
        extra_refs=extra_scalars_ref,
    )
    loop_carry = computer.compute(
        cu_q_lens_ref=cu_q_lens_ref,
        q_offsets_ref=q_offsets_ref,
        kv_cache_lens_ref=kv_cache_lens_ref,
        kv_new_lens_ref=kv_new_lens_ref,
        distribution_ref=distribution_ref,
    )
    count = loop_carry.count
    hbm_offset = loop_carry.hbm_offset
    steps = pl.cdiv(count, cfgs.batch_size) + hbm_offset
    schedule_ref.actual_steps[0] = steps
    # Step 3: Mask out unvisited steps.
    flush_to_hbm(
        count,
        schedule_ref,
        schedule_hbm_ref,
        hbm_offset,
        dma_sem,
        cfgs=cfgs,
    )


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    q_offsets: jax.Array,
    kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    distribution: jax.Array,
    cfgs: configs.RpaConfigs,
    *,
    computer_cls: type[BaseMetadataComputer] = BaseMetadataComputer,
    extra_scalars: Sequence[jax.Array] = (),
    interpret=False,
) -> RpaSchedule:
    """Generates RPA metadata schedule using the specified computer_cls."""
    schedule_shaped_dtype = computer_cls.get_rpa_schedule(cfgs)
    schedule_hbm = computer_cls.get_rpa_schedule(
        cfgs, multiplier=cfgs.max_schedule_size_multiplier
    )
    return pl.pallas_call(
        functools.partial(
            rpa_metadata_schedule_kernel,
            cfgs=cfgs,
            computer_cls=computer_cls,
        ),
        out_shape=schedule_hbm,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=[],
            out_specs=schedule_hbm.out_specs(),
            scratch_shapes=[
                schedule_shaped_dtype.scratch_shapes(),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        interpret=interpret,
        name="rpa_metadata_schedule",
    )(
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        distribution,
        extra_scalars,
    )
