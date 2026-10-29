from abc import ABC, abstractmethod
from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, ClassVar, TYPE_CHECKING

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.batched_rpa_new import configs, utils


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
  def num_fields() -> int:
    ...

  @abstractmethod
  def set_flags(self, fetch_val, wb_val):
    ...

  @property
  @abstractmethod
  def fetch_val(self):
    ...

  @property
  @abstractmethod
  def wb_val(self):
    ...


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

  @classmethod
  def create_shape_dtype(cls, shape):
    return cls(
        data=jax.ShapeDtypeStruct((np.prod(shape),), jnp.int32), shape=shape
    )

  def _get_pos(self, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
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
  def create_shape_dtype(cls, shape, struct_cls, struct_size):  # pyrefly: ignore[bad-override]
    assert struct_size == struct_cls.num_fields()
    return cls(
        data=jax.ShapeDtypeStruct((np.prod(shape) * struct_size,), jnp.int32),
        shape=shape,
        struct_cls=struct_cls,
        struct_size=struct_size,
    )

  def _get_pos(self, indices):
    strides = pl.strides_from_shape(self.shape)
    assert len(strides) == len(indices)

    pos = 0
    for stride, idx in zip(strides, indices):
      pos += stride * idx
    return pos * self.struct_size

  def __getitem__(self, indices):
    pos_start = self._get_pos(indices)
    return self.struct_cls(self.data, pos_start)


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
  dma_kv_cache: SmemWrapper  # [steps, batch, bkv_p_cache, 3]
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
  ):
    effective_max_steps = cfgs.max_steps_ub * multiplier

    idx_wrapper = SmemWrapper.create_shape_dtype(
        (effective_max_steps, cfgs.batch_size)
    )

    steps_wrapper = SmemWrapper.create_shape_dtype((effective_max_steps,))

    return cls(
        s_idx=idx_wrapper,
        q_idx=idx_wrapper,
        k_idx=idx_wrapper,
        is_last_k=idx_wrapper,
        do_writeback=idx_wrapper,
        dma_q=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, 2)
        ),
        dma_kv_cache=SmemWrapper.create_shape_dtype(
            (effective_max_steps, cfgs.batch_size, cfgs.bkv_p_cache, 3)
        ),
        dma_kv_new=SmemArrayOfStructs.create_shape_dtype(
            (
                effective_max_steps,
                cfgs.batch_size,
                cfgs.bkv_p_new,
            ),
            struct_cls=dma_kv_new_struct_cls,
            struct_size=cfgs.dma_kv_new_size,
        ),
        total_wait_kv_in=steps_wrapper,
        total_wait_kv_out=steps_wrapper,
        total_wait_q_in=steps_wrapper,
        total_wait_o_out=steps_wrapper,
        total_wait_lse_out=steps_wrapper,
        actual_steps=jax.ShapeDtypeStruct((1,), jnp.int32),
        cfgs=cfgs,
    )

  def get_dma_kv_cache(
      self,
      step: jax.typing.ArrayLike,
      batch_idx: jax.typing.ArrayLike,
      page_idx: jax.typing.ArrayLike,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    # 0: src_hbm, 1: dst_vmem, 2: size
    src_off = self.dma_kv_cache[step, batch_idx, page_idx, 0]
    dst_off = self.dma_kv_cache[step, batch_idx, page_idx, 1]
    sz = self.dma_kv_cache[step, batch_idx, page_idx, 2]
    return src_off, dst_off, sz

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

  for i in range(schedule_smem.cfgs.bkv_p_cache):
    schedule_smem.dma_kv_cache[step, b_idx, i, 0] = 0
    schedule_smem.dma_kv_cache[step, b_idx, i, 1] = 0
    schedule_smem.dma_kv_cache[step, b_idx, i, 2] = 0

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
  """Writes `num_steps` of metadata from `schedule_smem` to `schedule_hbm` at `hbm_offset`."""
  hbm_offset_aligned = pl.multiple_of(hbm_offset, 128)  # pyrefly: ignore[bad-argument-type]
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


def _compute_waits(
    schedule: RpaSchedule,
    start_step: jax.typing.ArrayLike,
    end_step: jax.typing.ArrayLike,
    *,
    cfgs: configs.RpaConfigs,
):
  """Computes total wait row/lane counts for a range of steps."""

  kv_bytes_per_token = cfgs.kv_bytes_per_token
  q_bytes_per_token = cfgs.q_bytes_per_token
  o_bytes_per_token = cfgs.o_bytes_per_token

  # Reshape buffers to (-1, 128) in u32 to perform contiguous 512-byte lane
  # waits (1 row of 128 u32 words = 512 bytes).
  num_lanes = pltpu.get_tpu_info().num_lanes
  dma_chunk_size = num_lanes * 4

  @jax.named_scope("compute_waits")
  def body(step, _):
    # KV IN
    kv_in_tokens = 0
    for b in range(cfgs.batch_size):
      for i in range(cfgs.bkv_p_cache):
        _, _, sz = schedule.get_dma_kv_cache(step, b, i)
        kv_in_tokens += sz
      for i in range(cfgs.bkv_p_new):
        kv_in_tokens += schedule.dma_kv_new[step, b, i].fetch_val

    schedule.total_wait_kv_in[step] = (
        kv_in_tokens * kv_bytes_per_token
    ) // dma_chunk_size

    # KV OUT
    kv_out_tokens = 0
    for b in range(cfgs.batch_size):
      do_writeback = schedule.do_writeback[step, b] == 1
      for i in range(cfgs.bkv_p_new):
        dma_entry = schedule.dma_kv_new[step, b, i]
        kv_out_tokens += jnp.where(do_writeback, dma_entry.wb_val, 0)

    schedule.total_wait_kv_out[step] = (
        kv_out_tokens * kv_bytes_per_token
    ) // dma_chunk_size

    # Q IN
    q_in_tokens = 0
    for b in range(cfgs.batch_size):
      _, q_sz = schedule.get_dma_q(step, b)
      q_in_tokens += q_sz
    schedule.total_wait_q_in[step] = (
        q_in_tokens * q_bytes_per_token
    ) // dma_chunk_size

    # O OUT
    o_out_tokens = 0
    for b in range(cfgs.batch_size):
      is_last_k = schedule.is_last_k[step, b] == 1
      _, q_sz = schedule.get_dma_q(step, b)
      o_out_tokens += jnp.where(is_last_k, q_sz, 0)
    schedule.total_wait_o_out[step] = (
        o_out_tokens * o_bytes_per_token
    ) // dma_chunk_size

    # LSE OUT
    lse_bytes_per_token = (
        cfgs.model.num_kv_heads
        * cfgs.lse_rows_per_token
        * 128
        * jnp.dtype(cfgs.serve.dtype_out).itemsize
    )
    schedule.total_wait_lse_out[step] = pl.cdiv(
        o_out_tokens * lse_bytes_per_token, dma_chunk_size
    )

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ComputeMetadata:
  """Scalars needed to build the QK mask, but not needed by the stitch."""

  causal_offset: list[jax.Array]
  local_k_start: list[jax.Array] | None = None
  local_k_end: list[jax.Array] | None = None
  interleave_meta: list[tuple[jax.Array, ...]] | None = None


class BaseMaskComputer(ABC):
  """Pluggable causal and valid-KV mask strategy for schedule and main kernels."""

  @classmethod
  @abstractmethod
  def get_causal_end_k_idx(
      cls,
      *,
      q_idx: jax.Array,
      q_sz_task: jax.Array,
      q_position: jax.Array,
      kv_cache_len: jax.Array,
      kv_new_len: jax.Array,
      cp_rank: jax.Array | int,
      cfgs: configs.RpaConfigs,
      q_start: jax.Array | int = 0,
  ) -> jax.Array:
    """Returns the exclusive upper bound KV block index for causal masking."""

  @classmethod
  @abstractmethod
  def fetch_compute_metadata(
      cls,
      step: jax.Array,
      schedule_ref: RpaSchedule,
      q_positions_ref: jax.Ref,
      kv_cache_lens_ref: jax.Ref,
      cp_rank_ref: jax.Ref,
      *,
      cfgs: configs.RpaConfigs,
      kv_new_lens_ref: jax.Ref | None = None,
      cu_q_lens_ref: jax.Ref | None = None,
  ) -> ComputeMetadata:
    """Fetches per-batch-item scalars needed for QK mask generation."""

  @classmethod
  @abstractmethod
  def generate_mask(
      cls,
      shape: tuple[int, int, int, int],
      *,
      bq_start: int,
      step_meta: ComputeMetadata,
      cfgs: configs.RpaConfigs,
  ) -> list[jax.Array]:
    """Generates the QK boolean mask list across the batch dimension."""


class ContiguousCausalMask(BaseMaskComputer):
  """Standard causal mask where query positions in a block are contiguous."""

  @classmethod
  def get_causal_end_k_idx(
      cls,
      *,
      q_idx: jax.Array,
      q_sz_task: jax.Array,
      q_position: jax.Array,
      kv_cache_len: jax.Array,
      kv_new_len: jax.Array,
      cp_rank: jax.Array | int,
      cfgs: configs.RpaConfigs,
      q_start: jax.Array | int = 0,
  ) -> jax.Array:
    del kv_cache_len, kv_new_len, cp_rank, q_start
    return (q_position + q_idx * cfgs.bq_sz + q_sz_task - 1) // cfgs.bkv_sz + 1

  @classmethod
  def fetch_compute_metadata(
      cls,
      step: jax.Array,
      schedule_ref: RpaSchedule,
      q_positions_ref: jax.Ref,
      kv_cache_lens_ref: jax.Ref,
      cp_rank_ref: jax.Ref,
      *,
      cfgs: configs.RpaConfigs,
      kv_new_lens_ref: jax.Ref | None = None,
      cu_q_lens_ref: jax.Ref | None = None,
  ) -> ComputeMetadata:
    del kv_new_lens_ref, cu_q_lens_ref
    causal_offset_list = []
    cp_size = cfgs.serve.cp_group_size
    cp_rank = cp_rank_ref[0]

    local_k_start_list = [] if not cfgs.serve.has_cache_kv else None
    local_k_end_list = [] if not cfgs.serve.has_new_kv else None

    for b_idx in range(cfgs.batch_size):
      s_idx = schedule_ref.s_idx[step, b_idx]
      is_valid = s_idx != -1
      q_idx = schedule_ref.q_idx[step, b_idx]
      k_idx = schedule_ref.k_idx[step, b_idx]
      if cfgs.ring_enabled:
        assert cp_size is not None
        hop = jax.lax.rem(k_idx, cp_size)
        k_idx = k_idx // cp_size
        src_rank = jax.lax.rem(cp_rank + cp_size - hop, cp_size)
      else:
        src_rank = cp_rank

      k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
      q_position = jnp.where(is_valid, q_positions_ref[s_idx], 0)

      # Causal base offset: K_base - Q_base
      q_base = q_idx * cfgs.bq_sz + q_position
      causal_offset = k_id - q_base
      causal_offset_list.append(causal_offset)

      if local_k_start_list is not None:
        kv_cache_len_val = jnp.where(is_valid, kv_cache_lens_ref[s_idx], 0)
        local_k_start_list.append(kv_cache_len_val - k_id)
      if local_k_end_list is not None:
        kv_cache_len_val = jnp.where(is_valid, kv_cache_lens_ref[s_idx], 0)
        if cfgs.ring_enabled:
          assert cp_size is not None
          kv_cache_len_val = utils.cp_local_cache_len(
              kv_cache_len_val, cp_size, src_rank, cfgs.serve.page_size
          )
        local_k_end_list.append(kv_cache_len_val - k_id)

    return ComputeMetadata(
        causal_offset=causal_offset_list,
        local_k_start=local_k_start_list,
        local_k_end=local_k_end_list,
    )

  @classmethod
  def generate_mask(
      cls,
      shape: tuple[int, int, int, int],
      *,
      bq_start: int,
      step_meta: ComputeMetadata,
      cfgs: configs.RpaConfigs,
  ) -> list[jax.Array]:
    b, k_heads, tq, s = shape

    if not cfgs.serve.has_new_kv and cfgs.model.sliding_window is None:
      return CacheOnlyMask.generate_mask(
          shape, bq_start=bq_start, step_meta=step_meta, cfgs=cfgs
      )

    masks = []
    kv_iota = jax.lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 2)
    q_iota = jax.lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
    q_iota //= cfgs.aligned_num_q_heads_per_kv_head
    q_kv_diff = q_iota - kv_iota

    for b_idx in range(b):
      offset = step_meta.causal_offset[b_idx] - bq_start
      mask_b = q_kv_diff >= offset
      if (sliding_window := cfgs.model.sliding_window) is not None:
        mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)

      if step_meta.local_k_start is not None:
        mask_b = jnp.logical_and(
            mask_b, kv_iota >= step_meta.local_k_start[b_idx]
        )
      if step_meta.local_k_end is not None:
        mask_b = jnp.logical_and(mask_b, kv_iota < step_meta.local_k_end[b_idx])

      masks.append(mask_b)

    return masks


class CacheOnlyMask(ContiguousCausalMask):
  """Mask for cache-only passes where all queries attend to all valid cached KV."""

  @classmethod
  def generate_mask(
      cls,
      shape: tuple[int, int, int, int],
      *,
      bq_start: int,
      step_meta: ComputeMetadata,
      cfgs: configs.RpaConfigs,
  ) -> list[jax.Array]:
    del bq_start, cfgs
    b, _, _, s = shape
    kv_iota = jax.lax.broadcasted_iota(jnp.int32, (1, 1, s), 2)
    assert step_meta.local_k_end is not None
    return [kv_iota < step_meta.local_k_end[b_idx] for b_idx in range(b)]


class BaseMetadataComputer:
  """Base metadata scheduler computing seq -> q -> k loops."""

  rpa_cases: ClassVar[tuple[configs.RpaCase, ...]] = (
      configs.RpaCase.DECODE,
      configs.RpaCase.MIXED,
  )
  ring_enabled: ClassVar[bool] = False
  mask_computer_cls: ClassVar[type[BaseMaskComputer]] = ContiguousCausalMask

  @property
  def cp_rank_val(self) -> jax.Array | int:
    return 0

  @classmethod
  def get_rpa_schedule(
      cls, cfgs: configs.RpaConfigs, multiplier: int = 1
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
    )

  @classmethod
  def get_ring_sems_spec(cls) -> Any:
    """Returns the Pallas semaphore specification for ring rotation, or None."""
    return None

  @classmethod
  def create_ring(
      cls, cfgs: configs.RpaConfigs, ring_sems: Any, *, kv_window_ref: jax.Ref
  ) -> Any:
    """Creates the ring communication handle for the main kernel, or None."""
    del cfgs, ring_sems, kv_window_ref
    return None

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
      update_kv_cache,
      kv_new_start=0,
      num_k=0,
      num_q=0,
      **kwargs,
  ) -> LoopCarry:
    cfgs = self.cfgs
    schedule = self.schedule
    count = carry.count
    step, target_lane = divmod(count, cfgs.batch_size)

    schedule.s_idx[step, target_lane] = s_idx
    schedule.q_idx[step, target_lane] = q_idx
    schedule.k_idx[step, target_lane] = k_idx

    is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
    schedule.is_last_k[step, target_lane] = is_last_k

    schedule.dma_q[step, target_lane, 0] = q_src
    schedule.dma_q[step, target_lane, 1] = q_sz_task

    kv_len_start = k_idx * cfgs.bkv_sz
    kv_p_start = k_idx * cfgs.bkv_p
    k_len = kv_cache_len + kv_new_len
    kv_left = k_len - kv_len_start
    if cfgs.serve.skip_kv_update:
      kv_left_frm_cache = kv_left
    else:
      kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)
    p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

    for i in range(cfgs.bkv_p_cache):
      dst_vmem = i << cfgs.serve.page_size_log2
      dma_sz = kv_left_frm_cache - dst_vmem
      dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

      src_hbm = jnp.minimum(p_offset + i, cfgs.serve.num_page_indices - 1)

      schedule.dma_kv_cache[step, target_lane, i, 0] = src_hbm
      schedule.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
      schedule.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
          dma_sz, cfgs.kv_token_align
      )

    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
    new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

    # Writeback logic: each new k block is written back by the first q block
    # that attends to it.
    q_wb = jnp.maximum(0, kv_len_start - kv_cache_len) // cfgs.bq_sz

    do_writeback = jnp.where(
        (new_sz > 0) & (q_idx == q_wb) & update_kv_cache, 1, 0
    )
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
        dma_entry.wb_vmem[...] = dst_vmem
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

    return self.advance_carry(carry)

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

  @jax.named_scope("q_loop")
  def q_loop(
      self,
      q_idx,
      carry: LoopCarry,
      *,
      s_idx,
      q_start,
      q_end,
      q_position,
      kv_cache_len,
      kv_new_len,
      num_k,
      num_q,
      update_kv_cache,
      kv_new_start=None,
  ) -> LoopCarry:
    cfgs = self.cfgs
    q_src = q_start + q_idx * cfgs.bq_sz
    q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)
    if kv_new_start is None:
      kv_new_start = q_start

    start_k_idx = 0
    if (sliding_window := cfgs.model.sliding_window) is not None:
      sw_start_idx = q_position + q_idx * cfgs.bq_sz - sliding_window + 1
      start_k_idx = jnp.maximum(0, sw_start_idx) // cfgs.bkv_sz

    end_k_idx_causal = self.mask_computer_cls.get_causal_end_k_idx(
        q_idx=q_idx,
        q_sz_task=q_sz_task,
        q_position=q_position,
        kv_cache_len=kv_cache_len,
        kv_new_len=kv_new_len,
        cp_rank=self.cp_rank_val,
        cfgs=cfgs,
        q_start=q_start,
    )
    end_k_idx = jnp.minimum(num_k, end_k_idx_causal)

    if not cfgs.serve.has_cache_kv:
      start_k_idx = jnp.maximum(start_k_idx, kv_cache_len // cfgs.bkv_sz)
      # The last Q block of a sequence visits every new-KV block so all new KV
      # pages are written to the cache. Any extra blocks beyond the causal limit
      # are masked out during attention compute.
      end_k_idx = jnp.where(
          (q_idx == num_q - 1) & update_kv_cache, num_k, end_k_idx
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
        kv_new_start=kv_new_start,
        end_k_idx=end_k_idx,
        num_k=num_k,
        num_q=num_q,
        update_kv_cache=update_kv_cache,
    )

    return jax.lax.fori_loop(start_k_idx, end_k_idx, k_loop_fn, carry)

  @jax.named_scope("seq_loop")
  def seq_loop(
      self,
      s_idx,
      carry: LoopCarry,
      *,
      cu_q_lens_ref,
      q_positions_ref,
      kv_cache_lens_ref,
      kv_new_lens_ref,
      kv_new_starts_ref,
      update_kv_cache_ref,
  ) -> LoopCarry:
    q_start = cu_q_lens_ref[s_idx]
    q_end = cu_q_lens_ref[s_idx + 1]
    q_len = q_end - q_start
    q_position = q_positions_ref[s_idx]
    kv_cache_len = kv_cache_lens_ref[s_idx]
    kv_new_len = kv_new_lens_ref[s_idx]
    k_len = kv_cache_len + kv_new_len
    update_kv_cache = update_kv_cache_ref[s_idx]

    num_q = pl.cdiv(q_len, self.cfgs.bq_sz)
    num_k = pl.cdiv(k_len, self.cfgs.bkv_sz)

    q_loop_fn = functools.partial(
        self.q_loop,
        s_idx=s_idx,
        q_start=q_start,
        q_end=q_end,
        q_position=q_position,
        kv_cache_len=kv_cache_len,
        kv_new_len=kv_new_len,
        num_k=num_k,
        num_q=num_q,
        update_kv_cache=update_kv_cache,
        kv_new_start=kv_new_starts_ref[s_idx],
    )

    return jax.lax.fori_loop(0, num_q, q_loop_fn, carry)

  def compute(
      self,
      cu_q_lens_ref: jax.Ref,
      q_positions_ref: jax.Ref,
      kv_cache_lens_ref: jax.Ref,
      kv_new_lens_ref: jax.Ref,
      kv_new_starts_ref: jax.Ref,
      update_kv_cache_ref: jax.Ref,
      distribution_ref: jax.Ref,
  ) -> LoopCarry:
    """Generates the metadata schedule across all sequences."""
    seq_loop_fn = functools.partial(
        self.seq_loop,
        cu_q_lens_ref=cu_q_lens_ref,
        q_positions_ref=q_positions_ref,
        kv_cache_lens_ref=kv_cache_lens_ref,
        kv_new_lens_ref=kv_new_lens_ref,
        kv_new_starts_ref=kv_new_starts_ref,
        update_kv_cache_ref=update_kv_cache_ref,
    )
    start_seq_idx, end_seq_idx = self.cfgs.seq_range(distribution_ref)  # pyrefly: ignore[bad-argument-type]
    init_carry = LoopCarry(0, 0)
    return jax.lax.fori_loop(
        start_seq_idx, end_seq_idx, seq_loop_fn, init_carry
    )


def rpa_metadata_schedule_kernel(
    ## Scalar prefetch.
    cu_q_lens_ref: jax.Ref,
    q_positions_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    kv_new_starts_ref: jax.Ref,
    update_kv_cache_ref: jax.Ref,
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
    q_positions_ref: [max_num_seqs]. Position of each sequence's first query
      token in the sequence's KV index space (cache tokens followed by new
      tokens); the causal mask is relative to it.
    kv_cache_lens_ref: [max_num_seqs]. Existing kv cache length of each
      sequence.
    kv_new_lens_ref: [max_num_seqs]. New kv length of each sequence.
    kv_new_starts_ref: [max_num_seqs]. Offset of each sequence's new KV in
      `new_kv_hbm`.
    update_kv_cache_ref: [max_num_seqs]. 1 if this sequence writes its new KV
      back to the paged cache.
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
      q_positions_ref=q_positions_ref,
      kv_cache_lens_ref=kv_cache_lens_ref,
      kv_new_lens_ref=kv_new_lens_ref,
      kv_new_starts_ref=kv_new_starts_ref,
      update_kv_cache_ref=update_kv_cache_ref,
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
    q_positions: jax.Array,
    kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    kv_new_starts: jax.Array,
    distribution: jax.Array,
    cfgs: configs.RpaConfigs,
    *,
    update_kv_cache: jax.Array | None = None,
    computer_cls: type[BaseMetadataComputer] = BaseMetadataComputer,
    extra_scalars: Sequence[jax.Array] = (),
    interpret: bool = False,
) -> RpaSchedule:
  """Generates RPA metadata schedule using the specified computer_cls."""
  if update_kv_cache is None:
    update_kv_cache = jnp.ones_like(kv_cache_lens, dtype=jnp.int32)
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
          num_scalar_prefetch=8,
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
      q_positions,
      kv_cache_lens,
      kv_new_lens,
      kv_new_starts,
      update_kv_cache,
      distribution,
      extra_scalars,
  )


class BaseScheduler(ABC):
  """Base control-plane scheduler specifying KV attention scope and lengths."""

  computer_cls: ClassVar[type[BaseMetadataComputer]] = BaseMetadataComputer
  has_cache_kv: ClassVar[bool] = True
  has_new_kv: ClassVar[bool] = True

  if TYPE_CHECKING:

    @property
    def cp_rank(self) -> jax.Array | None:
      ...

    @property
    def cp_group_size(self) -> int | None:
      ...

    @property
    def cp_kv_cache_interleave_size(self) -> int | None:
      ...

    @property
    def ring_axis_name(self) -> str | None:
      ...

    @property
    def ring_mesh_axis_names(self) -> tuple[str, ...] | None:
      ...

    @property
    def is_packed_lse(self) -> bool:
      ...

    @property
    def q_positions(self) -> jax.Array | None:
      ...

    @property
    def update_kv_cache(self) -> jax.Array | bool | None:
      ...

  else:
    cp_group_size = None
    cp_kv_cache_interleave_size = None
    ring_axis_name = None
    ring_mesh_axis_names = None
    is_packed_lse = False
    q_positions = None
    update_kv_cache = None

  @property
  def paged_new_kv(self) -> bool:
    return False

  @property
  def extra_scalars(self) -> tuple[jax.Array, ...]:
    return ()

  @property
  @abstractmethod
  def seq_lens(self) -> jax.Array:
    """Primary per-sequence length array for static shape/dtype validation."""

  def validate(self, model_cfgs: configs.ModelConfigs) -> None:
    """Validates scheduler configuration against model configs."""
    del model_cfgs

  @abstractmethod
  def get_kv_lens(
      self, q_lens: jax.Array, page_size: int = 1
  ) -> tuple[jax.Array, jax.Array]:
    """Returns `(kv_cache_lens, kv_new_lens)` for schedule and main kernels."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FullScheduler(BaseScheduler):
  """Standard full-attention scheduler attending both cached and new KV tokens.

  Attributes:
    kv_lens: [max_num_seqs] total KV length (cached + new) per sequence.
    q_positions: Optional [max_num_seqs] starting query position per sequence.
    update_kv_cache: Optional bool or [max_num_seqs] flag for KV cache writeback.
    is_packed_lse: Whether to store LSE in packed 1-row-per-token layout.
  """

  computer_cls: ClassVar[type[BaseMetadataComputer]] = BaseMetadataComputer
  has_new_kv: ClassVar[bool] = True
  if not TYPE_CHECKING:
    cp_rank = None

  kv_lens: jax.Array
  q_positions: jax.Array | None = None
  update_kv_cache: jax.Array | bool | None = None
  is_packed_lse: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )

  @property
  def seq_lens(self) -> jax.Array:
    return self.kv_lens

  def get_kv_lens(
      self, q_lens: jax.Array, page_size: int = 1
  ) -> tuple[jax.Array, jax.Array]:
    del page_size
    return self.kv_lens - q_lens, q_lens