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
# ==============================================================================

"""Module for deciding how memory should be allocated and DMAed."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from vllm_torchtpu.kernels.routing_moe.common import layout_util
from vllm_torchtpu.kernels.routing_moe.gmm import config


def map_memory_constraint(obj, memory_space):
  return jax.tree.map(
      lambda x: pltpu.with_memory_space_constraint(x, memory_space), obj
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
  """Stores scalar data used for routing and pipeline scheudling."""

  # NOTE: because gm_id is offsetted by num_buffers, it is not recommended
  # to directly access these values as it is error prone. Instead, it is
  # recommended to access them via get_* class methods.
  _gm_id_to_group_id: jax.Array | hijax.TransformedRef
  _gm_id_to_m_offset: jax.Array | hijax.TransformedRef
  _gm_id_to_num_rows: jax.Array | hijax.TransformedRef
  gather_idx: jax.Array | hijax.TransformedRef | None
  scatter_idx: jax.Array | hijax.TransformedRef | None
  group_cum_sum: jax.Array | hijax.TransformedRef
  group_offset: jax.Array | hijax.TransformedRef
  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  @classmethod
  def create(
      cls,
      gather_idx: jax.Array | None,
      scatter_idx: jax.Array | None,
      group_cum_sum: jax.Array,
      group_offset: jax.Array,
      cfg: config.RoutingGmmConfig,
  ) -> MetadataRef:

    # Padd indices by tile_m to avoid any out of bound access at last steps.
    if gather_idx is not None:
      gather_idx = jnp.pad(gather_idx, (0, cfg.tile.tile_m))
    if scatter_idx is not None:
      scatter_idx = jnp.pad(scatter_idx, (0, cfg.tile.tile_m))

    return cls(
        _gm_id_to_group_id=jnp.empty((cfg.max_num_gm,), jnp.int32),
        _gm_id_to_m_offset=jnp.empty((cfg.max_num_gm,), jnp.int32),
        _gm_id_to_num_rows=jnp.empty((cfg.max_num_gm,), jnp.int32),
        gather_idx=gather_idx,
        scatter_idx=scatter_idx,
        group_cum_sum=group_cum_sum,
        group_offset=group_offset,
        cfg=cfg,
    )

  def _real_gm_id(self, gm_id: jax.Array | int) -> jax.Array | int:
    return gm_id + self.cfg.tile.num_max_buffers

  def get_num_rows(self, gm_id: jax.Array | int) -> jax.Array:
    return self._gm_id_to_num_rows[self._real_gm_id(gm_id)]

  def get_m_offset(self, gm_id: jax.Array | int) -> jax.Array:
    return self._gm_id_to_m_offset[self._real_gm_id(gm_id)]

  def get_group_id(self, gm_id: jax.Array | int) -> jax.Array:
    return self._gm_id_to_group_id[self._real_gm_id(gm_id)]

  def fill_metadata_and_return_num_gm(self) -> jax.Array:
    """Iterate group sizes to metadata about DMA and pipeline schedule logic."""

    # NOTE: Unlike GMM v2, there is no need for complex sublane alignment logic
    # as issuing one dma per row avoids any shared chunk or non-zero starting
    # offset issue that was found in GMM v2.

    group_start = self.group_offset[0]
    group_end = group_start + self.cfg.dim.size_local_groups
    num_buffers = self.cfg.tile.num_max_buffers

    # To avoid any runtime out-of-the-bounds check overhead, we intentially
    # start gm_id from num_buffers so that trying to DMA wait gm_id -
    # num_buffers does not trigger stall.
    for b in range(num_buffers):
      self._gm_id_to_group_id[b] = 0
      self._gm_id_to_m_offset[b] = 0
      self._gm_id_to_num_rows[b] = 0

    @jax.named_scope("write_tile")
    def write_tile(gm_id, offset, *, local_group_id, end):
      num_rows = jnp.minimum(end - offset, self.cfg.tile.tile_m)
      self._gm_id_to_group_id[gm_id] = local_group_id
      self._gm_id_to_m_offset[gm_id] = offset
      self._gm_id_to_num_rows[gm_id] = num_rows
      return offset + num_rows

    @jax.named_scope("write_group")
    def write_group(group_id, num_gm):
      start = self.group_cum_sum[group_id]
      end = self.group_cum_sum[group_id + 1]
      next_num_gm = num_gm + pl.cdiv(end - start, self.cfg.tile.tile_m)
      local_group_id = group_id - group_start

      jax.lax.fori_loop(
          num_gm,
          next_num_gm,
          functools.partial(write_tile, local_group_id=local_group_id, end=end),
          start,
      )
      return next_num_gm

    num_gm = jax.lax.fori_loop(group_start, group_end, write_group, num_buffers)

    # Similar to above, we zero out up-to last gm + num_buffers to ensure DMA
    # copy for gm_id + num_buffers does not tirgger out-of-the-bound errors.
    for b in range(num_buffers):
      self._gm_id_to_group_id[num_gm + b] = 0
      self._gm_id_to_m_offset[num_gm + b] = 0
      self._gm_id_to_num_rows[num_gm + b] = 0

    # Return real number of buffers minus left and right padding.
    return num_gm - num_buffers


class AbstractWeightRef(abc.ABC):
  """Abstract class that defines interfaces for rhs values."""

  @abc.abstractmethod
  def get_weight(self) -> jax.Array:
    ...

  @abc.abstractmethod
  def get_scale(self, replicate_size: int | None = None) -> jax.Array | None:
    """Returns scale array, optionally replicated across sublanes."""
    ...

  @abc.abstractmethod
  def get_bias(self) -> jax.Array:
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SingleWeightRef(AbstractWeightRef):
  """Dataclass for a single weights."""

  weight: Any
  scale: Any | None
  bias: Any | None

  def get_weight(self) -> jax.Array:
    return self.weight[...]

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    assert self.scale is not None
    if replicate_size is not None:
      # Perform zero-stride load for efficient broadcasting across sublanes.
      return self.scale[:, pl.ds(0, replicate_size, 0), :]
    return self.scale[...]

  def get_bias(self) -> jax.Array:
    assert self.bias is not None
    return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(AbstractWeightRef):
  """Dataclass for gate and up weights used in fused activation."""

  gate: SingleWeightRef
  up: SingleWeightRef

  def get_weight(self) -> jax.Array:
    w_gate = self.gate.get_weight()
    w_up = self.up.get_weight()
    return layout_util.large_interleave_lane(w_gate, w_up)

  def get_scale(self, replicate_size: int | None = None) -> jax.Array:
    s_gate = self.gate.get_scale(replicate_size)
    s_up = self.up.get_scale(replicate_size)

    return layout_util.large_interleave_lane(s_gate, s_up)

  def get_bias(self) -> jax.Array:
    b_gate = self.gate.get_bias()
    b_up = self.up.get_bias()
    return layout_util.large_interleave_lane(b_gate, b_up)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CustomBufferedRef(pltpu.BufferedRef):
  """Shared interface for both fused and non-fused BufferedRef."""

  metadata_ref: MetadataRef = dataclasses.field(metadata=dict(static=True))
  src_ref: Any = dataclasses.field(metadata=dict(static=True))
  dst_ref: Any = dataclasses.field(metadata=dict(static=True))

  cfg: config.RoutingGmmConfig = dataclasses.field(metadata=dict(static=True))

  def __post_init__(self):
    # pallas doesn't allow you to set buffer_count > 2 for output refs, so
    # we override to bypass this check.
    pass

  @classmethod
  def create(
      cls,
      *args: Any,
      cfg: config.RoutingGmmConfig,
      metadata: MetadataRef,
      src_ref=None,
      dst_ref=None,
      **kwargs,
  ):
    standard_ref = pltpu.BufferedRef.create(*args, **kwargs)
    return cls(
        metadata_ref=metadata,
        src_ref=src_ref,
        dst_ref=dst_ref,
        cfg=cfg,
        **{
            f.name: getattr(standard_ref, f.name)
            for f in dataclasses.fields(pltpu.BufferedRef)
        },
    )

  @abc.abstractmethod
  def branchless_copy_in(self, gm_id):
    ...

  @abc.abstractmethod
  def branchless_wait_in(self, gm_id):
    ...

  @abc.abstractmethod
  def branchless_copy_out(self, gm_id):
    ...

  @abc.abstractmethod
  def branchless_wait_out(self, gm_id):
    ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchedBufferedRef(CustomBufferedRef):
  """BufferedRef managed by emit_pipeline. Gated by dynamic branching."""

  def _no_op(self, gm_id: jax.Array | int):
    del gm_id

  branchless_copy_in = branchless_wait_in = branchless_copy_out = (
      branchless_wait_out
  ) = _no_op


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BranchlessBufferedRef(CustomBufferedRef):
  """BufferedRef that performs branchless DMA to allow matmul overlap."""

  def _slot(self, gm_id: jax.Array | int) -> jax.Array:
    return jnp.mod(gm_id, self.cfg.tile.num_xo_buffers)

  def branchless_copy_in(self, gm_id: jax.Array | int):
    assert self.sem_recvs is not None
    assert self.window_ref is not None
    assert self.src_ref is not None
    assert self.metadata_ref.gather_idx is not None

    slot = self._slot(gm_id)
    sem = self.sem_recvs.at[slot]
    vmem_dst = self.window_ref.at[slot]
    vmem_dst = pltpu.annotate(vmem_dst, no_hazard_no_deps=True)
    hbm_src = self.src_ref
    blocks = hbm_src.shape[1]
    row_idx = self.metadata_ref.gather_idx

    offset = self.metadata_ref.get_m_offset(gm_id)
    rows = self.metadata_ref.get_num_rows(gm_id)

    for r in range(self.cfg.tile.tile_m):
      size = jnp.where(r < rows, 1, 0)
      row = row_idx[offset + r]
      pltpu.make_async_copy(
          src_ref=hbm_src.at[pl.ds(row, size)],
          dst_ref=vmem_dst.at[pl.ds(r, size), :blocks],
          sem=sem,
      ).start()

  def branchless_copy_out(self, gm_id: jax.Array):
    assert self.sem_sends is not None
    assert self.window_ref is not None
    assert self.dst_ref is not None
    assert self.metadata_ref.scatter_idx is not None

    slot = self._slot(gm_id)
    sem = self.sem_sends.at[slot]
    vmem_src = self.window_ref.at[slot]
    vmem_src = pltpu.annotate(vmem_src, no_hazard_no_deps=True)
    hbm_dst = self.dst_ref
    blocks = hbm_dst.shape[1]
    row_idx = self.metadata_ref.scatter_idx

    offset = self.metadata_ref.get_m_offset(gm_id)
    rows = self.metadata_ref.get_num_rows(gm_id)

    for r in range(self.cfg.tile.tile_m):
      size = jnp.where(r < rows, 1, 0)
      row = row_idx[offset + r]
      pltpu.make_async_copy(
          src_ref=vmem_src.at[pl.ds(r, size), :blocks],
          dst_ref=hbm_dst.at[pl.ds(row, size)],
          sem=sem,
      ).start(priority=1)

  def branchless_wait_in(self, gm_id):
    assert self.sem_recvs is not None
    assert self.window_ref is not None

    slot = self._slot(gm_id)
    sem = self.sem_recvs.at[slot]
    hbm_src = self.src_ref

    rows = self.metadata_ref.get_num_rows(gm_id)
    copied = hbm_src.at[pl.ds(0, rows)]
    pltpu.make_async_copy(copied, copied, sem).wait()

  def branchless_wait_out(self, gm_id) -> None:
    assert self.sem_sends is not None
    assert self.window_ref is not None

    slot = self._slot(gm_id)
    sem = self.sem_sends.at[slot]
    hbm_dst = self.dst_ref
    rows = self.metadata_ref.get_num_rows(gm_id)
    copied = hbm_dst.at[pl.ds(0, rows)]
    pltpu.make_async_copy(copied, copied, sem).wait()

  def _no_op(self, ref, grid_indices):
    del ref, grid_indices

  copy_in = copy_out = wait_in = wait_out = _no_op


@dataclasses.dataclass(frozen=True)
class IndexMapProvider:
  """Return function of index map used by emit pipeline."""

  metadata_ref: MetadataRef
  cfg: config.RoutingGmmConfig

  def act_index(self, n_id, gm_id, k_id, *, hbm_shape):
    del n_id, k_id

    num_chunks = hbm_shape[1]
    m_offset = self.metadata_ref.get_m_offset(gm_id)
    m_size = self.metadata_ref.get_num_rows(gm_id)
    return (pl.ds(m_offset, m_size), pl.ds(0, num_chunks), 0, 0)

  def weight_index(self, n_id, gm_id, k_id):
    del n_id, k_id
    return (self.metadata_ref.get_group_id(gm_id), 0, 0)

  def scale_index(self, n_id, gm_id, k_id):
    del n_id, k_id
    return (self.metadata_ref.get_group_id(gm_id), 0, 0, 0)


@dataclasses.dataclass(frozen=True)
class BlockSpecFactory:
  """Create block specs. Index map can be ignored in some custom casese."""

  metadata_ref: MetadataRef
  cfg: config.RoutingGmmConfig
  index_map_provider: IndexMapProvider

  def _act_block_shape(self, shape):
    num_chunks = layout_util.to_bank_conflict_aware_stride(shape[1])
    return (
        pl.BoundedSlice(self.cfg.tile.tile_m),
        pl.BoundedSlice(num_chunks),
        *shape[2:],
    )

  def _act_spec(self, hbm_shape):
    index_map = functools.partial(
        self.index_map_provider.act_index, hbm_shape=hbm_shape
    )
    return pl.BlockSpec(
        block_shape=self._act_block_shape(hbm_shape),
        index_map=index_map,
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.tile.num_xo_buffers, use_lookahead=True
        ),
        memory_space=pltpu.VMEM,
    )

  def get_x_spec(self):
    return self._act_spec(self.cfg.x.compact_shape)

  def get_out_spec(self):
    return self._act_spec(self.cfg.out.compact_shape)

  def get_weight_spec(self):
    block_shape = (None, self.cfg.tile.tile_k, self.cfg.tile.tile_n)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.weight_index,
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.tile.num_w_buffers, use_lookahead=True
        ),
        memory_space=pltpu.VMEM,
    )

  def get_scale_spec(self):
    block_shape = (None, 1, 1, self.cfg.tile.tile_n)
    return pl.BlockSpec(
        block_shape=block_shape,
        index_map=self.index_map_provider.scale_index,
        pipeline_mode=pl.Buffered(
            buffer_count=self.cfg.tile.num_w_buffers, use_lookahead=True
        ),
        memory_space=pltpu.VMEM,
    )


def create_allocs(
    x_hbm_ref: hijax.TransformedRef,
    w_hbm_ref: SingleWeightRef,
    out_hbm_ref: hijax.TransformedRef,
    metadata_ref: MetadataRef,
    cfg: config.RoutingGmmConfig,
) -> tuple[CustomBufferedRef, AbstractWeightRef, CustomBufferedRef]:
  """Create BufferedRef allocation object for arguments of the kernel."""

  index_map_provider = IndexMapProvider(metadata_ref, cfg)
  spec_creater = BlockSpecFactory(metadata_ref, cfg, index_map_provider)

  scale_bref = None
  if cfg.w.has_scale:
    assert w_hbm_ref.scale is not None
    scale_bref = pltpu.BufferedRef.input(
        spec=spec_creater.get_scale_spec(),
        dtype_or_type=w_hbm_ref.scale.dtype,
        buffer_count=cfg.tile.num_w_buffers,
        use_lookahead=True,
        grid_rank=3,
    )

  weight_bref = pltpu.BufferedRef.input(
      spec=spec_creater.get_weight_spec(),
      dtype_or_type=w_hbm_ref.weight.dtype,
      buffer_count=cfg.tile.num_w_buffers,
      use_lookahead=True,
      grid_rank=3,
  )
  w_bref = SingleWeightRef(weight=weight_bref, scale=scale_bref, bias=None)

  if cfg.fuse_act is not None:
    w_bref = FusedWeightsRef(gate=w_bref, up=w_bref)

  def _get_buffered_ref(mode: config.RoutingMode) -> type[CustomBufferedRef]:
    match mode:
      case config.RoutingMode.LOCAL:
        return BranchlessBufferedRef
      case config.RoutingMode.DIRECT:
        return BranchedBufferedRef
      case _:
        raise NotImplementedError("Only direct or local routing is supported.")

  x_bref = _get_buffered_ref(cfg.x.mode).input(
      spec=spec_creater.get_x_spec(),
      dtype_or_type=x_hbm_ref.dtype,
      buffer_count=cfg.tile.num_xo_buffers,
      use_lookahead=True,
      grid_rank=3,
      metadata=metadata_ref,
      src_ref=x_hbm_ref,
      cfg=cfg,
  )
  out_bref = _get_buffered_ref(cfg.out.mode).output(
      spec=spec_creater.get_out_spec(),
      dtype_or_type=out_hbm_ref.dtype,
      buffer_count=cfg.tile.num_xo_buffers,
      use_lookahead=True,
      grid_rank=3,
      cfg=cfg,
      metadata=metadata_ref,
      dst_ref=out_hbm_ref,
  )

  return x_bref, w_bref, out_bref


def get_specs(
    x_buf: CustomBufferedRef,
    w_buf: AbstractWeightRef,
    out_buf: CustomBufferedRef,
) -> tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec]:
  """Fetch block specs of a given buffered refs."""

  x_spec = x_buf.spec
  w_spec = jax.tree.map(
      lambda bref: bref.spec,
      w_buf,
      is_leaf=lambda x: isinstance(x, pltpu.BufferedRefBase),
  )
  out_spec = out_buf.spec

  return (x_spec, w_spec, out_spec)


def transform_weight_ref(
    w_hbm_ref: SingleWeightRef,
    cfg: config.RoutingGmmConfig,
) -> AbstractWeightRef:

  if cfg.fuse_act is not None:
    return FusedWeightsRef(
        gate=w_hbm_ref,
        up=jax.tree.map(lambda x: x.at[..., cfg.dim.size_out_n :], w_hbm_ref),
    )
  else:
    return w_hbm_ref
