"""Context Parallel (CP) DMA schedule computation and schedulers for RPA."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, ClassVar

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.batched_rpa_new import (
    configs,
    schedule,
    utils,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HeadAlongSublaneDmaNewCP(schedule.DmaNew):
  """Like HeadAlongSublaneDmaNew but with separate fetch/wb flags for CP.

  wb_val encodes per-page ownership (dma_sz if this rank owns the page,
  else 0), so bref_override copy_out needs no CP-specific logic.
  """

  wb_hbm = schedule.FieldOffset(0)
  fetch_hbm = schedule.FieldOffset(1)
  fetch_vmem = schedule.FieldOffset(2)
  wb_vmem = schedule.FieldOffset(2)
  _fetch_flags = schedule.FieldOffset(3)
  _wb_flags = schedule.FieldOffset(4)

  @staticmethod
  def num_fields() -> int:
    return 5

  @property
  def fetch_val(self):
    return self._fetch_flags[...]

  @property
  def wb_val(self):
    return self._wb_flags[...]

  def set_flags(self, fetch_val, wb_val):
    self._fetch_flags[...] = fetch_val
    self._wb_flags[...] = wb_val


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RingSems:
  """The ring's semaphores."""

  dma_sems: Any  # [NUM_INPUT_SLOTS, 2] send/recv per buffer slot.
  sync_sem: Any  # credit-based flow control semaphore.


class RingAttention:
  """Rotates the KV window around the PCP ring during the cache phase.

  Each rank owns one page-stripe of the KV cache. Step `t` is hop `t % P` of
  one KV block: at hop 0 the rank fetches its own block from HBM, and at each
  later hop the block it holds arrived from the previous rank, so after P
  hops its Q has attended the whole cache under a single online softmax.

  Flow control is a single counting semaphore:
    * each rank starts by granting its predecessor `n_buffer` credits, since
      every slot is free;
    * the sender spends one before each rotation;
    * the receiver grants one back after consuming a rotated block.

  A send at hop `r < P - 1` is always matched by a receive at hop `r + 1`, so
  grants and spends balance over the run and the kernel drains the initial
  `n_buffer` on the way out.
  """

  n_buffer: int = 3

  def __init__(
      self, cfgs: configs.RpaConfigs, sems: RingSems, *, kv_window_ref: jax.Ref
  ):
    assert configs.NUM_INPUT_SLOTS == self.n_buffer, (
        f"RingAttention requires NUM_INPUT_SLOTS == {self.n_buffer} for triple"
        f" buffering, got {configs.NUM_INPUT_SLOTS}"
    )
    assert cfgs.serve.cp_group_size is not None
    self.cfgs = cfgs
    self.size: int = cfgs.serve.cp_group_size
    self.kv_window = kv_window_ref
    self.sems = sems
    my_id = lax.axis_index(cfgs.serve.ring_axis_name)
    self.next_id = self._device_id(lax.rem(my_id + 1, self.size))
    self.prev_id = self._device_id(lax.rem(my_id + self.size - 1, self.size))
    self._pending_send = None

  def _device_id(self, rank):
    names = self.cfgs.serve.ring_mesh_axis_names
    if names is None:
      return (rank,)
    return tuple(
        rank
        if name == self.cfgs.serve.ring_axis_name
        else lax.axis_index(name)
        for name in names
    )

  def initial_handshake(self):
    """Tell the predecessor every slot is free. Call before the pipeline."""
    pl.semaphore_signal(
        self.sems.sync_sem,
        self.n_buffer,
        device_id=self.prev_id,
        device_id_type=pl.DeviceIdType.MESH,
    )

  def drain_credits(self):
    """Consume the successor's initial grants. Call after the pipeline."""
    pl.semaphore_wait(self.sems.sync_sem, self.n_buffer)

  def _remote_copy(self, src_slot, dst_slot):
    return pltpu.make_async_remote_copy(
        src_ref=self.kv_window.at[src_slot],
        dst_ref=self.kv_window.at[dst_slot],
        send_sem=self.sems.dma_sems.at[src_slot, 0],
        recv_sem=self.sems.dma_sems.at[dst_slot, 1],
        device_id=self.next_id,
        device_id_type=pl.DeviceIdType.MESH,
    )

  def receive_and_forward(self, step, schedule_ref, total_steps):
    """Take this step's rotated block, then pass it on."""
    del total_steps
    slot = lax.rem(step, self.n_buffer)
    next_slot = lax.rem(step + 1, self.n_buffer)

    is_valid = schedule_ref.s_idx[step, 0] != -1
    hop = lax.rem(schedule_ref.k_idx[step, 0], self.size)
    receives = jnp.logical_and(is_valid, hop > 0)
    sends = jnp.logical_and(is_valid, hop < self.size - 1)

    @pl.when(receives)
    def _wait_recv():
      self._remote_copy(slot, slot).wait_recv()

    send_op = self._remote_copy(slot, next_slot)

    @pl.when(sends)
    def _start_send():
      # Spend a credit before writing into the receiver's buffer.
      pl.semaphore_wait(self.sems.sync_sem, 1)
      send_op.start()

    self._pending_send = (sends, receives, send_op, slot)

  def finish(self, step: int | jax.Array, schedule_ref, total_steps):
    """Close out the step's hop: drain pipelined sends, release credit."""
    assert self._pending_send is not None
    sends, receives, send_op, _ = self._pending_send
    self._pending_send = None

    # Overlapped send draining: wait for the send from (step - 1),
    # which has already had a full step of compute to complete in the
    # background.
    prev_step = step - 1
    prev_sends = jnp.logical_and(
        prev_step >= 0,
        lax.rem(schedule_ref.k_idx[jnp.maximum(prev_step, 0), 0], self.size)
        < self.size - 1,
    )
    prev_slot = lax.rem(jnp.maximum(prev_step, 0), self.n_buffer)

    @pl.when(prev_sends)
    def _wait_prev_send():
      self._remote_copy(prev_slot, prev_slot).wait_send()

    # If this is the last step in the pipeline, also drain the current step's
    # send.
    is_last_step = step == total_steps - 1
    drain_last = jnp.logical_and(is_last_step, sends)

    @pl.when(drain_last)
    def _wait_last_send():
      send_op.wait_send()

    @pl.when(receives)
    def _release_slot():
      pl.semaphore_signal(
          self.sems.sync_sem,
          1,
          device_id=self.prev_id,
          device_id_type=pl.DeviceIdType.MESH,
      )


class CPMetadataComputer(schedule.BaseMetadataComputer):
  """Context Parallel (CP) metadata scheduler: overrides only k_loop and schema."""

  @classmethod
  def get_rpa_schedule(
      cls, cfgs: configs.RpaConfigs, multiplier: int = 1
  ) -> schedule.RpaSchedule:
    """Returns the RpaSchedule shape/dtype struct with CP-specific DMA struct."""
    dma_kv_new_struct_cls = (
        HeadAlongSublaneDmaNewCP
        if cfgs.serve.kv_layout == configs.KVLayout.HEAD_ALONG_SUBLANE
        else schedule.SeqAlongLaneDmaNew
    )
    return schedule.RpaSchedule.create_shape_dtype(
        cfgs,
        dma_kv_new_struct_cls=dma_kv_new_struct_cls,
        multiplier=multiplier,
    )

  @property
  def rank(self):
    return self.extra_refs[0][0] if self.extra_refs else 0

  @property
  def new_kv_page_indices_ref(self):
    return self.extra_refs[1] if len(self.extra_refs) > 1 else None

  @jax.named_scope("k_loop_cp")
  def k_loop(
      self,
      k_idx,
      carry: schedule.LoopCarry,
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
  ) -> schedule.LoopCarry:
    del q_end, num_k, kwargs
    cfgs = self.cfgs
    sched = self.schedule
    count = carry.count
    step, target_lane = divmod(count, cfgs.batch_size)

    sched.s_idx[step, target_lane] = s_idx
    sched.q_idx[step, target_lane] = q_idx
    sched.k_idx[step, target_lane] = k_idx

    is_last_k = jnp.where(k_idx == end_k_idx - 1, 1, 0)
    sched.is_last_k[step, target_lane] = is_last_k

    sched.dma_q[step, target_lane, 0] = q_src
    sched.dma_q[step, target_lane, 1] = q_sz_task

    kv_len_start = k_idx * cfgs.bkv_sz
    kv_p_start = k_idx * cfgs.bkv_p
    k_len = kv_cache_len + kv_new_len
    kv_left = k_len - kv_len_start
    kv_left_frm_cache = jnp.maximum(kv_cache_len - kv_len_start, 0)
    p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

    cp_group_size = cfgs.serve.cp_group_size
    assert cp_group_size is not None

    for i in range(cfgs.bkv_p_cache):
      dst_vmem = i << cfgs.serve.page_size_log2
      dma_sz = kv_left_frm_cache - dst_vmem
      dma_sz = jnp.clip(dma_sz, 0, cfgs.serve.page_size)

      if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        p_idx = kv_p_start + i
        local_page_i = jnp.minimum(
            p_idx // cp_group_size,
            cfgs.serve.pages_per_seq - 1,
        )
        src_hbm_cp = s_idx * cfgs.serve.pages_per_seq + local_page_i
        dma_sz = jnp.where(p_idx % cp_group_size == self.rank, dma_sz, 0)
        sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm_cp
        sched.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
        sched.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
            dma_sz, cfgs.kv_token_align
        )
      else:
        src_hbm = jnp.minimum(p_offset + i, cfgs.serve.num_page_indices - 1)
        sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm
        sched.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
        sched.dma_kv_cache[step, target_lane, i, 2] = 0

    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
    new_sz = jnp.minimum(cfgs.bkv_sz - bkv_sz_cache, kv_left_frm_new)

    q_wb = num_q - 1
    do_writeback = jnp.where(
        (new_sz > 0) & (q_idx == q_wb) & update_kv_cache, 1, 0
    )
    sched.do_writeback[step, target_lane] = do_writeback
    new_kv_pos = kv_new_len - kv_left_frm_new
    src_hbm = kv_new_start + new_kv_pos

    def fill_dma_kv_new(i, dst_vmem, dma_sz):
      dma_entry = sched.dma_kv_new[step, target_lane, i]
      if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        align = cfgs.kv_token_align
        if cfgs.serve.paged_new_kv:
          new_kv_page_indices_ref = self.new_kv_page_indices_ref
          assert new_kv_page_indices_ref is not None
          new_pages_per_seq = (
              new_kv_page_indices_ref.shape[0] // cfgs.serve.num_seqs
          )
          row = s_idx * new_pages_per_seq
          cache_pages = pl.cdiv(bkv_sz_cache, cfgs.serve.page_size)
          new_tok_offset = new_kv_pos % cfgs.serve.page_size
          num_pages_to_fetch = jnp.where(
              new_sz > 0,
              (new_tok_offset + new_sz - 1) // cfgs.serve.page_size + 1,
              0,
          )
          fetch_val = jnp.where(i < num_pages_to_fetch, cfgs.serve.page_size, 0)
          new_page_start = (
              new_kv_pos - new_tok_offset
          ) + i * cfgs.serve.page_size
          page = jnp.minimum(
              new_page_start >> cfgs.serve.page_size_log2,
              new_pages_per_seq - 1,
          )
          dma_entry.fetch_hbm[...] = (
              new_kv_page_indices_ref[row + page] << cfgs.serve.page_size_log2
          )
          dma_entry.fetch_vmem[...] = (cache_pages + i) * cfgs.serve.page_size
        elif i == 0:
          src, _, fetch_val = utils.align_span(src_hbm, new_sz, align)
          dma_entry.fetch_hbm[...] = src
          dma_entry.fetch_vmem[...] = utils.align_to(bkv_sz_cache, align)
          fetch_val = jnp.where(new_sz > 0, fetch_val, 0)
        else:
          fetch_val = 0

        # Writeback: one DMA per page, since page_indices scatters them, and
        # only for the pages this rank owns.
        tok_idx = kv_len_start + dst_vmem
        p_idx = jnp.minimum(
            tok_idx >> cfgs.serve.page_size_log2,
            cfgs.serve.pages_per_seq * cp_group_size - 1,
        )
        p_off = tok_idx & cfgs.serve.page_size_mask
        dst, lead, wb_val = utils.align_span(p_off, dma_sz, align)
        dma_entry.wb_hbm[...] = (
            (s_idx * cfgs.serve.pages_per_seq + p_idx // cp_group_size)
            << cfgs.serve.page_size_log2
        ) | dst
        dma_entry.wb_vmem[...] = dst_vmem - lead
        dma_entry.set_flags(
            fetch_val,
            jnp.where(
                (dma_sz > 0) & (p_idx % cp_group_size == self.rank), wb_val, 0
            ),
        )
      else:
        tok_idx = kv_len_start + dst_vmem
        p_idx = jnp.minimum(
            tok_idx >> cfgs.serve.page_size_log2,
            cfgs.serve.pages_per_seq * cp_group_size - 1,
        )
        p_off = tok_idx & cfgs.serve.page_size_mask
        local_slot = p_idx // cp_group_size
        dst_hbm = (
            (s_idx * cfgs.serve.pages_per_seq + local_slot)
            << cfgs.serve.page_size_log2
        ) | p_off
        wb_val = jnp.where(
            p_idx % cp_group_size == self.rank, dma_sz, jnp.int32(0)
        )

        fetch_hbm = src_hbm
        if cfgs.serve.paged_new_kv:
          new_kv_page_indices_ref = self.new_kv_page_indices_ref
          assert new_kv_page_indices_ref is not None
          new_pages_per_seq = (
              new_kv_page_indices_ref.shape[0] // cfgs.serve.num_seqs
          )
          row = s_idx * new_pages_per_seq
          rel_off = new_kv_pos + (dst_vmem - bkv_sz_cache)
          page = jnp.minimum(
              rel_off >> cfgs.serve.page_size_log2, new_pages_per_seq - 1
          )
          fetch_hbm = (
              new_kv_page_indices_ref[row + page] << cfgs.serve.page_size_log2
          ) | (rel_off & cfgs.serve.page_size_mask)
        dma_entry.fetch_hbm[...] = fetch_hbm
        dma_entry.fetch_vmem[...] = dst_vmem
        dma_entry.wb_hbm[...] = dst_hbm
        dma_entry.wb_vmem[...] = dst_vmem
        dma_entry.set_flags(dma_sz, wb_val)

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


class RingCacheMetadataComputer(schedule.BaseMetadataComputer):
  """PCP Ring metadata computer rotating page-striped KV cache across ranks."""

  rpa_cases: ClassVar[tuple[configs.RpaCase, ...]] = (configs.RpaCase.MIXED,)
  ring_enabled: ClassVar[bool] = True

  @classmethod
  def get_ring_sems_spec(cls) -> RingSems:
    return RingSems(
        dma_sems=pltpu.SemaphoreType.DMA((configs.NUM_INPUT_SLOTS, 2)),
        sync_sem=pltpu.SemaphoreType.REGULAR,
    )

  @classmethod
  def create_ring(
      cls,
      cfgs: configs.RpaConfigs,
      ring_sems: RingSems,
      *,
      kv_window_ref: jax.Ref,
  ) -> RingAttention:
    return RingAttention(cfgs, ring_sems, kv_window_ref=kv_window_ref)

  @jax.named_scope("q_loop_cp_ring")
  def q_loop(
      self,
      q_idx,
      carry: schedule.LoopCarry,
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
  ) -> schedule.LoopCarry:
    del q_position, kv_new_len, num_k, num_q, update_kv_cache
    cfgs = self.cfgs
    cp_size = cfgs.serve.cp_group_size
    assert cp_size is not None

    # Every rank walks rank 0's stripe length so all ranks execute identical
    # step counts and rotate in lockstep.
    local_kv_cache_len = utils.cp_local_cache_len(
        kv_cache_len, cp_size, 0, cfgs.serve.page_size
    )
    num_k = pl.cdiv(local_kv_cache_len, cfgs.bkv_sz)
    num_k_groups = pl.cdiv(num_k, cfgs.batch_size)
    end_k_idx = num_k_groups * cp_size * cfgs.batch_size

    q_src = q_start + q_idx * cfgs.bq_sz
    q_sz_task = jnp.clip(q_end - q_src, 0, cfgs.bq_sz)

    k_loop_fn = functools.partial(
        self.k_loop,
        s_idx=s_idx,
        q_idx=q_idx,
        q_src=q_src,
        q_sz_task=q_sz_task,
        kv_cache_len=local_kv_cache_len,
        num_k=num_k,
    )
    return jax.lax.fori_loop(0, end_k_idx, k_loop_fn, carry)

  @jax.named_scope("k_loop_cp_ring")
  def k_loop(  # pyrefly: ignore[bad-override]
      self,
      k_idx,
      carry: schedule.LoopCarry,
      *,
      s_idx,
      q_idx,
      q_src,
      q_sz_task,
      kv_cache_len,
      num_k,
      **kwargs,
  ) -> schedule.LoopCarry:
    del kwargs
    cfgs = self.cfgs
    sched = self.schedule
    count = carry.count
    step, target_lane = divmod(count, cfgs.batch_size)

    cp_group_size = cfgs.serve.cp_group_size
    assert cp_group_size is not None

    step_within_q = k_idx // cfgs.batch_size
    k_group = step_within_q // cp_group_size
    hop = lax.rem(step_within_q, cp_group_size)
    is_round_0 = hop == 0
    local_k_idx = k_group * cfgs.batch_size + target_lane
    is_valid_task = local_k_idx < num_k
    loop_k_idx = local_k_idx * cp_group_size + hop
    is_last_k = jnp.where(
        (hop == cp_group_size - 1) & (local_k_idx == num_k - 1), 1, 0
    )

    sched.s_idx[step, target_lane] = jnp.where(is_valid_task, s_idx, -1)
    sched.q_idx[step, target_lane] = jnp.where(is_valid_task, q_idx, 0)
    sched.k_idx[step, target_lane] = jnp.where(is_valid_task, loop_k_idx, 0)
    sched.is_last_k[step, target_lane] = is_last_k
    sched.do_writeback[step, target_lane] = 0

    sched.dma_q[step, target_lane, 0] = jnp.where(is_valid_task, q_src, 0)
    sched.dma_q[step, target_lane, 1] = jnp.where(is_valid_task, q_sz_task, 0)

    kv_len_start = local_k_idx * cfgs.bkv_sz
    kv_p_start = local_k_idx * cfgs.bkv_p
    kv_left_frm_cache = jnp.where(
        is_valid_task, jnp.maximum(kv_cache_len - kv_len_start, 0), 0
    )
    p_offset = s_idx * cfgs.serve.pages_per_seq + kv_p_start

    for i in range(cfgs.bkv_p_cache):
      dst_vmem = i << cfgs.serve.page_size_log2
      dma_sz = jnp.clip(kv_left_frm_cache - dst_vmem, 0, cfgs.serve.page_size)
      src_hbm = jnp.minimum(p_offset + i, cfgs.serve.num_page_indices - 1)
      # Only round 0 (hop == 0) fetches from HBM; subsequent rounds receive KV
      # from peer devices over the ICI ring.
      fetch_sz = jnp.where(is_round_0, dma_sz, 0)
      sched.dma_kv_cache[step, target_lane, i, 0] = src_hbm
      sched.dma_kv_cache[step, target_lane, i, 1] = dst_vmem
      sched.dma_kv_cache[step, target_lane, i, 2] = utils.align_to(
          fetch_sz, cfgs.kv_token_align
      )

    for i in range(cfgs.bkv_p_new):
      dma_entry = sched.dma_kv_new[step, target_lane, i]
      dma_entry.fetch_hbm[...] = 0
      dma_entry.fetch_vmem[...] = 0
      dma_entry.wb_hbm[...] = 0
      dma_entry.wb_vmem[...] = 0
      dma_entry.set_flags(0, 0)

    return self.advance_carry(carry)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StripedCacheScheduler(schedule.BaseScheduler):
  """Scheduler attending only to this rank's page-striped KV cache (DCP cache).

  Attributes:
    cache_lens: [max_num_seqs] global cached KV length per sequence.
    cp_rank: [1] rank of the current device in the CP group.
    cp_group_size: Number of devices in the CP group.
    q_positions: Optional [max_num_seqs] starting query position per sequence.
    is_packed_lse: Whether to store LSE in packed 1-row-per-token layout.
  """

  computer_cls: ClassVar[type[schedule.BaseMetadataComputer]] = (
      schedule.BaseMetadataComputer
  )
  has_new_kv: ClassVar[bool] = False

  cache_lens: jax.Array
  cp_rank: jax.Array
  cp_group_size: int = dataclasses.field(metadata=dict(static=True))
  q_positions: jax.Array | None = None
  is_packed_lse: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )

  @property
  def seq_lens(self) -> jax.Array:
    return self.cache_lens

  def validate(self, model_cfgs: configs.ModelConfigs) -> None:
    if model_cfgs.sliding_window is not None:
      raise ValueError(
          "Context Parallel does not support sliding window right now"
      )

  def get_kv_lens(
      self, q_lens: jax.Array, page_size: int = 1
  ) -> tuple[jax.Array, jax.Array]:
    kv_cache_lens = utils.cp_local_cache_len(
        self.cache_lens, self.cp_group_size, self.cp_rank[0], page_size
    )
    return kv_cache_lens, jnp.zeros_like(q_lens)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RingCacheScheduler(StripedCacheScheduler):
  """Scheduler rotating page-striped KV cache across the PCP ring.

  Attributes:
    ring_axis_name: Mesh axis name of the PCP ring.
    ring_mesh_axis_names: Full tuple of mesh axis names when inside a multi-axis
      shard_map.
    is_packed_lse: Defaults to True for ring cache phase.
  """

  computer_cls: ClassVar[type[schedule.BaseMetadataComputer]] = (
      RingCacheMetadataComputer
  )
  has_new_kv: ClassVar[bool] = False

  ring_axis_name: str = dataclasses.field(
      default="pcp", metadata=dict(static=True)
  )
  ring_mesh_axis_names: tuple[str, ...] | None = dataclasses.field(
      default=None, metadata=dict(static=True)
  )
  is_packed_lse: bool = dataclasses.field(
      default=True, metadata=dict(static=True)
  )

  @property
  def extra_scalars(self) -> tuple[jax.Array, ...]:
    return (self.cp_rank,)

  def get_kv_lens(
      self, q_lens: jax.Array, page_size: int = 1
  ) -> tuple[jax.Array, jax.Array]:
    del page_size
    return self.cache_lens, jnp.zeros_like(q_lens)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StripedNewTokensScheduler(schedule.BaseScheduler):
  """Scheduler attending only to new KV tokens and writing page-striped cache.

  Attributes:
    new_lens: [max_num_seqs] number of new KV tokens per sequence.
    cache_lens: [max_num_seqs] global cached KV length prior to the new tokens.
    cp_rank: [1] rank of the current device in the CP group.
    cp_group_size: Number of devices in the CP group.
    q_positions: Optional [max_num_seqs] starting query position per sequence.
    update_kv_cache: Optional bool or [max_num_seqs] flag for KV cache writeback.
    new_kv_page_indices: Optional page table mapping logical new-KV pages to
      physical pages in the all-gathered new_kv buffer.
    is_packed_lse: Whether to store LSE in packed 1-row-per-token layout.
  """

  computer_cls: ClassVar[type[schedule.BaseMetadataComputer]] = (
      CPMetadataComputer
  )
  has_cache_kv: ClassVar[bool] = False
  has_new_kv: ClassVar[bool] = True

  new_lens: jax.Array
  cache_lens: jax.Array
  cp_rank: jax.Array
  cp_group_size: int = dataclasses.field(metadata=dict(static=True))
  q_positions: jax.Array | None = None
  update_kv_cache: jax.Array | bool | None = None
  new_kv_page_indices: jax.Array | None = None
  is_packed_lse: bool = dataclasses.field(
      default=False, metadata=dict(static=True)
  )

  @property
  def seq_lens(self) -> jax.Array:
    return self.new_lens

  @property
  def paged_new_kv(self) -> bool:
    return self.new_kv_page_indices is not None

  @property
  def extra_scalars(self) -> tuple[jax.Array, ...]:
    if self.new_kv_page_indices is None:
      return (self.cp_rank,)
    return (self.cp_rank, self.new_kv_page_indices)

  def validate(self, model_cfgs: configs.ModelConfigs) -> None:
    if model_cfgs.sliding_window is not None:
      raise ValueError(
          "Context Parallel does not support sliding window right now"
      )
    if self.new_kv_page_indices is not None:
      max_num_seqs = self.new_lens.shape[0]
      if self.new_kv_page_indices.shape[0] % max_num_seqs != 0:
        raise ValueError(
            f"Expected {self.new_kv_page_indices.shape[0]=} to be divisible by "
            f"{max_num_seqs=}."
        )

  def get_kv_lens(
      self, q_lens: jax.Array, page_size: int = 1
  ) -> tuple[jax.Array, jax.Array]:
    del q_lens, page_size
    return self.cache_lens, self.new_lens