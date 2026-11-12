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
"""Async DMA pipelines, DCP8 remote copies, and 2-level hierarchical all-reduce."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, cast

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.megakernel.core.collectives import (
    PeerTopology,
    RemoteCopyBatch,
    make_remote_copy_batch,
)

from .constants import TP_AXIS_NAME


class AsyncCopyHandle(Protocol):
    """Pallas async DMA copy handle protocol."""

    def start(self) -> None:
        """Start async DMA copy."""
        ...

    def wait(self) -> None:
        """Wait for async DMA copy to complete."""
        ...


class BufferSlotRef(Protocol):
    """Indexable and mutable Pallas VMEM buffer slot reference."""

    def __getitem__(self, idx: object) -> jax.Array:
        """Read slice from buffer slot."""
        ...

    def __setitem__(self, idx: object, val: jax.Array) -> None:
        """Write slice into buffer slot."""
        ...


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .shapes import Qwen38Config


@dataclasses.dataclass(frozen=True, slots=True)
class HierAllReduceBuffers:
    """Scratchpad buffer and semaphore references for hierarchical all-reduce."""

    ar_send: jax.Array
    ar_recv: jax.Array
    ar_stage_send: jax.Array
    ar_stage_recv: jax.Array
    ar_ssem1: jax.Array
    ar_rsem1: jax.Array
    ar_ssem2: jax.Array
    ar_rsem2: jax.Array

    @classmethod
    def from_scratch(cls, scratch: Mapping[str, jax.Array]) -> HierAllReduceBuffers:
        """Extract all-reduce buffer and semaphore references from scratch map."""
        return cls(
            ar_send=scratch["ar_send"],
            ar_recv=scratch["ar_recv"],
            ar_stage_send=scratch["ar_stage_send"],
            ar_stage_recv=scratch["ar_stage_recv"],
            ar_ssem1=scratch["ar_ssem1"],
            ar_rsem1=scratch["ar_rsem1"],
            ar_ssem2=scratch["ar_ssem2"],
            ar_rsem2=scratch["ar_rsem2"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DcpQueryBuffers:
    """Scratchpad buffer and semaphore references for DCP query replication."""

    q_local_send: jax.Array
    q_dcp_recv: jax.Array
    dcp_q_ssem: jax.Array
    dcp_q_rsem: jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class DcpLseBuffers:
    """Scratchpad buffer and semaphore references for DCP LSE exchange."""

    lse_part_send: jax.Array
    lse_part_recv: jax.Array
    dcp_lse_ssem: jax.Array
    dcp_lse_rsem: jax.Array


def _send_ring_copy_batch(
    send_buf: jax.Array,
    targets: tuple[Sequence[object], Sequence[int | jax.Array]],
    sems: tuple[Sequence[object], Sequence[object]],
    *,
    axis_name: str | None,
) -> RemoteCopyBatch:
    """Build and start a remote copy batch from `send_buf`."""
    dst_slices, peer_ids = targets
    src_slices = [send_buf.at[idx] for idx, _ in enumerate(peer_ids)]
    ssem_slices, rsem_slices = sems
    batch = make_remote_copy_batch(
        src_slices,
        dst_slices,
        peer_ids,
        ssem_slices,
        rsem_slices,
        axis_name=axis_name,
    )
    batch.start()
    return batch


def _dcp_ring_replicate(
    bufs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    payload: jax.Array,
    me: jax.Array,
    cfg: Qwen38Config,
    *,
    axis_name: str | None,
) -> RemoteCopyBatch:
    """Replicate `payload` across the `cfg.DCP` ring peers."""
    send_buf, recv_buf, ssem, rsem = bufs
    if cfg.DCP <= 1:
        recv_buf[0] = payload
        return make_remote_copy_batch([], [], [], [], [], axis_name=axis_name)
    topo_dcp = PeerTopology(rank=me, world_size=cfg.R, group_size=cfg.DCP)
    dcp_slot = topo_dcp.intra_rank
    recv_buf[dcp_slot] = payload
    n_peers = cfg.DCP - 1
    for idx in range(n_peers):
        send_buf[idx] = payload
    return _send_ring_copy_batch(
        send_buf,
        (
            [recv_buf.at[dcp_slot] for _ in range(n_peers)],
            [topo_dcp.ring_next_peer(idx + 1) for idx in range(n_peers)],
        ),
        (
            [ssem.at[idx] for idx in range(n_peers)],
            [rsem.at[idx] for idx in range(n_peers)],
        ),
        axis_name=axis_name,
    )


class Qwen38HierAllReduce:
    """Two-level hierarchical TP32 all-reduce (`4x8`: 4-rank intra-group + 8-way XOR).

    Composes `PeerTopology` and `make_remote_copy_batch` from
    `megakernel.core.collectives` without adding model-specific branching
    to the shared communication library.
    """

    def __init__(
        self,
        buffers: HierAllReduceBuffers,
        me: jax.Array,
        cfg: Qwen38Config,
        *,
        axis_name: str | None = TP_AXIS_NAME,
    ) -> None:
        """Initializes the instance."""
        self.buffers = buffers
        self.ar_send = buffers.ar_send
        self.ar_recv = buffers.ar_recv
        self.ar_stage_send = buffers.ar_stage_send
        self.ar_stage_recv = buffers.ar_stage_recv
        self.ar_ssem1 = buffers.ar_ssem1
        self.ar_rsem1 = buffers.ar_rsem1
        self.ar_ssem2 = buffers.ar_ssem2
        self.ar_rsem2 = buffers.ar_rsem2
        self.me = me
        self.cfg = cfg
        self.axis_name = axis_name
        self.group_size = min(cfg.hier_group, cfg.R)
        self.num_groups = cfg.R // self.group_size
        self.topo = PeerTopology(rank=me, world_size=cfg.R, group_size=self.group_size)

    def allreduce(self, slot: int | jax.Array, mat_f32: jax.Array) -> jax.Array:
        """All-reduce `(B, H)` float32 partials across all `R` ranks."""
        mat_bf16 = mat_f32.astype(jnp.bfloat16)
        acc = mat_bf16.astype(jnp.float32)

        if self.group_size > 1:
            n_intra = self.group_size - 1
            for idx in range(n_intra):
                self.ar_send[idx, :, :] = mat_bf16
            _send_ring_copy_batch(
                self.ar_send,
                (
                    [self.ar_recv.at[slot, idx] for idx in range(n_intra)],
                    [self.topo.ring_next_peer(idx + 1) for idx in range(n_intra)],
                ),
                (
                    [self.ar_ssem1.at[slot, idx] for idx in range(n_intra)],
                    [self.ar_rsem1.at[slot, idx] for idx in range(n_intra)],
                ),
                axis_name=self.axis_name,
            ).wait()
            for idx in range(n_intra):
                acc = acc + self.ar_recv[slot, idx, :, :].astype(jnp.float32)

        if self.num_groups > 1:
            n_inter = self.num_groups - 1
            for idx in range(n_inter):
                self.ar_stage_send[idx, :, :] = acc.astype(jnp.float32)
            _send_ring_copy_batch(
                self.ar_stage_send,
                (
                    [self.ar_stage_recv.at[slot, idx] for idx in range(n_inter)],
                    [self.topo.inter_xor_peer(idx + 1) for idx in range(n_inter)],
                ),
                (
                    [self.ar_ssem2.at[slot, idx] for idx in range(n_inter)],
                    [self.ar_rsem2.at[slot, idx] for idx in range(n_inter)],
                ),
                axis_name=self.axis_name,
            ).wait()
            for idx in range(n_inter):
                acc = acc + self.ar_stage_recv[slot, idx, :, :].astype(jnp.float32)

        return acc


def global_barrier(
    b_sem: jax.Array,
    me: jax.Array,
    cfg: Qwen38Config,
    *,
    axis_name: str | None = TP_AXIS_NAME,
    sem_offset: int = 0,
) -> None:
    """Two-stage hierarchical barrier matching the 4x8 all-reduce topology."""
    g = min(cfg.hier_group, cfg.R)
    topo = PeerTopology(rank=me, world_size=cfg.R, group_size=g)
    for step in range(1, g):
        peer = topo.ring_next_peer(step)
        dev_id: dict[str, int | jax.Array] | tuple[int | jax.Array, ...] = (
            {axis_name: peer} if axis_name is not None else (peer,)
        )
        pl.semaphore_signal(
            b_sem.at[sem_offset],
            1,
            device_id=dev_id,
            device_id_type=pl.DeviceIdType.MESH,
        )
    if g > 1:
        pl.semaphore_wait(b_sem.at[sem_offset], g - 1)

    num_groups = cfg.R // g
    for mask in range(1, num_groups):
        peer = topo.inter_xor_peer(mask)
        dev_id2: dict[str, int | jax.Array] | tuple[int | jax.Array, ...] = (
            {axis_name: peer} if axis_name is not None else (peer,)
        )
        pl.semaphore_signal(
            b_sem.at[sem_offset + 1],
            1,
            device_id=dev_id2,
            device_id_type=pl.DeviceIdType.MESH,
        )
    if num_groups > 1:
        pl.semaphore_wait(b_sem.at[sem_offset + 1], num_groups - 1)


def dcp_group_query_start(
    buffers: DcpQueryBuffers,
    q_rot_bf16: jax.Array,
    me: jax.Array,
    cfg: Qwen38Config,
    *,
    axis_name: str | None = TP_AXIS_NAME,
) -> RemoteCopyBatch:
    """Launch async remote copies to replicate local `Q_R` heads across DCP8."""
    return _dcp_ring_replicate(
        (
            buffers.q_local_send,
            buffers.q_dcp_recv,
            buffers.dcp_q_ssem,
            buffers.dcp_q_rsem,
        ),
        q_rot_bf16,
        me,
        cfg,
        axis_name=axis_name,
    )


def dcp_group_query_wait(copies: RemoteCopyBatch) -> None:
    """Wait for all DCP query remote copies to complete."""
    copies.wait()


def dcp_lse_exchange(
    buffers: DcpLseBuffers,
    local_stats: tuple[jax.Array, jax.Array, jax.Array],
    me: jax.Array,
    cfg: Qwen38Config,
    *,
    axis_name: str | None = TP_AXIS_NAME,
) -> None:
    """Exchange partial `(O_num, m, l)` tensors across the 8-rank DCP group."""
    m_local, l_local, acc_local = local_stats
    packed = jnp.concatenate(
        [acc_local, m_local[:, :, None], l_local[:, :, None]], axis=-1
    )
    _dcp_ring_replicate(
        (
            buffers.lse_part_send,
            buffers.lse_part_recv,
            buffers.dcp_lse_ssem,
            buffers.dcp_lse_rsem,
        ),
        packed,
        me,
        cfg,
        axis_name=axis_name,
    ).wait()


def _start_async_copy(src: object, dst: object, sem: object) -> AsyncCopyHandle:
    """Create and start a Pallas async DMA copy."""
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    return cast("AsyncCopyHandle", cp)


def _wait_async_copy(src: object, dst: object, sem: object) -> None:
    """Create a matching Pallas async DMA copy descriptor and wait for completion."""
    pltpu.make_async_copy(src, dst, sem).wait()


def _slice_4d_htile(
    hbm_ref: jax.Array, idx0: int | jax.Array, ht_idx: int | jax.Array
) -> object:
    """Return HBM slice `hbm_ref.at[idx0, ht_idx, :, :]`."""
    d2, d3 = hbm_ref.shape[2], hbm_ref.shape[3]
    return hbm_ref.at[idx0, ht_idx, pl.ds(0, d2), pl.ds(0, d3)]


def _slice_5d_expert_htile(
    hbm_ref: jax.Array,
    coords: tuple[int | jax.Array, int | jax.Array, int | jax.Array],
) -> object:
    """Return HBM slice `hbm_ref.at[layer_idx, expert_id, ht_idx, :, :]`."""
    layer_idx, expert_id, ht_idx = coords
    d3, d4 = hbm_ref.shape[3], hbm_ref.shape[4]
    return hbm_ref.at[layer_idx, expert_id, ht_idx, pl.ds(0, d3), pl.ds(0, d4)]


def _slice_kv_page(
    pool_hbm: Sequence[jax.Array],
    att_idx: int,
    page_idx: int | jax.Array,
) -> object:
    """Return HBM slice `pool_hbm[att_idx].at[page_idx, :, :, :, :]`."""
    reg_hbm = pool_hbm[att_idx]
    _, d1, d2, d3, d4 = reg_hbm.shape
    return reg_hbm.at[
        page_idx,
        pl.ds(0, d1),
        pl.ds(0, d2),
        pl.ds(0, d3),
        pl.ds(0, d4),
    ]


def copy_3d_tile(
    hbm_ref: jax.Array, idx0: int | jax.Array, vmem_dst: object, sem: object
) -> AsyncCopyHandle:
    """Async copy a 2D slice `hbm_ref[idx0, :, :]` between HBM and scratchpad."""
    d1, d2 = hbm_ref.shape[1], hbm_ref.shape[2]
    return _start_async_copy(
        hbm_ref.at[idx0, pl.ds(0, d1), pl.ds(0, d2)], vmem_dst, sem
    )


def copy_4d_htile(
    hbm_ref: jax.Array,
    idx0: int | jax.Array,
    ht_idx: int | jax.Array,
    vmem_dst: object,
    sem: object,
) -> AsyncCopyHandle:
    """Async copy a 2D H-tile slice `hbm_ref[idx0, ht_idx, :, :]`."""
    return _start_async_copy(_slice_4d_htile(hbm_ref, idx0, ht_idx), vmem_dst, sem)


def wait_4d_htile(
    hbm_ref: jax.Array,
    idx0: int | jax.Array,
    ht_idx: int | jax.Array,
    vmem_dst: object,
    sem: object,
) -> None:
    """Wait for async copy of a 2D H-tile slice `hbm_ref[idx0, ht_idx, :, :]`."""
    _wait_async_copy(_slice_4d_htile(hbm_ref, idx0, ht_idx), vmem_dst, sem)


def copy_5d_expert_htile(
    hbm_ref: jax.Array,
    coords: tuple[int | jax.Array, int | jax.Array, int | jax.Array],
    vmem_dst: object,
    sem: object,
) -> AsyncCopyHandle:
    """Async copy routed expert H-tile `hbm_ref[layer_idx, expert_id, ht_idx]`."""
    return _start_async_copy(_slice_5d_expert_htile(hbm_ref, coords), vmem_dst, sem)


def wait_5d_expert_htile(
    hbm_ref: jax.Array,
    coords: tuple[int | jax.Array, int | jax.Array, int | jax.Array],
    vmem_dst: object,
    sem: object,
) -> None:
    """Wait for async copy of expert H-tile `hbm_ref[layer_idx, expert_id, ht_idx]`."""
    _wait_async_copy(_slice_5d_expert_htile(hbm_ref, coords), vmem_dst, sem)


def copy_4d_expert_full(
    hbm_ref: jax.Array,
    layer_idx: int | jax.Array,
    expert_id: int | jax.Array,
    vmem_dst: object,
    sem: object,
) -> AsyncCopyHandle:
    """Async copy routed expert down-projection `hbm_ref[layer_idx, expert_id]`."""
    return copy_4d_htile(hbm_ref, layer_idx, expert_id, vmem_dst, sem)


def _gdn_state_slices(
    hbm_states: tuple[jax.Array, jax.Array],
    gdn_idx: int | jax.Array,
) -> tuple[object, object]:
    """Return HBM slices for GDN conv and recurrent states at `gdn_idx`."""
    conv_hbm, rec_hbm = hbm_states
    _, b_dim, ck_dim, qkv_dim = conv_hbm.shape
    _, _, vh_dim, kd_dim, vd_dim = rec_hbm.shape
    conv_slice = conv_hbm.at[
        gdn_idx, pl.ds(0, b_dim), pl.ds(0, ck_dim), pl.ds(0, qkv_dim)
    ]
    rec_slice = rec_hbm.at[
        gdn_idx,
        pl.ds(0, b_dim),
        pl.ds(0, vh_dim),
        pl.ds(0, kd_dim),
        pl.ds(0, vd_dim),
    ]
    return conv_slice, rec_slice


def _copy_gdn_states(
    hbm_states: tuple[jax.Array, jax.Array],
    gdn_idx: int | jax.Array,
    vmem_slots: tuple[object, object],
    sems: tuple[object, object],
    *,
    to_hbm: bool,
) -> tuple[AsyncCopyHandle, AsyncCopyHandle]:
    """Start async DMAs transferring GDN conv and recurrent states."""
    conv_slice, rec_slice = _gdn_state_slices(hbm_states, gdn_idx)
    conv_vmem_slot, rec_vmem_slot = vmem_slots
    sem_conv, sem_rec = sems
    if to_hbm:
        return (
            _start_async_copy(conv_vmem_slot, conv_slice, sem_conv),
            _start_async_copy(rec_vmem_slot, rec_slice, sem_rec),
        )
    return (
        _start_async_copy(conv_slice, conv_vmem_slot, sem_conv),
        _start_async_copy(rec_slice, rec_vmem_slot, sem_rec),
    )


def copy_gdn_states_in(
    hbm_states: tuple[jax.Array, jax.Array],
    gdn_idx: int | jax.Array,
    vmem_slots: tuple[object, object],
    sems: tuple[object, object],
) -> tuple[AsyncCopyHandle, AsyncCopyHandle]:
    """Start async DMAs to load GDN conv and recurrent states for `gdn_idx`."""
    return _copy_gdn_states(hbm_states, gdn_idx, vmem_slots, sems, to_hbm=False)


def copy_gdn_states_out(
    vmem_slots: tuple[object, object],
    hbm_states: tuple[jax.Array, jax.Array],
    gdn_idx: int | jax.Array,
    sems: tuple[object, object],
) -> tuple[AsyncCopyHandle, AsyncCopyHandle]:
    """Start async DMAs to write updated GDN conv and recurrent states back to HBM."""
    return _copy_gdn_states(hbm_states, gdn_idx, vmem_slots, sems, to_hbm=True)


def copy_kv_page_in(
    pool_hbm: Sequence[jax.Array],
    att_idx: int,
    page_idx: int | jax.Array,
    kv_page_slot: object,
    sem: object,
) -> AsyncCopyHandle:
    """Start async DMA to fetch one physical KV page `(2, K_R, D, page_size)`."""
    return _start_async_copy(
        _slice_kv_page(pool_hbm, att_idx, page_idx), kv_page_slot, sem
    )


def wait_kv_page_in(
    pool_hbm: Sequence[jax.Array],
    att_idx: int,
    page_idx: int | jax.Array,
    kv_page_slot: object,
    sem: object,
) -> None:
    """Wait for async DMA of one physical KV page `(2, K_R, D, page_size)`."""
    _wait_async_copy(_slice_kv_page(pool_hbm, att_idx, page_idx), kv_page_slot, sem)


def copy_kv_page_out(
    kv_page_slot: object,
    pool_hbm: Sequence[jax.Array],
    att_idx: int,
    page_idx: int | jax.Array,
    sem: object,
) -> AsyncCopyHandle:
    """Start async DMA to write one physical KV page `(2, K_R, D, page_size)`."""
    return _start_async_copy(
        kv_page_slot, _slice_kv_page(pool_hbm, att_idx, page_idx), sem
    )
