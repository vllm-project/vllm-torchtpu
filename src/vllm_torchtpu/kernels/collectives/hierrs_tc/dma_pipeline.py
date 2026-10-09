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
"""Remote DMAs and accumulate pipelines of the micro-batched kernel."""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.hierrs_tc.config import Config
from vllm_torchtpu.kernels.collectives.hierrs_tc.topology import ChunkLocator, Topology


def scoped(name: str):
    # Not visible in xprof yet: Mosaic emits trace_start at level 10.
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with jax.named_scope(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RemoteWaitBufferedRef(pltpu.BufferedRef):
    """BufferedRef whose copy_in first waits for a peer's write to land."""

    index_fn_with_recv_sem: Callable[..., Any] | None = dataclasses.field(
        metadata={"static": True}, default=None
    )

    @classmethod
    def from_ref(
        cls,
        ref: pltpu.BufferedRef,
        *,
        index_fn_with_recv_sem: Callable | None = None,
    ):
        return cls(
            index_fn_with_recv_sem=index_fn_with_recv_sem,
            **{
                f.name: getattr(ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    def copy_in(self, src_ref, grid_indices):
        if self.index_fn_with_recv_sem is None:
            super().copy_in(src_ref, grid_indices)
            return
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        chunk_slice, sem, size = self.index_fn_with_recv_sem(grid_indices, src_ref)

        window_ref_slice = (slot, slice(None), pl.ds(0, size))
        if sem is not None:
            pltpu.make_async_copy(
                self.window_ref.at[window_ref_slice],
                self.window_ref.at[window_ref_slice],
                sem,
            ).wait()

        hbm_array_ref = src_ref[0] if isinstance(src_ref, (tuple, list)) else src_ref
        assert self.sem_recvs is not None
        pltpu.make_async_copy(
            hbm_array_ref.at[chunk_slice],
            self.window_ref.at[window_ref_slice],
            self.sem_recvs.at[slot],
        ).start()

    def wait_in(self, src_ref, grid_indices):
        if self.index_fn_with_recv_sem is None:
            super().wait_in(src_ref, grid_indices)
            return
        assert self.window_ref is not None
        wait_slot = self.current_wait_in_slot
        _, _, size = self.index_fn_with_recv_sem(grid_indices, src_ref)

        window_ref_slice = (wait_slot, slice(None), pl.ds(0, size))
        assert self.sem_recvs is not None
        pltpu.make_async_copy(
            self.window_ref.at[window_ref_slice],
            self.window_ref.at[window_ref_slice],
            self.sem_recvs.at[wait_slot],
        ).wait()


class DmaManager:
    def __init__(
        self,
        config: Config,
        topo: Topology,
        locator: ChunkLocator,
        recv_bref,
        run_bref,
        out_bref,
        phase1_send_sems,
        phase1_recv_sems,
        phase2_send_sems,
        phase2_recv_sems,
    ):
        self.config = config
        self.topo = topo
        self.locator = locator
        self.recv_bref = recv_bref
        self.run_bref = run_bref
        self.out_bref = out_bref
        self.phase1_send_sems = phase1_send_sems
        self.phase1_recv_sems = phase1_recv_sems
        self.phase2_send_sems = phase2_send_sems
        self.phase2_recv_sems = phase2_recv_sems

    def start_phase1_d2d_copies(self, src, dst, mb_idx):
        """Start DMA transfer of chunks that belong to the `mb_idx`-th micro batch
        from `src` to the partner's `dst`."""
        ops = []
        mb_start = mb_idx * self.config.mb_size
        mb_slice_size = self.config.mb_size
        partner_chunks = self.locator.get_phase1_chunk_idxes(self.topo.partner_id)
        for chip_idx, c_neigh in enumerate(partner_chunks):
            src_slice = self.locator.get_slice(
                chunk_idx=c_neigh, start=mb_start, size=mb_slice_size
            )
            dst_slice = self.locator.get_packed_slice(
                chunk_idx=c_neigh, start=mb_start, size=mb_slice_size
            )
            op = pltpu.make_async_remote_copy(
                src_ref=src.at[src_slice],
                dst_ref=dst.at[dst_slice],
                send_sem=self.phase1_send_sems.at[chip_idx, mb_idx],
                recv_sem=self.phase1_recv_sems.at[chip_idx, mb_idx],
                device_id=self.topo.partner_id,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )
            op.start()
            ops.append(op)
        return ops

    def start_phase2_c2c_copies(self, mb_idx, step_idx, src, dst):
        send_sems = self.phase2_send_sems
        recv_sems = self.phase2_recv_sems
        mb_ops = []
        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0

        for op_idx in range(num_ops_in_step):
            for hcube_dim_idx in range(self.config.num_hcube_dims):
                dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims

                chunk_start = (
                    mb_idx * self.config.mb_size
                    + hcube_dim_idx * self.config.hc_chunk_size
                )
                k_size = self.config.hc_chunk_size

                neigh_device_id = self.topo.get_neighbor_device_id(dim)
                neighbor_chunk_idx = self.locator.get_phase2_chunk_idx(
                    neigh_device_id, step_idx, op_idx, hcube_dim_idx
                )

                # Sender and receiver share chunk parity: one packed slice
                # addresses both ends.
                mb_slice = self.locator.get_packed_slice(
                    neighbor_chunk_idx, chunk_start, k_size
                )
                data_op = pltpu.make_async_remote_copy(
                    src_ref=src.at[mb_slice],
                    dst_ref=dst.at[mb_slice],
                    send_sem=send_sems.at[step_idx, mb_idx, hcube_dim_idx, op_idx],
                    recv_sem=recv_sems.at[step_idx, mb_idx, hcube_dim_idx, op_idx],
                    device_id=neigh_device_id,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
                data_op.start()
                mb_ops.append(data_op)
        return mb_ops

    @scoped("p1_accum")
    def run_phase1_accumulate_pipeline(
        self,
        src1,
        src2,
        dst,
        in1_index_fn,
        in2_index_fn,
        out_index_fn,
        hbm_index_fn,
        block_size,
        mb_idx,
    ):
        # src1 and dst are packed; src2 is the full-size input.
        def accum_body(s1_ref, s2_ref, d_ref):
            d_ref[...] = s1_ref[...] + s2_ref[...]

        grid = (self.config.num_chips,)

        def in_index_fn_with_recv_sem(grid_indices, ref):
            hbm_index, size = hbm_index_fn(grid_indices, ref)
            (chip_idx,) = grid_indices
            sem = self.phase1_recv_sems.at[chip_idx, mb_idx]
            return hbm_index, sem, size

        in1_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=in1_index_fn,
        )
        in2_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=in2_index_fn,
        )
        out_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=out_index_fn,
        )

        s1_bref = RemoteWaitBufferedRef.from_ref(
            self.recv_bref.with_spec(in1_spec),
            index_fn_with_recv_sem=in_index_fn_with_recv_sem,
        )
        s2_bref = RemoteWaitBufferedRef.from_ref(self.run_bref.with_spec(in2_spec))
        d_bref = RemoteWaitBufferedRef.from_ref(self.out_bref.with_spec(out_spec))

        pltpu.emit_pipeline(
            accum_body,
            grid=grid,
            in_specs=[in1_spec, in2_spec],
            out_specs=[out_spec],
        )(src1, src2, dst, allocations=[s1_bref, s2_bref, d_bref])

    @scoped("p2_accum")
    def run_phase2_accumulate_pipeline(
        self,
        src1,
        src2,
        dst,
        in_index_fn,
        out_index_fn,
        hbm_index_fn,
        block_size,
        mb_idx,
        step_idx,
    ):
        def accum_body(s1_ref, s2_ref, d_ref):
            d_ref[...] = s1_ref[...] + s2_ref[...]

        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0
        grid = (num_ops_in_step, self.config.num_hcube_dims)

        def in_index_fn_with_recv_sem(grid_indices, ref):
            hbm_index, size = hbm_index_fn(grid_indices, ref)
            op_idx, hcube_dim_idx = grid_indices
            sem = self.phase2_recv_sems.at[step_idx, mb_idx, hcube_dim_idx, op_idx]
            return hbm_index, sem, size

        in_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=in_index_fn,
        )
        out_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, block_size),
            index_map=out_index_fn,
        )

        s1_bref = RemoteWaitBufferedRef.from_ref(
            self.recv_bref.with_spec(in_spec),
            index_fn_with_recv_sem=in_index_fn_with_recv_sem,
        )
        s2_bref = RemoteWaitBufferedRef.from_ref(self.run_bref.with_spec(in_spec))
        d_bref = RemoteWaitBufferedRef.from_ref(self.out_bref.with_spec(out_spec))

        pltpu.emit_pipeline(
            accum_body,
            grid=grid,
            in_specs=[in_spec, in_spec],
            out_specs=[out_spec],
        )(src1, src2, dst, allocations=[s1_bref, s2_bref, d_bref])
