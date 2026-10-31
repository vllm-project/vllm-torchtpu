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
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.hierrs_tc.config import (
    FP8_E4M3_MAX,
    SCALE_LANE,
    Config,
)
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
        fp8_send_buf=None,
        fp8_recv_buf=None,
        scale_send_buf=None,
        scale_recv_buf=None,
        scale_p2_send_sems=None,
        scale_p2_recv_sems=None,
        fp8_send_bref=None,
        scale_send_bref=None,
        fp8_recv_bref=None,
        scale_bref=None,
        fp8_static_scale=None,
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
        self.fp8_send_buf = fp8_send_buf
        self.fp8_recv_buf = fp8_recv_buf
        self.scale_send_buf = scale_send_buf
        self.scale_recv_buf = scale_recv_buf
        self.scale_p2_send_sems = scale_p2_send_sems
        self.scale_p2_recv_sems = scale_p2_recv_sems
        self.fp8_send_bref = fp8_send_bref
        self.scale_send_bref = scale_send_bref
        self.fp8_recv_bref = fp8_recv_bref
        self.scale_bref = scale_bref
        self.fp8_static_scale = fp8_static_scale
        self.skip_scale_dma = config.skip_scale_dma

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

    def start_phase2_c2c_copies(self, mb_idx, step_idx, src, dst, fp8=False):
        """Returns (data_op, scale_op) pairs; scale_op is None if none is sent."""
        if fp8:
            src, dst = self.fp8_send_buf, self.fp8_recv_buf
        send_sems, recv_sems = self.phase2_send_sems, self.phase2_recv_sems
        send_scale = fp8 and not self.skip_scale_dma
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
                scale_op = None
                if send_scale:
                    scale_op = self._start_scale_copy(
                        neigh_device_id,
                        neighbor_chunk_idx,
                        step_idx,
                        mb_idx,
                        hcube_dim_idx,
                        op_idx,
                    )
                mb_ops.append((data_op, scale_op))
        return mb_ops

    def _start_scale_copy(
        self, neigh_device_id, chunk_idx, step_idx, mb_idx, hcube_dim_idx, op_idx
    ):
        assert self.scale_send_buf is not None
        assert self.scale_recv_buf is not None
        assert self.scale_p2_send_sems is not None
        assert self.scale_p2_recv_sems is not None
        sem_idx = (step_idx, mb_idx, hcube_dim_idx, op_idx)
        scale_slice = self._scale_slice(chunk_idx, *sem_idx)
        op = pltpu.make_async_remote_copy(
            src_ref=self.scale_send_buf.at[scale_slice],
            dst_ref=self.scale_recv_buf.at[scale_slice],
            send_sem=self.scale_p2_send_sems.at[sem_idx],
            recv_sem=self.scale_p2_recv_sems.at[sem_idx],
            device_id=neigh_device_id,
            device_id_type=pl.DeviceIdType.LOGICAL,
        )
        op.start()
        return op

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

    def _scale_slot(self, step_idx, mb_idx, hcube_dim_idx, op_idx):
        max_ops = 2 ** (self.config.num_hcube_dims - 1)
        return (
            step_idx
            * self.config.num_micro_batches
            * self.config.num_hcube_dims
            * max_ops
            + mb_idx * self.config.num_hcube_dims * max_ops
            + hcube_dim_idx * max_ops
            + op_idx
        )

    def _scale_slice(self, chunk_idx, step_idx, mb_idx, hcube_dim_idx, op_idx):
        slot = self._scale_slot(step_idx, mb_idx, hcube_dim_idx, op_idx)
        return pl.ds(chunk_idx, 1), pl.ds(slot * SCALE_LANE, SCALE_LANE)

    def _neighbor_chunk_idx(self, step_idx, op_idx, hcube_dim_idx):
        dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
        return self.locator.get_phase2_chunk_idx(
            self.topo.get_neighbor_device_id(dim), step_idx, op_idx, hcube_dim_idx
        )

    def _quantize_scale(self, data_f32):
        """Returns (scale, 1 / scale), with data * (1 / scale) in the FP8 range."""
        if self.fp8_static_scale is not None:
            return 1.0 / self.fp8_static_scale, self.fp8_static_scale
        # The scale ignores non-finite values so one cannot poison the chunk;
        # the values themselves still propagate, as on the BF16 wire.
        amax = jnp.max(jnp.abs(jnp.where(jnp.isfinite(data_f32), data_f32, 0.0)))
        scale = jnp.where(amax == 0.0, 1.0, amax / FP8_E4M3_MAX)
        return scale, 1.0 / scale

    @scoped("quant_stage")
    def quantize_chunks_to_fp8_staging(self, src_hbm, mb_idx, step_idx):
        assert self.run_bref is not None
        assert self.fp8_send_bref is not None
        assert self.scale_send_bref is not None
        assert self.fp8_send_buf is not None
        assert self.scale_send_buf is not None

        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0
        if num_ops_in_step == 0:
            return
        grid = (num_ops_in_step, self.config.num_hcube_dims)

        def quant_body(bf16_ref, fp8_ref, scale_ref=None):
            data_f32 = bf16_ref[...].astype(jnp.float32)
            scale, inv_scale = self._quantize_scale(data_f32)
            fp8_ref[...] = jnp.clip(
                data_f32 * inv_scale, -FP8_E4M3_MAX, FP8_E4M3_MAX
            ).astype(jnp.float8_e4m3fn)
            if scale_ref is not None:
                scale_ref[...] = jnp.full((1, SCALE_LANE), scale, jnp.float32)

        def send_data_index_fn(op_idx, hcube_dim_idx):
            neigh_chunk_idx = self._neighbor_chunk_idx(step_idx, op_idx, hcube_dim_idx)
            mb_col_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (self.locator.pack(neigh_chunk_idx), mb_col_idx)

        def send_scale_index_fn(op_idx, hcube_dim_idx):
            neigh_chunk_idx = self._neighbor_chunk_idx(step_idx, op_idx, hcube_dim_idx)
            return (
                neigh_chunk_idx,
                self._scale_slot(step_idx, mb_idx, hcube_dim_idx, op_idx),
            )

        bf16_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, self.config.hc_chunk_size),
            index_map=send_data_index_fn,
        )
        fp8_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, self.config.hc_chunk_size),
            index_map=send_data_index_fn,
        )
        scale_spec = pl.BlockSpec(
            block_shape=(1, SCALE_LANE), index_map=send_scale_index_fn
        )

        src_bref = RemoteWaitBufferedRef.from_ref(self.run_bref.with_spec(bf16_spec))
        fp8_out_bref = RemoteWaitBufferedRef.from_ref(
            self.fp8_send_bref.with_spec(fp8_spec)
        )
        scale_out_bref = RemoteWaitBufferedRef.from_ref(
            self.scale_send_bref.with_spec(scale_spec)
        )

        if self.skip_scale_dma:
            pltpu.emit_pipeline(
                quant_body,
                grid=grid,
                in_specs=[bf16_spec],
                out_specs=[fp8_spec],
            )(src_hbm, self.fp8_send_buf, allocations=[src_bref, fp8_out_bref])
        else:
            pltpu.emit_pipeline(
                quant_body,
                grid=grid,
                in_specs=[bf16_spec],
                out_specs=[fp8_spec, scale_spec],
            )(
                src_hbm,
                self.fp8_send_buf,
                self.scale_send_buf,
                allocations=[src_bref, fp8_out_bref, scale_out_bref],
            )

    @scoped("p2_dequant_accum")
    def run_phase2_dequant_accumulate_pipeline(
        self, running_sum_ref, dst_ref, mb_idx, step_idx, is_last_step
    ):
        assert self.fp8_recv_bref is not None
        assert self.scale_bref is not None
        assert self.scale_p2_recv_sems is not None
        assert self.fp8_recv_buf is not None
        assert self.scale_recv_buf is not None
        assert self.run_bref is not None
        assert self.out_bref is not None

        exponent = self.config.num_hcube_dims - 1 - step_idx
        num_ops_in_step = 1 << exponent if exponent >= 0 else 0
        if num_ops_in_step == 0:
            return
        grid = (num_ops_in_step, self.config.num_hcube_dims)

        def accum_body(fp8_ref, run_ref, *rest):
            if self.skip_scale_dma:
                (d_ref,) = rest
                scale_ref = None
            else:
                scale_ref, d_ref = rest
            if self.fp8_static_scale is not None:
                scale = 1.0 / self.fp8_static_scale
            else:
                assert scale_ref is not None
                scale = scale_ref[0, 0]
            recv_dq = (fp8_ref[...].astype(jnp.float32) * scale).astype(jnp.bfloat16)
            d_ref[...] = recv_dq + run_ref[...]

        data_index_fn = self.locator.make_phase2_index_fn(step_idx, mb_idx)
        out_index_fn = (
            self.locator.make_phase2_out_index_fn(step_idx, mb_idx)
            if is_last_step
            else self.locator.make_phase2_index_fn(step_idx, mb_idx)
        )

        def my_chunk_idx(op_idx, hcube_dim_idx):
            return self.locator.get_phase2_chunk_idx(
                self.topo.cur_id, step_idx, op_idx, hcube_dim_idx
            )

        def scale_index_fn(op_idx, hcube_dim_idx):
            return (
                my_chunk_idx(op_idx, hcube_dim_idx),
                self._scale_slot(step_idx, mb_idx, hcube_dim_idx, op_idx),
            )

        def fp8_recv_sem_fn(grid_indices, ref):
            op_idx, hcube_dim_idx = grid_indices
            chunk_idx = my_chunk_idx(op_idx, hcube_dim_idx)
            mb_start = mb_idx * self.config.mb_size
            mb_start_idx = mb_start + hcube_dim_idx * self.config.hc_chunk_size
            chunk_slice = self.locator.get_packed_slice(
                chunk_idx, mb_start_idx, self.config.hc_chunk_size
            )
            sem = self.phase2_recv_sems.at[step_idx, mb_idx, hcube_dim_idx, op_idx]
            return chunk_slice, sem, self.config.hc_chunk_size

        def scale_recv_sem_fn(grid_indices, ref):
            op_idx, hcube_dim_idx = grid_indices
            scale_slice = self._scale_slice(
                my_chunk_idx(op_idx, hcube_dim_idx),
                step_idx,
                mb_idx,
                hcube_dim_idx,
                op_idx,
            )
            sem = self.scale_p2_recv_sems.at[step_idx, mb_idx, hcube_dim_idx, op_idx]
            return scale_slice, sem, SCALE_LANE

        fp8_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, self.config.hc_chunk_size),
            index_map=data_index_fn,
        )
        run_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, self.config.hc_chunk_size),
            index_map=data_index_fn,
        )
        out_spec = pl.BlockSpec(
            block_shape=(self.config.seq_chunk_size, self.config.hc_chunk_size),
            index_map=out_index_fn,
        )
        scale_spec = pl.BlockSpec(block_shape=(1, SCALE_LANE), index_map=scale_index_fn)

        fp8_bref = RemoteWaitBufferedRef.from_ref(
            self.fp8_recv_bref.with_spec(fp8_spec),
            index_fn_with_recv_sem=fp8_recv_sem_fn,
        )
        run_bref = RemoteWaitBufferedRef.from_ref(self.run_bref.with_spec(run_spec))
        scale_bref = RemoteWaitBufferedRef.from_ref(
            self.scale_bref.with_spec(scale_spec),
            index_fn_with_recv_sem=scale_recv_sem_fn,
        )
        out_bref = RemoteWaitBufferedRef.from_ref(self.out_bref.with_spec(out_spec))

        dst = dst_ref if is_last_step else running_sum_ref

        if self.skip_scale_dma:
            pltpu.emit_pipeline(
                accum_body,
                grid=grid,
                in_specs=[fp8_spec, run_spec],
                out_specs=[out_spec],
            )(
                self.fp8_recv_buf,
                running_sum_ref,
                dst,
                allocations=[fp8_bref, run_bref, out_bref],
            )
        else:
            pltpu.emit_pipeline(
                accum_body,
                grid=grid,
                in_specs=[fp8_spec, run_spec, scale_spec],
                out_specs=[out_spec],
            )(
                self.fp8_recv_buf,
                running_sum_ref,
                self.scale_recv_buf,
                dst,
                allocations=[fp8_bref, run_bref, scale_bref, out_bref],
            )
