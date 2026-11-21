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
"""Device and chunk indexing; devices 2k and 2k+1 are the cores of chip k."""

from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.hierrs_tc.config import Config


class Topology:
    def __init__(self, axis_name: str):
        self.cur_id = jax.lax.axis_index(axis_name)
        self.cur_chip_id = self.cur_id // 2
        self.cur_chiplet_bit = self.cur_id % 2
        self.partner_id = jax.lax.select(
            self.cur_chiplet_bit == 0, self.cur_id + 1, self.cur_id - 1
        )

    def get_device_id(self, chip_id, chiplet_bit):
        return chip_id * 2 + chiplet_bit

    def get_neighbor_chip_id(self, dim):
        return self.cur_chip_id ^ (1 << dim)

    def get_neighbor_device_id(self, dim):
        return self.get_device_id(self.get_neighbor_chip_id(dim), self.cur_chiplet_bit)


# Half `h` reduces along 2x2 hypercube dim `h`, then dim `1 - h`. After phase 1
# each device holds one chunk per chip: step 0 keeps chunks matching on dim `h`,
# and step 1 keeps only its own chip's chunk.
def step0_send_chips(dim0_bit: Any, dim1_bit: Any, h: int) -> list[Any]:
    return (
        [(1 - dim0_bit) + 2 * j for j in range(2)]
        if h == 0
        else [2 * (1 - dim1_bit) + j for j in range(2)]
    )


def step0_keep_chips(dim0_bit: Any, dim1_bit: Any, h: int) -> list[Any]:
    return (
        [dim0_bit + 2 * j for j in range(2)]
        if h == 0
        else [2 * dim1_bit + j for j in range(2)]
    )


def step1_send_chip(dim0_bit: Any, dim1_bit: Any, h: int) -> Any:
    return dim0_bit + 2 * (1 - dim1_bit) if h == 0 else 2 * dim1_bit + (1 - dim0_bit)


def tpu_info() -> Any:
    """Active TensorCore `TpuInfo`, falling back to tpu7x on TPU-less CPU tests."""
    try:
        return pltpu.get_tpu_info()
    except ValueError:
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def epilogue_vmem_limit_bytes() -> int:
    """Reserves 15/16 of TensorCore VMEM (60 MiB on 64 MiB tpu7x) for fused ops."""
    return (int(tpu_info().vmem_capacity_bytes) * 15) // 16


def rank_ordered_rows(
    order: tuple[int, ...],
    chip: int,
    bit: Any,
    chunk_rows: int,
    blk: Any,
    block_rows: int,
) -> Any:
    """Row slice in `order` for `(chip, bit)` and streaming block `blk`."""
    src = jnp.where(bit == 0, order[2 * chip], order[2 * chip + 1])
    return pl.ds(src * chunk_rows + blk * block_rows, block_rows)


def wait_recv(buf: Any, sem: Any) -> None:
    pltpu.make_async_copy(buf, buf, sem).wait()


def signal_peers(
    sem: Any,
    peers: Iterable[Any],
    device_id_type: Any = pl.DeviceIdType.LOGICAL,
) -> None:
    for p in peers:
        pl.semaphore_signal(sem, 1, device_id=p, device_id_type=device_id_type)


class ChunkLocator:
    def __init__(
        self, config: Config, topo: Topology, order: tuple[int, ...] | None = None
    ):
        self.config = config
        self.topo = topo
        # Input row block order[c] holds chunk c; None is the identity.
        identity = tuple(range(config.num_devices))
        self.order = None if order is None or tuple(order) == identity else order

    def input_chunk(self, chunk_idx):
        """The input row block that holds `chunk_idx`.

        Only the input is permuted: every working buffer is indexed by chunk.
        """
        if self.order is None:
            return chunk_idx
        if isinstance(chunk_idx, int):
            return self.order[chunk_idx]
        block = jnp.int32(self.order[0])
        for c, b in enumerate(self.order[1:], start=1):
            block = jnp.where(chunk_idx == c, jnp.int32(b), block)
        return block

    def get_slice(self, chunk_idx, start, size):
        return (
            pl.ds(chunk_idx * self.config.seq_chunk_size, self.config.seq_chunk_size),
            pl.ds(start, size),
        )

    def pack(self, chunk_idx):
        # Working buffers only ever hold this device's chunk parity, so they
        # are allocated at half the rows. Not for the input operand.
        return chunk_idx // 2

    def get_packed_slice(self, chunk_idx, start, size):
        return self.get_slice(self.pack(chunk_idx), start, size)

    def get_phase1_chunk_idx(self, device_id, chip_idx):
        chiplet_bit = device_id % 2
        return chip_idx * 2 + chiplet_bit

    def get_phase1_chunk_idxes(self, device_id):
        chiplet_bit = device_id % 2
        return [chip_idx * 2 + chiplet_bit for chip_idx in range(self.config.num_chips)]

    def get_phase2_chunk_idx(self, device_id, step_idx, chunk_group_idx, hcube_dim_idx):
        dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
        chip_id = device_id // 2
        my_dim_bit = (chip_id >> dim) & 1

        prev_dims = [
            (hcube_dim_idx + j) % self.config.num_hcube_dims for j in range(step_idx)
        ]
        future_dims = [
            (hcube_dim_idx + j) % self.config.num_hcube_dims
            for j in range(step_idx + 1, self.config.num_hcube_dims)
        ]

        my_base_chunk_idx = self.get_hcube_chunk_idx(
            device_id, chunk_group_idx, future_dims, prev_dims, dim, my_dim_bit
        )
        chiplet_bit = device_id % 2
        return my_base_chunk_idx * 2 + chiplet_bit

    def get_hcube_chunk_idx(
        self,
        device_id,
        chunk_group_idx,
        future_dims,
        prev_dims,
        target_dim,
        dim_val,
    ):
        # Reduced dims take this chip's bits, pending dims take
        # chunk_group_idx's bits, and target_dim takes dim_val.
        chip_id = device_id // 2
        base = 0
        for d in prev_dims:
            bit = (chip_id >> d) & 1
            base |= bit << d
        for bit_pos, d in enumerate(future_dims):
            bit = (chunk_group_idx >> bit_pos) & 1
            base |= bit << d
        base |= dim_val << target_dim
        return base

    def make_phase1_index_fn(self, mb_idx):
        def phase1_index_fn(chip_idx):
            c_me = self.get_phase1_chunk_idx(self.topo.cur_id, chip_idx)
            return (self.input_chunk(c_me), mb_idx)

        return phase1_index_fn

    def make_phase1_packed_index_fn(self, mb_idx):
        def phase1_packed_index_fn(chip_idx):
            return (chip_idx, mb_idx)

        return phase1_packed_index_fn

    def make_phase1_in_index_fn_with_recv_sem(self, mb_idx):
        def phase1_in_index_fn_with_recv_sem(grid_indices, ref):
            (chip_idx,) = grid_indices
            return (
                self.get_slice(
                    chip_idx, mb_idx * self.config.mb_size, self.config.mb_size
                ),
                self.config.mb_size,
            )

        return phase1_in_index_fn_with_recv_sem

    def make_phase2_in_index_fn_with_recv_sem(self, step_idx, mb_idx):
        def phase2_in_index_fn_with_recv_sem(grid_indices, ref):
            chunk_group_idx, hcube_dim_idx = grid_indices
            my_chunk_idx = self.get_phase2_chunk_idx(
                self.topo.cur_id, step_idx, chunk_group_idx, hcube_dim_idx
            )
            chunk_slice = self.get_packed_slice(
                my_chunk_idx,
                mb_idx * self.config.mb_size
                + hcube_dim_idx * self.config.hc_chunk_size,
                self.config.hc_chunk_size,
            )
            return chunk_slice, self.config.hc_chunk_size

        return phase2_in_index_fn_with_recv_sem

    def make_phase2_index_fn(self, step_idx, mb_idx):
        def phase2_index_fn(chunk_group_idx, hcube_dim_idx):
            my_chunk_idx = self.get_phase2_chunk_idx(
                self.topo.cur_id, step_idx, chunk_group_idx, hcube_dim_idx
            )
            mb_start_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (self.pack(my_chunk_idx), mb_start_idx)

        return phase2_index_fn

    def last_step_landing_row(self, chip_id, hcube_dim_idx):
        """Packed row chunk that column half `hcube_dim_idx` of the last
        hypercube step lands in, on chip `chip_id`.

        The last step's chunk would otherwise land on one of the slots step 0
        uses, and nothing stops a sender whose own step 0 is done from writing
        it while step-0 data from a late chip is still due there. Step 0 only
        fills rows whose bit along dim `hcube_dim_idx` equals the chip's own,
        so flipping that bit names a row step 0 never touches: the fix costs
        no extra memory. Two hypercube steps only (see hier_rs_kernel).
        """
        return chip_id ^ (1 << hcube_dim_idx)

    def make_phase2_last_step_index_fn(self, mb_idx):
        def phase2_last_step_index_fn(chunk_group_idx, hcube_dim_idx):
            row = self.last_step_landing_row(self.topo.cur_chip_id, hcube_dim_idx)
            mb_start_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (row, mb_start_idx)

        return phase2_last_step_index_fn

    def make_phase2_last_step_in_index_fn_with_recv_sem(self, mb_idx):
        def phase2_last_step_in_index_fn_with_recv_sem(grid_indices, ref):
            _, hcube_dim_idx = grid_indices
            row = self.last_step_landing_row(self.topo.cur_chip_id, hcube_dim_idx)
            chunk_slice = self.get_slice(
                row,
                mb_idx * self.config.mb_size
                + hcube_dim_idx * self.config.hc_chunk_size,
                self.config.hc_chunk_size,
            )
            return chunk_slice, self.config.hc_chunk_size

        return phase2_last_step_in_index_fn_with_recv_sem

    def make_phase2_out_index_fn(self, step_idx, mb_idx):
        def phase2_out_index_fn(chunk_group_idx, hcube_dim_idx):
            mb_start_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (0, mb_start_idx)

        return phase2_out_index_fn
