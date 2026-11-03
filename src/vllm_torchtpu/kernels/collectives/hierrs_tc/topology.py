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

import jax
from jax.experimental import pallas as pl

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


class ChunkLocator:
    def __init__(self, config: Config, topo: Topology):
        self.config = config
        self.topo = topo

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
            return (c_me, mb_idx)

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

    def make_phase2_out_index_fn(self, step_idx, mb_idx):
        def phase2_out_index_fn(chunk_group_idx, hcube_dim_idx):
            mb_start_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (0, mb_start_idx)

        return phase2_out_index_fn
