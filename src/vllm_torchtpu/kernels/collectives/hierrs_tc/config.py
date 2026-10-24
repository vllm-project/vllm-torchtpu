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
"""Static configuration of the micro-batched hierarchical reduce-scatter."""

import dataclasses
import math

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


def next_multiple_of(val: int, multiple: int) -> int:
    return ((val + multiple - 1) // multiple) * multiple


_MB_STAGE_TARGET_BYTES = 2 << 20
# Above this, hc_chunk_size shrinks to a single vector width at hidden 4096.
_MAX_MICRO_BATCHES = 8
# At one micro-batch the kernel can return the previous call's result: nothing
# orders an accumulate's write of running_sum_ref before the next phase-2
# start reads it, and a second iteration hides that by timing. Only visible
# with a fresh input per call.
_MIN_SAFE_MICRO_BATCHES = 2
_MIN_LANES_PER_HC_CHUNK = 512


def pick_num_micro_batches(
    local_seq_len: int, hidden_dim_size: int, itemsize: int
) -> int:
    stages = (local_seq_len * hidden_dim_size * itemsize) // _MB_STAGE_TARGET_BYTES
    mb = 1 << max(0, stages.bit_length() - 1)
    mb = min(mb, max(1, hidden_dim_size // _MIN_LANES_PER_HC_CHUNK))
    mb = max(mb, _MIN_SAFE_MICRO_BATCHES)
    return int(min(max(mb, 1), _MAX_MICRO_BATCHES))


def get_capped_bounds(start, length, max_size):
    capped_start = min(start, max_size)
    capped_end = min(start + length, max_size)
    return capped_start, capped_end - capped_start


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Config:
    """Static shape of one micro-batched reduce-scatter call.

    mb_size and hc_chunk_size round up to a vector width; the ragged tail is
    clamped at each slice with get_capped_bounds.
    """

    num_devices: int
    hidden_dim_size: int
    local_seq_len: int  # after padding
    dtype: jnp.dtype
    _num_micro_batches: int | None = None

    def __post_init__(self):
        assert self.cores_per_chip == 2, (
            "This kernel architecture strictly supports 2 cores per chip, but"
            f" found {self.cores_per_chip}."
        )
        assert (self.num_chips & (self.num_chips - 1)) == 0, (
            f"num_chips {self.num_chips} must be a power of 2"
        )
        assert self.num_hcube_dims >= 1, (
            f"num_hcube_dims {self.num_hcube_dims} must be >= 1"
        )
        assert (self.num_micro_batches - 1) * self.mb_size < self.hidden_dim_size, (
            "Unsupported micro-batches config: num_micro_batches="
            f"{self.num_micro_batches} mb_size={self.mb_size} exceeds "
            f"hidden_dim_size={self.hidden_dim_size}"
        )

    @property
    def cores_per_chip(self) -> int:
        return pltpu.get_tpu_info().chip_version.num_physical_tensor_cores_per_chip

    @property
    def num_chips(self) -> int:
        return self.num_devices // self.cores_per_chip

    @property
    def num_hcube_dims(self) -> int:
        return int(math.log2(self.num_chips))

    @property
    def num_micro_batches(self) -> int:
        if self._num_micro_batches is not None:
            return self._num_micro_batches
        return pick_num_micro_batches(
            self.local_seq_len,
            self.hidden_dim_size,
            jnp.dtype(self.dtype).itemsize,
        )

    @property
    def vector_width(self) -> int:
        return pltpu.get_tpu_info().num_lanes

    @property
    def seq_chunk_size(self) -> int:
        return self.local_seq_len // self.num_devices

    @property
    def mb_size(self) -> int:
        return next_multiple_of(
            self.hidden_dim_size // self.num_micro_batches, self.vector_width
        )

    @property
    def hc_chunk_size(self) -> int:
        return next_multiple_of(
            self.mb_size // max(1, self.num_hcube_dims), self.vector_width
        )

    @property
    def packing_factor(self) -> int:
        return 4 // jnp.dtype(self.dtype).itemsize
