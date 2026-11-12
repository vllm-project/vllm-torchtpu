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
# The work buffers persist across calls. At one micro-batch the phase-2 copy
# can read running_sum_ref before the accumulate's write to it lands, and so
# send the previous call's sums. A second micro-batch hides this by timing;
# it shows only with a fresh input per call.
_MIN_SAFE_MICRO_BATCHES = 2
_MIN_LANES_PER_HC_CHUNK = 512
_LANES = 128


def splits_hidden_evenly(
    hidden_dim_size: int, num_micro_batches: int, num_devices: int
) -> bool:
    """Whether every micro-batch splits into one lane-aligned phase-2 chunk per
    hypercube dim. If not, a sender trims a chunk that its receiver waits for
    in full, and every device deadlocks."""
    num_hcube_dims = int(math.log2(num_devices // 2))
    return hidden_dim_size % (num_micro_batches * num_hcube_dims * _LANES) == 0


def pick_num_micro_batches(
    local_seq_len: int, hidden_dim_size: int, itemsize: int, *, num_devices: int
) -> int:
    stages = (local_seq_len * hidden_dim_size * itemsize) // _MB_STAGE_TARGET_BYTES
    mb = 1 << max(0, stages.bit_length() - 1)
    mb = min(mb, max(1, hidden_dim_size // _MIN_LANES_PER_HC_CHUNK))
    mb = max(mb, _MIN_SAFE_MICRO_BATCHES)
    mb = int(min(max(mb, 1), _MAX_MICRO_BATCHES))
    while mb > _MIN_SAFE_MICRO_BATCHES and not splits_hidden_evenly(
        hidden_dim_size, mb, num_devices
    ):
        mb //= 2
    return mb


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Config:
    """Static shape of one micro-batched reduce-scatter call."""

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
        assert splits_hidden_evenly(
            self.hidden_dim_size, self.num_micro_batches, self.num_devices
        ), (
            f"hidden_dim_size={self.hidden_dim_size} does not split into "
            f"{self.num_micro_batches} micro-batches of {self.num_hcube_dims} "
            f"{_LANES}-lane chunks"
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
            num_devices=self.num_devices,
        )

    @property
    def seq_chunk_size(self) -> int:
        return self.local_seq_len // self.num_devices

    @property
    def mb_size(self) -> int:
        return self.hidden_dim_size // self.num_micro_batches

    @property
    def hc_chunk_size(self) -> int:
        return self.mb_size // self.num_hcube_dims

    @property
    def packing_factor(self) -> int:
        return 4 // jnp.dtype(self.dtype).itemsize
