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
"""Trace-time policies, distinct from shape/layout eligibility contracts.

256-token tiles and eight logical pipeline lanes originate in the shared
RPA decode defaults. They trade MXU utilization against live VMEM and schedule
size; neither is an identity with MXU columns or hardware sublanes. See the
sparse decode guide for the matched single-variable measurements.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs

from .abi import RPA_WAIT_ROW_WORDS


@dataclasses.dataclass(frozen=True)
class DecodePolicy:
    """Immutable compilation choices; never selected by an environment knob."""

    bkv_tokens: int = 256
    pipeline_lanes: int = 8
    schedule_copy_words: int = RPA_WAIT_ROW_WORDS

    @property
    def block(self) -> configs.BlockSizes:
        # One query per task is decode semantics, unlike the tuning choices.
        return configs.BlockSizes(
            bq_sz=1, bq_c_sz=1, bkv_sz=self.bkv_tokens, batch_size=self.pipeline_lanes
        )


DEFAULT_POLICY = DecodePolicy()
DECODE_BLOCK = DEFAULT_POLICY.block


def use_fused_page_metadata(blocks, lengths, table, ratio, page_size):
    """Bound a resident integer page gather by the native vector/VMEM layout."""
    # JAX drops singleton batch dimensions from take_along_axis indices;
    # Mosaic's gather lowering requires them to match the output dimensions.
    if (
        blocks.ndim != 2
        or min(blocks.shape) <= 0
        or blocks.shape[0] == 1
        or table.ndim != 2
        or table.shape[0] != blocks.shape[0]
        or table.shape[1] <= 0
        or lengths.shape != (blocks.shape[0],)
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table))
        or ratio <= 0
        or page_size <= 0
        or page_size % ratio
    ):
        return False
    # TorchTPU traces under an abstract TPU mesh with a CPU JAX host backend.
    device = jax.sharding.get_abstract_mesh().abstract_device
    if (device is None and jax.default_backend() != "tpu") or (
        device is not None and device.platform != "tpu"
    ):
        return False
    info = pltpu.get_tpu_info()
    if table.shape[1] > info.num_lanes:
        return False
    from .planning import VMEM_USABLE_FRACTION

    sublanes = info.get_sublane_tiling(jnp.int32)
    rows = (blocks.shape[0] + sublanes - 1) // sublanes * sublanes
    columns = (blocks.shape[1] + info.num_lanes) // info.num_lanes * info.num_lanes
    # Bound inputs, output, selected groups, logical/physical gathers and masks.
    vector_bytes = (
        rows * (8 * columns + info.num_lanes + 1) * jnp.dtype(jnp.int32).itemsize
    )
    return vector_bytes <= info.vmem_capacity_bytes * VMEM_USABLE_FRACTION
