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

# (query shape, KV heads, compression ratio, selected groups, page payload).
# For this FP8 geometry, four lanes retain the 256-token arithmetic while
# halving lane-dependent schedule preparation. Two lanes did not improve
# the eight-live case and were slower with sixteen live requests.
_FP8_POLICIES = {
    ((16, 24, 256), 2, 4, 512, (1664, 1, 4, 256)): DecodePolicy(pipeline_lanes=4),
}


def default_policy_for(
    query: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    *,
    kv_heads: int,
    ratio: int,
    groups: int,
    info: pltpu.TpuInfo,
) -> DecodePolicy:
    """Select measured geometry; the caller still applies resource admission."""
    if (
        info.chip_version != pltpu.ChipVersion.TPU_7X
        or cache.dtype != jnp.float8_e4m3fn
    ):
        return DEFAULT_POLICY
    key = (query.shape, kv_heads, ratio, groups, cache.shape[1:])
    return _FP8_POLICIES.get(key, DEFAULT_POLICY)


_PAGE_METADATA_SHAPES = {((16, 512), (16, 41), 4, 1664)}


def use_fused_page_metadata(blocks, lengths, table, ratio, page_size):
    """Admit the measured compact page table and decode token bucket."""
    if (
        (blocks.shape, table.shape, ratio, page_size) not in _PAGE_METADATA_SHAPES
        or lengths.shape != (blocks.shape[0],)
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table))
    ):
        return False
    return pltpu.get_tpu_info().chip_version == pltpu.ChipVersion.TPU_7X


# Match the compiled BF16 output-projection reduction used by the decode
# baseline. A full-K dot changes partial-sum rounding near BF16 midpoints.
# Key: (padded decode tokens, input width, output width).
OUTPUT_PROJECTION_REDUCTION_BLOCKS = {(16, 6144, 2560): 2048}
