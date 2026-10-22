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
