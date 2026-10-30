# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The prefill stage's step sizes: the rows per core a step pads to."""

from types import SimpleNamespace

from vllm_torchtpu.core.pp_chunks import chunk_buckets, chunk_granularity
from vllm_torchtpu.megakernel.qwen38.stage import QwenPrefillStage


def _stage(max_tokens=32768):
    stage = object.__new__(QwenPrefillStage)
    stage.config = SimpleNamespace(max_num_batched_tokens=max_tokens)
    stage.cores, stage.chunk, stage.granule = 8, 128, 256
    stage._sizes = None
    return stage


def test_without_scheduler_sizes_steps_pad_to_powers_of_two():
    stage = _stage()
    assert stage.buckets() == [256, 512, 1024, 2048, 4096]
    assert stage._rows_for(17920) == 4096


def test_steps_pad_to_the_scheduler_buckets_in_whole_ring_rounds():
    stage = _stage()
    granule = chunk_granularity(640, 32768, True)
    stage.size_steps([*chunk_buckets(32768, granule), 32768])
    # Powers of two below the first bucket (short steps), then each bucket in
    # whole rounds of the MoE prologue's ring (512 rows).
    assert stage.buckets() == [256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    assert stage._rows_for(1000) == 256
    assert stage._rows_for(4480) == 1024
    assert stage._rows_for(17920) == 2560
    assert stage._rows_for(32768) == 4096
