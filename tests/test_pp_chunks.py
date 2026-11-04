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
from vllm_torchtpu.core.pp_chunks import (
    chunk_buckets,
    chunk_granularity,
    chunk_pairs,
    schedule_pairs,
    step_limit,
)


def test_granularity_is_an_eighth_step_in_whole_blocks_when_aligned():
    assert chunk_granularity(4096, 16384, True) == 4096
    assert chunk_granularity(4352, 16384, True) == 4352
    assert chunk_granularity(16, 16384, True) == 2048
    assert chunk_granularity(1024, 16384, True) == 2048
    assert chunk_granularity(4096, 8192, True) == 4096


def test_granularity_ignores_the_block_without_alignment_or_above_a_step():
    assert chunk_granularity(4352, 16384, False) == 2048
    assert chunk_granularity(1114112, 16384, True) == 2048
    assert chunk_granularity(8192, 4096, True) == 512


def test_buckets_cover_every_granule_multiple():
    assert chunk_buckets(16384, 4096) == [4096, 8192, 12288]
    assert chunk_buckets(16384, 4352) == [4352, 8704, 13056]
    assert chunk_buckets(4096, 512) == [512, 1024, 1536, 2048, 2560, 3072, 3584]
    assert chunk_buckets(4096, 4096) == []


def test_schedule_pairs_sums_chunk_pairs_over_sequences():
    # One sequence of q new tokens behind a prefix is exactly chunk_pairs.
    schedule = (0, 256, 256)
    assert schedule_pairs([16384], [32768 + 16384], 256, 256) == chunk_pairs(
        schedule, 16384, 32768
    )
    # Sequences add up; empty ones contribute nothing.
    assert schedule_pairs([4096, 0, 1], [4096, 100, 8193], 256, 256) == (
        chunk_pairs(schedule, 4096, 0) + chunk_pairs(schedule, 1, 8192)
    )
    assert schedule_pairs([], [], 256, 256) == 0


def test_chunk_pairs_count_the_kernel_schedule_of_a_chunk():
    schedule = (10000, 256, 256)
    # 16 query blocks; block i attends to (i + 1) KV blocks.
    assert chunk_pairs(schedule, 4096, 0) == sum(range(1, 17))
    # Behind 8K tokens each block also attends to the 32 prefix blocks.
    assert chunk_pairs(schedule, 4096, 8192) == sum(range(1, 17)) + 16 * 32
    assert chunk_pairs(schedule, 0, 8192) == 0
    # 16K behind 32K needs 64 * 128 + 2080 pairs of 256x256 tiles.
    assert chunk_pairs((13436, 256, 256), 16384, 32768) == 10272
    # No schedule capacity known.
    assert chunk_pairs(None, 16384, 32768) == 0


def test_step_limit_spreads_pending_work_over_the_stages_in_granules():
    # 32K over 8 stages: 4K steps.
    assert step_limit(32768, 8, 2048, 16384) == 4096
    # 9K over 8 stages is 1152 tokens a stage, rounded up to a granule.
    assert step_limit(9216, 8, 1024, 16384) == 2048
    assert step_limit(40000, 8, 2048, 16384) == 6144


def test_step_limit_floors_at_one_granule():
    assert step_limit(1, 8, 2048, 16384) == 2048
    assert step_limit(8192, 8, 2048, 16384) == 2048


def test_step_limit_is_none_without_pending_work():
    assert step_limit(0, 8, 2048, 16384) is None
    assert step_limit(-5, 8, 2048, 16384) is None


def test_step_limit_is_none_when_the_spread_is_a_full_step():
    assert step_limit(8 * 16384, 8, 2048, 16384) is None
    assert step_limit(4 * 65536, 8, 2048, 16384) is None


def test_step_limit_clamps_to_the_full_step():
    # 14000 tokens a stage round up to 17408 with 4352-token granules, past
    # the 16K step.
    assert step_limit(8 * 14000, 8, 4352, 16384) == 16384
