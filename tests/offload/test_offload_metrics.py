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
"""Unit tests for Raiden KV offload per-tier hit-source attribution."""

from __future__ import annotations

from typing import Any

import pytest
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.metrics.cache_hit_source import CacheHitSource

from vllm_torchtpu.offload.metrics import (
    SubBlockSourceCounts,
    compute_load_tokens_by_source,
    split_external_hit,
)

_KEY_0 = make_offload_key(b"key0", 0)
_KEY_1 = make_offload_key(b"key1", 0)


@pytest.mark.parametrize(
    ("end_token", "tokens_per_chunk", "load_key_ranges", "key_sources", "expected"),
    [
        # A 64-token key of four 16-token sub-blocks: 3 HOST + 1 DISK sub-blocks
        # form 2 runs -> 48 HOST + 16 DISK.
        pytest.param(
            64,
            64,
            [(0, 64, _KEY_0)],
            {_KEY_0: ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1))},
            {CacheHitSource.HOST: 48, CacheHitSource.DISK: 16},
            id="one key in two tiers",
        ),
        # The range [0, 48) of a 64-token chunk excludes its last sub-block,
        # [48, 64).
        pytest.param(
            48,
            64,
            [(0, 48, _KEY_0)],
            {
                _KEY_0: (
                    (CacheHitSource.HOST, 2),
                    (CacheHitSource.P2P, 1),
                    (CacheHitSource.DISK, 1),
                )
            },
            {CacheHitSource.HOST: 32, CacheHitSource.P2P: 16},
            id="partial tail",
        ),
        # 8192-token chunks of 512 sub-blocks: a uniform HOST key, and a key
        # of 384 HOST + 128 DISK sub-blocks in 2 runs.
        pytest.param(
            16384,
            8192,
            [(0, 8192, _KEY_0), (8192, 16384, _KEY_1)],
            {
                _KEY_0: ((CacheHitSource.HOST, 512),),
                _KEY_1: ((CacheHitSource.HOST, 384), (CacheHitSource.DISK, 128)),
            },
            {CacheHitSource.HOST: 8192 + 6144, CacheHitSource.DISK: 2048},
            id="512 sub-blocks per key",
        ),
    ],
)
def test_compute_load_tokens_by_source_splits_mixed_key_by_sub_block(
    end_token: int,
    tokens_per_chunk: int,
    load_key_ranges: list[tuple[int, int, OffloadKey]],
    key_sources: dict[OffloadKey, SubBlockSourceCounts],
    expected: dict[CacheHitSource, int],
) -> None:
    assert (
        compute_load_tokens_by_source(
            start_token=0,
            end_token=end_token,
            load_key_ranges=load_key_ranges,
            key_sources=key_sources,
            tokens_per_chunk_by_group={0: tokens_per_chunk},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mamba_sources", "expected"),
    [
        pytest.param(
            ((CacheHitSource.HOST, 4),),
            {CacheHitSource.HOST: 24, CacheHitSource.P2P: 8},
            id="recurrent state in host DRAM",
        ),
        # Mamba's indivisible checkpoint has one byte slice in DISK, so the
        # whole prefix [0, 32) depends on that DISK checkpoint.
        pytest.param(
            ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1)),
            {CacheHitSource.DISK: 32},
            id="recurrent state partly on disk",
        ),
    ],
)
def test_compute_load_tokens_by_source_hybrid_overlap_uses_slowest_tier(
    mamba_sources: SubBlockSourceCounts,
    expected: dict[CacheHitSource, int],
) -> None:
    # Hybrid Attention (group 0) + Mamba (group 1) over [0, 32):
    # k_a0=(HOST*4), k_a1=(HOST*2, P2P*2), and k_m1 covers the whole prefix.
    k_a0 = make_offload_key(b"a0", 0)
    k_a1 = make_offload_key(b"a1", 0)
    k_m1 = make_offload_key(b"m1", 1)
    assert (
        compute_load_tokens_by_source(
            start_token=0,
            end_token=32,
            load_key_ranges=[(0, 16, k_a0), (16, 32, k_a1), (0, 32, k_m1)],
            key_sources={
                k_a0: ((CacheHitSource.HOST, 4),),
                k_a1: ((CacheHitSource.HOST, 2), (CacheHitSource.P2P, 2)),
                k_m1: mamba_sources,
            },
            tokens_per_chunk_by_group={0: 16, 1: 16},
            mamba_group_ids={1},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("start_token", "end_token", "key_range", "expected"),
    [
        # The HBM hit ends inside the chunk, so the load starts mid-chunk.
        (16, 64, (16, 64), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
        (16, 64, (0, 64), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
        # The hit starts and ends inside sub-blocks.
        (8, 40, (0, 64), {CacheHitSource.HOST: 24, CacheHitSource.DISK: 8}),
        (40, 64, (0, 64), {CacheHitSource.DISK: 24}),
        # The second chunk of a prompt: tokens [64, 128).
        (80, 128, (64, 128), {CacheHitSource.HOST: 16, CacheHitSource.DISK: 32}),
    ],
)
def test_compute_load_tokens_by_source_clips_mixed_key_to_hit_range(
    start_token: int,
    end_token: int,
    key_range: tuple[int, int],
    expected: dict[CacheHitSource, int],
) -> None:
    # A 64-token key with four 16-token sub-blocks: two in host DRAM, two on disk.
    key = make_offload_key(b"mixed", 0)
    assert (
        compute_load_tokens_by_source(
            start_token=start_token,
            end_token=end_token,
            load_key_ranges=[(*key_range, key)],
            key_sources={key: ((CacheHitSource.HOST, 2), (CacheHitSource.DISK, 2))},
            tokens_per_chunk_by_group={0: 64},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("tokens_per_chunk_by_group", "mamba_group_ids", "num_warnings"),
    [
        pytest.param({}, (), 1, id="group without a chunk size"),
        pytest.param({0: 62}, (), 1, id="chunk size not divisible by sub-blocks"),
        pytest.param({0: 64}, (0,), 0, id="mamba checkpoint"),
    ],
)
def test_compute_load_tokens_by_source_unsplit_key_uses_slowest_tier(
    tokens_per_chunk_by_group: dict[int, int],
    mamba_group_ids: tuple[int, ...],
    num_warnings: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "vllm_torchtpu.offload.metrics.logger.warning_once",
        lambda *args: warnings.append(args),
    )
    key = make_offload_key(b"mixed", 0)
    assert compute_load_tokens_by_source(
        start_token=0,
        end_token=64,
        load_key_ranges=[(0, 64, key)],
        key_sources={key: ((CacheHitSource.HOST, 3), (CacheHitSource.DISK, 1))},
        tokens_per_chunk_by_group=tokens_per_chunk_by_group,
        mamba_group_ids=mamba_group_ids,
    ) == {CacheHitSource.DISK: 64}
    assert len(warnings) == num_warnings


def test_split_external_hit_sums_to_hit_count() -> None:
    assert split_external_hit({CacheHitSource.HOST: 32}, 48) == {
        CacheHitSource.HOST: 32,
        CacheHitSource.EXTERNAL_UNSPECIFIED: 16,
    }
    assert split_external_hit(
        {CacheHitSource.DISK: 8, CacheHitSource.EXTERNAL_UNSPECIFIED: 8}, 32
    ) == {CacheHitSource.DISK: 8, CacheHitSource.EXTERNAL_UNSPECIFIED: 24}
    assert split_external_hit({CacheHitSource.P2P: 16}, 16) == {CacheHitSource.P2P: 16}
    assert split_external_hit({}, 16) == {CacheHitSource.EXTERNAL_UNSPECIFIED: 16}
