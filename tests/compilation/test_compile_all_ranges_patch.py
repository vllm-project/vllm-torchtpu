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
"""The order in which TPU_PARALLEL_PRECOMPILE compiles a subgraph's ranges.

Each local rank starts on a different range, and the ranks of a host share
what they compile through the Tier-2 cache. The biggest range takes far longer
than the others, so startup is bounded by when it starts: local rank 0 must
start on it, not reach it after the smallest one.
"""

from types import SimpleNamespace

import pytest
from vllm.compilation import piecewise_backend
from vllm.config.utils import Range

from vllm_torchtpu import _patch_vllm_compile_all_ranges

SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.fixture
def compile_all_ranges(monkeypatch):
    monkeypatch.setenv("TPU_PARALLEL_PRECOMPILE", "1")
    monkeypatch.setattr(
        piecewise_backend, "create_concrete_args", lambda graph, size: [size]
    )
    # Restored on teardown, so the patch does not leak into other tests.
    monkeypatch.setattr(
        piecewise_backend.PiecewiseBackend,
        "compile_all_ranges",
        piecewise_backend.PiecewiseBackend.compile_all_ranges,
    )
    _patch_vllm_compile_all_ranges()
    return piecewise_backend.PiecewiseBackend.compile_all_ranges


def _compile_order(compile_all_ranges, monkeypatch, local_rank):
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    order = []

    def compile(graph, args, *rest, compile_range, **kwargs):
        order.append(compile_range.end)
        return lambda *a: a

    # In vLLM's order: smallest range first.
    backend = SimpleNamespace(
        graph=object(),
        range_entries={
            Range(s, s): piecewise_backend.RangeEntry(compile_range=Range(s, s))
            for s in SIZES
        },
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(compile=compile),
            inductor_config={},
            is_encoder=False,
        ),
        compilation_config=None,
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        _log_compile_start=lambda compile_range: None,
    )
    compile_all_ranges(backend)
    return order


def test_local_rank_0_starts_on_the_biggest_range(compile_all_ranges, monkeypatch):
    assert _compile_order(compile_all_ranges, monkeypatch, 0)[0] == 4096


def test_each_local_rank_starts_on_a_different_range(compile_all_ranges, monkeypatch):
    firsts = [
        _compile_order(compile_all_ranges, monkeypatch, rank)[0] for rank in range(8)
    ]
    assert firsts == [4096, 2048, 1024, 512, 256, 128, 64, 32]


def test_every_rank_compiles_every_range(compile_all_ranges, monkeypatch):
    for rank in range(8):
        order = _compile_order(compile_all_ranges, monkeypatch, rank)
        assert sorted(order) == SIZES


def test_biggest_first_off_keeps_vllm_order(compile_all_ranges, monkeypatch):
    monkeypatch.setenv("TPU_PRECOMPILE_BIGGEST_FIRST", "0")
    firsts = [
        _compile_order(compile_all_ranges, monkeypatch, rank)[0] for rank in range(8)
    ]
    # vLLM's order: local rank 0 starts on the smallest range.
    assert firsts == SIZES[:8]
    for rank in range(8):
        order = _compile_order(compile_all_ranges, monkeypatch, rank)
        assert sorted(order) == SIZES
