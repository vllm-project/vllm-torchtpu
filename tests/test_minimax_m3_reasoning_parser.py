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
"""The MiniMax-M3 reasoning parser whose marker searches run in C.

It must answer exactly as vLLM's parser does, so every test compares the two
on the same inputs, including markers that straddle a search window edge.
"""

import random

import pytest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import (
    MiniMaxM3ReasoningParser as UpstreamParser,
)

from vllm_torchtpu import minimax_m3_reasoning_parser
from vllm_torchtpu.minimax_m3_reasoning_parser import (
    MiniMaxM3ReasoningParser,
    find_token_sequence,
)
from vllm_torchtpu.reasoning_parsers import register_reasoning_parsers

_START, _END = 7, 9


def _random_ids(rng: random.Random, n: int) -> list[int]:
    # A 10-id vocabulary puts many partial and full marker matches in a row.
    return [rng.randrange(10) for _ in range(n)]


@pytest.mark.parametrize("seed", range(200))
def test_searches_match_upstream(seed):
    rng = random.Random(seed)
    ids = _random_ids(rng, rng.choice([0, 1, 3, 50, 4096, 4097, 9000]))
    marker = _random_ids(rng, rng.randrange(0, 4))
    for seq in (ids, tuple(ids)):
        assert MiniMaxM3ReasoningParser._rfind_token_sequence(
            seq, marker
        ) == UpstreamParser._rfind_token_sequence(seq, marker)
        assert MiniMaxM3ReasoningParser._contains_token_sequence(
            seq, marker
        ) == UpstreamParser._contains_token_sequence(seq, marker)


@pytest.mark.parametrize("window", [1, 2, 5])
def test_marker_across_window_edge(monkeypatch, window):
    monkeypatch.setattr(minimax_m3_reasoning_parser, "_RFIND_WINDOW", window)
    ids = [0] * 11
    ids[4:7] = [1, 2, 3]
    for last in (False, True):
        assert find_token_sequence(ids, [1, 2, 3], last=last) == 4
    ids[8:11] = [1, 2, 3]
    assert find_token_sequence(ids, [1, 2, 3], last=True) == 8
    assert find_token_sequence(ids, [1, 2, 3], last=False) == 4
    assert find_token_sequence(ids, [3, 1], last=True) == -1


def _parser(cls):
    parser = cls.__new__(cls)
    parser._start_token_ids = (_START,)
    parser._end_token_ids = (_END,)
    return parser


@pytest.mark.parametrize("seed", range(50))
def test_is_reasoning_end_matches_upstream(seed):
    rng = random.Random(seed)
    # A served prompt: <mm:think> in the system prompt, </mm:think> per turn.
    ids = [rng.randrange(_START) for _ in range(rng.randrange(1, 20000))]
    for _ in range(rng.randrange(4)):
        ids[rng.randrange(len(ids))] = rng.choice([_START, _END])
    assert _parser(MiniMaxM3ReasoningParser).is_reasoning_end(ids) == _parser(
        UpstreamParser
    ).is_reasoning_end(ids)


def test_registered_as_minimax_m3(monkeypatch):
    monkeypatch.setattr(ReasoningParserManager, "reasoning_parsers", {})
    monkeypatch.setattr(
        ReasoningParserManager,
        "lazy_parsers",
        dict(ReasoningParserManager.lazy_parsers),
    )
    register_reasoning_parsers()
    assert (
        ReasoningParserManager.get_reasoning_parser("minimax_m3")
        is MiniMaxM3ReasoningParser
    )
