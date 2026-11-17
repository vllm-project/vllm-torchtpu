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
"""tpu_frontend.lazy_reasoning_count, installed on vLLM's real DelegatingParser.

vLLM's parse_delta and count_reasoning_tokens are replaced by recorders before
the patch wraps them, so a test sees which counts the patch lets through. The
parser engine pieces the zero shortcut inspects are real vLLM classes; their
state machine is a stand-in config built from vLLM's own enums.
"""

from types import SimpleNamespace

import pytest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import ParserState, Transition
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine

from vllm_torchtpu import frontend_fix

REASONING = EventType.REASONING_CHUNK


def _bare(cls):
    """An instance of ``cls`` (or of a subclass with nothing abstract left)
    without running its __init__."""
    if getattr(cls, "__abstractmethods__", None):
        cls = type(cls.__name__, (cls,), {"__abstractmethods__": frozenset()})
    return object.__new__(cls)


def _config(transitions=(), content_events=None):
    """A parser engine config: ``transitions`` are (source state, events,
    next state) triples, read the way StreamingParserEngine reads them."""
    return SimpleNamespace(
        initial_state=ParserState.CONTENT,
        transitions={
            (src, f"term{i}"): Transition(next_state=dst, events=events)
            for i, (src, events, dst) in enumerate(transitions)
        },
        content_events=content_events or {},
    )


def _parser(config, *, streaming_count_valid=False, engine=None):
    """A DelegatingParser whose reasoning parser is a ParserEngine adapter."""
    parser_engine = _bare(ParserEngine)
    parser_engine._engine = _bare(StreamingParserEngine) if engine is None else engine
    parser_engine.parser_engine_config = config
    adapter = _bare(ParserEngineReasoningAdapter)
    adapter._parser_engine = parser_engine
    adapter._streaming_count_valid = streaming_count_valid
    return SimpleNamespace(_reasoning_parser=adapter)


# Reachable from CONTENT: a transition into a state whose content is reasoning.
CAN_REASON = _config(
    transitions=[(ParserState.CONTENT, (), ParserState.REASONING)],
    content_events={ParserState.REASONING: REASONING},
)
# REASONING maps to reasoning content, but no transition ever reaches it.
NEVER_REASONS = _config(content_events={ParserState.REASONING: REASONING})


@pytest.fixture
def counts(monkeypatch):
    """Install the patch over recorders; returns the token-id lengths vLLM's
    own count was called with. vLLM's count is half the ids, so a stale or
    skipped value is visible."""
    calls = []

    def count(self, token_ids):
        calls.append(len(token_ids))
        return len(token_ids) // 2

    monkeypatch.setattr(DelegatingParser, "parse_delta", lambda self, *a, **k: None)
    monkeypatch.setattr(DelegatingParser, "count_reasoning_tokens", count)
    monkeypatch.setattr(frontend_fix, "_INSTALLED", False)
    monkeypatch.setattr(frontend_fix, "_ZERO_CACHE", {})
    assert frontend_fix.maybe_install_lazy_reasoning_count()
    return calls


def _stream(parser, chunks, continuous_usage, *, request_positional=False):
    """Feed ``chunks`` token ids one per chunk and count after each, as the
    serving loop does; returns the count after every chunk."""
    request = SimpleNamespace(
        stream_options=SimpleNamespace(continuous_usage_stats=continuous_usage)
    )
    ids, seen = [], []
    for i in range(chunks):
        ids.append(i)
        finished = i == chunks - 1
        if request_positional:
            DelegatingParser.parse_delta(parser, "t", [i], request, finished=finished)
        else:
            DelegatingParser.parse_delta(
                parser, "t", [i], request=request, finished=finished
            )
        seen.append(DelegatingParser.count_reasoning_tokens(parser, tuple(ids)))
    return seen


@pytest.mark.parametrize("request_positional", [False, True])
def test_intermediate_counts_are_skipped_without_continuous_usage(
    counts, request_positional
):
    parser = _parser(CAN_REASON)
    seen = _stream(parser, 6, False, request_positional=request_positional)
    # Only the final chunk is counted, and its count is vLLM's own.
    assert counts == [6]
    assert seen == [0, 0, 0, 0, 0, 3]


def test_continuous_usage_counts_every_chunk(counts):
    seen = _stream(_parser(CAN_REASON), 6, True)
    assert counts == [1, 2, 3, 4, 5, 6]
    assert seen == [0, 1, 1, 2, 2, 3]


def test_a_valid_streaming_count_is_always_read(counts):
    # The adapter's own count is an attribute read then; nothing to skip.
    seen = _stream(_parser(CAN_REASON, streaming_count_valid=True), 4, False)
    assert counts == [1, 2, 3, 4]
    assert seen == [0, 1, 1, 2]


def test_a_config_that_never_reasons_counts_zero_without_parsing(counts):
    assert _stream(_parser(NEVER_REASONS), 4, True) == [0, 0, 0, 0]
    assert counts == []


@pytest.mark.parametrize(
    "config",
    [
        CAN_REASON,
        # A reachable transition that itself emits a reasoning chunk.
        _config(transitions=[(ParserState.CONTENT, (REASONING,), ParserState.CONTENT)]),
    ],
)
def test_a_config_that_can_reason_is_never_counted_as_zero(counts, config):
    assert not frontend_fix._config_never_reasons(config)
    assert _stream(_parser(config), 4, True) == [0, 1, 1, 2]
    assert counts == [1, 2, 3, 4]


def test_the_reachability_walk_follows_transitions():
    assert frontend_fix._config_never_reasons(NEVER_REASONS)
    # Reasoning content two transitions away is still reachable.
    two_hops = _config(
        transitions=[
            (ParserState.CONTENT, (), ParserState.TOOL_PREAMBLE),
            (ParserState.TOOL_PREAMBLE, (), ParserState.REASONING),
        ],
        content_events={ParserState.REASONING: REASONING},
    )
    assert not frontend_fix._config_never_reasons(two_hops)


# Not vLLM's engine: a class that only shares its name, and a subclass.
_NAMESAKE = type("StreamingParserEngine", (), {})
_SUBCLASS = type("SubEngine", (StreamingParserEngine,), {})


@pytest.mark.parametrize(
    "engine_cls", [_NAMESAKE, _SUBCLASS], ids=["namesake", "subclass"]
)
def test_only_vllms_streaming_engine_gets_the_zero_shortcut(counts, engine_cls):
    """The reachability proof models vLLM's StreamingParserEngine exactly; any
    other engine is counted normally, whatever its class is called."""
    seen = _stream(_parser(NEVER_REASONS, engine=_bare(engine_cls)), 2, True)
    assert counts == [1, 2]
    assert seen == [0, 1]
