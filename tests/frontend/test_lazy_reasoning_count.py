# SPDX-License-Identifier: Apache-2.0
"""Lazy reasoning-token counting in chat streaming.

The stream is driven the way ``OpenAIServingChat.chat_completion_stream_generator``
drives it: ``parse_delta`` on each delta, then ``count_reasoning_tokens`` on all
ids generated so far. The parser is vLLM's Qwen3 engine parser behind
``DelegatingParser``, with a prompt that already ends reasoning, so the stock
count re-parses the whole output on every call.
"""

import inspect
import multiprocessing
import types

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.parser.abstract_parser import DelegatingParser
from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.engine.registered_adapters import (
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
)

from vllm_torchtpu.frontend import lazy_reasoning_count

_SPECIAL_TOKENS = {
    "<think>": 200,
    "</think>": 201,
    "<tool_call>": 202,
    "</tool_call>": 203,
    "<|im_start|>": 204,
    "<|im_end|>": 205,
}
_ID_TO_TEXT = {v: k for k, v in _SPECIAL_TOKENS.items()}
_THINK_START = _SPECIAL_TOKENS["<think>"]
_THINK_END = _SPECIAL_TOKENS["</think>"]
_IM_START = _SPECIAL_TOKENS["<|im_start|>"]


class _CharTokenizer:
    """One token per ASCII character, plus the Qwen3 markup tokens."""

    all_special_tokens = list(_SPECIAL_TOKENS)
    all_special_ids = list(_SPECIAL_TOKENS.values())

    def get_vocab(self) -> dict[str, int]:
        return dict(_SPECIAL_TOKENS)

    def encode(self, text: str, **kwargs) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs) -> str:
        return "".join(_ID_TO_TEXT.get(i) or chr(i) for i in ids)


class _Qwen3Parser(DelegatingParser):
    reasoning_parser_cls = Qwen3ParserReasoningAdapter
    tool_parser_cls = Qwen3ParserToolAdapter


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


# An agentic turn whose prompt closes an empty think block.
_PROMPT = (
    [_IM_START]
    + _ids("assistant\n")
    + [_THINK_START]
    + _ids("\n\n")
    + [_THINK_END]
    + _ids("\n\n")
)
# Output with a stray </think> so the stock count is nonzero and changes
# mid-stream.
_OUTPUT = _ids("Let me check the file. " * 20) + [_THINK_END] + _ids("Done. " * 40)


def _request(**stream_options) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options=stream_options or None,
    )


def _stream(request: ChatCompletionRequest, chunk: int = 4) -> list[int]:
    """Returns the count the stream generator stores after every delta."""
    tokenizer = _CharTokenizer()
    parser = _Qwen3Parser(tokenizer)
    generated: list[int] = []
    counts = []
    for start in range(0, len(_OUTPUT), chunk):
        delta = _OUTPUT[start : start + chunk]
        parser.parse_delta(
            delta_text=tokenizer.decode(delta),
            delta_token_ids=delta,
            request=request,
            prompt_token_ids=_PROMPT,
            finished=start + chunk >= len(_OUTPUT),
        )
        generated.extend(delta)
        counts.append(parser.count_reasoning_tokens(tuple(generated)))
    assert not parser._reasoning_parser._streaming_count_valid
    return counts


@pytest.fixture
def stock_vllm(monkeypatch):
    """Stock vLLM methods, restored after the test. Another test in the
    session may already have installed the patch through register_layers."""
    for cls, name in (
        (DelegatingParser, "parse_delta"),
        (DelegatingParser, "count_reasoning_tokens"),
        (OpenAIServingChat, "__init__"),
    ):
        monkeypatch.setattr(cls, name, inspect.unwrap(getattr(cls, name)))
    monkeypatch.setattr(lazy_reasoning_count, "_installed", False)
    monkeypatch.setattr(lazy_reasoning_count, "_force_include_usage", False)
    monkeypatch.delenv("VLLM_TPU_LAZY_REASONING_COUNT", raising=False)


@pytest.fixture
def recounts(monkeypatch):
    """Counts the expensive recounts done by the reasoning adapter."""
    calls = []
    orig = ParserEngineReasoningAdapter.count_reasoning_tokens

    def counting(self, token_ids):
        calls.append(len(token_ids))
        return orig(self, token_ids)

    monkeypatch.setattr(
        ParserEngineReasoningAdapter, "count_reasoning_tokens", counting
    )
    return calls


def test_pinned_vllm_supports_the_patch():
    """Fails when a vLLM bump renames what the patch relies on."""
    assert lazy_reasoning_count.find_unsupported_vllm_api() is None
    parser = _Qwen3Parser(_CharTokenizer())
    assert isinstance(parser._reasoning_parser, ParserEngineReasoningAdapter)
    assert parser._reasoning_parser._streaming_count_valid is False


def test_final_count_matches_stock_and_skips_intermediate(stock_vllm, recounts):
    request = _request(include_usage=True)
    stock = _stream(request)
    assert len(recounts) == len(stock)
    assert stock[-1] > 0

    recounts.clear()
    lazy_reasoning_count.maybe_install_lazy_reasoning_count()
    lazy = _stream(request)

    assert lazy[-1] == stock[-1]
    assert recounts == [len(_OUTPUT)]


@pytest.mark.parametrize("force_include_usage", [False, True])
def test_continuous_usage_counts_every_delta(stock_vllm, recounts, force_include_usage):
    """Per-chunk usage exposes every count: none may be skipped."""
    if force_include_usage:
        request = _request()
        lazy_reasoning_count._force_include_usage = True
    else:
        request = _request(include_usage=True, continuous_usage_stats=True)
    stock = _stream(request)

    recounts.clear()
    lazy_reasoning_count.maybe_install_lazy_reasoning_count()
    assert _stream(request) == stock
    assert len(recounts) == len(stock)


def test_force_include_usage_is_read_from_the_chat_server(stock_vllm, monkeypatch):
    def init(self, *, enable_force_include_usage=False):
        self.enable_force_include_usage = enable_force_include_usage

    monkeypatch.setattr(OpenAIServingChat, "__init__", init)
    lazy_reasoning_count.maybe_install_lazy_reasoning_count()

    OpenAIServingChat(enable_force_include_usage=False)
    assert not lazy_reasoning_count._force_include_usage
    OpenAIServingChat(enable_force_include_usage=True)
    assert lazy_reasoning_count._force_include_usage


def test_env_off_leaves_vllm_unpatched(stock_vllm, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_LAZY_REASONING_COUNT", "0")
    stock_count = DelegatingParser.count_reasoning_tokens

    lazy_reasoning_count.maybe_install_lazy_reasoning_count()

    assert DelegatingParser.count_reasoning_tokens is stock_count
    assert not lazy_reasoning_count._installed


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("MainProcess", True),
        ("ApiServer_0", True),
        ("APIServer_DPRank_1", True),
        ("EngineCore_DP0", False),
    ],
)
def test_api_server_process_names(monkeypatch, name, expected):
    monkeypatch.setattr(
        multiprocessing, "current_process", lambda: types.SimpleNamespace(name=name)
    )
    assert lazy_reasoning_count._is_api_server_process() is expected
