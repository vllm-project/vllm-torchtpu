# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_REASONING_COUNT_FAST: same reasoning-token counts as vLLM's
re-parse, without the delta and tool-call rebuild.

Needs a Qwen3-family tokenizer (``<think>``, ``<tool_call>`` added tokens):
REASONING_COUNT_TEST_TOKENIZER names a local directory or a hub id.
"""

import os
import time

import pytest

TOKENIZER = os.environ.get("REASONING_COUNT_TEST_TOKENIZER", "Qwen/Qwen3-0.6B")

TOOL = (
    "<tool_call>\n<function=Bash>\n<parameter=command>\nls -la /tmp > out.txt 2>&1\n"
    "</parameter>\n<parameter=description>\nlist files\n</parameter>\n</function>\n"
    "</tool_call>"
)
DENSE_BODY = "<tool_call>\n<function=Write>\n<parameter=content>\n" + (
    '<li class="a">item</li>\n' * 600
)
OUTPUTS = {
    "prose": "The scheduler admits a request when its blocks fit. " * 40,
    "reasoning_then_content": "<think>\nplan the answer\n</think>\n\nThe answer is 4.",
    "reasoning_then_tool": "<think>\nI should list the files.\n</think>\n\n" + TOOL,
    "tool_inside_reasoning": "I should list the files first.\n" + TOOL,
    "content_then_think": "Plain content. <think>late reasoning</think> more content",
    "tool_then_text": TOOL + "\nDone, the files are listed.",
    "dense_open_tool_call": DENSE_BODY,
    "dense_closed_tool_call": DENSE_BODY + "</parameter>\n</function>\n</tool_call>",
}
TEMPLATE_KWARGS = [{"enable_thinking": False}, {"enable_thinking": True}, None]


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("vllm")
    from vllm.tokenizers import get_tokenizer

    try:
        return get_tokenizer(TOKENIZER)
    except Exception as e:  # no network / no cache
        pytest.skip(f"{TOKENIZER} tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def parser_cls():
    from vllm.parser.parser_manager import ParserManager

    # The decode's flags: --tool-call-parser qwen3_coder --reasoning-parser
    # qwen3, without --enable-auto-tool-choice.
    return ParserManager.get_parser(
        tool_parser_name="qwen3_coder",
        reasoning_parser_name="qwen3",
        enable_auto_tools=False,
        model_name="Qwen/Qwen3.8-Flash-Next-FP8",
    )


def _adapter(parser_cls, tokenizer, kwargs):
    parser = parser_cls(tokenizer, None, chat_template_kwargs=kwargs, model_config=None)
    return parser._reasoning_parser


def _orig():
    from vllm.parser.engine.adapters import ParserEngineReasoningAdapter

    fn = ParserEngineReasoningAdapter.count_reasoning_tokens
    return getattr(fn, "__wrapped__", fn)


@pytest.mark.parametrize(
    "kwargs", TEMPLATE_KWARGS, ids=["think_off", "think_on", "none"]
)
def test_counts_match_reparse(tokenizer, parser_cls, kwargs):
    from vllm_torchtpu.reasoning_count import count_reasoning_tokens_fast

    orig = _orig()
    counts = {}
    for name, text in OUTPUTS.items():
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        want = orig(_adapter(parser_cls, tokenizer, kwargs), ids)
        adapter = _adapter(parser_cls, tokenizer, kwargs)
        got = count_reasoning_tokens_fast(adapter, ids, orig)
        assert got == want, (name, got, want)
        # The cached counting engine resets between requests.
        assert count_reasoning_tokens_fast(adapter, ids, orig) == want, name
        counts[name] = got
    if kwargs == {"enable_thinking": True}:
        # Not a vacuous 0 == 0: reasoning is counted where the output has it.
        assert counts["reasoning_then_content"] > 0
        assert counts["reasoning_then_tool"] > 0


def test_streamed_reasoning_and_empty_ids_take_vllm_path(tokenizer, parser_cls):
    from vllm_torchtpu.reasoning_count import count_reasoning_tokens_fast

    calls = []

    def orig(adapter, ids):
        calls.append(ids)
        return -1

    adapter = _adapter(parser_cls, tokenizer, {"enable_thinking": True})
    assert count_reasoning_tokens_fast(adapter, (), orig) == -1
    adapter._streaming_count_valid = True
    assert count_reasoning_tokens_fast(adapter, (1, 2, 3), orig) == -1
    assert calls == [(), (1, 2, 3)]


def test_reasoning_reachable():
    pytest.importorskip("vllm")
    from vllm.parser.qwen3 import qwen3_config

    from vllm_torchtpu.reasoning_count import reasoning_reachable

    assert not reasoning_reachable(qwen3_config(thinking=False))
    assert reasoning_reachable(qwen3_config(thinking=True))


def _boom(*args, **kwargs):
    raise AssertionError("work the count does not need")


@pytest.mark.parametrize("thinking", [False, True], ids=["think_off", "think_on"])
def test_dense_tool_call_skips_rebuild(tokenizer, parser_cls, monkeypatch, thinking):
    from vllm.parser.engine.parser_engine import ParserEngine

    from vllm_torchtpu.reasoning_count import count_reasoning_tokens_fast

    text = OUTPUTS["dense_open_tool_call"]
    if thinking:
        text = "<think>\nwrite the page\n</think>\n\n" + text
    ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    kwargs = {"enable_thinking": thinking}
    t0 = time.perf_counter()
    want = _orig()(_adapter(parser_cls, tokenizer, kwargs), ids)
    t_orig = time.perf_counter() - t0

    monkeypatch.setattr(ParserEngine, "_events_to_delta", _boom)
    monkeypatch.setattr(ParserEngine, "_build_extracted_result", _boom)
    if not thinking:
        # No reasoning state is reachable: nothing is decoded or parsed.
        monkeypatch.setattr(ParserEngine, "_feed", _boom)
    adapter = _adapter(parser_cls, tokenizer, kwargs)
    t0 = time.perf_counter()
    got = count_reasoning_tokens_fast(adapter, ids, _orig())
    t_fast = time.perf_counter() - t0
    assert got == want
    if thinking:
        assert got > 0
    print(f"{len(ids)} tokens: re-parse {t_orig:.3f} s, fast {t_fast:.4f} s")


def test_patch_installs_behind_flag(tokenizer, parser_cls, monkeypatch):
    from vllm.parser.engine.adapters import ParserEngineReasoningAdapter

    from vllm_torchtpu import reasoning_count

    orig = ParserEngineReasoningAdapter.count_reasoning_tokens
    monkeypatch.setattr(ParserEngineReasoningAdapter, "count_reasoning_tokens", orig)
    monkeypatch.setenv("VLLM_TPU_REASONING_COUNT_FAST", "0")
    reasoning_count.patch_reasoning_count_fast()
    assert ParserEngineReasoningAdapter.count_reasoning_tokens is orig

    monkeypatch.setenv("VLLM_TPU_REASONING_COUNT_FAST", "1")
    reasoning_count.patch_reasoning_count_fast()
    patched = ParserEngineReasoningAdapter.count_reasoning_tokens
    assert patched is not orig and patched.__wrapped__ is orig
    reasoning_count.patch_reasoning_count_fast()  # idempotent
    assert ParserEngineReasoningAdapter.count_reasoning_tokens is patched

    ids = tuple(
        tokenizer.encode(OUTPUTS["reasoning_then_tool"], add_special_tokens=False)
    )
    kwargs = {"enable_thinking": True}
    assert patched(_adapter(parser_cls, tokenizer, kwargs), ids) == orig(
        _adapter(parser_cls, tokenizer, kwargs), ids
    )
