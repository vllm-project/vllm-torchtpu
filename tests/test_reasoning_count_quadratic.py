# SPDX-License-Identifier: Apache-2.0
"""The served Qwen3 parsers' stream-end reasoning count on a long tool call.

Reproduces the API-server freeze (j-2514e030): ``count_reasoning_tokens``
re-parses the whole response with tool parsing on, and every chunk of a tool
call re-converts the call's full argument text, so the cost grows with the
square of the argument length. With VLLM_TPU_REASONING_COUNT_SKIP_TOOLS the
re-parse skips tool parsing: same count, linear cost.

One token per character (vLLM's mock tokenizer); the argument is a multi-line
file, as in a large file write of an agentic trace (each line is a chunk).
"""

import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm.parser.parser_manager")

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter  # noqa: E402
from vllm.parser.parser_manager import ParserManager  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.core import reasoning_count_patch  # noqa: E402

VOCAB = {"<think>": 200, "</think>": 201, "<tool_call>": 202, "</tool_call>": 203}


def tokenizer():
    id_to_text = {v: k for k, v in VOCAB.items()}
    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.get_vocab.return_value = dict(VOCAB)
    tok.decode.side_effect = lambda ids: "".join(
        id_to_text.get(i, chr(i) if i < 128 else f"<{i}>") for i in ids
    )
    tok.all_special_tokens = list(VOCAB)
    tok.all_special_ids = list(VOCAB.values())
    return tok


def response(argument_chars):
    """Token ids: 40 reasoning chars, </think>, one tool call writing a file."""
    line = "<div class=\"row\"><span>{i}</span></div>\n"  # markup: 40 chars per line
    body = (line * (argument_chars // len(line) + 1))[:argument_chars]
    head = "\n<function=write_file>\n<parameter=path>\n/tmp/a.txt\n</parameter>\n<parameter=content>\n"
    tail = "\n</parameter>\n</function>\n"
    ids = [ord(c) for c in "thinking about the file to write it now"]
    ids += [201, 202] + [ord(c) for c in head + body + tail] + [203]
    return ids, 39


def count_seconds(argument_chars):
    parser = ParserManager.get_parser(
        tool_parser_name="qwen3_coder", reasoning_parser_name="qwen3", enable_auto_tools=True
    )(tokenizer())
    ids, expected = response(argument_chars)
    start = time.perf_counter()
    count = parser.count_reasoning_tokens(ids)
    return count, expected, time.perf_counter() - start


@pytest.fixture
def original_method():
    saved = ParserEngineReasoningAdapter.count_reasoning_tokens
    flag = getattr(ParserEngineReasoningAdapter, "_tpu_count_skips_tools", False)
    yield
    ParserEngineReasoningAdapter.count_reasoning_tokens = saved
    ParserEngineReasoningAdapter._tpu_count_skips_tools = flag


def test_tool_reparse_is_quadratic_and_the_patch_is_linear(monkeypatch, original_method):
    sizes = (2000, 8000)
    unpatched = [count_seconds(n) for n in sizes]
    monkeypatch.setattr(envs, "VLLM_TPU_REASONING_COUNT_SKIP_TOOLS", True)
    reasoning_count_patch.patch_reasoning_count_skip_tools()
    patched = [count_seconds(n) for n in sizes]
    for (count, expected, _), (pcount, _, _) in zip(unpatched, patched):
        assert count == expected and pcount == expected
    grow = unpatched[1][2] / unpatched[0][2]
    pgrow = patched[1][2] / patched[0][2]
    print(f"unpatched {[round(t, 3) for *_, t in unpatched]} s (x{grow:.1f} for 4x chars), "
          f"patched {[round(t, 3) for *_, t in patched]} s (x{pgrow:.1f})")
    # 4x the argument: the tool re-parse grows ~16x, the patched count ~4x.
    assert grow > 8
    assert pgrow < 7
    assert patched[1][2] < unpatched[1][2] / 4
