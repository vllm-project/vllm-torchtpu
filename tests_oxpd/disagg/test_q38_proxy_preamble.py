# SPDX-License-Identifier: Apache-2.0
"""PROXY_Q38_PREAMBLE_REPEATS: a shared text preamble for decode-cache accuracy gates."""

import importlib
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "examples/disagg"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
proxy = importlib.import_module("toy_proxy_server")


def test_off_by_default_leaves_the_request_unchanged(monkeypatch):
    monkeypatch.delenv("PROXY_Q38_PREAMBLE_REPEATS", raising=False)
    body = {"prompt": "Q: 2+3?", "max_tokens": 4}
    assert proxy._q38_apply_preamble(body) is body


def test_every_prompt_form_gets_the_same_prefix(monkeypatch):
    monkeypatch.setenv("PROXY_Q38_PREAMBLE_REPEATS", "3")
    text = proxy._q38_apply_preamble({"prompt": ""})["prompt"]
    assert text.count("[Padding block") == 3 and text.endswith("\n")
    assert proxy._q38_apply_preamble({"prompt": "Q"})["prompt"] == text + "Q"
    assert proxy._q38_apply_preamble({"prompt": ["a", "b"]})["prompt"] == [text + "a", text + "b"]
    assert proxy._q38_apply_preamble({"prompt": [1, 2, 3]})["prompt"] == [1, 2, 3]
    chat = proxy._q38_apply_preamble({"messages": [{"role": "user", "content": "hi"}]})
    assert chat["messages"][0] == {"role": "system", "content": text.rstrip("\n")}
    assert chat["messages"][1] == {"role": "user", "content": "hi"}
    sys_chat = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}]}
    out = proxy._q38_apply_preamble(sys_chat)
    assert out["messages"][0]["content"] == text + "Be brief."
    assert sys_chat["messages"][0]["content"] == "Be brief."  # the caller's dict is not mutated
    # Two different requests share the identical preamble prefix.
    a = proxy._q38_apply_preamble({"prompt": "first question"})["prompt"]
    b = proxy._q38_apply_preamble({"prompt": "another one"})["prompt"]
    assert a[: len(text)] == b[: len(text)] == text


def test_one_repeat_is_one_block(monkeypatch):
    monkeypatch.setenv("PROXY_Q38_PREAMBLE_REPEATS", "1")
    out = proxy._q38_apply_preamble({"prompt": "Q"})["prompt"]
    assert out.count("[Padding block") == 1 and out.endswith("Q")


def test_prefill_usage_log_reports_prompt_and_cached_tokens():
    full = {"usage": {"prompt_tokens": 95000, "prompt_tokens_details": {"cached_tokens": 81920}}}
    assert proxy._prefill_usage_log(full) == " prompt_tokens=95000 cached_tokens=81920"
    assert proxy._prefill_usage_log({"usage": {"prompt_tokens": 7}}) == " prompt_tokens=7"
    assert proxy._prefill_usage_log({"usage": {"prompt_tokens": 7, "prompt_tokens_details": None}}) == " prompt_tokens=7"
    assert proxy._prefill_usage_log({}) == ""
    assert proxy._prefill_usage_log([]) == ""
