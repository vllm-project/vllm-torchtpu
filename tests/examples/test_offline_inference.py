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
"""Unit tests for the chat-template default in examples/offline_inference.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _load_offline_inference():
    path = Path(__file__).resolve().parents[2] / "examples" / "offline_inference.py"
    spec = importlib.util.spec_from_file_location("offline_inference", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


offline_inference = _load_offline_inference()


def _make_fake_llm(tokenizer):
    output = SimpleNamespace(
        prompt="The capital of France is",
        outputs=[SimpleNamespace(text="Paris")],
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                profiler_config=SimpleNamespace(torch_profiler_dir=None)
            )
        ),
        get_default_sampling_params=lambda: SimpleNamespace(
            temperature=0.0, max_tokens=None, top_p=None, top_k=None
        ),
        get_tokenizer=MagicMock(return_value=tokenizer),
        chat=MagicMock(return_value=[output]),
        generate=MagicMock(return_value=[output]),
    )
    return llm


def _base_args(*, use_chat_template=None, chat_template_kwargs=None):
    return {
        "model": "google/gemma-4-31B-it",
        "prompt": ["The capital of France is"],
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "use_chat_template": use_chat_template,
        "chat_template_kwargs": chat_template_kwargs or {},
    }


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--use-chat-template"], True),
        (["--no-use-chat-template"], False),
    ],
    ids=["default", "use-chat-template", "no-use-chat-template"],
)
def test_parser_use_chat_template_flag(monkeypatch, argv, expected):
    # EngineArgs.add_cli_args builds a DeviceConfig, which needs the TPU torch
    # device; only the example's own flags matter here.
    monkeypatch.setattr(offline_inference.EngineArgs, "add_cli_args", lambda _: None)
    args = offline_inference.create_parser().parse_args(argv)
    assert args.use_chat_template is expected


@pytest.mark.parametrize(
    ("use_chat_template", "chat_template_kwargs", "tokenizer", "expect_chat"),
    [
        (None, {}, SimpleNamespace(chat_template="<bos>{{ messages }}"), True),
        (None, {}, SimpleNamespace(chat_template=None), False),
        (None, {}, SimpleNamespace(), False),
        (True, {}, SimpleNamespace(chat_template=None), True),
        (False, {}, SimpleNamespace(chat_template="<bos>{{ messages }}"), False),
        (
            None,
            {"enable_thinking": False},
            SimpleNamespace(chat_template=None),
            True,
        ),
        (
            False,
            {"enable_thinking": False},
            SimpleNamespace(chat_template="<bos>{{ messages }}"),
            False,
        ),
    ],
    ids=[
        "default-with-chat-template",
        "default-without-chat-template",
        "default-missing-chat-template-attr",
        "explicit-use-chat-template",
        "explicit-no-use-chat-template",
        "chat-template-kwargs-safeguard",
        "explicit-no-use-chat-template-overrides-kwargs",
    ],
)
def test_main_selects_chat_or_generate(
    monkeypatch, use_chat_template, chat_template_kwargs, tokenizer, expect_chat
):
    llm = _make_fake_llm(tokenizer)
    monkeypatch.setattr(offline_inference, "LLM", lambda **_: llm)

    offline_inference.main(
        _base_args(
            use_chat_template=use_chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
    )

    if expect_chat:
        llm.chat.assert_called_once()
        llm.generate.assert_not_called()
    else:
        llm.generate.assert_called_once()
        llm.chat.assert_not_called()


@pytest.mark.parametrize(
    ("use_chat_template", "chat_template_kwargs"),
    [
        (True, {}),
        (False, {}),
        (None, {"enable_thinking": False}),
    ],
    ids=["explicit-true", "explicit-false", "kwargs-safeguard"],
)
def test_main_does_not_load_tokenizer_when_mode_already_known(
    monkeypatch, use_chat_template, chat_template_kwargs
):
    llm = _make_fake_llm(SimpleNamespace(chat_template="<bos>{{ messages }}"))
    monkeypatch.setattr(offline_inference, "LLM", lambda **_: llm)

    offline_inference.main(
        _base_args(
            use_chat_template=use_chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
    )

    llm.get_tokenizer.assert_not_called()
