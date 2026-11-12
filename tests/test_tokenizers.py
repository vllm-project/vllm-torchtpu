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
"""The DeepSeek-V4 tokenizer wrapper that honors ``continue_final_message``.

vLLM renders DeepSeek-V4 chat through the checkpoint's ``encode_messages``
rather than a Jinja template, and drops ``continue_final_message`` on the way.
A prefill of a trailing assistant turn therefore gets ``eos_token`` appended,
which asks the model to generate past end-of-sentence. The wrapper's whole job
is to translate that flag into the per-message ``wo_eos`` marker the encoder
does read.

The wrapper is built per instance (vLLM constructs the DeepSeek-V4 tokenizer
class dynamically from the checkpoint's HF backend), so these tests stand a
fake base class in for it. That is what the wrapper subclasses in production
too, only with a real tokenizer.
"""

import copy
import pickle
from unittest.mock import MagicMock

import pytest

from vllm_torchtpu.tokenizers import (
    TpuDeepseekV4Tokenizer,
    _tpu_deepseek_v4_tokenizer,
    register_tokenizers,
)


class _FakeBaseTokenizer:
    """Stands in for vLLM's dynamically built DeepSeek-V4 tokenizer.

    Records what ``apply_chat_template`` was ultimately called with, which is
    the only thing the wrapper is supposed to change.
    """

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, tools=None, **kwargs):
        self.seen.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return "rendered"


@pytest.fixture
def base():
    return _FakeBaseTokenizer()


@pytest.fixture
def tokenizer(base):
    return _tpu_deepseek_v4_tokenizer(base)


def _last_call(base):
    assert base.seen, "the wrapper never delegated to the base tokenizer"
    return base.seen[-1]


# ---------------------------------------------------------------------------
# What the wrapper is for.
# ---------------------------------------------------------------------------


def test_trailing_assistant_turn_is_marked_wo_eos(tokenizer, base):
    """The case the module exists for: a prefilled assistant turn is handed
    to the encoder with wo_eos set, so it renders without eos_token and the
    model continues that turn instead of starting a new one."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial answ"},
    ]

    assert (
        tokenizer.apply_chat_template(messages, continue_final_message=True)
        == "rendered"
    )

    conversation = _last_call(base)["kwargs"]["conversation"]
    assert conversation[-1] == {
        "role": "assistant",
        "content": "partial answ",
        "wo_eos": True,
    }
    # Earlier turns end normally; only the one being continued skips eos.
    assert "wo_eos" not in conversation[0]


def test_continue_final_message_is_not_forwarded(tokenizer, base):
    """The base encoder builds its config from thinking_mode/drop_thinking/
    reasoning_effort and does not know this flag; leaving it in kwargs would
    reach the checkpoint's encode_messages as an unexpected argument."""
    tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "x"}], continue_final_message=True
    )

    assert "continue_final_message" not in _last_call(base)["kwargs"]


def test_trailing_user_turn_is_left_alone(tokenizer, base):
    """A normal request ends on the user turn. There is nothing to continue,
    so no message gets marked even though the flag was set."""
    messages = [{"role": "user", "content": "hi"}]

    tokenizer.apply_chat_template(messages, continue_final_message=True)

    conversation = _last_call(base)["kwargs"]["conversation"]
    assert conversation == [{"role": "user", "content": "hi"}]


def test_empty_conversation_does_not_index_out_of_range(tokenizer, base):
    """conversation[-1] is only safe behind the emptiness check. A request
    with no messages must not raise IndexError on its way to the encoder."""
    tokenizer.apply_chat_template([], continue_final_message=True)

    assert _last_call(base)["kwargs"]["conversation"] == []


# ---------------------------------------------------------------------------
# The pass-through path: every ordinary request goes through here.
# ---------------------------------------------------------------------------


def test_without_the_flag_nothing_is_added(tokenizer, base):
    """The path every ordinary request takes. The messages list is forwarded
    by identity and no conversation kwarg is invented, so a caller that never
    asked for continuation sees stock vLLM behaviour."""
    messages = [{"role": "assistant", "content": "done"}]

    tokenizer.apply_chat_template(messages, thinking_mode="off")

    call = _last_call(base)
    assert call["messages"] is messages
    assert call["kwargs"] == {"thinking_mode": "off"}
    # No conversation kwarg is invented when the flag is absent.
    assert "conversation" not in call["kwargs"]


def test_flag_set_false_behaves_like_absent(tokenizer, base):
    """An explicit False must be indistinguishable from omitting the flag.
    kwargs.pop returns False for both, so neither reaches the encoder."""
    tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "done"}], continue_final_message=False
    )

    kwargs = _last_call(base)["kwargs"]
    assert "conversation" not in kwargs
    assert "continue_final_message" not in kwargs


def test_tools_and_other_kwargs_reach_the_base_unchanged(tokenizer, base):
    """The wrapper intercepts one keyword and must be transparent to the
    rest: tools is positional in the super() call, and encoder settings like
    reasoning_effort have to survive untouched."""
    tools = [{"type": "function", "function": {"name": "f"}}]

    tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "x"}],
        tools,
        continue_final_message=True,
        reasoning_effort="high",
    )

    call = _last_call(base)
    assert call["tools"] is tools
    assert call["kwargs"]["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Aliasing. The wrapper edits a conversation the caller still holds.
# ---------------------------------------------------------------------------


def test_the_caller_s_messages_are_not_mutated(tokenizer):
    """wo_eos goes on a copy. vLLM reuses the request's message list after
    rendering, so writing the marker in place would leak a private encoder
    flag into the caller's data and into any later render of it."""
    messages = [{"role": "assistant", "content": "partial"}]
    before = copy.deepcopy(messages)

    tokenizer.apply_chat_template(messages, continue_final_message=True)

    assert messages == before, "wo_eos leaked into the caller's message dict"


def test_an_explicit_conversation_kwarg_wins_over_messages(tokenizer, base):
    """vLLM passes the rendered conversation separately when it has one. The
    marked copy has to be made from that, or the mark lands on a list the base
    tokenizer will not look at."""
    messages = [{"role": "user", "content": "ignored"}]
    conversation = [{"role": "assistant", "content": "the real one"}]

    tokenizer.apply_chat_template(
        messages, conversation=conversation, continue_final_message=True
    )

    sent = _last_call(base)["kwargs"]["conversation"]
    assert sent[-1]["content"] == "the real one"
    assert sent[-1]["wo_eos"] is True
    assert conversation == [{"role": "assistant", "content": "the real one"}]


# ---------------------------------------------------------------------------
# Construction and pickling. Workers are separate processes, so the tokenizer
# is pickled to reach them.
# ---------------------------------------------------------------------------


def test_wrapper_subclasses_the_tokenizer_it_was_given(tokenizer, base):
    """The subclass is built per instance from whatever class the checkpoint
    produced, so it inherits every method the real tokenizer has and only
    overrides apply_chat_template."""
    assert isinstance(tokenizer, _FakeBaseTokenizer)
    assert type(tokenizer) is not type(base)
    assert type(tokenizer).__name__ == "_TpuDeepseekV4Tokenizer"


def test_the_original_tokenizer_is_left_untouched(base):
    """copy.copy then reassigning __class__ marks the copy, not the original.
    The caller may still be holding and using the tokenizer it passed in."""
    _tpu_deepseek_v4_tokenizer(base)

    assert type(base) is _FakeBaseTokenizer


def test_reduce_rebuilds_through_the_factory(tokenizer, base):
    """The wrapper class is created per instance, so it cannot be looked up by
    name at unpickle time. __reduce__ has to name the factory instead."""
    func, args = tokenizer.__reduce__()

    assert func is _tpu_deepseek_v4_tokenizer
    assert args == (base,)


def test_pickle_round_trip_keeps_the_wrapping(tokenizer):
    """vLLM sends the tokenizer to worker processes, so it must pickle. The
    assertion is that the revived object still wraps: a __reduce__ that
    returned the bare tokenizer would pickle cleanly and silently drop the
    fix on every worker."""
    revived = pickle.loads(pickle.dumps(tokenizer))

    assert type(revived).__name__ == "_TpuDeepseekV4Tokenizer"
    revived.apply_chat_template(
        [{"role": "assistant", "content": "x"}], continue_final_message=True
    )
    assert revived.seen[-1]["kwargs"]["conversation"][-1]["wo_eos"] is True


# ---------------------------------------------------------------------------
# Registration, which is off by default.
# ---------------------------------------------------------------------------


def test_registration_is_skipped_unless_the_env_var_is_set(monkeypatch):
    """Off by default, so tokenizer behaviour matches stock vLLM unless a
    config opts in."""
    registry = MagicMock()
    monkeypatch.setattr("vllm.tokenizers.registry.TokenizerRegistry", registry)
    monkeypatch.delenv("TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", raising=False)

    register_tokenizers()

    registry.register.assert_not_called()


def test_registration_binds_the_deepseek_v4_tokenizer_mode(monkeypatch):
    """With the flag on, the entry point is registered under the
    deepseek_v4 tokenizer mode by module path and class name, which is how
    vLLM's registry resolves it in a worker process."""
    registry = MagicMock()
    monkeypatch.setattr("vllm.tokenizers.registry.TokenizerRegistry", registry)
    monkeypatch.setenv("TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", "1")

    register_tokenizers()

    registry.register.assert_called_once_with(
        "deepseek_v4", "vllm_torchtpu.tokenizers", "TpuDeepseekV4Tokenizer"
    )


def test_from_pretrained_returns_a_wrapped_tokenizer(monkeypatch):
    """The registry entry point builds vLLM's tokenizer, then wraps it."""
    inner = _FakeBaseTokenizer()
    deepseek_cls = MagicMock()
    deepseek_cls.from_pretrained.return_value = inner
    monkeypatch.setattr("vllm.tokenizers.deepseek_v4.DeepseekV4Tokenizer", deepseek_cls)

    wrapped = TpuDeepseekV4Tokenizer.from_pretrained("some/model", trust=True)

    deepseek_cls.from_pretrained.assert_called_once_with("some/model", trust=True)
    assert type(wrapped).__name__ == "_TpuDeepseekV4Tokenizer"
    assert isinstance(wrapped, _FakeBaseTokenizer)
