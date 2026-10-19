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
"""DeepSeek-V4 / V4.1 tokenizers that honor ``continue_final_message``.

vLLM renders DeepSeek-V4 and V4.1 chat by calling ``encode_messages`` instead
of a Jinja template, and builds its encode config from ``thinking_mode``,
``drop_thinking`` and ``reasoning_effort`` alone. ``continue_final_message``
is dropped, so a trailing assistant turn is always rendered with ``eos_token``
and a request that prefills one asks the model to generate past
end-of-sentence.

Both encoders read a per-message ``wo_eos`` flag and have a template for that
case. This module registers tokenizers that set the flag. vLLM resolves
``tokenizer_mode=auto`` to ``deepseek_v4`` for ``DeepseekV4ForCausalLM`` and to
``deepseek_v41`` for ``DeepseekV41ForCausalLM``, so one entry point is
registered per mode.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Mapping, Sequence
from typing import Any

from vllm.tokenizers.protocol import TokenizerLike


def _mark_continued_turn(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with ``wo_eos`` on a trailing assistant turn.

    Setting ``wo_eos`` tells the encoder to omit ``eos_token`` so the model
    continues that turn. Earlier turns and a trailing user turn are left as
    they are.
    """
    marked = [dict(m) for m in messages]
    if marked and marked[-1].get("role") == "assistant":
        marked[-1]["wo_eos"] = True
    return marked


def _tpu_deepseek_v4_tokenizer(tokenizer: Any) -> Any:
    """Return a copy of ``tokenizer`` that honors ``continue_final_message``.

    The concrete class is built per instance because vLLM constructs its own
    DeepSeek-V4 / V4.1 tokenizer classes dynamically from the checkpoint's HF
    backend.
    """
    base_cls = tokenizer.__class__

    class _TpuDeepseekV4Tokenizer(base_cls):
        @functools.wraps(base_cls.apply_chat_template)
        def apply_chat_template(self, messages, *args, **kwargs):
            # vLLM's renderer passes both ``messages`` and ``conversation``.
            # The V4 encoder renders ``conversation`` when it is there and
            # ``messages`` otherwise; the V4.1 encoder renders ``messages``
            # and ignores the kwarg. Mark both so either encoder sees it.
            if kwargs.pop("continue_final_message", False):
                marked_messages = _mark_continued_turn(messages)
                if "conversation" in kwargs:
                    kwargs["conversation"] = _mark_continued_turn(
                        kwargs["conversation"]
                    )
                else:
                    kwargs["conversation"] = marked_messages
                messages = marked_messages
            return super().apply_chat_template(messages, *args, **kwargs)

        def __reduce__(self):
            return _tpu_deepseek_v4_tokenizer, (tokenizer,)

    tpu_tokenizer = copy.copy(tokenizer)
    tpu_tokenizer.__class__ = _TpuDeepseekV4Tokenizer
    return tpu_tokenizer


class TpuDeepseekV4Tokenizer(TokenizerLike):
    """Registry entry point for ``tokenizer_mode=deepseek_v4``."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer

        return _tpu_deepseek_v4_tokenizer(
            DeepseekV4Tokenizer.from_pretrained(*args, **kwargs)
        )


class TpuDeepseekV41Tokenizer(TokenizerLike):
    """Registry entry point for ``tokenizer_mode=deepseek_v41``."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

        return _tpu_deepseek_v4_tokenizer(
            DeepseekV41Tokenizer.from_pretrained(*args, **kwargs)
        )


def register_tokenizers() -> None:
    from vllm_torchtpu import envs

    if not envs.TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE:
        return

    from vllm.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.register("deepseek_v4", __name__, "TpuDeepseekV4Tokenizer")
    TokenizerRegistry.register("deepseek_v41", __name__, "TpuDeepseekV41Tokenizer")
