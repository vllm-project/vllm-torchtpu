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
"""Prime the incremental detokenizer with a short prompt tail.

vLLM's ``FastIncrementalDetokenizer`` primes a ``tokenizers`` ``DecodeStream``
with the whole prompt, and the stream's first ``step`` decodes all of it on the
API server's event loop: about 0.4 ms per 1k prompt tokens before the first
token streams. The prompt only serves as left context for that step.

A byte-level decoder's text is ``from_utf8_lossy`` of the concatenated token
bytes, so left context matters only through UTF-8 synchronization. When the
last prompt tokens decode to non-empty text that does not end in U+FFFD, they
end on a character boundary, and every later step emits the same text from
them as from the whole prompt. ``FastIncrementalDetokenizer`` looks
``DecodeStream`` up on ``tokenizers.decoders`` when it builds a stream so that
backends can replace it; :func:`install` puts :class:`PromptTailDecodeStream`
there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tokenizers.decoders

# Prompt tokens kept as left context when they end on a character boundary.
_TAIL_TOKENS = 8


class PromptTailDecodeStream:
    """``DecodeStream`` that keeps only a prompt tail which decodes the same.

    The stream is built at the first ``step``, which is the first call that
    sees the tokenizer.
    """

    _decode_stream: Any = None  # the class this one replaces, set by install()

    def __init__(
        self, ids: Sequence[int] | None = None, skip_special_tokens: bool = False
    ):
        self._ids = ids
        self._skip_special_tokens = skip_special_tokens
        self._stream = None

    def step(self, tokenizer: Any, id: Any) -> str | None:
        if self._stream is None:
            ids = self._ids
            # getattr: a replacement backend's tokenizer may have no decoder.
            decoder = getattr(tokenizer, "decoder", None)
            if ids is not None and isinstance(decoder, tokenizers.decoders.ByteLevel):
                ids = prompt_tail(tokenizer, ids, self._skip_special_tokens)
            self._stream = self._decode_stream(
                ids=ids, skip_special_tokens=self._skip_special_tokens
            )
            self._ids = None
        return self._stream.step(tokenizer, id)


def prompt_tail(
    tokenizer: Any, ids: Sequence[int], skip_special_tokens: bool
) -> Sequence[int]:
    """The last prompt tokens if they end on a character boundary, else ``ids``.

    Only for byte-level decoders (see the module docstring).
    """
    if len(ids) <= _TAIL_TOKENS:
        return ids
    tail = list(ids[-_TAIL_TOKENS:])
    text = tokenizer.decode(tail, skip_special_tokens=skip_special_tokens)
    if not text or text.endswith("\ufffd"):
        return ids
    return tail


def install() -> None:
    current = tokenizers.decoders.DecodeStream
    if current is PromptTailDecodeStream:
        return
    PromptTailDecodeStream._decode_stream = current
    tokenizers.decoders.DecodeStream = PromptTailDecodeStream
