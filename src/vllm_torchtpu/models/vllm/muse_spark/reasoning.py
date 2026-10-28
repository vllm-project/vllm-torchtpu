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
"""Muse Spark reasoning parser for the pinned vLLM.

Backport of the Muse Spark profile of ``MuseReasoningParser`` in
``vllm/reasoning/muse_reasoning_parser.py`` from
https://github.com/vllm-project/vllm/pull/62622, layered on the pinned
``MuseGlimmerReasoningParser``. Streaming keeps the pinned tool handoff because
the upstream one feeds the ``muse`` tool parser, which the pin does not ship.
Drop this module and its ``musespark1x`` registration once the vLLM pin
includes that PR.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import regex as re
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.muse_glimmer_reasoning_parser import (
    MuseGlimmerReasoningParser,
    _trim_open_body,
)

_EOM = "<|eom|>"
_EOT = "<|eot|>"
_TRAILING_RECIPIENT_RE = re.compile(r"to=(?P<recipient>[^\s<]+)\s*$")
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_ASSISTANT_TURN_OPEN = "<|start|>assistant"

_MUSE_HEADER_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?(?:to=(?P<recipient>[^\s<]+))?<\|message\|>"
)
_MUSE_BOUNDARY_RE = re.compile(r"<\|eom\|>|<\|eot\|>|<\|start\|>")
_MUSE_OTHER_ROLE_RE = re.compile(r"<\|start\|>(?!assistant(?:\s|<|$))[^\s<]+")
_MUSE_NON_TOOL_RECIPIENTS = (None, "self", "user", "commentary", "developer")


def _muse_messages(text: str) -> Iterator[tuple[str | None, str, bool, str]]:
    """Yield recipient, body, completion status, and framed text per message."""
    headers = list(_MUSE_HEADER_RE.finditer(text))
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        boundary = _MUSE_BOUNDARY_RE.search(text, header.end(), end)
        body_end = boundary.start() if boundary else end
        closed = boundary is not None or i + 1 < len(headers)
        yield (
            header.group("recipient"),
            text[header.end() : body_end],
            closed,
            text[header.start() : end],
        )


def _muse_current_turn(text: str) -> str:
    # Prompt examples and earlier turns must not trigger a tool handoff.
    roles = list(_MUSE_OTHER_ROLE_RE.finditer(text))
    if roles:
        start = text.find(_ASSISTANT_TURN_OPEN, roles[-1].end())
        return text[start:] if start != -1 else ""
    return text


class MuseSpark1xReasoningParser(MuseGlimmerReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._marker_ids: tuple[int | None, set[int]] | None = None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count reasoning-body tokens, excluding header and marker tokens."""
        if self._marker_ids is None:
            vocab = self.vocab
            boundaries = {vocab[m] for m in (_EOM, _EOT, "<|start|>") if m in vocab}
            self._marker_ids = (vocab.get("<|message|>"), boundaries)
        message_id, boundary_ids = self._marker_ids

        count = 0
        header_start = 0
        in_reasoning = False
        for i, token_id in enumerate(token_ids):
            if token_id == message_id:
                header = self.model_tokenizer.decode(list(token_ids[header_start:i]))
                match = _TRAILING_RECIPIENT_RE.search(header)
                recipient = match.group("recipient") if match else None
                in_reasoning = recipient in (None, "self")
                header_start = i + 1
            elif token_id in boundary_ids:
                in_reasoning = False
                header_start = i + 1
            elif in_reasoning:
                count += 1
        return count

    @classmethod
    def _tool_channel_remainder(cls, text: str) -> str:
        """Text from the first tool-channel header onward, framing INCLUDED.
        ``DelegatingParser.parse_delta`` rebuilds ``current_text`` from whatever
        this parser returns as ``.content`` on the transition delta and commits
        it; anything not returned is destroyed. It must start AT the
        ``to=<name><|message|>`` header -- handing over the text after the header
        loses the recipient, and the tool parser then sees a bare ``<|message|>``,
        classifies it as the content channel, and leaks the ATEM markup.
        """
        scoped = _muse_current_turn(text)
        headers = list(_MUSE_HEADER_RE.finditer(scoped))
        for header, (recipient, body, _, _) in zip(headers, _muse_messages(scoped)):
            if recipient not in _MUSE_NON_TOOL_RECIPIENTS and (
                _FUNCTION_CALLS_OPEN in body or "<atem:invoke" in body
            ):
                return scoped[header.start() :]
        return ""

    @staticmethod
    def _classify_bodies(text: str) -> tuple[str, str]:
        """Split ``text`` into (reasoning_body, content_body), channel-aware.
        Framing markers and tool channels contribute nothing -- the tool parser
        owns those. A body ends at ``<|eom|>`` / ``<|eot|>``, at the next channel
        header, or at end-of-text (an OPEN body, which is held back).
        """
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for recipient, body, closed, _ in _muse_messages(text):
            if not closed:
                body = _trim_open_body(body)
            if recipient in (None, "self"):
                reasoning_parts.append(body)
            elif recipient in ("user", "commentary"):
                content_parts.append(body)
        return "".join(reasoning_parts), "".join(content_parts)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        messages = list(_muse_messages(model_output))
        if not messages:
            return None, model_output or None
        reasoning = (
            "".join(
                body
                for recipient, body, _, _ in messages
                if recipient in (None, "self")
            )
            or None
        )
        has_tool_call = any(
            recipient not in _MUSE_NON_TOOL_RECIPIENTS
            and (_FUNCTION_CALLS_OPEN in body or "<atem:invoke" in body)
            for recipient, body, _, _ in messages
        )
        if has_tool_call:
            content = "".join(
                framed
                for recipient, _, _, framed in messages
                if recipient not in (None, "self", "developer")
            )
        else:
            content = "".join(
                body
                for recipient, body, _, _ in messages
                if recipient in ("user", "commentary")
            )
        return reasoning, content or None
