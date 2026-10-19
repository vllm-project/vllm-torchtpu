# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable, Sequence

import regex as re
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

_EOM = "<|eom|>"
_EOT = "<|eot|>"
_START = "<|start|>"
_MESSAGE = "<|message|>"
_FUNCTION_CALLS_OPEN = "<atem:function_calls>"
_ASSISTANT_TURN_OPEN = "<|start|>assistant"
_REASONING_RECIPIENTS = (None, "self")
_CONTENT_RECIPIENTS = ("user", "commentary")
_NON_TOOL_RECIPIENTS = (*_REASONING_RECIPIENTS, *_CONTENT_RECIPIENTS, "developer")
_CHANNEL_HEADER_RE = re.compile(r"(?:to=(?P<recipient>[^\s<]+))?<\|message\|>")
_HEADER_PAT = r"(?:to=[^\s<]+)?<\|message\|>"
_TRAILING_RECIPIENT_RE = re.compile(r"to=(?P<recipient>[^\s<]+)\s*$")
_REASONING_HEADER_PAT = r"(?:to=self|(?<!to=[^\s<]*))<\|message\|>"
_REASONING_HEADER_RE = re.compile(_REASONING_HEADER_PAT)
_COLLAPSE_RE = re.compile(
    r"<\|eom\|>(?:(?!" + _REASONING_HEADER_PAT + r").)*?" + _REASONING_HEADER_PAT,
    re.DOTALL,
)
_REASONING_BODY_PAT = r"(?:(?!" + _HEADER_PAT + r").)*?"
_REASONING_RE = re.compile(
    _REASONING_HEADER_PAT + r"(" + _REASONING_BODY_PAT + r")<\|eom\|>", re.DOTALL
)
_CONTENT_RE = re.compile(
    r"to=(?:user|commentary)<\|message\|>(.*?)(?=<\|eot\|>|<\|eom\|>|$)", re.DOTALL
)
_STRIP_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?"
    + _REASONING_HEADER_PAT
    + _REASONING_BODY_PAT
    + r"<\|eom\|>",
    re.DOTALL,
)
_STRIP_OPEN_REASONING_RE = re.compile(
    r"(?:<\|start\|>assistant\s*)?"
    + _REASONING_HEADER_PAT
    + r"(?:(?!<\|eom\|>)(?!"
    + _HEADER_PAT
    + r").)*"
    r"(?=" + _HEADER_PAT + r"|$)",
    re.DOTALL,
)
_OPEN_REASONING_RE = re.compile(
    _REASONING_HEADER_PAT + r"((?:(?!<\|eom\|>)(?!" + _HEADER_PAT + r").)*)"
    r"(?=" + _HEADER_PAT + r"|$)",
    re.DOTALL,
)
_HOLDBACK_MARKERS = (_EOM, _EOT, _START, _MESSAGE)
_OPEN_TAIL_HEADER_RE = re.compile(r"[\s](?:t|to|to=[^\s<]*)$")


def _current_assistant_turn(text: str) -> str:
    """Return only the text generated in the current assistant turn."""
    idx = text.rfind(_ASSISTANT_TURN_OPEN)
    return text[idx + len(_ASSISTANT_TURN_OPEN) :] if idx != -1 else text


def _trim_open_body(body: str) -> str:
    """Hold back any tail of a still-growing body that could still be framing."""
    while True:
        trimmed = body
        for marker in _HOLDBACK_MARKERS:
            for k in range(min(len(marker) - 1, len(trimmed)), 0, -1):
                if trimmed.endswith(marker[:k]):
                    trimmed = trimmed[:-k]
                    break
            else:
                continue
            break
        header_tail = _OPEN_TAIL_HEADER_RE.search(trimmed)
        if header_tail is not None:
            trimmed = trimmed[: header_tail.start()]
        if trimmed == body:
            return body
        body = trimmed


class MuseSpark1xReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self._emitted_reasoning: str = ""
        self._emitted_content: str = ""
        self._tool_handoff_done: bool = False
        self._marker_ids: tuple[int | None, set[int]] | None = None

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        """Preserve Muse's ATEM framing tokens in the decoded output."""
        request.skip_special_tokens = False
        return request

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Whether the model has left reasoning and opened a TOOL channel."""
        text = self.model_tokenizer.decode(input_ids)
        remainder = self._tool_channel_remainder(text)
        return _FUNCTION_CALLS_OPEN in remainder or "<atem:invoke" in remainder

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.is_reasoning_end(list(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count reasoning-body tokens, excluding header and marker tokens."""
        if self._marker_ids is None:
            vocab = self.vocab
            boundaries = {vocab[m] for m in (_EOM, _EOT, _START) if m in vocab}
            self._marker_ids = (vocab.get(_MESSAGE), boundaries)
        message_id, boundary_ids = self._marker_ids

        count = 0
        header_start = 0
        in_reasoning = False
        for i, token_id in enumerate(token_ids):
            if token_id == message_id:
                header = self.model_tokenizer.decode(list(token_ids[header_start:i]))
                match = _TRAILING_RECIPIENT_RE.search(header)
                recipient = match.group("recipient") if match else None
                in_reasoning = recipient in _REASONING_RECIPIENTS
                header_start = i + 1
            elif token_id in boundary_ids:
                in_reasoning = False
                header_start = i + 1
            elif in_reasoning:
                count += 1
        return count

    @classmethod
    def _scoped_turn(cls, text: str) -> str:
        """Current assistant turn with reasoning spans removed."""
        scoped = _current_assistant_turn(text)
        scoped = _STRIP_REASONING_RE.sub("", scoped)
        return _STRIP_OPEN_REASONING_RE.sub("", scoped)

    @classmethod
    def _tool_channel_remainder(cls, text: str) -> str:
        """Text from the first tool-channel header onward, framing INCLUDED."""
        scoped = cls._scoped_turn(text)
        for match in _CHANNEL_HEADER_RE.finditer(scoped):
            if match.group("recipient") not in _NON_TOOL_RECIPIENTS:
                return scoped[match.start() :]
        return ""

    @staticmethod
    def _classify_bodies(text: str) -> tuple[str, str]:
        """Split ``text`` into (reasoning_body, content_body), channel-aware."""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        pos = 0
        n = len(text)
        while pos < n:
            match = _CHANNEL_HEADER_RE.search(text, pos)
            if not match:
                break
            recipient = match.group("recipient")
            body_start = match.end()
            eom = text.find(_EOM, body_start)
            eot = text.find(_EOT, body_start)
            terminators = [p for p in (eom, eot) if p != -1]
            next_header = _CHANNEL_HEADER_RE.search(text, body_start)
            if next_header is not None:
                terminators.append(next_header.start())
            body_end = min(terminators) if terminators else n
            body = text[body_start:body_end]
            if not terminators:
                body = _trim_open_body(body)
            if recipient in _REASONING_RECIPIENTS:
                reasoning_parts.append(body)
            elif (
                recipient in _CONTENT_RECIPIENTS
                and _FUNCTION_CALLS_OPEN not in body
                and "<atem:invoke" not in body
            ):
                content_parts.append(body)
            if terminators and body_end in (eom, eot):
                pos = body_end + len(_EOM if body_end == eom else _EOT)
            else:
                pos = body_end
        return "".join(reasoning_parts), "".join(content_parts)

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        """Promote un-surfaced content when the stream ends mid-reasoning."""
        _, content_body = self._classify_bodies(previous_text)
        remainder = content_body[len(self._emitted_content) :]
        return remainder or None

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        collapsed = _COLLAPSE_RE.sub("\n", model_output)
        matches = _REASONING_RE.findall(collapsed)
        reasoning = "\n".join(matches) if matches else None
        open_match = _OPEN_REASONING_RE.search(model_output)
        if open_match and open_match.group(1):
            partial = open_match.group(1)
            reasoning = f"{reasoning}\n{partial}" if reasoning else partial
        remainder = _STRIP_REASONING_RE.sub("", model_output)
        remainder = _STRIP_OPEN_REASONING_RE.sub("", remainder)
        if "<atem:invoke" in remainder or _FUNCTION_CALLS_OPEN in remainder:
            return reasoning, (remainder or None)
        content_bodies = _CONTENT_RE.findall(model_output)
        if content_bodies:
            content = "".join(content_bodies) or None
        elif _REASONING_HEADER_RE.search(model_output):
            content = None
        else:
            content = model_output or None
            reasoning = None
        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Channel-aware streaming split of reasoning vs content."""
        curr_reason, curr_content = self._classify_bodies(current_text)
        reasoning_delta = ""
        if curr_reason.startswith(self._emitted_reasoning) and len(curr_reason) > len(
            self._emitted_reasoning
        ):
            reasoning_delta = curr_reason[len(self._emitted_reasoning) :]
            self._emitted_reasoning = curr_reason
        content_delta = ""
        if curr_content.startswith(self._emitted_content) and len(curr_content) > len(
            self._emitted_content
        ):
            content_delta = curr_content[len(self._emitted_content) :]
            self._emitted_content = curr_content
        handoff = ""
        if not self._tool_handoff_done:
            remainder = self._tool_channel_remainder(current_text)
            if _FUNCTION_CALLS_OPEN in remainder or "<atem:invoke" in remainder:
                handoff = remainder
                self._tool_handoff_done = True
        if handoff:
            return DeltaMessage(reasoning=reasoning_delta or None, content=handoff)
        if reasoning_delta and content_delta:
            return DeltaMessage(reasoning=reasoning_delta, content=content_delta)
        if reasoning_delta:
            return DeltaMessage(reasoning=reasoning_delta)
        if content_delta:
            return DeltaMessage(content=content_delta)
        return None
