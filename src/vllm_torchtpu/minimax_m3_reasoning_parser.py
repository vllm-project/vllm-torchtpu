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
"""vLLM's MiniMax-M3 reasoning parser with its marker searches in C.

vLLM 0.32.0's ``MiniMaxM3ReasoningParser`` finds ``<mm:think>`` and
``</mm:think>`` with a Python loop that builds two tuples per position. The
chat serving path calls ``is_reasoning_end`` on the whole prompt twice per
request (``resolve_reasoning_ended`` before the stream starts, ``parse_delta``
on the first output). The M3 chat template writes ``<mm:think>`` only in the
system prompt unless an assistant turn carries reasoning, so the start-marker
search walks back over the whole prompt: 13.4 ms per call on 74k prompt ids,
0.81 ms with ``Sequence.index``. Results are unchanged.
"""

from collections.abc import Iterator, Sequence

from vllm.reasoning.minimax_m3_reasoning_parser import (
    MiniMaxM3ReasoningParser as _UpstreamMiniMaxM3ReasoningParser,
)

# Backward searches go window by window, so a marker near the end (the last
# ``</mm:think>``) is found without touching the rest of the prompt.
_RFIND_WINDOW = 4096


def _index(token_ids: Sequence[int], value: int, start: int, stop: int) -> int:
    """``token_ids.index(value, start, stop)``, or ``stop`` when absent."""
    try:
        return token_ids.index(value, start, stop)
    except ValueError:
        return stop


def _marker_starts(
    token_ids: Sequence[int], marker: tuple[int, ...], start: int, stop: int
) -> Iterator[int]:
    """Ascending positions in [start, stop) where ``marker`` begins."""
    i = _index(token_ids, marker[0], start, stop)
    while i < stop:
        if tuple(token_ids[i : i + len(marker)]) == marker:
            yield i
        i = _index(token_ids, marker[0], i + 1, stop)


def find_token_sequence(
    token_ids: Sequence[int], marker_ids: Sequence[int], *, last: bool
) -> int:
    """Start of the first (``last``: the last) ``marker_ids`` run in
    ``token_ids``, or -1."""
    marker = tuple(marker_ids)
    if not marker or len(marker) > len(token_ids):
        return -1
    stop = len(token_ids) - len(marker) + 1
    if not last:
        return next(_marker_starts(token_ids, marker, 0, stop), -1)
    for hi in range(stop, 0, -_RFIND_WINDOW):
        found = max(
            _marker_starts(token_ids, marker, max(0, hi - _RFIND_WINDOW), hi),
            default=-1,
        )
        if found >= 0:
            return found
    return -1


class MiniMaxM3ReasoningParser(_UpstreamMiniMaxM3ReasoningParser):
    @staticmethod
    def _contains_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        return find_token_sequence(token_ids, marker_ids, last=False) >= 0

    @staticmethod
    def _rfind_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> int:
        return find_token_sequence(token_ids, marker_ids, last=True)
