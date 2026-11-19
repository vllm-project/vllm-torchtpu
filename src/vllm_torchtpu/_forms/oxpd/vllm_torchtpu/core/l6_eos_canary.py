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
"""q38 L6-HOST end-token canary (log-only; changes no token, no program, no timing path that matters).

IX saw greedy batch-1 decode emit the end token (<|im_end|>, id 248044) in the middle of a sentence and asked whether
its PR11 / PR14 ports cause it. Our served bench forces every request's output length (aiperf records show
osl_mismatch 0), so an emitted end token never ends a request there and the symptom cannot show in bench records.
This canary counts end-token EMISSIONS in the decode worker instead.

``VLLM_TPU_L6_EOS_CANARY=<id>[+<id>...]`` (',' also accepted; unset or 0 = off) names the end-token ids. Once per
worker step, in ``_modify_prev_results`` after the discarded requests' tokens are cleared, rank 0 counts the
previous step's valid output tokens and how many of them are one of the ids, into the current l6_ab window:
``counts=...,eos.end=<n>,eos.tok=<n>`` on the ``q38 L6AB role=worker`` line, plus ``eos.tok1`` / ``eos.end1`` for
solo-phase steps (exactly one request produced tokens), where IX's empty batch-1 answers reproduce. The l6_ab reader then gives the
end-token rate per 1,000 output tokens per class (lever on vs off). The tokens belong to the previous step, so a
window's first step counts the last step of the window before (1 / VLLM_TPU_L6_AB_STEPS of the tokens). It reports
only while an L6 lever is set (l6_ab.ACTIVE), because the windows carry the counts.
"""

from __future__ import annotations

import os

from vllm_torchtpu.core import l6_ab as _l6_ab

ENV = "VLLM_TPU_L6_EOS_CANARY"


def _parse(raw: str) -> frozenset[int]:
    raw = (raw or "").strip()
    if raw in ("", "0"):
        return frozenset()
    ids = frozenset(int(x) for x in raw.replace("+", ",").split(",") if x.strip())
    if any(i < 0 for i in ids):
        raise ValueError(f"{ENV} ids must be >= 0, got {raw!r}")
    return ids


IDS: frozenset[int] = _parse(os.environ.get(ENV, ""))
ON = bool(IDS)
_rank: list[int | None] = [None]


def _is_rank0() -> bool:
    if _rank[0] is None:
        _rank[0] = _l6_ab._rank_world()[0]
    return _rank[0] == 0


def count(valid_sampled_token_ids: list[list[int]]) -> None:
    """Count the step's valid output tokens and end-token emissions (rank 0, while l6_ab windows are active)."""
    if not _l6_ab.ACTIVE or not _is_rank0():
        return
    tok = 0
    end = 0
    active = 0
    ids = IDS
    for seq in valid_sampled_token_ids:
        if seq:
            active += 1
        tok += len(seq)
        for t in seq:
            if t in ids:
                end += 1
    if tok:
        _l6_ab.note("eos", "tok", tok)
    if end:
        _l6_ab.note("eos", "end", end)
    if active == 1:
        # solo phase (exactly one request produced tokens this step): IX's empty batch-1 answers reproduce there
        _l6_ab.note("eos", "tok1", tok)
        if end:
            _l6_ab.note("eos", "end1", end)
