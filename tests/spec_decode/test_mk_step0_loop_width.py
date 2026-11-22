# SPDX-License-Identifier: Apache-2.0
"""Eagle3Proposer._loop_takes_width: when MK draft step 0's carries stay at the
request capacity instead of being trimmed to the loop bucket p."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

BUCKETS = (4, 8, 16, 24)


def _stub(mk_sizes=()):
    bucket = lambda n: min(b for b in BUCKETS if b >= n)  # noqa: E731
    mk = lambda n: object() if n in mk_sizes else None  # noqa: E731
    return SimpleNamespace(_loop_bucket=bucket, _mk_fused_loop_for=mk)


@pytest.mark.parametrize(
    "width,rows,mk_loop,mk_sizes,want",
    [
        (16, 4, False, (), True),  # capacity is a bucket: untrimmed
        (16, 16, False, (), True),
        (12, 4, False, (), False),  # not a warmed bucket: trim to p
        (4, 8, False, (), False),  # narrower than p
        (16, 4, True, (4, 8, 16), True),  # MK loop at p and at width
        (16, 4, True, (4, 8), False),  # MK at p only: keep p's MK loop
        (16, 4, True, (16,), False),  # MK at width only: keep p's stock loop
        (16, 4, True, (), True),  # stock at both
    ],
)
def test_loop_takes_width(monkeypatch, width, rows, mk_loop, mk_sizes, want):
    monkeypatch.setenv("VLLM_TPU_MK_DRAFT_LOOP", "1" if mk_loop else "0")
    got = Eagle3Proposer._loop_takes_width(_stub(mk_sizes), width, rows)
    assert got is want
