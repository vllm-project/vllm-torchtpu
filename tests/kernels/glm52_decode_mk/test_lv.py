# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""glm52_decode_mk lever LV (live-row collectives): the static row plans and the adapter's full-bucket dispatch.
Pure Python / host jnp, no TPU. The on-device checks (oracle, bit identity vs levers B, the Mosaic row-tiling
hazard, timing) are the campaign's (stageB/lv-prod)."""

import dataclasses

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.glm52_decode_mk import collectives as CO
from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import presets as PR
from vllm_torchtpu.kernels.glm52_decode_mk import vllm_adapter as A

ROWS_FLOORS = [(16, 0), (32, 16), (32, 0), (48, 32), (64, 48)]


def _safe(o, m):
    """The Mosaic-exact row pieces of a (16, 128)-tiled bf16 buffer (collectives.AR_SAFE)."""
    return (m % 16 == 0 and o % 16 == 0) or (m == 12 and o % 16 == 0) or (m == 4 and o % 4 == 0)


@pytest.mark.parametrize("n", range(4, 65, 4))
def test_ar_pieces_cover_exactly_and_are_safe(n):
    covered = []
    for o, m in CO.ar_pieces(n):
        assert _safe(o, m), (n, o, m)
        covered += list(range(o, o + m))
    assert covered == list(range(n))


@pytest.mark.parametrize("rows,floor", ROWS_FLOORS)
@pytest.mark.parametrize("eight", ["split", "round"])
def test_lv_ar_rows_is_an_option_and_covers_t(rows, floor, eight):
    opts = CO.lv_ar_options(rows, floor, eight)
    assert opts[-1] == rows and all(n % 4 == 0 and n >= min(floor, rows) for n in opts)
    if eight == "round":
        assert all(n % 16 != 8 for n in opts)
    for T in range(1, rows + 1):
        n = int(CO.lv_ar_rows(jnp.int32(T), rows, floor, eight))
        assert n in opts and T <= n <= rows, (T, n, opts)


@pytest.mark.parametrize("rows", [(16, 32), (32,), (16,)])
def test_preset_lv_floor_is_the_next_smaller_bucket(rows):
    for t in rows:
        kw = PR.tiles_kwargs(t, PR.parse("B,LV"), rows)
        below = [r for r in rows if r < t]
        assert kw["lv"] and kw["lv_floor"] == (max(below) if below else 0)
        assert kw["lv_eight"] == PR.LV_EIGHT and kw["lv_loop"] == PR.lv_loop(t)
    assert "lv" not in PR.tiles_kwargs(16, PR.parse("B"))


def test_fills_bucket():
    assert [A.fills_bucket(T, 16) for T in (4, 8, 12, 13, 15, 16)] == [False] * 3 + [True] * 3
    assert [A.fills_bucket(T, 32) for T in (17, 24, 28, 29, 32)] == [False] * 3 + [True] * 2


def _mk(lv_full, built_full=(16, 32)):
    mk = object.__new__(A.Glm52DecodeMK)
    mk.cfg = dataclasses.replace(C.FULL, TMAX=32, BM=8)
    mk.rows = [16, 32]
    mk._steps = {16: object(), 32: object()}
    mk._steps_full = {t: object() for t in built_full} if lv_full else {}
    mk.lv_full = lv_full
    return mk


def test_decision_marks_full_bucket_steps():
    mk = _mk(True)
    # (scheduled tokens, drafts, in prompt) per request: MTP-3 verify requests of 4 rows; one short request of 2
    cases = [([(4, 3, False)], 16, False), ([(4, 3, False)] * 2, 16, False), ([(4, 3, False)] * 3, 16, False),
             ([(4, 3, False)] * 4, 16, True), ([(4, 3, False)] * 3 + [(2, 1, False)], 16, True),
             ([(4, 3, False)] * 5, 32, False), ([(4, 3, False)] * 7, 32, False), ([(4, 3, False)] * 8, 32, True),
             ([(4, 3, False)] * 7 + [(2, 1, False)], 32, True), ([(4, 3, False)] * 6 + [(3, 2, False)], 32, False)]
    for reqs, want_t, want_full in cases:
        b, why, _ = mk.decision(reqs)
        assert why == "mk" and int(b) == want_t and isinstance(b, A.StepBucket), (reqs, b, why)
        assert b.full == want_full, (reqs, b.full)
    b, why, _ = mk.decision([(5, 0, True)])
    assert b is None and why == "prefill"
    off = _mk(False)
    b, _, _ = off.decision([(4, 3, False)] * 4)
    assert int(b) == 16 and not b.full
