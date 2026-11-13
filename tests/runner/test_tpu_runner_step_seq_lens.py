"""Padded request slots carry zero sequence length, not a previous step's (q38 SHARD-TP).

The QSA indexer takes its short-context fast path only when ``max(seq_lens)``
over the step's request slots fits the top-k budget; a stale padded slot from an
earlier long step used to push a short step onto the scored path.
fill_step_seq_lens zeroes the padded slots only under the flag
(VLLM_TPU_ZERO_PADDED_SEQ_LENS, ``zero_padded=True``).
"""

import ast
import pathlib

import numpy as np
import pytest

tpu_runner = pytest.importorskip("vllm_torchtpu.runner.tpu_runner")

RUNNER = pathlib.Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu" / "runner" / "tpu_runner.py"


def test_padded_slots_are_zeroed_after_a_longer_step():
    seq_lens = np.zeros(8, np.int32)
    tpu_runner.fill_step_seq_lens(
        seq_lens, np.array([100000, 90000, 80000, 70000]), np.array([6400, 6400, 6400, 6400]),
        zero_padded=True,
    )
    assert seq_lens.tolist() == [106400, 96400, 86400, 76400, 0, 0, 0, 0]
    # A later, shorter step with fewer requests: no stale lengths remain.
    tpu_runner.fill_step_seq_lens(seq_lens, np.array([1000]), np.array([500]), zero_padded=True)
    assert seq_lens.tolist() == [1500, 0, 0, 0, 0, 0, 0, 0]
    assert seq_lens.max() // 4 <= 512  # the indexer's fast-path test now sees the short step


def test_padded_slots_zeroed_when_enabled():
    seq = np.full(8, 99, np.int32)
    tpu_runner.fill_step_seq_lens(
        seq, np.array([5, 7], np.int32), np.array([1, 2], np.int32), zero_padded=True
    )
    assert seq.tolist() == [6, 9, 0, 0, 0, 0, 0, 0]


def test_padded_slots_kept_by_default():
    seq = np.full(4, 99, np.int32)
    tpu_runner.fill_step_seq_lens(seq, np.array([5], np.int32), np.array([1], np.int32))
    assert seq.tolist() == [6, 99, 99, 99]


def test_prepare_inputs_uses_the_helper():
    tree = ast.parse(RUNNER.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "fill_step_seq_lens"]
    assert calls, "tpu_runner no longer fills seq_lens through fill_step_seq_lens"
    assert all(
        any(k.arg == "zero_padded" for k in c.keywords) for c in calls
    ), "a fill_step_seq_lens call ignores VLLM_TPU_ZERO_PADDED_SEQ_LENS"
    raw = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) and t.value.attr == "seq_lens_np" for t in n.targets)]
    assert not raw, "a direct seq_lens_np[...] = ... write bypasses the padded-slot reset"
