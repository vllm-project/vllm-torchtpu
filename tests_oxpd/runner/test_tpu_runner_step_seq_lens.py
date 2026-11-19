# SPDX-License-Identifier: Apache-2.0
"""Padded request slots carry zero sequence length, not a previous step's (q38 SHARD-TP).

The QSA indexer takes its short-context fast path only when ``max(seq_lens)``
over the step's request slots fits the top-k budget; a stale padded slot from an
earlier long step used to push a short step onto the scored path.
"""

import ast
import pathlib

import numpy as np

from vllm_torchtpu.runner.tpu_runner import fill_step_seq_lens

RUNNER = pathlib.Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu" / "runner" / "tpu_runner.py"


def test_padded_slots_are_zeroed_after_a_longer_step():
    seq_lens = np.zeros(8, np.int32)
    fill_step_seq_lens(seq_lens, np.array([100000, 90000, 80000, 70000]), np.array([6400, 6400, 6400, 6400]))
    assert seq_lens.tolist() == [106400, 96400, 86400, 76400, 0, 0, 0, 0]
    # A later, shorter step with fewer requests: no stale lengths remain.
    fill_step_seq_lens(seq_lens, np.array([1000]), np.array([500]))
    assert seq_lens.tolist() == [1500, 0, 0, 0, 0, 0, 0, 0]
    assert seq_lens.max() // 4 <= 512  # the indexer's fast-path test now sees the short step


def test_prepare_inputs_uses_the_helper():
    tree = ast.parse(RUNNER.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "fill_step_seq_lens"]
    assert calls, "tpu_runner no longer fills seq_lens through fill_step_seq_lens"
    raw = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) and t.value.attr == "seq_lens_np" for t in n.targets)]
    assert not raw, "a direct seq_lens_np[...] = ... write bypasses the padded-slot reset"
