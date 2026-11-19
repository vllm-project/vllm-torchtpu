# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_POSTFWD_FUSED (q38 DEC-FB-3): runner-level checks of the fused
post-forward programs' warm-shape gating, on CPU.

A shape joins the warm set only after its fused call returned. A warmup
compile that raises leaves the shape unwarmed, so serving takes the separate
calls for it instead of retrying the compile on the step path.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    subtract_num_rejected_tokens,
    subtract_substitute_extract,
)

K = 3
STRIDE = 1 + K
MAX_REQS = 8
T = 16  # token bucket for 4 requests x (1 + K)
REQS = ["r0", "r1", "r2", "r3"]
SRC_POS = {"r0": 2, "r1": 0, "r2": 3, "r3": 1}


def _runner(warmup_done):
    g = np.random.default_rng(7)
    runner = object.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    runner.input_batch = SimpleNamespace(req_ids=list(REQS))
    runner._last_sequence_layout_plan = None
    runner.warmup_done = warmup_done
    runner._pre_async_results = SimpleNamespace(
        req_id_to_index_copy=dict(SRC_POS),
        spec_decode_num_rejected_tokens=torch.tensor([1, 2, 0, 3], dtype=torch.int32),
        next_tokens_tpu=torch.from_numpy(
            g.integers(1, 150000, MAX_REQS * STRIDE).astype(np.int32)
        ),
    )
    runner.input_ids = torch.from_numpy(g.integers(1, 150000, T).astype(np.int32))
    runner.position_ids = torch.from_numpy(g.integers(0, 5000, T).astype(np.int64))
    return runner


def _step_inputs(runner):
    seq_lens = torch.arange(100, 100 + MAX_REQS, dtype=torch.int32)
    seq_idx = torch.full((MAX_REQS,), -1, dtype=torch.int32)
    pos_idx = torch.full((T,), -1, dtype=torch.int32)
    for i, rid in enumerate(REQS):
        seq_idx[i] = SRC_POS[rid]
        pos_idx[i * STRIDE : (i + 1) * STRIDE] = SRC_POS[rid]
    pending = (runner._pre_async_results.spec_decode_num_rejected_tokens, seq_idx, pos_idx)
    target = torch.zeros(T, dtype=torch.int32)
    target[: len(REQS) * K] = torch.tensor(
        [i * STRIDE + j for i in range(len(REQS)) for j in range(K)], dtype=torch.int32
    )
    md = SimpleNamespace(
        final_logits_indices=torch.arange(T, dtype=torch.int32),
        target_logits_indices=target,
        draft_token_ids=None,
    )
    return seq_lens, pending, md


def _call(runner):
    seq_lens, pending, md = _step_inputs(runner)
    out = TPUModelRunner._postfwd_fused_prepare(
        runner,
        seq_lens,
        pending,
        0,
        len(REQS),
        np.full(len(REQS), STRIDE, dtype=np.int32),
        md,
    )
    return out, seq_lens, pending, md


class _Fused:
    """Stands in for the compiled `_postfwd_subst_fused`: counts calls and can
    raise on the first one (a failed torch_tpu warmup compile)."""

    def __init__(self, fail_first):
        self.calls = 0
        self.fail_first = fail_first

    def __call__(self, *args):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("simulated torch_tpu compile failure")
        return subtract_substitute_extract(*args)


def test_failed_warmup_compile_leaves_shape_cold(monkeypatch):
    fused = _Fused(fail_first=True)
    monkeypatch.setattr(tpu_runner, "_postfwd_subst_fused", fused)
    runner = _runner(warmup_done=False)
    with pytest.raises(RuntimeError, match="simulated"):
        _call(runner)
    assert not runner.__dict__.get("_postfwd_warm_keys"), "a failed compile warmed its shape"
    # Serving: the same shape must take the separate calls, not retry the compile.
    runner.warmup_done = True
    pos_before = runner.position_ids.clone()
    out, seq_lens, pending, _ = _call(runner)
    assert fused.calls == 1, "serving retried the fused compile of an unwarmed shape"
    assert "_postfwd_fused_out" not in runner.__dict__
    ref_seq, ref_pos = subtract_num_rejected_tokens(seq_lens, pos_before, *pending)
    assert torch.equal(out, ref_seq) and torch.equal(runner.position_ids, ref_pos)
    assert runner._postfwd_stats == [0, 1]


def test_successful_warmup_compile_warms_shape(monkeypatch):
    fused = _Fused(fail_first=False)
    monkeypatch.setattr(tpu_runner, "_postfwd_subst_fused", fused)
    runner = _runner(warmup_done=False)
    _call(runner)
    assert len(runner._postfwd_warm_keys) == 1
    runner.__dict__.pop("_postfwd_fused_out")
    runner.warmup_done = True
    pos_before = runner.position_ids.clone()
    ids_before = runner.input_ids.clone()
    out, seq_lens, pending, md = _call(runner)
    assert fused.calls == 2, "serving skipped the fused program of a warmed shape"
    start, ids, drafts = runner._postfwd_fused_out
    cur, pre = runner._postfwd_subst_cache[1], runner._postfwd_subst_cache[2]
    ref = subtract_substitute_extract(
        seq_lens, pos_before, *pending, ids_before, cur, pre,
        runner._pre_async_results.next_tokens_tpu,
        md.final_logits_indices, md.target_logits_indices,
    )
    assert start == 0
    for a, b in zip((out, runner.position_ids, ids, drafts), ref):
        assert torch.equal(a, b)
    assert runner._postfwd_stats == [2, 0]
