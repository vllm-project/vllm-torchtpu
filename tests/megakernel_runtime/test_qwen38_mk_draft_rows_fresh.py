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
"""VLLM_TPU_MK_DRAFT_ROWS_FRESH: no eager programs between MK draft step 0 and the
fused loop.

Before, a c16 pure-decode step paid two torch_tpu execute_as_strided programs here:
`last_token_indices[:capacity]` (step 0's draft_rows) and the `[:p]` trims of step 0's
outputs plus the loop's `positions + 1`. torch_tpu materializes a view even at full
length, so at p == capacity the fix must hand on the tensors themselves, with no slice
op at all. Values must be bitwise those of the flag-off path, and the compiled
programs must see the same input shapes (no new compile key, nothing to warm).
"""

from __future__ import annotations

import contextlib
import types

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_torchtpu import envs
from vllm_torchtpu.spec_decode import eagle3
from vllm_torchtpu.spec_decode import mtp_fused_loop as fl

HIDDEN = 24
# View-producing ops: each one, consumed by a compiled program, is an eager
# execute_as_strided program on torch_tpu. Plus the eager add the loop used to pay.
VIEW_OPS = ("slice", "as_strided", "alias", "view", "select", "narrow", "pad",
            "constant_pad_nd", "add")


class _StubMk(torch.nn.Module):
    """Deterministic stand-in for the MK draft call: per-row carry and greedy token."""

    def mk_draft_step(self, input_ids, positions, hidden, rows_per_request):
        carry = torch.tanh(hidden * 1.3 + positions[:, None].to(hidden.dtype) * 0.01)
        tokens = (input_ids * 7 + positions.to(input_ids.dtype)) % 1000
        return carry, tokens


def _step0_module():
    m = fl._MkDraftStep0.__new__(fl._MkDraftStep0)
    torch.nn.Module.__init__(m)
    m.prefix, m.mtp, m.rows_per_request = "", _StubMk(), 4
    return m


class _OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[str] = []
        self.paused = False

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if not self.paused:
            self.ops.append(func.__name__)
        return func(*args, **(kwargs or {}))


def _proposer(monkeypatch, cap, log=None):
    """An Eagle3Proposer with just what `_run_mk_draft_step0` reads; the step-0
    program is the real `_MkDraftStep0.forward`, run eagerly (op logging paused)."""
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(eagle3, name, lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(eagle3, "set_forward_context",
                        lambda *a, **k: contextlib.nullcontext())
    p = eagle3.Eagle3Proposer.__new__(eagle3.Eagle3Proposer)
    p.runner = types.SimpleNamespace(
        num_reqs_max_model_len=cap, num_reqs_most_model_len=cap,
        _dp_num_tokens_across_dp=lambda n: None, mesh=None,
    )
    p.draft_vllm_config = None
    p._seed_extras = {}
    p._draft_rows_len = lambda chunk: cap
    p._build_draft_attn_metadata = lambda **kw: {}
    p._draft_moe_ctx = contextlib.nullcontext
    module = _step0_module()
    seen = []

    def step0(input_ids, positions, hidden, draft_rows):
        seen.append(tuple((tuple(t.shape), t.dtype, t.stride())
                          for t in (input_ids, positions, hidden, draft_rows)))
        if log is not None:
            log.paused = True
        try:
            return module.forward(input_ids, positions, hidden, draft_rows)
        finally:
            if log is not None:
                log.paused = False

    p._mk_draft_step0 = step0
    p._mk_draft_step0_warm = True
    return p, seen


def _inputs(T, p, n, seed=0):
    g = torch.Generator().manual_seed(seed + 131 * T + p)
    input_ids = torch.randint(0, 50000, (T,), generator=g, dtype=torch.int32)
    positions = torch.randint(0, 200000, (T,), generator=g, dtype=torch.int32)
    hidden = torch.randn(T, HIDDEN, generator=g, dtype=torch.bfloat16)
    # Each request's last of 4 rows; filler past the real requests, as prep leaves it.
    lti = torch.arange(p, dtype=torch.int32) * 4 + 3
    lti[n:] = 0
    return input_ids, positions, hidden, lti


def _run(monkeypatch, flag, cap, T, p, n, extras=False, log=None):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_ROWS_FRESH", flag)
    prop, seen = _proposer(monkeypatch, cap, log)
    input_ids, positions, hidden, lti = _inputs(T, p, n)
    chunk = types.SimpleNamespace(num_reqs=n)
    if extras:
        # The seed program's draft_rows: the padded last_token_indices, fresh.
        prop._seed_extras[id(chunk)] = (
            eagle3.Eagle3Proposer._padded_draft_rows(prop, lti, cap).clone(), None
        )
    if log is not None:
        with log:
            out = prop._run_mk_draft_step0(chunk, input_ids, positions, hidden, lti, None)
    else:
        out = prop._run_mk_draft_step0(chunk, input_ids, positions, hidden, lti, None)
    return out, seen, prop, chunk


# (cap, p) over buckets 4/8/16 (and 32 for c32), T = 4 rows per request bucket.
CASES = [(cap, p, n, T)
         for cap in (4, 8, 16, 32)
         for p in (4, 8, 16, 32) if p <= cap
         for n in sorted({max(1, p // 2 + 1), p})
         for T in sorted({4 * p, 4 * cap})]


@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("cap,p,n,T", CASES)
def test_bitwise_flag_off_vs_on(monkeypatch, cap, p, n, T, extras):
    off, seen_off, _, _ = _run(monkeypatch, False, cap, T, p, n, extras=False)
    on, seen_on, _, _ = _run(monkeypatch, True, cap, T, p, n, extras=extras)
    assert off[3] is None
    for a, b in zip(off[:3], on[:3]):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                           b.view(torch.int16) if b.dtype == torch.bfloat16 else b)
    # The loop's step-1 positions: step 0's in-graph +1 at p == cap, the add otherwise.
    ref_next = off[2] + 1
    got_next = eagle3.Eagle3Proposer._loop_step1_positions(on[3], on[2], on[2])
    assert torch.equal(ref_next, got_next)
    if p == cap:
        assert on[3] is not None
    else:
        assert on[3] is None
    # Same program inputs flag on and off: no new compile key.
    assert seen_off == seen_on


@pytest.mark.parametrize("cap", [4, 8, 16, 32])
def test_full_capacity_emits_no_slice(monkeypatch, cap):
    """p == capacity: nothing but the step-0 program runs; its outputs are handed on
    as the very tensors it returned, and draft_rows is a tensor that exists already."""
    log = _OpLog()
    on, _, prop, chunk = _run(monkeypatch, True, cap, 4 * cap, cap, cap - 1,
                              extras=False, log=log)
    assert not [o for o in log.ops if o.split(".")[0] in VIEW_OPS], log.ops
    nxt = eagle3.Eagle3Proposer._loop_step1_positions(on[3], on[2], on[2])
    assert nxt is on[3]
    # Flag off, the same step pays the views and the loop's add.
    log_off = _OpLog()
    off, _, _, _ = _run(monkeypatch, False, cap, 4 * cap, cap, cap - 1, log=log_off)
    eagle3.Eagle3Proposer._loop_step1_positions(off[3], off[2], off[2])
    assert [o for o in log_off.ops if o.split(".")[0] in ("slice", "alias", "view")]


def test_full_capacity_emits_no_slice_with_extras(monkeypatch):
    log = _OpLog()
    _run(monkeypatch, True, 16, 64, 16, 16, extras=True, log=log)
    assert not [o for o in log.ops if o.split(".")[0] in VIEW_OPS], log.ops


@pytest.mark.parametrize("cap,p", [(16, 8), (16, 4), (8, 4), (32, 16)])
def test_below_capacity_draft_rows_from_seed(monkeypatch, cap, p):
    """p < capacity: draft_rows is the seed program's tensor (no pad program); the
    output trims stay, as one eager program like before."""
    log = _OpLog()
    _run(monkeypatch, True, cap, 4 * p, p, p, extras=True, log=log)
    ops = [o.split(".")[0] for o in log.ops]
    assert "constant_pad_nd" not in ops and "pad" not in ops, ops
    assert "add" not in ops


def test_draft_rows_choice(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_ROWS_FRESH", True)
    prop, _ = _proposer(monkeypatch, 16)
    chunk = types.SimpleNamespace(num_reqs=5)
    lti16 = torch.arange(16, dtype=torch.int32)
    lti8 = torch.arange(8, dtype=torch.int32)
    assert prop._mk_step0_draft_rows(chunk, lti16) is lti16
    pad = prop._mk_step0_draft_rows(chunk, lti8)
    assert pad.tolist() == list(range(8)) + [-1] * 8
    seed_rows = pad.clone()
    prop._seed_extras[id(chunk)] = (seed_rows, None)
    assert prop._mk_step0_draft_rows(chunk, lti8) is seed_rows
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_ROWS_FRESH", False)
    off = prop._mk_step0_draft_rows(chunk, lti16)
    assert off is not lti16 and torch.equal(off, lti16)


def test_padded_carry_keeps_the_add():
    """If the loop pads the positions carry (a different tensor reaches the loop),
    the step-0 +1 output does not match it: fall back to the add."""
    pos = torch.arange(8, dtype=torch.int32)
    padded = torch.nn.functional.pad(pos, (0, 8))
    got = eagle3.Eagle3Proposer._loop_step1_positions(pos + 1, pos, padded)
    assert torch.equal(got, padded + 1)
    assert eagle3.Eagle3Proposer._loop_step1_positions(None, pos, pos).tolist() == \
        (pos + 1).tolist()


def test_step0_module_outputs(monkeypatch):
    m = _step0_module()
    ids, pos, hid, lti = _inputs(64, 16, 16)
    rows = eagle3.Eagle3Proposer._padded_draft_rows(None, lti, 16)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_ROWS_FRESH", False)
    off = m.forward(ids, pos, hid, rows)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_ROWS_FRESH", True)
    on = m.forward(ids, pos, hid, rows)
    assert len(off) == 3 and len(on) == 4
    for a, b in zip(off, on[:3]):
        assert torch.equal(a, b)
    assert torch.equal(on[3], off[2] + 1) and on[3].dtype == off[2].dtype


@pytest.mark.parametrize("cap", [8, 16, 32])
def test_warmup_coverage_no_new_program_inputs(monkeypatch, cap):
    """Every (capacity, p, T) a server can hit: the step-0 program's inputs (shape,
    dtype, stride) are those of the flag-off path, with or without the seed rows, and
    the loop's inputs keep their [p] shapes. So the flag adds no compile key and needs
    no warmup variant (a missed one cost ~50 compiles mid-serve before)."""
    for p in (4, 8, 16, 32):
        if p > cap:
            continue
        for n in range(1, p + 1):
            for T in sorted({4 * p, 4 * cap}):
                off, s_off, _, _ = _run(monkeypatch, False, cap, T, p, n)
                for extras in (False, True):
                    on, s_on, _, _ = _run(monkeypatch, True, cap, T, p, n, extras=extras)
                    assert s_on == s_off, (cap, p, n, T, extras)
                    nxt = eagle3.Eagle3Proposer._loop_step1_positions(on[3], on[2], on[2])
                    for a, b in ((off[0], on[0]), (off[1], on[1]), (off[2] + 1, nxt)):
                        assert a.shape == b.shape and a.dtype == b.dtype
                        assert a.is_contiguous() and b.is_contiguous()
                    np.testing.assert_array_equal(off[1].numpy(), on[1].numpy())
