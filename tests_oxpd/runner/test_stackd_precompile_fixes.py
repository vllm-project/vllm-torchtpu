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
"""q38 DEC-FB-3 STACK-D warm-up fixes (CPU).

1. Lever 3 alias fix: `_precompile_draft_rows_carry_and_token` warms
   `_draft_rows_carry_token_fresh` with two DISTINCT tensors, as serving passes
   them (step 0's untrimmed last_hidden and hidden). The real precompile runs
   against the real fresh-carry body, recompiled with Dynamo's eager backend and
   dynamic=False (the device backend is not available on CPU); a serving-shaped
   call with distinct tensors then compiles 0 new graphs. Warming with one
   tensor twice (the old call) leaves that call to compile 1 more.
2. VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS=N: the real rejection-sampler
   precompile visits exactly the (padded_logits_length, padded_num_reqs) pairs
   of the uncapped sweep with num_reqs <= the bucket covering N and
   length <= the bucket covering N*(K+1); with N=0 it visits the full sweep.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import tpu_runner as tr
from vllm_torchtpu.spec_decode import eagle3

H = 8


# ---------------------------------------------------------------- 1. lever 3 alias fix
def _fresh_body():
    """The real `_draft_rows_carry_token_fresh` body under its decorators."""
    fn = eagle3.Eagle3Proposer._draft_rows_carry_token_fresh
    fn = getattr(fn, "__wrapped__", fn)  # _static_shape_args
    return torch._dynamo.eval_frame.innermost_fn(fn)  # the torch.compile'd function


def _fake_proposer(monkeypatch, caps=(8,), paddings=(4, 8), reqs=(1, 2, 4)):
    runner = SimpleNamespace(
        num_tokens_paddings=list(paddings),
        num_reqs_paddings=list(reqs),
        max_num_reqs=max(reqs),
        _hidden_states_dtype=torch.float32,
        device=torch.device("cpu"),
        num_reqs_max_model_len=None,
        num_reqs_most_model_len=caps[0],
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    p = SimpleNamespace(runner=runner, draft_model=SimpleNamespace(draft_id_to_target_id=None))
    p._draft_hidden_size = lambda: H
    p._draft_rows_carry_and_token = lambda positions, idx, hid: (positions, idx)

    def body(positions, indices, last_hidden):  # stands in for the lm-head token path
        indices = indices.clamp_min(0)
        return torch.index_select(positions, positions.ndim - 1, indices), last_hidden.sum(-1).to(torch.int32)

    p._carry_and_token_body = body
    compiled = torch.compile(backend="eager", fullgraph=True, dynamic=False)(_fresh_body())
    p._draft_rows_carry_token_fresh = lambda *a: compiled(p, *a)
    monkeypatch.setattr(eagle3, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.setattr(eagle3, "set_model_tag", lambda tag: contextlib.nullcontext())
    monkeypatch.setenv("VLLM_TPU_DRAFT_CARRY_ROWS_FRESH", "1")
    return p


def _graphs() -> int:
    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


def _serving_call(p, T=8, rows=4, cap=8):
    positions = torch.zeros(T, dtype=torch.int32)
    idx = torch.zeros(rows, dtype=torch.int64)
    last_hidden = torch.randn(cap, H)
    hidden = torch.randn(cap, H)  # a distinct tensor, as serving passes it
    return p._draft_rows_carry_token_fresh(positions, idx, last_hidden, hidden)


def test_precompile_warms_the_serving_signature(monkeypatch):
    torch._dynamo.reset()
    p = _fake_proposer(monkeypatch)
    eagle3.Eagle3Proposer._precompile_draft_rows_carry_and_token(p)
    before = _graphs()
    out = _serving_call(p)
    assert len(out) == 3
    assert _graphs() - before == 0, "a serving-shaped call compiled a new graph after the precompile"


def test_precompile_passes_distinct_tensors(monkeypatch):
    p = _fake_proposer(monkeypatch)
    calls = []
    p._draft_rows_carry_token_fresh = lambda *a: calls.append(a) or (a[0], a[1], a[3][: a[1].shape[0]])
    eagle3.Eagle3Proposer._precompile_draft_rows_carry_and_token(p)
    assert calls, "the fresh carry was not warmed"
    assert all(c[2] is not c[3] for c in calls), "last_hidden and hidden were the same tensor"


def test_aliased_warmup_leaves_serving_to_compile_once():
    """The mechanism the fix removes, shown on the real body: warm aliased, serve distinct -> +1."""
    torch._dynamo.reset()
    p = SimpleNamespace()
    p._carry_and_token_body = lambda pos, idx, lh: (pos[: idx.shape[0]], lh.sum(-1).to(torch.int32))
    compiled = torch.compile(backend="eager", fullgraph=True, dynamic=False)(_fresh_body())
    full = torch.zeros(8, H)
    compiled(p, torch.zeros(8, dtype=torch.int32), torch.zeros(4, dtype=torch.int64), full, full)
    before = _graphs()
    compiled(p, torch.zeros(8, dtype=torch.int32), torch.zeros(4, dtype=torch.int64), torch.randn(8, H), torch.randn(8, H))
    assert _graphs() - before == 1


# ---------------------------------------------------------------- 2. precompile sweep cap
class _Recorder:
    synthetic_mode = False
    synthetic_conditional_rates = None

    def __init__(self):
        self.pairs = []

    def __call__(self, **kw):
        if not kw.get("do_sampling"):
            self.pairs.append((int(kw["draft_token_ids"].shape[0]), int(kw["num_draft_tokens"].shape[0])))
        return torch.zeros(1)


def _fake_runner(paddings, reqs, max_reqs, k=3):
    rs = _Recorder()
    r = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=k),
        rejection_sampler=rs,
        num_tokens_paddings=list(paddings),
        num_reqs_paddings=list(reqs),
        max_num_reqs=max_reqs,
        vocab_size=16,
        _hidden_states_dtype=torch.float32,
        device=torch.device("cpu"),
        _greedy_head_owner=None,
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    r._topk_candidate_blocks = lambda: 0
    r._spec_verify_bucket_reachable = lambda t, n: tr.TPUModelRunner._spec_verify_bucket_reachable(r, t, n)
    r.spec_bonus_and_target_logits = lambda lg, b, t: (torch.zeros(b.shape[0], dtype=torch.int32), lg)
    r.spec_gather_bonus_and_target_logits = lambda lg, b, t: (None, lg)
    return r, rs


PADS = [8, 16, 32, 48, 64, 128, 256]
REQS = [1, 2, 4, 8, 16, 32, 64]


def _sweep(monkeypatch, cap):
    monkeypatch.setattr(tr, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.delenv("VLLM_TPU_VERIFY_FUSED", raising=False)
    if cap is None:
        monkeypatch.delenv("VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS", str(cap))
    r, rs = _fake_runner(PADS, REQS, max_reqs=64)
    tr.TPUModelRunner._precompile_rejection_sampler(r)
    return rs.pairs


def _bucket(values, n):
    return next((v for v in values if v >= n), values[-1])


@pytest.mark.parametrize("cap", [32, 24, 3])
def test_cap_visits_exactly_the_capped_subset(monkeypatch, cap):
    full = _sweep(monkeypatch, None)
    capped = _sweep(monkeypatch, cap)
    rb = _bucket(REQS, cap)  # the request bucket a step with <= cap requests pads to
    lb = _bucket(PADS, min(cap * 4, PADS[-1]))  # K = 3 -> 4 rows per request
    want = [(t, n) for t, n in full if n <= rb and t <= lb]
    assert capped == want
    assert len(capped) < len(full)
    # every reachable pair a step with <= cap requests can produce is still warmed
    for t, n in full:
        if n <= rb and t <= lb:
            assert (t, n) in capped


def test_cap_off_and_zero_keep_the_full_sweep(monkeypatch):
    full = _sweep(monkeypatch, None)
    assert _sweep(monkeypatch, 0) == full
    assert len(full) > 10
