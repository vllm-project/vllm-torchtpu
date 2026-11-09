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
"""CPU tests for the VLLM_TPU_MUSE_MEGAKERNEL=1 hooks in the TPU model runner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput

from vllm_torchtpu.models.vllm.muse_spark import megakernel_fastpath as mk_fp
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _row(gen, k=3, seed=0):
    """One request's (generated tokens, top ids, top logprobs) megakernel row."""
    rng = np.random.default_rng(seed)
    top_ids, top_lps = [], []
    for j, tok in enumerate(gen):
        ids = rng.permutation(np.arange(100, 100 + k + 2))[: k + 1]
        ids[j % (k + 1)] = tok
        top_ids.append(ids.astype(np.int32))
        top_lps.append(-np.sort(rng.random(k + 1)).astype(np.float32))
    return gen, np.stack(top_ids), np.stack(top_lps)


def test_megakernel_logprobs_rank_and_cu():
    rows = [_row([7, 8], seed=1), _row([9], seed=2)]
    lp = tpu_runner._megakernel_logprobs(rows, k=2)
    assert lp.cu_num_generated_tokens == [0, 2, 3]
    assert lp.logprob_token_ids.shape == (3, 3)
    flat = [
        (tok, ids, lps)
        for gen, ids_r, lps_r in rows
        for tok, ids, lps in zip(gen, ids_r, lps_r)
    ]
    for n, (tok, ids, lps) in enumerate(flat):
        hit = int(np.nonzero(ids == tok)[0][0])
        assert lp.logprob_token_ids[n, 0] == tok
        np.testing.assert_array_equal(lp.logprob_token_ids[n, 1:], ids[:2])
        assert lp.logprobs[n, 0] == lps[hit]
        np.testing.assert_array_equal(lp.logprobs[n, 1:], lps[:2])
        assert lp.sampled_token_ranks[n] == hit + 1

    gen, ids, lps = _row([5])
    ids = np.where(ids == 5, 1, ids)
    with pytest.raises(RuntimeError, match="not in its own top"):
        tpu_runner._megakernel_logprobs([(gen, ids, lps)], k=2)


@pytest.mark.parametrize("with_vllm_logprobs", [False, True])
def test_megakernel_merge_logprobs(with_vllm_logprobs):
    rows = [_row([7, 8], seed=3)]
    n_vllm = 2
    if not with_vllm_logprobs:
        # No request asked for logprobs: vLLM's value passes through.
        assert tpu_runner._megakernel_merge_logprobs(None, rows, n_vllm, None) is None
        merged = tpu_runner._megakernel_merge_logprobs(None, rows, n_vllm, 2)
        assert merged.cu_num_generated_tokens == [0, 2, 2, 2]
        assert merged.logprob_token_ids.shape == (2, 3)
        return
    old = LogprobsLists(
        logprob_token_ids=np.full((3, 3), 11, np.int32),
        logprobs=np.full((3, 3), -1.0, np.float32),
        sampled_token_ranks=np.ones(3, np.int32),
        cu_num_generated_tokens=[0, 1, 3],
    )
    merged = tpu_runner._megakernel_merge_logprobs(old, rows, n_vllm, 2)
    # One cu entry per request: megakernel rows first, then vLLM's shifted.
    assert merged.cu_num_generated_tokens == [0, 2, 3, 5]
    np.testing.assert_array_equal(merged.logprob_token_ids[2:], old.logprob_token_ids)
    assert merged.logprob_token_ids[0, 0] == 7 and merged.logprob_token_ids[1, 0] == 8


class _FakeInputBatch:
    def __init__(self, req_ids):
        self.req_ids = list(req_ids)
        self.num_reqs = len(req_ids)
        self.max_num_logprobs = None

    def swap_states(self, i, j):
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]


def _fastpath_owning(owned):
    fp = object.__new__(mk_fp.MuseDecodeFastPath)
    fp.slots = mk_fp.SlotTable(8)
    for r in owned:
        st = fp.slots.acquire(r)
        assert st is not None
        st.owned = True
    return fp


@pytest.mark.parametrize(
    "req_ids,owned,num_decode,num_windowed,expected_order,expected_counts",
    [
        # Nothing owned: untouched.
        (["a", "b", "c"], [], 2, 3, ["a", "b", "c"], (0, 2, 3)),
        # Owned decode rows move ahead of the vLLM decode and windowed rows.
        (
            ["a", "b", "c", "d", "e"],
            ["b", "d"],
            3,
            4,
            ["b", "d", "a", "c", "e"],
            (2, 4, 4),
        ),
        # All owned.
        (["a", "b"], ["a", "b"], 2, 2, ["a", "b"], (2, 2, 2)),
        (
            ["p", "q", "r", "s"],
            ["s", "q"],
            2,
            2,
            ["q", "s", "p", "r"],
            (2, 3, 3),
        ),
    ],
)
def test_move_owned_to_front(
    req_ids, owned, num_decode, num_windowed, expected_order, expected_counts
):
    fp = _fastpath_owning(owned)
    runner = SimpleNamespace(input_batch=_FakeInputBatch(req_ids))
    counts = fp.move_owned_to_front(runner, num_decode, num_windowed)
    assert counts == expected_counts
    owned_first = runner.input_batch.req_ids[: counts[0]]
    assert sorted(owned_first) == sorted(owned)
    # Owned rows keep their relative order; so do the rest.
    assert runner.input_batch.req_ids == expected_order


class _FakeMegakernel:
    """Fake fastpath: one megakernel step returns fixed tokens and drafts."""

    timer = None
    spec = True

    def __init__(self, res):
        self.res = res
        self.calls = []

    def run_decode(self, runner, indices, scheduler_output):
        self.calls.append(list(indices))
        return {r: self.res[r] for r in runner.input_batch.req_ids[: len(indices)]}


@pytest.mark.parametrize("mk_front", [1, 2])
def test_mixed_step_prepends_owned_and_shifts_discard(monkeypatch, mk_front):
    """vLLM sampled rows [mk_front, n); the owned rows' tokens and drafts go first
    and discard indices (full-batch space) still name the right rows."""
    monkeypatch.setenv("VLLM_TPU_MUSE_MEGAKERNEL", "1")
    owned = ["m0", "m1"][:mk_front]
    req_ids = owned + ["v0", "v1", "v2"]
    res = {r: ([10 + n, 20 + n], None, None, [30 + n]) for n, r in enumerate(owned)}
    fake = _FakeMegakernel(res)
    monkeypatch.setattr(mk_fp, "_INSTANCE", fake)

    runner = object.__new__(TPUModelRunner)
    runner.input_batch = _FakeInputBatch(req_ids)
    runner._mk_scheduler_output = None
    runner.spec_decode_manager = SimpleNamespace(_draft_token_ids=[[1], [2], [3]])
    # vLLM's output covers only its rows; row v1 (full index mk_front + 1) is a
    # partial prefill, so it is discarded and empty.
    discard_full = [mk_front + 1]
    shifted = [i - mk_front for i in discard_full]
    vllm_tokens = [[5], [], [6]]
    assert all(not vllm_tokens[i] for i in shifted)
    out = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=vllm_tokens,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    runner._megakernel_prepend_owned(out, mk_front)
    assert fake.calls == [list(range(mk_front))]
    assert out.sampled_token_ids == [res[r][0] for r in owned] + vllm_tokens
    assert runner.spec_decode_manager._draft_token_ids == [res[r][3] for r in owned] + [
        [1],
        [2],
        [3],
    ]
    runner._megakernel_check_output(out, discard_full, "mixed")
    # The same empty row without its discard index is an error.
    with pytest.raises(RuntimeError, match="no token"):
        runner._megakernel_check_output(out, [], "mixed")
    assert out.logprobs is None


def test_warmup_request_predicate():
    assert tpu_runner._is_warmup_req_id("__spec_warmup_b4_0__")
    assert tpu_runner._is_warmup_req_id("__mtp_prefill_warmup__")
    assert not tpu_runner._is_warmup_req_id("cmpl-123")
    assert not tpu_runner._is_warmup_req_id(None)


def test_vllm_rows_discard_logits(monkeypatch):
    """The skip-logits shortcut fires only when every vLLM-path row is a partial
    prefill: never for warmup steps or for rows with scheduled drafts, whose
    verify chunk must not get empty (zero-size) logits.
    """
    monkeypatch.setenv("VLLM_TPU_MUSE_MEGAKERNEL", "1")
    monkeypatch.setattr(mk_fp, "_INSTANCE", SimpleNamespace(ready=True))
    req_ids = ["m0", "p0", "p1"]
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        input_batch=SimpleNamespace(
            num_reqs=3,
            req_ids=req_ids,
            logits_processing_needs_token_ids=np.zeros(3, bool),
        ),
        requests={
            "m0": SimpleNamespace(num_computed_tokens=500, num_tokens=501),
            "p0": SimpleNamespace(num_computed_tokens=0, num_tokens=4096),
            "p1": SimpleNamespace(num_computed_tokens=2048, num_tokens=4096),
        },
    )

    def sched(spec=None, p1_tokens=1024):
        return SimpleNamespace(
            num_scheduled_tokens={"m0": 8, "p0": 1024, "p1": p1_tokens},
            scheduled_spec_decode_tokens=spec or {},
        )

    fn = tpu_runner._megakernel_vllm_rows_discard_logits
    # The owned row m0 (mk_front=1) is ignored, even with drafts.
    assert fn(runner, sched(spec={"m0": [1] * 7}), 1)
    # p1's prompt completes this step, so it samples.
    assert not fn(runner, sched(p1_tokens=2048), 1)
    # A vLLM-path row with drafts always gets logits.
    assert not fn(runner, sched(spec={"p0": [1, 2]}), 1)
    # Warmup steps always build logits.
    runner.input_batch.req_ids = ["m0", "__spec_warmup_b2_0__", "p1"]
    runner.requests["__spec_warmup_b2_0__"] = runner.requests["p0"]
    assert not fn(runner, sched(), 1)
    runner.input_batch.req_ids = req_ids
    # No megakernel, or not ready, or async scheduling: off.
    runner.scheduler_config.async_scheduling = True
    assert not fn(runner, sched(), 1)
    runner.scheduler_config.async_scheduling = False
    monkeypatch.setattr(mk_fp, "_INSTANCE", SimpleNamespace(ready=False))
    assert not fn(runner, sched(), 1)
    monkeypatch.setattr(mk_fp, "_INSTANCE", None)
    assert not fn(runner, sched(), 1)
