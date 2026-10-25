# SPDX-License-Identifier: Apache-2.0
"""hostov IT2: the steady-state device carry must reproduce the upload path exactly, and the
host-side check must refuse every step where it would not."""
import types

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.runner import decode_carry as dc
from vllm_torchtpu.runner.tpu_runner import _padded_substitution_indices
from vllm_torchtpu.runner.tpu_runner_async_output import (
    carry_subtract_and_substitute_draft_token_ids,
    subtract_and_substitute_draft_token_ids,
)

K = 3
D = K + 1


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    # Default off on ix-unified-port; these tests cover the enabled path (VLLM_TPU_DECODE_CARRY=1).
    monkeypatch.setattr(dc, "ENABLED", True)


def _indices(num_reqs, target, padded_total):
    seq_idx = np.full(target, -1, dtype=np.int32)
    seq_idx[:num_reqs] = np.arange(num_reqs)
    pos_idx = np.full(padded_total, -1, dtype=np.int32)
    pos_idx[: num_reqs * D] = np.repeat(np.arange(num_reqs), D)
    cur = np.arange(num_reqs * D, dtype=np.int32)
    src = np.arange(num_reqs * D, dtype=np.int32)  # prev index i -> span [i*D, i*D+D)
    subst = _padded_substitution_indices(padded_total, cur, src)
    return torch.from_numpy(seq_idx), torch.from_numpy(pos_idx), torch.from_numpy(subst)


@pytest.mark.parametrize("num_reqs,target,padded_total,mrope", [
    (1, 4, 4, False), (2, 4, 8, True), (3, 4, 16, True), (3, 8, 16, False),
])
def test_carry_matches_upload_path_over_many_steps(num_reqs, target, padded_total, mrope):
    rng = np.random.default_rng(num_reqs * 100 + padded_total)
    total = num_reqs * D
    seq_idx, pos_idx, subst = _indices(num_reqs, target, padded_total)
    stale = np.array([0] * num_reqs + list(rng.integers(5, 50, target - num_reqs)), dtype=np.int32)
    logits_idx = torch.arange(padded_total, dtype=torch.int32)
    target_logits_idx = torch.tensor([0, 1, 2] * max(1, padded_total // 3), dtype=torch.int32)[: padded_total - 1]

    def host_arrays(S):
        seq = stale.copy(); seq[:num_reqs] = S
        nc = S - D
        pos = np.zeros(padded_total, dtype=np.int32)
        pos[:total] = np.repeat(nc, D) + np.tile(np.arange(D), num_reqs)
        p = torch.from_numpy(pos)
        return torch.from_numpy(seq), (p.unsqueeze(0).repeat(3, 1) if mrope else p)

    S = rng.integers(1000, 90000, num_reqs).astype(np.int32)
    r_prev = rng.integers(0, D, num_reqs).astype(np.int32)
    # step 0: the upload path, as the runner runs it today
    seq_h, pos_h = host_arrays(S)
    ids0 = torch.zeros(padded_total, dtype=torch.int32)
    nt = torch.from_numpy(rng.integers(1, 1000, num_reqs * D).astype(np.int32))
    prev = subtract_and_substitute_draft_token_ids(
        seq_h, pos_h, torch.from_numpy(r_prev), seq_idx, pos_idx, ids0, subst, nt, logits_idx, target_logits_idx)
    for step in range(30):
        r_now = rng.integers(0, D, num_reqs).astype(np.int32)
        S = S + D - r_prev  # the scheduler's optimistic advance, corrected by last step's rejections
        seq_h, pos_h = host_arrays(S)
        nt = torch.from_numpy(rng.integers(1, 1000, num_reqs * D).astype(np.int32))
        ref = subtract_and_substitute_draft_token_ids(
            seq_h, pos_h, torch.from_numpy(r_now), seq_idx, pos_idx, torch.zeros(padded_total, dtype=torch.int32),
            subst, nt, logits_idx, target_logits_idx)
        got = carry_subtract_and_substitute_draft_token_ids(
            prev[0], prev[1], D, torch.from_numpy(r_now), seq_idx, pos_idx, prev[2], subst, nt,
            logits_idx, target_logits_idx)
        for a, b in zip(got, ref):
            assert torch.equal(a, b), (step, a, b)
        prev = got
        r_prev = r_now


class _Pre:
    def __init__(self, req_ids):
        self.spec_decode_num_rejected_tokens = torch.zeros(4, dtype=torch.int32)
        self.req_id_to_index_copy = {r: i for i, r in enumerate(req_ids)}
        self.next_tokens_tpu = torch.zeros(16, dtype=torch.int32)


def _runner(req_ids, target=4):
    r = types.SimpleNamespace()
    r.input_batch = types.SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids))
    r.seq_lens_np = np.zeros(16, dtype=np.int32)
    r.seq_lens_np[len(req_ids):target] = 77  # stale padding
    r.positions_np = np.zeros(64, dtype=np.int32)
    r.uses_mrope = False
    r._pcp_enabled = False
    r._last_sequence_layout_plan = types.SimpleNamespace(kind=SequenceLayoutKind.ALL)
    r.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
    r.position_ids = torch.zeros(len(req_ids) * D, dtype=torch.int32)
    r.input_ids = torch.zeros(len(req_ids) * D, dtype=torch.int32)
    return r


def _set_host(r, S):
    n = len(S)
    r.seq_lens_np[:n] = S
    r.positions_np[: n * D] = np.repeat(np.asarray(S) - D, D) + np.tile(np.arange(D), n)


def _steady(req_ids=("a", "b"), r_prev=(1, 0), advance=None):
    """Runner after a recorded step N, with step N+1's host arrays set; returns (runner, kwargs)."""
    n = len(req_ids)
    r = _runner(req_ids)
    S = np.array([5000, 7000][:n], dtype=np.int32)
    _set_host(r, S)
    pre_n = _Pre(req_ids)
    dc.tag_results(pre_n)
    r._pre_async_results = pre_n
    kw = dict(start_index=0, num_reqs=n, num_scheduled_tokens_per_req_list=[D] * n,
              total_num_scheduled_tokens=n * D, padded_total_num_scheduled_tokens=n * D, target_num_reqs=4)
    assert dc.record(r, sub_pairs=tuple((i, D) for i in range(n)), seq_lens=torch.zeros(4), **kw)
    # sample_tokens(N): _modify_prev_results reads back step N-1's rejections (the ones subtracted at N)
    dc.record_rejected(r, pre_n, dict(zip(req_ids, r_prev)))
    pre_n1 = _Pre(req_ids)
    dc.tag_results(pre_n1)
    r._pre_async_results = pre_n1
    adv = advance if advance is not None else [D - x for x in r_prev]
    _set_host(r, S + np.array(adv, dtype=np.int32))
    kw["layout_plan"] = types.SimpleNamespace(kind=SequenceLayoutKind.ALL)
    return r, kw


def test_check_accepts_the_steady_state():
    r, kw = _steady()
    st = dc.check(r, **kw)
    assert st is not None and st.step_delta == D


def test_check_refuses_when_the_scheduler_advanced_differently():
    r, kw = _steady(advance=[D, D])  # a ignores its rejected draft (e.g. capped near max_model_len)
    assert dc.check(r, **kw) is None


def test_check_refuses_stale_rejections():
    r, kw = _steady()
    r._decode_carry_rej = (r._decode_carry_rej[0] + 99, r._decode_carry_rej[1])
    assert dc.check(r, **kw) is None


def test_check_refuses_changed_padding_slot():
    r, kw = _steady()
    r.seq_lens_np[3] = 78
    assert dc.check(r, **kw) is None


def test_check_refuses_changed_batch():
    r, kw = _steady()
    r.input_batch.req_ids = ["a", "c"]
    assert dc.check(r, **kw) is None
    r, kw = _steady()
    kw["num_scheduled_tokens_per_req_list"] = [D, 1]
    assert dc.check(r, **kw) is None


def test_check_refuses_a_position_jump():
    r, kw = _steady()
    r.positions_np[2] += 1
    assert dc.check(r, **kw) is None


def test_check_refuses_unmapped_request():
    r, kw = _steady()
    del r._pre_async_results.req_id_to_index_copy["b"]
    assert dc.check(r, **kw) is None


def test_disabled_never_carries_the_steady_state(monkeypatch):
    # ix-unified-port: the carry is default off; off must refuse even a step it would carry.
    r, kw = _steady()
    monkeypatch.setattr(dc, "ENABLED", False)
    assert dc.check(r, **kw) is None
    kw.pop("layout_plan")
    assert dc.record(r, sub_pairs=((0, D), (1, D)), seq_lens=torch.zeros(4), **kw) is False
