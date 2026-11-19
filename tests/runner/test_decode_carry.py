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
"""The decode carry (runner/decode_carry.py): it runs by rule, the carried
program reproduces the upload path exactly, the host-side check refuses every
step where it would not, and _prepare_inputs wires it in.
"""

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
QWEN35 = ["Qwen3_5MoeForConditionalGeneration"]


@pytest.fixture
def qwen3_5_draft_path_rule(monkeypatch):
    """The registry holding Qwen3.5's rule, as register_models leaves it, on
    the checkpoint and device the rule was validated on."""
    from vllm_torchtpu.models.qwen3_5 import hooks
    from vllm_torchtpu.spec_decode import draft_path_rules

    monkeypatch.setattr(draft_path_rules, "_RULES", [hooks.mtp_draft_path])
    monkeypatch.setattr(hooks, "has_calibrated_shape", lambda _model_config: True)
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")


def _engine(*, arch=QWEN35, method="mtp", max_num_seqs=1, async_scheduling=True):
    ns = types.SimpleNamespace
    scheduler_config = ns(max_num_seqs=max_num_seqs, async_scheduling=async_scheduling)
    vllm_config = ns(
        model_config=ns(architectures=arch),
        speculative_config=None if method is None else ns(method=method),
    )
    return scheduler_config, vllm_config


def test_the_rule_holds_for_a_batch1_async_qwen3_5_mtp_engine(
    qwen3_5_draft_path_rule,
):
    assert dc.rule_holds(*_engine()) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"arch": ["Qwen3NextForCausalLM"]},
        {"method": None},
        {"method": "eagle3"},
        {"max_num_seqs": 2},
        {"async_scheduling": False},
    ],
)
def test_the_rule_is_off_outside_it(qwen3_5_draft_path_rule, kwargs):
    assert dc.rule_holds(*_engine(**kwargs)) is False


def test_off_never_carries_records_or_tags():
    """A runner whose rule does not hold (decided in __init__) never checks,
    records or tags."""
    r = types.SimpleNamespace(_decode_carry_enabled=False, _pre_async_results=None)
    kw = dict(
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req_list=[4],
        total_num_scheduled_tokens=4,
        padded_total_num_scheduled_tokens=4,
        target_num_reqs=4,
    )
    assert dc.check(r, layout_plan=None, **kw) is None
    assert dc.record(r, sub_pairs=((0, 4),), seq_lens=torch.zeros(4), **kw) is False
    res = types.SimpleNamespace(decode_carry_seq=None)
    dc.tag_results(r, res)
    assert res.decode_carry_seq is None


def test_carry_fast_positions_match():
    """The cached mrope position view returns the same values as the torch
    slice, follows in-place updates and refreshes on a replaced buffer."""
    buf = torch.arange(3 * 16, dtype=torch.int64).reshape(3, 16)
    runner = types.SimpleNamespace(
        uses_mrope=True,
        mrope_positions=types.SimpleNamespace(cpu=buf),
        _mrope_np_view=None,
    )
    ref = np.asarray(buf[:, :5].numpy(), dtype=np.int64)
    fast = dc._host_positions(runner, 5)
    assert fast.dtype == ref.dtype == np.int64 and np.array_equal(fast, ref)
    buf[1, 2] = 999  # the cached view sees in-place updates
    assert dc._host_positions(runner, 5)[1, 2] == 999
    new = torch.zeros(3, 16, dtype=torch.int64)
    runner.mrope_positions.cpu = new  # a replaced buffer refreshes the view
    assert dc._host_positions(runner, 5).sum() == 0


def _indices(num_reqs, target, padded_total):
    seq_idx = np.full(target, -1, dtype=np.int32)
    seq_idx[:num_reqs] = np.arange(num_reqs)
    pos_idx = np.full(padded_total, -1, dtype=np.int32)
    pos_idx[: num_reqs * D] = np.repeat(np.arange(num_reqs), D)
    cur = np.arange(num_reqs * D, dtype=np.int32)
    src = np.arange(num_reqs * D, dtype=np.int32)  # prev index i -> span [i*D, i*D+D)
    subst = _padded_substitution_indices(padded_total, cur, src)
    return torch.from_numpy(seq_idx), torch.from_numpy(pos_idx), torch.from_numpy(subst)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "num_reqs,target,padded_total,mrope",
    [
        (1, 4, 4, False),
        (2, 4, 8, True),
        (3, 4, 16, True),
        (3, 8, 16, False),
    ],
)
def test_carry_matches_upload_path_over_many_steps(
    num_reqs, target, padded_total, mrope, compiled
):
    """`compiled` traces the carry program in one graph, as the runner's
    dynamic=False program is, with step_delta a Python int baked in."""
    carry = (
        torch.compile(
            carry_subtract_and_substitute_draft_token_ids,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        if compiled
        else carry_subtract_and_substitute_draft_token_ids
    )
    rng = np.random.default_rng(num_reqs * 100 + padded_total)
    total = num_reqs * D
    seq_idx, pos_idx, subst = _indices(num_reqs, target, padded_total)
    stale = np.array(
        [0] * num_reqs + list(rng.integers(5, 50, target - num_reqs)), dtype=np.int32
    )
    logits_idx = torch.arange(padded_total, dtype=torch.int32)
    target_logits_idx = torch.tensor(
        [0, 1, 2] * max(1, padded_total // 3), dtype=torch.int32
    )[: padded_total - 1]

    def host_arrays(S):
        seq = stale.copy()
        seq[:num_reqs] = S
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
        seq_h,
        pos_h,
        torch.from_numpy(r_prev),
        seq_idx,
        pos_idx,
        ids0,
        subst,
        nt,
        logits_idx,
        target_logits_idx,
    )
    for step in range(30):
        r_now = rng.integers(0, D, num_reqs).astype(np.int32)
        S = (
            S + D - r_prev
        )  # the scheduler's optimistic advance, corrected by last step's rejections
        seq_h, pos_h = host_arrays(S)
        nt = torch.from_numpy(rng.integers(1, 1000, num_reqs * D).astype(np.int32))
        ref = subtract_and_substitute_draft_token_ids(
            seq_h,
            pos_h,
            torch.from_numpy(r_now),
            seq_idx,
            pos_idx,
            torch.zeros(padded_total, dtype=torch.int32),
            subst,
            nt,
            logits_idx,
            target_logits_idx,
        )
        got = carry(
            prev[0],
            prev[1],
            D,
            torch.from_numpy(r_now),
            seq_idx,
            pos_idx,
            prev[2],
            subst,
            nt,
            logits_idx,
            target_logits_idx,
        )
        for a, b in zip(got, ref):
            assert torch.equal(a, b), (step, a, b)
        prev = got
        r_prev = r_now


class _Pre:
    def __init__(self, req_ids):
        self.spec_decode_num_rejected_tokens = torch.zeros(4, dtype=torch.int32)
        self.req_id_to_index_copy = {r: i for i, r in enumerate(req_ids)}
        self.next_tokens_tpu = torch.zeros(16, dtype=torch.int32)
        self.decode_carry_seq = None


def _runner(req_ids, target=4):
    r = types.SimpleNamespace()
    r._decode_carry_enabled = True
    r._decode_carry_state = None
    r._decode_carry_rej = None
    r._mrope_np_view = None
    r.input_batch = types.SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids))
    r.seq_lens_np = np.zeros(16, dtype=np.int32)
    r.seq_lens_np[len(req_ids) : target] = 77  # stale padding
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
    """Runner after a recorded step N, with step N+1's host arrays set; returns
    (runner, kwargs)."""
    n = len(req_ids)
    r = _runner(req_ids)
    S = np.array([5000, 7000][:n], dtype=np.int32)
    _set_host(r, S)
    pre_n = _Pre(req_ids)
    dc.tag_results(r, pre_n)
    r._pre_async_results = pre_n
    kw = dict(
        start_index=0,
        num_reqs=n,
        num_scheduled_tokens_per_req_list=[D] * n,
        total_num_scheduled_tokens=n * D,
        padded_total_num_scheduled_tokens=n * D,
        target_num_reqs=4,
    )
    assert dc.record(
        r, sub_pairs=tuple((i, D) for i in range(n)), seq_lens=torch.zeros(4), **kw
    )
    # sample_tokens(N): _modify_prev_results reads back step N-1's rejections
    # (the ones subtracted at N)
    dc.record_rejected(r, pre_n, dict(zip(req_ids, r_prev)))
    pre_n1 = _Pre(req_ids)
    dc.tag_results(r, pre_n1)
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
    r, kw = _steady(
        advance=[D, D]
    )  # a ignores its rejected draft (e.g. capped near max_model_len)
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


def test_disabled_never_carries_the_steady_state():
    # Off (any engine outside the rule) refuses even a step it would carry.
    r, kw = _steady()
    r._decode_carry_enabled = False
    assert dc.check(r, **kw) is None
    kw.pop("layout_plan")
    assert (
        dc.record(r, sub_pairs=((0, D), (1, D)), seq_lens=torch.zeros(4), **kw) is False
    )


# The runner wiring: _prepare_inputs with the carry on, against the upload path.


class _TpuTyped(torch.Tensor):
    """A CPU tensor whose ``device`` reads as a TPU device, so _prepare_inputs
    takes its TPU-only fused subtraction (the only path the carry runs on)."""

    @property
    def device(self):
        return types.SimpleNamespace(type="tpu")


_SUBTRACT_PROGRAMS = (
    "_compiled_subtract_and_substitute_draft_token_ids",
    "_compiled_carry_subtract_and_substitute_draft_token_ids",
    "_compiled_subtract_and_substitute_placeholder_token",
    "_compiled_subtract_num_rejected_tokens",
)


@pytest.fixture
def eager_subtract_programs(monkeypatch):
    """The compiled subtraction programs run their eager bodies, and TPU events
    are stand-ins; yields the names of the programs called, in order."""
    from runner_input_test_utils import _fake_tpu_events

    import vllm_torchtpu.runner.tpu_runner_async_output as async_output

    _fake_tpu_events(monkeypatch)
    called: list[str] = []
    for name in _SUBTRACT_PROGRAMS:
        body = getattr(async_output, name).__wrapped__._torchdynamo_orig_callable

        def run(*args, _body=body, _name=name):
            called.append(_name)
            return _body(*args)

        monkeypatch.setattr(async_output, name, run)
    return called


def _prepare_runner(carry_on):
    """A batch-1 async MTP decode runner (K=3) on the _prepare_inputs stand-in
    of the runner tests, its sequence lengths staged on 'TPU'."""
    from runner_input_test_utils import _make_runner

    from vllm_torchtpu.runner.speculative_decoding_manager import (
        SpeculativeDecodingManager,
    )
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = _make_runner(
        num_computed_tokens=[20],
        prompt_tokens=[20],
        scheduled_tokens=[D],
        token_paddings=[4, 8, 16],
        pcp_size=1,
    )
    runner.scheduler_config.async_scheduling = True
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
    runner.spec_decode_manager = SpeculativeDecodingManager(runner)
    runner._decode_carry_enabled = carry_on
    runner._decode_carry_state = None
    runner._decode_carry_rej = None
    runner._mrope_np_view = None
    runner._prepare_async_token_substitution_indices = types.MethodType(
        TPUModelRunner._prepare_async_token_substitution_indices, runner
    )
    runner._async_subst_prep_cache = {}
    runner._last_sequence_layout_plan = None
    runner._async_subst_dev_cache = {}
    runner._cached_subst_key = None
    runner._cached_substitution_indices = None
    runner.seq_lens_cpu = runner.seq_lens_cpu.as_subclass(_TpuTyped)
    return runner


def _results(runner, rejected, step):
    """The previous step's async results for the one request: `rejected` of its
    K drafts rejected, its next tokens distinct per step."""
    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncPreResults

    bound = runner.max_num_reqs * D
    pre = AsyncPreResults(
        req_ids=["req0"],
        next_tokens_tpu=torch.arange(bound, dtype=torch.int32) + 100 * step,
        request_seq_lens=[],
        discard_sampled_tokens_req_indices=[],
        req_id_to_index_copy={"req0": 0},
        spec_decode_num_rejected_tokens=torch.tensor(
            [rejected] + [0] * (runner.max_num_reqs - 1), dtype=torch.int32
        ),
        num_draft_per_req={0: K},
    )
    dc.tag_results(runner, pre)
    return pre


def _decode_steps(called, *, carry_on, verify_drafts_at_last_step=True):
    """Two steady async-spec decode steps through _prepare_inputs, as the
    runner runs them: step N consumes step N-1's results (2 drafts rejected),
    _modify_prev_results reads that count back, and step N+1 consumes step N's
    results (1 rejected), the host having advanced by 1 + K - 2. Returns the
    runner, each step's (seq_lens, position_ids, input_ids, draft_token_ids)
    and the subtraction programs called per step."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = _prepare_runner(carry_on)
    outs, calls = [], []
    pre_prev = _results(runner, rejected=2, step=0)
    for step, verify in ((1, True), (2, verify_drafts_at_last_step)):
        runner._pre_async_results = pre_prev
        scheduler_output = types.SimpleNamespace(
            total_num_scheduled_tokens=D,
            num_scheduled_tokens={"req0": D},
            scheduled_spec_decode_tokens={"req0": [1, 2, 3]} if verify else {},
        )
        del called[:]
        out = TPUModelRunner._prepare_inputs(runner, scheduler_output, 0, 0)
        md = out[7]
        outs.append(
            (
                out[0]["layer.0"].seq_lens.tolist(),
                runner.position_ids.tolist(),
                runner.input_ids.tolist(),
                None if md is None else md.draft_token_ids.tolist(),
            )
        )
        calls.append(list(called))
        rejected = int(pre_prev.spec_decode_num_rejected_tokens[0])
        dc.record_rejected(runner, pre_prev, {"req0": rejected})
        pre_prev = _results(runner, rejected=1, step=step)
        runner.input_batch.num_computed_tokens_cpu[0] += D - rejected
    return runner, outs, calls


def test_prepare_inputs_carries_a_steady_step_like_the_upload_path(
    eager_subtract_programs,
):
    """With the carry on, the second steady step feeds the previous step's
    corrected device tensors to the carry program instead of uploading, and
    its sequence lengths, positions, input ids and drafts equal the upload
    path's."""
    upload, upload_outs, upload_calls = _decode_steps(
        eager_subtract_programs, carry_on=False
    )
    carried, carried_outs, carried_calls = _decode_steps(
        eager_subtract_programs, carry_on=True
    )
    assert upload_calls == [["_compiled_subtract_and_substitute_draft_token_ids"]] * 2
    assert carried_calls == [
        ["_compiled_subtract_and_substitute_draft_token_ids"],
        ["_compiled_carry_subtract_and_substitute_draft_token_ids"],
    ]
    assert carried_outs == upload_outs
    # 20 computed + 4 scheduled - 2 rejected, then + 4 - 1.
    assert upload_outs[0][0][0] == 22 and upload_outs[1][0][0] == 25
    assert carried._decode_carry_state is not None
    assert upload._decode_carry_state is None


def test_prepare_inputs_reuploads_when_the_fused_path_is_not_taken(
    eager_subtract_programs,
):
    """A carried step whose subtraction does not take the draft-token program
    (no drafts to verify) uploads its inputs after all, matches the upload path
    and clears the carried state."""
    _, upload_outs, _ = _decode_steps(
        eager_subtract_programs, carry_on=False, verify_drafts_at_last_step=False
    )
    carried, carried_outs, carried_calls = _decode_steps(
        eager_subtract_programs, carry_on=True, verify_drafts_at_last_step=False
    )
    assert carried_calls[1] == ["_compiled_subtract_and_substitute_placeholder_token"]
    assert carried_outs == upload_outs
    assert carried._decode_carry_state is None
