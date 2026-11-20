# Copyright 2025 Google LLC
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

import contextlib
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from runner_output_test_utils import (
    _assert_same_logprobs_lists,
    _blocking_logprobs_lists,
)
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner import tpu_runner_async_output as async_output
from vllm_torchtpu.runner.tpu_runner import _spec_logprob_row_tokens
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    LogprobsChunk,
    _compiled_assemble_spec_and_rejected,
    _compiled_extract_draft_token_ids,
    _compiled_subtract_num_rejected_tokens,
    assemble_spec_next_tokens,
    compute_num_rejected,
    extract_draft_token_ids,
    logprobs_lists,
    subtract_num_rejected_tokens,
)


@pytest.mark.tpu_test
def test_assemble_spec_next_tokens(device):
    # 2 requests, K=3. Rejection rows: req0 has 2 accepted drafts + bonus (3
    # valid cols), req1 has bonus only (1 valid). Bonus = last non-(-1) col.
    next_tokens = torch.tensor(
        [
            [11, 12, 99, INVALID_TOKEN_ID],
            [77, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
        ],
        dtype=torch.int32,
        device=device,
    )
    drafts = torch.tensor(
        [[201, 202, 203], [211, 212, 213]], dtype=torch.int32, device=device
    )

    src = assemble_spec_next_tokens(next_tokens, drafts, num_reqs=2)

    # Per req: [bonus, draft_1..K], row-major flat. bonus = [99, 77].
    expected = torch.tensor(
        [99, 201, 202, 203, 77, 211, 212, 213], dtype=torch.int32, device=device
    )
    assert torch.equal(src, expected)


@pytest.mark.tpu_test
def test_assemble_spec_next_tokens_ignores_padded_rows(device):
    # next_tokens / drafts are bucket-padded to 3 rows; only 2 real requests.
    next_tokens = torch.tensor(
        [[5, INVALID_TOKEN_ID], [6, INVALID_TOKEN_ID], [0, INVALID_TOKEN_ID]],
        dtype=torch.int32,
        device=device,
    )
    drafts = torch.tensor([[1], [2], [9]], dtype=torch.int32, device=device)

    src = assemble_spec_next_tokens(next_tokens, drafts, num_reqs=2)

    # K=1; bonus = [5, 6]; padded 3rd row dropped.
    expected = torch.tensor([5, 1, 6, 2], dtype=torch.int32, device=device)
    assert torch.equal(src, expected)


@pytest.mark.tpu_test
def test_subtract_num_rejected_tokens(device):
    # 2 reqs, K=3, n_sched=4 each → positions flat length 8. req0 rejected 1
    # draft, req1 rejected none; seq_lens/positions advanced optimistically.
    seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [6, 7, 8, 9, 16, 17, 18, 19], dtype=torch.int32, device=device
    )
    num_rejected = torch.tensor([1, 0], dtype=torch.int32, device=device)
    seq_idx = torch.tensor([0, 1], dtype=torch.int32, device=device)
    pos_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)

    new_seq, new_pos = subtract_num_rejected_tokens(
        seq_lens, positions, num_rejected, seq_idx, pos_idx
    )

    # req0 over-counted by 1 (corrected), req1 unchanged.
    assert new_seq.tolist() == [9, 20]
    assert new_pos.tolist() == [5, 6, 7, 8, 16, 17, 18, 19]


@pytest.mark.tpu_test
def test_subtract_num_rejected_tokens_skips_invalid(device):
    # A -1 index leaves the slot unchanged (e.g. a new request with no prior
    # spec step).
    seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=device)
    positions = torch.tensor([5, 6], dtype=torch.int32, device=device)
    num_rejected = torch.tensor([2], dtype=torch.int32, device=device)
    seq_idx = torch.tensor([-1, 0], dtype=torch.int32, device=device)
    pos_idx = torch.tensor([-1, 0], dtype=torch.int32, device=device)

    new_seq, new_pos = subtract_num_rejected_tokens(
        seq_lens, positions, num_rejected, seq_idx, pos_idx
    )

    assert new_seq.tolist() == [10, 18]  # slot 0 unchanged, slot 1 -= 2
    assert new_pos.tolist() == [5, 4]


@pytest.mark.tpu_test
def test_extract_draft_token_ids(device):
    input_ids = torch.tensor(
        [10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int32, device=device
    )
    # logits_indices picks the sampled+draft positions; target+1 selects the
    # draft positions among them.
    logits_indices = torch.tensor([1, 2, 3, 5, 6], dtype=torch.int32, device=device)
    target_logits_indices = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)

    out = extract_draft_token_ids(input_ids, logits_indices, target_logits_indices)

    # input_ids[logits_indices] = [11, 12, 13, 15, 16]; then index [1, 2, 4]
    # (target + 1) -> [12, 13, 16].
    expected = torch.tensor([12, 13, 16], dtype=torch.int32, device=device)
    assert torch.equal(out, expected)


@pytest.mark.tpu_test
def test_compute_num_rejected(device):
    # K=3. req0: 3 accepted + bonus (4 valid) -> 0 rejected; req1: 1 accepted +
    # bonus (2 valid) -> 2; req2: bonus only (1 valid) -> 3.
    next_tokens = torch.tensor(
        [[10, 11, 12, 99], [20, 21, -1, -1], [30, -1, -1, -1]],
        dtype=torch.int32,
        device=device,
    )
    num_draft = torch.tensor([3, 3, 3], dtype=torch.int32, device=device)

    out = compute_num_rejected(next_tokens, num_draft)

    assert out.tolist() == [0, 2, 3]


@pytest.mark.tpu_test
def test_compute_num_rejected_variable_draft(device):
    # Mixed scheduled draft counts (near max-model-len). req0 K=3 all accepted
    # -> 0; req1 K=1, draft rejected (1 valid = bonus only) -> 1.
    next_tokens = torch.tensor(
        [[10, 11, 12, 99], [20, -1, -1, -1]], dtype=torch.int32, device=device
    )
    num_draft = torch.tensor([3, 1], dtype=torch.int32, device=device)

    out = compute_num_rejected(next_tokens, num_draft)

    assert out.tolist() == [0, 1]


# Spec-decode logprobs. A verify chunk of 3 real requests padded to 4, K=3:
# drafts [3, 1, 0], so request 0 owns logits rows 0-3, request 1 rows 4-5 and
# request 2 row 6; row 7 is padding. Request 0 accepted 1 draft + bonus,
# request 1 both slots, request 2 its bonus. The padded request carries junk.
_VERIFY_NEXT_TOKENS = [
    [11, 12, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
    [21, 22, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
    [31, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
    [99, 98, 97, 96],
]
_VERIFY_DRAFT_LENGTHS = [3, 1, 0, 0]
_VERIFY_BONUS_INDICES = [3, 5, 6, 0]


def test_spec_logprob_row_tokens_places_each_slot_on_its_row():
    row_tokens = _spec_logprob_row_tokens(
        torch.tensor(_VERIFY_NEXT_TOKENS, dtype=torch.int32),
        torch.tensor(_VERIFY_BONUS_INDICES, dtype=torch.int32),
        torch.tensor(_VERIFY_DRAFT_LENGTHS, dtype=torch.int32),
        torch.zeros(8, dtype=torch.int32),
    )
    assert row_tokens.shape == (8, 1)
    assert row_tokens.dtype == torch.int64
    # Rejected slots, the padded request and the padding row read token 0;
    # in particular the padded request's junk does not overwrite row 0.
    assert row_tokens.flatten().tolist() == [11, 12, 0, 0, 21, 22, 31, 0]


def _row_logprobs(num_rows, base=0):
    """Rows whose values encode their own index, max_num_logprobs=1."""
    rows = torch.arange(num_rows, dtype=torch.int32) + base
    return LogprobsTensors(
        logprob_token_ids=torch.stack([rows, rows + 1000], dim=1),
        logprobs=torch.stack([-rows.float(), -rows.float() - 0.5], dim=1),
        selected_token_ranks=rows.clone(),
    )


def test_logprobs_lists_keeps_accepted_rows_in_request_order():
    verify = LogprobsChunk(_row_logprobs(8), row_starts=np.array([0, 4, 6]))
    no_drafts = LogprobsChunk(_row_logprobs(4, base=100), np.arange(2))
    out = logprobs_lists([verify, no_drafts], gen_lens=[2, 2, 1, 1, 1])
    assert out.cu_num_generated_tokens == [0, 2, 4, 5, 6, 7]
    assert out.logprob_token_ids[:, 0].tolist() == [0, 1, 4, 5, 6, 100, 101]
    assert out.sampled_token_ranks.tolist() == [0, 1, 4, 5, 6, 100, 101]
    # The scheduler takes each request's rows with slice_request.
    assert out.slice_request(1, 2).logprob_token_ids[:, 0].tolist() == [4, 5]
    assert out.slice_request(3, 1).logprob_token_ids[:, 0].tolist() == [100]


def test_get_output_attaches_spec_logprobs():
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor(_VERIFY_NEXT_TOKENS, dtype=torch.int32)],
        chunk_real_lens=[3],
        copy_ready_event=None,
        completed=True,
        logprobs=[LogprobsChunk(_row_logprobs(8), row_starts=np.array([0, 4, 6]))],
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["a", "b", "c"],
            req_id_to_index={"a": 0, "b": 1, "c": 2},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state,
        # A discarded request keeps its rows in the layout, like vLLM's
        # rejection sampler; the scheduler never reads them.
        discard_sampled_tokens_req_indices=[1],
    ).get_output()
    assert out.sampled_token_ids == [[11, 12], [], [31]]
    assert out.logprobs.cu_num_generated_tokens == [0, 2, 4, 5]
    assert out.logprobs.slice_request(0, 2).logprob_token_ids[:, 0].tolist() == [
        0,
        1,
    ]
    assert out.logprobs.slice_request(2, 1).logprobs[:, 0].tolist() == [-6.0]


def _bucketed_logprobs(num_rows, num_logprobs, seed):
    """Logprobs of a bucketed (padded) decode chunk, as gather_logprobs
    returns them."""
    g = torch.Generator().manual_seed(seed)
    return LogprobsTensors(
        logprob_token_ids=torch.randint(
            0, 1000, (num_rows, num_logprobs + 1), generator=g, dtype=torch.int64
        ).to(torch.int32),
        logprobs=-torch.rand((num_rows, num_logprobs + 1), generator=g),
        selected_token_ranks=torch.randint(
            1, 100, (num_rows,), generator=g, dtype=torch.int64
        ).to(torch.int32),
    )


@pytest.mark.parametrize("num_logprobs", [0, 20])
@pytest.mark.parametrize("real_rows", [[5], [16, 3], [16, 16, 7]])
def test_logprobs_lists_without_gen_lens_matches_the_blocking_path(
    num_logprobs, real_rows
):
    """gen_lens=None, one generated token per request: each chunk's first `n`
    bucketed rows, in chunk order."""
    chunks = [
        LogprobsChunk(_bucketed_logprobs(16, num_logprobs, seed), np.arange(n))
        for seed, n in enumerate(real_rows)
    ]
    out = logprobs_lists(chunks, gen_lens=None)
    _assert_same_logprobs_lists(
        out,
        _blocking_logprobs_lists([(c.tensors, len(c.row_starts)) for c in chunks]),
    )


def test_get_output_attaches_plain_logprobs():
    logprobs = _bucketed_logprobs(8, 0, 0)
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[7], [8], [9], [0], [0], [0], [0], [0]])],
        chunk_real_lens=[3],
        copy_ready_event=None,
        completed=True,
        logprobs=[LogprobsChunk(logprobs, np.arange(3))],
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["a", "b", "c"],
            req_id_to_index={"a": 0, "b": 1, "c": 2},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state,
        discard_sampled_tokens_req_indices=[1],
    ).get_output()
    assert out.sampled_token_ids == [[7], [], [9]]
    _assert_same_logprobs_lists(out.logprobs, _blocking_logprobs_lists([(logprobs, 3)]))


@pytest.mark.parametrize(
    "num_rows, logprobs",
    [
        (2, None),
        # No real rows: get_output returns before it assembles logprobs.
        (0, [LogprobsChunk(_bucketed_logprobs(4, 0, 1), np.arange(0))]),
    ],
    ids=["no-logprobs-requested", "empty-output"],
)
def test_get_output_leaves_logprobs_none(num_rows, logprobs):
    req_ids = ["a", "b"][:num_rows]
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[1], [2], [0], [0]])],
        chunk_real_lens=[num_rows],
        copy_ready_event=None,
        completed=True,
        logprobs=logprobs,
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=copy_state,
        discard_sampled_tokens_req_indices=[],
    ).get_output()
    assert out.sampled_token_ids == [[1], [2]][:num_rows]
    assert out.logprobs is None


def test_from_device_chunks_copies_logprobs_and_waits_for_them(monkeypatch):
    calls = []
    event = MagicMock()
    event.record.side_effect = lambda: calls.append("event.record")
    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=lambda: event), raising=False
    )
    to_cpu_nonblocking = LogprobsTensors.to_cpu_nonblocking
    host_copies = []

    def spy_to_cpu_nonblocking(tensors):
        calls.append("logprobs.to_cpu_nonblocking")
        # On CPU the real method returns its input. Return a new tuple, as a
        # copy from the device does, so the checks below can tell the host
        # copy from the device tensors.
        host_copies.append(to_cpu_nonblocking(tensors)._replace())
        return host_copies[-1]

    monkeypatch.setattr(LogprobsTensors, "to_cpu_nonblocking", spy_to_cpu_nonblocking)
    tokens = [torch.zeros((8, 1), dtype=torch.int32)]
    device_logprobs = _bucketed_logprobs(8, 2, 3)
    state = AsyncTPUCopyState.from_device_chunks(
        tokens, [5], logprobs=[LogprobsChunk(device_logprobs, np.arange(5))]
    )
    # The logprobs copy is issued before the copy event is recorded, so the
    # wait on that event covers it, and the state keeps the host copy. The
    # device sources stay alive until the event is waited on.
    assert calls == ["logprobs.to_cpu_nonblocking", "event.record"]
    (host,) = state.logprobs
    assert host.tensors is host_copies[0]
    assert host.tensors is not device_logprobs
    assert np.array_equal(host.row_starts, np.arange(5))
    assert state._logprobs_tpu is not None
    assert state._logprobs_tpu[0] is device_logprobs
    state.wait()
    event.synchronize.assert_called_once_with()
    assert state._logprobs_tpu is None
    empty = AsyncTPUCopyState.from_device_chunks(tokens, [5])
    assert empty.logprobs is None and empty._logprobs_tpu is None
    # get_output needs one row start per real request in each chunk.
    short = LogprobsChunk(device_logprobs, np.arange(4))
    with pytest.raises(AssertionError):
        AsyncTPUCopyState.from_device_chunks(tokens, [5], logprobs=[short])


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_logprobs", [0, 2])
@pytest.mark.parametrize("real_rows", [[5], [8, 3]])
def test_deferred_logprobs_from_device_match_a_blocking_cpu_copy(
    device, real_rows, num_logprobs
):
    """Bucketed logprobs on the device, copied by from_device_chunks next to
    the sampled tokens: get_output() returns the same rows, values and dtypes
    as a blocking .cpu() copy of the same device tensors, cut to the real
    rows and concatenated the way sample_tokens did before the copy was
    deferred."""
    tokens = [
        (torch.arange(8, dtype=torch.int32).view(8, 1) + 100 * i).to(device)
        for i in range(len(real_rows))
    ]
    on_device = [
        LogprobsTensors(
            *(t.to(device) for t in _bucketed_logprobs(8, num_logprobs, seed)[:3])
        )
        for seed in range(len(real_rows))
    ]
    state = AsyncTPUCopyState.from_device_chunks(
        tokens,
        real_rows,
        logprobs=[LogprobsChunk(t, np.arange(n)) for t, n in zip(on_device, real_rows)],
    )
    # Host tensors from the start; get_output() reads them after it waits.
    assert all(c.tensors.logprobs.device.type == "cpu" for c in state.logprobs)
    req_ids = [f"r{i}" for i in range(sum(real_rows))]
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[],
            logprobs=None,
        ),
        copy_state=state,
        discard_sampled_tokens_req_indices=[],
    ).get_output()
    blocking = [
        (LogprobsTensors(*(x.cpu() for x in t[:3])), n)
        for t, n in zip(on_device, real_rows)
    ]
    _assert_same_logprobs_lists(out.logprobs, _blocking_logprobs_lists(blocking))
    assert out.sampled_token_ids == [
        [100 * i + j] for i, n in enumerate(real_rows) for j in range(n)
    ]


def _eagle3_drafter_stand_in(**fields):
    """An Eagle3Proposer carrying only the given fields: the runner's fused
    assembly takes its fast path only for a bucketed Eagle3 drafter."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    drafter = object.__new__(Eagle3Proposer)
    for name, value in fields.items():
        setattr(drafter, name, value)
    return drafter


@pytest.mark.tpu_test
def test_compiled_async_spec_helpers_match_eager(device):
    if device.type != "tpu":
        pytest.skip("TPU compiled helpers require TPU device")
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    # 1. _compiled_assemble_spec_and_rejected
    #    (B=4, p=8, num_reqs=2, K=3, max_num_reqs=8)
    next_tokens = torch.tensor(
        [
            [10, 11, 99, -1],
            [20, -1, -1, -1],
            [700, -1, -1, -1],
            [800, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    drafts_p = torch.zeros((8, 3), dtype=torch.int32, device=device)
    drafts_p[0] = torch.tensor([101, 102, 103], dtype=torch.int32, device=device)
    drafts_p[1] = torch.tensor([201, 202, 203], dtype=torch.int32, device=device)
    draft_lengths = torch.tensor([3, 3, 0, 0], dtype=torch.int32, device=device)
    bound = 8 * 4
    src_pad, rej_pad = _compiled_assemble_spec_and_rejected(
        next_tokens, drafts_p, draft_lengths, bound, 8
    )
    assert src_pad.shape == (bound,)
    assert rej_pad.shape == (8,)
    eager_src = assemble_spec_next_tokens(next_tokens, drafts_p, num_reqs=2)
    eager_rej = compute_num_rejected(next_tokens, draft_lengths[:2])
    assert torch.equal(src_pad[:8].cpu(), eager_src.cpu())
    assert torch.equal(rej_pad[:2].cpu(), eager_rej.cpu())

    # Also verify TPUModelRunner._assemble_async_spec_substitution fast path.
    runner = object.__new__(TPUModelRunner)
    runner.device = device
    runner.max_num_reqs = 8
    runner._zero_draft_lengths_cache = {}
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    drafts_view = drafts_p[:2]
    runner.drafter = _eagle3_drafter_stand_in(
        _last_padded_drafts_per_chunk=[drafts_p],
        _last_propose_device_out=drafts_view,
    )
    state = SimpleNamespace(
        num_reqs_list=[2],
        spec_decode_metadata_list=[
            SimpleNamespace(
                draft_lengths=draft_lengths,
                draft_lengths_cpu=np.array([3, 3], dtype=np.int32),
            )
        ],
    )
    chunks, next_idx, spec_rej, draft_counts = runner._assemble_async_spec_substitution(
        drafts_view, [next_tokens], state
    )
    assert len(chunks) == 1 and chunks[0].shape == (bound,)
    assert torch.equal(chunks[0][:8].cpu(), eager_src.cpu())
    assert torch.equal(spec_rej[:2].cpu(), eager_rej.cpu())
    assert next_idx == {0: 0, 1: 1}
    assert draft_counts == {0: 3, 1: 3}

    # 2. _compiled_subtract_num_rejected_tokens
    seq_lens = torch.tensor([10, 20, 0, 0], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [6, 7, 8, 9, 16, 17, 18, 19], dtype=torch.int32, device=device
    )
    seq_idx = torch.tensor([0, 1, -1, -1], dtype=torch.int32, device=device)
    pos_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device)
    c_seq, c_pos = _compiled_subtract_num_rejected_tokens(
        seq_lens, positions, rej_pad, seq_idx, pos_idx
    )
    e_seq, e_pos = subtract_num_rejected_tokens(
        seq_lens, positions, rej_pad, seq_idx, pos_idx
    )
    assert torch.equal(c_seq.cpu(), e_seq.cpu())
    assert torch.equal(c_pos.cpu(), e_pos.cpu())

    # 3. _compiled_extract_draft_token_ids
    input_ids = torch.arange(10, 26, dtype=torch.int32, device=device)
    logits_idx = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7] + [0] * 8, dtype=torch.int32, device=device
    )
    target_idx = torch.tensor(
        [0, 1, 2, 4, 5, 6] + [0] * 10, dtype=torch.int32, device=device
    )
    c_ext = _compiled_extract_draft_token_ids(input_ids, logits_idx, target_idx)
    e_ext = extract_draft_token_ids(input_ids, logits_idx, target_idx)
    assert torch.equal(c_ext.cpu(), e_ext.cpu())

    # 4. _compiled_pad_non_spec_next_tokens + 2-chunk _assemble_async_spec_substitution
    from vllm_torchtpu.runner.tpu_runner_async_output import (
        _compiled_pad_non_spec_next_tokens,
    )

    sel_non_spec = torch.tensor(
        [[301], [302], [0], [0]], dtype=torch.int64, device=device
    )
    padded_non_spec = _compiled_pad_non_spec_next_tokens(sel_non_spec, 3)
    assert padded_non_spec.shape == (4, 4)
    assert padded_non_spec.dtype == torch.int32
    assert padded_non_spec.cpu()[:2].tolist() == [[301, -1, -1, -1], [302, -1, -1, -1]]

    drafts_p1 = torch.tensor(
        [
            [31, 32, 33],
            [41, 42, 43],
            [0, 0, 0],
            [0, 0, 0],
        ]
        + [[0, 0, 0]] * 12,
        dtype=torch.int32,
        device=device,
    )
    runner.drafter = _eagle3_drafter_stand_in(
        _last_padded_drafts_per_chunk=[drafts_p, drafts_p1],
        _last_propose_device_out=drafts_view,
    )
    state_2chunk = SimpleNamespace(
        num_reqs_list=[2, 1],
        spec_decode_metadata_list=[
            SimpleNamespace(
                draft_lengths=draft_lengths,
                draft_lengths_cpu=np.array([3, 3], dtype=np.int32),
            ),
            None,
        ],
    )
    chunks2, next_idx2, spec_rej2, draft_counts2 = (
        runner._assemble_async_spec_substitution(
            drafts_view, [next_tokens, padded_non_spec], state_2chunk
        )
    )
    assert len(chunks2) == 1 and chunks2[0].shape == (bound,)
    assert next_idx2 == {0: 0, 1: 1, 2: 4}
    assert draft_counts2 == {0: 3, 1: 3}
    # Chunk 0 reqs 0..1 at indices 0..1
    assert torch.equal(chunks2[0][:8].cpu(), eager_src.cpu())
    assert torch.equal(spec_rej2[:2].cpu(), eager_rej.cpu())
    # Chunk 1 req 2 (prefill) at bucketed index B0=4 -> offset 4*(1+3)=16
    assert chunks2[0][16:20].cpu().tolist() == [301, 31, 32, 33]
    assert int(spec_rej2[4].cpu()) == 0


def test_subtract_and_substitute_draft_token_ids_and_two_chunk_mamba_offsets():
    """Verify fused `_compiled_subtract_and_substitute_draft_token_ids`,
    `_compiled_subtract_and_substitute_placeholder_token`, and 2-chunk Mamba
    read-offset fusion match the separate reference operations."""
    from vllm_torchtpu.runner.tpu_runner import (
        _compiled_assemble_two_chunks_spec_rejected_and_read_offsets,
    )
    from vllm_torchtpu.runner.tpu_runner_async_output import (
        _compiled_subtract_and_substitute_draft_token_ids,
        _compiled_subtract_and_substitute_placeholder_token,
        extract_draft_token_ids,
        subtract_and_substitute_draft_token_ids,
        subtract_and_substitute_placeholder_token,
        subtract_num_rejected_tokens,
    )

    seq_lens = torch.tensor([12, 24, 0, 0], dtype=torch.int32)
    positions_1d = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int32)
    positions_2d = positions_1d.unsqueeze(0).expand(3, -1).clone()
    num_rejected = torch.tensor([1, 2, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    seq_idx = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    pos_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)

    from vllm_torchtpu.runner.tpu_runner import _padded_substitution_indices

    input_ids = torch.tensor([-1, -1, -1, -1, -1, -1, 777, 888], dtype=torch.int32)
    # Req 0 occupies tokens 0..3 (from next_tokens_tpu[0..3]), Req 1 occupies
    # 4..5 (from next_tokens_tpu[4..5]), slots 6..7 are padded
    subst_indices = torch.from_numpy(
        _padded_substitution_indices(
            8,
            np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
            np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        )
    )
    next_tokens_tpu = torch.tensor(
        [100, 101, 102, 103, 200, 201, 202, 203] + [999] * 24, dtype=torch.int32
    )
    logits_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    target_logits_indices = torch.tensor([0, 1, 2, 4, 0, 0, 0, 0], dtype=torch.int32)

    ref_sl, ref_pos1 = subtract_num_rejected_tokens(
        seq_lens.clone(), positions_1d.clone(), num_rejected, seq_idx, pos_idx
    )
    _, ref_pos2 = subtract_num_rejected_tokens(
        seq_lens.clone(), positions_2d.clone(), num_rejected, seq_idx, pos_idx
    )
    ref_ids = torch.tensor([100, 101, 102, 103, 200, 201, 777, 888], dtype=torch.int32)
    ref_drafts = extract_draft_token_ids(ref_ids, logits_indices, target_logits_indices)

    for fn_pl, fn_dr in (
        (
            subtract_and_substitute_placeholder_token,
            subtract_and_substitute_draft_token_ids,
        ),
        (
            _eager(_compiled_subtract_and_substitute_placeholder_token),
            _eager(_compiled_subtract_and_substitute_draft_token_ids),
        ),
    ):
        sl_a, pos_a, ids_a = fn_pl(
            seq_lens.clone(),
            positions_1d.clone(),
            num_rejected,
            seq_idx,
            pos_idx,
            input_ids.clone(),
            subst_indices,
            next_tokens_tpu,
        )
        assert torch.equal(sl_a, ref_sl)
        assert torch.equal(pos_a, ref_pos1)
        assert torch.equal(ids_a, ref_ids)

        sl_b, pos_b, ids_b, drafts_b = fn_dr(
            seq_lens.clone(),
            positions_2d.clone(),
            num_rejected,
            seq_idx,
            pos_idx,
            input_ids.clone(),
            subst_indices,
            next_tokens_tpu,
            logits_indices,
            target_logits_indices,
        )
        assert torch.equal(sl_b, ref_sl)
        assert torch.equal(pos_b, ref_pos2)
        assert torch.equal(ids_b, ref_ids)
        assert torch.equal(drafts_b, ref_drafts)

    # Verify 2-chunk fused assembly + mamba read offsets
    nt0 = torch.tensor(
        [[50, 51, -1, -1], [60, -1, -1, -1], [0, -1, -1, -1], [0, -1, -1, -1]],
        dtype=torch.int32,
    )
    d0 = torch.zeros((8, 3), dtype=torch.int32)
    d0[0] = torch.tensor([501, 502, 503], dtype=torch.int32)
    d0[1] = torch.tensor([601, 602, 603], dtype=torch.int32)
    dl0 = torch.tensor([3, 3, 0, 0], dtype=torch.int32)
    nt1 = torch.tensor(
        [[70, -1, -1, -1], [0, -1, -1, -1], [0, -1, -1, -1], [0, -1, -1, -1]],
        dtype=torch.int32,
    )
    d1 = torch.zeros((8, 3), dtype=torch.int32)
    d1[0] = torch.tensor([701, 702, 703], dtype=torch.int32)
    dl1 = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    read_offsets = torch.zeros((8,), dtype=torch.int32)
    mamba_idx = torch.tensor([[2, 5, 0, 0]], dtype=torch.int32)

    flat2, rej2 = _eager(_compiled_assemble_two_chunks_spec_rejected_and_read_offsets)(
        nt0, d0, dl0, nt1, d1, dl1, read_offsets, mamba_idx, 32, 8
    )
    assert flat2[:4].tolist() == [51, 501, 502, 503]
    assert flat2[4:8].tolist() == [60, 601, 602, 603]
    assert flat2[16:20].tolist() == [70, 701, 702, 703]
    assert rej2[:2].tolist() == [2, 3]
    # Accepted counts - 1: req 0 accepted 2 tokens -> offset 1; req 1 accepted
    # 1 token -> offset 0
    assert int(read_offsets[2]) == 1
    assert int(read_offsets[5]) == 0


class _TpuTyped(torch.Tensor):
    """A CPU tensor whose ``device`` reads as a TPU device, so the runner's
    TPU-only assembly branches run on CPU (with the programs' eager bodies)."""

    @property
    def device(self):
        return SimpleNamespace(type="tpu")


def _eager(program):
    return program.__wrapped__._torchdynamo_orig_callable


def _verify_output():
    """A 4-row rejection output (K = 3): 3, 0 and 1 drafts accepted, one
    padding row."""
    return torch.tensor(
        [
            [10, 11, 12, 13],
            [20, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
            [30, 31, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
            [0, INVALID_TOKEN_ID, INVALID_TOKEN_ID, INVALID_TOKEN_ID],
        ],
        dtype=torch.int32,
    )


def _padded_drafts(rows, k=3):
    return 100 * torch.arange(1, rows + 1, dtype=torch.int32).unsqueeze(
        1
    ) + torch.arange(k, dtype=torch.int32)


@pytest.mark.parametrize("draft_rows", [2, 4, 8])
@pytest.mark.parametrize("num_groups", [1, 2])
def test_one_chunk_assembly_with_read_offsets_matches_the_separate_programs(
    draft_rows, num_groups
):
    """The one-chunk program that folds in the mamba read-offset scatter
    gives the eager assembly's substitution source and rejected counts and
    the verify scatter's read offsets, for drafts with fewer, as many and
    more rows than the sampled output, and one or two mamba groups."""
    from vllm_torchtpu.runner.tpu_runner import (
        _compiled_assemble_spec_rejected_and_read_offsets,
        _update_read_offsets_verify_impl,
    )
    from vllm_torchtpu.spec_decode.utils import fit_rows

    k, max_num_reqs = 3, 8
    bound = max_num_reqs * (1 + k)
    nt = _verify_output()
    drafts = _padded_drafts(draft_rows)
    draft_lengths = torch.tensor([3, 3, 3, 0], dtype=torch.int32)
    groups = torch.tensor(
        [[5, 7, 9, 0, 0, 0, 0, 0], [11, 13, 15, 0, 0, 0, 0, 0]], dtype=torch.int32
    )
    stacked = groups[0] if num_groups == 1 else groups
    offsets = torch.full((16,), -7, dtype=torch.int32)
    src, rejected = _eager(_compiled_assemble_spec_rejected_and_read_offsets)(
        nt, drafts, draft_lengths, offsets, stacked, bound, max_num_reqs
    )
    want_offsets = torch.full((16,), -7, dtype=torch.int32)
    _update_read_offsets_verify_impl(want_offsets, stacked, nt)
    assert torch.equal(offsets, want_offsets)
    assert offsets[[5, 7, 9]].tolist() == [3, 0, 1]
    rows = nt.shape[0] * (1 + k)
    assert src.shape == (bound,)
    assert torch.equal(src[:rows], assemble_spec_next_tokens(nt, fit_rows(drafts, 4)))
    assert not src[rows:].any()
    assert rejected.shape == (max_num_reqs,)
    assert torch.equal(rejected[:4], compute_num_rejected(nt, draft_lengths))
    assert not rejected[4:].any()


def _assembly_runner(drafts_view, padded):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = object.__new__(TPUModelRunner)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 8
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    runner._adaptive_verification = None
    runner._zero_draft_lengths_cache = {}
    runner.drafter = _eagle3_drafter_stand_in(
        _last_padded_drafts_per_chunk=padded, _last_propose_device_out=drafts_view
    )
    return runner


def test_assembly_folds_deferred_read_offsets_into_the_one_chunk_program(
    monkeypatch,
):
    """With deferred mamba indices, the one-chunk assembly runs the program
    that scatters the read offsets too (not the separate scatter), with the
    same source, rejected counts and offsets as the general path."""
    from vllm_torchtpu.runner.tpu_runner import (
        TPUModelRunner,
        _compiled_assemble_spec_rejected_and_read_offsets,
    )

    ran = []

    def fused(*args):
        ran.append("fused")
        return _eager(_compiled_assemble_spec_rejected_and_read_offsets)(*args)

    def separate(*_args, **_kwargs):
        raise AssertionError("the separate read-offset scatter ran")

    monkeypatch.setattr(
        tpu_runner, "_compiled_assemble_spec_rejected_and_read_offsets", fused
    )
    monkeypatch.setattr(TPUModelRunner, "_update_mamba_slot_read_offsets", separate)
    padded = [_padded_drafts(8)]
    drafts_view = padded[0][:3]
    runner = _assembly_runner(drafts_view, padded)
    runner.mamba_slot_read_offsets = torch.zeros(16, dtype=torch.int32)
    md = SimpleNamespace(
        draft_lengths=torch.tensor([3, 3, 3, 0], dtype=torch.int32),
        draft_lengths_cpu=np.array([3, 3, 3], dtype=np.int32),
    )
    state = SimpleNamespace(num_reqs_list=[3], spec_decode_metadata_list=[md])
    indices = torch.tensor([5, 7, 9, 0, 0, 0, 0, 0], dtype=torch.int32)
    nt = _verify_output()
    chunks, next_idx, rejected, counts = runner._assemble_async_spec_substitution(
        drafts_view,
        [nt.as_subclass(_TpuTyped)],
        state,
        deferred_mamba_indices=[indices],
    )
    assert ran == ["fused"]
    assert runner.mamba_slot_read_offsets[[5, 7, 9]].tolist() == [3, 0, 1]
    assert next_idx == {0: 0, 1: 1, 2: 2} and counts == {0: 3, 1: 3, 2: 3}
    want = assemble_spec_next_tokens(nt, padded[0], 3)
    assert torch.equal(chunks[0][: want.shape[0]], want)
    assert (
        rejected[:3].tolist() == compute_num_rejected(nt, md.draft_lengths[:3]).tolist()
    )


@pytest.mark.parametrize("draft_rows", [4, 8])
def test_prefill_bootstrap_program_matches_the_general_bootstrap(
    monkeypatch, draft_rows
):
    """The compiled bootstrap (its eager body) parks each request's sampled
    token and its drafts where the general bootstrap does, with no rejected
    drafts and no drafts counted for this step."""
    monkeypatch.setattr(
        async_output,
        "_compiled_assemble_prefill_bootstrap",
        _eager(async_output._compiled_assemble_prefill_bootstrap),
    )
    n, k = 3, 3
    padded = [_padded_drafts(draft_rows)]
    drafts_view = padded[0][:n]
    selected = torch.tensor([[71], [72], [73], [0]], dtype=torch.int64)
    general = _assembly_runner(drafts_view, padded)._assemble_async_prefill_bootstrap(
        drafts_view, [selected], [n]
    )
    fused = _assembly_runner(drafts_view, padded)._assemble_async_prefill_bootstrap(
        drafts_view, [selected.as_subclass(_TpuTyped)], [n]
    )
    assert len(general[0]) == 1 and len(fused[0]) == 1
    assert fused[0][0].shape == (8 * (1 + k),)
    assert fused[1] == general[1] == {0: 0, 1: 1, 2: 2}
    assert fused[3] == general[3] == {0: 0, 1: 0, 2: 0}
    for src, _, _, _ in (general, fused):
        assert [src[0][r * (1 + k) : (r + 1) * (1 + k)].tolist() for r in range(n)] == [
            [71 + r] + padded[0][r].tolist() for r in range(n)
        ]
    assert not general[2].any() and not fused[2].any()


_ASSEMBLY_PROGRAMS = (
    "_compiled_pad_non_spec_next_tokens",
    "_compiled_assemble_prefill_bootstrap",
    "_compiled_assemble_spec_and_rejected",
    "_compiled_assemble_two_chunks_spec_and_rejected",
    "_compiled_subtract_num_rejected_tokens",
    "_compiled_subtract_and_substitute_placeholder_token",
    "_compiled_subtract_and_substitute_draft_token_ids",
    "_compiled_carry_subtract_and_substitute_draft_token_ids",
)
# The rejected-draft subtraction programs, warmed at every token bucket.
_SUBTRACT_PROGRAMS = (
    "_compiled_subtract_num_rejected_tokens",
    "_compiled_subtract_and_substitute_placeholder_token",
    "_compiled_subtract_and_substitute_draft_token_ids",
)


def _assembly_warmup_calls(
    monkeypatch,
    *,
    async_scheduling=True,
    most=None,
    async_drafter=True,
    runner_fields=None,
    with_dtypes=False,
    with_modes=False,
    programs=None,
):
    """Run _precompile_async_spec_assembly on a stub runner and record the
    shapes (and, `with_dtypes`, the dtypes) each assembly program was warmed
    with. `with_modes` records each call's grad mode and arguments instead;
    `programs` runs the named programs in place of the recorder."""
    calls: dict[str, list[tuple]] = {name: [] for name in _ASSEMBLY_PROGRAMS}

    def spec(a):
        if not isinstance(a, torch.Tensor):
            return a
        return (tuple(a.shape), a.dtype) if with_dtypes else tuple(a.shape)

    def recorder(name):
        def record(*args):
            if with_modes:
                calls[name].append((torch.is_grad_enabled(), args))
            else:
                calls[name].append(tuple(spec(a) for a in args))
            return (torch.zeros(1),)

        return record

    for name in _ASSEMBLY_PROGRAMS:
        program = (programs or {}).get(name)
        monkeypatch.setattr(async_output, name, program or recorder(name))
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    runner = SimpleNamespace(
        _is_async_drafter=async_drafter,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        max_num_reqs=8,
        num_reqs_paddings=[2, 4, 8],
        num_tokens_paddings=[8, 16, 32],
        num_reqs_max_model_len=8,
        most_model_len=None if most is None else 128,
        num_reqs_most_model_len=most,
        uses_mrope=False,
        device=torch.device("cpu"),
        drafter=None,
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _decode_carry_enabled=False,
        _dp_lockstep_enabled=lambda: False,
    )
    for name, value in (runner_fields or {}).items():
        setattr(runner, name, value)
    tpu_runner.TPUModelRunner._precompile_async_spec_assembly(runner)  # type: ignore[arg-type]
    return calls


# The step after a spec verify step: sample_tokens assembles the next tokens and
# the rejected counts, and execute_model (under its no_grad) subtracts them.
_SAMPLE_TOKENS_PROGRAMS = (
    "_compiled_pad_non_spec_next_tokens",
    "_compiled_assemble_prefill_bootstrap",
    "_compiled_assemble_spec_and_rejected",
    "_compiled_assemble_two_chunks_spec_and_rejected",
)
_EXECUTE_MODEL_PROGRAMS = (
    "_compiled_subtract_num_rejected_tokens",
    "_compiled_subtract_and_substitute_placeholder_token",
    "_compiled_subtract_and_substitute_draft_token_ids",
)


def test_async_spec_assembly_warmup_takes_each_programs_serving_guards(monkeypatch):
    """Dynamo guards on grad mode and on a repeated input. sample_tokens runs
    the assembly programs in its caller's grad mode and execute_model runs the
    rejected-token programs under no_grad, so each is warmed in its own mode;
    the draft-token program gets the final and target logits indices as two
    tensors, as serving passes them."""
    calls = _assembly_warmup_calls(monkeypatch, most=4, with_modes=True)
    for names, grad in (
        (_SAMPLE_TOKENS_PROGRAMS, True),
        (_EXECUTE_MODEL_PROGRAMS, False),
    ):
        for name in names:
            assert calls[name], name
            assert {mode for mode, _ in calls[name]} == {grad}, name
    for _, args in calls["_compiled_subtract_and_substitute_draft_token_ids"]:
        assert args[8] is not args[9]


class _StepStopped(Exception):
    pass


class _EntryRunner:
    """A runner stand-in that runs `call` at the entry point's first read of
    it, inside the grad mode the entry point gives its body, then stops."""

    def __init__(self, call):
        self._call = call

    def __getattr__(self, name):
        self._call()
        raise _StepStopped


def _inside(entry, call) -> None:
    """Run `call` inside TPUModelRunner.`entry` (execute_model or
    sample_tokens), in the grad mode serving gives that entry's body."""
    with pytest.raises(_StepStopped):
        getattr(tpu_runner.TPUModelRunner, entry)(_EntryRunner(call), None)


def _counting_compile(name, compiles):
    """The real program `name`, compiled whole on a backend that records each
    compile; dynamo's guards decide whether a call reuses a graph, whatever
    the backend."""
    from vllm_torchtpu.spec_decode.eagle3 import _static_shape_args

    def backend(gm, example_inputs):
        compiles.append(name)
        return gm.forward

    fn = inspect.unwrap(getattr(async_output, name))
    return _static_shape_args(
        torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
    )


def test_the_warmed_rejected_token_programs_serve_without_compiling(monkeypatch):
    """The real placeholder and draft-token programs, warmed by
    _precompile_async_spec_assembly and then called from inside execute_model
    as _prepare_inputs calls them, and the real one-chunk assembly called from
    inside sample_tokens: no serving call compiles."""
    torch._dynamo.reset()
    try:
        compiles: list[str] = []
        names = (*_EXECUTE_MODEL_PROGRAMS[1:], "_compiled_assemble_spec_and_rejected")
        programs = {name: _counting_compile(name, compiles) for name in names}
        _assembly_warmup_calls(monkeypatch, programs=programs)
        warmed = len(compiles)
        knr, n, bound, k = 8, 16, 8 * (1 + 3), 3

        def subtract_args():
            return (
                torch.zeros(knr, dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
                torch.zeros(knr, dtype=torch.int32),
                torch.from_numpy(np.full(knr, -1, dtype=np.int32)),
                torch.from_numpy(np.full(n, -1, dtype=np.int32)),
                torch.zeros(n, dtype=torch.int32),
                # Destinations in row 0, sources in row 1 (-1: padding).
                torch.from_numpy(
                    np.stack(
                        (np.zeros(n, dtype=np.int32), np.full(n, -1, dtype=np.int32))
                    )
                ),
                torch.zeros(bound, dtype=torch.int32),
            )

        def prepare_inputs():
            async_output._compiled_subtract_and_substitute_placeholder_token(
                *subtract_args()
            )
            # Final and target logits indices: two tensors of the metadata.
            async_output._compiled_subtract_and_substitute_draft_token_ids(
                *subtract_args(),
                torch.zeros(n, dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
            )

        _inside("execute_model", prepare_inputs)

        def assemble():
            async_output._compiled_assemble_spec_and_rejected(
                torch.zeros((4, k + 1), dtype=torch.int32),
                torch.zeros((8, k), dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                bound,
                knr,
            )

        _inside("sample_tokens", assemble)
        assert len(compiles) == warmed, compiles[warmed:]
    finally:
        torch._dynamo.reset()


def test_async_spec_assembly_warmup_covers_every_bucket(monkeypatch):
    """The assembly programs the step after a spec verify runs are warmed at
    every request bucket, every two-chunk split that fits max_num_reqs, and
    every token bucket for each request-count layout."""
    calls = _assembly_warmup_calls(monkeypatch, most=4)
    k, max_num_reqs, bound = 3, 8, 8 * (1 + 3)
    assert calls["_compiled_pad_non_spec_next_tokens"] == [
        ((n, 1), k) for n in (2, 4, 8)
    ]
    boot_reqs = {c[0][0] for c in calls["_compiled_assemble_prefill_bootstrap"]}
    assert boot_reqs == {2, 4, 8}
    assert all(c[2] == bound for c in calls["_compiled_assemble_prefill_bootstrap"])
    asm_reqs = {c[0][0] for c in calls["_compiled_assemble_spec_and_rejected"]}
    assert asm_reqs == {2, 4, 8}
    two = calls["_compiled_assemble_two_chunks_spec_and_rejected"]
    assert two and all(c[0][0] + c[3][0] <= max_num_reqs for c in two)
    assert {(c[0][0], c[3][0]) for c in two} >= {(2, 2), (2, 4), (4, 4)}
    for name in _SUBTRACT_PROGRAMS:
        warmed = {(c[0][0], c[1][-1]) for c in calls[name]}
        # Every token bucket, for both the max-model-len and the
        # most-model-len request layouts.
        assert warmed == {(knr, t) for knr in (8, 4) for t in (8, 16, 32)}, name


@pytest.mark.parametrize("drafter_kind", ["mtp", "dflash"])
def test_substituting_programs_warm_the_drafters_source_dtype(
    monkeypatch, drafter_kind
):
    """The step after a spec verify substitutes from the parked source, which
    keeps the drafts' dtype: int32 from the MTP/EAGLE drafter, int64 from
    DFlash's argmax. Both substituting programs are warmed with each dtype the
    drafter parks, at every request layout and token bucket; for the MTP
    drafter that stays int32 only."""
    from vllm_torchtpu.spec_decode.dflash import DFlashProposer
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    nt = torch.zeros((2, 4), dtype=torch.int32)
    argmax_drafts = torch.zeros((2, 3, 5)).argmax(dim=-1)
    assert assemble_spec_next_tokens(nt, argmax_drafts).dtype == torch.int64
    assert assemble_spec_next_tokens(nt, argmax_drafts.int()).dtype == torch.int32
    if drafter_kind == "mtp":
        drafter = _sharded_mtp_drafter(
            SimpleNamespace(
                max_num_reqs=8,
                num_reqs_paddings=[2, 4, 8],
                num_tokens_paddings=[8, 16, 32],
            ),
            lockstep=False,
        )
        assert isinstance(drafter, Eagle3Proposer)
    else:
        drafter = object.__new__(DFlashProposer)
    calls = _assembly_warmup_calls(
        monkeypatch, most=4, runner_fields=dict(drafter=drafter), with_dtypes=True
    )
    want = {torch.int32} if drafter_kind == "mtp" else {torch.int32, torch.int64}
    for name in (
        "_compiled_subtract_and_substitute_placeholder_token",
        "_compiled_subtract_and_substitute_draft_token_ids",
    ):
        warmed = {(c[0][0], c[5][0], c[7][1]) for c in calls[name]}
        assert warmed == {
            ((knr,), (t,), dt) for knr in (8, 4) for t in (8, 16, 32) for dt in want
        }, name


@pytest.mark.parametrize("uses_mrope", [False, True])
@pytest.mark.parametrize("lockstep", [False, True])
@pytest.mark.parametrize("paddings", [[4, 8, 16, 32], [2, 4, 16, 32]])
def test_decode_carry_program_is_warmed_at_every_carried_shape(
    monkeypatch, paddings, lockstep, uses_mrope
):
    """The decode carry's form of the draft-token program serves a carried step:
    one request (max_num_seqs == 1) with its bonus token and 1 to K drafts,
    padded to the first token bucket at or above that count (under EP-DP
    lockstep, to the DP-wide bucket, which can be any), at either request
    layout the stub runner gives (8 and 4), with step_delta = 1 + K baked in.
    The warm-up compiles exactly those shapes, with the arguments the
    draft-token program takes there. With K = 3 the carried counts 2 to 4 pad
    to bucket 4 alone, or to 2 and 4."""
    calls = _assembly_warmup_calls(
        monkeypatch,
        most=4,
        runner_fields=dict(
            _decode_carry_enabled=True,
            _dp_lockstep_enabled=lambda: lockstep,
            num_tokens_paddings=paddings,
            uses_mrope=uses_mrope,
        ),
    )
    k = 3
    buckets = (
        set(paddings)
        if lockstep
        else {min(t for t in paddings if t >= n) for n in range(2, k + 2)}
    )
    served = {(knr, t) for knr in (8, 4) for t in buckets}
    carry = calls["_compiled_carry_subtract_and_substitute_draft_token_ids"]
    assert {(c[0][0], c[1][-1]) for c in carry} == served
    assert len(carry) == len(served)
    assert all(c[2] == 1 + k for c in carry)
    assert all(len(c[1]) == (2 if uses_mrope else 1) for c in carry)
    draft = calls["_compiled_subtract_and_substitute_draft_token_ids"]
    assert {c[:2] + c[3:] for c in carry} == {
        c for c in draft if (c[0][0], c[1][-1]) in served
    }


def test_decode_carry_warmup_takes_the_serving_guards(monkeypatch):
    """execute_model runs the carry under its no_grad and passes the final and
    target logits indices as two tensors; dynamo guards on both, so the
    warm-up does the same."""
    calls = _assembly_warmup_calls(
        monkeypatch, with_modes=True, runner_fields=dict(_decode_carry_enabled=True)
    )
    carry = calls["_compiled_carry_subtract_and_substitute_draft_token_ids"]
    assert carry
    for grad, args in carry:
        assert not grad
        assert args[9] is not args[10]


def test_the_warmed_decode_carry_serves_a_carried_step_without_compiling(
    monkeypatch,
):
    """The real carry program, warmed by _precompile_async_spec_assembly and
    then called from inside execute_model as _prepare_inputs calls it on a
    carried step (one request, K = 3 drafts, token bucket 8): no compile."""
    name = "_compiled_carry_subtract_and_substitute_draft_token_ids"
    torch._dynamo.reset()
    try:
        compiles: list[str] = []
        _assembly_warmup_calls(
            monkeypatch,
            programs={name: _counting_compile(name, compiles)},
            runner_fields=dict(_decode_carry_enabled=True),
        )
        warmed = len(compiles)
        assert warmed
        knr, n, bound, k = 8, 8, 8 * (1 + 3), 3

        def prepare_inputs():
            async_output._compiled_carry_subtract_and_substitute_draft_token_ids(
                torch.zeros(knr, dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
                1 + k,
                torch.zeros(knr, dtype=torch.int32),
                torch.from_numpy(np.full(knr, -1, dtype=np.int32)),
                torch.from_numpy(np.full(n, -1, dtype=np.int32)),
                torch.zeros(n, dtype=torch.int32),
                torch.from_numpy(
                    np.stack(
                        (np.zeros(n, dtype=np.int32), np.full(n, -1, dtype=np.int32))
                    )
                ),
                torch.zeros(bound, dtype=torch.int32),
                # Final and target logits indices: two tensors of the metadata.
                torch.zeros(n, dtype=torch.int32),
                torch.zeros(n, dtype=torch.int32),
            )

        _inside("execute_model", prepare_inputs)
        assert len(compiles) == warmed, compiles[warmed:]
    finally:
        torch._dynamo.reset()


def test_decode_carry_program_is_not_warmed_without_the_carry(monkeypatch):
    calls = _assembly_warmup_calls(monkeypatch, most=4)
    assert not calls["_compiled_carry_subtract_and_substitute_draft_token_ids"]


def test_async_spec_assembly_warmup_needs_async_scheduling(monkeypatch):
    calls = _assembly_warmup_calls(monkeypatch, async_scheduling=False)
    assert all(not c for c in calls.values())


def test_async_spec_assembly_warmup_needs_an_async_drafter(monkeypatch):
    calls = _assembly_warmup_calls(monkeypatch, async_drafter=False)
    assert all(not c for c in calls.values())


# A 32-request engine: request buckets 8/16/32, loop buckets 16/32/64.
_LOCKSTEP_FIELDS = dict(
    max_num_reqs=32,
    num_reqs_paddings=[8, 16, 32],
    num_tokens_paddings=[16, 32, 64],
    num_reqs_max_model_len=32,
)


def _sharded_mtp_drafter(runner, lockstep):
    """A TP-sharded Eagle3 drafter on `runner`, under EP-DP lockstep or not."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    runner._dp_lockstep_enabled = lambda: lockstep
    runner._dp_step_max_reqs = 0
    drafter = object.__new__(Eagle3Proposer)
    drafter.runner = runner
    drafter._draft_replicated = False
    return drafter


def _served_assembly_shapes(runner, drafter, lockstep, max_count=None):
    """The shapes the compiled assembly takes while serving: per chunk, its
    request bucket and the drafter's padded draft rows (the program runs when
    the drafts cover the sampled rows); per two-chunk step, both chunks of one
    step, whose request buckets fit max_num_reqs. Under lockstep every step's
    DP-wide largest chunk m is walked, with every chunk of at most m requests.
    `max_count` caps a chunk's requests (a layout's SMEM-capped width)."""
    top = runner.max_num_reqs
    count_cap = top if max_count is None else max_count
    one, two = set(), set()
    for m in range(1, count_cap + 1) if lockstep else [0]:
        runner._dp_step_max_reqs = m
        chunks = set()
        for n in range(1, (m if lockstep else count_cap) + 1):
            shape = (
                tpu_runner._get_padded_num_reqs_with_upper_limit(n, top),
                drafter._loop_bucket(n),
            )
            if shape[1] >= shape[0]:
                chunks.add(shape)
        one |= chunks
        two |= {(c0, c1) for c0 in chunks for c1 in chunks if c0[0] + c1[0] <= top}
    runner._dp_step_max_reqs = 0
    return one, two


@pytest.mark.parametrize("lockstep", [False, True])
def test_async_spec_assembly_warmup_covers_the_drafters_loop_buckets(
    monkeypatch, lockstep
):
    """Under EP-DP lockstep the drafter pads a chunk's drafts to the loop
    bucket of the step's DP-wide largest chunk, so a chunk pairs with every
    loop bucket from its own up; the warm-up compiles each pair the serving
    assembly takes, and off lockstep the same pairs as before."""
    runner = SimpleNamespace(**_LOCKSTEP_FIELDS)
    drafter = _sharded_mtp_drafter(runner, lockstep)
    calls = _assembly_warmup_calls(
        monkeypatch, runner_fields=dict(_LOCKSTEP_FIELDS, drafter=drafter)
    )
    one, two = _served_assembly_shapes(runner, drafter, lockstep)
    if lockstep:
        assert (8, 32) in one and (16, 32) in one
    else:
        assert one == {(8, 16), (16, 16), (32, 32)}
    for name in (
        "_compiled_assemble_spec_and_rejected",
        "_compiled_assemble_prefill_bootstrap",
    ):
        assert {(c[0][0], c[1][0]) for c in calls[name]} == one, name
    warmed_two = {
        ((c[0][0], c[1][0]), (c[3][0], c[4][0]))
        for c in calls["_compiled_assemble_two_chunks_spec_and_rejected"]
    }
    assert warmed_two == two


_READ_OFFSET_PROGRAMS = (
    "_reset_read_offsets_compiled",
    "_clear_read_offsets_compiled",
    "_update_read_offsets_verify_compiled",
    "_compiled_assemble_spec_rejected_and_read_offsets",
    "_compiled_assemble_two_chunks_spec_rejected_and_read_offsets",
)


def _read_offset_warmup_calls(monkeypatch, lockstep, async_scheduling, **fields):
    """Run the spec-decode half of _precompile_mamba_rollback_helpers on a
    32-request stub runner (`fields` override its layout) and record each
    read-offset program's argument shapes and dtypes."""
    calls: dict[str, list[tuple]] = {name: [] for name in _READ_OFFSET_PROGRAMS}

    def recorder(name):
        def record(*args):
            calls[name].append(
                tuple(
                    (tuple(a.shape), a.dtype) if isinstance(a, torch.Tensor) else a
                    for a in args
                )
            )
            return (torch.zeros(1), torch.zeros(1))

        return record

    for name in _READ_OFFSET_PROGRAMS:
        monkeypatch.setattr(tpu_runner, name, recorder(name))
    for name in (
        "_rollback_offsets_seed_compiled",
        "_rollback_offsets_migrate_compiled",
    ):
        monkeypatch.setattr(tpu_runner, name, lambda *_a: None)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    runner = SimpleNamespace(
        **_LOCKSTEP_FIELDS,
        mamba_slot_read_offsets=torch.zeros(64, dtype=torch.int32),
        _mamba_ckpt_window=1,
        _pool_block_split=1,
        _mamba_copy_plan=[(0, [])],
        kv_cache_config=None,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        num_reqs_most_model_len=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _is_async_drafter=True,
        _read_offset_scratch={},
        _bucket_len=tpu_runner.TPUModelRunner._bucket_len,
        _precompile_timed=lambda _label: contextlib.nullcontext(),
    )
    for name, value in fields.items():
        setattr(runner, name, value)
    runner._read_offset_reset_scratch = (
        tpu_runner.TPUModelRunner._read_offset_reset_scratch.__get__(runner)
    )
    runner.drafter = _sharded_mtp_drafter(runner, lockstep)
    tpu_runner.TPUModelRunner._precompile_mamba_rollback_helpers(runner)  # type: ignore[arg-type]
    return runner, calls


def _warmed_read_offset_shapes(calls):
    """(rows, draft rows, width) of each one-chunk warm-up, ((rows, draft
    rows), (rows, draft rows), width) of each two-chunk one, and (rows,
    width, dtype) of each verify scatter."""
    one = {
        (c[0][0][0], c[1][0][0], c[4][0][0])
        for c in calls["_compiled_assemble_spec_rejected_and_read_offsets"]
    }
    two = {
        ((c[0][0][0], c[1][0][0]), (c[3][0][0], c[4][0][0]), c[7][0][0])
        for c in calls["_compiled_assemble_two_chunks_spec_rejected_and_read_offsets"]
    }
    verify = {
        (c[2][0][0], c[1][0][0], str(c[2][1]))
        for c in calls["_update_read_offsets_verify_compiled"]
    }
    return one, two, verify


@pytest.mark.parametrize("lockstep", [False, True])
@pytest.mark.parametrize("async_scheduling", [True, False])
def test_mamba_read_offset_warmup_covers_every_spec_shape(
    monkeypatch, lockstep, async_scheduling
):
    """The spec-decode half of _precompile_mamba_rollback_helpers warms the
    read-offset reset, clear and verify scatters at every batch width, and,
    under async scheduling, the assembly programs that fold the read-offset
    scatter in at every shape the serving assembly takes for that width."""
    runner, calls = _read_offset_warmup_calls(monkeypatch, lockstep, async_scheduling)

    widths = [8, 16, 32]
    for name in ("_reset_read_offsets_compiled", "_clear_read_offsets_compiled"):
        assert sorted(
            c[2 if name.startswith("_reset") else 1][0] for c in calls[name]
        ) == [(w,) for w in widths], name
    warmed_one, warmed_two, warmed_verify = _warmed_read_offset_shapes(calls)
    assert warmed_verify == {
        (b, w, str(dt))
        for w in widths
        for b in widths
        if b <= w
        for dt in (torch.int32, torch.int64)
    }
    one, two = _served_assembly_shapes(runner, runner.drafter, lockstep)
    if not async_scheduling:
        assert not warmed_one and not warmed_two
        return
    assert warmed_one == {(b, p, w) for b, p in one for w in widths if b <= w}
    assert warmed_two == {(c0, c1, w) for c0, c1 in two for w in widths if c0[0] <= w}


@pytest.mark.parametrize("lockstep", [False, True])
@pytest.mark.parametrize(
    ("max_model_len_reqs", "most_model_len_reqs"), [(20, None), (32, 12)]
)
def test_read_offset_warmup_covers_a_width_an_smem_cap_set(
    monkeypatch, lockstep, max_model_len_reqs, most_model_len_reqs
):
    """When the SMEM cap leaves a layout's width below max_num_reqs at a
    count that is not a request bucket (20 or 12 of 32), a chunk of up to that
    many requests has its sampled rows padded to the next request bucket,
    above the width. Every (rows, draft rows, width) the assembly and verify
    scatters take while serving, for each layout's width, is warmed."""
    runner, calls = _read_offset_warmup_calls(
        monkeypatch,
        lockstep,
        True,
        num_reqs_max_model_len=max_model_len_reqs,
        num_reqs_most_model_len=most_model_len_reqs,
    )
    warmed_one, warmed_two, warmed_verify = _warmed_read_offset_shapes(calls)
    top = runner.max_num_reqs
    for width in {max_model_len_reqs, most_model_len_reqs} - {None}:
        rows = {
            tpu_runner._get_padded_num_reqs_with_upper_limit(n, top)
            for n in range(1, width + 1)
        }
        assert max(rows) > width or width == top
        one, two = _served_assembly_shapes(
            runner, runner.drafter, lockstep, max_count=width
        )
        assert {(b, p, width) for b, p in one} <= warmed_one
        assert {(c0, c1, width) for c0, c1 in two} <= warmed_two
        assert {
            (b, width, str(dt)) for b in rows for dt in (torch.int32, torch.int64)
        } <= warmed_verify


def test_the_warmed_read_offset_programs_serve_without_compiling(monkeypatch):
    """Dynamo guards on grad mode. The real stack, reset, clear and verify
    programs, warmed by _precompile_mamba_rollback_helpers and then served as
    a step serves them: the reset from execute_model and the clear and verify
    from sample_tokens, each through its no_grad method, and the stack both
    there and in sample_tokens' assembly, in that entry's own mode. No serving
    call compiles."""
    bases = (
        "_stack_mamba_groups",
        "_reset_read_offsets",
        "_clear_read_offsets",
        "_update_read_offsets_verify",
    )
    compiles: list[str] = []
    for base in bases:

        def backend(gm, example_inputs, base=base):
            compiles.append(base)
            return gm.forward

        program = torch.compile(
            getattr(tpu_runner, f"{base}_impl"),
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
        # Off TPU the serving wrappers call the impl; both names get the program.
        monkeypatch.setattr(tpu_runner, f"{base}_compiled", program)
        monkeypatch.setattr(tpu_runner, f"{base}_impl", program)
    for name in (
        "_rollback_offsets_seed_compiled",
        "_rollback_offsets_migrate_compiled",
    ):
        monkeypatch.setattr(tpu_runner, name, lambda *_a: None)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **k: None)
    width, k = 16, 3
    # Two mamba groups, so the groups are stacked by a program.
    groups = [torch.zeros(width, dtype=torch.int32) for _ in range(2)]
    runner = SimpleNamespace(
        max_num_reqs=16,
        num_reqs_paddings=[8, 16],
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=None,
        mamba_slot_read_offsets=torch.zeros(64, dtype=torch.int32),
        _mamba_ckpt_window=1,
        _pool_block_split=1,
        _mamba_copy_plan=[(0, []), (1, [])],
        kv_cache_config=None,
        speculative_config=SimpleNamespace(num_speculative_tokens=k),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _is_async_drafter=True,
        _read_offset_scratch={},
        _bucket_len=tpu_runner.TPUModelRunner._bucket_len,
        _precompile_timed=lambda _label: contextlib.nullcontext(),
        _attn_metadata_builder_ctx=SimpleNamespace(
            mamba_state_indices=None,
            unified_mamba_state_indices=groups,
            _stacked_mamba_groups=None,
            _stacked_mamba_groups_ids=None,
        ),
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
        requests={"a": None, "b": None},
        _mamba_offset_seeded=set(),
        _dp_lockstep_enabled=lambda: False,
    )
    for method in (
        "_read_offset_reset_scratch",
        "_mamba_state_index_groups",
        "_reset_read_offsets_for_new_requests",
        "_update_mamba_slot_read_offsets",
    ):
        setattr(
            runner, method, getattr(tpu_runner.TPUModelRunner, method).__get__(runner)
        )

    def execute_model():
        runner._reset_read_offsets_for_new_requests(0, 2)

    def sample_tokens():
        runner._update_mamba_slot_read_offsets(groups, None)
        for rows in (8, width):
            for dtype in (torch.int32, torch.int64):
                next_tokens = torch.zeros((rows, k + 1), dtype=dtype)
                runner._update_mamba_slot_read_offsets(groups, next_tokens)
        tpu_runner._stack_or_single_mamba_groups(groups)

    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(
            cache_size_limit=1024, accumulated_recompile_limit=1024
        ):
            tpu_runner.TPUModelRunner._precompile_mamba_rollback_helpers(runner)  # type: ignore[arg-type]
            assert set(compiles) == set(bases), compiles
            warmed = len(compiles)
            _inside("execute_model", execute_model)
            _inside("sample_tokens", sample_tokens)
        assert compiles[warmed:] == []
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize(
    ("async_scheduling", "async_drafter", "sampled", "discarded", "deferred"),
    [
        (True, True, [torch.zeros(1)], [], True),
        (False, True, [torch.zeros(1)], [], False),
        (True, False, [torch.zeros(1)], [], False),
        (True, True, [], [], False),
        (True, True, [torch.zeros(1)], [0], False),
    ],
)
def test_prev_rollback_waits_for_the_drafter_only_on_plain_async_drafter_steps(
    async_scheduling, async_drafter, sampled, discarded, deferred
):
    """The previous step's rollback moves after the drafter dispatch only
    under async scheduling with an async drafter, on a step that sampled
    tokens and discards no request."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _is_async_drafter=async_drafter,
    )
    assert TPUModelRunner._defers_prev_rollback(runner, sampled, discarded) is deferred


def test_fused_assembly_stays_off_under_adaptive_verification():
    """The compiled assembly takes the drafter's padded drafts only when the
    drafts are the drafter's last device output, one block per chunk, and
    adaptive verification is off; the prefill bootstrap has no adaptive
    guard."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    drafts = torch.zeros(4, 3)
    padded = [torch.zeros(8, 3)]
    drafter = object.__new__(Eagle3Proposer)
    drafter._last_propose_device_out = drafts
    drafter._last_padded_drafts_per_chunk = padded
    runner = SimpleNamespace(drafter=drafter, _adaptive_verification=None)
    runner._drafter_padded_drafts = lambda d, n: TPUModelRunner._drafter_padded_drafts(
        runner, d, n
    )
    fused = TPUModelRunner._fused_assembly_padded_drafts
    assert fused(runner, drafts, 1) is padded
    assert fused(runner, drafts, 2) is None
    assert fused(runner, torch.zeros(4, 3), 1) is None
    runner._adaptive_verification = object()
    assert fused(runner, drafts, 1) is None
    assert TPUModelRunner._drafter_padded_drafts(runner, drafts, 1) is padded
    runner.drafter = SimpleNamespace(
        _last_propose_device_out=drafts, _last_padded_drafts_per_chunk=padded
    )
    assert TPUModelRunner._drafter_padded_drafts(runner, drafts, 1) is None


def test_sample_tokens_rolls_back_through_the_rule_after_the_drafter():
    """The call site in TPUModelRunner: defer_modify_prev comes from
    _defers_prev_rollback, the early rollback runs only when it is False, and
    the deferred rollback runs after the drafter's propose call (structural:
    sample_tokens needs a TPU-backed runner to execute)."""
    import ast

    import vllm_torchtpu.runner.tpu_runner as tr

    tree = ast.parse(inspect.getsource(tr))
    fn = next(
        f
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef)
        and any(
            isinstance(n, ast.Name) and n.id == "defer_modify_prev" for n in ast.walk(f)
        )
    )
    assigns = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "defer_modify_prev" for t in n.targets
        )
    ]
    assert len(assigns) == 1
    call = assigns[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert call.func.attr == "_defers_prev_rollback"

    def rollbacks_under(test_src):
        out = []
        for n in ast.walk(fn):
            if isinstance(n, ast.If) and ast.unparse(n.test) == test_src:
                out += [
                    c.lineno
                    for c in ast.walk(n)
                    if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "_modify_prev_results"
                ]
        return out

    early = rollbacks_under("is_async and (not defer_modify_prev)")
    late = rollbacks_under("defer_modify_prev")
    propose = [
        c.lineno
        for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr == "propose_draft_token_ids"
    ]
    assert len(early) == 1 and len(late) == 1 and propose
    assert early[0] < min(propose) < late[0]
