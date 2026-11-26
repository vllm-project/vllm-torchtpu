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

from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
import torch
from runner_output_test_utils import (
    _assert_same_logprobs_lists,
    _blocking_logprobs_lists,
)
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.runner.routed_experts import PendingRoutedExperts
from vllm_torchtpu.runner.tpu_runner import _spec_logprob_row_tokens
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    LogprobsChunk,
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


def test_get_output_hands_each_requests_sampled_tokens_to_routed_experts():
    """Routed experts get each request's sampled tokens, and an empty list for
    a partial prefill whose token was discarded."""
    routed_experts = create_autospec(PendingRoutedExperts, instance=True)
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([5, 6, 7], dtype=torch.int32)],
        chunk_real_lens=[3],
        copy_ready_event=None,
        completed=True,
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=ModelRunnerOutput(
            req_ids=["a", "b", "c"],
            req_id_to_index={"a": 0, "b": 1, "c": 2},
            sampled_token_ids=[],
        ),
        copy_state=copy_state,
        discard_sampled_tokens_req_indices=[1],
        routed_experts=routed_experts,
    ).get_output()

    routed_experts.process_output.assert_called_once_with([[5], [], [7]])
    assert out.aux_output_connector_output is routed_experts.process_output.return_value
