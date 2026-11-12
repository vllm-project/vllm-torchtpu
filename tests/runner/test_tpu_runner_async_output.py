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
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from runner_input_test_utils import (
    _assert_same_logprobs_lists,
    _blocking_logprobs_lists,
)
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.runner.tpu_runner import _spec_logprob_row_tokens
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    SpecLogprobsChunk,
    assemble_spec_next_tokens,
    compute_num_rejected,
    extract_draft_token_ids,
    spec_logprobs_lists,
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


def test_spec_logprobs_lists_keeps_accepted_rows_in_request_order():
    verify = SpecLogprobsChunk(_row_logprobs(8), row_starts=np.array([0, 4, 6]))
    no_drafts = SpecLogprobsChunk(_row_logprobs(4, base=100), np.arange(2))
    out = spec_logprobs_lists([verify, no_drafts], gen_lens=[2, 2, 1, 1, 1])
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
        spec_logprobs=[
            SpecLogprobsChunk(_row_logprobs(8), row_starts=np.array([0, 4, 6]))
        ],
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
def test_spec_logprobs_lists_without_gen_lens_matches_the_blocking_path(
    num_logprobs, real_rows
):
    """gen_lens=None, one generated token per request: each chunk's first `n`
    bucketed rows, in chunk order. A single chunk is sliced, not copied."""
    chunks = [
        SpecLogprobsChunk(_bucketed_logprobs(16, num_logprobs, seed), np.arange(n))
        for seed, n in enumerate(real_rows)
    ]
    out = spec_logprobs_lists(chunks, gen_lens=None)
    _assert_same_logprobs_lists(
        out,
        _blocking_logprobs_lists([(c.tensors, len(c.row_starts)) for c in chunks]),
    )
    shared = np.shares_memory(out.logprobs, chunks[0].tensors.logprobs.numpy())
    assert shared == (len(chunks) == 1)


def test_get_output_attaches_plain_logprobs():
    logprobs = _bucketed_logprobs(8, 0, 0)
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[7], [8], [9], [0], [0], [0], [0], [0]])],
        chunk_real_lens=[3],
        copy_ready_event=None,
        completed=True,
        spec_logprobs=[SpecLogprobsChunk(logprobs, np.arange(3))],
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
    "num_rows, spec_logprobs",
    [
        (2, None),
        # No real rows: get_output returns before it assembles logprobs.
        (0, [SpecLogprobsChunk(_bucketed_logprobs(4, 0, 1), np.arange(0))]),
    ],
    ids=["no-logprobs-requested", "empty-output"],
)
def test_get_output_leaves_logprobs_none(num_rows, spec_logprobs):
    req_ids = ["a", "b"][:num_rows]
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[1], [2], [0], [0]])],
        chunk_real_lens=[num_rows],
        copy_ready_event=None,
        completed=True,
        spec_logprobs=spec_logprobs,
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
    monkeypatch.setattr(torch, "tpu", SimpleNamespace(Event=MagicMock), raising=False)
    tokens = [torch.zeros((8, 1), dtype=torch.int32)]
    device_logprobs = _bucketed_logprobs(8, 2, 3)
    state = AsyncTPUCopyState.from_device_chunks(
        tokens, [5], spec_logprobs=[SpecLogprobsChunk(device_logprobs, np.arange(5))]
    )
    # The host copies ride the same async D2H as the sampled tokens; the
    # device sources stay alive until the copy event is waited on.
    (host,) = state.spec_logprobs
    assert np.array_equal(host.row_starts, np.arange(5))
    assert torch.equal(host.tensors.logprobs, device_logprobs.logprobs)
    assert state._logprobs_tpu is not None
    assert state._logprobs_tpu[0] is device_logprobs
    state.copy_ready_event.record.assert_called_once_with()
    state.wait()
    state.copy_ready_event.synchronize.assert_called_once_with()
    assert state._logprobs_tpu is None
    empty = AsyncTPUCopyState.from_device_chunks(tokens, [5])
    assert empty.spec_logprobs is None and empty._logprobs_tpu is None
    # get_output needs one row start per real request in each chunk.
    short = SpecLogprobsChunk(device_logprobs, np.arange(4))
    with pytest.raises(AssertionError):
        AsyncTPUCopyState.from_device_chunks(tokens, [5], spec_logprobs=[short])
