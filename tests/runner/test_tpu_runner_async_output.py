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
from unittest import mock

import numpy as np
import pytest
import torch
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


# VLLM_TPU_PAD_EAGER_GLUE: subtract_num_rejected_tokens_padded.

_PAD_MAX_NUM_REQS = 32
_PAD_TOKEN_BUCKETS = (16, 32, 64, 128)
_PAD_REQ_BUCKETS = (8, 32)


def _subtract_case(rng, num_reqs, num_tokens, num_prev, device, mrope=False):
    """Random inputs as the runner builds them: `num_reqs` live requests in a
    `num_tokens`-slot batch, each mapped (or not, -1) to one of the previous
    step's `num_prev` rejection counts."""
    seq_lens = torch.from_numpy(rng.integers(100, 10_000, num_reqs, dtype=np.int32))
    pos_shape = (3, num_tokens) if mrope else (num_tokens,)
    positions = torch.from_numpy(rng.integers(100, 10_000, pos_shape, dtype=np.int32))
    num_rejected = torch.from_numpy(rng.integers(0, 4, num_prev, dtype=np.int32))
    seq_idx = rng.integers(-1, num_prev, num_reqs, dtype=np.int32)
    pos_idx = rng.integers(-1, num_prev, num_tokens, dtype=np.int32)
    return tuple(
        t.to(device)
        for t in (
            seq_lens,
            positions,
            num_rejected,
            torch.from_numpy(seq_idx),
            torch.from_numpy(pos_idx),
        )
    )


@pytest.mark.parametrize("mrope", [False, True])
def test_subtract_num_rejected_tokens_padded_matches_unpadded(device, mrope):
    import vllm_torchtpu.runner.tpu_runner_async_output as ao

    rng = np.random.default_rng(0)
    for num_reqs in _PAD_REQ_BUCKETS:
        for num_tokens in _PAD_TOKEN_BUCKETS:
            for num_prev in (1, 2, 7, 19, _PAD_MAX_NUM_REQS):
                args = _subtract_case(
                    rng, num_reqs, num_tokens, num_prev, device, mrope
                )
                ref_seq, ref_pos = subtract_num_rejected_tokens(*args)
                seq, pos = ao.subtract_num_rejected_tokens_padded(
                    *args, pad_len=_PAD_MAX_NUM_REQS
                )
                assert seq.dtype == ref_seq.dtype and pos.dtype == ref_pos.dtype
                assert torch.equal(seq.cpu(), ref_seq.cpu())
                assert torch.equal(pos.cpu(), ref_pos.cpu())


def _record_keys(monkeypatch, ao, name, keys):
    impl = getattr(ao, name)

    def recorded(*args):
        keys.add(tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor)))
        return impl(*args)

    monkeypatch.setattr(ao, name, recorded)


def test_subtract_num_rejected_tokens_padded_bounds_program_keys(monkeypatch):
    """Flag off, the one eager program's key is the product of the three input
    lengths; flag on, the seq_lens half sees one key per request bucket, the
    positions half one per token bucket, and the pad one per previous count."""
    import vllm_torchtpu.runner.tpu_runner_async_output as ao

    pad_keys, seq_keys, pos_keys = set(), set(), set()
    _record_keys(monkeypatch, ao, "_pad_num_rejected_impl", pad_keys)
    _record_keys(monkeypatch, ao, "_subtract_seq_lens_impl", seq_keys)
    _record_keys(monkeypatch, ao, "_subtract_positions_impl", pos_keys)
    unpadded_keys = set()
    rng = np.random.default_rng(1)
    for num_reqs in _PAD_REQ_BUCKETS:
        for num_tokens in _PAD_TOKEN_BUCKETS:
            for num_prev in range(1, _PAD_MAX_NUM_REQS + 1):
                args = _subtract_case(rng, num_reqs, num_tokens, num_prev, "cpu")
                unpadded_keys.add(tuple(tuple(a.shape) for a in args))
                ao.subtract_num_rejected_tokens_padded(*args, pad_len=_PAD_MAX_NUM_REQS)
    n_buckets = len(_PAD_REQ_BUCKETS) * len(_PAD_TOKEN_BUCKETS)
    assert len(unpadded_keys) == n_buckets * _PAD_MAX_NUM_REQS
    assert len(seq_keys) == len(_PAD_REQ_BUCKETS)
    assert len(pos_keys) == len(_PAD_TOKEN_BUCKETS)
    # Every count below the pad length; the full-length one skips the pad.
    assert len(pad_keys) == _PAD_MAX_NUM_REQS - 1


def test_subtract_num_rejected_tokens_padded_fault_control(monkeypatch):
    """The equality check catches a pad that moves the real counts (a left
    pad), and does not depend on the pad value: the padded tail is never
    indexed, so a nonzero right pad is still exact."""
    import vllm_torchtpu.runner.tpu_runner_async_output as ao

    def check(pad_fn) -> bool:
        monkeypatch.setattr(ao, "_pad_num_rejected_impl", pad_fn)
        rng = np.random.default_rng(2)
        for num_prev in (1, 5, 17):
            args = _subtract_case(rng, 8, 64, num_prev, "cpu")
            ref = subtract_num_rejected_tokens(*args)
            out = ao.subtract_num_rejected_tokens_padded(
                *args, pad_len=_PAD_MAX_NUM_REQS
            )
            if not all(torch.equal(a, b) for a, b in zip(ref, out)):
                return False
        return True

    def left_pad(nr, pad_len):
        return torch.nn.functional.pad(nr, (pad_len - nr.shape[0], 0))

    def nonzero_right_pad(nr, pad_len):
        return torch.nn.functional.pad(nr, (0, pad_len - nr.shape[0]), value=7)

    assert not check(left_pad)
    assert check(nonzero_right_pad)


def test_precompile_subtract_num_rejected_covers_runtime_keys(monkeypatch):
    """Every key a runtime sweep reaches is one the precompile warmed."""
    import vllm_torchtpu.runner.tpu_runner_async_output as ao
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    warmed = {"pad": set(), "seq": set(), "pos": set()}
    _record_keys(monkeypatch, ao, "_pad_num_rejected_impl", warmed["pad"])
    _record_keys(monkeypatch, ao, "_subtract_seq_lens_impl", warmed["seq"])
    _record_keys(monkeypatch, ao, "_subtract_positions_impl", warmed["pos"])
    runner = mock.MagicMock(
        max_num_reqs=_PAD_MAX_NUM_REQS,
        num_reqs_max_model_len=_PAD_REQ_BUCKETS[0],
        num_reqs_most_model_len=_PAD_REQ_BUCKETS[1],
        num_tokens_paddings=list(_PAD_TOKEN_BUCKETS),
        device=torch.device("cpu"),
    )
    runner._precompile_timed.return_value = contextlib.nullcontext()
    TPUModelRunner._precompile_subtract_num_rejected(runner)
    precompiled = {k: set(v) for k, v in warmed.items()}
    for v in warmed.values():
        v.clear()
    rng = np.random.default_rng(3)
    for num_reqs in _PAD_REQ_BUCKETS:
        for num_tokens in _PAD_TOKEN_BUCKETS:
            for num_prev in range(1, _PAD_MAX_NUM_REQS + 1):
                ao.subtract_num_rejected_tokens_padded(
                    *_subtract_case(rng, num_reqs, num_tokens, num_prev, "cpu"),
                    pad_len=_PAD_MAX_NUM_REQS,
                )
    for k in warmed:
        assert warmed[k] <= precompiled[k], k
