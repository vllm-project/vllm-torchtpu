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
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner import tpu_runner_async_output as async_output
from vllm_torchtpu.runner.tpu_runner import _spec_logprob_row_tokens
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
    SpecLogprobsChunk,
    _compiled_assemble_spec_and_rejected,
    _compiled_extract_draft_token_ids,
    _compiled_subtract_num_rejected_tokens,
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


@pytest.mark.tpu_test
def test_compiled_async_spec_helpers_match_eager(device):
    if device.type != "tpu":
        pytest.skip("TPU compiled helpers require TPU device")
    from types import SimpleNamespace

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
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    drafts_view = drafts_p[:2]
    runner.drafter = SimpleNamespace(
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
    runner.drafter = SimpleNamespace(
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
            _compiled_subtract_and_substitute_placeholder_token.__wrapped__._torchdynamo_orig_callable,
            _compiled_subtract_and_substitute_draft_token_ids.__wrapped__._torchdynamo_orig_callable,
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

    flat2, rej2 = (
        _compiled_assemble_two_chunks_spec_rejected_and_read_offsets.__wrapped__._torchdynamo_orig_callable(
            nt0, d0, dl0, nt1, d1, dl1, read_offsets, mamba_idx, 32, 8
        )
    )
    assert flat2[:4].tolist() == [51, 501, 502, 503]
    assert flat2[4:8].tolist() == [60, 601, 602, 603]
    assert flat2[16:20].tolist() == [70, 701, 702, 703]
    assert rej2[:2].tolist() == [2, 3]
    # Accepted counts - 1: req 0 accepted 2 tokens -> offset 1; req 1 accepted
    # 1 token -> offset 0
    assert int(read_offsets[2]) == 1
    assert int(read_offsets[5]) == 0


_ASSEMBLY_PROGRAMS = (
    "_compiled_pad_non_spec_next_tokens",
    "_compiled_assemble_prefill_bootstrap",
    "_compiled_assemble_spec_and_rejected",
    "_compiled_assemble_two_chunks_spec_and_rejected",
    "_compiled_subtract_num_rejected_tokens",
    "_compiled_subtract_and_substitute_placeholder_token",
    "_compiled_subtract_and_substitute_draft_token_ids",
)


def _assembly_warmup_calls(
    monkeypatch, *, async_scheduling=True, most=None, async_drafter=True
):
    """Run _precompile_async_spec_assembly on a stub runner and record the
    shapes each assembly program was warmed with."""
    calls: dict[str, list[tuple]] = {name: [] for name in _ASSEMBLY_PROGRAMS}

    def recorder(name):
        def record(*args):
            calls[name].append(
                tuple(
                    tuple(a.shape) if isinstance(a, torch.Tensor) else a for a in args
                )
            )
            return (torch.zeros(1),)

        return record

    for name in _ASSEMBLY_PROGRAMS:
        monkeypatch.setattr(async_output, name, recorder(name))
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
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    tpu_runner.TPUModelRunner._precompile_async_spec_assembly(runner)  # type: ignore[arg-type]
    return calls


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
    for name in _ASSEMBLY_PROGRAMS[4:]:
        warmed = {(c[0][0], c[1][-1]) for c in calls[name]}
        # Every token bucket, for both the max-model-len and the
        # most-model-len request layouts.
        assert warmed == {(knr, t) for knr in (8, 4) for t in (8, 16, 32)}, name


def test_async_spec_assembly_warmup_needs_async_scheduling(monkeypatch):
    calls = _assembly_warmup_calls(monkeypatch, async_scheduling=False)
    assert all(not c for c in calls.values())


def test_async_spec_assembly_warmup_needs_an_async_drafter(monkeypatch):
    calls = _assembly_warmup_calls(monkeypatch, async_drafter=False)
    assert all(not c for c in calls.values())
