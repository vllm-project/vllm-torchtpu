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

"""Tests for speculative decoding bookkeeping fold under TPU_FOLD_SPEC_BOOKKEEPING."""

import contextlib
import types
from typing import Any, cast

import numpy as np
import pytest
import torch

import vllm_torchtpu.layers.adapter.sample.rejection_sampler as rejection_sampler_mod
import vllm_torchtpu.runner.tpu_runner as tpu_runner_mod
import vllm_torchtpu.runner.tpu_runner_async_output as async_out_mod
from tests.runner.runner_input_test_utils import _make_runner
from tests.spec_decode.test_mtp_full_draft_loop import (
    _build_b8_chunk,
    _DummyMtpDraft,
    _fake_get_forward_context,
    _fake_set_forward_context,
    _make_b8_test_proposer,
)
from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.static_decode_buffers import (
    fold_draft_loop_spec_bookkeeping,
    fold_verify_and_draft_prep,
)
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _PrecomputedDraftPrep,
)
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    assemble_spec_next_tokens,
    compute_async_draft_seeds_and_indices,
    compute_num_rejected,
    draft_seed_input_ids,
    extract_draft_token_ids,
    subtract_num_rejected_tokens,
)
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs


def subtract_and_substitute_placeholder_token(
    input_ids: torch.Tensor,
    token_in_tpu_cur_input_indices: torch.Tensor,
    token_in_tpu_pre_next_tokens_indices: torch.Tensor,
    next_tokens: torch.Tensor,
    rejected_tokens_per_req: torch.Tensor,
    num_reqs: int,
    max_num_tokens: int,
    start_index: int = 0,
) -> torch.Tensor:
    """Reference helper for next-step placeholder token substitution."""
    cur_indices = token_in_tpu_cur_input_indices
    pre_indices = token_in_tpu_pre_next_tokens_indices
    valid = pre_indices >= 0
    valid_cur = cur_indices[valid]
    valid_pre = pre_indices[valid]
    out = input_ids.clone()
    out[valid_cur] = next_tokens[valid_pre]
    return out


def subtract_and_substitute_draft_token_ids(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids: torch.Tensor,
    substitution_indices: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference helper for next-step subtraction and draft token extraction."""
    new_seq_lens, new_positions = subtract_num_rejected_tokens(
        seq_lens=seq_lens,
        positions=positions,
        num_rejected=num_rejected,
        seq_lens_subtract_indices=seq_lens_subtract_indices,
        positions_subtract_indices=positions_subtract_indices,
    )
    new_input_ids = subtract_and_substitute_placeholder_token(
        input_ids=input_ids,
        token_in_tpu_cur_input_indices=substitution_indices[0],
        token_in_tpu_pre_next_tokens_indices=substitution_indices[1],
        next_tokens=next_tokens_tpu,
        rejected_tokens_per_req=num_rejected,
        num_reqs=int(seq_lens.shape[0]),
        max_num_tokens=int(input_ids.shape[0]),
    )
    new_draft_token_ids = extract_draft_token_ids(
        input_ids=new_input_ids,
        logits_indices=logits_indices,
        target_logits_indices=target_logits_indices,
    )
    return new_seq_lens, new_positions, new_input_ids, new_draft_token_ids


def test_fold_spec_bookkeeping_env_default_false() -> None:
    """TPU_FOLD_SPEC_BOOKKEEPING defaults to False."""
    assert envs.TPU_FOLD_SPEC_BOOKKEEPING is False


@pytest.mark.parametrize(
    "accepted_counts",
    [(0, 3, 1, 2), (3, 3, 3, 3), (0, 0, 0, 0), (2, 1, 0, 3)],
)
def test_fold_verify_and_draft_prep_matches_unfolded(
    accepted_counts: tuple[int, int, int, int],
) -> None:
    """Verify fold_verify_and_draft_prep produces bit-identical tensors to
    compute_async_draft_seeds_and_indices + draft_seed_input_ids,
    compute_num_rejected, and mamba read-offset updates across 0..K accepted
    draft tokens, while masking non-sentinel padded rows (>= num_reqs)."""
    num_reqs = 4
    batch_bucket = 8
    knr = 8
    loop_bucket = 16
    max_num_reqs = 8
    spec_k = 3
    query_len = 1 + spec_k
    total_tokens = batch_bucket * query_len

    next_tokens = torch.full(
        (batch_bucket, spec_k + 1), INVALID_TOKEN_ID, dtype=torch.int32
    )
    for req_idx, num_acc in enumerate(accepted_counts):
        for col_idx in range(num_acc + 1):
            next_tokens[req_idx, col_idx] = 1000 + req_idx * 10 + col_idx
    # Populate padded rows (>= num_reqs) with non-sentinel tokens to verify
    # active_req_mask prevents corrupting Mamba slots on padded columns.
    next_tokens[num_reqs:, :] = 777

    draft_lengths = torch.zeros(batch_bucket, dtype=torch.int32)
    draft_lengths[:num_reqs] = spec_k

    qsl_np = np.minimum(np.arange(knr + 1, dtype=np.int32), num_reqs) * query_len
    query_start_loc = torch.from_numpy(qsl_np)

    pad_req_idx_np = np.minimum(np.arange(loop_bucket, dtype=np.int64), num_reqs - 1)
    pad_req_idx = torch.from_numpy(pad_req_idx_np)

    chunk_input_ids = torch.arange(100, 100 + total_tokens, dtype=torch.int32)
    base_positions_per_req = np.array([20, 40, 60, 80, 0, 0, 0, 0], dtype=np.int32)
    positions_np = np.repeat(base_positions_per_req, query_len) + np.tile(
        np.arange(query_len, dtype=np.int32), batch_bucket
    )
    positions = torch.from_numpy(positions_np)
    base_seq_lens = torch.from_numpy(base_positions_per_req + query_len)

    read_offsets_folded = torch.zeros(32, dtype=torch.int32)
    read_offsets_folded[20:28] = 99
    read_offsets_ref = torch.zeros(32, dtype=torch.int32)
    stacked_groups = torch.tensor(
        [
            [2, 5, 8, 11, 20, 21, 22, 23],
            [3, 6, 9, 12, 24, 25, 26, 27],
        ],
        dtype=torch.int32,
    )

    (
        draft_input_ids,
        gather_indices,
        num_rejected_knr,
        pos_step1,
        seq_lens_step1,
        bonus_full,
        spec_num_rejected,
    ) = fold_verify_and_draft_prep(
        next_tokens=next_tokens,
        chunk_input_ids=chunk_input_ids,
        draft_lengths=draft_lengths,
        query_start_loc=query_start_loc,
        pad_req_idx=pad_req_idx,
        positions=positions,
        base_seq_lens=base_seq_lens,
        advance_draft_positions=True,
        max_num_reqs=max_num_reqs,
        read_offsets=read_offsets_folded,
        stacked_groups=stacked_groups,
    )

    # Unfolded reference via compute_async_draft_seeds_and_indices and
    # draft_seed_input_ids.
    eff_b = min(batch_bucket, knr)
    ref_seed_ids_b, ref_last_tok_idx_b, ref_num_rej_b = (
        compute_async_draft_seeds_and_indices(
            next_tokens[:eff_b],
            draft_lengths[:eff_b],
            query_start_loc[: eff_b + 1],
        )
    )
    ref_gather_indices = torch.index_select(ref_last_tok_idx_b, 0, pad_req_idx)
    ref_seed_ids = torch.index_select(ref_seed_ids_b, 0, pad_req_idx)
    ref_draft_input_ids = draft_seed_input_ids(
        chunk_input_ids, ref_gather_indices, ref_seed_ids
    )
    assert torch.equal(gather_indices, ref_gather_indices)
    assert torch.equal(draft_input_ids, ref_draft_input_ids)
    assert torch.equal(num_rejected_knr, ref_num_rej_b)
    assert torch.equal(bonus_full[:num_reqs, 0], ref_seed_ids_b[:num_reqs])

    expected_rejected = torch.tensor(
        [spec_k - acc for acc in accepted_counts], dtype=torch.int32
    )
    ref_rejected = compute_num_rejected(
        next_tokens[:num_reqs], draft_lengths[:num_reqs]
    )
    assert torch.equal(ref_rejected, expected_rejected)
    assert torch.equal(num_rejected_knr[:num_reqs], expected_rejected)
    assert torch.equal(spec_num_rejected[:num_reqs], expected_rejected)
    assert int(num_rejected_knr[num_reqs:].abs().sum()) == 0

    ref_pos_step1 = torch.index_select(positions, -1, ref_gather_indices) + 1
    ref_seq_lens_step1 = base_seq_lens + 1 - ref_num_rej_b
    assert torch.equal(pos_step1, ref_pos_step1)
    assert torch.equal(seq_lens_step1, ref_seq_lens_step1)

    for req_idx, num_acc in enumerate(accepted_counts):
        expected_last_tok_idx = (req_idx + 1) * query_len - 1 - (spec_k - num_acc)
        assert int(gather_indices[req_idx]) == expected_last_tok_idx
        expected_bonus = 1000 + req_idx * 10 + num_acc
        assert int(bonus_full[req_idx, 0]) == expected_bonus
        assert int(draft_input_ids[expected_last_tok_idx]) == expected_bonus
        expected_pos_step1 = int(base_positions_per_req[req_idx]) + num_acc + 1
        assert int(pos_step1[req_idx]) == expected_pos_step1
        expected_sl_step1 = int(base_positions_per_req[req_idx]) + num_acc + 1 + 1
        assert int(seq_lens_step1[req_idx]) == expected_sl_step1

    for req_idx, num_acc in enumerate(accepted_counts):
        slot_g0 = int(stacked_groups[0, req_idx])
        slot_g1 = int(stacked_groups[1, req_idx])
        read_offsets_ref[slot_g0] = num_acc
        read_offsets_ref[slot_g1] = num_acc
    # Padded slots 20..27 must be reset to 0 despite next_tokens[num_reqs:] == 777.
    assert int(read_offsets_folded[20:28].abs().sum()) == 0
    assert torch.equal(read_offsets_folded, read_offsets_ref)


def test_fold_draft_loop_epilogue_matches_next_step_subtraction_and_substitution() -> (
    None
):
    """Verify fold_draft_loop_spec_bookkeeping produces the exact same
    src_padded, next_seq_lens, next_positions, next_input_ids, and
    next_draft_token_ids as the unfolded assemble + next-step
    subtract_and_substitute_draft_token_ids pipeline."""
    num_reqs = 4
    batch_bucket = 8
    knr = 8
    loop_bucket = 16
    max_num_reqs = 8
    spec_k = 3
    query_len = 1 + spec_k
    padded_total_tokens = batch_bucket * query_len
    assemble_bound = max_num_reqs * query_len

    accepted_counts = (2, 0, 3, 1)
    next_tokens = torch.full(
        (batch_bucket, spec_k + 1), INVALID_TOKEN_ID, dtype=torch.int32
    )
    for req_idx, num_acc in enumerate(accepted_counts):
        for col_idx in range(num_acc + 1):
            next_tokens[req_idx, col_idx] = 500 + req_idx * 10 + col_idx

    draft_lengths = torch.zeros(batch_bucket, dtype=torch.int32)
    draft_lengths[:num_reqs] = spec_k
    qsl_np = np.minimum(np.arange(knr + 1, dtype=np.int32), num_reqs) * query_len
    query_start_loc = torch.from_numpy(qsl_np)
    pad_req_idx = torch.from_numpy(
        np.minimum(np.arange(loop_bucket, dtype=np.int64), num_reqs - 1)
    )
    chunk_input_ids = torch.arange(1, 1 + padded_total_tokens, dtype=torch.int32)
    base_computed = np.array([12, 28, 44, 60, 0, 0, 0, 0], dtype=np.int32)
    positions_step_t = torch.from_numpy(
        np.repeat(base_computed, query_len)
        + np.tile(np.arange(query_len, dtype=np.int32), batch_bucket)
    )
    seq_lens_step_t = torch.from_numpy(base_computed + query_len)

    (
        _draft_in,
        _gather_idx,
        _num_rej_knr,
        pos_step1,
        seq_lens_step1,
        bonus_full,
        spec_num_rejected,
    ) = fold_verify_and_draft_prep(
        next_tokens=next_tokens,
        chunk_input_ids=chunk_input_ids,
        draft_lengths=draft_lengths,
        query_start_loc=query_start_loc,
        pad_req_idx=pad_req_idx,
        positions=positions_step_t,
        base_seq_lens=seq_lens_step_t,
        advance_draft_positions=True,
        max_num_reqs=max_num_reqs,
    )

    seq_lens_end = seq_lens_step1 + (spec_k - 2)
    drafts = torch.arange(2000, 2000 + loop_bucket * spec_k, dtype=torch.int32).reshape(
        loop_bucket, spec_k
    )

    req_expand_np = np.zeros(padded_total_tokens, dtype=np.int64)
    offsets_np = np.zeros(padded_total_tokens, dtype=np.int32)
    valid_tok_np = np.zeros(padded_total_tokens, dtype=np.int32)
    for req_idx in range(num_reqs):
        tok_start = req_idx * query_len
        tok_end = tok_start + query_len
        req_expand_np[tok_start:tok_end] = req_idx
        offsets_np[tok_start:tok_end] = np.arange(query_len, dtype=np.int32)
    valid_tok_np[: num_reqs * query_len] = 1
    valid_req_np = np.zeros(knr, dtype=np.int32)
    valid_req_np[:num_reqs] = 1
    valid_draft_np = np.zeros((batch_bucket, spec_k), dtype=np.int32)
    valid_draft_np[:num_reqs, :] = 1

    (
        drafts_out,
        src_padded,
        next_seq_lens,
        next_positions,
        next_input_ids,
        next_draft_token_ids,
    ) = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1,
        seq_lens_end=seq_lens_end,
        bonus_tokens=bonus_full,
        assemble_bound=assemble_bound,
        num_loop_steps=spec_k - 1,
        pos_req_expand_idx=torch.from_numpy(req_expand_np),
        pos_step1_offsets=torch.from_numpy(offsets_np),
        valid_token_mask=torch.from_numpy(valid_tok_np),
        valid_req_mask=torch.from_numpy(valid_req_np),
        valid_draft_mask=torch.from_numpy(valid_draft_np),
    )
    assert torch.equal(drafts_out, drafts)

    ref_src_unpadded = assemble_spec_next_tokens(next_tokens, drafts, num_reqs)
    assert torch.equal(src_padded[: num_reqs * query_len], ref_src_unpadded)

    optimistic_computed_next = base_computed.copy()
    optimistic_computed_next[:num_reqs] += query_len
    optimistic_seq_lens = torch.zeros(knr, dtype=torch.int32)
    optimistic_seq_lens[:num_reqs] = torch.from_numpy(
        optimistic_computed_next[:num_reqs] + query_len
    )
    optimistic_positions = torch.zeros(padded_total_tokens, dtype=torch.int32)
    for req_idx in range(num_reqs):
        for offset in range(query_len):
            optimistic_positions[req_idx * query_len + offset] = (
                int(optimistic_computed_next[req_idx]) + offset
            )
    optimistic_input_ids = torch.zeros(padded_total_tokens, dtype=torch.int32)

    seq_sub_idx = torch.full((knr,), -1, dtype=torch.int32)
    seq_sub_idx[:num_reqs] = torch.arange(num_reqs, dtype=torch.int32)
    pos_sub_idx = torch.full((padded_total_tokens,), -1, dtype=torch.int32)
    for req_idx in range(num_reqs):
        pos_sub_idx[req_idx * query_len : (req_idx + 1) * query_len] = req_idx

    cur_in_idx = torch.arange(padded_total_tokens, dtype=torch.int64)
    pre_nt_idx = torch.full((padded_total_tokens,), -1, dtype=torch.int64)
    pre_nt_idx[: num_reqs * query_len] = torch.arange(
        num_reqs * query_len, dtype=torch.int64
    )
    substitution_indices = torch.stack([cur_in_idx, pre_nt_idx], dim=0)

    logits_indices = torch.arange(padded_total_tokens, dtype=torch.int32)
    target_logits_list: list[int] = []
    for req_idx in range(batch_bucket):
        for offset in range(spec_k):
            if req_idx < num_reqs:
                target_logits_list.append(req_idx * query_len + offset)
            else:
                target_logits_list.append(num_reqs * query_len)
    target_logits_indices = torch.tensor(target_logits_list, dtype=torch.int32)

    (
        ref_next_seq_lens,
        ref_next_positions,
        ref_next_input_ids,
        ref_next_draft_token_ids,
    ) = subtract_and_substitute_draft_token_ids(
        seq_lens=optimistic_seq_lens,
        positions=optimistic_positions,
        num_rejected=spec_num_rejected,
        seq_lens_subtract_indices=seq_sub_idx,
        positions_subtract_indices=pos_sub_idx,
        input_ids=optimistic_input_ids,
        substitution_indices=substitution_indices,
        next_tokens_tpu=src_padded,
        logits_indices=logits_indices,
        target_logits_indices=target_logits_indices,
    )

    assert torch.equal(next_seq_lens, ref_next_seq_lens)
    assert torch.equal(next_positions, ref_next_positions)
    assert torch.equal(next_input_ids, ref_next_input_ids)
    assert torch.equal(next_draft_token_ids, ref_next_draft_token_ids)


def test_fold_draft_loop_epilogue_src_only_fallback() -> None:
    """When steady-state next-step masks are omitted, fold_draft_loop_spec_bookkeeping
    folds src_padded assembly and returns a 2-tuple (drafts, src_padded)."""
    drafts = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
    pos_step1 = torch.tensor([5, 9], dtype=torch.int32)
    seq_lens_end = torch.tensor([6, 10], dtype=torch.int32)
    bonus_tokens = torch.tensor([[10], [20]], dtype=torch.int32)
    out = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1,
        seq_lens_end=seq_lens_end,
        bonus_tokens=bonus_tokens,
        assemble_bound=12,
        num_loop_steps=2,
    )
    assert len(out) == 2
    assert torch.equal(out[0], drafts)
    expected_src = torch.tensor(
        [10, 11, 12, 13, 20, 21, 22, 23, 0, 0, 0, 0], dtype=torch.int32
    )
    assert torch.equal(out[1], expected_src)


def test_fold_draft_loop_epilogue_mrope_2d_positions_and_step1_seq_lens() -> None:
    """Verify fold_draft_loop_spec_bookkeeping supports 2D MRoPE positions
    (3, p) -> (3, T) and seq_lens_is_step1=True."""
    num_reqs = 2
    batch_bucket = 4
    loop_bucket = 4
    spec_k = 3
    query_len = 1 + spec_k
    padded_total_tokens = batch_bucket * query_len
    assemble_bound = batch_bucket * query_len

    drafts = torch.tensor(
        [[101, 102, 103], [201, 202, 203], [0, 0, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    pos_1d = np.array([15, 32, 32, 32], dtype=np.int32)
    pos_step1_2d = torch.from_numpy(np.stack([pos_1d, pos_1d, pos_1d], axis=0))
    seq_lens_step1 = torch.tensor([16, 33, 0, 0], dtype=torch.int32)
    bonus_tokens = torch.tensor([[100], [200], [-1], [-1]], dtype=torch.int32)

    req_expand_np = np.zeros(padded_total_tokens, dtype=np.int64)
    offsets_np = np.zeros(padded_total_tokens, dtype=np.int32)
    valid_tok_np = np.zeros(padded_total_tokens, dtype=np.int32)
    for req_idx in range(num_reqs):
        tok_start = req_idx * query_len
        tok_end = tok_start + query_len
        req_expand_np[tok_start:tok_end] = min(req_idx, loop_bucket - 1)
        offsets_np[tok_start:tok_end] = np.arange(query_len, dtype=np.int32)
    valid_tok_np[: num_reqs * query_len] = 1
    valid_req_np = np.array([1, 1, 0, 0], dtype=np.int32)
    valid_draft_np = np.zeros((batch_bucket, spec_k), dtype=np.int32)
    valid_draft_np[:num_reqs, :] = 1

    (
        drafts_out,
        src_padded,
        next_seq_lens,
        next_positions,
        next_input_ids,
        next_draft_token_ids,
    ) = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1_2d,
        seq_lens_end=seq_lens_step1,
        bonus_tokens=bonus_tokens,
        assemble_bound=assemble_bound,
        num_loop_steps=spec_k - 1,
        pos_req_expand_idx=torch.from_numpy(req_expand_np),
        pos_step1_offsets=torch.from_numpy(offsets_np),
        valid_token_mask=torch.from_numpy(valid_tok_np),
        valid_req_mask=torch.from_numpy(valid_req_np),
        valid_draft_mask=torch.from_numpy(valid_draft_np),
        seq_lens_is_step1=True,
    )

    assert torch.equal(drafts_out, drafts)
    assert next_positions.shape == (3, padded_total_tokens)
    expected_pos_row = torch.tensor(
        [15, 16, 17, 18, 32, 33, 34, 35, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    for row_idx in range(3):
        assert torch.equal(next_positions[row_idx], expected_pos_row)
    expected_next_seq_lens = torch.tensor(
        [16 + spec_k, 33 + spec_k, 0, 0], dtype=torch.int32
    )
    assert torch.equal(next_seq_lens, expected_next_seq_lens)
    expected_next_input_ids = torch.tensor(
        [100, 101, 102, 103, 200, 201, 202, 203, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    assert torch.equal(next_input_ids, expected_next_input_ids)
    expected_next_drafts = torch.tensor(
        [101, 102, 103, 201, 202, 203, 0, 0, 0, 0, 0, 0],
        dtype=torch.int32,
    )
    assert torch.equal(next_draft_token_ids, expected_next_drafts)
    assert torch.equal(src_padded[:8], expected_next_input_ids[:8])


def test_fold_draft_loop_epilogue_warmup_mismatched_num_reqs_and_token_paddings() -> (
    None
):
    """Verify 1D valid_draft_mask of length padded_total_tokens produces
    next_draft_token_ids of shape (4,) when num_reqs = 1, padded_num_reqs = 8."""
    num_reqs = 1
    padded_num_reqs = 8
    knr = 8
    spec_k = 3
    padded_total_tokens = 4
    assemble_bound = padded_num_reqs * (1 + spec_k)

    drafts = torch.tensor(
        [[11, 12, 13], [99, 98, 97], [0, 0, 0], [0, 0, 0]],
        dtype=torch.int32,
    )
    pos_1d = np.array([20, 20, 20, 20], dtype=np.int32)
    pos_step1_2d = torch.from_numpy(np.stack([pos_1d, pos_1d, pos_1d], axis=0))
    seq_lens_end = torch.tensor([22, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    bonus_tokens = torch.tensor(
        [[10], [-1], [-1], [-1], [-1], [-1], [-1], [-1]], dtype=torch.int32
    )

    req_expand_np = np.zeros(padded_total_tokens, dtype=np.int64)
    offsets_np = np.arange(padded_total_tokens, dtype=np.int32)
    valid_tok_np = np.ones(padded_total_tokens, dtype=np.int32)
    valid_req_np = np.zeros(knr, dtype=np.int32)
    valid_req_np[:num_reqs] = 1
    valid_draft_np = np.zeros(padded_total_tokens, dtype=np.int32)
    valid_draft_np[: min(num_reqs * spec_k, padded_total_tokens)] = 1

    (
        drafts_out,
        src_padded,
        next_seq_lens,
        next_positions,
        next_input_ids,
        next_draft_token_ids,
    ) = fold_draft_loop_spec_bookkeeping(
        drafts=drafts,
        pos_step1=pos_step1_2d,
        seq_lens_end=seq_lens_end,
        bonus_tokens=bonus_tokens,
        assemble_bound=assemble_bound,
        num_loop_steps=spec_k - 1,
        pos_req_expand_idx=torch.from_numpy(req_expand_np),
        pos_step1_offsets=torch.from_numpy(offsets_np),
        valid_token_mask=torch.from_numpy(valid_tok_np),
        valid_req_mask=torch.from_numpy(valid_req_np),
        valid_draft_mask=torch.from_numpy(valid_draft_np),
        seq_lens_is_step1=False,
    )

    assert torch.equal(drafts_out, drafts)
    assert src_padded.shape == (assemble_bound,)
    assert next_seq_lens.shape == (knr,)
    assert next_positions.shape == (3, padded_total_tokens)
    assert next_input_ids.shape == (padded_total_tokens,)
    assert next_draft_token_ids.shape == (padded_total_tokens,)
    assert torch.equal(
        next_draft_token_ids, torch.tensor([11, 12, 13, 0], dtype=torch.int32)
    )

    target_logits_argmax = torch.tensor([11, 12, 99, 10], dtype=torch.int32)
    accepted = next_draft_token_ids == target_logits_argmax
    assert torch.equal(
        accepted, torch.tensor([True, True, False, False], dtype=torch.bool)
    )

    logits_indices = torch.arange(padded_total_tokens, dtype=torch.int32)
    target_logits_indices = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
    ref_extracted = extract_draft_token_ids(
        next_input_ids, logits_indices, target_logits_indices
    )
    assert torch.equal(
        next_draft_token_ids[: num_reqs * spec_k],
        ref_extracted[: num_reqs * spec_k],
    )


def _fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Should not build greedy fold prep when all_greedy=False")


def _build_test_padded_params(
    cur_start: int, cur_end: int, bonus_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = cur_end - cur_start
    device = bonus_logits.device
    return (
        torch.full((count,), 0.8, dtype=torch.float32, device=device),
        torch.full((count,), -1, dtype=torch.int32, device=device),
        torch.full((count,), 1.0, dtype=torch.float32, device=device),
    )


class _FakeRejectionSampler:
    synthetic_mode = False
    synthetic_conditional_rates = None

    def __call__(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        return torch.tensor([[5, 6, -1, -1]], dtype=torch.int32)


def test_sample_spec_verify_chunk_non_greedy_skips_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _sample_spec_verify_chunk does not run greedy fold prep when
    all_greedy=False even with TPU_FOLD_SPEC_BOOKKEEPING=True."""
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    monkeypatch.setattr(
        TPUModelRunner, "_build_fold_verify_and_draft_prep_inputs", _fail_if_called
    )
    draft_chunk = types.SimpleNamespace(
        num_reqs=1,
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        position_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        attn_ctx=types.SimpleNamespace(
            seq_lens=torch.tensor([14], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        ),
        sequence_layout_plan=None,
        _precomputed_draft_prep=None,
        _last_fused_loop_pos_sl=None,
    )
    fake_runner = types.SimpleNamespace(
        rejection_sampler=_FakeRejectionSampler(),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        _is_async_drafter=True,
        input_batch=types.SimpleNamespace(top_k_cpu=np.array([0], dtype=np.int32)),
        spec_gather_bonus_and_target_logits=lambda logits, b_idx, t_idx: (
            logits[b_idx],
            logits[t_idx],
        ),
        _build_padded_sampling_params=_build_test_padded_params,
        _sampler_topk_candidates=lambda *_a, **_kw: 0,
        _sampler_applies_topk_topp=lambda *_a, **_kw: False,
        sample_from_logits_func=lambda *_a, **_kw: torch.tensor([6], dtype=torch.int32),
        spec_prepare_non_greedy_rejection_inputs=lambda *_a, **_kw: (
            torch.zeros((3, 8), dtype=torch.float32),
            torch.zeros((3, 8), dtype=torch.float32),
        ),
    )
    metadata = types.SimpleNamespace(
        draft_token_ids=torch.tensor([2, 3, 4], dtype=torch.int32),
        draft_lengths=torch.tensor([3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0], dtype=torch.int32),
        group_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    logits = torch.zeros((4, 8), dtype=torch.float32)
    out = TPUModelRunner._sample_spec_verify_chunk(
        cast(Any, fake_runner),
        logits=logits,
        md=metadata,
        grammar_output=None,
        scheduler_output=cast(Any, None),
        cur_start_idx=0,
        cur_end_idx=1,
        all_greedy=False,
        sampling_generator=None,
        top_tokens=False,
        draft_chunk=cast(Any, draft_chunk),
        mamba_indices=None,
        can_defer_mamba_offsets=True,
    )
    assert draft_chunk._precomputed_draft_prep is None
    assert torch.equal(out, torch.tensor([[5, 6, -1, -1]], dtype=torch.int32))


def test_real_eagle3_proposer_fold_verify_attribute_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiate real Eagle3Proposer and run fold-verify builder."""
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    fake_vllm_config = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            draft_tensor_parallel_size=8,
            draft_model_config=types.SimpleNamespace(
                max_model_len=2048,
                get_hidden_size=lambda: 64,
                hf_text_config=types.SimpleNamespace(model_type="qwen3_5_moe_mtp"),
            ),
            draft_parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
        ),
        model_config=types.SimpleNamespace(
            dtype=torch.bfloat16,
            max_model_len=2048,
            hf_text_config=types.SimpleNamespace(model_type="qwen3_5_moe"),
        ),
        cache_config=types.SimpleNamespace(block_size=16),
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=8,
            max_num_batched_tokens=64,
            async_scheduling=True,
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
        compilation_config=types.SimpleNamespace(),
    )
    fake_runner_for_init = types.SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_model_len=2048,
        max_num_reqs=8,
        max_num_tokens=64,
        num_tokens_paddings=[16, 32, 64],
        _dp_lockstep_enabled=lambda: False,
    )

    real_proposer = Eagle3Proposer(
        vllm_config=cast(Any, fake_vllm_config),
        runner=cast(Any, fake_runner_for_init),
    )

    required_proposer_attrs = (
        "_async_pad_req_idx_cache",
        "_zero_rej_dev_cache",
        "_loop_qsl_reqdist_cache",
        "_mtp_full_draft_loop",
        "_mtp_full_draft_loop_warm",
        "_mtp_fused_loop",
        "_mtp_fused_loop_warm",
        "_share_mtp_indices",
        "_draft_kv_shared",
        "advance_draft_positions",
    )
    for attr_name in required_proposer_attrs:
        assert hasattr(real_proposer, attr_name), (
            f"Missing {attr_name} on Eagle3Proposer"
        )
    assert isinstance(real_proposer._async_pad_req_idx_cache, dict)

    draft_chunk = types.SimpleNamespace(
        num_reqs=2,
        start_index=0,
        req_ids=["r0", "r1"],
        input_ids=torch.arange(100, 116, dtype=torch.int32),
        position_ids=torch.arange(16, dtype=torch.int64),
        query_start_loc_np=np.array([0, 4, 8], dtype=np.int32),
        sequence_layout_plan=types.SimpleNamespace(kind=SequenceLayoutKind.ALL),
        attn_ctx=types.SimpleNamespace(
            use_max_model_len=False,
            seq_lens_cpu=np.array([20, 30, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            seq_lens=torch.tensor([20, 30, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
            query_start_loc=torch.tensor(
                [0, 4, 8, 8, 8, 8, 8, 8, 8], dtype=torch.int32
            ),
        ),
        _precomputed_draft_prep=None,
        _last_fused_loop_pos_sl=None,
    )
    runner_ns = types.SimpleNamespace(
        device=torch.device("cpu"),
        drafter=real_proposer,
        _is_async_drafter=True,
        max_num_reqs=8,
        uses_mrope=False,
        requests={},
        input_batch=types.SimpleNamespace(req_ids=["r0", "r1"]),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        num_reqs_max_model_len=8,
        num_reqs_most_model_len=8,
        mamba_slot_read_offsets=None,
        _mamba_state_pos=None,
        _last_step_used_static_decode=False,
        _folded_steady_state_prep=None,
        _fold_epilogue_mask_cache={},
    )

    fold_inputs = TPUModelRunner._build_fold_verify_and_draft_prep_inputs(
        cast(Any, runner_ns),
        draft_chunk=cast(Any, draft_chunk),
        mamba_indices=None,
        can_defer_mamba_offsets=False,
    )
    assert fold_inputs is not None
    assert len(fold_inputs) == 11
    assert (16, 2, "cpu") in real_proposer._async_pad_req_idx_cache


@pytest.mark.parametrize("step0_in_loop", [False, True])
def test_fold_spec_bookkeeping_b8_equivalence(
    monkeypatch: pytest.MonkeyPatch, step0_in_loop: bool
) -> None:
    """Verify C3 fold bookkeeping at B=8 produces bit-identical drafts,
    src_padded, next_seq_lens, next_positions, next_input_ids, and
    next_draft_token_ids as the unfolded baseline, parametrized over
    step0_in_loop (exercising _run_mtp_fused_loop and _run_mtp_full_draft_loop)."""
    import vllm.forward_context as vllm_fwd_ctx

    import vllm_torchtpu.spec_decode.eagle3 as eagle3_mod

    monkeypatch.setattr(
        eagle3_mod, "set_model_tag", lambda *_a, **_kw: contextlib.nullcontext()
    )
    monkeypatch.setattr(eagle3_mod, "set_forward_context", _fake_set_forward_context)
    monkeypatch.setattr(vllm_fwd_ctx, "get_forward_context", _fake_get_forward_context)
    monkeypatch.setattr(
        eagle3_mod,
        "set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    for method_name in (
        "_draft_seed_input_ids",
        "_draft_gather_carries",
        "_draft_propose_token",
        "_draft_combine_hidden_states",
    ):
        raw_fn = getattr(Eagle3Proposer, method_name).__wrapped__.__wrapped__
        monkeypatch.setattr(Eagle3Proposer, method_name, raw_fn)

    max_num_reqs = 8
    spec_k = 3
    query_len = 1 + spec_k
    hidden_size = 4
    draft_model = _DummyMtpDraft(hidden_size=hidden_size, vocab_size=64)

    proposer_ref, _ = _make_b8_test_proposer(
        draft_model=draft_model,
        max_num_reqs=max_num_reqs,
        spec_k=spec_k,
        step0_in_loop=step0_in_loop,
    )
    proposer_test, specializer = _make_b8_test_proposer(
        draft_model=draft_model,
        max_num_reqs=max_num_reqs,
        spec_k=spec_k,
        step0_in_loop=step0_in_loop,
    )
    proposer_ref._draft_tp_matches_target = True
    proposer_test._draft_tp_matches_target = True
    proposer_ref._async_pad_req_idx_cache = {}
    proposer_test._async_pad_req_idx_cache = {}

    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", step0_in_loop)
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)

    # Warmup at B=1 if step0_in_loop.
    if specializer is not None:
        proposer_test.runner.input_batch.num_reqs = 1
        warmup_chunk = _build_b8_chunk(
            num_reqs=1,
            max_num_reqs=max_num_reqs,
            token_counts_per_req=(4,),
            padded_total_tokens=4,
            hidden_size=hidden_size,
            spec_k=spec_k,
        )
        warmup_chunk._precomputed_draft_prep = None
        warmup_chunk._last_fused_loop_pos_sl = None
        proposer_test.draft_chunks = [warmup_chunk]
        _ = proposer_test.propose(
            sampled_token_ids=[[15]],
            discard_sampled_tokens_req_indices=[],
            num_rejected_tokens_np=np.array([0], dtype=np.int32),
            scheduler_output=None,
            return_device=True,
        )
        assert specializer.specialized_gather_width == 8

    # Run B=8 with fold_verify_and_draft_prep + propose + epilogue.
    token_counts = (4, 4, 4, 4, 4, 4, 4, 4)
    rejected_counts = (0, 1, 2, 3, 1, 0, 2, 3)
    next_tokens_dev = torch.full(
        (max_num_reqs, spec_k + 1), INVALID_TOKEN_ID, dtype=torch.int32
    )
    for req_idx in range(max_num_reqs):
        acc_count = spec_k - rejected_counts[req_idx]
        for col_idx in range(acc_count):
            next_tokens_dev[req_idx, col_idx] = 200 + req_idx * 10 + col_idx
        next_tokens_dev[req_idx, acc_count] = 30 + req_idx

    chunk_ref = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    chunk_ref._precomputed_draft_prep = None
    chunk_ref._last_fused_loop_pos_sl = None
    chunk_test = _build_b8_chunk(
        num_reqs=max_num_reqs,
        max_num_reqs=max_num_reqs,
        token_counts_per_req=token_counts,
        padded_total_tokens=32,
        hidden_size=hidden_size,
        spec_k=spec_k,
    )
    chunk_test._last_fused_loop_pos_sl = None

    runner_ns = types.SimpleNamespace(
        device=torch.device("cpu"),
        drafter=proposer_test,
        _is_async_drafter=True,
        max_num_reqs=max_num_reqs,
        uses_mrope=False,
        requests={},
        input_batch=types.SimpleNamespace(
            req_ids=[f"req_{req_idx}" for req_idx in range(max_num_reqs)]
        ),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=spec_k),
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        num_reqs_max_model_len=max_num_reqs,
        num_reqs_most_model_len=max_num_reqs,
        mamba_slot_read_offsets=None,
        _mamba_state_pos=None,
        _last_step_used_static_decode=False,
        _folded_steady_state_prep=None,
        _fold_epilogue_mask_cache={},
    )
    fold_inputs = TPUModelRunner._build_fold_verify_and_draft_prep_inputs(
        cast(Any, runner_ns),
        draft_chunk=cast(Any, chunk_test),
        mamba_indices=None,
        can_defer_mamba_offsets=False,
    )
    assert fold_inputs is not None
    (
        f_chunk_ids,
        f_qsl,
        f_pad_req_idx,
        f_positions,
        f_base_sl,
        f_adv_pos,
        f_max_reqs,
        _f_ro,
        _f_sg,
        f_epilogue_tensors,
        f_steady_key,
    ) = fold_inputs

    prep_out = fold_verify_and_draft_prep(
        next_tokens=next_tokens_dev,
        chunk_input_ids=f_chunk_ids,
        draft_lengths=chunk_test.draft_lengths,
        query_start_loc=f_qsl,
        pad_req_idx=f_pad_req_idx,
        positions=f_positions,
        base_seq_lens=f_base_sl,
        advance_draft_positions=f_adv_pos,
        max_num_reqs=f_max_reqs,
    )
    chunk_test._precomputed_draft_prep = _PrecomputedDraftPrep(
        draft_input_ids=prep_out[0],
        gather_indices=prep_out[1],
        num_rejected_knr=prep_out[2],
        pos_step1=prep_out[3],
        seq_lens_step1=prep_out[4],
        bonus_full=prep_out[5],
        spec_num_rejected=prep_out[6],
        mamba_offsets_updated=False,
        pos_req_expand_idx=f_epilogue_tensors[0],
        pos_step1_offsets=f_epilogue_tensors[1],
        valid_token_mask=f_epilogue_tensors[2],
        valid_req_mask=f_epilogue_tensors[3],
        valid_draft_mask=f_epilogue_tensors[4],
        steady_state_key=f_steady_key,
    )

    proposer_ref.runner.input_batch.num_reqs = max_num_reqs
    proposer_test.runner.input_batch.num_reqs = max_num_reqs
    proposer_ref.draft_chunks = [chunk_ref]
    proposer_test.draft_chunks = [chunk_test]

    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", step0_in_loop)
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", False)
    drafts_ref = proposer_ref.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=None,
        return_device=True,
        next_tokens_per_chunk=[next_tokens_dev],
    )

    monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", step0_in_loop)
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    drafts_test = proposer_test.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=None,
        return_device=True,
        next_tokens_per_chunk=[next_tokens_dev],
    )
    assert isinstance(drafts_ref, torch.Tensor)
    assert isinstance(drafts_test, torch.Tensor)
    assert torch.equal(drafts_test, drafts_ref)

    prep_after = chunk_test._precomputed_draft_prep
    assert prep_after is not None
    assert prep_after.src_padded is not None
    ref_src = assemble_spec_next_tokens(next_tokens_dev, drafts_ref, max_num_reqs)
    assert torch.equal(prep_after.src_padded[: max_num_reqs * query_len], ref_src)
    assert prep_after.next_input_ids is not None
    assert torch.equal(prep_after.next_input_ids[: max_num_reqs * query_len], ref_src)

    # Unfolded reference next-step subtraction and substitution.
    base_computed = np.array(
        [20 + req_idx * 7 for req_idx in range(max_num_reqs)], dtype=np.int32
    )
    optimistic_computed_next = base_computed.copy() + query_len
    optimistic_seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
    optimistic_seq_lens[:max_num_reqs] = torch.from_numpy(
        optimistic_computed_next[:max_num_reqs] + query_len
    )
    padded_total_tokens = max_num_reqs * query_len
    optimistic_positions = torch.zeros(padded_total_tokens, dtype=torch.int32)
    for req_idx in range(max_num_reqs):
        for offset in range(query_len):
            optimistic_positions[req_idx * query_len + offset] = (
                int(optimistic_computed_next[req_idx]) + offset
            )
    optimistic_input_ids = torch.zeros(padded_total_tokens, dtype=torch.int32)

    seq_sub_idx = torch.full((max_num_reqs,), -1, dtype=torch.int32)
    seq_sub_idx[:max_num_reqs] = torch.arange(max_num_reqs, dtype=torch.int32)
    pos_sub_idx = torch.full((padded_total_tokens,), -1, dtype=torch.int32)
    for req_idx in range(max_num_reqs):
        pos_sub_idx[req_idx * query_len : (req_idx + 1) * query_len] = req_idx

    cur_in_idx = torch.arange(padded_total_tokens, dtype=torch.int64)
    pre_nt_idx = torch.full((padded_total_tokens,), -1, dtype=torch.int64)
    pre_nt_idx[: max_num_reqs * query_len] = torch.arange(
        max_num_reqs * query_len, dtype=torch.int64
    )
    substitution_indices = torch.stack([cur_in_idx, pre_nt_idx], dim=0)

    logits_indices = torch.arange(padded_total_tokens, dtype=torch.int32)
    target_logits_list: list[int] = []
    for req_idx in range(max_num_reqs):
        for offset in range(spec_k):
            target_logits_list.append(req_idx * query_len + offset)
    target_logits_indices = torch.tensor(target_logits_list, dtype=torch.int32)

    (
        ref_next_seq_lens,
        ref_next_positions,
        ref_next_input_ids,
        ref_next_draft_token_ids,
    ) = subtract_and_substitute_draft_token_ids(
        seq_lens=optimistic_seq_lens,
        positions=optimistic_positions,
        num_rejected=prep_after.spec_num_rejected,
        seq_lens_subtract_indices=seq_sub_idx,
        positions_subtract_indices=pos_sub_idx,
        input_ids=optimistic_input_ids,
        substitution_indices=substitution_indices,
        next_tokens_tpu=prep_after.src_padded,
        logits_indices=logits_indices,
        target_logits_indices=target_logits_indices,
    )

    assert prep_after.next_seq_lens is not None
    assert prep_after.next_positions is not None
    assert prep_after.next_draft_token_ids is not None
    assert torch.equal(prep_after.next_seq_lens, ref_next_seq_lens)
    assert torch.equal(prep_after.next_positions, ref_next_positions)
    assert torch.equal(prep_after.next_input_ids, ref_next_input_ids)
    assert torch.equal(
        prep_after.next_draft_token_ids[: max_num_reqs * spec_k],
        ref_next_draft_token_ids,
    )


def test_assemble_async_spec_substitution_fold_early_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _assemble_async_spec_substitution returns precomputed folded
    tensors on CPU/TPU when TPU_FOLD_SPEC_BOOKKEEPING=1, and runs standard
    assembly when off."""
    runner = _make_runner(
        num_computed_tokens=[10],
        prompt_tokens=[10],
        scheduled_tokens=[4],
        token_paddings=[4],
        max_num_reqs=1,
    )
    src_padded = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    spec_rejected = torch.tensor([1], dtype=torch.int32)
    precomputed = _PrecomputedDraftPrep(
        draft_input_ids=torch.zeros(4, dtype=torch.int32),
        gather_indices=torch.zeros(1, dtype=torch.int64),
        num_rejected_knr=torch.zeros(1, dtype=torch.int32),
        pos_step1=torch.zeros(1, dtype=torch.int32),
        seq_lens_step1=torch.zeros(1, dtype=torch.int32),
        bonus_full=torch.zeros((1, 1), dtype=torch.int32),
        spec_num_rejected=spec_rejected,
        src_padded=src_padded,
    )
    draft_chunk = types.SimpleNamespace(_precomputed_draft_prep=precomputed)
    metadata = types.SimpleNamespace(
        draft_lengths_cpu=np.array([3], dtype=np.int32),
    )
    state = types.SimpleNamespace(
        num_reqs_list=[1],
        spec_decode_metadata_list=[metadata],
        draft_chunks=[draft_chunk],
    )

    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    runner.max_num_reqs = 1
    nt_chunk = torch.tensor([[10, 11, -1, -1]], dtype=torch.int32)
    res = TPUModelRunner._assemble_async_spec_substitution(
        runner,
        drafts=torch.tensor([[11, 12, 13]], dtype=torch.int32),
        next_tokens_per_chunk=[nt_chunk],
        state=state,
    )
    assert len(res) == 4
    assert res[0] == [src_padded]
    assert res[1] == {0: 0}
    assert res[2] is spec_rejected
    assert res[3] == {0: 3}
    assert runner._folded_steady_state_prep is precomputed


def test_prepare_inputs_core_folded_steady_state_and_skips_h2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _prepare_inputs_core consumes precomputed folded steady-state
    buffers directly and skips host H2D transfers and subtraction."""
    runner = _make_runner(
        num_computed_tokens=[16],
        prompt_tokens=[16],
        scheduled_tokens=[4],
        token_paddings=[4],
        max_num_reqs=1,
        pcp_size=1,
    )
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=3)
    runner.scheduler_config = types.SimpleNamespace(async_scheduling=True)

    next_sl = torch.tensor([24], dtype=torch.int32)
    next_pos = torch.tensor([20, 21, 22, 23], dtype=torch.int32)
    next_in_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    next_drafts = torch.tensor([102, 103, 104], dtype=torch.int32)
    spec_rejected = torch.tensor([0], dtype=torch.int32)
    src_padded = torch.tensor([101, 102, 103, 104], dtype=torch.int32)

    steady_key = (1, 1, 1, 4, 4)
    precomputed = _PrecomputedDraftPrep(
        draft_input_ids=torch.zeros(4, dtype=torch.int32),
        gather_indices=torch.zeros(1, dtype=torch.int64),
        num_rejected_knr=torch.zeros(1, dtype=torch.int32),
        pos_step1=torch.zeros(1, dtype=torch.int32),
        seq_lens_step1=torch.zeros(1, dtype=torch.int32),
        bonus_full=torch.zeros((1, 1), dtype=torch.int32),
        spec_num_rejected=spec_rejected,
        src_padded=src_padded,
        next_seq_lens=next_sl,
        next_positions=next_pos,
        next_input_ids=next_in_ids,
        next_draft_token_ids=next_drafts,
        steady_state_key=steady_key,
    )
    runner._folded_steady_state_prep = precomputed
    req_id = runner.input_batch.req_ids[0]
    runner._pre_async_results = types.SimpleNamespace(
        spec_decode_num_rejected_tokens=spec_rejected,
        next_tokens_tpu=src_padded,
        req_id_to_index_copy={req_id: 0},
    )
    captured_metadata = types.SimpleNamespace(
        target_logits_indices=torch.zeros(3, dtype=torch.int32),
        final_logits_indices=torch.zeros(4, dtype=torch.int32),
        draft_token_ids=None,
    )
    runner.spec_decode_manager = types.SimpleNamespace(
        get_spec_decode_metadata=lambda *_a, **_kw: captured_metadata
    )

    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={req_id: 4},
        scheduled_spec_decode_tokens={req_id: (1, 2, 3)},
        scheduled_new_reqs=None,
        finished_req_ids=None,
        scheduled_cached_reqs=None,
    )

    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", False)
    TPUModelRunner._prepare_inputs_core(
        runner,
        scheduler_output=scheduler_output,
        start_index=0,
        num_decode_reqs=1,
        use_static_decode=False,
    )
    assert runner._fused_async_subst_done is True
    assert runner.input_ids is next_in_ids
    assert runner.position_ids is next_pos
    assert captured_metadata.draft_token_ids is next_drafts


@pytest.mark.parametrize(
    "fallback_reason",
    [
        "batch_membership_change",
        "non_uniform_request",
        "steady_key_mismatch",
        "static_verify_enabled",
    ],
)
def test_prepare_inputs_core_folded_steady_state_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    fallback_reason: str,
) -> None:
    """Verify _prepare_inputs_core cleanly falls back when folded prerequisites
    are violated (including when TPU_STATIC_DECODE_INPUTS_VERIFY=1)."""
    runner = _make_runner(
        num_computed_tokens=[16],
        prompt_tokens=[16],
        scheduled_tokens=[4],
        token_paddings=[4],
        max_num_reqs=1,
        pcp_size=1,
    )
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=3)
    runner.scheduler_config = types.SimpleNamespace(async_scheduling=True)

    next_sl = torch.tensor([24], dtype=torch.int32)
    next_pos = torch.tensor([20, 21, 22, 23], dtype=torch.int32)
    next_in_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
    next_drafts = torch.tensor([102, 103, 104], dtype=torch.int32)
    spec_rejected = torch.tensor([0], dtype=torch.int32)
    src_padded = torch.tensor([101, 102, 103, 104], dtype=torch.int32)

    steady_key = (
        (1, 1, 1, 4, 4)
        if fallback_reason != "steady_key_mismatch"
        else (999, 1, 1, 4, 4)
    )
    precomputed = _PrecomputedDraftPrep(
        draft_input_ids=torch.zeros(4, dtype=torch.int32),
        gather_indices=torch.zeros(1, dtype=torch.int64),
        num_rejected_knr=torch.zeros(1, dtype=torch.int32),
        pos_step1=torch.zeros(1, dtype=torch.int32),
        seq_lens_step1=torch.zeros(1, dtype=torch.int32),
        bonus_full=torch.zeros((1, 1), dtype=torch.int32),
        spec_num_rejected=spec_rejected,
        src_padded=src_padded,
        next_seq_lens=next_sl,
        next_positions=next_pos,
        next_input_ids=next_in_ids,
        next_draft_token_ids=next_drafts,
        steady_state_key=steady_key,
    )
    runner._folded_steady_state_prep = precomputed
    req_id = runner.input_batch.req_ids[0]
    runner._pre_async_results = types.SimpleNamespace(
        spec_decode_num_rejected_tokens=spec_rejected,
        next_tokens_tpu=src_padded,
        req_id_to_index_copy={req_id: 0},
    )
    current_req_id = (
        req_id if fallback_reason != "batch_membership_change" else "different_req"
    )
    runner.input_batch.req_ids = [current_req_id]
    runner.input_batch.req_id_to_index = {current_req_id: 0}
    captured_metadata = types.SimpleNamespace(
        target_logits_indices=torch.zeros(3, dtype=torch.int32),
        final_logits_indices=torch.zeros(4, dtype=torch.int32),
        draft_token_ids=None,
    )
    runner.spec_decode_manager = types.SimpleNamespace(
        get_spec_decode_metadata=lambda *_a, **_kw: captured_metadata
    )

    sched_count = 4 if fallback_reason != "non_uniform_request" else 2
    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=sched_count,
        num_scheduled_tokens={current_req_id: sched_count},
        scheduled_spec_decode_tokens={
            current_req_id: (1, 2, 3) if sched_count == 4 else (1,)
        },
        scheduled_new_reqs=None,
        finished_req_ids=None,
        scheduled_cached_reqs=None,
    )

    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    monkeypatch.setattr(
        envs,
        "TPU_STATIC_DECODE_INPUTS_VERIFY",
        fallback_reason == "static_verify_enabled",
    )
    TPUModelRunner._prepare_inputs_core(
        runner,
        scheduler_output=scheduler_output,
        start_index=0,
        num_decode_reqs=1,
        use_static_decode=False,
    )
    assert runner._fused_async_subst_done is False
    assert runner.input_ids is not next_in_ids


@pytest.mark.parametrize("top_tokens", [False, True])
def test_sample_spec_verify_chunk_fold_branches(
    monkeypatch: pytest.MonkeyPatch, top_tokens: bool
) -> None:
    """Verify _sample_spec_verify_chunk correctly executes the fold helper
    for both top_tokens and logits paths."""
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    draft_chunk = types.SimpleNamespace(
        num_reqs=1,
        start_index=0,
        input_ids=torch.arange(100, 104, dtype=torch.int32),
        position_ids=torch.arange(4, dtype=torch.int64),
        attn_ctx=types.SimpleNamespace(
            seq_lens=torch.tensor([10], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        ),
        sequence_layout_plan=types.SimpleNamespace(kind=SequenceLayoutKind.ALL),
        _precomputed_draft_prep=None,
        _last_fused_loop_pos_sl=None,
    )
    proposer = object.__new__(Eagle3Proposer)
    proposer._draft_tp_matches_target = True
    proposer._loop_bucket = lambda _num: 16
    proposer._async_pad_req_idx_cache = {}
    proposer._mtp_full_draft_loop = None
    proposer._draft_kv_shared = False

    fake_runner = types.SimpleNamespace(
        device=torch.device("cpu"),
        drafter=proposer,
        rejection_sampler=_FakeRejectionSampler(),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        _is_async_drafter=True,
        input_batch=types.SimpleNamespace(req_ids=["req_0"]),
        max_num_reqs=8,
        uses_mrope=False,
        requests={},
        mamba_slot_read_offsets=None,
        _fold_epilogue_mask_cache={},
    )
    fake_runner._build_fold_verify_and_draft_prep_inputs = (
        TPUModelRunner._build_fold_verify_and_draft_prep_inputs.__get__(fake_runner)
    )
    fake_runner._run_fold_greedy_verify_and_draft_prep = (
        TPUModelRunner._run_fold_greedy_verify_and_draft_prep.__get__(fake_runner)
    )

    metadata = types.SimpleNamespace(
        draft_token_ids=torch.tensor([2, 3, 4], dtype=torch.int32),
        draft_lengths=torch.tensor([3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0], dtype=torch.int32),
        group_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    logits = (
        torch.tensor([10, 11, 12, 13], dtype=torch.int32)
        if top_tokens
        else torch.zeros((4, 64), dtype=torch.float32)
    )

    selected = TPUModelRunner._sample_spec_verify_chunk(
        cast(Any, fake_runner),
        logits=logits,
        md=metadata,
        grammar_output=None,
        scheduler_output=cast(Any, None),
        cur_start_idx=0,
        cur_end_idx=1,
        all_greedy=True,
        sampling_generator=None,
        top_tokens=top_tokens,
        draft_chunk=cast(Any, draft_chunk),
        mamba_indices=None,
        can_defer_mamba_offsets=True,
    )
    assert isinstance(selected, torch.Tensor)
    assert isinstance(draft_chunk._precomputed_draft_prep, _PrecomputedDraftPrep)


def test_dflash_drafter_rejection_at_init_and_prep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DFlash drafter is rejected with NotImplementedError at runner init
    via the post-construction Eagle3Proposer check, and returns None from
    _build_fold_verify_and_draft_prep_inputs."""
    monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", True)
    monkeypatch.setattr(tpu_runner_mod, "static_verify_q_len", lambda _cfg: 4)
    monkeypatch.setattr(
        tpu_runner_mod,
        "DFlashProposer",
        lambda *_a, **_kw: object.__new__(DFlashProposer),
    )

    runner = types.SimpleNamespace(
        speculative_config=types.SimpleNamespace(
            method="dflash",
            rejection_sample_method=None,
        ),
        _pp_is_last=True,
        device=torch.device("cpu"),
        vllm_config=types.SimpleNamespace(),
    )
    expected_match = (
        r"TPU_FOLD_SPEC_BOOKKEEPING=1 is only supported with "
        r"Eagle3Proposer, got DFlashProposer\."
    )
    with pytest.raises(NotImplementedError, match=expected_match):
        TPUModelRunner._init_speculative_decoding(cast(Any, runner))

    dflash_drafter = object.__new__(DFlashProposer)
    runner_with_dflash = types.SimpleNamespace(drafter=dflash_drafter)
    prep_inputs = TPUModelRunner._build_fold_verify_and_draft_prep_inputs(
        cast(Any, runner_with_dflash),
        draft_chunk=cast(Any, types.SimpleNamespace()),
        mamba_indices=None,
        can_defer_mamba_offsets=False,
    )
    assert prep_inputs is None


@pytest.mark.parametrize("step0_in_loop", [False, True])
def test_runner_two_consecutive_spec_steps_folded_matches_unfolded(
    monkeypatch: pytest.MonkeyPatch,
    step0_in_loop: bool,
) -> None:
    """End-to-end two-step runner equivalence test (`num_reqs = 2 < max_num_reqs = 4`):
    wires `_make_runner`, `SpeculativeDecodingManager`, `RejectionSampler`, and
    `Eagle3Proposer` across two consecutive async spec-verify steps, asserting
    bit-for-bit equality between `TPU_FOLD_SPEC_BOOKKEEPING=0` and `=1` on Step 1
    verification/drafts/Mamba slot read offsets and Step 2 `seq_lens`,
    `position_ids`, substituted `input_ids`, `draft_token_ids`, and Step 2
    rejection sampling, while proving Step 2 skips `subtract_num_rejected_tokens`,
    `_apply_async_token_substitution`, and `extract_draft_token_ids`."""
    import vllm.forward_context as vllm_fwd_ctx

    import vllm_torchtpu.spec_decode.eagle3 as eagle3_mod

    monkeypatch.setattr(
        eagle3_mod, "set_model_tag", lambda *_a, **_kw: contextlib.nullcontext()
    )
    monkeypatch.setattr(eagle3_mod, "set_forward_context", _fake_set_forward_context)
    monkeypatch.setattr(vllm_fwd_ctx, "get_forward_context", _fake_get_forward_context)
    monkeypatch.setattr(
        eagle3_mod,
        "set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    for method_name in (
        "_draft_seed_input_ids",
        "_draft_gather_carries",
        "_draft_propose_token",
        "_draft_combine_hidden_states",
    ):
        raw_fn = getattr(Eagle3Proposer, method_name).__wrapped__.__wrapped__
        monkeypatch.setattr(Eagle3Proposer, method_name, raw_fn)

    # Unwrap backend="tpu" helpers so the unfolded reference runs natively on CPU.
    monkeypatch.setattr(
        rejection_sampler_mod,
        "_greedy_rejection_sample_with_segment",
        rejection_sampler_mod._greedy_rejection_sample_with_segment.__wrapped__,
    )
    monkeypatch.setattr(
        tpu_runner_mod,
        "_substitute_placeholder_token",
        tpu_runner_mod._substitute_placeholder_token.__wrapped__,
    )
    monkeypatch.setattr(
        TPUModelRunner,
        "spec_bonus_and_target_logits",
        TPUModelRunner.spec_bonus_and_target_logits.__wrapped__,
    )

    num_reqs = 2
    max_num_reqs = 4
    spec_k = 3
    query_len = 1 + spec_k
    vocab_size = 64
    hidden_size = 4
    draft_model = _DummyMtpDraft(hidden_size=hidden_size, vocab_size=vocab_size)

    def _run_two_steps(fold_enabled: bool) -> dict[str, Any]:
        monkeypatch.setattr(envs, "TPU_DRAFT_STEP0_IN_LOOP", step0_in_loop)
        monkeypatch.setattr(envs, "TPU_FOLD_SPEC_BOOKKEEPING", fold_enabled)
        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", False)

        runner = _make_runner(
            num_computed_tokens=[16, 24],
            prompt_tokens=[16, 24],
            scheduled_tokens=[query_len, query_len],
            token_paddings=[8, 16],
            max_num_reqs=max_num_reqs,
            pcp_size=1,
        )
        runner.scheduler_config.async_scheduling = True
        runner.speculative_config = types.SimpleNamespace(
            method="mtp",
            num_speculative_tokens=spec_k,
            rejection_sample_method=None,
        )
        runner.vllm_config.speculative_config = runner.speculative_config
        runner.spec_decode_manager = SpeculativeDecodingManager(runner)
        runner.rejection_sampler = RejectionSampler(
            runner.speculative_config, runner.device
        )
        runner._is_async_drafter = True
        runner._dp_lockstep_enabled = lambda: False
        runner._dp_num_tokens_across_dp = lambda _num_tok: None
        runner._last_sequence_layout_plan = None
        runner._prepare_async_token_substitution_indices = types.MethodType(
            TPUModelRunner._prepare_async_token_substitution_indices, runner
        )
        runner._apply_async_token_substitution = types.MethodType(
            TPUModelRunner._apply_async_token_substitution, runner
        )
        runner.spec_bonus_and_target_logits = types.MethodType(
            TPUModelRunner.spec_bonus_and_target_logits, runner
        )
        runner._build_fold_verify_and_draft_prep_inputs = types.MethodType(
            TPUModelRunner._build_fold_verify_and_draft_prep_inputs, runner
        )
        runner._run_fold_greedy_verify_and_draft_prep = types.MethodType(
            TPUModelRunner._run_fold_greedy_verify_and_draft_prep, runner
        )
        runner._drafts_per_chunk = TPUModelRunner._drafts_per_chunk

        proposer, _ = _make_b8_test_proposer(
            draft_model=draft_model,
            max_num_reqs=max_num_reqs,
            spec_k=spec_k,
            step0_in_loop=step0_in_loop,
        )
        proposer.runner = runner
        proposer._draft_tp_matches_target = True
        proposer._async_pad_req_idx_cache = {}
        runner.drafter = proposer

        # Pre-seed Mamba slot read offsets; slots 11, 13 are padded (>= num_reqs)
        # and start dirty to verify both paths reset padded slots to 0.
        runner._combined_request_distribution_cpu = torch.zeros(6, dtype=torch.int32)
        runner.mamba_slot_read_offsets = torch.zeros(16, dtype=torch.int32)
        runner.mamba_slot_read_offsets[11] = 7
        runner.mamba_slot_read_offsets[13] = 9
        mamba_indices = [torch.tensor([3, 7, 11, 13], dtype=torch.int32)]

        # Step 1 initial tokens: bonus + 3 initial drafts per request.
        init_req0 = [10, 11, 12, 13]
        init_req1 = [20, 21, 22, 23]
        runner.input_batch.token_ids_cpu[0, 16:20] = init_req0
        runner.input_batch.token_ids_cpu[1, 24:28] = init_req1

        sched1 = types.SimpleNamespace(
            total_num_scheduled_tokens=num_reqs * query_len,
            num_scheduled_tokens={"req0": query_len, "req1": query_len},
            scheduled_spec_decode_tokens={
                "req0": (11, 12, 13),
                "req1": (21, 22, 23),
            },
            scheduled_new_reqs=[],
            finished_req_ids=set(),
            scheduled_cached_reqs=types.SimpleNamespace(
                resumed_req_ids=set(), new_block_ids=[]
            ),
        )

        # --- STEP 1: _prepare_inputs_core -> verify -> propose -> assemble ---
        (
            attn_md_s1,
            _logits_idx_s1,
            _padded_nreqs_s1,
            _nreqs_s1,
            _end_idx_s1,
            _cur_in_idx_s1,
            _pre_nt_idx_s1,
            spec_md_s1,
            _req_major_ids_s1,
            layout_plan_s1,
        ) = TPUModelRunner._prepare_inputs_core(
            runner,
            scheduler_output=sched1,
            start_index=0,
            num_decode_reqs=0,
            use_static_decode=False,
        )
        assert spec_md_s1 is not None
        runner._last_sequence_layout_plan = layout_plan_s1
        padded_tot_s1 = int(runner.input_ids.shape[0])
        spec_md_s1.draft_token_ids = spec_md_s1.upload_deferred_draft_token_ids(
            runner.device
        )
        input_ids_s1 = runner.input_ids

        hidden_s1 = (
            torch.arange(padded_tot_s1 * hidden_size, dtype=torch.float32).reshape(
                padded_tot_s1, hidden_size
            )
            * 0.125
        )
        draft_chunk_s1 = DraftChunkInputs(
            input_ids=input_ids_s1,
            position_ids=runner.position_ids,
            query_start_loc_np=runner.query_start_loc_np[: num_reqs + 1].copy(),
            attn_ctx=runner._attn_metadata_builder_ctx,
            start_index=0,
            num_reqs=num_reqs,
            aux_hidden_states=[],
            sequence_layout_plan=layout_plan_s1,
            hidden_states=hidden_s1,
            draft_lengths=spec_md_s1.draft_lengths,
            attn_metadata=attn_md_s1,
        )

        # Deterministic Step 1 target logits:
        # - req0 (rows 0..3): argmax [11, 12, 41, 42] -> 2 accepted, 1 rejected.
        # - req1 (rows 4..7): argmax [51, 22, 23, 52] -> 0 accepted, 3 rejected.
        s1_argmax = [11, 12, 41, 42, 51, 22, 23, 52]
        logits_s1 = torch.zeros((padded_tot_s1, vocab_size), dtype=torch.float32)
        for row_i, tok_id in enumerate(s1_argmax):
            logits_s1[row_i, tok_id] = 10.0

        next_tokens_s1 = TPUModelRunner._sample_spec_verify_chunk(
            runner,
            logits=logits_s1,
            md=spec_md_s1,
            grammar_output=None,
            scheduler_output=cast(Any, sched1),
            cur_start_idx=0,
            cur_end_idx=num_reqs,
            all_greedy=True,
            sampling_generator=None,
            top_tokens=False,
            draft_chunk=draft_chunk_s1 if fold_enabled else None,
            mamba_indices=mamba_indices if fold_enabled else None,
            can_defer_mamba_offsets=fold_enabled,
        )
        pre_s1 = draft_chunk_s1._precomputed_draft_prep
        if not (pre_s1 is not None and pre_s1.mamba_offsets_updated):
            TPUModelRunner._update_mamba_slot_read_offsets(
                runner, mamba_indices, next_tokens_s1, num_reqs
            )
        mamba_after_s1 = runner.mamba_slot_read_offsets.clone()

        proposer.draft_chunks = [draft_chunk_s1]
        drafts_s1 = proposer.propose(
            sampled_token_ids=None,
            discard_sampled_tokens_req_indices=[],
            num_rejected_tokens_np=None,
            scheduler_output=sched1,
            return_device=True,
            next_tokens_per_chunk=[next_tokens_s1],
        )
        assert isinstance(drafts_s1, torch.Tensor)

        state_s1 = types.SimpleNamespace(
            num_reqs_list=[num_reqs],
            spec_decode_metadata_list=[spec_md_s1],
            draft_chunks=[draft_chunk_s1],
        )
        nt_chunks, next_tok_idx, spec_rej_dev, _num_draft_map = (
            TPUModelRunner._assemble_async_spec_substitution(
                runner,
                drafts=drafts_s1,
                next_tokens_per_chunk=[next_tokens_s1],
                state=state_s1,
            )
        )
        runner._pre_async_results = types.SimpleNamespace(
            next_tokens_tpu=nt_chunks[0],
            spec_decode_num_rejected_tokens=spec_rej_dev,
            req_id_to_index_copy={
                runner.input_batch.req_ids[req_i]: next_tok_idx[req_i]
                for req_i in range(num_reqs)
            },
        )

        # --- STEP 2: optimistic scheduler advance + steady-state handoff ---
        runner.input_batch.num_computed_tokens_cpu[:] += query_len
        runner.input_batch.token_ids_cpu[0, 20:24] = -1
        runner.input_batch.token_ids_cpu[1, 28:32] = -1

        sched2 = types.SimpleNamespace(
            total_num_scheduled_tokens=num_reqs * query_len,
            num_scheduled_tokens={"req0": query_len, "req1": query_len},
            scheduled_spec_decode_tokens={
                "req0": (-1, -1, -1),
                "req1": (-1, -1, -1),
            },
            scheduled_new_reqs=[],
            finished_req_ids=set(),
            scheduled_cached_reqs=types.SimpleNamespace(
                resumed_req_ids=set(), new_block_ids=[]
            ),
        )

        counts = {"subtract": 0, "subst": 0, "extract": 0}
        real_subtract = async_out_mod.subtract_num_rejected_tokens
        real_extract = async_out_mod.extract_draft_token_ids
        real_subst = runner._apply_async_token_substitution

        def counted_subtract(*args: Any, **kwargs: Any) -> Any:
            counts["subtract"] += 1
            return real_subtract(*args, **kwargs)

        def counted_extract(*args: Any, **kwargs: Any) -> Any:
            counts["extract"] += 1
            return real_extract(*args, **kwargs)

        def counted_subst(*args: Any, **kwargs: Any) -> Any:
            counts["subst"] += 1
            return real_subst(*args, **kwargs)

        monkeypatch.setattr(
            async_out_mod, "subtract_num_rejected_tokens", counted_subtract
        )
        monkeypatch.setattr(async_out_mod, "extract_draft_token_ids", counted_extract)
        runner._apply_async_token_substitution = counted_subst

        (
            attn_md_s2,
            _logits_idx_s2,
            _padded_nreqs_s2,
            _nreqs_s2,
            _end_idx_s2,
            cur_in_idx,
            pre_nt_idx,
            spec_md_s2,
            _req_major_ids_s2,
            layout_plan_s2,
        ) = TPUModelRunner._prepare_inputs_core(
            runner,
            scheduler_output=sched2,
            start_index=0,
            num_decode_reqs=0,
            use_static_decode=False,
        )
        assert spec_md_s2 is not None
        runner._last_sequence_layout_plan = layout_plan_s2
        padded_tot_s2 = int(runner.input_ids.shape[0])
        used_fused_subst = runner._fused_async_subst_done
        if runner._fused_async_subst_done:
            runner._fused_async_subst_done = False
            input_ids_s2 = runner.input_ids
        else:
            input_ids_s2 = runner._apply_async_token_substitution(
                runner.input_ids, cur_in_idx, pre_nt_idx
            )
        if len(cur_in_idx) > 0 and spec_md_s2.draft_token_ids is None:
            spec_md_s2.draft_token_ids = async_out_mod.extract_draft_token_ids(
                input_ids_s2,
                spec_md_s2.final_logits_indices,
                spec_md_s2.target_logits_indices,
            )
        assert spec_md_s2.draft_token_ids is not None

        # Deterministic Step 2 target logits to verify rejection sampling on Step 2.
        logits_s2 = torch.zeros((padded_tot_s2, vocab_size), dtype=torch.float32)
        for row_i in range(padded_tot_s2):
            tok_id = (
                int(input_ids_s2[row_i + 1].item())
                if row_i % query_len == 0
                else (35 + row_i) % vocab_size
            )
            logits_s2[row_i, tok_id] = 10.0

        next_tokens_s2 = TPUModelRunner._sample_spec_verify_chunk(
            runner,
            logits=logits_s2,
            md=spec_md_s2,
            grammar_output=None,
            scheduler_output=cast(Any, sched2),
            cur_start_idx=0,
            cur_end_idx=num_reqs,
            all_greedy=True,
            sampling_generator=None,
            top_tokens=False,
        )

        return {
            "next_tokens_s1": next_tokens_s1[:num_reqs].clone(),
            "drafts_s1": drafts_s1.clone(),
            "mamba_after_s1": mamba_after_s1,
            "seq_lens_s2": attn_md_s2["layer.0"].seq_lens.clone(),
            "positions_s2": runner.position_ids.clone(),
            "input_ids_s2": input_ids_s2.clone(),
            "draft_token_ids_s2": spec_md_s2.draft_token_ids[
                : num_reqs * spec_k
            ].clone(),
            "next_tokens_s2": next_tokens_s2[:num_reqs].clone(),
            "used_fused_subst": used_fused_subst,
            "counts": counts,
        }

    unfolded = _run_two_steps(fold_enabled=False)
    folded = _run_two_steps(fold_enabled=True)

    assert unfolded["used_fused_subst"] is False
    assert unfolded["counts"] == {"subtract": 1, "subst": 1, "extract": 1}
    assert folded["used_fused_subst"] is True
    assert folded["counts"] == {"subtract": 0, "subst": 0, "extract": 0}

    assert int(folded["mamba_after_s1"][11].item()) == 0
    assert int(folded["mamba_after_s1"][13].item()) == 0
    assert int(folded["mamba_after_s1"][3].item()) == 2
    assert int(folded["mamba_after_s1"][7].item()) == 0
    assert torch.equal(folded["mamba_after_s1"], unfolded["mamba_after_s1"])

    assert torch.equal(folded["next_tokens_s1"], unfolded["next_tokens_s1"])
    assert torch.equal(folded["drafts_s1"], unfolded["drafts_s1"])
    assert torch.equal(folded["seq_lens_s2"], unfolded["seq_lens_s2"])
    assert torch.equal(folded["positions_s2"], unfolded["positions_s2"])
    assert torch.equal(folded["input_ids_s2"], unfolded["input_ids_s2"])
    assert torch.equal(folded["draft_token_ids_s2"], unfolded["draft_token_ids_s2"])
    assert torch.equal(folded["next_tokens_s2"], unfolded["next_tokens_s2"])
