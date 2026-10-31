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
"""Unit tests for TPU_STATIC_DECODE_INPUTS and TPU_STATIC_DECODE_INPUTS_VERIFY."""

import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output
from vllm.v1.worker.block_table import BlockTable

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.static_decode_buffers import (
    StaticDecodeBufferTracker,
    build_fresh_verify_reference,
    verify_static_decode_PrepareInputs,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


@dataclass
class _MockAttnMetadata:
    seq_lens: Any
    query_start_loc: Any
    request_distribution: Any
    block_tables: Any
    input_positions: Any = None
    mamba_state_indices: Any = None
    mamba_ckpt_indices: Any = None
    mamba_slot_read_offsets: Any = None
    mamba_request_distribution: Any = None


@dataclass
class _MockSpecMetadata:
    draft_token_ids: Any
    draft_lengths: Any
    bonus_logits_indices: Any
    target_logits_indices: Any
    final_logits_indices: Any
    segment_ids: Any
    group_indices: Any
    deferred_draft_token_ids_cpu: Any = None


def _reference_prepare_host_buffers(
    req_ids: list[str],
    num_computed_tokens: np.ndarray,
    token_ids_table: np.ndarray,
    query_len: int,
    padded_num_tokens: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_reqs = len(req_ids)
    total_tokens = num_reqs * query_len
    eff_tokens = (
        padded_num_tokens
        if padded_num_tokens is not None and padded_num_tokens >= total_tokens
        else total_tokens
    )
    computed_slice = num_computed_tokens[:num_reqs].astype(np.int32)
    positions_ref = np.zeros(eff_tokens, dtype=np.int32)
    input_ids_ref = np.zeros(eff_tokens, dtype=np.int32)
    for req_idx in range(num_reqs):
        base_pos = int(computed_slice[req_idx])
        for offset in range(query_len):
            tok_slot = req_idx * query_len + offset
            pos_val = base_pos + offset
            positions_ref[tok_slot] = pos_val
            input_ids_ref[tok_slot] = int(token_ids_table[req_idx, pos_val])
    seq_lens_ref = computed_slice + query_len
    qsl_ref = np.arange(num_reqs + 1, dtype=np.int32) * query_len
    return input_ids_ref, positions_ref, seq_lens_ref, qsl_ref


def test_env_flags_default_false() -> None:
    """Static decode input env flags default to False."""
    assert envs.TPU_STATIC_DECODE_INPUTS is False
    assert envs.TPU_STATIC_DECODE_INPUTS_VERIFY is False


def test_static_decode_varying_accepted_tokens() -> None:
    """Verify incremental buffer updates across steps with 0..3 accepted drafts."""
    max_num_reqs = 8
    max_model_len = 128
    spec_k = 3
    query_len = 1 + spec_k
    tracker = StaticDecodeBufferTracker(
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_reqs * query_len,
        block_sizes=(16,),
    )
    rng = np.random.default_rng(42)
    token_ids_table = rng.integers(
        100, 5000, size=(max_num_reqs, max_model_len), dtype=np.int32
    )
    req_ids = ["req_0", "req_1", "req_2", "req_3"]
    num_computed = np.array([16, 32, 48, 64], dtype=np.int32)

    accepted_advances = (
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([1, 4, 2, 3], dtype=np.int32),
        np.array([4, 1, 3, 2], dtype=np.int32),
        np.array([2, 2, 1, 4], dtype=np.int32),
        np.array([3, 3, 4, 1], dtype=np.int32),
    )
    for step_idx, advance in enumerate(accepted_advances):
        num_computed = num_computed + advance
        in_ids, pos, seq_lens, qsl, same_batch, _boundary = tracker.update_step_buffers(
            req_ids=req_ids,
            num_computed_tokens=num_computed,
            token_ids_table=token_ids_table,
            query_len=query_len,
            padded_num_tokens=20,
        )
        ref_in, ref_pos, ref_sl, ref_qsl = _reference_prepare_host_buffers(
            req_ids=req_ids,
            num_computed_tokens=num_computed,
            token_ids_table=token_ids_table,
            query_len=query_len,
            padded_num_tokens=20,
        )
        assert same_batch is (step_idx > 0)
        assert np.array_equal(in_ids, ref_in)
        assert np.array_equal(pos, ref_pos)
        assert np.array_equal(seq_lens, ref_sl)
        assert np.array_equal(qsl, ref_qsl)


def test_static_decode_requests_joining_and_leaving() -> None:
    """Verify batch membership transitions (requests joining and leaving)."""
    max_num_reqs = 8
    max_model_len = 128
    query_len = 4
    tracker = StaticDecodeBufferTracker(
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_reqs * query_len,
        block_sizes=(16,),
    )
    rng = np.random.default_rng(7)
    token_ids_table = rng.integers(
        10, 9000, size=(max_num_reqs, max_model_len), dtype=np.int32
    )

    reqs_step1 = ["req_a", "req_b", "req_c"]
    comp_step1 = np.array([10, 20, 30], dtype=np.int32)
    in1, pos1, sl1, qsl1, same1, boundary1 = tracker.update_step_buffers(
        reqs_step1, comp_step1, token_ids_table, query_len
    )
    ref_in1, ref_pos1, ref_sl1, ref_qsl1 = _reference_prepare_host_buffers(
        reqs_step1, comp_step1, token_ids_table, query_len
    )
    assert not same1
    assert boundary1
    assert np.array_equal(in1, ref_in1)
    assert np.array_equal(pos1, ref_pos1)
    assert np.array_equal(sl1, ref_sl1)
    assert np.array_equal(qsl1, ref_qsl1)

    reqs_step2 = ["req_a", "req_b", "req_c", "req_d"]
    comp_step2 = np.array([13, 22, 34, 8], dtype=np.int32)
    in2, pos2, sl2, qsl2, same2, boundary2 = tracker.update_step_buffers(
        reqs_step2, comp_step2, token_ids_table, query_len
    )
    ref_in2, ref_pos2, ref_sl2, ref_qsl2 = _reference_prepare_host_buffers(
        reqs_step2, comp_step2, token_ids_table, query_len
    )
    assert not same2
    assert boundary2
    assert np.array_equal(in2, ref_in2)
    assert np.array_equal(pos2, ref_pos2)
    assert np.array_equal(sl2, ref_sl2)
    assert np.array_equal(qsl2, ref_qsl2)

    comp_step3 = np.array([15, 25, 35, 11], dtype=np.int32)
    in3, pos3, sl3, qsl3, same3, boundary3 = tracker.update_step_buffers(
        reqs_step2, comp_step3, token_ids_table, query_len
    )
    ref_in3, ref_pos3, ref_sl3, ref_qsl3 = _reference_prepare_host_buffers(
        reqs_step2, comp_step3, token_ids_table, query_len
    )
    assert same3
    assert not boundary3
    assert tracker.metadata_reused_last_step
    assert np.array_equal(in3, ref_in3)
    assert np.array_equal(pos3, ref_pos3)
    assert np.array_equal(sl3, ref_sl3)
    assert np.array_equal(qsl3, ref_qsl3)

    reqs_step4 = ["req_a", "req_c", "req_d"]
    comp_step4 = np.array([18, 38, 14], dtype=np.int32)
    in4, pos4, sl4, qsl4, same4, boundary4 = tracker.update_step_buffers(
        reqs_step4, comp_step4, token_ids_table, query_len
    )
    ref_in4, ref_pos4, ref_sl4, ref_qsl4 = _reference_prepare_host_buffers(
        reqs_step4, comp_step4, token_ids_table, query_len
    )
    assert not same4
    assert boundary4
    assert np.array_equal(in4, ref_in4)
    assert np.array_equal(pos4, ref_pos4)
    assert np.array_equal(sl4, ref_sl4)
    assert np.array_equal(qsl4, ref_qsl4)


def test_static_decode_block_boundary_crossing() -> None:
    """Verify block boundary detection triggers rebuild only when crossing
    block_size."""
    tracker = StaticDecodeBufferTracker(
        max_num_reqs=4,
        max_num_tokens=16,
        block_sizes=(16,),
    )
    token_ids_table = np.arange(4 * 64, dtype=np.int32).reshape(4, 64)
    req_ids = ["req_0", "req_1"]

    tracker.update_step_buffers(
        req_ids, np.array([9, 19], dtype=np.int32), token_ids_table, query_len=1
    )
    _, _, _, _, same_batch, crossed = tracker.update_step_buffers(
        req_ids,
        np.array([10, 20], dtype=np.int32),
        token_ids_table,
        query_len=1,
    )
    assert same_batch
    assert not crossed
    assert tracker.metadata_reused_last_step

    _, _, _, _, same_batch2, crossed2 = tracker.update_step_buffers(
        req_ids,
        np.array([16, 20], dtype=np.int32),
        token_ids_table,
        query_len=1,
    )
    assert same_batch2
    assert crossed2
    assert not tracker.metadata_reused_last_step


def test_static_decode_inputs_verify_assertion() -> None:
    """Verify `verify_static_decode_PrepareInputs` passes on identical tensors and
    raises AssertionError when any tensor field diverges."""
    attn_md_ref = {
        "layer.0": _MockAttnMetadata(
            seq_lens=torch.tensor([12, 24], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
            block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        )
    }
    attn_md_static = {
        "layer.0": _MockAttnMetadata(
            seq_lens=torch.tensor([12, 24], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
            block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        )
    }
    spec_md_ref = _MockSpecMetadata(
        draft_token_ids=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32),
        draft_lengths=torch.tensor([3, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3, 7], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int32),
        final_logits_indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
        group_indices=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
    )
    spec_md_static = _MockSpecMetadata(
        draft_token_ids=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32),
        draft_lengths=torch.tensor([3, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3, 7], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int32),
        final_logits_indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
        group_indices=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
    )
    logits_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    ref_tuple = (
        attn_md_ref,
        logits_idx,
        2,
        2,
        2,
        [],
        [],
        spec_md_ref,
        None,
        None,
    )
    static_tuple = (
        attn_md_static,
        logits_idx.clone(),
        2,
        2,
        2,
        [],
        [],
        spec_md_static,
        None,
        None,
    )
    in_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32)
    pos_ids = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int32)

    verify_static_decode_PrepareInputs(
        ref_tuple,
        static_tuple,
        in_ids,
        in_ids.clone(),
        pos_ids,
        pos_ids.clone(),
    )

    attn_md_static["layer.0"].seq_lens = torch.tensor([12, 25], dtype=torch.int32)
    with pytest.raises(AssertionError, match="seq_lens mismatch"):
        verify_static_decode_PrepareInputs(
            ref_tuple,
            static_tuple,
            in_ids,
            in_ids.clone(),
            pos_ids,
            pos_ids.clone(),
        )


def test_static_decode_page_size_256_and_force_block_rebuild() -> None:
    """Verify page-size-256 block boundary crossing and force_block_rebuild
    invalidation (prefix-cache / CoW / scheduler new_block_ids)."""
    tracker = StaticDecodeBufferTracker(
        max_num_reqs=4,
        max_num_tokens=16,
        block_sizes=(256,),
    )
    token_ids_table = np.arange(4 * 512, dtype=np.int32).reshape(4, 512)
    req_ids = ["req_0", "req_1"]

    tracker.update_step_buffers(
        req_ids,
        np.array([250, 100], dtype=np.int32),
        token_ids_table,
        query_len=4,
    )
    _, _, _, _, same_batch, crossed = tracker.update_step_buffers(
        req_ids,
        np.array([252, 104], dtype=np.int32),
        token_ids_table,
        query_len=4,
    )
    assert same_batch
    assert not crossed
    assert tracker.metadata_reused_last_step

    # Crossing 256 (seq_len = 253 + 4 = 257 > 256) triggers block rebuild.
    _, _, _, _, same_batch2, crossed2 = tracker.update_step_buffers(
        req_ids,
        np.array([253, 108], dtype=np.int32),
        token_ids_table,
        query_len=4,
    )
    assert same_batch2
    assert crossed2
    assert not tracker.metadata_reused_last_step

    # Explicit force_block_rebuild (e.g. CoW / new_block_ids) invalidates even
    # within the same 256-token page.
    _, _, _, _, same_batch3, crossed3 = tracker.update_step_buffers(
        req_ids,
        np.array([256, 111], dtype=np.int32),
        token_ids_table,
        query_len=4,
        force_block_rebuild=True,
    )
    assert same_batch3
    assert crossed3
    assert not tracker.metadata_reused_last_step


def test_static_decode_block_straddle_with_async_rejected_window() -> None:
    """Verify `max_rejected > 0` forces `block_boundary_crossed=True` whenever
    `[seq_len - max_rejected - 1, seq_len - 1]` straddles a KV/Mamba block
    boundary (e.g. `seq_len = 258` with `max_rejected = 3` and `block_size = 256`).
    """
    tracker = StaticDecodeBufferTracker(
        max_num_reqs=4,
        max_num_tokens=16,
        block_sizes=(256,),
    )
    token_ids_table = np.arange(4 * 512, dtype=np.int32).reshape(4, 512)
    req_ids = ["req_0"]

    # Step 1: num_computed = 248 -> seq_len = 252 (all in block 0).
    tracker.update_step_buffers(
        req_ids,
        np.array([248], dtype=np.int32),
        token_ids_table,
        query_len=4,
        max_rejected=3,
    )
    # Step 2: num_computed = 250 -> seq_len = 254 (low=250//256=0, high=253//256=0).
    _, _, _, _, same2, crossed2 = tracker.update_step_buffers(
        req_ids,
        np.array([250], dtype=np.int32),
        token_ids_table,
        query_len=4,
        max_rejected=3,
    )
    assert same2
    assert not crossed2
    assert tracker.metadata_reused_last_step

    # Step 3: num_computed = 254 -> seq_len = 258 (high=257//256=1, low=254//256=0).
    # Crosses 256 AND straddles [254, 257].
    _, _, _, _, same3, crossed3 = tracker.update_step_buffers(
        req_ids,
        np.array([254], dtype=np.int32),
        token_ids_table,
        query_len=4,
        max_rejected=3,
    )
    assert same3
    assert crossed3
    assert not tracker.metadata_reused_last_step

    # Step 4: num_computed = 255 -> seq_len = 259 (high=258//256=1, low=255//256=0).
    # Even though num_blocks is still 2, [255, 258] still straddles 256!
    _, _, _, _, same4, crossed4 = tracker.update_step_buffers(
        req_ids,
        np.array([255], dtype=np.int32),
        token_ids_table,
        query_len=4,
        max_rejected=3,
    )
    assert same4
    assert crossed4
    assert not tracker.metadata_reused_last_step

    # Step 5: num_computed = 256 -> seq_len = 260 (high=259//256=1, low=256//256=1).
    # First step settled completely inside block 1: rebuilds once to cache block 1.
    _, _, _, _, same5, crossed5 = tracker.update_step_buffers(
        req_ids,
        np.array([256], dtype=np.int32),
        token_ids_table,
        query_len=4,
        max_rejected=3,
    )
    assert same5
    assert crossed5

    # Step 6: num_computed = 260 -> seq_len = 264 (settled in block 1): reuses!
    _, _, _, _, same6, crossed6 = tracker.update_step_buffers(
        req_ids,
        np.array([260], dtype=np.int32),
        token_ids_table,
        query_len=4,
        max_rejected=3,
    )
    assert same6
    assert not crossed6
    assert tracker.metadata_reused_last_step


def test_verify_corrupt_field_catches_every_device_field() -> None:
    """Verify `corrupt_field=<field>` causes `verify_static_decode_PrepareInputs`
    to raise `AssertionError` for every device-consumed field."""
    in_ids = torch.tensor([11, 12, 13, 14, 21, 22, 23, 24], dtype=torch.int32)
    pos_ids = torch.tensor([8, 9, 10, 11, 20, 21, 22, 23], dtype=torch.int32)
    logits_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    umi_list = [torch.tensor([5, 9], dtype=torch.int32)]
    slot_map = {0: torch.tensor([100, 101, 102, 103], dtype=torch.int32)}

    ref_layer_md = _MockAttnMetadata(
        input_positions=pos_ids,
        seq_lens=torch.tensor([12, 24], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        block_tables=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        mamba_state_indices=torch.tensor([5, 9], dtype=torch.int32),
        mamba_ckpt_indices=torch.tensor(
            [[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32
        ),
        mamba_slot_read_offsets=torch.tensor([0, 2, 1, 0], dtype=torch.int32),
        mamba_request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    static_layer_md = _MockAttnMetadata(
        input_positions=pos_ids.clone(),
        seq_lens=torch.tensor([12, 24], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        block_tables=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        mamba_state_indices=torch.tensor([5, 9], dtype=torch.int32),
        mamba_ckpt_indices=torch.tensor(
            [[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32
        ),
        mamba_slot_read_offsets=torch.tensor([0, 2, 1, 0], dtype=torch.int32),
        mamba_request_distribution=torch.tensor([2, 2, 2], dtype=torch.int32),
    )
    spec_md_ref = _MockSpecMetadata(
        draft_token_ids=torch.tensor([12, 13, 14, 22, 23, 24], dtype=torch.int32),
        deferred_draft_token_ids_cpu=np.array([12, 13, 14, 22, 23, 24], dtype=np.int32),
        draft_lengths=torch.tensor([3, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3, 7], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int32),
        final_logits_indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
        group_indices=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
    )
    spec_md_static = _MockSpecMetadata(
        draft_token_ids=torch.tensor([12, 13, 14, 22, 23, 24], dtype=torch.int32),
        deferred_draft_token_ids_cpu=np.array([12, 13, 14, 22, 23, 24], dtype=np.int32),
        draft_lengths=torch.tensor([3, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3, 7], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int32),
        final_logits_indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32),
        group_indices=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32),
    )
    ref_tuple = (
        {"layer.0": ref_layer_md},
        logits_idx,
        2,
        2,
        2,
        [],
        [],
        spec_md_ref,
        None,
        None,
    )
    static_tuple = (
        {"layer.0": static_layer_md},
        logits_idx.clone(),
        2,
        2,
        2,
        [],
        [],
        spec_md_static,
        None,
        None,
    )

    # Uncorrupted passes cleanly.
    verify_static_decode_PrepareInputs(
        ref_tuple,
        static_tuple,
        in_ids,
        in_ids.clone(),
        pos_ids,
        pos_ids.clone(),
        ref_unified_mamba_state_indices=umi_list,
        static_unified_mamba_state_indices=[umi_list[0].clone()],
        ref_slot_mappings=slot_map,
        static_slot_mappings={0: slot_map[0].clone()},
        corrupt_field="",
    )

    fields_to_check = (
        ("input_ids", "input_ids mismatch"),
        ("positions", "position_ids mismatch"),
        ("input_positions", "input_positions mismatch"),
        ("seq_lens", "seq_lens mismatch"),
        ("query_start_loc", "query_start_loc mismatch"),
        ("request_distribution", "request_distribution mismatch"),
        ("slot_mapping", "slot_mapping"),
        ("block_tables", "block_tables mismatch"),
        ("logits_indices", "logits_indices mismatch"),
        ("spec_decode_metadata", "draft_token_ids mismatch"),
        ("draft_token_ids", "draft_token_ids mismatch"),
        ("deferred_draft_token_ids_cpu", "deferred_draft_token_ids_cpu mismatch"),
        ("draft_lengths", "draft_lengths mismatch"),
        ("bonus_logits_indices", "bonus_logits_indices mismatch"),
        ("target_logits_indices", "target_logits_indices mismatch"),
        ("final_logits_indices", "final_logits_indices mismatch"),
        ("mamba_slot_read_offsets", "mamba_slot_read_offsets mismatch"),
        ("mamba_state_indices", "mamba_state_indices mismatch"),
        ("unified_mamba_state_indices", "unified_mamba_state_indices"),
        ("mamba_ckpt_indices", "mamba_ckpt_indices mismatch"),
        ("mamba_request_distribution", "mamba_request_distribution mismatch"),
        ("nonexistent_field_xyz", "unknown corrupt target field"),
    )
    for field_name, expected_msg in fields_to_check:
        with pytest.raises(AssertionError, match=expected_msg):
            verify_static_decode_PrepareInputs(
                ref_tuple,
                static_tuple,
                in_ids,
                in_ids.clone(),
                pos_ids,
                pos_ids.clone(),
                ref_unified_mamba_state_indices=umi_list,
                static_unified_mamba_state_indices=[umi_list[0].clone()],
                ref_slot_mappings=slot_map,
                static_slot_mappings={0: slot_map[0].clone()},
                corrupt_field=field_name,
            )


class _MockBlockTableObj:
    def __init__(self, arr: np.ndarray) -> None:
        self._tensor = torch.from_numpy(arr)
        self.num_blocks_per_row = np.count_nonzero(arr, axis=1).astype(np.int32)

    def get_cpu_tensor(self) -> Any:
        return self._tensor


class _MockGroupBuilder:
    def __init__(
        self,
        gid: int,
        is_mamba_group: bool,
        target_block_size: int,
        block_table_obj: _MockBlockTableObj,
        target_num_blocks: int,
    ) -> None:
        self.kv_cache_group_id = gid
        self.is_mamba_group = is_mamba_group
        self.target_block_size = target_block_size
        self._bt_obj = block_table_obj
        self._tnb = target_num_blocks

    def _block_table_geometry(self, _ctx: Any) -> tuple[_MockBlockTableObj, int]:
        return self._bt_obj, self._tnb


class _MockSpecDecodeManager:
    def get_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
        defer_draft_token_ids: bool = False,
    ) -> _MockSpecMetadata:
        num_reqs = len(num_draft_tokens)
        spec_k = int(num_draft_tokens[0])
        total_tok = num_reqs * (1 + spec_k)
        target_idx = []
        bonus_idx = []
        for req_i in range(num_reqs):
            base = req_i * (1 + spec_k)
            for step_j in range(spec_k):
                target_idx.append(base + step_j)
            bonus_idx.append(base + spec_k)
        return _MockSpecMetadata(
            draft_token_ids=(
                None
                if defer_draft_token_ids
                else torch.zeros(num_reqs * spec_k, dtype=torch.int32)
            ),
            draft_lengths=torch.from_numpy(num_draft_tokens.copy()),
            bonus_logits_indices=torch.tensor(bonus_idx, dtype=torch.int32),
            target_logits_indices=torch.tensor(target_idx, dtype=torch.int32),
            final_logits_indices=torch.arange(total_tok, dtype=torch.int32),
            segment_ids=torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int32), spec_k
            ),
            group_indices=torch.arange(num_reqs * spec_k, dtype=torch.int32),
        )


def test_build_fresh_verify_reference_end_to_end_post_substitution() -> None:
    """Verify `build_fresh_verify_reference` computes post-rejection `seq_lens`,
    `positions`, post-substitution `input_ids`, post-extraction `draft_token_ids`,
    `block_tables`, and unified-pool `mamba_state_indices` / `mamba_ckpt_indices`
    from scratch and catches any 1-element corruption."""
    num_reqs = 2
    spec_k = 3
    query_len = 1 + spec_k
    total_tokens = num_reqs * query_len

    token_ids_cpu = np.zeros((4, 512), dtype=np.int32)
    token_ids_cpu[0, :64] = np.arange(1000, 1064, dtype=np.int32)
    token_ids_cpu[1, :64] = np.arange(2000, 2064, dtype=np.int32)
    num_computed_cpu = np.array([20, 30, 0, 0], dtype=np.int32)

    bt_attn = _MockBlockTableObj(
        np.array([[10, 11, 0, 0], [20, 21, 0, 0]], dtype=np.int32)
    )
    bt_mamba = _MockBlockTableObj(
        np.array([[31, 32, 33, 34], [41, 42, 43, 44]], dtype=np.int32)
    )
    attn_group = types.SimpleNamespace(
        layer_names=["layers.0.attn"],
        metadata_builders=[
            _MockGroupBuilder(0, False, 256, bt_attn, target_num_blocks=4)
        ],
    )
    mamba_group = types.SimpleNamespace(
        layer_names=["layers.1.mamba"],
        metadata_builders=[
            _MockGroupBuilder(1, True, 256, bt_mamba, target_num_blocks=4)
        ],
    )

    # Previous step async results: req_0 rejected 1 draft, req_1 rejected 2 drafts.
    # Substituted tokens: req_0 gets [901..904], req_1 gets [801..804].
    next_tokens_tpu = torch.tensor(
        [901, 902, 903, 904, 801, 802, 803, 804], dtype=torch.int32
    )
    spec_num_rejected = torch.tensor([1, 2, 0, 0], dtype=torch.int32)
    pre_async = types.SimpleNamespace(
        next_tokens_tpu=next_tokens_tpu,
        spec_decode_num_rejected_tokens=spec_num_rejected,
        req_id_to_index_copy={"req_0": 0, "req_1": 1},
        num_draft_per_req={0: spec_k, 1: spec_k},
    )

    mock_runner = types.SimpleNamespace(
        reorder_batch_threshold=1,
        most_model_len=None,
        num_reqs_max_model_len=2,
        num_reqs_most_model_len=2,
        max_num_reqs=4,
        num_reqs_paddings=[2, 4],
        _num_static_verify_reqs=0,
        input_batch=types.SimpleNamespace(
            num_reqs=2,
            req_ids=["req_0", "req_1"],
            num_computed_tokens_cpu=num_computed_cpu,
            token_ids_cpu=token_ids_cpu,
            block_table={0: bt_attn, 1: bt_mamba},
        ),
        uses_mrope=False,
        requests={},
        scheduler_config=types.SimpleNamespace(async_scheduling=True),
        speculative_config=types.SimpleNamespace(
            method="mtp", num_speculative_tokens=spec_k
        ),
        _pre_async_results=pre_async,
        spec_decode_manager=_MockSpecDecodeManager(),
        mamba_slot_read_offsets=torch.tensor([0, 2, 1, 0], dtype=torch.int32),
        _unified_kv_layout=True,
        _has_mamba_state=False,
        _mamba_ckpt_window=4,
        attn_groups=[[attn_group, mamba_group]],
        device="cpu",
    )
    sched_out = types.SimpleNamespace(
        num_scheduled_tokens={"req_0": 4, "req_1": 4},
        scheduled_spec_decode_tokens={
            "req_0": [-1, -1, -1],
            "req_1": [-1, -1, -1],
        },
        scheduled_new_reqs=[],
    )
    layout_plan = types.SimpleNamespace(
        global_padded_num_tokens=total_tokens,
        token_slice=slice(0, total_tokens),
        logits_indices_cpu=None,
    )

    ref_pack = build_fresh_verify_reference(
        mock_runner,
        sched_out,
        start_index=0,
        num_decode_reqs=0,
        num_windowed_reqs=2,
        layout_plan=layout_plan,
        padded_num_reqs_override=2,
    )

    # Check post-rejection seq_lens: [20 + 4 - 1, 30 + 4 - 2] = [23, 32]
    ref_attn_md = ref_pack["ref_result"][0]
    expected_seq_lens = torch.tensor([23, 32], dtype=torch.int32)
    assert torch.equal(ref_attn_md["layers.0.attn"].seq_lens, expected_seq_lens)
    assert torch.equal(ref_attn_md["layers.1.mamba"].seq_lens, expected_seq_lens)

    # Check post-rejection positions:
    # req_0: [20, 21, 22, 23] - 1 = [19, 20, 21, 22]
    # req_1: [30, 31, 32, 33] - 2 = [28, 29, 30, 31]
    expected_pos = torch.tensor([19, 20, 21, 22, 28, 29, 30, 31], dtype=torch.int32)
    assert torch.equal(ref_pack["ref_position_ids"], expected_pos)

    # Check post-substitution input_ids and extracted draft_token_ids:
    assert torch.equal(ref_pack["ref_input_ids_post"], next_tokens_tpu)
    expected_drafts = torch.tensor([902, 903, 904, 802, 803, 804], dtype=torch.int32)
    assert torch.equal(ref_pack["ref_result"][7].draft_token_ids, expected_drafts)

    # Check unified-pool mamba_state_indices and mamba_ckpt_indices:
    expected_mamba_state = torch.tensor([31, 41], dtype=torch.int32)
    assert torch.equal(
        ref_attn_md["layers.1.mamba"].mamba_state_indices, expected_mamba_state
    )
    assert len(ref_pack["ref_unified_mamba_state_indices"]) == 1
    assert torch.equal(
        ref_pack["ref_unified_mamba_state_indices"][0], expected_mamba_state
    )

    # Verify passes when static output matches ref_pack, and fails when
    # cached_attn_metadata forgets to populate unified_mamba_state_indices ([]).
    verify_static_decode_PrepareInputs(
        ref_pack["ref_result"],
        ref_pack["ref_result"],
        ref_pack["ref_input_ids_post"],
        ref_pack["ref_input_ids_post"].clone(),
        ref_pack["ref_position_ids"],
        ref_pack["ref_position_ids"].clone(),
        ref_unified_mamba_state_indices=ref_pack["ref_unified_mamba_state_indices"],
        static_unified_mamba_state_indices=[expected_mamba_state.clone()],
    )
    with pytest.raises(
        AssertionError, match="unified_mamba_state_indices length mismatch"
    ):
        verify_static_decode_PrepareInputs(
            ref_pack["ref_result"],
            ref_pack["ref_result"],
            ref_pack["ref_input_ids_post"],
            ref_pack["ref_input_ids_post"].clone(),
            ref_pack["ref_position_ids"],
            ref_pack["ref_position_ids"].clone(),
            ref_unified_mamba_state_indices=ref_pack["ref_unified_mamba_state_indices"],
            static_unified_mamba_state_indices=[],
        )


def test_build_fresh_verify_reference_step_context() -> None:
    """Verify `build_fresh_verify_reference` populates `step_context` and
    includes `step_context` in `AssertionError` messages when a mismatch occurs."""
    num_reqs = 2
    spec_k = 3
    query_len = 1 + spec_k
    total_tokens = num_reqs * query_len

    token_ids_cpu = np.zeros((4, 512), dtype=np.int32)
    num_computed_cpu = np.array([255, 16, 0, 0], dtype=np.int32)
    bt_mamba = _MockBlockTableObj(
        np.array([[31, 32, 33, 34], [41, 42, 43, 44]], dtype=np.int32)
    )
    mamba_group = types.SimpleNamespace(
        layer_names=["language_model.model.layers.0.linear_attn"],
        metadata_builders=[
            _MockGroupBuilder(0, True, 256, bt_mamba, target_num_blocks=4)
        ],
    )
    base_ns = types.SimpleNamespace(
        reorder_batch_threshold=1,
        most_model_len=None,
        num_reqs_max_model_len=2,
        num_reqs_most_model_len=2,
        max_num_reqs=4,
        num_reqs_paddings=[2, 4],
        _num_static_verify_reqs=0,
        input_batch=types.SimpleNamespace(
            num_reqs=2,
            req_ids=["req_0", "req_1"],
            num_computed_tokens_cpu=num_computed_cpu,
            token_ids_cpu=token_ids_cpu,
            block_table={0: bt_mamba},
        ),
        uses_mrope=False,
        requests={
            "req_0": types.SimpleNamespace(
                num_prompt_tokens=255, mrope_position_delta=None
            ),
            "req_1": types.SimpleNamespace(
                num_prompt_tokens=10, mrope_position_delta=None
            ),
        },
        scheduler_config=types.SimpleNamespace(async_scheduling=False),
        speculative_config=types.SimpleNamespace(
            method="mtp", num_speculative_tokens=spec_k
        ),
        _pre_async_results=None,
        spec_decode_manager=_MockSpecDecodeManager(),
        mamba_slot_read_offsets=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        _unified_kv_layout=True,
        _has_mamba_state=False,
        _mamba_ckpt_window=4,
        attn_groups=[[mamba_group]],
        device="cpu",
    )
    sched_out = types.SimpleNamespace(
        num_scheduled_tokens={"req_0": 4, "req_1": 4},
        scheduled_spec_decode_tokens={
            "req_0": [101, 102, 103],
            "req_1": [201, 202, 203],
        },
        scheduled_new_reqs=[],
    )
    layout_plan = types.SimpleNamespace(
        global_padded_num_tokens=total_tokens,
        token_slice=slice(0, total_tokens),
        logits_indices_cpu=None,
    )

    ref_pack = build_fresh_verify_reference(
        base_ns,
        sched_out,
        start_index=0,
        num_decode_reqs=0,
        num_windowed_reqs=2,
        layout_plan=layout_plan,
        padded_num_reqs_override=2,
    )
    step_ctx = ref_pack["step_context"]
    assert step_ctx["num_reqs"] == 2
    assert step_ctx["first_decode_after_prefill"] == [True, False]
    assert step_ctx["is_uniform_spec_step"] is True

    with pytest.raises(AssertionError) as exc_info:
        verify_static_decode_PrepareInputs(
            ref_pack["ref_result"],
            ref_pack["ref_result"],
            ref_pack["ref_input_ids_post"],
            ref_pack["ref_input_ids_post"].clone(),
            ref_pack["ref_position_ids"],
            ref_pack["ref_position_ids"].clone(),
            ref_unified_mamba_state_indices=ref_pack["ref_unified_mamba_state_indices"],
            static_unified_mamba_state_indices=[
                tensor_item.clone()
                for tensor_item in ref_pack["ref_unified_mamba_state_indices"]
            ],
            corrupt_field="seq_lens",
            step_context=step_ctx,
        )
    err_str = str(exc_info.value)
    assert "seq_lens mismatch" in err_str
    assert "step_context=" in err_str
    assert "first_decode_after_prefill" in err_str


class _MockPropertySpec:
    """Simulates `UniformTypeKVCacheSpecs` where `block_size` is a `@property`
    on the class rather than an instance `__dict__` attribute."""

    def __init__(self, real_bs: int) -> None:
        self._real_bs = real_bs

    @property
    def block_size(self) -> int:
        return self._real_bs


def test_get_static_decode_tracker_extracts_property_block_size() -> None:
    """Verify `TPUModelRunner._get_static_decode_tracker` extracts `@property`
    `block_size=640` from wrapper specs instead of falling back to 16."""
    prop_spec = _MockPropertySpec(640)
    assert "block_size" not in vars(prop_spec)
    mock_runner = types.SimpleNamespace(
        _static_decode_tracker=None,
        max_num_reqs=8,
        positions_np=np.zeros(32, dtype=np.int32),
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[types.SimpleNamespace(kv_cache_spec=prop_spec)]
        ),
        attn_groups=(),
        cache_config=types.SimpleNamespace(block_size=16),
    )
    built_tracker = TPUModelRunner._get_static_decode_tracker(mock_runner)
    extracted_bs = built_tracker.block_sizes[0]
    assert extracted_bs == 640

    # With block_sizes=(640,), stepping from 20 to 100 in steps of 4 never
    # crosses a 640-token block boundary, so reuse_md happens on every step
    # after the initial step!
    tracker = StaticDecodeBufferTracker(
        max_num_reqs=8,
        max_num_tokens=32,
        block_sizes=(extracted_bs,),
    )
    token_ids_table = np.zeros((8, 1024), dtype=np.int32)
    req_ids = [f"req_{idx}" for idx in range(8)]
    reused_steps = 0
    rebuilt_steps = 0
    for step_idx in range(20):
        num_comp = np.full(8, 20 + step_idx * 4, dtype=np.int32)
        _, _, _, _, same_reqs, crossed = tracker.update_step_buffers(
            req_ids,
            num_comp,
            token_ids_table,
            query_len=4,
            max_rejected=3,
        )
        if same_reqs and not crossed:
            reused_steps += 1
        else:
            rebuilt_steps += 1
    assert rebuilt_steps == 1
    assert reused_steps == 19


def test_warmup_spec_decode_cleanup_resets_static_decode_tracker() -> None:
    tracker = StaticDecodeBufferTracker(
        max_num_reqs=4,
        max_num_tokens=16,
        block_sizes=(16,),
    )
    token_ids_table = np.zeros((4, 128), dtype=np.int32)
    tracker.update_step_buffers(
        ["req_0", "req_1"],
        np.array([10, 20], dtype=np.int32),
        token_ids_table,
        query_len=4,
        max_rejected=3,
    )
    assert tracker.prev_req_ids == ("req_0", "req_1")

    runner = types.SimpleNamespace(
        requests={},
        input_batch=types.SimpleNamespace(
            req_id_to_index={},
            num_reqs=0,
        ),
        encoder_cache={},
        _mamba_slot_by_req_id={},
        spec_decode_manager=None,
        drafter=None,
        _pre_async_results="dirty",
        mm_embed_inputs="dirty",
        _static_decode_tracker=tracker,
        _pending_static_verify=("dirty",),
        _last_step_used_static_decode=True,
    )
    TPUModelRunner._warmup_spec_decode_cleanup(runner, [])
    assert tracker.prev_req_ids is None
    assert runner._pending_static_verify is None
    assert runner._last_step_used_static_decode is False


def _make_real_block_table_runner(
    *,
    num_computed_tokens: list[int],
    scheduled_tokens: list[int],
    initial_blocks_per_req: list[list[int]],
    max_num_reqs: int = 8,
    block_size: int = 16,
    spec_k: int = 0,
    async_scheduling: bool = False,
    uses_mrope: bool = False,
) -> tuple[Any, BlockTable, list[int]]:
    """Construct a runner wired with a real `vllm.v1.worker.block_table.BlockTable`,
    real `AttentionMetadataBuilder`, and real `SpeculativeDecodingManager`."""
    runner = _make_runner(
        num_computed_tokens=num_computed_tokens,
        prompt_tokens=[64] * len(scheduled_tokens),
        scheduled_tokens=scheduled_tokens,
        token_paddings=[8, 16, 32],
        pcp_size=1,
        max_num_reqs=max_num_reqs,
        block_size=block_size,
        uses_mrope=uses_mrope,
    )
    if uses_mrope:
        runner._calc_mrope_positions = types.MethodType(
            TPUModelRunner._calc_mrope_positions, runner
        )
    max_num_blocks_per_req = 8
    real_bt = BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=runner.max_num_tokens,
        pin_memory=False,
        device=torch.device("cpu"),
        cp_kv_cache_interleave_size=1,
    )
    for row_idx, block_ids in enumerate(initial_blocks_per_req):
        real_bt.add_row(block_ids, row_idx)
    runner.input_batch.block_table = [real_bt]

    spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["layer.0"],
        vllm_config=runner.vllm_config,
        device=runner.device,
        runner=runner,
        kv_cache_group_id=0,
    )
    attn_group = types.SimpleNamespace(
        layer_names=["layer.0"],
        metadata_builders=[builder],
    )
    runner.attn_groups = [[attn_group]]

    build_counter = [0]
    real_build = TPUModelRunner._build_attention_metadata

    def counted_build_attention_metadata(self_runner: Any, **kwargs: Any) -> Any:
        build_counter[0] += 1
        return real_build(self_runner, **kwargs)

    runner._build_attention_metadata = types.MethodType(
        counted_build_attention_metadata, runner
    )
    runner.scheduler_config.async_scheduling = async_scheduling
    if spec_k > 0:
        runner.speculative_config = types.SimpleNamespace(
            method="mtp",
            num_speculative_tokens=spec_k,
        )
        runner.vllm_config.speculative_config = runner.speculative_config
        runner.spec_decode_manager = SpeculativeDecodingManager(runner)
    return runner, real_bt, build_counter


def test_runner_prepare_inputs_static_vs_default_decode_and_spec_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise `TPUModelRunner._prepare_inputs` with `TPU_STATIC_DECODE_INPUTS=1`
    and `TPU_STATIC_DECODE_INPUTS_VERIFY=1` vs `TPU_STATIC_DECODE_INPUTS=0` on
    real `BlockTable` instances across multi-step decode and spec-verify steps."""
    # Part 1: Multi-step pure decode (q_len == 1) on real BlockTable.
    runner_static, _bt_static, build_cnt_static = _make_real_block_table_runner(
        num_computed_tokens=[4, 20],
        scheduled_tokens=[1, 1],
        initial_blocks_per_req=[[10], [20, 21]],
    )
    runner_default, _bt_default, build_cnt_default = _make_real_block_table_runner(
        num_computed_tokens=[4, 20],
        scheduled_tokens=[1, 1],
        initial_blocks_per_req=[[10], [20, 21]],
    )

    for step_idx in range(3):
        for runner_inst in (runner_static, runner_default):
            runner_inst.input_batch.num_computed_tokens_cpu[:] = np.array(
                [4 + step_idx, 20 + step_idx], dtype=np.int32
            )
        sched_out = _scheduler_output([1, 1])

        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", True)
        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", True)
        out_static = TPUModelRunner._prepare_inputs(
            runner_static, sched_out, start_index=0, num_decode_reqs=2
        )

        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", False)
        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", False)
        out_default = TPUModelRunner._prepare_inputs(
            runner_default, sched_out, start_index=0, num_decode_reqs=2
        )

        assert runner_static._last_step_used_static_decode is True
        assert runner_static._pending_static_verify is None
        verify_static_decode_PrepareInputs(
            out_default,
            out_static,
            runner_default.input_ids,
            runner_static.input_ids,
            runner_default.position_ids,
            runner_static.position_ids,
        )
        if step_idx == 0:
            assert build_cnt_static[0] == 1
        else:
            # Real BlockTable reuse fired: _build_attention_metadata was skipped!
            assert build_cnt_static[0] == 1
            assert (
                runner_static._static_decode_tracker.metadata_reused_last_step is True
            )
    assert build_cnt_default[0] == 3

    # Part 2: Multi-step uniform spec-verify (q_len == 4, async_scheduling=True)
    # verifying both attention metadata reuse and fresh deferred draft tokens.
    spec_k = 3
    runner_spec_static, _bt_spec_s, spec_build_cnt_s = _make_real_block_table_runner(
        num_computed_tokens=[4, 20],
        scheduled_tokens=[4, 4],
        initial_blocks_per_req=[[10], [20, 21]],
        spec_k=spec_k,
        async_scheduling=True,
    )
    runner_spec_default, _bt_spec_d, _spec_build_cnt_d = _make_real_block_table_runner(
        num_computed_tokens=[4, 20],
        scheduled_tokens=[4, 4],
        initial_blocks_per_req=[[10], [20, 21]],
        spec_k=spec_k,
        async_scheduling=True,
    )

    drafts_per_step = [
        [[101, 102, 103], [201, 202, 203]],
        [[301, 302, 303], [401, 402, 403]],
        [[501, 502, 503], [601, 602, 603]],
    ]
    computed_per_step = [
        np.array([4, 20], dtype=np.int32),
        np.array([7, 22], dtype=np.int32),
        np.array([10, 25], dtype=np.int32),
    ]
    for step_idx, (step_drafts, step_comp) in enumerate(
        zip(drafts_per_step, computed_per_step)
    ):
        for runner_inst in (runner_spec_static, runner_spec_default):
            runner_inst.input_batch.num_computed_tokens_cpu[:] = step_comp
            for row_idx in range(2):
                base_c = int(step_comp[row_idx])
                runner_inst.input_batch.token_ids_cpu[
                    row_idx, base_c + 1 : base_c + 1 + spec_k
                ] = np.array(step_drafts[row_idx], dtype=np.int32)
            runner_inst.spec_decode_manager._draft_token_ids = step_drafts

        sched_out = _scheduler_output([4, 4])
        sched_out.scheduled_spec_decode_tokens = {
            "req0": step_drafts[0],
            "req1": step_drafts[1],
        }

        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", True)
        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", True)
        out_s = TPUModelRunner._prepare_inputs(
            runner_spec_static, sched_out, start_index=0, num_decode_reqs=0
        )

        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", False)
        monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", False)
        out_d = TPUModelRunner._prepare_inputs(
            runner_spec_default, sched_out, start_index=0, num_decode_reqs=0
        )

        # Ensure deferred_draft_token_ids_cpu is refreshed every step (not stale).
        spec_md_s = out_s[7]
        spec_md_d = out_d[7]
        expected_deferred = np.array(step_drafts[0] + step_drafts[1], dtype=np.int32)
        assert np.array_equal(
            spec_md_s.deferred_draft_token_ids_cpu[:6], expected_deferred
        )
        assert np.array_equal(
            spec_md_s.deferred_draft_token_ids_cpu,
            spec_md_d.deferred_draft_token_ids_cpu,
        )
        assert spec_build_cnt_s[0] == 1
        if step_idx > 0:
            assert (
                runner_spec_static._static_decode_tracker.metadata_reused_last_step
                is True
            )


def test_runner_prepare_inputs_block_boundary_crossing_and_append_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify KV block boundary crossing and `BlockTable.append_row` invalidate
    cached attention metadata and rebuild block tables accurately."""
    runner_static, bt_static, build_cnt_static = _make_real_block_table_runner(
        num_computed_tokens=[14, 20],
        scheduled_tokens=[1, 1],
        initial_blocks_per_req=[[10], [20, 21]],
    )
    runner_default, bt_default, _build_cnt_default = _make_real_block_table_runner(
        num_computed_tokens=[14, 20],
        scheduled_tokens=[1, 1],
        initial_blocks_per_req=[[10], [20, 21]],
    )
    sched_out = _scheduler_output([1, 1])

    # Step 1: seq_lens = [15, 21] -> initial build.
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", True)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", True)
    TPUModelRunner._prepare_inputs(
        runner_static, sched_out, start_index=0, num_decode_reqs=2
    )
    assert build_cnt_static[0] == 1

    # Step 2: seq_lens = [16, 22] -> still 1 block for req0, 2 blocks for req1 (reuses)
    for runner_inst in (runner_static, runner_default):
        runner_inst.input_batch.num_computed_tokens_cpu[:] = np.array(
            [15, 21], dtype=np.int32
        )
    TPUModelRunner._prepare_inputs(
        runner_static, sched_out, start_index=0, num_decode_reqs=2
    )
    assert build_cnt_static[0] == 1
    assert runner_static._static_decode_tracker.metadata_reused_last_step is True

    # Step 3: req0 crosses block_size=16 (seq_len=17) and gets block 12 via append_row.
    bt_static.append_row([12], 0)
    bt_default.append_row([12], 0)
    for runner_inst in (runner_static, runner_default):
        runner_inst.input_batch.num_computed_tokens_cpu[:] = np.array(
            [16, 22], dtype=np.int32
        )

    out_static = TPUModelRunner._prepare_inputs(
        runner_static, sched_out, start_index=0, num_decode_reqs=2
    )
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", False)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", False)
    out_default = TPUModelRunner._prepare_inputs(
        runner_default, sched_out, start_index=0, num_decode_reqs=2
    )

    assert build_cnt_static[0] == 2
    assert runner_static._static_decode_tracker.metadata_reused_last_step is False
    bt_2d = out_static[0]["layer.0"].block_tables.reshape(8, 8)
    assert int(bt_2d[0, 1]) == 12
    verify_static_decode_PrepareInputs(
        out_default,
        out_static,
        runner_default.input_ids,
        runner_static.input_ids,
        runner_default.position_ids,
        runner_static.position_ids,
    )


def test_runner_prepare_inputs_request_departure_c4_to_c2_with_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify request departure (`c=4 -> c=2` inside `target_num_reqs=8`) zeroes
    padded tail slots consistently without false mismatch under `VERIFY=1`."""
    runner_static, bt_static, build_cnt_static = _make_real_block_table_runner(
        num_computed_tokens=[10, 20, 30, 40],
        scheduled_tokens=[1, 1, 1, 1],
        initial_blocks_per_req=[[1], [2, 3], [4, 5], [6, 7, 8]],
    )
    runner_default, bt_default, _build_cnt_default = _make_real_block_table_runner(
        num_computed_tokens=[10, 20, 30, 40],
        scheduled_tokens=[1, 1, 1, 1],
        initial_blocks_per_req=[[1], [2, 3], [4, 5], [6, 7, 8]],
    )

    sched_c4 = _scheduler_output([1, 1, 1, 1])
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", True)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", True)
    TPUModelRunner._prepare_inputs(
        runner_static, sched_c4, start_index=0, num_decode_reqs=4
    )
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", False)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", False)
    TPUModelRunner._prepare_inputs(
        runner_default, sched_c4, start_index=0, num_decode_reqs=4
    )
    assert build_cnt_static[0] == 1

    # Now req1 and req3 depart; batch condenses to 2 requests: [req0, req2].
    for runner_inst, bt_inst in (
        (runner_static, bt_static),
        (runner_default, bt_default),
    ):
        bt_inst.move_row(2, 1)
        bt_inst.clear_row(2)
        bt_inst.clear_row(3)
        runner_inst.input_batch.num_reqs = 2
        runner_inst.input_batch.req_ids = ["req0", "req2"]
        runner_inst.input_batch.req_id_to_index = {"req0": 0, "req2": 1}
        runner_inst.input_batch.num_computed_tokens_cpu = np.array(
            [11, 31], dtype=np.int32
        )
        runner_inst.input_batch.token_ids_cpu[1] = (
            runner_inst.input_batch.token_ids_cpu[2]
        )

    sched_c2 = types.SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"req0": 1, "req2": 1},
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
        finished_req_ids={"req1", "req3"},
        kv_cache_block_copies=None,
        scheduled_cached_reqs=types.SimpleNamespace(
            resumed_req_ids=set(),
            new_block_ids=[],
        ),
    )

    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", True)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", True)
    out_static_c2 = TPUModelRunner._prepare_inputs(
        runner_static, sched_c2, start_index=0, num_decode_reqs=2
    )

    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", False)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", False)
    out_default_c2 = TPUModelRunner._prepare_inputs(
        runner_default, sched_c2, start_index=0, num_decode_reqs=2
    )

    # Padded tail of seq_lens [2:8] and position_ids [2:8] must be zeroed.
    assert out_static_c2[0]["layer.0"].seq_lens.tolist() == [12, 32, 0, 0, 0, 0, 0, 0]
    assert runner_static.position_ids.tolist() == [11, 31, 0, 0, 0, 0, 0, 0]
    verify_static_decode_PrepareInputs(
        out_default_c2,
        out_static_c2,
        runner_default.input_ids,
        runner_static.input_ids,
        runner_default.position_ids,
        runner_static.position_ids,
    )


def test_runner_prepare_inputs_verify_waits_staging_fence_before_poisoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_prepare_inputs` with `TPU_STATIC_DECODE_INPUTS_VERIFY=1` waits on
    `_input_staging_fence` BEFORE writing `-1` poison values into host staging
    buffers so in-flight `non_blocking=True` uploads are never corrupted."""
    runner, _bt, _build_cnt = _make_real_block_table_runner(
        num_computed_tokens=[5, 10],
        scheduled_tokens=[1, 1],
        initial_blocks_per_req=[[1], [2]],
    )
    runner.positions_np[:2] = np.array([4, 9], dtype=np.int32)
    runner.input_ids_cpu[:2] = torch.tensor([104, 209], dtype=torch.int32)
    runner.seq_lens_np[:2] = np.array([5, 10], dtype=np.int32)
    runner.query_start_loc_np[:3] = np.array([0, 1, 2], dtype=np.int32)

    snapshots_at_fence_sync: dict[str, list[int]] = {}

    def sync_and_snapshot() -> None:
        snapshots_at_fence_sync["positions"] = runner.positions_np[:2].tolist()
        snapshots_at_fence_sync["input_ids"] = runner.input_ids_cpu[:2].tolist()
        snapshots_at_fence_sync["seq_lens"] = runner.seq_lens_np[:2].tolist()
        snapshots_at_fence_sync["query_start_loc"] = runner.query_start_loc_np[
            :3
        ].tolist()

    runner._input_staging_fence = types.SimpleNamespace(synchronize=sync_and_snapshot)

    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", True)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", True)
    TPUModelRunner._prepare_inputs(
        runner, _scheduler_output([1, 1]), start_index=0, num_decode_reqs=2
    )

    assert snapshots_at_fence_sync == {
        "positions": [4, 9],
        "input_ids": [104, 209],
        "seq_lens": [5, 10],
        "query_start_loc": [0, 1, 2],
    }


def test_runner_prepare_inputs_empty_new_block_tuples_and_mrope_departure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify (1) per-request empty `new_block_ids` tuples like `[([], [])]` do
    not falsely trigger `force_block_rebuild` and disable attention-metadata
    reuse, and (2) `uses_mrope=True` zeroes padded tail positions on `c=4 -> c=2`
    departure under `TPU_STATIC_DECODE_INPUTS_VERIFY=1`."""
    # Part 1: CachedRequestData.new_block_ids with empty per-KV-group tuples.
    runner, _bt, build_cnt = _make_real_block_table_runner(
        num_computed_tokens=[4, 20],
        scheduled_tokens=[1, 1],
        initial_blocks_per_req=[[10], [20, 21]],
    )
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS", True)
    monkeypatch.setattr(envs, "TPU_STATIC_DECODE_INPUTS_VERIFY", True)

    for step_idx in range(2):
        runner.input_batch.num_computed_tokens_cpu[:] = np.array(
            [4 + step_idx, 20 + step_idx], dtype=np.int32
        )
        sched_out = _scheduler_output([1, 1])
        sched_out.scheduled_cached_reqs = types.SimpleNamespace(
            resumed_req_ids=set(),
            new_block_ids=[([], []), ([], [])],
        )
        TPUModelRunner._prepare_inputs(
            runner, sched_out, start_index=0, num_decode_reqs=2
        )

    assert build_cnt[0] == 1
    assert runner._static_decode_tracker.metadata_reused_last_step is True

    # Part 2: uses_mrope=True request departure (c=4 -> c=2) zeroes [3, 2:8].
    runner_mrope, bt_mrope, _ = _make_real_block_table_runner(
        num_computed_tokens=[10, 20, 30, 40],
        scheduled_tokens=[1, 1, 1, 1],
        initial_blocks_per_req=[[1], [2], [3], [4]],
        max_num_reqs=8,
        uses_mrope=True,
    )
    TPUModelRunner._prepare_inputs(
        runner_mrope, _scheduler_output([1, 1, 1, 1]), start_index=0, num_decode_reqs=4
    )
    bt_mrope.move_row(2, 1)
    bt_mrope.clear_row(2)
    bt_mrope.clear_row(3)
    runner_mrope.input_batch.num_reqs = 2
    runner_mrope.input_batch.req_ids = ["req0", "req2"]
    runner_mrope.input_batch.req_id_to_index = {"req0": 0, "req2": 1}
    runner_mrope.input_batch.num_computed_tokens_cpu = np.array(
        [11, 31], dtype=np.int32
    )
    runner_mrope.input_batch.token_ids_cpu[1] = runner_mrope.input_batch.token_ids_cpu[
        2
    ]
    sched_c2 = types.SimpleNamespace(
        total_num_scheduled_tokens=2,
        num_scheduled_tokens={"req0": 1, "req2": 1},
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
        finished_req_ids={"req1", "req3"},
        kv_cache_block_copies=None,
        scheduled_cached_reqs=types.SimpleNamespace(
            resumed_req_ids=set(),
            new_block_ids=[([],), ([],)],
        ),
    )
    TPUModelRunner._prepare_inputs(
        runner_mrope, sched_c2, start_index=0, num_decode_reqs=2
    )
    assert runner_mrope.position_ids.shape == (3, 8)
    assert runner_mrope.position_ids.tolist() == [[11, 31, 0, 0, 0, 0, 0, 0]] * 3
