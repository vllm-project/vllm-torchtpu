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

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output

from vllm_torchtpu.layers.core.sequence_layout import (
    SequenceLayoutKind,
)
from vllm_torchtpu.runner.speculative_decoding_manager import SpeculativeDecodingManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

_PCP_LAYOUT_RANK = "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank"
_PCP_LAYOUT_WORLD_SIZE = (
    "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_world_size"
)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "computed,prompt,scheduled,pcp_size,mtp_enabled,rejected",
    [
        (8, 8, 1, 2, True, True),  # Decode without a draft.
        (8, 8, 2, 2, True, True),  # K=1 target verification.
        (7, 8, 2, 2, True, True),  # A chunk crossing the prompt boundary.
        (9, 8, 1, 2, True, True),  # A resumed request already past prefill.
        (0, 8, 4, 2, True, False),  # Intermediate prefill chunk.
        (7, 8, 1, 2, True, False),  # Final prompt token produces the first output.
        (0, 8, 8, 2, True, False),  # Unchunked prefill.
        (8, 8, 2, 2, False, False),  # PCP without MTP.
        (8, 8, 2, 1, True, False),  # MTP without PCP.
        (8, 8, 2, 1, False, False),  # Neither PCP nor MTP.
    ],
)
def test_execute_model_pcp_mtp_prefill_only_boundary(
    computed, prompt, scheduled, pcp_size, mtp_enabled, rejected, async_scheduling
):
    # Put a valid prefill first so validation must cover the whole batch.
    scheduled_tokens = [4, scheduled]
    runner = _make_runner(
        num_computed_tokens=[0, computed],
        prompt_tokens=[8, prompt],
        scheduled_tokens=scheduled_tokens,
        token_paddings=[256],
        mtp_enabled=mtp_enabled,
        pcp_size=pcp_size,
    )
    runner.execute_model_state = None
    runner.scheduler_config.async_scheduling = async_scheduling
    # The fixture already represents the batch after the scheduler update.
    runner._update_states = lambda _: None
    runner._flush_disjoint_async_results = lambda: None

    class ForwardBoundaryReached(Exception):
        pass

    def stop_before_forward(_):
        raise ForwardBoundaryReached

    runner._reorder_batch_for_rpa = stop_before_forward
    if rejected:
        with pytest.raises(RuntimeError, match="PCP.*MTP.*req1"):
            TPUModelRunner.execute_model(runner, _scheduler_output(scheduled_tokens))
    else:
        with pytest.raises(ForwardBoundaryReached):
            TPUModelRunner.execute_model(runner, _scheduler_output(scheduled_tokens))


@pytest.mark.parametrize("bucket", [1, 4])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
@pytest.mark.parametrize("hybrid", [False, True])
def test_prepare_inputs_routes_reordered_verify_per_chunk(bucket, chunk_size, hybrid):
    """Protect the real prepare-inputs boundary, not a hand-made distribution.

    Both RPA routes must keep prefill in MIXED. Hybrid GDN always includes
    verification, independently of whether the RPA bucket is widened.
    """
    scheduled = [16, 3, 1, 3, 1]
    runner = _make_runner(
        num_computed_tokens=[0, 8, 8, 8, 8],
        prompt_tokens=[16, 8, 8, 8, 8],
        scheduled_tokens=scheduled,
        token_paddings=[256],
        pcp_size=1,
    )
    runner.reorder_batch_threshold = bucket
    runner.num_reqs_max_model_len = chunk_size
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    runner.spec_decode_manager = SpeculativeDecodingManager(runner)
    if hybrid:
        runner.mamba_slot_read_offsets = torch.zeros(8, dtype=torch.int32)
        runner._combined_request_distribution_cpu = torch.zeros(6, dtype=torch.int32)
    sched = _scheduler_output(scheduled)
    sched.scheduled_spec_decode_tokens = {"req1": [1, 2], "req3": [3, 4]}
    batch = runner.input_batch

    def swap_states(i, j):
        batch.req_ids[i], batch.req_ids[j] = batch.req_ids[j], batch.req_ids[i]
        batch.req_id_to_index[batch.req_ids[i]] = i
        batch.req_id_to_index[batch.req_ids[j]] = j
        for array in (
            batch.num_computed_tokens_cpu,
            batch.num_prompt_tokens,
            batch.token_ids_cpu,
        ):
            array[[i, j]] = array[[j, i]]
        table = batch.block_table[0]._table
        table[[i, j]] = table[[j, i]]

    batch.swap_states = swap_states
    num_decode, num_windowed = TPUModelRunner._reorder_batch_for_rpa(runner, sched)
    assert (num_decode, num_windowed) == (2, 4)
    assert [sched.num_scheduled_tokens[r] for r in batch.req_ids] == [1, 1, 3, 3, 16]
    # Independent per-request expectations in the reordered batch.
    rpa_membership = [True, True, bucket > 1, bucket > 1, False]
    gdn_membership = [True, True, True, True, False]
    start = 0
    while start < 5:
        prepared = TPUModelRunner._prepare_inputs(
            runner, sched, start, num_decode, num_windowed
        )
        metadata = prepared[0]["layer.0"]
        end = prepared[4]
        expected_decode = sum(rpa_membership[start:end])
        assert metadata.request_distribution.tolist() == [
            expected_decode,
            expected_decode,
            end - start,
        ]
        expected_lengths = [1, 1, 3, 3, 16][start:end]
        assert (
            torch.diff(metadata.query_start_loc)[: end - start].tolist()
            == expected_lengths
        )
        if hybrid:
            expected_windowed = sum(gdn_membership[start:end])
            assert (
                runner._attn_metadata_builder_ctx.mamba_request_distribution.tolist()
                == [expected_windowed, expected_windowed, end - start]
            )
        start = end


def _run_dummy_run(monkeypatch, runner, *, num_tokens=256):
    def capture_forward_context(attn_metadata, *_args, **_kwargs):
        runner._captured_dummy_attn_metadata = attn_metadata
        return contextlib.nullcontext()

    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_forward_context",
        capture_forward_context,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors",
        lambda *_args, **_kwargs: None,
    )

    runner.maybe_select_dummy_loras = lambda *_args, **_kwargs: contextlib.nullcontext()
    runner.forward_model = MagicMock(
        side_effect=lambda input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors=None: (
            torch.zeros((positions.shape[-1], 4), dtype=torch.float32),
            None,
        )
    )

    TPUModelRunner._dummy_run(
        runner,
        num_tokens,
        runner.num_reqs_max_model_len,
        runner.input_batch.block_table[0].max_num_blocks_per_req,
        use_max_model_len=True,
    )
    return getattr(runner, "_attn_metadata_builder_ctx", None)


def test_prepare_inputs_builds_rank_local_partial_layout(monkeypatch):
    scheduled = [32]
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[256],
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 256
    assert runner.position_ids.shape[0] == 256
    torch.testing.assert_close(
        runner.input_ids[:16].cpu(), torch.arange(116, 132, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.input_ids[16:].cpu(), torch.zeros(240, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids[:16].cpu(), torch.arange(16, 32, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids[16:].cpu(), torch.zeros(240, dtype=torch.int32)
    )
    assert not hasattr(
        runner._attn_metadata_builder_ctx, "pcp_attention_metadata_by_gid"
    )
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(
        logits_indices.cpu(), torch.tensor([271] + [-1] * 7, dtype=torch.int32)
    )


def test_prepare_inputs_pcp_long_context_step_is_not_bounded(monkeypatch):
    # One 64K-token step of one request: far past the batched kernel's SMEM
    # schedule, which no layer here runs.
    scheduled = [65536]
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[65536],
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )

    assert runner._attention_schedule_capacity() is None
    md = attn_metadata["layer.0"]
    assert md.sequence_layout_protocol == "pcp_streaming"
    assert runner.seq_lens_np[0] == 65536


def test_prepare_inputs_pcp_mtp_snapshots_request_major_tokens_before_pack(monkeypatch):
    scheduled = [25]
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[256],
        mtp_enabled=True,
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    prepared = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )

    request_major_input_ids, sequence_layout_plan = prepared[-2:]
    assert sequence_layout_plan is not None
    assert request_major_input_ids is not None
    assert request_major_input_ids.shape == (512,)
    torch.testing.assert_close(
        request_major_input_ids[:25].cpu(), torch.arange(100, 125, dtype=torch.int32)
    )
    torch.testing.assert_close(
        request_major_input_ids[25:].cpu(), torch.zeros(487, dtype=torch.int32)
    )
    # The target input still owns the packed rank-local representation.
    torch.testing.assert_close(
        runner.input_ids[:9].cpu(), torch.arange(116, 125, dtype=torch.int32)
    )


def test_prepare_inputs_pcp_mtp_returns_each_chunks_own_plan(monkeypatch):
    scheduled = [17, 25]
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[32],
        mtp_enabled=True,
    )
    runner.num_reqs_max_model_len = 1

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    first_prepared = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )
    first_input_ids, first_plan = first_prepared[-2:]
    assert first_plan is not None
    first_packed_to_request = first_plan._packed_to_request_major_token_indices.copy()

    second_prepared = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 1, 0
    )
    second_input_ids, second_plan = second_prepared[-2:]

    assert second_plan is not None
    assert first_plan is not second_plan
    assert first_plan.global_padded_num_tokens == 64
    assert second_plan.global_padded_num_tokens == 64
    assert not np.array_equal(
        first_plan._packed_to_request_major_token_indices,
        second_plan._packed_to_request_major_token_indices,
    )
    np.testing.assert_array_equal(
        first_plan._packed_to_request_major_token_indices,
        first_packed_to_request,
    )
    torch.testing.assert_close(
        first_input_ids[:17].cpu(), torch.arange(100, 117, dtype=torch.int32)
    )
    torch.testing.assert_close(
        second_input_ids[:25].cpu(), torch.arange(200, 225, dtype=torch.int32)
    )


def test_prepare_inputs_builds_rank_local_multi_request_partial_layout(monkeypatch):
    scheduled = [512, 512]
    runner = _make_runner(
        num_computed_tokens=[0, 0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[512],
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 512
    assert runner.position_ids.shape[0] == 512
    torch.testing.assert_close(
        runner.input_ids[:16].cpu(), torch.arange(116, 132, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.input_ids[256:272].cpu(), torch.arange(216, 232, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids[:16].cpu(), torch.arange(16, 32, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids[256:272].cpu(), torch.arange(16, 32, dtype=torch.int32)
    )
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(
        logits_indices.cpu(),
        torch.tensor([767, 1023] + [-1] * 6, dtype=torch.int32),
    )


def test_prepare_inputs_builds_rank_local_unaligned_partial_layout(monkeypatch):
    scheduled = [25]
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[256],
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 256
    torch.testing.assert_close(
        runner.input_ids[:9].cpu(), torch.arange(116, 125, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.input_ids[9:].cpu(), torch.zeros(247, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids[:9].cpu(), torch.arange(16, 25, dtype=torch.int32)
    )
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(
        logits_indices.cpu(),
        torch.tensor([264] + [-1] * 7, dtype=torch.int32),
    )


def test_prepare_inputs_pcp_mrope_keeps_rank_major_real_positions(monkeypatch):
    scheduled = [25]
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=scheduled,
        scheduled_tokens=scheduled,
        token_paddings=[256],
        uses_mrope=True,
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )

    md = attn_metadata["layer.0"]
    assert runner.position_ids.shape == (3, 256)
    expected = torch.stack(
        (
            torch.arange(16, 25, dtype=torch.int32),
            torch.arange(1016, 1025, dtype=torch.int32),
            torch.arange(2016, 2025, dtype=torch.int32),
        )
    )
    torch.testing.assert_close(runner.position_ids[:, :9].cpu(), expected)
    torch.testing.assert_close(
        runner.position_ids[:, 9:].cpu(), torch.zeros((3, 247), dtype=torch.int32)
    )
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"


def test_dummy_run_partial_layout_decode_like_metadata_routes_to_streaming(monkeypatch):
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[512],
        scheduled_tokens=[32],
        token_paddings=[256],
    )

    ctx = _run_dummy_run(monkeypatch, runner)

    assert ctx.sequence_layout_descriptor.kind is SequenceLayoutKind.PARTIAL
    assert ctx.sequence_layout_descriptor.protocol == "pcp_streaming"
    torch.testing.assert_close(
        ctx.request_distribution.cpu(), torch.tensor([8, 8, 8], dtype=torch.int32)
    )
    torch.testing.assert_close(
        ctx.query_start_loc.cpu(), torch.arange(9, dtype=torch.int32)
    )
    torch.testing.assert_close(ctx.seq_lens.cpu(), torch.ones(8, dtype=torch.int32))
    assert runner.forward_model.call_args.kwargs["input_ids"].shape == (256,)
    torch.testing.assert_close(
        runner.forward_model.call_args.kwargs["positions"][:16].cpu(),
        torch.zeros(16, dtype=torch.int32),
    )


def test_dummy_run_marks_inactive_request_rows_after_cache_init(monkeypatch):
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[64],
        scheduled_tokens=[1],
        token_paddings=[256],
    )

    ctx = _run_dummy_run(monkeypatch, runner, num_tokens=3)
    md = runner._captured_dummy_attn_metadata["layer.0"]

    assert ctx is not None
    torch.testing.assert_close(
        md.query_start_loc.cpu(),
        torch.tensor([0, 1, 2, 3, 3, 3, 3, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        md.request_distribution.cpu(), torch.tensor([3, 3, 3], dtype=torch.int32)
    )


def test_dummy_run_marks_inactive_request_rows_before_cache_init(monkeypatch):
    runner = _make_runner(
        num_computed_tokens=[0],
        prompt_tokens=[64],
        scheduled_tokens=[1],
        token_paddings=[256],
    )
    runner.kv_cache_config = None
    runner._attn_layer_names = ["layer.0"]

    ctx = _run_dummy_run(monkeypatch, runner, num_tokens=3)
    md = runner._captured_dummy_attn_metadata["layer.0"]

    assert ctx is None
    torch.testing.assert_close(
        md.query_start_loc.cpu(),
        torch.tensor([0, 1, 2, 3, 3, 3, 3, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        md.request_distribution.cpu(), torch.tensor([3, 3, 3], dtype=torch.int32)
    )
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    assert md.sequence_layout_version == 1


def test_dummy_run_preinit_and_postinit_share_metadata_pytree(monkeypatch):
    import jax

    def make_runner():
        return _make_runner(
            num_computed_tokens=[0],
            prompt_tokens=[64],
            scheduled_tokens=[1],
            token_paddings=[256],
        )

    postinit_runner = make_runner()
    _run_dummy_run(monkeypatch, postinit_runner, num_tokens=3)
    postinit_md = postinit_runner._captured_dummy_attn_metadata["layer.0"]

    preinit_runner = make_runner()
    preinit_runner.kv_cache_config = None
    preinit_runner._attn_layer_names = ["layer.0"]
    _run_dummy_run(monkeypatch, preinit_runner, num_tokens=3)
    preinit_md = preinit_runner._captured_dummy_attn_metadata["layer.0"]

    assert jax.tree_util.tree_structure(preinit_md) == (
        jax.tree_util.tree_structure(postinit_md)
    )
    for field in (
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
    ):
        preinit_value = getattr(preinit_md, field)
        postinit_value = getattr(postinit_md, field)
        assert preinit_value.shape == postinit_value.shape
        assert preinit_value.dtype == postinit_value.dtype


def test_prepare_inputs_batch_flat_decode_can_leave_rank_with_only_padding(monkeypatch):
    scheduled = [1, 1]
    runner = _make_runner(
        num_computed_tokens=[16, 16],
        prompt_tokens=[16, 16],
        scheduled_tokens=scheduled,
        token_paddings=[2],
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 2
    )

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 2
    assert runner.position_ids.shape[0] == 2
    torch.testing.assert_close(
        runner.input_ids.cpu(), torch.zeros(2, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids.cpu(), torch.zeros(2, dtype=torch.int32)
    )
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(
        md.request_distribution.cpu(), torch.tensor([2, 2, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        logits_indices.cpu(), torch.tensor([0, 1] + [-1] * 6, dtype=torch.int32)
    )


def test_prepare_inputs_pcp_owner_does_not_rotate_with_absolute_q_start(monkeypatch):
    scheduled = [1]
    runner = _make_runner(
        num_computed_tokens=[16],
        prompt_tokens=[64],
        scheduled_tokens=scheduled,
        token_paddings=[8],
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    _, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 1
    )
    torch.testing.assert_close(
        logits_indices.cpu(), torch.tensor([0] + [-1] * 7, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids.cpu()[:1], torch.tensor([16], dtype=torch.int32)
    )

    runner.input_batch.num_computed_tokens_cpu[0] = 32
    _, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 1
    )

    torch.testing.assert_close(
        logits_indices.cpu(), torch.tensor([0] + [-1] * 7, dtype=torch.int32)
    )
    torch.testing.assert_close(
        runner.position_ids.cpu()[:1], torch.tensor([32], dtype=torch.int32)
    )


def test_prepare_inputs_batch_flat_owner_preserves_history_positions_and_logits(
    monkeypatch,
):
    scheduled = [25, 23]
    runner = _make_runner(
        num_computed_tokens=[100, 200],
        prompt_tokens=[125, 223],
        scheduled_tokens=scheduled,
        token_paddings=[32],
    )

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0
    )

    expected_ids = torch.cat(
        (
            torch.arange(216, 225, dtype=torch.int32),
            torch.arange(400, 407, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
        )
    )
    expected_positions = torch.cat(
        (
            torch.arange(116, 125, dtype=torch.int32),
            torch.arange(200, 207, dtype=torch.int32),
            torch.zeros(16, dtype=torch.int32),
        )
    )
    torch.testing.assert_close(runner.input_ids.cpu(), expected_ids)
    torch.testing.assert_close(runner.position_ids.cpu(), expected_positions)
    torch.testing.assert_close(
        logits_indices.cpu(),
        torch.tensor([40, 31] + [-1] * 6, dtype=torch.int32),
    )

    md = attn_metadata["layer.0"]
    torch.testing.assert_close(
        md.query_start_loc.cpu()[:3], torch.tensor([0, 25, 48], dtype=torch.int32)
    )
    torch.testing.assert_close(
        md.seq_lens.cpu()[:2], torch.tensor([125, 223], dtype=torch.int32)
    )
    assert not hasattr(runner._attn_metadata_builder_ctx, "token_owner_start_offsets")


def _make_attention_capacity_runner(monkeypatch, pcp_size):
    from jax.experimental.pallas import tpu as pltpu
    from vllm.config import CacheConfig

    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl

    # Run the real capacity arithmetic using JAX's static chip specification;
    # neither the runner nor this fixture initializes a TPU device.
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    impl = MagicMock(spec=PallasAttentionBackendImpl)
    impl.runs_batched_rpa_schedule.return_value = pcp_size == 1
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.get_layers_from_vllm_config",
        lambda _config, _layer_type: {"layer.0": SimpleNamespace(impl=impl)},
    )
    runner = object.__new__(TPUModelRunner)
    runner._attention_runs_batched_kernel = None
    runner._attention_capacity = {}
    runner.parallel_config = SimpleNamespace(prefill_context_parallel_size=pcp_size)
    runner.vllm_config = SimpleNamespace(parallel_config=runner.parallel_config)
    runner.cache_config = CacheConfig(block_size=256)
    runner.cache_config.kv_cache_layout = "LBNHC"
    runner._attention_kernel_block_size = 256
    runner.model_config = SimpleNamespace(
        dtype=torch.bfloat16, get_num_attention_heads=lambda _: 32
    )
    runner.num_kv_heads = 2
    runner.head_size = 256
    runner.max_num_reqs = 64
    runner.num_reqs_max_model_len = 64
    runner.max_model_len = 262144
    runner.most_model_len = None
    runner.num_reqs_most_model_len = None
    runner._attention_kv_cache_group_id = None
    runner.kv_cache_dtype = torch.float8_e4m3fn
    return runner


@pytest.mark.parametrize(
    "q_lens,kv_lens,num_decode",
    [
        ([32768], [32768], 0),
        ([32768], [65536], 0),
        ([1, 32768], [262144, 65536], 1),
        ([1] * 64, [262144] * 64, 64),
    ],
    ids=["prefill", "history", "mixed", "decode"],
)
def test_pcp_streaming_does_not_use_batched_smem_capacity(
    monkeypatch, q_lens, kv_lens, num_decode
):
    runner = _make_attention_capacity_runner(monkeypatch, pcp_size=8)
    runner.seq_lens_np = np.asarray(kv_lens, dtype=np.int32)
    runner._check_attention_schedule(len(q_lens), np.asarray(q_lens), num_decode)
    # The backend also serves ordinary RPA, but PCP dispatches the distinct
    # compact streaming schedule for every request distribution.
    assert runner._attention_schedule_capacity() is None
    assert runner._attention_runs_batched_kernel is False
