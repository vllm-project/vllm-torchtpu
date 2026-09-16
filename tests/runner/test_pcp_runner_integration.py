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
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import (
    SequenceLayoutKind, create_sequence_layout_planner)
from vllm_torchtpu.runner.speculative_decoding_manager import \
    SpeculativeDecodingManager
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

_PCP_LAYOUT_RANK = (
    "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank")
_PCP_LAYOUT_WORLD_SIZE = ("vllm_torchtpu.layers.core.pcp_sequence_layout."
                          "_get_native_pcp_world_size")


class _BlockTable:

    def __init__(self, table: torch.Tensor):
        self._table = table
        self.max_num_blocks_per_req = table.shape[1]

    def get_cpu_tensor(self):
        return self._table


def _make_runner(*,
                 num_computed_tokens,
                 prompt_tokens,
                 scheduled_tokens,
                 token_paddings,
                 uses_mrope=False,
                 pcp_mtp_k1_enabled=False,
                 pcp_size=2):
    num_reqs = len(scheduled_tokens)
    max_model_len = max(64, max(prompt_tokens) + 16)
    max_num_reqs = 8
    block_size = 16
    target_num_reqs = 8
    max_num_tokens = max(token_paddings)

    runner = SimpleNamespace()
    runner.device = torch.device("cpu")
    runner.most_model_len = None
    runner.num_reqs_max_model_len = target_num_reqs
    runner.num_reqs_most_model_len = None
    runner.num_tokens_paddings = list(token_paddings)
    runner.max_num_reqs = max_num_reqs
    runner.max_num_tokens = max_num_tokens
    runner.max_model_len = max_model_len
    runner.block_size = block_size
    runner.uses_mrope = uses_mrope
    runner.supports_mm_inputs = False
    runner.lora_config = None
    runner.speculative_config = None
    runner.reorder_batch_threshold = 1
    runner._pcp_mtp_k1_enabled = pcp_mtp_k1_enabled
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=pcp_size,
        cp_kv_cache_interleave_size=block_size,
        decode_context_parallel_size=1,
        pipeline_parallel_size=1,
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config,
        scheduler_config=runner.scheduler_config,
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}))
    runner.sequence_layout_planner = create_sequence_layout_planner(
        runner.vllm_config)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=8,
                                          num_gpu_blocks_override=None)
    runner.mesh = SimpleNamespace(shape={
        "attn_dp": 1,
        "expert": 1,
        "model": 1,
    })

    runner.input_ids_cpu = torch.zeros(max_num_tokens, dtype=torch.int32)
    runner.positions_cpu = torch.zeros(max_num_tokens, dtype=torch.int32)
    runner.positions_np = runner.positions_cpu.numpy()
    runner.query_start_loc_cpu = torch.zeros(max_num_tokens + 1,
                                             dtype=torch.int32)
    runner.query_start_loc_np = runner.query_start_loc_cpu.numpy()
    runner.seq_lens_cpu = torch.zeros(max_num_tokens, dtype=torch.int32)
    runner.seq_lens_np = runner.seq_lens_cpu.numpy()
    runner.arange_np = np.arange(max_num_tokens, dtype=np.int64)
    runner._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
    runner._decode_device_cache_key = None
    # The staging-tensor fence is recorded by execute_model; these tests
    # call _prepare_inputs directly, so the wait must find an empty fence.
    runner._input_staging_fence = None
    runner._wait_input_staging_fence = types.MethodType(
        TPUModelRunner._wait_input_staging_fence, runner)
    runner._cached_query_start_loc = None
    runner._cached_logits_indices = None
    runner._cached_request_distribution = None
    runner._dp_target_bucket = None
    runner._has_mamba_state = False
    # One pipeline stage: the runner embeds its own inputs.
    runner._pp_is_first = True
    runner._pp_is_last = True
    # The real schedule check; the stub registers no attention layer, so no
    # kernel bounds the step.
    runner._attention_capacity = {}
    runner._attention_runs_batched_kernel = None
    runner._attention_schedule_capacity = types.MethodType(
        TPUModelRunner._attention_schedule_capacity, runner)
    runner._check_attention_schedule = types.MethodType(
        TPUModelRunner._check_attention_schedule, runner)
    # Spec-decode mamba rollback is inactive in these PCP layout tests; the
    # runner leaves the read-offset buffer unset (None) unless speculative
    # decoding is enabled, so _prepare_inputs / dummy_run skip the windowed
    # GDN distribution path and just pass None through to the metadata.
    runner.mamba_slot_read_offsets = None
    # Affine checkpoint addressing (no per-checkpoint blocks) in these tests.
    runner._mamba_ckpt_window = 1
    runner._unified_kv_layout = False
    # Default to layer-major KV cache (None bundle) for PCP integration layout tests.
    runner._kv_cache_bundle = None
    # Bind the real seed-copy collector; an empty copy plan makes it a no-op
    # (no mamba align mode in these layout tests).
    runner._mamba_copy_plan = []
    runner._collect_mamba_state_seed_copies = (
        TPUModelRunner._collect_mamba_state_seed_copies.__get__(runner))
    runner._prepare_async_token_substitution_indices = (
        lambda *_args, **_kwargs:
        (np.array([], dtype=np.int32), np.array([], dtype=np.int32)))
    runner.empty_slot_mappings = {0: torch.empty(0)}
    if uses_mrope:

        def make_buffer(rows, cols, *, dtype):
            return SimpleNamespace(cpu=torch.zeros((rows, cols), dtype=dtype))

        runner._make_buffer = make_buffer
        runner.mrope_positions = make_buffer(3,
                                             max_num_tokens + 1,
                                             dtype=torch.int32)

        def fill_mrope_positions(_scheduler_output):
            positions = torch.arange(max_num_tokens + 1, dtype=torch.int32)
            runner.mrope_positions.cpu[0] = positions
            runner.mrope_positions.cpu[1] = positions + 1000
            runner.mrope_positions.cpu[2] = positions + 2000

        runner._calc_mrope_positions = fill_mrope_positions

    token_ids_cpu = torch.zeros((num_reqs, max_model_len), dtype=torch.int32)
    for req_idx in range(num_reqs):
        token_ids_cpu[req_idx] = torch.arange(
            100 * (req_idx + 1),
            100 * (req_idx + 1) + max_model_len,
            dtype=torch.int32,
        )

    pages_per_req = max(4, (max(prompt_tokens) + block_size - 1) // block_size)
    block_tables = torch.zeros((num_reqs, pages_per_req), dtype=torch.int32)
    next_block = 0
    for req_idx, seq_len in enumerate(prompt_tokens):
        num_blocks = (
            max(seq_len, num_computed_tokens[req_idx] +
                scheduled_tokens[req_idx]) + block_size - 1) // block_size
        block_tables[req_idx, :num_blocks] = torch.arange(next_block,
                                                          next_block +
                                                          num_blocks,
                                                          dtype=torch.int32)
        next_block += num_blocks

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=[f"req{i}" for i in range(num_reqs)],
        req_id_to_index={f"req{i}": i
                         for i in range(num_reqs)},
        num_computed_tokens_cpu=np.asarray(num_computed_tokens,
                                           dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt_tokens, dtype=np.int32),
        token_ids_cpu=token_ids_cpu.numpy(),
        token_ids_cpu_tensor=token_ids_cpu,
        block_table=[_BlockTable(block_tables)],
    )
    runner.input_batch = input_batch

    spec = FullAttentionSpec(block_size=block_size,
                             num_kv_heads=1,
                             head_size=128,
                             dtype=torch.bfloat16,
                             page_size_padded=block_size * 128 * 2)
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)
        ],
        has_mamba_layers=True,
    )

    def fake_build_attention_metadata(**_kwargs):
        ctx = runner._attn_metadata_builder_ctx
        md = AttentionMetadata(
            input_positions=(ctx.position_ids_override
                             if ctx.position_ids_override is not None else
                             runner.position_ids),
            block_tables=torch.zeros((target_num_reqs * 4, ),
                                     dtype=torch.int32),
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
            mamba_state_indices=ctx.mamba_state_indices,
            sequence_layout_kind=ctx.sequence_layout_descriptor.kind.value,
            sequence_layout_protocol=ctx.sequence_layout_descriptor.protocol,
            sequence_layout_version=ctx.sequence_layout_descriptor.version,
        )
        return {"layer.0": md}, None

    runner._build_attention_metadata = fake_build_attention_metadata
    return runner


def _scheduler_output(scheduled_tokens):
    return SimpleNamespace(
        total_num_scheduled_tokens=sum(scheduled_tokens),
        num_scheduled_tokens={
            f"req{i}": int(n)
            for i, n in enumerate(scheduled_tokens)
        },
        scheduled_spec_decode_tokens={},
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("bucket", [1, 4])
@pytest.mark.parametrize("chunk_size", [1, 2, 8])
@pytest.mark.parametrize("hybrid", [False, True])
def test_prepare_inputs_routes_reordered_verify_per_chunk(
        bucket, chunk_size, hybrid):
    """Protect the real prepare-inputs boundary, not a hand-made distribution.

    Both RPA routes must keep prefill in MIXED. Hybrid GDN always includes
    verification, independently of whether the RPA bucket is widened.
    """
    scheduled = [16, 3, 1, 3, 1]
    runner = _make_runner(num_computed_tokens=[0, 8, 8, 8, 8],
                          prompt_tokens=[16, 8, 8, 8, 8],
                          scheduled_tokens=scheduled,
                          token_paddings=[256],
                          pcp_size=1)
    runner.reorder_batch_threshold = bucket
    runner.num_reqs_max_model_len = chunk_size
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    runner.spec_decode_manager = SpeculativeDecodingManager(runner)
    if hybrid:
        runner.mamba_slot_read_offsets = torch.zeros(8, dtype=torch.int32)
        runner._combined_request_distribution_cpu = torch.zeros(
            6, dtype=torch.int32)
    sched = _scheduler_output(scheduled)
    sched.scheduled_spec_decode_tokens = {"req1": [1, 2], "req3": [3, 4]}
    batch = runner.input_batch

    def swap_states(i, j):
        batch.req_ids[i], batch.req_ids[j] = batch.req_ids[j], batch.req_ids[i]
        batch.req_id_to_index[batch.req_ids[i]] = i
        batch.req_id_to_index[batch.req_ids[j]] = j
        for array in (batch.num_computed_tokens_cpu, batch.num_prompt_tokens,
                      batch.token_ids_cpu):
            array[[i, j]] = array[[j, i]]
        table = batch.block_table[0]._table
        table[[i, j]] = table[[j, i]]

    batch.swap_states = swap_states
    num_decode, num_windowed = TPUModelRunner._reorder_batch_for_rpa(
        runner, sched)
    assert (num_decode, num_windowed) == (2, 4)
    assert [sched.num_scheduled_tokens[r]
            for r in batch.req_ids] == [1, 1, 3, 3, 16]
    # Independent per-request expectations in the reordered batch.
    rpa_membership = [True, True, bucket > 1, bucket > 1, False]
    gdn_membership = [True, True, True, True, False]
    start = 0
    while start < 5:
        prepared = TPUModelRunner._prepare_inputs(runner, sched, start,
                                                  num_decode, num_windowed)
        metadata = prepared[0]["layer.0"]
        end = prepared[4]
        expected_decode = sum(rpa_membership[start:end])
        assert metadata.request_distribution.tolist() == [
            expected_decode, expected_decode, end - start
        ]
        expected_lengths = [1, 1, 3, 3, 16][start:end]
        assert torch.diff(
            metadata.query_start_loc)[:end -
                                      start].tolist() == expected_lengths
        if hybrid:
            expected_windowed = sum(gdn_membership[start:end])
            assert runner._attn_metadata_builder_ctx.mamba_request_distribution.tolist(
            ) == [expected_windowed, expected_windowed, end - start]
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

    runner.maybe_select_dummy_loras = (
        lambda *_args, **_kwargs: contextlib.nullcontext())
    runner.forward_model = MagicMock(
        side_effect=lambda input_ids, positions, inputs_embeds,
        intermediate_tensors=None: (torch.zeros(
            (positions.shape[-1], 4), dtype=torch.float32), None))

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
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=scheduled,
                          scheduled_tokens=scheduled,
                          token_paddings=[256])

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0)

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 256
    assert runner.position_ids.shape[0] == 256
    torch.testing.assert_close(runner.input_ids[:16].cpu(),
                               torch.arange(116, 132, dtype=torch.int32))
    torch.testing.assert_close(runner.input_ids[16:].cpu(),
                               torch.zeros(240, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids[:16].cpu(),
                               torch.arange(16, 32, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids[16:].cpu(),
                               torch.zeros(240, dtype=torch.int32))
    assert not hasattr(runner._attn_metadata_builder_ctx,
                       "pcp_attention_metadata_by_gid")
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(
        logits_indices.cpu(), torch.tensor([271] + [-1] * 7,
                                           dtype=torch.int32))


def test_prepare_inputs_pcp_long_context_step_is_not_bounded(monkeypatch):
    # One 64K-token step of one request: far past the batched kernel's SMEM
    # schedule, which no layer here runs.
    scheduled = [65536]
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=scheduled,
                          scheduled_tokens=scheduled,
                          token_paddings=[65536])

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0)

    assert runner._attention_schedule_capacity() is None
    md = attn_metadata["layer.0"]
    assert md.sequence_layout_protocol == "pcp_streaming"
    assert runner.seq_lens_np[0] == 65536


def test_prepare_inputs_pcp_mtp_snapshots_request_major_tokens_before_pack(
        monkeypatch):
    scheduled = [25]
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=scheduled,
                          scheduled_tokens=scheduled,
                          token_paddings=[256],
                          pcp_mtp_k1_enabled=True)

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    prepared = TPUModelRunner._prepare_inputs(runner,
                                              _scheduler_output(scheduled), 0,
                                              0)

    request_major_input_ids, sequence_layout_plan = prepared[-2:]
    assert sequence_layout_plan is not None
    assert request_major_input_ids is not None
    assert request_major_input_ids.shape == (512, )
    torch.testing.assert_close(request_major_input_ids[:25].cpu(),
                               torch.arange(100, 125, dtype=torch.int32))
    torch.testing.assert_close(request_major_input_ids[25:].cpu(),
                               torch.zeros(487, dtype=torch.int32))
    # The target input still owns the packed rank-local representation.
    torch.testing.assert_close(runner.input_ids[:9].cpu(),
                               torch.arange(116, 125, dtype=torch.int32))


def test_prepare_inputs_pcp_mtp_returns_each_chunks_own_plan(monkeypatch):
    scheduled = [17, 25]
    runner = _make_runner(num_computed_tokens=[0, 0],
                          prompt_tokens=scheduled,
                          scheduled_tokens=scheduled,
                          token_paddings=[32],
                          pcp_mtp_k1_enabled=True)
    runner.num_reqs_max_model_len = 1

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    first_prepared = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0)
    first_input_ids, first_plan = first_prepared[-2:]
    assert first_plan is not None
    first_packed_to_request = (
        first_plan._packed_to_request_major_token_indices.copy())

    second_prepared = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 1, 0)
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
    torch.testing.assert_close(first_input_ids[:17].cpu(),
                               torch.arange(100, 117, dtype=torch.int32))
    torch.testing.assert_close(second_input_ids[:25].cpu(),
                               torch.arange(200, 225, dtype=torch.int32))


def test_prepare_inputs_builds_rank_local_multi_request_partial_layout(
        monkeypatch):
    scheduled = [512, 512]
    runner = _make_runner(num_computed_tokens=[0, 0],
                          prompt_tokens=scheduled,
                          scheduled_tokens=scheduled,
                          token_paddings=[512])

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0)

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 512
    assert runner.position_ids.shape[0] == 512
    torch.testing.assert_close(runner.input_ids[:16].cpu(),
                               torch.arange(116, 132, dtype=torch.int32))
    torch.testing.assert_close(runner.input_ids[256:272].cpu(),
                               torch.arange(216, 232, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids[:16].cpu(),
                               torch.arange(16, 32, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids[256:272].cpu(),
                               torch.arange(16, 32, dtype=torch.int32))
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(
        logits_indices.cpu(),
        torch.tensor([767, 1023] + [-1] * 6, dtype=torch.int32),
    )


def test_prepare_inputs_builds_rank_local_unaligned_partial_layout(
        monkeypatch):
    scheduled = [25]
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=scheduled,
                          scheduled_tokens=scheduled,
                          token_paddings=[256])

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0)

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 256
    torch.testing.assert_close(runner.input_ids[:9].cpu(),
                               torch.arange(116, 125, dtype=torch.int32))
    torch.testing.assert_close(runner.input_ids[9:].cpu(),
                               torch.zeros(247, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids[:9].cpu(),
                               torch.arange(16, 25, dtype=torch.int32))
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(
        logits_indices.cpu(),
        torch.tensor([264] + [-1] * 7, dtype=torch.int32),
    )


def test_prepare_inputs_pcp_mrope_keeps_rank_major_real_positions(monkeypatch):
    scheduled = [25]
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=scheduled,
                          scheduled_tokens=scheduled,
                          token_paddings=[256],
                          uses_mrope=True)

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 0)

    md = attn_metadata["layer.0"]
    assert runner.position_ids.shape == (3, 256)
    expected = torch.stack((
        torch.arange(16, 25, dtype=torch.int32),
        torch.arange(1016, 1025, dtype=torch.int32),
        torch.arange(2016, 2025, dtype=torch.int32),
    ))
    torch.testing.assert_close(runner.position_ids[:, :9].cpu(), expected)
    torch.testing.assert_close(runner.position_ids[:, 9:].cpu(),
                               torch.zeros((3, 247), dtype=torch.int32))
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"


def test_dummy_run_partial_layout_decode_like_metadata_routes_to_streaming(
        monkeypatch):
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=[512],
                          scheduled_tokens=[32],
                          token_paddings=[256])

    ctx = _run_dummy_run(monkeypatch, runner)

    assert ctx.sequence_layout_descriptor.kind is SequenceLayoutKind.PARTIAL
    assert ctx.sequence_layout_descriptor.protocol == "pcp_streaming"
    torch.testing.assert_close(ctx.request_distribution.cpu(),
                               torch.tensor([8, 8, 8], dtype=torch.int32))
    torch.testing.assert_close(ctx.query_start_loc.cpu(),
                               torch.arange(9, dtype=torch.int32))
    torch.testing.assert_close(ctx.seq_lens.cpu(),
                               torch.ones(8, dtype=torch.int32))
    assert runner.forward_model.call_args.kwargs["input_ids"].shape == (256, )
    torch.testing.assert_close(
        runner.forward_model.call_args.kwargs["positions"][:16].cpu(),
        torch.zeros(16, dtype=torch.int32),
    )


def test_dummy_run_marks_inactive_request_rows_after_cache_init(monkeypatch):
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=[64],
                          scheduled_tokens=[1],
                          token_paddings=[256])

    ctx = _run_dummy_run(monkeypatch, runner, num_tokens=3)
    md = runner._captured_dummy_attn_metadata["layer.0"]

    assert ctx is not None
    torch.testing.assert_close(
        md.query_start_loc.cpu(),
        torch.tensor([0, 1, 2, 3, 3, 3, 3, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(md.request_distribution.cpu(),
                               torch.tensor([3, 3, 3], dtype=torch.int32))


def test_dummy_run_marks_inactive_request_rows_before_cache_init(monkeypatch):
    runner = _make_runner(num_computed_tokens=[0],
                          prompt_tokens=[64],
                          scheduled_tokens=[1],
                          token_paddings=[256])
    runner.kv_cache_config = None
    runner._attn_layer_names = ["layer.0"]

    ctx = _run_dummy_run(monkeypatch, runner, num_tokens=3)
    md = runner._captured_dummy_attn_metadata["layer.0"]

    assert ctx is None
    torch.testing.assert_close(
        md.query_start_loc.cpu(),
        torch.tensor([0, 1, 2, 3, 3, 3, 3, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(md.request_distribution.cpu(),
                               torch.tensor([3, 3, 3], dtype=torch.int32))
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    assert md.sequence_layout_version == 1


def test_dummy_run_preinit_and_postinit_share_metadata_pytree(monkeypatch):
    import jax

    def make_runner():
        return _make_runner(num_computed_tokens=[0],
                            prompt_tokens=[64],
                            scheduled_tokens=[1],
                            token_paddings=[256])

    postinit_runner = make_runner()
    _run_dummy_run(monkeypatch, postinit_runner, num_tokens=3)
    postinit_md = postinit_runner._captured_dummy_attn_metadata["layer.0"]

    preinit_runner = make_runner()
    preinit_runner.kv_cache_config = None
    preinit_runner._attn_layer_names = ["layer.0"]
    _run_dummy_run(monkeypatch, preinit_runner, num_tokens=3)
    preinit_md = preinit_runner._captured_dummy_attn_metadata["layer.0"]

    assert jax.tree_util.tree_structure(preinit_md) == (
        jax.tree_util.tree_structure(postinit_md))
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


def test_prepare_inputs_batch_flat_decode_can_leave_rank_with_only_padding(
        monkeypatch):
    scheduled = [1, 1]
    runner = _make_runner(num_computed_tokens=[16, 16],
                          prompt_tokens=[16, 16],
                          scheduled_tokens=scheduled,
                          token_paddings=[2])

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 1)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    attn_metadata, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 2)

    md = attn_metadata["layer.0"]
    assert runner.input_ids.shape[0] == 2
    assert runner.position_ids.shape[0] == 2
    torch.testing.assert_close(runner.input_ids.cpu(),
                               torch.zeros(2, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids.cpu(),
                               torch.zeros(2, dtype=torch.int32))
    assert md.sequence_layout_kind == SequenceLayoutKind.PARTIAL.value
    assert md.sequence_layout_protocol == "pcp_streaming"
    torch.testing.assert_close(md.request_distribution.cpu(),
                               torch.tensor([2, 2, 2], dtype=torch.int32))
    torch.testing.assert_close(
        logits_indices.cpu(), torch.tensor([0, 1] + [-1] * 6,
                                           dtype=torch.int32))


def test_prepare_inputs_pcp_owner_does_not_rotate_with_absolute_q_start(
        monkeypatch):
    scheduled = [1]
    runner = _make_runner(num_computed_tokens=[16],
                          prompt_tokens=[64],
                          scheduled_tokens=scheduled,
                          token_paddings=[8])

    monkeypatch.setattr(_PCP_LAYOUT_RANK, lambda: 0)
    monkeypatch.setattr(_PCP_LAYOUT_WORLD_SIZE, lambda: 2)

    _, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 1)
    torch.testing.assert_close(logits_indices.cpu(),
                               torch.tensor([0] + [-1] * 7, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids.cpu()[:1],
                               torch.tensor([16], dtype=torch.int32))

    runner.input_batch.num_computed_tokens_cpu[0] = 32
    _, logits_indices, *_ = TPUModelRunner._prepare_inputs(
        runner, _scheduler_output(scheduled), 0, 1)

    torch.testing.assert_close(logits_indices.cpu(),
                               torch.tensor([0] + [-1] * 7, dtype=torch.int32))
    torch.testing.assert_close(runner.position_ids.cpu()[:1],
                               torch.tensor([32], dtype=torch.int32))


def test_prepare_inputs_batch_flat_owner_preserves_history_positions_and_logits(
        monkeypatch):
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
        runner, _scheduler_output(scheduled), 0, 0)

    expected_ids = torch.cat((
        torch.arange(216, 225, dtype=torch.int32),
        torch.arange(400, 407, dtype=torch.int32),
        torch.zeros(16, dtype=torch.int32),
    ))
    expected_positions = torch.cat((
        torch.arange(116, 125, dtype=torch.int32),
        torch.arange(200, 207, dtype=torch.int32),
        torch.zeros(16, dtype=torch.int32),
    ))
    torch.testing.assert_close(runner.input_ids.cpu(), expected_ids)
    torch.testing.assert_close(runner.position_ids.cpu(), expected_positions)
    torch.testing.assert_close(
        logits_indices.cpu(),
        torch.tensor([40, 31] + [-1] * 6, dtype=torch.int32),
    )

    md = attn_metadata["layer.0"]
    torch.testing.assert_close(md.query_start_loc.cpu()[:3],
                               torch.tensor([0, 25, 48], dtype=torch.int32))
    torch.testing.assert_close(md.seq_lens.cpu()[:2],
                               torch.tensor([125, 223], dtype=torch.int32))
    assert not hasattr(runner._attn_metadata_builder_ctx,
                       "token_owner_start_offsets")
