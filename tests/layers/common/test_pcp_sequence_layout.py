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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_logits_indices, build_pcp_rank_major_token_order)
from vllm_torchtpu.layers.common.pcp_sequence_layout import (
    PcpSequenceLayoutEligibility, PcpSequenceLayoutMode,
    PcpSequenceLayoutPlanner)
from vllm_torchtpu.layers.common.sequence_layout import SequenceLayoutKind


def _eligibility(
    *,
    pcp_size=4,
    interleave_size=16,
    dcp_size=1,
    pipeline_parallel_size=1,
    async_scheduling=False,
    speculative_enabled=False,
    is_kv_producer=None,
):
    return PcpSequenceLayoutEligibility(
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        dcp_size=dcp_size,
        pipeline_parallel_size=pipeline_parallel_size,
        async_scheduling=async_scheduling,
        speculative_enabled=speculative_enabled,
        is_kv_producer=is_kv_producer,
    )


def _input_batch(*, computed, prompt):
    req_ids = [f"req{i}" for i in range(len(computed))]
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={
            req_id: idx
            for idx, req_id in enumerate(req_ids)
        },
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )


def _scheduler_output(scheduled):
    return SimpleNamespace(num_scheduled_tokens={
        f"req{i}": int(tokens)
        for i, tokens in enumerate(scheduled)
    })


def _evaluate(
        *,
        eligibility=None,
        computed,
        prompt,
        scheduled,
        num_tokens_paddings=(16, 32, 64),
        max_num_tokens=64,
):
    eligibility = eligibility or _eligibility()
    scheduled_np = np.asarray(scheduled, dtype=np.int32)
    return eligibility.evaluate_runner_chunk(
        input_batch=_input_batch(computed=computed, prompt=prompt),
        scheduler_output=_scheduler_output(scheduled),
        start_index=0,
        num_reqs=len(scheduled),
        num_scheduled_tokens_per_req=scheduled_np,
        num_tokens_paddings=num_tokens_paddings,
        max_num_tokens=max_num_tokens,
    )


def test_from_vllm_config_normalizes_and_records_kv_role():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4,
            cp_kv_cache_interleave_size=16,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(is_kv_producer=False),
    )

    eligibility = PcpSequenceLayoutEligibility.from_vllm_config(vllm_config)

    assert eligibility.enabled
    assert eligibility.pcp_size == 4
    assert eligibility.interleave_size == 16
    assert eligibility.is_kv_producer is False


def test_evaluate_runner_chunk_classifies_streaming_prefill():
    decision = _evaluate(computed=[0], prompt=[64], scheduled=[64])

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.pcp_size == 4
    assert decision.interleave_size == 16
    assert decision.pcp_alignment is None
    assert decision.local_required_tokens == 16
    assert decision.local_padded_tokens == 16
    assert decision.global_padded_tokens == 64
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([16, 16, 16, 16]))
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0]))
    np.testing.assert_array_equal(decision.token_owner_start_offsets_per_req,
                                  np.asarray([0]))


def test_evaluate_runner_chunk_accepts_decode_only_query_spans():
    decision = _evaluate(
        computed=[64, 64],
        prompt=[64, 64],
        scheduled=[1, 1],
        num_tokens_paddings=(2, 4),
        max_num_tokens=4,
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 2
    assert decision.local_padded_tokens == 2
    assert decision.global_padded_tokens == 8
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([2, 0, 0, 0]))
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([64, 64]))
    np.testing.assert_array_equal(decision.token_owner_start_offsets_per_req,
                                  np.asarray([0, 1]))


def test_evaluate_runner_chunk_disabled_when_pcp_is_off():
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=1),
        computed=[0],
        prompt=[64],
        scheduled=[64],
    )

    assert decision.mode is PcpSequenceLayoutMode.DISABLED


def test_evaluate_runner_chunk_accepts_mixed_prefill_decode_batch():
    decision = _evaluate(
        computed=[0, 64],
        prompt=[64, 64],
        scheduled=[64, 1],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 17
    assert decision.local_padded_tokens == 32
    assert decision.global_padded_tokens == 128
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([17, 16, 16, 16]))
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0, 64]))
    np.testing.assert_array_equal(decision.token_owner_start_offsets_per_req,
                                  np.asarray([0, 64]))


def test_evaluate_runner_chunk_rejects_boundary_crossing_chunk():
    with pytest.raises(NotImplementedError, match="boundary-crossing"):
        _evaluate(
            computed=[32],
            prompt=[64],
            scheduled=[64],
        )


def test_evaluate_runner_chunk_accepts_unaligned_q_len():
    decision = _evaluate(
        computed=[0],
        prompt=[32],
        scheduled=[32],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16
    assert decision.local_padded_tokens == 16
    assert decision.global_padded_tokens == 64
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([16, 16, 0, 0]))


def test_evaluate_runner_chunk_accepts_unaligned_q_start():
    decision = _evaluate(
        computed=[16],
        prompt=[80],
        scheduled=[64],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([16, 16, 16, 16]))


def test_batch_flat_owner_is_independent_from_request_absolute_position():
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=2),
        computed=[16],
        prompt=[32],
        scheduled=[16],
        num_tokens_paddings=(16, ),
        max_num_tokens=16,
    )

    assert decision.spans[0].absolute_query_start == 16
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([16]))
    np.testing.assert_array_equal(decision.token_owner_start_offsets_per_req,
                                  np.asarray([0]))
    assert decision.absolute_query_start_offsets_per_req is not (
        decision.token_owner_start_offsets_per_req)
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([16, 0]))

    token_order, inverse_order = build_pcp_rank_major_token_order(
        [16],
        pcp_size=2,
        interleave_size=16,
        padded_num_tokens=32,
        token_owner_start_offsets_per_req=(
            decision.token_owner_start_offsets_per_req),
    )
    np.testing.assert_array_equal(token_order[:16], np.arange(16))
    np.testing.assert_array_equal(token_order[16:], np.full(16, -1))
    np.testing.assert_array_equal(inverse_order, np.arange(16))
    np.testing.assert_array_equal(
        build_pcp_logits_indices(
            [16],
            pcp_size=2,
            interleave_size=16,
            padded_num_tokens=32,
            token_owner_start_offsets_per_req=(
                decision.token_owner_start_offsets_per_req),
        ),
        np.asarray([15]),
    )


def test_evaluate_runner_chunk_classifies_multi_active_streaming_prefill():
    decision = _evaluate(
        computed=[0, 0],
        prompt=[1024, 1024],
        scheduled=[1024, 1024],
        num_tokens_paddings=(256, 512),
        max_num_tokens=512,
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.pcp_alignment is None
    assert decision.local_required_tokens == 512
    assert decision.local_padded_tokens == 512
    assert decision.global_padded_tokens == 2048
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([512, 512, 512, 512]))
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0, 0]))
    np.testing.assert_array_equal(decision.token_owner_start_offsets_per_req,
                                  np.asarray([0, 1024]))


def test_batch_flat_owner_balances_32_by_256_across_pcp8():
    num_reqs = 32
    q_len = 256
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=8, interleave_size=256),
        computed=[0] * num_reqs,
        prompt=[q_len] * num_reqs,
        scheduled=[q_len] * num_reqs,
        num_tokens_paddings=(1024, ),
        max_num_tokens=1024,
    )

    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req,
        np.arange(num_reqs, dtype=np.int64) * q_len,
    )
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.full(8, 1024, dtype=np.int32))
    assert decision.local_required_tokens == 1024
    assert decision.local_padded_tokens == 1024
    assert decision.global_padded_tokens == 8192


def test_batch_flat_owner_balances_four_by_8192_across_pcp8():
    num_reqs = 4
    q_len = 8192
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=8, interleave_size=256),
        computed=[0] * num_reqs,
        prompt=[q_len] * num_reqs,
        scheduled=[q_len] * num_reqs,
        num_tokens_paddings=(4096, ),
        max_num_tokens=4096,
    )

    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req,
        np.arange(num_reqs, dtype=np.int64) * q_len,
    )
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.full(8, 4096, dtype=np.int32))
    assert decision.local_required_tokens == 4096
    assert decision.local_padded_tokens == 4096
    assert decision.global_padded_tokens == 4 * q_len


def test_evaluate_runner_chunk_accepts_unaligned_multi_active_q_len():
    decision = _evaluate(
        computed=[0, 0],
        prompt=[1024, 512],
        scheduled=[1024, 512],
        num_tokens_paddings=(256, 512),
        max_num_tokens=512,
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 384
    assert decision.local_padded_tokens == 512
    assert decision.global_padded_tokens == 2048
    np.testing.assert_array_equal(decision.local_token_counts,
                                  np.asarray([384, 384, 384, 384]))


def test_evaluate_runner_chunk_rejects_local_bucket_overflow():
    with pytest.raises(ValueError, match="max compile bucket"):
        _evaluate(
            computed=[0],
            prompt=[1024],
            scheduled=[1024],
            num_tokens_paddings=(128, ),
            max_num_tokens=128,
        )


@pytest.mark.parametrize(
    ("eligibility", "error_type", "message"),
    [
        (_eligibility(dcp_size=2), NotImplementedError, "DCP"),
        (_eligibility(pipeline_parallel_size=2), NotImplementedError,
         "pipeline parallelism"),
        (_eligibility(speculative_enabled=True), NotImplementedError,
         "speculative decoding"),
        (_eligibility(is_kv_producer=False), NotImplementedError,
         "KV consumer"),
        (_eligibility(interleave_size=0), ValueError, "interleave_size > 0"),
    ],
)
def test_evaluate_runner_chunk_rejects_unsupported_runtime_config(
        eligibility, error_type, message):
    with pytest.raises(error_type, match=message):
        _evaluate(eligibility=eligibility,
                  computed=[0],
                  prompt=[64],
                  scheduled=[64])


def test_evaluate_runner_chunk_accepts_async_non_speculative_runtime_config():
    decision = _evaluate(
        eligibility=_eligibility(async_scheduling=True),
        computed=[0],
        prompt=[64],
        scheduled=[64],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16


def _planner_runner_stub(*, scheduled, computed, prompt, token_paddings):
    max_tokens = max(token_paddings)
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        input_ids_cpu=torch.arange(max_tokens * 4, dtype=torch.int32),
        positions_cpu=torch.arange(max_tokens * 4, dtype=torch.int32),
        uses_mrope=False,
        supports_mm_inputs=False,
        positions_np=None,
        arange_np=np.arange(max_tokens * 4, dtype=np.int64),
        num_tokens_paddings=list(token_paddings),
        max_num_tokens=max_tokens,
        _dp_target_bucket=None,
    )
    runner.positions_np = runner.positions_cpu.numpy()
    req_ids = [f"req{i}" for i in range(len(scheduled))]
    runner.input_batch = SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={
            req_id: idx
            for idx, req_id in enumerate(req_ids)
        },
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )
    return runner


def test_pcp_sequence_layout_planner_returns_partial_plan(monkeypatch):
    runner = _planner_runner_stub(
        scheduled=[25],
        computed=[0],
        prompt=[25],
        token_paddings=[256],
    )
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=2))

    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.pcp_sequence_layout._get_native_pcp_rank",
        lambda: 1)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.pcp_sequence_layout."
        "_get_native_pcp_world_size", lambda: 2)

    plan = planner.prepare_real(
        runner=runner,
        scheduler_output=_scheduler_output([25]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.asarray([25], dtype=np.int32),
        total_num_scheduled_tokens=25,
        use_max_model_len=True,
        target_num_reqs=8,
        padded_num_reqs=8,
    )

    assert plan.kind is SequenceLayoutKind.PARTIAL
    assert plan.descriptor.protocol == "pcp_streaming"
    assert plan.global_padded_num_tokens == 512
    assert plan.local_padded_num_tokens == 256
    assert plan.local_num_tokens == 9
    assert plan.token_slice == slice(256, 512)
    assert plan.requires_hidden_state_gather is True
    assert plan.local_index_for_request_major_token(0) is None
    assert plan.local_index_for_request_major_token(16) == 0
    assert plan.local_index_for_request_major_token(24) == 8
    torch.testing.assert_close(
        plan.logits_indices_cpu,
        torch.tensor([264] + [-1] * 7, dtype=torch.int32),
    )
    torch.testing.assert_close(
        plan.logits_local_indices_cpu,
        torch.tensor([8] + [0] * 7, dtype=torch.int32),
    )
    torch.testing.assert_close(
        plan.logits_owner_mask_cpu,
        torch.tensor([True] + [False] * 7, dtype=torch.bool),
    )

    monkeypatch.setattr(
        "vllm_torchtpu.layers.common.pcp_sequence_layout."
        "_pcp_all_reduce_sum", lambda tensor: tensor)
    hidden_states = torch.arange(256 * 2, dtype=torch.float32).reshape(256, 2)
    selected = planner.maybe_select_logits_hidden_states(
        hidden_states,
        plan,
        plan.logits_indices_cpu,
    )
    expected = torch.zeros((8, 2), dtype=torch.float32)
    expected[0] = hidden_states[8]
    torch.testing.assert_close(selected, expected)


def test_pcp_sequence_layout_planner_returns_all_when_disabled():
    runner = _planner_runner_stub(
        scheduled=[25],
        computed=[0],
        prompt=[25],
        token_paddings=[32, 64],
    )
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=1))

    plan = planner.prepare_real(
        runner=runner,
        scheduler_output=_scheduler_output([25]),
        start_index=0,
        num_reqs=1,
        num_scheduled_tokens_per_req=np.asarray([25], dtype=np.int32),
        total_num_scheduled_tokens=25,
        use_max_model_len=True,
        target_num_reqs=8,
        padded_num_reqs=8,
    )

    assert plan.kind is SequenceLayoutKind.ALL
    assert plan.token_slice == slice(0, 32)
    assert plan.global_padded_num_tokens == 32
    assert plan.local_padded_num_tokens == 32
    assert plan.requires_hidden_state_gather is False


def test_pcp_planner_preinit_and_dummy_share_the_v1_layout_contract():
    planner = PcpSequenceLayoutPlanner(
        _eligibility(pcp_size=8, interleave_size=256))

    assert planner.requires_backend_preinit is True
    assert planner.backend_preinit_world_size == 8

    before_cache = planner.prepare_dummy(
        num_tokens=4096,
        num_reqs=32,
        kv_cache_initialized=False,
    )
    assert before_cache.kind is SequenceLayoutKind.PARTIAL
    assert before_cache.descriptor.protocol == "pcp_streaming"
    assert before_cache.descriptor.version == 1

    after_cache = planner.prepare_dummy(
        num_tokens=4096,
        num_reqs=32,
        kv_cache_initialized=True,
    )
    assert after_cache.kind is SequenceLayoutKind.PARTIAL
    assert after_cache.descriptor.protocol == "pcp_streaming"
    assert after_cache.descriptor.version == 1
    assert after_cache.token_slice == slice(0, 4096)
    assert after_cache.global_num_tokens == 8 * 4096
    assert after_cache.global_padded_num_tokens == 8 * 4096
    assert after_cache.local_num_tokens == 4096
    assert after_cache.local_padded_num_tokens == 4096
    assert after_cache.requires_hidden_state_gather is True

    assert before_cache == after_cache
