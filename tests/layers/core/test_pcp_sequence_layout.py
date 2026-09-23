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

import vllm_torchtpu.layers.core.sequence_layout as sequence_layout
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_logits_indices,
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
    PcpSequenceLayoutEligibility,
    PcpSequenceLayoutMode,
    PcpSequenceLayoutPlanner,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    AllSequenceLayoutPlanner,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
    SequenceLayoutPlan,
)


def _eligibility(
    *,
    pcp_size=4,
    interleave_size=16,
    dcp_size=1,
    pipeline_parallel_size=1,
    async_scheduling=False,
    is_kv_producer=None,
):
    return PcpSequenceLayoutEligibility(
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        dcp_size=dcp_size,
        pipeline_parallel_size=pipeline_parallel_size,
        async_scheduling=async_scheduling,
        is_kv_producer=is_kv_producer,
    )


def _input_batch(*, computed, prompt):
    req_ids = [f"req{i}" for i in range(len(computed))]
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
        num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
    )


def _scheduler_output(scheduled):
    return SimpleNamespace(
        num_scheduled_tokens={
            f"req{i}": int(tokens) for i, tokens in enumerate(scheduled)
        }
    )


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
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([16, 16, 16, 16])
    )
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0])
    )


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
    np.testing.assert_array_equal(decision.local_token_counts, np.asarray([2, 0, 0, 0]))
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([64, 64])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0, 1])
    )


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
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([17, 16, 16, 16])
    )
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0, 64])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0, 64])
    )


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
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([16, 16, 0, 0])
    )


def test_evaluate_runner_chunk_accepts_unaligned_q_start():
    decision = _evaluate(
        computed=[16],
        prompt=[80],
        scheduled=[64],
    )

    assert decision.mode is PcpSequenceLayoutMode.STREAMING
    assert decision.local_required_tokens == 16
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([16, 16, 16, 16])
    )


def test_batch_flat_owner_is_independent_from_request_absolute_position():
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=2),
        computed=[16],
        prompt=[32],
        scheduled=[16],
        num_tokens_paddings=(16,),
        max_num_tokens=16,
    )

    assert decision.spans[0].absolute_query_start == 16
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([16])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0])
    )
    assert decision.absolute_query_start_offsets_per_req is not (
        decision.token_owner_start_offsets_per_req
    )
    np.testing.assert_array_equal(decision.local_token_counts, np.asarray([16, 0]))

    token_order, inverse_order = build_pcp_rank_major_token_order(
        [16],
        pcp_size=2,
        interleave_size=16,
        padded_num_tokens=32,
        token_owner_start_offsets_per_req=(decision.token_owner_start_offsets_per_req),
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
                decision.token_owner_start_offsets_per_req
            ),
        ),
        np.asarray([15]),
    )


@pytest.mark.parametrize(
    ("q_lens", "pcp_size", "interleave_size", "padded_num_tokens", "owner_starts"),
    [
        ([2, 3], 2, 1, 6, [0, 2]),
        ([3, 3], 2, 2, 8, [0, 3]),
        ([2, 5, 1], 4, 2, 16, [0, 2, 7]),
        ([1], 4, 2, 8, [5]),
    ],
)
def test_pcp_token_orders_are_inverses_and_each_request_has_one_owner(
    q_lens, pcp_size, interleave_size, padded_num_tokens, owner_starts
):
    packed_to_request, request_to_packed = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=padded_num_tokens,
        token_owner_start_offsets_per_req=owner_starts,
    )
    total_num_tokens = sum(q_lens)
    np.testing.assert_array_equal(
        packed_to_request[request_to_packed],
        np.arange(total_num_tokens),
    )

    request_gathers = torch.tensor(
        np.cumsum(q_lens, dtype=np.int64) - 1,
        dtype=torch.int64,
    )
    request_tokens = torch.arange(total_num_tokens, dtype=torch.int64)
    local_padded_num_tokens = padded_num_tokens // pcp_size
    owner_counts = torch.zeros(len(q_lens), dtype=torch.int64)
    for rank in range(pcp_size):
        local_start = rank * local_padded_num_tokens
        plan = SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(local_start, local_start + local_padded_num_tokens),
            global_num_tokens=total_num_tokens,
            global_padded_num_tokens=padded_num_tokens,
            local_num_tokens=int(
                np.count_nonzero(
                    packed_to_request[
                        local_start : local_start + local_padded_num_tokens
                    ]
                    >= 0
                )
            ),
            local_padded_num_tokens=local_padded_num_tokens,
            _packed_to_request_major_token_indices=packed_to_request,
            _request_major_to_packed_token_indices=request_to_packed,
        )

        local_tokens, local_gathers = plan.localize_token_tensor_and_gather_indices(
            request_tokens,
            request_gathers,
            num_valid_gathers=len(q_lens),
        )
        if plan.local_num_tokens == 0:
            torch.testing.assert_close(local_tokens, torch.zeros_like(local_tokens))
            torch.testing.assert_close(
                local_gathers, torch.full_like(local_gathers, -1)
            )
        owner_counts += local_gathers.ge(0)

    torch.testing.assert_close(owner_counts, torch.ones_like(owner_counts))


def test_partial_plan_ignores_padded_request_aligned_gathers():
    packed_to_request, request_to_packed = build_pcp_rank_major_token_order(
        [2, 3],
        pcp_size=2,
        interleave_size=1,
        padded_num_tokens=6,
        token_owner_start_offsets_per_req=[0, 2],
    )
    request_tokens = torch.tensor([10, 11, 20, 21, 22])
    request_gathers = torch.tensor([1, 4, 0])

    expected = [
        (torch.tensor([10, 20, 22]), torch.tensor([-1, 2, -1])),
        (torch.tensor([11, 21, 0]), torch.tensor([0, -1, -1])),
    ]
    for rank, (expected_tokens, expected_gathers) in enumerate(expected):
        local_start = rank * 3
        plan = SequenceLayoutPlan(
            descriptor=PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR,
            token_slice=slice(local_start, local_start + 3),
            global_num_tokens=5,
            global_padded_num_tokens=6,
            local_num_tokens=3 if rank == 0 else 2,
            local_padded_num_tokens=3,
            _packed_to_request_major_token_indices=packed_to_request,
            _request_major_to_packed_token_indices=request_to_packed,
        )

        local_tokens, local_gathers = plan.localize_token_tensor_and_gather_indices(
            request_tokens, request_gathers, num_valid_gathers=2
        )

        torch.testing.assert_close(local_tokens, expected_tokens)
        torch.testing.assert_close(local_gathers, expected_gathers)


def _carry_plan(descriptor: SequenceLayoutDescriptor) -> SequenceLayoutPlan:
    return SequenceLayoutPlan(
        descriptor=descriptor,
        token_slice=slice(0, 3),
        global_num_tokens=3,
        global_padded_num_tokens=3,
        local_num_tokens=3,
        local_padded_num_tokens=3,
    )


def test_default_plan_skips_pcp_carry_collective(monkeypatch):
    plan = AllSequenceLayoutPlanner().prepare_dummy(
        num_tokens=3,
        num_reqs=3,
        kv_cache_initialized=False,
    )
    local_tensor = torch.tensor([[99.0], [20.0], [7.0]])
    owner_mask = torch.tensor([False, True, False])
    monkeypatch.setattr(
        sequence_layout,
        "_pcp_all_reduce_sum",
        lambda _tensor: pytest.fail("default layout entered PCP collective"),
        raising=False,
    )

    result = plan.aggregate_request_aligned_tensor(local_tensor, owner_mask)

    torch.testing.assert_close(result, local_tensor)


def test_pcp_plan_masks_non_owner_before_request_aligned_reduce(monkeypatch):
    plan = _carry_plan(PCP_STREAMING_SEQUENCE_LAYOUT_DESCRIPTOR)
    local_tensor = torch.tensor([[99.0], [20.0], [7.0]])
    owner_mask = torch.tensor([False, True, False])
    reduced_inputs = []

    def fake_reduce(masked_tensor):
        reduced_inputs.append(masked_tensor.clone())
        remote_owner = torch.zeros_like(masked_tensor)
        remote_owner[0, 0] = 10
        return masked_tensor + remote_owner

    monkeypatch.setattr(
        sequence_layout, "_pcp_all_reduce_sum", fake_reduce, raising=False
    )

    result = plan.aggregate_request_aligned_tensor(local_tensor, owner_mask)

    torch.testing.assert_close(reduced_inputs[0], torch.tensor([[0.0], [20.0], [0.0]]))
    torch.testing.assert_close(result, torch.tensor([[10.0], [20.0], [0.0]]))


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
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([512, 512, 512, 512])
    )
    np.testing.assert_array_equal(
        decision.absolute_query_start_offsets_per_req, np.asarray([0, 0])
    )
    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req, np.asarray([0, 1024])
    )


def test_batch_flat_owner_balances_32_by_256_across_pcp8():
    num_reqs = 32
    q_len = 256
    decision = _evaluate(
        eligibility=_eligibility(pcp_size=8, interleave_size=256),
        computed=[0] * num_reqs,
        prompt=[q_len] * num_reqs,
        scheduled=[q_len] * num_reqs,
        num_tokens_paddings=(1024,),
        max_num_tokens=1024,
    )

    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req,
        np.arange(num_reqs, dtype=np.int64) * q_len,
    )
    np.testing.assert_array_equal(
        decision.local_token_counts, np.full(8, 1024, dtype=np.int32)
    )
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
        num_tokens_paddings=(4096,),
        max_num_tokens=4096,
    )

    np.testing.assert_array_equal(
        decision.token_owner_start_offsets_per_req,
        np.arange(num_reqs, dtype=np.int64) * q_len,
    )
    np.testing.assert_array_equal(
        decision.local_token_counts, np.full(8, 4096, dtype=np.int32)
    )
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
    np.testing.assert_array_equal(
        decision.local_token_counts, np.asarray([384, 384, 384, 384])
    )


def test_evaluate_runner_chunk_rejects_local_bucket_overflow():
    with pytest.raises(ValueError, match="max compile bucket"):
        _evaluate(
            computed=[0],
            prompt=[1024],
            scheduled=[1024],
            num_tokens_paddings=(128,),
            max_num_tokens=128,
        )


@pytest.mark.parametrize(
    ("eligibility", "error_type", "message"),
    [
        (_eligibility(dcp_size=2), NotImplementedError, "DCP"),
        (
            _eligibility(pipeline_parallel_size=2),
            NotImplementedError,
            "pipeline parallelism",
        ),
        (_eligibility(is_kv_producer=False), NotImplementedError, "KV consumer"),
        (_eligibility(interleave_size=0), ValueError, "interleave_size > 0"),
    ],
)
def test_evaluate_runner_chunk_rejects_unsupported_runtime_config(
    eligibility, error_type, message
):
    with pytest.raises(error_type, match=message):
        _evaluate(eligibility=eligibility, computed=[0], prompt=[64], scheduled=[64])


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
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
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
        "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_rank", lambda: 1
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.core.pcp_sequence_layout._get_native_pcp_world_size",
        lambda: 2,
    )

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
    expected_packed_to_request = np.full(512, -1, dtype=np.int64)
    expected_packed_to_request[:16] = np.arange(16)
    expected_packed_to_request[256:265] = np.arange(16, 25)
    np.testing.assert_array_equal(
        plan._packed_to_request_major_token_indices,
        expected_packed_to_request,
    )
    np.testing.assert_array_equal(
        plan._request_major_to_packed_token_indices,
        np.concatenate((np.arange(16), np.arange(256, 265))),
    )
    local_tokens, local_gathers = plan.localize_token_tensor_and_gather_indices(
        torch.arange(25, dtype=torch.int32),
        torch.tensor([24]),
        num_valid_gathers=1,
    )
    torch.testing.assert_close(
        local_tokens[:9], torch.arange(16, 25, dtype=torch.int32)
    )
    torch.testing.assert_close(local_tokens[9:], torch.zeros(247, dtype=torch.int32))
    torch.testing.assert_close(local_gathers, torch.tensor([8]))
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
        "vllm_torchtpu.layers.core.pcp_sequence_layout._pcp_all_reduce_sum",
        lambda tensor: tensor,
    )
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
    planner = PcpSequenceLayoutPlanner(_eligibility(pcp_size=8, interleave_size=256))

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
