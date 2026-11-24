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
import torch

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)
from vllm_torchtpu.runner.pcp import PcpManager
from vllm_torchtpu.runner.pcp.interleaved_sharding_plan import (
    InterleavedPcpShardingPlan,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner_with_async_state(req_ids, previous_req_positions=None, pcp_manager=None):
    if previous_req_positions is None:
        pre_async_results = None
    else:
        pre_async_results = SimpleNamespace(
            req_id_to_index_copy=dict(previous_req_positions),
            spec_decode_num_rejected_tokens=None,
        )
    return SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=list(req_ids)),
        _pre_async_results=pre_async_results,
        speculative_config=None,
        pcp_manager=pcp_manager,
    )


def _current_async_substitution_indices(
    *,
    req_ids,
    previous_req_positions,
    scheduled_tokens,
    pcp_manager=None,
):
    runner = _runner_with_async_state(
        req_ids, previous_req_positions, pcp_manager=pcp_manager
    )
    return TPUModelRunner._prepare_async_token_substitution_indices(
        runner,
        start_index=0,
        num_reqs=len(req_ids),
        num_scheduled_tokens_per_req=np.asarray(scheduled_tokens, dtype=np.int32),
    )


def _expected_pcp_local_async_substitution_indices(
    *,
    req_ids,
    previous_req_positions,
    scheduled_tokens,
    pcp_size,
    interleave_size,
    local_padded_tokens,
    pcp_rank,
):
    """Expected non-spec async substitution indices after PCP token packing."""
    owner_starts = np.concatenate(
        ([0], np.cumsum(scheduled_tokens[:-1], dtype=np.int64))
    )
    _, inverse_order = build_pcp_rank_major_token_order(
        np.asarray(scheduled_tokens, dtype=np.int32),
        pcp_size,
        interleave_size,
        int(local_padded_tokens) * int(pcp_size),
        token_owner_start_offsets_per_req=owner_starts,
    )

    local_start = int(pcp_rank) * int(local_padded_tokens)
    local_end = local_start + int(local_padded_tokens)
    cur_indices: list[int] = []
    src_indices: list[int] = []
    req_major_start = 0
    for req_id, n_sched in zip(req_ids, scheduled_tokens):
        n_sched = int(n_sched)
        previous_position = previous_req_positions.get(req_id)
        if previous_position is not None:
            for offset in range(n_sched):
                packed_index = int(inverse_order[req_major_start + offset])
                if local_start <= packed_index < local_end:
                    cur_indices.append(packed_index - local_start)
                    src_indices.append(int(previous_position) + offset)
        req_major_start += n_sched

    return np.asarray(cur_indices, dtype=np.int32), np.asarray(
        src_indices, dtype=np.int32
    )


def _prepared_pcp_manager(
    *,
    req_ids,
    computed,
    prompt,
    scheduled_tokens,
    pcp_size,
    interleave_size,
    local_padded_tokens,
    pcp_rank,
) -> PcpManager:
    total_tokens = int(np.sum(scheduled_tokens))
    cap = int(local_padded_tokens) * int(pcp_size)
    positions_cpu = torch.arange(cap, dtype=torch.int32)
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        input_ids_cpu=torch.arange(cap, dtype=torch.int32),
        positions_cpu=positions_cpu,
        positions_np=positions_cpu.numpy(),
        arange_np=np.arange(cap, dtype=np.int64),
        uses_mrope=False,
        supports_mm_inputs=False,
        num_tokens_paddings=[int(local_padded_tokens)],
        max_num_tokens=int(local_padded_tokens),
        _dp_target_bucket=None,
        input_batch=SimpleNamespace(
            req_ids=list(req_ids),
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            num_computed_tokens_cpu=np.asarray(computed, dtype=np.int32),
            num_prompt_tokens=np.asarray(prompt, dtype=np.int32),
        ),
    )
    mgr = PcpManager(
        InterleavedPcpShardingPlan(
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            interleave_size=interleave_size,
            max_num_batched_tokens=cap,
        )
    )
    mgr.prepare_real(
        runner=runner,
        scheduler_output=SimpleNamespace(
            num_scheduled_tokens=dict(zip(req_ids, scheduled_tokens)),
            scheduled_encoder_inputs={},
        ),
        start_index=0,
        num_reqs=len(req_ids),
        num_scheduled_tokens_per_req=np.asarray(scheduled_tokens, dtype=np.int32),
        total_num_scheduled_tokens=total_tokens,
        use_max_model_len=True,
        target_num_reqs=len(req_ids),
        padded_num_reqs=len(req_ids),
    )
    return mgr


def test_prefill_only_without_prior_async_results_has_no_substitution():
    cur_indices, src_indices = _current_async_substitution_indices(
        req_ids=["long_prefill"],
        previous_req_positions=None,
        scheduled_tokens=[64],
    )

    assert cur_indices.tolist() == []
    assert src_indices.tolist() == []


def test_prefill_only_new_request_ignores_disjoint_prior_async_results():
    cur_indices, src_indices = _current_async_substitution_indices(
        req_ids=["new_long_prefill"],
        previous_req_positions={"finished_short": 0},
        scheduled_tokens=[64],
    )

    assert cur_indices.tolist() == []
    assert src_indices.tolist() == []


def test_decode_only_rank0_indices_match_pcp_local_layout_when_aligned():
    req_ids = ["decode0", "decode1", "decode2"]
    scheduled_tokens = [1, 1, 1]
    previous_req_positions = {
        "decode0": 0,
        "decode1": 1,
        "decode2": 2,
    }
    pcp_manager = _prepared_pcp_manager(
        req_ids=req_ids,
        computed=[16, 16, 16],
        prompt=[16, 16, 16],
        scheduled_tokens=scheduled_tokens,
        pcp_size=4,
        interleave_size=4,
        local_padded_tokens=32,
        pcp_rank=0,
    )

    current = _current_async_substitution_indices(
        req_ids=req_ids,
        previous_req_positions=previous_req_positions,
        scheduled_tokens=scheduled_tokens,
        pcp_manager=pcp_manager,
    )
    expected = _expected_pcp_local_async_substitution_indices(
        req_ids=req_ids,
        previous_req_positions=previous_req_positions,
        scheduled_tokens=scheduled_tokens,
        pcp_size=4,
        interleave_size=4,
        local_padded_tokens=32,
        pcp_rank=0,
    )

    np.testing.assert_array_equal(current[0], expected[0])
    np.testing.assert_array_equal(current[1], expected[1])


def test_mixed_prefill_decode_requires_pcp_local_destination_indices():
    """Verify mixed prefill/decode maps decode substitutions to rank-local slots."""
    req_ids = ["new_long_prefill", "decode0", "decode1", "decode2"]
    scheduled_tokens = [16, 1, 1, 1]
    previous_req_positions = {
        "decode0": 0,
        "decode1": 1,
        "decode2": 2,
    }
    for rank, expected_cur in ((0, [4, 5, 6]), (1, [])):
        pcp_manager = _prepared_pcp_manager(
            req_ids=req_ids,
            computed=[0, 16, 16, 16],
            prompt=[16, 16, 16, 16],
            scheduled_tokens=scheduled_tokens,
            pcp_size=4,
            interleave_size=4,
            local_padded_tokens=32,
            pcp_rank=rank,
        )
        current = _current_async_substitution_indices(
            req_ids=req_ids,
            previous_req_positions=previous_req_positions,
            scheduled_tokens=scheduled_tokens,
            pcp_manager=pcp_manager,
        )
        expected = _expected_pcp_local_async_substitution_indices(
            req_ids=req_ids,
            previous_req_positions=previous_req_positions,
            scheduled_tokens=scheduled_tokens,
            pcp_size=4,
            interleave_size=4,
            local_padded_tokens=32,
            pcp_rank=rank,
        )
        assert expected[0].tolist() == expected_cur
        np.testing.assert_array_equal(current[0], expected[0])
        np.testing.assert_array_equal(current[1], expected[1])
