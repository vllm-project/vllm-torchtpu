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
"""`InterleavedPcpShardingPlan`: rank-major packing of the step's flat token
stream, request-major -> local rows, and the sampling rows."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.pcp.interleaved_sharding_plan import (
    InterleavedPcpShardingPlan,
)

PCP = 4
INTERLEAVE = 16
TOKEN_PADDINGS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def _runner(num_computed, num_prompt, q_lens):
    num_tokens = sum(q_lens)
    cap = num_tokens
    input_ids = torch.zeros(cap, dtype=torch.int32)
    # Token `t` of the request-major stream has id `t + 1`, position `1000 + t`.
    input_ids[:num_tokens] = torch.arange(1, num_tokens + 1, dtype=torch.int32)
    positions = torch.zeros(cap, dtype=torch.int64)
    positions[:num_tokens] = torch.arange(1000, 1000 + num_tokens)
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.asarray(num_computed, dtype=np.int32),
            num_prompt_tokens=np.asarray(num_prompt, dtype=np.int32),
        ),
        num_tokens_paddings=TOKEN_PADDINGS,
        input_ids_cpu=input_ids,
        positions_cpu=positions,
        positions_np=positions.numpy(),
        arange_np=np.arange(cap, dtype=np.int64),
    )


def _prepare(pcp_rank, q_lens, num_computed=None, num_prompt=None):
    num_computed = num_computed or [0] * len(q_lens)
    num_prompt = num_prompt or [c + q for c, q in zip(num_computed, q_lens)]
    runner = _runner(num_computed, num_prompt, q_lens)
    plan = InterleavedPcpShardingPlan(
        pcp_size=PCP,
        pcp_rank=pcp_rank,
        dcp_size=1,
        interleave_size=INTERLEAVE,
        max_num_batched_tokens=TOKEN_PADDINGS[-1],
    )
    plan.prepare_real(
        runner=runner,
        scheduler_output=None,
        start_index=0,
        num_reqs=len(q_lens),
        num_scheduled_tokens_per_req=q_lens,
        total_num_scheduled_tokens=sum(q_lens),
        use_max_model_len=False,
        target_num_reqs=8,
        padded_num_reqs=8,
    )
    return plan, runner


def _owner(token: int) -> int:
    return (token // INTERLEAVE) % PCP


# Prefills, a prefill that is not a multiple of the interleave, and decodes.
Q_LENS = [100, 37, 1, 64, 1]
NUM_COMPUTED = [0, 0, 50, 0, 9]
NUM_PROMPT = [100, 37, 50, 64, 9]


@pytest.mark.parametrize("pcp_rank", range(PCP))
def test_rank_holds_its_interleave_chunks_of_the_flat_stream(pcp_rank):
    plan, runner = _prepare(pcp_rank, Q_LENS, NUM_COMPUTED, NUM_PROMPT)
    local_ids = runner.input_ids_cpu[plan.token_slice].numpy()
    local_pos = runner.positions_np[plan.token_slice]
    expected = [t for t in range(sum(Q_LENS)) if _owner(t) == pcp_rank]
    n = len(expected)
    assert plan.local_num_tokens == n
    np.testing.assert_array_equal(local_ids[:n], np.asarray(expected) + 1)
    np.testing.assert_array_equal(local_pos[:n], np.asarray(expected) + 1000)
    assert not local_ids[n:].any()
    assert plan.local_padded_num_tokens in TOKEN_PADDINGS
    assert plan.global_padded_num_tokens == PCP * plan.local_padded_num_tokens
    # The attention metadata stays global request-major.
    assert plan.local_num_reqs == len(Q_LENS)
    assert plan.local_block_rows is None
    assert (
        plan.build_metadata(
            runner=runner, block_table_obj=None, target_num_reqs=8, target_num_blocks=1
        )
        is None
    )


@pytest.mark.parametrize("pcp_rank", range(PCP))
def test_request_major_to_local(pcp_rank):
    plan, runner = _prepare(pcp_rank, Q_LENS, NUM_COMPUTED, NUM_PROMPT)
    local_ids = runner.input_ids_cpu[plan.token_slice]
    for t in range(sum(Q_LENS)):
        local = plan.local_index_for_request_major_token(t)
        if _owner(t) == pcp_rank:
            assert local is not None and int(local_ids[local]) == t + 1
        else:
            assert local is None


def test_each_last_token_has_exactly_one_owner():
    last_tokens = np.cumsum(Q_LENS) - 1
    owners = np.zeros(len(Q_LENS), dtype=int)
    for pcp_rank in range(PCP):
        plan, runner = _prepare(pcp_rank, Q_LENS, NUM_COMPUTED, NUM_PROMPT)
        local_ids = runner.input_ids_cpu[plan.token_slice]
        mask = plan.logits_owner_mask_cpu.numpy()
        assert not mask[len(Q_LENS) :].any()
        for i, t in enumerate(last_tokens):
            if mask[i]:
                owners[i] += 1
                local = int(plan.logits_local_indices_cpu[i])
                assert int(local_ids[local]) == t + 1
    np.testing.assert_array_equal(owners, 1)


def test_max_local_num_tokens_bounds_a_full_step():
    q_lens = [TOKEN_PADDINGS[-1]]
    plan, _ = _prepare(0, q_lens)
    assert plan.local_num_tokens <= plan.max_local_num_tokens


def test_rejects_a_request_crossing_its_prompt_boundary():
    with pytest.raises(NotImplementedError, match="prompt/decode boundary"):
        _prepare(0, [10], num_computed=[0], num_prompt=[5])


def test_rejects_an_unscheduled_request():
    with pytest.raises(NotImplementedError, match="scheduled tokens"):
        _prepare(0, [10, 0])


def test_rejects_a_zero_interleave():
    with pytest.raises(ValueError, match="cp_kv_cache_interleave_size"):
        InterleavedPcpShardingPlan(
            pcp_size=PCP,
            pcp_rank=0,
            dcp_size=1,
            interleave_size=0,
            max_num_batched_tokens=16,
        )


def test_prepare_dummy():
    plan = InterleavedPcpShardingPlan(
        pcp_size=PCP,
        pcp_rank=1,
        dcp_size=1,
        interleave_size=INTERLEAVE,
        max_num_batched_tokens=TOKEN_PADDINGS[-1],
    )
    plan.prepare_dummy(num_tokens=64, num_reqs=4, max_num_tokens=4096)
    assert plan.token_slice == slice(0, 64)
    assert plan.local_padded_num_tokens == 64
    assert plan.global_padded_num_tokens == PCP * 64
    assert plan.local_num_reqs == 4
    assert plan.logits_local_indices_cpu is None
