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
"""PCP manager: rank-local request count vs the metadata's request bucket.

Every split prefill is two DualChunkSwap chunks on each rank, so a full batch
of `max_num_seqs` requests is up to `2 * max_num_seqs` local requests. The
runner sizes its attention request bucket by `local_reqs_per_req`; a bucket
of `max_num_seqs` crashed serving (`13 > 8` with 5 prefills + 3 decodes).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import HeadTailPcpShardingPlan

MAX_NUM_SEQS = 8
TOKEN_PADDINGS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def _manager(pcp_size, pcp_rank=0):
    return HeadTailPcpShardingPlan(
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
        dcp_size=8,
        page_size=64,
        workspace_tokens=1 << 20,
        max_num_batched_tokens=TOKEN_PADDINGS[-1],
        max_num_seqs=MAX_NUM_SEQS,
    )


def _runner(num_computed, num_prompt, q_lens, capacity):
    cap = 8 * sum(q_lens) + 8192
    positions = torch.zeros(cap, dtype=torch.int64)
    qsl = np.zeros(capacity + 1, dtype=np.int32)
    qsl[1 : len(q_lens) + 1] = np.cumsum(q_lens)
    qsl[len(q_lens) + 1 :] = qsl[len(q_lens)]
    seq_lens = np.zeros(capacity, dtype=np.int32)
    seq_lens[: len(q_lens)] = np.asarray(num_computed) + np.asarray(q_lens)
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.asarray(num_computed, dtype=np.int32),
            num_prompt_tokens=np.asarray(num_prompt, dtype=np.int32),
        ),
        num_tokens_paddings=TOKEN_PADDINGS,
        input_ids_cpu=torch.zeros(cap, dtype=torch.int32),
        positions_cpu=positions,
        positions_np=positions.numpy(),
        arange_np=np.arange(cap, dtype=np.int64),
        query_start_loc_np=qsl,
        seq_lens_np=seq_lens,
    )


def _prepare(manager, num_prefills, target_num_reqs):
    # `num_prefills` prefills, the rest of the batch decodes.
    num_decodes = MAX_NUM_SEQS - num_prefills
    q_lens = [300 + 7 * i for i in range(num_prefills)] + [1] * num_decodes
    num_computed = [0] * num_prefills + [500] * num_decodes
    num_prompt = [q + 100 for q in q_lens[:num_prefills]] + [500] * num_decodes
    runner = _runner(num_computed, num_prompt, q_lens, target_num_reqs)
    manager.prepare_real(
        runner=runner,
        scheduler_output=None,
        start_index=0,
        num_reqs=MAX_NUM_SEQS,
        num_scheduled_tokens_per_req=q_lens,
        total_num_scheduled_tokens=sum(q_lens),
        use_max_model_len=False,
        target_num_reqs=target_num_reqs,
        padded_num_reqs=MAX_NUM_SEQS,
    )
    return manager


def test_pcp_manager_needs_two_local_reqs_per_req():
    assert _manager(4).local_reqs_per_req == 2


def test_bucket_of_max_num_seqs_overflows():
    """The serving crash: 5 prefills + 3 decodes is 13 local requests."""
    with pytest.raises(RuntimeError, match="13 > 8"):
        _prepare(_manager(4), num_prefills=5, target_num_reqs=MAX_NUM_SEQS)


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
@pytest.mark.parametrize("num_prefills", range(MAX_NUM_SEQS + 1))
def test_local_reqs_fit_the_scaled_bucket(pcp_size, num_prefills):
    target = MAX_NUM_SEQS * _manager(pcp_size).local_reqs_per_req
    for rank in range(pcp_size):
        plan = _prepare(_manager(pcp_size, rank), num_prefills, target)
        assert plan.local_num_reqs <= target
        assert plan.local_num_split_reqs == 2 * num_prefills
        assert plan.rep_num_reqs == MAX_NUM_SEQS - num_prefills
        assert len(plan.host_metadata["ws_page_offset"]) == target


@pytest.fixture
def small_chunks(monkeypatch):
    # 2-page gather chunks, so the workspace is not rounded up to 128 pages.
    monkeypatch.setattr(envs, "TPU_PCP_KV_CHUNK_TOKENS", 1024)


def _ws_manager(workspace_tokens):
    # page 64 x DCP 8: 512-token virtual pages.
    return HeadTailPcpShardingPlan(
        pcp_size=2,
        pcp_rank=0,
        dcp_size=8,
        page_size=64,
        workspace_tokens=workspace_tokens,
        max_num_batched_tokens=TOKEN_PADDINGS[-1],
        max_num_seqs=MAX_NUM_SEQS,
    )


@pytest.mark.parametrize(
    "is_prefilling,seq_lens,expected",
    [
        # 3 + 3 + 3 pages of a 8-page workspace: the third prefill waits.
        ([True, True, True], [1500, 1100, 1300], 2),
        # Decodes take no workspace.
        ([False, True, False, True], [9000, 2000, 9000, 2000], 4),
        # A first request alone always runs.
        ([True, True], [4096, 1], 1),
        ([True, True], [4096, 0], 2),
    ],
)
def test_chunk_stops_at_the_kv_workspace(
    is_prefilling, seq_lens, expected, small_chunks
):
    manager = _ws_manager(4096)
    assert manager.num_ws_pages == 8
    got = manager.num_reqs_in_chunk(
        np.asarray(is_prefilling), np.asarray(seq_lens), capacity=64
    )
    assert got == expected


def test_chunk_keeps_the_request_bucket_limit():
    manager = _ws_manager(1 << 20)
    # 5 prefills are 10 local requests, past a bucket of 9.
    assert manager.num_reqs_in_chunk([True] * 5, [100] * 5, capacity=9) == 4


@pytest.mark.parametrize("workspace_tokens", [4096, 4097, 5000, 8191])
def test_workspace_holds_the_requested_tokens(workspace_tokens, small_chunks):
    """Rounded up, so a `max_model_len <= workspace_tokens` request fits."""
    manager = _ws_manager(workspace_tokens)
    assert manager.num_ws_pages * manager.virtual_page_size >= workspace_tokens
