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

import pytest
import torch
from runner_input_test_utils import _BlockTable, _make_runner, _scheduler_output

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


@pytest.mark.parametrize(
    "max_model_len,expected_budget,expected_chunks",
    [(1048576, 47, [47, 17]), (65536, 766, [64])],
)
@pytest.mark.parametrize("dcp_size", [1, 8])
@pytest.mark.parametrize("scheduled_per_req", [1, 4])
def test_smem_page_table_budget_splits_short_decode_batch(
    max_model_len, expected_budget, expected_chunks, dcp_size, scheduled_per_req
):
    """Reproduce the 1M SMEM clamp using real CPU input preparation.

    The token store only needs the actual P8K+D1K context. The old global
    budget is checked first, then final rank-local tables refresh the cap.
    Four scheduled tokens also exercise the K=3 verification window's input
    path; this harness does not execute the MTP drafter or a TPU kernel.
    """
    scheduled_tokens = [scheduled_per_req] * 64
    computed = 8192 + 1024 - scheduled_per_req
    runner = _make_runner(
        num_computed_tokens=[computed] * 64,
        prompt_tokens=[8192 + 1024] * 64,
        scheduled_tokens=scheduled_tokens,
        token_paddings=[64, 256],
        pcp_size=1,
        max_num_reqs=64,
        block_size=384,
    )
    runner.max_model_len = max_model_len
    runner.parallel_config.decode_context_parallel_size = dcp_size
    budget = PallasAttentionBackend.get_max_num_seqs(
        runner.max_model_len, runner.block_size
    )
    assert budget == expected_budget
    spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    pages = spec.max_num_blocks_per_req(runner.vllm_config, max_model_len)
    runner.input_batch.block_table[0] = _BlockTable(
        torch.zeros((64, pages), dtype=torch.int32)
    )
    runner.max_num_blocks_per_req = pages
    TPUModelRunner._refresh_smem_request_caps(runner)
    if dcp_size == 8:
        expected_chunks = [64]
    scheduler_output = _scheduler_output(scheduled_tokens)
    assert TPUModelRunner._count_input_chunks(runner, scheduler_output) == (
        len(expected_chunks),
        max(expected_chunks),
    )

    chunks = []
    start_index = 0
    while start_index < runner.input_batch.num_reqs:
        result = TPUModelRunner._prepare_inputs(
            runner, scheduler_output, start_index, num_decode_reqs=64
        )
        num_reqs, end_index = result[3:5]
        assert end_index == start_index + num_reqs
        ctx = runner._attn_metadata_builder_ctx
        assert ctx.use_max_model_len
        assert ctx.start_index == start_index
        assert ctx.seq_lens[:num_reqs].tolist() == [8192 + 1024] * num_reqs
        assert ctx.query_start_loc[: num_reqs + 1].tolist() == [
            i * scheduled_per_req for i in range(num_reqs + 1)
        ]
        # Check the second chunk gathers requests 47..63, not 0..16 again.
        expected_tokens = runner.input_batch.token_ids_cpu_tensor[
            start_index:end_index, computed : computed + scheduled_per_req
        ].flatten()
        torch.testing.assert_close(
            runner.input_ids[: num_reqs * scheduled_per_req], expected_tokens
        )
        chunks.append(num_reqs)
        start_index = end_index

    assert chunks == expected_chunks
    assert sum(chunks) == 64
