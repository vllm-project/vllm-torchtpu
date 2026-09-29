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
"""The shared top-k buffer must hold the request-major rows of the largest
warm-up bucket under PCP x DCP.

Config mirrors the GLM-5.2 run: PCP=2, DCP=16, interleave=4,
max_num_batched_tokens=1024. Warm-up compiles every bucket up to
max_num_batched_tokens *per PCP rank*, so its top bucket has 1024 local rows
and 2 * 1024 = 2048 request-major rows, while upstream allocates the buffer
with max_num_batched_tokens = 1024 rows.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    PcpSequenceLayoutEligibility, PcpSequenceLayoutPlanner)

PCP = 2
DCP = 16
INTERLEAVE = 4
MAX_NUM_BATCHED_TOKENS = 1024
TOPK = 8
HEAD_DIM = 128
N_HEAD = 4
PREFIX = "model.layers.0.self_attn.indexer.k_cache"


def _warmup_top_bucket_plan():
    planner = PcpSequenceLayoutPlanner(
        PcpSequenceLayoutEligibility(
            pcp_size=PCP,
            interleave_size=INTERLEAVE,
            dcp_size=DCP,
            pipeline_parallel_size=1,
            async_scheduling=False,
            is_kv_producer=None,
        ))
    # `_precompile_backbone` calls `_dummy_run(num_tokens)` for every entry of
    # `num_tokens_paddings`, whose last entry is max_num_batched_tokens.
    return planner.prepare_dummy(num_tokens=MAX_NUM_BATCHED_TOKENS,
                                 num_reqs=1,
                                 kv_cache_initialized=True)


def _upstream_topk_buffer():
    # deepseek_v2.py: torch.empty(max_num_batched_tokens, index_topk, int32)
    return torch.full((MAX_NUM_BATCHED_TOKENS, TOPK), -1, dtype=torch.int32)


def _run_indexer_write(topk_indices_buffer, local_rows, request_major_rows):
    """Run the real `VllmTPUSparseAttnIndexer.forward_oot` DCP branch with the
    top-k kernel stubbed to return `request_major_rows` rows, which is what
    it returns after `gather_to_request_major` under PCP."""
    kernel_out = torch.arange(request_major_rows,
                              dtype=torch.int32)[:, None].expand(
                                  request_major_rows, TOPK).contiguous()
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(prefix=PREFIX,
                                kv_cache=torch.zeros(4,
                                                     8,
                                                     1,
                                                     256,
                                                     dtype=torch.uint8)),
        topk_op_dcp=lambda *args: kernel_out,
        topk_op=None,
        dcp_size=DCP,
        topk_tokens=TOPK,
        topk_indices_buffer=topk_indices_buffer,
    )
    metadata = MagicMock(spec=AttentionMetadata)
    metadata.request_distribution = torch.zeros(3, dtype=torch.int32)
    metadata.pcp_request_major_to_packed_token_indices = None
    context = SimpleNamespace(attn_metadata={PREFIX: metadata})

    with patch.object(mla_attention_op,
                      "get_forward_context",
                      return_value=context):
        return mla_attention_op.VllmTPUSparseAttnIndexer.forward_oot(
            indexer,
            hidden_states=torch.zeros(local_rows, 16),
            q_values=torch.zeros(local_rows, N_HEAD,
                                 HEAD_DIM).to(torch.float8_e4m3fn),
            k=torch.zeros(local_rows, HEAD_DIM),
            weights=torch.zeros(local_rows, N_HEAD),
        )


def _rows_attention_receives(indexer_out, num_query_rows):
    # PallasMLAttentionBackendImpl.forward narrows the top-k to the gathered
    # query rows (`topk_indices[:ql_nope_flat.shape[0]]`), and
    # sparse_mla_attention_dcp raises ValueError unless the two match.
    return indexer_out[:num_query_rows].shape[0]


def test_warmup_top_bucket_is_pcp_times_max_num_batched_tokens():
    plan = _warmup_top_bucket_plan()
    assert plan.local_padded_num_tokens == MAX_NUM_BATCHED_TOKENS
    assert plan.global_padded_num_tokens == PCP * MAX_NUM_BATCHED_TOKENS
    assert (plan.global_padded_num_tokens > _upstream_topk_buffer().shape[0])


def test_topk_reaches_attention_intact_at_top_warmup_bucket():
    """Fails before the fix: the upstream-sized buffer truncates the write to
    1024 rows, so attention gets 1024 top-k rows for 2048 query rows."""
    plan = _warmup_top_bucket_plan()
    num_query_rows = plan.global_padded_num_tokens  # already a DCP multiple
    buf = _upstream_topk_buffer()
    grow = getattr(mla_attention_op, "_grow_topk_buffer", None)
    if grow is not None:
        # What VllmTPUMultiHeadLatentAttentionWrapper.__init__ does to the
        # shared buffer under PCP x DCP.
        buf = grow(buf, PCP, DCP)
    out = _run_indexer_write(buf,
                             local_rows=plan.local_padded_num_tokens,
                             request_major_rows=num_query_rows)
    topk = out[:num_query_rows]
    assert _rows_attention_receives(out, num_query_rows) == num_query_rows
    # Every request-major row survives, including the second PCP chunk's.
    assert torch.equal(topk[:, 0],
                       torch.arange(num_query_rows, dtype=torch.int32))


@pytest.mark.skipif(not hasattr(mla_attention_op, "_grow_topk_buffer"),
                    reason="needs the overflow check from the fix")
def test_ungrown_buffer_raises_instead_of_truncating():
    plan = _warmup_top_bucket_plan()
    with pytest.raises(ValueError, match="top-k buffer is short"):
        _run_indexer_write(_upstream_topk_buffer(),
                           local_rows=plan.local_padded_num_tokens,
                           request_major_rows=plan.global_padded_num_tokens)
