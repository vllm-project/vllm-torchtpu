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
"""PCP manager KV workspace sizing follows the MLA cache's page size."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.pcp_kv_gather import pcp_chunk_pages
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import (
    HeadTailPcpShardingPlan,
    mla_kv_page_size,
)
from vllm_torchtpu.runner.pcp.manager import validate_pcp_manager_config


def _manager(page_size, workspace_tokens=262144, dcp_size=8):
    return HeadTailPcpShardingPlan(
        pcp_size=4,
        pcp_rank=0,
        dcp_size=dcp_size,
        page_size=page_size,
        workspace_tokens=workspace_tokens,
        max_num_batched_tokens=4096,
        max_num_seqs=8,
    )


def _mla(block_size):
    return MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.uint8
    )


def _config(*specs):
    return SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"l{i}"], kv_cache_spec=s)
            for i, s in enumerate(specs)
        ]
    )


@pytest.mark.parametrize(
    "workspace_tokens, dcp_size", [(262144, 8), (1048576, 8), (1048576, 4), (8192, 8)]
)
def test_set_page_size_resizes_workspace(workspace_tokens, dcp_size):
    # `cache_config.block_size` (16) differs from the unified MLA page (1024).
    m = _manager(16, workspace_tokens, dcp_size)
    m.set_page_size(1024)
    assert m.page_size == 1024
    assert m.virtual_page_size == 1024 * dcp_size
    chunk_pages = pcp_chunk_pages(1024, dcp_size)
    assert m.num_ws_pages % chunk_pages == 0
    # Covers the requested tokens, overshooting by less than one chunk.
    ws_tokens = m.num_ws_pages * m.virtual_page_size
    assert ws_tokens >= min(workspace_tokens, m.virtual_page_size)
    assert ws_tokens < max(workspace_tokens, m.virtual_page_size) + (
        chunk_pages * m.virtual_page_size
    )


def test_page_size_16_vs_1024_is_64x_workspace():
    # The pre-fix sizing: 262144 tokens at 16-token pages is 2048 workspace
    # pages, i.e. 16.7M tokens once each page really holds 1024 per rank.
    m = _manager(16)
    assert m.num_ws_pages == 2048
    m.set_page_size(1024)
    assert m.num_ws_pages == 32


def test_mla_kv_page_size_plain_and_uniform_groups():
    assert mla_kv_page_size(_config(_mla(1024))) == 1024
    uniform = UniformTypeKVCacheSpecs(
        block_size=1024, kv_cache_specs={"a": _mla(1024), "b": _mla(1024)}
    )
    assert mla_kv_page_size(_config(uniform)) == 1024
    # Non-MLA groups are ignored.
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    assert mla_kv_page_size(_config(_mla(1024), full)) == 1024


@pytest.mark.parametrize("specs", [(), (_mla(1024), _mla(512))])
def test_mla_kv_page_size_requires_one_size(specs):
    with pytest.raises(ValueError, match="one MLA KV cache block size"):
        mla_kv_page_size(_config(*specs))


def _pcp_manager_config(max_model_len):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            use_mla=True,
            hf_text_config=SimpleNamespace(index_topk=2048),
            max_model_len=max_model_len,
            is_multimodal_model=False,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=8,
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            is_moe_model=False,
            enable_expert_parallel=False,
        ),
        speculative_config=None,
        lora_config=None,
    )


def test_max_model_len_must_fit_the_kv_workspace(monkeypatch):
    """One request's context must fit the workspace on its own; the runner
    chunks a step's split prefills to fit it otherwise."""
    monkeypatch.setattr(envs, "TPU_PCP_KV_WORKSPACE_TOKENS", 1 << 20)
    validate_pcp_manager_config(_pcp_manager_config(1 << 20))
    with pytest.raises(ValueError, match="TPU_PCP_KV_WORKSPACE_TOKENS"):
        validate_pcp_manager_config(_pcp_manager_config((1 << 20) + 1))
