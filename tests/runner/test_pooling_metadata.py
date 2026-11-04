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


from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.model_executor.layers.pooler.tokwise.methods import StepPool
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from vllm_torchtpu.runner.tpu_pooling import (
    PromptTokenIdsTensor,
    TPUMeanPool,
    build_pooling_metadata,
    install_tpu_pooling_patches,
)


@pytest.fixture(autouse=True)
def setup_patches():
    install_tpu_pooling_patches()


def test_prompt_token_ids_tensor_dual_interface():
    """Verify PromptTokenIdsTensor supports both tensor operations and dict mapping."""
    raw_tensor = torch.tensor(
        [[101, 2054, 102, 0], [101, 1037, 2005, 102]], dtype=torch.int32
    )
    req_map = {
        "req-alpha": [101, 2054, 102],
        "req-beta": [101, 1037, 2005, 102],
    }

    tensor = PromptTokenIdsTensor(raw_tensor, req_map)

    # Tensor indexing and slicing
    assert tensor.shape == (2, 4)
    assert tensor.dtype == torch.int32
    assert torch.equal(tensor[0, :3], torch.tensor([101, 2054, 102], dtype=torch.int32))
    assert torch.equal(
        tensor[1, :4], torch.tensor([101, 1037, 2005, 102], dtype=torch.int32)
    )

    # Dict interface operations
    assert tensor["req-alpha"] == [101, 2054, 102]
    assert tensor["req-beta"] == [101, 1037, 2005, 102]
    assert "req-alpha" in tensor
    assert "req-unknown" not in tensor
    assert tensor.get("req-alpha") == [101, 2054, 102]
    assert tensor.get("req-missing", []) == []

    # Dict keys, values, items
    assert set(tensor.keys()) == {"req-alpha", "req-beta"}
    assert list(tensor.values()) == [[101, 2054, 102], [101, 1037, 2005, 102]]
    assert dict(tensor.items()) == req_map

    # Slicing retains dict mapping
    sub = tensor[0:1]
    assert isinstance(sub, PromptTokenIdsTensor)
    assert sub["req-alpha"] == [101, 2054, 102]


def test_build_pooling_metadata_from_runner_requests():
    """Verify build_pooling_metadata accurately extracts request metadata."""
    req1 = MagicMock()
    req1.req_id = "req-1"
    req1.prompt_token_ids = [10, 20, 30]
    req1.num_prompt_tokens = 3
    req1.prompt_token_ids_len = 3
    req1.pooling_params = PoolingParams(task="embed")
    req1.pooling_states = PoolingStates()

    req2 = MagicMock()
    req2.req_id = "req-2"
    req2.prompt_token_ids = [40, 50]
    req2.num_prompt_tokens = 2
    req2.prompt_token_ids_len = 2
    req2.pooling_params = PoolingParams(task="embed")
    req2.pooling_states = PoolingStates()

    runner_requests = {"req-1": req1, "req-2": req2}

    batch = MagicMock()
    batch.requests = None
    batch.req_ids = ["req-1", "req-2"]
    batch.num_reqs = 2

    metadata = build_pooling_metadata(batch, runner_requests)

    # Prompt lengths
    assert torch.equal(metadata.prompt_lens, torch.tensor([3, 2], dtype=torch.int32))

    # Prompt token IDs dual tensor
    assert isinstance(metadata.prompt_token_ids_cpu, PromptTokenIdsTensor)
    assert metadata.prompt_token_ids_cpu.shape == (2, 3)
    assert metadata.prompt_token_ids_cpu["req-1"] == [10, 20, 30]
    assert metadata.prompt_token_ids_cpu["req-2"] == [40, 50]

    # Persistent pooling states
    assert len(metadata.pooling_states) == 2
    assert metadata.pooling_states[0] is req1.pooling_states
    assert metadata.pooling_states[1] is req2.pooling_states


def test_build_pooling_metadata_preemption_and_persistence():
    """Verify pooling_states persist across successive chunks."""
    req = MagicMock()
    req.req_id = "req-stream"
    req.prompt_token_ids = [1, 2, 3, 4]
    req.num_prompt_tokens = 4
    req.prompt_token_ids_len = 4
    req.pooling_params = PoolingParams(task="embed")
    req.pooling_states = PoolingStates()

    runner_requests = {"req-stream": req}
    batch = MagicMock()
    batch.requests = None
    batch.req_ids = ["req-stream"]
    batch.num_reqs = 1

    # Chunk 1
    meta1 = build_pooling_metadata(batch, runner_requests)
    p_state1 = meta1.pooling_states[0]
    p_state1.mean_pool_sum = torch.tensor([2.5, 3.5])
    p_state1.mean_pool_count = 2

    # Chunk 2
    meta2 = build_pooling_metadata(batch, runner_requests)
    p_state2 = meta2.pooling_states[0]

    # Must be the exact same state object preserving intermediate sum
    assert p_state2 is p_state1
    assert torch.equal(p_state2.mean_pool_sum, torch.tensor([2.5, 3.5]))
    assert p_state2.mean_pool_count == 2


def test_tpu_mean_pool_single_pass():
    """Verify TPUMeanPool produces correct embeddings in single-pass prefill."""
    pooler = TPUMeanPool()
    dim = 8

    # 2 requests: req1 has 4 tokens, req2 has 6 tokens
    prompt_lens = [4, 6]
    total_tokens = sum(prompt_lens)
    hidden_states = torch.randn(total_tokens, dim, dtype=torch.float32)

    req1 = MagicMock()
    req1.req_id = "r1"
    req1.prompt_token_ids = list(range(4))
    req1.num_prompt_tokens = 4
    req1.prompt_token_ids_len = 4
    req1.pooling_params = PoolingParams(task="embed")
    req1.pooling_states = PoolingStates()

    req2 = MagicMock()
    req2.req_id = "r2"
    req2.prompt_token_ids = list(range(6))
    req2.num_prompt_tokens = 6
    req2.prompt_token_ids_len = 6
    req2.pooling_params = PoolingParams(task="embed")
    req2.pooling_states = PoolingStates()

    batch = MagicMock()
    batch.requests = None
    batch.req_ids = ["r1", "r2"]
    batch.num_reqs = 2

    metadata = build_pooling_metadata(batch, {"r1": req1, "r2": req2})
    metadata.build_pooling_cursor(
        num_scheduled_tokens_np=np.array([4, 6], dtype=np.int32),
        seq_lens_cpu=torch.tensor([4, 6], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    out = pooler.forward(hidden_states, metadata)

    assert out.shape == (2, dim)
    expected_r1 = hidden_states[:4].mean(dim=0)
    expected_r2 = hidden_states[4:].mean(dim=0)

    assert torch.allclose(out[0], expected_r1, atol=1e-5)
    assert torch.allclose(out[1], expected_r2, atol=1e-5)


def test_tpu_mean_pool_chunked_prefill_online_accumulation():
    """Verify TPUMeanPool online FP32 accumulation across chunked prefill steps."""
    pooler = TPUMeanPool()
    dim = 4
    total_tokens = 10

    # Request of 10 tokens split into chunks: [4, 4, 2]
    all_hidden_states = torch.randn(total_tokens, dim, dtype=torch.float32)
    expected_mean = all_hidden_states.mean(dim=0)

    req = MagicMock()
    req.req_id = "chunked-req"
    req.prompt_token_ids = list(range(total_tokens))
    req.num_prompt_tokens = total_tokens
    req.prompt_token_ids_len = total_tokens
    req.pooling_params = PoolingParams(task="embed")
    req.pooling_states = PoolingStates()

    runner_requests = {"chunked-req": req}
    batch = MagicMock()
    batch.requests = None
    batch.req_ids = ["chunked-req"]
    batch.num_reqs = 1

    # Chunk 1: 4 tokens (0 -> 4, optimistic seq_len = 4)
    hs_chunk1 = all_hidden_states[0:4]
    meta1 = build_pooling_metadata(batch, runner_requests)
    meta1.build_pooling_cursor(
        num_scheduled_tokens_np=np.array([4], dtype=np.int32),
        seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    out1 = pooler.forward(hs_chunk1, meta1)
    assert out1 == [None]
    assert req.pooling_states.mean_pool_count == 4
    assert torch.allclose(req.pooling_states.mean_pool_sum, hs_chunk1.sum(dim=0))

    # Chunk 2: 4 tokens (4 -> 8, optimistic seq_len = 8)
    hs_chunk2 = all_hidden_states[4:8]
    meta2 = build_pooling_metadata(batch, runner_requests)
    meta2.build_pooling_cursor(
        num_scheduled_tokens_np=np.array([4], dtype=np.int32),
        seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    out2 = pooler.forward(hs_chunk2, meta2)
    assert out2 == [None]
    assert req.pooling_states.mean_pool_count == 8
    assert torch.allclose(
        req.pooling_states.mean_pool_sum, (hs_chunk1 + hs_chunk2).sum(dim=0)
    )

    # Chunk 3: 2 tokens (8 -> 10, optimistic seq_len = 10, finished!)
    hs_chunk3 = all_hidden_states[8:10]
    meta3 = build_pooling_metadata(batch, runner_requests)
    meta3.build_pooling_cursor(
        num_scheduled_tokens_np=np.array([2], dtype=np.int32),
        seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    out3 = pooler.forward(hs_chunk3, meta3)
    assert len(out3) == 1
    assert out3[0] is not None
    assert torch.allclose(out3[0], expected_mean, atol=1e-5)

    # Verify state was cleaned up to prevent memory leaks
    assert req.pooling_states.mean_pool_sum is None
    assert req.pooling_states.mean_pool_count == 0


def test_embedding_pooler_head_safe_forward_with_nones():
    """Verify patched EmbeddingPoolerHead handles None elements from chunked prefill."""
    head = EmbeddingPoolerHead()

    p1 = PoolingParams(task="embed")
    p2 = PoolingParams(task="embed")
    meta = PoolingMetadata(
        prompt_lens=torch.tensor([4, 4], dtype=torch.int32),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[p1, p2],
        pooling_states=[],
    )

    # Item 0 is None (unfinished chunk), Item 1 is completed embedding tensor
    emb2 = torch.randn(4)
    pooled_data = [None, emb2]

    out = head.forward(pooled_data, meta)
    assert len(out) == 2
    assert out[0] is None
    assert isinstance(out[1], torch.Tensor)
    assert torch.allclose(out[1], emb2)


def test_steppool_metadata_tag_resolution():
    """Verify StepPool metadata resolution and tag filtering on prompt_token_ids_cpu."""
    prompt_tokens = [101, 1000, 2000, 102]
    req = MagicMock()
    req.req_id = "step-req"
    req.prompt_token_ids = prompt_tokens
    req.num_prompt_tokens = len(prompt_tokens)
    req.prompt_token_ids_len = len(prompt_tokens)
    # step_tag_id = 1000, which occurs at token index 1
    req.pooling_params = PoolingParams(task="token_embed", step_tag_id=1000)
    req.pooling_states = PoolingStates()

    runner_requests = {"step-req": req}
    batch = MagicMock()
    batch.requests = None
    batch.req_ids = ["step-req"]
    batch.num_reqs = 1

    metadata = build_pooling_metadata(batch, runner_requests)
    metadata.build_pooling_cursor(
        num_scheduled_tokens_np=np.array([4], dtype=np.int32),
        seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    # 1. Dual interface: dict lookup by req_id
    assert metadata.prompt_token_ids_cpu["step-req"] == prompt_tokens
    assert metadata.prompt_token_ids_cpu.get("step-req") == prompt_tokens

    # 2. StepPool uses get_prompt_token_ids_cpu()
    cpu_tokens_list = metadata.get_prompt_token_ids_cpu()
    assert len(cpu_tokens_list) == 1
    assert torch.equal(
        cpu_tokens_list[0], torch.tensor(prompt_tokens, dtype=torch.int32)
    )

    # 3. Test StepPool forward pass executes without D2H sync error
    from vllm.config import set_current_vllm_config

    mock_vllm_config = MagicMock()
    mock_vllm_config.scheduler_config.enable_chunked_prefill = False
    with set_current_vllm_config(mock_vllm_config):
        step_pool = StepPool()
        hidden_states = torch.randn(4, 16, dtype=torch.float32)
        step_out = step_pool.forward(hidden_states, metadata)

    assert len(step_out) == 1
    assert step_out[0].shape == (1, 16)  # Exactly 1 step token extracted
    assert torch.allclose(
        step_out[0][0], hidden_states[1]
    )  # Corresponds to index 1 (token 1000)


def test_last_pool_indices():
    """Verify pooling_cursor.last_token_indices_gpu picks each sequence's final token.

    LastPool relies on these indices.
    """
    # 3 requests with lengths 3, 5, 2 in a packed batch (total 10 tokens)
    prompt_lens = [3, 5, 2]
    total_tokens = sum(prompt_lens)
    dim = 8
    hidden_states = torch.randn(total_tokens, dim, dtype=torch.float32)

    runner_requests = {}
    req_ids = []
    for i, plen in enumerate(prompt_lens):
        rid = f"req-{i}"
        req_ids.append(rid)
        r = MagicMock()
        r.req_id = rid
        r.prompt_token_ids = list(range(plen))
        r.num_prompt_tokens = plen
        r.prompt_token_ids_len = plen
        r.pooling_params = PoolingParams(task="embed")
        r.pooling_states = PoolingStates()
        runner_requests[rid] = r

    batch = MagicMock()
    batch.requests = None
    batch.req_ids = req_ids
    batch.num_reqs = len(req_ids)

    metadata = build_pooling_metadata(batch, runner_requests)
    metadata.build_pooling_cursor(
        num_scheduled_tokens_np=np.array(prompt_lens, dtype=np.int32),
        seq_lens_cpu=torch.tensor(prompt_lens, dtype=torch.int32),
        device=torch.device("cpu"),
    )

    # 1. Verify cursor indices
    cursor = metadata.get_pooling_cursor()
    # Sequence boundaries in packed hidden_states:
    # seq 0: indices [0, 1, 2] -> last index is 2
    # seq 1: indices [3, 4, 5, 6, 7] -> last index is 7
    # seq 2: indices [8, 9] -> last index is 9
    expected_last_indices = torch.tensor([2, 7, 9], dtype=torch.int64)
    assert torch.equal(cursor.last_token_indices_gpu, expected_last_indices)

    # 2. Test LastPool forward pass
    last_pool = LastPool()
    pooled = last_pool.forward(hidden_states, metadata)

    assert pooled.shape == (3, dim)
    assert torch.allclose(pooled[0], hidden_states[2])
    assert torch.allclose(pooled[1], hidden_states[7])
    assert torch.allclose(pooled[2], hidden_states[9])


def test_pooling_metadata_chunk_slice():
    """Slicing metadata to a chunk window keeps dict lookup and shared states."""
    prompt_lens = [3, 5, 2]
    runner_requests = {}
    req_ids = []
    for i, plen in enumerate(prompt_lens):
        rid = f"req-{i}"
        req_ids.append(rid)
        r = MagicMock()
        r.req_id = rid
        r.prompt_token_ids = list(range(100 * i, 100 * i + plen))
        r.num_prompt_tokens = plen
        r.prompt_token_ids_len = plen
        r.pooling_params = PoolingParams(task="embed")
        r.pooling_states = PoolingStates()
        runner_requests[rid] = r

    batch = MagicMock()
    batch.requests = None
    batch.req_ids = req_ids
    batch.num_reqs = len(req_ids)

    # Second chunk of the batch: requests [1, 3)
    sliced = build_pooling_metadata(batch, runner_requests)[1:3]

    assert torch.equal(sliced.prompt_lens, torch.tensor([5, 2], dtype=torch.int32))
    assert isinstance(sliced.prompt_token_ids_cpu, PromptTokenIdsTensor)
    assert sliced.prompt_token_ids_cpu.shape[0] == 2
    assert sliced.prompt_token_ids_cpu["req-1"] == list(range(100, 105))
    assert sliced.pooling_states[0] is runner_requests["req-1"].pooling_states
    assert sliced.pooling_states[1] is runner_requests["req-2"].pooling_states

    # Cursor built from the chunk's own scheduled tokens now lines up.
    sliced.build_pooling_cursor(
        num_scheduled_tokens_np=np.array([5, 2], dtype=np.int32),
        seq_lens_cpu=torch.tensor([5, 2], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    cursor = sliced.get_pooling_cursor()
    assert torch.equal(
        cursor.last_token_indices_gpu, torch.tensor([4, 6], dtype=torch.int64)
    )
