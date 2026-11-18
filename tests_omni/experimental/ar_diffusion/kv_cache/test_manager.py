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
"""TPU component tests for ARDiffusionKVCache and paged_write_attn."""

from __future__ import annotations

import pytest
import torch
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionPagedLayerInputs,
    paged_write_attn,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

AR_KV_MAX_REL_L2 = 3.0e-2
BLOCK = 16
N_HEADS = 4
HEAD_DIM = 64
ACTION_LEN = 3
CROSS_LEN = 8
BRANCH = "positive"


def _dense_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * (
        HEAD_DIM**-0.5
    )
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, value.float())


def _commit_chunks(
    kv: ARDiffusionKVCache,
    state: ARDiffusionKVState,
    k_chunks: list[torch.Tensor],
    v_chunks: list[torch.Tensor],
    q_dummy: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for k_c, v_c in zip(k_chunks, v_chunks, strict=True):
        layer_ctx = state.get_kv_caches(BRANCH, seq_len=BLOCK, commit_current=True)[0]
        layer_ctx.forward_ctx.prepare(device=device, action_len=0, query_len=BLOCK)
        inputs = layer_ctx.to_layer_inputs()
        assert isinstance(inputs, ARDiffusionPagedLayerInputs), (
            f"commit_chunks: expected ARDiffusionPagedLayerInputs, "
            f"got {type(inputs).__name__}"
        )
        paged_write_attn(
            inputs,
            q_dummy[0],
            k_c.to(device=device, dtype=dtype)[0],
            v_c.to(device=device, dtype=dtype)[0],
            None,
            None,
            HEAD_DIM**-0.5,
        )
        state.commit_paged_context(BRANCH)


def test_ar_diffusion_engine_kv_cache() -> None:
    """AR-Diffusion Engine KV Cache

    - TPU path: Preallocate ARDiffusionKVCache (ChunkWindowSpec +
      ARDiffusionKVState) on device=torch.device("tpu") (bfloat16) with
      window_chunks=2, sink_chunks=1, cross-attention KV, and scratch action
      slots, then run multi-chunk paged_write_attn rollouts and session close.
    - Setup check: Assert kv.num_blocks > 0, KV pools and cross-attention
      tensors reside on "tpu", adapter.completed_chunks == 3, and end_request
      restores free blocks.
    - Outcome check: Assert TPU paged_write_attn output over [sink,
      recent_window, current_video, action_scratch] and cross-attention KV
      match CPU float32 dense reference within bf16 relative-L2 tolerance.
    """
    torch.manual_seed(0)
    cfg = ARDiffusionKVConfig(
        enable=True,
        chunk_size=BLOCK,
        window_chunks=2,
        sink_chunks=1,
    )
    tpu_device = torch.device("tpu")
    kv = ARDiffusionKVCache(
        cfg,
        num_layers=1,
        num_kv_heads=N_HEADS,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        block_size=BLOCK,
        max_model_len=256,
        available_bytes=1 << 25,
        kv_branches=(ARDiffusionKVBranchSpec(BRANCH, 0),),
        session_capacity=2,
        frames_per_block=2,
        max_scratch_tokens_per_branch=BLOCK,
        cross_attention_lengths={"text": CROSS_LEN},
        device=tpu_device,
    )
    assert kv.num_blocks > 0, f"Expected num_blocks > 0, got {kv.num_blocks}"
    assert kv.key_cache(0).device.type == "tpu", (
        f"Expected key_cache on tpu, got {kv.key_cache(0).device.type}"
    )
    assert kv.value_cache(0).device.type == "tpu", (
        f"Expected value_cache on tpu, got {kv.value_cache(0).device.type}"
    )
    initial_free = kv.manager.block_pool.get_num_free_blocks()

    cross_k_cpu = torch.randn(1, CROSS_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32)
    cross_v_cpu = torch.randn(1, CROSS_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32)
    kv.populate_cross_attention(
        "s1",
        "text",
        BRANCH,
        [
            (
                cross_k_cpu.to(device=tpu_device, dtype=torch.bfloat16),
                cross_v_cpu.to(device=tpu_device, dtype=torch.bfloat16),
            )
        ],
    )
    cross_read = kv.read_cross_attention_kv("s1", "text", 0, BRANCH)
    assert cross_read["k"].device.type == "tpu", (
        f"Expected cross-attention key on tpu, got {cross_read['k'].device.type}"
    )
    assert_parity(
        cross_read["k"].cpu(),
        cross_k_cpu,
        name="AR-Diffusion cross-attention key TPU bf16 vs CPU fp32",
        max_rel_l2=AR_KV_MAX_REL_L2,
    )

    adapter = kv.begin_request("r-pos")
    state = ARDiffusionKVState(kv, "s1", {BRANCH: adapter}, num_layers=1)

    k_hist = [
        torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=torch.float32) for _ in range(3)
    ]
    v_hist = [
        torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=torch.float32) for _ in range(3)
    ]
    q_dummy = torch.zeros(
        1, BLOCK, N_HEADS, HEAD_DIM, device=tpu_device, dtype=torch.bfloat16
    )

    with torch.no_grad():
        _commit_chunks(kv, state, k_hist, v_hist, q_dummy, tpu_device, torch.bfloat16)
        assert adapter.completed_chunks == 3, (
            f"Expected completed_chunks == 3, got {adapter.completed_chunks}"
        )

        cur_k = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=torch.float32)
        cur_v = torch.randn(1, BLOCK, N_HEADS, HEAD_DIM, dtype=torch.float32)
        act_k = torch.randn(1, ACTION_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32)
        act_v = torch.randn(1, ACTION_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32)
        query = torch.randn(
            1, BLOCK + ACTION_LEN, N_HEADS, HEAD_DIM, dtype=torch.float32
        )

        layer_ctx = state.get_kv_caches(BRANCH, seq_len=BLOCK, commit_current=False)[0]
        layer_ctx.forward_ctx.prepare(
            device=tpu_device, action_len=ACTION_LEN, query_len=query.shape[1]
        )
        inputs = layer_ctx.to_layer_inputs()
        assert isinstance(inputs, ARDiffusionPagedLayerInputs), (
            f"final query: expected ARDiffusionPagedLayerInputs, "
            f"got {type(inputs).__name__}"
        )
        tpu_out = paged_write_attn(
            inputs,
            query.to(device=tpu_device, dtype=torch.bfloat16)[0],
            cur_k.to(device=tpu_device, dtype=torch.bfloat16)[0],
            cur_v.to(device=tpu_device, dtype=torch.bfloat16)[0],
            act_k.to(device=tpu_device, dtype=torch.bfloat16)[0],
            act_v.to(device=tpu_device, dtype=torch.bfloat16)[0],
            HEAD_DIM**-0.5,
        ).unsqueeze(0)

    # window_chunks=2 counts the current chunk, so only the newest history chunk
    # stays next to the sink; chunk 1 has been evicted.
    ref_k = torch.cat([k_hist[0], k_hist[2], cur_k, act_k], dim=1)
    ref_v = torch.cat([v_hist[0], v_hist[2], cur_v, act_v], dim=1)
    ref_out = _dense_attention(query, ref_k, ref_v)

    assert_parity(
        tpu_out.cpu(),
        ref_out,
        name="AR-Diffusion paged_write_attn TPU bf16 vs CPU fp32",
        max_rel_l2=AR_KV_MAX_REL_L2,
    )

    kv.end_request(adapter)
    kv.release_cross_attention("s1")
    restored_free = kv.manager.block_pool.get_num_free_blocks()
    assert restored_free == initial_free, (
        f"Expected free blocks restored to {initial_free}, got {restored_free}"
    )
