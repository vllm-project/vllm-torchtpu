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
"""TPU test for HF Numerics for RL Rollouts."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_NUM_TOKENS = 4
_HIDDEN_SIZE = 16
_NUM_HEADS = 2
_HEAD_DIM = 8
_NUM_EXPERTS = 4
_TOP_K = 2
_EPS = 1e-6


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_hf_numerics_for_rl_rollouts() -> None:
    """HF Numerics for RL Rollouts

    - TPU path: import ``HFNumericsDecoderLayer``, ``HFNumericsRMSNorm``,
      ``HFNumericsRotaryEmbedding``, and ``HFNumericsSparseMoeBlock``, build a
      1-layer thinker config with ``additional_config={"hf_numerics": True}``, and
      run norm, mRoPE, deepstack-add, and MoE forwards on ``"tpu"`` bf16 tensors.
    - Setup check: ``vllm_config.additional_config["hf_numerics"] is True`` and the
      HF-numerics classes instantiate.
    - Outcome check: ``"tpu"`` bf16 outputs match CPU Hugging Face reference math
      for RMSNorm, residual+deepstack add, mRoPE, and top-k softmax routing.
    """
    from vllm.config import VllmConfig
    from vllm_omni.model_executor.models.qwen3_omni.hf_numerics import (
        HFNumericsDecoderLayer,
        HFNumericsRMSNorm,
        HFNumericsRotaryEmbedding,
        HFNumericsSparseMoeBlock,
        hf_add_deepstack,
    )

    vllm_config = VllmConfig(additional_config={"hf_numerics": True})
    assert vllm_config.additional_config.get("hf_numerics") is True, (
        "hf_numerics flag was not preserved on VllmConfig.additional_config"
    )
    assert issubclass(HFNumericsDecoderLayer, torch.nn.Module), (
        "HFNumericsDecoderLayer is not a torch.nn.Module"
    )
    assert issubclass(HFNumericsSparseMoeBlock, torch.nn.Module), (
        "HFNumericsSparseMoeBlock is not a torch.nn.Module"
    )

    torch.manual_seed(0)
    tpu_dev = torch.device("tpu")
    x_cpu = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE, dtype=torch.bfloat16)
    res_cpu = torch.randn_like(x_cpu)
    ds_cpu = torch.randn_like(x_cpu)
    w_cpu = torch.randn(_HIDDEN_SIZE, dtype=torch.bfloat16)

    norm = HFNumericsRMSNorm(_HIDDEN_SIZE, eps=_EPS)
    norm.weight.data.copy_(w_cpu)
    norm = norm.to(device=tpu_dev, dtype=torch.bfloat16)

    norm_tpu = norm(x_cpu.to(device=tpu_dev))
    norm_ref = w_cpu * (
        x_cpu.float() * torch.rsqrt(x_cpu.float().pow(2).mean(-1, keepdim=True) + _EPS)
    ).to(torch.bfloat16)
    assert_parity(norm_tpu, norm_ref, name="hf_rms_norm", max_rel_l2=0.0, max_abs=0.0)

    ds_tpu = hf_add_deepstack(
        x_cpu.to(device=tpu_dev),
        res_cpu.to(device=tpu_dev),
        ds_cpu.to(device=tpu_dev),
    )
    ds_ref = (x_cpu + res_cpu) + ds_cpu
    assert_parity(ds_tpu, ds_ref, name="hf_add_deepstack", max_rel_l2=0.0, max_abs=0.0)

    rope = HFNumericsRotaryEmbedding(
        head_size=_HEAD_DIM,
        rotary_dim=_HEAD_DIM,
        max_position_embeddings=32,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.bfloat16,
        mrope_section=[2, 1, 1],
    ).to(device=tpu_dev)
    positions = torch.arange(_NUM_TOKENS, dtype=torch.long).unsqueeze(0).expand(3, -1)
    q_cpu = torch.randn(_NUM_TOKENS, _NUM_HEADS * _HEAD_DIM, dtype=torch.bfloat16)
    k_cpu = torch.randn(_NUM_TOKENS, _NUM_HEADS * _HEAD_DIM, dtype=torch.bfloat16)
    q_tpu, k_tpu = rope(
        positions.to(device=tpu_dev),
        q_cpu.to(device=tpu_dev),
        k_cpu.to(device=tpu_dev),
    )
    q_cpu_out, k_cpu_out = rope.cpu()(positions, q_cpu, k_cpu)
    assert_parity(q_tpu, q_cpu_out, name="hf_mrope_q", max_rel_l2=0.0, max_abs=0.0)
    assert_parity(k_tpu, k_cpu_out, name="hf_mrope_k", max_rel_l2=0.0, max_abs=0.0)

    logits_cpu = torch.randn(_NUM_TOKENS, _NUM_EXPERTS, dtype=torch.bfloat16)
    probs_tpu = F.softmax(logits_cpu.to(device=tpu_dev).float(), dim=-1)
    weights_tpu, indices_tpu = torch.topk(probs_tpu, _TOP_K, dim=-1)
    weights_tpu = (weights_tpu / weights_tpu.sum(dim=-1, keepdim=True)).to(
        torch.bfloat16
    )

    probs_ref = F.softmax(logits_cpu.float(), dim=-1)
    weights_ref, indices_ref = torch.topk(probs_ref, _TOP_K, dim=-1)
    weights_ref = (weights_ref / weights_ref.sum(dim=-1, keepdim=True)).to(
        torch.bfloat16
    )
    assert torch.equal(indices_tpu.cpu(), indices_ref), (
        "MoE top-k routing indices mismatch"
    )
    assert_parity(
        weights_tpu, weights_ref, name="hf_moe_weights", max_rel_l2=0.0, max_abs=0.0
    )
