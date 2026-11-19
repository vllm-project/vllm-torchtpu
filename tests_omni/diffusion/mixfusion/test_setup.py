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
"""TPU tests for MixFusion mixed-resolution diffusion step batching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

NUM_HEADS = 4
HEAD_DIM = 32
MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_mixfusion_mixed_resolution_step_batching() -> None:
    """MixFusion (Mixed-Resolution Step Batching).

    - TPU path: import ``validate_mixfusion_config`` and ``fits_chunk_budget``,
      validate MixFusion config admission and GCD chunk budgeting for
      mixed-resolution requests, and run ragged block-diagonal packed attention
      on ``torch.device("tpu")``.
    - Setup check: ``current_omni_platform.device_type == "tpu"``,
      ``validate_mixfusion_config`` requires step execution on supported
      pipelines, and ``fits_chunk_budget`` enforces ``min_chunk_tokens``.
    - Outcome check: split outputs of the packed mixed-resolution TPU forward
      match separate per-resolution CPU FP32 attention via ``assert_parity``.
    """
    from vllm_omni.diffusion.mixfusion.scheduling import (
        MixFusionRequestInfo,
        fits_chunk_budget,
    )
    from vllm_omni.diffusion.mixfusion.setup import validate_mixfusion_config

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    enabled_cfg = SimpleNamespace(
        enable_mixfusion=True,
        model_class_name="QwenImagePipeline",
        diffusion_kv_mode=None,
    )
    assert validate_mixfusion_config(enabled_cfg, supported=lambda _: True) is True, (
        "Expected MixFusion validation to require step execution"
    )
    with pytest.raises(ValueError, match="does not support MixFusion"):
        validate_mixfusion_config(enabled_cfg, supported=lambda _: False)

    compatible = [
        MixFusionRequestInfo(
            num_images=1,
            image_tokens=512,
            min_chunk_tokens=256,
            max_chunks=128,
        ),
        MixFusionRequestInfo(
            num_images=1,
            image_tokens=1024,
            min_chunk_tokens=256,
            max_chunks=128,
        ),
    ]
    assert fits_chunk_budget(compatible) is True, (
        "Expected 512 + 1024 token batch (GCD=512) to fit MixFusion budget"
    )
    incompatible = [
        MixFusionRequestInfo(
            num_images=1,
            image_tokens=300,
            min_chunk_tokens=256,
            max_chunks=128,
        ),
        MixFusionRequestInfo(
            num_images=1,
            image_tokens=350,
            min_chunk_tokens=256,
            max_chunks=128,
        ),
    ]
    assert fits_chunk_budget(incompatible) is False, (
        "Expected 300 + 350 token batch (GCD=50 < 256) to fail MixFusion budget"
    )

    torch.manual_seed(0)
    tpu_device = torch.device("tpu")
    len_a, len_b = 16, 32
    q_a = torch.randn(1, NUM_HEADS, len_a, HEAD_DIM, dtype=torch.float32)
    k_a = torch.randn(1, NUM_HEADS, len_a, HEAD_DIM, dtype=torch.float32)
    v_a = torch.randn(1, NUM_HEADS, len_a, HEAD_DIM, dtype=torch.float32)
    q_b = torch.randn(1, NUM_HEADS, len_b, HEAD_DIM, dtype=torch.float32)
    k_b = torch.randn(1, NUM_HEADS, len_b, HEAD_DIM, dtype=torch.float32)
    v_b = torch.randn(1, NUM_HEADS, len_b, HEAD_DIM, dtype=torch.float32)

    ref_a = F.scaled_dot_product_attention(q_a, k_a, v_a)
    ref_b = F.scaled_dot_product_attention(q_b, k_b, v_b)

    total = len_a + len_b
    mask = torch.zeros((total, total), dtype=torch.bool)
    mask[:len_a, :len_a] = True
    mask[len_a:, len_a:] = True

    q_packed = torch.cat([q_a, q_b], dim=2).to(dtype=torch.bfloat16, device=tpu_device)
    k_packed = torch.cat([k_a, k_b], dim=2).to(dtype=torch.bfloat16, device=tpu_device)
    v_packed = torch.cat([v_a, v_b], dim=2).to(dtype=torch.bfloat16, device=tpu_device)
    out_packed = F.scaled_dot_product_attention(
        q_packed, k_packed, v_packed, attn_mask=mask.to(tpu_device)
    )

    assert_parity(
        out_packed[:, :, :len_a, :],
        ref_a,
        name="mixfusion_packed_req_a",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
    assert_parity(
        out_packed[:, :, len_a:, :],
        ref_b,
        name="mixfusion_packed_req_b",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
