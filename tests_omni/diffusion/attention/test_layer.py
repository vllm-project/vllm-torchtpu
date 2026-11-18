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
"""TPU tests for diffusion attention runtime KV-cache quantization."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 5.0e-2
MAX_ABS = 1.0e-1


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=("TpuFlashAttentionBackend does not support kv_cache_dtype='fp8' on TPU"),
)
def test_diffusion_attention_runtime_quantization() -> None:
    """Diffusion Attention Runtime Quantization.

    - TPU path: Activate ``OmniDiffusionConfig`` with
      ``diffusion_kv_cache_dtype="fp8"``, ``diffusion_kv_cache_skip_steps="0"``,
      and ``diffusion_kv_cache_skip_layers="1"`` under
      ``set_current_diffusion_config``, then instantiate and run
      ``Attention(num_heads=4, head_size=128)`` on ``"tpu"`` bfloat16 tensors.
    - Setup check: Assert ``current_omni_platform.device_name == "tpu"`` and
      ``od_config.diffusion_kv_cache_dtype == "fp8"`` before ``Attention()``,
      then verify ``_kv_cache_dtype == "fp8"`` and skip sets ``{0}`` / ``{1}``.
    - Outcome check: Run ``attn(q_tpu, k_tpu, v_tpu)`` on ``"tpu"`` and assert
      relative-L2 / max-abs parity against CPU float32 SDPA reference.
    """
    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        dtype=torch.bfloat16,
        diffusion_kv_cache_dtype="fp8",
        diffusion_kv_cache_skip_steps="0",
        diffusion_kv_cache_skip_layers="1",
    )
    assert current_omni_platform.device_name == "tpu", (
        f"Expected device_name == 'tpu', got {current_omni_platform.device_name!r}"
    )
    assert od_config.diffusion_kv_cache_dtype == "fp8", (
        f"Expected diffusion_kv_cache_dtype == 'fp8', "
        f"got {od_config.diffusion_kv_cache_dtype!r}"
    )

    scale = 1.0 / math.sqrt(128)
    with set_current_diffusion_config(od_config):
        attn = Attention(
            num_heads=4,
            head_size=128,
            causal=False,
            softmax_scale=scale,
            prefix="blocks.0.attn",
        ).to("tpu")

    assert attn._kv_cache_dtype == "fp8", (
        f"Expected _kv_cache_dtype == 'fp8', got {attn._kv_cache_dtype!r}"
    )
    assert attn._kv_cache_skip_steps == {0}, (
        f"Expected _kv_cache_skip_steps == {{0}}, got {attn._kv_cache_skip_steps}"
    )
    assert attn._kv_cache_skip_layers == {1}, (
        f"Expected _kv_cache_skip_layers == {{1}}, got {attn._kv_cache_skip_layers}"
    )

    torch.manual_seed(0)
    q_cpu = torch.randn(1, 64, 4, 128, dtype=torch.bfloat16)
    k_cpu = torch.randn(1, 64, 4, 128, dtype=torch.bfloat16)
    v_cpu = torch.randn(1, 64, 4, 128, dtype=torch.bfloat16)

    out_tpu = attn(q_cpu.to("tpu"), k_cpu.to("tpu"), v_cpu.to("tpu"))
    ref_out = (
        F.scaled_dot_product_attention(
            q_cpu.float().transpose(1, 2),
            k_cpu.float().transpose(1, 2),
            v_cpu.float().transpose(1, 2),
            scale=scale,
        )
        .transpose(1, 2)
        .contiguous()
    )
    assert_parity(
        out_tpu,
        ref_out,
        name="diffusion_attention_kv_fp8",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
