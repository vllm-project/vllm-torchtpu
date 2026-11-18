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
import math

import pytest
import torch
import torch.nn.functional as F
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2e-2
MAX_ABS = 5e-2


def _sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q = query.detach().cpu().float().transpose(1, 2)
    k = key.detach().cpu().float().transpose(1, 2)
    v = value.detach().cpu().float().transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, scale=scale).transpose(1, 2)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "AttentionImpl.forward has no TPU dispatch branch on OmniTpuPlatform "
        "and raises NotImplementedError"
    ),
)
def test_flash_attention_deterministic_mode() -> None:
    """FlashAttention Deterministic Mode

    - TPU path: run FlashAttentionImpl twice under fa_deterministic=True on TPU bf16.
    - Setup check: assert impl.fa_deterministic is True and Attention preserves it.
    - Outcome check: assert bitwise equality across TPU calls and parity vs CPU fp32.
    """
    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        fa_deterministic=True,
        diffusion_attention_config={"default": {"backend": "FLASH_ATTN"}},
    )
    scale = 1.0 / math.sqrt(128)
    with set_current_diffusion_config(od_config):
        impl = FlashAttentionImpl(num_heads=4, head_size=128, softmax_scale=scale)
        assert impl.fa_deterministic is True, "expected impl.fa_deterministic is True"

        torch.manual_seed(0)
        shape = (2, 64, 4, 128)
        q_cpu = torch.randn(shape, dtype=torch.bfloat16)
        k_cpu = torch.randn(shape, dtype=torch.bfloat16)
        v_cpu = torch.randn(shape, dtype=torch.bfloat16)
        ref = _sdpa_reference(q_cpu, k_cpu, v_cpu, scale=scale)

        tpu = torch.device("tpu")
        q_tpu, k_tpu, v_tpu = q_cpu.to(tpu), k_cpu.to(tpu), v_cpu.to(tpu)
        out1 = impl.forward(q_tpu, k_tpu, v_tpu)
        out2 = impl.forward(q_tpu, k_tpu, v_tpu)

        attn = Attention(num_heads=4, head_size=128, causal=False, softmax_scale=scale)
        assert isinstance(attn.attention, FlashAttentionImpl), (
            f"expected FlashAttentionImpl, got {type(attn.attention).__name__}"
        )
        assert attn.attention.fa_deterministic is True, (
            "expected Attention(...) to preserve fa_deterministic=True, "
            f"got {attn.attention.fa_deterministic!r}"
        )

    assert torch.equal(out1.cpu(), out2.cpu()), (
        "repeated deterministic FlashAttention calls produced different outputs"
    )
    assert_parity(
        out1,
        ref,
        name="flash_attention_deterministic_mode",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
