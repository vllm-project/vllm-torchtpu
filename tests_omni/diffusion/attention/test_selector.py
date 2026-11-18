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
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.selector import _cached_get_backend_cls
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir
from vllm_torchtpu.omni.attention import (
    TpuFlashAttentionBackend,
    TpuFlashAttentionImpl,
    TpuSDPABackend,
    TpuSDPAImpl,
)

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 1e-2
MAX_ABS = 2e-2


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


@pytest.mark.parametrize(
    (
        "case_id",
        "env_backend",
        "attn_cfg",
        "role",
        "role_category",
        "expected_backend",
        "expected_impl",
    ),
    [
        (
            "platform_default",
            None,
            None,
            "self",
            None,
            TpuFlashAttentionBackend,
            TpuFlashAttentionImpl,
        ),
        (
            "env_torch_sdpa",
            "TORCH_SDPA",
            None,
            "self",
            None,
            TpuSDPABackend,
            TpuSDPAImpl,
        ),
        (
            "per_role_self",
            None,
            {
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"cross": {"backend": "TORCH_SDPA"}},
            },
            "self",
            None,
            TpuFlashAttentionBackend,
            TpuFlashAttentionImpl,
        ),
        (
            "per_role_cross_category",
            None,
            {
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"cross": {"backend": "TORCH_SDPA"}},
            },
            "ltx2.audio_to_video",
            "cross",
            TpuSDPABackend,
            TpuSDPAImpl,
        ),
    ],
    ids=[
        "platform_default",
        "env_torch_sdpa",
        "per_role_self",
        "per_role_cross_category",
    ],
)
def test_diffusion_attention_backend_selection(
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    env_backend: str | None,
    attn_cfg: dict | None,
    role: str,
    role_category: str | None,
    expected_backend: type,
    expected_impl: type,
) -> None:
    """Diffusion Attention Backend Selection

    - TPU path: construct Attention under default/env/role configs on TPU bf16.
    - Setup check: clear _cached_get_backend_cls; assert backend and impl types.
    - Outcome check: compare TPU bf16 output against CPU fp32 SDPA via assert_parity.
    """
    _cached_get_backend_cls.cache_clear()
    if env_backend is not None:
        monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", env_backend)
    else:
        monkeypatch.delenv("DIFFUSION_ATTENTION_BACKEND", raising=False)

    od_config = OmniDiffusionConfig.from_kwargs(
        model=tiny_diffusion_dir(),
        diffusion_attention_config=attn_cfg,
    )
    with set_current_diffusion_config(od_config):
        attn = Attention(
            num_heads=4,
            head_size=128,
            causal=False,
            softmax_scale=1.0 / math.sqrt(128),
            role=role,
            role_category=role_category,
        )

    assert attn.attn_backend is expected_backend, (
        f"{case_id}: expected {expected_backend.__name__}, got {attn.attn_backend}"
    )
    assert type(attn.attention) is expected_impl, (
        f"{case_id}: expected {expected_impl.__name__}, "
        f"got {type(attn.attention).__name__}"
    )

    torch.manual_seed(0)
    shape = (2, 64, 4, 128)
    q_cpu = torch.randn(shape, dtype=torch.bfloat16)
    k_cpu = torch.randn(shape, dtype=torch.bfloat16)
    v_cpu = torch.randn(shape, dtype=torch.bfloat16)
    ref = _sdpa_reference(q_cpu, k_cpu, v_cpu, scale=1.0 / math.sqrt(128))

    tpu = torch.device("tpu")
    with torch.no_grad():
        actual = attn(q_cpu.to(tpu), k_cpu.to(tpu), v_cpu.to(tpu))
    assert_parity(
        actual,
        ref,
        name=f"diffusion_attention_backend_selection_{case_id}",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
