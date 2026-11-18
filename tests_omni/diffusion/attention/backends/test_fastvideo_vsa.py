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
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    FastVideoVSABackend,
    FastVideoVSAImpl,
)
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.data import OmniDiffusionConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

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
def test_fastvideo_vsa_sparse_attention() -> None:
    """FastVideo VSA sparse attention

    - TPU path: configure FASTVIDEO_VSA (topk=2) and run FastVideoVSAImpl on TPU bf16.
    - Setup check: assert spec.backend == "FASTVIDEO_VSA", impl.topk == 2, backend.
    - Outcome check: compare all-blocks VSA output on TPU bf16 vs CPU fp32 SDPA.
    """
    od_config = OmniDiffusionConfig.from_kwargs(
        model=tiny_diffusion_dir(),
        diffusion_attention_backend="FASTVIDEO_VSA",
        fastvideo_vsa_topk=2,
    )
    backend_cls, spec = get_attn_backend_for_role(
        role="self",
        head_size=128,
        attention_config=od_config.diffusion_attention_config,
    )
    assert spec is not None, (
        "expected AttentionSpec for role='self' with FASTVIDEO_VSA, got None"
    )
    assert spec.backend == "FASTVIDEO_VSA", (
        f"expected AttentionSpec backend 'FASTVIDEO_VSA', got {spec.backend!r}"
    )

    scale = 1.0 / math.sqrt(128)
    impl = FastVideoVSAImpl(
        num_heads=4,
        head_size=128,
        softmax_scale=scale,
        backend_kwargs={"topk": 2, "fallback_on_error": False},
    )
    assert impl.topk == 2, f"expected impl.topk == 2, got {impl.topk}"

    torch.manual_seed(0)
    shape = (1, 512, 4, 128)
    q_cpu = torch.randn(shape, dtype=torch.bfloat16)
    k_cpu = torch.randn(shape, dtype=torch.bfloat16)
    v_cpu = torch.randn(shape, dtype=torch.bfloat16)
    ref = _sdpa_reference(q_cpu, k_cpu, v_cpu, scale=scale)

    tpu = torch.device("tpu")
    attn_metadata = AttentionMetadata(
        extra={
            "vsa_dit_seq_shape": (8, 8, 8),
            "preserve_vsa_all_blocks": True,
        }
    )
    actual = impl.forward(q_cpu.to(tpu), k_cpu.to(tpu), v_cpu.to(tpu), attn_metadata)

    assert backend_cls is FastVideoVSABackend, (
        f"expected FastVideoVSABackend, got {backend_cls.__name__}"
    )
    assert_parity(
        actual,
        ref,
        name="fastvideo_vsa_sparse_attention",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
