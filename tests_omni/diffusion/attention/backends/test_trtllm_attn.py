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
from vllm_omni.diffusion.attention.backends.trtllm_attn import (
    TrtllmAttentionBackend,
    TrtllmAttentionImpl,
)
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    set_forward_context,
    set_forward_context_denoise_timestep,
)

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
@pytest.mark.parametrize(
    ("case_id", "skip_softmax_spec", "layer_calibration"),
    [
        ("calibration_free", {"threshold": 0.0}, (None, None)),
        (
            "calibrated_gated",
            {"target_sparsity": 0.2, "disabled_until_timestep": 0.8},
            (1.0, 1.0),
        ),
    ],
    ids=["calibration_free", "calibrated_gated"],
)
def test_skip_softmax_trtllm_attn(
    case_id: str,
    skip_softmax_spec: dict[str, float],
    layer_calibration: tuple[float | None, float | None],
) -> None:
    """Skip-softmax block sparsity (TRTLLM_ATTN)

    - TPU path: configure skip_softmax and run TrtllmAttentionImpl on TPU bf16 Q/K/V.
    - Setup check: assert impl.skip.enabled, _resolve_skip_factor(64), and backend.
    - Outcome check: compare TPU bf16 output against CPU fp32 SDPA via assert_parity.
    """
    od_config = OmniDiffusionConfig.from_kwargs(
        model=tiny_diffusion_dir(),
        diffusion_attention_config={
            "default": {"backend": "TRTLLM_ATTN", "skip_softmax": skip_softmax_spec}
        },
    )
    default_spec = od_config.diffusion_attention_config.default
    assert default_spec is not None, (
        f"{case_id}: expected non-None diffusion_attention_config.default, got None"
    )
    scale = 1.0 / math.sqrt(128)
    impl = TrtllmAttentionImpl(
        num_heads=4,
        head_size=128,
        softmax_scale=scale,
        backend_kwargs=default_spec.backend_kwargs(),
    )
    # (None, None) keeps SkipSoftmaxConfig's uncalibrated defaults.
    impl.set_layer_calibration(*layer_calibration)

    assert impl.skip.enabled is True, f"{case_id}: expected impl.skip.enabled is True"

    torch.manual_seed(0)
    shape = (2, 64, 4, 128)
    q_cpu = torch.randn(shape, dtype=torch.bfloat16)
    k_cpu = torch.randn(shape, dtype=torch.bfloat16)
    v_cpu = torch.randn(shape, dtype=torch.bfloat16)
    ref = _sdpa_reference(q_cpu, k_cpu, v_cpu, scale=scale)

    tpu = torch.device("tpu")
    with set_forward_context(omni_diffusion_config=od_config):
        set_forward_context_denoise_timestep(0.5)
        factor = impl._resolve_skip_factor(64)
        assert factor is not None, (
            f"{case_id}: expected non-None _resolve_skip_factor(64)"
        )
        actual = impl.forward(q_cpu.to(tpu), k_cpu.to(tpu), v_cpu.to(tpu))

    backend_cls, _ = get_attn_backend_for_role(
        role="self",
        head_size=128,
        attention_config=od_config.diffusion_attention_config,
    )
    assert backend_cls is TrtllmAttentionBackend, (
        f"{case_id}: expected TrtllmAttentionBackend, got {backend_cls.__name__}"
    )
    assert_parity(
        actual,
        ref,
        name=f"skip_softmax_trtllm_attn_{case_id}",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
