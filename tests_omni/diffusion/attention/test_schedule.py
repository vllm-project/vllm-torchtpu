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
"""TPU tests for diffusion per-step and sigma-window attention schedules."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

NUM_HEADS = 4
HEAD_SIZE = 64
SEQ_LEN = 32
MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_diffusion_attention_schedules(omni_ctx: None) -> None:
    """Diffusion Attention Schedules (Per-Step & Sigma-Window).

    - TPU path: import ``schedule`` helpers, resolve per-step ranges and
      sigma windows, reject combined step + sigma schedules, and execute the
      selected attention backend on ``torch.device("tpu")``.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and TPU
      backend lookup resolves ``TpuSDPABackend`` for the scheduled profile.
    - Outcome check: step and sigma selectors return the expected profile name
      inside ranges and ``None`` in gaps, mutual exclusion raises ``ValueError``,
      and TPU attention output matches CPU FP32 SDPA via ``assert_parity``.
    """
    from vllm_omni.diffusion.attention.schedule import (
        checked_sigma,
        parse_attention_schedule,
        parse_attention_sigma_schedule,
        select_attention_profile,
        select_attention_profile_by_sigma,
        validate_attention_schedule,
        validate_attention_sigma_schedule,
        validate_request_attention_schedules,
    )
    from vllm_omni.diffusion.data import AttentionScheduleConfig

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    step_sched = parse_attention_schedule([{"start": 0, "end": 2, "profile": "sdpa"}])
    assert step_sched is not None, "parse_attention_schedule returned None"
    validate_attention_schedule(step_sched, profiles=("sdpa",), total_steps=4)
    assert select_attention_profile(step_sched, 0, total_steps=4) == "sdpa", (
        "Expected 'sdpa' profile at step 0"
    )
    assert select_attention_profile(step_sched, 2, total_steps=4) is None, (
        "Expected None (base backend) in step gap at step 2"
    )

    sigma_sched = parse_attention_sigma_schedule(
        [{"low": 0.5, "high": 1.0, "profile": "sdpa"}]
    )
    assert sigma_sched is not None, "parse_attention_sigma_schedule returned None"
    validate_attention_sigma_schedule(sigma_sched, profiles=("sdpa",))
    assert (
        select_attention_profile_by_sigma(sigma_sched, checked_sigma(0.75)) == "sdpa"
    ), "Expected 'sdpa' profile at sigma=0.75"
    assert select_attention_profile_by_sigma(sigma_sched, checked_sigma(0.2)) is None, (
        "Expected None (base backend) at sigma=0.2"
    )

    service_cfg = AttentionScheduleConfig(
        profiles={"sdpa": {"backend": "TORCH_SDPA"}},
        default=step_sched,
    )
    conflicting = OmniDiffusionSamplingParams(
        attention_schedule=list(step_sched),
        attention_sigma_schedule=list(sigma_sched),
    )
    with pytest.raises(ValueError, match="cannot combine"):
        validate_request_attention_schedules(conflicting, service_cfg)

    backend_cls = current_omni_platform.get_diffusion_attn_backend_cls(
        "TORCH_SDPA", head_size=HEAD_SIZE
    )
    assert backend_cls.endswith("TpuSDPABackend"), (
        f"Expected TpuSDPABackend for TORCH_SDPA profile, got {backend_cls!r}"
    )

    torch.manual_seed(0)
    tpu_device = torch.device("tpu")
    q_cpu = torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)
    k_cpu = torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)
    v_cpu = torch.randn(1, SEQ_LEN, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)
    ref = F.scaled_dot_product_attention(
        q_cpu.transpose(1, 2),
        k_cpu.transpose(1, 2),
        v_cpu.transpose(1, 2),
    ).transpose(1, 2)

    attn = Attention(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        softmax_scale=HEAD_SIZE**-0.5,
        causal=False,
    ).to(tpu_device)
    tpu_out = attn(
        q_cpu.to(dtype=torch.bfloat16, device=tpu_device),
        k_cpu.to(dtype=torch.bfloat16, device=tpu_device),
        v_cpu.to(dtype=torch.bfloat16, device=tpu_device),
    )
    assert_parity(
        tpu_out,
        ref,
        name="diffusion_attention_schedule_tpu",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
