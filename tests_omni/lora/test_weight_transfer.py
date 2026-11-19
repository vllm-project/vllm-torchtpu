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
"""TPU tests for engine-side LoRA merging (diffusion and AR)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 2.0e-2
MAX_ABS = 5.0e-2


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_engine_side_lora_merging() -> None:
    """Engine-Side LoRA Merging (Diffusion & AR).

    - TPU path: import ``MergedLoRAIPCWeightTransferEngine``,
      ``ARMergedLoRAManager``, and ``MergedLinearWeights``; merge a rank-4
      adapter directly into a dense bf16 linear weight on ``torch.device("tpu")``,
      run a TPU forward pass, and restore pristine base weights via ``close()``.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``MergedLoRAIPCWeightTransferEngine.update_info_cls`` is registered.
    - Outcome check: merged TPU forward matches CPU FP32 ``x @ (W + B @ A).T``
      via ``assert_parity``, and ``close()`` restores bit-identical base weights.
    """
    from vllm_omni.lora.merged_manager import ARMergedLoRAManager
    from vllm_omni.lora.merged_weights import MergedLinearWeights
    from vllm_omni.lora.weight_transfer import (
        MergedLoRAIPCUpdateInfo,
        MergedLoRAIPCWeightTransferEngine,
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )
    assert (
        MergedLoRAIPCWeightTransferEngine.update_info_cls is MergedLoRAIPCUpdateInfo
    ), "Unexpected update_info_cls on MergedLoRAIPCWeightTransferEngine"
    assert ARMergedLoRAManager is not None, "ARMergedLoRAManager unavailable"

    torch.manual_seed(0)
    tpu_device = torch.device("tpu")
    base_cpu = torch.randn(16, 16, dtype=torch.bfloat16) * 0.1
    lora_a_cpu = torch.randn(4, 16, dtype=torch.float32) * 0.1
    lora_b_cpu = torch.randn(16, 4, dtype=torch.float32) * 0.1

    linear = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(base_cpu)
    linear = linear.to(tpu_device)

    binding = SimpleNamespace(
        base_layer=linear,
        _diffusion_lora_active_slices=(True,),
        output_slices=(16,),
        _merged_lora_scales=(1.0,),
        lora_a_stacked=(lora_a_cpu.view(1, 1, 4, 16).to(tpu_device),),
        lora_b_stacked=(lora_b_cpu.view(1, 1, 16, 4).to(tpu_device),),
    )
    bank = MergedLinearWeights()
    prepared = bank.prepare({"proj": binding})
    bank.apply(prepared)

    x_cpu = torch.randn(2, 16, dtype=torch.float32)
    ref_merged = x_cpu @ (base_cpu.float() + lora_b_cpu @ lora_a_cpu).T
    tpu_out = linear(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))
    assert_parity(
        tpu_out,
        ref_merged,
        name="engine_side_lora_merged_forward",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )

    bank.close()
    assert torch.equal(linear.weight.cpu(), base_cpu), (
        "MergedLinearWeights.close() did not restore exact base weights"
    )
