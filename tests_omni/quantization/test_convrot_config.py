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
"""TPU tests for vllm-omni ConvRot quantization configuration and linear execution."""

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

CONVROT_MAX_REL_L2 = {
    "convrot_int8": 3.0e-2,
    "convrot_int4": 1.2e-1,
}
CONVROT_MAX_ABS = {
    "convrot_int8": 2.0e-1,
    "convrot_int4": 8.0e-1,
}


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="vllm_omni.quantization.convrot_config is not in vllm-omni 0.32.0rc1",
)
@pytest.mark.parametrize(
    "method, rotation_group_size, int8_layers",
    [
        ("convrot_int8", 256, None),
        ("convrot_int4", 64, ["blocks.0.int8_proj"]),
    ],
    ids=["convrot_int8", "convrot_int4"],
)
def test_convrot_quantization(
    method: str,
    rotation_group_size: int,
    int8_layers: list[str] | None,
) -> None:
    """ConvRot Quantization

    - TPU path: resolve DiffusionConvRotConfig (convrot_int8 / convrot_int4) via
      OmniDiffusionConfig, construct a 256x128 ReplicatedLinear on TPU, load
      outlier-heavy BF16 weights, process weights, and run forward.
    - Setup check: current_omni_platform.device_type == "tpu", resolved config
      matches method/rotation_group_size, and ignored_layers uses
      UnquantizedLinearMethod.
    - Outcome check: output is on TPU and matches CPU FP32 F.linear within
      per-mode W8A8/W4A4 limits.
    """
    from vllm_omni.quantization.convrot_config import DiffusionConvRotConfig

    assert current_omni_platform.device_type == "tpu", (
        f"{method}: expected device_type 'tpu', got {current_omni_platform.device_type}"
    )

    quant_spec: dict[str, object] = {
        "method": method,
        "rotation_group_size": rotation_group_size,
        "ignored_layers": ["ignored_proj"],
    }
    if int8_layers is not None:
        quant_spec["int8_layers"] = int8_layers

    tpu_device = torch.device("tpu")
    od_config = OmniDiffusionConfig(
        dtype=torch.bfloat16,
        quantization_config=quant_spec,
    )
    quant_config = od_config.quantization_config
    assert isinstance(quant_config, DiffusionConvRotConfig), (
        f"{method}: expected DiffusionConvRotConfig, got {type(quant_config).__name__}"
    )
    assert quant_config.get_name() == method, (
        f"{method}: get_name()={quant_config.get_name()!r} != {method!r}"
    )
    assert quant_config.rotation_group_size == rotation_group_size, (
        f"{method}: rotation_group_size={quant_config.rotation_group_size} != "
        f"{rotation_group_size}"
    )

    vllm_config = create_diffusion_vllm_config(tpu_device, od_config)
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        ignored_layer = ReplicatedLinear(
            256,
            128,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="ignored_proj",
            return_bias=False,
            disable_tp=True,
        )
        assert isinstance(ignored_layer.quant_method, UnquantizedLinearMethod), (
            f"{method}: ignored layer resolved "
            f"{type(ignored_layer.quant_method).__name__}"
        )

        layer = ReplicatedLinear(
            256,
            128,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="blocks.0.proj",
            return_bias=False,
            disable_tp=True,
        ).to(tpu_device)

        torch.manual_seed(0)
        w_ref_fp32 = torch.randn(128, 256, dtype=torch.float32) * 0.05
        w_ref_fp32[:, 0] *= 20.0
        x_cpu = torch.randn(4, 256, dtype=torch.float32) * 0.5
        layer.weight.data.copy_(w_ref_fp32.to(dtype=torch.bfloat16, device=tpu_device))

        layer.quant_method.process_weights_after_loading(layer)
        out_tpu = layer(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))

    assert out_tpu.device.type == "tpu", (
        f"{method}: output device {out_tpu.device.type} != 'tpu'"
    )
    ref_out = F.linear(x_cpu, w_ref_fp32)
    assert_parity(
        out_tpu.cpu(),
        ref_out,
        name=method,
        max_rel_l2=CONVROT_MAX_REL_L2[method],
        max_abs=CONVROT_MAX_ABS[method],
    )
