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
"""TPU tests for vllm-omni BitsAndBytes W4 quantization config and linear execution."""

from collections.abc import Generator

import pytest
import torch
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

BNB_W4_MAX_REL_L2 = 1.5e-1
BNB_W4_MAX_ABS = 6.0e-1


@pytest.fixture(scope="module")
def dist_ctx() -> Generator[None, None, None]:
    """Single-rank gloo world and model-parallel groups for this module."""
    owns_world = not torch.distributed.is_initialized()
    owns_model_parallel = False
    if owns_world:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            backend="gloo",
        )
    try:
        owns_model_parallel = not model_parallel_is_initialized()
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)
        yield
    finally:
        if owns_model_parallel:
            destroy_model_parallel()
        if owns_world:
            destroy_distributed_environment()


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "DiffusionBitsAndBytesConfig.get_quant_method raises NotImplementedError "
        "off CUDA"
    ),
)
@pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
def test_bitsandbytes_w4_quantization(dist_ctx: None, quant_type: str) -> None:
    """BitsAndBytes W4 Quantization

    - TPU path: resolve DiffusionBitsAndBytesConfig (nf4 / fp4 with
      compress_statistics=True) via OmniDiffusionConfig, construct a 128x128
      ReplicatedLinear on TPU, process BF16 weights, and run forward.
    - Setup check: current_omni_platform.device_type == "tpu",
      DiffusionBitsAndBytesConfig quant_type matches, and ignored_layers
      resolves UnquantizedLinearMethod.
    - Outcome check: output is on TPU and matches CPU FP32 F.linear within 4-bit
      NF4/FP4 limits.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"{quant_type}: expected device_type 'tpu', got "
        f"{current_omni_platform.device_type}"
    )

    tpu_device = torch.device("tpu")
    od_config = OmniDiffusionConfig(
        dtype=torch.bfloat16,
        quantization_config={
            "method": "bitsandbytes",
            "quant_type": quant_type,
            "compress_statistics": True,
            "ignored_layers": ["ignored_proj"],
        },
    )
    quant_config = od_config.quantization_config
    assert isinstance(quant_config, DiffusionBitsAndBytesConfig), (
        f"{quant_type}: expected DiffusionBitsAndBytesConfig, got "
        f"{type(quant_config).__name__}"
    )
    assert quant_config.quant_type == quant_type, (
        f"{quant_type}: quant_type={quant_config.quant_type!r} != {quant_type!r}"
    )
    assert quant_config.compress_statistics is True, (
        f"{quant_type}: compress_statistics={quant_config.compress_statistics} != True"
    )

    vllm_config = create_diffusion_vllm_config(tpu_device, od_config)
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config):
        ignored_layer = ReplicatedLinear(
            128,
            128,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="ignored_proj",
            return_bias=False,
            disable_tp=True,
        )
        assert isinstance(ignored_layer.quant_method, UnquantizedLinearMethod), (
            f"{quant_type}: ignored layer resolved "
            f"{type(ignored_layer.quant_method).__name__}"
        )

        layer = ReplicatedLinear(
            128,
            128,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="blocks.0.proj",
            return_bias=False,
            disable_tp=True,
        ).to(tpu_device)

        torch.manual_seed(0)
        w_ref_fp32 = torch.randn(128, 128, dtype=torch.float32) * 0.1
        x_cpu = torch.randn(4, 128, dtype=torch.float32) * 0.5
        layer.weight.data.copy_(w_ref_fp32.to(dtype=torch.bfloat16, device=tpu_device))

        layer.quant_method.process_weights_after_loading(layer)
        out_tpu = layer(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))

    assert out_tpu.device.type == "tpu", (
        f"{quant_type}: output device {out_tpu.device.type} != 'tpu'"
    )
    ref_out = F.linear(x_cpu, w_ref_fp32)
    assert_parity(
        out_tpu.cpu(),
        ref_out,
        name=f"bitsandbytes_{quant_type}",
        max_rel_l2=BNB_W4_MAX_REL_L2,
        max_abs=BNB_W4_MAX_ABS,
    )
