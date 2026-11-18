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
"""TPU tests for vllm-omni INT8 quantization configuration and linear execution."""

from collections.abc import Callable, Generator

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
from vllm_omni.quantization.int8_config import DiffusionInt8Config

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

INT8_MAX_REL_L2 = 3.0e-2
INT8_MAX_ABS = 1.5e-1


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


def _load_random_bf16_weight(
    layer: ReplicatedLinear, device: torch.device
) -> torch.Tensor:
    """Online quantization: load a random bf16 weight; the reference stays fp32."""
    w_ref_fp32 = torch.randn(128, 128, dtype=torch.float32) * 0.1
    layer.weight.data.copy_(w_ref_fp32.to(dtype=torch.bfloat16, device=device))
    return w_ref_fp32


def _load_random_int8_checkpoint(
    layer: ReplicatedLinear, device: torch.device
) -> torch.Tensor:
    """Serialized checkpoint: random int8 weight with a per-row 1/64 scale."""
    w_int8 = torch.randint(-64, 64, (128, 128), dtype=torch.int8)
    w_scale = torch.full((128, 1), 1.0 / 64.0, dtype=torch.float32)
    layer.weight.data.copy_(w_int8.to(device=device))
    layer.weight_scale.data.copy_(w_scale.to(device=device))
    return w_int8.to(torch.float32) * w_scale


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "DiffusionInt8Config.get_quant_method raises NotImplementedError off CUDA/NPU"
    ),
)
@pytest.mark.parametrize(
    "mode, is_serialized, load_weight",
    [
        ("online", False, _load_random_bf16_weight),
        ("offline", True, _load_random_int8_checkpoint),
    ],
    ids=["online", "offline"],
)
def test_int8_quantization(
    dist_ctx: None,
    mode: str,
    is_serialized: bool,
    load_weight: Callable[[ReplicatedLinear, torch.device], torch.Tensor],
) -> None:
    """Int8 Quantization

    - TPU path: resolve DiffusionInt8Config via OmniDiffusionConfig in online
      or offline W8A8 mode, construct a 128x128 ReplicatedLinear on TPU, process
      weights, and execute forward.
    - Setup check: current_omni_platform.device_type == "tpu", DiffusionInt8Config
      flags match the mode, and ignored_layers resolves UnquantizedLinearMethod.
    - Outcome check: output is on TPU and matches CPU FP32 F.linear within W8A8
      INT8 limits.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"{mode}: expected device_type 'tpu', got {current_omni_platform.device_type}"
    )

    tpu_device = torch.device("tpu")
    od_config = OmniDiffusionConfig(
        dtype=torch.bfloat16,
        quantization_config={
            "method": "int8",
            "is_checkpoint_int8_serialized": is_serialized,
            "ignored_layers": ["ignored_proj"],
        },
    )
    quant_config = od_config.quantization_config
    assert isinstance(quant_config, DiffusionInt8Config), (
        f"{mode}: expected DiffusionInt8Config, got {type(quant_config).__name__}"
    )
    assert quant_config.is_checkpoint_int8_serialized is is_serialized, (
        f"{mode}: is_checkpoint_int8_serialized="
        f"{quant_config.is_checkpoint_int8_serialized} != {is_serialized}"
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
            f"{mode}: ignored layer resolved "
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
        x_cpu = torch.randn(4, 128, dtype=torch.float32) * 0.5
        w_ref_fp32 = load_weight(layer, tpu_device)

        layer.quant_method.process_weights_after_loading(layer)
        out_tpu = layer(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))

    assert out_tpu.device.type == "tpu", (
        f"{mode}: output device {out_tpu.device.type} != 'tpu'"
    )
    ref_out = F.linear(x_cpu, w_ref_fp32)
    assert_parity(
        out_tpu.cpu(),
        ref_out,
        name=f"int8_{mode}",
        max_rel_l2=INT8_MAX_REL_L2,
        max_abs=INT8_MAX_ABS,
    )
