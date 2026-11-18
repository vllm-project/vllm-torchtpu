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
"""TPU tests for vllm-omni FP8 quantization configuration and linear execution."""

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
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.fp8_config import DiffusionFp8Config

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

FP8_MAX_REL_L2 = 5.0e-2
FP8_MAX_ABS = 2.0e-1


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


def _load_bf16_weight(
    layer: ReplicatedLinear, w_src_fp32: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Online quantization: load a bf16 weight; the reference stays fp32."""
    layer.weight.data.copy_(w_src_fp32.to(dtype=torch.bfloat16, device=device))
    return w_src_fp32


def _load_fp8_checkpoint(
    layer: ReplicatedLinear, w_src_fp32: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Serialized checkpoint: FP8 weight with unit weight and input scales."""
    w_fp8 = w_src_fp32.to(torch.float8_e4m3fn)
    layer.weight.data.copy_(w_fp8.to(device=device))
    layer.weight_scale.data.fill_(1.0)
    layer.input_scale.data.fill_(1.0)
    return w_fp8.to(torch.float32)


@pytest.mark.parametrize(
    "mode, quant_spec, expected_cls, expected_serialized, expected_scheme, load_weight",
    [
        pytest.param(
            "online_dynamic",
            {
                "method": "fp8",
                "activation_scheme": "dynamic",
                "ignored_layers": ["ignored_proj"],
            },
            DiffusionFp8Config,
            False,
            "dynamic",
            _load_bf16_weight,
            id="online_dynamic",
            marks=pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason=(
                    "choose_scaled_mm_linear_kernel raises ValueError on "
                    "PlatformEnum.OOT; TPU diffusion has no FP8 linear kernel"
                ),
            ),
        ),
        pytest.param(
            "serialized_static",
            {
                "method": "fp8",
                "is_checkpoint_fp8_serialized": True,
                "activation_scheme": "static",
                "ignored_layers": ["ignored_proj"],
            },
            Fp8Config,
            True,
            "static",
            _load_fp8_checkpoint,
            id="serialized_static",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AttributeError,
                reason=(
                    "Fp8LinearMethod.process_weights_after_loading calls "
                    "torch.ops._C.static_scaled_fp8_quant (CUDA custom op "
                    "missing on TPU)"
                ),
            ),
        ),
    ],
)
def test_fp8_quantization(
    dist_ctx: None,
    mode: str,
    quant_spec: dict[str, object],
    expected_cls: type,
    expected_serialized: bool,
    expected_scheme: str,
    load_weight: Callable[[ReplicatedLinear, torch.Tensor, torch.device], torch.Tensor],
) -> None:
    """FP8 Quantization

    - TPU path: resolve online dynamic (DiffusionFp8Config) or serialized static
      (Fp8Config) via OmniDiffusionConfig + create_diffusion_vllm_config, build
      a 128x128 ReplicatedLinear on TPU, process weights, and run forward.
    - Setup check: current_omni_platform.device_type == "tpu", resolved config
      class/flags match the mode, and ignored_layers uses UnquantizedLinearMethod.
    - Outcome check: output is on TPU and matches CPU FP32 F.linear within W8A8
      FP8 limits.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"{mode}: expected device_type 'tpu', got {current_omni_platform.device_type}"
    )

    tpu_device = torch.device("tpu")
    od_config = OmniDiffusionConfig(
        dtype=torch.bfloat16,
        quantization_config=quant_spec,
    )
    quant_config = od_config.quantization_config
    assert isinstance(quant_config, expected_cls), (
        f"{mode}: expected {expected_cls.__name__}, got {type(quant_config).__name__}"
    )
    assert quant_config.is_checkpoint_fp8_serialized is expected_serialized, (
        f"{mode}: is_checkpoint_fp8_serialized="
        f"{quant_config.is_checkpoint_fp8_serialized} != {expected_serialized}"
    )
    assert quant_config.activation_scheme == expected_scheme, (
        f"{mode}: activation_scheme={quant_config.activation_scheme!r} != "
        f"{expected_scheme!r}"
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
        w_src_fp32 = torch.randn(128, 128, dtype=torch.float32) * 0.1
        x_cpu = torch.randn(4, 128, dtype=torch.float32) * 0.5
        w_ref_fp32 = load_weight(layer, w_src_fp32, tpu_device)

        layer.quant_method.process_weights_after_loading(layer)
        out_tpu = layer(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))

    assert out_tpu.device.type == "tpu", (
        f"{mode}: output device {out_tpu.device.type} != 'tpu'"
    )
    ref_out = F.linear(x_cpu, w_ref_fp32)
    assert_parity(
        out_tpu.cpu(),
        ref_out,
        name=f"fp8_{mode}",
        max_rel_l2=FP8_MAX_REL_L2,
        max_abs=FP8_MAX_ABS,
    )
