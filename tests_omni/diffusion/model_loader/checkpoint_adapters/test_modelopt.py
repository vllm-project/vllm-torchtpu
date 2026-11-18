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
"""TPU tests for vllm-omni ModelOpt quantization and checkpoint adapters."""

from collections.abc import Generator
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.utils.network_utils import get_open_port
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.checkpoint_adapters import (
    ModelOptFp8CheckpointAdapter,
    ModelOptMixedPrecisionCheckpointAdapter,
    ModelOptNvFp4CheckpointAdapter,
    get_checkpoint_adapter,
)
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MODELOPT_MAX_REL_L2 = {
    "FP8": 5.0e-2,
    "NVFP4": 1.5e-1,
    "MIXED_PRECISION": 5.0e-2,
}
MODELOPT_MAX_ABS = {
    "FP8": 2.0e-1,
    "NVFP4": 6.0e-1,
    "MIXED_PRECISION": 2.0e-1,
}


class _AdapterProbeModel(nn.Module):
    """Minimal module exposing BF16 ignored and FP8 quantized target parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.ignored_proj = nn.Module()
        self.ignored_proj.weight = nn.Parameter(
            torch.empty(128, 128, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].proj = nn.Module()
        self.blocks[0].proj.weight = nn.Parameter(
            torch.empty(128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )


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
    raises=ValueError,
    reason=(
        "select_linear_kernel raises ValueError on PlatformEnum.OOT; "
        "ModelOpt kernels require CUDA SM89+/SM100+"
    ),
)
@pytest.mark.parametrize(
    "quant_algo, quant_spec, expected_cfg_cls, expected_adapter_cls",
    [
        (
            "FP8",
            {
                "method": "modelopt",
                "quant_algo": "FP8",
                "exclude_modules": ["ignored_proj"],
            },
            ModelOptFp8Config,
            ModelOptFp8CheckpointAdapter,
        ),
        (
            "NVFP4",
            {
                "method": "modelopt",
                "quant_algo": "NVFP4",
                "exclude_modules": ["ignored_proj"],
            },
            ModelOptNvFp4Config,
            ModelOptNvFp4CheckpointAdapter,
        ),
        (
            "MIXED_PRECISION",
            {
                "method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": ["ignored_proj"],
                "quantized_layers": {
                    "blocks.0.proj": {"quant_algo": "FP8"},
                },
            },
            ModelOptMixedPrecisionConfig,
            ModelOptMixedPrecisionCheckpointAdapter,
        ),
    ],
    ids=["FP8", "NVFP4", "MIXED_PRECISION"],
)
def test_modelopt_quantization(
    dist_ctx: None,
    quant_algo: str,
    quant_spec: dict[str, object],
    expected_cfg_cls: type,
    expected_adapter_cls: type,
) -> None:
    """ModelOpt Quantization

    - TPU path: resolve ModelOpt*Config via OmniDiffusionConfig, adapt serialized
      weights/scales with get_checkpoint_adapter, build a 128x128 ReplicatedLinear
      on TPU, load adapted tensors, process weights, and run forward.
    - Setup check: current_omni_platform.device_type == "tpu", ModelOpt*Config
      and ModelOpt*CheckpointAdapter match quant_algo, adapter dequantizes
      ignored full-precision targets while preserving quantized weights, and
      excluded ReplicatedLinear resolves UnquantizedLinearMethod.
    - Outcome check: output is on TPU and matches CPU FP32 F.linear within
      FP8/NVFP4 limits.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"{quant_algo}: expected device_type 'tpu', got "
        f"{current_omni_platform.device_type}"
    )

    tpu_device = torch.device("tpu")
    od_config = OmniDiffusionConfig(
        dtype=torch.bfloat16,
        quantization_config=quant_spec,
    )
    quant_config = od_config.quantization_config
    assert isinstance(quant_config, expected_cfg_cls), (
        f"{quant_algo}: expected {expected_cfg_cls.__name__}, got "
        f"{type(quant_config).__name__}"
    )

    torch.manual_seed(0)
    w_ref_fp32 = torch.randn(128, 128, dtype=torch.float32) * 0.1
    w_fp8 = w_ref_fp32.to(torch.float8_e4m3fn)
    w_ref_fp32 = w_fp8.to(torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)
    raw_weights = [
        ("ignored_proj.weight", w_fp8.clone()),
        ("ignored_proj.weight_scale", scale.clone()),
        ("blocks.0.proj.weight", w_fp8.clone()),
        ("blocks.0.proj.weight_scale", scale.clone()),
    ]
    probe_model = _AdapterProbeModel()
    source = SimpleNamespace(subfolder="transformer", prefix="transformer.")
    adapter = get_checkpoint_adapter(
        probe_model,
        source,
        quant_config,
        use_safetensors=True,
    )
    assert isinstance(adapter, expected_adapter_cls), (
        f"{quant_algo}: expected {expected_adapter_cls.__name__}, got "
        f"{type(adapter).__name__}"
    )
    adapted = dict(adapter.adapt(raw_weights))
    assert adapted["ignored_proj.weight"].dtype == torch.bfloat16, (
        f"{quant_algo}: expected dequantized bfloat16 for ignored_proj.weight, "
        f"got {adapted['ignored_proj.weight'].dtype}"
    )
    assert adapted["blocks.0.proj.weight"].dtype == torch.float8_e4m3fn, (
        f"{quant_algo}: expected preserved float8_e4m3fn for blocks.0.proj.weight, "
        f"got {adapted['blocks.0.proj.weight'].dtype}"
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
            f"{quant_algo}: excluded layer resolved "
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

        x_cpu = torch.randn(4, 128, dtype=torch.float32) * 0.5
        layer.weight.data.copy_(adapted["blocks.0.proj.weight"].to(device=tpu_device))
        layer.weight_scale.data.copy_(
            adapted["blocks.0.proj.weight_scale"].to(device=tpu_device)
        )
        layer.quant_method.process_weights_after_loading(layer)
        out_tpu = layer(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))

    assert out_tpu.device.type == "tpu", (
        f"{quant_algo}: output device {out_tpu.device.type} != 'tpu'"
    )
    ref_out = F.linear(x_cpu, w_ref_fp32)
    assert_parity(
        out_tpu.cpu(),
        ref_out,
        name=f"modelopt_{quant_algo}",
        max_rel_l2=MODELOPT_MAX_REL_L2[quant_algo],
        max_abs=MODELOPT_MAX_ABS[quant_algo],
    )
