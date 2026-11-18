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
"""TPU tests for per-component quantization config routing."""

from __future__ import annotations

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
from vllm_omni.quantization import (
    ComponentQuantizationConfig,
    build_quant_config,
    resolve_component_quant_config,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

FP8_MAX_REL_L2 = 1.2e-1
FP8_MAX_ABS = 2.0e-1
BF16_MAX_REL_L2 = 2.0e-2
BF16_MAX_ABS = 4.0e-2


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


def test_per_component_quantization_config(dist_ctx: None) -> None:
    """Per-component quantization config.

    - TPU path: Verify per-component dict parsing via ``build_quant_config`` and
      instantiate ``ComponentQuantizationConfig`` routing
      ``transformer.blocks.0.to_q`` to ``VllmFp8Config`` (TPU FP8) and
      ``transformer.blocks.1.to_q`` to unquantized BF16 on ``"tpu"``.
    - Setup check: Assert longest-prefix resolution via
      ``resolve_component_quant_config``, ``VllmFp8LinearMethodTPU`` with
      ``torch.float8_e4m3fn`` weight after ``process_weights_after_loading``,
      and ``UnquantizedLinearMethod`` with ``torch.bfloat16`` weight.
    - Outcome check: Execute both layers on ``"tpu"`` bfloat16 activations and
      assert relative-L2 / max-abs parity against CPU float32 ``F.linear``.
    """
    parsed = build_quant_config({"transformer": {"method": "fp8"}, "vae": None})
    assert isinstance(parsed, ComponentQuantizationConfig), (
        f"Expected ComponentQuantizationConfig, got {type(parsed).__name__}"
    )
    assert resolve_component_quant_config(parsed, "transformer.blocks.0") is not None, (
        "Expected transformer prefix to resolve to a non-None quant config"
    )
    assert resolve_component_quant_config(parsed, "vae.decoder") is None, (
        "Expected vae prefix to resolve to None"
    )

    cfg = ComponentQuantizationConfig(
        {
            "transformer.blocks.0": VllmFp8Config(is_checkpoint_fp8_serialized=True),
            "transformer": None,
            "vae": None,
        }
    )
    quant_cfg = resolve_component_quant_config(cfg, "transformer.blocks.0.to_q")
    bf16_cfg = resolve_component_quant_config(cfg, "transformer.blocks.1.to_q")
    assert isinstance(quant_cfg, VllmFp8Config), (
        "Expected longest prefix 'transformer.blocks.0' to resolve to VllmFp8Config"
    )
    assert bf16_cfg is None, "Expected 'transformer.blocks.1.to_q' to resolve to None"

    quant_layer = ReplicatedLinear(
        128,
        128,
        bias=False,
        quant_config=quant_cfg,
        params_dtype=torch.bfloat16,
        prefix="transformer.blocks.0.to_q",
    )
    bf16_layer = ReplicatedLinear(
        128,
        128,
        bias=False,
        quant_config=bf16_cfg,
        params_dtype=torch.bfloat16,
        prefix="transformer.blocks.1.to_q",
    )
    assert isinstance(quant_layer.quant_method, VllmFp8LinearMethodTPU), (
        f"Expected VllmFp8LinearMethodTPU, "
        f"got {type(quant_layer.quant_method).__name__}"
    )
    assert isinstance(bf16_layer.quant_method, UnquantizedLinearMethod), (
        f"Expected UnquantizedLinearMethod, "
        f"got {type(bf16_layer.quant_method).__name__}"
    )

    torch.manual_seed(0)
    w_raw = torch.randn(128, 128, dtype=torch.float32) * 0.1
    scale = float(w_raw.abs().max().item()) / 448.0
    w_fp8 = (w_raw / scale).to(torch.float8_e4m3fn)
    w_fp8_ref = w_fp8.float() * scale
    w_bf16 = torch.randn(128, 128, dtype=torch.bfloat16) * 0.1
    x_cpu = torch.randn(2, 16, 128, dtype=torch.bfloat16)

    with torch.no_grad():
        quant_layer.weight.copy_(w_fp8)
        quant_layer.weight_scale.fill_(scale)
        bf16_layer.weight.copy_(w_bf16)

    quant_layer = quant_layer.to("tpu")
    bf16_layer = bf16_layer.to("tpu")
    quant_layer.quant_method.process_weights_after_loading(quant_layer)
    assert quant_layer.weight.dtype == torch.float8_e4m3fn, (
        f"Expected quant_layer.weight.dtype == torch.float8_e4m3fn, "
        f"got {quant_layer.weight.dtype}"
    )
    assert bf16_layer.weight.dtype == torch.bfloat16, (
        f"Expected bf16_layer.weight.dtype == torch.bfloat16, "
        f"got {bf16_layer.weight.dtype}"
    )

    x_tpu = x_cpu.to("tpu")
    quant_out, _ = quant_layer(x_tpu)
    bf16_out, _ = bf16_layer(x_tpu)

    assert_parity(
        quant_out,
        F.linear(x_cpu.float(), w_fp8_ref),
        name="component_quant_fp8",
        max_rel_l2=FP8_MAX_REL_L2,
        max_abs=FP8_MAX_ABS,
    )
    assert_parity(
        bf16_out,
        F.linear(x_cpu.float(), w_bf16.float()),
        name="component_unquant_bf16",
        max_rel_l2=BF16_MAX_REL_L2,
        max_abs=BF16_MAX_ABS,
    )
