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
"""TPU tests for quantization factory methods (TorchAO, msModelSlim)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.quantization import ComponentQuantizationConfig, build_quant_config

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 5.0e-2
MAX_ABS = 1.0e-1


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="torchao is not installed in the TPU omni image",
)
def test_torchao_quantization() -> None:
    """TorchAO Quantization.

    - TPU path: Build per-component ``torchao_float8_weight_only`` config via
      ``build_quant_config`` and execute a tiny ``ReplicatedLinear(128, 64)``
      on ``"tpu"`` in bfloat16.
    - Setup check: Assert ``isinstance(cfg, ComponentQuantizationConfig)``,
      resolved transformer config has ``get_name() == "torchao"``, and
      ``layer.quant_method`` is ``TorchAOLinearMethod``.
    - Outcome check: Load FP8 weight-only tensor onto ``"tpu"``, run
      ``layer(x_tpu)``, and assert relative-L2 / max-abs parity against CPU
      float32 ``F.linear``.
    """
    cfg = build_quant_config({"transformer": {"method": "torchao_float8_weight_only"}})
    assert isinstance(cfg, ComponentQuantizationConfig), (
        f"Expected ComponentQuantizationConfig, got {type(cfg).__name__}"
    )
    transformer_cfg = cfg.resolve("transformer.proj")
    assert transformer_cfg is not None, "Resolved transformer config is None"
    assert transformer_cfg.get_name() == "torchao", (
        f"Expected get_name() == 'torchao', got {transformer_cfg.get_name()!r}"
    )

    from vllm.model_executor.layers.quantization.torchao import TorchAOLinearMethod

    layer = ReplicatedLinear(
        128,
        64,
        bias=False,
        quant_config=transformer_cfg,
        params_dtype=torch.bfloat16,
        prefix="transformer.proj",
    ).to("tpu")
    assert isinstance(layer.quant_method, TorchAOLinearMethod), (
        f"Expected TorchAOLinearMethod, got {type(layer.quant_method).__name__}"
    )

    torch.manual_seed(0)
    ref_weight = torch.randn(64, 128, dtype=torch.float32) * 0.1
    x_cpu = torch.randn(2, 16, 128, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(ref_weight.to(torch.float8_e4m3fn).to("tpu"))
    out_tpu, _ = layer(x_cpu.to("tpu"))
    ref_out = F.linear(x_cpu.float(), ref_weight)
    assert_parity(
        out_tpu,
        ref_out,
        name="torchao_fp8_weight_only",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Ascend msModelSlim ('ascend') quantization is not registered on TPU",
)
def test_msmodelslim_quantization() -> None:
    """msModelSlim Quantization.

    - TPU path: Resolve ``ascend`` quantization via ``OmniDiffusionConfig`` and
      ``build_quant_config("ascend")``, then execute a tiny
      ``ReplicatedLinear(64, 64)`` on ``"tpu"`` in bfloat16.
    - Setup check: Assert ``cfg is not None``, ``cfg.get_name() == "ascend"``,
      and ``layer.quant_method`` is not ``UnquantizedLinearMethod``.
    - Outcome check: Load static INT8 weights and scales onto ``"tpu"``, call
      ``process_weights_after_loading``, run ``layer(x_tpu)``, and assert
      relative-L2 / max-abs parity against CPU float32 ``F.linear``.
    """
    od_config = OmniDiffusionConfig(model="test", quantization_config="ascend")
    cfg = od_config.quantization_config
    assert cfg is not None, "Expected non-None quantization_config for 'ascend'"
    assert cfg.get_name() == "ascend", (
        f"Expected get_name() == 'ascend', got {cfg.get_name()!r}"
    )

    layer = ReplicatedLinear(
        64,
        64,
        bias=False,
        quant_config=cfg,
        params_dtype=torch.bfloat16,
        prefix="transformer.proj",
    ).to("tpu")
    assert not isinstance(layer.quant_method, UnquantizedLinearMethod), (
        "Expected quantized linear method, got UnquantizedLinearMethod"
    )

    torch.manual_seed(0)
    ref_weight = torch.randn(64, 64, dtype=torch.float32) * 0.1
    x_cpu = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    layer.quant_method.process_weights_after_loading(layer)
    out_tpu, _ = layer(x_cpu.to("tpu"))
    ref_out = F.linear(x_cpu.float(), ref_weight)
    assert_parity(
        out_tpu,
        ref_out,
        name="msmodelslim_ascend",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
