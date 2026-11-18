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
"""TPU tests for SVDQuant NVFP4 quantization config."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.svdquant_config import (
    DiffusionSVDQuantConfig,
    DiffusionSVDQuantLinearMethod,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 5.0e-2
MAX_ABS = 1.0e-1


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="SVDQuant NVFP4 requires a CUDA device (_assert_supported)",
)
def test_svdquant_quantization() -> None:
    """SVDQuant Quantization.

    - TPU path: Construct ``DiffusionSVDQuantConfig`` via
      ``build_quant_config("svdquant", rank=16, precision="nvfp4")`` and
      instantiate + execute ``ReplicatedLinear(64, 64)`` on ``"tpu"`` in
      bfloat16.
    - Setup check: Assert ``isinstance(cfg, DiffusionSVDQuantConfig)`` and
      ``cfg.rank == 16`` before layer construction, and
      ``isinstance(layer.quant_method, DiffusionSVDQuantLinearMethod)`` after.
    - Outcome check: Populate synthetic SVDQuant parameters on ``"tpu"``, call
      ``process_weights_after_loading``, run ``layer(x_tpu)``, and assert parity
      against CPU float32 NVFP4-dequant + low-rank residual reference.
    """
    cfg = build_quant_config("svdquant", rank=16, precision="nvfp4")
    assert isinstance(cfg, DiffusionSVDQuantConfig), (
        f"Expected DiffusionSVDQuantConfig, got {type(cfg).__name__}"
    )
    assert cfg.rank == 16, f"Expected rank=16, got {cfg.rank}"
    assert cfg.precision == "nvfp4", (
        f"Expected precision='nvfp4', got {cfg.precision!r}"
    )

    layer = ReplicatedLinear(
        64,
        64,
        bias=False,
        quant_config=cfg,
        params_dtype=torch.bfloat16,
        prefix="transformer.proj",
    ).to("tpu")
    assert isinstance(layer.quant_method, DiffusionSVDQuantLinearMethod), (
        f"Expected DiffusionSVDQuantLinearMethod, "
        f"got {type(layer.quant_method).__name__}"
    )

    torch.manual_seed(0)
    x_cpu = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    proj_down = torch.randn(64, 16, dtype=torch.bfloat16) * 0.05
    proj_up = torch.randn(64, 16, dtype=torch.bfloat16) * 0.05
    with torch.no_grad():
        layer.qweight.zero_()
        layer.wscales.fill_(1)
        layer.proj_down.copy_(proj_down.to("tpu"))
        layer.proj_up.copy_(proj_up.to("tpu"))
        layer.smooth_factor.fill_(1.0)
        layer.wcscales.fill_(1.0)
        layer.wtscale.fill_(1.0)

    layer.quant_method.process_weights_after_loading(layer)
    out_tpu, _ = layer(x_cpu.to("tpu"))
    ref_out = F.linear(x_cpu.float() @ proj_down.float(), proj_up.float())
    assert_parity(
        out_tpu,
        ref_out,
        name="svdquant_nvfp4",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
