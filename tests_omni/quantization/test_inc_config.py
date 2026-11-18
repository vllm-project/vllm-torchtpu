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
"""TPU tests for Intel Neural Compressor / AutoRound quantization config."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.inc_config import OmniINCConfig

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 5.0e-2
MAX_ABS = 1.0e-1


@pytest.mark.parametrize(
    ("data_type", "group_size", "packing_format"),
    [
        pytest.param(
            "int",
            128,
            "auto_round:auto_gptq",
            id="w4a16_int",
            marks=pytest.mark.xfail(
                strict=True,
                raises=NotImplementedError,
                reason=(
                    "AutoRound W4A16 requires Marlin/GPTQ kernels that are "
                    "not implemented on TPU"
                ),
            ),
        ),
        pytest.param(
            "mx_fp",
            32,
            "auto_round:llm_compressor",
            id="mxfp4",
            marks=pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason=("AutoRound MXFP4 fails in init_mxfp4_linear_kernel on TPU"),
            ),
        ),
    ],
)
def test_autoround_quantization(
    data_type: str,
    group_size: int,
    packing_format: str,
) -> None:
    """AutoRound Quantization.

    - TPU path: Build ``OmniINCConfig`` via ``build_quant_config("auto-round", ...)``
      and run a tiny ``ReplicatedLinear(128, 64)`` on ``"tpu"`` in bfloat16.
    - Setup check: Assert ``isinstance(cfg, OmniINCConfig)`` and verify
      ``weight_bits``, ``group_size``, and ``data_type`` match the scheme before
      constructing the linear layer.
    - Outcome check: Load packed weights and scales onto ``"tpu"``, run
      ``process_weights_after_loading`` and ``layer(x_tpu)``, and assert
      relative-L2 / max-abs parity against CPU float32 ``F.linear``.
    """
    cfg = build_quant_config(
        "auto-round",
        weight_bits=4,
        group_size=group_size,
        sym=True,
        data_type=data_type,
        packing_format=packing_format,
    )
    assert isinstance(cfg, OmniINCConfig), (
        f"Expected OmniINCConfig, got {type(cfg).__name__}"
    )
    assert cfg.weight_bits == 4, f"Expected weight_bits=4, got {cfg.weight_bits}"
    assert cfg.group_size == group_size, (
        f"Expected group_size={group_size}, got {cfg.group_size}"
    )
    assert cfg.data_type == data_type, (
        f"Expected data_type={data_type!r}, got {cfg.data_type!r}"
    )

    layer = ReplicatedLinear(
        128,
        64,
        bias=False,
        quant_config=cfg,
        params_dtype=torch.bfloat16,
        prefix="transformer.proj",
    ).to("tpu")
    torch.manual_seed(0)
    ref_weight = torch.randn(64, 128, dtype=torch.float32) * 0.1
    x_cpu = torch.randn(2, 16, 128, dtype=torch.bfloat16)

    layer.quant_method.process_weights_after_loading(layer)
    out_tpu, _ = layer(x_cpu.to("tpu"))
    ref_out = F.linear(x_cpu.float(), ref_weight)
    assert_parity(
        out_tpu,
        ref_out,
        name=f"autoround_{data_type}",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
