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
"""The Qwen3.5 rule that installs the fused GDN gated norm + FP8 quantization
after loading: the engine it was measured on (Qwen3.5-397B-A17B on TPU7x)
routes its model through the kernel bridge at the measured value heads per
device, and any other model, Qwen3.5 checkpoint or device is left as it was.
Also every IX leg of the validation entries keeps both fused kernels."""

import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from vllm_torchtpu.models.qwen3_5 import hooks
from vllm_torchtpu.models.qwen3_5.kernels import gated_norm_quant_op, measured
from vllm_torchtpu.runner import tpu_runner

QWEN35 = ["Qwen3_5MoeForConditionalGeneration"]
# Qwen3.5-397B-A17B's text config (the measured checkpoint) and
# Qwen3.5-35B-A3B's (the model main's integration checks serve).
SHAPE_397B = {"num_experts": 512, "hidden_size": 4096, "num_hidden_layers": 60}
SHAPE_35B = {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40}


def _model_config(architectures, shape):
    return SimpleNamespace(
        architectures=architectures, hf_text_config=SimpleNamespace(**shape)
    )


@pytest.mark.parametrize(
    "architectures, shape, chip, routed",
    [
        (None, None, "7x", False),
        (["Qwen3NextForCausalLM"], SHAPE_397B, "7x", False),
        (["Qwen3_5ForConditionalGeneration"], SHAPE_397B, "7x", False),
        (QWEN35, SHAPE_35B, "7x", False),
        (QWEN35, {**SHAPE_397B, "num_experts": 256}, "7x", False),
        (QWEN35, SHAPE_397B, "v6e", False),
        (QWEN35, SHAPE_397B, None, False),
        (QWEN35, SHAPE_397B, "7x", True),
    ],
)
def test_the_runner_routes_the_measured_model_only(
    monkeypatch, architectures, shape, chip, routed
):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    install = Mock(return_value=3)
    monkeypatch.setattr(gated_norm_quant_op, "install_if_supported", install)
    model = torch.nn.Linear(2, 2)
    model_config = (
        None if architectures is None else _model_config(architectures, shape)
    )
    got = tpu_runner._install_gdn_fused_norm_quant(model, model_config)
    if routed:
        install.assert_called_once_with(model, value_heads=measured.GDN_VALUE_HEADS)
        assert got == 3
    else:
        install.assert_not_called()
        assert got == 0


def test_main_ci_model_never_queries_the_device(monkeypatch):
    """The checkpoint shape is checked first, so a 35B engine never reads the
    accelerator type."""
    chip = Mock(return_value="7x")
    monkeypatch.setattr(hooks, "_chip_family", chip)
    assert not measured.is_measured_model(_model_config(QWEN35, SHAPE_35B))
    chip.assert_not_called()


# 397B attention: 32 query and 2 kv heads; GDN: 64 value heads.
HEADS_397B, KV_HEADS_397B, VALUE_HEADS_397B = 32, 2, 64

VALIDATION = pathlib.Path(hooks.__file__).parent / "validation"
POINTS = sorted(VALIDATION.glob("p[0-9][0-9]-*.yaml"))


def test_validation_points_are_present():
    assert [p.name.split("-")[0] for p in POINTS] == ["p01", "p05", "p08", "p10", "p12"]


@pytest.mark.parametrize("path", POINTS, ids=lambda p: p.stem)
def test_every_ix_leg_keeps_both_fused_kernels(monkeypatch, path):
    """Each IX point's prefill and decode legs (397B on TPU7x, TP1, TP2 or
    TP8) are the measured model at measured head counts, so both kernels
    still run there; the gate entries reuse these legs."""
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")
    with path.open() as f:
        legs = yaml.safe_load(f)["legs"]
    assert measured.is_measured_model(_model_config(QWEN35, SHAPE_397B))
    tps = []
    for leg in legs.values():
        args = leg["serve_args"]
        tp = int(args[args.index("--tensor-parallel-size") + 1])
        tps.append(tp)
        attention = (HEADS_397B // tp, max(1, KV_HEADS_397B // tp))
        assert attention in measured.ATTENTION_HEADS
        assert VALUE_HEADS_397B // tp in measured.GDN_VALUE_HEADS
        assert "--quantization" in args and "--language-model-only" in args
    assert tps[0] == 1 and tps[1] in (2, 8)


def _gdn_layer(num_v_heads, tp_size):
    layer = Mock(spec=gated_norm_quant_op.QwenGatedDeltaNetAttention)
    layer.num_v_heads, layer.tp_size, layer.prefix = num_v_heads, tp_size, "gdn"
    return layer


@pytest.mark.parametrize(
    "num_v_heads, tp_size, routed",
    [
        (64, 1, True),
        (64, 2, True),
        (64, 8, True),
        # The 397B at TP4; Qwen3.5-35B-A3B (32 value heads) at TP1 and TP4.
        (64, 4, False),
        (32, 1, True),
        (32, 4, True),
        (32, 2, False),
    ],
)
def test_install_if_supported_routes_the_measured_value_heads_only(
    monkeypatch, num_v_heads, tp_size, routed
):
    """The value-head check alone (the model rule above keeps the 35B out:
    its TP1 and TP4 layers have the 397B's TP2 and TP8 head counts)."""
    monkeypatch.setattr(gated_norm_quant_op, "unsupported_reason", lambda layer: None)
    install = Mock(return_value=1)
    monkeypatch.setattr(gated_norm_quant_op, "install", install)
    layer = _gdn_layer(num_v_heads, tp_size)
    model = SimpleNamespace(modules=lambda: iter([layer]))
    got = gated_norm_quant_op.install_if_supported(
        model, value_heads=measured.GDN_VALUE_HEADS
    )
    assert got == int(routed)
    assert install.called is routed
    # Without value_heads (install's other callers) the heads do not matter.
    install.reset_mock()
    assert gated_norm_quant_op.install_if_supported(model) == 1
