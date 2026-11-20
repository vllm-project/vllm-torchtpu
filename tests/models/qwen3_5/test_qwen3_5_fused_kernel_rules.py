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
"""The Qwen3.5 rule that installs the fused kernels: the engine they were
measured on (Qwen3.5-397B-A17B on TPU7x) installs the fused q/k norm + rope
and routes every model vLLM's loader finishes through the fused GDN gated norm
+ FP8 quantization bridge at the measured value heads per device; any other
model, Qwen3.5 checkpoint or device is left as it was. Also every leg of the
validation entries keeps both fused kernels."""

import inspect
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml
from vllm.model_executor.model_loader import base_loader

from vllm_torchtpu.models.qwen3_5 import hooks
from vllm_torchtpu.models.qwen3_5.kernels import (
    fused_qk_norm_rope,
    gated_norm_quant_op,
    measured,
)

QWEN35 = ["Qwen3_5MoeForConditionalGeneration"]
# Qwen3.5-397B-A17B's text config (the measured checkpoint) and
# Qwen3.5-35B-A3B's (the model main's integration checks serve).
SHAPE_397B = {"num_experts": 512, "hidden_size": 4096, "num_hidden_layers": 60}
SHAPE_35B = {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40}


def _model_config(architectures, shape):
    return SimpleNamespace(
        architectures=architectures, hf_text_config=SimpleNamespace(**shape)
    )


RULE_CASES = [
    (None, None, "7x", False),
    (["Qwen3NextForCausalLM"], SHAPE_397B, "7x", False),
    (["Qwen3_5ForConditionalGeneration"], SHAPE_397B, "7x", False),
    (QWEN35, SHAPE_35B, "7x", False),
    (QWEN35, {**SHAPE_397B, "num_experts": 256}, "7x", False),
    (QWEN35, SHAPE_397B, "v6e", False),
    (QWEN35, SHAPE_397B, None, False),
    (QWEN35, SHAPE_397B, "7x", True),
]


@pytest.mark.parametrize("architectures, shape, chip, routed", RULE_CASES)
def test_loaded_models_route_where_measured_only(
    monkeypatch, architectures, shape, chip, routed
):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    install = Mock(return_value=3)
    monkeypatch.setattr(gated_norm_quant_op, "install_if_supported", install)
    model = torch.nn.Linear(2, 2)
    model_config = (
        None if architectures is None else _model_config(architectures, shape)
    )
    got = gated_norm_quant_op.route_measured_model(model, model_config)
    if routed:
        install.assert_called_once_with(model, value_heads=measured.GDN_VALUE_HEADS)
        assert got == 3
    else:
        install.assert_not_called()
        assert got == 0


@pytest.mark.parametrize("architectures, shape, chip, installs", RULE_CASES)
def test_the_rule_installs_both_kernels_where_measured_only(
    monkeypatch, architectures, shape, chip, installs
):
    """The registered patch: the measured engine installs the q/k norm + rope
    patch and the post-load GDN routing and completes (None); any other config
    installs neither and stays eligible for the next stage (False)."""
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    qk = Mock()
    after_loading = Mock()
    monkeypatch.setattr(fused_qk_norm_rope, "install_qk_norm_rope_patch", qk)
    monkeypatch.setattr(gated_norm_quant_op, "install_after_loading", after_loading)
    model_config = (
        None if architectures is None else _model_config(architectures, shape)
    )
    got = measured.maybe_install_fused_norm_kernels(model_config)
    assert got is (None if installs else False)
    assert qk.called is installs
    assert after_loading.called is installs


def test_the_loader_routes_every_model_after_its_weights_are_processed(monkeypatch):
    """install_after_loading wraps vLLM's post-processing once: each model the
    loader finishes is processed first, then routed with the loader's config."""
    calls = []

    def process(model, model_config, target_device):
        calls.append(("process", model, model_config, target_device))

    monkeypatch.setattr(base_loader, "process_weights_after_loading", process)
    monkeypatch.setattr(
        gated_norm_quant_op,
        "route_measured_model",
        lambda model, model_config: calls.append(("route", model, model_config)),
    )
    gated_norm_quant_op.install_after_loading()
    wrapped = base_loader.process_weights_after_loading
    gated_norm_quant_op.install_after_loading()
    assert base_loader.process_weights_after_loading is wrapped
    model, model_config, device = object(), object(), torch.device("cpu")
    base_loader.process_weights_after_loading(model, model_config, device)
    assert calls == [
        ("process", model, model_config, device),
        ("route", model, model_config),
    ]


def test_vllms_loader_calls_the_post_processing_by_its_module_name():
    """The routing replaces ``base_loader.process_weights_after_loading``; vLLM's
    loader looks that name up in its module when it loads a model (under its
    tracing decorator)."""
    load_model = inspect.unwrap(base_loader.BaseModelLoader.load_model)
    assert "process_weights_after_loading" in load_model.__code__.co_names


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
def test_every_entry_leg_keeps_both_fused_kernels(path):
    """Each point entry's prefill and decode legs run the 397B attention and
    GDN layers at measured head counts (TP1, TP2 or TP8), so both kernels
    still run there; the gate entries reuse these legs. The checkpoint is the
    launcher's ``$MODEL``, which the entries do not name."""
    with path.open() as f:
        legs = yaml.safe_load(f)["legs"]
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
        # The 397B at TP4: not routed.
        (64, 4, False),
        # Qwen3.5-35B-A3B (32 value heads) at TP1 and TP4: routed by the value
        # heads alone; at TP2: not routed.
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
