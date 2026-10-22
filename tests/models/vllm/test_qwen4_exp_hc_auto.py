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
"""Calibrated HC choices are independent of numerical/resource admission."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

MEASURED = HcHardware("7x", 64 << 20, 128, 16, 8, 256)


@pytest.fixture
def boundaries(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    calls = []

    def mix(x, weight, down, up):
        calls.append("mix")
        return x[:, :2560], x

    def combine(x, block, gates):
        calls.append("combine")
        return x

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: MEASURED)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a: mix)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", lambda *a: combine)
    return calls


def meta_module(**kwargs):
    with torch.device("meta"):
        return GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
            ),
            **kwargs,
        )


def route_view(module):
    """Pure tensor-metadata harness for a target TPU; never allocates a device.

    Production guards are executed unmodified, while dtype/policy/factory facts
    come from the real constructor. These are CPU policy tests, not TPU runs.
    """
    view = SimpleNamespace(**vars(module))
    for name in (
        "hc_norm",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
    ):
        if hasattr(module, name):
            weight = getattr(module, name).weight
            setattr(
                view,
                name,
                SimpleNamespace(
                    weight=SimpleNamespace(
                        dtype=weight.dtype, device=SimpleNamespace(type="tpu")
                    )
                ),
            )
    return view


def route_inputs(tokens, shape=None, dtype=torch.bfloat16):
    shape = shape or (tokens, 10240)

    def metadata(shape):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=SimpleNamespace(type="tpu"),
        )

    return metadata(shape), metadata((*shape[:-1], 2560))


def mix_selected(module, x):
    return GatedResidual._use_prefill_mix(route_view(module), x)


def combine_selected(module, block, x, norm):
    return GatedResidual._use_prefill_combine(route_view(module), block, x, norm)


def test_default_auto_dispatch(boundaries):
    module = meta_module()
    assert module.prefill_full_tile == module.prefill_combine_tile == "auto"
    x, block = route_inputs(1024)
    assert mix_selected(module, x)
    assert combine_selected(module, block, x, x)


@pytest.mark.parametrize(
    "full,combine,expected",
    [
        (0, "auto", ["combine"]),
        ("auto", 0, ["mix"]),
        ("auto", None, ["mix", "combine"]),
        (0, None, []),
        (128, None, ["mix", "combine"]),
        (512, "auto", ["combine"]),
    ],
)
def test_independent_modes(full, combine, expected, boundaries):
    module = meta_module(prefill_full_tile=full, prefill_combine_tile=combine)
    x, block = route_inputs(1024)
    selected = []
    if mix_selected(module, x):
        selected.append("mix")
    if combine_selected(module, block, x, x):
        selected.append("combine")
    assert selected == expected


@pytest.mark.parametrize(
    "tokens",
    [
        1,
        32,
        128,
        256,
        512,
        640,
        768,
        1023,
        1024,
        1152,
        1536,
        2048,
        4096,
        8192,
        8320,
        127,
        129,
        257,
        1025,
    ],
)
def test_auto_minimum_threshold_and_full_tile_alignment(tokens, boundaries):
    module = meta_module()
    x, block = route_inputs(tokens)
    selected = tokens >= 1024 and tokens % 128 == 0
    assert mix_selected(module, x) is selected
    assert combine_selected(module, block, x, x) is selected


@pytest.mark.parametrize(
    "change",
    [
        {"chip": "6e"},
        {"chip": "unknown"},
        {"vmem_bytes": 16 << 20},
        {"vmem_bytes": 128 << 20},
        {"lanes": 256},
        {"bf16_rows": 8},
        {"fp32_rows": 16},
        {"mxu_columns": 128},
    ],
)
def test_unmeasured_hardware_falls_back(change, boundaries, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    facts = dataclasses.replace(MEASURED, **change)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: facts)
    module = meta_module()
    assert module._prefill_full_op is module._prefill_combine_op is None


@pytest.mark.parametrize(
    "change",
    [
        {"hc_count": 3},
        {"hidden_size": 128},
        {"hc_lowrank": 128},
        {"hc_per_branch_norm": False},
        {"rms_norm_eps": 1e-5},
    ],
)
def test_unmeasured_geometry_falls_back(change, boundaries):
    config = dict(hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True)
    config.update(change)
    with torch.device("meta"):
        module = GatedResidual(HyperConnectionConfig(**config))
    assert module._prefill_full_op is module._prefill_combine_op is None


def test_resource_admission_still_independent(boundaries, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import hc_resources

    monkeypatch.setattr(hc_resources, "supports_full_mix", lambda *a: False)
    module = meta_module()
    assert module._prefill_full_op is None
    assert module._prefill_combine_op is not None
    x, block = route_inputs(1024)
    assert not mix_selected(module, x)
    assert combine_selected(module, block, x, x)


def test_combine_only_does_not_require_mix_weights(boundaries):
    module = meta_module(use_mix=False)
    x, block = route_inputs(1024)
    assert module._prefill_full_op is None
    assert combine_selected(module, block, x, x)
    assert module._prefill_combine_op is not None


@pytest.mark.parametrize(
    "name,mix,combine",
    [
        ("hc_norm", False, False),
        ("input_mix_weight_down", False, True),
        ("input_mix_weight_up", False, True),
        ("block_inject_weight", True, False),
    ],
)
def test_relevant_weight_dtype_guards(name, mix, combine, boundaries):
    module = meta_module()
    getattr(module, name).to(dtype=torch.float32)
    x, block = route_inputs(1024)
    assert mix_selected(module, x) is mix
    assert combine_selected(module, block, x, x) is combine


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_input_dtype_fallback(dtype, boundaries):
    module = meta_module()
    x, block = route_inputs(1024, dtype=dtype)
    assert not mix_selected(module, x)
    assert not combine_selected(module, block, x, x)


@pytest.mark.parametrize("shape", [(10240,), (1024, 1, 10240)])
def test_rank_fallback(shape, boundaries):
    module = meta_module()
    x, block = route_inputs(1024, shape=shape)
    assert not mix_selected(module, x)
    assert not combine_selected(module, block, x, x)


def test_combine_individual_input_metadata(boundaries):
    module = meta_module()
    x, block = route_inputs(1024)
    rank3, _ = route_inputs(1024, shape=(1024, 1, 10240))
    norm32, _ = route_inputs(1024, dtype=torch.float32)
    one, one_block = route_inputs(1)
    assert not combine_selected(module, rank3, x, x)
    assert not combine_selected(module, block, x, rank3)
    assert not combine_selected(module, one_block, x, x)
    assert not combine_selected(module, block, x, one)
    assert not combine_selected(module, block, x, norm32)


@pytest.mark.parametrize(
    "value", [True, False, -1, -128, "128", "AUTO", "", 1.5, [], {}]
)
@pytest.mark.parametrize("component", ["full", "combine"])
def test_invalid_modes_rejected(value, component):
    kwargs = {
        "prefill_full_tile" if component == "full" else "prefill_combine_tile": value
    }
    with pytest.raises(ValueError, match="HC prefill mode"):
        GatedResidual(HyperConnectionConfig(), **kwargs)


def test_unaligned_forced_mix_rejected():
    with pytest.raises(ValueError, match="multiple of 128"):
        GatedResidual(HyperConnectionConfig(), prefill_full_tile=16)


def test_no_backend_initialization_and_frozen_unknown_facts(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    def forbidden():
        raise AssertionError("CPU construction attempted hardware discovery")

    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(is_initialized=lambda: False), raising=False
    )
    monkeypatch.setattr(vllm_adapter.pltpu, "get_tpu_info", forbidden)
    with torch.device("cpu"):
        module = GatedResidual(HyperConnectionConfig(hidden_size=128))
    assert module.prefill_full_tile == module.prefill_combine_tile == "auto"
    assert module._prefill_full_op is module._prefill_combine_op is None
    # Moving a module does not refresh a captured policy; no implicit init/query.
    module.to(dtype=torch.float32)
    assert module._auto_mix_min_tokens is module._auto_combine_min_tokens is None


def test_auto_fallback_actual_math_and_no_fused_call(monkeypatch):
    from tests.models.vllm.test_qwen4_exp_math import hc_cuda_source_reference
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    def forbidden(*args):
        pytest.fail("CPU fallback reached fused HC")

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: MEASURED)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a: forbidden)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", lambda *a: forbidden)
    with torch.device("cpu"):
        torch.random.default_generator.manual_seed(1961)
        module = GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
            )
        )
        x = torch.randn(8, 10240).bfloat16()
        block = torch.randn(8, 2560).bfloat16()
        expected = hc_cuda_source_reference(module, x, block)
        mixed, residual = module.mix(x)
        result = module.combine(block, residual)
    for actual, ref in zip((mixed, result), expected, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def test_auto_cpu_device_falls_back_even_after_hardware_capture(boundaries):
    module = meta_module()
    with FakeTensorMode():
        x = torch.empty(1024, 10240, device="cpu", dtype=torch.bfloat16)
        block = torch.empty(1024, 2560, device="cpu", dtype=torch.bfloat16)
        assert not module._use_prefill_mix(x)
        assert not module._use_prefill_combine(block, x, x)
