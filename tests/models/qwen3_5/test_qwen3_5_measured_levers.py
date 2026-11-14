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
"""Qwen3.5's kernel levers apply only where they were measured.

The kernel numerics rule (the quantize helpers' reciprocal scale, the grouped
matmul's streamed whole-expert slabs, its tuned FP8 MoE tiling and rhs-scale
hoist) and the few-token MoE kernel change the last bits of the forward pass.
They were measured for Qwen3.5-397B-A17B on TPU7x, the kernel at EP8. Every
other Qwen3.5 checkpoint, device and expert-parallel size (main's CI model,
Qwen3.5-35B-A3B at EP4, among them) keeps main's arithmetic and code, and
every leg of the validation entries keeps what it ran."""

import pathlib
from types import SimpleNamespace

import pytest
import yaml

from vllm_torchtpu.kernels.megablox import gmm_v2
from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores as scores
from vllm_torchtpu.layers.core import fused_moe_gmm
from vllm_torchtpu.models.qwen3_5 import hooks, settings

pytestmark = pytest.mark.cpu_test

VALIDATION = pathlib.Path(hooks.__file__).parent / "validation"
POINTS = sorted(VALIDATION.glob("p[0-9][0-9]-*.yaml"))
GATES = sorted(VALIDATION.glob("gate-*.yaml"))

QWEN3_5_ARCHS = sorted(settings.ARCHITECTURES)
SHAPE_397B = dict(hooks._CALIBRATED_SHAPE)
# Main's CI checkpoint, Qwen3.5-35B-A3B(-FP8): its text config's shape.
SHAPE_35B = {"num_experts": 256, "hidden_size": 2048, "num_hidden_layers": 40}
ON = (True, True, True)
MAINS = (False, False, False)


@pytest.fixture(autouse=True)
def _mains_default(monkeypatch):
    monkeypatch.setattr(util, "RECIPROCAL_SCALE", False)
    monkeypatch.setattr(gmm_v2, "RHS_STREAM", False)
    monkeypatch.setattr(gmm_v2, "TUNED_FP8_MOE", False)
    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")
    # The few-token kernel's armed cap and the grouped path's routed-ids
    # switch it sets; both start unarmed and are restored after each test.
    monkeypatch.setattr(scores, "_ARMED_MAX_TOKENS", 0)
    monkeypatch.setattr(scores, "_LAYERS", [])
    monkeypatch.setattr(fused_moe_gmm, "_MOE_LOWTOK_MAX_TOKENS", 0)


def _levers() -> tuple[bool, bool, bool]:
    return util.RECIPROCAL_SCALE, gmm_v2.RHS_STREAM, gmm_v2.TUNED_FP8_MOE


def _model(shape: dict, arch: str = QWEN3_5_ARCHS[0]) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=[arch], hf_text_config=SimpleNamespace(**shape)
    )


def _engine(shape: dict, ep: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=_model(shape),
        parallel_config=SimpleNamespace(enable_expert_parallel=ep),
    )


def _arm(monkeypatch, engine, ep_size: int) -> int:
    """The few-token kernel's prebuild for one MoE layer of `engine` at
    `ep_size`; returns the row cap the grouped path is armed with."""
    import vllm.config

    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: engine)
    layer = SimpleNamespace(
        layer_name="model.layers.0.mlp.experts",
        # Not expert-parallel itself, so the prebuild stops at the layer's
        # verdict after the rule has armed (or not armed) the kernel.
        moe_config=SimpleNamespace(
            ep_size=ep_size, moe_parallel_config=SimpleNamespace(use_ep=False)
        ),
    )
    assert scores.prebuild(SimpleNamespace(), layer, "silu") is None
    assert scores.armed_max_tokens() == fused_moe_gmm._MOE_LOWTOK_MAX_TOKENS
    return fused_moe_gmm._MOE_LOWTOK_MAX_TOKENS


# Kernel numerics.


@pytest.mark.parametrize("arch", QWEN3_5_ARCHS)
def test_measured_checkpoint_on_tpu7x_takes_the_kernel_numerics(arch):
    assert hooks.is_measured_target(_model(SHAPE_397B, arch)) is True
    assert hooks.apply_kernel_numerics(_model(SHAPE_397B, arch)) is None
    assert _levers() == ON


@pytest.mark.parametrize("arch", QWEN3_5_ARCHS)
def test_mains_ci_checkpoint_keeps_mains_arithmetic(arch):
    assert hooks.is_measured_target(_model(SHAPE_35B, arch)) is False
    assert hooks.apply_kernel_numerics(_model(SHAPE_35B, arch)) is None
    assert _levers() == MAINS


@pytest.mark.parametrize("key", sorted(SHAPE_397B))
def test_a_checkpoint_that_differs_in_one_dimension_keeps_mains_arithmetic(key):
    shape = dict(SHAPE_397B, **{key: SHAPE_397B[key] // 2})
    assert hooks.apply_kernel_numerics(_model(shape)) is None
    assert _levers() == MAINS


@pytest.mark.parametrize("chip", ["v6e", "v5p", "v5e", None])
def test_the_measured_checkpoint_on_another_device_keeps_mains_arithmetic(
    monkeypatch, chip
):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    assert hooks.apply_kernel_numerics(_model(SHAPE_397B)) is None
    assert _levers() == MAINS


def test_mains_ci_checkpoint_never_reads_the_device(monkeypatch):
    """The shape decides first, so a 35B engine does not query the chip."""

    def _fail():
        raise AssertionError("the chip was read for a checkpoint that is not measured")

    monkeypatch.setattr(hooks, "_chip_family", _fail)
    assert hooks.apply_kernel_numerics(_model(SHAPE_35B)) is None
    assert _levers() == MAINS


# The few-token MoE kernel.


@pytest.mark.parametrize(
    "shape,ep_size,chip,cap",
    [
        (SHAPE_397B, 8, "7x", 448),
        (SHAPE_397B, 4, "7x", 0),
        (SHAPE_397B, 2, "7x", 0),
        (SHAPE_397B, 8, "v6e", 0),
        (SHAPE_35B, 4, "7x", 0),
        (SHAPE_35B, 8, "7x", 0),
        (SHAPE_35B, 1, "7x", 0),
    ],
    ids=[
        "397b-ep8-tpu7x",
        "397b-ep4-tpu7x",
        "397b-ep2-tpu7x",
        "397b-ep8-v6e",
        "35b-ep4-tpu7x(main-ci)",
        "35b-ep8-tpu7x",
        "35b-ep1-tpu7x",
    ],
)
def test_few_token_kernel_is_armed_only_where_it_was_measured(
    monkeypatch, shape, ep_size, chip, cap
):
    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    assert _arm(monkeypatch, _engine(shape), ep_size) == cap
    # Unarmed, no layer is recorded either: main's MoE path, no summary line.
    assert len(scores._LAYERS) == (1 if cap else 0)


def test_few_token_kernel_needs_expert_parallelism(monkeypatch):
    assert _arm(monkeypatch, _engine(SHAPE_397B, ep=False), 8) == 0
    assert scores._LAYERS == []


# The validation entries: every leg keeps what the record jobs ran.


def _flag(args: list[str], name: str, default: str) -> str:
    return args[args.index(name) + 1] if name in args else default


def _expert_parallel_size(args: list[str]) -> int:
    """vLLM's expert-parallel size for a leg: its TP x DP x PCP ranks, or 1
    without --enable-expert-parallel."""
    if "--enable-expert-parallel" not in args:
        return 1
    return (
        int(_flag(args, "--tensor-parallel-size", "1"))
        * int(_flag(args, "--data-parallel-size", "1"))
        * int(_flag(args, "--prefill-context-parallel-size", "1"))
    )


def _legs(path: pathlib.Path) -> dict:
    with path.open() as f:
        entry = yaml.safe_load(f)
    if "base_point" not in entry:
        assert "397b" in entry["recipe"]["base"]
        return entry["legs"]
    # A gate takes its legs from its base point; its overrides do not change
    # the layout.
    (point,) = [p for p in POINTS if p.name.startswith(f"p{entry['base_point']:02d}-")]
    overrides = entry["legs"]["decode"].get("serve_args_override", {})
    assert not {"--tensor-parallel-size", "--data-parallel-size"} & set(overrides)
    return _legs(point)


@pytest.mark.parametrize("path", POINTS + GATES, ids=lambda p: p.stem)
def test_validation_legs_keep_the_levers_they_ran(monkeypatch, path):
    legs = _legs(path)
    # Both legs serve the measured checkpoint on TPU7x: the kernel numerics.
    assert hooks.apply_kernel_numerics(_model(SHAPE_397B)) is None
    assert _levers() == ON
    # The decode leg is EP8, where the few-token kernel runs up to 448 rows.
    decode_ep = _expert_parallel_size(legs["decode"]["serve_args"])
    assert decode_ep == 8
    assert _arm(monkeypatch, _engine(SHAPE_397B), decode_ep) == 448
    # The prefill leg is PP8 x TP1, one expert-parallel rank: no expert
    # parallelism, so the kernel does not serve it (as before).
    assert _expert_parallel_size(legs["prefill"]["serve_args"]) == 1
