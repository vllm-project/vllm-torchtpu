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
"""CPU tests for the TPU Beam model.

Pins the checkpoint weight mapping, the registry entry, and the BF16 rounding
points the torch ops share with upstream's Triton kernels
(vllm/models/beam/nvidia/{layernorm,attention_ops}.py), transcribed below as
references.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import ModelRegistry

import vllm_torchtpu.models.vllm.beam.model as beam
from vllm_torchtpu.layers import register_layers

# ---------------------------------------------------------------------------
# Weight mapping
# ---------------------------------------------------------------------------
# Beam checkpoints carry two `gate_proj` names that are not the dense SwiGLU
# gate: the per-expert `mlp.experts.N.gate_proj` (loaded by the MoE runner) and
# the attention output gate `self_attn.gate_proj`. Folding either into
# `mlp.gate_up_proj` silently corrupts the weights while the checkpoint still
# loads.


def _map(merged_qkv_gate: bool, name: str) -> tuple[str, int | str | None]:
    ((mapped, out),) = beam._weights_mapper(merged_qkv_gate).apply(
        [(name, torch.empty(0))]
    )
    return mapped, getattr(out, "shard_id", None)


@pytest.mark.parametrize(
    ("checkpoint_name", "expected_name", "expected_shard"),
    [
        (
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.3.self_attn.qkv_proj.weight",
            "v",
        ),
        (
            "model.layers.3.self_attn.gate_proj.weight",
            "model.layers.3.self_attn.output_gate_proj.weight",
            None,
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
    ],
)
def test_split_layout_mapping(checkpoint_name, expected_name, expected_shard):
    """More TP ranks than KV heads: QKV projection plus a separate gate."""
    assert _map(False, checkpoint_name) == (expected_name, expected_shard)


@pytest.mark.parametrize(
    ("checkpoint_name", "expected_shard"),
    [
        ("model.layers.3.self_attn.q_proj.weight", 0),
        ("model.layers.3.self_attn.k_proj.weight", 1),
        ("model.layers.3.self_attn.v_proj.weight", 2),
        ("model.layers.3.self_attn.gate_proj.weight", 3),
    ],
)
def test_merged_layout_stacks_attention(checkpoint_name, expected_shard):
    """TP up to the KV head count: q/k/v and the output gate in one projection."""
    assert _map(True, checkpoint_name) == (
        "model.layers.3.self_attn.qkv_gate_proj.weight",
        expected_shard,
    )


@pytest.mark.parametrize("merged_qkv_gate", [False, True])
@pytest.mark.parametrize(
    ("checkpoint_name", "expected_name", "expected_shard"),
    [
        (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
    ],
)
def test_dense_mlp_is_fused_in_both_layouts(
    merged_qkv_gate, checkpoint_name, expected_name, expected_shard
):
    assert _map(merged_qkv_gate, checkpoint_name) == (expected_name, expected_shard)


@pytest.mark.parametrize("merged_qkv_gate", [False, True])
@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "model.layers.1.mlp.experts.7.gate_proj.weight",
        "model.layers.1.mlp.experts.7.up_proj.weight",
        "model.layers.1.mlp.experts.7.down_proj.weight",
        "model.layers.1.mlp.experts.7.gate_proj.weight_scale",
        "model.layers.1.self_attn.gate_bias",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
    ],
)
def test_experts_router_and_gate_bias_pass_through(merged_qkv_gate, checkpoint_name):
    assert _map(merged_qkv_gate, checkpoint_name) == (checkpoint_name, None)


@pytest.mark.parametrize("merged_qkv_gate", [False, True])
def test_attention_gate_is_never_a_gate_up_shard(merged_qkv_gate):
    mapped, _ = _map(merged_qkv_gate, "model.layers.1.self_attn.gate_proj.weight")
    assert "gate_up_proj" not in mapped


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_beam_resolves_to_tpu_model():
    """An upstream registry change must not bring back the CUDA model."""
    arch = "BeamForCausalLM"
    previous_model = ModelRegistry.models.get(arch)
    try:
        register_layers()
        assert ModelRegistry.models[arch].load_model_cls() is beam.BeamForCausalLM
    finally:
        if previous_model is None:
            ModelRegistry.models.pop(arch, None)
        else:
            ModelRegistry.models[arch] = previous_model


# ---------------------------------------------------------------------------
# Rounding points, against references transcribed from the Triton kernels
# ---------------------------------------------------------------------------
BF16 = torch.bfloat16


def _boundary_math_reference(x_normed, residual, x_scale, residual_scale):
    """`_boundary_math` after its post-norm affine rounds to the activation dtype."""
    dtype = x_normed.dtype
    x = x_normed.to(torch.float32)
    r = residual.to(dtype).to(torch.float32)
    x = (x * x_scale).to(dtype).to(torch.float32)
    r = (r * residual_scale).to(dtype).to(torch.float32)
    return (x + r).to(dtype)


def _output_gate_kernel_reference(x, gate, bias, heads, dim):
    """`_output_gate_kernel`."""
    dtype = gate.dtype
    rows = x.shape[0]
    g = gate.reshape(rows, heads, dim).to(torch.float32)
    b = bias.to(torch.float32)[None, :, None]
    g = (g + b).to(dtype).to(torch.float32)
    sigmoid = (1.0 / (1.0 + torch.exp(-g))).to(dtype).to(torch.float32)
    y = x.reshape(rows, heads, dim).to(torch.float32) * sigmoid
    return y.to(x.dtype).reshape(rows, heads * dim)


def _fma(a, b, c):
    """FP32 fused multiply-add: one rounding, via an exact FP64 product."""
    return (a.double() * b.double() + c.double()).to(torch.float32)


def _qk_rope_reference(y, positions, cache, rotary_dim):
    """The NeoX `ROTARY` branch of `_qk_norm_rope_kernel` on the rounded norm."""
    dtype = y.dtype
    y = y.to(torch.float32)
    dim = y.shape[-1]
    half = rotary_dim // 2
    c = torch.arange(dim)
    partner = torch.where(c < half, c + half, c - half).clamp(0, dim - 1)
    second = c >= half
    offset = c % half
    other = y[..., partner]
    in_rotary = c < rotary_dim
    cos = torch.where(in_rotary, cache[positions][:, offset], 0.0)[:, None, :]
    sin = torch.where(in_rotary, cache[positions][:, half + offset], 0.0)[:, None, :]
    rotated = torch.where(second, _fma(other, sin, y * cos), _fma(y, cos, -other * sin))
    return torch.where(in_rotary, rotated, y).to(dtype)


def _qk_ssmax_reference(y, positions):
    """The `SSMAX` branch of `_qk_norm_rope_kernel` on the rounded query norm."""
    p = positions.to(torch.float32) + 1
    scale = torch.exp(0.1 * torch.log1p(p / 2048.0))
    return (y.to(torch.float32) * scale[:, None, None]).to(y.dtype)


@pytest.mark.parametrize("layer_id", [0, 1, 25, 51])
@pytest.mark.parametrize("is_attention", [True, False])
def test_depth_scaled_coeffs(layer_id, is_attention):
    num_adds = 2 * (layer_id + 1) + (0 if is_attention else 1)
    denominator = math.sqrt(num_adds)
    assert beam._depth_scaled_coeffs(layer_id, is_attention) == (
        1.0 / denominator,
        math.sqrt(num_adds - 1) / denominator,
    )


@pytest.mark.parametrize("layer_id", [0, 7, 51])
@pytest.mark.parametrize("is_attention", [True, False])
def test_scaled_residual_add_matches_boundary_math(layer_id, is_attention):
    torch.manual_seed(layer_id)
    x_normed = torch.randn(33, 512).to(BF16)
    residual = (torch.randn(33, 512) * 8).to(BF16)
    scales = beam._depth_scaled_coeffs(layer_id, is_attention)
    actual = beam._scaled_residual_add(x_normed, residual, scales)
    expected = _boundary_math_reference(x_normed, residual, *scales)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_output_gate_matches_kernel(packed):
    torch.manual_seed(0)
    rows, heads, dim = 17, 6, 128
    x = torch.randn(rows, heads * dim).to(BF16)
    # In the merged layout the gate is a column slice of the packed projection.
    gate = (torch.randn(rows, (2 if packed else 1) * heads * dim) * 8).to(BF16)
    gate = gate[:, -heads * dim :]
    bias = torch.randn(heads).to(BF16)
    attention = SimpleNamespace(num_heads=heads, head_dim=dim, gate_bias=bias)
    actual = beam.BeamAttention._apply_output_gate(attention, x, gate)
    expected = _output_gate_kernel_reference(x, gate, bias, heads, dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ssmax_matches_kernel():
    torch.manual_seed(0)
    y = torch.randn(17, 6, 128).to(BF16)
    positions = torch.tensor([0, 1, 2047, 2048, 65535, 1048575] + list(range(11)))
    actual = beam._ssmax(y, positions)
    torch.testing.assert_close(
        actual, _qk_ssmax_reference(y, positions), rtol=0, atol=0
    )


@pytest.mark.parametrize("rotary_dim", [128, 64])
def test_rotate_neox_matches_kernel(rotary_dim):
    torch.manual_seed(0)
    rows, heads, dim = 17, 6, 128
    y = torch.randn(rows, heads, dim).to(BF16)
    positions = torch.arange(rows) * 3
    angle = torch.randn(rows * 3, rotary_dim // 2)
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    # As BeamAttention.forward reads the FP32 cos/sin cache.
    cos, sin = cache.index_select(0, positions).unsqueeze(-2).chunk(2, dim=-1)
    actual = beam._rotate_neox(y, cos, sin, rotary_dim)
    expected = _qk_rope_reference(y, positions, cache, rotary_dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
