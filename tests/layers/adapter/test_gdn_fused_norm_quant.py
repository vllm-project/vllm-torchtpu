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
"""TPU_GDN_FUSED_NORM_QUANT: the GDN output projection gives the same output
with the fused gated norm and quantization on and off.

The torch<->JAX op bridge is replaced by a CPU bridge that runs the same JAX
functions with the Pallas kernel in interpret mode.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated

from vllm_torchtpu.kernels.quantized_matmul import util as quantized_matmul_util
from vllm_torchtpu.layers.adapter import gated_norm_quant_op
from vllm_torchtpu.layers.adapter.custom_ops import gdn_attention_op
from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
    VllmGatedDeltaNetAttention,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)

HEADS = 4
HEAD_DIM = 128
HIDDEN = 256
EPS = 1e-6


def _torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    tensor = tensor.detach()
    if tensor.dtype == torch.float8_e4m3fn:
        raw = tensor.view(torch.uint8).numpy()
        return jnp.asarray(raw.view(ml_dtypes.float8_e4m3fn))
    if tensor.dtype == torch.bfloat16:
        return jnp.asarray(tensor.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(tensor.numpy())


def _jax_to_torch(array: jax.Array) -> torch.Tensor:
    if array.dtype == jnp.float8_e4m3fn:
        raw = np.array(array).view(np.uint8)
        return torch.from_numpy(raw).view(torch.float8_e4m3fn)
    if array.dtype == jnp.bfloat16:
        return torch.from_numpy(np.array(array.astype(jnp.float32))).bfloat16()
    return torch.from_numpy(np.array(array))


def _cpu_jax_op(name, fn):
    del name

    def run(*tensors):
        with pltpu.force_tpu_interpret_mode():
            outputs = jax.jit(fn)(*(_torch_to_jax(t) for t in tensors))
        return jax.tree.map(_jax_to_torch, outputs)

    run.register_fake = lambda fake: None
    return run


@pytest.fixture(autouse=True)
def cpu_op_bridge(monkeypatch):
    monkeypatch.setattr(gated_norm_quant_op.pallas, "jax_op", _cpu_jax_op)
    monkeypatch.setattr(gated_norm_quant_op, "_op_cache", {})
    monkeypatch.setenv("TPU_GDN_FUSED_NORM_QUANT", "1")
    monkeypatch.setenv("TPU_GATED_NORM_TILE", "32")


class _Fp8OutProj(torch.nn.Module):
    """A per-channel FP8 RowParallelLinear after VllmFp8LinearMethodTPU
    loading: weight [in, out], scale [out], unfused forward through
    xla_quantized_matmul like VllmFp8LinearMethodTPU.apply."""

    def __init__(self, seed: int = 0):
        super().__init__()
        weight = jax.random.normal(jax.random.key(seed), (HIDDEN, HEADS * HEAD_DIM))
        w_q, w_scale = quantized_matmul_util.quantize_tensor(
            weight * 0.05, jnp.float8_e4m3fn
        )
        self.weight = torch.nn.Parameter(_jax_to_torch(w_q.T), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            _jax_to_torch(w_scale.reshape(-1)), requires_grad=False
        )
        setattr(self, WEIGHT_FLIPPED_ATTR, True)
        self.bias = None
        self.input_is_parallel = True
        self.reduce_results = True
        self.tp_size = 1
        self.quant_method = SimpleNamespace(w8a16_max_rows=0)
        self._unfused_matmul = _cpu_jax_op(
            "unfused", quantized_matmul_util.xla_quantized_matmul
        )

    def forward(self, x):
        return self._unfused_matmul(x, self.weight, self.weight_scale), None


def _gdn_layer() -> SimpleNamespace:
    with set_current_vllm_config(VllmConfig()):
        norm = RMSNormGated(
            HEAD_DIM,
            eps=EPS,
            group_size=None,
            norm_before_gate=True,
            activation="silu",
            dtype=torch.bfloat16,
        )
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(HEAD_DIM))
    return SimpleNamespace(
        num_v_heads=HEADS,
        tp_size=1,
        head_v_dim=HEAD_DIM,
        norm=norm,
        out_proj=_Fp8OutProj(),
        conv1d=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(8, 1, 4))),
    )


def _load_gdn_layer(monkeypatch) -> SimpleNamespace:
    """A GDN layer after VllmGatedDeltaNetAttention's post-load step."""
    monkeypatch.setattr(gdn_attention_op, "fresh_buffer", lambda w: w.data)
    layer = _gdn_layer()
    VllmGatedDeltaNetAttention.process_weights_after_loading(layer, torch.bfloat16)
    return layer


def _inputs(num_tokens: int):
    torch.manual_seed(num_tokens)
    core_attn_out = torch.randn(num_tokens, HEADS * HEAD_DIM).bfloat16()
    z = (2.0 * torch.randn(num_tokens, HEADS, HEAD_DIM)).bfloat16()
    return core_attn_out, z


def _output_projection(layer, core_attn_out, z):
    with set_current_vllm_config(VllmConfig()):
        return VllmGatedDeltaNetAttention._apply_output_projection(
            layer, core_attn_out, z, core_attn_out.shape[0]
        )


@pytest.mark.parametrize("num_tokens", [4, 32, 48, 64, 80, 96])
def test_fused_matches_unfused(monkeypatch, num_tokens):
    layer = _load_gdn_layer(monkeypatch)
    core_attn_out, z = _inputs(num_tokens)
    fused = _output_projection(layer, core_attn_out, z)
    assert len(gated_norm_quant_op._op_cache) == 2  # the fused ops ran

    monkeypatch.setenv("TPU_GDN_FUSED_NORM_QUANT", "0")
    unfused = _output_projection(layer, core_attn_out, z)
    assert fused.shape == unfused.shape == (num_tokens, HIDDEN)
    assert fused.dtype == unfused.dtype == torch.bfloat16
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)


def test_gated_norm_quant_fake_returns_2d_rows(monkeypatch):
    registered_fakes = {}

    class _CapturedOp:
        def __init__(self, name):
            self._name = name

        def __call__(self, *args):
            raise AssertionError("fake test should not execute op")

        def register_fake(self, fake_fn):
            registered_fakes[self._name] = fake_fn

    def _capture_jax_op(name, fn):
        del fn
        return _CapturedOp(name)

    monkeypatch.setattr(gated_norm_quant_op.pallas, "jax_op", _capture_jax_op)
    monkeypatch.setattr(gated_norm_quant_op, "_op_cache", {})
    # Sub-32 tile values normalize to the 32-sublane 2D FP8 minimum in op naming.
    monkeypatch.setenv("TPU_GATED_NORM_TILE", "16")
    gated_norm_quant_op.prebuild_ops(EPS)
    fake_fn = registered_fakes["pallas::gdn_gated_norm_quant_e1em06_t32"]
    attn_out = torch.empty((12, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    gate = torch.empty_like(attn_out)
    weight = torch.empty((HEAD_DIM,), dtype=torch.bfloat16)
    q_fake, s_fake = fake_fn(attn_out, gate, weight)
    assert q_fake.shape == (12, HEADS * HEAD_DIM)
    assert q_fake.dtype == torch.float8_e4m3fn
    assert s_fake.shape == (12, 1)
    assert s_fake.dtype == torch.float32


@pytest.mark.parametrize(
    ("reduce_results", "tp_size", "expect_all_reduce"),
    [(True, 2, True), (False, 2, False), (True, 1, False)],
)
def test_all_reduce_matches_row_parallel_linear(
    monkeypatch, reduce_results, tp_size, expect_all_reduce
):
    calls = []

    def _record_all_reduce(tensor):
        calls.append(tensor.shape)
        return tensor

    monkeypatch.setattr(
        gated_norm_quant_op, "tensor_model_parallel_all_reduce", _record_all_reduce
    )
    layer = _load_gdn_layer(monkeypatch)
    layer.out_proj.reduce_results = reduce_results
    layer.out_proj.tp_size = tp_size
    core_attn_out, z = _inputs(8)
    _output_projection(layer, core_attn_out, z)
    assert calls == ([(8, HIDDEN)] if expect_all_reduce else [])


def _set_gate_first(layer):
    layer.norm.norm_before_gate = False


def _set_sigmoid_gate(layer):
    layer.norm.activation = "sigmoid"


def _set_grouped_norm(layer):
    layer.norm.group_size = 64


def _set_bf16_weight(layer):
    layer.out_proj.weight = torch.nn.Parameter(
        layer.out_proj.weight.bfloat16(), requires_grad=False
    )


def _set_blockwise_scale(layer):
    layer.out_proj.weight_scale = torch.nn.Parameter(
        torch.ones(1, 4, 1, HIDDEN), requires_grad=False
    )


def _set_vllm_layout(layer):
    setattr(layer.out_proj, KEEP_VLLM_LAYOUT_ATTR, True)


def _set_bias(layer):
    layer.out_proj.bias = torch.nn.Parameter(torch.zeros(HIDDEN))


def _set_w8a16_bucket(layer):
    layer.out_proj.quant_method.w8a16_max_rows = 8


def _set_unaligned_head_dim(layer):
    layer.norm.weight = torch.nn.Parameter(torch.ones(64, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "make_unsupported",
    [
        _set_gate_first,
        _set_sigmoid_gate,
        _set_grouped_norm,
        _set_bf16_weight,
        _set_blockwise_scale,
        _set_vllm_layout,
        _set_bias,
        _set_w8a16_bucket,
        _set_unaligned_head_dim,
    ],
)
def test_unsupported_layers_keep_unfused_path(make_unsupported):
    layer = _gdn_layer()
    assert gated_norm_quant_op.can_fuse(layer.norm, layer.out_proj, 8)
    make_unsupported(layer)
    assert not gated_norm_quant_op.can_fuse(layer.norm, layer.out_proj, 8)


def test_non_bf16_activation_keeps_unfused_path():
    layer = _gdn_layer()
    assert not gated_norm_quant_op.can_fuse(
        layer.norm, layer.out_proj, 8, torch.float32
    )
    assert not gated_norm_quant_op.can_fuse(
        layer.norm, layer.out_proj, 8, torch.float16
    )


def test_prebuild_ops_validates_eps_and_tile(monkeypatch):
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        gated_norm_quant_op.prebuild_ops(0.0)
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        gated_norm_quant_op.prebuild_ops(float("nan"))
    monkeypatch.setenv("TPU_GATED_NORM_TILE", "-8")
    with pytest.raises(ValueError, match="max_tile must be positive"):
        gated_norm_quant_op.prebuild_ops(EPS)


def test_w8a16_requires_warmup_when_compiling(monkeypatch):
    layer = _gdn_layer()
    layer.out_proj.quant_method.w8a16_max_rows = 8
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        gated_norm_quant_op.shape_variants, "aot_cache_tag", lambda: None
    )
    assert not gated_norm_quant_op.can_fuse(layer.norm, layer.out_proj, 16)
    monkeypatch.setattr(
        gated_norm_quant_op.shape_variants, "aot_cache_tag", lambda: "warmup"
    )
    assert gated_norm_quant_op.can_fuse(layer.norm, layer.out_proj, 16)


def test_w8a16_applies_only_to_covered_buckets():
    layer = _gdn_layer()
    layer.out_proj.quant_method.w8a16_max_rows = 8
    assert not gated_norm_quant_op.can_fuse(layer.norm, layer.out_proj, 8)
    assert gated_norm_quant_op.can_fuse(layer.norm, layer.out_proj, 16)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_GDN_FUSED_NORM_QUANT")
    layer = _load_gdn_layer(monkeypatch)
    assert not gated_norm_quant_op.can_fuse(layer.norm, layer.out_proj, 8)
    core_attn_out, z = _inputs(8)
    _output_projection(layer, core_attn_out, z)
    assert gated_norm_quant_op._op_cache == {}


def _fail_op_creation(name, fn):
    raise AssertionError(f"{name} was created inside the forward")


def test_weight_loading_builds_ops_and_forward_only_looks_them_up(monkeypatch):
    # Dynamo cannot trace pallas.jax_op (it calls inspect.signature), so the
    # traced forward must not create ops.
    layer = _load_gdn_layer(monkeypatch)
    assert set(gated_norm_quant_op._op_cache) == {
        "pallas::gdn_gated_norm_quant_e1em06_t32",
        "pallas::gdn_fp8_matmul_on_quantized_rows",
    }
    monkeypatch.setattr(gated_norm_quant_op.pallas, "jax_op", _fail_op_creation)
    for num_tokens in (4, 32, 48, 64):
        core_attn_out, z = _inputs(num_tokens)
        output = _output_projection(layer, core_attn_out, z)
        assert output.shape == (num_tokens, HIDDEN)
    assert len(gated_norm_quant_op._op_cache) == 2


def test_forward_without_prebuilt_ops_raises(monkeypatch):
    monkeypatch.setattr(gated_norm_quant_op.pallas, "jax_op", _fail_op_creation)
    layer = _gdn_layer()
    core_attn_out, z = _inputs(8)
    with pytest.raises(RuntimeError, match="prebuild_ops"):
        _output_projection(layer, core_attn_out, z)
