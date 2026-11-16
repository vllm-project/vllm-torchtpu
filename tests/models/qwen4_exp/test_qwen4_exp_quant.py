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
"""Tests for the Qwen4Exp W8A8 FP8 attention projections."""

from __future__ import annotations

import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)

from vllm_torchtpu.models.qwen4_exp import quant
from vllm_torchtpu.models.qwen4_exp.ops import fp8_linear

HIDDEN = 256
OUT = 384


@pytest.fixture(autouse=True)
def tensor_parallel_group(monkeypatch):
    from vllm.distributed import parallel_state

    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(parallel_state, "_TP", group)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_rank", lambda: 0)


@pytest.fixture
def chip(monkeypatch):
    def set_chip(name: str | None) -> None:
        version = None if name is None else pltpu.ChipVersion(name)
        monkeypatch.setattr(quant, "_chip_version", lambda: version)

    set_chip("7x")
    return set_chip


def make_linear(in_features=HIDDEN, out_features=OUT):
    return ReplicatedLinear(
        in_features,
        out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=None,
        prefix="self_attn.q_proj",
        return_bias=False,
        disable_tp=True,
    )


def sqnr_db(result: np.ndarray, reference: np.ndarray) -> float:
    noise = np.linalg.norm(result - reference)
    return float(20 * np.log10(np.linalg.norm(reference) / max(noise, 1e-30)))


def random_inputs(rows, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(OUT, HIDDEN, generator=generator) * 0.05).to(torch.bfloat16)
    # A few outlier channels, as hidden states have, stress per-token scales.
    x = torch.randn(rows, HIDDEN, generator=generator)
    x[:, :4] *= 20
    return weight, x.to(torch.bfloat16)


def test_weight_quantization_layout_and_scale():
    weight, _ = random_inputs(1)
    weight[3] = 0
    w_q, scale = quant.quantize_weight_per_channel(weight)
    assert w_q.dtype == torch.float8_e4m3fn
    assert w_q.shape == (HIDDEN, OUT)  # (k, n) for the contraction
    assert scale.dtype == torch.float32 and scale.shape == (OUT,)
    assert scale[3] > 0 and torch.all(w_q[:, 3].float() == 0)
    amax = w_q.float().abs().amax(dim=0)
    assert torch.all((amax == quant.FP8_MAX) | (amax == 0))
    dequant = (w_q.float() * scale).t()
    assert sqnr_db(dequant.numpy(), weight.float().numpy()) > 28


def test_tiny_rows_stay_finite():
    # Normal float32 values whose absmax / 448 would be subnormal (flushed to
    # zero on TPU) or whose 448 / absmax would overflow float32.
    weight, x = random_inputs(4)
    weight, x = weight.float(), x.float()
    weight[1], x[1] = 0, 0
    weight[1, :3] = x[1, :3] = 1e-38
    weight[2, :3] = x[2, :3] = 3e-36
    w_q, scale = quant.quantize_weight_per_channel(weight)
    assert torch.all(torch.isfinite(w_q.float())) and torch.all(torch.isfinite(scale))
    assert torch.all(scale >= torch.finfo(torch.float32).tiny)
    out = fp8_linear.fp8_linear_jax(
        jnp.asarray(x.numpy(), dtype=jnp.bfloat16),
        jnp.asarray(w_q.float().numpy(), dtype=jnp.float8_e4m3fn),
        jnp.asarray(scale.numpy()),
    )
    assert np.all(np.isfinite(np.asarray(out, dtype=np.float32)))


def test_weight_quantization_rejects_non_matrix():
    with pytest.raises(ValueError, match="2-D"):
        quant.quantize_weight_per_channel(torch.zeros(4, dtype=torch.bfloat16))


@pytest.mark.parametrize("rows", [1, 16, 64])
def test_fp8_linear_matches_bf16(rows):
    weight, x = random_inputs(rows)
    w_q, scale = quant.quantize_weight_per_channel(weight)
    result = fp8_linear.fp8_linear_jax(
        jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16),
        jnp.asarray(w_q.float().numpy(), dtype=jnp.float8_e4m3fn),
        jnp.asarray(scale.numpy()),
    )
    assert result.dtype == jnp.bfloat16 and result.shape == (rows, OUT)
    reference = x.float().numpy() @ weight.float().numpy().T
    assert sqnr_db(np.asarray(result, dtype=np.float32), reference) > 25


def test_fp8_linear_zero_rows_stay_finite():
    weight, x = random_inputs(8)
    x[2] = 0
    w_q, scale = quant.quantize_weight_per_channel(weight)
    result = np.asarray(
        fp8_linear.fp8_linear_jax(
            jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(w_q.float().numpy(), dtype=jnp.float8_e4m3fn),
            jnp.asarray(scale.numpy()),
        ),
        dtype=np.float32,
    )
    assert np.all(np.isfinite(result)) and np.all(result[2] == 0)


def dot_operand_dtypes(rows, w8a16_max_rows):
    jaxpr = jax.make_jaxpr(fp8_linear.fp8_linear_jax, static_argnums=3)(
        jax.ShapeDtypeStruct((rows, HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((HIDDEN, OUT), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((OUT,), jnp.float32),
        w8a16_max_rows,
    )
    dots = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "dot_general"]
    assert len(dots) == 1
    return [v.aval.dtype for v in dots[0].invars]


def test_w8a8_feeds_fp8_to_both_matmul_operands():
    assert dot_operand_dtypes(64, 0) == [jnp.float8_e4m3fn] * 2


def test_empty_bucket_takes_w8a8_branch():
    assert dot_operand_dtypes(0, 64) == [jnp.float8_e4m3fn] * 2
    weight, _ = random_inputs(1)
    w_q, scale = quant.quantize_weight_per_channel(weight)
    out = fp8_linear.fp8_linear_jax(
        jnp.zeros((0, HIDDEN), jnp.bfloat16),
        jnp.asarray(w_q.float().numpy(), dtype=jnp.float8_e4m3fn),
        jnp.asarray(scale.numpy()),
        64,
    )
    assert out.shape == (0, OUT)


def test_small_buckets_select_w8a16():
    assert dot_operand_dtypes(16, 16) == [jnp.bfloat16] * 2
    assert dot_operand_dtypes(17, 16) == [jnp.float8_e4m3fn] * 2
    default = quant.DEFAULT_W8A16_MAX_ROWS
    assert dot_operand_dtypes(default, default) == [jnp.bfloat16] * 2
    assert dot_operand_dtypes(default + 1, default) == [jnp.float8_e4m3fn] * 2


@pytest.mark.parametrize(
    ("name", "native"),
    [(None, False), ("v5e", False), ("v6e", False), ("7", True), ("7x", True)],
)
def test_native_fp8_self_selection(chip, name, native):
    chip(name)
    assert quant.has_native_fp8_matmul() is native
    layer = make_linear()
    assert quant.enable_fp8_projection(layer) is native
    expected = quant.Qwen4ExpFp8LinearMethod if native else UnquantizedLinearMethod
    assert type(layer.quant_method) is expected


def test_enable_rejects_quantized_layer(chip):
    layer = make_linear()
    assert quant.enable_fp8_projection(layer)
    with pytest.raises(ValueError, match="unquantized BF16"):
        quant.enable_fp8_projection(layer)


def test_bf16_checkpoint_load_and_quantize(chip):
    layer = make_linear()
    quant.enable_fp8_projection(layer, w8a16_max_rows=8)
    assert layer.quant_method.w8a16_max_rows == 8
    weight, _ = random_inputs(1)
    # Loading is untouched: the BF16 parameter takes the checkpoint tensor.
    assert layer.weight.dtype == torch.bfloat16
    layer.weight.weight_loader(layer.weight, weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (HIDDEN, OUT)
    # The FP8 weight owns its buffer; the BF16 storage is not kept alive.
    assert layer.weight.untyped_storage().nbytes() == layer.weight.nbytes
    assert hasattr(layer.weight, "weight_loader")
    assert layer.weight_scale.shape == (OUT,)
    w_q, scale = quant.quantize_weight_per_channel(weight)
    assert torch.equal(layer.weight.float(), w_q.float())
    assert torch.equal(layer.weight_scale, scale)
    # Reload paths re-enter the hook; it must not quantize twice.
    before = layer.weight
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight is before


def test_keep_vllm_layout_stores_n_by_k(chip):
    from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR

    layer = make_linear()
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    quant.enable_fp8_projection(layer)
    weight, _ = random_inputs(1)
    layer.weight.weight_loader(layer.weight, weight)
    layer.quant_method.process_weights_after_loading(layer)
    w_q, scale = quant.quantize_weight_per_channel(weight, transpose=False)
    assert layer.weight.shape == (OUT, HIDDEN)
    assert torch.equal(layer.weight.float(), w_q.float())
    assert torch.equal(layer.weight_scale, scale)


def test_rejects_non_bf16_weight(chip):
    layer = ReplicatedLinear(
        HIDDEN,
        OUT,
        bias=False,
        params_dtype=torch.float32,
        quant_config=None,
        prefix="self_attn.o_proj",
        disable_tp=True,
    )
    quant.enable_fp8_projection(layer)
    with pytest.raises(ValueError, match="expects a BF16 weight"):
        layer.quant_method.process_weights_after_loading(layer)


@pytest.fixture
def tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU")
    return torch.device("tpu")


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("rows", "w8a16_max_rows"), [(4, 0), (512, 0), (16, 32), (64, None), (128, None)]
)
def test_projection_forward_tpu(chip, tpu, rows, w8a16_max_rows):
    layer = make_linear()
    if w8a16_max_rows is None:
        quant.enable_fp8_projection(layer)
    else:
        quant.enable_fp8_projection(layer, w8a16_max_rows=w8a16_max_rows)
    weight, x = random_inputs(rows, seed=rows)
    layer.weight.weight_loader(layer.weight, weight)
    layer = layer.to(tpu)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight.untyped_storage().nbytes() == layer.weight.nbytes
    result = layer(x.to(tpu)).cpu().float().numpy()
    reference = x.float().numpy() @ weight.float().numpy().T
    assert sqnr_db(result, reference) > 25


@pytest.mark.tpu_test
@pytest.mark.parametrize("keep_layout", [False, True])
def test_projection_edge_inputs_tpu(chip, tpu, keep_layout):
    from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR

    layer = make_linear()
    if keep_layout:
        setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    quant.enable_fp8_projection(layer)
    weight, x = random_inputs(128, seed=5)
    weight[1] = 0
    weight[1, :3] = 1e-38  # subnormal on TPU after scaling
    x[1], x[2] = 0, 0
    x[1, :3] = 1e-38
    x[2, :3] = 3e-36
    layer.weight.weight_loader(layer.weight, weight)
    layer = layer.to(tpu)
    layer.quant_method.process_weights_after_loading(layer)
    assert torch.all(torch.isfinite(layer.weight.cpu().float()))
    assert torch.all(torch.isfinite(layer.weight_scale.cpu()))
    result = layer(x.to(tpu)).cpu().float()
    assert torch.all(torch.isfinite(result))
    reference = x.float().numpy() @ weight.float().numpy().T
    assert sqnr_db(result.numpy(), reference) > 25
    empty = layer(x[:0].to(tpu)).cpu()
    assert empty.shape == (0, OUT)


@pytest.mark.tpu_test
def test_projection_traces_under_torch_compile_tpu(chip, tpu):
    layer = make_linear()
    quant.enable_fp8_projection(layer)
    weight, x = random_inputs(512, seed=3)
    layer.weight.weight_loader(layer.weight, weight)
    layer = layer.to(tpu)
    layer.quant_method.process_weights_after_loading(layer)
    eager = layer(x.to(tpu)).cpu()
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x.to(tpu)).cpu(), eager, rtol=0, atol=0)


@pytest.mark.tpu_test
def test_w8a8_lowers_to_fp8_matmul_tpu(tpu):
    del tpu
    if not quant.has_native_fp8_matmul():
        pytest.skip("No validated native FP8 MXU path on this TPU")
    hlo = (
        jax.jit(fp8_linear.fp8_linear_jax)
        .lower(
            jax.ShapeDtypeStruct((512, HIDDEN), jnp.bfloat16),
            jax.ShapeDtypeStruct((HIDDEN, OUT), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((OUT,), jnp.float32),
        )
        .compile()
        .as_text()
    )
    # Operand element types of every matmul, fused computations included
    # (instruction names are unique within an HLO module).
    types = dict(re.findall(r"%([\w.\-]+) = (\w+)\[", hlo))
    matmuls = [
        [types[name] for name in re.findall(r"%([\w.\-]+)", m.group(1))]
        for m in re.finditer(r"\b(?:convolution|dot)\(([^)]*)\)", hlo)
    ]
    assert matmuls == [["f8e4m3fn", "f8e4m3fn"]], hlo
    # The FP8 weight enters the MXU as is, never widened first.
    assert f"convert(f8e4m3fn[{HIDDEN},{OUT}]" not in hlo, hlo
