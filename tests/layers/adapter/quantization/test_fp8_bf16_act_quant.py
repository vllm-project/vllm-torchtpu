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
"""Tests for FP8 activation quantization scale dtype (TPU_ACT_QUANT_SCALE_DTYPE)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.quantized_matmul import util
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
from vllm_torchtpu.layers.adapter.quantization import fp8
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantLinearConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    resolve_online_fp8,
)
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import attach_online_fp8

_K, _N = 256, 128


def _weights(seed: int = 0) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((_K, _N)).astype(np.float32)
    scale = np.abs(w).max(axis=0) / 448.0
    w_q = jnp.asarray(w / scale, dtype=jnp.float8_e4m3fn)
    return w_q, jnp.asarray(scale, dtype=jnp.float32)


def test_quantize_block_bf16_hlo_and_roundtrip() -> None:
    x = jax.random.normal(jax.random.key(0), (64, 512), dtype=jnp.bfloat16)

    for fp32_scale_dtype in (None, "float32"):
        lowered_fp32 = jax.jit(
            lambda a, sd=fp32_scale_dtype: util.quantize_block(
                a, axis=1, target_dtype=jnp.float8_e4m3fn, scale_dtype=sd
            )
        ).lower(x)
        assert "f32[64,512]{1,0} divide(" in lowered_fp32.as_text("hlo")

    lowered_bf16 = jax.jit(
        lambda a: util.quantize_block(
            a, axis=1, target_dtype=jnp.float8_e4m3fn, scale_dtype="bfloat16"
        )
    ).lower(x)
    hlo_text = lowered_bf16.as_text("hlo")
    assert "bf16[64,512]{1,0} divide(" in hlo_text
    assert "f32[64,512]{1,0} divide(" not in hlo_text

    x_q_fp32, x_s_fp32 = util.quantize_block(
        x, axis=1, target_dtype=jnp.float8_e4m3fn, scale_dtype="float32"
    )
    x_q, x_s = util.quantize_block(
        x, axis=1, target_dtype=jnp.float8_e4m3fn, scale_dtype="bfloat16"
    )
    assert x_q.dtype == jnp.float8_e4m3fn
    assert x_s.dtype == jnp.float32
    assert jnp.array_equal(x_s, x_s.astype(jnp.bfloat16).astype(jnp.float32))
    assert jnp.allclose(x_s, x_s_fp32, rtol=2**-7, atol=0.0)

    x_deq_fp32 = x_q_fp32.astype(jnp.float32) * x_s_fp32
    x_deq = x_q.astype(jnp.float32) * x_s
    rel_err = jnp.mean(jnp.abs(x.astype(jnp.float32) - x_deq)) / jnp.mean(
        jnp.abs(x.astype(jnp.float32))
    )
    assert float(rel_err) < 0.05

    # Compare bf16-division output directly against the fp32-division path:
    # double rounding can shift at most 1 fp8_e4m3fn ULP plus 1 bf16 scale ULP,
    # and mean drift across the tensor stays << 1%.
    fp8_eps = float(jnp.finfo(jnp.float8_e4m3fn).eps)
    bf16_eps = float(jnp.finfo(jnp.bfloat16).eps)
    assert jnp.allclose(
        x_deq,
        x_deq_fp32,
        rtol=fp8_eps + bf16_eps,
        atol=(2**-9) * float(jnp.max(x_s_fp32)),
    )
    bf16_vs_fp32_rel_err = jnp.mean(jnp.abs(x_deq - x_deq_fp32)) / jnp.mean(
        jnp.abs(x_deq_fp32)
    )
    assert float(bf16_vs_fp32_rel_err) < 0.01


def test_quantize_block_edge_cases() -> None:
    # Zero and near-subnormal bf16 rows must not produce NaN or Inf.
    x_edge = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1e-37, -1e-37, 0.0, 0.0],
            [1.0, -2.0, 4.0, -8.0],
        ],
        dtype=jnp.bfloat16,
    )
    q, s = util.quantize_block(
        x_edge, axis=1, target_dtype=jnp.float8_e4m3fn, scale_dtype="bfloat16"
    )
    assert not jnp.isnan(q.astype(jnp.float32)).any()
    assert not jnp.isnan(s).any()
    assert jnp.all(q[0] == 0)
    assert jnp.all(q[1] == 0)

    # float16 small-magnitude rows (1e-4 / 448 underflows fp16) still divide in fp32.
    x_fp16 = (
        jax.random.uniform(jax.random.key(1), (8, 128), minval=0.5, maxval=1.0) * 1e-4
    ).astype(jnp.float16)
    q_fp16, s_fp16 = util.quantize_block(
        x_fp16, axis=1, target_dtype=jnp.float8_e4m3fn, scale_dtype="bfloat16"
    )
    deq_fp16 = q_fp16.astype(jnp.float32) * s_fp16
    assert not jnp.isnan(deq_fp16).any()
    assert jnp.allclose(deq_fp16, x_fp16.astype(jnp.float32), rtol=0.1, atol=1e-5)

    # Integer quantization ignores bfloat16 scale_dtype.
    q_i8_off, s_i8_off = util.quantize_block(
        x_edge[2:], axis=1, target_dtype=jnp.int8, scale_dtype="float32"
    )
    q_i8_on, s_i8_on = util.quantize_block(
        x_edge[2:], axis=1, target_dtype=jnp.int8, scale_dtype="bfloat16"
    )
    assert jnp.array_equal(q_i8_off, q_i8_on)
    assert jnp.array_equal(s_i8_off, s_i8_on)

    with pytest.raises(ValueError, match="Unsupported scale_dtype"):
        util.quantize_block(
            x_edge, axis=1, target_dtype=jnp.float8_e4m3fn, scale_dtype="float16"
        )


@pytest.mark.parametrize(
    "env_val,expected",
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("bfloat16", "bfloat16"),
        ("BFLOAT16", "bfloat16"),
        ("float32", "float32"),
    ],
)
def test_tpu_act_quant_scale_dtype_env_parsing(
    monkeypatch: pytest.MonkeyPatch, env_val: str | None, expected: str | None
) -> None:
    if env_val is None:
        monkeypatch.delenv("TPU_ACT_QUANT_SCALE_DTYPE", raising=False)
    else:
        monkeypatch.setenv("TPU_ACT_QUANT_SCALE_DTYPE", env_val)
    assert expected == envs.TPU_ACT_QUANT_SCALE_DTYPE


@pytest.mark.parametrize("bad_val", ["1", "true", "float16", "fp8"])
def test_tpu_act_quant_scale_dtype_env_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, bad_val: str
) -> None:
    monkeypatch.setenv("TPU_ACT_QUANT_SCALE_DTYPE", bad_val)
    with pytest.raises(ValueError, match="Invalid value"):
        _ = envs.TPU_ACT_QUANT_SCALE_DTYPE


@pytest.mark.parametrize("scale_dtype", [None, "float32", "bfloat16"])
def test_quantized_matmul_jax_forwards_act_quant_scale_dtype(
    monkeypatch: pytest.MonkeyPatch, scale_dtype: str | None
) -> None:
    seen: dict[str, str | None] = {}
    real = linear_common.xla_quantized_matmul

    def spy(x, w_q, w_s, quantize_activation=True, *, act_quant_scale_dtype=None):
        seen["act_quant_scale_dtype"] = act_quant_scale_dtype
        return real(
            x,
            w_q,
            w_s,
            quantize_activation=quantize_activation,
            act_quant_scale_dtype=act_quant_scale_dtype,
        )

    monkeypatch.setattr(linear_common, "xla_quantized_matmul", spy)
    w_q, w_s = _weights()
    out = _quantized_matmul_jax(
        jnp.ones((16, _K), jnp.bfloat16),
        w_q,
        w_s,
        act_quant_scale_dtype=scale_dtype,
    )
    assert out.shape == (16, _N)
    assert out.dtype == jnp.bfloat16
    assert seen["act_quant_scale_dtype"] == scale_dtype


@pytest.mark.parametrize(
    "env_val,expected_cfg,expected_method",
    [
        (None, None, "float32"),
        ("", None, "float32"),
        ("bfloat16", "bfloat16", "bfloat16"),
        ("float32", "float32", "float32"),
    ],
)
def test_fp8_linear_method_uses_linear_config_without_reading_envs(
    monkeypatch: pytest.MonkeyPatch,
    env_val: str | None,
    expected_cfg: str | None,
    expected_method: str,
) -> None:
    if env_val is None:
        monkeypatch.delenv("TPU_ACT_QUANT_SCALE_DTYPE", raising=False)
    else:
        monkeypatch.setenv("TPU_ACT_QUANT_SCALE_DTYPE", env_val)

    mock_linear = MagicMock(spec=LinearBase)
    mock_linear.output_size = _N
    linear_config = VllmQuantLinearConfig(None, mock_linear)
    assert linear_config.act_quant_scale_dtype == expected_cfg

    # Set an invalid env value afterwards: if VllmFp8LinearMethodTPU read envs
    # directly instead of linear_config, constructing it would raise ValueError.
    monkeypatch.setenv("TPU_ACT_QUANT_SCALE_DTYPE", "invalid_if_envs_read")
    method = VllmFp8LinearMethodTPU(
        Fp8Config(is_checkpoint_fp8_serialized=True), linear_config=linear_config
    )
    assert method.act_quant_scale_dtype == expected_method

    seen: dict[str, str | None] = {}

    def fake_quantized_matmul(
        x, w_q, w_s, w8a16_max_rows=0, act_quant_scale_dtype=None
    ):
        del w_s, w8a16_max_rows
        seen["act_quant_scale_dtype"] = act_quant_scale_dtype
        return torch.zeros((x.shape[0], w_q.shape[-1]), dtype=x.dtype)

    monkeypatch.setattr(fp8, "quantized_matmul", fake_quantized_matmul)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((_K, _N), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((_N,), dtype=torch.float32), requires_grad=False
    )
    method.apply(layer, torch.ones((16, _K), dtype=torch.bfloat16))
    assert seen["act_quant_scale_dtype"] == expected_method


def test_resolve_online_fp8_forwards_linear_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPU_ACT_QUANT_SCALE_DTYPE", "bfloat16")
    monkeypatch.setenv("TPU_FP8_LINEAR_W8A16_MAX_ROWS", "32")
    config = VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["proj"],
        }
    )
    attach_online_fp8(
        config,
        SimpleNamespace(
            additional_config={
                "tpu_online_fp8": {"enabled": True, "targets": ["proj"], "exclude": []}
            }
        ),
    )
    mock_linear = MagicMock(spec=LinearBase)
    mock_linear.output_size = _N
    method = resolve_online_fp8(config, mock_linear, "proj", UnquantizedLinearMethod())
    assert isinstance(method, VllmFp8LinearMethodTPU)
    assert method.linear_config is not None
    assert method.act_quant_scale_dtype == "bfloat16"
    assert method.w8a16_max_rows == 32


@pytest.mark.parametrize("scale_dtype", [None, "bfloat16", "float32"])
def test_quantized_matmul_op_accepts_act_quant_scale_dtype(
    scale_dtype: str | None,
) -> None:
    op = linear_common._get_quantized_matmul_op()
    with torch.device("meta"):
        x = torch.empty((16, _K), dtype=torch.bfloat16)
        w_q = torch.empty((_K, _N), dtype=torch.float8_e4m3fn)
        w_s = torch.empty((_N,), dtype=torch.float32)
        out = op(x, w_q, w_s, 0, scale_dtype)
    assert out.shape == (16, _N)
    assert out.dtype == torch.bfloat16


@pytest.mark.tpu_test
def test_fp8_linear_tpu_runtime_and_compile_scale_dtypes(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if device.type != "tpu":
        pytest.skip("Pallas quantized matmul bridge requires TPU.")

    gen = torch.Generator(device="cpu").manual_seed(42)
    x_cpu = torch.randn((64, _K), generator=gen, dtype=torch.bfloat16)
    w_f32 = torch.randn((_K, _N), generator=gen, dtype=torch.float32)
    w_scale_cpu = w_f32.abs().amax(dim=0).clamp(min=1e-6) / 448.0
    w_q_cpu = (w_f32 / w_scale_cpu).to(torch.float8_e4m3fn)
    ref = x_cpu.float() @ (w_q_cpu.float() * w_scale_cpu)

    x_tpu = x_cpu.to(device)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_q_cpu.to(device), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_scale_cpu.to(device), requires_grad=False)

    outs: dict[str | None, torch.Tensor] = {}
    for env_val in (None, "", "bfloat16", "float32"):
        if env_val is None:
            monkeypatch.delenv("TPU_ACT_QUANT_SCALE_DTYPE", raising=False)
        else:
            monkeypatch.setenv("TPU_ACT_QUANT_SCALE_DTYPE", env_val)

        linear_config = VllmQuantLinearConfig()
        method = VllmFp8LinearMethodTPU(
            Fp8Config(is_checkpoint_fp8_serialized=True),
            linear_config=linear_config,
        )
        out_eager = method.apply(layer, x_tpu).cpu()

        torch._dynamo.reset()
        compiled_apply = torch.compile(
            lambda inp, m=method: m.apply(layer, inp),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        out_compiled = compiled_apply(x_tpu).cpu()
        assert torch.equal(out_eager, out_compiled)

        rel_err = (out_eager.float() - ref).norm() / ref.norm()
        assert rel_err.item() < 0.05
        outs[env_val] = out_eager

    assert torch.equal(outs[None], outs[""])
    assert torch.equal(outs[None], outs["float32"])
    bf16_vs_fp32 = (
        outs["bfloat16"].float() - outs["float32"].float()
    ).norm() / ref.norm()
    assert bf16_vs_fp32.item() < 0.025
    assert not torch.equal(outs["bfloat16"], outs["float32"])
