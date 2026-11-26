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

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
    xla_quantized_matmul,
)
from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUMultiHeadLatentAttentionWrapper,
    arm_split_qkv_a_proj,
    split_qkv_a_proj_widths,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_N_ATTR,
    FP8_SPLIT_SCALES_ATTR,
    FP8_SPLIT_WEIGHTS_ATTR,
    KEEP_VLLM_LAYOUT_ATTR,
    PREQUANT_ATTR,
    PrequantSlot,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8LinearMethodTPU,
    split_fp8_linear_columns,
)


def _make_fp8_linear(
    k_dim: int,
    n_dim: int,
    *,
    seed: int = 11,
    with_bias: bool = False,
    scale_2d: bool = False,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.nn.Module, VllmFp8LinearMethodTPU, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    method = object.__new__(VllmFp8LinearMethodTPU)
    layer = torch.nn.Module()
    layer.quant_method = method  # type: ignore[assignment]
    w_f32 = rng.normal(0.0, 1.3, (k_dim, n_dim)).astype(np.float32)
    w = torch.from_numpy(w_f32).to(weight_dtype)
    s_np = rng.uniform(0.02, 0.37, (n_dim,)).astype(np.float32)
    s = torch.from_numpy(s_np)
    if scale_2d:
        s = s.reshape(1, n_dim)
    layer.register_parameter("weight", torch.nn.Parameter(w, requires_grad=False))
    layer.register_parameter("weight_scale", torch.nn.Parameter(s, requires_grad=False))
    if with_bias:
        layer.register_parameter(
            "bias", torch.nn.Parameter(torch.ones(n_dim), requires_grad=False)
        )
    else:
        layer.register_parameter("bias", None)
    return layer, method, w.clone(), s.clone()


def _make_wrapper(
    fused_layer: torch.nn.Module | None,
    *,
    q_lora_rank: int | None = 32,
    kv_lora_rank: int = 16,
    qk_rope_head_dim: int = 8,
    gate_is_fused: bool = False,
    kimi_preprocess: object | None = None,
) -> VllmTPUMultiHeadLatentAttentionWrapper:
    wrapper = object.__new__(VllmTPUMultiHeadLatentAttentionWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.q_lora_rank = q_lora_rank
    wrapper.kv_lora_rank = kv_lora_rank
    wrapper.qk_rope_head_dim = qk_rope_head_dim
    wrapper.gate_is_fused = gate_is_fused
    wrapper.kimi_preprocess = kimi_preprocess
    wrapper.fused_qkv_a_proj = fused_layer
    return wrapper


@pytest.mark.parametrize(
    ("q_lora", "kv_dim", "chunk", "expected"),
    [
        (2048, 576, 1024, (1024, 1024, 576)),
        (2048, 576, 2048, (2048, 576)),
        (2048, 576, 4096, (2048, 576)),
        (1536, 576, 1024, (1024, 512, 576)),
        (1024, 576, 512, (512, 512, 576)),
        (0, 576, 1024, ()),
        (2048, 0, 1024, ()),
        (2048, 576, 0, ()),
        (2048, 576, -128, ()),
    ],
)
def test_should_partition_qkv_a_widths_into_disjoint_column_slices(
    q_lora: int, kv_dim: int, chunk: int, expected: tuple[int, ...]
) -> None:
    assert split_qkv_a_proj_widths(q_lora, kv_dim, chunk) == expected


def test_should_split_fp8_linear_columns_preserving_bytes_and_detaching_monolith(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, _, orig_w, orig_s = _make_fp8_linear(32, 56, seed=3)
    built_ops: list[str] = []
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_prequantized_matmul_op",
        lambda: built_ops.append("matmul"),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_quantize_fp8_row_op",
        lambda: built_ops.append("quant"),
    )

    widths = (16, 16, 24)
    assert split_fp8_linear_columns(layer, widths) is True
    assert built_ops == ["matmul", "quant"]

    # Monolithic weight & scale are detached so total parameter count stays exact.
    assert getattr(layer, "weight") is None  # noqa: B009 -- mirror PyTorch attribute API
    assert getattr(layer, "weight_scale") is None  # noqa: B009 -- mirror PyTorch attribute API
    w_names = getattr(layer, FP8_SPLIT_WEIGHTS_ATTR)
    s_names = getattr(layer, FP8_SPLIT_SCALES_ATTR)
    assert len(w_names) == len(widths) == len(s_names)

    w_parts = [getattr(layer, name) for name in w_names]
    s_parts = [getattr(layer, name) for name in s_names]
    assert sum(p.numel() for p in w_parts) == orig_w.numel()
    assert sum(p.numel() for p in s_parts) == orig_s.numel()
    assert torch.equal(
        torch.cat(w_parts, dim=1).view(torch.uint8),
        orig_w.view(torch.uint8),
    )
    assert torch.equal(torch.cat(s_parts, dim=0), orig_s)
    # Disjoint device storage pointers (not views into a shared buffer).
    ptrs = {p.data_ptr() for p in w_parts}
    assert len(ptrs) == len(widths)
    assert orig_w.data_ptr() not in ptrs

    # Idempotent: splitting a second time returns False without mutating parts.
    assert split_fp8_linear_columns(layer, widths) is False


@pytest.mark.parametrize(
    ("kwargs", "widths", "extra_attr"),
    [
        ({}, (56,), None),
        ({}, (32, 23), None),
        ({}, (16, 0, 40), None),
        ({"with_bias": True}, (32, 24), None),
        ({"scale_2d": True}, (32, 24), None),
        ({"weight_dtype": torch.bfloat16}, (32, 24), None),
        ({}, (32, 24), (KEEP_VLLM_LAYOUT_ATTR, True)),
        ({}, (32, 24), (FP8_PALLAS_N_ATTR, 56)),
    ],
)
def test_should_reject_unsupported_linear_when_splitting_columns(
    kwargs: dict,
    widths: tuple[int, ...],
    extra_attr: tuple[str, object] | None,
) -> None:
    layer, _, orig_w, orig_s = _make_fp8_linear(16, 56, **kwargs)
    if extra_attr is not None:
        setattr(layer, extra_attr[0], extra_attr[1])
    assert split_fp8_linear_columns(layer, widths) is False
    assert getattr(layer, FP8_SPLIT_WEIGHTS_ATTR, None) is None
    assert torch.equal(
        layer.weight.view(torch.uint8)
        if layer.weight.dtype == torch.float8_e4m3fn
        else layer.weight,
        orig_w.view(torch.uint8) if orig_w.dtype == torch.float8_e4m3fn else orig_w,
    )
    assert torch.equal(layer.weight_scale, orig_s)


def test_should_match_monolithic_fp8_matmul_for_prequant_and_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    k_dim, n_dim = 64, 80
    widths = (32, 32, 16)
    layer, method, orig_w, orig_s = _make_fp8_linear(k_dim, n_dim, seed=29)

    def _quant_cpu(x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_j = jnp.asarray(x_t.to(torch.float32).numpy(), dtype=jnp.bfloat16)
        xq, xs = quantize_tensor(x_j, jnp.float8_e4m3fn)
        xq_t = torch.from_numpy(
            np.asarray(jax.lax.bitcast_convert_type(xq, jnp.uint8)).copy()
        ).view(torch.float8_e4m3fn)
        xs_t = torch.from_numpy(np.asarray(xs, dtype=np.float32).copy())
        return xq_t, xs_t

    def _preq_mm_cpu(
        xq_t: torch.Tensor,
        xs_t: torch.Tensor,
        wq_t: torch.Tensor,
        ws_t: torch.Tensor,
    ) -> torch.Tensor:
        xq_j = jax.lax.bitcast_convert_type(
            jnp.asarray(xq_t.view(torch.uint8).numpy()), jnp.float8_e4m3fn
        )
        wq_j = jax.lax.bitcast_convert_type(
            jnp.asarray(wq_t.view(torch.uint8).numpy()), jnp.float8_e4m3fn
        )
        out = xla_prequantized_matmul(
            xq_j,
            jnp.asarray(xs_t.numpy(), dtype=jnp.float32),
            wq_j,
            jnp.asarray(ws_t.numpy(), dtype=jnp.float32),
            jnp.bfloat16,
        )
        out_u16 = np.asarray(jax.lax.bitcast_convert_type(out, jnp.uint16)).copy()
        return torch.from_numpy(out_u16).view(torch.bfloat16)

    quant_calls = 0

    def _counted_quant(x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal quant_calls
        quant_calls += 1
        return _quant_cpu(x_t)

    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_prequantized_matmul_op",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_quantize_fp8_row_op",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_fp8_row",
        _counted_quant,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.prequantized_matmul",
        _preq_mm_cpu,
    )

    assert split_fp8_linear_columns(layer, widths) is True

    rng = np.random.default_rng(41)
    x_np = rng.normal(0.0, 2.5, (2, 5, k_dim)).astype(np.float32)
    # Anti-cancellation row check: zero row + sharp negative spike row.
    x_np[0, 0, :] = 0.0
    x_np[1, 3, 7] = -37.5
    x_bf = torch.from_numpy(x_np).to(torch.bfloat16)

    w_j = jax.lax.bitcast_convert_type(
        jnp.asarray(orig_w.view(torch.uint8).numpy()), jnp.float8_e4m3fn
    )
    s_j = jnp.asarray(orig_s.numpy(), dtype=jnp.float32)
    x_j = jnp.asarray(
        x_bf.reshape(-1, k_dim).to(torch.float32).numpy(), dtype=jnp.bfloat16
    )
    ref_j = xla_quantized_matmul(x_j, w_j, s_j)
    ref_u16 = np.asarray(jax.lax.bitcast_convert_type(ref_j, jnp.uint16))

    # Fallback path (no PrequantSlot): activation must be quantized exactly ONCE.
    out_fb = method.apply(layer, x_bf)
    assert quant_calls == 1
    assert out_fb.shape == (2, 5, n_dim)
    assert out_fb.dtype == torch.bfloat16
    assert np.array_equal(out_fb.reshape(-1, n_dim).view(torch.uint16).numpy(), ref_u16)

    # Paired PrequantSlot path: consumes prequantized input and clears slot.
    slot = PrequantSlot()
    slot.value = _quant_cpu(x_bf.reshape(-1, k_dim))
    setattr(layer, PREQUANT_ATTR, slot)
    out_pq = method.apply(layer, x_bf)
    assert quant_calls == 1
    assert slot.value is None
    assert np.array_equal(out_pq.reshape(-1, n_dim).view(torch.uint16).numpy(), ref_u16)

    # Positive control: perturbing one split scale breaks exact match.
    s_names = getattr(layer, FP8_SPLIT_SCALES_ATTR)
    orig_scale_val = getattr(layer, s_names[1]).data[0].item()
    getattr(layer, s_names[1]).data[0] = orig_scale_val * 2.0 + 0.5
    out_bad = method.apply(layer, x_bf)
    assert not np.array_equal(
        out_bad.reshape(-1, n_dim).view(torch.uint16).numpy(), ref_u16
    )


def test_should_arm_only_eligible_mla_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_prequantized_matmul_op",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_quantize_fp8_row_op",
        lambda: None,
    )
    ok_lin, _, _, _ = _make_fp8_linear(16, 56, seed=1)
    gate_lin, _, _, _ = _make_fp8_linear(16, 56, seed=2)
    kimi_lin, _, _, _ = _make_fp8_linear(16, 56, seed=3)
    mismatch_lin, _, _, _ = _make_fp8_linear(16, 40, seed=4)

    root = torch.nn.Module()
    root.ok = _make_wrapper(ok_lin, q_lora_rank=32, kv_lora_rank=16, qk_rope_head_dim=8)
    root.gated = _make_wrapper(gate_lin, gate_is_fused=True)
    root.kimi = _make_wrapper(kimi_lin, kimi_preprocess=object())
    root.sharded = _make_wrapper(mismatch_lin)
    root.no_lora = _make_wrapper(None, q_lora_rank=None)

    assert arm_split_qkv_a_proj(root, chunk_cols=16) == 1
    assert getattr(ok_lin, FP8_SPLIT_WEIGHTS_ATTR) == (
        "weight_split_0",
        "weight_split_1",
        "weight_split_2",
    )
    assert [
        getattr(ok_lin, n).shape for n in getattr(ok_lin, FP8_SPLIT_WEIGHTS_ATTR)
    ] == [
        (16, 16),
        (16, 16),
        (16, 24),
    ]
    assert getattr(gate_lin, FP8_SPLIT_WEIGHTS_ATTR, None) is None
    assert getattr(kimi_lin, FP8_SPLIT_WEIGHTS_ATTR, None) is None
    assert getattr(mismatch_lin, FP8_SPLIT_WEIGHTS_ATTR, None) is None
    assert arm_split_qkv_a_proj(root, chunk_cols=0) == 0


def test_should_reject_non_fp8_method_and_handle_runtime_bias_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = torch.nn.Module()
    plain.quant_method = object()  # type: ignore[assignment]
    assert split_fp8_linear_columns(plain, (8, 8)) is False

    layer, method, _, _ = _make_fp8_linear(16, 32, seed=9)
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_prequantized_matmul_op",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.get_quantize_fp8_row_op",
        lambda: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.quantize_fp8_row",
        lambda x: (
            torch.zeros(x.shape, dtype=torch.float8_e4m3fn),
            torch.ones((x.shape[0], 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.quantization.fp8.prequantized_matmul",
        lambda xq, xs, w, s: torch.ones(
            (xq.shape[0], w.shape[1]), dtype=torch.bfloat16
        ),
    )
    assert split_fp8_linear_columns(layer, (16, 16)) is True
    bias = torch.arange(32, dtype=torch.float32) * 0.25
    out = method.apply(layer, torch.zeros((3, 16), dtype=torch.float32), bias=bias)
    assert out.dtype == torch.float32
    assert torch.allclose(out, 1.0 + bias.unsqueeze(0))


def test_should_raise_when_quantize_fp8_row_called_before_op_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(linear_common, "_quantize_fp8_row_op", None)
    with pytest.raises(RuntimeError, match="quantize_fp8_row op not built"):
        linear_common.quantize_fp8_row(torch.zeros((2, 16), dtype=torch.bfloat16))
