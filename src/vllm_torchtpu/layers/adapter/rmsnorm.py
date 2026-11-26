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
"""Prefill-sized RMSNorm and pre-quantization pairing as one Pallas pass."""

from collections.abc import Callable
from threading import Lock
from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.kernels import fused_add_rmsnorm as kernel
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_WEIGHT,
    PREQUANT_ATTR,
    PrequantSlot,
    get_prequantized_matmul_op,
)

MIN_TOKENS = 512
MIN_HIDDEN = 4096

_ops: dict[tuple[str, float, bool], Callable[..., Any]] = {}
_ops_lock = Lock()


def _eps_tag(eps: float) -> str:
    return f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")


def _cached_op(
    key: tuple[str, float, bool],
    name: str,
    fn: Callable[..., Any],
    fake: Callable[..., Any],
) -> Callable[..., Any]:
    op = _ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        if key not in _ops:
            built = pallas.jax_op(name, fn)
            built.register_fake(fake)
            _ops[key] = built
        return _ops[key]


def _get_op(eps: float, has_residual: bool) -> Callable[..., Any]:
    tag = f"pallas::fused_add_rmsnorm_r{int(has_residual)}_{_eps_tag(eps)}"
    key = ("norm", float(eps), has_residual)
    if has_residual:

        def fn(
            x: jax.Array, residual: jax.Array, weight: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            return kernel.fused_add_rms_norm(x, residual, weight, eps=float(eps))

        return _cached_op(
            key, tag, fn, lambda x, r, _w: (torch.empty_like(x), torch.empty_like(r))
        )

    def fn2(x: jax.Array, weight: jax.Array) -> jax.Array:
        return kernel.fused_add_rms_norm(x, None, weight, eps=float(eps))

    return _cached_op(key, tag, fn2, lambda x, _w: torch.empty_like(x))


def _get_quant_op(eps: float, emit_y: bool) -> Callable[..., Any]:
    def fn(
        x: jax.Array, residual: jax.Array, weight: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        y, y_q, scale, new_res = kernel.fused_add_rms_norm_quant(
            x, residual, weight, eps=float(eps), emit_y=emit_y
        )
        return (jnp.empty((0,), dtype=x.dtype) if y is None else y), y_q, scale, new_res

    def fake(x, residual, _weight):
        y_fake = (
            torch.empty_like(x)
            if emit_y
            else torch.empty((0,), dtype=x.dtype, device=x.device)
        )
        return (
            y_fake,
            torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
            torch.empty_like(residual),
        )

    return _cached_op(
        ("quant", float(eps), emit_y),
        f"pallas::fused_add_rmsnorm_quant_y{int(emit_y)}_{_eps_tag(eps)}",
        fn,
        fake,
    )


def _pallas_capable(norm: RMSNorm) -> bool:
    return (
        norm.has_weight
        and norm.variance_size_override is None
        and norm.hidden_size % 128 == 0
        and norm.hidden_size >= MIN_HIDDEN
    )


class TpuRMSNorm(RMSNorm):
    _min_rows: int
    _pallas_ok: bool
    _emit_y: bool
    _prequant_slot: PrequantSlot | None
    _op_quant: Callable[..., Any] | None
    _op_res: Callable[..., Any] | None
    _op_nores: Callable[..., Any] | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _init_tpu_rmsnorm_attrs(self)

    def disarm_prequant(self) -> None:
        self._prequant_slot = None
        self._op_quant = None

    def is_prequant_armed(self) -> bool:
        return self._prequant_slot is not None

    def _use_pallas(self, x: torch.Tensor, residual: torch.Tensor | None) -> bool:
        if (
            not self._pallas_ok
            or x.device.type != "tpu"
            or (torch.compiler.is_compiling() and not shape_variants.in_warmup())
            or x.dtype != self.weight.dtype
            or x.shape[-1] != self.hidden_size
            or not x.is_contiguous()
        ):
            return False
        if residual is not None and (
            residual.dtype != x.dtype
            or residual.shape != x.shape
            or not residual.is_contiguous()
        ):
            return False
        rows = x.numel() // self.hidden_size
        return rows > shape_variants.bucket_gate_edge(
            self._min_rows
        ) and kernel.supports(rows, self.hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if not self._use_pallas(x, residual):
            return super().forward(x, residual)
        if self._prequant_slot is not None and residual is not None and x.dim() == 2:
            assert self._op_quant is not None
            y, y_q, scale, new_res = self._op_quant(x, residual, self.weight)
            self._prequant_slot.value = (y_q, scale)
            return (y if self._emit_y else x), new_res
        assert self._op_nores is not None and self._op_res is not None
        is_2d = x.dim() == 2
        x2 = x if is_2d else x.reshape(-1, self.hidden_size)
        if residual is None:
            out = self._op_nores(x2, self.weight)
            return out if is_2d else out.reshape(x.shape)
        res2 = residual if is_2d else residual.reshape(-1, self.hidden_size)
        y, new_res = self._op_res(x2, res2, self.weight)
        return (y, new_res) if is_2d else (y.reshape(x.shape), new_res.reshape(x.shape))


def _init_tpu_rmsnorm_attrs(norm: TpuRMSNorm) -> None:
    norm._min_rows = max(MIN_TOKENS, kernel.DEFAULT_BLOCK_ROWS)
    norm._pallas_ok = _pallas_capable(norm)
    norm._emit_y = True
    norm._prequant_slot = None
    norm._op_quant = None
    norm._op_res = _get_op(norm.variance_epsilon, True) if norm._pallas_ok else None
    norm._op_nores = _get_op(norm.variance_epsilon, False) if norm._pallas_ok else None


def _upgrade_model_rmsnorms(model: torch.nn.Module) -> int:
    upgraded = 0
    for module in model.modules():
        if type(module) is RMSNorm and _pallas_capable(module):
            module.__class__ = TpuRMSNorm
            _init_tpu_rmsnorm_attrs(module)  # type: ignore[arg-type]
            upgraded += 1
    return upgraded


def _per_channel_fp8_linear(linear: torch.nn.Module | None) -> bool:
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    if (
        not isinstance(linear, LinearBase)
        or not isinstance(linear.quant_method, VllmFp8LinearMethodTPU)
        or FP8_PALLAS_WEIGHT in linear._parameters
    ):
        return False
    w, s, bias = (linear._parameters.get(k) for k in ("weight", "weight_scale", "bias"))
    return (
        isinstance(w, torch.Tensor)
        and w.dtype == torch.float8_e4m3fn
        and isinstance(s, torch.Tensor)
        and s.dim() == 1
        and bias is None
    )


def _pair(
    norm: torch.nn.Module | None, linear: torch.nn.Module | None, emit_y: bool
) -> bool:
    if (
        not isinstance(norm, TpuRMSNorm)
        or not norm._pallas_ok
        or norm.weight.dtype != torch.bfloat16
        or not _per_channel_fp8_linear(linear)
    ):
        return False
    assert linear is not None
    get_prequantized_matmul_op()
    slot = PrequantSlot()
    norm._emit_y, norm._prequant_slot = emit_y, slot
    norm._op_quant = _get_quant_op(norm.variance_epsilon, emit_y)
    linear.__dict__[PREQUANT_ATTR] = slot
    return True


def arm_prequant(model: torch.nn.Module) -> int:
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUMultiHeadLatentAttentionWrapper,
    )

    _upgrade_model_rmsnorms(model)
    paired = 0
    for layer in model.modules():
        ln_in, ln_post, attn, mlp = (
            layer._modules.get(k)
            for k in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp")
        )
        if (
            ln_in is None
            or attn is None
            or layer.__dict__.get("use_sequence_parallel_moe")
        ):
            continue
        wrappers = [
            m
            for m in attn.modules()
            if isinstance(m, VllmTPUMultiHeadLatentAttentionWrapper)
            and m.q_lora_rank is not None
            and m.fused_qkv_a_proj is not None
            and m.kimi_preprocess is None
        ]
        if len(wrappers) == 1:
            w = wrappers[0]
            emit_y = (w.indexer is not None and not w.skip_topk) or (
                w.g_proj is not None
            )
            paired += int(_pair(ln_in, w.fused_qkv_a_proj, emit_y))
        if (
            mlp is not None
            and ln_post is not None
            and "gate_up_proj" in mlp._modules
            and "experts" not in mlp._modules
        ):
            paired += int(_pair(ln_post, mlp._modules["gate_up_proj"], False))
    return paired


def disarm_post_attention_prequant(layer: torch.nn.Module) -> None:
    ln_post = getattr(layer, "post_attention_layernorm", None)
    if isinstance(ln_post, TpuRMSNorm):
        ln_post.disarm_prequant()
    gate_up = getattr(getattr(layer, "mlp", None), "gate_up_proj", None)
    if gate_up is not None and hasattr(gate_up, PREQUANT_ATTR):
        delattr(gate_up, PREQUANT_ATTR)
