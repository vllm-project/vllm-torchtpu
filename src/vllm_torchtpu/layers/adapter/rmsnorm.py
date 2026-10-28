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
"""Out-of-tree RMSNorm dispatching prefill buckets to one-pass Pallas kernels.

Token-count gating (`rows >= min_rows` and `kernel.supports(rows, hidden)`)
lives inside the exported `pallas.jax_op` JAX function rather than in PyTorch
`forward`, so Dynamo traces a single FX graph across all token buckets without
shape specialization while each concrete bucket compiles to either the one-pass
Pallas kernel (prefill/mixed) or the reference XLA HLO (decode/small buckets).
"""

from collections.abc import Callable
from threading import Lock
from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import fused_add_rmsnorm as kernel
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor
from vllm_torchtpu.layers.adapter.linear_common import (
    PREQUANT_ATTR,
    PrequantSlot,
    get_prequantized_matmul_op,
)

_ops: dict[tuple[float, bool, int], Callable[..., Any]] = {}
_quant_ops: dict[tuple[float, bool, int], Callable[..., Any]] = {}
_ops_lock = Lock()


def _eps_tag(eps: float) -> str:
    return f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")


def _get_op(eps: float, has_residual: bool, min_rows: int) -> Callable[..., Any]:
    """Build (outside Dynamo) the fused add + RMSNorm custom op."""
    key = (float(eps), has_residual, int(min_rows))
    op = _ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _ops.get(key)
        if op is not None:
            return op
        tag = _eps_tag(eps)
        eps_val = float(eps)
        min_r = int(min_rows)
        if has_residual:

            def fn(
                x: jax.Array, residual: jax.Array, weight: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                rows, hidden = x.shape
                if rows >= min_r and kernel.supports(rows, hidden):
                    res = kernel.fused_add_rms_norm(x, residual, weight, eps=eps_val)
                    assert isinstance(res, tuple)
                    return res
                ref = kernel.reference(x, residual, weight, eps_val)
                assert isinstance(ref, tuple)
                return ref

            def fake(
                x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                del weight
                return torch.empty_like(x), torch.empty_like(residual)

        else:

            def fn(x: jax.Array, weight: jax.Array) -> jax.Array:  # type: ignore[misc]
                rows, hidden = x.shape
                if rows >= min_r and kernel.supports(rows, hidden):
                    res = kernel.fused_add_rms_norm(x, None, weight, eps=eps_val)
                    assert not isinstance(res, tuple)
                    return res
                ref = kernel.reference(x, None, weight, eps_val)
                assert not isinstance(ref, tuple)
                return ref

            def fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
                del weight
                return torch.empty_like(x)

        op = pallas.jax_op(
            f"pallas::fused_add_rmsnorm_r{int(has_residual)}_m{min_r}_{tag}", fn
        )
        op.register_fake(fake)
        _ops[key] = op
        return op


def _get_quant_op(eps: float, emit_y: bool, min_rows: int) -> Callable[..., Any]:
    """Build (outside Dynamo) the fused add + RMSNorm + per-row FP8 quant op."""
    key = (float(eps), emit_y, int(min_rows))
    op = _quant_ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _quant_ops.get(key)
        if op is not None:
            return op
        tag = _eps_tag(eps)
        eps_val = float(eps)
        min_r = int(min_rows)

        def fn(
            x: jax.Array, residual: jax.Array, weight: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            rows, hidden = x.shape
            if rows >= min_r and kernel.supports(rows, hidden):
                y, y_q, scale, new_res = kernel.fused_add_rms_norm_quant(
                    x, residual, weight, eps=eps_val, emit_y=emit_y
                )
                if y is None:
                    ref = kernel.reference(x, residual, weight, eps_val)
                    assert isinstance(ref, tuple)
                    y = ref[0]
                return y, y_q, scale, new_res
            ref = kernel.reference(x, residual, weight, eps_val)
            assert isinstance(ref, tuple)
            y, new_res = ref
            y_q, scale = quantize_tensor(y, jnp.float8_e4m3fn)
            return y, y_q, scale, new_res

        def fake(
            x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            del weight
            return (
                torch.empty_like(x),
                torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
                torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
                torch.empty_like(residual),
            )

        op = pallas.jax_op(
            f"pallas::fused_add_rmsnorm_quant_y{int(emit_y)}_m{min_r}_{tag}", fn
        )
        op.register_fake(fake)
        _quant_ops[key] = op
        return op


def _per_channel_fp8_linear(linear: torch.nn.Module) -> bool:
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    qm = getattr(linear, "quant_method", None)
    w: Any = getattr(linear, "weight", None)
    s: Any = getattr(linear, "weight_scale", None)
    return (
        isinstance(qm, VllmFp8LinearMethodTPU)
        and isinstance(w, torch.Tensor)
        and w.dtype == torch.float8_e4m3fn
        and isinstance(s, torch.Tensor)
        and s.dim() == 1
        and getattr(linear, "bias", None) is None
    )


def _pair(
    norm: torch.nn.Module | None,
    linear: torch.nn.Module | None,
    emit_y: bool,
) -> bool:
    if not isinstance(norm, TpuRMSNorm) or not norm._pallas_ok or linear is None:
        return False
    if norm._prequant_slot is not None or not _per_channel_fp8_linear(linear):
        return False
    get_prequantized_matmul_op()
    slot = PrequantSlot()
    norm._prequant_slot = slot
    norm._op_quant = _get_quant_op(norm.variance_epsilon, emit_y, norm._min_rows)
    setattr(linear, PREQUANT_ATTR, slot)
    return True


def arm_prequant(model: torch.nn.Module) -> int:
    """Pair residual-stream norms with adjacent per-channel FP8 linears.

    Pairs:
      1. ``input_layernorm`` -> MLA ``fused_qkv_a_proj`` (``emit_y=True`` when
         ``indexer`` or ``g_proj`` also reads ``y``).
      2. Dense MLP ``post_attention_layernorm`` -> ``mlp.gate_up_proj``
         (``emit_y=False``).
      3. MoE ``post_attention_layernorm`` -> ``mlp.shared_experts.gate_up_proj``
         (``emit_y=True`` so router and routed experts still read bf16 ``y``),
         provided the MoE runner executes ``shared_experts`` directly without
         sequence-parallel redistribution or fused gate+shared weights.

    Returns:
        Number of ``(norm, linear)`` pairs armed.
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUMultiHeadLatentAttentionWrapper,
    )

    n = 0
    for layer in model.modules():
        ln_in = getattr(layer, "input_layernorm", None)
        ln_post = getattr(layer, "post_attention_layernorm", None)
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if (
            ln_in is None
            or attn is None
            or getattr(layer, "use_sequence_parallel_moe", False)
        ):
            continue
        wrappers = [
            m
            for m in attn.modules()
            if isinstance(m, VllmTPUMultiHeadLatentAttentionWrapper)
        ]
        if len(wrappers) == 1:
            w = wrappers[0]
            if (
                getattr(w, "kimi_preprocess", None) is None
                and w.q_lora_rank is not None
                and w.fused_qkv_a_proj is not None
            ):
                emit_y = w.indexer is not None or getattr(w, "g_proj", None) is not None
                n += int(_pair(ln_in, w.fused_qkv_a_proj, emit_y))
        if mlp is None:
            continue
        if (
            getattr(mlp, "gate_up_proj", None) is not None
            and getattr(mlp, "experts", None) is None
        ):
            n += int(_pair(ln_post, mlp.gate_up_proj, False))
        elif not getattr(mlp, "is_sequence_parallel", False):
            shared_mlp = getattr(mlp, "shared_experts", None)
            gate_up = (
                getattr(shared_mlp, "gate_up_proj", None)
                if shared_mlp is not None
                else None
            )
            runners = [m for m in mlp.modules() if isinstance(m, MoERunner)]
            if (
                gate_up is not None
                and len(runners) == 1
                and runners[0].shared_experts is not None
                and getattr(runners[0].shared_experts, "_layer", None) is shared_mlp
                and not getattr(runners[0], "_fse_fuse_gate", False)
            ):
                n += int(_pair(ln_post, gate_up, True))
    return n


def keep_upstream_path(model: torch.nn.Module) -> int:
    """Pin every ``TpuRMSNorm`` in ``model`` to the upstream path."""
    n = 0
    for m in model.modules():
        if isinstance(m, TpuRMSNorm) and m._pallas_ok:
            m._pallas_ok = False
            n += 1
    return n


@RMSNorm.register_oot
class TpuRMSNorm(RMSNorm):
    """RMSNorm dispatching eligible residual-stream norms via Pallas custom op."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._min_rows = max(
            envs.TPU_FUSED_ADD_RMSNORM_MIN_TOKENS, kernel.DEFAULT_BLOCK_ROWS
        )
        self._pallas_ok = (
            self.has_weight
            and self.variance_size_override is None
            and self.hidden_size % 128 == 0
            and self.hidden_size >= envs.TPU_FUSED_ADD_RMSNORM_MIN_HIDDEN
        )
        self._prequant_slot: PrequantSlot | None = None
        self._op_quant: Callable[..., Any] | None = None
        if self._pallas_ok:
            self._op_res = _get_op(self.variance_epsilon, True, self._min_rows)
            self._op_nores = _get_op(self.variance_epsilon, False, self._min_rows)

    def _can_dispatch(self, x: torch.Tensor, residual: torch.Tensor | None) -> bool:
        if not self._pallas_ok or x.device.type != "tpu":
            return False
        if x.dtype != self.weight.dtype or x.shape[-1] != self.hidden_size:
            return False
        if not x.is_contiguous():
            return False
        return residual is None or (
            residual.dtype == x.dtype
            and residual.shape == x.shape
            and residual.is_contiguous()
        )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self._can_dispatch(x, residual):
            return super().forward(x, residual)
        if self._prequant_slot is not None and residual is not None and x.dim() == 2:
            assert self._op_quant is not None
            y, y_q, scale, new_res = self._op_quant(x, residual, self.weight.data)
            self._prequant_slot.value = (y_q, scale)
            return y, new_res
        if x.dim() == 2:
            if residual is None:
                return self._op_nores(x, self.weight.data)
            return self._op_res(x, residual, self.weight.data)
        shape = x.shape
        x2 = x.reshape(-1, self.hidden_size)
        if residual is None:
            return self._op_nores(x2, self.weight.data).reshape(shape)
        y, new_res = self._op_res(
            x2, residual.reshape(-1, self.hidden_size), self.weight.data
        )
        return y.reshape(shape), new_res.reshape(shape)
