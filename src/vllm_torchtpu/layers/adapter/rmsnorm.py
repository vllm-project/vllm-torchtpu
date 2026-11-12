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
"""``TPU_FUSED_ADD_RMSNORM``: prefill-sized RMSNorm as one Pallas pass.

At prefill/mixed token counts XLA splits ``fused_add_rms_norm`` into a row
reduction and an apply pass that re-reads both inputs (and recomputes the
add), so each norm moves ~1.75x the bytes it needs. Rows at or above
``TPU_FUSED_ADD_RMSNORM_MIN_TOKENS`` go to ``kernels/fused_add_rmsnorm``,
which reads ``x``/``residual`` once and writes ``y``/``residual`` once. Smaller
(decode) batches keep the upstream path untouched.

Imported only when the flag is set, so the flag-off model builds the stock
``RMSNorm`` class.
"""

from threading import Lock

import jax
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import fused_add_rmsnorm as kernel
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_WEIGHT,
    PREQUANT_ATTR,
    PrequantSlot,
)

_ops: dict[tuple[float, bool], object] = {}
_ops_lock = Lock()


def _get_op(eps: float, has_residual: bool):
    """Build (once, at model construction -- outside Dynamo) the torch op."""
    key = (float(eps), has_residual)
    op = _ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _ops.get(key)
        if op is not None:
            return op
        tag = f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")
        if has_residual:

            def fn(
                x: jax.Array, residual: jax.Array, weight: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                return kernel.fused_add_rms_norm(x, residual, weight, eps=float(eps))

            def fake(x, residual, weight):
                return torch.empty_like(x), torch.empty_like(residual)

        else:

            def fn(x: jax.Array, weight: jax.Array) -> jax.Array:
                return kernel.fused_add_rms_norm(x, None, weight, eps=float(eps))

            def fake(x, weight):
                return torch.empty_like(x)

        op = pallas.jax_op(f"pallas::fused_add_rmsnorm_r{int(has_residual)}_{tag}", fn)
        op.register_fake(fake)
        _ops[key] = op
        return op


_quant_ops: dict[tuple[float, bool], object] = {}


def _get_quant_op(eps: float, emit_y: bool):
    """The add + RMSNorm + per-row fp8 quant torch op (built at arming)."""
    key = (float(eps), emit_y)
    op = _quant_ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _quant_ops.get(key)
        if op is not None:
            return op
        tag = f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")

        def fn(
            x: jax.Array, residual: jax.Array, weight: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            y, y_q, scale, new_res = kernel.fused_add_rms_norm_quant(
                x, residual, weight, eps=float(eps), emit_y=emit_y
            )
            if y is None:
                # No second consumer: y is dead and XLA drops this. Kept so a
                # consumer nobody paired still reads correct values.
                y = kernel.reference(x, residual, weight, float(eps))[0]
            return y, y_q, scale, new_res

        def fake(x, residual, weight):
            return (
                torch.empty_like(x),
                torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
                torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
                torch.empty_like(residual),
            )

        op = pallas.jax_op(f"pallas::fused_add_rmsnorm_quant_y{int(emit_y)}_{tag}", fn)
        op.register_fake(fake)
        _quant_ops[key] = op
        return op


def _per_channel_fp8_linear(linear) -> bool:
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    qm = getattr(linear, "quant_method", None)
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    return (
        isinstance(qm, VllmFp8LinearMethodTPU)
        and w is not None
        and w.dtype == torch.float8_e4m3fn
        and s is not None
        and s.dim() == 1
        and getattr(linear, "bias", None) is None
    )


def _pair(norm, linear, emit_y: bool) -> bool:
    if not isinstance(norm, TpuRMSNorm) or not norm._pallas_ok:
        return False
    if not _per_channel_fp8_linear(linear):
        return False
    if getattr(linear, FP8_PALLAS_WEIGHT, None) is not None:
        # Its unpaired path lifts the padded kernel copy, the paired one would
        # not: the two sides of the rows branch must lift the same buffers.
        return False
    if linear.quant_method.w8a16_max_rows >= norm._min_rows:
        # Buckets the norm's Pallas path takes would run the linear as W8A16
        # (no activation quant), so the prequantized input would change it.
        return False
    # Both ops are built here, at load, outside Dynamo: the forward only calls them.
    from vllm_torchtpu.layers.adapter.linear_common import get_prequantized_matmul_op

    get_prequantized_matmul_op()
    slot = PrequantSlot()
    norm._prequant_slot = slot
    norm._op_quant = _get_quant_op(norm.variance_epsilon, emit_y)
    setattr(linear, PREQUANT_ATTR, slot)
    return True


def arm_prequant(model: torch.nn.Module) -> int:
    """TPU_FUSED_ADD_RMSNORM_QUANT: pair each decoder layer's residual-stream
    norms with the per-channel FP8 linear that reads their output; returns the
    number of pairs.

    - ``input_layernorm`` -> the MLA ``fused_qkv_a_proj``. ``y`` is still
      written when the indexer (``wk_weights_proj``) or ``g_proj`` reads it.
    - ``post_attention_layernorm`` -> a dense MLP's ``gate_up_proj``. MoE
      layers are left alone: their router reads ``y`` and the fused MoE ops
      quantize with their own numerics.

    Call on the target model only (the drafter's norms stay upstream).
    """
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
                w.kimi_preprocess is None
                and w.q_lora_rank is not None
                and w.fused_qkv_a_proj is not None
            ):
                emit_y = w.indexer is not None or w.g_proj is not None
                n += _pair(ln_in, w.fused_qkv_a_proj, emit_y)
        if (
            mlp is not None
            and getattr(mlp, "gate_up_proj", None) is not None
            and getattr(mlp, "experts", None) is None
        ):
            n += _pair(ln_post, mlp.gate_up_proj, False)
    return n


def keep_upstream_path(model: torch.nn.Module) -> int:
    """Pin every TpuRMSNorm in ``model`` to the upstream path; returns the count.

    For modules traced once for all token buckets (the spec-decode draft
    model): the token-count branch in ``forward`` would specialize their graph.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, TpuRMSNorm) and m._pallas_ok:
            m._pallas_ok = False
            n += 1
    return n


@RMSNorm.register_oot
class TpuRMSNorm(RMSNorm):
    """RMSNorm whose prefill-sized calls run the one-pass Pallas kernel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._min_rows = max(
            envs.TPU_FUSED_ADD_RMSNORM_MIN_TOKENS, kernel.DEFAULT_BLOCK_ROWS
        )
        self._pallas_ok = (
            self.has_weight
            and self.variance_size_override is None
            and self.hidden_size % 128 == 0
            # Only the residual-stream norms: the small q/kv latent norms read
            # strided slices XLA fuses into their neighbours.
            and self.hidden_size >= envs.TPU_FUSED_ADD_RMSNORM_MIN_HIDDEN
        )
        # TPU_FUSED_ADD_RMSNORM_QUANT: set by arm_prequant.
        self._prequant_slot = None
        self._op_quant = None
        if self._pallas_ok:
            self._op_res = _get_op(self.variance_epsilon, True)
            self._op_nores = _get_op(self.variance_epsilon, False)

    def _use_pallas(self, x: torch.Tensor, residual: torch.Tensor | None) -> bool:
        if not self._pallas_ok or x.device.type != "tpu":
            return False
        if x.dtype != self.weight.dtype or x.shape[-1] != self.hidden_size:
            return False
        if not x.is_contiguous():
            return False
        if residual is not None and (
            residual.dtype != x.dtype
            or residual.shape != x.shape
            or not residual.is_contiguous()
        ):
            return False
        rows = x.numel() // self.hidden_size
        return rows >= self._min_rows and kernel.supports(rows, self.hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if not self._use_pallas(x, residual):
            return super().forward(x, residual)
        if self._prequant_slot is not None and residual is not None and x.dim() == 2:
            # The same rows predicate as the plain Pallas path above: no new
            # token-count branch. The paired linear takes (y_q, scale).
            y, y_q, scale, new_res = self._op_quant(x, residual, self.weight.data)
            self._prequant_slot.value = (y_q, scale)
            return y, new_res
        if x.dim() == 2:
            # No view ops on the 2-D residual stream: a view the backend
            # lowers as a relayout would cost a full pass at these sizes.
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
