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

import contextlib
from collections.abc import Callable
from threading import Lock
from typing import Any

import jax
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import fused_add_rmsnorm as kernel
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_N_ATTR,
    FP8_PALLAS_WEIGHT,
    KEEP_VLLM_LAYOUT_ATTR,
    PREQUANT_ATTR,
    PrequantSlot,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_ops: dict[tuple[float, bool], Callable[..., Any]] = {}
_ops_lock = Lock()


def _slice_rows() -> int:
    """TPU_FUSED_ADD_RMSNORM_SLICES: the kernels' ``slice_rows``."""
    return kernel.SLICE_ROWS if envs.TPU_FUSED_ADD_RMSNORM_SLICES else 0


def _get_op(eps: float, has_residual: bool) -> Callable[..., Any]:
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
        slice_rows = _slice_rows()
        if has_residual:

            def fn(
                x: jax.Array, residual: jax.Array, weight: jax.Array
            ) -> tuple[jax.Array, jax.Array]:
                return kernel.fused_add_rms_norm(
                    x, residual, weight, eps=float(eps), slice_rows=slice_rows
                )

            def fake(x, residual, weight):
                return torch.empty_like(x), torch.empty_like(residual)

        else:

            def fn(x: jax.Array, weight: jax.Array) -> jax.Array:
                return kernel.fused_add_rms_norm(
                    x, None, weight, eps=float(eps), slice_rows=slice_rows
                )

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
        slice_rows = _slice_rows()

        def fn(
            x: jax.Array, residual: jax.Array, weight: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            y, y_q, scale, new_res = kernel.fused_add_rms_norm_quant(
                x,
                residual,
                weight,
                eps=float(eps),
                emit_y=emit_y,
                slice_rows=slice_rows,
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


_quant_moe_ops: dict[float, object] = {}


def _get_quant_moe_op(eps: float):
    """The add + RMSNorm op that also emits the stage-A MoE prefill kernel's
    fp8 rows and scales (TPU_FUSED_ADD_RMSNORM_QUANT_MOE; built at arming)."""
    key = float(eps)
    op = _quant_moe_ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _quant_moe_ops.get(key)
        if op is not None:
            return op
        tag = f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")
        slice_rows = _slice_rows()

        def fn(
            x: jax.Array, residual: jax.Array, weight: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return kernel.fused_add_rms_norm_quant_moe(
                x, residual, weight, eps=float(eps), slice_rows=slice_rows
            )

        def fake(x, residual, weight):
            return (torch.empty_like(x), *_quant_moe_fake(x, residual))

        op = pallas.jax_op(f"pallas::fused_add_rmsnorm_quant_moe_{tag}", fn)
        op.register_fake(fake)
        _quant_moe_ops[key] = op
        return op


def _quant_moe_fake(x, residual):
    rows, hidden = x.shape
    return (
        torch.empty(
            (rows, hidden // 128, 128), dtype=torch.float8_e4m3fn, device=x.device
        ),
        torch.empty((rows,), dtype=torch.float32, device=x.device),
        torch.empty_like(residual),
    )


_quant_moe_router_ops: dict[float, object] = {}


def _get_quant_moe_router_op(eps: float):
    """``_get_quant_moe_op``'s op that also computes the router logits from
    its own y (TPU_FUSED_ADD_RMSNORM_SLICES=2; built at arming). Returns
    ``(y, x_q, x_scale, new_residual, logits)``; the kernel does not write y,
    and the y returned here is dead unless something besides the gate reads
    it, so XLA drops it."""
    key = float(eps)
    op = _quant_moe_router_ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _quant_moe_router_ops.get(key)
        if op is not None:
            return op
        tag = f"{float(eps):.3e}".replace(".", "p").replace("-", "m").replace("+", "")
        slice_rows = _slice_rows()

        def fn(
            x: jax.Array, residual: jax.Array, weight: jax.Array, gate: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            _, x_q, x_scale, new_res, logits = kernel.fused_add_rms_norm_quant_moe(
                x, residual, weight, gate, eps=float(eps), slice_rows=slice_rows
            )
            y = kernel.reference(x, residual, weight, float(eps))[0]
            return y, x_q, x_scale, new_res, logits

        def fake(x, residual, weight, gate):
            return (
                torch.empty_like(x),
                *_quant_moe_fake(x, residual),
                torch.empty((x.shape[0], gate.shape[0]), dtype=x.dtype, device=x.device),
            )

        op = pallas.jax_op(f"pallas::fused_add_rmsnorm_quant_moe_router_{tag}", fn)
        op.register_fake(fake)
        _quant_moe_router_ops[key] = op
        return op


def _router_gate(runner, hidden: int):
    """The runner's gate when it is the plain bf16 ``F.linear`` the router
    variant reproduces: no bias, no fused shared-expert gate, the stock
    unquantized method on the ``[experts, hidden]`` weight, and none of the
    GPU-only GEMM tiers. None otherwise."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    gate = runner.gate
    if gate is None or getattr(runner, "_fse_fuse_gate", False):
        return None
    weight = getattr(gate, "weight", None)
    if (
        type(getattr(gate, "quant_method", None)) is not UnquantizedLinearMethod
        or getattr(gate, "bias", None) is not None
        or weight is None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or weight.shape[1] != hidden
        or weight.shape[0] % 128
    ):
        return None
    gpu_tiers = (
        "allow_ll_bf16_gemm",
        "allow_fp32_router_gemm",
        "allow_bf16x3_router_gemm",
        "allow_cublas_router_gemm",
    )
    if any(getattr(gate, name, False) for name in gpu_tiers):
        return None
    return gate


def _read_router_logits(gate, slot: PrequantSlot) -> None:
    """Makes ``gate`` return the logits its norm left in ``slot``, converted
    to the gate's output dtype as its own matmul output is, and empty the
    slot. The norm fills it on every path, so no trace reads the gate's own
    weight; with an empty slot the gate runs as before."""
    upstream = gate.forward
    out_dtype = getattr(gate, "out_dtype", None)

    def forward(x: torch.Tensor):
        logits = slot.value
        if logits is None:
            return upstream(x)
        slot.value = None
        if out_dtype is not None and logits.dtype != out_dtype:
            logits = logits.to(out_dtype)
        return logits, None

    gate.forward = forward


def arm_prequant_moe(model: torch.nn.Module) -> int:
    """TPU_FUSED_ADD_RMSNORM_QUANT_MOE: pair each MoE layer's
    post_attention_layernorm with its moe_tp_decode-armed experts, so on the
    norm's Pallas (prefill-sized) branch the norm emits the stage-A prefill
    kernel's own fp8 rows and scales and the op skips its quantization. The
    bf16 y is still returned (the router reads it). Returns the pair count.

    Call after moe_tp_decode.arm, on the target model only (the drafter's
    norms stay upstream). Pairs only where every bucket the norm's Pallas
    branch takes (`rows > bucket_gate_edge(_min_rows)`) is above the op's
    prefill cut-over, so a filled slot always meets the prefill kernel.

    TPU_FUSED_ADD_RMSNORM_SLICES=2: where the runner's gate is the plain bf16
    matmul, the norm also takes an alias of the gate weight as its own buffer
    (it never holds the gate module) and leaves the router logits in a slot
    the gate returns instead of computing them: from its kernel on the Pallas
    branch, from the gate's own ``F.linear`` on y otherwise. Every trace then
    lifts the weight under the norm's name, on both sides of the token-count
    branch."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm_torchtpu.layers.adapter.moe_tp_decode import (
        PREQUANT_MOE_ATTR,
        arm_prequant_op,
    )

    n = routed = 0
    for layer in model.modules():
        ln_post = getattr(layer, "post_attention_layernorm", None)
        mlp = getattr(layer, "mlp", None)
        if not isinstance(ln_post, TpuRMSNorm) or not ln_post._pallas_ok or mlp is None:
            continue
        if ln_post._prequant_slot is not None:
            continue  # already paired with a dense linear (TPU_FUSED_ADD_RMSNORM_QUANT)
        runners = [m for m in mlp.modules() if isinstance(m, MoERunner)]
        if len(runners) != 1:
            continue
        experts = runners[0].routed_experts
        cut_over = arm_prequant_op(experts)
        if cut_over is None or envs.bucket_gate_edge(ln_post._min_rows) < cut_over:
            continue
        ln_post._op_quant_moe = _get_quant_moe_op(ln_post.variance_epsilon)
        slot = PrequantSlot()
        ln_post._prequant_moe_slot = slot
        setattr(experts, PREQUANT_MOE_ATTR, slot)
        if envs.TPU_FUSED_ADD_RMSNORM_SLICES >= 2:
            gate = _router_gate(runners[0], ln_post.hidden_size)
            if gate is not None:
                ln_post.register_buffer(
                    "_router_weight", gate.weight.detach(), persistent=False
                )
                ln_post._op_quant_moe = _get_quant_moe_router_op(
                    ln_post.variance_epsilon
                )
                ln_post._router_slot = PrequantSlot()
                _read_router_logits(gate, ln_post._router_slot)
                routed += 1
        n += 1
    if envs.TPU_FUSED_ADD_RMSNORM_SLICES >= 2:
        logger.info(
            "TPU_FUSED_ADD_RMSNORM_SLICES=2: %d of %d MoE norms compute their "
            "router logits",
            routed,
            n,
        )
    return n


_norm_qkva_ops: dict[tuple, object] = {}
# The sequence-parallel prefill runs its layers on each core's share of the
# step's rows, so a norm there sees 64 rows or fewer of a long step; its
# attention wrapper projects those rows itself (the token split's gather).
_norm_qkva_off = False


@contextlib.contextmanager
def norm_qkva_off():
    """TPU_FUSED_NORM_QKVA's fused prologue stays off inside this block."""
    global _norm_qkva_off
    previous, _norm_qkva_off = _norm_qkva_off, True
    try:
        yield
    finally:
        _norm_qkva_off = previous


# `attention._retain_buffers`, bound at arming (outside Dynamo).
_retain_buffers: Callable[..., torch.Tensor] | None = None


def _get_norm_qkva_op(
    eps: float, q_eps: float, kv_eps: float, q_rank: int, kv_rank: int, emit_q8: bool
):
    """The TPU_FUSED_NORM_QKVA torch op (built at arming): returns
    ``(y, new_residual, q_c, kv_c_normed, k_pe)`` and with ``emit_q8``
    ``(q_c_fp8, q_c_scale)``."""
    from vllm_torchtpu.kernels import fused_norm_qkva

    key = (float(eps), float(q_eps), float(kv_eps), q_rank, kv_rank, emit_q8)
    op = _norm_qkva_ops.get(key)
    if op is not None:
        return op
    with _ops_lock:
        op = _norm_qkva_ops.get(key)
        if op is not None:
            return op
        tag = "_".join(
            f"{e:.3e}".replace(".", "p").replace("-", "m").replace("+", "")
            for e in key[:3]
        )

        def fn(
            x: jax.Array,
            residual: jax.Array,
            ln_w: jax.Array,
            w_q: jax.Array,
            w_s: jax.Array,
            qa_w: jax.Array,
            kva_w: jax.Array,
        ):
            return fused_norm_qkva.fused_norm_qkva(
                x,
                residual,
                ln_w,
                w_q,
                w_s,
                qa_w,
                kva_w,
                eps=float(eps),
                q_eps=float(q_eps),
                kv_eps=float(kv_eps),
                q_rank=q_rank,
                kv_rank=kv_rank,
                emit_q8=emit_q8,
            )

        # jax_op takes a fixed-length tuple return, not tuple[jax.Array, ...].
        fn.__annotations__["return"] = tuple[(jax.Array,) * (7 if emit_q8 else 5)]

        def fake(x, residual, ln_w, w_q, w_s, qa_w, kva_w):
            rows = x.shape[0]
            rope = w_q.shape[1] - q_rank - kv_rank
            outs = [
                torch.empty_like(x),
                torch.empty_like(residual),
                x.new_empty((rows, q_rank)),
                x.new_empty((rows, kv_rank)),
                x.new_empty((rows, rope)),
            ]
            if emit_q8:
                outs.append(
                    torch.empty((rows, q_rank), dtype=w_q.dtype, device=x.device)
                )
                outs.append(
                    torch.empty((rows, 1), dtype=torch.float32, device=x.device)
                )
            return tuple(outs)

        op = pallas.jax_op(
            f"pallas::fused_norm_qkva_r{q_rank}_{kv_rank}_q{int(emit_q8)}_{tag}", fn
        )
        op.register_fake(fake)
        _norm_qkva_ops[key] = op
        return op


def _plain_norm(norm) -> bool:
    w = getattr(norm, "weight", None)
    return (
        norm is not None
        and getattr(norm, "has_weight", False)
        and getattr(norm, "variance_size_override", None) is None
        and w is not None
        and w.dtype == torch.bfloat16
    )


def _plain_fp8_weight(linear, prequant_ok: bool) -> bool:
    """A per-channel FP8 linear whose every call is ``xla_quantized_matmul``
    (or ``fp8_skinny``, or with a PREQUANT slot ``xla_prequantized_matmul``)
    on its own canonical ``[k, n]`` weight."""
    return (
        _per_channel_fp8_linear(linear)
        and getattr(linear, FP8_PALLAS_WEIGHT, None) is None
        and not getattr(linear, KEEP_VLLM_LAYOUT_ATTR, False)
        and (prequant_ok or getattr(linear, PREQUANT_ATTR, None) is None)
    )


def arm_norm_qkva(model: torch.nn.Module) -> int:
    """TPU_FUSED_NORM_QKVA: pair each decoder layer's ``input_layernorm`` with
    its MLA wrapper, so at decode buckets (at most ``fused_norm_qkva.MAX_ROWS``
    rows) the norm runs ``kernels/fused_norm_qkva`` -- the add + RMSNorm,
    ``fused_qkv_a_proj`` and the q_a / kv_a latent norms in one call -- and
    leaves ``(q_c, kv_c_normed, k_pe)`` in a slot the wrapper reads instead of
    running the three ops. Returns the pair count.

    The norm holds aliases of the four weights it reads (the projection's
    weight and scale, the two latent norm weights), never the modules. Both
    sides of the rows branch lift both sets: the side that does not read a set
    retains it (``attention._retain_buffers``), so the warmup can hand buckets
    between the traces.

    Where the wrapper calls ``q_b_proj`` itself (no q_b-overlap indexer), the
    norm also emits q_c's fp8 quant above 16 rows (where ``fp8_skinny`` stops
    and q_b runs ``xla_quantized_matmul``), handed to ``q_b_proj`` through its
    PREQUANT slot: the q_b amax and convert passes go away, bit for bit.

    Call on the target model only (the drafter is traced once for every
    bucket and cannot branch on rows), after the post-load hooks (the rope
    permutation replaces the projection's parameters).
    """
    global _retain_buffers
    from vllm_torchtpu.kernels import fused_norm_qkva
    from vllm_torchtpu.layers.adapter.attention import (
        _retain_buffers as retain_buffers,
    )
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmTPUMultiHeadLatentAttentionWrapper,
    )
    from vllm_torchtpu.layers.adapter.linear_common import get_prequantized_matmul_op

    max_rows = fused_norm_qkva.MAX_ROWS
    _retain_buffers = retain_buffers
    n = 0
    for layer in model.modules():
        ln_in = getattr(layer, "input_layernorm", None)
        attn = getattr(layer, "self_attn", None)
        if (
            not isinstance(ln_in, TpuRMSNorm)
            or attn is None
            or getattr(layer, "use_sequence_parallel_moe", False)
            or not _plain_norm(ln_in)
        ):
            continue
        wrappers = [
            m
            for m in attn.modules()
            if isinstance(m, VllmTPUMultiHeadLatentAttentionWrapper)
        ]
        if len(wrappers) != 1:
            continue
        w = wrappers[0]
        qkv = w.fused_qkv_a_proj
        if (
            w.kimi_preprocess is not None
            or w.q_lora_rank is None
            or qkv is None
            or w.gate_is_fused
            or not _plain_norm(w.q_a_layernorm)
            or not _plain_norm(w.kv_a_layernorm)
            # TPU_FUSED_ADD_RMSNORM_QUANT's slot (prefill rows) may be on it.
            or not _plain_fp8_weight(qkv, prequant_ok=True)
            or getattr(qkv, FP8_PALLAS_N_ATTR, None) is not None
        ):
            continue
        q_rank, kv_rank = w.q_lora_rank, w.kv_lora_rank
        rope = w.qk_rope_head_dim
        hidden = ln_in.hidden_size
        if tuple(qkv.weight.shape) != (
            hidden,
            q_rank + kv_rank + rope,
        ) or not fused_norm_qkva.supports(
            max_rows, hidden, qkv.weight.shape[1], q_rank, kv_rank, rope
        ):
            continue
        # The layers whose wrapper runs q_b_proj itself (the forward's
        # non-overlap branch).
        direct_qb = not w.qb_overlap or bool(
            envs.TPU_QB_OVERLAP_HONORS_SKIP_TOPK and w.skip_topk
        )
        qb_slot = None
        if direct_qb and _plain_fp8_weight(w.q_b_proj, prequant_ok=False):
            get_prequantized_matmul_op()
            qb_slot = PrequantSlot()
            setattr(w.q_b_proj, PREQUANT_ATTR, qb_slot)
        ln_in.register_buffer("_nq_w", qkv.weight.detach(), persistent=False)
        # The kernel reads the vectors as [1, n] rows. A 1-D vector's layout is
        # tiled by 1024, so reshaping it per call is a real relayout: 0.6 us
        # each where XLA puts the result in HBM (16- and 32-row programs,
        # vm-b9794158-e94e-45e4-9204). Rows built once here need none.
        for name, t in (
            ("_nq_lnw", ln_in.weight),
            ("_nq_ws", qkv.weight_scale.float()),
            ("_nq_qa", w.q_a_layernorm.weight),
            ("_nq_kva", w.kv_a_layernorm.weight),
        ):
            ln_in.register_buffer(
                name, t.detach().reshape(1, -1).clone(), persistent=False
            )
        ln_in._nq_ops = {
            emit: _get_norm_qkva_op(
                ln_in.variance_epsilon,
                w.q_a_layernorm.variance_epsilon,
                w.kv_a_layernorm.variance_epsilon,
                q_rank,
                kv_rank,
                emit,
            )
            for emit in ((False, True) if qb_slot is not None else (False,))
        }
        ln_in._nq_max_rows = max_rows
        slot = PrequantSlot()
        ln_in._nq_slot = slot
        w._nq_slot = slot
        w._nq_qb_slot = qb_slot
        n += 1
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
        # TPU_FUSED_ADD_RMSNORM_QUANT_MOE: set by arm_prequant_moe.
        self._prequant_moe_slot = None
        self._op_quant_moe = None
        self._router_slot = None
        # TPU_FUSED_NORM_QKVA: set by arm_norm_qkva.
        self._nq_slot = None
        self._nq_ops: dict[bool, Any] = {}
        self._nq_max_rows = 0
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
        # Cap rule: `rows > edge` (the bucket below `_min_rows`).
        return rows > envs.bucket_gate_edge(self._min_rows) and kernel.supports(
            rows, self.hidden_size
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        out = self._forward(x, residual)
        if self._router_slot is not None and self._router_slot.value is None:
            y = out if residual is None else out[0]
            self._router_slot.value = torch.nn.functional.linear(y, self._router_weight)
        return out

    def _norm_qkva_applies(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> bool:
        """TPU_FUSED_NORM_QKVA's rows branch: decode buckets (`T <= cap`, a
        power of 2), below every Pallas norm bucket."""
        return (
            self._nq_slot is not None
            and not _norm_qkva_off
            and residual is not None
            and x.dim() == 2
            and x.device.type == "tpu"
            and x.dtype == torch.bfloat16
            and residual.dtype == x.dtype
            and residual.shape == x.shape
            and x.shape[-1] == self.hidden_size
            and x.shape[0] <= self._nq_max_rows
            and not self._use_pallas(x, residual)
        )

    def _nq_buffers(self) -> tuple[torch.Tensor, ...]:
        return self._nq_lnw, self._nq_w, self._nq_ws, self._nq_qa, self._nq_kva

    def _forward(self, x: torch.Tensor, residual: torch.Tensor | None):
        if self._nq_slot is not None:
            if self._norm_qkva_applies(x, residual):
                assert residual is not None
                # q_c's fp8 quant only where q_b_proj runs its XLA matmul.
                emit_q8 = True in self._nq_ops and x.shape[0] > fp8_skinny.MAX_M
                outs = self._nq_ops[emit_q8](
                    x,
                    residual,
                    self._nq_lnw,
                    self._nq_w,
                    self._nq_ws,
                    self._nq_qa,
                    self._nq_kva,
                )
                self._nq_slot.value = tuple(outs[2:])
                # The unfused side reads the norm's own weight; lift it here too.
                assert _retain_buffers is not None
                return _retain_buffers(outs[0], self.weight), outs[1]
            out = self._forward_unfused(x, residual)
            # The other side of the rows branch reads these; lift them here too.
            assert _retain_buffers is not None
            if residual is None:
                return _retain_buffers(out, *self._nq_buffers())
            y, new_res = out
            y = _retain_buffers(y, *self._nq_buffers())
            return y, new_res
        return self._forward_unfused(x, residual)

    def _forward_unfused(self, x: torch.Tensor, residual: torch.Tensor | None):
        if not self._use_pallas(x, residual):
            return super().forward(x, residual)
        if self._prequant_slot is not None and residual is not None and x.dim() == 2:
            # The same rows predicate as the plain Pallas path above: no new
            # token-count branch. The paired linear takes (y_q, scale).
            assert self._op_quant is not None
            y, y_q, scale, new_res = self._op_quant(x, residual, self.weight.data)
            self._prequant_slot.value = (y_q, scale)
            return y, new_res
        if (
            self._prequant_moe_slot is not None
            and residual is not None
            and x.dim() == 2
        ):
            # Same rows predicate again. The paired MoE op takes (x_q, x_scale)
            # on its stage-A prefill kernel; y still feeds the router, unless
            # the norm computed the router logits for the gate.
            assert self._op_quant_moe is not None
            if self._router_slot is not None:
                y, x_q, x_scale, new_res, logits = self._op_quant_moe(
                    x, residual, self.weight.data, self._router_weight
                )
                self._router_slot.value = logits
            else:
                y, x_q, x_scale, new_res = self._op_quant_moe(
                    x, residual, self.weight.data
                )
            self._prequant_moe_slot.value = (x_q, x_scale)
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
