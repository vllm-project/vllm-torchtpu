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
"""Fused gated q split + QK RMS norm + RoPE for ``Qwen3NextAttention``.

vLLM fuses these on CUDA only. With ``TPU_FUSED_QK_NORM_ROPE`` set, token
buckets in ``[TPU_FUSED_QK_NORM_ROPE_MIN_ROWS, MAX_ROWS]`` run the
``qk_norm_rope_gate`` kernel, which tiles large row counts in VMEM, instead of
the eager split, per-head norms and rope; buckets outside that range run the
JAX reference inside the same custom op so the FX trace carries no Python row
fork and works for both ``backbone`` and ``eagle_head`` graphs.
"""

import functools
from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.qk_norm_rope_gate import (
    qk_norm_rope_gate,
    reference as qk_norm_rope_gate_reference,
)
from vllm_torchtpu.kernels.experimental.qk_norm_rope_gate.kernel import (
    qk_norm_rope_gate_v2,
    v2_supported,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The largest token bucket that takes the kernel.
MAX_ROWS = 16384

# The (1 + w) buffer TPU_NORM_WEIGHT_HOIST keeps on each GemmaRMSNorm, when that
# feature is present and on.
_HOISTED_WEIGHT_ATTR = "_tpu_weight_plus_one"

_ops: dict[tuple[int, int, int, int, float], Callable] = {}
_inplace_ops: dict[tuple[int, int, int, int, float, int], Callable] = {}
_INPLACE_VMEM_LIMIT = 48 * 1024 * 1024


def _op_key(layer: torch.nn.Module) -> tuple[int, int, int, int, float]:
    return (
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        layer.rotary_emb.rotary_dim,
        float(layer.q_norm.variance_epsilon),
    )


def _build_op(key: tuple[int, int, int, int, float]) -> None:
    if key in _ops:
        return
    heads, kv_heads, dim, rot, eps = key

    def fn(
        q_gate: jax.Array,
        k: jax.Array,
        q_w: jax.Array,
        k_w: jax.Array,
        cos_sin: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        min_rows = envs.TPU_FUSED_QK_NORM_ROPE_MIN_ROWS
        impl = (
            qk_norm_rope_gate
            if min_rows <= q_gate.shape[0] <= MAX_ROWS
            else qk_norm_rope_gate_reference
        )
        return impl(
            q_gate,
            k,
            q_w,
            k_w,
            cos_sin,
            heads=heads,
            kv_heads=kv_heads,
            dim=dim,
            rot=rot,
            eps=eps,
        )

    op = pallas.jax_op(f"pallas::qk_norm_rope_gate_{len(_ops)}", fn)

    def fake(q_gate, k, q_w, k_w, cos_sin):
        rows = q_gate.shape[0]
        return (
            q_gate.new_empty((rows, heads * dim)),
            k.new_empty((rows, kv_heads * dim)),
            q_gate.new_empty((rows, heads * dim)),
        )

    op.register_fake(fake)
    _ops[key] = op


def _build_inplace_op(key: tuple[int, int, int, int, float], tile: int) -> None:
    """TPU_QK_NORM_ROPE_INPLACE_TILE (DN3): the op takes the whole qkv and the
    kernel reads q_gate and k in place, so torch_tpu materializes no sliced
    copy of q_gate for the call. Same math, element for element."""
    if key + (tile,) in _inplace_ops:
        return
    heads, kv_heads, dim, rot, eps = key
    qg_cols, k_cols = heads * 2 * dim, kv_heads * dim

    def fn(
        qkv: jax.Array, q_w: jax.Array, k_w: jax.Array, cos_sin: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        rows = qkv.shape[0]
        kw = dict(heads=heads, kv_heads=kv_heads, dim=dim, rot=rot, eps=eps)
        if envs.TPU_FUSED_QK_NORM_ROPE_MIN_ROWS <= rows <= MAX_ROWS:
            packed = envs.TPU_QK_NORM_ROPE_Q_PACKED and heads % kv_heads == 0 and (heads // kv_heads) % 2 == 0
            q, k, g = qk_norm_rope_gate_v2(
                qkv,
                q_w,
                k_w,
                cos_sin,
                tile=tile,
                rope="concat",
                vmem_limit=_INPLACE_VMEM_LIMIT if tile > 128 else 0,
                name="qk_norm_rope_gate",
                q_packed=packed,
                **kw,
            )
            if packed:
                # TPU_QK_NORM_ROPE_Q_PACKED: the kernel wrote q in RPAm's packed kv-major order; hand
                # torch the same values token-major. XLA folds this permutation against the PCP entry's
                # re-pack (TPU_PCP_Q_PACK_OUTSIDE), so q is never relayouted.
                q = q.swapaxes(0, 1).reshape(rows, heads * dim)
            return q, k, g
        return qk_norm_rope_gate_reference(
            qkv[:, :qg_cols], qkv[:, qg_cols : qg_cols + k_cols], q_w, k_w, cos_sin, **kw
        )

    op = pallas.jax_op(f"pallas::qk_norm_rope_gate_inplace{tile}_{len(_inplace_ops)}", fn)

    def fake(qkv, q_w, k_w, cos_sin):
        rows = qkv.shape[0]
        return (
            qkv.new_empty((rows, heads * dim)),
            qkv.new_empty((rows, k_cols)),
            qkv.new_empty((rows, heads * dim)),
        )

    op.register_fake(fake)
    _inplace_ops[key + (tile,)] = op


def _weight_plus_one(norm: torch.nn.Module) -> torch.Tensor:
    hoisted = getattr(norm, _HOISTED_WEIGHT_ATTR, None)
    if hoisted is None:
        hoisted = norm.weight.float() + 1.0
        setattr(norm, _HOISTED_WEIGHT_ATTR, hoisted)
    return hoisted


def _unsupported_reason(layer: torch.nn.Module, model_config) -> str | None:
    mm_config = model_config.multimodal_config if model_config is not None else None
    if not layer.attn_output_gate:
        return "the attention has no output gate"
    if not layer.rotary_emb.is_neox_style:
        return "the rope is not NeoX style"
    if mm_config is not None and not mm_config.language_model_only:
        return "multimodal positions need M-RoPE; serve with --language-model-only"
    return None


def _log_armed_rows(vllm_config: VllmConfig) -> None:
    spec = vllm_config.speculative_config
    rows_per_request = 1 + (spec.num_speculative_tokens if spec is not None else 0)
    max_decode_rows = vllm_config.scheduler_config.max_num_seqs * rows_per_request
    logger.info_once(
        "[Attention] TPU_FUSED_QK_NORM_ROPE: buckets of <= %d rows run the fused "
        "q/k norm + rope kernel, i.e. decode steps of <= %d requests at %d rows "
        "each; the largest decode step has %d rows and runs the %s path.",
        MAX_ROWS,
        MAX_ROWS // rows_per_request,
        rows_per_request,
        max_decode_rows,
        "fused" if max_decode_rows <= MAX_ROWS else "eager",
    )


def _prebuild(layer: torch.nn.Module, model_config) -> bool:
    reason = _unsupported_reason(layer, model_config)
    if reason is not None:
        logger.warning_once(
            "[Attention] TPU_FUSED_QK_NORM_ROPE: %s; running the eager q/k norm "
            "and rope.",
            reason,
        )
        return False
    rot = layer.rotary_emb.rotary_dim
    assert 0 < rot <= layer.head_dim and rot % 2 == 0, (rot, layer.head_dim)
    _weight_plus_one(layer.q_norm)
    _weight_plus_one(layer.k_norm)
    _build_op(_op_key(layer))
    tile = envs.TPU_QK_NORM_ROPE_INPLACE_TILE
    if tile:
        if v2_supported(layer.num_heads, layer.num_kv_heads, layer.head_dim, rot, "concat"):
            _build_inplace_op(_op_key(layer), tile)
            logger.info_once(
                "[Attention] TPU_QK_NORM_ROPE_INPLACE_TILE=%d: the q/k norm + rope "
                "kernel reads q_gate and k in place from qkv.",
                tile,
            )
        else:
            logger.warning_once(
                "[Attention] TPU_QK_NORM_ROPE_INPLACE_TILE ignored: %d heads, %d kv "
                "heads of %d do not tile in place.",
                layer.num_heads,
                layer.num_kv_heads,
                layer.head_dim,
            )
    _log_armed_rows(get_current_vllm_config())
    return True


def project_qkv_gate(
    layer: torch.nn.Module, qkv: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``Qwen3NextAttention._project_qkv_gate`` through the fused kernel."""
    q_gate, k, v = qkv.split([layer.q_size * 2, layer.kv_size, layer.kv_size], dim=-1)
    if positions.ndim == 2:
        # Text-only M-RoPE: the three position rows agree.
        positions = positions[0]
    cos_sin = layer.rotary_emb.cos_sin_cache.index_select(0, positions.to(torch.int64))
    tile = envs.TPU_QK_NORM_ROPE_INPLACE_TILE
    if tile and _op_key(layer) + (tile,) in _inplace_ops:
        # Built by _prebuild. The op reads q_gate and k inside qkv; v stays a view.
        q, k, gate = _inplace_ops[_op_key(layer) + (tile,)](
            qkv,
            _weight_plus_one(layer.q_norm),
            _weight_plus_one(layer.k_norm),
            cos_sin,
        )
        return q, k, v, gate
    # Built by _prebuild: registering an op inside a traced forward is not allowed.
    q, k, gate = _ops[_op_key(layer)](
        q_gate,
        k,
        _weight_plus_one(layer.q_norm),
        _weight_plus_one(layer.k_norm),
        cos_sin,
    )
    return q, k, v, gate


def install_qk_norm_rope_patch() -> None:
    """Route ``Qwen3NextAttention`` to the fused kernel custom op."""
    if not envs.TPU_FUSED_QK_NORM_ROPE:
        return
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    original_init = Qwen3NextAttention.__init__
    original_project = Qwen3NextAttention._project_qkv_gate

    @functools.wraps(original_init)
    def __init__(self, config, model_config=None, *args, **kwargs):
        original_init(self, config, model_config, *args, **kwargs)
        self.fused_qk_norm_rope = _prebuild(self, model_config)

    @functools.wraps(original_project)
    def _project_qkv_gate(self, qkv, positions):
        if self.fused_qk_norm_rope:
            return project_qkv_gate(self, qkv, positions)
        return original_project(self, qkv, positions)

    Qwen3NextAttention.__init__ = __init__
    Qwen3NextAttention._project_qkv_gate = _project_qkv_gate
