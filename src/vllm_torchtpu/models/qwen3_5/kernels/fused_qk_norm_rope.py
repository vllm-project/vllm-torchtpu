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

vLLM fuses these on CUDA only. Where the kernel was measured (the rule in
:func:`maybe_install_qk_norm_rope_patch` and the per-device head counts in
:func:`_unsupported_reason`, both from ``measured``), token buckets in
``[MIN_ROWS, MAX_ROWS]`` run the ``qk_norm_rope_gate`` kernel, which tiles
large row counts in VMEM, instead of the eager split, per-head norms and
rope; buckets outside that range run the JAX reference inside the same custom
op so the FX trace carries no Python row fork and works for both ``backbone``
and ``eagle_head`` graphs. Other models that use ``Qwen3NextAttention``
(Qwen3-Next, and every Qwen3.5 checkpoint or device the kernel was not
measured on) keep vLLM's eager path.
"""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import jax
import torch
from torch_tpu._internal import pallas
from vllm.config import ModelConfig, VllmConfig, get_current_vllm_config

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5.kernels import measured
from vllm_torchtpu.models.qwen3_5.kernels.qk_norm_rope_gate import (
    qk_norm_rope_gate,
)
from vllm_torchtpu.models.qwen3_5.kernels.qk_norm_rope_gate import (
    reference as qk_norm_rope_gate_reference,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

logger = init_logger(__name__)

# The token buckets that take the kernel; the others run the JAX reference.
MIN_ROWS = 0
MAX_ROWS = 16384

# Marks the wrapper install_qk_norm_rope_patch puts on Qwen3NextAttention.
_INSTALLED_ATTR = "_tpu_fused_qk_norm_rope"

# The (1 + w) buffer the Qwen3.5 model patch's norm weight hoist keeps on each
# GemmaRMSNorm after loading; without it _weight_plus_one computes (1 + w).
_HOISTED_WEIGHT_ATTR = "_tpu_weight_plus_one"

_ops: dict[tuple[int, int, int, int, float], Callable] = {}


def _op_key(layer: torch.nn.Module) -> tuple[int, int, int, int, float]:
    layer = cast("Qwen3NextAttention", layer)
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
        impl = (
            qk_norm_rope_gate
            if MIN_ROWS <= q_gate.shape[0] <= MAX_ROWS
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


def _weight_plus_one(norm: torch.nn.Module) -> torch.Tensor:
    """The norm's (1 + w): the hoisted buffer when the model patch set one
    after loading, else computed from the current weight. Never cached here:
    the layer is built before its weights load, so a value taken then holds
    the initial weight, not the checkpoint's."""
    hoisted = getattr(norm, _HOISTED_WEIGHT_ATTR, None)
    if hoisted is not None:
        return hoisted
    return cast(torch.Tensor, norm.weight).float() + 1.0


def _unsupported_reason(layer: torch.nn.Module, model_config) -> str | None:
    layer = cast("Qwen3NextAttention", layer)
    mm_config = model_config.multimodal_config if model_config is not None else None
    if not layer.attn_output_gate:
        return "the attention has no output gate"
    if not layer.rotary_emb.is_neox_style:
        return "the rope is not NeoX style"
    if mm_config is not None and not mm_config.language_model_only:
        return "multimodal positions need M-RoPE; serve with --language-model-only"
    heads = (layer.num_heads, layer.num_kv_heads)
    if heads not in measured.ATTENTION_HEADS:
        return (
            f"{heads[0]} query and {heads[1]} kv heads per device were not "
            "measured (only the Qwen3.5-397B-A17B legs at TP1, TP2 and TP8)"
        )
    return None


def _log_armed_rows(vllm_config: VllmConfig) -> None:
    spec = vllm_config.speculative_config
    rows_per_request = 1 + (spec.num_speculative_tokens if spec is not None else 0)
    max_decode_rows = vllm_config.scheduler_config.max_num_seqs * rows_per_request
    logger.info_once(
        "[Attention] Fused q/k norm + rope: buckets of <= %d rows run the fused "
        "q/k norm + rope kernel, i.e. decode steps of <= %d requests at %d rows "
        "each; the largest decode step has %d rows and runs the %s path.",
        MAX_ROWS,
        MAX_ROWS // rows_per_request,
        rows_per_request,
        max_decode_rows,
        "fused" if max_decode_rows <= MAX_ROWS else "eager",
    )


def _prebuild(layer: torch.nn.Module, model_config) -> bool:
    layer = cast("Qwen3NextAttention", layer)
    reason = _unsupported_reason(layer, model_config)
    if reason is not None:
        logger.warning_once(
            "[Attention] Fused q/k norm + rope: %s; running the eager q/k norm "
            "and rope.",
            reason,
        )
        return False
    rot = layer.rotary_emb.rotary_dim
    assert 0 < rot <= layer.head_dim and rot % 2 == 0, (rot, layer.head_dim)
    _build_op(_op_key(layer))
    _log_armed_rows(get_current_vllm_config())
    return True


def project_qkv_gate(
    layer: torch.nn.Module, qkv: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``Qwen3NextAttention._project_qkv_gate`` through the fused kernel."""
    layer = cast("Qwen3NextAttention", layer)
    q_gate, k, v = qkv.split([layer.q_size * 2, layer.kv_size, layer.kv_size], dim=-1)
    if positions.ndim == 2:
        # Text-only M-RoPE: the three position rows agree.
        positions = positions[0]
    cos_sin = layer.rotary_emb.cos_sin_cache.index_select(0, positions.to(torch.int64))
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
    """Route ``Qwen3NextAttention`` to the fused kernel custom op.

    Idempotent. Each layer arms the fused path at construction
    (:func:`_prebuild`) and keeps vLLM's eager path when the kernel cannot
    compute it or was not measured at its head counts.
    """
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    if getattr(Qwen3NextAttention._project_qkv_gate, _INSTALLED_ATTR, False):
        return
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

    setattr(_project_qkv_gate, _INSTALLED_ATTR, True)
    Qwen3NextAttention.__init__ = __init__
    Qwen3NextAttention._project_qkv_gate = _project_qkv_gate


def maybe_install_qk_norm_rope_patch(model_config: ModelConfig | None) -> bool | None:
    """Qwen3.5 rule: the engine the kernel was measured on
    (``measured.is_measured_model``: Qwen3.5-397B-A17B on TPU7x) runs the fused
    q/k norm + rope.

    Registered as a model-specific patch (``patch_registry``). Returns False
    until such a model config is available, so other models, including
    Qwen3-Next, which shares ``Qwen3NextAttention``, and every other Qwen3.5
    checkpoint or device, never install it and keep vLLM's eager path. The
    patch is class-wide, so the MTP draft's attention, built after the
    target's config is seen, takes the fused path too. A config without
    architectures (the stand-ins callers hand the patch registry) is not a
    Qwen3.5 config.
    """
    if not measured.is_measured_model(model_config):
        return False
    install_qk_norm_rope_patch()
    logger.info("Applied TPU patch: Qwen3.5 fused q/k norm + rope.")
    return None
