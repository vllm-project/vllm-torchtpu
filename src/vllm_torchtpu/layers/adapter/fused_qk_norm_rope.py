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
buckets of at most :data:`MAX_ROWS` rows run the ``qk_norm_rope_gate`` kernel,
which holds its rows whole in VMEM, instead of the eager split, per-head norms
and rope; larger buckets keep the eager path. The fork needs a trace per side,
and only the backbone is re-traced per bucket group, so a layer compiled
outside it -- the MTP draft's -- keeps the eager path.
"""

import functools
from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.qk_norm_rope_gate import qk_norm_rope_gate
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The largest token bucket that takes the kernel.
MAX_ROWS = 64

# The (1 + w) buffer TPU_NORM_WEIGHT_HOIST keeps on each GemmaRMSNorm, when that
# feature is present and on. The buckets above MAX_ROWS run the norm modules,
# which then read it, and every bucket has to lift the same inputs for the
# compiled buckets to be handed between traces.
_HOISTED_WEIGHT_ATTR = "_tpu_weight_plus_one"

_ops: dict[tuple[int, int, int, int, float], Callable] = {}


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
        return qk_norm_rope_gate(
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
    hoisted = getattr(norm, _HOISTED_WEIGHT_ATTR, None)
    return norm.weight.float() + 1.0 if hoisted is None else hoisted


def _unsupported_graph_reason() -> str | None:
    """Why the graph this layer is compiled into cannot fork on the bucket.

    The fork reads the token count, which leaves a shape guard in the trace.
    Only the backbone is re-traced per bucket group
    (`TPUModelRunner._precompile_backbone`); a module built under another model
    tag -- the MTP or EAGLE draft, "eagle_head" -- compiles every bucket from
    its one trace, and the buckets on the far side of the fork would raise
    `ShapeSpecializationError` at its first forward.
    """
    # Read at call time: `set_model_tag` rebinds the module global.
    from vllm.compilation import backends

    if backends.model_tag != "backbone":
        return (
            f"the {backends.model_tag!r} graph is traced once for every bucket, "
            "so it cannot carry the row fork"
        )
    return None


def _unsupported_reason(layer: torch.nn.Module, model_config) -> str | None:
    reason = _unsupported_graph_reason()
    if reason is not None:
        return reason
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
    _build_op(_op_key(layer))
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
    """Route small buckets of ``Qwen3NextAttention`` to the fused kernel."""
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
        if self.fused_qk_norm_rope and qkv.shape[0] <= MAX_ROWS:
            return project_qkv_gate(self, qkv, positions)
        return original_project(self, qkv, positions)

    Qwen3NextAttention.__init__ = __init__
    Qwen3NextAttention._project_qkv_gate = _project_qkv_gate
