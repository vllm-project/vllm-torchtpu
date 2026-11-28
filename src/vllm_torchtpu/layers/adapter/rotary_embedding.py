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
"""FP32 and Tier-1 in-graph rotary embedding for TPU as out-of-tree CustomOps.

vLLM's CUDA rotary kernel (``csrc/pos_encoding_kernels.cu``) loads the
(activation-dtype) cos/sin cache entries and the q/k values, upcasts all of
them to ``float``, rotates, and casts once on store. The PyTorch reference,
``ApplyRotaryEmb.forward_static``, only does that when
``enable_fp32_compute=True``; the default is ``False`` and
``RotaryEmbedding.forward_static`` never passes it, so on TPU the rotation
``x1 * cos - x2 * sin`` runs in BF16. For RL rollouts the resulting drift in
q/k shows up as trainer/sampler log-prob mismatch.

Two registrations restore the CUDA numerics, following the vLLM
device-extension design (``docs/design/custom_op.md``) rather than patching
vLLM internals. ``CustomOp.__new__`` substitutes an out-of-tree class by the
upstream class name, so each upstream class that needs the fix gets one entry:

* ``ApplyRotaryEmb``: constructed with ``enable_fp32_compute=True``. Covers
  every caller that goes through an instance, in particular
  ``MRotaryEmbedding`` (Qwen3.5 / Qwen3-VL) and its subclasses.
* ``RotaryEmbedding`` and every upstream subclass that inherits its
  ``forward_native`` (Llama3, YaRN, NTK, linear, Gemma4 scaling, ...; found by
  walking the class hierarchy): ``forward_static`` calls
  ``ApplyRotaryEmb.forward_static`` directly, so the instance flag does not
  reach it. The TPU subclass replays ``forward_static`` with
  ``enable_fp32_compute=True`` and strips trailing identity ``(cos=1, sin=0)``
  pairs from NeoX ``cos_sin_cache`` buffers.

``forward_native`` is overridden as well as ``forward_oot``/``forward_tpu`` so
the numerics do not depend on ``CompilationConfig.custom_ops`` enabling the op:
CUDA runs the FP32 kernel in both cases, and so should TPU.

Tier-1 in-graph RoPE (on by default; opt out via
``--additional-config '{"tpu_rope_in_graph": false}'``):

* Verified static-forward classes (``TIER1_STATIC_ROPE_CLASSES``) and 3D
  multimodal classes (``TIER1_MROPE_CLASSES``), both derived from
  ``rope_in_graph.rope_class_registry``, mix in ``Tier1InGraphRopeMixin``,
  installing a 0-row ``_compute_cos_sin_cache`` stub + ``RopeInGraphSpec`` at
  ``__init__`` so ``load_model()`` allocates 0 B peak and steady-state HBM.
* Without it, all ``Tpu*`` classes keep upstream
  ``_compute_cos_sin_cache()`` and FP32 table-gather rotation (with NeoX
  trailing-identity compaction on static classes), matching legacy execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers import rotary_embedding as upstream_rope
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    MRotaryEmbeddingInterleaved,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
    Ernie4_5_VLRotaryEmbedding,
)

from vllm_torchtpu import rope_in_graph
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@CustomOp.register_oot(name="ApplyRotaryEmb")
class TpuApplyRotaryEmb(ApplyRotaryEmb):
    """``ApplyRotaryEmb`` that always rotates in FP32, like the CUDA kernel."""

    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
        **kwargs,
    ) -> None:
        # Mirror the upstream positional signature so callers passing
        # enable_fp32_compute positionally are overridden too, not rejected.
        del enable_fp32_compute  # always FP32 on TPU
        super().__init__(
            enforce_enable=enforce_enable,
            is_neox_style=is_neox_style,
            enable_fp32_compute=True,
            **kwargs,
        )

    def forward_oot(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_native(x, cos, sin)

    forward_tpu = forward_oot  # pyrefly: ignore[bad-override]


def _maybe_compact_neox_rope_cache(self: RotaryEmbedding) -> None:
    """Strip trailing identity ``(cos=1, sin=0)`` angle pairs on NeoX caches."""
    if not self.is_neox_style or self.rotary_dim <= 0 or self.rotary_dim % 2 != 0:
        return
    cache = self._buffers.get("cos_sin_cache")
    if (
        cache is None
        or cache.is_meta
        or cache.ndim != 2
        or cache.shape[0] == 0
        or cache.shape[1] != self.rotary_dim
    ):
        return

    half = self.rotary_dim // 2
    cos_full, sin_full = cache[:, :half], cache[:, half:]
    # Standard RoPE rotates every pair up to half - 1; exit early without
    # scanning the full cache when the last angle pair is not identity.
    if bool((cos_full[:, -1] != 1).any() or (sin_full[:, -1] != 0).any()):
        return

    active = ((cos_full != 1) | (sin_full != 0)).any(dim=0).nonzero(as_tuple=False)
    if active.numel() == 0:
        return
    rot_angles = int(active[-1].item()) + 1

    compact = torch.cat(
        (cos_full[:, :rot_angles], sin_full[:, :rot_angles]), dim=-1
    ).contiguous()
    self.register_buffer("cos_sin_cache", compact, persistent=False)
    logger.info_once(
        "Compacted %s NeoX cos_sin_cache width %d -> %d (active_angles=%d).",
        type(self).__name__,
        self.rotary_dim,
        2 * rot_angles,
        rot_angles,
    )


def _apply_compact_neox_rope_fp32(
    x_3d: torch.Tensor,
    rotary_dim: int,
    cos_f32: torch.Tensor,
    sin_f32: torch.Tensor,
) -> torch.Tensor:
    """Rotate active leading pairs of each NeoX rotary half-block in FP32.

    Args:
        x_3d: Input of shape ``[num_tokens, num_heads, head_size]`` with
            ``rotary_dim <= head_size``.
        rotary_dim: Width of the NeoX rotary prefix within ``head_size``.
        cos_f32: FP32 cosines of shape ``[num_tokens, 1, rot_angles]``.
        sin_f32: FP32 sines of shape ``[num_tokens, 1, rot_angles]``.

    Returns:
        Tensor matching ``x_3d`` shape and dtype with ``[..., :rot_angles]``
        and ``[..., half : half + rot_angles]`` (``half = rotary_dim // 2``)
        rotated in FP32 and ``[..., rot_angles:half]`` /
        ``[..., half + rot_angles:]`` passed through unchanged.
    """
    orig_dtype = x_3d.dtype
    half = rotary_dim // 2
    rot_angles = cos_f32.shape[-1]
    x1_rot = x_3d[..., :rot_angles].to(torch.float32)
    x1_pass = x_3d[..., rot_angles:half]
    x2_rot = x_3d[..., half : half + rot_angles].to(torch.float32)
    x2_pass = x_3d[..., half + rot_angles :]
    o1_rot = (x1_rot * cos_f32 - x2_rot * sin_f32).to(orig_dtype)
    o2_rot = (x2_rot * cos_f32 + x1_rot * sin_f32).to(orig_dtype)
    return torch.cat((o1_rot, x1_pass, o2_rot, x2_pass), dim=-1)


def _rotary_forward_fp32(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``RotaryEmbedding.forward_static`` with ``enable_fp32_compute=True``.

    The cos/sin cache is still matched to the query dtype first: the CUDA
    kernel reads the activation-dtype cache and upcasts the loaded entries,
    so the rounded cache values are part of the reference numerics.
    """
    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    # `_maybe_compact_neox_rope_cache` narrows `cos_sin_cache` below `rotary_dim`
    # when trailing NeoX angle pairs are exact `(cos=1, sin=0)`.
    if self.is_neox_style and cos.shape[-1] < self.rotary_dim // 2:
        cos_f32 = cos.unsqueeze(1).to(torch.float32)
        sin_f32 = sin.unsqueeze(1).to(torch.float32)
        query = _apply_compact_neox_rope_fp32(
            query.view(num_tokens, -1, self.head_size),
            self.rotary_dim,
            cos_f32,
            sin_f32,
        ).reshape(query.shape)
        if key is not None:
            key = _apply_compact_neox_rope_fp32(
                key.view(num_tokens, -1, self.head_size),
                self.rotary_dim,
                cos_f32,
                sin_f32,
            ).reshape(key.shape)
        return query, key

    query = rope_in_graph.rotate_fp32(
        query,
        cos,
        sin,
        num_tokens=num_tokens,
        head_size=self.head_size,
        rotary_dim=self.rotary_dim,
        is_neox_style=self.is_neox_style,
    )
    if key is not None:
        key = rope_in_graph.rotate_fp32(
            key,
            cos,
            sin,
            num_tokens=num_tokens,
            head_size=self.head_size,
            rotary_dim=self.rotary_dim,
            is_neox_style=self.is_neox_style,
        )
    return query, key


def _static_tier1_forward(
    self: Any,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if self._in_graph_spec is not None:
        return rope_in_graph.in_graph_spec_forward(
            self, positions, query, key, offsets=offsets
        )
    if offsets is not None:
        positions = positions + offsets
    return _rotary_forward_fp32(self, positions, query, key)


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _uses_static_rotary_forward(cls: type[RotaryEmbedding]) -> bool:
    """True if ``cls.forward_native`` resolves to ``RotaryEmbedding``'s.

    Classes with their own forward_native (MRotaryEmbedding family, FoPE,
    DeepSeek scaling, ...) do not run the static path; the MRoPE family is
    covered through TpuApplyRotaryEmb instead.
    """
    for c in cls.__mro__:
        if "forward_native" in c.__dict__:
            return c is RotaryEmbedding
    return False


def _discover_static_rope_classes() -> tuple[type[RotaryEmbedding], ...]:
    """Upstream RotaryEmbedding classes whose forward is the static path.

    Discovered from the class hierarchy (``vllm.model_executor.layers.
    rotary_embedding`` imports every variant), so new upstream scaling
    variants are picked up without editing this file.
    """
    upstream_prefix = upstream_rope.__name__ + "."
    found = {
        c
        for c in _all_subclasses(RotaryEmbedding)
        if c.__module__.startswith(upstream_prefix)
        and _uses_static_rotary_forward(c)
        and c.__name__ not in op_registry_oot
    }
    return (RotaryEmbedding, *sorted(found, key=lambda c: c.__name__))


_REGISTRY = rope_in_graph.rope_class_registry()
TIER1_STATIC_ROPE_CLASSES: frozenset[type[RotaryEmbedding]] = cast(
    frozenset[type[RotaryEmbedding]],
    frozenset(c for c, e in _REGISTRY.items() if e.tier1 == "static"),
)

# `vllm_omni/patch.py` (vllm-omni==0.32.0rc1) replaces `MRotaryEmbedding` with
# `OmniMRotaryEmbedding`, a subclass of upstream `MRotaryEmbedding`. Register
# both the original class and any patched alias so Tier-1 covers either entry.
TIER1_MROPE_CLASSES: tuple[type[MRotaryEmbedding], ...] = cast(
    tuple[type[MRotaryEmbedding], ...],
    tuple(c for c, e in _REGISTRY.items() if e.tier1 == "mrope"),
)

_STATIC_ROPE_CLASSES = _discover_static_rope_classes()


def _make_tpu_rope_class(base: type[RotaryEmbedding]) -> type[RotaryEmbedding]:
    assert _uses_static_rotary_forward(base), (
        f"{base.__name__} defines its own forward_native; it needs its own TPU "
        "override instead of the shared RotaryEmbedding.forward_static replay."
    )
    if base in TIER1_STATIC_ROPE_CLASSES:
        attrs: dict[str, Any] = {
            "__module__": __name__,
            "__doc__": f"``{base.__name__}`` in-graph & FP32 RoPE on TPU.",
            "_on_table_cache_init": _maybe_compact_neox_rope_cache,
            "forward_native": _static_tier1_forward,
            "forward_oot": _static_tier1_forward,
            "forward_tpu": _static_tier1_forward,
        }
        bases: tuple[type, ...] = (rope_in_graph.Tier1InGraphRopeMixin, base)
    else:

        def _tpu_rope_init(self: RotaryEmbedding, *args, **kwargs) -> None:
            base.__init__(self, *args, **kwargs)
            _maybe_compact_neox_rope_cache(self)

        attrs = {
            "__module__": __name__,
            "__doc__": (
                f"``{base.__name__}`` rotating in FP32 (CUDA numerics), "
                "stripping trailing identity NeoX cos/sin pairs."
            ),
            "__init__": _tpu_rope_init,
            "forward_native": _rotary_forward_fp32,
            "forward_oot": _rotary_forward_fp32,
            "forward_tpu": _rotary_forward_fp32,
        }
        bases = (base,)
    tpu_cls = type(f"Tpu{base.__name__}", bases, attrs)
    return CustomOp.register_oot(name=base.__name__)(tpu_cls)


def _mrope_table_forward(
    base: type[MRotaryEmbedding],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """``base``'s table forward as ``(self, positions, query, key, offsets)``."""
    if base is MRotaryEmbeddingInterleaved:
        # Overrides `forward` directly, without `offsets`.
        return lambda self, positions, query, key, offsets: base.forward(
            self, positions, query, key
        )
    if base is Ernie4_5_VLRotaryEmbedding:
        # Its `forward_native` takes no `offsets`.
        return lambda self, positions, query, key, offsets: base.forward_native(
            self, positions, query, key
        )
    return base.forward_native


def _make_tpu_mrope_class(base: type[MRotaryEmbedding]) -> type[MRotaryEmbedding]:
    table_forward = _mrope_table_forward(base)
    # Upstream `MRotaryEmbeddingInterleaved.forward` accepts `key=None`
    # (cross-layer KV sharing); the other M-RoPE forwards require a key.
    requires_key = base is not MRotaryEmbeddingInterleaved

    def _mrope_forward(
        self: Any,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._in_graph_spec is None:
            return table_forward(self, positions, query, key, offsets)
        if requires_key and key is None:
            raise ValueError(f"{base.__name__} requires key states")
        return rope_in_graph.in_graph_spec_forward(
            self, positions, query, key, offsets=offsets
        )

    attrs: dict[str, Any] = {
        "__module__": __name__,
        "__doc__": f"``{base.__name__}`` Tier-1 in-graph M-RoPE on TPU.",
        "forward_native": _mrope_forward,
        "forward_oot": _mrope_forward,
        "forward_tpu": _mrope_forward,
    }
    if base is MRotaryEmbeddingInterleaved:
        # MRotaryEmbeddingInterleaved overrides `forward` directly, bypassing
        # CustomOp._forward_method.
        attrs["forward"] = _mrope_forward
    tpu_cls = type(
        f"Tpu{base.__name__}", (rope_in_graph.Tier1InGraphRopeMixin, base), attrs
    )
    return CustomOp.register_oot(name=base.__name__)(tpu_cls)


TPU_ROPE_CLASSES: dict[str, type[RotaryEmbedding]] = {
    base.__name__: _make_tpu_rope_class(base) for base in _STATIC_ROPE_CLASSES
}
TPU_MROPE_CLASSES: dict[str, type[MRotaryEmbedding]] = {
    base.__name__: _make_tpu_mrope_class(base) for base in TIER1_MROPE_CLASSES
}
TpuRotaryEmbedding = TPU_ROPE_CLASSES["RotaryEmbedding"]

logger.info(
    "Registered TPU rotary embeddings: ApplyRotaryEmb, %s.",
    ", ".join((*TPU_ROPE_CLASSES, *TPU_MROPE_CLASSES)),
)
