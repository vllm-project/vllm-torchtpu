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
"""Tier-1 in-graph & FP32 rotary embedding out-of-tree CustomOps for TPU.

Registers out-of-tree (`@CustomOp.register_oot`) TPU subclasses for:
* `ApplyRotaryEmb`: forces `enable_fp32_compute=True` (CUDA kernel numerics).
* Static-forward `RotaryEmbedding` subclasses (`TPU_ROPE_CLASSES`): discovered
  from the upstream MRO for FP32 rotation, and equipped with the Tier-1 0-row
  `_compute_cos_sin_cache` stub + `RopeInGraphSpec` on verified classes
  (`TIER1_STATIC_ROPE_CLASSES`) so `load_model()` allocates `0 B` peak and
  steady-state HBM when `TPU_ROPE_IN_GRAPH=1`.
* 3D multimodal `MRotaryEmbedding` subclasses (`TPU_MROPE_CLASSES`:
  `MRotaryEmbedding`, `MRotaryEmbeddingInterleaved`,
  `Ernie4_5_VLRotaryEmbedding`, `BailingMRotaryEmbedding`): same Tier-1 0-row
  cache stub and pure-HLO per-channel modality selection.

When `TPU_ROPE_IN_GRAPH=0`, all `Tpu*` classes fall back to upstream
`_compute_cos_sin_cache()` and FP32 table-gather rotation for bit-exact parity
with legacy precomputed-table execution.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers import rotary_embedding as upstream_rope
from vllm.model_executor.layers.rotary_embedding import (
    BailingMRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    Gemma4RotaryEmbedding,
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    MRotaryEmbedding,
    MRotaryEmbeddingInterleaved,
    NTKScalingRotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.bailing_mrope import (
    _validate_bailing_positions,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
    Ernie4_5_VLRotaryEmbedding,
)

from vllm_torchtpu import envs, rope_patch
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@CustomOp.register_oot(name="ApplyRotaryEmb")
class TpuApplyRotaryEmb(ApplyRotaryEmb):
    """`ApplyRotaryEmb` that always rotates in FP32, like the CUDA kernel."""

    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
        **kwargs: Any,
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


def _rotary_forward_fp32(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`RotaryEmbedding.forward_static` with `enable_fp32_compute=True`.

    The cos/sin cache is still matched to the query dtype first: the CUDA
    kernel reads the activation-dtype cache and upcasts the loaded entries,
    so the rounded cache values are part of the reference numerics.
    """
    cos_sin_cache = self._match_cos_sin_cache_dtype(query)
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    query_rot = ApplyRotaryEmb.forward_static(
        query_rot, cos, sin, self.is_neox_style, enable_fp32_compute=True
    )
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = ApplyRotaryEmb.forward_static(
            key_rot, cos, sin, self.is_neox_style, enable_fp32_compute=True
        )
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def _static_tier1_forward(
    self: RotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if getattr(self, "_in_graph_spec", None) is not None:
        return rope_patch._in_graph_spec_forward(
            self, positions, query, key, offsets=offsets
        )
    if offsets is not None:
        positions = positions + offsets
    return _rotary_forward_fp32(self, positions, query, key)


def _tier1_get_cos_sin(
    self: RotaryEmbeddingBase,
    seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(self, "_in_graph_spec", None) is not None:
        return rope_patch._in_graph_spec_get_cos_sin(self, seqlen)
    return RotaryEmbeddingBase.get_cos_sin(self, seqlen)


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _uses_static_rotary_forward(cls: type[RotaryEmbedding]) -> bool:
    """True if `cls.forward_native` resolves to `RotaryEmbedding`'s."""
    for c in cls.__mro__:
        if "forward_native" in c.__dict__:
            return c is RotaryEmbedding
    return False


def _discover_static_rope_classes() -> tuple[type[RotaryEmbedding], ...]:
    """Upstream RotaryEmbedding classes whose forward is the static path."""
    upstream_prefix = upstream_rope.__name__ + "."
    found = {
        c
        for c in _all_subclasses(RotaryEmbedding)
        if c.__module__.startswith(upstream_prefix)
        and _uses_static_rotary_forward(c)
        and c.__name__ not in op_registry_oot
    }
    return (RotaryEmbedding, *sorted(found, key=lambda c: c.__name__))


TIER1_STATIC_ROPE_CLASSES: frozenset[type[RotaryEmbedding]] = frozenset(
    {
        RotaryEmbedding,
        Llama3RotaryEmbedding,
        Gemma4RotaryEmbedding,
        NTKScalingRotaryEmbedding,
        LinearScalingRotaryEmbedding,
        DynamicNTKScalingRotaryEmbedding,
        DynamicNTKAlphaRotaryEmbedding,
        YaRNScalingRotaryEmbedding,
    }
)

_ORIG_MROPE_CLS: type[MRotaryEmbedding] = next(
    (
        c
        for c in MRotaryEmbedding.__mro__
        if c.__name__ == "MRotaryEmbedding" and issubclass(c, RotaryEmbeddingBase)
    ),
    MRotaryEmbedding,
)

TIER1_MROPE_CLASSES: tuple[type[MRotaryEmbedding], ...] = tuple(
    dict.fromkeys(
        (
            _ORIG_MROPE_CLS,
            MRotaryEmbedding,
            MRotaryEmbeddingInterleaved,
            Ernie4_5_VLRotaryEmbedding,
            BailingMRotaryEmbedding,
        )
    )
)

_STATIC_ROPE_CLASSES = _discover_static_rope_classes()


def _attach_tier1_lifecycle(
    attrs: dict[str, Any],
    base: type[RotaryEmbeddingBase],
    holder: list[type],
) -> None:
    """Adds outermost-`__init__` 0-row cache stub + `_apply` device propagation."""

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        use_tier1 = type(self) is holder[0] and envs.TPU_ROPE_IN_GRAPH
        if use_tier1:
            self._tpu_skip_cos_sin_cache_alloc = True
        try:
            base.__init__(self, *args, **kwargs)
        finally:
            self.__dict__.pop("_tpu_skip_cos_sin_cache_alloc", None)
        if use_tier1:
            rope_patch.install_tier1_in_graph_rope(
                self, device=torch.get_default_device()
            )

    def _compute_cos_sin_cache(self: Any) -> torch.Tensor:
        if getattr(self, "_tpu_skip_cos_sin_cache_alloc", False):
            return torch.empty((0, self.rotary_dim), dtype=torch.float32, device="cpu")
        return base._compute_cos_sin_cache(self)

    def _apply(self: Any, fn: Any, recurse: bool = True) -> Any:
        res = base._apply(self, fn, recurse=recurse)
        rope_patch.apply_in_graph_cache_transform(self, fn)
        return res

    attrs["__init__"] = __init__
    attrs["_compute_cos_sin_cache"] = _compute_cos_sin_cache
    attrs["_apply"] = _apply
    attrs["get_cos_sin"] = _tier1_get_cos_sin


def _make_tpu_rope_class(base: type[RotaryEmbedding]) -> type[RotaryEmbedding]:
    assert _uses_static_rotary_forward(base), (
        f"{base.__name__} defines its own forward_native; it needs its own TPU "
        "override instead of the shared RotaryEmbedding.forward_static replay."
    )
    holder: list[type] = []
    if base in TIER1_STATIC_ROPE_CLASSES:
        attrs: dict[str, Any] = {
            "__module__": __name__,
            "__doc__": f"`{base.__name__}` in-graph & FP32 RoPE on TPU.",
            "forward_native": _static_tier1_forward,
            "forward_oot": _static_tier1_forward,
            "forward_tpu": _static_tier1_forward,
        }
        _attach_tier1_lifecycle(attrs, base, holder)
    else:
        attrs = {
            "__module__": __name__,
            "__doc__": f"`{base.__name__}` rotating in FP32 (CUDA numerics).",
            "forward_native": _rotary_forward_fp32,
            "forward_oot": _rotary_forward_fp32,
            "forward_tpu": _rotary_forward_fp32,
        }
    tpu_cls = type(f"Tpu{base.__name__}", (base,), attrs)
    holder.append(tpu_cls)
    return CustomOp.register_oot(name=base.__name__)(tpu_cls)


def _make_tpu_mrope_class(base: type[MRotaryEmbedding]) -> type[MRotaryEmbedding]:
    holder: list[type] = []

    if base is BailingMRotaryEmbedding:

        def _mrope_forward(
            self: Any,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            offsets: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if getattr(self, "_in_graph_spec", None) is not None:
                if key is None:
                    raise ValueError("Bailing M-RoPE requires key states")
                _validate_bailing_positions(positions)
                return rope_patch._in_graph_spec_forward(
                    self, positions, query, key, offsets=offsets
                )
            return BailingMRotaryEmbedding.forward_native(
                self, positions, query, key, offsets
            )

    elif base is MRotaryEmbeddingInterleaved:

        def _mrope_forward(
            self: Any,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            offsets: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if getattr(self, "_in_graph_spec", None) is not None:
                return rope_patch._in_graph_spec_forward(
                    self, positions, query, key, offsets=offsets
                )
            if offsets is not None:
                positions = positions + offsets
            return MRotaryEmbeddingInterleaved.forward(self, positions, query, key)

    elif base is Ernie4_5_VLRotaryEmbedding:

        def _mrope_forward(
            self: Any,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            offsets: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if getattr(self, "_in_graph_spec", None) is not None:
                assert key is not None
                return rope_patch._in_graph_spec_forward(
                    self, positions, query, key, offsets=offsets
                )
            if offsets is not None:
                positions = positions + offsets
            return Ernie4_5_VLRotaryEmbedding.forward_native(
                self, positions, query, key
            )

    else:

        def _mrope_forward(
            self: Any,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            offsets: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if getattr(self, "_in_graph_spec", None) is not None:
                assert key is not None
                return rope_patch._in_graph_spec_forward(
                    self, positions, query, key, offsets=offsets
                )
            return MRotaryEmbedding.forward_native(self, positions, query, key, offsets)

    attrs: dict[str, Any] = {
        "__module__": __name__,
        "__doc__": f"`{base.__name__}` Tier-1 in-graph M-RoPE on TPU.",
        "forward_native": _mrope_forward,
        "forward_oot": _mrope_forward,
        "forward_tpu": _mrope_forward,
    }
    if base is MRotaryEmbeddingInterleaved:
        # MRotaryEmbeddingInterleaved overrides `forward` directly, bypassing
        # CustomOp._forward_method.
        attrs["forward"] = _mrope_forward
    _attach_tier1_lifecycle(attrs, base, holder)
    tpu_cls = type(f"Tpu{base.__name__}", (base,), attrs)
    holder.append(tpu_cls)
    return CustomOp.register_oot(name=base.__name__)(tpu_cls)


TPU_ROPE_CLASSES: dict[str, type[RotaryEmbedding]] = {
    base.__name__: _make_tpu_rope_class(base) for base in _STATIC_ROPE_CLASSES
}
TPU_MROPE_CLASSES: dict[str, type[MRotaryEmbedding]] = {
    base.__name__: _make_tpu_mrope_class(base) for base in TIER1_MROPE_CLASSES
}
TpuRotaryEmbedding = TPU_ROPE_CLASSES["RotaryEmbedding"]
TpuMRotaryEmbedding = TPU_MROPE_CLASSES[MRotaryEmbedding.__name__]

logger.info(
    "Registered TPU rotary embeddings: ApplyRotaryEmb, %s.",
    ", ".join((*TPU_ROPE_CLASSES, *TPU_MROPE_CLASSES)),
)
