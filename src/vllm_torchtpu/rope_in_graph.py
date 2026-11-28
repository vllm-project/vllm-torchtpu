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
"""In-graph RoPE specification, registry, evaluator, cache stand-in, and Tier-1 mixin.

Tier 1 (`layers/adapter/rotary_embedding.py`, `mla_attention_op.py`) mixes in
`Tier1InGraphRopeMixin` to install a 0-row `_compute_cos_sin_cache` stub and a
`RopeInGraphSpec` + zero-HBM `InGraphCosSinCache` at `__init__` time. The
post-load sweep for plain `nn.Module` rotary classes and export recovery lives
in `vllm_torchtpu.rope_patch`.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.custom_op import op_registry_oot
from vllm.model_executor.layers import rotary_embedding as upstream
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
    Ernie4_5_VLRotaryEmbedding,
)

from vllm_torchtpu import rope_in_graph_flag
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class CosSinSegment:
    """Rows `[start, start + length)` of a cache: `[cos | sin](p * inv_freq) * mscale`.

    `p` counts from 0 at `start`; `inv_freq` holds one value per frequency.
    """

    start: int
    length: int
    inv_freq: tuple[float, ...]
    mscale: float = 1.0


@dataclasses.dataclass(frozen=True)
class RopeInGraphSpec:
    """Analytical in-graph specification for a rotary embedding module."""

    segments: tuple[CosSinSegment, ...]
    num_rows: int
    rotary_dim: int
    cache_dtype: torch.dtype
    match_query_dtype: bool = True
    is_neox_style: bool = True
    is_mrope: bool = False
    dim_modalities: tuple[int, ...] | None = None
    active_angles: int | None = None


@dataclasses.dataclass(frozen=True)
class RopeClassEntry:
    """How one upstream rotary class is computed in-graph."""

    # None: out of scope, the class keeps its table and forward.
    build_spec: Callable[[Any], RopeInGraphSpec | None] | None = None
    # Which OOT wrapper converts it at __init__; None: post-load sweep only.
    tier1: Literal["static", "mrope", "custom_op"] | None = None
    # M-RoPE only: (module, rotary_dim // 2) -> per-frequency modality.
    dim_modalities: Callable[[Any, int], tuple[int, ...] | None] | None = None


def mrope_aliases(mrope_cls: type) -> tuple[type, ...]:
    """Both `MRotaryEmbedding` and any plugin alias (e.g. `vllm_omni/patch.py`
    replaces `MRotaryEmbedding` with `OmniMRotaryEmbedding`)."""
    return tuple(
        dict.fromkeys(
            (
                mrope_cls,
                *(c for c in mrope_cls.__mro__ if c.__name__ == "MRotaryEmbedding"),
            )
        )
    )


def unwrap_oot_rope_class(cls: type) -> type:
    """Unwraps one registered OOT `CustomOp` subclass layer to its upstream base."""
    for base in cls.__bases__:
        if op_registry_oot.get(base.__name__) is cls:
            return base
    return cls


def to_f32_tuple(inv_freq: torch.Tensor) -> tuple[float, ...]:
    """Returns `inv_freq` flattened to a tuple of its fp32 values."""
    return tuple(float(x) for x in inv_freq.detach().cpu().float().flatten().tolist())


def _scaled_num_rows(max_position_embeddings: int, scaling_factor: float) -> int:
    return max(0, math.ceil(float(max_position_embeddings) * float(scaling_factor)))


def _is_power_of_two(val: float) -> bool:
    return math.isfinite(val) and val > 0.0 and math.frexp(val)[0] == 0.5


def _section_to_modalities(
    section: Sequence[int],
    half: int,
    *,
    interleaved: bool,
) -> tuple[int, ...]:
    if interleaved:
        out_interleaved: list[int] = []
        for ch in range(half):
            if ch % 3 == 1 and ch < int(section[1]) * 3:
                out_interleaved.append(1)
            elif ch % 3 == 2 and len(section) > 2 and ch < int(section[2]) * 3:
                out_interleaved.append(2)
            else:
                out_interleaved.append(0)
        return tuple(out_interleaved)
    out_contig: list[int] = []
    for mod_idx, count in enumerate(section):
        out_contig.extend([mod_idx] * int(count))
    return tuple(out_contig)


def _mrope_section_modalities(rope: Any, half: int) -> tuple[int, ...] | None:
    sec = rope.mrope_section
    if not sec or len(sec) > 3 or sum(int(x) for x in sec) != half:
        return None
    interleaved = bool(rope.mrope_interleaved)
    if interleaved and len(sec) < 2:
        return None
    return _section_to_modalities(sec, half, interleaved=interleaved)


def _mrope_dim_modalities(rope: Any, half: int) -> tuple[int, ...] | None:
    mrope_dim = getattr(rope, "mrope_dim", None)
    return tuple(int(x) for x in mrope_dim[:half]) if mrope_dim is not None else None


def _ernie_modalities(rope: Any, half: int) -> tuple[int, ...] | None:
    sec = rope.mrope_section
    if not sec or len(sec) != 3:
        return None
    sec_h, sec_w, sec_t = int(sec[0]), int(sec[1]), int(sec[2])
    if sec_h != sec_w or sec_h + sec_w + sec_t != half:
        return None
    return tuple(([1, 2] * sec_h) + ([0] * sec_t))


def _bailing_modalities(rope: Any, half: int) -> tuple[int, ...] | None:
    spatial = int(rope.spatial_size)
    return tuple(
        [1 if i % 2 == 0 else 2 for i in range(spatial)] + [0] * (half - spatial)
    )


def compute_mrope_dim_modalities(module: nn.Module) -> tuple[int, ...] | None:
    """Returns the per-frequency modality `{0, 1, 2}` for `[3, N]` M-RoPE positions."""
    half = int(getattr(module, "rotary_dim", 0)) // 2
    if half <= 0:
        return None

    if isinstance(module, RotaryEmbeddingBase):
        entry = registry_entry(module)
        if entry is not None and entry.dim_modalities is not None:
            return entry.dim_modalities(module, half)
        return None

    mrope_dim = _mrope_dim_modalities(module, half)
    if mrope_dim is not None:
        return mrope_dim

    mrope_section = getattr(module, "mrope_section", None)
    if not mrope_section:
        return None
    interleaved = bool(getattr(module, "mrope_interleaved", False))
    if interleaved and len(mrope_section) < 2:
        return None
    return _section_to_modalities(mrope_section, half, interleaved=interleaved)


def active_rope_angles(inv_freq: tuple[float, ...], declared: int | None) -> int | None:
    """Leading angle pairs that actually rotate, or None when all of them do.

    Proportional RoPE (Gemma-4 global layers) zero-pads `inv_freq` for the pairs
    it does not rotate, so their cos/sin rows are the identity (1, 0) at every
    position and rotating them is a no-op in fp32 (`x * 1.0 - y * 0.0 == x`).
    Only exact trailing zeros count. A module that declares `rope_angles`
    (Gemma4RotaryEmbedding) must agree with the zero count, else the shortcut
    is disabled.
    """
    rot = len(inv_freq)
    while rot > 0 and inv_freq[rot - 1] == 0.0:
        rot -= 1
    if rot == 0 or rot == len(inv_freq):
        return None
    if declared is not None and int(declared) != rot:
        return None
    return rot


def _one_segment_spec(
    rope: Any,
    inv_freq: torch.Tensor,
    num_rows: int,
    *,
    mscale: float = 1.0,
    match_query_dtype: bool = True,
    is_mrope: bool = False,
    dim_modalities: tuple[int, ...] | None = None,
) -> RopeInGraphSpec:
    inv_tuple = to_f32_tuple(inv_freq)
    # Identity shortcut (`cos == 1, sin == 0`) holds only when `mscale == 1.0`.
    active_angles = (
        active_rope_angles(inv_tuple, getattr(rope, "rope_angles", None))
        if mscale == 1.0
        else None
    )
    return RopeInGraphSpec(
        segments=(CosSinSegment(0, num_rows, inv_tuple, mscale),),
        num_rows=num_rows,
        rotary_dim=int(rope.rotary_dim),
        cache_dtype=rope.dtype,
        match_query_dtype=match_query_dtype,
        is_neox_style=bool(rope.is_neox_style),
        is_mrope=is_mrope,
        dim_modalities=dim_modalities,
        active_angles=active_angles,
    )


def _base_spec(rope: Any) -> RopeInGraphSpec:
    return _one_segment_spec(
        rope,
        rope._compute_inv_freq(rope.base),
        int(rope.max_position_embeddings),
    )


def _linear_spec(rope: Any) -> RopeInGraphSpec | None:
    # XLA's AlgebraicSimplifier rewrites `p / sf` to `p * (1 / sf)`, which is
    # only bit-exact when `sf` is a positive power of two. For a power of two,
    # `(p / sf) * f == p * (f / sf)` in float32, so the factor is folded into
    # the constants; non-power-of-two factors keep the precomputed table.
    if not all(_is_power_of_two(float(sf)) for sf in rope.scaling_factors):
        return None
    base_inv = rope._compute_inv_freq(rope.base)
    segs: list[CosSinSegment] = []
    offset = 0
    for sf in rope.scaling_factors:
        sf_f = float(sf)
        seg_len = _scaled_num_rows(rope.max_position_embeddings, sf_f)
        segs.append(CosSinSegment(offset, seg_len, to_f32_tuple(base_inv / sf_f)))
        offset += seg_len
    return RopeInGraphSpec(
        segments=tuple(segs),
        num_rows=offset,
        rotary_dim=int(rope.rotary_dim),
        cache_dtype=rope.dtype,
        match_query_dtype=True,
        is_neox_style=bool(rope.is_neox_style),
    )


def _dynamic_ntk_spec(rope: Any) -> RopeInGraphSpec:
    rotary_dim = int(rope.rotary_dim)
    sf_f = float(rope.scaling_factor)
    scaled_base = float(rope.base) * (
        (sf_f * float(rope.max_position_embeddings) / float(rope.max_trained_positions))
        - (sf_f - 1.0)
    ) ** (float(rotary_dim) / (float(rotary_dim) - 2.0))
    return _one_segment_spec(
        rope,
        rope._compute_inv_freq(scaled_base),
        int(rope.max_position_embeddings),
    )


def _ntk_alpha_spec(rope: Any) -> RopeInGraphSpec:
    rotary_dim = int(rope.rotary_dim)
    scaled_base = float(rope.base) * (
        float(rope.scaling_alpha) ** (float(rotary_dim) / (float(rotary_dim) - 2.0))
    )
    return _one_segment_spec(
        rope,
        rope._compute_inv_freq(scaled_base),
        int(rope.max_position_embeddings),
    )


def _yarn_spec(rope: Any, *, match_query_dtype: bool = True) -> RopeInGraphSpec:
    num_rows = _scaled_num_rows(rope.max_position_embeddings, rope.scaling_factor)
    return _one_segment_spec(
        rope,
        rope._compute_inv_freq(rope.scaling_factor),
        num_rows,
        mscale=float(rope.mscale),
        match_query_dtype=match_query_dtype,
    )


def longrope_segments(rope: Any) -> tuple[CosSinSegment, CosSinSegment]:
    """Returns the `(short, long)` `CosSinSegment` pair for a LongRoPE module."""
    short_len = int(rope.original_max_position_embeddings)
    long_len = int(rope.max_position_embeddings)
    short_inv = to_f32_tuple(rope._compute_inv_freq(rope.short_factor))
    long_inv = to_f32_tuple(rope._compute_inv_freq(rope.long_factor))
    short_mscale = float(getattr(rope, "short_mscale", 1.0) or 1.0)
    long_mscale = float(getattr(rope, "long_mscale", 1.0) or 1.0)
    return (
        CosSinSegment(0, short_len, short_inv, short_mscale),
        CosSinSegment(short_len, long_len, long_inv, long_mscale),
    )


def _longrope_spec(rope: Any) -> RopeInGraphSpec:
    short_seg, long_seg = longrope_segments(rope)
    return RopeInGraphSpec(
        segments=(short_seg, long_seg),
        num_rows=short_seg.length + long_seg.length,
        rotary_dim=int(rope.rotary_dim),
        cache_dtype=rope.long_short_cos_sin_cache.dtype,
        match_query_dtype=False,
        is_neox_style=True,
    )


def _mrope_spec(
    rope: Any,
    *,
    match_query_dtype: bool,
    sectionless_ok: bool = False,
    yarn_ok: bool = False,
) -> RopeInGraphSpec | None:
    if yarn_ok and rope.scaling_factor is not None:
        inv = rope._compute_inv_freq(rope.scaling_factor)
        num_rows = _scaled_num_rows(rope.max_position_embeddings, rope.scaling_factor)
        mscale = float(rope.mscale)
    else:
        inv = rope._compute_inv_freq(rope.base)
        num_rows = int(rope.max_position_embeddings)
        mscale = 1.0
    dim_mods = compute_mrope_dim_modalities(rope)
    if dim_mods is None and (not sectionless_ok or bool(rope.mrope_section)):
        return None
    return _one_segment_spec(
        rope,
        inv,
        num_rows,
        mscale=mscale,
        match_query_dtype=match_query_dtype,
        is_mrope=True,
        dim_modalities=dim_mods,
    )


@functools.cache
def _build_registry(mrope_cls: type) -> Mapping[type, RopeClassEntry]:
    mrope_entry = RopeClassEntry(
        build_spec=functools.partial(
            _mrope_spec, match_query_dtype=True, sectionless_ok=True, yarn_ok=True
        ),
        tier1="mrope",
        dim_modalities=_mrope_section_modalities,
    )
    entries: dict[type, RopeClassEntry] = {
        upstream.RotaryEmbedding: RopeClassEntry(build_spec=_base_spec, tier1="static"),
        upstream.Llama3RotaryEmbedding: RopeClassEntry(
            build_spec=_base_spec, tier1="static"
        ),
        upstream.Gemma4RotaryEmbedding: RopeClassEntry(
            build_spec=_base_spec, tier1="static"
        ),
        upstream.NTKScalingRotaryEmbedding: RopeClassEntry(
            build_spec=_base_spec, tier1="static"
        ),
        upstream.LinearScalingRotaryEmbedding: RopeClassEntry(
            build_spec=_linear_spec, tier1="static"
        ),
        upstream.DynamicNTKScalingRotaryEmbedding: RopeClassEntry(
            build_spec=_dynamic_ntk_spec, tier1="static"
        ),
        upstream.DynamicNTKAlphaRotaryEmbedding: RopeClassEntry(
            build_spec=_ntk_alpha_spec, tier1="static"
        ),
        upstream.YaRNScalingRotaryEmbedding: RopeClassEntry(
            build_spec=_yarn_spec, tier1="static"
        ),
        upstream.DeepseekScalingRotaryEmbedding: RopeClassEntry(
            build_spec=functools.partial(_yarn_spec, match_query_dtype=False),
            tier1="custom_op",
        ),
        upstream.Phi3LongRoPEScaledRotaryEmbedding: RopeClassEntry(
            build_spec=_longrope_spec, tier1=None
        ),
    }
    # Aliases first so a plugin alias (`OmniMRotaryEmbedding`) precedes
    # `MRotaryEmbedding` in `TIER1_MROPE_CLASSES`.
    for alias in mrope_aliases(mrope_cls):
        entries[alias] = mrope_entry
    entries.update(
        {
            upstream.MRotaryEmbeddingInterleaved: RopeClassEntry(
                build_spec=functools.partial(_mrope_spec, match_query_dtype=False),
                tier1="mrope",
                dim_modalities=_mrope_dim_modalities,
            ),
            Ernie4_5_VLRotaryEmbedding: RopeClassEntry(
                build_spec=functools.partial(
                    _mrope_spec, match_query_dtype=False, sectionless_ok=True
                ),
                tier1="mrope",
                dim_modalities=_ernie_modalities,
            ),
            upstream.BailingMRotaryEmbedding: RopeClassEntry(
                build_spec=functools.partial(_mrope_spec, match_query_dtype=True),
                tier1="mrope",
                dim_modalities=_bailing_modalities,
            ),
            upstream.FourierRotaryEmbedding: RopeClassEntry(),
            upstream.Llama4VisionRotaryEmbedding: RopeClassEntry(),
            upstream.DualChunkRotaryEmbedding: RopeClassEntry(),
            upstream.DeepseekV4ScalingRotaryEmbedding: RopeClassEntry(),
        }
    )
    return types.MappingProxyType(entries)


def rope_class_registry() -> Mapping[type, RopeClassEntry]:
    """The one table from exact upstream class to spec builder."""
    # Re-read `upstream.MRotaryEmbedding` each call: a plugin may alias it later.
    return _build_registry(upstream.MRotaryEmbedding)


def registry_entry(module: nn.Module) -> RopeClassEntry | None:
    """Returns the registry entry for `module`'s unwrapped exact class, or None."""
    return rope_class_registry().get(unwrap_oot_rope_class(type(module)))


def spec_from_module(module: nn.Module) -> RopeInGraphSpec | None:
    """Extracts an exact-class `RopeInGraphSpec` from `module` on CPU in FP32.

    Dispatches strictly on the unwrapped exact upstream class in
    `rope_class_registry()` (never `isinstance`) and returns `None` for
    out-of-scope or unrecognized classes.
    """
    existing = getattr(module, "_in_graph_spec", None)
    if isinstance(existing, RopeInGraphSpec):
        return existing
    entry = registry_entry(module)
    if entry is None or entry.build_spec is None:
        return None
    with torch.device("cpu"):
        return entry.build_spec(module)


def _build_device_vector_f32(
    values: tuple[float, ...],
    device: torch.device,
) -> torch.Tensor:
    """Materializes host-computed fp32 `inv_freq` values as in-graph constants.

    Each value is emitted as an fp32 scalar constant and stacked so the exact
    host-computed values reach the compiled graph unchanged rather than being
    recomputed on the device.
    """
    return torch.stack(
        [torch.full((), float(v), dtype=torch.float32, device=device) for v in values]
    )


def _select_mrope_positions(
    pos_2d: torch.Tensor,
    dim_modalities: tuple[int, ...],
) -> torch.Tensor:
    """Selects per-frequency positions `[N, D/2]` from `pos_2d[3, N]` in pure HLO."""
    half = len(dim_modalities)
    dev = pos_2d.device
    ch = torch.arange(half, dtype=torch.int32, device=dev).unsqueeze(0)
    t_pos = pos_2d[0].unsqueeze(1).expand(-1, half)
    h_pos = pos_2d[1].unsqueeze(1).expand(-1, half)
    w_pos = pos_2d[2].unsqueeze(1).expand(-1, half)

    is_h = torch.zeros((1, half), dtype=torch.bool, device=dev)
    is_w = torch.zeros((1, half), dtype=torch.bool, device=dev)
    for idx, mod in enumerate(dim_modalities):
        if mod == 1:
            is_h = is_h | (ch == idx)
        elif mod == 2:
            is_w = is_w | (ch == idx)
    return torch.where(is_w, w_pos, torch.where(is_h, h_pos, t_pos))


def _trig_rows(
    freqs: torch.Tensor,
    half: Literal["cos", "sin"] | None,
) -> torch.Tensor:
    if half == "cos":
        return freqs.cos()
    if half == "sin":
        return freqs.sin()
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def eval_spec_cos_sin(
    spec: RopeInGraphSpec,
    pos_for_freq: torch.Tensor,
    target_dtype: torch.dtype,
    *,
    half: Literal["cos", "sin"] | None = None,
    active_angles: int | None = None,
) -> torch.Tensor:
    """Evaluates `[cos | sin]` (or one `half`) rows for 1-D or 2-D positions."""
    dev = pos_for_freq.device
    pos_2d = pos_for_freq.unsqueeze(-1) if pos_for_freq.ndim == 1 else pos_for_freq
    out: torch.Tensor | None = None
    for seg in spec.segments:
        rel = pos_2d - seg.start if seg.start else pos_2d
        inv_vals = (
            seg.inv_freq[:active_angles] if active_angles is not None else seg.inv_freq
        )
        inv = _build_device_vector_f32(inv_vals, dev)
        freqs = rel.to(torch.float32) * inv
        seg_rows = _trig_rows(freqs, half)
        if seg.mscale != 1.0:
            seg_rows = seg_rows * float(seg.mscale)
        if out is None:
            out = seg_rows
        else:
            if pos_for_freq.ndim == 1:
                in_seg = (pos_for_freq >= seg.start).unsqueeze(-1)
            elif half is None:
                cond = pos_2d >= seg.start
                in_seg = torch.cat((cond, cond), dim=-1)
            else:
                in_seg = pos_2d >= seg.start
            out = torch.where(in_seg, seg_rows, out)
    assert out is not None
    rows = out.to(spec.cache_dtype)
    if target_dtype != spec.cache_dtype:
        rows = rows.to(target_dtype)
    return rows


def rotate_fp32(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_tokens: int,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
    active_angles: int | None = None,
) -> torch.Tensor:
    """Rotates `x[..., :rotary_dim]` in FP32 and preserves trailing channels.

    With `active_angles` set (zero-padded proportional RoPE), repeats the FP32
    arithmetic of `ApplyRotaryEmb.forward_static(..., enable_fp32_compute=True)`
    on the leading active pairs only, matching full-width rotation bit-for-bit
    (identity pairs contribute `x * 1.0 - y * 0.0 == x`).
    """
    shape = x.shape
    x = x.view(num_tokens, -1, head_size)
    if active_angles is None:
        # Skip the no-op slice / cat on full-RoPE heads.
        x_rot = x if rotary_dim == head_size else x[..., :rotary_dim]
        out = ApplyRotaryEmb.forward_static(
            x_rot, cos, sin, is_neox_style, enable_fp32_compute=True
        )
        if rotary_dim < head_size:
            out = torch.cat((out, x[..., rotary_dim:]), dim=-1)
        return out.reshape(shape)

    c = cos.unsqueeze(-2).float()
    s = sin.unsqueeze(-2).float()
    if is_neox_style:
        half = rotary_dim // 2
        x1 = x[..., :active_angles].float()
        x2 = x[..., half : half + active_angles].float()
        o1 = (x1 * c - x2 * s).to(x.dtype)
        o2 = (x2 * c + x1 * s).to(x.dtype)
        out = torch.cat(
            (o1, x[..., active_angles:half], o2, x[..., half + active_angles :]),
            dim=-1,
        )
    else:  # GPT-J: pairs (2i, 2i+1), so active channels are the leading 2*active
        width = 2 * active_angles
        xr = x[..., :width].float()
        o1 = xr[..., ::2] * c - xr[..., 1::2] * s
        o2 = xr[..., 1::2] * c + xr[..., ::2] * s
        rot = torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)
        out = torch.cat((rot, x[..., width:]), dim=-1)
    return out.reshape(shape)


def _prepare_in_graph_positions(
    positions: torch.Tensor,
    dim_modalities: tuple[int, ...] | None,
    *,
    is_mrope: bool = False,
) -> tuple[torch.Tensor, int]:
    """Returns `(pos_for_freq, num_tokens)` for 1-D or 3-axis M-RoPE positions."""
    if positions.ndim >= 2:
        if dim_modalities is not None:
            if positions.shape[0] != 3:
                raise ValueError(
                    "In-graph M-RoPE requires positions with leading dimension 3 "
                    f"(T/H/W), got shape {tuple(positions.shape)}"
                )
            pos_2d = positions.reshape(3, -1)
            return _select_mrope_positions(pos_2d, dim_modalities), pos_2d.shape[-1]
        if is_mrope:
            raise ValueError(
                "In-graph M-RoPE requires mrope_section for multi-dimensional "
                f"positions, got shape {tuple(positions.shape)}"
            )
    pos_1d = positions.flatten()
    return pos_1d, pos_1d.shape[0]


def in_graph_spec_forward(
    self: Any,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unified in-graph rotary forward driven by `self._in_graph_spec`."""
    spec: RopeInGraphSpec = self._in_graph_spec
    # Upstream MRotaryEmbedding / BailingMRotaryEmbedding ignore `offsets`.
    if offsets is not None and not spec.is_mrope:
        positions = positions + offsets

    active = spec.active_angles
    dim_modalities = spec.dim_modalities
    if active is not None and dim_modalities is not None:
        dim_modalities = dim_modalities[:active]

    pos_for_freq, num_tokens = _prepare_in_graph_positions(
        positions, dim_modalities, is_mrope=spec.is_mrope
    )

    target_dtype = query.dtype if spec.match_query_dtype else spec.cache_dtype
    cos_sin = eval_spec_cos_sin(spec, pos_for_freq, target_dtype, active_angles=active)
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = rotate_fp32(
        query,
        cos,
        sin,
        num_tokens=num_tokens,
        head_size=self.head_size,
        rotary_dim=spec.rotary_dim,
        is_neox_style=spec.is_neox_style,
        active_angles=active,
    )
    if key is not None:
        key = rotate_fp32(
            key,
            cos,
            sin,
            num_tokens=num_tokens,
            head_size=self.head_size,
            rotary_dim=spec.rotary_dim,
            is_neox_style=spec.is_neox_style,
            active_angles=active,
        )
    return query, key


def in_graph_spec_get_cos_sin(
    self: Any,
    seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes `(cos, sin)` for `[0, seqlen)` from `self._in_graph_spec`."""
    spec: RopeInGraphSpec = self._in_graph_spec
    cache = getattr(self, "cos_sin_cache", None)
    device = (
        cache.device
        if isinstance(cache, InGraphCosSinCache)
        else getattr(self, "_in_graph_rope_device", torch.device("cpu"))
    )
    target_dtype = (
        cache.dtype if isinstance(cache, InGraphCosSinCache) else spec.cache_dtype
    )
    positions = torch.arange(seqlen, dtype=torch.int32, device=device)
    cos_sin = eval_spec_cos_sin(spec, positions, target_dtype)
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin


def _normalize_device(dev: torch.device) -> torch.device:
    if dev.type not in ("cpu", "meta") and dev.index is None:
        return torch.device(dev.type, 0)
    return dev


class InGraphCosSinCache:
    """Stands in for a rotary cos/sin buffer, computing looked-up rows in-graph.

    Supports what rotary forwards do with their cache: indexing rows by a
    position tensor or slice (`cache[idx]`, `cache[:seqlen]`,
    `cache.index_select(0, idx)`, `torch.index_select(cache, 0, idx)`), reading
    `.device` / `.dtype` / `.shape`, and `.to(device, dtype)`. Any other use
    raises, so a forward that needs the whole table fails at trace time.
    """

    def __init__(
        self,
        segments: Sequence[CosSinSegment],
        dtype: torch.dtype,
        num_rows: int,
        width: int,
        half: Literal["cos", "sin"] | None = None,
        *,
        device: torch.device | None = None,
        cache_dtype: torch.dtype | None = None,
    ) -> None:
        self.segments = tuple(segments)
        # "cos" or "sin" for a table holding only that half, else `[cos | sin]`.
        self._half: Literal["cos", "sin"] | None = half
        self.dtype = dtype
        self.cache_dtype = dtype if cache_dtype is None else cache_dtype
        self.device = (
            torch.device("cpu") if device is None else _normalize_device(device)
        )
        self.shape = torch.Size((num_rows, width))
        self.ndim = 2
        self._spec = RopeInGraphSpec(
            self.segments,
            num_rows,
            rotary_dim=width * (2 if half else 1),
            cache_dtype=self.cache_dtype,
        )

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def to(self, *args: Any, **kwargs: Any) -> InGraphCosSinCache:
        device: torch.device | str | None = kwargs.get("device")
        dtype: torch.dtype | None = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, (torch.device, str)):
                device = arg
            elif isinstance(arg, torch.Tensor):
                device = arg.device
                dtype = arg.dtype
        resolved_device = (
            torch.device(device)
            if isinstance(device, str)
            else (self.device if device is None else device)
        )
        if resolved_device.type == self.device.type and resolved_device.index is None:
            resolved_device = self.device
        else:
            resolved_device = _normalize_device(resolved_device)
        resolved_dtype = self.dtype if dtype is None else dtype
        if resolved_device == self.device and resolved_dtype == self.dtype:
            return self
        return InGraphCosSinCache(
            self.segments,
            resolved_dtype,
            *self.shape,
            half=self._half,
            device=resolved_device,
            cache_dtype=self.cache_dtype,
        )

    def index_select(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("rotary cache rows are selected on dim 0")
        return self._rows(index)

    def __getitem__(self, index: Any) -> torch.Tensor:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            idx = torch.arange(start, stop, step, dtype=torch.int64, device=self.device)
            return self._rows(idx)
        if not isinstance(index, torch.Tensor):
            raise NotImplementedError(
                f"rotary cache indexed by {type(index).__name__}; only position "
                "tensors or slices can be computed in-graph"
            )
        return self._rows(index)

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        if func in (torch.index_select, torch.Tensor.index_select):
            named = dict(zip(("input", "dim", "index"), args), **(kwargs or {}))
            return named["input"].index_select(named["dim"], named["index"])
        raise NotImplementedError(
            f"{getattr(func, '__name__', func)} on an in-graph rotary cache"
        )

    def _rows(self, index: torch.Tensor) -> torch.Tensor:
        if index.ndim == 1:
            return eval_spec_cos_sin(self._spec, index, self.dtype, half=self._half)
        # eval_spec_cos_sin reads a 2-D input as per-frequency positions.
        rows = eval_spec_cos_sin(
            self._spec, index.reshape(-1), self.dtype, half=self._half
        )
        return rows.reshape(*index.shape, rows.shape[-1])


def install_tier1_in_graph_rope(module: nn.Module) -> RopeInGraphSpec | None:
    """Finalizes a Tier-1 OOT rotary module after `super().__init__` completes.

    Returns the installed `RopeInGraphSpec`, or `None` if `module`'s config is
    not eligible for in-graph evaluation (the caller falls back to the
    precomputed table).
    """
    rope: Any = module
    spec = spec_from_module(module)
    if spec is None:
        logger.warning_once(
            "[rope_in_graph] Tier-1 rotary class %s produced no in-graph spec; "
            "falling back to the precomputed cos_sin_cache table "
            "(additional_config[%r]=false disables in-graph RoPE globally).",
            type(module).__qualname__,
            rope_in_graph_flag.KEY,
        )
        return None
    scaling_factors = getattr(rope, "scaling_factors", None)
    if isinstance(scaling_factors, Sequence) and not isinstance(
        scaling_factors, (str, bytes)
    ):
        rope._scaling_factor_to_offset = {
            float(sf): seg.start
            for sf, seg in zip(scaling_factors, spec.segments, strict=True)
        }
    had_cache = "cos_sin_cache" in module._buffers
    for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
        module._buffers.pop(buf_name, None)
        module.__dict__.pop(buf_name, None)
    rope.cos_sin_cache_bf16 = None
    resolved_device = _normalize_device(torch.get_default_device())
    rope._in_graph_spec = spec
    rope._in_graph_rope_device = resolved_device
    rope._tpu_in_graph_tier1 = True
    if had_cache:
        rope.cos_sin_cache = InGraphCosSinCache(
            spec.segments,
            spec.cache_dtype,
            spec.num_rows,
            spec.rotary_dim,
            device=resolved_device,
            cache_dtype=spec.cache_dtype,
        )
    return spec


def _restore_upstream_table(rope: Any) -> None:
    for attr in (
        "cos_sin_cache",
        "_in_graph_spec",
        "_in_graph_rope_device",
        "_tpu_in_graph_tier1",
    ):
        rope.__dict__.pop(attr, None)
    cache = rope._compute_cos_sin_cache()
    if not rope.use_flashinfer:
        cache = cache.to(rope.dtype)
    rope.register_buffer("cos_sin_cache", cache, persistent=False)


def require_cos_sin_table(module: nn.Module) -> None:
    """Exempts `module` from the post-load sweep so `cos_sin_cache` stays a tensor.

    For callers that pass the table to a kernel. `module` must not be a Tier-1
    class: those are already converted at `__init__`.
    """
    # TODO(#2747): remove once the DeepSeek-V4 Pallas kernels take per-token
    # cos/sin.
    cast(Any, module)._tpu_rope_table_required = True


class Tier1InGraphRopeMixin:
    """Mixin for Tier-1 0-row cache stub, `_apply` propagation, and `get_cos_sin`.

    Place before the upstream `RotaryEmbeddingBase` subclass in the bases
    tuple. When `tpu_rope_in_graph` is set, `__init__` activates the Tier-1
    0-row cache stub; a module whose `spec_from_module` returns `None` falls
    back to the precomputed table and `_on_table_cache_init`.
    """

    _in_graph_spec: RopeInGraphSpec | None = None

    def _on_table_cache_init(self) -> None:
        """Hook run after `super().__init__` when keeping a precomputed table."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        use_tier1 = rope_in_graph_flag.in_graph_enabled(
            get_current_vllm_config_or_none()
        )
        if use_tier1:
            self._tpu_skip_cos_sin_cache_alloc = True
        try:
            super().__init__(*args, **kwargs)
        finally:
            self.__dict__.pop("_tpu_skip_cos_sin_cache_alloc", None)
        if use_tier1:
            spec = install_tier1_in_graph_rope(cast(nn.Module, self))
            if spec is not None:
                return
            # Fallback: rebuild the precomputed cache exactly as
            # `RotaryEmbeddingBase.__init__` would have when `init_cache` ran.
            rope: Any = self
            if "cos_sin_cache" in rope._buffers:
                _restore_upstream_table(rope)
        self._on_table_cache_init()

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        if getattr(self, "_tpu_skip_cos_sin_cache_alloc", False):
            rotary_dim = cast(Any, self).rotary_dim
            return torch.empty((0, rotary_dim), dtype=torch.float32, device="cpu")
        return cast(Any, super())._compute_cos_sin_cache()

    def _apply(self, fn: Any, recurse: bool = True) -> Any:
        res = cast(Any, super())._apply(fn, recurse=recurse)
        apply_in_graph_cache_transform(cast(nn.Module, self), fn)
        return res

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._in_graph_spec is not None:
            return in_graph_spec_get_cos_sin(self, seqlen)
        return cast(Any, super()).get_cos_sin(seqlen)


def apply_in_graph_cache_transform(module: nn.Module, fn: Any) -> None:
    """Propagates `nn.Module._apply` (`to` / `to_empty`) to `InGraphCosSinCache`."""
    cache = module.__dict__.get("cos_sin_cache")
    if isinstance(cache, InGraphCosSinCache):
        probe = fn(torch.empty(0, dtype=cache.dtype, device=cache.device))
        new_cache = cache.to(device=probe.device, dtype=probe.dtype)
        module.__dict__["cos_sin_cache"] = new_cache
        module.__dict__["_in_graph_rope_device"] = new_cache.device
