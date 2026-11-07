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
"""Pure in-graph RotaryEmbedding and MRotaryEmbedding patching.

Eliminates static `cos_sin_cache` module buffers on eligible rotary embedding
modules (both 1-D standard `RotaryEmbedding` and 2-D `[3, num_tokens]` multimodal
`MRotaryEmbedding` / `MRotaryEmbeddingInterleaved`), computing `cos` and `sin`
directly in-graph from `positions` and `inv_freq`.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys
import types
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn


def _allowed_static_rope_forwards(rotary_cls: type) -> tuple[Any, ...]:
    """Allowed `forward_native` callables for static-forward `RotaryEmbedding`s."""
    # Look up via sys.modules rather than importing: importing rotary_embedding
    # runs vllm_torchtpu.layers package-init and globally registers Tpu* OOT
    # subclasses, which AOT export does not want. If nothing loaded the module,
    # no instance can use _rotary_forward_fp32.
    tpu_rope_mod = sys.modules.get("vllm_torchtpu.layers.adapter.rotary_embedding")
    if tpu_rope_mod is not None:
        return (rotary_cls.forward_native, tpu_rope_mod._rotary_forward_fp32)
    return (rotary_cls.forward_native,)


def is_in_graph_rope_eligible(module: nn.Module) -> bool:
    """Returns True if `module`'s cos/sin depend only on `positions` and `inv_freq`.

    Holds for:
    - `RotaryEmbedding`, its TPU out-of-tree `Tpu*` wrappers, and subclasses
      overriding only `_compute_inv_freq` (e.g. Llama3, NTK, Gemma4).
    - `MRotaryEmbedding` and `MRotaryEmbeddingInterleaved` without YaRN scaling
      (`scaling_factor is None`), supporting both 1-D `[num_tokens]` and 2-D
      `[3, num_tokens]` (T/H/W modality) positions.

    Excludes variants whose cache scales positions or values (YaRN `mscale`,
    linear and dynamic-NTK scaling) and variants with custom forwards (DeepSeek,
    Fourier) that read `cos_sin_cache` directly; cache-backed variants are
    handled via `InGraphCosSinCache` instead.
    """
    if hasattr(module, "_in_graph_rope_inv_freq"):
        return True
    try:
        from vllm.model_executor.layers.rotary_embedding import (
            MRotaryEmbedding,
            MRotaryEmbeddingInterleaved,
            RotaryEmbedding,
        )
        from vllm.model_executor.layers.rotary_embedding.base import (
            RotaryEmbeddingBase,
        )

        cls: Any = type(module)
        if isinstance(module, RotaryEmbeddingBase):
            if (
                isinstance(module, RotaryEmbedding)
                and cls.forward_native in _allowed_static_rope_forwards(RotaryEmbedding)
                and cls._compute_cos_sin_cache
                is RotaryEmbeddingBase._compute_cos_sin_cache
            ):
                return True
            if (
                isinstance(module, MRotaryEmbedding)
                and getattr(module, "scaling_factor", None) is None
            ):
                return cls in (MRotaryEmbedding, MRotaryEmbeddingInterleaved)
            return False
    except ImportError:
        pass
    return (
        hasattr(module, "_compute_inv_freq")
        and hasattr(module, "base")
        and hasattr(module, "rotary_dim")
        and hasattr(module, "head_size")
        and hasattr(module, "is_neox_style")
        and getattr(module, "scaling_factor", None) is None
        and getattr(module, "scaling_factors", None) is None
        and getattr(module, "scaling_alpha", None) is None
        and not hasattr(module, "cos_coef")
        and "cos_sin_cache" in getattr(module, "_buffers", {})
    )


def compute_mrope_dim_modalities(module: nn.Module) -> tuple[int, ...] | None:
    """Returns the per-frequency modality `{0, 1, 2}` for `[3, N]` M-RoPE positions."""
    half = int(getattr(module, "rotary_dim", 0)) // 2
    if half <= 0:
        return None

    mrope_dim = getattr(module, "mrope_dim", None)
    if mrope_dim is not None:
        return tuple(int(x) for x in mrope_dim[:half])

    mrope_section = getattr(module, "mrope_section", None)
    if not mrope_section:
        return None

    if getattr(module, "mrope_interleaved", False):
        out: list[int] = []
        for ch in range(half):
            if ch % 3 == 1 and ch < mrope_section[1] * 3:
                out.append(1)
            elif ch % 3 == 2 and len(mrope_section) > 2 and ch < mrope_section[2] * 3:
                out.append(2)
            else:
                out.append(0)
        return tuple(out)

    out = []
    for mod_idx, count in enumerate(mrope_section):
        out.extend([mod_idx] * int(count))
    return tuple(out)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """Applies rotary embedding to `x[num_tokens, num_heads, rotary_dim]`.

    Rotates in FP32 like vLLM's CUDA kernel (and the runtime TPU path in
    `vllm_torchtpu.layers.adapter.rotary_embedding`); `cos`/`sin` arrive
    already rounded to the cache dtype.
    """
    try:
        from vllm.model_executor.layers.rotary_embedding.base import ApplyRotaryEmb

        return ApplyRotaryEmb.forward_static(
            x, cos, sin, is_neox_style, enable_fp32_compute=True
        )
    except ImportError:
        orig_dtype = x.dtype
        x = x.float()
        cos_u = cos.unsqueeze(-2).float()
        sin_u = sin.unsqueeze(-2).float()
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
            o1 = x1 * cos_u - x2 * sin_u
            o2 = x2 * cos_u + x1 * sin_u
            return torch.cat((o1, o2), dim=-1).to(orig_dtype)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos_u - x2 * sin_u
        o2 = x2 * cos_u + x1 * sin_u
        return torch.stack((o1, o2), dim=-1).flatten(-2).to(orig_dtype)


def _build_device_vector_f32(
    values: tuple[float, ...],
    device: torch.device,
) -> torch.Tensor:
    """Materializes the host-computed fp32 `inv_freq` as an in-graph constant.

    The values are embedded bit-exactly (one scalar constant per frequency,
    folded by XLA into a single literal) rather than recomputed on the device
    as `base ** (2i / d)`: TPU `pow` is only accurate to ~8 ulp, which at
    position p becomes an angle error of `p * inv_freq * 5e-7` (up to 0.06 rad
    at 1M positions), while the embedded constants reproduce a CPU-built table
    bit-exactly after bf16 rounding.
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


def _in_graph_cos_sin(
    pos_for_freq: torch.Tensor,
    inv_freq: tuple[float, ...],
    cache_dtype: torch.dtype,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Returns `[cos | sin]` rows in `dtype` for `[N]` or `[N, D/2]` positions."""
    inv = _build_device_vector_f32(inv_freq, pos_for_freq.device)
    freqs = (
        torch.outer(pos_for_freq.to(torch.float32), inv)
        if pos_for_freq.ndim == 1
        else pos_for_freq.to(torch.float32) * inv
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(cache_dtype).to(dtype)


def _in_graph_rope_forward(
    self: Any,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """In-graph `RotaryEmbedding` / `MRotaryEmbedding` forward, no `cos_sin_cache`."""
    if offsets is not None:
        positions = positions + offsets

    inv_freq = self._in_graph_rope_inv_freq
    dim_modalities = getattr(self, "_in_graph_rope_dim_modalities", None)
    if positions.ndim >= 2 and dim_modalities is not None and positions.shape[0] == 3:
        pos_2d = positions.reshape(3, -1)
        num_tokens = pos_2d.shape[-1]
        pos_for_freq = _select_mrope_positions(pos_2d, dim_modalities)
    else:
        pos_for_freq = positions.flatten()
        num_tokens = pos_for_freq.shape[0]

    cos_sin = _in_graph_cos_sin(
        pos_for_freq, inv_freq, self._in_graph_rope_cache_dtype, query.dtype
    )
    cos, sin = cos_sin.chunk(2, dim=-1)
    query = _rotate_in_graph(self, query, cos, sin, num_tokens)
    if key is not None:
        key = _rotate_in_graph(self, key, cos, sin, num_tokens)
    return query, key


def _rotate_in_graph(
    self: Any,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Rotates `x[..., :rotary_dim]` in-graph and preserves any trailing channels."""
    shape = x.shape
    x = x.view(num_tokens, -1, self.head_size)
    rotary_dim = self.rotary_dim
    # Skip the no-op slice / cat on full-RoPE heads.
    x_rot = x if rotary_dim == self.head_size else x[..., :rotary_dim]
    out = _apply_rotary_emb(x_rot, cos, sin, self.is_neox_style)
    if rotary_dim < self.head_size:
        out = torch.cat((out, x[..., rotary_dim:]), dim=-1)
    return out.reshape(shape)


def _in_graph_get_cos_sin(
    self: Any,
    seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes `(cos, sin)` for `[0, seqlen)` without a preallocated cache."""
    device = getattr(self, "_in_graph_rope_device", torch.device("cpu"))
    positions = torch.arange(seqlen, dtype=torch.int32, device=device)
    cos_sin = _in_graph_cos_sin(
        positions,
        self._in_graph_rope_inv_freq,
        self._in_graph_rope_cache_dtype,
        self._in_graph_rope_cache_dtype,
    )
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin


def drop_unused_rope_cache_aliases(model: nn.Module) -> None:
    """Drops upstream CUDA `_rope_cos_sin_cache` aliases (DFlash / DSpark).

    Upstream `Qwen3DFlashModel` (`vllm/model_executor/models/qwen3_dflash.py`)
    and `Gemma4DSparkModel` (`vllm/model_executor/models/gemma4_dspark.py`)
    stash `attn0.rotary_emb.cos_sin_cache` as `_rope_cos_sin_cache` for CUDA's
    `ops.rotary_embedding` path; TPU drafters call `rotary_emb(...)` directly,
    so keeping the alias pins the original full-length buffer even after
    in-graph patching or `TPU_ROPE_CACHE_TRUNCATE`.
    """
    for _, module in model.named_modules():
        module.__dict__.pop("_rope_cos_sin_cache", None)


def patch_in_graph_rotary_embedding(
    model: nn.Module,
    *,
    _skip_ids: set[int] | None = None,
) -> int:
    """Replaces `cos_sin_cache` buffer lookup with pure in-graph RoPE on `model`.

    Scoped strictly to `model`'s module instances:
    - Computes `inv_freq` on CPU as a Python tuple constant
      (`_in_graph_rope_inv_freq`) so FX / StableHLO / `torch.compile` embeds it
      as an inline constant without triggering eager TPU compilations at load.
    - Precomputes `_in_graph_rope_dim_modalities` for `MRotaryEmbedding` and
      `MRotaryEmbeddingInterleaved` so 2-D `[3, num_tokens]` multimodal positions
      are indexed in-graph without `cos_sin_cache`.
    - Binds `_in_graph_rope_forward` to `forward_native`, `forward_oot`,
      `forward_tpu`, and `_forward_method` (plus `forward` for
      `MRotaryEmbeddingInterleaved`, and `get_cos_sin` for vision towers) on the
      instance only, never mutating vLLM classes.
    - Removes `cos_sin_cache`, `cos_sin_cache_bf16`, and
      `_tpu_compact_cos_sin_cache` from `module._buffers` and `module.__dict__`.
    """
    skip_ids = (
        _collect_deepseek_v4_pallas_ropes(model) if _skip_ids is None else _skip_ids
    )
    drop_unused_rope_cache_aliases(model)
    mrope_interleaved_cls: type | None = None
    try:
        from vllm.model_executor.layers.rotary_embedding import (
            MRotaryEmbeddingInterleaved,
        )

        mrope_interleaved_cls = MRotaryEmbeddingInterleaved
    except ImportError:
        pass

    num_patched = 0
    for _, module in model.named_modules():
        if id(module) in skip_ids or hasattr(module, "_in_graph_rope_inv_freq"):
            continue
        if not is_in_graph_rope_eligible(module):
            continue
        # Eligible modules carry RoPE attributes nn.Module's typing cannot see.
        rope: Any = module
        cache: Any = getattr(rope, "cos_sin_cache", None)
        cache_dtype = cache.dtype if isinstance(cache, torch.Tensor) else torch.float32
        cache_device = (
            cache.device if isinstance(cache, torch.Tensor) else torch.device("cpu")
        )
        with torch.device("cpu"):
            inv_freq_tensor = rope._compute_inv_freq(rope.base).detach().cpu().float()
        rope._in_graph_rope_inv_freq = tuple(inv_freq_tensor.tolist())
        rope._in_graph_rope_cache_dtype = cache_dtype
        rope._in_graph_rope_device = cache_device
        rope._in_graph_rope_dim_modalities = compute_mrope_dim_modalities(rope)
        bound = types.MethodType(_in_graph_rope_forward, rope)
        rope.forward_native = bound
        rope.forward_oot = bound
        rope.forward_tpu = bound
        rope._forward_method = bound
        rope.get_cos_sin = types.MethodType(_in_graph_get_cos_sin, rope)
        if (
            mrope_interleaved_cls is not None
            and isinstance(rope, mrope_interleaved_cls)
        ) or getattr(rope, "mrope_dim", None) is not None:
            rope.forward = bound
        for buf_name in (
            "cos_sin_cache",
            "cos_sin_cache_bf16",
            "_tpu_compact_cos_sin_cache",
        ):
            if hasattr(module, "_buffers"):
                module._buffers.pop(buf_name, None)
            module.__dict__.pop(buf_name, None)
        module.__dict__.pop("_tpu_compact_rope_angles", None)
        num_patched += 1
    return (
        num_patched
        + patch_cos_sin_cache_buffers(model, _skip_ids=skip_ids)
        + patch_cos_sin_table_pairs(model)
    )


@dataclasses.dataclass(frozen=True)
class CosSinSegment:
    """Rows `[start, start + length)` of a cache: `[cos | sin](p * inv_freq) * mscale`.

    `p` counts from 0 at `start`; `inv_freq` holds one value per frequency.
    """

    start: int
    length: int
    inv_freq: tuple[float, ...]
    mscale: float = 1.0


def _call_compute_inv_freq(module: Any) -> torch.Tensor | None:
    """Calls `module._compute_inv_freq` on CPU with the argument its class expects."""
    fn = getattr(module, "_compute_inv_freq", None)
    if not callable(fn):
        return None
    with torch.device("cpu"):
        # MRotaryEmbedding names its parameter `base` even when `scaling_factor`
        # delegates to `YaRNScalingRotaryEmbedding._compute_inv_freq(self, sf)`.
        try:
            from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

            if (
                isinstance(module, MRotaryEmbedding)
                and getattr(module, "scaling_factor", None) is not None
            ):
                out = fn(module.scaling_factor)
                return out if isinstance(out, torch.Tensor) and out.ndim == 1 else None
        except ImportError:
            pass

        params = list(inspect.signature(fn).parameters.values())
        if not params:
            out = fn()
        elif len(params) == 1 and hasattr(module, params[0].name):
            out = fn(getattr(module, params[0].name))
        elif hasattr(module, "base"):
            out = fn(module.base)
        else:
            return None
    return out if isinstance(out, torch.Tensor) and out.ndim == 1 else None


def _extract_raw_cos_sin_segments(
    rope: Any,
    num_rows: int,
) -> list[tuple[int, int, torch.Tensor, float]] | None:
    """Derives `(start, length, inv_freq, mscale)` segments from `rope`'s attributes."""
    # FoPE mutates `cos_sin_cache` on first forward using learned `cos_coef`/`sin_coef`.
    if hasattr(rope, "cos_coef") or getattr(rope, "update_cache", False):
        return None
    fn = getattr(rope, "_compute_inv_freq", None)
    if not callable(fn):
        return None

    with torch.device("cpu"):
        if (
            hasattr(rope, "short_factor")
            and hasattr(rope, "long_factor")
            and hasattr(rope, "original_max_position_embeddings")
            and hasattr(rope, "max_position_embeddings")
        ):
            short_len = int(rope.original_max_position_embeddings)
            long_len = int(rope.max_position_embeddings)
            if short_len + long_len != num_rows:
                return None
            short_inv = fn(rope.short_factor)
            long_inv = fn(rope.long_factor)
            short_mscale = float(getattr(rope, "short_mscale", 1.0) or 1.0)
            long_mscale = float(getattr(rope, "long_mscale", 1.0) or 1.0)
            return [
                (0, short_len, short_inv, short_mscale),
                (short_len, long_len, long_inv, long_mscale),
            ]

        scaling_factors = getattr(rope, "scaling_factors", None)
        if (
            isinstance(scaling_factors, Sequence)
            and not isinstance(scaling_factors, (str, bytes))
            and hasattr(rope, "base")
            and hasattr(rope, "max_position_embeddings")
        ):
            base_inv = fn(rope.base)
            if not isinstance(base_inv, torch.Tensor) or base_inv.ndim != 1:
                return None
            raw_linear: list[tuple[int, int, torch.Tensor, float]] = []
            offset = 0
            for sf in scaling_factors:
                sf_f = float(sf)
                seg_len = int(
                    torch.arange(
                        rope.max_position_embeddings * sf_f, dtype=torch.float32
                    ).numel()
                )
                raw_linear.append((offset, seg_len, base_inv / sf_f, 1.0))
                offset += seg_len
            return raw_linear if offset == num_rows else None

        if (
            hasattr(rope, "max_trained_positions")
            and getattr(rope, "scaling_factor", None) is not None
            and hasattr(rope, "max_position_embeddings")
            and hasattr(rope, "rotary_dim")
            and hasattr(rope, "base")
            and int(rope.rotary_dim) > 2
        ):
            sf = float(rope.scaling_factor)
            scaled_base = float(rope.base) * (
                (
                    sf
                    * float(rope.max_position_embeddings)
                    / float(rope.max_trained_positions)
                )
                - (sf - 1.0)
            ) ** (float(rope.rotary_dim) / (float(rope.rotary_dim) - 2.0))
            inv = fn(scaled_base)
            if not isinstance(inv, torch.Tensor) or inv.ndim != 1:
                return None
            return [(0, num_rows, inv, 1.0)]

        if (
            getattr(rope, "scaling_alpha", None) is not None
            and hasattr(rope, "rotary_dim")
            and hasattr(rope, "base")
            and int(rope.rotary_dim) > 2
        ):
            scaled_base = float(rope.base) * (
                float(rope.scaling_alpha)
                ** (float(rope.rotary_dim) / (float(rope.rotary_dim) - 2.0))
            )
            inv = fn(scaled_base)
            if not isinstance(inv, torch.Tensor) or inv.ndim != 1:
                return None
            return [(0, num_rows, inv, 1.0)]

    inv = _call_compute_inv_freq(rope)
    if inv is None:
        return None
    mscale = float(getattr(rope, "mscale", 1.0) or 1.0)
    return [(0, num_rows, inv, mscale)]


def _sample_segment_row_indices(start: int, length: int) -> torch.Tensor:
    """Sorted unique int64 row indices sampling prefix, interior, and tail."""
    if length <= 4096:
        return torch.arange(start, start + length, dtype=torch.int64)
    head = torch.arange(start, start + min(256, length), dtype=torch.int64)
    strided = (torch.arange(4096, dtype=torch.int64) * (length - 1)) // 4095 + start
    tail = torch.arange(start + max(0, length - 8), start + length, dtype=torch.int64)
    return torch.unique(torch.cat((head, strided, tail)), sorted=True)


def _module_cos_sin_segments(
    module: nn.Module,
    cache: torch.Tensor,
    *,
    _verified_memo: dict[Any, list[CosSinSegment]] | None = None,
) -> list[CosSinSegment] | None:
    """Extracts analytical `CosSinSegment`s from `module`'s RoPE attributes.

    Uses `module._compute_inv_freq` and `mscale` (or `short_factor` / `long_factor`,
    `scaling_factors`, DynamicNTK `scaled_base`) directly rather than fitting
    phases from buffer values, and verifies that the analytical formula reproduces
    `cache` within dtype rounding error.
    """
    if (
        cache.ndim != 2
        or not cache.dtype.is_floating_point
        or cache.shape[0] == 0
        or cache.shape[1] == 0
        or cache.shape[1] % 2 != 0
    ):
        return None

    raw_segments = _extract_raw_cos_sin_segments(module, int(cache.shape[0]))
    if raw_segments is None:
        return None

    half = cache.shape[1] // 2
    candidate_segments: list[CosSinSegment] = []
    for start, length, inv_t, mscale in raw_segments:
        inv_cpu = inv_t.detach().cpu().float().flatten()
        if inv_cpu.numel() != half:
            return None
        candidate_segments.append(
            CosSinSegment(
                start=start,
                length=length,
                inv_freq=tuple(float(x) for x in inv_cpu.tolist()),
                mscale=mscale,
            )
        )

    is_known_vllm_rope = False
    try:
        from vllm.model_executor.layers.rotary_embedding.base import (
            RotaryEmbeddingBase,
        )

        is_known_vllm_rope = isinstance(module, RotaryEmbeddingBase)
    except ImportError:
        pass

    memo_key = (
        type(module),
        tuple(cache.shape),
        cache.dtype,
        tuple(candidate_segments),
    )
    if is_known_vllm_rope and _verified_memo is not None and memo_key in _verified_memo:
        return _verified_memo[memo_key]

    eps = torch.finfo(cache.dtype).eps
    eps32 = torch.finfo(torch.float32).eps
    # `.cpu()` is a raw PJRT D2H DMA with zero eager XLA compilations; sample and
    # cast on CPU so verification stays O(1) in `max_position` too.
    rows_cpu = cache.detach().cpu()
    for seg in candidate_segments:
        if seg.length <= 0:
            continue
        if seg.start < 0 or seg.start + seg.length > rows_cpu.shape[0]:
            return None
        idx = _sample_segment_row_indices(seg.start, seg.length)
        actual = rows_cpu[idx].float()
        inv_cpu = torch.tensor(seg.inv_freq, dtype=torch.float32)
        t = (idx - seg.start).float()
        freqs = torch.outer(t, inv_cpu)
        approx = torch.cat((freqs.cos() * seg.mscale, freqs.sin() * seg.mscale), dim=-1)
        tol = abs(seg.mscale) * (4 * eps + 8 * eps32 * freqs.abs().repeat(1, 2))
        if not bool(((approx - actual).abs() <= tol + 1e-6).all()):
            return None

    if is_known_vllm_rope and _verified_memo is not None:
        _verified_memo[memo_key] = candidate_segments
    return candidate_segments


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
        self.half = half
        self.dtype = dtype
        self.cache_dtype = dtype if cache_dtype is None else cache_dtype
        self.device = (
            torch.device("cpu") if device is None else _normalize_device(device)
        )
        self.shape = torch.Size((num_rows, width))
        self.ndim = 2

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
            half=self.half,
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
        kwargs = dict(kwargs or {})
        if func in (torch.index_select, torch.Tensor.index_select):
            cache, dim, index = (*args, *kwargs.values())[:3]
            return cache.index_select(dim, index)
        raise NotImplementedError(
            f"{getattr(func, '__name__', func)} on an in-graph rotary cache"
        )

    def _rows(self, index: torch.Tensor) -> torch.Tensor:
        device = index.device
        out: torch.Tensor | None = None
        for seg in self.segments:
            pos = index - seg.start if seg.start else index
            inv = _build_device_vector_f32(seg.inv_freq, device)
            freqs = pos.to(torch.float32).unsqueeze(-1) * inv
            if self.half == "cos":
                rows = freqs.cos()
            elif self.half == "sin":
                rows = freqs.sin()
            else:
                rows = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
            if seg.mscale != 1.0:
                rows = rows * float(seg.mscale)
            if out is None:
                out = rows
            else:
                in_seg = (index >= seg.start).unsqueeze(-1)
                out = torch.where(in_seg, rows, out)
        assert out is not None
        return out.to(self.cache_dtype).to(self.dtype)


_DSV4_PALLAS_OWNER_SPECS: tuple[tuple[str, str], ...] = (
    (
        "vllm_torchtpu.models.vllm.deepseek_v4.attention",
        "VllmDeepseekV4MLAAttention",
    ),
    (
        "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer",
        "VllmDeepseekV4Indexer",
    ),
    (
        "vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor",
        "VllmDeepseekCompressor",
    ),
)


def _collect_deepseek_v4_pallas_ropes(*models: nn.Module) -> set[int]:
    """Object ids of rotary modules consumed as raw tensors by DSv4 Pallas ops."""
    # TODO(#2747): Remove once DeepSeek-V4 Pallas kernels consume per-token cos_sin.
    dsv4_rope_cls: type | None = None
    try:
        from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
            DeepseekV4ScalingRotaryEmbedding,
        )

        dsv4_rope_cls = DeepseekV4ScalingRotaryEmbedding
    except ImportError:
        pass

    # Look up via sys.modules rather than importing: importing DSv4 modules
    # requires torch_tpu/JAX and registers Pallas ops globally. Any model
    # containing an instance of an owner class has already imported its module;
    # direct getattr (no default) fails loudly if a class is renamed (R-4.1).
    pallas_owner_classes: list[type] = []
    for mod_name, cls_name in _DSV4_PALLAS_OWNER_SPECS:
        owner_mod = sys.modules.get(mod_name)
        if owner_mod is not None:
            pallas_owner_classes.append(getattr(owner_mod, cls_name))
    pallas_owner_tuple = tuple(pallas_owner_classes)

    skip_ids: set[int] = set()
    for model in models:
        for mod in model.modules():
            if dsv4_rope_cls is not None and isinstance(mod, dsv4_rope_cls):
                skip_ids.add(id(mod))
            if pallas_owner_tuple and isinstance(mod, pallas_owner_tuple):
                for attr in ("rotary_emb", "indexer_rotary_emb"):
                    sub = getattr(mod, attr, None)
                    if isinstance(sub, nn.Module):
                        skip_ids.add(id(sub))
                mla_attn = getattr(mod, "mla_attn", None)
                if mla_attn is not None and mla_attn is not mod:
                    for attr in ("rotary_emb", "indexer_rotary_emb"):
                        sub = getattr(mla_attn, attr, None)
                        if isinstance(sub, nn.Module):
                            skip_ids.add(id(sub))
    return skip_ids


def _recover_rope_segment_from_table(
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> CosSinSegment | None:
    """Content-based fallback recovering `(inv_freq, mscale)` from `(cos, sin)`.

    `cos`/`sin` `[n, w]` must hold `mscale * cos/sin(p * f_j)` for row `p` and
    column `j`. `mscale` is read from row 0 (`cos[0] == mscale`, `sin[0] == 0`)
    in the table's dtype (so a bf16 table yields a bf16-rounded `mscale`, which
    is acceptable for this last-resort fallback), each `f_j` is unwrapped over
    rows 1, 2, 4, ... so its error shrinks with the row, and the rebuilt table
    must match within dtype rounding + float32 drift.
    """
    if (
        cos.ndim != 2
        or cos.shape != sin.shape
        or cos.shape[0] < 2
        or cos.shape[1] == 0
        or not cos.dtype.is_floating_point
        or not sin.dtype.is_floating_point
    ):
        return None
    n = int(cos.shape[0])
    c = cos.detach().cpu().double()
    s = sin.detach().cpu().double()

    eps = max(torch.finfo(cos.dtype).eps, torch.finfo(sin.dtype).eps)
    mscale = float(c[0].median().item())
    if mscale <= 0.0:
        return None
    row0_tol = max(1e-3, 2.0 * eps) * max(1.0, mscale)
    if (
        float((c[0] - mscale).abs().max()) > row0_tol
        or float(s[0].abs().max()) > row0_tol
    ):
        return None
    if abs(mscale - 1.0) <= 2.0 * eps:
        mscale = 1.0

    freq = torch.atan2(s[1], c[1])
    row = 1
    while 2 * row < n:
        row *= 2
        angle = torch.atan2(s[row], c[row])
        turns = torch.round((row * freq - angle) / (2 * math.pi))
        freq = (angle + 2 * math.pi * turns) / row
    inv_freq = freq.float()
    angles = torch.outer(torch.arange(n, dtype=torch.float32), inv_freq)
    err = max(
        float((angles.cos() * mscale - c.float()).abs().max()),
        float((angles.sin() * mscale - s.float()).abs().max()),
    )
    tol = (1e-3 + eps) * max(1.0, mscale)
    if err > tol:
        return None
    return CosSinSegment(
        start=0,
        length=n,
        inv_freq=tuple(float(x) for x in inv_freq.tolist()),
        mscale=mscale,
    )


def patch_cos_sin_cache_buffers(
    model: nn.Module,
    *,
    _skip_ids: set[int] | None = None,
) -> int:
    """Replaces remaining rotary `cos_sin_cache` buffers with `InGraphCosSinCache`.

    Matches modules analytically via `_module_cos_sin_segments` first (YaRN,
    DeepSeek, LongRoPE, Linear, DynamicNTK, M-RoPE+YaRN, ...) and falls back to
    content-based `_recover_rope_segment_from_table` only when analytical
    extraction is unavailable. Returns the number of modules patched.
    """
    skip_ids = (
        _collect_deepseek_v4_pallas_ropes(model) if _skip_ids is None else _skip_ids
    )
    num_patched = 0
    verified_memo: dict[Any, list[CosSinSegment]] = {}
    fallback_memo: dict[tuple[Any, ...], CosSinSegment | None] = {}
    for _, module in model.named_modules():
        # FoPE mutates `cos_sin_cache` on first forward using learned
        # `cos_coef` / `sin_coef`.
        if (
            id(module) in skip_ids
            or hasattr(module, "cos_coef")
            or getattr(module, "update_cache", False)
        ):
            continue
        patched = False
        for buf_name, buf in list(module._buffers.items()):
            if buf is None or "cos_sin_cache" not in buf_name:
                continue
            segments = _module_cos_sin_segments(
                module, buf, _verified_memo=verified_memo
            )
            if (
                segments is None
                and buf.ndim == 2
                and buf.dtype.is_floating_point
                and buf.shape[1] > 0
                and buf.shape[1] % 2 == 0
            ):
                memo_key = (
                    buf.untyped_storage().data_ptr(),
                    buf.storage_offset(),
                    tuple(buf.shape),
                    tuple(buf.stride()),
                    buf.dtype,
                )
                if memo_key not in fallback_memo:
                    half = buf.shape[1] // 2
                    fallback_memo[memo_key] = _recover_rope_segment_from_table(
                        buf[:, :half], buf[:, half:]
                    )
                recovered = fallback_memo[memo_key]
                if recovered is not None:
                    segments = [recovered]
            if segments is None:
                continue
            del module._buffers[buf_name]
            setattr(
                module,
                buf_name,
                InGraphCosSinCache(
                    segments,
                    buf.dtype,
                    buf.shape[0],
                    buf.shape[1],
                    device=buf.device,
                    cache_dtype=buf.dtype,
                ),
            )
            patched = True
        num_patched += patched
    return num_patched


def patch_cos_sin_table_pairs(model: nn.Module) -> int:
    """Replaces paired `cos`/`sin` rotary table buffers with in-graph rows.

    For rotary modules that keep the two halves as separate `[max_len, d / 2]`
    buffers (Muse Spark's), indexed by position: each pair that
    `_recover_rope_segment_from_table` recognizes becomes two
    `InGraphCosSinCache` halves, so `@main` takes no table input. Returns the
    number of modules patched.
    """
    num_patched = 0
    for _, module in model.named_modules():
        cos, sin = module._buffers.get("cos"), module._buffers.get("sin")
        if cos is None or sin is None:
            continue
        segment = _recover_rope_segment_from_table(cos, sin)
        if segment is None:
            continue
        for half, buf in (("cos", cos), ("sin", sin)):
            del module._buffers[half]
            setattr(
                module,
                half,
                InGraphCosSinCache(
                    [segment],
                    buf.dtype,
                    *buf.shape,
                    half=half,
                    device=buf.device,
                    cache_dtype=buf.dtype,
                ),
            )
        num_patched += 1
    return num_patched
