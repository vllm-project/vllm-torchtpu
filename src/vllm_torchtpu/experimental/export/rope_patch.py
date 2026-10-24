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
"""Pure in-graph RotaryEmbedding and MRotaryEmbedding patch for AOT StableHLO export.

Eliminates the `cos_sin_cache` module buffer on eligible rotary embedding modules
(both 1-D standard `RotaryEmbedding` and 2-D `[3, num_tokens]` multimodal
`MRotaryEmbedding` / `MRotaryEmbeddingInterleaved`) during export so that `@main`
does not take an extra static cache buffer input, computing `cos` and `sin`
directly in-graph from `positions` and `inv_freq`.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


def is_in_graph_rope_eligible(module: nn.Module) -> bool:
    """Returns True if `module`'s cos/sin depend only on `positions` and `inv_freq`.

    Holds for:
    - `RotaryEmbedding` and subclasses overriding only `_compute_inv_freq`
      (e.g. Llama3, NTK, Gemma4).
    - `MRotaryEmbedding` and `MRotaryEmbeddingInterleaved` without YaRN scaling
      (`scaling_factor is None`), supporting both 1-D `[num_tokens]` and 2-D
      `[3, num_tokens]` (T/H/W modality) positions.

    Excludes variants whose cache scales positions or values (YaRN `mscale`,
    linear and dynamic-NTK scaling) and variants with custom forwards (DeepSeek,
    Fourier) that read `cos_sin_cache` directly.
    """
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
        if (
            isinstance(module, RotaryEmbedding)
            and cls.forward_native is RotaryEmbedding.forward_native
            and cls._compute_cos_sin_cache is RotaryEmbeddingBase._compute_cos_sin_cache
        ):
            return True
        if (
            isinstance(module, MRotaryEmbedding)
            and getattr(module, "scaling_factor", None) is None
        ):
            return cls in (MRotaryEmbedding, MRotaryEmbeddingInterleaved)
    except ImportError:
        pass
    return (
        hasattr(module, "_compute_inv_freq")
        and hasattr(module, "base")
        and hasattr(module, "rotary_dim")
        and hasattr(module, "head_size")
        and hasattr(module, "is_neox_style")
        and getattr(module, "scaling_factor", None) is None
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
    """Applies rotary embedding to `x[num_tokens, num_heads, rotary_dim]`."""
    try:
        from vllm.model_executor.layers.rotary_embedding.base import ApplyRotaryEmb

        return ApplyRotaryEmb.forward_static(x, cos, sin, is_neox_style)
    except ImportError:
        cos_u = cos.unsqueeze(-2).to(x.dtype)
        sin_u = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
            o1 = x1 * cos_u - x2 * sin_u
            o2 = x2 * cos_u + x1 * sin_u
            return torch.cat((o1, o2), dim=-1)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos_u - x2 * sin_u
        o2 = x2 * cos_u + x1 * sin_u
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def _build_device_vector_f32(
    values: tuple[float, ...],
    device: torch.device,
    *,
    base: float | None = None,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Materializes a 1-D float32 frequency vector as pure in-graph HLO ops."""
    if base is not None and rotary_dim is not None and rotary_dim == len(values) * 2:
        dim_idx = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
        std_inv = 1.0 / (float(base) ** (dim_idx / float(rotary_dim)))
        if device.type != "cpu":
            return std_inv
        ref = torch.tensor(values, dtype=torch.float32, device="cpu")
        if torch.allclose(std_inv, ref, rtol=1e-5, atol=1e-6):
            return std_inv
    if device.type == "cpu":
        return torch.tensor(values, dtype=torch.float32, device=device)
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
    *,
    base: float | None = None,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    """Returns `[cos | sin]` rows in `dtype` for `[N]` or `[N, D/2]` positions."""
    inv = _build_device_vector_f32(
        inv_freq,
        pos_for_freq.device,
        base=base,
        rotary_dim=rotary_dim,
    )
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

    dim_modalities = getattr(self, "_in_graph_rope_dim_modalities", None)
    if positions.ndim >= 2 and dim_modalities is not None and positions.shape[0] == 3:
        pos_2d = positions.reshape(3, -1)
        num_tokens = pos_2d.shape[-1]
        pos_for_freq = _select_mrope_positions(pos_2d, dim_modalities)
    else:
        pos_for_freq = positions.flatten()
        num_tokens = pos_for_freq.shape[0]

    cos_sin = _in_graph_cos_sin(
        pos_for_freq,
        self._in_graph_rope_inv_freq,
        self._in_graph_rope_cache_dtype,
        query.dtype,
        base=float(getattr(self, "base", 10000.0)),
        rotary_dim=int(getattr(self, "rotary_dim", 0)),
    )
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = _apply_rotary_emb(
        query[..., : self.rotary_dim], cos, sin, self.is_neox_style
    )
    query = torch.cat((query_rot, query[..., self.rotary_dim :]), dim=-1).reshape(
        query_shape
    )
    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = _apply_rotary_emb(
            key[..., : self.rotary_dim], cos, sin, self.is_neox_style
        )
        key = torch.cat((key_rot, key[..., self.rotary_dim :]), dim=-1).reshape(
            key_shape
        )
    return query, key


def patch_in_graph_rotary_embedding(model: nn.Module) -> int:
    """Replaces `cos_sin_cache` buffer lookup with pure in-graph RoPE on `model`.

    Scoped strictly to the model instance being exported:
    - Computes `inv_freq` as a Python tuple constant (`_in_graph_rope_inv_freq`)
      so FX / StableHLO tracing embeds it as an inline constant rather than a
      lifted `@main` buffer argument.
    - Precomputes `_in_graph_rope_dim_modalities` for `MRotaryEmbedding` and
      `MRotaryEmbeddingInterleaved` so 2-D `[3, num_tokens]` multimodal positions
      are indexed in-graph without `cos_sin_cache`.
    - Binds `_in_graph_rope_forward` to `module.forward_native` and
      `module._forward_method` (plus `module.forward` for `MRotaryEmbeddingInterleaved`)
      on the instance only, never mutating vLLM classes.
    - Removes `cos_sin_cache` and `cos_sin_cache_bf16` from `module._buffers`
      and `module.__dict__`.
    """
    num_patched = 0
    for _, module in model.named_modules():
        if not is_in_graph_rope_eligible(module):
            continue
        # Eligible modules carry RoPE attributes nn.Module's typing cannot see.
        rope: Any = module
        cache: Any = getattr(rope, "cos_sin_cache", None)
        cache_dtype = cache.dtype if isinstance(cache, torch.Tensor) else torch.float32
        inv_freq_tensor = rope._compute_inv_freq(rope.base).detach().float().cpu()
        rope._in_graph_rope_inv_freq = tuple(inv_freq_tensor.tolist())
        rope._in_graph_rope_cache_dtype = cache_dtype
        rope._in_graph_rope_dim_modalities = compute_mrope_dim_modalities(rope)
        bound = types.MethodType(_in_graph_rope_forward, rope)
        rope.forward_native = bound
        rope._forward_method = bound
        if (
            type(rope).__name__ == "MRotaryEmbeddingInterleaved"
            or getattr(rope, "mrope_dim", None) is not None
        ):
            rope.forward = bound
        for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
            if hasattr(module, "_buffers"):
                module._buffers.pop(buf_name, None)
            module.__dict__.pop(buf_name, None)
        num_patched += 1
    return num_patched + patch_cos_sin_cache_buffers(model)


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
    """Calls `module._compute_inv_freq` with the attribute its signature names."""
    fn = getattr(module, "_compute_inv_freq", None)
    if not callable(fn):
        return None
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


def _module_cos_sin_segments(
    module: nn.Module, cache: torch.Tensor
) -> list[CosSinSegment] | None:
    """Extracts analytical `CosSinSegment`s from `module`'s RoPE attributes.

    Uses `module._compute_inv_freq` and `mscale` (or `short_factor` / `long_factor`
    on LongRoPE modules) directly rather than fitting phases from buffer values,
    and verifies that the analytical formula reproduces `cache` within dtype
    rounding error.
    """
    if (
        cache.ndim != 2
        or not cache.dtype.is_floating_point
        or cache.shape[0] == 0
        or cache.shape[1] % 2 != 0
    ):
        return None
    rope: Any = module
    fn = getattr(rope, "_compute_inv_freq", None)
    if not callable(fn):
        return None

    raw_segments: list[tuple[int, int, torch.Tensor, float]] = []
    if (
        hasattr(rope, "short_factor")
        and hasattr(rope, "long_factor")
        and hasattr(rope, "original_max_position_embeddings")
        and hasattr(rope, "max_position_embeddings")
    ):
        short_len = int(rope.original_max_position_embeddings)
        long_len = int(rope.max_position_embeddings)
        if short_len + long_len != cache.shape[0]:
            return None
        short_inv = fn(rope.short_factor)
        long_inv = fn(rope.long_factor)
        short_mscale = float(getattr(rope, "short_mscale", 1.0) or 1.0)
        long_mscale = float(getattr(rope, "long_mscale", 1.0) or 1.0)
        raw_segments.append((0, short_len, short_inv, short_mscale))
        raw_segments.append((short_len, long_len, long_inv, long_mscale))
    else:
        inv = _call_compute_inv_freq(rope)
        if inv is None:
            return None
        mscale = float(getattr(rope, "mscale", 1.0) or 1.0)
        raw_segments.append((0, cache.shape[0], inv, mscale))

    half = cache.shape[1] // 2
    eps = torch.finfo(cache.dtype).eps
    eps32 = torch.finfo(torch.float32).eps
    rows = cache.detach().to("cpu", torch.float32)
    segments: list[CosSinSegment] = []
    for start, length, inv_t, mscale in raw_segments:
        inv_cpu = inv_t.detach().to("cpu", torch.float32).flatten()
        if inv_cpu.numel() != half:
            return None
        t = torch.arange(length, dtype=torch.float32)
        freqs = torch.outer(t, inv_cpu)
        approx = torch.cat((freqs.cos() * mscale, freqs.sin() * mscale), dim=-1)
        seg = rows[start : start + length]
        tol = abs(mscale) * (4 * eps + 8 * eps32 * freqs.abs().repeat(1, 2))
        if not bool(((approx - seg).abs() <= tol + 1e-6).all()):
            return None
        segments.append(
            CosSinSegment(
                start=start,
                length=length,
                inv_freq=tuple(float(x) for x in inv_cpu.tolist()),
                mscale=mscale,
            )
        )
    return segments


class InGraphCosSinCache:
    """Stands in for a rotary cos/sin buffer, computing looked-up rows in-graph.

    Supports what rotary forwards do with their cache: indexing rows by a
    position tensor (`cache[idx]`, `cache.index_select(0, idx)`,
    `torch.index_select(cache, 0, idx)`) and `.to(device, dtype)`. Any other use
    raises, so a forward that needs the whole table fails at trace time.
    """

    def __init__(
        self,
        segments: Sequence[CosSinSegment],
        dtype: torch.dtype,
        num_rows: int,
        width: int,
    ) -> None:
        self.segments = tuple(segments)
        self.dtype = dtype
        self.shape = torch.Size((num_rows, width))
        self.ndim = 2

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def to(self, *args: Any, **kwargs: Any) -> InGraphCosSinCache:
        dtype = kwargs.get("dtype")
        for arg in args:
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, torch.Tensor):
                dtype = arg.dtype
        if dtype is None or dtype == self.dtype:
            return self
        return InGraphCosSinCache(self.segments, dtype, *self.shape)

    def index_select(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("rotary cache rows are selected on dim 0")
        return self._rows(index)

    def __getitem__(self, index: Any) -> torch.Tensor:
        if not isinstance(index, torch.Tensor):
            raise NotImplementedError(
                f"rotary cache indexed by {type(index).__name__}; only position "
                "tensors can be computed in-graph"
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
            rows = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
            if seg.mscale != 1.0:
                rows = rows * float(seg.mscale)
            if out is None:
                out = rows
            else:
                in_seg = (index >= seg.start).unsqueeze(-1)
                out = torch.where(in_seg, rows, out)
        assert out is not None
        return out.to(self.dtype)


def patch_cos_sin_cache_buffers(model: nn.Module) -> int:
    """Replaces every remaining rotary cos/sin cache buffer with in-graph rows.

    Covers rotary variants whose forward reads the table itself (YaRN, DeepSeek,
    LongRoPE, ...): each buffer whose name contains `cos_sin_cache` and whose
    rows `_module_cos_sin_segments` derives from the module's own `_compute_inv_freq`
    and `mscale` becomes an `InGraphCosSinCache`, so `@main` takes no table
    input. Returns the number of modules patched.
    """
    num_patched = 0
    for _, module in model.named_modules():
        patched = False
        for buf_name, buf in list(module._buffers.items()):
            if buf is None or "cos_sin_cache" not in buf_name:
                continue
            segments = _module_cos_sin_segments(module, buf)
            if segments is None:
                continue
            del module._buffers[buf_name]
            setattr(
                module,
                buf_name,
                InGraphCosSinCache(segments, buf.dtype, *buf.shape),
            )
            patched = True
        num_patched += patched
    return num_patched
