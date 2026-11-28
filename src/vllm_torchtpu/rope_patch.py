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
"""Tier-2 post-load and export sweep converting remaining rotary caches in-graph.

Converts rotary modules that Tier 1 (`vllm_torchtpu.rope_in_graph`) did not
handle at `__init__` time (plain `nn.Module` classes such as
`Phi3LongRoPEScaledRotaryEmbedding`, and modules built outside a
`tpu_rope_in_graph` config) via `rope_in_graph.spec_from_module`. With
`allow_table_recovery=True` (export only), also handles non-vLLM modules and
converts paired `(cos, sin)` buffers through `patch_cos_sin_table_pairs` and
`_recover_rope_segment_from_table` rather than `spec_from_module`.
"""

from __future__ import annotations

import inspect
import math
import types
from typing import Any

import torch
import torch.nn as nn
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbeddingInterleaved,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase

from vllm_torchtpu import rope_in_graph
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def is_in_graph_rope_eligible(module: nn.Module) -> bool:
    """Returns True if `module`'s cos/sin depend only on `positions` and `inv_freq`.

    Holds for:
    - `RotaryEmbedding`, its TPU out-of-tree `Tpu*` wrappers, and subclasses
      overriding only `_compute_inv_freq` (e.g. Llama3, NTK, Gemma4).
    - `MRotaryEmbedding`, `MRotaryEmbeddingInterleaved`, `Ernie4_5_VLRotaryEmbedding`,
      and `BailingMRotaryEmbedding` without YaRN scaling (`scaling_factor is None`),
      supporting both 1-D `[num_tokens]` and 2-D `[3, num_tokens]` positions.

    Excludes variants whose cache scales positions or values (YaRN `mscale`,
    linear and dynamic-NTK scaling) and variants with custom forwards (DeepSeek,
    Fourier) that read `cos_sin_cache` directly; cache-backed variants are
    handled via `RopeInGraphSpec` / `InGraphCosSinCache`.
    """
    if isinstance(module, RotaryEmbeddingBase):
        cls: Any = rope_in_graph.unwrap_oot_rope_class(type(module))
        if (
            isinstance(module, RotaryEmbedding)
            and cls.forward_native is RotaryEmbedding.forward_native
            and cls._compute_cos_sin_cache is RotaryEmbeddingBase._compute_cos_sin_cache
        ):
            return True
        entry = rope_in_graph.registry_entry(module)
        return (
            entry is not None
            and entry.tier1 == "mrope"
            and getattr(module, "scaling_factor", None) is None
        )
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


def drop_unused_rope_cache_aliases(model: nn.Module) -> None:
    """Drops upstream CUDA `_rope_cos_sin_cache` aliases (DFlash / DSpark).

    Upstream `Qwen3DFlashModel` (`vllm/model_executor/models/qwen3_dflash.py`)
    and `Gemma4DSparkModel` (`vllm/model_executor/models/gemma4_dspark.py`)
    stash `attn0.rotary_emb.cos_sin_cache` as `_rope_cos_sin_cache` for CUDA's
    `ops.rotary_embedding` path; TPU drafters call `rotary_emb(...)` directly
    and never read the alias. In serving it is already an `InGraphCosSinCache`
    (Tier 1 converts the shared module at `__init__`); in export it is the
    full-length table, which the alias would keep alive.
    """
    for _, module in model.named_modules():
        module.__dict__.pop("_rope_cos_sin_cache", None)


def _export_one_segment_spec(
    rope: Any,
    cache: Any,
) -> rope_in_graph.RopeInGraphSpec:
    # One-segment spec with `is_mrope=False` and `match_query_dtype=True`
    # reproduces the legacy export forward/get_cos_sin op for op (offsets
    # always added, N-D positions without modalities flattened).
    with torch.device("cpu"):
        inv_freq = rope_in_graph.to_f32_tuple(rope._compute_inv_freq(rope.base))
    num_rows = int(cache.shape[0]) if isinstance(cache, torch.Tensor) else 0
    cache_dtype = cache.dtype if isinstance(cache, torch.Tensor) else torch.float32
    return rope_in_graph.RopeInGraphSpec(
        segments=(rope_in_graph.CosSinSegment(0, num_rows, inv_freq),),
        num_rows=num_rows,
        rotary_dim=int(rope.rotary_dim),
        cache_dtype=cache_dtype,
        match_query_dtype=True,
        is_neox_style=bool(rope.is_neox_style),
        is_mrope=False,
        dim_modalities=rope_in_graph.compute_mrope_dim_modalities(rope),
        active_angles=rope_in_graph.active_rope_angles(
            inv_freq, getattr(rope, "rope_angles", None)
        ),
    )


def patch_in_graph_rotary_embedding(
    model: nn.Module,
    *,
    allow_table_recovery: bool,
) -> int:
    """Tier-2 post-load & export sweep replacing remaining `cos_sin_cache` buffers.

    Modules already converted at `__init__` time by Tier 1 (`_tpu_in_graph_tier1`)
    and modules marked with `rope_in_graph.require_cos_sin_table` are skipped.
    Remaining eligible modules (plain-`nn.Module` `Phi3LongRoPE`, CPU-traced
    `torch.export` models) are converted in-place via per-class specs.

    `allow_table_recovery` has no default, so each caller chooses. With `True`
    (export path), the sweep also attempts attribute/table-content recovery for
    non-vLLM modules and paired `(cos, sin)` tables (e.g. Muse Spark). With
    `False` (serving path), any table without a per-class spec is left untouched
    as a tensor.
    """
    drop_unused_rope_cache_aliases(model)
    num_patched = 0
    for _, module in model.named_modules():
        if getattr(module, "_tpu_rope_table_required", False):
            continue
        if getattr(module, "_in_graph_spec", None) is not None:
            continue
        if not is_in_graph_rope_eligible(module):
            continue
        rope: Any = module
        cache: Any = getattr(rope, "cos_sin_cache", None)
        cache_device = (
            cache.device if isinstance(cache, torch.Tensor) else torch.device("cpu")
        )
        spec = rope_in_graph.spec_from_module(module)
        if spec is None:
            if not allow_table_recovery:
                continue
            spec = _export_one_segment_spec(rope, cache)

        rope._in_graph_spec = spec
        rope._in_graph_rope_device = cache_device
        bound = types.MethodType(rope_in_graph.in_graph_spec_forward, rope)
        rope.forward_native = bound
        rope.forward_oot = bound
        rope.forward_tpu = bound
        rope._forward_method = bound
        rope.get_cos_sin = types.MethodType(
            rope_in_graph.in_graph_spec_get_cos_sin, rope
        )
        if (
            isinstance(rope, MRotaryEmbeddingInterleaved)
            or getattr(rope, "mrope_dim", None) is not None
        ):
            rope.forward = bound
        for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
            if hasattr(module, "_buffers"):
                module._buffers.pop(buf_name, None)
            module.__dict__.pop(buf_name, None)
        num_patched += 1
    return (
        num_patched
        + patch_cos_sin_cache_buffers(model, allow_table_recovery=allow_table_recovery)
        + (patch_cos_sin_table_pairs(model) if allow_table_recovery else 0)
    )


def _call_compute_inv_freq(module: Any) -> torch.Tensor | None:
    """Calls `module._compute_inv_freq` on CPU with the argument its class expects."""
    fn = getattr(module, "_compute_inv_freq", None)
    if not callable(fn):
        return None
    with torch.device("cpu"):
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
) -> list[rope_in_graph.CosSinSegment] | None:
    """Derives `CosSinSegment`s for synthetic modules."""
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
            return list(rope_in_graph.longrope_segments(rope))

    inv = _call_compute_inv_freq(rope)
    if inv is None:
        return None
    inv_vals = rope_in_graph.to_f32_tuple(inv)
    mscale = float(getattr(rope, "mscale", 1.0) or 1.0)
    return [rope_in_graph.CosSinSegment(0, num_rows, inv_vals, mscale)]


def _sample_segment_row_indices(start: int, length: int) -> torch.Tensor:
    """Sorted unique int64 row indices sampling prefix, interior, and tail."""
    if length <= 4096:
        return torch.arange(start, start + length, dtype=torch.int64)
    head = torch.arange(start, start + min(256, length), dtype=torch.int64)
    strided = (torch.arange(4096, dtype=torch.int64) * (length - 1)) // 4095 + start
    tail = torch.arange(start + max(0, length - 8), start + length, dtype=torch.int64)
    return torch.unique(torch.cat((head, strided, tail)), sorted=True)


def _is_known_vllm_rotary_instance(module: nn.Module) -> bool:
    return isinstance(
        module, (RotaryEmbeddingBase, *rope_in_graph.rope_class_registry())
    )


def _is_out_of_scope_vllm_rope(module: nn.Module) -> bool:
    entry = rope_in_graph.registry_entry(module)
    return entry is not None and entry.build_spec is None


def _module_cos_sin_segments(
    module: nn.Module,
    cache: torch.Tensor,
    *,
    _verified_memo: dict[Any, list[rope_in_graph.CosSinSegment]] | None = None,
    allow_table_recovery: bool = False,
) -> list[rope_in_graph.CosSinSegment] | None:
    """Extracts analytical `CosSinSegment`s from `module` and verifies `cache`."""
    if (
        cache.ndim != 2
        or not cache.dtype.is_floating_point
        or cache.shape[0] == 0
        or cache.shape[1] == 0
        or cache.shape[1] % 2 != 0
    ):
        return None

    is_known_vllm = _is_known_vllm_rotary_instance(module)
    candidate_segments: list[rope_in_graph.CosSinSegment]
    if is_known_vllm:
        spec = rope_in_graph.spec_from_module(module)
        if (
            spec is None
            or spec.num_rows != int(cache.shape[0])
            or spec.rotary_dim != int(cache.shape[1])
        ):
            return None
        candidate_segments = list(spec.segments)
    elif not allow_table_recovery:
        return None
    else:
        raw_segments = _extract_raw_cos_sin_segments(module, int(cache.shape[0]))
        if raw_segments is None:
            return None
        half = cache.shape[1] // 2
        if any(len(seg.inv_freq) != half for seg in raw_segments):
            return None
        candidate_segments = raw_segments

    memo_key = (
        type(module),
        tuple(cache.shape),
        cache.dtype,
        tuple(candidate_segments),
    )
    if is_known_vllm and _verified_memo is not None and memo_key in _verified_memo:
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
        # 2.5e-4 floor absorbs ~1.5e-4 OpenMP thread-slice jitter in PyTorch's
        # multi-threaded CPU `einsum` + `cos`/`sin` on high-core-count hosts,
        # preventing non-deterministic cross-worker splits under TP/DP/EP.
        tol = abs(seg.mscale) * (
            max(4 * eps, 2.5e-4) + 8 * eps32 * freqs.abs().repeat(1, 2)
        )
        if not bool(((approx - actual).abs() <= tol).all()):
            return None

    if is_known_vllm and _verified_memo is not None:
        _verified_memo[memo_key] = candidate_segments
    return candidate_segments


def _recover_rope_segment_from_table(
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> rope_in_graph.CosSinSegment | None:
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
    return rope_in_graph.CosSinSegment(
        start=0,
        length=n,
        inv_freq=tuple(float(x) for x in inv_freq.tolist()),
        mscale=mscale,
    )


def patch_cos_sin_cache_buffers(
    model: nn.Module,
    *,
    allow_table_recovery: bool = False,
) -> int:
    """Replaces remaining rotary `cos_sin_cache` buffers with `InGraphCosSinCache`.

    Matches `vLLM` rotary modules strictly via exact-class `spec_from_module` +
    sampled CPU verification. When `allow_table_recovery=True` (export path
    only), also runs attribute/content-based recovery for non-`RotaryEmbeddingBase`
    model-local buffers.
    """
    num_patched = 0
    verified_memo: dict[Any, list[rope_in_graph.CosSinSegment]] = {}
    fallback_memo: dict[tuple[Any, ...], rope_in_graph.CosSinSegment | None] = {}
    for _, module in model.named_modules():
        if (
            getattr(module, "_tpu_rope_table_required", False)
            or getattr(module, "_tpu_in_graph_tier1", False)
            or hasattr(module, "cos_coef")
            or getattr(module, "update_cache", False)
            or _is_out_of_scope_vllm_rope(module)
        ):
            continue
        is_vllm_rotary_base = isinstance(module, RotaryEmbeddingBase)
        patched = False
        for buf_name, buf in list(module._buffers.items()):
            if buf is None or "cos_sin_cache" not in buf_name:
                continue
            segments = _module_cos_sin_segments(
                module,
                buf,
                _verified_memo=verified_memo,
                allow_table_recovery=allow_table_recovery,
            )
            if (
                allow_table_recovery
                and segments is None
                and not is_vllm_rotary_base
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
                if is_vllm_rotary_base:
                    logger.warning_once(
                        "[rope_patch] RotaryEmbedding %s.%s buffer %r "
                        "(shape=%s, dtype=%s) could not be converted to in-graph "
                        "RoPE; keeping precomputed table.",
                        type(module).__module__,
                        type(module).__qualname__,
                        buf_name,
                        tuple(buf.shape),
                        buf.dtype,
                    )
                continue
            del module._buffers[buf_name]
            setattr(
                module,
                buf_name,
                rope_in_graph.InGraphCosSinCache(
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
        if getattr(module, "_tpu_rope_table_required", False) or isinstance(
            module, RotaryEmbeddingBase
        ):
            continue
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
                rope_in_graph.InGraphCosSinCache(
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
