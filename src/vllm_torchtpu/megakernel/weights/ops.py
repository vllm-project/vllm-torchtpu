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
"""Minimal `ArrayOps` and `WeightSource` protocols for generic weight loading.

The core weight loader (`megakernel.weights`) never imports `torch` or `vllm`
and never branches on the concrete source type. Instead, every `WeightSource`
exposes an `ops: ArrayOps` implementation with the 12-method protocol:

- `slice_axis(x, axis, start, stop)` — slice `x` along `axis` in `[start, stop)`.
- `concat(xs, axis)` — concatenate `xs` along `axis` into a contiguous buffer.
- `reshape(x, shape)` — reshape `x` to `shape` (contiguous view where supported).
- `transpose(x, perm)` — permute dimensions of `x` according to `perm`.
- `pad(x, pad_widths)` — zero-pad trailing dimensions with `(before, after)` pairs.
- `cast(x, dtype)` — convert `x` to `dtype` (`FP8`, `BF16`, `F32`, `I32`).
- `is_fp8(x)` — return whether `x` has an FP8 (`float8_e4m3fn`) dtype.
- `dequantize(x, scale, dtype)` — dequantize `x` by `scale` in `F32` and cast
  to `dtype`.
- `quantize_fp8(x)` — per-output-channel symmetric FP8 (`e4m3fn`) quantization of
  a `[K, N]` weight tensor, returning `(w_fp8 [K, N], scale [N] f32)`.
- `zeros(shape, dtype)` — allocate a zero-filled tensor of `shape` and `dtype`.
- `stack_layers(xs)` — stack per-layer tensors along leading axis `0` into a
  newly allocated contiguous base buffer (not a strided view).
- `own(x)` — materialize `x` as a contiguous base buffer (`storage_offset == 0`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from ..comm.replication import (
    shard_count,
    slice_rank_ranges,
    split_sharded_width,
    unsharded_target_ranges,
)
from ..decls import HFParamDecl, VLLMAlias
from ..exec_spec import ExecSpec
from ..ops import F32, FP8


class ArrayOps(Protocol):
    """Minimal array-operations interface used by the generic weight loader."""

    def slice_axis(self, x: Any, axis: int, start: int, stop: int) -> Any:
        """Slice `x` along `axis` over `[start, stop)`."""
        ...

    def concat(self, xs: Sequence[Any], axis: int) -> Any:
        """Concatenate `xs` along `axis`."""
        ...

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        """Reshape `x` to `shape`."""
        ...

    def transpose(self, x: Any, perm: tuple[int, ...]) -> Any:
        """Permute axes of `x` by `perm`."""
        ...

    def pad(self, x: Any, pad_widths: Sequence[tuple[int, int]]) -> Any:
        """Zero-pad trailing axes of `x` with `(before, after)` widths."""
        ...

    def cast(self, x: Any, dtype: Any) -> Any:
        """Cast `x` to canonical `dtype` (`FP8`, `BF16`, `F32`, `I32`)."""
        ...

    def is_fp8(self, x: Any) -> bool:
        """Return whether `x` has an FP8 (`float8_e4m3fn`) dtype."""
        ...

    def dequantize(self, x: Any, scale: Any, dtype: Any) -> Any:
        """Dequantize `x` by `scale` in `F32` and cast to `dtype`."""
        ...

    def quantize_fp8(self, x: Any) -> tuple[Any, Any]:
        """Quantize `[K, N]` `x` per column to `(w_fp8 [K, N], scale [N] f32)`."""
        ...

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any:
        """Return a zero-filled tensor of `shape` and `dtype`."""
        ...

    def stack_layers(self, xs: Sequence[Any]) -> Any:
        """Stack per-layer arrays along axis `0` into an owned base buffer."""
        ...

    def own(self, x: Any) -> Any:
        """Return `x` as a contiguous base buffer (`storage_offset == 0`)."""
        ...


class WeightSource(Protocol):
    """Source-agnostic weight provider for a single rank (`tp_rank`)."""

    @property
    def ops(self) -> ArrayOps:
        """Return the `ArrayOps` implementation for tensors from this source."""
        ...

    def has_layer_alias(self, lyr: int, ord_idx: int, source_name: str) -> bool:
        """Return whether `source_name` exists on layer `lyr` (`ord_idx`)."""
        ...

    def resolve_alias(
        self,
        lyr: int,
        ord_idx: int,
        alias: VLLMAlias,
        target_decls: Sequence[HFParamDecl],
        scale_decls: Mapping[str, HFParamDecl],
        *,
        R: int,
        tp_rank: int,
        fuse_groups: Mapping[str, tuple[str, ...]] | None = None,
    ) -> dict[str, Any]:
        """Extract and slice `alias` on layer `lyr` into `target_decls`."""
        ...

    def read_model_param(self, decl: HFParamDecl, *, R: int, tp_rank: int) -> Any:
        """Extract a model-level parameter (`final_norm`, `embed_tokens`, `lm_head`)."""
        ...

    def read_layer_ecopy(
        self,
        lyr: int,
        g: Any,
        ex: ExecSpec,
        *,
        expert_placement: Any = None,
        experts_path: str = "mlp.experts",
    ) -> Any | None:
        """Return `[E]` `f32` `ecopy` vector for layer `lyr`, or `None` if inactive."""
        ...

    def supports_passthrough(self) -> bool:
        """Return whether dynamic per-layer 3D base buffers can be passed through."""
        ...


class JaxArrayOps:
    """`ArrayOps` implementation backed by `jax.numpy`."""

    def slice_axis(self, x: jax.Array, axis: int, start: int, stop: int) -> jax.Array:
        ax = axis % x.ndim
        sl = [slice(None)] * x.ndim
        sl[ax] = slice(start, stop)
        return x[tuple(sl)]

    def concat(self, xs: Sequence[jax.Array], axis: int) -> jax.Array:
        return jnp.concatenate(list(xs), axis=axis)

    def reshape(self, x: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        return jnp.reshape(x, shape)

    def transpose(self, x: jax.Array, perm: tuple[int, ...]) -> jax.Array:
        return jnp.transpose(x, perm)

    def pad(self, x: jax.Array, pad_widths: Sequence[tuple[int, int]]) -> jax.Array:
        full_pad = [(0, 0)] * (x.ndim - len(pad_widths)) + list(pad_widths)
        if any(b > 0 or a > 0 for b, a in full_pad):
            return jnp.pad(x, tuple(full_pad))
        return x

    def cast(self, x: jax.Array, dtype: Any) -> jax.Array:
        target = jnp.dtype(dtype)
        return x if x.dtype == target else x.astype(target)

    def is_fp8(self, x: jax.Array) -> bool:
        return jnp.dtype(x.dtype) == jnp.dtype(FP8)

    def dequantize(self, x: jax.Array, scale: jax.Array, dtype: Any) -> jax.Array:
        return (x.astype(F32) * scale.astype(F32)).astype(jnp.dtype(dtype))

    def quantize_fp8(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        w_nk_f32 = jnp.transpose(x, (1, 0)).astype(F32)
        amax = jnp.max(jnp.abs(w_nk_f32), axis=-1, keepdims=True)
        scale = jnp.where(amax == 0.0, jnp.ones_like(amax), amax * (1.0 / 448.0))
        w_fp8_kn = jnp.transpose(
            jnp.clip(w_nk_f32 / scale, -448.0, 448.0).astype(FP8), (1, 0)
        )
        return w_fp8_kn, jnp.squeeze(scale, axis=-1).astype(F32)

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> jax.Array:
        return jnp.zeros(shape, dtype=jnp.dtype(dtype))

    def stack_layers(self, xs: Sequence[jax.Array]) -> jax.Array:
        return jnp.stack(list(xs), axis=0)

    def own(self, x: jax.Array) -> jax.Array:
        return jnp.asarray(x)


def target_axis_widths(decl: HFParamDecl, R: int, split_axis: int) -> tuple[int, int]:
    """Return `(full_width, sharded_width)` of `decl` along `split_axis`."""
    if len(decl.shape) == 3 and decl.transform.kv_repeat:
        full_kv = int(decl.shape[0] * decl.shape[-1])
        shard_kv = int(max(1, decl.shape[0] // R) * decl.shape[-1])
        return (full_kv, shard_kv) if split_axis != 0 else (int(decl.shape[1]),) * 2
    dim_ax = split_axis % len(decl.shape)
    full_w = int(decl.shape[dim_ax])
    tf = decl.transform
    if tf.channel_splits is not None and dim_ax == len(decl.shape) - 1:
        return full_w, split_sharded_width(R, tf.channel_splits, tf.split_groups)
    if tf.shard_axis is not None and (tf.shard_axis % len(decl.shape)) == dim_ax:
        return full_w, full_w // shard_count(R, tf.shard_groups)
    return full_w, full_w


def apply_decl_pad(ops: ArrayOps, x: Any, decl: HFParamDecl) -> Any:
    """Apply `decl.transform.pad` (`(before, target_width)`) via `ops.pad`."""
    pad_spec = decl.transform.pad
    if pad_spec is None:
        return x
    shape = tuple(int(d) for d in x.shape)
    offset = len(shape) - len(pad_spec)
    widths: list[tuple[int, int]] = []
    for idx, (before, target_w) in enumerate(pad_spec):
        cur = shape[offset + idx]
        widths.append((before, max(0, target_w - cur - before)))
    return ops.pad(x, widths)


def slice_alias_targets(
    ops: ArrayOps,
    alias: VLLMAlias,
    w_buf: Any,
    s_buf: Any | None,
    target_decls: Sequence[HFParamDecl],
    scale_decls: Mapping[str, HFParamDecl],
    *,
    R: int,
    tp_rank: int,
    fuse_groups: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Slice `(w_buf, s_buf)` across `target_decls` driven by `VLLMAlias` metadata."""
    is_w_fp8 = ops.is_fp8(w_buf)
    if is_w_fp8 != alias.already_fp8:
        raise ValueError(
            f"Alias {alias.source!r} declared already_fp8={alias.already_fp8}, "
            f"got is_fp8={is_w_fp8} (dtype={w_buf.dtype})"
        )
    d0 = target_decls[0]
    if len(target_decls) == 1 and d0.transform.flatten_tail and len(d0.shape) == 2:
        k_dim = int(d0.shape[0])
        w_buf = ops.cast(w_buf, d0.dtype)
        t_sh = tuple(int(dim) for dim in w_buf.shape)
        if len(t_sh) != 2 or t_sh[0] != k_dim:
            if math.prod(t_sh) % k_dim != 0:
                raise ValueError(
                    f"Alias {alias.source!r} shape {t_sh} not divisible by {k_dim}"
                )
            n_ch = int(math.prod(t_sh) // k_dim)
            w_buf = ops.transpose(ops.reshape(w_buf, (n_ch, k_dim)), (1, 0))

    split_axis = alias.shard_axis if alias.shard_axis is not None else -1
    widths = [target_axis_widths(d, R, split_axis) for d in target_decls]
    total_full = sum(fw for fw, _ in widths)
    total_sharded = sum(sw for _, sw in widths)
    expected_w = total_sharded if alias.already_sharded else total_full
    w_shape = tuple(int(d) for d in w_buf.shape)
    actual_w = w_shape[split_axis % len(w_shape)]
    if actual_w != expected_w:
        raise ValueError(
            f"Alias {alias.source!r} (already_sharded={alias.already_sharded}) "
            f"axis {split_axis} width {actual_w} != expected {expected_w} "
            f"(sharded={total_sharded}, unsharded={total_full})"
        )

    expected_s_w = expected_w if split_axis == -1 else int(d0.shape[-1])
    if s_buf is not None:
        s_len = int(s_buf.shape[-1])
        if s_len != expected_s_w:
            raise ValueError(
                f"Scale for alias {alias.source!r} has trailing width {s_len}, "
                f"expected {expected_s_w}"
            )

    f_grp = d0.transform.fuse_into
    s_grp: str | None = None
    s_first: HFParamDecl | None = None
    if s_buf is not None and alias.scale_targets:
        s_first = scale_decls[alias.scale_targets[0]]
        s_grp = s_first.transform.fuse_into
        s_fast_ok = (
            s_grp is not None
            and fuse_groups is not None
            and fuse_groups.get(s_grp) == tuple(alias.scale_targets)
            and all(scale_decls[sn].transform.pad is None for sn in alias.scale_targets)
        )
    else:
        s_fast_ok = not alias.scale_targets

    if (
        alias.already_sharded
        and alias.already_fp8
        and f_grp is not None
        and fuse_groups is not None
        and tuple(d.name for d in target_decls) == fuse_groups.get(f_grp)
        and all(
            d.transform.pad is None and jnp.dtype(d.dtype) == jnp.dtype(FP8)
            for d in target_decls
        )
        and s_fast_ok
    ):
        out: dict[str, Any] = {f_grp: w_buf}
        if s_buf is not None and s_grp is not None and s_first is not None:
            out[s_grp] = ops.cast(s_buf, s_first.dtype)
        return out

    scale_by_target: dict[str, HFParamDecl] = {}
    for sn in alias.scale_targets:
        sd = scale_decls[sn]
        if sd.transform.scale_for is not None:
            scale_by_target[sd.transform.scale_for] = sd

    out = {}
    off = 0
    single = len(target_decls) == 1
    for d, (full_w, sharded_w) in zip(target_decls, widths, strict=True):
        if alias.already_sharded:
            rngs = [(off, off + sharded_w)]
            off += sharded_w
        else:
            rngs = unsharded_target_ranges(
                d, off, full_w, sharded_w, split_axis, R, tp_rank
            )
            off += full_w

        w_t = slice_rank_ranges(
            ops, w_buf, split_axis, rngs, actual_w if single else -1
        )
        s_t = (
            s_buf
            if s_buf is None or split_axis != -1
            else slice_rank_ranges(ops, s_buf, -1, rngs, expected_s_w if single else -1)
        )

        want_fp8 = jnp.dtype(d.dtype) == jnp.dtype(FP8)
        if alias.already_fp8 and not want_fp8:
            if s_t is None:
                raise ValueError(
                    f"Target {d.name!r} has dtype {d.dtype} in FP8 alias "
                    f"{alias.source!r} but no scale slice is available"
                )
            s_2d = ops.reshape(ops.cast(s_t, F32), (1, sharded_w))
            w_t = ops.dequantize(w_t, s_2d, d.dtype)
        elif not alias.already_fp8 and want_fp8:
            w_t, s_t = ops.quantize_fp8(w_t)
        elif not alias.already_fp8:
            w_t = ops.cast(w_t, d.dtype)

        if d.transform.flatten_tail and len(d.shape) == 2:
            w_t = ops.reshape(ops.own(w_t), (-1,))

        w_t = apply_decl_pad(ops, w_t, d)
        out[d.name] = w_t

        sd = scale_by_target.get(d.name)
        if sd is not None:
            if s_t is None:
                raise ValueError(
                    f"Target {d.name!r} requires scale {sd.name!r} in alias "
                    f"{alias.source!r}, but no scale was provided"
                )
            out[sd.name] = apply_decl_pad(ops, ops.cast(s_t, sd.dtype), sd)

    return out


__all__ = [
    "ArrayOps",
    "JaxArrayOps",
    "WeightSource",
    "apply_decl_pad",
    "slice_alias_targets",
    "target_axis_widths",
]
