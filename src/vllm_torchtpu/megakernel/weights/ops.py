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
exposes an `ops: ArrayOps` implementation (`slice_axis`, `concat`, `reshape`,
`transpose`, `pad`, `cast`, `is_fp8`, `dequantize`, `quantize_fp8`, `zeros`,
`stack_layers`, `own`) and alias-resolution hooks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from ..decls import HFParamDecl, VLLMAlias
from ..exec_spec import ExecSpec
from ..ops import F32, FP8


class ArrayOps(Protocol):
    """Minimal array-operations interface used by the generic weight loader."""

    def slice_axis(self, x: Any, axis: int, start: int, stop: int) -> Any: ...
    def concat(self, xs: Sequence[Any], axis: int) -> Any: ...
    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any: ...
    def transpose(self, x: Any, perm: tuple[int, ...]) -> Any: ...
    def pad(self, x: Any, pad_widths: Sequence[tuple[int, int]]) -> Any: ...
    def cast(self, x: Any, dtype: Any) -> Any: ...
    def is_fp8(self, x: Any) -> bool: ...
    def dequantize(self, x: Any, scale: Any, dtype: Any) -> Any: ...
    def quantize_fp8(self, x: Any) -> tuple[Any, Any]: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any) -> Any: ...
    def stack_layers(self, xs: Sequence[Any]) -> Any: ...
    def own(self, x: Any) -> Any: ...


class WeightSource(Protocol):
    """Source-agnostic weight provider for a single rank (`tp_rank`)."""

    @property
    def ops(self) -> ArrayOps: ...
    def has_layer_alias(self, lyr: int, ord_idx: int, source_name: str) -> bool: ...
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
    ) -> dict[str, Any]: ...
    def read_model_param(self, decl: HFParamDecl, *, R: int, tp_rank: int) -> Any: ...
    def read_layer_ecopy(
        self,
        lyr: int,
        g: Any,
        ex: ExecSpec,
        *,
        expert_placement: Any = None,
        experts_path: str = "mlp.experts",
    ) -> Any | None: ...
    def supports_passthrough(self) -> bool: ...


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
        nkv = int(decl.shape[0])
        if nkv < R and R % nkv != 0:
            raise ValueError(
                f"HFParamDecl {decl.name!r} kv_repeat nkv={nkv} does not divide R={R}"
            )
        if nkv > R and nkv % R != 0:
            raise ValueError(
                f"HFParamDecl {decl.name!r} kv_repeat nkv={nkv} "
                f"is not divisible by R={R}"
            )
        full_kv = int(nkv * decl.shape[-1])
        shard_kv = int(max(1, nkv // R) * decl.shape[-1])
        return (full_kv, shard_kv) if split_axis != 0 else (int(decl.shape[1]),) * 2
    dim_ax = split_axis % len(decl.shape)
    full_w = int(decl.shape[dim_ax])
    tf = decl.transform
    if tf.channel_splits is not None and dim_ax == len(decl.shape) - 1:
        if sum(tf.channel_splits) != full_w or any(
            w <= 0 or w % R != 0 for w in tf.channel_splits
        ):
            raise ValueError(
                f"HFParamDecl {decl.name!r} channel_splits={tf.channel_splits!r} "
                f"must sum to {full_w} with each split divisible by R={R}"
            )
        return full_w, full_w // R
    if tf.shard_axis is not None and (tf.shard_axis % len(decl.shape)) == dim_ax:
        if full_w % R != 0:
            raise ValueError(
                f"HFParamDecl {decl.name!r} axis {dim_ax} width {full_w} "
                f"is not divisible by R={R}"
            )
        return full_w, full_w // R
    return full_w, full_w


def apply_decl_pad(ops: ArrayOps, x: Any, decl: HFParamDecl) -> Any:
    """Apply `decl.transform.pad` (`(before, target_width)`) via `ops.pad`."""
    if decl.transform.pad is None:
        return x
    shape = tuple(int(d) for d in x.shape)
    offset = len(shape) - len(decl.transform.pad)
    widths: list[tuple[int, int]] = []
    for idx, (before, target_w) in enumerate(decl.transform.pad):
        if before == 0 and target_w == 0:
            widths.append((0, 0))
            continue
        cur = shape[offset + idx]
        after = target_w - cur - before
        if before < 0 or after < 0:
            raise ValueError(
                f"HFParamDecl {decl.name!r} axis {offset + idx} width {cur} "
                f"with pad before={before} exceeds target_width={target_w}"
            )
        widths.append((before, after))
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
    any_want_fp8 = any(jnp.dtype(d.dtype) == jnp.dtype(FP8) for d in target_decls)
    if not alias.already_fp8 and is_w_fp8:
        raise ValueError(
            f"Alias {alias.source!r} declared already_fp8=False, "
            f"got is_fp8=True (dtype={w_buf.dtype})"
        )
    if alias.already_fp8 and not is_w_fp8 and any_want_fp8:
        raise ValueError(
            f"Alias {alias.source!r} declared already_fp8=True, "
            f"got is_fp8=False (dtype={w_buf.dtype})"
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
    norm_split_axis = split_axis % len(w_shape)
    is_trailing_split = norm_split_axis == len(w_shape) - 1
    actual_w = w_shape[norm_split_axis]
    if actual_w != expected_w:
        raise ValueError(
            f"Alias {alias.source!r} (already_sharded={alias.already_sharded}) "
            f"axis {split_axis} width {actual_w} != expected {expected_w} "
            f"(sharded={total_sharded}, unsharded={total_full})"
        )

    expected_s_w = expected_w if is_trailing_split else int(d0.shape[-1])
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
        and is_w_fp8
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

    scale_by_target = {
        scale_decls[sn].transform.scale_for: scale_decls[sn]
        for sn in alias.scale_targets
        if scale_decls[sn].transform.scale_for is not None
    }

    out = {}
    off = 0
    for d, (full_w, sharded_w) in zip(target_decls, widths, strict=True):
        tf = d.transform
        has_unsharded_cs = (
            not alias.already_sharded
            and tf.channel_splits is not None
            and is_trailing_split
            and full_w != sharded_w
        )
        if has_unsharded_cs:
            assert tf.channel_splits is not None
            w_parts = []
            s_parts = []
            cs_off = off
            for cw in tf.channel_splits:
                cw_r = cw // R
                cs_start = cs_off + tp_rank * cw_r
                cs_stop = cs_start + cw_r
                w_parts.append(ops.slice_axis(w_buf, split_axis, cs_start, cs_stop))
                if s_buf is not None:
                    s_parts.append(ops.slice_axis(s_buf, -1, cs_start, cs_stop))
                cs_off += cw
            off += full_w
            w_t = ops.concat(w_parts, axis=norm_split_axis)
            s_t = ops.concat(s_parts, axis=-1) if s_buf is not None else None
        else:
            if not alias.already_sharded:
                if tf.kv_repeat and len(d.shape) == 3 and full_w != sharded_w:
                    nkv = int(d.shape[0])
                    kv_head = tp_rank // (R // nkv) if nkv < R else tp_rank
                    start = off + kv_head * sharded_w
                    stop = start + sharded_w
                elif tf.shard_axis is not None and full_w != sharded_w:
                    start = off + tp_rank * sharded_w
                    stop = start + sharded_w
                else:
                    start, stop = off, off + full_w
                off += full_w
            else:
                start, stop = off, off + sharded_w
                off += sharded_w

            w_t = (
                w_buf
                if len(target_decls) == 1 and start == 0 and stop == actual_w
                else ops.slice_axis(w_buf, split_axis, start, stop)
            )
            if s_buf is None:
                s_t = None
            elif not is_trailing_split or (
                len(target_decls) == 1 and start == 0 and stop == expected_s_w
            ):
                s_t = s_buf
            else:
                s_t = ops.slice_axis(s_buf, -1, start, stop)

        want_fp8 = jnp.dtype(d.dtype) == jnp.dtype(FP8)
        if is_w_fp8 and not want_fp8:
            if s_t is None:
                raise ValueError(
                    f"Target {d.name!r} has dtype {d.dtype} in FP8 alias "
                    f"{alias.source!r} but no scale slice is available"
                )
            s_f32 = ops.cast(s_t, F32)
            s_bcast = ops.reshape(s_f32, (*s_f32.shape[:-1], 1, int(s_f32.shape[-1])))
            w_t = ops.dequantize(w_t, s_bcast, d.dtype)
        elif not is_w_fp8 and want_fp8:
            w_t, s_t = ops.quantize_fp8(w_t)
        elif not is_w_fp8:
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
