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
"""Internal helper to shard, pad, and fuse canonical MK-shaped block parameters.

Takes tensors already keyed by canonical `HFParamDecl.name` in canonical
`HFParamDecl.shape` (with optional leading layer dimension) and applies the
declaration's rank sharding, KV-head repetition, head reduction, padding, and
`fuse_into` concatenation across `R` ranks. Raw HF checkpoint key resolution,
`split_from` slicing, and `hf_shape -> shape` permutation are handled by the
checkpoint loader rather than this helper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp

from ..decls import HFParamDecl


def _validate_decl_and_array(
    decl: HFParamDecl, arr: jax.Array, known_names: set[str]
) -> None:
    tf = decl.transform
    if tf.quantize not in (None, "fp8", "per_channel_fp8"):
        raise ValueError(
            f"apply_hf_transforms does not support quantize={tf.quantize!r} "
            f"on {decl.name!r}"
        )
    if tf.quantize is not None and jnp.dtype(arr.dtype) != jnp.dtype(decl.dtype):
        raise ValueError(
            f"apply_hf_transforms does not quantize on the fly: {decl.name!r} "
            f"has quantize={tf.quantize!r} and requires dtype {decl.dtype}, "
            f"got {arr.dtype}"
        )
    if tf.scale_for is not None and tf.scale_for not in known_names:
        raise ValueError(
            f"apply_hf_transforms got unknown scale_for={tf.scale_for!r} "
            f"on {decl.name!r}"
        )
    if tf.head_reduce not in (None, "mean"):
        raise ValueError(
            f"apply_hf_transforms does not support head_reduce={tf.head_reduce!r} "
            f"on {decl.name!r}"
        )
    if (tf.split_from is None) != (tf.split_slice is None):
        raise ValueError(
            f"apply_hf_transforms requires both split_from and split_slice "
            f"or neither on {decl.name!r}"
        )
    if tf.expert_axis is not None and not (0 <= tf.expert_axis < len(decl.shape)):
        raise ValueError(
            f"apply_hf_transforms got out-of-range expert_axis={tf.expert_axis!r} "
            f"on {decl.name!r} with shape {decl.shape}"
        )
    rank = len(decl.shape)
    if arr.ndim not in (rank, rank + 1) or arr.shape[-rank:] != decl.shape:
        raise ValueError(
            f"apply_hf_transforms expected {decl.name!r} to have trailing "
            f"canonical shape {decl.shape}, got {arr.shape}"
        )


def _shard_and_transform_single(
    arr: jax.Array,
    decl: HFParamDecl,
    R: int,
    known_names: set[str],
) -> jax.Array:
    """Apply one `HFParamDecl`'s sharding, repetition, reduction, and padding."""
    _validate_decl_and_array(decl, arr, known_names)
    tf = decl.transform
    has_layer_dim = arr.ndim == len(decl.shape) + 1

    if tf.channel_splits is not None:
        parts = []
        off = 0
        for width in tf.channel_splits:
            sl = arr[..., off : off + width]
            off += width
            w_r = width // R
            lead = sl.shape[:-1]
            reshaped = sl.reshape((*lead, R, w_r))
            perm = (len(lead), *range(len(lead)), len(lead) + 1)
            parts.append(jnp.transpose(reshaped, perm))
        cat = jnp.concatenate(parts, axis=-1)
        if tf.flatten_tail:
            return cat.reshape((*cat.shape[:-2], cat.shape[-2] * cat.shape[-1]))
        return cat

    if tf.shard_axis is None:
        if tf.replicate_ranks:
            return jnp.broadcast_to(arr[None, ...], (R, *arr.shape))
        return arr

    per_layer_axis = tf.shard_axis % len(decl.shape)
    full_axis = per_layer_axis + (1 if has_layer_dim else 0)

    if tf.kv_repeat:
        nkv = arr.shape[full_axis]
        G = R // nkv
        perm = (
            full_axis,
            *(i for i in range(arr.ndim) if i != full_axis),
        )
        moved = jnp.transpose(arr, perm)
        out = jnp.repeat(moved, G, axis=0)
    else:
        dim_size = arr.shape[full_axis]
        local_size = dim_size // R
        new_shape = (
            *arr.shape[:full_axis],
            R,
            local_size,
            *arr.shape[full_axis + 1 :],
        )
        reshaped = arr.reshape(new_shape)
        perm = (
            full_axis,
            *(i for i in range(reshaped.ndim) if i != full_axis),
        )
        out = jnp.transpose(reshaped, perm)
        if tf.head_reduce == "mean":
            out = out.mean(axis=full_axis + 1)

    if tf.pad is not None:
        pad_spec = [(0, 0)] * out.ndim
        offset = out.ndim - len(tf.pad)
        for idx, (before, target_or_after) in enumerate(tf.pad):
            cur = out.shape[offset + idx]
            after = max(0, target_or_after - cur - before)
            pad_spec[offset + idx] = (before, after)
        if any(b > 0 or a > 0 for b, a in pad_spec):
            out = jnp.pad(out, tuple(pad_spec))
    return out


def apply_hf_transforms(
    decls: Sequence[HFParamDecl],
    hf: Mapping[str, jax.Array],
    *,
    R: int,
) -> dict[str, jax.Array]:
    """Shard, pad, and fuse canonical MK-shaped `hf` tensors across `R` ranks."""
    unique_decls: dict[str, HFParamDecl] = {}
    for d in decls:
        unique_decls.setdefault(d.name, d)

    known_names = set(unique_decls)
    sharded: dict[str, jax.Array] = {}
    fuse_groups: dict[str, list[HFParamDecl]] = {}
    for d in unique_decls.values():
        if d.name not in hf:
            raise ValueError(
                f"apply_hf_transforms missing canonical parameter key {d.name!r}"
            )
        sharded[d.name] = _shard_and_transform_single(hf[d.name], d, R, known_names)
        if d.transform.fuse_into is not None:
            fuse_groups.setdefault(d.transform.fuse_into, []).append(d)

    out: dict[str, jax.Array] = {}
    for d in unique_decls.values():
        if d.transform.fuse_into is None:
            out[d.name] = sharded[d.name]
    for target, members in fuse_groups.items():
        members_sorted = sorted(members, key=lambda m: m.transform.fuse_order)
        axis = members_sorted[0].transform.fuse_axis
        out[target] = jnp.concatenate(
            [sharded[m.name] for m in members_sorted], axis=axis
        )
    return out
