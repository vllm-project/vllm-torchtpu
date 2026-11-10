# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pack the per-step PCP all-RPA metadata pytree into one int32 tensor.

TPU_PCP_RPA_HOIST_META computes the metadata once per forward pass in its own
custom op and hands it to every PCP attention layer. A torch custom op takes a
fixed list of tensors, so the pytree (leaves of several shapes and dtypes, each
with a leading PCP-rank axis) travels as one ``[pcp, total]`` int32 tensor.
Both ops derive the same layout from ``jax.eval_shape`` of the metadata
function on identical input shapes, so no layout crosses the op boundary.
"""

import math
import os

import jax
import jax.numpy as jnp

_ALIGN = 1024  # elements: one (8, 128) int32 tile


def body_unpack_enabled() -> bool:
    """TPU_PCP_META_BODY_UNPACK=1 (default off): the attention layers unpack the packed
    metadata per rank inside the PCP shard_map (``unpack_local``) instead of before it.
    Unpacked outside, every leaf crosses the shard_map as a per-rank [1, n] array with
    one-row (1, 128) tiles (e.g. the bool leaves' compare at 1/8 sublane use). Each leaf
    then starts at a multiple of 1024 elements of the packed row (zero padding between
    leaves), so the per-rank row is read as whole (8, 128) tiles. Pack and unpack derive
    the layout from this module, so it stays consistent across the op boundary."""
    return os.environ.get("TPU_PCP_META_BODY_UNPACK", "0") == "1"


def _span(n: int) -> int:
    """Packed-row elements a leaf of ``n`` elements per rank occupies."""
    return -(-n // _ALIGN) * _ALIGN if body_unpack_enabled() else n


class PackedMeta:
    """The packed [P, W] row plus its layout, handed to the PCP wrapper unopened
    (TPU_PCP_META_BODY_UNPACK); the wrapper unpacks it per rank in its shard_map body."""

    def __init__(self, packed, layout):
        self.packed = packed
        self.layout = layout


def _to_i32(x):
    if x.dtype == jnp.int32:
        return x
    if x.dtype == jnp.bool_:
        return x.astype(jnp.int32)
    if jnp.dtype(x.dtype).itemsize == 4:
        return jax.lax.bitcast_convert_type(x, jnp.int32)
    if jnp.issubdtype(x.dtype, jnp.integer) and jnp.dtype(x.dtype).itemsize < 4:
        return x.astype(jnp.int32)
    raise TypeError(f"cannot pack PCP metadata leaf of dtype {x.dtype}")


def _from_i32(x, dtype):
    dtype = jnp.dtype(dtype)
    if dtype == jnp.int32:
        return x
    if dtype == jnp.bool_:
        return x != 0
    if dtype.itemsize == 4:
        return jax.lax.bitcast_convert_type(x, dtype)
    return x.astype(dtype)


def pack(tree):
    """Pytree of [P, ...] arrays -> [P, total] int32."""
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError("empty PCP metadata")
    p = leaves[0].shape[0]
    if not body_unpack_enabled():
        return jnp.concatenate([_to_i32(x).reshape(p, -1) for x in leaves], axis=1)
    parts = []
    for x in leaves:
        v = _to_i32(x).reshape(p, -1)
        pad = _span(v.shape[1]) - v.shape[1]
        parts.append(jnp.pad(v, ((0, 0), (0, pad))) if pad else v)
    return jnp.concatenate(parts, axis=1)


def packed_width(shape_tree) -> int:
    return sum(_span(math.prod(s.shape[1:])) for s in jax.tree.leaves(shape_tree))


def unpack(packed, shape_tree):
    """[P, total] int32 -> pytree shaped like ``shape_tree`` (ShapeDtypeStructs)."""
    leaves, treedef = jax.tree.flatten(shape_tree)
    width = packed_width(shape_tree)
    if packed.shape[1] != width:
        raise ValueError(
            f"packed PCP metadata width {packed.shape[1]} != layout width {width}"
        )
    out, off = [], 0
    for s in leaves:
        n = math.prod(s.shape[1:])
        out.append(_from_i32(packed[:, off : off + n].reshape(s.shape), s.dtype))
        off += n
    return jax.tree.unflatten(treedef, out)


def unpack_local(packed_local, shape_tree):
    """TPU_PCP_META_BODY_UNPACK, inside the shard_map: this rank's [1, W'] row (W' >= the
    layout's aligned width; the padded tail is never read) -> pytree of per-rank leaves
    shaped ``s.shape[1:]``. The [W' / 128, 128] view of the row is a bitcast (same byte
    order, full (8, 128) tiles) and every leaf is a run of whole rows of it."""
    leaves, treedef = jax.tree.flatten(shape_tree)
    width = packed_local.shape[-1]
    if width % 128 or width < packed_width(shape_tree):
        raise ValueError(f"packed PCP metadata width {width} cannot hold layout width {packed_width(shape_tree)}")
    # The barrier keeps XLA from folding each leaf's slice back onto the one-row-tile [1, W'] operand
    # (reshape(slice(reshape)) -> slice); the view itself is a bitcast.
    rows = jax.lax.optimization_barrier(packed_local.reshape(width // 128, 128))
    out, off = [], 0
    for s in leaves:
        n = math.prod(s.shape[1:])
        span = _span(n)
        blk = rows[off // 128 : (off + span) // 128, :].reshape(span)
        out.append(_from_i32(blk[:n].reshape(s.shape[1:]), s.dtype))
        off += span
    return jax.tree.unflatten(treedef, out)
