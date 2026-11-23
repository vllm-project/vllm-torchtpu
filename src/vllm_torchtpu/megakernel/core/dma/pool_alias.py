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
"""Multi-shape VMEM pool aliasing over physical `(sublanes, lanes)`-tiled TPU memory.

Every view returned by `view()` aliases the underlying VMEM buffer without
copying and must be overwritten before reading.

Public `ref.reshape` lowers to `tpu.memref_reshape` (logical row-major order,
which cannot alias physical VMEM tiles across target shapes or ranks), while
`ref.bitcast` changes only the element dtype. Instead of monkey-patching Mosaic
internal helpers, this module defines a standalone JAX ref primitive
(`reinterpret_cast_p`) and registers its TPU Mosaic lowering rule via
`lowering.register_lowering_rule`, emitting `tpu.reinterpret_cast` with an
explicit `#tpu.tiled` layout followed by `tpu.erase_memref_layout`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jax import tree_util
from jax._src import core as jax_core
from jax._src import state
from jax._src.lib import tpu
from jax._src.pallas.mosaic import lowering
from jax.experimental.pallas import tpu as pltpu
from jax.extend import core as jex_core
from jax.extend.mlir import ir

reinterpret_cast_p = jex_core.Primitive("vllm_tpu_reinterpret_cast")
reinterpret_cast_p.ref_primitive = True
# Required by JAX's jaxpr effect checker (`make_jaxpr_effects` / `check_jaxpr`):
# a primitive returning a non-invar `AbstractRef` must set `ref_allocating` so
# subsequent effects on the returned alias pass validation.
reinterpret_cast_p.ref_allocating = True


@reinterpret_cast_p.def_impl
def _reinterpret_cast_impl(*args: Any, **params: Any) -> Any:
    del args, params
    raise NotImplementedError(
        "pool_alias.view requires TPU Mosaic lowering (interpret=False)"
    )


@reinterpret_cast_p.def_effectful_abstract_eval
def _reinterpret_cast_abstract_eval(
    ref_aval: Any,
    *flat_transforms: Any,
    shape: tuple[int, ...],
    dtype: jnp.dtype[Any],
    transforms_tree: Any,
) -> tuple[Any, set[Any]]:
    del flat_transforms, transforms_tree
    out_aval = state.AbstractRef(
        jax_core.ShapedArray(shape, dtype),
        memory_space=ref_aval.memory_space,
    )
    return out_aval, {state.ReadEffect(0), state.WriteEffect(0)}


def _dtype_bits(dt: jnp.dtype[Any]) -> int:
    """Return bitwidth for supported 8-, 16-, or 32-bit numeric dtypes."""
    if jnp.issubdtype(dt, jnp.integer):
        bits = jnp.iinfo(dt).bits
    elif jnp.issubdtype(dt, jnp.floating):
        bits = jnp.finfo(dt).bits
    else:
        raise ValueError(
            f"pool_alias supports 8-, 16-, or 32-bit numeric dtypes, got {dt.name}"
        )
    if bits not in (8, 16, 32):
        raise ValueError(
            f"pool_alias supports 8-, 16-, or 32-bit numeric dtypes, "
            f"got {dt.name} ({bits}-bit)"
        )
    return bits


def _tiled_layout_attr(shape: tuple[int, ...], dtype: jnp.dtype[Any]) -> ir.Attribute:
    num_sublanes, num_lanes = pltpu.Tiling.COMPACT.shape
    bits = _dtype_bits(dtype)
    packing = 32 // bits
    rows = num_sublanes * packing
    columns = (shape[-1] + num_lanes - 1) // num_lanes
    strides = [columns, 1]
    # Only the two minor dims are tiled: leading dims advance by their full size.
    extent = (shape[-2] + rows - 1) // rows
    for dimension in reversed(shape[:-2]):
        strides.insert(0, strides[0] * extent)
        extent = dimension
    inner = f"({packing},1)" if packing > 1 else ""
    return ir.Attribute.parse(
        f"#tpu.tiled<({rows},{num_lanes}){inner},[{','.join(map(str, strides))}]>"
    )


@lowering.register_lowering_rule(reinterpret_cast_p)
def _reinterpret_cast_lowering(
    ctx: lowering.LoweringRuleContext,
    base_ref: ir.Value,
    *flat_transforms: ir.Value,
    shape: tuple[int, ...],
    dtype: jnp.dtype[Any],
    transforms_tree: Any,
) -> ir.Value:
    transforms = tree_util.tree_unflatten(transforms_tree, flat_transforms)
    sliced_ref, _ = lowering._transform_ref(
        base_ref, ctx.avals_in[0], ctx.block_shapes[0], transforms
    )
    erased_type = ir.MemRefType(ctx.aval_to_ir_type(ctx.avals_out[0]))
    tiled_type = ir.MemRefType.get(
        shape,
        erased_type.element_type,
        layout=_tiled_layout_attr(shape, dtype),
        memory_space=erased_type.memory_space,
    )
    alias = tpu.reinterpret_cast(tiled_type, sliced_ref)
    return tpu.erase_memref_layout(erased_type, alias)


def storage_size(shape: Sequence[int], dtype: Any) -> int:
    """Bytes occupied by native `(sublanes, lanes)`-tiled VMEM layout."""
    dims = tuple(int(d) for d in shape)
    if len(dims) < 2 or any(d < 0 for d in dims):
        raise ValueError(
            f"pool_alias.storage_size requires shape with >= 2 non-negative dims, "
            f"got {dims}"
        )
    dt = jnp.dtype(dtype)
    bits = _dtype_bits(dt)
    num_sublanes, num_lanes = pltpu.Tiling.COMPACT.shape
    rows = num_sublanes * (32 // bits)
    row_tiles = (dims[-2] + rows - 1) // rows
    lane_tiles = (dims[-1] + num_lanes - 1) // num_lanes
    return (
        math.prod(dims[:-2]) * row_tiles * lane_tiles * (num_sublanes * num_lanes * 4)
    )


def view(ref: Any, shape: Sequence[int], dtype: Any) -> Any:
    """Return a `shape`/`dtype` view aliasing `ref`'s physical tiled storage.

    Args:
        ref: Source VMEM reference (or tile-aligned slice) with at least 2
            positive dimensions holding the exact bytes required by `shape`
            and `dtype`.
        shape: Target view shape with at least 2 positive dimensions.
        dtype: Target 8-, 16-, or 32-bit numeric dtype.

    Returns:
        An `AbstractRef` tracer aliasing `ref`'s physical VMEM tiles. The
        returned view must be overwritten before it is read.

    Raises:
        ValueError: If `ref`, `shape`, or `dtype` is invalid, requires
            edge-tile padding, or does not match `ref`'s byte size.
    """
    target_shape = tuple(int(d) for d in shape)
    if len(target_shape) < 2 or any(d <= 0 for d in target_shape):
        raise ValueError(
            f"pool_alias.view requires shape with >= 2 positive dims, "
            f"got {target_shape}"
        )
    dt = jnp.dtype(dtype)
    bits = _dtype_bits(dt)
    num_sublanes, num_lanes = pltpu.Tiling.COMPACT.shape
    tile_rows = num_sublanes * (32 // bits)
    nbytes = math.prod(target_shape) * (bits // 8)
    tiled_bytes = storage_size(target_shape, dt)
    if tiled_bytes != nbytes:
        raise ValueError(
            f"pool_alias.view{target_shape} {dt.name} pads to "
            f"{tiled_bytes} bytes but aliases only {nbytes}; pad "
            f"shape[-2] to a multiple of {tile_rows} and shape[-1] to a "
            f"multiple of {num_lanes}"
        )
    ref_shape = tuple(int(d) for d in ref.shape)
    if len(ref_shape) < 2 or any(d <= 0 for d in ref_shape):
        raise ValueError(
            f"pool_alias.view requires source ref with >= 2 positive dims, "
            f"got {ref_shape}"
        )
    ref_dtype = jnp.dtype(ref.dtype)
    ref_bits = _dtype_bits(ref_dtype)
    ref_nbytes = math.prod(ref_shape) * (ref_bits // 8)
    ref_tiled_bytes = storage_size(ref_shape, ref_dtype)
    if ref_nbytes != nbytes or ref_tiled_bytes != nbytes:
        raise ValueError(
            f"pool_alias.view{target_shape} {dt.name} requires {nbytes} "
            f"contiguous tiled bytes, got source ref {ref_shape} "
            f"{ref_dtype.name} ({ref_nbytes} logical bytes, "
            f"{ref_tiled_bytes} tiled bytes)"
        )
    if isinstance(ref, state.TransformedRef):
        base_ref, transforms = ref.ref, ref.transforms
    else:
        base_ref, transforms = ref, ()
    flat_transforms, transforms_tree = tree_util.tree_flatten(transforms)
    return reinterpret_cast_p.bind(
        base_ref,
        *flat_transforms,
        shape=target_shape,
        dtype=dt,
        transforms_tree=transforms_tree,
    )
