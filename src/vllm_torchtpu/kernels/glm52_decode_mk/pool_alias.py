# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Experimental raw VMEM views; every new view must be overwritten before reading.

Vendored from https://github.com/Inferact/tpu-megakernels (pool_alias.py @ aa0094e), code identical to the copies
in glm5_mk_dcp (md5 71235f4695eca722d6c3cc6ac94a8181) and explore/clean-sheet-qwen (CSQ-P1, j-563b2830 N5).
Pinned to jax 0.11.2: it patches the Mosaic lowering's `_reshape_memref`. Interpret mode does not use it.
"""

import dataclasses
import math

from jax import tree_util
from jax._src.lib import tpu
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic import lowering
from jax._src.state import types


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Reinterpret(types.ReshapeTransform):
    pass


_original = lowering._reshape_memref


def _lower(ref, transform, aval, block_shape):
    if not isinstance(transform, Reinterpret):
        return _original(ref, transform, aval, block_shape)
    transform._validate_shape(tuple(block_shape))
    original_type = ir.MemRefType(ref.type)
    bits = aval.dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    columns = transform.shape[-1] // 128
    strides = [columns, 1]
    minor_dimension = transform.shape[-2]
    for dimension in reversed(transform.shape[:-2]):
        minor_tiles = (minor_dimension + rows - 1) // rows
        strides.insert(0, strides[0] * minor_tiles)
        minor_dimension = dimension
    inner = f"({packing},1)" if packing > 1 else ""
    layout = ir.Attribute.parse(
        f"#tpu.tiled<({rows},128){inner},[{','.join(map(str, strides))}]>"
    )
    result_type = ir.MemRefType.get(
        transform.shape,
        original_type.element_type,
        layout=layout,
        memory_space=original_type.memory_space,
    )
    alias = tpu.reinterpret_cast(result_type, ref)
    erased = ir.MemRefType.get(
        transform.shape,
        original_type.element_type,
        memory_space=original_type.memory_space,
    )
    return tpu.erase_memref_layout(erased, alias), transform.shape


lowering._reshape_memref = _lower


def storage_size(shape, dtype):
    bits = dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096


def view(ref, shape, dtype):
    ref = ref.bitcast(dtype)
    return types.TransformedRef(ref.ref, (*ref.transforms, Reinterpret(tuple(shape))))
