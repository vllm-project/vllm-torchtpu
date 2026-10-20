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
"""Multi-shape VMEM pool aliasing over physical `(8, 128)`-tiled TPU memory.

Every new view returned by `view()` aliases the underlying flat VMEM buffer and
must be overwritten before reading.

Why a private Mosaic lowering hook (`lowering._reshape_memref`) is needed:
public `ref.reshape` lowers to `tpu.memref_reshape`, which reshapes in logical
row-major order and cannot alias the physical `(8, 128)`-tiled VMEM layout,
while `ref.bitcast` only changes the element dtype without changing the shape.
`Reinterpret` lowers instead to `tpu.reinterpret_cast` with an explicit
`#tpu.tiled` layout attribute followed by `tpu.erase_memref_layout`. The hook is
installed lazily and idempotently on first call to `view()`.
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
    """Ref transform marking a raw `(8, 128)`-tiled VMEM shape reinterpretation."""


_original = lowering._reshape_memref
_installed = False


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


def _install() -> None:
    global _installed, _original
    if not _installed:
        # Chain the hook that is current at install time, not at import time:
        # another module (e.g. a reference kernel with its own pool_alias) may
        # have installed its `_reshape_memref` hook after this module loaded.
        _original = lowering._reshape_memref
        # pyrefly: ignore[bad-assignment]
        lowering._reshape_memref = _lower
        _installed = True


def storage_size(shape, dtype):
    """Bytes occupied by `(8, 128)`-tiled layout (rounding second-minor dim)."""
    bits = dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096


def view(ref, shape, dtype):
    """Return a `shape`/`dtype` view aliasing `ref`'s physical tiled storage.

    The returned view must be fully overwritten before it is read.
    """
    _install()
    ref = ref.bitcast(dtype)
    # pyrefly: ignore[bad-argument-count]
    return types.TransformedRef(ref.ref, (*ref.transforms, Reinterpret(tuple(shape))))
