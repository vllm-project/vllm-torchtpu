# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
#   src/vllm_torchtpu/megakernel/common/pool_alias.py
"""Raw VMEM views; every new view must be overwritten before reading.

Calling :func:`view` patches JAX's private
``jax._src.pallas.mosaic.lowering._reshape_memref`` so a :class:`Reinterpret`
transform lowers to a ``tpu.reinterpret_cast`` with an explicit tiled layout;
other reshapes keep the default lowering.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import tree_util
from jax._src.lib import tpu
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic import lowering
from jax._src.state import types

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True, slots=True)
class Reinterpret(types.ReshapeTransform):
    """Reshape transform that lowers via tiled ``tpu.reinterpret_cast``."""

    if TYPE_CHECKING:

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Initialize view transform."""


tree_util.register_dataclass(Reinterpret)
_ORIGINAL = vars(lowering)["_reshape_memref"]


def _lower(
    ref: ir.Value,
    transform: types.Transform,
    aval: jax.core.ShapedArray,
    block_shape: Sequence[int],
) -> tuple[ir.Value, tuple[int, ...]]:
    """Lower ``Reinterpret`` views to an explicit tiled memref cast."""
    if not isinstance(transform, Reinterpret):
        return _ORIGINAL(ref, transform, aval, block_shape)
    transform.transform_type(jax.core.ShapedArray(tuple(block_shape), aval.dtype))
    original_type = ir.MemRefType(ref.type)
    bits = aval.dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    columns = transform.shape[-1] // 128
    # Tile strides: the last two dims are (rows, 128)-tiled, the leading
    # dims dense over them.
    strides = [columns, 1]
    stride = (transform.shape[-2] + rows - 1) // rows * columns
    for dimension in reversed(transform.shape[:-2]):
        strides.insert(0, stride)
        stride *= dimension
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


if vars(lowering)["_reshape_memref"] is not _lower:
    vars(lowering)["_reshape_memref"] = _lower


def storage_size(shape: Sequence[int], dtype: jax.typing.DTypeLike) -> int:
    """Return the padded tile byte footprint for ``shape`` and ``dtype``."""
    resolved_dtype = jnp.dtype(dtype)
    bits = resolved_dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096


def view(
    ref: types.TransformedRef,
    shape: Sequence[int],
    dtype: jax.typing.DTypeLike,
) -> types.TransformedRef:
    """Reinterpret ``ref`` as ``shape`` and ``dtype`` over tiled VMEM."""
    bitcast_ref = ref.bitcast(dtype)
    return types.TransformedRef(
        bitcast_ref.ref,
        (*bitcast_ref.transforms, Reinterpret(tuple(shape))),
    )
