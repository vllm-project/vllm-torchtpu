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
# Source provenance: inferact/k3-megakernel@30c9ba966b src/vllm_torchtpu/megakernel/common/pool_alias.py

"""Raw VMEM views; every new view must be overwritten before reading.

Source provenance: Adapted from inferact/k3-megakernel@30c9ba966b.

Calling :func:`view` patches JAX's private
``jax._src.pallas.mosaic.lowering._reshape_memref`` so a :class:`Reinterpret`
transform lowers to a ``tpu.reinterpret_cast`` with an explicit tiled layout;
other reshapes keep the stock lowering.
"""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import Any

import jax.numpy as jnp
from jax import tree_util
from jax._src.lib import tpu
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic import lowering
from jax._src.state import types

_Dynamic = Any


@dataclasses.dataclass(frozen=True, slots=True)
class Reinterpret(types.ReshapeTransform):
    """Manage reinterpret instance and state."""


_original: _Dynamic = None
_patched_box = [False]
_registered_box = [False]


def _ensure_patched() -> None:
    """Ensure reshape lowering is patched."""
    if not _registered_box[0]:
        tree_util.register_dataclass(Reinterpret)
        _registered_box[0] = True
    if not _patched_box[0]:
        globals()["_original"] = vars(lowering)["_reshape_memref"]
        vars(lowering)["_reshape_memref"] = _lower
        _patched_box[0] = True


def _lower(
    ref: _Dynamic, transform: _Dynamic, aval: _Dynamic, block_shape: _Dynamic
) -> _Dynamic:
    """Execute lower logic."""
    if not isinstance(transform, Reinterpret):
        return _original(ref, transform, aval, block_shape)
    operator.attrgetter("_validate_shape")(transform)(tuple(block_shape))
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


def storage_size(shape: _Dynamic, dtype: _Dynamic) -> _Dynamic:
    """Execute storage size logic."""
    dtype = jnp.dtype(dtype)
    bits = dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096


def view(ref: _Dynamic, shape: _Dynamic, dtype: _Dynamic) -> _Dynamic:
    """Execute view logic."""
    _ensure_patched()
    ref = ref.bitcast(dtype)
    return types.TransformedRef(ref.ref, (*ref.transforms, Reinterpret(tuple(shape))))
