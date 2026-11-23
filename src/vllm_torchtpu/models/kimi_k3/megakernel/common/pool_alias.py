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
"""Raw tiled arena views; every new view must be overwritten before reading.

Importing patches JAX's private ``jax._src.pallas.mosaic.lowering._reshape_memref``
so a :class:`Reinterpret` transform lowers to a ``tpu.reinterpret_cast`` with an
explicit tiled layout; other reshapes keep the standard lowering.

TODO: delete this local copy once ``megakernel/core/dma/pool_alias.py``
supports opt-in padded views for ``_PaddedBf16Rows`` and non-32-aligned
``uint8`` expert scale tensors (112 and 24 rows).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp
from jax import tree_util
from jax._src.pallas.mosaic import lowering
from jax._src.state import types
from jax.experimental.mosaic.dialects import tpu
from jax.extend.mlir import ir

TransformedRef = types.TransformedRef

if TYPE_CHECKING:
    from collections.abc import Sequence

    import jax


class _TransformedView(Protocol):
    """Protocol for bitcast reference views carrying ``ref`` and ``transforms``."""

    @property
    def ref(self) -> object:
        """Return underlying state reference."""
        ...

    @property
    def transforms(self) -> tuple[types.Transform, ...]:
        """Return accumulated reference transforms."""
        ...


class BitcastableRef(Protocol):
    """Protocol for Pallas references supporting element ``bitcast``."""

    def bitcast(self, dtype: jnp.dtype) -> object:
        """Reinterpret reference elements as ``dtype``."""
        ...


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Reinterpret(types.ReshapeTransform):
    """Ref transform marking a raw ``(8, 128)``-tiled arena shape reinterpretation."""

    if TYPE_CHECKING:
        shape: tuple[int, ...]

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Initialize reinterpret transform for ``shape``."""
            ...


_original = lowering._reshape_memref


def _lower(
    ref: ir.Value,
    transform: types.ReshapeTransform,
    aval: types.AbstractRef,
    block_shape: tuple[int, ...],
) -> tuple[object, tuple[int, ...]]:
    """Lower :class:`Reinterpret` transforms to tiled ``tpu.reinterpret_cast`` ops."""
    if not isinstance(transform, Reinterpret):
        return _original(ref, transform, aval, block_shape)
    transform._validate_shape(tuple(block_shape))
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


# pyrefly: ignore[bad-assignment]
lowering._reshape_memref = _lower


def storage_size(shape: Sequence[int], dtype: jax.typing.DTypeLike) -> int:
    """Return padded byte footprint of ``shape`` under ``(8, 128)``-tiled storage."""
    resolved_dtype = jnp.dtype(dtype)
    bits = resolved_dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096


def view(
    ref: BitcastableRef,
    shape: Sequence[int],
    dtype: jax.typing.DTypeLike,
) -> types.TransformedRef:
    """Return an unpadded ``shape`` and ``dtype`` tiled view over ``ref``."""
    aliased = cast("_TransformedView", ref.bitcast(jnp.dtype(dtype)))
    return types.TransformedRef(
        aliased.ref, (*aliased.transforms, Reinterpret(tuple(shape)))
    )
