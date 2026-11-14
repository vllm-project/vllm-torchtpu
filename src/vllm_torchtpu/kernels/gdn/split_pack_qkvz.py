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
"""Single-pass Pallas split and layout pack for GDN ``in_proj_qkvz`` outputs.

``in_proj_qkvz`` produces a 2D ``[tokens, qkv_size + local_num_v_heads * head_v_dim]``
tensor tiled as ``T(8, 128)`` along ``(tokens, cols)``. Downstream, ``gdn/v3``
reads ``qkv`` in compact ``float32[tokens, 1, qkv_size]`` (``T(1, 128)``) layout,
while ``RMSNormGated`` reads ``z`` as 3D
``[tokens, local_num_v_heads, head_v_dim]`` (``T(8, 128)`` along
``(heads, head_dim)``). Splitting ``mixed_qkvz`` in PyTorch/XLA materializes two
separate multi-pass HBM re-tilings per GDN layer: an ``f32[tokens, qkv_size]``
intermediate followed by a ``T(8, 128) -> T(1, 128)`` reshape copy, plus a
full-width ``mixed_qkvz`` sublane transpose and slice for ``z``.

``split_pack_gdn_qkvz`` reads ``mixed_qkvz`` once per token tile and writes both
target layouts directly from VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MAX_TILE = 64
_VMEM_BUDGET_FRACTION = 0.5
_INTERPRET_VMEM_CAPACITY_BYTES = 16 * 1024 * 1024


def legal_row_tile(rows: int, max_tile: int) -> int:
    """Largest multiple of 8 at or below ``max_tile`` that divides ``rows``."""
    if rows < 8 or rows % 8 != 0:
        raise ValueError(f"rows must be a positive multiple of 8, got {rows}.")
    if max_tile < 8 or max_tile % 8 != 0:
        raise ValueError(f"max_tile must be a positive multiple of 8, got {max_tile}.")
    for tile in range(min(rows, max_tile), 7, -8):
        if rows % tile == 0:
            return tile
    raise AssertionError("Unreachable: rows and max_tile are multiples of 8 >= 8.")


def _double_buffered_row_bytes(
    qkv_size: int,
    local_num_v_heads: int,
    head_v_dim: int,
    dtype_itemsize: int,
) -> int:
    packing = 4 // dtype_itemsize
    z_sublane_tile = 8 * packing
    aligned_v_heads = (
        (local_num_v_heads + z_sublane_tile - 1) // z_sublane_tile
    ) * z_sublane_tile
    total_cols = qkv_size + local_num_v_heads * head_v_dim
    in_bytes = total_cols * dtype_itemsize
    out_qkv_bytes = qkv_size * 4
    out_z_bytes = aligned_v_heads * head_v_dim * dtype_itemsize
    return 2 * (in_bytes + out_qkv_bytes + out_z_bytes)


def can_split_pack_qkvz(
    num_tokens: int,
    qkv_size: int,
    local_num_v_heads: int,
    head_v_dim: int,
    *,
    dtype_itemsize: int = 2,
    vmem_capacity_bytes: int,
) -> bool:
    """Return True when ``split_pack_gdn_qkvz`` meets alignment and VMEM bounds."""
    if num_tokens < 8 or num_tokens % 8 != 0:
        return False
    if qkv_size <= 0 or qkv_size % 128 != 0:
        return False
    if local_num_v_heads < 8 or local_num_v_heads % 8 != 0:
        return False
    if head_v_dim <= 0 or head_v_dim % 128 != 0:
        return False
    if dtype_itemsize not in (2, 4):
        return False
    min_tile_bytes = 8 * _double_buffered_row_bytes(
        qkv_size, local_num_v_heads, head_v_dim, dtype_itemsize
    )
    return min_tile_bytes <= int(_VMEM_BUDGET_FRACTION * vmem_capacity_bytes)


def _split_pack_gdn_qkvz_kernel(
    qkvz_ref,
    out_qkv_ref,
    out_z_ref,
    *,
    qkv_size: int,
) -> None:
    tile_rows = qkvz_ref.shape[0]
    local_num_v_heads = out_z_ref.shape[1]
    head_v_dim = out_z_ref.shape[2]
    # Slice VMEM in 8- or 16-row sub-tiles so live f32/bf16 intermediates fit in
    # vector registers without spilling back to VMEM scratch on wide GDN layers.
    step = 16 if qkvz_ref.dtype == jnp.bfloat16 and tile_rows % 16 == 0 else 8
    for r in range(0, tile_rows, step):
        row_slice = slice(r, r + step)
        out_qkv_ref[row_slice, :, :] = (
            qkvz_ref[row_slice, :qkv_size]
            .astype(jnp.float32)
            .reshape(step, 1, qkv_size)
        )
        out_z_ref[row_slice, :, :] = qkvz_ref[row_slice, qkv_size:].reshape(
            step, local_num_v_heads, head_v_dim
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "qkv_size",
        "local_num_v_heads",
        "head_v_dim",
        "max_tile",
        "vmem_capacity_bytes",
        "interpret",
    ),
)
def split_pack_gdn_qkvz(
    mixed_qkvz: jax.Array,
    *,
    qkv_size: int,
    local_num_v_heads: int,
    head_v_dim: int,
    max_tile: int = DEFAULT_MAX_TILE,
    vmem_capacity_bytes: int | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Split ``mixed_qkvz`` into ``(float32[B, 1, qkv_size], dtype[B, n_v, d_v])``.

    Args:
        mixed_qkvz: 2D array ``[tokens, qkv_size + local_num_v_heads * head_v_dim]``
            in ``bfloat16`` or ``float32``.
        qkv_size: Rank-local width of concatenated ``[q, k, v]`` channels.
        local_num_v_heads: Rank-local value head count.
        head_v_dim: Per-head value channel dimension.
        max_tile: Maximum row tile per Pallas grid step.
        vmem_capacity_bytes: Optional VMEM capacity override; defaults to chip
            VMEM capacity on TPU or 16 MiB in ``interpret`` mode.
        interpret: Run ``pallas_call`` in CPU interpret mode for unit testing.

    Returns:
        Tuple ``(qkv_compact_f32, z_3d)`` matching ``mixed_qkvz[:, :qkv_size]``
        cast/reshaped to ``float32[tokens, 1, qkv_size]`` and
        ``mixed_qkvz[:, qkv_size:].reshape(tokens, local_num_v_heads, head_v_dim)``.
    """
    if mixed_qkvz.ndim != 2:
        raise ValueError(f"mixed_qkvz must be 2D, got shape {mixed_qkvz.shape}.")
    if mixed_qkvz.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(
            f"mixed_qkvz dtype must be bfloat16 or float32, got {mixed_qkvz.dtype}."
        )
    num_tokens, total_cols = mixed_qkvz.shape
    z_size = local_num_v_heads * head_v_dim
    if total_cols != qkv_size + z_size:
        raise ValueError(
            f"mixed_qkvz width {total_cols} != qkv_size ({qkv_size}) + "
            f"local_num_v_heads * head_v_dim ({z_size})."
        )
    dtype_itemsize = jnp.dtype(mixed_qkvz.dtype).itemsize
    if vmem_capacity_bytes is not None:
        vmem_cap = vmem_capacity_bytes
    elif interpret:
        vmem_cap = _INTERPRET_VMEM_CAPACITY_BYTES
    else:
        vmem_cap = pltpu.get_tpu_info().vmem_capacity_bytes

    if not can_split_pack_qkvz(
        num_tokens,
        qkv_size,
        local_num_v_heads,
        head_v_dim,
        dtype_itemsize=dtype_itemsize,
        vmem_capacity_bytes=vmem_cap,
    ):
        raise ValueError(
            "split_pack_gdn_qkvz alignment or VMEM precondition not met: "
            f"num_tokens={num_tokens}, qkv_size={qkv_size}, "
            f"local_num_v_heads={local_num_v_heads}, head_v_dim={head_v_dim}."
        )

    row_bytes = _double_buffered_row_bytes(
        qkv_size, local_num_v_heads, head_v_dim, dtype_itemsize
    )
    vmem_budget = int(_VMEM_BUDGET_FRACTION * vmem_cap)
    vmem_max_rows = ((vmem_budget // row_bytes) // 8) * 8
    effective_max_tile = max(8, min(max_tile, vmem_max_rows))
    tile_rows = legal_row_tile(num_tokens, effective_max_tile)

    return pl.pallas_call(
        functools.partial(_split_pack_gdn_qkvz_kernel, qkv_size=qkv_size),
        grid=(num_tokens // tile_rows,),
        in_specs=[pl.BlockSpec((tile_rows, total_cols), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((tile_rows, 1, qkv_size), lambda i: (i, 0, 0)),
            pl.BlockSpec(
                (tile_rows, local_num_v_heads, head_v_dim), lambda i: (i, 0, 0)
            ),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, 1, qkv_size), jnp.float32),
            jax.ShapeDtypeStruct(
                (num_tokens, local_num_v_heads, head_v_dim), mixed_qkvz.dtype
            ),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name=f"gdn_split_pack_qkvz_t{tile_rows}",
    )(mixed_qkvz)
