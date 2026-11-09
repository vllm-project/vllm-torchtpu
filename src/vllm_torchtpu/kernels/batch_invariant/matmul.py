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
"""Batch-invariant dense matmul on TPU.

XLA picks a dot tiling from the full operand shapes, so the same row of `x`
gets a different K accumulation order when the number of rows changes. This
kernel fixes tile_k and tile_n from the weight shape alone and accumulates the
K tiles in a fixed order in fp32, so a row's result never depends on how many
other rows share the call or where it sits.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(x_ref, w_ref, o_ref, acc_ref, *, nk: int, precision, rhs_nk: bool):
    k = pl.program_id(2)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    dims = (((1,), (1,)), ((), ())) if rhs_nk else (((1,), (0,)), ((), ()))
    acc_ref[...] += jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        dims,
        preferred_element_type=jnp.float32,
        precision=precision,
    )

    @pl.when(k == nk - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def _vmem_capacity_bytes() -> int:
    if jax.default_backend() == "tpu":
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    return 64 * 1024 * 1024


def vmem_budget_bytes() -> int:
    return min(40 * 1024 * 1024, (_vmem_capacity_bytes() * 5) // 8)


def _divisor_tile(dim: int, cands: tuple[int, ...]) -> int | None:
    for c in cands:
        if dim % c == 0:
            return c
    return None


def default_tile_k(kdim: int) -> int:
    """K tile from K alone: the only tile that changes a row's result."""
    return _divisor_tile(kdim, (2048, 1024, 512, 256, 128)) or kdim


def lane_tile(n: int, cap: int = 2048) -> int:
    """Largest multiple of 128 up to cap that divides n (a multiple of 128)."""
    for t in range(min(n, cap) // 128 * 128, 0, -128):
        if n % t == 0:
            return t
    raise ValueError(f"{n} is not a multiple of 128")


def default_tiles(
    m: int, kdim: int, n: int, x_bytes: int, w_bytes: int, o_bytes: int
) -> tuple[int, int, int]:
    """(tile_m, tile_n, tile_k); tile_m and tile_n only affect speed."""
    tk = default_tile_k(kdim)
    # Partial edge blocks are slow on v7x: callers pad n to a multiple of 128.
    tn = lane_tile(n) if n % 128 == 0 else n
    tm = 8
    while tm < m and tm < 1024:
        tm *= 2

    def vmem(tm_: int) -> int:
        return (
            2 * tm_ * tk * x_bytes
            + 2 * tk * tn * w_bytes
            + tm_ * tn * (2 * o_bytes + 4)
        )

    vmem_budget = vmem_budget_bytes()
    while tm > 8 and vmem(tm) > vmem_budget:
        tm //= 2
    return tm, tn, tk


@functools.partial(
    jax.jit,
    static_argnames=(
        "out_dtype",
        "tile_m",
        "tile_n",
        "tile_k",
        "highest",
        "rhs_nk",
        "interpret",
    ),
)
def det_matmul(
    x: jax.Array,
    w: jax.Array,
    *,
    out_dtype=None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
    highest: bool = False,
    rhs_nk: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """`x [M, K] @ w [K, N]` (or `w [N, K]` if rhs_nk) with M independent tiling."""
    m, kdim = x.shape
    n, kdim_w = w.shape if rhs_nk else w.shape[::-1]
    assert kdim == kdim_w, (x.shape, w.shape)
    out_dtype = out_dtype or x.dtype
    d_tm, d_tn, d_tk = default_tiles(
        m,
        kdim,
        n,
        x.dtype.itemsize,
        w.dtype.itemsize,
        jnp.dtype(out_dtype).itemsize,
    )
    tm, tn, tk = tile_m or d_tm, tile_n or d_tn, tile_k or d_tk
    # Edge blocks are partial: their garbage rows/columns only reach output
    # rows/columns that Pallas clips, since every output element is one row dot.
    m_pad = pl.cdiv(m, 8) * 8
    if m_pad != m:
        x = jnp.pad(x, ((0, m_pad - m), (0, 0)))
    nk = kdim // tk
    precision = jax.lax.Precision.HIGHEST if highest else jax.lax.Precision.DEFAULT
    out = pl.pallas_call(
        functools.partial(_kernel, nk=nk, precision=precision, rhs_nk=rhs_nk),
        grid=(pl.cdiv(m_pad, tm), pl.cdiv(n, tn), nk),
        in_specs=[
            pl.BlockSpec((tm, tk), lambda i, j, k: (i, k)),
            pl.BlockSpec((tn, tk), lambda i, j, k: (j, k))
            if rhs_nk
            else pl.BlockSpec((tk, tn), lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec((tm, tn), lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m_pad, n), out_dtype),
        scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=min(56 * 1024 * 1024, (_vmem_capacity_bytes() * 7) // 8),
        ),
        interpret=interpret,
    )(x, w)
    return out[:m] if m_pad != m else out
