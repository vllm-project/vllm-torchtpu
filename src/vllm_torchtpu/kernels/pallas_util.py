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
"""Shared Pallas TPU VMEM, DMA, dtype-packing, and alignment helper functions."""

import math
import re
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cdiv(a, b):
    """Ceiling integer division `(a + b - 1) // b` for Python ints or JAX tracers."""
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    """Round `x` up to a multiple of `a` via Python/JAX floor-division `cdiv`.

    Accepts both static Python `int` values and non-negative JAX scalar tracers.
    Explicitly multiplies `cdiv(x, a) * a` (required by MLA v2 `async_copy` sizes
    so Mosaic sees the static alignment factor `a`, e.g. `128`, and avoids TPU
    compile errors). Use `pl_align_to` instead only when a kernel schedule relies
    on `pl.cdiv` (`lax.div` truncation toward zero on traced scalars).
    """
    return cdiv(x, a) * a


def pl_align_to(x, a):
    """Round `x` up to a multiple of `a` via `pl.cdiv` (`lax.div` on tracers).

    Accepts static Python `int` values or JAX scalar tracers in Pallas schedules
    (such as `batched_rpa`) that lower `pl.cdiv` through `lax.div` rather than
    floor-division `//`.
    """
    return pl.cdiv(x, a) * a


def get_dtype_bitwidth(dtype) -> int:
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype) -> int:
    return 32 // get_dtype_bitwidth(dtype)


def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= `x`."""
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
    """Return the numeric TPU generation, or `-1` if not on TPU."""
    kind = jax.devices()[0].device_kind
    match = re.match(r"^TPU[^\d]*(\d+)", kind)
    if match is None:
        return -1
    return int(match.group(1))


_CHIP_VERSION_BY_TPU_GEN = {
    5: pltpu.ChipVersion.TPU_V5P,
    6: pltpu.ChipVersion.TPU_V6E,
    7: pltpu.ChipVersion.TPU_7X,
    8: pltpu.ChipVersion.TPU_8I,
}


def get_tpu_info_for_version(tpu_version: int):
    """Return a `pltpu.TpuInfo` record for `tpu_version` without requiring a TPU."""
    chip = _CHIP_VERSION_BY_TPU_GEN.get(tpu_version)
    if chip is None:
        raise NotImplementedError(f"Unsupported tpu_version={tpu_version}.")
    return pltpu.get_tpu_info_for_chip(chip, 1)


def align_sublane_rows(rows: int, dtype, info) -> int:
    """Upper-bound VMEM second-minor row count under Mosaic sublane tiling.

    Reads the full sublane tile height from `info.get_sublane_tiling(dtype)`.
    When `rows < full_tile`, Mosaic uses a compact power-of-two sub-tile of at
    least `get_dtype_packing(dtype)` rows. When `rows >= full_tile`, rounding up
    to `full_tile` is an upper bound (Mosaic uses the large tile only when `rows`
    is a multiple of it, so shapes like `uint8[24, 128]` or `uint8[136, 128]`
    incur no extra sublane padding in practice).
    """
    assert rows >= 0
    if rows == 0:
        return 0
    full_tile = info.get_sublane_tiling(dtype)
    if rows < full_tile:
        return max(get_dtype_packing(dtype), next_power_of_2(rows))
    return align_to(rows, full_tile)


def array_vmem_bytes(shape: Sequence[int], dtype, info) -> int:
    """Upper-bound VMEM bytes occupied by an array of `shape` and `dtype`.

    Reads lane width and dtype sublane tiling from `info` (`pltpu.TpuInfo`).
    1D arrays occupy a single lane-padded row; higher-rank arrays pad the minor
    axis to `info.num_lanes` and bound the second-minor axis via
    `align_sublane_rows` (an upper bound when `shape[-2] >= full_tile` is not a
    multiple of `full_tile`).
    """
    if math.prod(shape) == 0:
        return 0
    bitwidth = get_dtype_bitwidth(dtype)
    if len(shape) == 1:
        elems = align_to(shape[0], info.num_lanes)
    else:
        lanes = align_to(shape[-1], info.num_lanes)
        sublanes = align_sublane_rows(shape[-2], dtype, info)
        elems = math.prod(shape[:-2]) * sublanes * lanes
    return cdiv(elems * bitwidth, 8)


def broadcast_minor(src, shape):
    """Broadcast `src` across minor lane multiples to match `shape`."""
    if src.shape == shape:
        return src
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % num_lanes == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation in Mosaic.
    return jnp.concatenate(
        [src for _ in range(target_minor // src.shape[-1])], axis=-1
    )[..., : shape[-1]]


def strided_load(ref, start, sz, step, *, dtype=None):
    """Load `sz // step` strided 2D rows from a 32-bit VMEM `ref`."""
    assert get_dtype_packing(ref.dtype) == 1
    assert len(ref.shape) == 2
    rows, cols = ref.shape
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert cols % num_lanes == 0
    folds = cols // num_lanes
    ref = ref.reshape(rows * folds, num_lanes)
    start *= folds
    sz *= folds
    step *= folds
    assert sz % step == 0
    vec = jnp.concat(
        [ref[pl.ds(start + i, sz // step, step)] for i in range(folds)], axis=1
    )
    if dtype is not None:
        vec = pltpu.bitcast(vec, dtype)
    return vec


def strided_store(ref, start, sz, step, val):
    """Store `val` into `sz // step` strided 2D rows of a 32-bit VMEM `ref`."""
    assert get_dtype_packing(ref.dtype) == 1
    assert ref.dtype == val.dtype
    assert len(ref.shape) == 2
    assert sz % step == 0
    rows, cols = ref.shape
    assert val.shape == (sz // step, cols)
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert cols % num_lanes == 0
    folds = cols // num_lanes
    ref = ref.reshape(rows * folds, num_lanes)
    start *= folds
    sz *= folds
    step *= folds
    for i in range(folds):
        ref[pl.ds(start + i, sz // step, step)] = val[
            :, i * num_lanes : (i + 1) * num_lanes
        ]


def async_copy(src, dst, sem, wait: bool, *, debug_mode: bool = False):
    """Start or wait on an async TPU DMA transfer unless `debug_mode` is set."""
    if debug_mode:
        return
    cp = pltpu.make_async_copy(src, dst, sem)
    if wait:
        cp.wait()
    else:
        cp.start()


def has_bank_conflicts(stride: int, distance: int = 24, num_banks: int = 32) -> bool:
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False
