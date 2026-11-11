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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def pl_align_to(x, a):
    """Align `x` to `a` via `pl.cdiv` (`lax.div` on traced values)."""
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


def align_second_minor_sublanes(rows: int, dtype) -> int:
    """Align `shape[-2]` to Mosaic's TPU VMEM sublane tile for `dtype`."""
    assert rows > 0
    packing = get_dtype_packing(dtype)
    full_tile = 8 * packing
    tile = full_tile if rows >= full_tile else max(packing, next_power_of_2(rows))
    return align_to(rows, tile)


def get_tpu_version() -> int:
    """Return the numeric TPU generation, or `-1` if not on TPU."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    if kind.endswith("p") or kind.endswith("e"):
        kind = kind[:-1]
    if kind == "TPU7x":
        return 7
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


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
