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
"""Utility, validation, and VMEM/DMA helper functions for ragged paged attention v3."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def next_power_of_2(x: int):
    """Finds the smallest power of 2 >= x using bit manipulation.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
    """
    assert x > 0
    if x == 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
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


def _get_num_lanes() -> int:
    try:
        return pltpu.get_tpu_info().num_lanes
    except ValueError:
        return 128


def broadcast_minor(src, shape):
    """Broadcast `src` across minor lane multiples to match `shape`."""
    if src.shape == shape:
        return src
    num_lanes = _get_num_lanes()
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
    num_lanes = _get_num_lanes()
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
    num_lanes = _get_num_lanes()
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


def get_span_int_dtype(
    q_dtype,
    pages_per_seq: int,
    page_size: int,
    bkv_csz: int,
    *,
    tpu_version: int | None = None,
):
    """Return `int16` if the `bkv_csz`-aligned KV span fits on TPU v6+, else `int32`."""
    if tpu_version is None:
        tpu_version = get_tpu_version()
    max_kv_span = align_to(pages_per_seq * page_size, bkv_csz) - 1
    if (
        get_dtype_packing(q_dtype) != 1
        and tpu_version >= 6
        and max_kv_span <= jnp.iinfo(jnp.int16).max
    ):
        return jnp.int16
    return jnp.int32


def validate_common_inputs(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    page_size: int,
    kv_packing: int,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    chunk_prefill_size: int | None = None,
    block_sizes_by_name: tuple[tuple[str, tuple[int, int, int, int] | None], ...] = (),
    vmem_limit_bytes: int | None = None,
) -> None:
    """Validate cache-layout-independent Q/K/V, metadata, and block-size inputs."""
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(
            f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}"
        )
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(
            f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}"
        )
    if not (
        jnp.issubdtype(q.dtype, jnp.floating)
        and jnp.issubdtype(k.dtype, jnp.floating)
        and jnp.issubdtype(v.dtype, jnp.floating)
    ):
        raise ValueError(
            f"Expected floating-point Q/K/V dtypes, got {q.dtype=}, {k.dtype=},"
            f" {v.dtype=}."
        )
    if k.dtype != v.dtype:
        raise ValueError(f"Expected {k.dtype=} to equal {v.dtype=}.")

    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]
    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(
            f"Expected {actual_num_q_heads=} to be divisible by {actual_num_kv_heads=}."
        )

    if not (
        jnp.int32
        == kv_lens.dtype
        == page_indices.dtype
        == cu_q_lens.dtype
        == distribution.dtype
    ):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}"
        )
    if not (
        len(kv_lens.shape)
        == len(page_indices.shape)
        == len(cu_q_lens.shape)
        == len(distribution.shape)
        == 1
    ):
        raise ValueError(
            f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}, {distribution.shape=}"
        )

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(
            f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
        )
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    for prefix, block_sizes in block_sizes_by_name:
        if block_sizes is None:
            continue
        bq_sz, bkv_sz, bq_csz, bkv_csz = block_sizes
        if not (bq_csz > 0 and bq_sz % bq_csz == 0):
            raise ValueError(
                f"{prefix} {bq_csz=} and {bq_sz=} must satisfy (0 < bq_csz and bq_sz"
                " % bq_csz == 0)."
            )
        if not (bkv_csz > 0 and bkv_sz % bkv_csz == 0):
            raise ValueError(
                f"{prefix} {bkv_csz=} and {bkv_sz=} must satisfy (0 < bkv_csz and"
                " bkv_sz % bkv_csz == 0)."
            )
        if bkv_sz % page_size != 0:
            raise ValueError(f"{prefix} {bkv_sz=} must be divisible by {page_size=}.")
        if bkv_csz % page_size != 0:
            raise ValueError(f"{prefix} {bkv_csz=} must be divisible by {page_size=}.")
