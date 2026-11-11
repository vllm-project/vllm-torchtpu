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
"""RPA-specific configuration, validation, and bundle helpers for RPA v3."""

from collections.abc import Callable
from enum import Enum

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.pallas_util import (
    align_to,
    cdiv,
    get_dtype_packing,
    get_tpu_version,
)


class RpaCase(Enum):
    """Represents the different execution passes for Ragged Paged Attention.

    - DECODE: Sequences are in decode-only mode (`q_len == 1`).
    - PREFILL: Sequences are in chunked-prefill-only mode (`q_len` static).
    - MIXED: Sequences can be a mix of prefill and decode (`q_len` dynamic).
    """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self) -> str:
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(self, distribution, *, has_prefill_pass: bool = True):
        assert distribution.shape == (3,)
        if self == RpaCase.DECODE:
            return 0, distribution[0]
        if self == RpaCase.PREFILL:
            return distribution[0], distribution[1]
        if self == RpaCase.MIXED:
            start = distribution[1] if has_prefill_pass else distribution[0]
            return start, distribution[2]
        raise ValueError(f"Unsupported RPA case: {self}")


def get_span_int_dtype(
    q_dtype,
    pages_per_seq: int,
    page_size: int,
    bkv_csz: int = 1,
    *,
    tpu_version: int | None = None,
):
    """Return `int16` if the `bkv_csz`-aligned KV span fits on TPU v6+, else `int32`."""
    if tpu_version is None:
        tpu_version = get_tpu_version()
    max_kv_len = align_to(pages_per_seq * page_size, bkv_csz)
    if (
        get_dtype_packing(q_dtype) != 1
        and tpu_version >= 6
        and max_kv_len <= jnp.iinfo(jnp.int16).max
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


def validate_dynamic_common_inputs(
    queries: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> None:
    """Validate runtime `distribution`, `q_len`, `kv_len`, and `page_indices` ranges."""
    max_num_tokens = queries.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    i, j, k = distribution
    if not (i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")
    if k > max_num_seqs:
        raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")
    if cu_q_lens[k] > max_num_tokens:
        raise ValueError(f"Total q tokens {cu_q_lens[k]} must be <= {max_num_tokens=}.")

    for seq_i in range(k):
        q_len = cu_q_lens[seq_i + 1] - cu_q_lens[seq_i]
        kv_len = kv_lens[seq_i]
        if not (0 < q_len <= kv_len):
            raise ValueError(f"Require 0 < {q_len=} <= {kv_len=} at sequence {seq_i}.")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Require {page_cnt=} <= {pages_per_seq=} at sequence {seq_i} where"
                f" {kv_len=} and {page_size=}."
            )
        for p in range(page_cnt):
            page_idx = page_indices[seq_i * pages_per_seq + p]
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence"
                    f" {seq_i} where {kv_len=} and {page_size=}."
                )


def run_bundled_rpa(
    rpa_fn: Callable[..., tuple[jax.Array, jax.Array]],
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache_bundle: jax.Array,
    layer_idx: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    **rpa_kwargs,
) -> tuple[jax.Array, jax.Array]:
    """Flatten a block-major `[num_pages, num_layers, ...]` bundle and run `rpa_fn`."""
    num_blocks, num_layers = kv_cache_bundle.shape[:2]
    flat_bundle = kv_cache_bundle.reshape(
        num_blocks * num_layers, *kv_cache_bundle.shape[2:]
    )
    out, new_flat_bundle = rpa_fn(
        queries,
        keys,
        values,
        flat_bundle,
        kv_lens,
        page_indices * num_layers + layer_idx,
        cu_q_lens,
        distribution,
        **rpa_kwargs,
    )
    return out, new_flat_bundle.reshape(kv_cache_bundle.shape)
