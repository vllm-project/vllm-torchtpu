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
"""Logical <-> packed cache pages for the MSA tests."""

import jax.numpy as jnp
import numpy as np

CACHE_DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn]
PACKING = {jnp.bfloat16: 2, jnp.float8_e4m3fn: 4}


def rounded(x: np.ndarray, dtype) -> np.ndarray:
    """``x`` as stored in a cache of `dtype`, back in float32."""
    return np.asarray(jnp.asarray(x).astype(dtype).astype(jnp.float32))


def pack(rows: np.ndarray, dtype) -> jnp.ndarray:
    """``[pages, S, W]`` logical rows -> ``[pages, S / p, p, W]`` cache."""
    pages, page_size, width = rows.shape
    packing = PACKING[dtype]
    return jnp.asarray(
        rows.reshape(pages, page_size // packing, packing, width)
    ).astype(dtype)


def unpack(cache: jnp.ndarray) -> np.ndarray:
    """``[pages, S / p, p, W]`` cache -> float32 ``[pages, S, W]``."""
    pages, rows, packing, width = cache.shape
    return np.asarray(cache.astype(jnp.float32)).reshape(pages, rows * packing, width)


def pack_index(keys: np.ndarray, dtype) -> jnp.ndarray:
    """``[pages, S, D]`` index keys -> the index cache's dim-major pages
    ``[pages, D / p, p, S]`` (`msa_indexer.dim_major_rows`)."""
    keys = np.asarray(keys)
    packing = PACKING[dtype]
    pages, page_size, dim = keys.shape
    return jnp.asarray(
        keys.transpose(0, 2, 1).reshape(pages, dim // packing, packing, page_size)
    ).astype(dtype)


def pack_index_fp4(nibbles, scales=None) -> jnp.ndarray:
    """``[pages, S, D]`` 4-bit index nibbles (and ``[pages, S]`` folded slot
    scales) -> dim-major uint8 pages ``[pages, D / 8 (+ 1), 4, S]``."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

    return msa_fp4.pack_rows(jnp.swapaxes(jnp.asarray(nibbles), 1, 2), scales)


def unpack_index(cache: jnp.ndarray) -> np.ndarray:
    """Dim-major ``[pages, D / p, p, S]`` index pages -> float32 ``[pages, S, D]``."""
    pages, rows, packing, slots = cache.shape
    keys = np.asarray(cache.astype(jnp.float32)).reshape(pages, rows * packing, slots)
    return keys.transpose(0, 2, 1)
