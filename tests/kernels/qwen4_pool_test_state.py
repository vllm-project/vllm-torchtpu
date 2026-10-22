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
"""Independent host fixtures for native main/BF16-index lane storage."""

import math

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import pallas as pl


def _upload_index(index: np.ndarray) -> jax.Array:
    """Transfer opaque tag planes without BF16 numerical materialization."""
    if jax.default_backend() == "cpu":
        return jnp.asarray(index)
    shape = index.shape
    planes = index.view(np.uint16).reshape(*shape[:-2], 1, 2, shape[-1])
    words = planes[..., 0, :].astype(np.uint32) | (
        planes[..., 1, :].astype(np.uint32) << 16
    )

    def kernel(source, target):
        target.bitcast(jnp.uint32)[...] = source[...]

    return pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec((1, *words.shape[1:]), lambda i: (i, 0, 0, 0, 0))],
        out_specs=pl.BlockSpec((1, *shape[1:]), lambda i: (i, 0, 0, 0, 0)),
        grid=(shape[0],),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.bfloat16),
    )(jnp.asarray(words))


def pack_state(main, compressed, ring, tags):
    """Reserve null ID0; co-own keys with main IDs, ring with separate IDs."""
    pages, block_size = main.shape[:2]
    slots, ring_rows, index_dim = ring.shape
    width = compressed.shape[-1]
    compressed_rows = math.prod(compressed.shape[1:-1])
    # Each physical BF16 token contains two lane rows. Int32 words use low
    # and high BF16 planes in that second-minor packing dimension.
    ring_tokens = math.ceil(ring_rows * index_dim / (2 * width))
    tag_tokens = math.ceil(math.prod(tags.shape[1:]) / width)
    alignment = math.lcm(16, 2, compressed_rows * 128 // block_size)
    rows = (
        math.ceil(max(compressed_rows, 2 * (ring_tokens + tag_tokens)) / alignment)
        * alignment
    )
    index = np.zeros((pages + slots + 1, rows, width), ml_dtypes.bfloat16)
    index[1 : pages + 1, :compressed_rows] = np.asarray(compressed).reshape(
        pages, compressed_rows, width
    )
    for slot in range(slots):
        page = pages + 1 + slot
        index[page].reshape(-1)[: ring_rows * index_dim] = np.asarray(
            ring[slot]
        ).reshape(-1)
        words = np.zeros((tag_tokens, width), np.uint32)
        words.reshape(-1)[: tags[slot].size] = np.asarray(
            tags[slot], np.uint32
        ).reshape(-1)
        planes = np.stack((words & 65535, words >> 16), axis=-2).astype(np.uint16)
        index[page, 2 * ring_tokens : 2 * (ring_tokens + tag_tokens)] = planes.view(
            ml_dtypes.bfloat16
        ).reshape(-1, width)
    main_pool = jnp.pad(main, ((1, 0),) + ((0, 0),) * (main.ndim - 1))
    return (
        main_pool,
        _upload_index(index.reshape(len(index), rows // 2, 1, 2, width)),
    ), np.arange(pages + 1, pages + slots + 1, dtype=np.int32)


def unpack_state(main_pool, index_pool, compressed_shape, ring_shape):
    """Decode lane words on host; do not reuse production writers/readers."""
    pages = main_pool.shape[0] - 1
    slots, ring_rows, index_dim = ring_shape
    width = index_pool.shape[-1]
    index = np.asarray(index_pool).reshape(index_pool.shape[0], -1, width)
    keys = index[1 : pages + 1, : math.prod(compressed_shape[1:-1])].reshape(
        compressed_shape
    )
    ring_tokens = math.ceil(ring_rows * index_dim / (2 * width))
    rings, tags = [], []
    for slot in range(slots):
        page = index[pages + 1 + slot]
        rings.append(
            page.reshape(-1)[: ring_rows * index_dim].reshape(ring_rows, index_dim)
        )
        planes = (
            page[2 * ring_tokens :]
            .view(np.uint16)
            .reshape(-1, 2, width)
            .astype(np.uint32)
        )
        words = planes[:, 0] | (planes[:, 1] << 16)
        tags.append(
            words.reshape(-1)[: ring_rows * 2].view(np.int32).reshape(ring_rows, 2)
        )
    return (
        main_pool[1:],
        jnp.asarray(keys),
        jnp.asarray(np.stack(rings)),
        jnp.asarray(np.stack(tags)),
    )
