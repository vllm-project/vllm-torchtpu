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
"""Experimental dense-decode batched metadata fusion.

Uses csa_metadata's exact address mapping and int32 overflow semantics.
Caller must validate one token per sequence, dense active prefix, and no
padded query suffix outside the timed region. No attention arithmetic changes.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import csa_metadata


def _kernel(top, pages, indices, lengths, *, page_size, num_pages, pages_per_seq):
    tile = top[...]
    token = pl.program_id(0) * tile.shape[0] + lax.broadcasted_iota(
        jnp.int32, (tile.shape[0], 1), 0
    )
    indices[...], lengths[...] = csa_metadata.map_indices(
        tile,
        pages[...],
        token,
        page_size=page_size,
        num_pages=num_pages,
        pages_per_seq=pages_per_seq,
    )


def decode_metadata(
    topk: jax.Array, pages: jax.Array, cu: jax.Array, *, page_size: int, num_pages: int
) -> tuple[jax.Array, jax.Array]:
    """Fuse top-k page lookup, padding scatter, and valid counts by sublane tile."""
    tokens, k = topk.shape
    seqs = cu.shape[0] - 1
    if (
        topk.dtype != jnp.int32
        or pages.dtype != jnp.int32
        or tokens > seqs
        or k != 512
        or pages.size % seqs
        or csa_metadata.get_device_name() != "TPU v7"
    ):
        raise NotImplementedError("Dense-decode metadata envelope mismatch")
    info = pltpu.get_tpu_info()
    rows, lanes = info.num_sublanes, info.num_lanes
    pps = pages.size // seqs
    padded_pps = pl.cdiv(pps, lanes) * lanes
    padded_tokens = pl.cdiv(tokens, rows) * rows
    page_rows = pages.reshape(seqs, pps)[:tokens]
    page_rows = jnp.pad(page_rows, ((0, padded_tokens - tokens), (0, padded_pps - pps)))
    top_rows = jnp.pad(topk, ((0, padded_tokens - tokens), (0, 0)), constant_values=-1)
    indices, lengths = pl.pallas_call(
        functools.partial(
            _kernel, page_size=page_size, num_pages=num_pages, pages_per_seq=pps
        ),
        in_specs=[
            pl.BlockSpec((rows, k), lambda t: (t, 0)),
            pl.BlockSpec((rows, padded_pps), lambda t: (t, 0)),
        ],
        out_specs=[
            pl.BlockSpec((rows, k), lambda t: (t, 0)),
            pl.BlockSpec((rows, 1), lambda t: (t, 0)),
        ],
        grid=(padded_tokens // rows,),
        out_shape=[
            jax.ShapeDtypeStruct((padded_tokens, k), jnp.int32),
            jax.ShapeDtypeStruct((padded_tokens, 1), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="CSA_DECODE_BATCH_METADATA",
    )(top_rows, page_rows)
    return indices[:tokens].reshape(-1), lengths[:tokens].reshape(-1)
