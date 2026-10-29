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
"""Typed state regions in Qwen4Exp's native attention-shaped pool.

Only requested regions cross HBM. Type reinterpretation and lane reshaping
take place inside Pallas; the persistent pool never gets a typed XLA view.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows
from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv


def read_region(pool, ids, *, elements, dtype, tok0=0):
    if len(ids) == 0 or elements == 0:
        return jnp.zeros((len(ids), elements), dtype)
    ntok, rows = region_rows(pool, elements, dtype)
    values = pool_adapters.gather_region(
        pool,
        jnp.maximum(ids, 0),
        tok0=tok0,
        ntok=ntok,
        out_dtype=dtype,
    )
    return values.reshape(len(ids), rows * pool.shape[-1])[:, :elements]


def write_region(pool, values, ids, *, tok0=0):
    """Write unique positive manager IDs; idle/null IDs leave all bytes intact."""
    count, elements = values.shape
    if count == 0 or elements == 0:
        return pool
    ntok, rows = region_rows(pool, elements, values.dtype)
    lanes = pool.shape[-1]
    values = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(
        count, rows, lanes
    )
    return pool_adapters.scatter_region(
        pool, values, ids, tok0=tok0, ntok=ntok, skip_null=True
    )


def pooled_short_conv(
    x: jax.Array,
    weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    *,
    dilation: int,
    state_len: int,
) -> tuple[jax.Array, jax.Array]:
    if x.shape[0] == 0 or len(ids) == 0:
        return pool, jnp.zeros_like(x)
    channels = x.shape[-1]
    history = (
        read_region(
            pool, ids, elements=state_len * channels, dtype=jnp.bfloat16
        ).reshape(len(ids), state_len, channels)
        if state_len
        else jnp.zeros((len(ids), 0, channels), jnp.bfloat16)
    )
    active = (ids > 0) & (starts[1:] > starts[:-1])
    slots = jnp.where(active, jnp.arange(len(ids), dtype=jnp.int32), -1)
    output, updated = short_conv(
        x,
        weight,
        history,
        starts,
        slots,
        initial,
        dilation=dilation,
    )
    if state_len:
        pool = write_region(
            pool, updated.reshape(len(ids), -1), jnp.where(active, ids, 0)
        )
    return pool, output
