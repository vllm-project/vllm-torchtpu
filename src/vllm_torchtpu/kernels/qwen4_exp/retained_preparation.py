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
"""Prepared-input boundary for tensor-derived QSA decode layouts."""

import jax
import jax.numpy as jnp

from .preparation_candidates import norm_with_multiplier, prepare_index_query
from .qsa import main_rope
from .query_reader_prepare import prepare_query_reader_layout
from .rope import select_rope_rows
from .tuned_block_sizes import _hardware, _vmem, use_query_prepare


def use_retained_preparation(
    qgate: jax.Array,
    key: jax.Array,
    index_qk: jax.Array,
    nq: jax.Array,
    nk: jax.Array,
    niq: jax.Array,
    multiplier: jax.Array | None,
    trig: jax.Array,
    *,
    rotary_dim: int,
) -> bool:
    if (
        nq.ndim != 1
        or nk.shape != nq.shape
        or niq.ndim != 1
        or min(nq.shape) <= 0
        or min(niq.shape) <= 0
        or multiplier is None
        or multiplier.shape != nq.shape
        or multiplier.dtype != jnp.float32
        or qgate.ndim != 2
        or key.ndim != 2
        or index_qk.ndim != 2
        or min(qgate.shape) <= 0
        or min(key.shape) <= 0
        or key.shape[0] != len(qgate)
        or index_qk.shape[0] != len(qgate)
        or qgate.shape[1] % (2 * len(nq))
        or key.shape[1] % len(nq)
        or index_qk.shape[1] % len(niq)
        or index_qk.shape[1] <= len(niq)
        or any(
            x.dtype != jnp.bfloat16 for x in (qgate, key, index_qk, nq, nk, niq, trig)
        )
        or rotary_dim > len(niq)
    ):
        return False
    heads = qgate.shape[1] // (2 * len(nq))
    kv_heads = key.shape[1] // len(nq)
    if heads % kv_heads:
        return False
    hw = _hardware()
    if hw is None:
        return False
    shaped = jax.ShapeDtypeStruct((len(qgate), heads, 2 * len(nq)), qgate.dtype)
    if not use_query_prepare(shaped, nq, trig, rotary_dim):
        return False
    query_memory = _vmem(
        (kv_heads, len(qgate), heads // kv_heads, 1, len(nq)), jnp.float32, hw
    )
    index_memory = 6 * _vmem(index_qk.shape, jnp.float32, hw)
    return query_memory + index_memory <= hw.vmem_bytes * 0.8


def produce_prepared(
    qgate: jax.Array,
    key: jax.Array,
    value: jax.Array,
    index_qk: jax.Array,
    nq: jax.Array,
    multiplier: jax.Array,
    niq: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    norm_eps: float,
    rotary_dim: int,
    rope_axes: tuple[int, ...] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    width = nq.shape[0]
    qgate = qgate.reshape(len(qgate), -1, 2 * width)
    key = key.reshape(len(key), -1, width)
    value = value.reshape(len(value), -1, width)
    q = prepare_query_reader_layout(
        qgate,
        nq,
        positions,
        trig,
        norm_eps,
        rotary_dim,
        rope_axes,
        kv_heads=key.shape[1],
    )
    key = main_rope(
        norm_with_multiplier(key, multiplier, norm_eps),
        positions,
        trig,
        rotary_dim,
        rope_axes,
    )
    angles = select_rope_rows(positions, trig, rotary_dim, rope_axes)
    iq = prepare_index_query(index_qk, niq, angles, norm_eps, rotary_dim)
    return q, key, value, qgate[..., width:], iq, index_qk[:, -niq.shape[0] :]
