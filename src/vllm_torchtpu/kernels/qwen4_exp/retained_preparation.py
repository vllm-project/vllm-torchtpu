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
"""Explicit prepared-input boundary for the validated B16 QSA decode path."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .preparation_candidates import norm_with_multiplier, prepare_index_query
from .qsa import main_rope
from .query_reader_prepare import prepare_query_reader_layout
from .tuned_block_sizes import _hardware


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
        multiplier is None
        or multiplier.shape != (256,)
        or multiplier.dtype != jnp.float32
        or qgate.shape != (16, 12288)
        or key.shape != (16, 512)
        or index_qk.shape != (16, 640)
        or nq.shape != (256,)
        or nk.shape != (256,)
        or niq.shape != (128,)
        or any(
            x.dtype != jnp.bfloat16 for x in (qgate, key, index_qk, nq, nk, niq, trig)
        )
        or trig.ndim != 2
        or trig.shape[0] == 0
        or trig.shape[1] != 128
        or rotary_dim != 64
    ):
        return False
    hardware = _hardware()
    return hardware is not None and hardware.chip_version == pltpu.ChipVersion.TPU_7X


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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    width = nq.shape[0]
    qgate = qgate.reshape(len(qgate), -1, 2 * width)
    key = key.reshape(len(key), -1, width)
    value = value.reshape(len(value), -1, width)
    q = prepare_query_reader_layout(qgate, nq, positions, trig, norm_eps, rotary_dim)
    key = main_rope(
        norm_with_multiplier(key, multiplier, norm_eps), positions, trig, rotary_dim
    )
    angles = jax.lax.optimization_barrier(trig[positions])
    iq = prepare_index_query(index_qk, niq, angles, norm_eps)
    return q, key, value, qgate[..., width:], iq, index_qk[:, -niq.shape[0] :]
