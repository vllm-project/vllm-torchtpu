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
"""Pick, per step, which copy of each routed MoE expert computes it.

Under expert parallelism a decode-sized grouped matmul is weight-read bound:
a rank's time follows the number of distinct experts it computes, and the
all-reduce after the layer waits for the busiest rank. With redundant expert
slots (``layers/adapter/moe_redundant.py``) some experts have a second copy on
another rank. This kernel assigns every expert the step touches to one copy:

1. experts with one copy go to it, counting one unit of load on its rank;
2. experts with several copies then go, in expert order, to the copy whose
   rank has the least load so far (ties: the native copy).

Every token routed to an expert uses the same copy, so no weights are read
twice. The assignment is a function of the routed ids alone, identical on
every rank. It runs on the scalar core: a handful of scalar loops over at most
``T * top_k`` routes and ``E`` experts, with everything in SMEM.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Routes (T * top_k) above which the kernel is not used (SMEM budget; larger
# batches are compute bound, where whole-expert moves are the wrong lever).
MAX_ROUTES = 2048

_UNHIT = -1
_HIT = -2


def _kernel(ids_ref, copies_ref, out_ref, choice_ref, load_ref, *, n, e, c, ep, slots):
    def clear(i, _):
        choice_ref[i] = _UNHIT
        return 0

    lax.fori_loop(0, e, clear, 0)
    for r in range(ep):
        load_ref[r] = 0

    def mark(i, _):
        x = ids_ref[i]

        @pl.when(x >= 0)
        def _():
            choice_ref[x] = _HIT

        return 0

    lax.fori_loop(0, n, mark, 0)

    def fixed(x, _):
        native = copies_ref[x * c]
        single = copies_ref[x * c + 1] < 0
        take = jnp.logical_and(choice_ref[x] == _HIT, single)

        @pl.when(take)
        def _():
            r = native // slots
            load_ref[r] = load_ref[r] + 1
            choice_ref[x] = native

        return 0

    lax.fori_loop(0, e, fixed, 0)

    def flexible(x, _):
        @pl.when(choice_ref[x] == _HIT)
        def _():
            best = copies_ref[x * c]
            best_load = load_ref[best // slots]
            for k in range(1, c):
                p = copies_ref[x * c + k]
                ok = p >= 0
                pl_load = load_ref[jnp.where(ok, p, 0) // slots]
                better = jnp.logical_and(ok, pl_load < best_load)
                best = jnp.where(better, p, best)
                best_load = jnp.where(better, pl_load, best_load)
            r = best // slots
            load_ref[r] = load_ref[r] + 1
            choice_ref[x] = best

        return 0

    lax.fori_loop(0, e, flexible, 0)

    def write(i, _):
        x = ids_ref[i]
        out_ref[i] = jnp.where(x >= 0, choice_ref[jnp.maximum(x, 0)], -1)
        return 0

    lax.fori_loop(0, n, write, 0)


def _reference(ids, copies, *, ep, slots):
    """Same assignment in plain jnp (CPU tests and odd shapes)."""
    num_experts, c = copies.shape
    valid = ids >= 0
    hit = jnp.zeros((num_experts,), jnp.bool_).at[jnp.maximum(ids, 0)].max(valid)
    native = copies[:, 0]
    single = copies[:, 1] < 0 if c > 1 else jnp.ones_like(hit)
    load = jnp.zeros((ep,), jnp.int32).at[native // slots].add(
        jnp.logical_and(hit, single).astype(jnp.int32)
    )
    choice = jnp.where(jnp.logical_and(hit, single), native, _UNHIT)

    def flexible(x, carry):
        load, choice = carry
        todo = jnp.logical_and(hit[x], jnp.logical_not(single[x]))
        best = native[x]
        best_load = load[best // slots]
        for k in range(1, c):
            p = copies[x, k]
            ok = p >= 0
            pl_load = load[jnp.where(ok, p, 0) // slots]
            better = jnp.logical_and(ok, pl_load < best_load)
            best = jnp.where(better, p, best)
            best_load = jnp.where(better, pl_load, best_load)
        load = jnp.where(todo, load.at[best // slots].add(1), load)
        choice = jnp.where(todo, choice.at[x].set(best), choice)
        return load, choice

    _, choice = lax.fori_loop(0, num_experts, flexible, (load, choice))
    return jnp.where(valid, choice[jnp.maximum(ids, 0)], -1)


@functools.partial(jax.jit, static_argnames=("ep", "slots", "interpret"))
def replica_assign(
    ids: jax.Array,
    copies: jax.Array,
    *,
    ep: int,
    slots: int,
    interpret: bool = False,
) -> jax.Array:
    """Physical slot ids ``[T, K]`` for logical routed ids ``[T, K]``.

    Negative ids (rows the grouped matmul skips) take no part in the
    assignment and come back as ``-1``.

    ``copies[e]`` lists the physical slot ids holding logical expert ``e``:
    column 0 is the native copy, further columns replicas, ``-1`` for none.
    Physical id ``p`` lives on rank ``p // slots``. Above ``MAX_ROUTES``
    routes the native copies are used (no replicas).
    """
    num_tokens, topk = ids.shape
    num_experts, c = copies.shape
    ids = ids.astype(jnp.int32)
    copies = copies.astype(jnp.int32)
    n = num_tokens * topk
    if n > MAX_ROUTES or c < 2:
        return jnp.where(ids >= 0, copies[:, 0][jnp.maximum(ids, 0)], -1)
    if jax.default_backend() != "tpu" and not interpret:
        return _reference(ids.reshape(-1), copies, ep=ep, slots=slots).reshape(
            num_tokens, topk
        )
    kernel = functools.partial(_kernel, n=n, e=num_experts, c=c, ep=ep, slots=slots)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n,), jnp.int32),
        in_specs=[smem, smem],
        out_specs=smem,
        scratch_shapes=[
            pltpu.SMEM((num_experts,), jnp.int32),
            pltpu.SMEM((ep,), jnp.int32),
        ],
        interpret=interpret,
    )(ids.reshape(-1), copies.reshape(-1))
    return out.reshape(num_tokens, topk)
