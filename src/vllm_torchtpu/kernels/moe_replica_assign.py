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
2. experts with several copies then go, in order of first appearance among
   the routes, to the copy whose rank has the least load so far (ties: the
   native copy).

Every token routed to an expert uses the same copy, so no weights are read
twice. The assignment is a function of the routed ids alone, identical on
every rank. It runs on the scalar core: scalar loops over the ``T * top_k``
routes and the experts they touch, with everything in SMEM.
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


def _kernel(
    ids_ref, copies_ref, init_ref, out_ref, hit_ref, load_ref, *rest, n, c, ep, slots, refine
):
    # init_ref arrives as all -1 (_UNHIT), so no clearing loop over E; every
    # loop below runs over the routes or the experts they touch only.
    choice_ref = init_ref
    for r in range(ep):
        load_ref[r] = 0

    def mark(i, cnt):
        x = ids_ref[i]
        new = jnp.logical_and(x >= 0, choice_ref[jnp.maximum(x, 0)] == _UNHIT)

        @pl.when(new)
        def _():
            choice_ref[x] = _HIT
            hit_ref[cnt] = x

        return cnt + new.astype(jnp.int32)

    cnt = lax.fori_loop(0, n, mark, jnp.int32(0), unroll=8)

    def fixed(j, _):
        x = hit_ref[j]
        native = copies_ref[x * c]

        @pl.when(copies_ref[x * c + 1] < 0)
        def _():
            r = native // slots
            load_ref[r] = load_ref[r] + 1
            choice_ref[x] = native

        return 0

    lax.fori_loop(0, cnt, fixed, 0)

    def flexible(j, _):
        x = hit_ref[j]

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
            if refine:
                lst_ref, lcnt_ref, _ = rest
                k = lcnt_ref[r]
                lst_ref[r * lcap + k] = x
                lcnt_ref[r] = k + 1

        return 0

    if refine:
        lst_ref, lcnt_ref, flag_ref = rest
        lcap = min(n, copies_ref.shape[0] // c)
        for r in range(ep):
            lcnt_ref[r] = 0
    lax.fori_loop(0, cnt, flexible, 0)
    if refine:
        _refine(copies_ref, choice_ref, load_ref, lst_ref, lcnt_ref, flag_ref,
                c=c, ep=ep, slots=slots, lcap=lcap, iters=refine)

    def write(i, _):
        x = ids_ref[i]
        out_ref[i] = jnp.where(x >= 0, choice_ref[jnp.maximum(x, 0)], -1)
        return 0

    lax.fori_loop(0, n, write, 0, unroll=8)


def _refine(copies_ref, choice_ref, load_ref, lst_ref, lcnt_ref, flag_ref, *, c, ep, slots, lcap, iters):
    """Lower the busiest rank's load by moves of replicated experts.

    Each round takes the first rank A at the maximum load M and looks for
    either a direct move of one of A's replicated experts to another copy on a
    rank with load <= M - 2, or a two-step chain: e from A to B (load M - 1)
    while an expert e2 on B moves to a third rank C with load <= M - 2. A
    round without such a move ends the search (the maximum cannot drop through
    A). Experts stay whole; the result is a valid assignment either way.
    flag_ref[0] = 1 while searching, flag_ref[1] = found in this round."""
    flag_ref[0] = 1

    def move(e, p_new, frm, to, idx):
        # e sits at lst[frm][idx]: swap-remove it, append to `to`.
        last = lcnt_ref[frm] - 1
        lst_ref[frm * lcap + idx] = lst_ref[frm * lcap + last]
        lcnt_ref[frm] = last
        k = lcnt_ref[to]
        lst_ref[to * lcap + k] = e
        lcnt_ref[to] = k + 1
        load_ref[frm] = load_ref[frm] - 1
        load_ref[to] = load_ref[to] + 1
        choice_ref[e] = p_new

    def round_(it, _):
        @pl.when(flag_ref[0] == 1)
        def _():
            m = load_ref[0]
            a = jnp.int32(0)
            for r in range(1, ep):
                better = load_ref[r] > m
                a = jnp.where(better, r, a)
                m = jnp.where(better, load_ref[r], m)
            flag_ref[1] = 0

            def over_a(i, _):
                @pl.when(flag_ref[1] == 0)
                def _():
                    e = lst_ref[a * lcap + i]
                    for k in range(c):
                        p = copies_ref[e * c + k]
                        b = jnp.where(p >= 0, p, 0) // slots
                        ok = jnp.logical_and(p >= 0, b != a)
                        lb = load_ref[b]

                        @pl.when(jnp.logical_and(flag_ref[1] == 0, jnp.logical_and(ok, lb <= m - 2)))
                        def _():
                            move(e, p, a, b, i)
                            flag_ref[1] = 1

                        @pl.when(jnp.logical_and(flag_ref[1] == 0, jnp.logical_and(ok, lb == m - 1)))
                        def _():
                            def over_b(j, _):
                                @pl.when(flag_ref[1] == 0)
                                def _():
                                    e2 = lst_ref[b * lcap + j]
                                    for k2 in range(c):
                                        p2 = copies_ref[e2 * c + k2]
                                        cr = jnp.where(p2 >= 0, p2, 0) // slots
                                        ok2 = jnp.logical_and(
                                            p2 >= 0, jnp.logical_and(cr != b, cr != a)
                                        )

                                        @pl.when(jnp.logical_and(
                                            flag_ref[1] == 0,
                                            jnp.logical_and(ok2, load_ref[cr] <= m - 2),
                                        ))
                                        def _():
                                            move(e2, p2, b, cr, j)
                                            move(e, p, a, b, i)
                                            flag_ref[1] = 1

                                return 0

                            lax.fori_loop(0, lcnt_ref[b], over_b, 0)

                return 0

            lax.fori_loop(0, lcnt_ref[a], over_a, 0)

            @pl.when(flag_ref[1] == 0)
            def _():
                flag_ref[0] = 0

        return 0

    lax.fori_loop(0, iters, round_, 0)


def flat_table(copies: jax.Array, slots: int) -> jax.Array:
    """``[E, c]`` copies table -> ``[E * c * 2]`` (physical id, rank) pairs,
    ``(-1, -1)`` where an expert has no such copy. Built once at load time
    for `replica_assign_fast` (no per-step reshape, no scalar division)."""
    copies = jnp.asarray(copies, jnp.int32)
    rank = jnp.where(copies >= 0, copies // slots, -1)
    return jnp.stack([copies, rank], axis=-1).reshape(-1)


def _kernel_fast(tab_ref, ids_ref, out_ref, choice_ref, hit_ref, load_ref, *rest,
                 n, c, ep, num_experts, refine):
    """`_kernel` on a `flat_table`: same assignment, same output.

    Differences that cost nothing in the result: the choice array is cleared
    in-kernel (no ``[E]`` init input), copy ranks come from the table, and the
    route loops store unconditionally (the scalar core pays for branches)."""
    w = 2 * c
    for e in range(num_experts):
        choice_ref[e] = _UNHIT
    for r in range(ep):
        load_ref[r] = 0

    def mark(i, cnt):
        x = ids_ref[i]
        xs = jnp.maximum(x, 0)
        cur = choice_ref[xs]
        new = jnp.logical_and(x >= 0, cur == _UNHIT)
        choice_ref[xs] = jnp.where(new, _HIT, cur)
        hit_ref[cnt] = xs  # hit_ref has one spare entry past the last
        return cnt + new.astype(jnp.int32)

    cnt = lax.fori_loop(0, n, mark, jnp.int32(0), unroll=8)

    def fixed(j, _):
        x = hit_ref[j]
        single = tab_ref[x * w + 2] < 0
        r = tab_ref[x * w + 1]
        load_ref[r] = load_ref[r] + single.astype(jnp.int32)
        choice_ref[x] = jnp.where(single, tab_ref[x * w], _HIT)
        return 0

    lax.fori_loop(0, cnt, fixed, 0)

    if refine:
        lst_ref, lcnt_ref, flag_ref = rest
        lcap = min(n, num_experts)
        for r in range(ep):
            lcnt_ref[r] = 0

    def flexible(j, _):
        x = hit_ref[j]
        todo = choice_ref[x] == _HIT
        best = tab_ref[x * w]
        br = tab_ref[x * w + 1]
        best_load = load_ref[br]
        for k in range(1, c):
            p = tab_ref[x * w + 2 * k]
            pr = tab_ref[x * w + 2 * k + 1]
            ok = p >= 0
            pl_load = load_ref[jnp.maximum(pr, 0)]
            better = jnp.logical_and(ok, pl_load < best_load)
            best = jnp.where(better, p, best)
            br = jnp.where(better, pr, br)
            best_load = jnp.where(better, pl_load, best_load)
        load_ref[br] = load_ref[br] + todo.astype(jnp.int32)
        choice_ref[x] = jnp.where(todo, best, choice_ref[x])
        if refine:
            @pl.when(todo)
            def _():
                k = lcnt_ref[br]
                lst_ref[br * lcap + k] = x
                lcnt_ref[br] = k + 1

        return 0

    lax.fori_loop(0, cnt, flexible, 0)
    if refine:
        _refine_fast(tab_ref, choice_ref, load_ref, lst_ref, lcnt_ref, flag_ref,
                     c=c, ep=ep, lcap=lcap, iters=refine)

    def write(i, _):
        x = ids_ref[i]
        out_ref[i] = jnp.where(x >= 0, choice_ref[jnp.maximum(x, 0)], -1)
        return 0

    lax.fori_loop(0, n, write, 0, unroll=8)


def _refine_fast(tab_ref, choice_ref, load_ref, lst_ref, lcnt_ref, flag_ref, *, c, ep, lcap, iters):
    """`_refine` reading copy ranks from the flat table."""
    w = 2 * c
    flag_ref[0] = 1

    def move(e, p_new, frm, to, idx):
        last = lcnt_ref[frm] - 1
        lst_ref[frm * lcap + idx] = lst_ref[frm * lcap + last]
        lcnt_ref[frm] = last
        k = lcnt_ref[to]
        lst_ref[to * lcap + k] = e
        lcnt_ref[to] = k + 1
        load_ref[frm] = load_ref[frm] - 1
        load_ref[to] = load_ref[to] + 1
        choice_ref[e] = p_new

    def round_(it, _):
        @pl.when(flag_ref[0] == 1)
        def _():
            m = load_ref[0]
            a = jnp.int32(0)
            for r in range(1, ep):
                better = load_ref[r] > m
                a = jnp.where(better, r, a)
                m = jnp.where(better, load_ref[r], m)
            flag_ref[1] = 0

            def over_a(i, _):
                @pl.when(flag_ref[1] == 0)
                def _():
                    e = lst_ref[a * lcap + i]
                    for k in range(c):
                        p = tab_ref[e * w + 2 * k]
                        b = jnp.maximum(tab_ref[e * w + 2 * k + 1], 0)
                        ok = jnp.logical_and(p >= 0, b != a)
                        lb = load_ref[b]

                        @pl.when(jnp.logical_and(flag_ref[1] == 0, jnp.logical_and(ok, lb <= m - 2)))
                        def _():
                            move(e, p, a, b, i)
                            flag_ref[1] = 1

                        @pl.when(jnp.logical_and(flag_ref[1] == 0, jnp.logical_and(ok, lb == m - 1)))
                        def _():
                            def over_b(j, _):
                                @pl.when(flag_ref[1] == 0)
                                def _():
                                    e2 = lst_ref[b * lcap + j]
                                    for k2 in range(c):
                                        p2 = tab_ref[e2 * w + 2 * k2]
                                        cr = jnp.maximum(tab_ref[e2 * w + 2 * k2 + 1], 0)
                                        ok2 = jnp.logical_and(
                                            p2 >= 0, jnp.logical_and(cr != b, cr != a)
                                        )

                                        @pl.when(jnp.logical_and(
                                            flag_ref[1] == 0,
                                            jnp.logical_and(ok2, load_ref[cr] <= m - 2),
                                        ))
                                        def _():
                                            move(e2, p2, b, cr, j)
                                            move(e, p, a, b, i)
                                            flag_ref[1] = 1

                                return 0

                            lax.fori_loop(0, lcnt_ref[b], over_b, 0)

                return 0

            lax.fori_loop(0, lcnt_ref[a], over_a, 0)

            @pl.when(flag_ref[1] == 0)
            def _():
                flag_ref[0] = 0

        return 0

    lax.fori_loop(0, iters, round_, 0)


@functools.partial(jax.jit, static_argnames=("num_experts", "ep", "refine", "interpret"))
def replica_assign_fast(
    ids: jax.Array,
    table: jax.Array,
    *,
    num_experts: int,
    ep: int,
    refine: int = 0,
    interpret: bool = False,
) -> jax.Array:
    """`replica_assign` on a `flat_table` (same result for the same copies)."""
    num_tokens, topk = ids.shape
    c = table.shape[0] // (2 * num_experts)
    n = num_tokens * topk
    ids = ids.astype(jnp.int32)
    table = table.astype(jnp.int32)
    if n > MAX_ROUTES or c < 2:
        native = table.reshape(num_experts, c, 2)[:, 0, 0]
        return jnp.where(ids >= 0, native[jnp.maximum(ids, 0)], -1)
    if jax.default_backend() != "tpu" and not interpret:
        return _reference_ranked(ids.reshape(-1), table, num_experts=num_experts, c=c, ep=ep).reshape(
            num_tokens, topk
        )
    lcap = min(n, num_experts)
    extra = (
        [
            pltpu.SMEM((ep * lcap,), jnp.int32),
            pltpu.SMEM((ep,), jnp.int32),
            pltpu.SMEM((2,), jnp.int32),
        ]
        if refine
        else []
    )
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    out = pl.pallas_call(
        functools.partial(_kernel_fast, n=n, c=c, ep=ep, num_experts=num_experts, refine=refine),
        out_shape=jax.ShapeDtypeStruct((n,), jnp.int32),
        in_specs=[smem, smem],
        out_specs=smem,
        scratch_shapes=[
            pltpu.SMEM((num_experts,), jnp.int32),
            pltpu.SMEM((lcap + 1,), jnp.int32),
            pltpu.SMEM((ep,), jnp.int32),
            *extra,
        ],
        interpret=interpret,
        name="moe_replica_assign_fast",
    )(table, ids.reshape(-1))
    return out.reshape(num_tokens, topk)


def _reference_ranked(ids, table, *, num_experts, c, ep):
    """`_reference` on a flat table (ranks from the table)."""
    t = table.reshape(num_experts, c, 2)
    copies, ranks = t[:, :, 0], t[:, :, 1]
    valid = ids >= 0
    hit = jnp.zeros((num_experts,), jnp.bool_).at[jnp.maximum(ids, 0)].max(valid)
    native, nrank = copies[:, 0], ranks[:, 0]
    single = copies[:, 1] < 0
    load = jnp.zeros((ep,), jnp.int32).at[nrank].add(jnp.logical_and(hit, single).astype(jnp.int32))
    choice = jnp.where(jnp.logical_and(hit, single), native, _UNHIT)
    pos = jnp.arange(ids.shape[0], dtype=jnp.int32)
    first = jnp.full((num_experts,), ids.shape[0], jnp.int32).at[jnp.maximum(ids, 0)].min(
        jnp.where(valid, pos, ids.shape[0])
    )
    order = jnp.argsort(first, stable=True)

    def flexible(j, carry):
        load, choice = carry
        x = order[j]
        todo = jnp.logical_and(hit[x], jnp.logical_not(single[x]))
        best, br = native[x], nrank[x]
        best_load = load[br]
        for k in range(1, c):
            p, pr = copies[x, k], ranks[x, k]
            ok = p >= 0
            pl_load = load[jnp.maximum(pr, 0)]
            better = jnp.logical_and(ok, pl_load < best_load)
            best = jnp.where(better, p, best)
            br = jnp.where(better, pr, br)
            best_load = jnp.where(better, pl_load, best_load)
        load = jnp.where(todo, load.at[br].add(1), load)
        choice = jnp.where(todo, choice.at[x].set(best), choice)
        return load, choice

    _, choice = lax.fori_loop(0, num_experts, flexible, (load, choice))
    return jnp.where(valid, choice[jnp.maximum(ids, 0)], -1)


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
    # Experts in order of first appearance among the routes (as the kernel).
    pos = jnp.arange(ids.shape[0], dtype=jnp.int32)
    first = jnp.full((num_experts,), ids.shape[0], jnp.int32).at[
        jnp.maximum(ids, 0)
    ].min(jnp.where(valid, pos, ids.shape[0]))
    order = jnp.argsort(first, stable=True)

    def flexible(j, carry):
        load, choice = carry
        x = order[j]
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


@functools.partial(jax.jit, static_argnames=("ep", "slots", "refine", "interpret"))
def replica_assign(
    ids: jax.Array,
    copies: jax.Array,
    *,
    ep: int,
    slots: int,
    refine: int = 0,
    interpret: bool = False,
) -> jax.Array:
    """Physical slot ids ``[T, K]`` for logical routed ids ``[T, K]``.

    Negative ids (rows the grouped matmul skips) take no part in the
    assignment and come back as ``-1``.

    ``copies[e]`` lists the physical slot ids holding logical expert ``e``:
    column 0 is the native copy, further columns replicas, ``-1`` for none.
    Physical id ``p`` lives on rank ``p // slots``. Above ``MAX_ROUTES``
    routes the native copies are used (no replicas).

    refine > 0: after the greedy pass, up to that many rounds of moves that
    lower the busiest rank's load (`_refine`; TPU kernel only, the jnp
    reference keeps the greedy assignment).
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
    kernel = functools.partial(_kernel, n=n, c=c, ep=ep, slots=slots, refine=refine)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    lcap = min(n, num_experts)
    extra = (
        [
            pltpu.SMEM((ep * lcap,), jnp.int32),
            pltpu.SMEM((ep,), jnp.int32),
            pltpu.SMEM((2,), jnp.int32),
        ]
        if refine
        else []
    )
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n,), jnp.int32),
        in_specs=[smem, smem, smem],
        out_specs=smem,
        scratch_shapes=[
            pltpu.SMEM((min(n, num_experts),), jnp.int32),
            pltpu.SMEM((ep,), jnp.int32),
            *extra,
        ],
        interpret=interpret,
    )(ids.reshape(-1), copies.reshape(-1), jnp.full((num_experts,), _UNHIT, jnp.int32))
    return out.reshape(num_tokens, topk)
