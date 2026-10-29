# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The stage's routed experts: 64 per core plus copies, every route kept.

Per core (inside the stage's ``shard_map``):

1. The router's top 10 of each own row, renormalized; the core's routes
   sorted by expert (:func:`sorted_routes`). Rows (quantized per row to FP8)
   and the sorted routes are all-gathered, so every core sees the step's
   ``8n`` rows (``prologue.gather_and_share``, which runs the shared expert
   meanwhile).
2. Every core holds its 64 experts and copies of some of the previous
   core's (:func:`held_experts`); each step, the routes of the copied
   experts are split between the owner and the copy so that the busiest
   core's routes are as few as possible (:func:`balance_plan`).
3. The core's routes, grouped by expert, are runs of the gathered sorted
   routes (:func:`core_routes`), no sort needed. They run in one window, the
   smallest of :data:`WINDOWS` that holds the busiest core's routes (several
   windows of the largest past that): the window's rows are gathered, then
   one call (``experts.expert_products``) computes ``silu(gate) * up`` and
   the down projection from E2M1 weights requantized to FP8 in VMEM.
4. :func:`combine_scatter` reduce-scatters the tokens' sums back to the
   rows' owners, within each chip and then around a ring of chips
   (:mod:`ring`), combining each chunk where it is first added: each route's
   weighted output (BF16, as the reference rounds it) goes into its token's
   FP32 sum, kept as slabs (:data:`SLAB`). Windows after the first are summed
   by :func:`combine` first.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.qwen38.experts import (
    ROWS,
    SCALE_ALIGN,
    expert_products,
    per_row_fp8,
)
from vllm_torchtpu.megakernel.qwen38.prologue import gather_and_share
from vllm_torchtpu.megakernel.qwen38.reference import F32, Dims
from vllm_torchtpu.megakernel.qwen38.ring import RING_ID, Plan, Streams

COPIES = 32  # the previous core's experts each core also holds (every other one)
PLAN_TARGETS = 64  # busiest loads a step's balance plan tries at once
# An expert's fixed cost in experts.expert_products (its weight steps), in routes:
# the plan balances routes plus this per expert a core runs.
EXPERT_ROUTES = 280
# Window sizes, in shares of a core's mean routes: skewed routing takes a larger
# one instead of more windows.
WINDOWS = (1.25, 1.75, 2.5)
COMBINE_BLOCK = 64  # tokens per combine step
SCATTER_CHUNK = 32  # tokens per ring push of the sums: VMEM holds all their routes
SCATTER_GROUP = 4  # chunks on the wire per ring stream
META_ALIGN = 128  # a chunk's routes are copied to SMEM from this alignment
# A row as a [SLAB, H / SLAB] slab: whole (16, 128) BF16 / (8, 128) FP32 VMEM tiles.
SLAB = 16
ROW_SLAB = 32  # the same for FP8 rows: whole (32, 128) tiles
ROUTER_ROWS = 256  # rows per top-k step


def top_routes(logits, top_k: int, *, interpret=False):
    """Each row's ``top_k`` experts (lowest index first among equal logits)
    and their weights, the softmax over just those ``top_k`` (the full
    softmax's denominator cancels in the renormalization): ``[n, top_k]``
    each, from ``logits`` ``[n, E]`` FP32. One Pallas step per
    :data:`ROUTER_ROWS` rows: ``top_k`` rounds of a row maximum, its first
    index and a mask."""
    n, experts = logits.shape
    rows = min(ROUTER_ROWS, n)

    def kernel(logits_ref, ids_ref, weights_ref):
        x = logits_ref[...]
        expert = lax.broadcasted_iota(jnp.int32, x.shape, 1)
        lane = lax.broadcasted_iota(jnp.int32, ids_ref.shape, 1)
        ids = jnp.zeros(ids_ref.shape, jnp.int32)
        top = jnp.full(ids_ref.shape, -jnp.inf, F32)
        for k in range(top_k):
            best = jnp.max(x, axis=1, keepdims=True)
            first = jnp.min(
                jnp.where(x == best, expert, experts), axis=1, keepdims=True
            )
            ids = jnp.where(lane == k, first, ids)
            top = jnp.where(lane == k, best, top)
            x = jnp.where(expert == first, -jnp.inf, x)
        e = jnp.exp(top - top[:, :1])
        ids_ref[...] = ids
        weights_ref[...] = e / jnp.sum(e, axis=1, keepdims=True)

    ids, weights = pl.pallas_call(
        kernel,
        grid=(n // rows,),
        in_specs=[pl.BlockSpec((rows, experts), lambda i: (i, 0))],
        out_specs=[pl.BlockSpec((rows, 128), lambda i: (i, 0))] * 2,
        out_shape=(
            jax.ShapeDtypeStruct((n, 128), jnp.int32),
            jax.ShapeDtypeStruct((n, 128), F32),
        ),
        interpret=interpret,
        name="qwen38_top_routes",
    )(logits)
    return ids[:, :top_k], weights[:, :top_k]


def window(d: Dims, tokens: int, cores: int, share: float) -> int:
    """Routes per window: ``share`` of a core's mean share of the step's
    ``tokens * top_k`` routes, in whole 256-row tiles."""
    mean = tokens * d.top_k // cores
    return -(-int(mean * share) // 256) * 256


def held_experts(d: Dims, cores: int, core: int) -> list[int]:
    """The experts core ``core`` holds, in its weights' group order: its own
    ``E / cores``, then copies of every other one of the previous core's."""
    local = d.experts // cores
    previous = (core - 1) % cores
    return list(range(core * local, (core + 1) * local)) + [
        previous * local + j * (local // COPIES) for j in range(COPIES)
    ]


def balance_plan(counts, copies: int):
    """``[cores, copies]``: how many of each copied expert's routes core ``c``
    passes to core ``c + 1``, from the step's routes per expert ``counts``
    ``[cores, local]`` (copied experts at every ``local / copies``-th index),
    so the busiest core's load is as small as the copies allow. The same on
    every core. A core's load is its routes plus :data:`EXPERT_ROUTES` per
    expert it runs.

    A busiest load ``T`` is reachable iff going round the cycle (three times:
    the first core's inflow settles), each core can pass on its excess ``x_c
    = max(0, L_c + x_{c-1} - T)`` from its copied experts' loads;
    ``PLAN_TARGETS`` loads between the mean and the maximum are tried at once
    and the smallest reachable one is taken."""
    cores, local = counts.shape
    copied = counts[:, :: local // copies]

    def cost(routes):
        return routes + EXPERT_ROUTES * (routes > 0)

    load, cap = jnp.sum(cost(counts), axis=1), jnp.sum(cost(copied), axis=1)
    low = -(-jnp.sum(load) // cores)
    step = (jnp.max(load) - low) // (PLAN_TARGETS - 1) + 1
    target = low + step * jnp.arange(PLAN_TARGETS, dtype=jnp.int32)
    passed = [jnp.zeros_like(target)] * cores
    reachable = jnp.ones(target.shape, bool)
    for _ in range(3):
        for c in range(cores):
            passed[c] = jnp.maximum(load[c] + passed[c - 1] - target, 0)
            reachable &= passed[c] <= cap[c]
    best = jnp.argmax(reachable)  # the first: the largest target always is
    total = jnp.stack(passed)[:, best]  # [cores]
    # Whole copied experts first, in order; the last one split (the copy's
    # share of it pays the expert's fixed cost again).
    left = total[:, None] - (jnp.cumsum(cost(copied), axis=1) - cost(copied))
    return jnp.where(
        left >= cost(copied), copied, jnp.clip(left - EXPERT_ROUTES, 0, copied)
    )


def sorted_routes(ids, weights, scale, experts: int):
    """This core's routes sorted by expert, in token order within each, from
    its rows' ``top_k`` experts ``ids`` and their ``weights`` ``[n, top_k]``
    and the rows' FP8 scales ``scale`` ``[n]``: each route's index (``row *
    top_k + k``), weight and row scale ``[n * top_k]``, and the routes per
    expert ``[experts]``. Gathered from every core, they are what
    :func:`core_routes` splits."""
    n, top_k = ids.shape
    route = jnp.arange(n * top_k, dtype=jnp.int32)
    expert, route, weight, row_scale = lax.sort(
        (ids.reshape(-1), route, weights.reshape(-1), jnp.repeat(scale, top_k)),
        num_keys=1,
        is_stable=True,
    )
    bounds = jnp.searchsorted(
        expert, jnp.arange(experts + 1, dtype=jnp.int32), method="compare_all"
    )
    return route, weight, row_scale, jnp.diff(bounds).astype(jnp.int32)


def core_routes(counts, local: int, groups: int, me, per_core: int):
    """Core ``me``'s share of the step's routes, from every core's routes per
    expert ``counts`` ``[cores, E]`` (:func:`sorted_routes`, gathered
    core-major, ``per_core`` routes each). Group ``g`` takes its expert's
    routes in token order: each core's run of that expert, in core order; a
    copied expert's last routes run on the copy, as :func:`balance_plan`
    splits them. Returns the groups' runs (:func:`window_routes` reads them)
    as each run's first window position and its gathered index minus that
    position (``[groups * cores + 1]`` each, group-major, then one for the
    positions past the core's routes), ``sizes`` ``[groups]`` and
    ``busiest``, the most routes a core holds (the same on every core)."""
    cores, experts = counts.shape
    copies = groups - local
    stride = local // copies
    per_expert = jnp.sum(counts, axis=0)
    moved = balance_plan(per_expert.reshape(cores, local), copies)
    previous = (me - 1) % cores
    expert = jnp.concatenate(
        [
            me * local + jnp.arange(local, dtype=jnp.int32),
            previous * local + stride * jnp.arange(copies, dtype=jnp.int32),
        ]
    )
    own_moved = (
        jnp.zeros((local,), jnp.int32)
        .at[::stride]
        .set(lax.dynamic_index_in_dim(moved, me, keepdims=False))
    )
    count = per_expert[expert]
    copy_count = count[local:]
    lo = jnp.concatenate(
        [
            jnp.zeros((local,), jnp.int32),
            copy_count - lax.dynamic_index_in_dim(moved, previous, keepdims=False),
        ]
    )
    hi = jnp.concatenate([count[:local] - own_moved, copy_count])
    # Core c's run of a group's expert, as places in the expert's routes.
    runs = counts[:, expert]  # [cores, groups]
    first = jnp.cumsum(runs, axis=0) - runs
    begin = jnp.clip(lo[None], first, first + runs)
    length = jnp.clip(hi[None], first, first + runs) - begin
    places = jnp.cumsum(counts, axis=1) - counts  # each expert's run in its core's
    gathered = (
        jnp.arange(cores, dtype=jnp.int32)[:, None] * per_core
        + places[:, expert]
        + begin
        - first
    )
    length, gathered = length.T.reshape(-1), gathered.T.reshape(-1)
    start = jnp.cumsum(length) - length
    # Past the core's routes, a last run of offset zero: those positions read
    # distinct routes (one route read many times is slow to gather).
    total = jnp.sum(length, keepdims=True)
    start = jnp.concatenate([start, total])
    offset = jnp.concatenate([gathered - start[:-1], jnp.zeros((1,), jnp.int32)])
    passed = jnp.sum(moved, axis=1)
    busiest = jnp.max(
        jnp.sum(per_expert.reshape(cores, local), axis=1) - passed + jnp.roll(passed, 1)
    )
    return (start, offset), hi - lo, busiest


def window_routes(runs, start, size: int, routes: int):
    """The gathered indices ``[size]`` of the routes at window positions
    ``start, ..., start + size - 1`` from :func:`core_routes`' ``runs``
    (``routes`` gathered routes; past the core's routes each position's own
    index, clipped): each position plus the offset of the last run
    starting at or before it, as a cumulative sum of the offsets' changes."""
    first, offset = runs
    change = jnp.diff(offset, prepend=0)
    at = jnp.clip(first - start, 0, size)
    shift = jnp.cumsum(jnp.zeros((size + 1,), jnp.int32).at[at].add(change))[:size]
    return jnp.clip(start + jnp.arange(size, dtype=jnp.int32) + shift, 0, routes - 1)


def _route_window(chunk: int, top_k: int) -> int:
    """Words copied for a chunk's routes: all ``chunk * top_k`` of them from
    the :data:`META_ALIGN` boundary below the first."""
    return -(-(chunk * top_k + META_ALIGN) // 128) * 128


def chunk_routes(token, weight, total, span, tokens, chunk, top_k):
    """The window's routes in token order for :func:`combine_scatter`, from
    each sorted route's ``token`` and ``weight`` (``[>= span]``; past
    ``total``, other cores'): their tokens, places in the window and weights
    (``[span]`` padded by a route window), and each ``chunk`` of the
    ``tokens``' first entry and route count (``[tokens / chunk]``)."""
    position = jnp.arange(span, dtype=jnp.int32)
    key = jnp.where(position < total, token[:span], tokens)  # others' last
    key, position, weight = lax.sort(
        (key, position, weight[:span]), num_keys=1, is_stable=True
    )
    blocks = tokens // chunk
    live = jnp.bincount(key // chunk, length=blocks + 1)[:blocks].astype(jnp.int32)
    first = jnp.cumsum(live) - live
    padding = _route_window(chunk, top_k)
    padded = (jnp.pad(v, (0, padding)) for v in (key, position, weight))
    return (*padded, first.astype(jnp.int32), live)


def routed_experts(d: Dims, ring, axis: str, x, logits, expert_weights, shared_weights):
    """The routed experts' sum and the shared expert's output (BF16) for
    this core's rows ``x`` ``[n, H]`` (BF16, normalized) from their router
    ``logits`` ``[n, E]``. Core ``c`` of the cores along ``axis`` (``ring``:
    :func:`ring.ring_order`) holds :func:`held_experts` (groups), their
    ``expert_weights`` as :func:`experts.expert_products` takes them;
    ``shared_weights`` are the shared expert's (gate/up and down, as
    :func:`prologue.gather_and_share` takes them), computed while the rows
    are all-gathered. A copied expert's last routes (in token order) run on
    the copy, as :func:`balance_plan` splits them."""
    cores = len(ring)
    local = d.experts // cores
    groups = expert_weights[0].shape[0]
    ids, weights = top_routes(logits.astype(F32), d.top_k)
    # Rows quantized once per token (the expert GEMMs' FP8 operand) as
    # [ROW_SLAB, H / ROW_SLAB] slabs, whole FP8 tiles: half the all-gather.
    # Per-token metadata travels flat: [n, top_k] would pad to whole lane tiles.
    n = x.shape[0]
    xq, x_scale = per_row_fp8(x.astype(F32))
    slabs = xq.reshape(n, ROW_SLAB, d.hidden // ROW_SLAB)
    metadata = sorted_routes(ids, weights, x_scale.reshape(-1), d.experts)
    per_core = n * d.top_k
    gate_up, down = shared_weights
    x_all, (route_all, weight_all, scale_all, counts), shared = gather_and_share(
        slabs, metadata, x, gate_up, down, ring, axis
    )
    tokens = x_all.shape[0]
    routes = tokens * d.top_k
    spans = [window(d, tokens, cores, share) for share in WINDOWS]

    runs, sizes, busiest = core_routes(
        counts.reshape(cores, d.experts), local, groups, lax.axis_index(axis), per_core
    )
    total = jnp.sum(sizes)
    ends = jnp.cumsum(sizes)

    def window_of(start, size):
        """The routes at window positions ``[start, start + size)``: their
        global route indices (core-major), weights and row scales."""
        at = window_routes(runs, start, size, routes)
        return at // per_core * per_core + route_all[at], weight_all[at], scale_all[at]

    def products_from(start, span, route, row_scale):
        clipped = jnp.clip(ends, start, start + span)
        size = jnp.diff(clipped, prepend=start).astype(jnp.int32)
        rows = x_all[route[: span + ROWS] // d.top_k]
        return expert_products(
            rows, row_scale, clipped - size - start, size, expert_weights, span=span
        )

    def scatter(span, extra):
        route, weight, row_scale = window_of(jnp.int32(0), span + ROWS + SCALE_ALIGN)
        products = products_from(jnp.int32(0), span, route, row_scale)
        token = route // d.top_k
        routes = chunk_routes(
            token, weight, total, span, tokens, SCATTER_CHUNK, d.top_k
        )
        return combine_scatter(products, routes, extra, ring, axis, top_k=d.top_k)

    def several():
        span = spans[-1]
        # The later windows' combine takes each route's place in the order.
        order, weight, _ = window_of(jnp.int32(0), routes)
        place = jnp.arange(routes, dtype=jnp.int32)
        target = jnp.where(place < total, order, routes)
        position = jnp.full((routes,), -1, jnp.int32).at[target].set(place, mode="drop")
        weights = jnp.zeros((routes,), F32).at[target].set(weight, mode="drop")

        def one_window(start, partial):
            route, _, row_scale = window_of(start, span + ROWS + SCALE_ALIGN)
            products = products_from(start, span, route, row_scale)
            return combine(
                products,
                position.reshape(tokens, d.top_k),
                weights.reshape(tokens, d.top_k),
                partial,
                start,
            )

        _, extra = lax.while_loop(
            lambda carry: carry[0] < total,
            lambda carry: (carry[0] + span, one_window(*carry)),
            (jnp.int32(2 * span), one_window(jnp.int32(span), None)),
        )
        return scatter(span, extra)

    # Every core takes the same branch (its ring kernel must match its
    # neighbors'): the first window that holds the busiest core's routes.
    branch = jnp.searchsorted(jnp.asarray(spans), busiest)
    one = [functools.partial(scatter, span, None) for span in spans]
    routed = lax.switch(branch, [*one, several])
    return routed.reshape(-1, d.hidden), shared


def combine_scatter(
    products,
    routes,
    extra,
    ring,
    axis,
    *,
    top_k,
    chunk=SCATTER_CHUNK,
    group=SCATTER_GROUP,
    interpret=False,
):
    """The step's tokens' routed sums reduce-scattered to their owners along
    ``ring`` (:mod:`ring`): this core's rows ``[n, SLAB, H / SLAB]`` BF16,
    each the sum over the cores of ``extra[t]`` plus its routes' weighted
    rows of ``products`` ``[window, SLAB, H / SLAB]`` BF16 (sorted). The
    routes are :func:`chunk_routes`' (``chunk`` tokens, at most ``top_k``
    routes each); ``extra`` ``[tokens, SLAB, H / SLAB]`` FP32 the later
    windows' sums (:func:`combine`), or None. ``interpret`` runs the kernel on
    the host.

    Each ring iteration combines two chunks, the sibling's (pushed to it) and
    then its own: a chunk's routes reach SMEM two chunks ahead (a window
    from the :data:`META_ALIGN` boundary below its first) and their product
    rows VMEM one ahead, with room for every route of the chunk's tokens;
    each row is added, weighted, to its token's FP32 sum."""
    token_at, rows_at, weight_at, first, live = routes
    tokens = first.shape[0] * chunk
    _, slab, columns = products.shape
    plan = Plan(tuple(ring), tokens // len(ring), (slab, columns), chunk, group)
    slots, route_window = chunk * top_k, _route_window(chunk, top_k)

    def kernel(
        ring_ref,
        position_ref,
        first_ref,
        live_ref,
        token_hbm,
        rows_hbm,
        weight_hbm,
        products_hbm,
        *refs,
    ):
        later_hbm, out_hbm, *refs = refs if extra is not None else (None, *refs)
        rows_smem, token_smem, weight_smem, routed, sums, meta_sems, row_sems, *refs = (
            refs
        )
        # The later windows' sums: their HBM input, VMEM banks and semaphores.
        later = None if later_hbm is None else (later_hbm, *refs[:2])
        ring_scratch = refs if later is None else refs[2:]
        streams = Streams(plan, axis, ring_ref, position_ref, out_hbm, *ring_scratch)
        streams.enter()

        jobs = 2 * plan.iterations  # the sibling's chunk, then this core's

        def job_row(d, t):
            i = t // 2
            return jnp.where(
                t % 2 == 0, streams.sibling_row(d, i), streams.partial_row(d, i)
            )

        def chunk_of(d, t):
            return job_row(d, t) // chunk

        def window(d, t):
            """Chunk ``t``'s routes: the copied window's first word, and the
            offset of the chunk's first route in it."""
            start = first_ref[chunk_of(d, t)]
            base = pl.multiple_of(start // META_ALIGN * META_ALIGN, META_ALIGN)
            return base, start - base

        def metadata(d, t):
            words, m = pl.ds(window(d, t)[0], route_window), t % 3
            return [
                pltpu.make_async_copy(
                    source.at[words], target.at[d, m], meta_sems.at[d, f, m]
                )
                for f, (source, target) in enumerate(
                    (
                        (rows_hbm, rows_smem),
                        (token_hbm, token_smem),
                        (weight_hbm, weight_smem),
                    )
                )
            ]

        def later_copy(later, d, t):
            hbm, vmem, sems = later
            rows = pl.ds(job_row(d, t), chunk)
            return pltpu.make_async_copy(
                hbm.at[rows], vmem.at[d, t % 2], sems.at[d, t % 2]
            )

        def fetch(d, t):
            for copy in metadata(d, t):
                copy.start()

        def issue(d, t):
            """Chunk ``t``'s product rows (and later windows' sums) on their
            way, once its metadata has landed."""
            for copy in metadata(d, t):
                copy.wait()
            m, b = t % 3, t % 2
            offset = window(d, t)[1]

            @pl.loop(0, live_ref[chunk_of(d, t)])
            def _(r):
                pltpu.make_async_copy(
                    products_hbm.at[pl.ds(rows_smem[d, m, offset + r], 1)],
                    routed.at[d, b, pl.ds(r, 1)],
                    row_sems.at[d, b],
                ).start()

            if later is not None:
                later_copy(later, d, t).start()

        def combine_chunk(d, t):
            """Chunk ``t``'s FP32 sums into ``sums[d]``, the next chunks'
            inputs on their way."""
            pl.when(t + 2 < jobs)(lambda: fetch(d, t + 2))
            pl.when(t + 1 < jobs)(lambda: issue(d, t + 1))
            m, b = t % 3, t % 2
            landed = live_ref[chunk_of(d, t)]

            @pl.when(landed > 0)
            def _():
                arrived = routed.at[d, b, pl.ds(0, landed)]
                pltpu.make_async_copy(arrived, arrived, row_sems.at[d, b]).wait()

            if later is None:
                sums[d] = jnp.zeros(sums.shape[1:], F32)
            else:
                later_copy(later, d, t).wait()
                sums[d] = later[1][d, b]

            offset, row = window(d, t)[1], job_row(d, t)

            @pl.loop(0, landed)
            def _(r):
                token = token_smem[d, m, offset + r] - row
                route = weight_smem[d, m, offset + r] * routed[d, b, r].astype(F32)
                sums[d, token] = sums[d, token] + route

        def advance(d, i):
            combine_chunk(d, 2 * i)
            streams.give(d, i, sums.at[d])
            combine_chunk(d, 2 * i + 1)
            streams.finish(d, i, sums.at[d])

        for d in range(2):
            fetch(d, 0)
            fetch(d, 1)
            issue(d, 0)

        @pl.loop(0, plan.iterations)
        def _(i):
            for d in range(2):
                advance(d, i)

        streams.drain()

    rows = (slab, columns)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    extra_scratch = (
        []
        if extra is None
        else [
            pltpu.VMEM((2, 2, chunk, *rows), F32),
            pltpu.SemaphoreType.DMA((2, 2)),
        ]
    )
    row_bytes = slab * columns
    vmem = (
        4 * slots * 2 + 2 * chunk * 4 + (0 if extra is None else 4 * chunk * 4)
    ) * row_bytes
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((plan.rows, *rows), products.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(1,),
            in_specs=[hbm] * (4 if extra is None else 5),
            out_specs=hbm,
            scratch_shapes=[
                pltpu.SMEM((2, 3, route_window), jnp.int32),
                pltpu.SMEM((2, 3, route_window), jnp.int32),
                pltpu.SMEM((2, 3, route_window), F32),
                pltpu.VMEM((2, 2, slots, *rows), products.dtype),  # route rows
                pltpu.VMEM((2, chunk, *rows), F32),  # each stream's chunk sums
                pltpu.SemaphoreType.DMA((2, 3, 3)),
                pltpu.SemaphoreType.DMA((2, 2)),
                *extra_scratch,
                *plan.scratch_shapes(),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=RING_ID,
            vmem_limit_bytes=vmem + plan.vmem_bytes() + 2**24,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"qwen38_moe_combine_scatter_c{chunk}",
    )(
        *plan.prefetch(),
        first,
        live,
        token_at,
        rows_at,
        weight_at,
        products,
        *([] if extra is None else [extra]),
    )


def combine(
    products, position, weights, partial, start, *, block=COMBINE_BLOCK, interpret=False
):
    """``partial[t] + sum_k weights[t, k] * products[position[t, k] - start]``
    over the routes ``(t, k)`` whose sorted position falls in the window
    ``[start, start + len(products))``; ``products`` ``[window, SLAB, H /
    SLAB]`` BF16, ``position`` / ``weights`` ``[tokens, top_k]``, ``partial``
    ``[tokens, SLAB, H / SLAB]`` FP32 (updated in place), or None to start
    from zero.
    ``interpret`` runs the kernel on the host, for tests without a TPU.

    A step handles ``block`` tokens. Their windowed routes, compacted
    (``_routes``), reach SMEM two steps ahead; their product rows are DMAed
    one step ahead; the step adds each route's weighted row to its token. A
    row travels and sums as a ``[SLAB, H / SLAB]`` slab: any number of rows
    is a whole number of VMEM tiles."""
    tokens, top_k = weights.shape
    span, _, columns = products.shape
    hidden = SLAB * columns
    slots = block * top_k
    steps = tokens // block
    assert tokens % block == 0 and columns % 128 == 0
    assert partial is None or partial.shape == (tokens, SLAB, columns)
    rows_at, token_at, weight_at, live = _routes(position, weights, start, span, block)
    stride = rows_at.shape[0] // steps

    def kernel(live_ref, rows_hbm, token_hbm, weight_hbm, products_hbm, *refs):
        partial_ref, refs = (None, refs) if partial is None else (refs[0], refs[1:])
        out_ref, rows_smem, token_smem, weight_smem, rows, meta_sem, row_sem = refs
        step = pl.program_id(0)

        def metadata(block_id):
            m = block_id % 3
            words = pl.ds(block_id * stride, stride)
            return [
                pltpu.make_async_copy(source.at[words], target.at[m], meta_sem.at[i, m])
                for i, (source, target) in enumerate(
                    (
                        (rows_hbm, rows_smem),
                        (token_hbm, token_smem),
                        (weight_hbm, weight_smem),
                    )
                )
            ]

        def fetch(block_id):
            for copy in metadata(block_id):
                copy.start()

        def issue(block_id):
            for copy in metadata(block_id):
                copy.wait()
            m, b = block_id % 3, block_id % 2

            @pl.loop(0, live_ref[block_id])
            def _(i):
                pltpu.make_async_copy(
                    products_hbm.at[pl.ds(rows_smem[m, i], 1)],
                    rows.at[b, pl.ds(i, 1)],
                    row_sem.at[b],
                ).start()

        @pl.when(step == 0)
        def _():
            fetch(0)
            if steps > 1:
                fetch(1)
            issue(0)

        @pl.when(step + 2 < steps)
        def _():
            fetch(step + 2)

        @pl.when(step + 1 < steps)
        def _():
            issue(step + 1)

        m, b = step % 3, step % 2
        landed = live_ref[step]

        @pl.when(landed > 0)
        def _():
            arrived = rows.at[b, pl.ds(0, landed)]
            pltpu.make_async_copy(arrived, arrived, row_sem.at[b]).wait()

        out_ref[...] = (
            jnp.zeros_like(out_ref) if partial_ref is None else partial_ref[...]
        )

        @pl.loop(0, landed)
        def _(i):
            t = token_smem[m, i]
            out_ref[t] = out_ref[t] + weight_smem[m, i] * rows[b, i].astype(F32)

    sums = pl.BlockSpec((block, SLAB, columns), lambda s, *_: (s, 0, 0))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tokens, SLAB, columns), F32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(steps,),
            in_specs=[hbm, hbm, hbm, hbm] + ([] if partial is None else [sums]),
            out_specs=sums,
            scratch_shapes=[
                pltpu.SMEM((3, stride), jnp.int32),
                pltpu.SMEM((3, stride), jnp.int32),
                pltpu.SMEM((3, stride), F32),
                pltpu.VMEM((2, slots, SLAB, columns), products.dtype),
                pltpu.SemaphoreType.DMA((3, 3)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        input_output_aliases={} if partial is None else {5: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            # Both row banks, the double-buffered partial and output blocks.
            vmem_limit_bytes=(2 * slots * products.dtype.itemsize + 16 * block) * hidden
            + 2**22,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"qwen38_moe_combine_b{block}",
    )(
        live,
        rows_at,
        token_at,
        weight_at,
        products,
        *([] if partial is None else [partial]),
    )
    return out


def _routes(position, weights, start, span, block):
    """Each ``block`` of tokens' routes in the window, compacted in token
    order: their product rows, tokens (within the block) and weights
    (``[blocks * stride]``, ``stride`` = ``block * top_k`` in whole 128-word
    tiles), and each block's count."""
    tokens, top_k = position.shape
    steps = tokens // block
    slots = block * top_k
    stride = -(-slots // 128) * 128
    local = position - start
    alive = ((position >= 0) & (local >= 0) & (local < span)).reshape(steps, slots)
    token = jnp.broadcast_to(
        jnp.arange(block, dtype=jnp.int32)[:, None], (steps, block, top_k)
    )
    # The values ride the sort as payloads (a gather by the order is slow).
    _, *values = lax.sort(
        tuple(
            v.reshape(steps, slots)
            for v in (~alive, local.astype(jnp.int32), token, weights.astype(F32))
        ),
        dimension=1,
        is_stable=True,
        num_keys=1,
    )
    return (
        *(jnp.pad(v, ((0, 0), (0, stride - slots))).reshape(-1) for v in values),
        jnp.sum(alive, axis=1, dtype=jnp.int32),
    )
