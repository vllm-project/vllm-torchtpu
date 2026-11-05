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
"""Kimi-K3.1's prefill top-k selection on a stage core as one Pallas kernel
(:func:`select_words`): a core's window rows score their own sequence's
cached tokens and window tokens, select their ``index_topk`` best and pack
the selection as the flash kernel's words, with work bounded by each
sequence's real context, not the arena's capacity.

A grid step takes ``ROWS`` consecutive window rows (one sequence, or
padding past it). Scores go to a VMEM buffer as order-preserving int32 keys
(``key ^ INT32_MIN`` of ``dsa_prefill.order_keys``; an invalid candidate is
``INT32_MIN``) in compact columns: the arena columns ``[a0, a1)`` around the
sequence's cached range (``GROUP``-aligned), then every window column. A
score is ``2^(e - 127) * sum_h w_h ReLU(q_h . k)`` with the key's FP8 codes
straight from its gathered index page (the power-of-two scale commutes with
every rounding) and the heads summed as the decoder sums them
(``dsa_decode._head_sum``). The threshold is the bisection of
``dsa_prefill.topk_threshold`` over the compact columns; ties at it go to the
lower column (a lane prefix sum per group); each ``GROUP`` columns' bits pack
into 128 words by one MXU product with a power-of-two matrix (16 bits per
product column, exact in FP32). The step writes its rows' words for every
arena column below the window's cached total (zero outside its own range)
and for every window column; words above the cached total stay unwritten
(no sequence's attention reads them).
"""

from __future__ import annotations

import contextlib
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.megakernel.kimi_k3.dsa import INDEX_DIM, INDEX_HEADS
from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

ROWS = 8  # window rows a grid step selects for: one sublane tile
LANES = 128
GROUP = 4096  # columns one packing product covers: 128 words
TILE = 512  # columns a bisection loop step reads
RING = 32  # index pages in flight
ZERO_WORDS = 2048  # a zero tile's words: the copies that write the arena words a step's stage leaves out
INT32_MIN = -(2**31)


def pack_matrix() -> np.ndarray:
    """``[LANES, LANES]``: a 128-column chunk's bits (lane ``l``, word ``l //
    32``, bit ``l % 32``) times their powers of two, summed per word: bits
    ``0..15`` of word ``m`` into column ``m``, bits ``16..31`` into column
    ``LANES / 2 + m``."""
    matrix = np.zeros((LANES, LANES), np.float32)
    lane = np.arange(LANES)
    word, bit = lane // 32, lane % 32
    low = bit < 16
    matrix[lane[low], word[low]] = 2.0 ** bit[low]
    matrix[lane[~low], LANES // 2 + word[~low]] = 2.0 ** (bit[~low] - 16)
    return matrix


def _butterfly_rows(partials):
    """``[8, LANES]``: row ``r``'s sum over the 8 sublanes of ``partials[r]``
    (8 ``[8, LANES]`` values), each in the order of ``x + roll(x, 1)``,
    ``+ roll(., 2)``, ``+ roll(., 4)`` read at sublane ``r``, for all 8 rows at
    once: the rolls of rows whose read sublanes differ in parity (then in
    residue mod 4) share one value."""
    sublane = jax.lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
    single = [x + pltpu.roll(x, 1, 0) for x in partials]
    pairs = [jnp.where(sublane % 2 == 0, single[2 * p], single[2 * p + 1]) for p in range(4)]
    pairs = [x + pltpu.roll(x, 2, 0) for x in pairs]
    quads = [jnp.where((sublane % 4) // 2 == 0, pairs[2 * q], pairs[2 * q + 1]) for q in range(2)]
    quads = [x + pltpu.roll(x, 4, 0) for x in quads]
    return jnp.where(sublane < 4, quads[0], quads[1])


def _tree(values):
    """The pairwise sum of ``values`` (a power-of-two count): ``(v0 + v1) +
    (v2 + v3)``, and so on up."""
    while len(values) > 1:
        values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
    return values[0]


def _head_major_sums(logits, weights, block, packed):
    """Row sums of ``w_h ReLU(logit_h)`` from head-major logits (head ``h``'s
    ``[block, LANES]`` at rows ``h * block ..``), the heads summed in a pairwise
    tree: FP32 (``weights`` the same layout), or ``packed``: SM100-class BF16
    (each head's ReLU'd logit and its weight in BF16, heads ``p`` and ``p + 16``
    sharing a packed ``[2 block, LANES]`` tile, ``weights`` in that pair layout,
    the products and their sums in BF16, the two halves added in BF16)."""
    if not packed:
        return _tree([jnp.maximum(logits[h * block:(h + 1) * block], 0.0) * weights[h * block:(h + 1) * block]
                      for h in range(INDEX_HEADS)])
    half = INDEX_HEADS // 2
    terms = []
    for p in range(half):
        pair = jnp.concatenate((logits[p * block:(p + 1) * block], logits[(p + half) * block:(p + half + 1) * block]))
        pair = jnp.maximum(pair.astype(jnp.bfloat16), jnp.bfloat16(0))
        terms.append(pair * weights[2 * p * block:2 * (p + 1) * block])
    total = _tree(terms).astype(jnp.float32)
    return (total[:block] + total[block:]).astype(jnp.bfloat16).astype(jnp.float32)


def _head_sums(queries, weights, keys, block=ROWS, skip=(), head_matrix=None, butterfly=False):
    """``[block, LANES]`` FP32: row ``r``'s ``sum_h w_h ReLU(q_h . k)`` over the
    128 keys ``keys``, ``queries``/``weights`` ``[block * 32, ...]`` (row
    ``r``'s heads at rows ``32 r ..``), summed as ``dsa_decode._head_sum``;
    with ``head_matrix`` (``[block, block * 32]``, row ``r``'s head weights at
    its own 32 columns) as one FP32 product over the ReLU logits instead."""
    return _reduce_heads(_logits(queries, keys, skip), weights, block, skip, head_matrix, butterfly)


def _logits(queries, keys, skip=()):
    """``[block * 32, LANES]`` FP32: every row's and head's ``q_h . k``."""
    if "fp8" in skip:
        keys = keys.astype(jnp.bfloat16)
    return jax.lax.dot_general(queries, keys, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _reduce_heads(logits, weights, block=ROWS, skip=(), head_matrix=None, butterfly=False, order="rows"):
    """:func:`_head_sums` from its :func:`_logits`; ``order`` ``major`` or
    ``bf16``: :func:`_head_major_sums` (head-major logits)."""
    if "heads" in skip:
        return logits[:block]
    if order != "rows":
        return _head_major_sums(logits, weights, block, order == "bf16")
    if head_matrix is not None:
        return jax.lax.dot_general(head_matrix, jnp.maximum(logits, 0.0), (((1,), (0,)), ((), ())),
                                   precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if butterfly:
        groups = []
        for g in range(block // 8):
            partials = []
            for r in range(8 * g, 8 * g + 8):
                base = INDEX_HEADS * r
                terms = [jnp.maximum(logits[base + 8 * t:base + 8 * t + 8], 0.0)
                         * weights[base + 8 * t:base + 8 * t + 8] for t in range(INDEX_HEADS // 8)]
                partials.append((terms[0] + terms[1]) + (terms[2] + terms[3]))
            groups.append(_butterfly_rows(partials))
        return groups[0] if len(groups) == 1 else jnp.concatenate(groups, axis=0)
    sublane = jax.lax.broadcasted_iota(jnp.int32, (block, LANES), 0)
    out = jnp.zeros((block, LANES), jnp.float32)
    for r in range(block):
        base = INDEX_HEADS * r
        terms = [jnp.maximum(logits[base + 8 * t:base + 8 * t + 8], 0.0) * weights[base + 8 * t:base + 8 * t + 8]
                 for t in range(INDEX_HEADS // 8)]
        total = (terms[0] + terms[1]) + (terms[2] + terms[3])
        for shift in (1, 2, 4):
            total = total + pltpu.roll(total, shift, 0)
        if block > 8:  # every sublane holds the row's sum: one copy per 8 rows
            total = jnp.concatenate([total] * (block // 8), axis=0)
        out = jnp.where(sublane == r, total, out)
    return out


def _compare_exchange(values, high, low):
    """Order ``values[high] >= values[low]`` elementwise (in place)."""
    a, b = values[high], values[low]
    values[high], values[low] = jnp.maximum(a, b), jnp.minimum(a, b)


def _bitonic_sort(values):
    """``values`` (a power-of-two list of equal-shape arrays) sorted
    elementwise in descending order: a bitonic sorting network."""
    values, n, size = list(values), len(values), 2
    while size <= n:
        span = size // 2
        while span:
            for i in range(n):
                partner = i ^ span
                if partner > i:
                    if i & size:
                        _compare_exchange(values, partner, i)
                    else:
                        _compare_exchange(values, i, partner)
            span //= 2
        size *= 2
    return values


def _merge_top(kept, incoming):
    """The elementwise top ``len(kept)`` of two descending lists, descending:
    a bitonic half-cleaner, then a bitonic merge."""
    n = len(kept)
    values = [jnp.maximum(kept[i], incoming[n - 1 - i]) for i in range(n)]
    span = n // 2
    while span:
        for i in range(n):
            if i ^ span > i:
                _compare_exchange(values, i, i ^ span)
        span //= 2
    return values


NEG_INF_KEY = -0x7F800001  # the order key of -inf


def _key_floats(keys):
    """FP32 values ordered as int32 order ``keys`` (one op per max or min,
    against two for int32), invalid keys as -inf."""
    bits = jnp.maximum(keys, NEG_INF_KEY)
    return pltpu.bitcast(jnp.where(bits < 0, bits ^ 0x7FFFFFFF, bits), jnp.float32)


def _float_keys(values):
    """The order keys of :func:`_key_floats` values (-inf invalid again)."""
    bits = pltpu.bitcast(values, jnp.int32)
    keys = jnp.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return jnp.where(keys == NEG_INF_KEY, INT32_MIN, keys)


def _order_keys(scores):
    """int32 keys ordered as FP32 ``scores`` (-0 as +0)."""
    bits = pltpu.bitcast(scores, jnp.int32)
    bits = jnp.where(bits == INT32_MIN, 0, bits)
    return jnp.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


def _phase(name, on):
    """Open an XProf scope named ``name`` over the kernel code that follows
    (``KernelConfig.phase_scopes``); close it with ``__exit__``."""
    scope = jax.named_scope(name) if on else contextlib.nullcontext()
    scope.__enter__()
    return scope


def _kernel(order_ref, segment_ref, length_ref, position_ref, q_ref, w_ref, hm_ref, window_ref, window_scale_ref,
            pack_ref, pages_hbm, out_hbm, scores, maxima, pages, page_sems, stage, out_sems, flag_vmem, flag_smem,
            flag_sem, staged_even, staged_odd, scales_even, scales_odd, thresholds, zeros, *, count, tokens, capacity,
            page_tokens, topk, block=ROWS, fp8=False, mxu=False, narrow=False, butterfly=False, per_lane=False, skip=(),
            scopes=False, order="rows", score_capacity=None):
    score_capacity = capacity if score_capacity is None else score_capacity
    b = pl.program_id(0)
    first = position_ref[b * block]
    length = length_ref[0]
    offset = base = cached = end = jnp.int32(0)
    for s in range(count):
        start, size = segment_ref[s], segment_ref[count + s]
        inside = (start <= first) & (first < start + size)
        offset = jnp.where(inside, start, offset)
        end = jnp.where(inside, start + size, end)
        base = jnp.where(inside, segment_ref[3 * count + s], base)
        cached = jnp.where(inside, segment_ref[4 * count + s], cached)
    total = segment_ref[4 * count - 1] + segment_ref[5 * count - 1]  # the last slot's base + cached
    total_words = (total + GROUP - 1) // GROUP * (GROUP // 32)
    stop = jnp.minimum(end, length)  # rows from here on are padding
    block_live = first < stop
    cached = jnp.where(block_live, cached, 0)
    a0 = jnp.where(cached > 0, base // GROUP * GROUP, 0)
    a1 = jnp.where(cached > 0, (base + cached + GROUP - 1) // GROUP * GROUP, 0)
    base = jnp.where(cached > 0, base, 0)
    arena = a1 - a0
    window_columns = -(-tokens // GROUP) * GROUP  # the window region, padded to whole groups
    columns = arena + window_columns
    rows = first + jax.lax.broadcasted_iota(jnp.int32, (block, 1), 0)
    live = rows < stop
    lane = jax.lax.broadcasted_iota(jnp.int32, (block, LANES), 1)
    queries = q_ref[...].reshape(block * INDEX_HEADS, INDEX_DIM)
    weights = w_ref[...]
    head_matrix = hm_ref[...] if mxu else None
    invalid = jnp.full((block, LANES), INT32_MIN, jnp.int32)

    def head_sums(keys):
        return _reduce_heads(_logits(queries, keys, skip), weights, block, skip, head_matrix, butterfly, order)

    # The arena edges around the cached range, and every window column, start invalid.
    @pl.loop(0, (base - a0) // LANES)
    def _(c):
        scores[:, pl.ds(pl.multiple_of(c * LANES, LANES), LANES)] = invalid

    @pl.loop((base + cached - a0) // LANES, arena // LANES)
    def _(c):
        scores[:, pl.ds(pl.multiple_of(c * LANES, LANES), LANES)] = invalid

    @pl.loop(0, window_columns // LANES)
    def _(c):
        scores[:, pl.ds(pl.multiple_of(arena + c * LANES, LANES), LANES)] = invalid

    phase = _phase("select_score", scopes)

    # The cached tokens, page by page from the gathered index pages.
    count_pages = jnp.where("score" in skip, 0, cached // page_tokens)

    def copy(j, slot):
        page = order_ref[1 + base // page_tokens + j]
        return pltpu.make_async_copy(pages_hbm.at[page], pages.at[slot], page_sems.at[slot])

    dma = "dma" not in skip

    @pl.loop(0, jnp.minimum(count_pages, RING) if dma else 0)
    def _(j):
        copy(j, j).start()

    chunks_per_page = page_tokens // LANES
    staged, staged_scales = (staged_even, staged_odd), (scales_even, scales_odd)

    def product(p, buf, c):
        """Stage chunks ``c, c + 1`` of page ``p`` (one product): their
        logits and key scales in buffer ``buf``."""
        slot = p % RING
        keys = pltpu.bitcast(pages[slot, pl.ds(LANES * c, 2 * LANES), :], jnp.float8_e4m3fn)
        logits = _logits(queries, keys, skip)
        for d in range(2):
            exponents = pages[slot, pl.ds(page_tokens + c + d, 1), :].astype(jnp.int32)
            staged[buf][c + d] = logits[:, d * LANES:(d + 1) * LANES]
            staged_scales[buf][c + d] = pltpu.bitcast(exponents << 23, jnp.float32)

    def reduce(p, buf, c):
        """Chunk ``c`` of page ``p``'s scores from buffer ``buf``."""
        sums = _reduce_heads(staged[buf][c], weights, block, skip, head_matrix, butterfly, order)
        value = _order_keys(sums * staged_scales[buf][c])
        column = base - a0 + p * page_tokens + LANES * c
        scores[:, pl.ds(pl.multiple_of(column, LANES), LANES)] = jnp.where(live, value, INT32_MIN)

    def products(p, buf):
        for c in range(0, chunks_per_page, 2):
            product(p, buf, c)

    def overlapped(p, buf, q, next_buf):
        """Page ``p``'s scores and page ``q``'s logits, chunk pair by pair."""
        for c in range(0, chunks_per_page, 2):
            reduce(p, buf, c)
            reduce(p, buf, c + 1)
            product(q, next_buf, c)

    def fetch(p):
        @pl.when(dma & (p + RING < count_pages))
        def _():
            copy(p + RING, p % RING).start()

    def wait(p):
        if dma:  # a wait needs only the slot's semaphore and the page's size
            slot = p % RING
            pltpu.make_async_copy(pages_hbm.at[0], pages.at[slot], page_sems.at[slot]).wait()

    # Software-pipelined by hand: each step reduces a page on the VPU while
    # the MXU produces the next one's logits (two staging buffers).
    @pl.when(count_pages > 0)
    def _():
        wait(0)
        products(0, 0)
        fetch(0)

    pairs = jnp.maximum(count_pages - 1, 0) // 2

    @pl.loop(0, pairs)
    def _(i):
        p = 2 * i
        wait(p + 1)
        wait(p + 2)
        overlapped(p, 0, p + 1, 1)
        overlapped(p + 1, 1, p + 2, 0)
        fetch(p + 1)
        fetch(p + 2)

    tail = 2 * pairs

    @pl.when(count_pages - tail == 2)
    def _():
        wait(tail + 1)
        overlapped(tail, 0, tail + 1, 1)
        for c in range(chunks_per_page):
            reduce(tail + 1, 1, c)

    @pl.when(count_pages - tail == 1)
    def _():
        for c in range(chunks_per_page):
            reduce(tail, 0, c)

    # The window tokens of the sequence up to the block's last row.
    last = jnp.minimum(first + block, stop) - 1

    @pl.loop(offset // LANES, jnp.where(block_live & ("score" not in skip), last // LANES + 1, offset // LANES))
    def _(c):
        keys = window_ref[pl.ds(pl.multiple_of(c * LANES, LANES), LANES), :]
        if fp8:  # the FP8 codes, then each key's power-of-two scale
            keys = pltpu.bitcast(keys, jnp.float8_e4m3fn)
            scale = pltpu.bitcast(window_scale_ref[pl.ds(c, 1), :] << 23, jnp.float32)
            value = _order_keys(head_sums(keys) * scale)
        else:
            value = _order_keys(head_sums(keys))
        column = c * LANES + lane
        ok = live & (column >= offset) & (column <= rows)
        scores[:, pl.ds(pl.multiple_of(arena + c * LANES, LANES), LANES)] = jnp.where(ok, value, INT32_MIN)

    phase.__exit__(None, None, None)
    phase = _phase("select_threshold", scopes)

    # The k-th largest key: bisection over the key bits (unsigned domain).
    groups = jnp.where("bisect" in skip, 0, columns // GROUP)

    def count_where(test):
        """Per row, the columns whose key passes ``test``: a sweep of
        ``GROUP`` columns a loop step."""
        def step(g, acc):
            for t in range(GROUP // TILE):
                block = scores[:, pl.ds(pl.multiple_of(g * GROUP + t * TILE, TILE), TILE)]
                acc = acc + test(block).astype(jnp.int32)
            return acc

        acc = jax.lax.fori_loop(0, groups, step, jnp.zeros((block, TILE), jnp.int32))
        return jnp.sum(acc, axis=1, keepdims=True)

    def count_at_least(threshold):
        return count_where(lambda tile: tile >= threshold)

    def bisect(i, tau):
        candidate = tau | (jnp.int32(1) << (31 - i))
        return jnp.where(count_at_least(candidate ^ INT32_MIN) >= topk, candidate, tau)

    def to_scalar(value):
        """A ``[block, 1]`` int32's minimum as a scalar (through SMEM)."""
        flag_vmem[...] = jnp.broadcast_to(jnp.min(value, axis=0, keepdims=True), (block, LANES))
        flag_copy = pltpu.make_async_copy(flag_vmem, flag_smem, flag_sem.at[0])
        flag_copy.start()
        flag_copy.wait()
        return flag_smem[0, 0]

    def exact_threshold():
        """``(tau, quota, ties)`` from every column: bisection (narrowed)."""
        start, tau = jnp.int32(0), jnp.zeros((block, 1), jnp.int32)
        if narrow:
            start, tau = narrow_bounds()
        tau = jax.lax.fori_loop(start, 32, bisect, tau) ^ INT32_MIN
        quota = topk - count_where(lambda tile: tile > tau)
        ties = count_where(lambda tile: (tile == tau) & (tile > INT32_MIN))
        return tau, quota, ties

    def narrow_bounds():
        # Each group's lane maxima (its 32 columns l, l + 128, ...): the k-th
        # largest bounds the threshold below (k columns at least as large),
        # the row's largest above, so their common bits are the threshold's.
        @pl.loop(0, groups)
        def _(g):
            top = scores[:, pl.ds(pl.multiple_of(g * GROUP, LANES), LANES)]
            for t in range(1, GROUP // LANES):
                top = jnp.maximum(top, scores[:, pl.ds(pl.multiple_of(g * GROUP + t * LANES, LANES), LANES)])
            maxima[:, pl.ds(pl.multiple_of(g * LANES, LANES), LANES)] = top

        def count_maxima(test):
            def step(g, acc):
                return acc + test(maxima[:, pl.ds(pl.multiple_of(g * LANES, LANES), LANES)]).astype(jnp.int32)
            acc = jax.lax.fori_loop(0, groups, step, jnp.zeros((block, LANES), jnp.int32))
            return jnp.sum(acc, axis=1, keepdims=True)

        def bisect_maxima(i, bound):
            candidate = bound | (jnp.int32(1) << (31 - i))
            enough = count_maxima(lambda tile: tile >= (candidate ^ INT32_MIN)) >= topk
            return jnp.where(enough, candidate, bound)

        lower = jax.lax.fori_loop(0, 32, bisect_maxima, jnp.zeros((block, 1), jnp.int32))
        known = count_maxima(lambda tile: tile > INT32_MIN) >= topk
        lower = jnp.where(known, lower, 0)  # unsigned domain; no bound: 0
        def row_max(g, best):
            tile = maxima[:, pl.ds(pl.multiple_of(g * LANES, LANES), LANES)]
            return jnp.maximum(best, jnp.max(tile, axis=1, keepdims=True))

        upper = jax.lax.fori_loop(0, groups, row_max, jnp.full((block, 1), INT32_MIN, jnp.int32)) ^ INT32_MIN
        differ = lower ^ upper
        common = jnp.zeros((block, 1), jnp.int32)
        for i in range(32):  # leading bits where the bounds agree
            common = common + (jax.lax.shift_right_logical(differ, jnp.int32(31 - i)) == 0).astype(jnp.int32)
        common = jnp.where(live, common, 32)
        keep = jnp.where(common == 0, 0, jnp.int32(-1) << jnp.minimum(32 - common, 31))
        keep = jnp.where(common >= 32, -1, keep)
        return to_scalar(common), upper & keep

    if per_lane:
        # Each lane's 32 largest keys per row (every group's 32 columns of the
        # lane sorted, then merged in). Their k-th largest is the threshold
        # unless some lane dropped a valid key at or above it (its 32nd kept
        # key valid and not below it); then the block bisects every column.
        def merge_rows(lo):
            """Rows ``lo .. lo + 8``' kept keys (8 rows a pass: 32 kept vregs)."""
            def merge_group(g, kept):
                incoming = [_key_floats(scores[pl.ds(lo, ROWS), pl.ds(pl.multiple_of(g * GROUP + t * LANES, LANES),
                                                                       LANES)])
                            for t in range(GROUP // LANES)]
                return tuple(_merge_top(list(kept), _bitonic_sort(incoming)))

            empty = jnp.full((ROWS, LANES), -jnp.inf, jnp.float32)
            return [_float_keys(x) for x in jax.lax.fori_loop(0, groups, merge_group, (empty,) * (GROUP // LANES))]

        halves = [merge_rows(lo) for lo in range(0, block, ROWS)]
        kept = [jnp.concatenate([half[i] for half in halves]) if len(halves) > 1 else halves[0][i]
                for i in range(GROUP // LANES)]

        def count_kept(test):
            acc = sum(test(value).astype(jnp.int32) for value in kept)
            return jnp.sum(acc, axis=1, keepdims=True)

        def bisect_kept(i, bound):
            candidate = bound | (jnp.int32(1) << (31 - i))
            enough = count_kept(lambda value: value >= (candidate ^ INT32_MIN)) >= topk
            return jnp.where(enough, candidate, bound)

        tau = jax.lax.fori_loop(0, 32, bisect_kept, jnp.zeros((block, 1), jnp.int32)) ^ INT32_MIN
        quota = topk - count_kept(lambda value: value > tau)
        ties = count_kept(lambda value: (value == tau) & (value > INT32_MIN))
        dropped = jnp.max(((kept[-1] >= tau) & (kept[-1] > INT32_MIN)).astype(jnp.int32), axis=1, keepdims=True)
        overflow = (to_scalar(-dropped) < 0) & ("fallback" not in skip)

        # Through VMEM, so the fallback stays a branch (a cond over vectors
        # whose branches fold becomes a select Mosaic cannot lower).
        def keep(values):
            for i, value in enumerate(values):
                thresholds[i] = jnp.broadcast_to(value, (block, LANES))

        keep((tau, quota, ties))

        @pl.when(overflow)
        def _():
            keep(exact_threshold())

        tau, quota, ties = (thresholds[i][:, :1] for i in range(3))
    else:
        tau, quota, ties = exact_threshold()
    # Whether some row's ties overflow its quota (then ties are ranked by
    # column), as a scalar: through SMEM.
    ranked = to_scalar(-(ties > quota).astype(jnp.int32)) < 0

    phase.__exit__(None, None, None)
    phase = _phase("select_pack", scopes)

    # Zero this block's words of the window's arena outside its own range.
    zero = jnp.zeros((block, LANES), jnp.int32)

    @pl.loop(0, a0 // GROUP)
    def _(g):
        stage[:, pl.ds(pl.multiple_of(g * LANES, LANES), LANES)] = zero

    @pl.loop(a1 // GROUP, total_words // LANES)
    def _(g):
        stage[:, pl.ds(pl.multiple_of(g * LANES, LANES), LANES)] = zero

    # The stage's arena words above the window's cached total, which no block writes: once.
    @pl.when(b == 0)
    def _():
        @pl.loop(total_words // LANES, score_capacity // GROUP)
        def _(g):
            stage[:, pl.ds(pl.multiple_of(g * LANES, LANES), LANES)] = zero

        if score_capacity < capacity:
            zeros[...] = jnp.zeros(zeros.shape, jnp.int32)

    # Bits, ties to the lower column, packed per group.
    group_lane = jax.lax.broadcasted_iota(jnp.int32, (block, GROUP), 1)
    word_lane = jax.lax.broadcasted_iota(jnp.int32, (block, LANES), 1)
    matrix = pack_ref[...]
    chunks = GROUP // LANES

    def pack(g, bits):
        """Store group ``g``'s words: every chunk's 4 words (as two 16-bit
        halves) in one product of the chunks stacked over sublanes, each
        chunk's then rolled to its lanes ``4 t + m``."""
        ones = jnp.where(bits, 1.0, 0.0)
        stacked = jnp.concatenate([ones[:, t * LANES:(t + 1) * LANES] for t in range(chunks)])
        packed = jax.lax.dot_general(stacked.astype(jnp.bfloat16), matrix, (((1,), (0,)), ((), ())),
                                     preferred_element_type=jnp.float32)
        rolled = [packed[t * block:(t + 1) * block] for t in range(chunks)]
        rolled = [pltpu.roll(x, 4 * (t % (chunks // 2)), 1) if t % (chunks // 2) else x for t, x in enumerate(rolled)]
        while len(rolled) > 2:  # a tree: chunks t < 16 into one, the rest into the other
            rolled = [rolled[i] + rolled[i + 1] for i in range(0, len(rolled), 2)]
        low, high = (x.astype(jnp.int32) for x in rolled)
        low = low | (pltpu.roll(low, LANES // 2, 1) << 16)
        high = high | (pltpu.roll(high, LANES // 2, 1) << 16)
        words = jnp.where(word_lane < LANES // 2, low, pltpu.roll(high, LANES // 2, 1))
        word = jnp.where(g * GROUP < arena, (a0 + g * GROUP) // 32, (score_capacity + g * GROUP - arena) // 32)
        stage[:, pl.ds(pl.multiple_of(word, LANES), LANES)] = words

    pack_groups = jnp.where("pack" in skip, 0, columns // GROUP)
    floor = jnp.maximum(tau, INT32_MIN + 1)  # every valid key at or above tau

    def plain(g):
        pack(g, scores[:, pl.ds(pl.multiple_of(g * GROUP, GROUP), GROUP)] >= floor)

    @pl.when(jnp.logical_not(ranked))
    def _():
        # Two groups a step: one group's product latency hides behind the other's.
        @pl.loop(0, pack_groups // 2)
        def _(i):
            plain(2 * i)
            plain(2 * i + 1)

        @pl.when(pack_groups % 2 == 1)
        def _():
            plain(pack_groups - 1)

    @pl.when(ranked)
    def _():
        def ranked_pack(g, taken):
            tile = scores[:, pl.ds(pl.multiple_of(g * GROUP, GROUP), GROUP)]
            tie = ((tile == tau) & (tile > INT32_MIN)).astype(jnp.int32)
            prefix = tie
            shift = 1
            while shift < GROUP:
                prefix = prefix + jnp.where(group_lane >= shift, pltpu.roll(prefix, shift, 1), 0)
                shift *= 2
            chosen = jnp.where(taken + prefix - tie < quota, tie, 0)
            pack(g, (tile > tau) | (chosen > 0))
            return taken + prefix[:, GROUP - 1:GROUP]

        jax.lax.fori_loop(0, pack_groups, ranked_pack, jnp.zeros((block, 1), jnp.int32))

    phase.__exit__(None, None, None)
    phase = _phase("select_out", scopes)

    # Out: the block's rows' words, the stage's in one copy, or with a stage
    # of score_capacity arena words, its arena and window words in two and
    # the arena words above score_capacity from the zero tile.
    rows_out = pl.ds(pl.multiple_of(b * block, block), block)
    if score_capacity == capacity:
        outs = [pltpu.make_async_copy(stage, out_hbm.at[rows_out], out_sems.at[0])]
    else:
        held, window_words = score_capacity // 32, stage.shape[1] - score_capacity // 32
        outs = [
            pltpu.make_async_copy(stage.at[:, pl.ds(0, held)], out_hbm.at[rows_out, pl.ds(0, held)], out_sems.at[0]),
            pltpu.make_async_copy(stage.at[:, pl.ds(held, window_words)],
                                  out_hbm.at[rows_out, pl.ds(capacity // 32, window_words)], out_sems.at[0]),
        ]
        for at in range(held, capacity // 32, zeros.shape[1]):
            size = min(zeros.shape[1], capacity // 32 - at)
            outs.append(pltpu.make_async_copy(zeros.at[:, pl.ds(0, size)], out_hbm.at[rows_out, pl.ds(at, size)],
                                              out_sems.at[0]))
    for out in outs:
        out.start()
    for out in outs:
        out.wait()
    phase.__exit__(None, None, None)


VMEM_BUDGET = 62 * 1024**2  # of the kernel's 64 MiB, the rest for Mosaic's own scratch


def vmem_bytes(block: int, columns: int, tokens: int, page_tokens: int, stage_words: int) -> int:
    """The kernel's VMEM at ``block`` rows a step: the scores, the lane
    maxima, the word stage, the window's keys, the two logit stages and the
    page ring, plus Mosaic's own copies of a step's large values (pack
    tiles, stacked bits, iotas), fitted to TPU7x compiles: about 168 KiB a
    row beyond 2.25 MiB (Mosaic's totals: 66.89 MiB at 16 rows x 917504
    columns, 64.44 at 32 x 376832, both with an 8K window)."""
    own = max(0, block * 168 * 1024 - 2304 * 1024)
    return (block * columns * 4 + block * (columns // GROUP * LANES) * 4 + block * stage_words * 4
            + tokens * INDEX_DIM * 2 + 2 * (page_tokens // LANES) * block * INDEX_HEADS * LANES * 4
            + RING * (page_tokens + dsa_cache.SCALE_ROWS) * INDEX_DIM + own)


def stage_columns(capacity: int, score_capacity: int, tokens: int) -> int:
    """A step's word stage and zero tile, in words: ``score_capacity``'s
    arena words and the window's, plus ``ZERO_WORDS`` where
    ``score_capacity`` is short of ``capacity``."""
    return word_columns(score_capacity, tokens) + (ZERO_WORDS if score_capacity < capacity else 0)


def word_columns(capacity: int, tokens: int) -> int:
    """The words :func:`select_words` gives a row: the arena's, then the
    window's rounded up to whole ``GROUP`` columns."""
    return capacity // 32 + -(-tokens // GROUP) * LANES


def columns_from_words(words, segments, *, count: int, capacity: int, topk: int):
    """``(columns int32 [R, topk], counts int32 [R])``: each row's selected
    ``[arena | window]`` columns (arena column ``c``, window column
    ``capacity + t``), ascending and zero past the row's count, from its
    :func:`select_words` words (``[R, word_columns]``, this core's own rows,
    no exchange); the arena words above the window's cached total, which the
    kernel leaves unwritten, read as zero. ``segments``: the ``[5 * count]``
    descriptor."""
    rows, width = words.shape
    total = segments[4 * count - 1] + segments[5 * count - 1]
    word = jnp.arange(width)
    used = (word < (total + GROUP - 1) // GROUP * (GROUP // 32)) | (word >= capacity // 32)
    words = jnp.where(used[None], jax.lax.bitcast_convert_type(words, jnp.uint32), jnp.uint32(0))
    ends = jnp.cumsum(jax.lax.population_count(words).astype(jnp.int32), axis=1)
    slots = jnp.arange(topk, dtype=jnp.int32)
    at = jax.vmap(lambda row_ends: jnp.searchsorted(row_ends, slots, side="right"))(ends)
    at = jnp.minimum(at, width - 1).astype(jnp.int32)
    before = jnp.take_along_axis(jnp.pad(ends, ((0, 0), (1, 0)))[:, :-1], at, axis=1)
    rank = slots[None] - before  # the slot's set bit within its word
    bits = jnp.take_along_axis(words, at, axis=1)
    offset = jnp.zeros_like(at)
    for span in (16, 8, 4, 2, 1):
        low = jax.lax.population_count(bits & jnp.uint32((1 << span) - 1)).astype(jnp.int32)
        upper = rank >= low
        rank = jnp.where(upper, rank - low, rank)
        offset = jnp.where(upper, offset + span, offset)
        bits = jnp.where(upper, bits >> span, bits)
    counts = ends[:, -1]
    column = at * 32 + offset  # the window's words start at capacity / 32
    return jnp.where(slots[None] < counts[:, None], column, 0).astype(jnp.int32), counts


def candidate_columns(segments, length, positions, *, count: int, capacity: int, topk: int):
    """:func:`columns_from_words`' output for rows of at most ``topk``
    candidates (a window ``select_words`` skips as dense): every candidate,
    the sequence's cached arena run ``[base, base + cached)`` then its window
    columns ``capacity + [offset, row]``; none for padding rows."""
    rows = positions[:, None]
    offset = base = cached = jnp.zeros_like(rows)
    inside = jnp.zeros(rows.shape, bool)
    for s in range(count):
        start, size = segments[s], segments[count + s]
        here = (rows >= start) & (rows < start + size) & (rows < length[0])
        offset = jnp.where(here, start, offset)
        base = jnp.where(here, segments[3 * count + s], base)
        cached = jnp.where(here, segments[4 * count + s], cached)
        inside = inside | here
    counts = jnp.where(inside, cached + rows - offset + 1, 0)
    slots = jnp.arange(topk, dtype=jnp.int32)[None]
    column = jnp.where(slots < cached, base + slots, capacity + offset + slots - cached)
    return jnp.where(slots < counts, column, 0).astype(jnp.int32), counts[:, 0].astype(jnp.int32)


def select_words(queries, weights, window_keys, index_pages, order, segments, length, positions, *, count: int,
                 capacity: int, topk: int, skip: tuple[str, ...] = (), scheduling_group_id: int | None = None):
    """Each of this core's window rows' top ``topk`` candidates as
    ``dsa_prefill.pack_bits`` words over ``[arena | window]`` (``[R,
    capacity / 32 + W]`` uint32, ``W`` the window's words rounded up to 128,
    the words above the window's cached total zero), inside
    ``shard_map``. ``skip`` (timing only, wrong words): phases left out,
    of ``score``, ``bisect`` and ``pack``, or ``fallback`` (the per-lane
    threshold taken even where a lane may have dropped a key).

    ``queries`` ``[R, 32, 128]`` (FP8 values) and ``weights`` ``[R, 32]``
    FP32 as ``dsa_prefill.index_query`` returns them; ``window_keys`` ``[T,
    128]`` the window's dequantized keys; ``index_pages`` ``[slots, P + 32,
    128]`` uint8 the gathered index pages and ``order`` their arena order
    (``prefill_megakernel.prefix_page_order``: arena page ``i`` is slot
    ``order[1 + i]``); ``segments`` the ``[5 * count]`` descriptor; ``length``
    ``[1]``; ``positions`` ``[R]`` the rows' window positions (consecutive in
    blocks of ``ROWS``). ``scheduling_group_id``: the kernel call's
    ``_scheduling_group_id`` (the call only: its operands' ops, equal between
    calls, would leave the group with gaps once XLA merges them).

    The kernel's VMEM holds a ``[rows a step, columns]`` scores scratch, so
    the rows a step trade against the context it scores: :func:`row_steps`
    lists, from ``KernelConfig.dsa_select_rows`` down, each step size with
    the most context it fits, and the call takes the first that holds the
    window's real arena total (a switch on the descriptor). The words' layout
    stays ``capacity``'s."""
    rows, heads, dim = queries.shape
    tokens = window_keys.shape[0]
    page_tokens = index_pages.shape[1] - dsa_cache.SCALE_ROWS
    if (heads, dim) != (INDEX_HEADS, INDEX_DIM) or rows % ROWS:
        raise ValueError(f"select_words takes [R, {INDEX_HEADS}, {INDEX_DIM}] queries, R a multiple of {ROWS}")
    if capacity % GROUP or tokens % TILE or page_tokens % LANES or GROUP % page_tokens:
        raise ValueError("select_words needs a GROUP-multiple capacity, TILE-multiple windows and whole pages")
    operands = (queries, weights, window_keys, index_pages, order, segments, length, positions)
    variants = row_steps(rows, capacity, tokens, page_tokens, _config().dsa_select_rows)
    calls = [functools.partial(_select_call, count=count, capacity=capacity, topk=topk, skip=tuple(skip),
                               scheduling_group_id=scheduling_group_id, block=block, score_capacity=score_capacity)
             for block, score_capacity in variants]
    if len(calls) == 1:
        words = calls[0](*operands)
    else:
        total = segments[4 * count - 1] + segments[5 * count - 1]  # the arena tokens in use (last slot's base + cached)
        index = sum((total > score_capacity).astype(jnp.int32) for _, score_capacity in variants[:-1])
        words = jax.lax.switch(index, calls, *operands)
    return jax.lax.bitcast_convert_type(words, jnp.uint32)


def row_steps(rows: int, capacity: int, tokens: int, page_tokens: int, most: int) -> list[tuple[int, int]]:
    """``[(rows a step, score capacity)]``: from ``most`` rows a step down to
    ``ROWS``, each size that divides ``rows`` with the most arena columns (a
    ``GROUP`` multiple, at most ``capacity``) whose scores fit the kernel's
    VMEM, kept where it holds more than the larger step before it; the last
    holds ``capacity`` (``ROWS`` rows a step if no larger one does)."""
    window_columns = -(-tokens // GROUP) * GROUP
    variants: list[tuple[int, int]] = []
    block = most
    while block > ROWS:
        if rows % block == 0:
            fit = capacity
            while fit > 0 and vmem_bytes(block, fit + 2 * GROUP + window_columns, tokens, page_tokens,
                                         stage_columns(capacity, fit, tokens)) > VMEM_BUDGET:
                fit -= GROUP
            if fit > (variants[-1][1] if variants else 0):
                variants.append((block, fit))
            if fit == capacity:
                return variants
        block //= 2
    return variants + [(ROWS, capacity)]


def _select_call(queries, weights, window_keys, index_pages, order, segments, length, positions, *, count, capacity,
                 topk, skip, scheduling_group_id, block, score_capacity):
    """:func:`select_words`' kernel call at ``block`` rows a step, its scores
    scratch holding ``score_capacity`` arena columns: ``[R, word_columns]``
    int32."""
    rows, heads, dim = queries.shape
    tokens = window_keys.shape[0]
    page_tokens = index_pages.shape[1] - dsa_cache.SCALE_ROWS
    lanes = jnp.broadcast_to(weights.astype(jnp.float32).reshape(rows * heads, 1), (rows * heads, LANES))
    columns = score_capacity + 2 * GROUP + -(-tokens // GROUP) * GROUP
    stage_words = word_columns(score_capacity, tokens)
    zero_words = ZERO_WORDS if score_capacity < capacity else LANES
    config = _config()
    mxu, narrow = config.dsa_head_sums == "mxu", config.dsa_bisect == "narrow"
    butterfly, per_lane = config.dsa_head_sums == "butterfly", config.dsa_bisect == "lanes"
    head_order = config.dsa_head_sums if config.dsa_head_sums in ("major", "bf16") else "rows"
    # Row r's head weights at columns 32 (r % block) .. of its block's matrix.
    head_matrix = (jnp.eye(block, dtype=jnp.float32)[None, :, :, None]
                   * weights.astype(jnp.float32).reshape(rows // block, block, 1, heads)).reshape(rows, block * heads)
    fp8 = config.dsa_index_dots == "fp8"
    if fp8:
        # The window's dequantized keys re-encode to their own codes and scales.
        window_bytes, window_exponents = dsa_cache.quantize_index_keys(window_keys)
        window_operand = window_bytes
        window_scale = window_exponents.astype(jnp.int32).reshape(tokens // LANES, LANES)
        query_operand = queries.astype(jnp.float8_e4m3fn)
    else:
        window_operand = window_keys.astype(jnp.bfloat16)
        window_scale = jnp.zeros((tokens // LANES, LANES), jnp.int32)
        query_operand = queries.astype(jnp.bfloat16)
    if head_order != "rows":
        # Head-major within each step's rows: head h's rows at h * block ..,
        # the weights alike (bf16: heads p and p + 16 paired, in BF16).
        query_operand = query_operand.reshape(rows // block, block, heads, dim).swapaxes(1, 2).reshape(rows, heads, dim)
        w = weights.astype(jnp.float32).reshape(rows // block, block, heads).swapaxes(1, 2)
        if head_order == "bf16":
            w = w.reshape(rows // block, 2, heads // 2, block).swapaxes(1, 2).astype(jnp.bfloat16)
        lanes = jnp.broadcast_to(w.reshape(rows * heads, 1), (rows * heads, LANES))
    kernel = functools.partial(_kernel, count=count, tokens=tokens, capacity=capacity, page_tokens=page_tokens,
                               topk=topk, block=block, fp8=fp8, mxu=mxu, narrow=narrow, butterfly=butterfly,
                               per_lane=per_lane, skip=tuple(skip), scopes=config.phase_scopes, order=head_order,
                               score_capacity=score_capacity)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, word_columns(capacity, tokens)), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(rows // block,),
            in_specs=[
                pl.BlockSpec((block, heads, dim), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((block * heads, LANES), lambda b, *_: (b, 0)),
                pl.BlockSpec((block, block * heads), lambda b, *_: (b, 0)),
                pl.BlockSpec((tokens, dim), lambda b, *_: (0, 0)),
                pl.BlockSpec((tokens // LANES, LANES), lambda b, *_: (0, 0)),
                pl.BlockSpec((LANES, LANES), lambda b, *_: (0, 0)),
                hbm,
            ],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((block, columns), jnp.int32),
                pltpu.VMEM((block, columns // GROUP * LANES), jnp.int32),
                pltpu.VMEM((RING, page_tokens + dsa_cache.SCALE_ROWS, dim), jnp.uint8),
                pltpu.SemaphoreType.DMA((RING,)),
                pltpu.VMEM((block, stage_words), jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.VMEM((block, LANES), jnp.int32),
                pltpu.SMEM((block, LANES), jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.VMEM((page_tokens // LANES, block * heads, LANES), jnp.float32),
                pltpu.VMEM((page_tokens // LANES, block * heads, LANES), jnp.float32),
                pltpu.VMEM((page_tokens // LANES, 1, LANES), jnp.float32),
                pltpu.VMEM((page_tokens // LANES, 1, LANES), jnp.float32),
                pltpu.VMEM((3, block, LANES), jnp.int32),
                pltpu.VMEM((block, zero_words), jnp.int32),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=64 * 1024**2),
        name="kimi_dsa_prefill_select",
    )
    operands = (
        order.astype(jnp.int32),
        segments.astype(jnp.int32),
        length.astype(jnp.int32).reshape(1),
        positions.astype(jnp.int32),
        query_operand,
        lanes,
        head_matrix,
        window_operand,
        window_scale,
        jnp.asarray(pack_matrix(), jnp.bfloat16),
        index_pages,
    )
    with (contextlib.nullcontext() if scheduling_group_id is None
          else xla_metadata.set_xla_metadata(_scheduling_group_id=scheduling_group_id)):
        return call(*operands)


GATHER_COLLECTIVE = 51
CHUNKS = (32, 16, 8, 4, 2, 1)  # the arena words' copies, in 128-word units: the largest repeated, then one each
TRANSPOSE_ROWS = 256  # window rows of a transposed tile (two 128 x 128 blocks): sequences start on 256-row pages
TRANSPOSE_RING = 8  # tiles in flight


def _gather_kernel(segment_ref, words_hbm, gathered_hbm, out_hbm, send_sem, recv_sem, local_sem, done, tiles_in,
                   tiles_out, tile_sems, *, count, capacity, tokens, ranks, stripe):
    rank = jax.lax.axis_index("k3_sp")
    stripes = words_hbm.shape[0] // stripe
    total = segment_ref[4 * count - 1] + segment_ref[5 * count - 1]
    units = (total + GROUP - 1) // GROUP  # the arena's 128-word units in use
    window_at, window_words = capacity // 32, word_columns(capacity, tokens) - capacity // 32

    def own_units(row):
        """``(first, end)``: the arena units of window row ``row``'s
        sequence (its ``select_words`` range; none for padding)."""
        first = end = jnp.int32(0)
        for s in range(count):
            offset, size = segment_ref[s], segment_ref[count + s]
            base, cached = segment_ref[3 * count + s], segment_ref[4 * count + s]
            inside = (offset <= row) & (row < offset + size) & (cached > 0)
            first = jnp.where(inside, base // GROUP, first)
            end = jnp.where(inside, (base + cached + GROUP - 1) // GROUP, end)
        return first, end

    def window_row(sender, t):
        return t * ranks * stripe + sender * stripe

    def copies(sender, t):
        """Stripe ``t`` of core ``sender``'s rows: ``(repeat, copy(i) -> (src, dst))``
        of its sequence's arena units (the largest chunk repeated, then
        one of each smaller) and its window words, into the gathered rows."""
        local_rows = pl.ds(pl.multiple_of(t * stripe, stripe), stripe)
        window_rows = pl.ds(pl.multiple_of(window_row(sender, t), stripe), stripe)
        first, end = own_units(window_row(sender, t))
        n = end - first
        out, at = [], first
        for size in CHUNKS:
            repeat = n // size if size == CHUNKS[0] else ((n % CHUNKS[0]) & size) // size

            def one(i, at=at, size=size):
                span = pl.ds(pl.multiple_of((at + i * size) * LANES, LANES), size * LANES)
                return words_hbm.at[local_rows, span], gathered_hbm.at[window_rows, span]

            out.append((repeat, one))
            at = at + repeat * size
        span = pl.ds(window_at, window_words)
        out.append((1, lambda i: (words_hbm.at[local_rows, span], gathered_hbm.at[window_rows, span])))
        return out

    def remote(src, dst, peer):
        return pltpu.make_async_remote_copy(src, dst, send_sem, recv_sem, device_id=(peer,),
                                           device_id_type=pl.DeviceIdType.MESH)

    def each_copy(sender, action):
        @pl.loop(0, stripes)
        def _(t):
            for repeat, one in copies(sender, t):
                @pl.loop(0, repeat)
                def _(i, one=one):
                    action(*one(i))

    barrier = pltpu.get_barrier_semaphore()
    for d in range(1, ranks):
        pl.semaphore_signal(barrier, 1, device_id=((rank + d) % ranks,), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(barrier, ranks - 1)

    each_copy(rank, lambda src, dst: pltpu.make_async_copy(src, dst, local_sem).start())
    for d in range(1, ranks):
        each_copy(rank, lambda src, dst, peer=(rank + d) % ranks: remote(src, dst, peer).start())
    each_copy(rank, lambda src, dst: pltpu.make_async_copy(src, dst, local_sem).wait())
    for d in range(1, ranks):
        each_copy(rank, lambda src, dst, peer=(rank + d) % ranks: remote(src, dst, peer).wait_send())
        # Core (rank - d)'s copies into this core, as many bytes as it sends.
        each_copy((rank - d + ranks) % ranks, lambda src, dst: remote(src, dst, rank).wait_recv())

    # The used words, transposed to [words, T] in [128 words, TRANSPOSE_ROWS rows] tiles; a
    # tile of arena units outside its rows' sequence is zero.
    row_tiles = tokens // TRANSPOSE_ROWS
    n_tiles = (units + window_words // LANES) * row_tiles

    def tile(n):
        u, r = n // row_tiles, n % row_tiles
        first, end = own_units(r * TRANSPOSE_ROWS)
        inside = (u >= units) | ((first <= u) & (u < end))
        return jnp.where(u < units, u, window_at // LANES + u - units), r, inside

    def load(n, slot):
        u, r, _ = tile(n)
        return pltpu.make_async_copy(
            gathered_hbm.at[pl.ds(pl.multiple_of(r * TRANSPOSE_ROWS, TRANSPOSE_ROWS), TRANSPOSE_ROWS),
                            pl.ds(pl.multiple_of(u * LANES, LANES), LANES)],
            tiles_in.at[slot], tile_sems.at[0, slot])

    def store(n, slot):
        u, r, _ = tile(n)
        return pltpu.make_async_copy(
            tiles_out.at[slot],
            out_hbm.at[pl.ds(pl.multiple_of(u * LANES, LANES), LANES),
                       pl.ds(pl.multiple_of(r * TRANSPOSE_ROWS, TRANSPOSE_ROWS), TRANSPOSE_ROWS)],
            tile_sems.at[1, slot])

    @pl.loop(0, jnp.minimum(n_tiles, TRANSPOSE_RING - 1))
    def _(n):
        pl.when(tile(n)[2])(load(n, n).start)

    @pl.loop(0, n_tiles)
    def _(n):
        slot = n % TRANSPOSE_RING
        inside = tile(n)[2]
        pl.when(inside)(load(n, slot).wait)
        ahead = n + TRANSPOSE_RING - 1

        @pl.when((ahead < n_tiles) & tile(ahead)[2])
        def _():
            load(ahead, ahead % TRANSPOSE_RING).start()

        @pl.when(n >= TRANSPOSE_RING)
        def _():
            store(n - TRANSPOSE_RING, slot).wait()

        @pl.when(inside)
        def _():
            for j in range(TRANSPOSE_ROWS // LANES):
                tiles_out[slot, :, pl.ds(j * LANES, LANES)] = tiles_in[slot, pl.ds(j * LANES, LANES), :].T

        @pl.when(jnp.logical_not(inside))
        def _():
            tiles_out[slot] = jnp.zeros((LANES, TRANSPOSE_ROWS), jnp.int32)

        store(n, slot).start()

    @pl.loop(jnp.maximum(n_tiles - TRANSPOSE_RING, 0), n_tiles)
    def _(n):
        store(n, n % TRANSPOSE_RING).wait()

    for d in range(1, ranks):
        pl.semaphore_signal(done, 1, device_id=((rank + d) % ranks,), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(done, ranks - 1)


def gather_words(words, segments, *, count: int, capacity: int, tokens: int, ranks: int, stripe: int):
    """Every core's :func:`select_words` rows, transposed to ``[W, T]``
    uint32 (``W`` = :func:`word_columns`; window row order: core ``r``'s
    local rows ``[s stripe, (s + 1) stripe)`` are window rows ``s ranks
    stripe + r stripe ..``) as ``mla_flash_prefill_packed`` reads its
    ``select``, inside ``shard_map`` over ``k3_sp``: a row's words of its own
    sequence's arena units and its window words move (direct copies to every
    core, then 128 x 128 tile transposes), the window's other arena units
    below its cached total are zero, and the words above stay unwritten."""
    rows, width = words.shape
    if (width != word_columns(capacity, tokens) or rows * ranks != tokens or rows % stripe or stripe % 8
            or tokens % TRANSPOSE_ROWS):
        raise ValueError("gather_words takes select_words' rows of every core's stripes")
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    _, transposed = pl.pallas_call(
        functools.partial(_gather_kernel, count=count, capacity=capacity, tokens=tokens, ranks=ranks,
                          stripe=stripe),
        out_shape=(jax.ShapeDtypeStruct((tokens, width), jnp.int32),
                   jax.ShapeDtypeStruct((width, tokens), jnp.int32)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[hbm],
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.REGULAR,
                pltpu.VMEM((TRANSPOSE_RING, TRANSPOSE_ROWS, LANES), jnp.int32),
                pltpu.VMEM((TRANSPOSE_RING, LANES, TRANSPOSE_ROWS), jnp.int32),
                pltpu.SemaphoreType.DMA((2, TRANSPOSE_RING)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(collective_id=GATHER_COLLECTIVE),
        name="kimi_dsa_prefill_gather_words",
    )(segments.astype(jnp.int32), jax.lax.bitcast_convert_type(words, jnp.int32))
    return jax.lax.bitcast_convert_type(transposed, jnp.uint32)
