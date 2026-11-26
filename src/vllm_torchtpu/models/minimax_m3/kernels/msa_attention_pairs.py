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
"""Page-major MSA attention over a 4-bit cache, one KV head.

`msa_attention_lse` walks query tiles and attends every row of a tile to
the union of the pages its tokens select: a tile costs its union times its
rows. When a step's tokens select dispersed pages, nearly all of those
scores are masked (64 tokens of 16 selections each can span hundreds of
pages). This kernel walks pages instead: each (page, token) pair of the
selection is attended once, the selecting tokens' query rows gathered into
one block per page, so it costs the number of pairs whatever the dispersion.

Partial softmax sums add up across pages in any order because the scores
need no running max (`msa_attention_lse.MAX_FREE`): each token's ``acc`` and
``l`` are plain sums, accumulated in VMEM over a window of tokens. The page
preparation (4-bit unpacking, K / V scales folded into bf16 rows, the score
scale in the K scales, the visible slots as an MXU mask column) is
`msa_attention_lse`'s `attend_t`; the results differ from it by the order
of the f32 sums only.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_attention_lse,
    msa_fp4,
)

# Tokens one grid step accumulates: a page's selecting tokens are gathered
# within a window, so wider windows fill the per-page blocks (MXU rows)
# better; the accumulator is a WINDOW * H x D f32 array in VMEM.
WINDOW = 2048
# Token block widths: a page's run of pairs is cut into blocks of at most
# the widest, and a unit runs its blocks at the narrowest width holding its
# widest one. The kernel takes a window's blocks narrowest width first, so a
# unit's blocks share one.
_WIDTHS = (8, 12, 16, 20, 24, 28, 32)  # `_width`
# Blocks a loop step runs side by side (`unit`). Op replay, 4,096 rows x 16
# heads, us: 1891 / 1502 / 1163 / 1022 for 1 / 2 / 4 / 8 (uniform
# selections, j-7166bc3d, j-b5dbd63f). A unit's schedule is its MXU pushes
# plus a fixed tail (the s -> exp2 -> p @ v result latencies), which more
# blocks spread over more pushes: 16 -> 32 was -6 to -9 % at c4k-c8k shapes,
# +6 % at 1,024 tokens (op bench j-145c546e).
_UNIT = 32
# VMEM `_max_window` leaves beside the kernel's buffers for Mosaic's own: a
# 2,048-token window of 16 heads (35.6 MiB of buffers) compiles within 36 MiB,
# 8, 24, 32 and 48 heads at their windows within 48 MiB (deviceless tpu7x).
# It also holds the pair words' VMEM copy (2 x 132 KiB at a 2,048-token
# window of top-16 selections).
_VMEM_MARGIN = 2 * 1024 * 1024
# Blocks the width ordering places per loop step (a chain of dependent
# scalar loads, which the unrolled steps fill).
_PLACE_UNROLL = 4
# Slots of a block whose accumulator rows are loaded before any is stored.
_RMW_GROUP = 8
# Blocks whose page (and scale page) copies are in flight, the running ones
# included: a unit runs while the next one's copies land.
_RING = 2 * _UNIT
_NO_PAIR = jnp.iinfo(jnp.int32).max
# A sort key: (page key << `_token_bits` | token) << 8 | visible slots: page
# keys times the token field's span below 2^23 (`_max_window`).
_LIMIT_BITS = 8
# A pair word: its token | its visible slots << 16.
_TOKEN_BITS = 16
# A block word: its page key | its pairs << 24.
_KEY_BITS = 24
_KEY_MASK = (1 << _KEY_BITS) - 1
# The work list's kernel (`_work_kernel`) holds a window's pairs as [8, 128]
# tiles.
_LANES, _SUBLANES = 128, 8


def _kernel(
    # Scalar prefetch
    page_table_ref,  # page key -> physical page
    num_pairs_ref,  # [windows] valid pairs of each window
    width_blocks_ref,  # [windows * widths] blocks of each width (`_width`)
    # In
    blocks_ref,  # SMEM [P]: page key | block size << 24
    words_ref,  # SMEM [P]: each pair's token | its visible slots << 16
    words_vmem,  # [P / 128, 128]: the same words, for a block's limits as a vector
    q_hbm_ref,  # [T, H, D] in HBM
    kv_hbm_ref,  # [pages, S / 8, 4, G * 2D] uint8, in HBM
    scale_hbm_ref,  # [pages, R, S] f32, in HBM
    # Out
    o_hbm_ref,  # [H, T, D] head-major, in HBM
    # Scratch
    sums_ref,  # [W + widest, H + 8, D] f32 per token: its heads' acc rows, then
    #   head h's l in row h % 8, lane h; the spare tokens of a block's empty
    #   slots last
    kv_buf,  # [ring, 1, S / 8, 4, 2D] uint8
    scale_buf,  # [ring, 1, R, S] f32
    sem_ref,  # DMA [2, ring]
    order_ref,  # SMEM [P]: the window's blocks' first pairs, by width
    place_ref,  # SMEM [widths]: each width's next place in order_ref
    q_ref,  # [W, H, D]: the window's queries
    o_buf,  # [2, H, C, D]: output chunks on their way to HBM
    io_sem,  # DMA [3]: the queries' copy, the output chunks' copies
    *,
    num_heads: int,
    head: int,
    k_row: int,
    v_row: int,
    sm_scale: float,
):
    window, _, head_dim = q_ref.shape
    first_token = pl.multiple_of(pl.program_id(0) * window, window)
    q_copy = pltpu.make_async_copy(
        q_hbm_ref.at[pl.ds(first_token, window)], q_ref, io_sem.at[0]
    )
    q_copy.start()
    row_width = kv_buf.shape[-1]
    page_size = kv_buf.shape[2] * 8
    num_pairs = num_pairs_ref[pl.program_id(0)]
    counts = [
        width_blocks_ref[pl.program_id(0) * len(_WIDTHS) + w]
        for w in range(len(_WIDTHS))
    ]
    num_blocks = functools.reduce(jnp.add, counts)

    def place(i):
        """The blocks at pair ``i`` and the next few go to their widths' next
        places in `order_ref`; any past the last pair, to its last word."""
        for _ in range(_PLACE_UNROLL):
            real = i < num_pairs
            n = jnp.where(real, blocks_ref[i] >> _KEY_BITS, 0)
            width = jnp.where(real, _width(n), 0)
            at = place_ref[width]
            place_ref[width] = at + real.astype(jnp.int32)
            order_ref[jnp.where(real, at, order_ref.shape[0] - 1)] = i
            i = i + n
        return i

    place_ref[0] = 0
    for w in range(1, len(_WIDTHS)):
        place_ref[w] = place_ref[w - 1] + counts[w - 1]
    jax.lax.while_loop(lambda i: i < num_pairs, place, jnp.int32(0))

    def copies(slot, word):
        page = page_table_ref[word & _KEY_MASK]
        src = kv_hbm_ref.at[pl.ds(page, 1), :, :, pl.ds(head * row_width, row_width)]
        return (
            pltpu.make_async_copy(src, kv_buf.at[slot], sem_ref.at[0, slot]),
            pltpu.make_async_copy(
                scale_hbm_ref.at[pl.ds(page, 1)],
                scale_buf.at[slot],
                sem_ref.at[1, slot],
            ),
        )

    def issue(k):
        for copy in copies(k % _RING, blocks_ref[order_ref[k]]):
            copy.start()
        return k + 1

    sums_ref[...] = jnp.zeros_like(sums_ref)

    @pl.when(pl.program_id(0) == 0)
    def _clear_ring():  # a block of no pair may read a slot no copy wrote
        kv_buf[...] = jnp.zeros_like(kv_buf)
        scale_buf[...] = jnp.zeros_like(scale_buf)

    def attend(width, i, n, slot):
        """Pairs ``i .. i + n - 1`` of one page (``n <= width``): their
        accumulator tokens and ``[acc | l]`` terms. The rest of the width (the
        next page's pairs, or padding) sees no slot, and its zero terms go to
        spare tokens: a block's tokens are distinct."""
        rows, targets = [], []
        for j in range(width):
            token = words_ref[i + j] & ((1 << _TOKEN_BITS) - 1)
            rows.append(q_ref[token])
            targets.append(jnp.where(j < n, token, window + j))
        # The block's visible slots on lanes 0 .. n - 1: words i .. i + 127
        # are two rows of words_vmem, each rotated by i % 128.
        off = i % _LANES
        two = words_vmem[pl.ds(i // _LANES, 2), :]
        shift = (_LANES - off) % _LANES
        near = pltpu.roll(two[0:1], shift, 1)
        far = pltpu.roll(two[1:2], shift, 1)
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, _LANES), 1)
        limit = jnp.where(lane < _LANES - off, near, far) >> _TOKEN_BITS
        limit = jnp.pad(limit, ((0, 0), (0, head_dim - _LANES)))
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, head_dim), 1)
        limit = jnp.where(lane < n, limit, 0)
        # [q | onehot(token)] against [K | A]: A[slot, j] masks the slots
        # token j does not see (`msa_attention_lse.MXU_MASK`).
        shape = (width * num_heads, head_dim)
        row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
        col = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
        onehot = jnp.where(col == row // num_heads, 1.0, 0.0).astype(jnp.bfloat16)
        q = jnp.concatenate([jnp.concatenate(rows, axis=0), onehot], axis=1)
        page, sk, sv = msa_attention._fp4_page(
            kv_buf.at[slot], scale_buf.at[slot], k_row, v_row
        )
        k_col = (sk * (sm_scale * msa_attention_lse._LOG2_E)).reshape(page_size, 1)
        keys = (page[:, :head_dim] * k_col).astype(jnp.bfloat16)
        values = (page[:, head_dim:] * sv.reshape(page_size, 1)).astype(jnp.bfloat16)
        slot_id = jax.lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
        bias = jnp.where(slot_id < limit, 0.0, -msa_attention_lse._MASK_BIAS)
        k = jnp.concatenate([keys, bias.astype(jnp.bfloat16)], axis=1)
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )  # [width * H, S]
        p = jnp.exp2(s).astype(jnp.bfloat16)
        # l = 1 . p comes out of the p @ v matmul as the columns past D.
        v = jnp.concatenate([values, jnp.ones_like(values)], axis=1)
        return targets, jnp.dot(p, v, preferred_element_type=jnp.float32)

    def unit(width, blocks):
        """Blocks side by side: each is one chain of MXU and EUP latencies,
        which the others' fill. A block's tokens are distinct, so its adds
        into acc load a group of rows before storing any; blocks add in order
        (a token can be in several of them)."""
        terms = [attend(width, *b) for b in blocks]
        # Head h's l is lane h of its row's l columns: a token's heads fold
        # into one 8-row tile, head h in row h % 8.
        diagonal = jax.lax.broadcasted_iota(
            jnp.int32, (num_heads, head_dim), 0
        ) == jax.lax.broadcasted_iota(jnp.int32, (num_heads, head_dim), 1)
        for targets, pv in terms:
            for g in range(0, width, _RMW_GROUP):
                at = targets[g : g + _RMW_GROUP]
                acc = [sums_ref[a, :num_heads] for a in at]
                ls = [sums_ref[a, num_heads:] for a in at]
                for j, a in enumerate(at, g):
                    rows = pv[j * num_heads : (j + 1) * num_heads]
                    sums_ref[a, :num_heads] = acc[j - g] + rows[:, :head_dim]
                    heads = jnp.where(diagonal, rows[:, head_dim:], 0.0)
                    sums_ref[a, num_heads:] = ls[j - g] + heads.reshape(
                        -1, 8, head_dim
                    ).sum(axis=0)

    def body(u, issued):
        k = u * _UNIT
        issued = jax.lax.while_loop(
            lambda b: (b < k + _RING) & (b < num_blocks), issue, issued
        )
        # The unit's blocks; past the last one, blocks of no pair: they see
        # no slot of the page their ring slot holds (zeros or a copied page)
        # and add zeros.
        blocks = []
        for b in range(_UNIT):
            real = k + b < num_blocks
            at = jnp.where(real, order_ref[k + b], 0)
            n = jnp.where(real, blocks_ref[at] >> _KEY_BITS, 0)
            slot = (k + b) % _RING

            @pl.when(n > 0)
            def _wait(slot=slot):
                for copy in copies(slot, 0):
                    copy.wait()

            blocks.append((at, n, slot))
        widest = functools.reduce(jnp.maximum, [n for _, n, _ in blocks])
        for w, width in enumerate(_WIDTHS):
            fits = widest <= width
            if w:
                fits &= widest > _WIDTHS[w - 1]
            pl.when(fits)(functools.partial(unit, width, blocks))
        return issued

    q_copy.wait()
    jax.lax.fori_loop(0, -(-num_blocks // _UNIT), body, jnp.int32(0))

    # Head-major output: each head's rows are every (H + 8)-th sums row.
    tokens = o_buf.shape[2]
    chunks = window // tokens
    stride = sums_ref.shape[1]
    rows = sums_ref.reshape(-1, head_dim)

    def out_copy(c):
        return pltpu.make_async_copy(
            o_buf.at[c % 2],
            o_hbm_ref.at[
                :, pl.ds(pl.multiple_of(first_token + c * tokens, tokens), tokens)
            ],
            io_sem.at[1 + c % 2],
        )

    def finish(c, carry):
        @pl.when(c >= 2)
        def _free():
            out_copy(c - 2).wait()

        for h in range(num_heads):
            at = pl.ds(c * tokens * stride + h, tokens, stride=stride)
            l_at = pl.ds(c * tokens * stride + num_heads + h % 8, tokens, stride=stride)
            denom = jnp.maximum(
                rows[l_at][:, h : h + 1], msa_attention_lse._denom_floor()
            )
            o_buf[c % 2, h] = (rows[at] * (1.0 / denom)).astype(o_buf.dtype)
        out_copy(c).start()
        return carry

    jax.lax.fori_loop(0, chunks, finish, 0)
    for c in range(max(chunks - 2, 0), chunks):
        out_copy(c).wait()


def _width(n):
    """The narrowest of `_WIDTHS` (8, then steps of 4) holding ``n >= 1`` pairs."""
    return jnp.maximum((n - 1) >> 2, 1) - 1


def _token_bits(window):
    """Bits of a pair key's token field: a window's tokens."""
    return max(window - 1, 1).bit_length()


def _work_kernel(x_ref, blocks_ref, words_ref, stats_ref, *, token_bits):
    """One window's pair keys (``(key << token_bits | token) << _LIMIT_BITS |
    visible slots``, `_NO_PAIR` none), n = R * 128 of them, sorted and turned
    into `_work`'s block and pair words, plus its blocks of each width (lanes
    ``0 .. len(_WIDTHS) - 1`` of ``stats_ref``) and its valid pairs (lane 127).

    Pair i is held at row i % R (tile i % R // 8, sublane i % 8), lane i // R:
    the bitonic sort's most frequent partner distances (the low bits of i) are
    sublane or whole-tile moves, and a run's neighbours along i are in the
    same lane but at its ends. The sorted [R, 128] blocks transpose to the
    window's words in order."""
    rows = x_ref.shape[0]
    num = rows // _SUBLANES
    n = rows * _LANES
    lane = jax.lax.broadcasted_iota(jnp.int32, (_SUBLANES, _LANES), 1)
    sub = jax.lax.broadcasted_iota(jnp.int32, (_SUBLANES, _LANES), 0)
    tiles = [x_ref[pl.ds(v * _SUBLANES, _SUBLANES), :] for v in range(num)]

    def bit_kind(bit):
        if bit < _SUBLANES:
            return "s", bit
        if bit < rows:
            return "t", bit // _SUBLANES
        return "l", bit // rows

    def low(bit, v):
        """i & bit == 0 in tile v: a mask, or a static bool."""
        kind, d = bit_kind(bit)
        if kind == "s":
            return (sub & d) == 0
        if kind == "l":
            return (lane & d) == 0
        return not (v & d)

    # Bitonic sort, ascending.
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            kind, d = bit_kind(j)
            out = list(tiles)
            for v in range(num):
                up = True if k == n else low(k, v)
                if kind == "t":
                    if v & d:
                        continue
                    a, b = tiles[v], tiles[v + d]
                    lt = a < b
                    lo, hi = jnp.where(lt, a, b), jnp.where(lt, b, a)
                    if isinstance(up, bool):
                        out[v], out[v + d] = (lo, hi) if up else (hi, lo)
                    else:
                        out[v] = jnp.where(up, lo, hi)
                        out[v + d] = jnp.where(up, hi, lo)
                    continue
                x = tiles[v]
                axis, size = (0, _SUBLANES) if kind == "s" else (1, _LANES)
                first = low(j, v)
                partner = jnp.where(
                    first, pltpu.roll(x, size - d, axis), pltpu.roll(x, d, axis)
                )
                if up is True:
                    keep = first
                elif up is False:
                    keep = jnp.logical_not(first)
                else:
                    keep = first == up
                out[v] = jnp.where((x < partner) == keep, x, partner)
            tiles = out
            j //= 2
        k *= 2

    idx = [v * _SUBLANES + sub + rows * lane for v in range(num)]
    valid = [x != _NO_PAIR for x in tiles]
    key = [
        jnp.where(ok, x >> (token_bits + _LIMIT_BITS), 0) for x, ok in zip(tiles, valid)
    ]

    def row(x, r):  # sublane r of x on every sublane
        return jnp.broadcast_to(x[r : r + 1], x.shape)

    def neighbour(xs, step):
        """xs at i + step (step = +-1); the wrapped ends are don't-cares."""
        out = []
        for v, x in enumerate(xs):
            if step > 0:
                inner = pltpu.roll(x, _SUBLANES - 1, 0)  # x[s + 1]
                if v + 1 < num:
                    edge = row(xs[v + 1], 0)
                else:
                    edge = pltpu.roll(row(xs[0], 0), _LANES - 1, 1)  # next lane
                out.append(jnp.where(sub < _SUBLANES - 1, inner, edge))
            else:
                inner = pltpu.roll(x, 1, 0)  # x[s - 1]
                if v > 0:
                    edge = row(xs[v - 1], _SUBLANES - 1)
                else:
                    edge = pltpu.roll(row(xs[-1], _SUBLANES - 1), 1, 1)  # previous lane
                out.append(jnp.where(sub > 0, inner, edge))
        return out

    def scan(xs, op, neutral, reverse):
        """Inclusive scan of `op` along i (descending i with `reverse`)."""
        xs = list(xs)
        order = range(num - 1, -1, -1) if reverse else range(num)
        carry = None
        for v in order:  # along rows: sublanes, then tiles
            x = xs[v]
            d = 1
            while d < _SUBLANES:
                if reverse:
                    sh = jnp.where(
                        sub < _SUBLANES - d, pltpu.roll(x, _SUBLANES - d, 0), neutral
                    )
                else:
                    sh = jnp.where(sub >= d, pltpu.roll(x, d, 0), neutral)
                x = op(x, sh)
                d *= 2
            if carry is not None:
                x = op(x, carry)
            carry = row(x, 0 if reverse else _SUBLANES - 1)
            xs[v] = x
        # across lanes: each lane takes the scan of the lanes before it
        tot = carry
        d = 1
        while d < _LANES:
            if reverse:
                sh = jnp.where(
                    lane < _LANES - d, pltpu.roll(tot, _LANES - d, 1), neutral
                )
            else:
                sh = jnp.where(lane >= d, pltpu.roll(tot, d, 1), neutral)
            tot = op(tot, sh)
            d *= 2
        if reverse:
            before = jnp.where(
                lane < _LANES - 1, pltpu.roll(tot, _LANES - 1, 1), neutral
            )
        else:
            before = jnp.where(lane >= 1, pltpu.roll(tot, 1, 1), neutral)
        return [op(x, before) for x in xs]

    key_next = neighbour(key, 1)
    valid_next = neighbour([jnp.where(ok, 1, 0) for ok in valid], 1)
    key_prev = neighbour(key, -1)
    ends = [
        jnp.where((kk != kn) | (vn == 0) | (i == n - 1), i + 1, n)
        for kk, kn, vn, i in zip(key, key_next, valid_next, idx)
    ]
    run_end = scan(ends, jnp.minimum, n, True)
    starts = [
        jnp.where((kk != kp) | (i == 0), i, 0) for kk, kp, i in zip(key, key_prev, idx)
    ]
    run_start = scan(starts, jnp.maximum, 0, False)

    counts = [jnp.zeros((_SUBLANES, _LANES), jnp.int32) for _ in _WIDTHS]
    pairs = jnp.zeros((_SUBLANES, _LANES), jnp.int32)
    blocks, words = [], []
    for v in range(num):
        x, ok, i = tiles[v], valid[v], idx[v]
        size = jnp.minimum(run_end[v] - i, _WIDTHS[-1])
        start = ok & (((i - run_start[v]) & (_WIDTHS[-1] - 1)) == 0)
        width = _width(size)
        for w in range(len(_WIDTHS)):
            counts[w] = counts[w] + jnp.where(start & (width == w), 1, 0)
        pairs = pairs + jnp.where(ok, 1, 0)
        token = jnp.where(ok, (x >> _LIMIT_BITS) & ((1 << token_bits) - 1), 0)
        limit = jnp.where(ok, x & ((1 << _LIMIT_BITS) - 1), 0)
        blocks.append(key[v] | (size << _KEY_BITS))
        words.append(token | (limit << _TOKEN_BITS))
    stats = jnp.zeros((_SUBLANES, _LANES), jnp.int32)
    for w, c in enumerate(counts):
        stats = jnp.where(lane == w, jnp.sum(c), stats)
    stats_ref[...] = jnp.where(lane == _LANES - 1, jnp.sum(pairs), stats)
    blocks_ref[...] = jnp.concatenate(blocks, axis=0).T
    words_ref[...] = jnp.concatenate(words, axis=0).T


def _work(keys, local_key, local_limit, window, page_size, interpret=False):
    """Per window of `window` tokens: its (page key, token) pairs in key order,
    as two ``[windows * P]`` arrays: the page key with the size of the
    block starting there (the rest of its page's run, at most the widest
    width), and the pair word (`_TOKEN_BITS`); the valid pairs per window; and
    each window's blocks of each width (``[windows * widths]``).
    A page a token lists twice is one pair. The pairs are sorted and turned
    into words in one kernel per window (`_work_kernel`)."""
    num_tokens, topk = keys.shape
    windows = num_tokens // window
    earlier = jnp.arange(topk)[:, None] > jnp.arange(topk)[None, :]
    again = jnp.any((keys[:, :, None] == keys[:, None, :]) & earlier, axis=2)
    keys = jnp.where(again, -1, keys)
    tok = jnp.arange(num_tokens, dtype=jnp.int32)[:, None] % window
    limit = jnp.where(keys == local_key[:, None], local_limit[:, None], page_size)
    token_bits = _token_bits(window)
    pair = jnp.where(
        keys >= 0, (keys << token_bits | tok) << _LIMIT_BITS | limit, _NO_PAIR
    )
    length = window * topk
    # The kernel unrolls its sort over every tile of the window: sized for
    # `WINDOW` x top-16 (32,768 pairs, 128 KiB a window in VMEM).
    size = max(pl.next_power_of_2(length), _SUBLANES * _LANES)
    pair = jnp.pad(
        pair.reshape(windows, length),
        ((0, 0), (0, size - length)),
        constant_values=_NO_PAIR,
    )
    r = size // _LANES
    blocks, words, stats = pl.pallas_call(
        functools.partial(_work_kernel, token_bits=token_bits),
        grid=(windows,),
        in_specs=[pl.BlockSpec((None, r, _LANES), lambda w: (w, 0, 0))],
        out_specs=[
            pl.BlockSpec((None, _LANES, r), lambda w: (w, 0, 0)),
            pl.BlockSpec((None, _LANES, r), lambda w: (w, 0, 0)),
            pl.BlockSpec((None, _SUBLANES, _LANES), lambda w: (w, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((windows, _LANES, r), jnp.int32),
            jax.ShapeDtypeStruct((windows, _LANES, r), jnp.int32),
            jax.ShapeDtypeStruct((windows, _SUBLANES, _LANES), jnp.int32),
        ],
        interpret=interpret,
        name="minimax_m3_msa_pairs_work",
    )(pair.reshape(windows, r, _LANES))
    # Room past the last pair for the widest block and for a block's 128 words
    # in VMEM (`_kernel`'s limits), in whole 1,024-word SMEM blocks.
    pad = ((0, 0), (0, -(length + _LANES) % 1024 + _LANES))

    def flat(x):
        return jnp.pad(x.reshape(windows, size)[:, :length], pad).reshape(-1)

    return (
        flat(blocks),
        flat(words),
        stats[:, 0, _LANES - 1],
        stats[:, 0, : len(_WIDTHS)].reshape(-1),
    )


def _max_window(num_heads, head_dim, page_size, num_keys, window=WINDOW) -> int:
    """`window`, or fewer tokens (a multiple of 8): halved while the sort
    key's fields would not hold its pairs, and cut where its sums and queries
    would not fit the VMEM limit beside the copy ring, the output chunks and
    `_VMEM_MARGIN`."""
    token_bytes = (num_heads + 8) * head_dim * 4 + num_heads * head_dim * 2
    # A ring slot: a 4-bit page of [K | V] rows and its scale page.
    ring_bytes = _RING * page_size * (head_dim + msa_fp4.SCALE_ROWS * 4)
    out_bytes = 2 * num_heads * 256 * head_dim * 2  # chunks of at most 256 tokens
    budget = (
        msa_attention._VMEM_LIMIT_BYTES
        - _VMEM_MARGIN
        - ring_bytes
        - out_bytes
        - _WIDTHS[-1] * (num_heads + 8) * head_dim * 4
    )
    # Halved, not trimmed, for the key (a step's tokens split into whole
    # windows), down to the 1,024 tokens of the first version: larger page
    # tables go to the tiles (`supports`).
    keys = ((_NO_PAIR >> _LIMIT_BITS) - 1) // max(num_keys, 1)
    while 1 << _token_bits(window) > keys and window > 1024:
        window //= 2
    return min(window, max(budget, 0) // token_bytes) // 8 * 8


def supports(num_heads, head_dim, page_size, num_keys, window=WINDOW) -> bool:
    """Whether `msa_sparse_attention_pairs` takes calls of this shape: whole
    sublane tiles of query rows per token, a page's slots and a block's
    tokens within the D mask columns, a token's heads' l in one row, the pair
    and block words' fields, and a window of tokens in VMEM."""
    window = _max_window(num_heads, head_dim, page_size, num_keys, window)
    return (
        window >= 8
        and window < 1 << _TOKEN_BITS
        and num_heads <= head_dim
        and num_heads % 8 == 0
        and head_dim % 128 == 0
        and page_size <= head_dim
        and page_size < 1 << _LIMIT_BITS
        and _WIDTHS[-1] <= head_dim
        and num_keys <= _KEY_MASK
        and num_keys << _token_bits(window) < _NO_PAIR >> _LIMIT_BITS
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "head",
        "k_scale_row",
        "v_scale_row",
        "window",
        "interpret",
    ),
)
def msa_sparse_attention_pairs(
    q: jax.Array,
    kv_cache: jax.Array,
    selected: jax.Array,
    local_page: jax.Array,
    local_limit: jax.Array,
    page_table: jax.Array,
    *,
    sm_scale: float,
    scale_cache: jax.Array,
    head: int = 0,
    k_scale_row: int = 0,
    v_scale_row: int = 1,
    window: int = WINDOW,
    interpret: bool = False,
) -> jax.Array:
    """`msa_attention_lse.msa_sparse_attention_lse`'s head-major output
    ``[H, T, D]`` for a 4-bit cache (``q [T, H, D]``, ``selected [T, K]``
    page keys, -1 none; a token with no page gets zeros), page-major. Needs
    `msa_attention_lse.MAX_FREE`; returns no log-sum-exp."""
    if not msa_attention_lse.MAX_FREE:
        raise ValueError("page-major attention needs max-free scores")
    msa_attention.check_cache(kv_cache, scale_cache)
    num_tokens, num_heads, head_dim = q.shape
    page_size = msa_attention.page_tokens(kv_cache)
    num_keys = page_table.shape[0]
    window = min(
        _max_window(num_heads, head_dim, page_size, num_keys, window),
        -(-num_tokens // 8) * 8,
    )
    if not supports(num_heads, head_dim, page_size, num_keys, window):
        raise ValueError(
            f"{num_heads} heads, D {head_dim}, {page_size}-slot pages, "
            f"{num_keys} page keys, window {window}"
        )
    q, selected, local_page, local_limit = msa_attention_lse._pad_tokens(
        q, selected, local_page, local_limit, window
    )
    blocks, words, num_pairs, width_blocks = _work(
        selected, local_page, local_limit, window, page_size, interpret
    )
    windows = q.shape[0] // window
    tokens = math.gcd(window, 256)  # an output chunk
    # One buffer: with two of a 2,048-token window's blocks, a call of two
    # windows ran 155 ms instead of 0.67 ms (op bench j-ffe043b7, j-eef1a888).
    smem = pl.BlockSpec(
        (blocks.shape[0] // windows,),
        lambda w, *_: (w,),
        memory_space=pltpu.SMEM,
        pipeline_mode=pl.Buffered(1),
    )
    _, rows_per_page, packing, _ = kv_cache.shape
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            num_heads=num_heads,
            head=head,
            k_row=k_scale_row,
            v_row=v_scale_row,
            sm_scale=sm_scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(windows,),
            in_specs=[
                smem,
                smem,
                pl.BlockSpec(
                    (blocks.shape[0] // windows // _LANES, _LANES),
                    lambda w, *_: (w, 0),
                ),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[
                pltpu.VMEM(
                    (window + _WIDTHS[-1], num_heads + 8, head_dim), jnp.float32
                ),
                pltpu.VMEM(
                    (_RING, 1, rows_per_page, packing, 2 * head_dim), kv_cache.dtype
                ),
                pltpu.VMEM((_RING, 1) + scale_cache.shape[1:], jnp.float32),
                pltpu.SemaphoreType.DMA((2, _RING)),
                pltpu.SMEM((blocks.shape[0] // windows,), jnp.int32),
                pltpu.SMEM((len(_WIDTHS),), jnp.int32),
                pltpu.VMEM((window, num_heads, head_dim), q.dtype),
                pltpu.VMEM((2, num_heads, tokens, head_dim), q.dtype),
                pltpu.SemaphoreType.DMA((3,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(
            (num_heads, windows * window, head_dim), q.dtype
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=msa_attention._VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="minimax_m3_msa_sparse_attention_pairs",
    )(
        page_table,
        num_pairs,
        width_blocks,
        blocks,
        words,
        words.reshape(-1, _LANES),
        q,
        kv_cache,
        scale_cache,
    )
    return out[:, :num_tokens]
