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

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_attention_lse

# Tokens one grid step accumulates: a page's selecting tokens are gathered
# within a window, so wider windows fill the per-page blocks (MXU rows)
# better; the accumulators are two WINDOW * H x D f32 arrays in VMEM.
WINDOW = 1024
# Token block widths: a page's run of pairs is cut into blocks of at most
# the widest, and each block runs at the narrowest width holding it.
_WIDTHS = (8, 16, 24, 32)
# Blocks a loop step runs side by side (`unit`). Op replay, 4,096 rows x 16
# heads, us: 1891 / 1502 / 1163 / 1022 for 1 / 2 / 4 / 8 (uniform
# selections, j-7166bc3d, j-b5dbd63f).
_UNIT = 8
# Blocks whose page (and scale page) copies are in flight, the running ones included.
_RING = 16
_NO_PAIR = jnp.iinfo(jnp.int32).max
# A pair word: its token's first query row (token * H) | visible slots << 16.
_ROW_BITS = 16
# A sort key: (page key * window + token) << 8 | visible slots: page keys
# below 2^23 / window (8,192 at 1,024 tokens; `supports`).
_LIMIT_BITS = 8
# A block word: its page key | its pairs << 24.
_KEY_BITS = 24
_KEY_MASK = (1 << _KEY_BITS) - 1


def _kernel(
    # Scalar prefetch
    page_table_ref,  # page key -> physical page
    num_pairs_ref,  # [windows] valid pairs of each window
    # In
    pairs_ref,  # SMEM [1, 2, P]: page key | block size << 24; pair word
    q_ref,  # [W * H, D] token-major rows (t * H + h)
    kv_hbm_ref,  # [pages, S / 8, 4, G * 2D] uint8, in HBM
    scale_hbm_ref,  # [pages, R, S] f32, in HBM
    # Out
    o_ref,  # [H, W, D] head-major
    # Scratch
    acc_ref,  # [W * H, D] f32
    l_ref,  # [W * H, D] f32, every column l
    kv_buf,  # [ring, 1, S / 8, 4, 2D] uint8
    scale_buf,  # [ring, 1, R, S] f32
    sem_ref,  # DMA [2, ring]
    *,
    num_heads: int,
    head: int,
    k_row: int,
    v_row: int,
    sm_scale: float,
):
    head_dim = q_ref.shape[1]
    window = q_ref.shape[0] // num_heads
    row_width = kv_buf.shape[-1]
    page_size = kv_buf.shape[2] * 8
    num_pairs = num_pairs_ref[pl.program_id(0)]

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

    def issue(state):
        i, k = state
        word = pairs_ref[0, 0, i]
        for copy in copies(k % _RING, word):
            copy.start()
        return i + (word >> _KEY_BITS), k + 1

    acc_ref[...] = jnp.zeros_like(acc_ref)
    l_ref[...] = jnp.zeros_like(l_ref)

    @pl.when(pl.program_id(0) == 0)
    def _clear_ring():  # a block of no pair may read a slot no copy wrote
        kv_buf[...] = jnp.zeros_like(kv_buf)
        scale_buf[...] = jnp.zeros_like(scale_buf)

    def attend(width, i, n, slot):
        """Pairs ``i .. i + n - 1`` of one page (``n <= width``): their query
        rows' first rows and ``[acc | l]`` terms. The rest of the width (the
        next page's pairs, or padding) sees no slot: its terms are zero."""
        rows, starts = [], []
        lane = jax.lax.broadcasted_iota(jnp.int32, (1, head_dim), 1)
        limit = jnp.zeros((1, head_dim), jnp.int32)
        for j in range(width):
            word = pairs_ref[0, 1, i + j]
            start = pl.multiple_of(word & ((1 << _ROW_BITS) - 1), num_heads)
            rows.append(q_ref[pl.ds(start, num_heads), :])
            starts.append(start)
            limit = jnp.where(lane == j, word >> _ROW_BITS, limit)
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
        return starts, jnp.dot(p, v, preferred_element_type=jnp.float32)

    def unit(width, blocks):
        """Blocks side by side: each is one chain of MXU and EUP latencies,
        which the others' fill. Their adds into acc go in order (a token can
        be in several of them)."""
        terms = [attend(width, *b) for b in blocks]
        for starts, pv in terms:
            for j, start in enumerate(starts):
                rows = pv[j * num_heads : (j + 1) * num_heads]
                acc_ref[pl.ds(start, num_heads)] += rows[:, :head_dim]
                l_ref[pl.ds(start, num_heads)] += rows[:, head_dim:]

    def body(state):
        i, k, issued = state
        issued = jax.lax.while_loop(
            lambda s: (s[1] < k + _RING) & (s[0] < num_pairs), issue, issued
        )
        # The unit's blocks; past the last pair, blocks of no pair: they see
        # no slot of the page their ring slot holds (zeros or a copied page)
        # and add zeros.
        blocks, at, real = [], i, 0
        for b in range(_UNIT):
            n = jnp.where(at < num_pairs, pairs_ref[0, 0, at] >> _KEY_BITS, 0)
            slot = (k + b) % _RING

            @pl.when(n > 0)
            def _wait(slot=slot):
                for copy in copies(slot, 0):
                    copy.wait()

            blocks.append((at, n, slot))
            at, real = at + n, real + jnp.minimum(n, 1)
        widest = functools.reduce(jnp.maximum, [n for _, n, _ in blocks])
        for w, width in enumerate(_WIDTHS):
            fits = widest <= width
            if w:
                fits &= widest > _WIDTHS[w - 1]
            pl.when(fits)(functools.partial(unit, width, blocks))
        return at, k + real, issued

    zero = jnp.int32(0)
    jax.lax.while_loop(lambda s: s[0] < num_pairs, body, (zero, zero, (zero, zero)))

    # Head-major output: each head's rows are every H-th accumulator row.
    tokens = math.gcd(window, 256)

    def finish(c, carry):
        for h in range(num_heads):
            at = pl.ds(c * tokens * num_heads + h, tokens, stride=num_heads)
            denom = jnp.maximum(l_ref[at][:, :1], msa_attention_lse._denom_floor())
            o_ref[h, pl.ds(pl.multiple_of(c * tokens, tokens), tokens)] = (
                acc_ref[at] * (1.0 / denom)
            ).astype(o_ref.dtype)
        return carry

    jax.lax.fori_loop(0, window // tokens, finish, 0)


def _work(keys, local_key, local_limit, window, num_heads, page_size):
    """Per window of `window` tokens: its (page key, token) pairs in key order,
    ``[windows, 2, P]``: the page key with the size of the block starting
    there (the rest of its page's run, at most the widest width), and the
    pair word (`_ROW_BITS`); and the valid pairs per window. A pair a token
    lists twice counts once (its copy sees no slot)."""
    num_tokens, topk = keys.shape
    windows = num_tokens // window
    tok = jnp.arange(num_tokens, dtype=jnp.int32)[:, None] % window
    limit = jnp.where(keys == local_key[:, None], local_limit[:, None], page_size)
    # One sort operand: (key * window + token) << 8 | visible slots.
    pair = jnp.where(keys >= 0, (keys * window + tok) << _LIMIT_BITS | limit, _NO_PAIR)
    pair = jax.lax.sort(pair.reshape(windows, -1), dimension=1)
    valid = pair != _NO_PAIR
    limit = pair & ((1 << _LIMIT_BITS) - 1)
    pair = jnp.where(valid, pair >> _LIMIT_BITS, _NO_PAIR)
    word = (pair % window) * num_heads | (limit << _ROW_BITS)
    key = jnp.where(valid, pair // window, 0)
    dup = jnp.pad(pair[:, 1:] == pair[:, :-1], ((0, 0), (1, 0)))
    # Past the valid pairs, and a pair's copies: token 0, no slot.
    word = jnp.where(valid & ~dup, word, word & ((1 << _ROW_BITS) - 1))
    word = jnp.where(valid, word, 0)
    length = pair.shape[1]
    idx = jnp.arange(length, dtype=jnp.int32)[None]
    last = jnp.pad(key[:, 1:] != key[:, :-1], ((0, 0), (0, 1)), constant_values=True)
    run_end = jax.lax.cummin(jnp.where(last, idx + 1, length), axis=1, reverse=True)
    size = jnp.minimum(run_end - idx, _WIDTHS[-1])
    packed = jnp.stack([key | (size << _KEY_BITS), word], axis=1)
    packed = jnp.pad(packed, ((0, 0), (0, 0), (0, _WIDTHS[-1])))
    return packed, jnp.sum(valid, axis=1, dtype=jnp.int32)


def _max_window(num_heads, head_dim, window=WINDOW) -> int:
    """`window`, or fewer tokens (a multiple of 8) where its accumulators and
    double-buffered query and output blocks, beside a unit's live blocks
    ([q | onehot], s, p, [pv | l] per row), would not fit the VMEM limit."""
    row_bytes = num_heads * head_dim * (2 * 4 + 2 * 2 + 2 * 2)
    unit_bytes = _UNIT * _WIDTHS[-1] * num_heads * head_dim * (2 * 2 + 4 + 2 + 2 * 4)
    budget = msa_attention._VMEM_LIMIT_BYTES * 9 // 10 - unit_bytes
    return min(window, max(budget, 0) // row_bytes // 8 * 8)


def supports(num_heads, head_dim, page_size, num_keys, window=WINDOW) -> bool:
    """Whether `msa_sparse_attention_pairs` takes calls of this shape: whole
    sublane tiles of query rows per token, a page's slots and a block's
    tokens within the D mask columns, the pair and block words' fields, and
    a window of tokens in VMEM."""
    window = _max_window(num_heads, head_dim, window)
    return (
        window >= 8
        and num_heads % 8 == 0
        and head_dim % 128 == 0
        and page_size <= head_dim
        and page_size < 1 << _LIMIT_BITS
        and _WIDTHS[-1] <= head_dim
        and num_keys * window < _NO_PAIR >> _LIMIT_BITS
        and num_keys <= _KEY_MASK
        and window * num_heads <= 1 << _ROW_BITS
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
    window = min(_max_window(num_heads, head_dim, window), -(-num_tokens // 8) * 8)
    if not supports(num_heads, head_dim, page_size, page_table.shape[0], window):
        raise ValueError(
            f"{num_heads} heads, D {head_dim}, {page_size}-slot pages, "
            f"{page_table.shape[0]} page keys, window {window}"
        )
    q, selected, local_page, local_limit = msa_attention_lse._pad_tokens(
        q, selected, local_page, local_limit, window
    )
    packed, num_pairs = _work(
        selected, local_page, local_limit, window, num_heads, page_size
    )
    windows = q.shape[0] // window
    rows = window * num_heads
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
            num_scalar_prefetch=2,
            grid=(windows,),
            in_specs=[
                pl.BlockSpec(
                    (1,) + packed.shape[1:],
                    lambda w, *_: (w, 0, 0),
                    memory_space=pltpu.SMEM,
                ),
                pl.BlockSpec((rows, head_dim), lambda w, *_: (w, 0)),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec(
                (num_heads, window, head_dim), lambda w, *_: (0, w, 0)
            ),
            scratch_shapes=[
                pltpu.VMEM((rows, head_dim), jnp.float32),
                pltpu.VMEM((rows, head_dim), jnp.float32),
                pltpu.VMEM(
                    (_RING, 1, rows_per_page, packing, 2 * head_dim), kv_cache.dtype
                ),
                pltpu.VMEM((_RING, 1) + scale_cache.shape[1:], jnp.float32),
                pltpu.SemaphoreType.DMA((2, _RING)),
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
        packed,
        q.reshape(-1, head_dim),
        kv_cache,
        scale_cache,
    )
    return out[:, :num_tokens]
