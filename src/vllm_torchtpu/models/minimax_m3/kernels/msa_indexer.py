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
"""Pallas lightning-indexer block scores for MiniMax Sparse Attention.

For every query token and every KV page of its request, the score of the
page is the max over its tokens of ``index_q . index_k`` (no scale: only the
order matters). Each request's tokens are cut into query tiles of
`block_q` rows; a grid step scores one tile against `pages_per_step` pages
read straight from the paged index cache, so a request's index keys are read
once per tile. The cache packs consecutive tokens into 32-bit words
(`packed_page`); each packing slot is scored separately, which a max does not
mind.

Long (prefill) requests' scores are produced transposed,
``[pages, tiles * block_q]``, so each output block is lane-aligned, and
gathered back to ``[tokens, pages]``; the short (decode) requests' 128-page
groups are lane-aligned either way and come out token-major.
"""

import functools
import os
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4
from vllm_torchtpu.models.minimax_m3.kernels import page_table as page_table_lib
from vllm_torchtpu.models.minimax_m3.kernels.packed_page import (
    check_page_dtype,
    unpack_page,
)

DEFAULT_BLOCK_Q = 128
DEFAULT_LARGE_BLOCK_Q = 512
# TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP: pages per grid step of the
# large (prefill) class, a multiple of 32 (default 32). Wider steps amortize
# the per-step cost over more pages; the scores are bit-identical (each page
# maximum is one dot, whatever the grid).
LARGE_PAGES_PER_STEP_ENV = "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP"
DEFAULT_PAGES_PER_STEP = int(os.environ.get(LARGE_PAGES_PER_STEP_ENV, "32"))
if DEFAULT_PAGES_PER_STEP <= 0 or DEFAULT_PAGES_PER_STEP % 32:
    raise ValueError(
        f"{LARGE_PAGES_PER_STEP_ENV} must be a positive multiple of 32, got "
        f"{DEFAULT_PAGES_PER_STEP}"
    )
# The block_q class (decode, MTP verify) runs the v3 DMA ring; 128 pages per
# step keep more bytes in flight (j-dba77583: 32 req x 8 tokens at 128K,
# 1.95 -> 1.66 ms). The large prefill class keeps DEFAULT_PAGES_PER_STEP.
DEFAULT_SMALL_PAGES_PER_STEP = 128
DEFAULT_KERNEL = "v3"
# Rows per token-major query tile of the block_q class (see `block_scores`).
_SMALL_TILE_Q = 64
# Folded 4-bit and fp8 pages are scored transposed (`_chunk_maxima_t`), whose
# cost grows with the tile's rows: by default 32 holds a decode request of 4
# folded index heads x 8 (MTP) tokens; callers with one index head per rank
# pass `transposed_tile_q=8` (`msa.block_score_options`).
_SMALL_TILE_Q_TRANSPOSED = 32
# Large prefill tiles: v3's DMA ring, as the small class. v2's 32 BlockSpec
# page inputs cost ~1,900 scalar bundles per 32-page step beside ~2,500 of
# vector work (AOT); v3 is bit-identical and -17..-24% per indexer call at
# the c48 / c64 DCP 2 / c4 mixed-step shapes (bench j-5eb4a611).
DEFAULT_LARGE_KERNEL = "v3"
_TILE_SHIFT = 16
# tile_info = query row of the tile << _ROW_SHIFT | request of the tile
_ROW_SHIFT = 10
_SEQ_MASK = (1 << _ROW_SHIFT) - 1
# f32 query rows are read at element offsets that are multiples of the sublane tile.
_ROW_ALIGN = 8
_GROUP_MASK = (1 << _TILE_SHIFT) - 1

logger = init_logger(__name__)

# TPU_MINIMAX_M3_INDEXER_V2=1: kernel v3 scores with fp8 queries as well as
# fp8 keys, so the MXU runs fp8 x fp8 (twice the bf16 rate) instead of
# widening the keys to bf16. Each query row is scaled so its largest |q| is
# _FP8_Q_MAX before the cast, and its maxima are scaled back: a positive
# per-row scale keeps the order of the row's page scores, and the scaled-back
# scores stay comparable across rows and heads (`merge_head_candidates`).
# Not bit-identical to the bf16 queries: the query rounds to 3 mantissa bits.
INDEXER_V2_ENV = "TPU_MINIMAX_M3_INDEXER_V2"
_FP8_Q_MAX = 256.0


def fp8_queries_enabled() -> bool:
    return os.environ.get(INDEXER_V2_ENV, "0") == "1"


if DEFAULT_PAGES_PER_STEP != 32:
    logger.info(
        "%s=%d: MSA indexer prefill tiles score %d pages per grid step.",
        LARGE_PAGES_PER_STEP_ENV,
        DEFAULT_PAGES_PER_STEP,
        DEFAULT_PAGES_PER_STEP,
    )

if fp8_queries_enabled():
    logger.info(
        "%s=1: MSA indexer prefill block scores use fp8 queries (fp8 x fp8 MXU).",
        INDEXER_V2_ENV,
    )


def _kernel_v1(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    # In
    q_ref,  # [Bq, D]
    *refs,  # G x [1, S / p, p, D] index-cache pages, then the output
):
    pages_per_step = len(refs) - 1
    page_refs, o_ref = refs[:pages_per_step], refs[pages_per_step]
    q = q_ref[...].astype(jnp.bfloat16)
    maxima = []
    for page_ref in page_refs:
        page_max = None
        for keys in unpack_page(page_ref):
            s = jax.lax.dot_general(
                keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            m = jnp.max(s, axis=0, keepdims=True)
            page_max = m if page_max is None else jnp.maximum(page_max, m)
        maxima.append(page_max)
    o_ref[...] = jnp.concatenate(maxima, axis=0)


# Scores per dot in `_kernel_v2`: bounds the f32 ``[pages * S, block_q]``
# product kept in VMEM (2 MiB).
_MAX_SCORE_ELEMS = 512 * 1024


def _page_keys(page_ref) -> jax.Array:
    """``[1, S / p, p, D]`` page -> bf16 ``[S, D]`` keys in packing order.

    One width-changing bitcast of the packed 32-bit words: the token order
    inside the page changes, which a max over the page does not see.
    """
    words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
    return _words_keys(words, page_ref.dtype)


# Folded 4-bit pages: scale the keys (``[S, D]`` bf16, before the dot) rather
# than the f32 scores (``[S, Bq]``, after it): fewer elements per page, the
# same per-key scale (rounded to bf16 with the key, far finer than fp8 keys).
SCALE_KEYS = True


def _folded_keys(words: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A 4-bit page with its scales folded in (`msa_fp4.scales_in_page`):
    ``[S / 8 + 1, D]`` words -> (unscaled bf16 keys ``[S, D]``, the per-slot
    scales of the last word row as a ``[1, S]`` f32 lane vector)."""
    keys = pltpu.bitcast(words[:-1], jnp.int4).astype(jnp.bfloat16)
    return keys, pltpu.bitcast(words[-1:], jnp.float32)


def _scale_rows_to_column(rows: list[jax.Array]) -> jax.Array:
    """The group's ``[1, S]`` slot-scale rows -> one ``[n * S, 1]`` column in
    the score rows' (page-major) order, non-finite read as 0. One transpose
    of the stacked ``[n, S]`` rows per dot group: a ``[1, S] -> [S, 1]``
    transpose per page made the 4-bit indexer 50% slower than fp8."""
    stacked = rows[0] if len(rows) == 1 else jnp.concatenate(rows, axis=0)  # [n, S]
    stacked = jnp.where(jnp.isfinite(stacked), stacked, 0.0)
    cols = stacked.T  # [S, n]
    return jnp.concatenate([cols[:, g : g + 1] for g in range(len(rows))], axis=0)


def _words_keys(words: jax.Array, dtype) -> jax.Array:
    """Packed ``[S / p, D]`` words -> bf16 ``[S, D]`` keys. A 4-bit (uint8)
    cache's int4 values come out unscaled, in slot order (`msa_fp4`)."""
    if dtype == jnp.uint8:
        return pltpu.bitcast(words, jnp.int4).astype(jnp.bfloat16)
    return pltpu.bitcast(words, dtype).astype(jnp.bfloat16)


def _scale_column(scale_page: jax.Array, row: int) -> jax.Array:
    """``[8, 128]`` scale page -> ``[128, 1]`` scales of its slots for scale-page
    row `row`. An unwritten slot's scale may be any bits; non-finite reads as 0."""
    col = scale_page.T[:, row : row + 1]
    return jnp.where(jnp.isfinite(col), col, 0.0)


def _kernel_v2(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    # In
    q_ref,  # [Bq, D]
    *refs,  # G x [1, S / p, p, D] index-cache pages, [G x [1, 8, 128] scale pages], out
    scale_row: int | None = None,
):
    """`_kernel_v1` with the pages' keys unpacked by one bitcast each and
    stacked, so each dot keeps the query tile latched across many pages.

    With `scale_row` (4-bit caches) each page is followed, after all pages,
    by its scale page, and a key's score is scaled before the block max."""
    num_inputs = len(refs) - 1
    pages_per_step = num_inputs if scale_row is None else num_inputs // 2
    page_refs, o_ref = refs[:pages_per_step], refs[num_inputs]
    scale_refs = refs[pages_per_step:num_inputs]
    q = q_ref[...].astype(jnp.bfloat16)
    block_q = q.shape[0]
    page_size = _page_tokens(page_refs[0].shape[1], page_refs[0].dtype)
    pages_per_dot = max(
        1, min(pages_per_step, _MAX_SCORE_ELEMS // (page_size * block_q))
    )
    maxima = []
    folded = msa_fp4.scales_in_page(page_refs[0].shape[1], page_refs[0].dtype)
    for first in range(0, pages_per_step, pages_per_dot):
        group = page_refs[first : first + pages_per_dot]
        if folded:
            pairs = [_folded_keys(r.bitcast(jnp.uint32)[0, :, 0, :]) for r in group]
            keys = [k for k, _ in pairs]
        else:
            keys = [_page_keys(r) for r in group]
        keys = keys[0] if len(keys) == 1 else jnp.concatenate(keys, axis=0)
        col = _scale_rows_to_column([c for _, c in pairs]) if folded else None
        if folded and SCALE_KEYS:
            keys = keys * col.astype(jnp.bfloat16)
        s = jax.lax.dot_general(
            keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        if folded and not SCALE_KEYS:
            s = s * col
        elif scale_row is not None:
            cols = [
                _scale_column(r[0], scale_row)
                for r in scale_refs[first : first + pages_per_dot]
            ]
            s = s * (cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=0))
        maxima.append(jnp.max(s.reshape(len(group), page_size, block_q), axis=1))
    o_ref[...] = maxima[0] if len(maxima) == 1 else jnp.concatenate(maxima, axis=0)


def _page_tokens(rows_per_page: int, dtype) -> int:
    """Tokens per page: 4-bit (uint8) caches hold 8 per 32-bit word (a folded
    page's extra word row holds scales, not tokens)."""
    if dtype == jnp.uint8:
        return msa_fp4.data_rows(rows_per_page) * msa_fp4.TOKENS_PER_WORD
    return rows_per_page * (4 // jnp.dtype(dtype).itemsize)


# Pages per dot in `_kernel_v3`: 2 x 16 pages, paired along the contraction.
_PAGES_PER_CHUNK = 32
# `_kernel_v3` issues the ahead group's descriptors during its first chunks.
_DMA_ISSUE_CHUNKS = 2
# Page groups in `_kernel_v3`'s DMA ring.
_RING_SLOTS = 3


class DecodeOpt(NamedTuple):
    """`TPU_MINIMAX_M3_INDEXER_DECODE_OPT` for the transposed (decode) class."""

    issue: bool = False  # full groups start their copies without per-page checks
    slots: int = _RING_SLOTS  # page groups in the DMA ring
    quad: bool = False  # fp8 queries, four pages per MXU row
    # A page copy below the group's live pages reads its block-table entry
    # unclamped (it is inside the request's row).
    noclamp: bool = False
    # Bodies specialised on whether the ahead group is full: a full one starts
    # its copies unpredicated, in the same straight-line body.
    full: bool = False
    # The last scored chunk of a body runs as dots of this many pages, so the
    # maxima left after the body's last matmul are a few pages' (0: off).
    tail: int = 0
    # One body per count of scored chunks (1 .. all), not per power of two:
    # a group's partial last chunks are not scored.
    every_count: bool = False
    # Score token-major with the query tile as the stationary MXU operand
    # (pushed once per chunk) and the pages' keys streamed through it,
    # instead of pushing every page's keys as MXU weights; slot scales are
    # applied in f32 after the dot, as in the transposed scores
    # (``tm``: keys widened to bf16, ``tm8``: keys as fp8, mixed dot).
    token_scores: int = 0
    # The maxima of a body's last scored chunk are taken in the next grid
    # step's body (its scores wait in VMEM), where the lane maxima overlap
    # that body's matmuls instead of draining after the last one; the
    # scores stay resident in VMEM for the whole call.
    defer: bool = False
    # Token-major transposed tiles: each page's maxima by one cross-lane max
    # per score vreg, selected into the group's 128-lane output block
    # (`_fold_maxima_lanes`) instead of `_maxima_t`'s relayout.
    lanes: bool = False


_DECODE_OPT_LOGGED = False


def decode_opt() -> DecodeOpt:
    """Parse `TPU_MINIMAX_M3_INDEXER_DECODE_OPT` (read through envs, at trace
    time): comma-separated ``issue``, ``slots<N>``, ``quad``."""
    global _DECODE_OPT_LOGGED
    from vllm_torchtpu import envs

    spec = envs.TPU_MINIMAX_M3_INDEXER_DECODE_OPT
    opt = DecodeOpt()
    # "+" separates items too: EXTRA_SERVER_ENV-style lists split on commas.
    for item in filter(None, (x.strip() for x in spec.replace("+", ",").split(","))):
        if item == "issue":
            opt = opt._replace(issue=True)
        elif item == "quad":
            opt = opt._replace(quad=True)
        elif item == "noclamp":
            opt = opt._replace(noclamp=True)
        elif item == "full":
            opt = opt._replace(full=True)
        elif item == "counts":
            opt = opt._replace(every_count=True)
        elif item == "defer":
            opt = opt._replace(defer=True)
        elif item == "lanes":
            opt = opt._replace(lanes=True)
        elif item in ("tm", "tm8"):
            opt = opt._replace(token_scores=16 if item == "tm" else 8)
        elif item.startswith("tail") and item[4:] in ("4", "8", "16"):
            opt = opt._replace(tail=int(item[4:]))
        elif item.startswith("slots") and item[5:].isdigit() and int(item[5:]) >= 2:
            opt = opt._replace(slots=int(item[5:]))
        else:
            raise ValueError(
                f"TPU_MINIMAX_M3_INDEXER_DECODE_OPT: unknown item {item!r} in {spec!r}"
            )
    if spec and not _DECODE_OPT_LOGGED:
        _DECODE_OPT_LOGGED = True
        logger.info(
            "TPU_MINIMAX_M3_INDEXER_DECODE_OPT=%s on: MSA indexer decode tiles run "
            "issue=%s slots=%d quad=%s noclamp=%s full=%s tail=%d counts=%s tm=%d "
            "defer=%s lanes=%s.",
            spec,
            opt.issue,
            opt.slots,
            opt.quad,
            opt.noclamp,
            opt.full,
            opt.tail,
            opt.every_count,
            opt.token_scores,
            opt.defer,
            opt.lanes,
        )
    return opt


# Folded 4-bit and fp8 tiles of up to this many rows score transposed
# (`_chunk_maxima_t`, maxima across lanes); larger ones score token-major
# (maxima across sublanes), a folded page with its slot scales folded into
# the keys.
_TRANSPOSED_MAX_ROWS = 64


def _scores_transposed(block_q: int, rows_per_page: int, dtype) -> bool:
    """Whether `_kernel_v3` scores `block_q`-row tiles of these pages with
    `_chunk_maxima_t`."""
    fp8 = dtype == jnp.float8_e4m3fn and FP8_MXU_KEYS
    folded = msa_fp4.scales_in_page(rows_per_page, dtype)
    return (folded or fp8) and block_q <= _TRANSPOSED_MAX_ROWS


def _paired_query(q: jax.Array) -> jax.Array:
    """``[Bq, D]`` -> ``[[q, 0], [0, q]]`` of ``[2 Bq, 2 D]``."""
    zeros = jnp.zeros_like(q)
    return jnp.concatenate(
        [
            jnp.concatenate([q, zeros], axis=1),
            jnp.concatenate([zeros, q], axis=1),
        ],
        axis=0,
    )


def _chunk_maxima(q_pair, page_keys, num_pages, scale_cols=None) -> jax.Array:
    """``[num_pages, Bq]`` block maxima of `num_pages` pages against a query
    tile given as `_paired_query`.

    Pages ``j`` and ``h + j`` (``h = num_pages / 2``) share one MXU row,
    ``[k_j | k_h+j] @ q_pair^T``: both are scored with one contraction of
    ``2 D``, the MXU depth at D = 128. `page_keys(j)` is page ``j``'s bf16
    ``[S, D]`` keys (or fp8 keys with an fp8 `q_pair`). `scale_cols`
    (``[S, num_pages]`` f32, unscaled 4-bit keys) scales each key's score
    after the dot.
    """
    half = num_pages // 2
    block_q = q_pair.shape[0] // 2
    keys = [
        jnp.concatenate([page_keys(j), page_keys(half + j)], axis=1)
        for j in range(half)
    ]
    page_size = keys[0].shape[0]
    s = jax.lax.dot_general(
        jnp.concatenate(keys, axis=0),
        q_pair,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [half * S, 2 Bq]
    if scale_cols is not None:
        left, right = (
            jnp.concatenate([scale_cols[:, p : p + 1] for p in pages], axis=0)
            for pages in (range(half), range(half, num_pages))
        )  # [half * S, 1] each: page j's and page half + j's token scales
        col = jax.lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = s * jnp.where(col < block_q, left, right)
    m = jnp.max(s.reshape(half, page_size, 2 * block_q), axis=1)
    return jnp.concatenate([m[:, :block_q], m[:, block_q:]], axis=0)


# Keys enter the MXU as fp8 in the transposed path: fp8 caches as stored and
# 4-bit keys converted exactly (|k| <= 8), half the bytes of bf16 keys per
# transposed weight push, and no fp8 -> bf16 unpack (AOT, 128 pages at 8 rows:
# fp8 2,171 -> 819 bundles, 4-bit 1,323 -> 1,008). Products and their f32
# sums are those of bf16 keys.
FP8_MXU_KEYS = True


def _transposed_keys(buf, p, live, dtype, folded, masked):
    """Page `p` of `buf` as ``[S, D]`` MXU keys: 4-bit keys unscaled, fp8 keys
    of a page past the request's end (`masked`) zeroed, as stale VMEM may hold
    NaN, which would reach the paired page's scores through the zero half of
    the paired query."""
    rows = buf.shape[1]
    key_dtype = jnp.float8_e4m3fn if FP8_MXU_KEYS else jnp.bfloat16
    if folded:
        k = pltpu.bitcast(buf[p, : rows - 1], jnp.int4)
    else:
        w = buf[p]
        k = pltpu.bitcast(jnp.where(p < live, w, 0) if masked else w, dtype)
    return k if k.dtype == key_dtype else k.astype(key_dtype)


def _chunk_maxima_t(
    q_pair,
    buf,
    first,
    chunk,
    live,
    issue,
    *,
    dtype,
    folded,
    reduce=True,
    lanes_acc=None,
    lane_base=0,
) -> jax.Array:
    """``[Bq, chunk]`` block maxima of the pages ``buf[first : first + chunk]``
    with the scores computed transposed, ``[2 Bq, tokens]``: the MXU result
    is ``2 Bq`` rows rather than ``chunk * S / 2``. A folded 4-bit page's
    per-token scales are a lane vector (the page's last word row), applied by
    a sublane broadcast. `issue(j)` runs among the keys of page pair ``j``.
    """
    rows = buf.shape[1]
    half = chunk // 2
    block_q = q_pair.shape[0] // 2
    keys, scale_rows = [], [[], []]
    for j in range(half):
        issue(j)
        pair = []
        for side, p in enumerate((first + j, first + half + j)):
            # Only the right page of a pair can be past the end while its
            # partner is not.
            pair.append(_transposed_keys(buf, p, live, dtype, folded, side == 1))
            if folded:
                s = pltpu.bitcast(buf[p, rows - 1 :], jnp.float32)  # [1, S]
                # Stale VMEM past the request's end, unwritten slots: scale 0.
                if lanes_acc is not None:
                    # A page past the request's end only reaches its own
                    # (never read) maxima: its int4 keys are finite, so the
                    # zero half of q_pair keeps its partner's scores exact.
                    scale_rows[side].append(jnp.where(jnp.isfinite(s), s, 0.0))
                else:
                    scale_rows[side].append(
                        jnp.where(jnp.isfinite(s) & (p < live), s, 0.0)
                    )
        keys.append(jnp.concatenate(pair, axis=1))
    page_size = keys[0].shape[0]
    s = jax.lax.dot_general(
        q_pair,
        jnp.concatenate(keys, axis=0),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [2 Bq, half * S]
    if folded:
        left, right = (jnp.concatenate(r, axis=1) for r in scale_rows)
        if lanes_acc is not None:
            # Rows < Bq hold the left pages' scores: whole vregs (Bq % 8 == 0).
            s = jnp.concatenate([s[:block_q] * left, s[block_q:] * right], axis=0)
        else:
            row = jax.lax.broadcasted_iota(jnp.int32, s.shape, 0)
            s = s * jnp.where(row < block_q, left, right)
    if not reduce:
        return s
    if lanes_acc is not None:
        return _fold_maxima_lanes(lanes_acc, s, half, page_size, base=lane_base)
    return _maxima_t(s, half, page_size)


def _maxima_t(s, half, page_size) -> jax.Array:
    """`_chunk_maxima_t`'s ``[2 Bq, half * S]`` scores -> ``[Bq, 2 half]``."""
    block_q = s.shape[0] // 2
    m = jnp.max(s.reshape(2 * block_q, half, page_size), axis=2)  # [2 Bq, half]
    return jnp.concatenate([m[:block_q], m[block_q:]], axis=1)


def _fold_maxima_lanes(acc, s, half, page_size, base=0, mod=None) -> jax.Array:
    """`_maxima_t` without its relayout: each page's ``[Bq, S]`` score block
    is reduced across its lanes (one cross-lane max per vreg) and the column
    is selected into lane ``base + p`` of `acc` (``[Bq, W]``; with `mod`,
    every lane ``l`` with ``l % mod == base + p``). `_maxima_t`'s reshape
    to ``[2 Bq, half, S]`` transposes rows and pages through sublane
    shuffles (~5 per page) and moves the maxima from sublanes to lanes."""
    block_q = s.shape[0] // 2
    lane = jax.lax.broadcasted_iota(jnp.int32, acc.shape, 1)
    if mod is not None:
        lane = lane % mod
    for j in range(half):
        for side in range(2):
            v = s[
                side * block_q : (side + 1) * block_q,
                j * page_size : (j + 1) * page_size,
            ]
            col = jnp.max(v, axis=1, keepdims=True)  # [Bq, 1]
            acc = jnp.where(lane == base + side * half + j, col, acc)
    return acc


def _block_diag_query(q: jax.Array, n: int) -> jax.Array:
    """``[Bq, D]`` -> block-diagonal ``[n Bq, n D]`` (n copies of q)."""
    zeros = jnp.zeros_like(q)
    return jnp.concatenate(
        [
            jnp.concatenate([q if i == k else zeros for k in range(n)], axis=1)
            for i in range(n)
        ],
        axis=0,
    )


def _chunk_maxima_t4(q_quad, buf, first, chunk, live, issue) -> jax.Array:
    """`_chunk_maxima_t` for folded 4-bit pages and fp8 queries with four
    pages per MXU row: pages ``first + i * chunk / 4 + j`` (i = 0..3) share
    row ``j``, ``[k_0 | k_1 | k_2 | k_3] @ q_quad^T`` with a contraction of
    ``4 D``. int4 keys are exact in fp8; past-the-end pages get scale 0."""
    rows = buf.shape[1]
    quarter = chunk // 4
    block_q = q_quad.shape[0] // 4
    keys, scale_rows = [], [[], [], [], []]
    for j in range(quarter):
        issue(2 * j)
        issue(2 * j + 1)
        parts = []
        for side in range(4):
            p = first + side * quarter + j
            parts.append(
                pltpu.bitcast(buf[p, : rows - 1], jnp.int4).astype(jnp.float8_e4m3fn)
            )
            s = pltpu.bitcast(buf[p, rows - 1 :], jnp.float32)  # [1, S]
            scale_rows[side].append(jnp.where(jnp.isfinite(s) & (p < live), s, 0.0))
        keys.append(jnp.concatenate(parts, axis=1))
    page_size = keys[0].shape[0]
    s = jax.lax.dot_general(
        q_quad,
        jnp.concatenate(keys, axis=0),
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # [4 Bq, quarter * S]
    sc = [jnp.concatenate(r, axis=1) for r in scale_rows]  # [1, quarter * S] each
    row = jax.lax.broadcasted_iota(jnp.int32, s.shape, 0)
    scale = jnp.where(
        row < block_q,
        sc[0],
        jnp.where(row < 2 * block_q, sc[1], jnp.where(row < 3 * block_q, sc[2], sc[3])),
    )
    s = s * scale
    m = jnp.max(s.reshape(4 * block_q, quarter, page_size), axis=2)  # [4 Bq, quarter]
    return jnp.concatenate(
        [m[i * block_q : (i + 1) * block_q] for i in range(4)], axis=1
    )


def _v3_chunk(pages_per_step: int, block_q: int, page_size: int) -> int:
    """Pages per dot of `_kernel_v3`: the f32 ``[chunk / 2 * S, 2 Bq]``
    product within twice `_MAX_SCORE_ELEMS`."""
    chunk = min(_PAGES_PER_CHUNK, pages_per_step)
    while chunk > 2 and chunk * page_size * block_q > 2 * _MAX_SCORE_ELEMS:
        chunk //= 2
    return chunk


def _kernel_v3(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    seq_pages_ref,  # [max_num_seqs] pages of each request
    # In
    q_ref,  # [Bq, D]
    cache_ref,  # [pages, S / p, p, D] index cache, left in HBM
    # Out
    o_ref,  # [G, Bq], or [Bq, G] if `transposed`
    # Scratch
    buf,  # [R, G, S / p, D] ring of R groups of pages, as 32-bit words
    sems,  # DMA semaphores [R], one per ring slot
    *defer_refs,  # with `defer`: pending scores [2 Bq, chunk / 2 * S], SMEM [2]
    defer=False,
    transposed=False,
    fp8_q=False,
    fast_issue=False,
    quad=False,
    noclamp=False,
    full_bodies=False,
    tail=0,
    every_count=False,
    token_scores=0,
    lane_maxima=False,
):
    """`_kernel_v2` with the pages fetched by a hand-written DMA ring of
    `_RING_SLOTS` groups: one scalar address per page and no copy for pages
    past the request's end (their scores are never read). A step issues the
    group `_RING_SLOTS - 1` steps ahead: with one group ahead, each step
    waited out its pages' DMA latency (~1.3 us per step at 64 or 128 pages,
    bench j-9956cd34).

    Each ring slot has straight-line bodies (static buffers, so the
    scheduler sees the ahead group's copies do not touch the pages being
    scored), and the ahead group's descriptors are issued among this
    group's matmuls. A wait between chunks would drain the MXU pipeline, so
    all of the group's pages are waited for first.

    A folded 4-bit page (`msa_fp4.scales_in_page`) is ``S / 8`` word rows of
    int4 keys and one row of f32 slot scales (`_chunk_maxima_t`).
    """
    slots, pages_per_step, rows, head_dim = buf.shape
    ahead = slots - 1
    block_q = q_ref.shape[0]
    folded = msa_fp4.scales_in_page(rows, cache_ref.dtype)
    page_size = _page_tokens(rows, cache_ref.dtype)
    chunk = _v3_chunk(pages_per_step, block_q, page_size)
    num_keys = page_table_lib.num_keys(block_tables_ref)
    pages_per_seq = num_keys // seq_pages_ref.shape[0]
    step, num_steps = pl.program_id(0), pl.num_programs(0)
    # The pages as 32-bit words: a dense VMEM layout, so each key vreg is one
    # load (a packed [S / p, p, D] page tiles VMEM in (p, D) tiles).
    words = cache_ref.bitcast(jnp.uint32).reshape(cache_ref.shape[0], rows, head_dim)

    def group(w):
        """(first block-table entry, live pages) of work item `w`."""
        entry = work_ref[w]
        seq = tile_seq_ref[entry >> _TILE_SHIFT] & _SEQ_MASK
        first = (entry & _GROUP_MASK) * pages_per_step
        return seq * pages_per_seq + first, jnp.maximum(seq_pages_ref[seq] - first, 1)

    # A group's pages in one half of a packed table: one half test per group.
    run_safe = page_table_lib.run_safe(block_tables_ref, pages_per_seq, pages_per_step)

    def page(base, g, run=None):
        if run_safe:
            if run is None:
                run = page_table_lib.run_start(block_tables_ref, base)
            return page_table_lib.load_run(block_tables_ref, run, g)
        return page_table_lib.load(block_tables_ref, base + g)

    def copy(base, g, slot, run=None):
        if pages_per_seq % pages_per_step and not noclamp:
            # Keep the last group's block-table reads in its request's row.
            first = base % pages_per_seq
            col = jnp.minimum(first + g, pages_per_seq - 1) - first
            index = page_table_lib.load(block_tables_ref, base + col)
        else:
            index = page(base, g, run)
        return pltpu.make_async_copy(
            words.at[index],
            buf.at[slot, g],
            sems.at[slot],
        )

    def wait_pages(live, slot):
        """Wait for the slot's `live` page copies: the slot's semaphore counts
        their bytes, so one wait per set bit of `live` waits for them all
        (a wait per page cost two serial bundles each)."""
        live = jnp.minimum(live, pages_per_step)
        if fast_issue:
            full = live >= pages_per_step
            pl.when(full)(fake_copy(pages_per_step, slot).wait)
        n = 1
        while n <= pages_per_step:
            cond = (live & n) != 0
            if fast_issue:
                cond = cond & jnp.logical_not(full)
            pl.when(cond)(fake_copy(n, slot).wait)
            n *= 2

    def fake_copy(n, slot):
        """A descriptor of `n` pages' bytes on the slot's semaphore, to wait on."""
        return pltpu.make_async_copy(
            words.at[pl.ds(0, n)], buf.at[slot, pl.ds(0, n)], sems.at[slot]
        )

    def start(base, live, pages, slot, full=False):
        """Start the copies of a group's `pages` that are below its `live`
        pages, alternate pages on the second DMA queue (priority 1). With
        `full` (static) every page is live: no predicates."""
        # The group's table run, set up once for all its page reads.
        run = page_table_lib.run_start(block_tables_ref, base) if run_safe else None
        if full:
            for g in pages:
                copy(base, g, slot, run).start(priority=g % 2)
            return
        if fast_issue:
            full = live >= pages_per_step

            @pl.when(full)
            def _():
                # A full group: every page is below the request's last page,
                # inside its block-table row.
                for g in pages:
                    pltpu.make_async_copy(
                        words.at[page(base, g, run)],
                        buf.at[slot, g],
                        sems.at[slot],
                    ).start(priority=g % 2)

            for g in pages:
                pl.when((g < live) & jnp.logical_not(full))(
                    functools.partial(copy(base, g, slot, run).start, priority=g % 2)
                )
            return
        for g in pages:
            pl.when(g < live)(
                functools.partial(copy(base, g, slot, run).start, priority=g % 2)
            )

    def ahead_group(k):
        """(first entry, live pages) of work item `k`; none past the grid."""
        k_base, k_live = group(jnp.minimum(k, num_steps - 1))
        return k_base, jnp.where(k < num_steps, k_live, 0)

    @pl.when(step == 0)
    def _():
        for k in range(ahead):
            start(*ahead_group(k), range(pages_per_step), k)
        if defer:
            entry0 = work_ref[0]
            defer_refs[0][...] = jnp.zeros(defer_refs[0].shape, jnp.float32)
            defer_refs[1][0] = (entry0 >> _TILE_SHIFT) * block_q
            defer_refs[1][1] = (entry0 & _GROUP_MASK) * pages_per_step
            defer_refs[1][2] = 0

    base, live = group(step)
    next_base, next_live = ahead_group(step + ahead)
    if defer:
        # o_ref holds the whole [tiles * Bq, pages] output.
        pend_ref, pend_pos = defer_refs
        entry = work_ref[step]
        row0 = pl.multiple_of((entry >> _TILE_SHIFT) * block_q, _ROW_ALIGN)
        col0 = pl.multiple_of((entry & _GROUP_MASK) * pages_per_step, pages_per_step)
    if quad:
        q = q_ref[...].astype(jnp.float32)
        amax = jnp.max(jnp.abs(q), axis=1, keepdims=True)  # [Bq, 1]
        amax = jnp.where(amax > 0, amax, 1.0)
        q_pair = _block_diag_query(
            (q * (_FP8_Q_MAX / amax)).astype(jnp.float8_e4m3fn), 4
        )
        row_scale = amax * (1.0 / _FP8_Q_MAX)  # [Bq, 1]
    elif fp8_q:
        q = q_ref[...].astype(jnp.float32)
        amax = jnp.max(jnp.abs(q), axis=1, keepdims=True)  # [Bq, 1]
        amax = jnp.where(amax > 0, amax, 1.0)
        q_pair = _paired_query((q * (_FP8_Q_MAX / amax)).astype(jnp.float8_e4m3fn))
        row_scale = amax * (1.0 / _FP8_Q_MAX)  # [Bq, 1]
    else:
        q_pair = _paired_query(q_ref[...].astype(jnp.bfloat16))
        row_scale = None
    num_chunks = pages_per_step // chunk
    # The ahead group's descriptors, spread over the first chunks' page pairs
    # (in one chunk they crowd its scalar slots, in all chunks they issue
    # late: both slower, benches j-9956cd34 / j-2e0b290e).
    issue_chunks = min(_DMA_ISSUE_CHUNKS, num_chunks)
    per_pair = -(-pages_per_step // (issue_chunks * (chunk // 2)))

    def score_chunk(
        slot,
        first,
        share,
        next_full=False,
        size=None,
        pair_off=0,
        deferred=False,
        acc=None,
    ):
        """Score pages ``[first, first + size)`` (default a chunk), issuing
        the ahead group's descriptors `share` among the first page pairs of
        the chunk (`start`); `pair_off` is the first page pair's index in
        the chunk when a chunk runs as several smaller dots."""
        size = chunk if size is None else size

        def issue(j):
            j += pair_off
            if j < chunk // 2:
                start(
                    next_base,
                    next_live,
                    share[j * per_pair : (j + 1) * per_pair],
                    (slot + ahead) % slots,
                    next_full,
                )

        if quad:
            m = _chunk_maxima_t4(q_pair, buf.at[slot], first, chunk, live, issue)
            m = m * row_scale
            if transposed:
                o_ref[:, pl.ds(first, chunk)] = m
            else:
                o_ref[pl.ds(first, chunk), :] = m.T
            return
        if not token_scores and _scores_transposed(block_q, rows, cache_ref.dtype):
            m = _chunk_maxima_t(
                q_pair,
                buf.at[slot],
                first,
                size,
                live,
                issue,
                dtype=cache_ref.dtype,
                folded=folded,
                reduce=not deferred,
                lanes_acc=None if deferred else acc,
                lane_base=first,
            )  # [Bq, size]
            if acc is not None and not deferred:
                return m  # the group's [Bq, pages_per_step] block so far
            if deferred:
                pend_ref[...] = m
                pend_pos[0] = row0
                pend_pos[1] = col0
                pend_pos[2] = first // chunk
                return
            if defer:
                return m  # the body writes the group's block
            if row_scale is not None:
                m = m * row_scale
            if transposed:
                o_ref[:, pl.ds(first, size)] = m
            else:
                o_ref[pl.ds(first, size), :] = m.T
            return
        assert size == chunk, "split chunks need transposed scores"

        if folded:
            # Each page's slot scales as a column of one transpose for the
            # chunk; stale VMEM past the request's end and unwritten slots
            # get scale 0.
            scale_rows = jnp.concatenate(
                [
                    pltpu.bitcast(buf[slot, first + j, rows - 1 :], jnp.float32)
                    for j in range(chunk)
                ],
                axis=0,
            )  # [chunk, S]
            page = first + jax.lax.broadcasted_iota(jnp.int32, scale_rows.shape, 0)
            scale_cols = jnp.where(
                jnp.isfinite(scale_rows) & (page < live), scale_rows, 0.0
            ).T  # [S, chunk]
            if not fp8_q and not token_scores:
                scale_cols = scale_cols.astype(jnp.bfloat16)

        def page_keys(j):
            issue(j)
            if folded:
                k = pltpu.bitcast(buf[slot, first + j, : rows - 1], jnp.int4)
                if fp8_q or token_scores == 8:  # exact (|k| <= 8); scaled after the dot
                    return k.astype(jnp.float8_e4m3fn)
                if token_scores:  # exact; scaled after the dot
                    return k.astype(jnp.bfloat16)
                return k.astype(jnp.bfloat16) * scale_cols[:, j : j + 1]
            # A page past the request's end was never copied: its stale words
            # may be NaN or inf, which would reach its pair's score through
            # the zero half of q_pair.
            w = jnp.where(first + j < live, buf[slot, first + j], 0)
            k = pltpu.bitcast(w, cache_ref.dtype)
            return k if fp8_q else k.astype(jnp.bfloat16)

        m = _chunk_maxima(
            q_pair,
            page_keys,
            chunk,
            scale_cols if folded and (fp8_q or token_scores) else None,
        )
        if row_scale is not None:
            m = m * row_scale.T
        if transposed:
            o_ref[:, pl.ds(first, chunk)] = m.T
        else:
            o_ref[pl.ds(first, chunk), :] = m

    def flush_pending():
        """Write the pending chunk's maxima into its 128-lane group block
        (a dynamic lane offset must be a multiple of 128): read, select the
        chunk's lanes, write back."""
        width = min(128, pages_per_step)
        if lane_maxima:
            # The chunk's maxima at every chunk position of the 128-lane block.
            m = _fold_maxima_lanes(
                jnp.zeros((block_q, width), jnp.float32),
                pend_ref[...],
                chunk // 2,
                page_size,
                mod=chunk,
            )
        else:
            m = _maxima_t(pend_ref[...], chunk // 2, page_size)  # [Bq, chunk]
            m = jnp.concatenate([m] * (width // chunk), axis=1) if width > chunk else m
        rows_ = pl.ds(pl.multiple_of(pend_pos[0], _ROW_ALIGN), block_q)
        lanes = pl.ds(pl.multiple_of(pend_pos[1] + 0, width), width)
        lane = jax.lax.broadcasted_iota(jnp.int32, m.shape, 1)
        hit = lane // chunk == pend_pos[2] % (width // chunk)
        # The chunk's 128-lane block within the group.
        blk = pend_pos[2] // (width // chunk)
        lanes = pl.ds(pl.multiple_of(pend_pos[1] + blk * width, width), width)
        o_ref[rows_, lanes] = jnp.where(hit, m, o_ref[rows_, lanes])

    share = per_pair * (chunk // 2)
    tail_split = (
        0 < tail < chunk
        and chunk % tail == 0
        and not quad
        and not token_scores
        and _scores_transposed(block_q, rows, cache_ref.dtype)
    )

    def body(slot, scored, next_full=False):
        """Score the group's first `scored` chunks; the rest are past the
        request's last page (their scores are never read) and only issue
        their share of the ahead group's descriptors."""
        wait_pages(live, slot)
        done = []
        acc = jnp.zeros((block_q, pages_per_step), jnp.float32) if lane_maxima else None
        if defer:
            # The previous step's last chunk (a dummy at step 0: chunk 0's
            # own maxima overwrite it below).
            flush_pending()
        for c in range(num_chunks):
            pages = range(
                min(c * share, pages_per_step), min((c + 1) * share, pages_per_step)
            )
            if defer and c == scored - 1 and c > 0:
                score_chunk(slot, c * chunk, pages, next_full, deferred=True)
            elif c < scored - 1 or (c == scored - 1 and not tail_split):
                m = score_chunk(slot, c * chunk, pages, next_full, acc=acc)
                if lane_maxima:
                    acc = m
                elif defer:
                    done.append(m)
            elif c == scored - 1:
                # The last scored chunk as `tail`-page dots: the maxima
                # still to do after its last matmul are one small dot's.
                for off in range(0, chunk, tail):
                    score_chunk(
                        slot,
                        c * chunk + off,
                        pages,
                        next_full,
                        size=tail,
                        pair_off=off // 2,
                    )
            else:
                start(next_base, next_live, pages, (slot + ahead) % slots, next_full)
        if lane_maxima:
            # Lanes of chunks past the request's end (never read) and of the
            # deferred chunk (filled in by the next step's flush) stay zero.
            if defer:
                o_ref[pl.ds(row0, block_q), pl.ds(col0, pages_per_step)] = acc
            else:
                o_ref[...] = acc
        elif defer:
            # Chunks scored here (the deferred one and those past the
            # request's end are filled in later / never read).
            blk = jnp.concatenate(
                done
                + [jnp.zeros((block_q, pages_per_step - len(done) * chunk), jnp.float32)],
                axis=1,
            ) if len(done) * chunk < pages_per_step else jnp.concatenate(done, axis=1)
            o_ref[pl.ds(row0, block_q), pl.ds(col0, pages_per_step)] = blk

    # One straight-line body per slot and count of scored chunks, so that a
    # chunk's matmuls overlap the previous chunk's maxima.
    counts = sorted({num_chunks} | {2**k for k in range(1, 8) if 2**k < num_chunks})
    if every_count:
        counts = list(range(1, num_chunks + 1))
    live_chunks = jnp.minimum(-(-live // chunk), num_chunks)
    next_is_full = next_live >= pages_per_step
    for slot in range(slots):
        for i, n in enumerate(counts):
            lo = counts[i - 1] if i else 0
            cond = (step % slots == slot) & (live_chunks > lo) & (live_chunks <= n)
            if not full_bodies:
                pl.when(cond)(functools.partial(body, slot, n))
                continue
            pl.when(cond & next_is_full)(functools.partial(body, slot, n, True))
            pl.when(cond & jnp.logical_not(next_is_full))(
                functools.partial(body, slot, n)
            )
    if defer:
        pl.when(step == num_steps - 1)(flush_pending)


class FusedSelect(NamedTuple):
    """A tile class's local top-k picked inside its score kernel
    (TPU_MINIMAX_M3_INDEXER_FUSED_TOPK, kernel v3, column-major scores).

    The fields are `msa_dcp.local_candidates`' arguments; the selection runs
    the ops of its pretransposed kernel (`msa_dcp._candidates_kernel_t`) on
    the tile's ``[pages, block_q]`` scores, kept in VMEM over the tile's page
    groups, so the picks are bit-identical and the scores never reach HBM.
    """

    topk: int
    block_size: int
    init_blocks: int
    local_blocks: int
    dcp: int
    rank: jax.Array  # int32 [1], traced (one program for every rank)
    # (tile_seq [tiles], tile_row0 [tiles], block_q) -> int32 [tiles * block_q]
    # position of each column's query token
    pos: Callable | None
    widths: Callable  # padded pages -> static scan widths
    unrolled_width: int
    pick_unroll: int
    block_rows: int  # lanes per selection block (one width each)
    forced_local: float
    forced_init: float


def _select_tile(
    scores_ref,  # [P, Bq] f32 VMEM: the tile's block scores
    pos_ref,  # [1, Bq] each column's token position
    widths_ref,  # SMEM [tiles * Bq / sub_rows] scan width of each lane block
    rank_ref,  # SMEM [1]
    score_out_ref,  # [topk, Bq]
    block_out_ref,  # [topk, Bq]
    tile,
    *,
    select: FusedSelect,
    widths: tuple[int, ...],
    sub_rows: int,
):
    """`msa_dcp._candidates_kernel_t` (pretransposed) on each `sub_rows`-lane
    block of the tile: the same masks, picks and tie order."""
    block_q = pos_ref.shape[1]
    num_sub = block_q // sub_rows
    rank = rank_ref[0]
    dcp, topk = select.dcp, select.topk

    def run(j, width):
        lanes = (
            pl.ds(pl.multiple_of(j * sub_rows, sub_rows), sub_rows)
            if num_sub > 1
            else pl.ds(0, block_q)
        )
        num_visible = pos_ref[:, lanes] // select.block_size + 1  # [1, rows]
        s = scores_ref[pl.ds(0, width), lanes]
        cols = jax.lax.broadcasted_iota(jnp.int32, (width, sub_rows), 0)
        blocks = cols * dcp + rank
        visible = blocks < num_visible
        s = jnp.where(visible, s, -jnp.inf)
        s = jnp.where(
            visible & (blocks >= num_visible - select.local_blocks),
            select.forced_local,
            s,
        )
        s = jnp.where(visible & (blocks < select.init_blocks), select.forced_init, s)
        col_ids = cols.astype(jnp.float32)

        def pick(k, s):
            best_score = jnp.max(s, axis=0, keepdims=True)
            best = jnp.min(
                jnp.where(s == best_score, col_ids, float(width)),
                axis=0,
                keepdims=True,
            )
            score_out_ref[pl.ds(k, 1), lanes] = best_score
            block_out_ref[pl.ds(k, 1), lanes] = jnp.where(
                best_score > -jnp.inf, best.astype(jnp.int32) * dcp + rank, -1
            )
            return jnp.where(col_ids == best, -jnp.inf, s)

        if width <= select.unrolled_width:
            for k in range(topk):
                s = pick(k, s)
        else:
            jax.lax.fori_loop(0, topk, pick, s, unroll=select.pick_unroll)

    def lane_block(j):
        width_now = widths_ref[tile * num_sub + j]
        for width in widths:
            pl.when(width_now == width)(functools.partial(run, j, width))

    if num_sub == 1:
        lane_block(0)
    else:

        def loop_body(j, carry):
            lane_block(j)
            return carry

        jax.lax.fori_loop(0, num_sub, loop_body, 0)


def _kernel_v3_select(
    # Scalar prefetch
    work_ref,
    tile_seq_ref,
    block_tables_ref,
    seq_pages_ref,
    widths_ref,  # [tiles * Bq / sub_rows] scan width of each lane block
    rank_ref,  # [1]
    # In
    q_ref,
    cache_ref,
    pos_ref,  # [1, Bq]
    # Out
    score_out_ref,  # [topk, Bq]
    block_out_ref,  # [topk, Bq]
    # Scratch
    buf,
    sems,
    scores_ref,  # [P, Bq] f32: the tile's scores, all its page groups
    *,
    select: FusedSelect,
    widths: tuple[int, ...],
    sub_rows: int,
    **v3_kwargs,
):
    """`_kernel_v3` (column-major scores) writing each group's scores into a
    VMEM block of the whole tile instead of an HBM output; after the tile's
    last group, `_select_tile` picks its top-k. The output blocks follow the
    tile, so they are written back once per tile."""
    pages_per_step = buf.shape[1]
    step, num_steps = pl.program_id(0), pl.num_programs(0)
    entry = work_ref[step]
    col0 = pl.multiple_of((entry & _GROUP_MASK) * pages_per_step, pages_per_step)
    _kernel_v3(
        work_ref,
        tile_seq_ref,
        block_tables_ref,
        seq_pages_ref,
        q_ref,
        cache_ref,
        scores_ref.at[pl.ds(col0, pages_per_step)],
        buf,
        sems,
        **v3_kwargs,
    )
    tile = entry >> _TILE_SHIFT
    next_tile = work_ref[jnp.minimum(step + 1, num_steps - 1)] >> _TILE_SHIFT

    @pl.when((step == num_steps - 1) | (next_tile != tile))
    def _():
        _select_tile(
            scores_ref,
            pos_ref,
            widths_ref,
            rank_ref,
            score_out_ref,
            block_out_ref,
            tile,
            select=select,
            widths=widths,
            sub_rows=sub_rows,
        )


def _fused_select_call(
    select: FusedSelect,
    v3_kwargs: dict,
    *,
    work,
    tile_info,
    tile_seq,
    tile_row0,
    table,
    seq_pages,
    q_rows,
    index_cache,
    q_spec,
    grid,
    slots,
    pages_per_step,
    padded_pages,
    block_q,
    max_tiles,
    interpret,
) -> tuple[jax.Array, jax.Array]:
    """`_kernel_v3_select` over a tile class's work list: ``(picked scores,
    picked blocks)``, ``[topk, max_tiles * block_q]`` each, as
    `msa_dcp.local_candidates` returns them for its pretransposed scores
    (before its transpose). Columns of tiles past the class's live ones are
    not written."""
    _, rows_per_page, _, head_dim = index_cache.shape
    num_cols = max_tiles * block_q
    pos = select.pos(tile_seq, tile_row0, block_q).astype(jnp.int32)
    widths = tuple(select.widths(padded_pages))
    # Lane blocks of `block_rows` (as `local_candidates`' row blocks), or
    # the whole tile if it does not split into them: the width a block
    # scans changes its cost, never its picks.
    sub_rows = select.block_rows if block_q % select.block_rows == 0 else block_q
    num_sub = num_cols // sub_rows
    last_pos = jnp.max(pos.reshape(num_sub, sub_rows), axis=1)
    needed = -(-(last_pos // select.block_size + 1) // select.dcp)
    w = jnp.asarray(widths, jnp.int32)
    row_widths = jnp.min(jnp.where(w >= needed[:, None], w, padded_pages), axis=1)
    topk = select.topk

    def tile_block(w, work, *_):
        return (0, work[w] >> _TILE_SHIFT)

    out_spec = pl.BlockSpec((topk, block_q), tile_block)
    scores_bytes = padded_pages * block_q * 4
    return pl.pallas_call(
        functools.partial(
            _kernel_v3_select,
            select=select,
            widths=widths,
            sub_rows=sub_rows,
            **v3_kwargs,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=grid,
            in_specs=[
                q_spec,
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec((1, block_q), tile_block),
            ],
            out_specs=[out_spec, out_spec],
            scratch_shapes=[
                pltpu.VMEM(
                    (slots, pages_per_step, rows_per_page, head_dim),
                    jnp.uint32,
                ),
                pltpu.SemaphoreType.DMA((slots,)),
                pltpu.VMEM((padded_pages, block_q), jnp.float32),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((topk, num_cols), jnp.float32),
            jax.ShapeDtypeStruct((topk, num_cols), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=scores_bytes + (32 << 20),
        ),
        interpret=interpret,
        name="minimax_m3_msa_block_candidates",
    )(
        work,
        tile_info,
        table,
        seq_pages,
        row_widths,
        select.rank,
        q_rows,
        index_cache,
        pos.reshape(1, num_cols),
    )


_KERNELS = {"v1": _kernel_v1, "v2": _kernel_v2}


def lookup(table: jax.Array, index: jax.Array) -> jax.Array:
    """``table[index]`` for a small per-request table and in-range `index`
    (an out-of-range one reads 0), by a one-hot over its entries: XLA lowers
    a gather from a small table to a compare chain over one scalar operand
    per entry, ~0.6 us each (a [T] lookup into 64 requests' table: ~72 us,
    c64 decode twin j-0a66fa2e)."""
    onehot = index[..., None] == jnp.arange(table.shape[0], dtype=index.dtype)
    if table.dtype == jnp.bool_:
        return jnp.any(onehot & table, axis=-1)
    return jnp.sum(jnp.where(onehot, table, 0), axis=-1, dtype=table.dtype)


def _tiles(cu_q_lens: jax.Array, seq_in_class: jax.Array, block_q: int, max_tiles: int):
    """Query tiles of one tile class.

    A request's tiles start at its first token rounded down to a multiple of
    `_ROW_ALIGN` (the kernel reads query rows at aligned element offsets), so
    a tile may begin with up to `_ROW_ALIGN - 1` rows of the previous request
    and end with rows of the next; those rows' scores are never read.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    start = (cu_q_lens[:-1] // _ROW_ALIGN) * _ROW_ALIGN
    q_lens = jnp.where(seq_in_class, cu_q_lens[1:] - start, 0)
    tiles_per_seq = -(-q_lens // block_q)
    tile_ends = jnp.cumsum(tiles_per_seq)
    tile = jnp.arange(max_tiles, dtype=jnp.int32)
    tile_seq = jnp.minimum(
        jnp.searchsorted(tile_ends, tile, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    first_tile = tile_ends - tiles_per_seq
    tile_row0 = (
        lookup(start, tile_seq) + (tile - lookup(first_tile, tile_seq)) * block_q
    )
    return tile_seq, tile_row0, first_tile, start, tile_ends[-1]


class ClassScores(NamedTuple):
    """One tile class's scores (`_class_scores`) and where its rows sit."""

    scores: jax.Array  # [P, tiles * Bq], or [tiles * Bq, P] if token-major
    col: jax.Array  # [T] each query row's column
    tile_seq: jax.Array  # [tiles] request of each tile
    tile_row0: jax.Array  # [tiles] query row of each tile's first column
    num_tiles: jax.Array  # tiles in use
    first_tile: jax.Array  # [max_num_seqs] each request's first tile
    tile_start: jax.Array  # [max_num_seqs] query row of its first tile


def _class_scores(
    index_q,
    index_cache,
    seq_lens,
    block_tables,
    cu_q_lens,
    seq_in_class,
    *,
    block_q,
    max_tiles,
    pages_per_step,
    kernel,
    interpret,
    token_major=False,
    scale_cache=None,
    scale_row=None,
    q_pad=None,
    fp8_q=False,
    select: FusedSelect | None = None,
):
    """Transposed scores ``[P, tiles * block_q]`` of one tile class, and the
    column of each token (`ClassScores`); `token_major` (kernel v3) returns
    them as ``[tiles * block_q, P]`` instead, so a token's scores are one row.
    The query rows are padded by ``q_pad >= block_q`` rows (default
    `block_q`): classes sharing it share one padded copy.

    With `select` (kernel v3, column-major) the kernel picks each tile's
    local top-k itself: `scores` is then ``(picked scores [topk, tiles *
    block_q] f32, picked blocks [topk, tiles * block_q] i32)``, the
    transposed outputs of `msa_dcp.local_candidates` on the scores."""
    num_tokens, _, head_dim = index_q.shape
    _, rows_per_page, packing, _ = index_cache.shape
    page_size = _page_tokens(rows_per_page, index_cache.dtype)
    fp4 = scale_row is not None
    if fp4 and kernel == "v1":
        raise NotImplementedError("4-bit index caches need kernel v2 or v3")
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    num_groups_per_seq = -(-pages_per_seq // pages_per_step)
    # The kernels' SMEM table (16-bit with TPU_MINIMAX_M3_PT16).
    table = page_table_lib.prefetch(block_tables, index_cache.shape[0])
    padded_pages = num_groups_per_seq * pages_per_step

    tile_seq, tile_row0, first_tile, tile_start, num_tiles = _tiles(
        cu_q_lens, seq_in_class, block_q, max_tiles
    )
    # The kernel reads each tile's query rows straight from the token-ordered
    # queries at an element offset (a gather of [tiles * block_q, D] cost
    # ~0.5 ms per layer on an 8K prefill, twin j-52e2df18). Rows past a
    # request's end read the next request's queries; their columns are never
    # gathered back. f32 so that any row offset is a whole 32-bit row.
    if max_num_seqs > _SEQ_MASK + 1:
        raise NotImplementedError(
            f"at most {_SEQ_MASK + 1} requests, got {max_num_seqs}"
        )
    q_pad = block_q if q_pad is None else q_pad
    q_rows = jnp.pad(index_q[:, 0].astype(jnp.float32), ((0, q_pad), (0, 0)))
    tile_info = (
        jnp.clip(tile_row0, 0, num_tokens // _ROW_ALIGN * _ROW_ALIGN) << _ROW_SHIFT
    ) | tile_seq

    # Work list: every page group of every tile's request.
    seq_pages = -(-seq_lens // page_size)
    groups_per_tile = jnp.where(
        jnp.arange(max_tiles) < num_tiles,
        jnp.maximum(-(-lookup(seq_pages, tile_seq) // pages_per_step), 1),
        0,
    )
    group_ends = jnp.cumsum(groups_per_tile)
    w = jnp.arange(max_tiles * num_groups_per_seq, dtype=jnp.int32)
    work_tile = jnp.minimum(
        jnp.searchsorted(group_ends, w, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_tiles - 1,
    )
    # A one-hot lookup: gathers from these max_tiles-entry tables lowered to
    # compare/select chains over max_tiles HBM scalar operands each, ~0.6 us
    # per operand (20 us for this fusion in the decode twin, j-b312f26a).
    in_tile = work_tile[:, None] == jnp.arange(max_tiles, dtype=jnp.int32)[None, :]
    work_group = w - jnp.sum(
        jnp.where(in_tile, (group_ends - groups_per_tile)[None, :], 0), axis=1
    )
    work = (work_tile << _TILE_SHIFT) | jnp.clip(work_group, 0, _GROUP_MASK)

    def page_block(g):
        def index_map(w, work, tile_info, block_tables, *_):
            entry = work[w]
            seq = tile_info[entry >> _TILE_SHIFT] & _SEQ_MASK
            col = jnp.minimum(
                (entry & _GROUP_MASK) * pages_per_step + g, pages_per_seq - 1
            )
            page = page_table_lib.load(block_tables, seq * pages_per_seq + col)
            return (page, 0, 0, 0)

        return pl.BlockSpec((1, rows_per_page, packing, head_dim), index_map)

    def scale_block(g):
        page_map = page_block(g).index_map
        return pl.BlockSpec(
            (1, scale_cache.shape[1], msa_fp4.SCALE_LANES),
            lambda *args: page_map(*args)[:3],
        )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        # One step even with no tile in the class; its scores are never read.
        grid=(jnp.maximum(group_ends[-1], 1),),
        in_specs=[
            pl.BlockSpec(
                (pl.Element(block_q), pl.Element(head_dim)),
                lambda w, work, tile_info, *_: (
                    pl.multiple_of(
                        tile_info[work[w] >> _TILE_SHIFT] >> _ROW_SHIFT, _ROW_ALIGN
                    ),
                    0,
                ),
            ),
            *(page_block(g) for g in range(pages_per_step)),
            *(scale_block(g) for g in range(pages_per_step) if fp4),
        ],
        out_specs=pl.BlockSpec(
            (pages_per_step, block_q),
            lambda w, work, *_: (work[w] & _GROUP_MASK, work[w] >> _TILE_SHIFT),
        ),
    )
    out_shape = jax.ShapeDtypeStruct((padded_pages, max_tiles * block_q), jnp.float32)
    out_specs = grid_spec.out_specs
    if token_major:
        if kernel != "v3":
            raise NotImplementedError(
                f"token-major scores need kernel v3, got {kernel}"
            )
        out_shape = jax.ShapeDtypeStruct(out_shape.shape[::-1], jnp.float32)
        out_specs = pl.BlockSpec(
            (block_q, pages_per_step),
            lambda w, work, *_: (work[w] >> _TILE_SHIFT, work[w] & _GROUP_MASK),
        )
    params = pltpu.CompilerParams(
        dimension_semantics=("arbitrary",), disable_bounds_checks=kernel == "v3"
    )
    if select is not None and (kernel != "v3" or token_major or fp4):
        raise NotImplementedError(
            "a fused top-k needs kernel v3, column-major scores and pages "
            "with folded scales (or fp8 / bf16 pages)"
        )
    if kernel == "v3":
        if fp4:
            raise NotImplementedError(
                "kernel v3 reads 4-bit pages with folded scales only"
            )
        chunk = _v3_chunk(pages_per_step, block_q, page_size)
        if pages_per_step % chunk or chunk % 2:
            raise NotImplementedError(
                f"kernel v3 pairs pages in chunks of up to {_PAGES_PER_CHUNK}: "
                f"pages_per_step must be an even divisor or a multiple of it, "
                f"got {pages_per_step}"
            )
        dopt = DecodeOpt()
        folded = msa_fp4.scales_in_page(rows_per_page, index_cache.dtype)
        if folded and _scores_transposed(block_q, rows_per_page, index_cache.dtype):
            dopt = decode_opt()
            if dopt.quad and chunk % 4:
                dopt = dopt._replace(quad=False)
            # The resident output must fit VMEM next to the ring.
            if dopt.defer and (
                not token_major
                or select is not None
                or dopt.quad
                or dopt.tail
                or dopt.token_scores
                or out_shape.shape[0] * out_shape.shape[1] * 4 > (8 << 20)
            ):
                dopt = dopt._replace(defer=False)
        defer_scratch = []
        if dopt.defer:
            out_specs = pl.BlockSpec(out_shape.shape, lambda w, *_: (0, 0))
            defer_scratch = [
                pltpu.VMEM((2 * block_q, chunk // 2 * page_size), jnp.float32),
                pltpu.SMEM((3,), jnp.int32),
            ]
            # Two output buffers, the page ring, the pending scores, margin.
            ring = dopt.slots * pages_per_step * rows_per_page * head_dim * 4
            params = pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                disable_bounds_checks=True,
                vmem_limit_bytes=2 * out_shape.shape[0] * out_shape.shape[1] * 4
                + ring
                + (16 << 20),
            )
        v3_kwargs = dict(
            transposed=token_major,
            # fp8 x fp8 only for fp8 / 4-bit pages (keys exact in fp8).
            fp8_q=fp8_q and index_cache.dtype in (jnp.float8_e4m3fn, jnp.uint8),
            fast_issue=dopt.issue,
            quad=dopt.quad,
            noclamp=dopt.noclamp,
            full_bodies=dopt.full,
            tail=dopt.tail,
            every_count=dopt.every_count,
            token_scores=dopt.token_scores,
            defer=dopt.defer,
            lane_maxima=dopt.lanes
            and token_major
            and not (dopt.tail or dopt.quad or dopt.token_scores or fp8_q),
        )
        if select is not None:
            scores_t = _fused_select_call(
                select,
                v3_kwargs,
                work=work,
                tile_info=tile_info,
                tile_seq=tile_seq,
                tile_row0=tile_row0,
                table=table,
                seq_pages=seq_pages,
                q_rows=q_rows,
                index_cache=index_cache,
                q_spec=grid_spec.in_specs[0],
                grid=grid_spec.grid,
                slots=dopt.slots,
                pages_per_step=pages_per_step,
                padded_pages=padded_pages,
                block_q=block_q,
                max_tiles=max_tiles,
                interpret=interpret,
            )
        else:
            scores_t = pl.pallas_call(
                functools.partial(_kernel_v3, **v3_kwargs),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=4,
                    grid=grid_spec.grid,
                    in_specs=[grid_spec.in_specs[0], pl.BlockSpec(memory_space=pl.ANY)],
                    out_specs=out_specs,
                    scratch_shapes=[
                        pltpu.VMEM(
                            (dopt.slots, pages_per_step, rows_per_page, head_dim),
                            jnp.uint32,
                        ),
                        pltpu.SemaphoreType.DMA((dopt.slots,)),
                        *defer_scratch,
                    ],
                ),
                out_shape=out_shape,
                compiler_params=params,
                interpret=interpret,
                name="minimax_m3_msa_block_scores",
            )(work, tile_info, table, seq_pages, q_rows, index_cache)
    else:
        kernel_fn = _KERNELS[kernel]
        if fp4:
            kernel_fn = functools.partial(kernel_fn, scale_row=scale_row)
        scores_t = pl.pallas_call(
            kernel_fn,
            grid_spec=grid_spec,
            out_shape=out_shape,
            compiler_params=params,
            interpret=interpret,
            name="minimax_m3_msa_block_scores",
        )(
            work,
            tile_info,
            table,
            q_rows,
            *([index_cache] * pages_per_step),
            *([scale_cache] * (pages_per_step if fp4 else 0)),
        )

    # Back to token order: token t of request r sits in r's tiles at its offset.
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], tok, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    offset = tok - lookup(tile_start, seq)
    col = (lookup(first_tile, seq) + offset // block_q) * block_q + offset % block_q
    col = jnp.clip(col, 0, max_tiles * block_q - 1)
    return ClassScores(
        scores_t, col, tile_seq, tile_row0, num_tiles, first_tile, tile_start
    )


class _ClassPlan(NamedTuple):
    """`block_scores`' two tile classes. `small()` / `large()` score them
    (`ClassScores` each); `token_major` is the small class's layout."""

    small: Callable
    large: Callable
    small_block_q: int
    large_block_q: int
    large_pages: int
    max_large: int
    is_large: jax.Array
    token_major: bool


def is_prefill_step(num_tokens: int, max_num_seqs: int) -> bool:
    """More rows than a batch of decode requests: `block_scores` scores both
    tile classes unconditionally."""
    return num_tokens > max_num_seqs * _SMALL_TILE_Q


def kv4_large_block_q(index_cache: jax.Array, num_rows: int, default: int) -> int:
    """The large class's tile rows: `TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q`
    for a 4-bit index cache with folded scales (kernel v3, keys scaled to
    bf16 per tile) on a step of at least twice that many query rows (read
    when traced, through `envs`), else `default`."""
    from vllm_torchtpu import envs

    wide = envs.TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q
    folded = msa_fp4.scales_in_page(index_cache.shape[1], index_cache.dtype)
    if not wide or not folded or num_rows < 2 * wide:
        return default
    if wide % _ROW_ALIGN or wide <= _TRANSPOSED_MAX_ROWS:
        # Tiles of up to _TRANSPOSED_MAX_ROWS rows score transposed, scaling
        # f32 scores instead of bf16 keys: other bits.
        raise ValueError(
            f"TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q={wide}: want a multiple "
            f"of {_ROW_ALIGN} above {_TRANSPOSED_MAX_ROWS}"
        )
    global _KV4_LARGE_LOGGED
    if not _KV4_LARGE_LOGGED:
        _KV4_LARGE_LOGGED = True
        logger.info(
            "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q=%d: 4-bit MSA indexer prefill "
            "tiles of %d query rows.",
            wide,
            wide,
        )
    return wide


_KV4_LARGE_LOGGED = False


def class_plan(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    block_q: int = DEFAULT_BLOCK_Q,
    large_block_q: int = DEFAULT_LARGE_BLOCK_Q,
    pages_per_step: int = DEFAULT_PAGES_PER_STEP,
    small_pages_per_step: int = DEFAULT_SMALL_PAGES_PER_STEP,
    kernel: str = DEFAULT_KERNEL,
    large_kernel: str = DEFAULT_LARGE_KERNEL,
    scale_cache: jax.Array | None = None,
    scale_row: int | None = None,
    transposed_tile_q: int = _SMALL_TILE_Q_TRANSPOSED,
    interpret: bool = False,
    fp8_q: bool = False,
) -> _ClassPlan:
    """`block_scores`' tile classes, unscored (arguments as `block_scores`)."""
    num_tokens, num_heads, _ = index_q.shape
    if num_heads != 1:
        raise NotImplementedError(
            f"MSA indexer scores one index head per rank, got {num_heads}."
        )
    if msa_fp4.is_fp4(index_cache) and msa_fp4.scales_in_page(
        index_cache.shape[1], index_cache.dtype
    ):
        scale_cache = scale_row = None  # the pages carry their own scales
    elif msa_fp4.is_fp4(index_cache):
        if scale_cache is None or scale_row is None:
            raise ValueError("a 4-bit index cache needs scale_cache and scale_row")
        if isinstance(scale_cache, msa_fp4.SharedScales):
            raise NotImplementedError(
                "shared scale pages need index pages with folded scales"
            )
    else:
        check_page_dtype(index_cache.shape[2], index_cache.dtype)
        scale_row = None
    large_block_q = kv4_large_block_q(index_cache, num_tokens, large_block_q)
    max_num_seqs = seq_lens.shape[0]
    seq_ids = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    active = seq_ids < num_seqs
    is_large = active & (q_lens > block_q)
    score = functools.partial(
        _class_scores,
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        interpret=interpret,
        scale_cache=scale_cache,
        scale_row=scale_row,
        # One padded f32 copy of the queries for both classes (~32 us per
        # layer each at DCP 8, c64 twin j-0a66fa2e).
        q_pad=max(block_q, large_block_q),
    )
    # Large requests have more than block_q tokens each.
    max_large = min(max_num_seqs, num_tokens // (block_q + 1))
    # Kernel v3 reads 4-bit pages with folded scales only.
    unfolded_fp4 = scale_row is not None
    small_kernel = "v2" if kernel == "v3" and unfolded_fp4 else kernel
    large_kernel = "v2" if large_kernel == "v3" and unfolded_fp4 else large_kernel
    # Token-major scores make the gather back to token order a row gather; a
    # column gather first relayouts the whole padded [P, tiles * block_q] f32
    # block (36 us per layer in the c48 decode twin, j-a281ffb0). Needs
    # lane-aligned [tile, pages_per_step] blocks.
    token_major = small_kernel == "v3" and small_pages_per_step % 128 == 0
    # Token-major tiles of up to _SMALL_TILE_Q rows: decode requests have 8
    # (MTP) to 32 (folded index heads) rows, and a tile of up to 64 rows costs
    # the same as one of 8 (paired scores are 2 x tile lanes). A request of up
    # to block_q rows takes several tiles; none holds more than the batch.
    transposed = scale_row is None and _scores_transposed(
        1, index_cache.shape[1], index_cache.dtype
    )
    tile = (
        min(
            block_q,
            transposed_tile_q if transposed else _SMALL_TILE_Q,
            -(-num_tokens // _ROW_ALIGN) * _ROW_ALIGN,
        )
        if token_major
        else block_q
    )
    # Aligned tile starts add up to _ROW_ALIGN - 1 rows per request.
    tiles_per_request = -(-(block_q + _ROW_ALIGN - 1) // tile)
    small = functools.partial(
        score,
        active & ~is_large,
        block_q=tile,
        max_tiles=min(
            tiles_per_request * max_num_seqs,
            (num_tokens + (_ROW_ALIGN - 1) * max_num_seqs) // tile + max_num_seqs,
        ),
        pages_per_step=small_pages_per_step,
        kernel=small_kernel,
        token_major=token_major,
    )
    large = functools.partial(
        score,
        is_large,
        block_q=large_block_q,
        # +1 per request for the partial last tile, plus the up-to-7 rows
        # each aligned start adds.
        max_tiles=num_tokens // large_block_q
        + max_large
        + -(-(_ROW_ALIGN - 1) * max_large // large_block_q)
        + 1,
        pages_per_step=pages_per_step,
        kernel=large_kernel,
        # fp8 queries on the prefill tiles only: the decode tiles are bound by
        # the key stream into the MXU, not its rate, and gain nothing.
        fp8_q=fp8_q,
    )
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    return _ClassPlan(
        small=small,
        large=large,
        small_block_q=tile,
        large_block_q=large_block_q,
        large_pages=-(-pages_per_seq // pages_per_step) * pages_per_step,
        max_large=max_large,
        is_large=is_large,
        token_major=token_major,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_q",
        "large_block_q",
        "pages_per_step",
        "small_pages_per_step",
        "kernel",
        "large_kernel",
        "scale_row",
        "transposed_tile_q",
        "interpret",
        "fp8_q",
    ),
)
def block_scores(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    block_q: int = DEFAULT_BLOCK_Q,
    large_block_q: int = DEFAULT_LARGE_BLOCK_Q,
    pages_per_step: int = DEFAULT_PAGES_PER_STEP,
    small_pages_per_step: int = DEFAULT_SMALL_PAGES_PER_STEP,
    kernel: str = DEFAULT_KERNEL,
    large_kernel: str = DEFAULT_LARGE_KERNEL,
    scale_cache: jax.Array | None = None,
    scale_row: int | None = None,
    transposed_tile_q: int = _SMALL_TILE_Q_TRANSPOSED,
    interpret: bool = False,
    fp8_q: bool | None = None,
    row_order: jax.Array | None = None,
) -> jax.Array:
    """Block-max index scores ``[T, 1, P]`` (``P`` padded to the page group).

    `row_order` (int ``[N]``): return rows ``row_order`` of that result,
    ``[N, 1, P]``, from the same single gather that lays the kernel's tiles
    out (the caller's own row gather would be a second pass over the scores).

    Requests of at most `block_q` query tokens (decode, short chunks) are
    scored in `block_q`-row tiles; longer prefill chunks in `large_block_q`-row
    tiles, so a request's index keys are read once per large tile rather than
    once per `block_q` rows.

    Args:
      index_q: ``[T, 1, D]`` index queries (one index head per rank).
      index_cache: ``[pages, S / p, p, D]`` key-only cache (bf16 or fp8).
      seq_lens, block_tables, cu_q_lens, num_seqs: ragged batch metadata.
      kernel: ``"v2"`` (bitcast unpack, query latched across pages), ``"v3"``
        (v2 with a hand-written DMA ring for the pages) or the original
        ``"v1"``, kept as the reference arm for benchmarks. A 4-bit cache
        without folded scales runs v2 where v3 is asked for.
      large_kernel: the same choice (and fallback) for the `large_block_q` tiles.
      pages_per_step / small_pages_per_step: pages per grid step of the large
        and the block_q tile classes.
      scale_cache, scale_row: for a 4-bit (uint8) index cache without folded
        scales (`msa_fp4.scales_in_page`),
        the ``[pages, 8, 128]`` f32 scale cache and the row of the index-key
        scales. `index_q` must then be rotated (`msa_fp4.rotate_queries`).
      fp8_q: kernel v3 scores fp8 / 4-bit pages with fp8 queries
        (`INDEXER_V2_ENV`); None reads the env at trace time.
    Pages past a token's own page hold scores of keys it must not see, and
    pages past its request's last page are not written; callers mask both.
    """
    plan = class_plan(
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        num_seqs,
        block_q=block_q,
        large_block_q=large_block_q,
        pages_per_step=pages_per_step,
        small_pages_per_step=small_pages_per_step,
        kernel=kernel,
        large_kernel=large_kernel,
        scale_cache=scale_cache,
        scale_row=scale_row,
        transposed_tile_q=transposed_tile_q,
        interpret=interpret,
        fp8_q=fp8_queries_enabled() if fp8_q is None else fp8_q,
    )
    num_tokens = index_q.shape[0]
    max_num_seqs = seq_lens.shape[0]
    max_large, is_large, token_major = plan.max_large, plan.is_large, plan.token_major
    small, small_col, *_ = plan.small()
    # The classes may pad pages to different group multiples; pad to the wider.
    pages = max(small.shape[1 if token_major else 0], plan.large_pages)

    def small_only():
        col = small_col if row_order is None else small_col[row_order]
        rows = small[col] if token_major else small[:, col].T
        return jnp.pad(rows, ((0, 0), (0, pages - rows.shape[1])))

    if max_large == 0:
        return small_only()[:, None, :]

    def with_large():
        large, large_col, *_ = plan.large()
        small_t = small.T if token_major else small
        tok = jnp.arange(num_tokens, dtype=jnp.int32)
        seq = jnp.minimum(
            jnp.searchsorted(
                cu_q_lens[1:], tok, side="right", method="compare_all"
            ).astype(jnp.int32),
            max_num_seqs - 1,
        )
        # One gather over both classes' columns (XLA fuses the concatenate
        # into it) instead of a transposing gather per class and a select.
        col = jnp.where(lookup(is_large, seq), small_t.shape[1] + large_col, small_col)
        if row_order is not None:
            col = col[row_order]
        small_t, large = (
            jnp.pad(x, ((0, pages - x.shape[0]), (0, 0))) for x in (small_t, large)
        )
        return jnp.concatenate([small_t, large], axis=1)[:, col].T

    if is_prefill_step(num_tokens, max_num_seqs):
        # More rows than a batch of decode requests: a prefill step. A
        # conditional would wait for the gather (offloaded to the SparseCore)
        # and its neighbours' collectives at its end (c4 mixed twin +50 us
        # per layer, j-08c5e333).
        return with_large()[:, None, :]
    # Decode steps have no large request: skip its kernel call and gather.
    return jax.lax.cond(jnp.any(is_large), with_large, small_only)[:, None, :]
