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
"""Static top-k selectors for StreamIndex score buffers.

The QSA (query-sparse attention) indexer of Qwen4Exp (Qwen3.8 Flash) keeps the top-k
4-token blocks of each query row from these buffers.

A selector reads an i32 buffer of IEEE f32 score bits, ``[rows, width]``, and
returns each row's ``k`` highest-scoring column IDs. ``SparseCoreTopK`` runs
on the SparseCore. ``BisectTopK`` is an exact, sort-free TensorCore selector
for chips without a usable SparseCore.
``TensorCoreTopK`` is the ``lax.top_k`` reference.
"""

import abc
import dataclasses
import functools
from typing import ClassVar, Literal, overload

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.sparse_core.support import get_sparse_core_info
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.qwen4_exp.kernels.sparsecore_topk import (
    BITS_NEG_INF,
    LANES,
    sparsecore_topk,
)

logger = init_logger(__name__)

# The SparseCore layout `sparsecore_topk` is written for: it hard-codes 2 cores
# of 16 subcores (32 row groups per wave), 16-lane vectors, and DMA offsets
# aligned to at most a 32-byte granule.
_SC_TOPK_NUM_CORES = 2
_SC_TOPK_NUM_SUBCORES = 16
_SC_TOPK_MAX_DMA_GRANULE_BYTES = 32


def _has_usable_sparse_core() -> bool:
    """Whether this chip's SparseCore matches the layout of `sparsecore_topk`.

    `get_sparse_core_info()` already returns None outside the chips the
    SparseCore kernels are validated on (v5p, v6e, 7, 7x); the 2x16-subcore and
    16-lane check then also filters out the 8-lane SparseCores of v5p and v6e.
    """
    sc_info = get_sparse_core_info()
    return (
        sc_info is not None
        and sc_info.num_cores == _SC_TOPK_NUM_CORES
        and sc_info.num_subcores == _SC_TOPK_NUM_SUBCORES
        and sc_info.num_lanes == LANES
        and sc_info.dma_granule_size_bytes <= _SC_TOPK_MAX_DMA_GRANULE_BYTES
    )


class TopKBackend(abc.ABC):
    """Exact top-k over i32 buffers holding IEEE f32 score bits.

    Only columns below each row's length are eligible, and -inf marks an
    invalid score. Return i32 column IDs with the winners packed before a -1
    suffix; winner order is unspecified. Optional scores use the same i32 f32
    bit representation, -inf in unfilled slots. Implementations must be
    hashable: they are static JIT arguments.

    ``overlaps_scoring``: the selector runs on another engine, so chunk ``g``'s
    selection can overlap chunk ``g + 1``'s scoring under one XLA scheduling
    group.
    """

    overlaps_scoring: ClassVar[bool]

    @overload
    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: Literal[False],
    ) -> jax.Array: ...

    @overload
    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: Literal[True],
    ) -> tuple[jax.Array, jax.Array]: ...

    @overload
    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]: ...

    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        return self._select(
            score_bits,
            k,
            row_lengths,
            scheduling_group_id=scheduling_group_id,
            return_scores=return_scores,
        )

    @abc.abstractmethod
    def _select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]: ...


@dataclasses.dataclass(frozen=True)
class SparseCoreTopK(TopKBackend):
    """The SparseCore selector, overlapping the next chunk's scoring."""

    overlaps_scoring: ClassVar[bool] = True

    def _select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        return sparsecore_topk(
            score_bits,
            k,
            row_lengths=row_lengths,
            write_empty_rows=True,
            scheduling_group_id=scheduling_group_id,
            return_scores=return_scores,
        )


@dataclasses.dataclass(frozen=True)
class TensorCoreTopK(TopKBackend):
    """``lax.top_k`` on the TensorCore: the reference selector.

    XLA lowers ``lax.top_k`` over a wide row to a full sort, so it is much
    slower than ``BisectTopK`` at StreamIndex widths; it stays as the exactness
    reference for ``BisectTopK``. ``lax.top_k`` ranks -0.0 below
    +0.0; both are read as +0.0 here, so signed-zero ties also go to the lowest
    column, as in ``BisectTopK`` and ``torch.topk``.
    """

    overlaps_scoring: ClassVar[bool] = False

    def _select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        del scheduling_group_id
        with jax.named_scope("TensorCoreTopK"):
            return self._top_k(score_bits, k, row_lengths, return_scores)

    def _top_k(self, score_bits, k, row_lengths, return_scores):
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        cols = jnp.arange(scores.shape[1], dtype=jnp.int32)[None, :]
        scores = jnp.where(scores == 0.0, 0.0, scores)  # -0.0 -> +0.0
        scores = jnp.where(cols < row_lengths[:, None], scores, -jnp.inf)
        if scores.shape[1] < k:
            scores = jnp.pad(
                scores, ((0, 0), (0, k - scores.shape[1])), constant_values=-jnp.inf
            )
        values, ids = jax.lax.top_k(scores, k)
        # Sorted descending, so the -inf entries already form the suffix.
        valid = values > -jnp.inf
        ids = jnp.where(valid, ids.astype(jnp.int32), jnp.int32(-1))
        if not return_scores:
            return ids
        # The winners' own score bits (a -0.0 stays -0.0), as BisectTopK returns.
        taken = jnp.take_along_axis(score_bits, jnp.maximum(ids, 0), axis=1)
        return ids, jnp.where(valid, taken, jnp.int32(BITS_NEG_INF))


# Order-preserving u32 key of -inf. Every finite score and +inf maps above it.
_KEY_NEG_INF = 0x007FFFFF
# Columns per packing block. In-block ranks stay <= 256, exact in bf16.
_PACK_BLOCK = 256
# Narrowest width bucket above the all-win bucket, as a multiple of k.
_MIN_BUCKET_K_MULTIPLE = 4


def _round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


def _ordered_keys(score_bits: jax.Array) -> jax.Array:
    """Maps f32 score bits to u32 keys whose unsigned order is the float order.

    -0.0 maps to +0.0's key: the two compare equal, as in ``torch.topk`` (and
    unlike ``lax.top_k`` on TPU), so their ties also go to the lowest column.
    """
    bits = jax.lax.bitcast_convert_type(score_bits, jnp.uint32)
    bits = jnp.where(bits == jnp.uint32(0x80000000), jnp.uint32(0), bits)
    negative = (bits >> 31) == 1
    return jnp.where(negative, ~bits, bits | jnp.uint32(0x80000000))


def _inclusive_block_prefix(totals: jax.Array) -> jax.Array:
    """Row-wise inclusive prefix sum of ``f32 [rows, blocks]`` small integers.

    A triangular matmul instead of ``cumsum``, which XLA lowers poorly on TPU.
    Exact: the inputs are integers <= 256 and the sums stay far below 2**24.
    """
    blocks = jnp.arange(totals.shape[1])
    upper = (blocks[:, None] <= blocks[None, :]).astype(jnp.float32)
    return jnp.einsum("rn,nm->rm", totals, upper, precision=jax.lax.Precision.HIGHEST)


def _block_ranks(mask: jax.Array) -> jax.Array:
    """Inclusive rank of each column among its 256-column block's set columns.

    ``mask`` is bool ``[rows, blocks * 256]``; returns f32 ``[rows, blocks, 256]``.
    """
    num_rows, width = mask.shape
    lane = jnp.arange(_PACK_BLOCK)
    upper = (lane[:, None] <= lane[None, :]).astype(jnp.bfloat16)
    return jnp.einsum(
        "rnb,bc->rnc",
        mask.reshape(num_rows, width // _PACK_BLOCK, _PACK_BLOCK).astype(jnp.bfloat16),
        upper,
        preferred_element_type=jnp.float32,
    )


def _pack_winners(won: jax.Array, k: int, quota: jax.Array) -> jax.Array:
    """Packs each row's winning column IDs into ``k`` slots, ``-1`` after ``quota``.

    Winners come out in column order. Only matmuls, compares and reductions:
    XLA's TPU ``cumsum`` and scatter over wide rows cost more than the sort
    this selector replaces. ``won`` is bool ``[rows, blocks * 256]`` with
    ``quota <= k`` winners per row.
    """
    rank = _block_ranks(won)
    num_blocks = rank.shape[1]
    totals = rank[:, :, -1]
    block_end = _inclusive_block_prefix(totals)
    slots = jnp.arange(k, dtype=jnp.float32)[None, :]
    # Slot j's winner lives in the first block whose inclusive end exceeds j.
    block = jnp.sum(block_end[:, None, :] <= slots[:, :, None], axis=2)
    one_hot = jnp.arange(num_blocks)[None, None, :] == block[:, :, None]
    block_start = jnp.sum(
        jnp.where(one_hot, (block_end - totals)[:, None, :], 0.0), axis=2
    )
    block_rank = jnp.einsum(
        "rkn,rnc->rkc",
        one_hot.astype(jnp.bfloat16),
        rank.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    # The winner's lane is the number of lanes ranked at or below its offset.
    lane = jnp.sum(block_rank <= (slots - block_start)[:, :, None], axis=2)
    ids = block.astype(jnp.int32) * _PACK_BLOCK + lane.astype(jnp.int32)
    return jnp.where(jnp.arange(k)[None, :] < quota[:, None], ids, -1)


def _kth_largest_key(keys: jax.Array, quota: jax.Array, radix_bits: int) -> jax.Array:
    """The ``quota``-th largest u32 key per row: a radix search from the top bit.

    Each step settles ``radix_bits`` bits by counting the keys at or above the
    ``2**radix_bits - 1`` candidate digits; XLA fuses those counts into one pass
    over ``keys``. A row with ``quota == 0`` ends at ``0xffffffff``.
    """
    num_rows = keys.shape[0]

    def refine(step, threshold):
        shift = (32 - radix_bits * (step + 1)).astype(jnp.uint32)
        digit = jnp.zeros((num_rows,), jnp.uint32)
        for d in range(1, 1 << radix_bits):
            trial = threshold | jnp.left_shift(jnp.uint32(d), shift)
            count = jnp.sum(keys >= trial[:, None], axis=1, dtype=jnp.int32)
            # Counts fall as the digit grows: the digit is how many candidates
            # still keep `quota` keys at or above them.
            digit = digit + (count >= quota).astype(jnp.uint32)
        return threshold | jnp.left_shift(digit, shift)

    return jax.lax.fori_loop(
        0, 32 // radix_bits, refine, jnp.zeros((num_rows,), jnp.uint32)
    )


def _lowest_ties(tied: jax.Array, tie_quota: jax.Array) -> jax.Array:
    """The first ``tie_quota`` set columns of each row of bool ``tied``.

    One reduction finds the 256-column block holding the ``tie_quota``-th tie;
    a rank matmul over that block alone finds its column.
    """
    num_rows, width = tied.shape
    num_blocks = width // _PACK_BLOCK
    blocks = tied.reshape(num_rows, num_blocks, _PACK_BLOCK)
    totals = jnp.sum(blocks, axis=2, dtype=jnp.int32)
    tie_end = _inclusive_block_prefix(totals.astype(jnp.float32)).astype(jnp.int32)
    cut_block = jnp.minimum(
        jnp.sum(tie_end < tie_quota[:, None], axis=1, dtype=jnp.int32),
        num_blocks - 1,
    )
    boundary = jnp.take_along_axis(blocks, cut_block[:, None, None], axis=1)
    lane = jnp.arange(_PACK_BLOCK)
    rank = jnp.einsum(
        "rb,bc->rc",
        boundary[:, 0].astype(jnp.bfloat16),
        (lane[:, None] <= lane[None, :]).astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)
    before = jnp.take_along_axis(tie_end - totals, cut_block[:, None], axis=1)
    lane_cut = jnp.sum(rank < tie_quota[:, None] - before, axis=1, dtype=jnp.int32)
    cut = cut_block * _PACK_BLOCK + lane_cut
    cols = jnp.arange(width, dtype=jnp.int32)[None, :]
    return tied & (cols <= cut[:, None]) & (tie_quota > 0)[:, None]


def _bucket_widths(width: int, k: int) -> tuple[int, ...]:
    """Searched score widths of the active-width switch, narrowest first.

    Powers of two from ``4 * k`` (rounded up), then ``width`` itself; a power
    of two within 1.5x of ``width`` is dropped as not worth its own branch.
    Rows no longer than ``k`` take a separate all-win bucket ahead of these.
    """
    widths = []
    bucket = 1 << (_MIN_BUCKET_K_MULTIPLE * k - 1).bit_length()
    while 3 * bucket < 2 * width:
        widths.append(bucket)
        bucket *= 2
    return (*widths, width)


@dataclasses.dataclass(frozen=True)
class BisectTopK(TopKBackend):
    """Exact, sort-free TensorCore top-k for chips without a usable SparseCore.

    ``lax.top_k`` over a wide row lowers to a full sort. Instead, per row:

    1. Map scores to order-preserving u32 keys; ineligible columns get key 0.
    2. Find the k-th largest key with a radix search, ``radix_bits`` bits per
       step (``32 / radix_bits`` passes over the keys).
    3. Take every key above that threshold. If more keys tie at the threshold
       than the row has slots left, the lowest tied columns win, as in
       ``lax.top_k``; rows without such a surplus skip this step.
    4. Pack the winners' column IDs in column order with bf16 block-rank
       matmuls: no ``cumsum``, scatter or sort.

    ``width_buckets``: on calls with at least ``bucket_min_rows`` rows, run
    the search over the narrowest static width bucket that covers
    ``max(row_lengths)`` (``lax.switch``), so short contexts do not pay for a
    score buffer sized to ``max_model_len``; when that maximum is at most
    ``k``, every eligible column wins and no search runs. The switch costs
    about 10% at full width, which pays off for calls with hundreds of rows
    (prefill chunks) but not for small ones, so calls with fewer than
    ``bucket_min_rows`` rows stay flat.
    """

    overlaps_scoring: ClassVar[bool] = False
    radix_bits: int = 2
    width_buckets: bool = True
    bucket_min_rows: int = 128

    def __post_init__(self):
        # 8-bit digits would unroll 255 counts per pass for no gain over 2 bits.
        if self.radix_bits not in (1, 2, 4):
            raise ValueError(f"radix_bits must be 1, 2 or 4, got {self.radix_bits}")

    def _select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        del scheduling_group_id
        if score_bits.ndim != 2:
            raise ValueError(f"score_bits must be 2D, got {score_bits.shape}")
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        # Named scope: profiles attribute the selector's ops to "BisectTopK".
        with jax.named_scope("BisectTopK"):
            return self._top_k(score_bits, k, row_lengths, return_scores)

    def _top_k(self, score_bits, k, row_lengths, return_scores):
        width = score_bits.shape[1]
        lengths = jnp.clip(row_lengths.astype(jnp.int32), 0, width)
        # (prefix width, whether to search): each branch serves the rows whose
        # longest length is at most its prefix width.
        if width <= k:
            buckets = ((width, False),)
        elif not self.width_buckets or score_bits.shape[0] < self.bucket_min_rows:
            buckets = ((width, True),)
        else:
            buckets = ((k, False),) + tuple((w, True) for w in _bucket_widths(width, k))
        branches = [
            functools.partial(self._select_prefix, width=w, k=k, search=search)
            for w, search in buckets
        ]
        if len(branches) == 1:
            ids = branches[0](score_bits, lengths)
        else:
            active = jnp.max(lengths)
            index = sum((active > w).astype(jnp.int32) for w, _ in buckets[:-1])
            ids = jax.lax.switch(index, branches, score_bits, lengths)
        if not return_scores:
            return ids
        taken = jnp.take_along_axis(score_bits, jnp.maximum(ids, 0), axis=1)
        return ids, jnp.where(ids >= 0, taken, jnp.int32(BITS_NEG_INF))

    def _select_prefix(
        self,
        score_bits: jax.Array,
        lengths: jax.Array,
        *,
        width: int,
        k: int,
        search: bool,
    ) -> jax.Array:
        """Top-k over the first ``width`` columns (every row length fits)."""
        used = min(width, score_bits.shape[1])
        padded = _round_up(width, _PACK_BLOCK)
        keys = jnp.pad(
            _ordered_keys(score_bits[:, :used]), ((0, 0), (0, padded - used))
        )
        cols = jnp.arange(padded, dtype=jnp.int32)[None, :]
        eligible = (cols < lengths[:, None]) & (keys > jnp.uint32(_KEY_NEG_INF))
        quota = jnp.minimum(jnp.sum(eligible, axis=1, dtype=jnp.int32), k)
        if not search:
            # At most k eligible columns: all of them win.
            return _pack_winners(eligible, k, quota)
        keys = jnp.where(eligible, keys, jnp.uint32(0))
        threshold = _kth_largest_key(keys, quota, self.radix_bits)[:, None]
        above = keys > threshold
        tied = (keys == threshold) & eligible
        tie_quota = quota - jnp.sum(above, axis=1, dtype=jnp.int32)
        surplus = jnp.any(jnp.sum(tied, axis=1, dtype=jnp.int32) > tie_quota)
        won = jax.lax.cond(
            surplus,
            lambda: above | _lowest_ties(tied, tie_quota),
            lambda: above | tied,
        )
        return _pack_winners(won, k, quota)


def default_topk_backend() -> TopKBackend:
    """The SparseCore selector where it can run on this chip, else ``BisectTopK``."""
    if _has_usable_sparse_core():
        return SparseCoreTopK()
    logger.info_once(
        "No usable SparseCore for the StreamIndex top-k; selecting on the "
        "TensorCore with BisectTopK."
    )
    return BisectTopK()
