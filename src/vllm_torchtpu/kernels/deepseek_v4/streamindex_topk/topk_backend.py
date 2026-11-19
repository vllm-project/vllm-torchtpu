# SPDX-License-Identifier: Apache-2.0
"""Static selection backends for StreamIndex's score-buffer contract."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk


class TopKBackend(Protocol):
    """Select exact winners from i32 buffers containing IEEE float32 score bits.

    Only columns below each row length are eligible; -inf marks invalid scores.
    Return i32 column IDs, with valid winners packed before a -1 suffix. Winner
    order is unspecified; threshold ties may choose any eligible tied IDs. Optional
    scores use the same i32 float32-bit representation, with -inf padding.
    Backends own any input reinterpretation, hardware eligibility and tiling.
    Implementations must be hashable because they are static JIT arguments.
    """

    overlaps_scoring: ClassVar[bool]

    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]: ...


@dataclass(frozen=True)
class SparseCoreTopK:
    """The original selector, including its score/next-chunk overlap schedule.

    ``bounded_copy`` copies only each row's live prefix of the score buffer
    (see ``sparsecore_topk``); same winners.
    """

    overlaps_scoring: ClassVar[bool] = True
    bounded_copy: bool = False

    def select(
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
            bounded_copy=self.bounded_copy,
        )


def has_usable_sparse_core() -> bool:
    """Return True when the active TPU provides <=32 B SparseCore DMA granules."""
    try:
        sc = pltpu.get_tpu_info().sparse_core
    except ValueError:
        return True
    return sc is not None and getattr(sc, "dma_granule_size_bytes", 32) <= 32


_TC_TOPK_PREFIX_BUCKETS: tuple[int, ...] = (
    1024,
    2048,
    4096,
    16384,
    32768,
    65536,
)
_TC_TOPK_TOURNAMENT_LEAF_WIDTH = 4096
_TC_TOPK_VMEM_SUB_BATCH = 32


def _f32_to_desc_key(x: jax.Array) -> jax.Array:
    """Map IEEE float32 values to int32 keys ordering descending (+0.0 > -0.0)."""
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    sign_mask = jax.lax.shift_right_arithmetic(bits, jnp.int32(31))
    return ~((bits & jnp.int32(0x7FFFFFFF)) ^ sign_mask)


def _desc_key_to_f32(desc_key: jax.Array) -> jax.Array:
    """Invert ``_f32_to_desc_key`` back to exact IEEE float32 values."""
    u = ~desc_key
    sign_mask = jax.lax.shift_right_arithmetic(u, jnp.int32(31))
    bits = u ^ (sign_mask & jnp.int32(0x7FFFFFFF))
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _tournament_topk(
    scores: jax.Array,
    k: int,
    leaf_width: int = _TC_TOPK_TOURNAMENT_LEAF_WIDTH,
) -> tuple[jax.Array, jax.Array]:
    """Exact tournament ``lax.top_k`` using leaf ``top_k`` + 2-key sort."""
    batch_rows, width = scores.shape
    pad_w = (-width) % leaf_width
    if pad_w > 0:
        scores = jnp.pad(scores, ((0, 0), (0, pad_w)), constant_values=-jnp.inf)
    num_blocks = scores.shape[1] // leaf_width
    leaf_vals, leaf_local = jax.lax.top_k(
        scores.reshape(batch_rows * num_blocks, leaf_width), k
    )
    offsets = (jnp.arange(num_blocks, dtype=jnp.int32) * leaf_width)[None, :, None]
    cand_vals = leaf_vals.reshape(batch_rows, num_blocks * k)
    cand_ids = (
        leaf_local.reshape(batch_rows, num_blocks, k).astype(jnp.int32) + offsets
    ).reshape(batch_rows, num_blocks * k)

    cand_desc = _f32_to_desc_key(cand_vals)
    sorted_desc, sorted_ids = jax.lax.sort(
        (cand_desc, cand_ids),
        dimension=1,
        is_stable=False,
        num_keys=2,
    )
    return _desc_key_to_f32(sorted_desc[:, :k]), sorted_ids[:, :k].astype(jnp.int32)


def _exact_tournament_topk(
    scores: jax.Array,
    k: int,
    leaf_width: int = _TC_TOPK_TOURNAMENT_LEAF_WIDTH,
) -> tuple[jax.Array, jax.Array]:
    """Exact TensorCore top-k with strided k=16 VPU filter + tie-certified path."""
    batch_rows, width = scores.shape
    if k >= leaf_width or width <= leaf_width:
        vals, ids = jax.lax.top_k(scores, k)
        return vals, ids.astype(jnp.int32)

    num_lanes = 128
    k_lane = 16
    if k <= 512 and width >= 8192 and width % num_lanes == 0:
        lane_len = width // num_lanes
        if lane_len >= k_lane:
            lanes = (
                scores.reshape(batch_rows, lane_len, num_lanes)
                .swapaxes(1, 2)
                .reshape(batch_rows * num_lanes, lane_len)
            )
            leaf_vals, leaf_q = jax.lax.top_k(lanes, k_lane)
            leaf_vals = leaf_vals.reshape(batch_rows, num_lanes, k_lane)
            lane_ids = jnp.arange(num_lanes, dtype=jnp.int32)[None, :, None]
            leaf_ids = (
                leaf_q.reshape(batch_rows, num_lanes, k_lane).astype(jnp.int32)
                * num_lanes
                + lane_ids
            )
            leaf_desc = _f32_to_desc_key(leaf_vals)
            drop_desc = leaf_desc[:, :, k_lane - 1]
            drop_id = leaf_ids[:, :, k_lane - 1]
            cand_desc = leaf_desc.reshape(batch_rows, num_lanes * k_lane)
            cand_ids = leaf_ids.reshape(batch_rows, num_lanes * k_lane)

            sorted_desc, sorted_ids = jax.lax.sort(
                (cand_desc, cand_ids),
                dimension=1,
                is_stable=False,
                num_keys=2,
            )
            fast_desc = sorted_desc[:, :k]
            fast_ids = sorted_ids[:, :k]
            fast_vals = _desc_key_to_f32(fast_desc)
            kth_desc = fast_desc[:, k - 1 : k]
            kth_id = fast_ids[:, k - 1 : k]
            neg_inf_desc = jnp.int32(0x7F800000)
            certified = jnp.all(
                (kth_desc < drop_desc)
                | ((kth_desc == drop_desc) & (kth_id <= drop_id))
                | (drop_desc >= neg_inf_desc)
            )
            return jax.lax.cond(
                certified,
                lambda _: (fast_vals, fast_ids),
                lambda _: _tournament_topk(scores, k, leaf_width),
                None,
            )

    return _tournament_topk(scores, k, leaf_width)


@dataclass(frozen=True)
class TensorCoreTopK:
    """Bounded tournament selector for TPUs without usable SparseCore DMA."""

    overlaps_scoring: ClassVar[bool] = False
    enable_prefix_bucketing: bool = True

    def _select_tile(
        self,
        sliced_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        return_scores: bool,
        use_tournament: bool = True,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        scores = jax.lax.bitcast_convert_type(sliced_bits, jnp.float32)
        cols = jnp.arange(scores.shape[1], dtype=jnp.int32)[None, :]
        scores = jnp.where(cols < row_lengths[:, None], scores, -jnp.inf)
        if scores.shape[1] < k:
            scores = jnp.pad(
                scores, ((0, 0), (0, k - scores.shape[1])), constant_values=-jnp.inf
            )
        if use_tournament:
            values, ids = _exact_tournament_topk(scores, k)
        else:
            values, ids = jax.lax.top_k(scores, k)
        valid = values > -jnp.inf
        ids = jnp.where(valid, ids.astype(jnp.int32), jnp.int32(-1))
        if not return_scores:
            return ids
        values = jnp.where(valid, values, -jnp.inf)
        return ids, jax.lax.bitcast_convert_type(values, jnp.int32)

    def _select_width(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        width: int,
        *,
        return_scores: bool,
        use_tournament: bool = True,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        sliced_bits = score_bits[:, :width]
        batch_rows = sliced_bits.shape[0]
        sub_batch = _TC_TOPK_VMEM_SUB_BATCH
        if (
            use_tournament
            and width >= 8192
            and batch_rows > sub_batch
            and batch_rows % sub_batch == 0
        ):
            num_sub = batch_rows // sub_batch
            sub_bits = sliced_bits.reshape(num_sub, sub_batch, width)
            sub_lens = row_lengths.reshape(num_sub, sub_batch)
            neg_inf_bits = jax.lax.bitcast_convert_type(
                jnp.float32(-jnp.inf), jnp.int32
            )

            def _scan_body(carry, xs):
                b_bits, b_lens = xs

                def _empty_tile(_):
                    empty_ids = jnp.full((sub_batch, k), jnp.int32(-1), dtype=jnp.int32)
                    if not return_scores:
                        return empty_ids
                    empty_scores = jnp.full(
                        (sub_batch, k), neg_inf_bits, dtype=jnp.int32
                    )
                    return empty_ids, empty_scores

                def _active_tile(_):
                    return self._select_tile(
                        b_bits,
                        k,
                        b_lens,
                        return_scores=return_scores,
                        use_tournament=True,
                    )

                return carry, jax.lax.cond(
                    jnp.max(b_lens) <= 0, _empty_tile, _active_tile, None
                )

            _, out = jax.lax.scan(_scan_body, None, (sub_bits, sub_lens))
            if not return_scores:
                return out.reshape(batch_rows, k)
            out_ids, out_scores = out
            return out_ids.reshape(batch_rows, k), out_scores.reshape(batch_rows, k)

        return self._select_tile(
            sliced_bits,
            k,
            row_lengths,
            return_scores=return_scores,
            use_tournament=use_tournament,
        )

    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        del scheduling_group_id
        full_width = score_bits.shape[1]
        buckets = tuple(b for b in _TC_TOPK_PREFIX_BUCKETS if k <= b < full_width) + (
            full_width,
        )
        if not self.enable_prefix_bucketing:
            return self._select_width(
                score_bits,
                k,
                row_lengths,
                full_width,
                return_scores=return_scores,
                use_tournament=False,
            )
        if len(buckets) == 1:
            return self._select_width(
                score_bits,
                k,
                row_lengths,
                full_width,
                return_scores=return_scores,
                use_tournament=True,
            )
        max_len = jnp.maximum(jnp.max(row_lengths), k)
        branch_idx = jnp.sum(jnp.asarray(buckets[:-1], dtype=jnp.int32) < max_len)
        branches = [
            (
                lambda w: lambda bits, lens: self._select_width(
                    bits,
                    k,
                    lens,
                    w,
                    return_scores=return_scores,
                    use_tournament=True,
                )
            )(w)
            for w in buckets
        ]
        return jax.lax.switch(branch_idx, branches, score_bits, row_lengths)


def default_topk_backend() -> TopKBackend:
    """SparseCore selector where available, bounded TensorCore tournament otherwise."""
    mode = envs.VLLM_TPU_TOPK_BACKEND.strip().lower()
    if mode in ("xla", "tensorcore"):
        return TensorCoreTopK()
    if mode == "sparsecore":
        return SparseCoreTopK()
    if mode not in ("", "auto"):
        raise ValueError(
            f"Invalid VLLM_TPU_TOPK_BACKEND={mode!r}; expected '', 'auto', "
            "'xla', 'tensorcore', or 'sparsecore'."
        )
    return SparseCoreTopK() if has_usable_sparse_core() else TensorCoreTopK()
