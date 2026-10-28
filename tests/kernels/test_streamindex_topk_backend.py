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
"""Selection contract and backend injection through the real scoring pipeline."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.components import (
    StreamIndexComponents,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)


@dataclass(frozen=True)
class ReferenceTopK:
    """Test-only JAX selector; not a proposed TensorCore implementation."""

    overlaps_scoring: ClassVar[bool] = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        assert scheduling_group_id is None
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None, :] < row_lengths[:, None],
            scores,
            -jnp.inf,
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(values > -jnp.inf, ids, -1)
        bits = jax.lax.bitcast_convert_type(values, jnp.int32)
        return (ids, bits) if return_scores else ids


@pytest.mark.parametrize("return_scores", [False, True])
def test_reference_contract_masks_rows_and_pads(return_scores):
    scores = np.array([[2, 2, 1, 100], [1, 2, 3, 4]], np.float32)
    selected = ReferenceTopK().select(
        jnp.asarray(scores.view(np.int32)),
        3,
        jnp.array([2, 0], jnp.int32),
        scheduling_group_id=None,
        return_scores=return_scores,
    )
    ids = selected[0] if return_scores else selected
    np.testing.assert_array_equal(ids, [[0, 1, -1], [-1, -1, -1]])
    if return_scores:
        np.testing.assert_array_equal(
            np.asarray(selected[1]).view(np.float32),
            [[2, 2, -np.inf], [-np.inf, -np.inf, -np.inf]],
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("return_scores", [False, True])
def test_default_selector_preserves_sparsecore_results(return_scores):
    rng = np.random.default_rng(42)
    scores = rng.integers(-4, 5, (32, 512)).astype(np.float32)
    scores[:, 31] = -np.inf
    bits = jnp.asarray(scores.view(np.int32))
    lengths = jnp.array([0, 17, 128, 511] * 8, jnp.int32)
    expected = sparsecore_topk(bits, 128, lengths, return_scores=return_scores)
    actual = SparseCoreTopK().select(
        bits,
        128,
        lengths,
        scheduling_group_id=None,
        return_scores=return_scores,
    )
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(got, want)
    ids = np.asarray(actual[0] if return_scores else actual)
    for row, length in enumerate(np.asarray(lengths)):
        live = ids[row][ids[row] >= 0]
        assert np.all(live < length)
        assert len(live) == len(np.unique(live))
        eligible = scores[row, :length]
        eligible = eligible[eligible > -np.inf]
        np.testing.assert_array_equal(
            np.sort(scores[row, live]),
            np.sort(eligible)[-128:],
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("chunk_tokens", [None, 128])
@pytest.mark.parametrize("return_scores", [False, True])
def test_backend_replacement_keeps_real_streamindex_scores(chunk_tokens, return_scores):
    # All scores are zero: varying causal row lengths exercise ties, masks,
    # short rows and padding without a tolerance hiding selection errors.
    q = jnp.zeros((256, 4, 128), jnp.bfloat16)
    weights = jnp.zeros((256, 4), jnp.float32)
    cache = jnp.zeros((4, 32, 4, 256), jnp.uint8)
    args = (
        q,
        weights,
        cache,
        jnp.array([256, 257]),
        jnp.array([0, 1, 2, 3]),
        jnp.array([0, 128, 256]),
        jnp.array([0, 0, 2]),
    )
    kwargs = dict(
        k=128,
        compression_ratio=2,
        num_kv_pages_per_block=2,
        num_queries_per_block=(1, 128, 128),
        chunk_tokens=chunk_tokens,
        return_scores=return_scores,
    )
    default = streamindex_topk(*args, **kwargs)
    alternate = streamindex_topk(*args, topk_backend=ReferenceTopK(), **kwargs)
    for result in (default, alternate):
        ids = np.asarray(result[0] if return_scores else result)
        positions = np.r_[np.arange(128, 256), np.arange(129, 257)]
        lengths = np.minimum(positions // 2 + 1, np.repeat([128, 128], 128))
        for row, length in enumerate(lengths):
            live = ids[row][ids[row] >= 0]
            assert len(live) == min(length, 128)
            assert len(live) == len(np.unique(live))
            assert np.all(live < length)
            assert np.all(ids[row, len(live) :] == -1)
        if return_scores:
            scores = np.asarray(result[1]).view(np.float32)
            np.testing.assert_array_equal(scores[ids >= 0], 0)
            np.testing.assert_array_equal(scores[ids < 0], -np.inf)


@dataclass(frozen=True)
class DenseKeyReader:
    """BF16 records with an explicit FP32 scale, unlike the packed FP8 reader."""

    packing: int = 1
    scale_gain: float = 2.0

    def validate(self, cache, width, layout):
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout

        if layout != KVLayout.HEAD_ALONG_SUBLANE:
            raise NotImplementedError("DenseKeyReader requires HEAD_ALONG_SUBLANE")
        if self.packing != 1 or cache.shape[2:] != (1, 2 * width):
            raise ValueError(
                "DenseKeyReader requires one padded key/scale record per group"
            )
        if cache.dtype != jnp.bfloat16:
            raise ValueError("DenseKeyReader requires BF16 records")

    def read(self, ref, rows, width):
        records = ref[...].reshape(rows, 2 * width)
        return records[:, :width], records[:, width : width + 1].T.astype(
            jnp.float32
        ) * self.scale_gain


@dataclass(frozen=True)
class PrefixSchedule:
    """Nonidentity coordinates and row-dependent visibility, including empty rows."""

    def validate(self, compression_ratio):
        if compression_ratio != 1:
            raise ValueError("PrefixSchedule requires compression_ratio=1")

    def positions(self, positions):
        return positions - 3

    def visible_lengths(self, lengths):
        return jnp.where(lengths % 4 == 0, 0, jnp.maximum(lengths - 11, 0))

    def chunk_metadata(self, meta, chunk_idx):
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.metadata import (
            MetadataRef,
        )

        offset = jnp.sum(meta.num_steps[:chunk_idx])
        indices = jnp.arange(meta.batch_tile_idx.size) + offset
        return MetadataRef.create(
            meta.num_steps[chunk_idx : chunk_idx + 1],
            *(
                jnp.take(a, indices, mode="clip")
                for a in (meta.batch_tile_idx, meta.bq_idx, meta.bkv_idx)
            ),
        )


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize(
    "shift,offset,visible,local,sharded",
    [
        (-3, 0, [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]),
        (20, 0, [4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2]),
        (0, -10, [1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0]),
        (0, 10, [1, 2, 3, 4], [3, 3, 3, 3], [3, 3, 3, 3]),
    ],
    ids=[
        "negative-positions",
        "past-end-positions",
        "negative-lengths",
        "long-lengths",
    ],
)
def test_component_row_lengths_are_bounded(
    cp_size, shift, offset, visible, local, sharded
):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
        _effective_row_lengths,
    )

    calls = []

    class ObservingSchedule(PrefixSchedule):
        def positions(self, positions):
            calls.append("positions")
            return positions + shift

        def visible_lengths(self, lengths):
            calls.append("visible_lengths")
            np.testing.assert_array_equal(lengths, visible)
            return lengths + offset

    actual = _effective_row_lengths(
        jnp.array([4]),
        jnp.array([0, 4]),
        jnp.array([0, 0, 1]),
        num_tokens=4,
        num_positions=3,
        compression_ratio=1,
        cp_rank=cp_size - 1,
        cp_size=cp_size,
        components=StreamIndexComponents(DenseKeyReader(), ObservingSchedule()),
    )
    assert calls == ["positions", "visible_lengths"]
    np.testing.assert_array_equal(actual, local if cp_size == 1 else sharded)


@pytest.mark.parametrize(
    "cp_size,cp_rank,expected",
    [
        (1, 0, [2, 2, 2, 1, 0, 0]),
        (2, 0, [1, 1, 1, 1, 0, 0]),
        (2, 1, [1, 1, 1, 0, 0, 0]),
    ],
)
def test_default_row_lengths_keep_causal_and_padding_behavior(
    cp_size, cp_rank, expected
):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
        _effective_row_lengths,
    )

    actual = _effective_row_lengths(
        jnp.array([5, 2, 0]),
        jnp.array([0, 3, 4, 4]),
        jnp.array([0, 0, 2]),
        num_tokens=6,
        num_positions=16,
        compression_ratio=2,
        cp_size=cp_size,
        cp_rank=cp_rank,
    )
    np.testing.assert_array_equal(actual, expected)


def test_key_reader_decodes_window_and_scales():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout

    records = (
        jnp.arange(4 * 256, dtype=jnp.float32).astype(jnp.bfloat16).reshape(4, 1, 256)
    )
    reader = DenseKeyReader()
    reader.validate(records[None], 128, KVLayout.HEAD_ALONG_SUBLANE)
    keys, scales = reader.read(records, 4, 128)
    np.testing.assert_array_equal(keys, records[:, 0, :128])
    np.testing.assert_array_equal(
        scales, np.asarray(records[:, 0, 128:129], np.float32).T * 2
    )
    assert reader.packing == 1


def test_schedule_coordinates_visibility_and_chunk_window():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.metadata import MetadataRef

    schedule = PrefixSchedule()
    schedule.validate(1)
    np.testing.assert_array_equal(
        schedule.positions(jnp.array([60, 61, 62, 63])), [57, 58, 59, 60]
    )
    np.testing.assert_array_equal(
        schedule.visible_lengths(jnp.array([58, 59, 60, 61])), [47, 48, 0, 50]
    )
    meta = MetadataRef.create(
        jnp.array([2, 3]), jnp.arange(5), jnp.arange(5) + 10, jnp.arange(5) + 20
    )
    chunk = schedule.chunk_metadata(meta, 1)
    np.testing.assert_array_equal(chunk.num_steps, [3])
    for got, want in zip(
        (chunk.batch_tile_idx, chunk.bq_idx, chunk.bkv_idx),
        ([2, 3, 4], [12, 13, 14], [22, 23, 24]),
    ):
        np.testing.assert_array_equal(got[:3], want)


def _component_inputs(seq_lens, q_lens, padded_tokens):
    page_size, width, heads = 128, 128, 4
    pages_per_seq = (max(seq_lens) + page_size - 1) // page_size
    table = (
        np.arange(len(seq_lens) * pages_per_seq)[::-1]
        .reshape(len(seq_lens), pages_per_seq)
        .copy()
    )
    cache = np.zeros((table.size, page_size, 1, 2 * width), np.float32)
    for seq, pages in enumerate(table):
        for logical_page, physical_page in enumerate(pages):
            positions = logical_page * page_size + np.arange(page_size)
            cache[physical_page, :, 0, 0] = (positions * 17 + seq * 7) % 251 - 125
            cache[physical_page, :, 0, width] = np.where(positions % 3 == 0, 2, 0.5)
    q = np.zeros((padded_tokens, heads, width), np.float32)
    q[:, :, 0] = (np.arange(padded_tokens) % 3 + 1)[:, None]
    weights = np.broadcast_to(
        np.array([0.5, 1, 1.5, 2], np.float32), (padded_tokens, heads)
    ).copy()
    args = (
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(weights),
        jnp.asarray(cache, jnp.bfloat16),
        jnp.array(seq_lens, jnp.int32),
        jnp.array(table.reshape(-1), jnp.int32),
        jnp.array(np.r_[0, np.cumsum(q_lens)], jnp.int32),
        jnp.array([0, 0, len(seq_lens)], jnp.int32),
    )
    return args, (q, weights, cache, table)


def _assert_component_selection(result, inputs, seq_lens, q_lens, k, return_scores):
    q, weights, cache, table = inputs
    ids = np.asarray(result[0] if return_scores else result)
    scores = np.asarray(result[1]).view(np.float32) if return_scores else None
    row = 0
    for seq, (length, count) in enumerate(zip(seq_lens, q_lens)):
        records = cache[table[seq]].reshape(-1, cache.shape[-1])
        for t in range(count):
            # Independent host arithmetic; do not call the schedule or scoring helpers.
            causal_count = min(length, length - count + t - 3 + 1)
            visible = 0 if causal_count % 4 == 0 else max(causal_count - 11, 0)
            expected = (
                np.maximum(q[row] @ records[:visible, :128].T, 0).T @ weights[row]
            )
            expected *= records[:visible, 128] * 2
            live = ids[row][ids[row] >= 0]
            assert len(live) == min(visible, k)
            assert len(np.unique(live)) == len(live)
            assert np.all(live < visible)
            assert np.all(ids[row, len(live) :] == -1)
            np.testing.assert_array_equal(
                np.sort(expected[live]), np.sort(expected)[-k:]
            )
            if scores is not None:
                np.testing.assert_array_equal(scores[row, : len(live)], expected[live])
                assert np.all(np.isneginf(scores[row, len(live) :]))
            row += 1
    assert np.all(ids[row:] == -1)
    if scores is not None:
        assert np.all(np.isneginf(scores[row:]))


@pytest.mark.tpu_test
@pytest.mark.parametrize("enable_early_exit", [False, True])
def test_custom_schedule_short_candidates_use_full_visibility(enable_early_exit):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.components import (
        StreamIndexComponents,
    )

    args, host = _component_inputs((64,), (4,), 8)
    actual = streamindex_topk(
        *args,
        k=128,
        compression_ratio=1,
        num_kv_pages_per_block=1,
        num_queries_per_block=8,
        enable_early_exit=enable_early_exit,
        components=StreamIndexComponents(DenseKeyReader(), PrefixSchedule()),
        topk_backend=ReferenceTopK(),
    )
    _assert_component_selection(actual, host, (64,), (4,), 128, False)


@pytest.mark.tpu_test
@pytest.mark.parametrize("chunk_tokens", [None, 32])
@pytest.mark.parametrize("query_block", [8, 16])
def test_custom_components_score_chunked_rows_against_host(chunk_tokens, query_block):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.components import (
        StreamIndexComponents,
    )

    lengths, q_lens = (512, 0, 512), (24, 0, 32)
    args, host = _component_inputs(lengths, q_lens, 64)
    actual = streamindex_topk(
        *args,
        k=16,
        compression_ratio=1,
        num_kv_pages_per_block=1,
        num_queries_per_block=query_block,
        chunk_tokens=chunk_tokens,
        return_scores=True,
        components=StreamIndexComponents(DenseKeyReader(), PrefixSchedule()),
        topk_backend=ReferenceTopK(),
    )
    _assert_component_selection(actual, host, lengths, q_lens, 16, True)


@pytest.mark.parametrize("unsupported", ["packing", "dtype", "layout", "compression"])
def test_components_reject_unsupported_inputs(unsupported):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.components import (
        StreamIndexComponents,
    )
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout

    args, _ = _component_inputs((64,), (4,), 8)
    reader, ratio, layout = DenseKeyReader(), 1, KVLayout.HEAD_ALONG_SUBLANE
    if unsupported == "packing":
        reader = DenseKeyReader(packing=2)
    elif unsupported == "dtype":
        args = (*args[:2], args[2].astype(jnp.uint8), *args[3:])
    elif unsupported == "layout":
        layout = KVLayout.SEQ_ALONG_LANE
    else:
        ratio = 2
    with pytest.raises(
        (ValueError, NotImplementedError),
        match="DenseKeyReader|PrefixSchedule|HEAD_ALONG_SUBLANE",
    ):
        streamindex_topk(
            *args,
            k=128,
            compression_ratio=ratio,
            num_kv_pages_per_block=1,
            num_queries_per_block=8,
            kv_layout=layout,
            components=StreamIndexComponents(reader, PrefixSchedule()),
        )


@pytest.mark.parametrize("packing", [0, -1, 3, 512])
def test_component_packing_rejected_before_scoring(packing):
    class PermissiveReader(DenseKeyReader):
        def validate(self, cache, width, layout):
            pass

    args, _ = _component_inputs((64,), (4,), 8)
    with pytest.raises(ValueError, match="packing"):
        streamindex_topk(
            *args,
            k=128,
            compression_ratio=1,
            num_kv_pages_per_block=(1, 2, 1),
            num_queries_per_block=8,
            components=StreamIndexComponents(
                PermissiveReader(packing), PrefixSchedule()
            ),
        )


def test_unsupported_layout_skips_both_component_validators():
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout

    class ForbiddenReader(DenseKeyReader):
        def validate(self, cache, width, layout):
            pytest.fail("key validator must not receive an unsupported layout")

    class ForbiddenSchedule(PrefixSchedule):
        def validate(self, compression_ratio):
            pytest.fail("schedule validator must not receive an unsupported layout")

    args, _ = _component_inputs((64,), (4,), 8)
    with pytest.raises(NotImplementedError, match="HEAD_ALONG_SUBLANE"):
        streamindex_topk(
            *args,
            k=128,
            compression_ratio=1,
            num_kv_pages_per_block=1,
            num_queries_per_block=8,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            components=StreamIndexComponents(ForbiddenReader(), ForbiddenSchedule()),
        )
