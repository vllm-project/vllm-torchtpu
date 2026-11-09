# SPDX-License-Identifier: Apache-2.0
"""Merge only disjoint cache regions; preserve ordered writes otherwise."""

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as cache_ops


def _view(owner, offset, *, stride=12, page_rows=4):
    return cache_ops.PagedCacheRows(
        owner, ((4, page_rows, 128), torch.float32, stride * 512, offset * 512)
    )


@pytest.mark.parametrize(
    "offsets,strides,page_rows,merged",
    [
        ((0, 4, 8), (12, 12, 12), (4, 4, 4), True),
        ((8, 0, 4), (12, 12, 12), (4, 4, 4), True),
        ((0, 0), (12, 12), (4, 4), False),
        ((0, 1), (12, 12), (4, 4), False),
        ((0, 4), (12, 16), (4, 4), False),
        ((0, 4), (12, 12), (4, 2), False),
        ((0, 10), (12, 12), (4, 4), False),
    ],
)
def test_merged_writes_match_ordered_layer_publication(
    monkeypatch, offsets, strides, page_rows, merged
):
    # Meta owners exercise dispatch without allocating a TPU; replace only
    # the final hardware write with a CPU row writer.
    owner = torch.empty((64, 128), dtype=torch.int32, device="meta")
    caches = [
        _view(owner, off, stride=stride, page_rows=rows)
        for off, stride, rows in zip(offsets, strides, page_rows, strict=True)
    ]
    actual = torch.zeros((64, 128), dtype=torch.float32)
    expected = torch.zeros_like(actual)
    writes = []

    def publish(_cache, records, values):
        writes.append(records.clone())
        valid = records >= 0
        actual.index_copy_(0, records[valid].long(), values[valid])

    monkeypatch.setattr(cache_ops.PagedCacheRows, "_write_word_records", publish)
    slots = [torch.tensor([0, 1, 4, 5, -1, 7]) for _ in caches]
    valid = [torch.tensor([True, True, True, False, True, True]) for _ in caches]
    values = [torch.full((6, 128), float(i + 1)) for i in range(len(caches))]
    for off, stride, rows, ids, vals, ok in zip(
        offsets, strides, page_rows, slots, values, valid, strict=True
    ):
        for slot, value, keep in zip(ids.tolist(), vals, ok.tolist(), strict=True):
            if keep and slot >= 0:
                expected[off + slot // rows * stride + slot % rows] = value

    cache_ops.write_cache_layers(caches, slots, values, valid)

    assert len(writes) == (1 if merged else len(caches))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distinct_owners_keep_separate_writes(monkeypatch):
    owners = [
        torch.empty((64, 128), dtype=torch.int32, device="meta") for _ in range(2)
    ]
    caches = [_view(owners[0], 0), _view(owners[1], 4)]
    writes = []
    monkeypatch.setattr(
        cache_ops.PagedCacheRows,
        "_write_word_records",
        lambda cache, records, values: writes.append(cache.raw),
    )
    cache_ops.write_cache_layers(
        caches,
        [torch.tensor([0])] * 2,
        [torch.ones((1, 128))] * 2,
        [torch.tensor([True])] * 2,
    )
    assert len(writes) == 2
    assert writes[0] is owners[0]
    assert writes[1] is owners[1]
