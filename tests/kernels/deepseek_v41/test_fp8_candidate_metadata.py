# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed FP8 candidate metadata: SparseCore histogram windows vs XLA."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8 import (
    candidates as fp8_candidates,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS,
    PackedFP8Layout,
)

BLOCK = 8
LANES = 128


def _cdiv(a, b):
    return -(-a // b)


def _fake_tpu_info(monkeypatch, vmem_bytes):
    info = SimpleNamespace(
        num_lanes=LANES,
        sparse_core=SimpleNamespace(num_lanes=16, vmem_capacity_bytes=vmem_bytes),
    )
    monkeypatch.setattr(fp8_candidates.pltpu, "get_tpu_info", lambda: info)


# Decode bucket 288: offsets has tokens + 1 entries.
_DECODE = dict(offsets=289, requests=288)


@pytest.mark.parametrize("vmem_kib", [384, 512, 1024])
def test_histogram_window_fits_budget(monkeypatch, vmem_kib):
    _fake_tpu_info(monkeypatch, vmem_kib * 1024)
    budget = vmem_kib * 1024 // 2 // 4

    # 98K scores (12,352 blocks): one window spans the whole histogram.
    blocks = 98816 // BLOCK
    full = _cdiv(blocks, LANES) * LANES
    window = fp8_candidates._metadata_histogram_blocks(
        2, 2048, blocks, 98816 // PAGE_RECORDS, **_DECODE
    )
    assert window == full

    # 512K scores (65,536 blocks): equal lane-aligned windows within budget,
    # and no more windows than the largest fitting window needs.
    blocks = 524288 // BLOCK
    columns = 524288 // PAGE_RECORDS
    fixed = fp8_candidates._metadata_scratch_words(2, 2048, 0, columns, **_DECODE)
    window = fp8_candidates._metadata_histogram_blocks(
        2, 2048, blocks, columns, **_DECODE
    )
    widest = (budget - fixed) // 2 // LANES * LANES
    assert 0 < window <= widest and window % LANES == 0
    assert fixed + 2 * window <= budget
    windows = _cdiv(blocks, window)
    assert windows == _cdiv(blocks, widest) <= fp8_candidates._MAX_HISTOGRAM_WINDOWS


# 128 KiB: fixed scratch alone exceeds the budget. 192 KiB: 43 windows.
@pytest.mark.parametrize("vmem_kib", [128, 192])
def test_histogram_window_falls_back(monkeypatch, vmem_kib):
    _fake_tpu_info(monkeypatch, vmem_kib * 1024)
    assert (
        fp8_candidates._metadata_histogram_blocks(
            2, 2048, 524288 // BLOCK, 524288 // PAGE_RECORDS, **_DECODE
        )
        == 0
    )


def _inputs(width, per_query):
    """Paired and unpaired requests over ragged, partly unmapped page tables."""
    rng = np.random.default_rng(width + per_query)
    blocks = width // BLOCK
    counts = [2, 1, 2, 3, 2]
    lengths = [width, width // 3, width - 1000, 5000, width // 2 + 77]
    columns = _cdiv(width, PAGE_RECORDS)
    layout = PackedFP8Layout(len(counts) * columns)
    pages = rng.permutation(layout.pages).reshape(len(counts), columns)
    for request, length in enumerate(lengths):
        pages[request, _cdiv(length, PAGE_RECORDS) :] = -1
    pages[2, 1] = -1

    rows = []
    for count, length in zip(counts, lengths):
        visible = _cdiv(length, BLOCK)
        first = None
        for query in range(count):
            # Blocks across the full range cross every histogram window;
            # some lie past the request's mapped pages.
            ids = np.concatenate(
                (
                    rng.integers(0, blocks, per_query // 2),
                    rng.integers(0, visible, per_query // 2),
                )
            )
            if first is not None and query % 2:
                ids[: per_query // 2] = first[: per_query // 2]  # Shared blocks.
            ids[rng.random(per_query) < 0.05] = ids[0]  # Repeats in a query.
            ids[-rng.integers(1, per_query // 4) :] = -1  # Short top-k rows.
            ids[3] = blocks + 5  # Out of range.
            rng.shuffle(ids)
            first = ids
            rows.append(ids)
    rows.append(np.full(per_query, -1))  # Padded query row.
    requests = np.concatenate(
        [np.full(count, request) for request, count in enumerate(counts)] + [[-1]]
    )
    return (
        jnp.asarray(pages, jnp.int32),
        jnp.asarray(np.stack(rows), jnp.int32),
        jnp.asarray(requests, jnp.int32),
        layout,
    )


def _metadata(pages, candidates, requests, layout, width):
    # A fresh jit per call, so monkeypatched routing is retraced.
    run = jax.jit(
        lambda p, c, r: fp8_candidates.prepare_candidate_metadata(
            p, c, r, layout, BLOCK, width
        )
    )
    return jax.tree.map(np.asarray, run(pages, candidates, requests))


@pytest.mark.parametrize("split", [False, True], ids=["sized", "narrowed"])
@pytest.mark.parametrize("per_query", [2048, 2000], ids=["direct", "grouped"])
@pytest.mark.parametrize("width", [98816, 524288])
def test_candidate_metadata_matches_reference(monkeypatch, width, per_query, split):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore candidate metadata requires TPU")
    pages, candidates, requests, layout = _inputs(width, per_query)
    sized = fp8_candidates._metadata_histogram_blocks

    with monkeypatch.context() as patch:
        patch.setattr(fp8_candidates, "_metadata_histogram_blocks", lambda *a, **k: 0)
        expected = _metadata(pages, candidates, requests, layout, width)

    def no_reference(*args, **kwargs):
        raise AssertionError("candidate metadata fell back to the XLA reference")

    monkeypatch.setattr(fp8_candidates, "_canonical_candidate_blocks_ref", no_reference)
    if split:
        # At most a third of the histogram per window: three or more windows.
        full = _cdiv(width // BLOCK, LANES) * LANES
        forced = _cdiv(_cdiv(full, 3), LANES) * LANES
        monkeypatch.setattr(
            fp8_candidates,
            "_metadata_histogram_blocks",
            lambda *a, **k: min(sized(*a, **k), forced),
        )
    actual = _metadata(pages, candidates, requests, layout, width)

    for name, want, got in zip(expected._fields, expected, actual):
        np.testing.assert_array_equal(got, want, err_msg=name)
