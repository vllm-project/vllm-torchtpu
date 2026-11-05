# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare common SWA metadata with an independent NumPy address reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.swa_metadata import (
    common_swa_metadata,
    supports_swa_attention_metadata,
    swa_attention_metadata,
    swa_attention_records,
)

TOKENS, REQUESTS, WINDOW, PAGE, COLUMNS = 288, 48, 128, 64, 1544


@pytest.fixture(autouse=True)
def target_tpu_layout(monkeypatch):
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice(
            "TPU7x", num_cores=1, platform="tpu"
        ),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        target_info = pltpu.get_tpu_info()
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: target_info)


def _inputs(layout):
    counts = {
        "fixed_six": np.full(REQUESTS, 6, dtype=np.int32),
        "ragged_padded": np.tile(np.array([0, 1, 6, 3, 5, 7], np.int32), 8),
        "empty": np.zeros(REQUESTS, dtype=np.int32),
    }[layout]
    starts = np.concatenate((np.zeros(1, np.int32), counts.cumsum(dtype=np.int32)))
    bases = np.tile(np.array([0, 1, 63, 64, 127, 128, 130, 195], np.int32), 6)
    positions = np.zeros(TOKENS, dtype=np.int32)
    for request, count in enumerate(counts):
        positions[starts[request] : starts[request + 1]] = bases[request] + np.arange(
            count, dtype=np.int32
        )
    window_starts = np.maximum(bases - 4, 0)
    window_starts[::4] = bases[::4] + 2
    return starts, positions, window_starts


def _reference(starts, positions, window_starts):
    requests = np.minimum(
        np.searchsorted(starts[1:], np.arange(TOKENS), side="right"), REQUESTS - 1
    ).astype(np.int32)
    expected = {
        "request_ids": requests,
        "valid_tokens": np.arange(TOKENS) < starts[-1],
        "token_window_starts": window_starts[requests],
        "current_pages": np.minimum(positions // PAGE, COLUMNS - 1),
        "previous_pages": np.empty(TOKENS, np.int32),
        "oldest_pages": np.empty(TOKENS, np.int32),
        "current_offsets": positions % PAGE,
        "page_offsets": np.empty((TOKENS, WINDOW), np.int32),
        "page_selector": np.empty((TOKENS, WINDOW), np.int32),
        "is_current_page": np.empty((TOKENS, WINDOW), np.bool_),
        "is_previous_page": np.empty((TOKENS, WINDOW), np.bool_),
        "read_allowed": np.empty((TOKENS, WINDOW), np.bool_),
        "query_counts": np.diff(starts),
        "nonempty_requests": np.diff(starts) != 0,
    }
    for token, position in enumerate(positions):
        last_page = min(max(int(position), 0), COLUMNS * PAGE - 1) // PAGE
        expected["previous_pages"][token] = max(last_page - 1, 0)
        expected["oldest_pages"][token] = max(last_page - 2, 0)
        for column in range(WINDOW):
            logical = int(position) - WINDOW + 1 + column
            safe = min(max(logical, 0), COLUMNS * PAGE - 1)
            page, offset = divmod(safe, PAGE)
            if page == last_page:
                selector = 2
            elif page == max(last_page - 1, 0):
                selector = 1
            else:
                selector = 0
            expected["page_offsets"][token, column] = offset
            expected["page_selector"][token, column] = selector
            expected["is_current_page"][token, column] = page == last_page
            expected["is_previous_page"][token, column] = page == max(last_page - 1, 0)
            expected["read_allowed"][token, column] = (
                token < starts[-1]
                and logical >= 0
                and logical >= window_starts[requests[token]]
            )
    return expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("kernel", [common_swa_metadata, swa_attention_metadata])
@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("layout", ["fixed_six", "ragged_padded", "empty"])
def test_common_metadata_matches_request_and_window_reference(
    layout, enable_x64, kernel
):
    inputs = _inputs(layout)
    expected = _reference(*inputs)
    with jax.enable_x64(enable_x64):
        actual = kernel(
            *(jnp.asarray(value) for value in inputs),
            window_size=WINDOW,
            page_size=PAGE,
            table_columns=COLUMNS,
            interpret=True,
        )
    for name, value in actual._asdict().items():
        assert value.dtype == expected[name].dtype, name
        np.testing.assert_array_equal(value, expected[name], err_msg=name)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,requests,window,page,queries,expected",
    [
        (288, 48, 128, 64, 6, True),
        (48, 48, 128, 64, 1, True),
        (289, 48, 128, 64, 6, False),
        (0, 48, 128, 64, 6, False),
        (288, 48, 192, 64, 6, False),
        (288, 48, 128, 128, 6, False),
    ],
)
def test_decode_metadata_envelope(tokens, requests, window, page, queries, expected):
    assert (
        supports_swa_attention_metadata(tokens, requests, window, page, queries)
        is expected
    )


@pytest.mark.cpu_test
def test_common_metadata_rejects_non_power_of_two_page_size():
    with pytest.raises(AssertionError):
        common_swa_metadata(
            *(jnp.asarray(value) for value in _inputs("fixed_six")),
            window_size=WINDOW,
            page_size=48,
            table_columns=COLUMNS,
            interpret=True,
        )


def _records_reference(starts, positions, window_starts, table, stride, offset):
    """Direct per-window-position lookup, independent of three-page selection."""
    slots = np.full(positions.shape, -1, np.int32)
    records = np.full((len(positions), WINDOW), -1, np.int32)
    for token in range(int(starts[-1])):
        request = np.searchsorted(starts[1:], token, side="right")
        position = int(positions[token])
        block = int(table[request, position // PAGE])
        slots[token] = block * PAGE + position % PAGE
        if slots[token] < 0:
            continue
        for column in range(WINDOW):
            logical = position - WINDOW + 1 + column
            if logical < 0 or logical < window_starts[request]:
                continue
            block = int(table[request, logical // PAGE])
            records[token, column] = offset + block * stride + logical % PAGE
    return slots, slots >= 0, records


@pytest.mark.cpu_test
@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize(
    "layout,columns,stride,offset,blocks",
    [
        ("fixed_six", COLUMNS, PAGE, 0, "valid"),
        ("ragged_padded", COLUMNS, 256, 73, "valid"),
        ("empty", COLUMNS, PAGE, 0, "valid"),
        ("fixed_six", COLUMNS, 256, 4096, "invalid_current"),
        ("fixed_six", COLUMNS, 256, 300, "invalid_history"),
        ("ragged_padded", 1, 128, 1024, "valid"),
        ("fixed_six", 3, PAGE, 0, "valid"),
    ],
)
def test_final_records_match_direct_physical_lookup(
    layout, columns, stride, offset, blocks, enable_x64
):
    starts, positions, window_starts = _inputs(layout)
    # Deliberately non-monotonic physical placement and sequence/page boundaries.
    table = np.random.default_rng(17).integers(
        0, 30000, (REQUESTS, columns), dtype=np.int32
    )
    boundaries = np.array(
        [0, 1, 63, 64, 65, 127, 128, 129, columns * PAGE - 1], np.int32
    )
    positions[:] = np.resize(boundaries % (columns * PAGE), TOKENS)
    window_starts[::2] = 0
    window_starts[1::4] = columns * PAGE  # Some requests have no readable history.
    if blocks == "invalid_current":
        table[:] = -1
    elif blocks == "invalid_history":
        table[:, ::2] = -1
    # Padding positions can be stale or arbitrary, including overflow hazards.
    padding_count = TOKENS - int(starts[-1])
    positions[starts[-1] :] = np.resize(
        np.array(
            [columns * PAGE + 17, -17, np.iinfo(np.int32).min, np.iinfo(np.int32).max],
            np.int32,
        ),
        padding_count,
    )
    expected = _records_reference(
        starts, positions, window_starts, table, stride, offset
    )
    with jax.enable_x64(enable_x64):
        actual = swa_attention_records(
            *(jnp.asarray(v) for v in (starts, positions, window_starts, table)),
            window_size=WINDOW,
            page_size=PAGE,
            page_stride=stride,
            row_offset=offset,
            interpret=True,
        )
    for name, value, reference in zip(actual._fields, actual, expected):
        assert value.dtype == reference.dtype, name
        np.testing.assert_array_equal(value, reference, err_msg=name)
