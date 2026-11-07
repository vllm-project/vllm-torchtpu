# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for Stage 2: BlockEvictionTracker and hardware span attributes."""

import pytest

from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    compute_request_padding_tokens,
    extract_hardware_span_attributes,
)


def test_block_eviction_tracker_age_and_bounds():
    tracker = BlockEvictionTracker(maxsize=2)
    tracker.record_eviction("hash-1", now=100.0)
    tracker.record_eviction("hash-2", now=101.5)

    # Missed hashes include both hash-1 and hash-2; most recent eviction is hash-2 (101.5s)
    age_ms = tracker.time_since_eviction_ms(["hash-1", "hash-2"], now=102.0)
    assert age_ms == pytest.approx(500.0)

    # Insert hash-3, evicting hash-1 from the bounded LRU
    tracker.record_eviction("hash-3", now=103.0)
    assert tracker.time_since_eviction_ms(["hash-1"], now=104.0) is None
    assert tracker.time_since_eviction_ms(["hash-3"], now=104.0) == pytest.approx(
        1000.0
    )


def test_block_eviction_tracker_ignores_none_and_unknown():
    tracker = BlockEvictionTracker()
    tracker.record_eviction(None, now=10.0)
    assert len(tracker) == 0
    assert tracker.time_since_eviction_ms(["never-seen"], now=20.0) is None


@pytest.mark.parametrize(
    (
        "req_tokens",
        "total_scheduled",
        "padded_total",
        "expected_padding",
    ),
    [
        # Single request of 65 tokens padded to 128-token bucket -> 63 padded tokens
        (65, 65, 128, 63),
        # Two equal requests (50 + 50 = 100) padded to 128 -> 14 padded tokens each
        (50, 100, 128, 14),
        # Exact bucket match -> 0 padding
        (128, 128, 128, 0),
        # Invalid/empty -> 0 padding
        (0, 100, 128, 0),
    ],
)
def test_compute_request_padding_tokens(
    req_tokens, total_scheduled, padded_total, expected_padding
):
    assert (
        compute_request_padding_tokens(req_tokens, total_scheduled, padded_total)
        == expected_padding
    )


def test_extract_hardware_span_attributes():
    attrs = extract_hardware_span_attributes(
        kv_transfer_params={
            "dram_tokens": 512,
            "dram_bytes": 16777216,
            "dram_ms": 3.1,
            "remote_tokens": 256,
            "kv_transfer_ms": 4.25,
        },
        num_external_computed_tokens=768,
        time_since_eviction_ms=320.1256,
        lru_horizon_ms=14000.0,
        session_evicted_at_ms=1700000005000,
        tpu_shape_padding_tokens=63,
    )
    assert attrs == {
        "vllm.kv_cache.time_since_eviction_ms": 320.126,
        "vllm.kv_cache.lru_horizon_ms": 14000.0,
        "vllm.kv_cache.session_evicted_at_ms": 1700000005000,
        "vllm.tpu.shape_padding_tokens": 63,
        "vllm.kv_cache.external_computed_tokens": 768,
        "vllm.kv_cache.dram_tokens": 512,
        "vllm.kv_cache.dram_bytes": 16777216,
        "vllm.kv_cache.dram_ms": 3.1,
        "vllm.kv_cache.remote_tokens": 256,
        "vllm.kv_cache.transfer_ms": 4.25,
    }


def test_session_eviction_and_lru_horizon_for_joiner():
    tracker = BlockEvictionTracker()
    tracker.register_session_blocks("sess-A", ["blk-1", "blk-2"])
    assert tracker.session_evicted_at_ms("sess-A") is None
    assert tracker.lru_horizon_ms(now=50.0) is None

    tracker.record_eviction("blk-1", now=40.0, wall_time_ms=1700000040000)
    assert tracker.session_evicted_at_ms("sess-A") == 1700000040000
    assert tracker.lru_horizon_ms(now=54.0) == pytest.approx(14000.0)
