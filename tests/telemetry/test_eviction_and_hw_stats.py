# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""Unit tests for BlockEvictionTracker and hardware span attributes."""

import pytest

from vllm_torchtpu.telemetry.eviction_tracker import (
    BlockEvictionTracker,
    TpuPaddingCounterTracker,
    compute_request_padding_tokens,
    compute_tpu_bucket_padding_tokens,
    extract_hardware_span_attributes,
)


def test_block_eviction_tracker_age_and_bounds():
    tracker = BlockEvictionTracker(maxsize=2)
    tracker.record_eviction("hash-1", now=100.0)
    tracker.record_eviction("hash-2", now=101.5)

    # Missed hashes include both hash-1 and hash-2; most recent eviction is hash-2
    # (101.5s)
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
    tracker.register_session_blocks("sess-B", ["blk-3", "blk-4"])
    assert tracker.session_evicted_at_ms("sess-A") is None
    assert tracker.lru_horizon_ms(now=50.0, session_id="sess-A") is None

    tracker.record_eviction("blk-1", now=40.0, wall_time_ms=1700000040000)
    tracker.record_eviction("blk-3", now=52.0, wall_time_ms=1700000052000)

    assert tracker.session_evicted_at_ms("sess-A") == 1700000040000
    assert tracker.session_evicted_at_ms("sess-B") == 1700000052000
    assert tracker.lru_horizon_ms(now=54.0, session_id="sess-A") == pytest.approx(
        14000.0
    )
    assert tracker.lru_horizon_ms(now=54.0, session_id="sess-B") == pytest.approx(
        2000.0
    )
    assert tracker.lru_horizon_ms(now=54.0, session_id="sess-C") is None


def test_tpu_padding_counter_tracker():
    pad = TpuPaddingCounterTracker()
    pad.record_round(430, 512, is_prompt_round=True)
    pad.record_round(1, 16, is_prompt_round=False)
    pad.record_round(1, 16, is_prompt_round=False)

    assert pad.as_dict() == {
        "prompt_rounds": 1,
        "prompt_real_tokens": 430,
        "prompt_filler_tokens": 82,
        "reply_rounds": 2,
        "reply_real_tokens": 2,
        "reply_filler_tokens": 30,
    }


@pytest.mark.parametrize(
    ("scheduled_tokens", "expected_padding"),
    [
        (0, 0),
        (32, 0),
        (157, 99),
        (345, 167),
        (430, 82),
        (902, 122),
    ],
)
def test_compute_tpu_bucket_padding_tokens(scheduled_tokens, expected_padding):
    assert compute_tpu_bucket_padding_tokens(scheduled_tokens) == expected_padding


def test_dirty_sessions_stays_bounded_by_session_maxsize():
    tracker = BlockEvictionTracker(maxsize=8, session_maxsize=2)
    for idx in range(5):
        tracker.record_eviction(f"blk-{idx}", now=float(idx), session_id=f"s-{idx}")
    assert len(tracker._dirty_sessions) <= 2
    dirty = tracker.pop_dirty_session_state()
    assert set(dirty.keys()) == {"s-3", "s-4"}


def test_compute_tpu_bucket_padding_with_runner_paddings_table():
    paddings = [16, 64, 256, 1024, 4096, 6000]
    assert compute_tpu_bucket_padding_tokens(5000, paddings=paddings) == 1000
    assert compute_tpu_bucket_padding_tokens(6000, paddings=paddings) == 0
    assert compute_tpu_bucket_padding_tokens(7000, paddings=paddings) == 0


def test_eviction_state_sync_clear_watermarks_export_and_padding_merge():
    tracker = BlockEvictionTracker()
    tracker.record_eviction("b1", now=10.0, wall_time_ms=10000, session_id="s-a")
    tracker.record_eviction("b2", now=20.0, wall_time_ms=20000, session_id="s-b")
    tracker.record_eviction("b3", now=30.0, wall_time_ms=30000, session_id="s-c")

    # Dirty sessions flush once, then empty on immediate second call.
    dirty_1 = tracker.pop_dirty_session_state()
    assert set(dirty_1.keys()) == {"s-a", "s-b", "s-c"}
    assert tracker.pop_dirty_session_state() == {}

    # Export all vs recent limit=2 (newest two entries preserved in order).
    assert list(tracker.export_session_state().keys()) == ["s-a", "s-b", "s-c"]
    assert list(tracker.export_recent_session_state(limit=2).keys()) == ["s-b", "s-c"]

    # Receiver clears s-prewarmed at up_to_ts=15.0 BEFORE EngineCore sync arrives.
    replica_reader = BlockEvictionTracker()
    replica_reader.clear_session_eviction("s-prewarmed", up_to_ts=15.0)
    replica_reader.import_session_state(
        {
            "s-prewarmed": [12.0, 12000],  # older than clear watermark -> ignored
            "s-a": [10.0, 10000],
        }
    )
    assert replica_reader.session_evicted_at_ms("s-prewarmed") is None
    assert replica_reader.session_evicted_at_ms("s-a") == 10000

    # Clear s-a up_to_ts=12.0; older import at 11.0 is ignored, newer at 25.0 applies.
    replica_reader.clear_session_eviction("s-a", up_to_ts=12.0)
    assert replica_reader.session_evicted_at_ms("s-a") is None
    replica_reader.import_session_state({"s-a": [11.0, 11000]})
    assert replica_reader.session_evicted_at_ms("s-a") is None
    replica_reader.import_session_state({"s-a": [25.0, 25000]})
    assert replica_reader.session_evicted_at_ms("s-a") == 25000

    # Local re-eviction after clear drops the cleared watermark.
    replica_reader.clear_session_eviction("s-a", up_to_ts=25.0)
    replica_reader.record_eviction(
        "b-new", now=26.0, wall_time_ms=26000, session_id="s-a"
    )
    assert replica_reader.session_evicted_at_ms("s-a") == 26000
    replica_reader.clear()
    assert len(replica_reader) == 0

    # TpuPaddingCounterTracker monotonic snapshot merge & clear.
    pad = TpuPaddingCounterTracker()
    pad.update_from_dict(
        {"prompt_rounds": 3, "prompt_real_tokens": 500, "reply_rounds": 4}
    )
    pad.update_from_dict(
        {"prompt_rounds": 2, "prompt_real_tokens": 700, "reply_filler_tokens": 28}
    )
    assert pad.prompt_rounds == 3
    assert pad.prompt_real_tokens == 700
    assert pad.reply_rounds == 4
    assert pad.reply_filler_tokens == 28
    pad.clear()
    assert pad.as_dict() == {
        "prompt_rounds": 0,
        "prompt_real_tokens": 0,
        "prompt_filler_tokens": 0,
        "reply_rounds": 0,
        "reply_real_tokens": 0,
        "reply_filler_tokens": 0,
    }

    # P/D remote_tokens without dram_tokens does not mislabel external tokens as DRAM.
    pd_attrs = extract_hardware_span_attributes(
        kv_transfer_params={"remote_tokens": 256, "kv_transfer_ms": 1.5},
        num_external_computed_tokens=256,
    )
    assert pd_attrs["vllm.kv_cache.remote_tokens"] == 256
    assert "vllm.kv_cache.dram_tokens" not in pd_attrs
