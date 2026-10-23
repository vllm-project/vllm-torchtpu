# SPDX-License-Identifier: Apache-2.0
"""ix-unified-port: TPU_SCHED_DECODE_INFLIGHT_THROTTLE=1 (default 0) restores the final-lineage
decode admission throttle (_decode_inflight_reserved_blocks, fp2q-r5 0447ff58b)."""
from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler


# ---------------- TPU_SCHED_DECODE_INFLIGHT_THROTTLE ----------------


def _pas(consumer=True, inflight=0, running=0, max_running=16, dcp=4):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._is_kv_consumer = consumer
    s._inflight_prefills = [object()] * inflight
    s.running = [object()] * running
    s.max_num_running_reqs = max_running
    s.dcp_world_size = dcp
    return s


@pytest.fixture
def base7():
    with patch.object(Scheduler, "_inflight_prefill_reserved_blocks", return_value=7, create=True):
        yield


def _clear(monkeypatch):
    for v in ("TPU_SCHED_DECODE_INFLIGHT_THROTTLE", "TPU_SCHED_DECODE_MAX_STAGE3_INFLIGHT",
              "TPU_SCHED_DECODE_STAGE3_OVERPROVISION", "TPU_SCHED_DECODE_RESERVED_BLOCKS",
              "TPU_SCHED_DECODE_RESERVED_BLOCKS_PER_SEQ"):
        monkeypatch.delenv(v, raising=False)


def test_throttle_default_off_is_the_stock_reservation(monkeypatch, base7):
    _clear(monkeypatch)
    for inflight in (0, 16, 40):
        assert _pas(inflight=inflight, running=16)._decode_inflight_reserved_blocks() == 7
        assert _pas(inflight=inflight, running=16)._inflight_prefill_reserved_blocks() == 7


def test_throttle_on_dcp4_defaults(monkeypatch, base7):
    _clear(monkeypatch)
    monkeypatch.setenv("TPU_SCHED_DECODE_INFLIGHT_THROTTLE", "1")
    assert _pas(inflight=16, running=0)._decode_inflight_reserved_blocks() == 1 << 30
    assert _pas(inflight=15, running=0)._decode_inflight_reserved_blocks() == 128
    assert _pas(inflight=2, running=14)._decode_inflight_reserved_blocks() == 128
    assert _pas(inflight=10, running=10, max_running=8)._decode_inflight_reserved_blocks() == 160
    assert _pas(inflight=15, running=40, max_running=16)._decode_inflight_reserved_blocks() == 1 << 30
    assert _pas(consumer=False, inflight=16)._decode_inflight_reserved_blocks() == 7


def test_throttle_on_dcp1_defaults_and_overrides(monkeypatch, base7):
    _clear(monkeypatch)
    monkeypatch.setenv("TPU_SCHED_DECODE_INFLIGHT_THROTTLE", "1")
    assert _pas(inflight=6, dcp=1)._decode_inflight_reserved_blocks() == 1 << 30
    assert _pas(inflight=2, running=3, max_running=16, dcp=1)._decode_inflight_reserved_blocks() == 512
    assert _pas(inflight=5, running=15, max_running=16, dcp=1)._decode_inflight_reserved_blocks() == 1 << 30
    assert _pas(inflight=5, running=30, max_running=64, dcp=1)._decode_inflight_reserved_blocks() == 560
    monkeypatch.setenv("TPU_SCHED_DECODE_MAX_STAGE3_INFLIGHT", "0")
    monkeypatch.setenv("TPU_SCHED_DECODE_RESERVED_BLOCKS", "3")
    assert _pas(inflight=9, running=0, max_running=64, dcp=1)._decode_inflight_reserved_blocks() == 144


def test_throttle_reaches_the_vllm_hook(monkeypatch, base7):
    # vLLM 0.31 calls _inflight_prefill_reserved_blocks() for every async KV load; the throttle
    # must act through it, not only through the helper.
    _clear(monkeypatch)
    monkeypatch.setenv("TPU_SCHED_DECODE_INFLIGHT_THROTTLE", "1")
    s = _pas(inflight=16, running=0)
    s._kv_prefetch_in_waiting = False
    assert s._inflight_prefill_reserved_blocks() == 1 << 30
    s = _pas(inflight=2, running=3)
    s._kv_prefetch_in_waiting = False
    assert s._inflight_prefill_reserved_blocks() == 128
