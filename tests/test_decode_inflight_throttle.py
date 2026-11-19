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
"""The decode admission throttle (_decode_inflight_reserved_blocks): option
``tpu_scheduler.decode_inflight_throttle``, on a KV consumer with DCP > 1."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

# ---------------- decode admission throttle ----------------


def _pas(on=True, inflight=0, running=0, max_running=16):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    s._decode_throttle_switch = on
    s._inflight_prefills = [object()] * inflight
    s.running = [object()] * running
    s.max_num_running_reqs = max_running
    return s


@pytest.fixture
def base7():
    with patch.object(
        Scheduler, "_inflight_prefill_reserved_blocks", return_value=7, create=True
    ):
        yield


def test_throttle_off_is_the_stock_reservation(base7):
    for inflight in (0, 16, 40):
        s = _pas(on=False, inflight=inflight, running=16)
        s._kv_prefetch_in_waiting = False
        assert s._decode_inflight_reserved_blocks() == 7
        assert s._inflight_prefill_reserved_blocks() == 7


def test_throttle_on_dcp4_defaults(base7):
    assert _pas(inflight=16, running=0)._decode_inflight_reserved_blocks() == 1 << 30
    assert _pas(inflight=15, running=0)._decode_inflight_reserved_blocks() == 128
    assert _pas(inflight=2, running=14)._decode_inflight_reserved_blocks() == 128
    assert (
        _pas(inflight=10, running=10, max_running=8)._decode_inflight_reserved_blocks()
        == 160
    )
    assert (
        _pas(inflight=15, running=40, max_running=16)._decode_inflight_reserved_blocks()
        == 1 << 30
    )
    assert _pas(on=False, inflight=16)._decode_inflight_reserved_blocks() == 7


def test_throttle_reaches_the_vllm_hook(base7):
    # vLLM calls _inflight_prefill_reserved_blocks() for every async KV load;
    # the throttle must act through it, not only through the helper.
    s = _pas(inflight=16, running=0)
    s._kv_prefetch_in_waiting = False
    assert s._inflight_prefill_reserved_blocks() == 1 << 30
    s = _pas(inflight=2, running=3)
    s._kv_prefetch_in_waiting = False
    assert s._inflight_prefill_reserved_blocks() == 128


def _built(monkeypatch, options, kv_role="kv_consumer", dcp=4):
    """Runs PrefixAwareScheduler.__init__ over a stub base __init__."""

    def fake_init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_role=kv_role,
                is_kv_consumer=kv_role == "kv_consumer",
                is_kv_producer=kv_role == "kv_producer",
            ),
            additional_config={"tpu_scheduler": options},
        )
        self.connector = object()
        self.kv_cache_manager = MagicMock()
        self.dcp_world_size = dcp

    monkeypatch.setattr(AsyncScheduler, "__init__", fake_init)
    monkeypatch.setattr(PrefixAwareScheduler, "_enable_memo_xstep", lambda self: None)
    return PrefixAwareScheduler()


def test_the_throttle_is_the_option(monkeypatch):
    assert _built(monkeypatch, {})._decode_throttle_switch is False
    s = _built(monkeypatch, {"decode_inflight_throttle": True})
    assert s._decode_throttle_switch is True


@pytest.mark.parametrize("kv_role, dcp", [("kv_consumer", 1), ("kv_producer", 4)])
def test_the_throttle_is_refused_off_a_dcp_consumer(monkeypatch, kv_role, dcp):
    with pytest.raises(ValueError, match="decode_inflight_throttle"):
        _built(monkeypatch, {"decode_inflight_throttle": True}, kv_role, dcp)
