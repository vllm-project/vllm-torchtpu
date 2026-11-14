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
"""The decode admission throttle (_decode_inflight_reserved_blocks): on for a
KV consumer with DCP > 1."""

from unittest.mock import patch

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

# ---------------- decode admission throttle ----------------


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
    with patch.object(
        Scheduler, "_inflight_prefill_reserved_blocks", return_value=7, create=True
    ):
        yield


def test_throttle_off_at_dcp1_is_the_stock_reservation(base7):
    for inflight in (0, 16, 40):
        s = _pas(inflight=inflight, running=16, dcp=1)
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
    assert _pas(consumer=False, inflight=16)._decode_inflight_reserved_blocks() == 7


def test_throttle_reaches_the_vllm_hook(base7):
    # vLLM calls _inflight_prefill_reserved_blocks() for every async KV load;
    # the throttle must act through it, not only through the helper.
    s = _pas(inflight=16, running=0)
    s._kv_prefetch_in_waiting = False
    assert s._inflight_prefill_reserved_blocks() == 1 << 30
    s = _pas(inflight=2, running=3)
    s._kv_prefetch_in_waiting = False
    assert s._inflight_prefill_reserved_blocks() == 128
