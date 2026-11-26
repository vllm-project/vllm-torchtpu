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
"""VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH: refresh a request's resident host chain
at its finish (write-through) or at its write-back flush."""

import collections
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.offload.test_raiden_store_offload import _group_key


def _scheduler(write_back=None, lru_mode2=False):
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    s = TPURaidenOffloadingScheduler.__new__(TPURaidenOffloadingScheduler)
    s.config = SimpleNamespace(
        kv_group_configs=[
            SimpleNamespace(is_eagle_group=False),
            SimpleNamespace(is_eagle_group=True),
        ],
        blocks_per_chunk=1,
    )
    s._raiden_manager = MagicMock()
    s.manager = s._raiden_manager
    s._req_status = {}
    s.wb_counters = collections.Counter()
    s._write_back = write_back
    s._lru_refresh_on_finish = lru_mode2
    s._wb_blocks = collections.OrderedDict()
    s._wb_owners = {}
    s._block_pool = None
    return s


KEYS = [_group_key(i, 0) for i in range(3)]


def _add_req(s, req_id="r"):
    s._req_status[req_id] = SimpleNamespace(
        group_states=[
            SimpleNamespace(offload_keys=KEYS),
            SimpleNamespace(offload_keys=[_group_key(9, 1)]),
        ]
    )


def _finish(s, req_id="r"):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        OffloadingConnectorScheduler,
    )

    with patch.object(
        OffloadingConnectorScheduler, "request_finished", return_value=(False, None)
    ):
        s.request_finished(SimpleNamespace(request_id=req_id))


def test_flag_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH", raising=False)
    assert envs.VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH is False
    s = _scheduler()
    _add_req(s)
    _finish(s)
    s._raiden_manager.refresh_resident.assert_not_called()


def test_write_through_refreshes_at_finish(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH", "1")
    s = _scheduler()
    _add_req(s)
    _finish(s)
    # Eagle groups are skipped.
    s._raiden_manager.refresh_resident.assert_called_once_with(KEYS)


@pytest.mark.parametrize("lru_mode2", [False, True])
def test_write_back_defers_refresh_to_the_flush(monkeypatch, lru_mode2):
    from vllm_torchtpu.offload.raiden_connector import WriteBackPolicy

    monkeypatch.setenv("VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH", "1")
    s = _scheduler(WriteBackPolicy(), lru_mode2=lru_mode2)
    _add_req(s)
    _finish(s)
    # TPU_RAIDEN_LRU_REFRESH=2 refreshes at the finish as before; the flag
    # alone waits for the flush.
    assert s._raiden_manager.refresh_resident.call_count == (1 if lru_mode2 else 0)
    s._raiden_manager.refresh_resident.reset_mock()
    s._wb_blocks["r"] = {5}
    s._wb_owners[5] = {"r"}
    out = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        kv_cache_block_copies=None,
        num_scheduled_tokens={},
    )
    with (
        patch(
            "vllm_torchtpu.offload.raiden_connector._allocated_block_ids",
            return_value=set(),
        ),
        patch.object(type(s), "_wb_near_eviction", return_value=["r"]),
        patch("dataclasses.replace", side_effect=lambda o, **kw: kw),
    ):
        result = s._write_back_output(out)
    assert result["finished_req_ids"] == {"r"}
    if lru_mode2:
        s._raiden_manager.refresh_resident.assert_not_called()
    else:
        s._raiden_manager.refresh_resident.assert_called_once_with(KEYS)
