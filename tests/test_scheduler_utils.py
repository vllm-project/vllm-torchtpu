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
"""The helpers the TPU schedulers share (``core/scheduler_utils.py``)."""

from types import SimpleNamespace

import pytest
from vllm.config import KVTransferConfig

from vllm_torchtpu.core import scheduler_utils
from vllm_torchtpu.core.hybrid_retention import retention_wanted


@pytest.mark.parametrize(
    "kv_role,roles",
    [
        ("kv_producer", (True, False)),
        ("kv_consumer", (False, True)),
        ("kv_both", (False, False)),
        (None, (False, False)),
    ],
)
def test_one_helper_reads_the_kv_roles(kv_role, roles):
    """``scheduler_utils.kv_roles`` on vLLM's own KVTransferConfig: the roles
    both schedulers and hybrid-state retention read."""
    config = (
        None
        if kv_role is None
        else KVTransferConfig(kv_connector="TPURaidenConnector", kv_role=kv_role)
    )
    assert tuple(scheduler_utils.kv_roles(config)) == roles
    assert retention_wanted(SimpleNamespace(kv_transfer_config=config)) is roles[0]


def _r(rid, decoding, remaining=0):
    return SimpleNamespace(rid=rid, decoding=decoding, remaining=remaining)


def test_decoders_first_keeps_their_order_and_sorts_the_prefills_stably():
    running = [
        _r("p1", False, 30),
        _r("d1", True),
        _r("p2", False, 10),
        _r("p3", False, 30),
        _r("d2", True),
    ]
    alias = running
    decoding, prefilling = scheduler_utils.run_decoders_first(
        running, lambda r: r.decoding, lambda r: r.remaining
    )
    assert [r.rid for r in running] == ["d1", "d2", "p2", "p1", "p3"]
    # In place: vLLM holds the running list.
    assert running is alias
    assert [r.rid for r in decoding] == ["d1", "d2"]
    assert [r.rid for r in prefilling] == ["p2", "p1", "p3"]
    # Without a key the prefills keep their order.
    running = [_r("p1", False, 30), _r("d1", True), _r("p2", False, 10)]
    scheduler_utils.run_decoders_first(running, lambda r: r.decoding)
    assert [r.rid for r in running] == ["d1", "p1", "p2"]
