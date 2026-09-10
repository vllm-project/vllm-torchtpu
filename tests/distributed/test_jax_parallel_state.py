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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_torchtpu.distributed import jax_parallel_state as state


@pytest.fixture(autouse=True)
def isolated_pp_group(monkeypatch):
    monkeypatch.setattr(state, "_PP", None)


def test_pp_group_requires_initialization():
    with pytest.raises(AssertionError, match="not initialized"):
        state.get_pp_group()


@pytest.mark.parametrize("rank,size,first,last", [(0, 1, True, True),
                                                  (0, 3, True, False),
                                                  (1, 3, False, False),
                                                  (2, 3, False, True)])
def test_pp_rank_boundaries(rank, size, first, last):
    group = state.GroupCoordinator(rank, size)
    assert group.is_first_rank is first
    assert group.is_last_rank is last


@pytest.mark.parametrize("need_pp", [False, True])
def test_pp_transfer_initialization(monkeypatch, need_pp):
    server = Mock()
    start = Mock(return_value=server)
    monkeypatch.setattr(state.transfer, "start_transfer_server", start)
    device = SimpleNamespace(client=object())
    state.init_pp_distributed_environment("10.0.0.2", 2, 4, device, need_pp)
    group = state.get_pp_group()
    assert (group.rank_in_group, group.world_size) == (2, 4)
    assert group.connection is None
    if not need_pp:
        start.assert_not_called()
        assert group.transfer_server is None
        return
    start.assert_called_once_with(device.client, "10.0.0.2:5002",
                                  ["10.0.0.2:0", "10.0.0.2:0"])
    assert group.transfer_server is server
    state.connect("10.0.0.1", 1)
    server.connect.assert_called_once_with("10.0.0.1:5001")
    assert group.connection is server.connect.return_value
    tensors = {"hidden": object()}
    spec = {"hidden": object()}
    group.send_tensor_dict(27, tensors)
    server.await_pull.assert_called_once_with(27, tensors)
    result = group.recv_tensor_dict(28, spec)
    group.connection.pull.assert_called_once_with(28, spec)
    assert result is group.connection.pull.return_value
