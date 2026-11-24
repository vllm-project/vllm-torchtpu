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

import pickle

import pytest

from vllm_torchtpu.distributed import mesh_utils
from vllm_torchtpu.distributed.mesh_utils import SliceHostOrderError

# tpu7x 2x2x2 as libtpu numbers it: within a host, chip order 0,1,4,5,6,7,2,3.
POD0_FIRST = [0, 1, 4, 5, 6, 7, 2, 3, 8, 9, 12, 13, 14, 15, 10, 11]
POD1_FIRST = POD0_FIRST[8:] + POD0_FIRST[:8]
HOST_OF_RANK = [0] * 8 + [1] * 8


@pytest.fixture(autouse=True)
def no_expectation(monkeypatch):
    monkeypatch.setattr(mesh_utils, "_HOST_OF_RANK", None)


def test_hosts_sort_by_their_lowest_device_id():
    assert mesh_utils.hosts_in_device_order(POD0_FIRST, HOST_OF_RANK) == (0, 1)
    assert mesh_utils.hosts_in_device_order(POD1_FIRST, HOST_OF_RANK) == (1, 0)


def test_nothing_is_checked_without_an_expectation():
    mesh_utils.check_hosts_in_device_order(POD1_FIRST)


def test_a_binding_out_of_device_order_names_the_order_to_rebind_in():
    mesh_utils.expect_hosts_in_device_order(HOST_OF_RANK)
    mesh_utils.check_hosts_in_device_order(POD0_FIRST)
    with pytest.raises(SliceHostOrderError) as err:
        mesh_utils.check_hosts_in_device_order(POD1_FIRST)
    assert err.value.host_order == (1, 0)
    assert err.value.device_ids == tuple(POD1_FIRST)


def test_the_error_survives_the_trip_back_to_the_executor():
    err = pickle.loads(pickle.dumps(SliceHostOrderError((1, 0), POD1_FIRST)))
    assert (err.host_order, err.device_ids) == ((1, 0), tuple(POD1_FIRST))
