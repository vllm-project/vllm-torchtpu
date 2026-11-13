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
from vllm.config import ParallelConfig

from vllm_torchtpu import utils
from vllm_torchtpu.distributed.dp_world import (
    require_unified_dp_world,
    unified_dp_world,
)


@pytest.mark.parametrize(
    "ep,marked,expected",
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_unified_dp_world(ep, marked, expected):
    pc = ParallelConfig(data_parallel_size=2, enable_expert_parallel=ep)
    if marked:
        require_unified_dp_world(pc)
    assert unified_dp_world(pc) is expected
    assert utils.dp_engines_in_lockstep(pc) is expected


def test_mark_survives_pickling_to_workers():
    pc = ParallelConfig(data_parallel_size=2)
    require_unified_dp_world(pc)
    assert unified_dp_world(pickle.loads(pickle.dumps(pc)))


def test_single_engine_is_never_in_lockstep():
    pc = ParallelConfig(data_parallel_size=1)
    require_unified_dp_world(pc)
    assert not utils.dp_engines_in_lockstep(pc)
