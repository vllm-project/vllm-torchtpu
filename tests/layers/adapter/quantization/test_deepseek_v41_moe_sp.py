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
import torch

from vllm_torchtpu.layers.adapter.quantization import deepseek_v41 as mod


def _moe(*, sp, hash_table, naive):
    moe = object.__new__(mod.DeepseekV41MoE)
    object.__setattr__(moe, "use_sequence_parallel", sp)
    object.__setattr__(moe, "gate", SimpleNamespace(tid2eid=hash_table))
    object.__setattr__(moe, "experts", SimpleNamespace(do_naive_dispatch_combine=naive))
    return moe


@pytest.mark.parametrize(
    "sp,hash_table,naive,gathered",
    [
        (True, torch.zeros(4, 2), True, True),
        (False, torch.zeros(4, 2), True, False),
        (True, None, True, False),
        # An armed fused-EP layer routes this rank's own rows.
        (True, torch.zeros(4, 2), False, False),
    ],
)
def test_hash_ids_follow_sp_dispatch(monkeypatch, sp, hash_table, naive, gathered):
    ep = Mock()
    ep.all_gather.side_effect = lambda x, dim: torch.cat([x, x + 10], dim=dim)
    monkeypatch.setattr(mod, "get_ep_group", lambda: ep)
    seen = {}

    def upstream_forward(self, hidden_states, input_ids=None):
        seen["ids"] = input_ids
        return hidden_states

    monkeypatch.setattr(mod.DeepseekV4MoE, "forward", upstream_forward)
    ids = torch.tensor([1, 2])
    hidden = torch.ones(2, 3)
    assert _moe(sp=sp, hash_table=hash_table, naive=naive).forward(hidden, ids) is (
        hidden
    )
    if gathered:
        ep.all_gather.assert_called_once_with(ids, dim=0)
        torch.testing.assert_close(seen["ids"], torch.tensor([1, 2, 11, 12]))
    else:
        ep.all_gather.assert_not_called()
        assert seen["ids"] is ids
