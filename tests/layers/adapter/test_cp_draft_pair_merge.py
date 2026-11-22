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
"""cp_draft_attention's pair merge: what arming builds and that the armed
forward merges through it. Stubs; the kernel itself is
tests/kernels/test_pair_lse_merge.py."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.adapter import cp_draft_attention as cpd

TP = 8


def _layer(head_ranks):
    layer = nn.Module()
    layer.register_buffer("draft_dcp_rank", torch.tensor([1], dtype=torch.int32))
    group = SimpleNamespace(world_size=2, ranks=list(head_ranks), rank_in_group=1)
    layer.impl = SimpleNamespace(
        num_heads=8,
        head_size=128,
        sharded_draft_dcp=dict(head_group=group, ops=dict(attend="a")),
    )
    return layer


@pytest.fixture
def armed(monkeypatch):
    import vllm.distributed

    from vllm_torchtpu.distributed import pallas_all_reduce

    monkeypatch.setattr(
        vllm.distributed,
        "get_tp_group",
        lambda: SimpleNamespace(
            ranks=list(range(TP)), world_size=TP, rank_in_group=3, cpu_group=None
        ),
    )
    record = SimpleNamespace(mesh=object(), rank_ordered=True)
    monkeypatch.setattr(pallas_all_reduce, "armed_for", lambda ranks: record)
    built = []
    monkeypatch.setattr(
        cpd, "_pair_merge_op", lambda mesh, perm: built.append(perm) or "op"
    )
    monkeypatch.setattr(cpd, "_pair_selftest", lambda *a: True)
    pairs = [(2 * (r // 2), 2 * (r // 2) + 1) for r in range(TP)]

    def gather(seen, obj, group):
        if isinstance(obj, tuple):  # (refusal, head-group ranks) per rank
            seen[:] = [(obj[0], pairs[r]) for r in range(TP)]
        else:
            seen[:] = [obj] * TP

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    return built


def test_arming_builds_the_pairs_and_this_ranks_meta(armed):
    model = nn.ModuleList([_layer((2, 3)), _layer((2, 3))])
    model[1].impl.sharded_draft_dcp["head_group"] = model[0].impl.sharded_draft_dcp[
        "head_group"
    ]
    assert cpd.arm_pair_merge(model) == 2
    assert armed == [tuple((r, r ^ 1) for r in range(TP))]
    for layer in model:
        assert layer.impl.sharded_draft_dcp["ops"] == dict(attend="a", pair_merge="op")
        assert layer.draft_pair_meta.tolist() == [[2, 1]]  # peer 2, block 1


def test_a_four_rank_head_group_is_refused(armed):
    layer = _layer((0, 1, 2, 3))
    layer.impl.sharded_draft_dcp["head_group"].world_size = 4
    assert cpd.arm_pair_merge(nn.ModuleList([layer])) == 0
    assert not hasattr(layer, "draft_pair_meta")


def test_the_armed_forward_merges_through_the_pair_op():
    calls = []
    ops = dict(
        attend=lambda q, *a: (q, torch.zeros(q.shape[:2])),
        pair_merge=lambda out, lse, meta: calls.append(meta)
        or (out[:, :1], lse[:, :1]),
        block=lambda q, k, v, s, c, out, lse: out + 1,
    )
    query = torch.zeros(4, 1, 128)
    meta = torch.tensor([[0, 1]], dtype=torch.int32)
    out = cpd.forward_with_dcp(
        query=query,
        key=torch.zeros(4, 1, 128),
        value=torch.zeros(4, 1, 128),
        kv_cache=torch.zeros(1),
        attn_metadata=SimpleNamespace(
            seq_lens=None, block_tables=None, query_start_loc=None
        ),
        dcp_rank=torch.tensor([0]),
        dcp=2,
        kv_head_replicas=2,
        head_group=None,
        k_scale=None,
        v_scale=None,
        ops=ops,
        pair_meta=meta,
    )
    assert calls == [meta] and torch.equal(out, torch.ones(4, 1, 128))
