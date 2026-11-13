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
"""A graph calling a rank-dependent op gets a per-rank fingerprint."""

import torch
import torch.distributed as dist
import torch.fx as fx

from vllm_torchtpu.compilation import graph_fingerprint as gf


def sequence_parallel_chunk(x):
    return x[:1]


torch.fx.wrap("sequence_parallel_chunk")


class SP(torch.nn.Module):
    def forward(self, x):
        return sequence_parallel_chunk(x) + 1


class Plain(torch.nn.Module):
    def forward(self, x):
        return x + 1


def _fp(module, monkeypatch, rank):
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    return gf._graph_fingerprint(fx.symbolic_trace(module))


def test_rank_dependent_graph_fingerprint_differs_per_rank(monkeypatch):
    gm = fx.symbolic_trace(SP())
    assert gf.graph_is_rank_dependent(gm)
    assert _fp(SP(), monkeypatch, 0) != _fp(SP(), monkeypatch, 3)


def test_other_graphs_keep_one_fingerprint(monkeypatch):
    assert not gf.graph_is_rank_dependent(fx.symbolic_trace(Plain()))
    assert _fp(Plain(), monkeypatch, 0) == _fp(Plain(), monkeypatch, 3)


def test_mark_rank_dependent(monkeypatch):
    monkeypatch.setattr(gf, "_RANK_DEPENDENT_OPS", set(gf._RANK_DEPENDENT_OPS))
    assert not gf.graph_is_rank_dependent(fx.symbolic_trace(Plain()))

    class Custom(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.relu(x)

    gm = fx.symbolic_trace(Custom())
    assert not gf.graph_is_rank_dependent(gm)
    gf.mark_rank_dependent("relu")
    assert gf.graph_is_rank_dependent(gm)
