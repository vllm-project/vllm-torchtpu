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
"""Tests for the MoE combine's hierarchical reduce-scatter."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

RANKS = tuple(range(8))
HIDDEN = 512


class _Group:
    def __init__(self, ranks=RANKS):
        self.ranks = list(ranks)
        self.world_size = len(ranks)
        self.calls = 0

    def reduce_scatter(self, x, dim):
        assert dim == 0
        self.calls += 1
        return x[: x.shape[0] // self.world_size]


class _Op:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x[: x.shape[0] // 8]


@pytest.fixture
def armed(monkeypatch):
    op = _Op()
    monkeypatch.setattr(hrs, "_ARMED", hrs._Armed(op=op, ranks=RANKS))
    return op


def test_not_armed_uses_group(monkeypatch):
    monkeypatch.setattr(hrs, "_ARMED", None)
    group = _Group()
    hrs.moe_combine_reduce_scatter(group, torch.zeros(64, HIDDEN, dtype=torch.bfloat16))
    assert group.calls == 1


def test_armed_and_eligible_uses_kernel(armed):
    group = _Group()
    out = hrs.moe_combine_reduce_scatter(
        group, torch.zeros(64, HIDDEN, dtype=torch.bfloat16)
    )
    assert armed.calls == 1 and group.calls == 0
    assert out.shape == (8, HIDDEN)


@pytest.mark.parametrize(
    "x, group",
    [
        pytest.param(
            torch.zeros(64, HIDDEN, dtype=torch.float32), _Group(), id="not_bf16"
        ),
        pytest.param(
            torch.zeros(64, 4, 4, dtype=torch.bfloat16), _Group(), id="not_2d"
        ),
        pytest.param(
            torch.zeros(60, HIDDEN, dtype=torch.bfloat16),
            _Group(),
            id="rows_not_divisible",
        ),
        pytest.param(
            torch.zeros(56, HIDDEN, dtype=torch.bfloat16),
            _Group(),
            id="under_8_output_rows",
        ),
        pytest.param(
            torch.zeros(64, 2816, dtype=torch.bfloat16),
            _Group(),
            id="hidden_not_splittable",
        ),
        pytest.param(
            torch.zeros(64, HIDDEN, dtype=torch.bfloat16),
            _Group(ranks=range(8, 16)),
            id="other_group",
        ),
    ],
)
def test_ineligible_calls_use_group(armed, x, group):
    hrs.moe_combine_reduce_scatter(group, x)
    assert armed.calls == 0 and group.calls == 1


def test_prebuild_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(hrs, "_ARMED", None)
    monkeypatch.setattr(hrs.envs, "TPU_MOE_EP_HIER_RS", False, raising=False)
    monkeypatch.setattr(
        hrs,
        "_unsupported_world_reason",
        lambda: pytest.fail("must not inspect the world when disabled"),
    )
    assert hrs.prebuild_hier_reduce_scatter() is False


def test_world_reason_rejects_wrong_size(monkeypatch):
    import torch.distributed as dist
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import ep_mesh

    monkeypatch.setattr(
        ep_mesh,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3]),
    )
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(ranks=[0, 1, 2, 3])
    )
    assert "EP group of 4" in hrs._unsupported_world_reason()
