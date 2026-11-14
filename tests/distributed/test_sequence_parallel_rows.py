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
"""Row gate of sequence-parallel MoE (TPU_SP_MIN_ROWS) and its communicator."""

from unittest.mock import Mock

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed import sequence_parallel_rows as rows
from vllm_torchtpu.distributed import tpu_communicator as comm


@pytest.fixture
def communicator(monkeypatch):
    def init(instance, *args, **kwargs):
        instance.use_all2all = False

    monkeypatch.setattr(comm.DeviceCommunicatorBase, "__init__", init)
    return comm.TpuDeviceCommunicator()


@pytest.fixture
def ep_group(monkeypatch):
    group = Mock(world_size=4)
    group.all_gather.side_effect = lambda x, dim=0: torch.cat([x] * 4, dim=dim)
    group.reduce_scatter.side_effect = lambda x, dim=0: x.chunk(4, dim=dim)[0]
    group.all_reduce.side_effect = lambda x: x * 4
    monkeypatch.setattr(comm, "get_ep_group", lambda: group)
    return group


@pytest.mark.parametrize(
    "min_rows,num_tokens,expected",
    [(0, 8, False), (16, 8, True), (16, 15, True), (16, 16, False), (16, None, False)],
)
def test_replicated_rows_step(monkeypatch, min_rows, num_tokens, expected):
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", min_rows)
    assert rows.replicated_rows_step(num_tokens) is expected


def test_trace_outside_warmup_stays_sequence_parallel(monkeypatch):
    """Only the backbone warmup traces each side of the threshold apart."""
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", 16)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(rows.shape_variants, "aot_cache_tag", lambda: None)
    assert not rows.replicated_rows_step(8)
    monkeypatch.setattr(rows.shape_variants, "aot_cache_tag", lambda: "trace1")
    assert rows.replicated_rows_step(8)


def test_backbone_step_scopes_the_token_count(monkeypatch):
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", 16)
    assert not rows.replicated_rows_moe_step()
    with rows.backbone_step(8):
        assert rows.replicated_rows_moe_step()
    # Other forwards (the MTP drafter) run after the backbone step.
    assert not rows.replicated_rows_moe_step()
    with pytest.raises(RuntimeError), rows.backbone_step(8):
        raise RuntimeError
    assert not rows.replicated_rows_moe_step()


@pytest.mark.parametrize("min_rows,num_tokens", [(0, 8), (16, 16), (16, 64)])
def test_sequence_parallel_step_gathers_and_reduce_scatters(
    monkeypatch, communicator, ep_group, min_rows, num_tokens
):
    """At or above the threshold (or with it unset) the communicator is main's."""
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", min_rows)
    x = torch.arange(6.0).reshape(2, 3)
    logits = torch.ones(2, 5)
    with rows.backbone_step(num_tokens):
        hidden, gathered_logits = communicator.dispatch_router_logits(
            x, logits, is_sequence_parallel=True
        )
        combined = communicator.combine(torch.ones(8, 3), is_sequence_parallel=True)
    assert hidden.shape == (8, 3) and gathered_logits.shape == (8, 5)
    assert combined.shape == (2, 3)
    ep_group.all_reduce.assert_not_called()


def test_replicated_step_skips_gather_and_all_reduces(
    monkeypatch, communicator, ep_group
):
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", 16)
    x = torch.arange(6.0).reshape(2, 3)
    logits = torch.ones(2, 5)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    with rows.backbone_step(8):
        dispatched = communicator.dispatch_router_logits(
            x, logits, is_sequence_parallel=True
        )
        dispatched_topk = communicator.dispatch(
            x, topk_weights, topk_ids, is_sequence_parallel=True
        )
        combined = communicator.combine(x, is_sequence_parallel=True)
    assert all(a is b for a, b in zip(dispatched, (x, logits)))
    assert all(a is b for a, b in zip(dispatched_topk, (x, topk_weights, topk_ids)))
    torch.testing.assert_close(combined, x * 4)
    ep_group.all_gather.assert_not_called()
    ep_group.reduce_scatter.assert_not_called()


def test_replicated_step_leaves_data_parallel_moe_alone(
    monkeypatch, communicator, ep_group
):
    """The gate only concerns sequence-parallel MoE calls."""
    monkeypatch.setattr(envs, "TPU_SP_MIN_ROWS", 16)
    dp_group = Mock(world_size=2)
    dp_group.reduce_scatter.side_effect = lambda x, dim=0: x.chunk(2, dim=dim)[0]
    monkeypatch.setattr(comm, "get_dp_group", lambda: dp_group)
    with rows.backbone_step(8):
        combined = communicator.combine(torch.ones(4, 3), is_sequence_parallel=False)
    assert combined.shape == (2, 3)
    ep_group.all_reduce.assert_not_called()
