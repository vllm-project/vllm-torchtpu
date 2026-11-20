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
"""The FP8 MoE method's fused EP v3 forward: which path each step takes.

A layer fused EP v3 armed takes the fused kernel for a step at or above the
kernel's gate. Below it, and on a replicated step under tensor parallelism,
the method runs vLLM's EP dispatch, then the few-token scores op when it
serves the layer, else the routed grouped path, then the EP combine.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import (
    fused_moe_ep_v3,
    sequence_parallel,
    small_all_reduce_op,
    token_padding,
)
from vllm_torchtpu.layers.adapter.quantization import fp8, moe_lowtok_scores

pytestmark = pytest.mark.cpu_test

GATE = 64


class _Unread:
    """A row count that fails the test if the gate compares against it."""

    def __ge__(self, other):
        raise AssertionError("the row count was compared against the gate")


@pytest.mark.parametrize(
    "tp, sp_active, min_rows, rows, takes",
    [
        (1, False, 0, GATE - 1, False),
        (1, False, 0, GATE, True),
        # A replicated step under TP stays off the kernel, whatever its rows.
        (2, False, 0, 4 * GATE, False),
        # Every sequence-parallel step already has gate rows per rank: settled
        # without reading the traced row count.
        (2, True, 2 * GATE, _Unread(), True),
        (2, True, GATE, GATE - 1, False),
        (2, True, GATE, GATE, True),
    ],
)
def test_fused_kernel_takes(monkeypatch, tp, sp_active, min_rows, rows, takes):
    monkeypatch.setattr(fused_moe_ep_v3, "tensor_parallel_size", lambda: tp)
    monkeypatch.setattr(sequence_parallel, "is_active", lambda: sp_active)
    monkeypatch.setattr(sequence_parallel, "min_rows", lambda *args: min_rows)
    assert fp8._fused_kernel_takes(rows, GATE) is takes


class _EpGroup:
    def __init__(self):
        self.calls = []

    def dispatch_router_logits(self, x, logits, sequence_parallel_flag):
        self.calls.append(("dispatch", x.shape[0], sequence_parallel_flag))
        return x, logits

    def combine(self, out, sequence_parallel_flag):
        self.calls.append(("combine", out.shape[0], sequence_parallel_flag))
        return out


def _identity(out):
    return out


@pytest.fixture
def v3(monkeypatch):
    """An FP8 MoE method whose layer fused EP v3 armed, with the kernel, the
    EP group, the scores op and the grouped path recorded, not run."""
    import vllm.distributed.parallel_state as parallel_state

    calls = SimpleNamespace(fused=[], scores=[], grouped=[], ep=_EpGroup())

    def fused(x, *args):
        calls.fused.append(x.shape[0])
        return torch.zeros_like(x)

    def scores(op, layer, x, logits):
        calls.scores.append(x.shape[0])
        return torch.zeros_like(x)

    def grouped(**kwargs):
        calls.grouped.append(kwargs["hidden_states"].shape[0])
        return torch.zeros_like(kwargs["hidden_states"])

    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", False)
    monkeypatch.setattr(fused_moe_ep_v3, "_OP", object())
    monkeypatch.setattr(fused_moe_ep_v3, "fused_moe_ep", fused)
    monkeypatch.setattr(fused_moe_ep_v3, "step_min_tokens", lambda: GATE)
    monkeypatch.setattr(fused_moe_ep_v3, "retain_fused_ep_graph_inputs", _identity)
    monkeypatch.setattr(small_all_reduce_op, "retain", _identity)
    monkeypatch.setattr(token_padding, "retain_padding_graph_input", _identity)
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: calls.ep)
    monkeypatch.setattr(moe_lowtok_scores, "apply", scores)
    monkeypatch.setattr(fp8, "fused_moe_gmm", grouped)
    monkeypatch.setattr(
        fp8.moe_routing,
        "route",
        lambda layer, x, logits: (
            torch.ones(x.shape[0], 2),
            torch.zeros(x.shape[0], 2, dtype=torch.int32),
        ),
    )
    method = object.__new__(fp8.VllmFp8MoEMethodTPU)
    method._tpu_activation_str = "silu"
    method._tpu_fused_ep_eligible = True
    calls.method = method
    return calls


def _layer(sequence_parallel_flag):
    return SimpleNamespace(
        w13_weight=torch.zeros(2, 8, 16),
        w2_weight=torch.zeros(2, 8, 8),
        w13_weight_scale_inv=torch.ones(2, 1, 1, 16),
        w2_weight_scale_inv=torch.ones(2, 1, 1, 8),
        _experts_start=torch.zeros((), dtype=torch.int32),
        _gmm_primary_experts=None,
        _lowtok_has_copies=False,
        moe_config=SimpleNamespace(
            is_sequence_parallel=sequence_parallel_flag, experts_per_token=2
        ),
    )


@pytest.mark.parametrize(
    "tp, sp_active, min_rows, rows, scores_op, path",
    [
        (1, False, 0, GATE, False, "fused"),
        (1, False, 0, GATE - 1, True, "scores"),
        (1, False, 0, GATE - 1, False, "grouped"),
        # A replicated TP step is dispatched and combined, not fused.
        (2, False, 0, 4 * GATE, True, "scores"),
        # The threshold settles a sequence-parallel step at any row count.
        (2, True, 2 * GATE, 8, True, "fused"),
        (2, True, GATE, 8, True, "scores"),
    ],
)
def test_the_v3_forward_takes_one_path_per_step(
    monkeypatch, v3, tp, sp_active, min_rows, rows, scores_op, path
):
    monkeypatch.setattr(fused_moe_ep_v3, "tensor_parallel_size", lambda: tp)
    monkeypatch.setattr(sequence_parallel, "is_active", lambda: sp_active)
    monkeypatch.setattr(sequence_parallel, "min_rows", lambda *args: min_rows)
    v3.method._moe_lowtok_scores_op = Mock(name="scores_op") if scores_op else None
    flag = tp > 1
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    out = v3.method.apply_monolithic(_layer(flag), x, torch.zeros(rows, 2))
    assert out.shape == x.shape
    taken = {
        "fused": v3.fused,
        "scores": v3.scores,
        "grouped": v3.grouped,
    }
    assert {name: calls for name, calls in taken.items() if calls} == {path: [rows]}
    if path == "fused":
        assert v3.ep.calls == []
    else:
        # The EP exchange brackets the per-rank path, with the layer's flag.
        assert v3.ep.calls == [("dispatch", rows, flag), ("combine", rows, flag)]
