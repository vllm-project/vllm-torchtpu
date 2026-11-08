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
"""Unquantized (bf16) MoE layers on the ragged all-to-all EP exchange
(the path a2a_unquantized): which layers claim the exchange, and what the
forward hands the op."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import decode_switches
from vllm_torchtpu.layers.adapter import ragged_a2a_moe as ragged_a2a
from vllm_torchtpu.layers.adapter.quantization import unquantized


def _method(use_ep=True, has_bias=False):
    method = object.__new__(unquantized.VllmUnquantizedFusedMoEMethod)
    method.moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=use_ep), has_bias=has_bias
    )
    method._tpu_activation_str = "silu"
    method._a2a_unquantized = decode_switches.get().path("a2a_unquantized")
    return method


@pytest.mark.parametrize(
    "ragged,off,use_ep,has_bias,expected",
    [
        ("1", "", True, False, True),
        ("1", "a2a_unquantized", True, False, False),
        ("0", "", True, False, False),
        ("1", "", False, False, False),
        # A layer with expert biases keeps vLLM's exchange.
        ("1", "", True, True, False),
    ],
)
def test_layers_claim_the_exchange_only_when_armed(
    monkeypatch, ragged, off, use_ep, has_bias, expected
):
    monkeypatch.setenv("TPU_MOE_RAGGED_A2A", ragged)
    monkeypatch.setenv("TPU_DECODE_FAST_PATHS_OFF", off)
    monkeypatch.setattr(
        unquantized, "enable_pipelined_collective_and_compute", lambda: False
    )
    method = _method(use_ep, has_bias)
    assert unquantized._uses_ragged_a2a(method) is expected
    # Claiming the exchange removes vLLM's all-gather / reduce-scatter pair.
    assert method.supports_internal_mk is expected


@pytest.mark.parametrize("shared", [False, True])
def test_forward_hands_unscaled_experts_to_the_ragged_op(monkeypatch, shared):
    monkeypatch.setenv("TPU_MOE_RAGGED_A2A", "1")
    x = torch.zeros(4, 8)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    weights = torch.ones(4, 2)
    monkeypatch.setattr(
        unquantized.moe_routing, "route", lambda layer, x, logits: (weights, ids)
    )
    seen = {}

    def ragged(**kwargs):
        seen.update(kwargs)
        return "ragged"

    monkeypatch.setattr(unquantized, "ragged_a2a_fused_moe_gmm", ragged)
    monkeypatch.setattr(
        unquantized, "fused_moe_gmm", lambda **kwargs: pytest.fail("default path")
    )
    shared_output = torch.ones(4, 8) if shared else None
    layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 8, 16),
        w2_weight=torch.zeros(2, 8, 8),
        _experts_start=torch.zeros((), dtype=torch.int32),
        moe_config=SimpleNamespace(experts_per_token=2),
        _tpu_owner_shared_output=shared_output,
        _tpu_owner_combine=shared,
    )
    out = _method()._forward_monolithic_tpu(layer, x, torch.zeros(4, 2))
    assert out == "ragged"
    assert seen["w1_scale"] is None and seen["w2_scale"] is None
    assert seen["w1_bias"] is None and seen["w2_bias"] is None
    assert seen["hidden_states"] is x and seen["topk_ids"] is ids
    # The ragged op never skips routes by weight and takes no such argument.
    assert "skip_padded_tokens" not in seen
    assert seen["owner_combine"] is shared
    if shared:
        assert seen["shared_output"] is shared_output
    else:
        assert "shared_output" not in seen


@pytest.mark.parametrize("shared", [False, True])
def test_unscaled_op_takes_no_scale_operands(monkeypatch, shared):
    monkeypatch.setattr(ragged_a2a, "_OPS", {})
    monkeypatch.setattr(
        ragged_a2a, "get_dp_tp_mesh", lambda: SimpleNamespace(shape={"dp": 2, "tp": 4})
    )
    monkeypatch.setattr(ragged_a2a, "_log_capacity", lambda *args: None)
    calls = []

    def _ragged_a2a_moe(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(ragged_a2a, "ragged_a2a_moe", _ragged_a2a_moe)
    built = {}

    class _Op:
        def register_fake(self, fn):
            pass

    def _jax_op(name, fn, **kwargs):
        built.update(kwargs)
        fn(*range(len(kwargs["input_partition_specs"])))
        return _Op()

    monkeypatch.setattr(ragged_a2a.pallas, "jax_op", _jax_op)
    ragged_a2a._get_op(
        topk=10,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
        shared=shared,
        scaled=False,
    )
    assert len(built["input_partition_specs"]) == 6 + shared
    ((args, kwargs),) = calls
    # x, w1, w2, then None scales, then routing and experts_start.
    assert args == (0, 1, 2, None, None, 3, 4, 5)
    assert kwargs.get("shared") == (6 if shared else None)
