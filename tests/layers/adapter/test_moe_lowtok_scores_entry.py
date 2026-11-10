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
"""Arming the few-token kernel's scores entry on a MoE layer, and storing a
placement's copy slots apart from the primaries. CPU only."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import moe_lowtok_scores as scores_entry
from vllm_torchtpu.layers.adapter import moe_routing, token_padding

EP, LOCAL, COPIES, HIDDEN, INTER = 8, 64, 4, 512, 256


def _layer(slots=LOCAL, **changes):
    fp8 = torch.float8_e4m3fn
    layer = torch.nn.Module()
    for name, shape, dtype in (
        ("w13_weight", (slots, HIDDEN, 2 * INTER), fp8),
        ("w2_weight", (slots, INTER, HIDDEN), fp8),
        ("w13_weight_scale_inv", (slots, 1, 1, 2 * INTER), torch.float32),
        ("w2_weight_scale_inv", (slots, 1, 1, HIDDEN), torch.float32),
    ):
        # Each slot's values name the slot, so a split can be checked.
        values = torch.arange(slots, dtype=torch.float32)
        param = values.reshape(-1, *[1] * (len(shape) - 1)).expand(*shape).to(dtype)
        setattr(layer, name, torch.nn.Parameter(param.clone(), requires_grad=False))
    layer.global_num_experts = EP * LOCAL
    layer.renormalize = True
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.moe_config = SimpleNamespace(
        ep_rank=2,
        ep_size=EP,
        moe_parallel_config=SimpleNamespace(use_ep=True),
    )
    for key, value in changes.items():
        setattr(layer, key, value)
    return layer


@pytest.fixture
def scores_env(monkeypatch):
    monkeypatch.setenv("TPU_MOE_LOWTOK_FROM_SCORES", "1")
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "64")
    monkeypatch.delenv("TPU_MOE_LOWTOK_SCORES_STRICT", raising=False)
    monkeypatch.delenv("TPU_EXPERT_PLACEMENT", raising=False)


def test_armed_layer_supplies_its_tables(scores_env):
    layer = _layer()
    tables = scores_entry.arm(layer, "silu", adaptive=True)
    assert tables is not None and not tables.has_copies
    # Contiguous placement: rank 2 holds experts 128 .. 191.
    assert tables.tables[0] == 2
    assert tables.primary[130] == 2 and (tables.copy == -1).all()
    hidden = torch.zeros(5, HIDDEN, dtype=torch.bfloat16)
    operands = scores_entry.operands(layer, hidden)
    assert len(operands) == 9
    assert operands[0].tolist() == [0] * 5  # no padding state: no padded row
    # Without copies the op takes the primaries in the copy operands' place.
    assert operands[5] is layer.w13_weight


@pytest.mark.parametrize(
    "changes,adaptive,reason",
    [
        ({}, False, "adaptive"),
        ({"renormalize": False}, True, "renormalization"),
        ({"global_num_experts": 256}, True, "256 experts"),
    ],
)
def test_refusals(scores_env, monkeypatch, changes, adaptive, reason):
    layer = _layer(**changes)
    assert reason in scores_entry.scores_entry_refusal(layer, "silu", adaptive)
    assert scores_entry.arm(layer, "silu", adaptive) is None
    assert scores_entry.operands(layer, torch.zeros(1, HIDDEN)) == ()
    monkeypatch.setenv("TPU_MOE_LOWTOK_SCORES_STRICT", "1")
    with pytest.raises(RuntimeError, match="STRICT"):
        scores_entry.arm(layer, "silu", adaptive)


def test_unsupported_activation_is_refused(scores_env):
    assert "kernel" in scores_entry.scores_entry_refusal(_layer(), "gelu", True)


def test_cap_below_ep_size_or_replicated_tp_is_refused(scores_env, monkeypatch):
    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "4")
    assert "ep_size" in scores_entry.scores_entry_refusal(_layer(), "silu", True)

    monkeypatch.setenv("TPU_MOE_LOWTOK_MAX_TOKENS", "64")
    rep_layer = _layer(
        moe_config=SimpleNamespace(
            ep_rank=2,
            ep_size=EP,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, is_sequence_parallel=False, tp_size=2
            ),
        )
    )
    assert "replicated" in scores_entry.scores_entry_refusal(rep_layer, "silu", True)


def test_whole_expert_admits_single_lane_half_intermediate(scores_env, monkeypatch):
    fp8 = torch.float8_e4m3fn
    inter_128 = 128
    layer = _layer(
        w13_weight=torch.nn.Parameter(
            torch.zeros(LOCAL, HIDDEN, 2 * inter_128, dtype=fp8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros(LOCAL, inter_128, HIDDEN, dtype=fp8), requires_grad=False
        ),
        w13_weight_scale_inv=torch.nn.Parameter(
            torch.ones(LOCAL, 1, 1, 2 * inter_128, dtype=torch.float32),
            requires_grad=False,
        ),
        w2_weight_scale_inv=torch.nn.Parameter(
            torch.ones(LOCAL, 1, 1, HIDDEN, dtype=torch.float32), requires_grad=False
        ),
    )
    monkeypatch.delenv("TPU_MOE_LOWTOK_WHOLE_EXPERT", raising=False)
    assert "kernel" in scores_entry.scores_entry_refusal(layer, "silu", True)
    assert scores_entry.arm(layer, "silu", adaptive=True) is None

    monkeypatch.setenv("TPU_MOE_LOWTOK_WHOLE_EXPERT", "1")
    assert scores_entry.scores_entry_refusal(layer, "silu", True) is None
    tables = scores_entry.arm(layer, "silu", adaptive=True)
    assert tables is not None

    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    assert "TPU_MOE_LOWTOK_WHOLE_EXPERT" in TpuPlatform.additional_env_vars


def test_off_is_a_no_op(monkeypatch):
    monkeypatch.delenv("TPU_MOE_LOWTOK_FROM_SCORES", raising=False)
    layer = _layer()
    assert scores_entry.arm(layer, "silu", adaptive=False) is None
    assert scores_entry.operands(layer, torch.zeros(1, HIDDEN)) == ()


def test_padding_rows(monkeypatch):
    state = token_padding.TokenPaddingState.create(8, "cpu")
    monkeypatch.setattr(token_padding, "_padding_state", state)
    token_padding.mark_padding(3, 6)
    assert token_padding.local_padding_rows(6, "cpu").tolist() == [0, 0, 0, 1, 1, 1]


def test_copy_slots_are_stored_apart(monkeypatch):
    from vllm_torchtpu.layers.adapter.quantization import fp8

    layer = _layer(slots=LOCAL + COPIES)
    copies = fp8._split_copy_slots(layer)
    assert len(copies) == 4
    assert layer.w13_weight.shape[0] == LOCAL
    assert layer.w13_copy_weight.shape[0] == COPIES
    assert layer.w2_copy_weight_scale_inv.shape == (COPIES, 1, 1, HIDDEN)
    # The primaries keep slots 0 .. LOCAL-1; the copies are the slots after.
    assert layer.w2_weight_scale_inv[:, 0, 0, 0].tolist() == list(range(LOCAL))
    assert layer.w2_copy_weight_scale_inv[:, 0, 0, 0].tolist() == list(
        range(LOCAL, LOCAL + COPIES)
    )
    # A layer without copy slots is untouched.
    plain = _layer()
    assert fp8._split_copy_slots(plain) == []
    assert not hasattr(plain, "w13_copy_weight")


def test_unsplit_copy_slots_are_refused(monkeypatch):
    from vllm_torchtpu import expert_placement

    monkeypatch.setattr(expert_placement, "has_copies", lambda: True)
    with pytest.raises(NotImplementedError, match="FP8"):
        moe_routing.register_experts_start_buffer(
            _layer(slots=LOCAL + COPIES), device="cpu"
        )
