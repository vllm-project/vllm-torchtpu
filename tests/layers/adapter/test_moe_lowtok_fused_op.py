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
"""Which deployments arm the fused few-token collectives, which steps take
them, and how the FP8 MoE method hands a step to them.

The kernel itself runs in TPU interpret mode on 8 CPU devices in
tests/kernels/experimental/test_moe_lowtok_fused_collectives.py.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.layers.adapter import moe_lowtok_fused_op as op
from vllm_torchtpu.layers.adapter import small_all_reduce_op

pytestmark = pytest.mark.cpu_test

TOPK = 10


def _config(additional_config=None):
    return SimpleNamespace(additional_config=additional_config or {})


@pytest.fixture
def fresh(monkeypatch):
    """Unarmed module state, the op build stubbed, and rank tensors on the host."""
    for name, value in (
        ("_OPS", {}),
        ("_RANK", None),
        ("_DP", 0),
        ("_TP", 0),
        ("_ARMED", False),
        ("_CHECKED", False),
        ("_FP8_DISPATCH", False),
    ):
        monkeypatch.setattr(op, name, value)
    build = Mock(name="_build_op")
    monkeypatch.setattr(op, "_build_op", build)
    host_tensor = torch.tensor

    def tensor(*args, device=None, **kwargs):
        return host_tensor(*args, **kwargs)

    monkeypatch.setattr(op.torch, "tensor", tensor)
    return build


def _layout(monkeypatch, *, dp, tp, cores=8, mesh=True, groups_match=True, index=5):
    """The model-parallel groups and EP mesh prebuild reads; the groups check
    (a collective over the EP group) as a mock."""
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(
        op,
        "build_ep_mesh",
        lambda: SimpleNamespace(shape={EP_AXIS_NAME: cores}) if mesh else None,
    )
    monkeypatch.setattr(ps, "get_tp_group", lambda: SimpleNamespace(world_size=tp))
    monkeypatch.setattr(ps, "get_dp_group", lambda: SimpleNamespace(world_size=dp))
    monkeypatch.setattr(op, "ep_mesh_index", lambda: index)
    check = Mock(name="_groups_match_mesh", return_value=groups_match)
    monkeypatch.setattr(op, "_groups_match_mesh", check)
    return check


def test_dp4_tp2_arms_the_op(monkeypatch, fresh):
    check = _layout(monkeypatch, dp=4, tp=2, index=5)
    op.prebuild(_config(), TOPK)
    assert op._ARMED is True
    assert (op._DP, op._TP) == (4, 2)
    # Mesh index 5 is DP rank 2, TP rank 1 of the (dp, tp) view.
    assert op._RANK.tolist() == [2, 1]
    check.assert_called_once_with(5, 4, 2)
    fresh.assert_called_once_with(TOPK)


@pytest.mark.parametrize(
    "layout",
    [
        dict(dp=1, tp=8),
        dict(dp=8, tp=1),
        dict(dp=2, tp=4),
        dict(dp=4, tp=2, cores=16),
    ],
    ids=["tp8", "dp8_tp1", "dp2_tp4", "not_the_mesh"],
)
def test_other_layouts_leave_it_unarmed_without_the_groups_collective(
    monkeypatch, fresh, layout
):
    """TP8 (the other IX decode layout), TP other than a chip's pair, or a
    DP x TP that is not the EP mesh: the default MoE path, and no collective."""
    check = _layout(monkeypatch, **layout)
    op.prebuild(_config(), TOPK)
    assert op._ARMED is False
    assert op._RANK is None
    check.assert_not_called()
    fresh.assert_not_called()


def test_tp_groups_off_the_mesh_blocks_leave_it_unarmed(monkeypatch, fresh):
    _layout(monkeypatch, dp=4, tp=2, groups_match=False)
    op.prebuild(_config(), TOPK)
    assert op._ARMED is False
    fresh.assert_not_called()


def test_no_expert_parallel_mesh_leaves_it_unarmed(monkeypatch, fresh):
    _layout(monkeypatch, dp=4, tp=2, mesh=False)
    op.prebuild(_config(), TOPK)
    assert op._ARMED is False
    fresh.assert_not_called()


def test_prebuild_checks_the_layout_once(monkeypatch, fresh):
    check = _layout(monkeypatch, dp=4, tp=2)
    op.prebuild(_config(), TOPK)
    op.prebuild(_config(), TOPK)
    check.assert_called_once()
    fresh.assert_called_once()


def test_fp8_dispatch_is_on_by_default(monkeypatch, fresh):
    _layout(monkeypatch, dp=4, tp=2)
    op.prebuild(_config(), TOPK)
    assert op._FP8_DISPATCH is True


def test_fp8_dispatch_option_off(monkeypatch, fresh):
    _layout(monkeypatch, dp=4, tp=2)
    op.prebuild(_config({"tpu_moe": {"lowtok_fused_fp8_dispatch": False}}), TOPK)
    assert op._ARMED is True
    assert op._FP8_DISPATCH is False


def test_a_malformed_option_fails_at_prebuild(monkeypatch, fresh):
    _layout(monkeypatch, dp=4, tp=2)
    with pytest.raises(ValueError, match="lowtok_fused_fp8_dispatch"):
        op.prebuild(_config({"tpu_moe": {"lowtok_fused_fp8_dispatch": 1}}), TOPK)


def test_forms_are_the_dp_tp_decode_values():
    assert (op.DISPATCH_FORM, op.COMBINE_FORM) == ("hier", "hier")
    assert op.RING_DEPTH == 1
    assert op.GATHER_ROWS is True


@pytest.fixture
def replicated(monkeypatch):
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setattr(sequence_parallel, "_ACTIVE", False)


def _layer(copies=False):
    return SimpleNamespace(_lowtok_has_copies=copies)


def test_serves_a_replicated_step_of_a_scores_layer(monkeypatch, replicated):
    monkeypatch.setattr(op, "_ARMED", True)
    assert op.serves(_layer(), object(), is_sequence_parallel=True) is True


@pytest.mark.parametrize(
    "armed,scores_op,copies,sp,active",
    [
        (False, object(), False, True, False),
        (True, None, False, True, False),
        (True, object(), True, True, False),
        (True, object(), False, False, False),
        (True, object(), False, True, True),
    ],
    ids=["unarmed", "no_scores_entry", "copy_slots", "not_sp_layer", "sliced_step"],
)
def test_does_not_serve(monkeypatch, armed, scores_op, copies, sp, active):
    from vllm_torchtpu.layers.adapter import sequence_parallel

    monkeypatch.setattr(op, "_ARMED", armed)
    monkeypatch.setattr(sequence_parallel, "_ACTIVE", active)
    assert op.serves(_layer(copies), scores_op, is_sequence_parallel=sp) is False


def test_retain_is_the_identity_when_unarmed(monkeypatch):
    monkeypatch.setattr(op, "_RANK", None)
    out = torch.randn(4, 8)
    assert op.retain(out) is out


def test_retain_reads_the_rank_buffer_in_the_warmup(monkeypatch):
    """A live read: a NaN rank buffer poisons the result."""
    monkeypatch.setattr(op.shape_variants, "in_warmup", lambda: True)
    monkeypatch.setattr(op, "_RANK", torch.tensor([float("nan"), 0.0]))
    assert torch.isnan(op.retain(torch.zeros(4, 8))).all()
    monkeypatch.setattr(op, "_RANK", torch.tensor([3, 1], dtype=torch.int32))
    out = torch.randn(4, 8)
    torch.testing.assert_close(op.retain(out), out, rtol=0, atol=0)


# The FP8 MoE method's fused EP v3 entry, on a step the fused v3 kernel does
# not take (use_fused_kernel=False).


@pytest.fixture
def v3_method(monkeypatch):
    import vllm.distributed.parallel_state as ps

    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU

    ep = Mock(name="ep_group")
    ep.dispatch_router_logits.return_value = (
        torch.zeros(4, 8, dtype=torch.bfloat16),
        torch.zeros(4, 16, dtype=torch.bfloat16),
    )
    ep.combine.return_value = torch.full((4, 8), 7.0, dtype=torch.bfloat16)
    monkeypatch.setattr(ps, "get_ep_group", lambda: ep)
    monkeypatch.setattr(small_all_reduce_op, "_RANK", None)
    method = MagicMock(spec=VllmFp8MoEMethodTPU)
    method._moe_lowtok_scores_op = object()
    method.apply_monolithic_external_dispatch.return_value = torch.zeros(4, 8)
    layer = SimpleNamespace(moe_config=SimpleNamespace(is_sequence_parallel=True))

    def call(x, logits):
        return VllmFp8MoEMethodTPU._apply_owning_dispatch_v3(
            method, layer, x, logits, "silu", use_fused_kernel=False
        )

    return call, ep, method, layer


def test_v3_entry_hands_a_served_step_to_the_fused_op(monkeypatch, v3_method):
    call, ep, method, layer = v3_method
    fused = torch.full((4, 8), 3.0, dtype=torch.bfloat16)
    serves = Mock(return_value=True)
    apply = Mock(return_value=fused)
    monkeypatch.setattr(op, "serves", serves)
    monkeypatch.setattr(op, "apply", apply)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    logits = torch.randn(4, 16, dtype=torch.bfloat16)
    out = call(x, logits)
    torch.testing.assert_close(out, fused, rtol=0, atol=0)
    serves.assert_called_once_with(layer, method._moe_lowtok_scores_op, True)
    apply.assert_called_once_with(layer, x, logits)
    ep.dispatch_router_logits.assert_not_called()
    ep.combine.assert_not_called()


def test_v3_entry_keeps_vllms_dispatch_otherwise(monkeypatch, v3_method):
    call, ep, method, _ = v3_method
    monkeypatch.setattr(op, "serves", Mock(return_value=False))
    apply = Mock()
    monkeypatch.setattr(op, "apply", apply)
    out = call(
        torch.randn(4, 8, dtype=torch.bfloat16),
        torch.randn(4, 16, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(out, ep.combine.return_value, rtol=0, atol=0)
    apply.assert_not_called()
    ep.dispatch_router_logits.assert_called_once()
    method.apply_monolithic_external_dispatch.assert_called_once()
