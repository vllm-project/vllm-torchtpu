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
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.kernels.experimental.small_collectives import common as sc_common
from vllm_torchtpu.layers.adapter import moe_lowtok_fused_op as op
from vllm_torchtpu.layers.adapter import small_all_reduce_op

pytestmark = pytest.mark.cpu_test

TOPK = 10


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
    ):
        monkeypatch.setattr(op, name, value)
    build = Mock(name="_build_op")
    monkeypatch.setattr(op, "_build_op", build)
    host_tensor = torch.tensor

    def tensor(*args, device=None, **kwargs):
        return host_tensor(*args, **kwargs)

    monkeypatch.setattr(op.torch, "tensor", tensor)
    return build


def _layout(
    monkeypatch,
    *,
    dp,
    tp,
    cores=8,
    mesh=True,
    groups_match=True,
    index=5,
    chip_devices=2,
):
    """The model-parallel groups, EP mesh and devices per chip prebuild reads
    (TPU7x: two); the groups check (a collective over the EP group) as a
    mock."""
    import vllm.distributed.parallel_state as ps

    monkeypatch.setattr(sc_common, "devices_per_chip", lambda: chip_devices)

    monkeypatch.setattr(
        op,
        "build_ep_mesh",
        lambda: SimpleNamespace(shape={EP_AXIS_NAME: cores}) if mesh else None,
    )
    monkeypatch.setattr(ps, "get_tp_group", lambda: SimpleNamespace(world_size=tp))
    monkeypatch.setattr(ps, "get_dp_group", lambda: SimpleNamespace(world_size=dp))
    monkeypatch.setattr(op, "ep_mesh_index", lambda: index)
    check = Mock(name="groups_match_mesh", return_value=groups_match)
    monkeypatch.setattr(small_all_reduce_op, "groups_match_mesh", check)
    return check


def test_dp4_tp2_arms_the_op(monkeypatch, fresh):
    check = _layout(monkeypatch, dp=4, tp=2, index=5)
    op.prebuild(TOPK)
    assert op._ARMED is True
    assert (op._DP, op._TP) == (4, 2)
    # Mesh index 5 is DP rank 2, TP rank 1 of the (dp, tp) view.
    assert op._RANK.tolist() == [2, 1]
    check.assert_called_once_with(5, 2, 4)
    fresh.assert_called_once_with(TOPK)


@pytest.mark.parametrize(
    "layout",
    [
        dict(dp=1, tp=8),
        dict(dp=8, tp=1),
        dict(dp=2, tp=4),
        dict(dp=4, tp=2, cores=16),
        dict(dp=1, tp=2, cores=2),
        dict(dp=4, tp=2, chip_devices=1),
    ],
    ids=[
        "tp8",
        "dp8_tp1",
        "dp2_tp4",
        "not_the_mesh",
        "one_engine_tp2",
        "tp2_on_two_chips",
    ],
)
def test_other_layouts_leave_it_unarmed_without_the_groups_collective(
    monkeypatch, fresh, layout
):
    """TP8 (the other measured decode layout), TP other than a chip's pair, a DP x TP
    that is not the EP mesh, one engine on a chip's pair (nothing to
    dispatch over DP), or a TP pair across two single-device chips: the
    default MoE path, and no collective."""
    check = _layout(monkeypatch, **layout)
    op.prebuild(TOPK)
    assert op._ARMED is False
    assert op._RANK is None
    check.assert_not_called()
    fresh.assert_not_called()


@pytest.mark.parametrize(
    "chip,cores_per_device,armed",
    [
        (pltpu.ChipVersion.TPU_7X, 1, True),
        (pltpu.ChipVersion.TPU_V5P, 2, False),
    ],
    ids=["tpu7x", "v5p_megacore"],
)
def test_the_device_info_decides_and_a_disarmed_tp_pair_is_logged(
    monkeypatch, fresh, chip, cores_per_device, armed
):
    """The DP4 x TP2 layout arms on TPU7x's device info (two devices per chip)
    without a warning; where the device info reports one device per chip the
    op stays off and says so once, with the reported value."""
    devices_per_chip = sc_common.devices_per_chip
    _layout(monkeypatch, dp=4, tp=2)
    info = pltpu.get_tpu_info_for_chip(chip, cores_per_device)
    monkeypatch.setattr(sc_common, "devices_per_chip", devices_per_chip)
    monkeypatch.setattr(sc_common.pltpu, "get_tpu_info", lambda: info)
    logger = Mock(name="logger")
    monkeypatch.setattr(small_all_reduce_op, "logger", logger)
    op.prebuild(TOPK)
    assert op._ARMED is armed
    if armed:
        logger.warning_once.assert_not_called()
    else:
        logger.warning_once.assert_called_once()
        _, what, reported, tp, _ = logger.warning_once.call_args.args
        assert (what, reported, tp) == ("The moe_lowtok fused collectives", 1, 2)


def test_tp_groups_off_the_mesh_blocks_leave_it_unarmed(monkeypatch, fresh):
    _layout(monkeypatch, dp=4, tp=2, groups_match=False)
    op.prebuild(TOPK)
    assert op._ARMED is False
    fresh.assert_not_called()


def test_no_expert_parallel_mesh_leaves_it_unarmed(monkeypatch, fresh):
    _layout(monkeypatch, dp=4, tp=2, mesh=False)
    op.prebuild(TOPK)
    assert op._ARMED is False
    fresh.assert_not_called()


def test_prebuild_checks_the_layout_once(monkeypatch, fresh):
    check = _layout(monkeypatch, dp=4, tp=2)
    op.prebuild(TOPK)
    op.prebuild(TOPK)
    check.assert_called_once()
    fresh.assert_called_once()


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
    """Unarmed, even in the warmup and with the small collectives' mesh rank
    buffer set: that buffer is not this op's."""
    from vllm_torchtpu.compilation import shape_variants

    monkeypatch.setattr(shape_variants, "in_warmup", lambda: True)
    monkeypatch.setattr(op, "_RANK", None)
    monkeypatch.setattr(small_all_reduce_op, "_RANK", torch.tensor([float("nan")]))
    out = torch.randn(4, 8)
    assert op.retain(out) is out


def test_retain_reads_the_rank_buffer_in_the_warmup(monkeypatch):
    """A live read: a NaN rank buffer poisons the result."""
    from vllm_torchtpu.compilation import shape_variants

    monkeypatch.setattr(shape_variants, "in_warmup", lambda: True)
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


def _moe_layer(use_ep=True, topk=TOPK):
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            experts_per_token=topk,
            moe_parallel_config=SimpleNamespace(use_ep=use_ep),
        )
    )


def test_maybe_prebuild_arms_for_a_scores_layer_under_ep(monkeypatch):
    """The FP8 MoE method's weight processing calls it for every MoE layer."""
    prebuild = Mock(name="prebuild")
    monkeypatch.setattr(op, "prebuild", prebuild)
    op.maybe_prebuild(_moe_layer(), object())
    prebuild.assert_called_once_with(TOPK)


@pytest.mark.parametrize(
    "use_ep,scores_op",
    [(False, object()), (True, None)],
    ids=["no_expert_parallelism", "no_scores_entry"],
)
def test_maybe_prebuild_leaves_other_layers(monkeypatch, use_ep, scores_op):
    prebuild = Mock(name="prebuild")
    monkeypatch.setattr(op, "prebuild", prebuild)
    op.maybe_prebuild(_moe_layer(use_ep=use_ep), scores_op)
    prebuild.assert_not_called()


class _Armed(Exception):
    """Stops the FP8 MoE method's weight processing at the arming call."""


def test_fp8_moe_weight_processing_offers_the_layer_to_the_op(monkeypatch):
    """VllmFp8MoEMethodTPU.process_weights_after_loading hands the layer and
    its scores entry to maybe_prebuild once the scores entry is built."""
    from vllm_torchtpu.layers.adapter.quantization import fp8

    experts, hidden, inter = 1, 512, 256
    layer = torch.nn.Module()
    layer.activation = "silu"
    layer.moe_config = SimpleNamespace(
        experts_per_token=TOPK, moe_parallel_config=SimpleNamespace(use_ep=False)
    )
    for name, shape in (
        ("w13_weight", (experts, hidden, 2 * inter)),
        ("w2_weight", (experts, inter, hidden)),
    ):
        weight = torch.empty(*shape, dtype=torch.float8_e4m3fn)
        layer.register_parameter(name, torch.nn.Parameter(weight, requires_grad=False))
    method = object.__new__(fp8.VllmFp8MoEMethodTPU)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
    method.weight_block_size = None
    method.weight_scale_name = "weight_scale"

    def converted(layer, **_kwargs):
        return (
            layer.w13_weight,
            torch.ones(experts, 1, 1, 2 * inter),
            layer.w2_weight,
            torch.ones(experts, 1, 1, hidden),
            "float8_e4m3fn",
            None,
        )

    scores_op = object()
    arm = Mock(name="maybe_prebuild", side_effect=_Armed)
    monkeypatch.setattr(fp8, "RoutedExperts", torch.nn.Module)
    monkeypatch.setattr(fp8, "_process_fp8_moe_weights", converted)
    monkeypatch.setattr(
        fp8.moe_routing, "register_experts_start_buffer", lambda *a, **kw: None
    )
    monkeypatch.setattr(fp8.moe_lowtok_scores, "prebuild", lambda *a: scores_op)
    monkeypatch.setattr(op, "maybe_prebuild", arm)
    with pytest.raises(_Armed):
        method.process_weights_after_loading(layer)
    arm.assert_called_once_with(layer, scores_op)
    assert method._moe_lowtok_scores_op is scores_op
