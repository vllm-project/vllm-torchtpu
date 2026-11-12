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


class _FakeMesh:
    """Hashable EP mesh stand-in keyed on `EP_AXIS_NAME` for CPU adapter tests."""

    def __init__(self, axis_name: str, ep_size: int) -> None:
        self.axis_names = (axis_name,)
        self.shape = {axis_name: ep_size}


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
        moe_parallel_config=SimpleNamespace(
            use_ep=True,
            pcp_size=1,
            is_sequence_parallel=False,
        ),
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
    monkeypatch.delenv("TPU_MOE_LOWTOK_GATHER_LOGITS", raising=False)


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


def test_gather_logits_env_and_platform_registration(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    monkeypatch.delenv("TPU_MOE_LOWTOK_GATHER_LOGITS", raising=False)
    assert envs.TPU_MOE_LOWTOK_GATHER_LOGITS is False
    monkeypatch.setenv("TPU_MOE_LOWTOK_GATHER_LOGITS", "1")
    assert envs.TPU_MOE_LOWTOK_GATHER_LOGITS is True
    monkeypatch.setenv("TPU_MOE_LOWTOK_GATHER_LOGITS", "invalid")
    with pytest.raises(ValueError, match="TPU_MOE_LOWTOK_GATHER_LOGITS"):
        _ = envs.TPU_MOE_LOWTOK_GATHER_LOGITS
    for var_name in (
        "TPU_EXPERT_PLACEMENT",
        "TPU_MOE_LOWTOK_MAX_TOKENS",
        "TPU_MOE_LOWTOK_FROM_SCORES",
        "TPU_MOE_LOWTOK_SCORES_STRICT",
        "TPU_MOE_LOWTOK_GATHER_LOGITS",
    ):
        assert var_name in TpuPlatform.additional_env_vars


def test_adaptive_build_op_keys_and_forwards_gather_logits(monkeypatch):
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import vllm_adapter

    monkeypatch.setattr(vllm_adapter, "_OPS", {})
    captured = []
    warnings = []

    def fake_jax_op(name, fn, *, mesh, input_partition_specs, output_partition_specs):
        op = SimpleNamespace(
            name=name,
            fn=fn,
            register_fake=lambda f: None,
        )
        captured.append(op)
        return op

    monkeypatch.setattr(vllm_adapter, "jax_op", fake_jax_op)
    monkeypatch.setattr(
        vllm_adapter.logger,
        "warning_once",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )
    forwarded = []
    monkeypatch.setattr(
        vllm_adapter,
        "moe_lowtok_ep_serves",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        vllm_adapter,
        "moe_lowtok_scores_ep",
        lambda *args, **kwargs: forwarded.append(kwargs["gather_logits"]) or args[0],
    )
    fake_mesh = _FakeMesh(vllm_adapter.EP_AXIS_NAME, EP)

    # When lowtok_scores=False, lowtok_gather_logits must not split `_OPS`.
    op_plain_off = vllm_adapter._build_op(
        fake_mesh,
        topk=4,
        renormalize=True,
        activation="silu",
        mesh_expert_order=None,
        lowtok_max_tokens=64,
        lowtok_scores=False,
        lowtok_scores_copies=False,
        lowtok_gather_logits=False,
    )
    op_plain_on = vllm_adapter._build_op(
        fake_mesh,
        topk=4,
        renormalize=True,
        activation="silu",
        mesh_expert_order=None,
        lowtok_max_tokens=64,
        lowtok_scores=False,
        lowtok_scores_copies=False,
        lowtok_gather_logits=True,
    )
    assert op_plain_off is op_plain_on
    assert len(captured) == 1

    op_off = vllm_adapter._build_op(
        fake_mesh,
        topk=4,
        renormalize=True,
        activation="silu",
        mesh_expert_order=None,
        lowtok_max_tokens=64,
        lowtok_scores=True,
        lowtok_scores_copies=True,
        lowtok_gather_logits=False,
    )
    op_on = vllm_adapter._build_op(
        fake_mesh,
        topk=4,
        renormalize=True,
        activation="silu",
        mesh_expert_order=None,
        lowtok_max_tokens=64,
        lowtok_scores=True,
        lowtok_scores_copies=True,
        lowtok_gather_logits=True,
    )
    assert op_off is not op_on
    assert len(captured) == 3

    bf16_gating = SimpleNamespace(shape=(16, scores_entry.LANES), dtype=jnp.bfloat16)
    f32_gating = SimpleNamespace(shape=(16, scores_entry.LANES), dtype=jnp.float32)
    dummy = SimpleNamespace(shape=(16, HIDDEN), dtype=jnp.bfloat16)
    captured[1].fn(dummy, *((dummy,) * 4), bf16_gating, *((dummy,) * 10))
    captured[2].fn(dummy, *((dummy,) * 4), bf16_gating, *((dummy,) * 10))
    captured[2].fn(dummy, *((dummy,) * 4), f32_gating, *((dummy,) * 10))
    assert forwarded == [False, True, False]
    assert any("falling back to float32 score all-gather" in w for w in warnings)


def test_prebuild_adaptive_fused_moe_reads_gather_logits_env(scores_env, monkeypatch):
    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import vllm_adapter

    fake_mesh = _FakeMesh(vllm_adapter.EP_AXIS_NAME, EP)
    monkeypatch.setattr(vllm_adapter, "build_ep_mesh", lambda: fake_mesh)
    monkeypatch.setattr(vllm_adapter, "_max_node_tokens", lambda ep, pcp: 2048)
    monkeypatch.setattr(
        vllm_adapter,
        "fused_moe_ep_unsupported_reason",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        vllm_adapter,
        "_device_info",
        lambda: SimpleNamespace(smem_capacity_bytes=1 << 20),
    )
    monkeypatch.setattr(vllm_adapter.host, "chip_generation", lambda info: 7)
    monkeypatch.setattr(
        vllm_adapter.host,
        "select_weight_buffers",
        lambda *args, **kwargs: SimpleNamespace(
            w1_nbuf=2, w2_nbuf=2, vmem_bytes=1 << 20
        ),
    )
    monkeypatch.setattr(vllm_adapter, "_mesh_expert_order", lambda ep: None)
    monkeypatch.setattr(vllm_adapter, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(vllm_adapter, "ep_token_replica_groups", lambda **kwargs: None)
    monkeypatch.setattr(vllm_adapter, "_RANK_BUFFER", None)

    recorded_flags = []
    warnings = []
    monkeypatch.setattr(
        vllm_adapter.logger,
        "warning_once",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )

    def fake_build_op(*args, **kwargs):
        recorded_flags.append((kwargs["lowtok_scores"], kwargs["lowtok_gather_logits"]))
        return SimpleNamespace(name="fake_op")

    monkeypatch.setattr(vllm_adapter, "_build_op", fake_build_op)

    layer = _layer()
    monkeypatch.setenv("TPU_MOE_LOWTOK_GATHER_LOGITS", "0")
    vllm_adapter.prebuild_adaptive_fused_moe(
        layer,
        topk=4,
        renormalize=True,
        activation="silu",
        lowtok_scores=True,
    )
    monkeypatch.setenv("TPU_MOE_LOWTOK_GATHER_LOGITS", "1")
    vllm_adapter.prebuild_adaptive_fused_moe(
        layer,
        topk=4,
        renormalize=True,
        activation="silu",
        lowtok_scores=True,
    )
    vllm_adapter.prebuild_adaptive_fused_moe(
        layer,
        topk=4,
        renormalize=True,
        activation="silu",
        lowtok_scores=False,
    )
    assert recorded_flags == [(True, False), (True, True), (False, True)]
    assert any("TPU_MOE_LOWTOK_FROM_SCORES is not armed" in w for w in warnings)
