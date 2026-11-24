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

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import expert_banks
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.aggregate import KimiAggregate
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import (
    KernelConfig,
    resolve_decode_expert_banks,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.decode_ops import (
    DecodeOps,
    decode_weight_parts,
    pack_small_operands,
    unpack_small_operands,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import decode_megakernel
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_megakernel import (
    expert_family_slot,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.rank_loader import (
    PREFILL_SHARED,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.prefill import (
    PrefillProgram,
    _shared_expert_banks,
)

FAMILIES = expert_banks.EXPERT_FAMILIES
SIZES = expert_banks.SHARED_BANK_SIZES


def _banked_stream(boundaries):
    weights = {
        family: np.arange(92 * 6, dtype=np.int32).reshape(92, 2, 3) + i * 1000
        for i, family in enumerate(FAMILIES)
    }
    parts = {family: np.split(value, boundaries) for family, value in weights.items()}
    stream = [("decode", "router", np.arange(5))]
    for part in range(len(boundaries) + 1):
        for family in FAMILIES:
            stream.append(
                (
                    "decode",
                    f"{family}@{part}/{len(boundaries) + 1}",
                    parts[family][part],
                )
            )
    return weights, list(expert_banks.bank_expert_parts(iter(stream), SIZES))


@pytest.mark.parametrize("boundaries", [[], [42, 44], [43], [1, 50, 90]])
def test_should_select_every_layer_through_both_banks_when_parts_straddle_them(
    boundaries,
):
    weights, output = _banked_stream(boundaries)
    assert output[0][1] == "router"
    banks = {name: value for _, name, value in output[1:]}
    assert sorted(banks) == sorted(
        expert_banks.bank_name(f, b) for f in FAMILIES for b in range(len(SIZES))
    )
    for family in FAMILIES:
        assert [banks[expert_banks.bank_name(family, b)].shape[0] for b in (0, 1)] == [
            43,
            49,
        ]
        prefill, decode = [], []
        for slot in range(92):
            bank, _ = expert_banks.bank_of_slot(SIZES, slot)
            value = banks[expert_banks.bank_name(family, bank)]
            prefill.append(
                value[int(expert_family_slot(value.shape[0], jnp.int32(slot)))]
            )
            d_bank, d_index = decode_megakernel.expert_bank_slot(jnp.int32(slot))
            decode.append(
                banks[expert_banks.bank_name(family, int(d_bank))][int(d_index)]
            )
        np.testing.assert_array_equal(np.stack(prefill), weights[family])
        np.testing.assert_array_equal(np.stack(decode), weights[family])


def test_should_keep_the_global_and_per_group_slot_maps_when_shared_banks_exist():
    slots = jnp.arange(92)
    np.testing.assert_array_equal(expert_family_slot(92, slots), np.arange(92))
    np.testing.assert_array_equal(expert_family_slot(43, slots[:43]), np.arange(43))
    np.testing.assert_array_equal(
        expert_family_slot(8, jnp.arange(3, 11)), np.arange(8)
    )


def test_should_place_every_prefill_group_in_the_bank_holding_all_its_layers():
    groups = [expert_banks.group_bank(SIZES, g) for g in range(13)]
    assert groups == [0] * 6 + [1] * 7


def test_should_reject_bank_sizes_that_split_a_prefill_group():
    with pytest.raises(ValueError, match="straddles"):
        expert_banks.group_bank((46, 46), 6)
    with pytest.raises(ValueError, match="beyond"):
        expert_banks.bank_of_slot(SIZES, 92)


def test_should_pass_bank_operands_family_major_when_decode_weights_are_banked():
    weights = {"router": "r"}
    weights.update(
        {expert_banks.bank_name(f, b): f"{f}{b}" for f in FAMILIES for b in (0, 1)}
    )
    original = dict(weights)
    split, extra = decode_megakernel._split_expert_bank_operands(weights)  # noqa: SLF001 -- stack-internal operand order contract
    assert split == {"router": "r", **{f: f"{f}0" for f in FAMILIES}}
    assert extra == tuple(f"{f}1" for f in FAMILIES)
    assert weights == original
    unbanked = {"router": "r", **{f: f for f in FAMILIES}}
    same, none = decode_megakernel._split_expert_bank_operands(unbanked)  # noqa: SLF001 -- stack-internal operand order contract
    assert same is unbanked and none == ()


def test_should_hand_the_decoders_banks_to_the_prefill_when_both_run_ep16():
    weights = {name: object() for name in PREFILL_SHARED if name not in FAMILIES}
    weights.update(
        {expert_banks.bank_name(f, b): object() for f in FAMILIES for b in (0, 1)}
    )
    calls = []
    aggregate = object.__new__(KimiAggregate)
    aggregate.decoder = types.SimpleNamespace(
        load_weights=lambda: None, ops=types.SimpleNamespace(weights=weights)
    )
    aggregate.prefiller = types.SimpleNamespace(load_weights=calls.append)
    aggregate.rank, aggregate._loaded = 1, False  # noqa: SLF001 -- bypasses the device-building __init__
    aggregate._prefill_ep = aggregate._decode_ep = 16  # noqa: SLF001 -- bypasses the device-building __init__
    aggregate.load_weights()
    assert set(calls[0]) == set(weights)
    assert not set(calls[0]) & set(FAMILIES)


def test_should_pop_exactly_the_shared_banks_out_of_the_prefills_shared_tensors():
    shared = {"router": 1}
    shared.update(
        {expert_banks.bank_name(f, b): (f, b) for f in FAMILIES for b in (0, 1)}
    )
    banks = _shared_expert_banks(shared)
    assert banks == [{f: (f, b) for f in FAMILIES} for b in (0, 1)]
    assert shared == {"router": 1}
    assert _shared_expert_banks({"expert_gu": 0}) == []


def test_should_require_decode_ep16_for_decode_expert_banks():
    with pytest.raises(ValueError, match="decode_expert_parallel 16"):
        KernelConfig(decode_expert_banks=True, decode_expert_parallel=8)
    assert KernelConfig(
        decode_expert_banks=True, decode_expert_parallel=16
    ).decode_expert_banks


@pytest.mark.parametrize(
    ("prefill_ep", "decode_ep", "enabled", "expected"),
    [
        (16, 16, True, True),
        (16, 16, False, False),
        (16, 8, True, False),
        (8, 8, True, False),
    ],
)
def test_should_bank_decode_experts_only_when_both_sides_run_ep16_and_enabled(
    prefill_ep, decode_ep, enabled, expected
):
    assert (
        resolve_decode_expert_banks(
            prefill_ep=prefill_ep, decode_ep=decode_ep, enabled=enabled
        )
        is expected
    )


@pytest.mark.parametrize("banked", [False, True])
def test_should_stream_decode_experts_as_shared_banks_only_when_banked(banked):
    family_parts = [
        ("decode", f"{f}@0/1", np.zeros((92, 1), np.int8)) for f in FAMILIES
    ]
    loader = types.SimpleNamespace(
        iter_decode_rank=lambda presharded, rank, split_experts, attention_dp: iter(
            [("decode", "router", np.zeros(1))] + family_parts
        )
    )
    names = [
        n
        for _, n, _ in decode_weight_parts(
            loader, "p", 3, attention_dp=1, banked=banked
        )
    ]
    if banked:
        assert sorted(names) == sorted(
            ["router"]
            + [expert_banks.bank_name(f, b) for f in FAMILIES for b in (0, 1)]
        )
    else:
        assert names == ["router"] + [f"{f}@0/1" for f in FAMILIES]


def _program_with_banks(count):
    program = object.__new__(PrefillProgram)
    program._expert_banks = [{"expert_gu": f"bank{b}"} for b in range(count)]  # noqa: SLF001 -- bypasses the device-building __init__
    program.names = {"moe": ("router", "expert_gu")}
    program.weights = {"moe": {"router": "r", "expert_gu": "global"}}
    return program


@pytest.mark.parametrize(
    ("count", "group_ops", "first", "regular_p4", "regular_p5", "final"),
    [
        (2, (1, 11), "bank0", "bank0", "bank1", "bank1"),
        (13, (1,), "bank0", "bank5", "bank6", "bank12"),
    ],
)
def test_should_hand_each_execution_group_the_resident_bank_holding_it(
    count, group_ops, first, regular_p4, regular_p5, final
):
    program = _program_with_banks(count)
    assert program.group_op_banks() == group_ops
    selected = [program._flat("moe", expert_bank=g)[1] for g in (0, 5, 6, 12)]  # noqa: SLF001 -- the window ops' operand list contract
    assert selected == [first, regular_p4, regular_p5, final]
    assert _program_with_banks(0)._flat("moe", expert_bank=6) == ["r", "global"]  # noqa: SLF001 -- the window ops' operand list contract


def _decode_names(banked: bool, packed: bool) -> tuple[str, ...]:
    dense = set(decode_megakernel.NAMES + decode_megakernel.EPILOGUE)
    dense |= set(decode_megakernel.DENSE) | {"embedding", "lm_head"}
    if banked:
        dense = (dense - set(FAMILIES)) | {
            expert_banks.bank_name(f, b) for f in FAMILIES for b in (0, 1)
        }
    weights = {n: torch.zeros((1, 2, 3), dtype=torch.float32) for n in dense}
    if packed:
        weights, _ = pack_small_operands(weights)
    return tuple(sorted(weights))


def _ops_with(names: tuple[str, ...], pools: int) -> DecodeOps:
    ops = object.__new__(DecodeOps)
    ops._ops, ops.batch_sizes, ops.names = {}, (8,), names  # noqa: SLF001 -- bypasses the device-loading __init__
    ops.paged = types.SimpleNamespace()
    ops.unified_pools = [None] * pools
    ops.pool_map, ops.owner_table = (0,) * 24, tuple(range(32))
    ops._packed_layout = {}  # noqa: SLF001 -- bypasses the device-loading __init__
    ops.attention_dp = 1
    ops.mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))
    return ops


CONTROL = types.SimpleNamespace(key=lambda: "bank-control")


def test_should_register_the_plain_step_within_64_operands_when_banked_and_packed():
    names = _decode_names(banked=True, packed=True)
    assert len(names) + 2 + 24 + 1 <= 64
    _ops_with(names, pools=24)._build_plain(8, CONTROL, 0, False, False)  # noqa: SLF001 -- the op registration is the contract


def test_should_exceed_torch_operand_limit_when_banked_without_packing():
    names = _decode_names(banked=True, packed=False)
    assert len(names) + 2 + 24 + 1 == 68
    with pytest.raises(RuntimeError, match="64"):
        _ops_with(names, pools=24)._build_plain(8, CONTROL, 0, False, False)  # noqa: SLF001 -- the op registration is the contract


def test_should_restore_every_packed_family_bit_exactly():
    weights = {
        "attn_res": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "router_bias": torch.arange(4, dtype=torch.float32).reshape(1, 4) + 100,
        "attn_norm": torch.arange(5, dtype=torch.bfloat16).reshape(1, 5),
        "expert_gu#bank0": torch.ones((1, 2), dtype=torch.int32),
    }
    operands, layout = pack_small_operands(weights)
    assert sorted(operands) == [
        "expert_gu#bank0",
        "packed#torch.bfloat16",
        "packed#torch.float32",
    ]
    local = {n: jnp.asarray(t[0].float().numpy()) for n, t in operands.items()}
    restored = unpack_small_operands(local, layout)
    for name in ("attn_res", "router_bias", "attn_norm"):
        np.testing.assert_array_equal(
            np.asarray(restored[name]), weights[name][0].float().numpy()
        )
    assert "packed#torch.float32" not in restored
