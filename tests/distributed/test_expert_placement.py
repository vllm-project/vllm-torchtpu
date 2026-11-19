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
"""Static expert placement: LPT tables, their validation, and the checkpoint
relabelling (`distributed/expert_placement.py`)."""

import json
import re

import numpy as np
import pytest
import torch

from vllm_torchtpu.distributed import expert_placement as ep_place

E, EP, TP = 32, 8, 2
LAYERS = [3, 4]


def _skewed_rows(seed):
    rng = np.random.default_rng(seed)
    rows = rng.integers(10, 50, E)
    rows[rng.integers(E)] = 2000  # one hot expert
    return rows


def _group_load(rows, slot):
    load = np.zeros(EP // TP, np.int64)
    np.add.at(load, slot // (E // EP * TP), rows)
    return load


def _table(slots, **over):
    t = dict(
        format=ep_place.TABLE_FORMAT,
        num_experts=E,
        ep_size=EP,
        ep_tp_size=TP,
        layers={str(k): np.asarray(v).tolist() for k, v in slots.items()},
    )
    t.update(over)
    return t


def _load(table):
    return ep_place.ExpertPlacement.from_table(
        table, num_experts=E, moe_layers=LAYERS, ep_size=EP, ep_tp_size=TP
    )


def _checkpoint(seed):
    rng = np.random.default_rng(seed)
    out = []
    for layer in LAYERS:
        p = f"model.layers.{layer}.mlp"
        out.append((f"{p}.gate.weight", torch.randn(E, 16, dtype=torch.bfloat16)))
        out.append((f"{p}.gate.e_score_correction_bias", torch.randn(E)))
        out.append((f"{p}.shared_experts.gate_proj.weight", torch.randn(8, 16)))
        for e in range(E):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("weight", "weight_scale_inv"):
                    out.append((f"{p}.experts.{e}.{proj}.{suffix}", torch.randn(2, 2)))
    out.append(("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 16)))
    out.append(("model.norm.weight", torch.from_numpy(rng.standard_normal(16))))
    return out


def test_lpt_balances_the_groups_and_is_a_permutation():
    rows = _skewed_rows(0)
    slot = ep_place.lpt_slots(rows, EP, TP)
    assert sorted(slot) == list(range(E))
    before = _group_load(rows, np.arange(E))
    after = _group_load(rows, slot)
    assert after.max() <= before.max()
    # The hot expert's group holds nothing else it can shed.
    hot = int(np.argmax(rows))
    assert after.max() == _group_load(rows, slot)[slot[hot] // (E // EP * TP)]
    balanced = ep_place.lpt_slots(np.full(E, 7), EP, TP)
    assert _group_load(np.full(E, 7), balanced).tolist() == [56] * (EP // TP)


def test_identity_table_relabels_nothing():
    ckpt = _checkpoint(1)
    ident = _load(_table({k: np.arange(E) for k in LAYERS}))
    out = list(ident.relabel(iter(ckpt)))
    assert [n for n, _ in out] == [n for n, _ in ckpt]
    for (_, a), (_, b) in zip(ckpt, out):
        assert a.dtype == b.dtype and torch.equal(a, b)
    for name, _ in ckpt:
        m = re.search(r"\.mlp\.experts\.(\d+)\.", name)
        assert ident.physical_expert_id(name) == (m and int(m.group(1)))


def test_permutation_moves_experts_and_keeps_the_router():
    rng = np.random.default_rng(2)
    slots = {k: rng.permutation(E) for k in LAYERS}
    place = _load(_table(slots))
    ckpt = dict(_checkpoint(3))
    out = dict(place.relabel(iter(ckpt.items())))
    assert len(out) == len(ckpt)
    for layer, s in slots.items():
        p = f"model.layers.{layer}.mlp"
        # Routing stays logical: the ids move after the top-k.
        for name in (f"{p}.gate.weight", f"{p}.gate.e_score_correction_bias"):
            assert out[name] is ckpt[name]
        for e in range(E):
            src = f"{p}.experts.{e}.down_proj.weight_scale_inv"
            dst = f"{p}.experts.{s[e]}.down_proj.weight_scale_inv"
            assert torch.equal(out[dst], ckpt[src])
            assert place.physical_expert_id(src) == s[e]
    shared = "model.layers.3.mlp.shared_experts.gate_proj.weight"
    assert out[shared] is ckpt[shared]
    assert (
        out["model.layers.0.mlp.gate_proj.weight"]
        is ckpt["model.layers.0.mlp.gate_proj.weight"]
    )


@pytest.mark.parametrize(
    "table, match",
    [
        (_table({3: np.arange(E), 4: np.arange(E)}, format="x"), "is for"),
        (_table({3: np.arange(E), 4: np.arange(E)}, num_experts=16), "is for"),
        (_table({3: np.arange(E), 4: np.arange(E)}, ep_size=16), "is for"),
        (_table({3: np.arange(E), 4: np.arange(E)}, ep_tp_size=1), "is for"),
        (_table({3: np.arange(E)}), "MoE layers"),
        (_table({3: np.arange(E), 4: np.arange(E), 5: np.arange(E)}), "MoE layers"),
        (_table({3: np.arange(E), 4: np.r_[np.arange(E - 1), 0]}), "permutation"),
        (_table({3: np.arange(E), 4: np.arange(E - 1)}), "permutation"),
        (_table({3: np.arange(E), 4: np.arange(E) + 0.5}), "permutation"),
        (_table({3: np.arange(E)}, layers={"3": list(range(E)), "x": []}), "maps"),
    ],
)
def test_malformed_or_mismatched_tables_raise(table, match):
    with pytest.raises(ValueError, match=match):
        _load(table)


def test_table_tool_writes_lpt_of_the_dump(tmp_path):
    stats = dict(
        format=ep_place.STATS_FORMAT,
        num_experts=E,
        ep_size=EP,
        ep_tp_size=TP,
        layers={str(k): _skewed_rows(k).tolist() for k in LAYERS},
    )
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    ep_place.main([str(tmp_path / "stats.json"), str(tmp_path / "table.json")])
    place = _load(json.loads((tmp_path / "table.json").read_text()))
    for k in LAYERS:
        assert (
            place.slots[k].tolist()
            == ep_place.lpt_slots(_skewed_rows(k), EP, TP).tolist()
        )
    with pytest.raises(ValueError, match="dump"):
        ep_place.table_from_stats(dict(stats, format="x"))


def test_physical_ids_map_selected_experts_through_the_slots():
    slots = np.random.default_rng(4).permutation(E)
    layer = torch.nn.Module()
    ids = torch.randint(0, E, (5, 3), dtype=torch.int32)
    assert ep_place.physical_ids(layer, ids) is ids
    layer.register_buffer(ep_place.SLOTS_ATTR, torch.tensor(slots, dtype=torch.int32))
    got = ep_place.physical_ids(layer, ids)
    assert got.dtype == ids.dtype and torch.equal(
        got, torch.tensor(slots)[ids.long()].int()
    )


def _check_annotated(name, fn, **_kw):
    """torch_tpu's jax_op reads every operand's type from fn's annotations."""
    import functools
    import inspect
    from unittest.mock import MagicMock

    assert not isinstance(fn, functools.partial), name
    params = inspect.signature(fn).parameters.values()
    missing = [p.name for p in params if p.annotation is inspect.Parameter.empty]
    assert not missing, (name, missing)
    return MagicMock()


def test_ops_handed_to_jax_op_are_fully_annotated(monkeypatch):
    import jax
    import vllm.envs
    from jax.sharding import Mesh

    # moe_routing resolves its simulation strategy at import.
    monkeypatch.setattr(vllm.envs, "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "", False)
    from vllm_torchtpu.layers.adapter import fused_moe_ep, moe_prefill_sp, moe_tp_decode

    monkeypatch.setattr(moe_prefill_sp, "jax_op", _check_annotated)
    monkeypatch.setattr(fused_moe_ep, "jax_op", _check_annotated)
    monkeypatch.setattr(moe_tp_decode.pallas, "jax_op", _check_annotated)
    monkeypatch.setattr(moe_prefill_sp, "_LOAD_OPS", {})
    monkeypatch.setattr(fused_moe_ep, "_OPS", {})
    monkeypatch.setattr(moe_tp_decode, "_OPS", {})
    moe_prefill_sp._build_load_op(8)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("d",))
    fused_moe_ep._build_op(
        mesh, 8, True, "silu", None, True, has_score_bias=True, has_expert_slots=True
    )
    statics = moe_tp_decode._OpStatics(
        topk=8,
        routed_scale=1.0,
        activation="silu",
        use_sparse_core=False,
        onehot_threshold=0,
        use_ep=True,
        placed=True,
    )
    moe_tp_decode._get_op(mesh, statics)
