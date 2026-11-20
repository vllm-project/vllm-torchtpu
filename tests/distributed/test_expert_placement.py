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
"""Static expert placement: LPT tables and the table tool
(`distributed/expert_placement.py`)."""

import json

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
    assert json.loads((tmp_path / "table.json").read_text()) == dict(
        format=ep_place.TABLE_FORMAT,
        num_experts=E,
        ep_size=EP,
        ep_tp_size=TP,
        layers={
            str(k): ep_place.lpt_slots(_skewed_rows(k), EP, TP).tolist() for k in LAYERS
        },
    )
    with pytest.raises(ValueError, match="dump"):
        ep_place.table_from_stats(dict(stats, format="x"))


_HANDED: dict = {}


def _check_annotated(name, fn, **kw):
    """torch_tpu's jax_op reads every operand's type from fn's annotations;
    no op on these paths donates or aliases an operand."""
    import functools
    import inspect
    from unittest.mock import MagicMock

    assert not isinstance(fn, functools.partial), name
    params = inspect.signature(fn).parameters.values()
    missing = [p.name for p in params if p.annotation is inspect.Parameter.empty]
    assert not missing, (name, missing)
    assert not kw.get("donate_argnums"), (name, kw)
    _HANDED[name] = fn
    return MagicMock()


def _adapters(monkeypatch):
    import vllm.envs

    # moe_routing resolves its simulation strategy at import.
    monkeypatch.setattr(vllm.envs, "VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "", False)
    from vllm_torchtpu.layers.adapter import fused_moe_ep, moe_prefill_sp, moe_tp_decode

    return fused_moe_ep, moe_prefill_sp, moe_tp_decode


def test_ops_handed_to_jax_op_are_annotated_and_alias_nothing(monkeypatch):
    import jax
    import jax.numpy as jnp

    _, moe_prefill_sp, _ = _adapters(monkeypatch)
    monkeypatch.setattr(moe_prefill_sp, "jax_op", _check_annotated)
    monkeypatch.setattr(moe_prefill_sp, "_LOAD_OPS", {})
    _HANDED.clear()
    moe_prefill_sp._build_load_op(8)
    assert len(_HANDED) == 1, sorted(_HANDED)
    # The counter's program: no donated operand, no input/output alias.
    (count,) = (f for n, f in _HANDED.items() if "load_counts" in n)
    avals = (
        jax.ShapeDtypeStruct((E,), jnp.int32),
        jax.ShapeDtypeStruct((64, E), jnp.float32),
        jax.ShapeDtypeStruct((E,), jnp.float32),
        jax.ShapeDtypeStruct((64,), jnp.bool_),
    )
    lowered = jax.jit(count).lower(*avals)
    assert "aliasing_output" not in lowered.as_text()
    # A compare-and-sum, not a scatter (which the TPU offloads to the
    # SparseCore).
    assert "scatter" not in lowered.as_text()
    assert "buffer_donor" not in lowered.as_text()
    assert "input_output_alias" not in lowered.compile().as_text()


def test_load_counts_accumulate_into_the_buffer(monkeypatch):
    # The op is functional (a new tensor, nothing donated); the buffer takes
    # its value with copy_, so the update stays inside each traced graph.
    _, moe_prefill_sp, _ = _adapters(monkeypatch)
    mlp = torch.nn.Module()
    mlp.register_buffer(moe_prefill_sp.LOAD_ATTR, torch.arange(E, dtype=torch.int32))
    buf = getattr(mlp, moe_prefill_sp.LOAD_ATTR)
    returned = []

    def op(counts, *_args):
        returned.append(counts + 1)
        return returned[-1]

    setattr(mlp, moe_prefill_sp.LOAD_OP, op)
    moe_prefill_sp.accumulate_load(mlp, None, None, None)
    assert getattr(mlp, moe_prefill_sp.LOAD_ATTR) is buf
    assert returned[0] is not buf
    assert torch.equal(buf, torch.arange(E, dtype=torch.int32) + 1)
