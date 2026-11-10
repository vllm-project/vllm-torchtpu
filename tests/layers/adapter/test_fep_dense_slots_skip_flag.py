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
"""Config.dense_slots_skip_unrouted / dense_slots_weight_slots (default off / 3 slots): the
TPU_MOE_FUSED_EP_V3_CONFIG JSON reaches the dense-slots call, and dense_slots_moe hands both
settings to the program builder (CPU, no kernel is built)."""
from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.kernels.fused_moe.v3 import dense_slots as DS
from vllm_torchtpu.kernels.fused_moe.v3.layout import MESH_AXIS
from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as A

SHAPE = dict(hidden=4096, inter=1024, experts_per_shard=64)
EP = 8
DENSE = '"dense_slots":true,"dense_slots_fused":true,"dense_slots_merged":true'


class _Mesh:
    axis_names = (MESH_AXIS,)
    shape = {MESH_AXIS: EP}


def _adapter_config(monkeypatch, json_fields):
    for k in ("TPU_MOE_FEP_DENSE_SLOTS", "TPU_MOE_FEP_DENSE_SLOTS_MAX_TOKENS",
              "TPU_MOE_FEP_DENSE_SLOTS_FUSED", "TPU_MOE_FEP_DENSE_SLOTS_PALLAS_TABLES",
              "TPU_MOE_FEP_DENSE_SLOTS_MERGED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TPU_MOE_FUSED_EP_V3_CONFIG", "{" + json_fields + "}")
    calls = []
    monkeypatch.setattr(A._dense_slots, "dense_slots_moe",
                        lambda *a, **k: calls.append(k) or "dense")
    monkeypatch.setattr(A, "fused_ep_moe", lambda *a, **k: "v3")
    run = A.run_moe_v3(SimpleNamespace(axis_names=("d",), shape={"d": EP}), 10, True, "silu",
                       **SHAPE)
    got = run(SimpleNamespace(shape=(8 * 48, 4096)), "w1", "w2", "s1", "s2", "g", "r", "v")
    assert got == "dense"
    return calls[0]["config"]


def test_json_fields_reach_the_dense_call(monkeypatch):
    cfg = _adapter_config(monkeypatch, DENSE + ',"dense_slots_skip_unrouted":true,'
                          '"dense_slots_weight_slots":4')
    assert cfg.dense_slots_skip_unrouted is True
    assert cfg.dense_slots_weight_slots == 4


def test_json_default_is_off(monkeypatch):
    cfg = _adapter_config(monkeypatch, DENSE)
    assert cfg.dense_slots_skip_unrouted is False
    assert cfg.dense_slots_weight_slots == 0


class _Stop(Exception):
    pass


def _builder_kwargs(monkeypatch, config, **kw):
    got = {}

    def fake_build(**k):
        got.update(k)
        raise _Stop

    monkeypatch.setattr(DS, "build_kernel", fake_build)
    monkeypatch.setattr(DS, "SKIP_UNROUTED", False)
    monkeypatch.setattr(DS, "WEIGHT_SLOTS", 0)
    x = SimpleNamespace(shape=(EP * 48, 4096), dtype="bfloat16")
    w1 = SimpleNamespace(shape=(EP * 64, 4096, 2048))
    with pytest.raises(_Stop):
        DS.dense_slots_moe(x, w1, None, None, None, None, top_k=10, renormalize=True,
                           mesh=_Mesh(), config=config, fused=True, merged=True,
                           unpadded=True, **kw)
    return got


@pytest.mark.parametrize("fields,skip,slots", [
    ({}, False, 3),
    ({"dense_slots_skip_unrouted": True}, True, 3),
    ({"dense_slots_weight_slots": 4}, False, 4),
    ({"dense_slots_skip_unrouted": True, "dense_slots_weight_slots": 2}, True, 2),
])
def test_builder_gets_the_settings(monkeypatch, fields, skip, slots):
    k = _builder_kwargs(monkeypatch, Config(**fields))
    assert k["skip_unrouted"] is skip
    assert k["weight_slots"] == slots
    assert k["weight_prefetch"] == slots - 1


def test_bad_slot_count_raises():
    for bad in (1, 6, -1):
        with pytest.raises(ValueError):
            Config(dense_slots_weight_slots=bad)
