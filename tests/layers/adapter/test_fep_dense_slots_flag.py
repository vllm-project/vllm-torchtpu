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
"""TPU_MOE_FEP_DENSE_SLOTS (default off): unset, every call takes the v3 layer; set, calls of at
most TPU_MOE_FEP_DENSE_SLOTS_MAX_TOKENS rows per shard (default 64) take the dense-slots layer
with the caller's operands, larger calls keep v3; a bad threshold raises."""
from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as A

SHAPE = dict(hidden=4096, inter=1024, experts_per_shard=64)
EP = 8


class _Mesh:  # what run_moe_v3 reads of the mesh: the axis names and the ep width
    axis_names = ("d",)
    shape = {"d": EP}


def _run(monkeypatch, rows, env):
    for k in ("TPU_MOE_FEP_DENSE_SLOTS", "TPU_MOE_FEP_DENSE_SLOTS_MAX_TOKENS",
              "TPU_MOE_FEP_DENSE_SLOTS_FUSED", "TPU_MOE_FEP_DENSE_SLOTS_PALLAS_TABLES",
              "TPU_MOE_FEP_DENSE_SLOTS_MERGED",
              "TPU_MOE_FUSED_EP_V3_CONFIG"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    calls = []

    def dense(*a, **k):
        calls.append(("dense", a, k))
        return "dense"

    def v3(*a, **k):
        calls.append(("v3", a, k))
        return "v3"

    monkeypatch.setattr(A._dense_slots, "dense_slots_moe", dense)
    monkeypatch.setattr(A, "fused_ep_moe", v3)
    run = A.run_moe_v3(_Mesh(), 10, True, "silu", **SHAPE)
    x = SimpleNamespace(shape=(rows, 4096))
    ops = ("w1", "w2", "s1", "s2", "gating", "rank", "valid")
    got = run(x, *ops)
    return got, calls


def test_unset_is_v3(monkeypatch):
    for rows in (8 * 48, 8 * 56, 8 * 64, 8 * 1024):
        got, calls = _run(monkeypatch, rows, {})
        assert got == "v3" and [c[0] for c in calls] == ["v3"]


def test_on_takes_dense_up_to_the_threshold(monkeypatch):
    on = {"TPU_MOE_FEP_DENSE_SLOTS": "1"}
    for rows, want in ((8 * 48, "dense"), (8 * 56, "dense"), (8 * 64, "dense"),
                       (8 * 72, "v3"), (8 * 1024, "v3")):
        got, calls = _run(monkeypatch, rows, on)
        assert got == want, (rows, got)
    got, calls = _run(monkeypatch, 8 * 56, on)
    _, a, k = calls[0]
    assert a[1:6] == ("w1", "w2", "s1", "s2", "gating")
    assert k["shard_index"] == "rank" and k["valid_tokens"] == "valid"
    assert k["drop_nan_rows"] is True and k["fused"] is False and k["top_k"] == 10
    assert k["pallas_tables"] is False and k["merged"] is False and k["unpadded"] is False


def test_threshold_and_fused(monkeypatch):
    env = {"TPU_MOE_FEP_DENSE_SLOTS": "1", "TPU_MOE_FEP_DENSE_SLOTS_MAX_TOKENS": "128",
           "TPU_MOE_FEP_DENSE_SLOTS_FUSED": "1", "TPU_MOE_FEP_DENSE_SLOTS_PALLAS_TABLES": "1"}
    got, calls = _run(monkeypatch, 8 * 128, env)
    assert got == "dense" and calls[0][2]["fused"] is True
    assert calls[0][2]["pallas_tables"] is True
    got, _ = _run(monkeypatch, 8 * 136, env)
    assert got == "v3"


@pytest.mark.parametrize("bad", ["0", "60", "-8"])
def test_bad_threshold_raises(monkeypatch, bad):
    with pytest.raises(ValueError):
        _run(monkeypatch, 8 * 56, {"TPU_MOE_FEP_DENSE_SLOTS": "1",
                                   "TPU_MOE_FEP_DENSE_SLOTS_MAX_TOKENS": bad})


def test_config_fields_turn_it_on(monkeypatch):
    """The recipe passes TPU_MOE_FUSED_EP_V3_CONFIG but not new env names: the Config fields
    alone must turn the layer on, with the same threshold and variants."""
    cfg = ('{"combine_real_tokens": true, "dense_slots": true, "dense_slots_fused": true, '
           '"dense_slots_pallas_tables": true}')
    got, calls = _run(monkeypatch, 8 * 56, {"TPU_MOE_FUSED_EP_V3_CONFIG": cfg})
    assert got == "dense"
    assert calls[0][2]["fused"] is True and calls[0][2]["pallas_tables"] is True
    got, _ = _run(monkeypatch, 8 * 72, {"TPU_MOE_FUSED_EP_V3_CONFIG": cfg})
    assert got == "v3"
    got, _ = _run(monkeypatch, 8 * 56, {"TPU_MOE_FUSED_EP_V3_CONFIG": '{"combine_real_tokens": true}'})
    assert got == "v3"


def test_merged(monkeypatch):
    cfg = '{"dense_slots": true, "dense_slots_fused": true, "dense_slots_merged": true}'
    got, calls = _run(monkeypatch, 8 * 56, {"TPU_MOE_FUSED_EP_V3_CONFIG": cfg})
    assert got == "dense"
    k = calls[0][2]
    assert k["fused"] is True and k["merged"] is True and k["unpadded"] is True
    got, calls = _run(monkeypatch, 8 * 56, {"TPU_MOE_FEP_DENSE_SLOTS": "1",
                                            "TPU_MOE_FEP_DENSE_SLOTS_FUSED": "1",
                                            "TPU_MOE_FEP_DENSE_SLOTS_MERGED": "1"})
    assert got == "dense" and calls[0][2]["merged"] is True
