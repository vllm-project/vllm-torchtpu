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
"""The dense-slots layer is off by default: every call takes the v3 layer. With
the Config field dense_slots (set by the Qwen3.5 DP4 x TP2 layout table), calls
of at most dense_slots_max_tokens rows per shard (default 64) take the
dense-slots layer with the caller's operands, larger calls keep v3; a bad
threshold raises."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.layers.adapter import fused_moe_ep_v3 as A
from vllm_torchtpu.models.qwen3_5.kernels import dense_slots as DS

SHAPE = dict(hidden=4096, inter=1024, experts_per_shard=64)
EP = 8


class _Mesh:  # what run_moe_v3 reads of the mesh: the axis names and the ep width
    axis_names = ("d",)
    shape = {"d": EP}


def _run(monkeypatch, rows, overrides):
    monkeypatch.setattr(A, "_CONFIG_OVERRIDES", dict(overrides))
    calls = []

    def dense(*a, **k):
        calls.append(("dense", a, k))
        return "dense"

    def v3(*a, **k):
        calls.append(("v3", a, k))
        return "v3"

    monkeypatch.setattr(DS, "dense_slots_moe", dense)
    monkeypatch.setattr(A, "fused_ep_moe", v3)
    run = A.run_moe_v3(_Mesh(), 10, True, "silu", **SHAPE)
    x = SimpleNamespace(shape=(rows, 4096))
    ops = ("w1", "w2", "s1", "s2", "gating", "rank", "valid")
    got = run(x, *ops)
    return got, calls


def test_default_is_v3(monkeypatch):
    for rows in (8 * 48, 8 * 56, 8 * 64, 8 * 1024):
        got, calls = _run(monkeypatch, rows, {})
        assert got == "v3" and [c[0] for c in calls] == ["v3"]


def test_on_takes_dense_up_to_the_threshold(monkeypatch):
    on = {"dense_slots": True}
    for rows, want in (
        (8 * 48, "dense"),
        (8 * 56, "dense"),
        (8 * 64, "dense"),
        (8 * 72, "v3"),
        (8 * 1024, "v3"),
    ):
        got, calls = _run(monkeypatch, rows, on)
        assert got == want, (rows, got)
    got, calls = _run(monkeypatch, 8 * 56, on)
    _, a, k = calls[0]
    assert a[1:6] == ("w1", "w2", "s1", "s2", "gating")
    assert k["shard_index"] == "rank" and k["valid_tokens"] == "valid"
    assert k["drop_nan_rows"] is True and k["fused"] is False and k["top_k"] == 10
    assert (
        k["pallas_tables"] is False and k["merged"] is False and k["unpadded"] is False
    )


def test_threshold_and_fused(monkeypatch):
    fields = {
        "dense_slots": True,
        "dense_slots_max_tokens": 128,
        "dense_slots_fused": True,
        "dense_slots_pallas_tables": True,
    }
    got, calls = _run(monkeypatch, 8 * 128, fields)
    assert got == "dense" and calls[0][2]["fused"] is True
    assert calls[0][2]["pallas_tables"] is True
    got, _ = _run(monkeypatch, 8 * 136, fields)
    assert got == "v3"


@pytest.mark.parametrize("bad", [0, 60, -8])
def test_bad_threshold_raises(monkeypatch, bad):
    with pytest.raises(ValueError):
        _run(monkeypatch, 8 * 56, {"dense_slots": True, "dense_slots_max_tokens": bad})


def test_other_fields_leave_it_off(monkeypatch):
    got, _ = _run(monkeypatch, 8 * 56, {"combine_real_tokens": True})
    assert got == "v3"


def test_merged(monkeypatch):
    fields = {
        "dense_slots": True,
        "dense_slots_fused": True,
        "dense_slots_merged": True,
    }
    got, calls = _run(monkeypatch, 8 * 56, fields)
    assert got == "dense"
    k = calls[0][2]
    assert k["fused"] is True and k["merged"] is True and k["unpadded"] is True
