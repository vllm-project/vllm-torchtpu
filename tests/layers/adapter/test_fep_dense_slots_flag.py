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
of at most dense_slots_max_tokens rows per shard (default 64), a multiple of 8,
take the dense-slots layer with the caller's operands, other calls keep v3; a
bad threshold raises. The layer hands its program the shard's rows unpadded and
refuses a count that is not a multiple of 8 (CPU, no kernel is built)."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.kernels.fused_moe.v3.layout import MESH_AXIS
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
        (8 * 60, "v3"),
        (8 * 20, "v3"),
        (8 * 72, "v3"),
        (8 * 1024, "v3"),
    ):
        got, calls = _run(monkeypatch, rows, on)
        assert got == want, (rows, got)
    got, calls = _run(monkeypatch, 8 * 56, on)
    _, a, k = calls[0]
    assert a[1:6] == ("w1", "w2", "s1", "s2", "gating")
    assert k["shard_index"] == "rank" and k["valid_tokens"] == "valid"
    assert k["drop_nan_rows"] is True and k["top_k"] == 10
    assert k["side_effects"] is False
    assert k["config"].dense_slots is True


def test_threshold(monkeypatch):
    fields = {"dense_slots": True, "dense_slots_max_tokens": 128}
    got, _ = _run(monkeypatch, 8 * 128, fields)
    assert got == "dense"
    got, _ = _run(monkeypatch, 8 * 136, fields)
    assert got == "v3"


@pytest.mark.parametrize("bad", [0, 60, -8])
def test_bad_threshold_raises(monkeypatch, bad):
    with pytest.raises(ValueError):
        _run(monkeypatch, 8 * 56, {"dense_slots": True, "dense_slots_max_tokens": bad})


def test_other_fields_leave_it_off(monkeypatch):
    got, _ = _run(monkeypatch, 8 * 56, {"combine_real_tokens": True})
    assert got == "v3"


class _Stop(Exception):
    pass


def _builder_kwargs(monkeypatch, rows_per_shard, **kwargs):
    got = {}

    def fake_build(**k):
        got.update(k)
        raise _Stop

    monkeypatch.setattr(DS, "build_kernel", fake_build)
    mesh = SimpleNamespace(axis_names=(MESH_AXIS,), shape={MESH_AXIS: EP})
    x = SimpleNamespace(shape=(EP * rows_per_shard, 4096), dtype="bfloat16")
    w1 = SimpleNamespace(shape=(EP * 64, 4096, 2048))
    with pytest.raises(_Stop):
        DS.dense_slots_moe(
            x,
            w1,
            None,
            None,
            None,
            None,
            top_k=10,
            renormalize=True,
            mesh=mesh,
            config=Config(),
            **kwargs,
        )
    return got


@pytest.mark.parametrize("rows,padded_to", [(48, 64), (56, 64), (64, 64), (8, 32)])
def test_the_program_takes_the_shards_rows_unpadded(monkeypatch, rows, padded_to):
    k = _builder_kwargs(monkeypatch, rows)
    assert k["real_rows"] == rows
    assert k["tokens_per_shard"] == padded_to
    assert k["experts_per_shard"] == 64 and k["num_experts"] == EP * 64


@pytest.mark.parametrize("bridge", [False, True], ids=["own", "bridge"])
def test_the_program_takes_its_lowering_from_the_arguments(monkeypatch, bridge):
    """The program is built in the form its caller asks for: a shard index
    handed in selects the operand form of the peer addresses, and
    side_effects is what it declares; nothing is read from the kernel
    package's state."""
    lowering = dict(shard_index=object(), side_effects=False) if bridge else {}
    k = _builder_kwargs(monkeypatch, 64, **lowering)
    assert k["shard_index_operand"] is bridge
    assert k["side_effects"] is not bridge


@pytest.mark.parametrize("rows", [20, 60])
def test_rows_not_a_multiple_of_8_are_refused(monkeypatch, rows):
    with pytest.raises(ValueError, match="multiple of 8"):
        _builder_kwargs(monkeypatch, rows)
