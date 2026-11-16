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
"""Unit tests for scripts/moe_expert_placement.py."""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "moe_expert_placement.py"
)
_spec = importlib.util.spec_from_file_location("moe_expert_placement", _SCRIPT)
moe_expert_placement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moe_expert_placement)

EXPERTS, SHARDS = 8, 2
# Experts 0 and 1 are hot, so contiguous blocks put both on rank 0.
HOT = np.array([40, 40, 2, 2, 2, 2, 2, 2], dtype=np.int64)
EVEN = np.full(EXPERTS, 12, dtype=np.int64)
# A slot costs its row count: each layer evens out the rows per rank.
ROW_COSTS = (
    "--floor-us=0",
    "--visit-us=0",
    "--tile-us=0",
    "--short-tile-us=0",
    "--row-us=1",
)


def _snapshot(directory, names, loads, indices=None, steps=100):
    """One cumulative snapshot of `steps` steps at constant per-step loads."""
    counts = np.zeros((len(names), 3, EXPERTS), dtype=np.int64)
    for row, load in enumerate(loads):
        counts[row, 0] = load * steps
        counts[row, 1] = load * load * steps
        counts[row, 2, 0] = steps
    fields = {
        "counts": counts,
        "layer_names": np.array(names),
        "step": steps,
        "wall": 0.0,
    }
    if indices is not None:
        fields["layer_indices"] = np.array(indices, dtype=np.int64)
    np.savez(directory / "counts_ep0_0000.npz", **fields)


def _build(monkeypatch, tmp_path, layers, *flags):
    out = tmp_path / "placement.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe_expert_placement.py",
            f"--shards={SHARDS}",
            f"--layers={layers}",
            "--skip=0",
            f"--out={out}",
            *flags,
            str(tmp_path),
        ],
    )
    moe_expert_placement.main()
    return json.loads(out.read_text())


def test_row_costs_put_the_hot_experts_on_different_ranks(monkeypatch, tmp_path):
    _snapshot(tmp_path, ["model.layers.3.mlp.experts"], [HOT], indices=[3])
    table = _build(monkeypatch, tmp_path, "0:10", *ROW_COSTS)
    assert (table["experts"], table["shards"]) == (EXPERTS, SHARDS)
    ranks = table["layers"]["3"]
    assert sorted(e for held in ranks for e in held) == list(range(EXPERTS))
    assert [len(held) for held in ranks] == [EXPERTS // SHARDS] * SHARDS
    assert not any({0, 1} <= set(held) for held in ranks)
    # The table is one TPU_EXPERT_PLACEMENT reads.
    expert_placement = pytest.importorskip(
        "vllm_torchtpu.layers.adapter.expert_placement"
    )
    assert expert_placement.parse(table).slot_order(3) == tuple(
        e for held in ranks for e in held
    )


def test_a_drafter_layer_takes_the_table_number_the_tap_recorded(monkeypatch, tmp_path):
    """A `mtp.layers.0` drafter of a 6-layer target is table layer 6; it must
    not be read as target layer 0 and take its place."""
    _snapshot(
        tmp_path,
        ["model.layers.0.mlp.experts", "mtp.layers.0.mlp.experts"],
        [EVEN, HOT],
        indices=[0, 6],
    )
    table = _build(monkeypatch, tmp_path, "0:6", *ROW_COSTS, "--min-gain-us=10")
    assert set(table["layers"]) == {"6"}


def test_snapshots_without_table_numbers_skip_drafter_layers(monkeypatch, tmp_path):
    _snapshot(
        tmp_path,
        ["model.layers.0.mlp.experts", "mtp.layers.0.mlp.experts"],
        [EVEN, HOT],
    )
    table = _build(monkeypatch, tmp_path, "0:6", *ROW_COSTS, "--min-gain-us=10")
    assert table["layers"] == {}


def test_two_layers_with_one_table_number_are_refused(monkeypatch, tmp_path):
    _snapshot(
        tmp_path,
        ["model.layers.3.mlp.experts", "model.layers.3.extra.experts"],
        [HOT, HOT],
        indices=[3, 3],
    )
    with pytest.raises(SystemExit, match="both table layer 3"):
        _build(monkeypatch, tmp_path, "0:10", *ROW_COSTS)
