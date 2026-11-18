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
NAME = "model.layers.3.mlp.experts"


def _counts(loads, steps):
    """A run's cumulative counts after `steps` steps at constant per-step loads."""
    counts = np.zeros((len(loads), 3, EXPERTS), dtype=np.int64)
    for row, load in enumerate(loads):
        counts[row, 0] = load * steps
        counts[row, 1] = load * load * steps
        counts[row, 2, 0] = steps
    return counts


def _write(directory, names, counts, indices, rank=0, seq=0):
    fields = {
        "counts": counts,
        "layer_names": np.array(names),
        "step": int(counts[0, 2, 0]),
        "wall": 0.0,
    }
    if indices is not None:
        fields["layer_indices"] = np.array(indices, dtype=np.int64)
    np.savez(directory / f"counts_ep{rank}_{seq:04d}.npz", **fields)


def _snapshot(directory, names, loads, indices, steps=100, rank=0, seq=0):
    _write(directory, names, _counts(loads, steps), indices, rank, seq)


def _build(monkeypatch, tmp_path, layers, *flags):
    out = tmp_path / "placement.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe_expert_placement.py",
            f"--layers={layers}",
            "--skip=0",
            f"--out={out}",
            *flags,
            str(tmp_path),
        ],
    )
    if not any(flag.startswith("--shards") for flag in flags):
        sys.argv.insert(1, f"--shards={SHARDS}")
    moe_expert_placement.main()
    return json.loads(out.read_text())


def test_row_costs_put_the_hot_experts_on_different_ranks(monkeypatch, tmp_path):
    _snapshot(tmp_path, [NAME], [HOT], [3])
    table = _build(monkeypatch, tmp_path, "0:10", *ROW_COSTS)
    assert (table["experts"], table["shards"]) == (EXPERTS, SHARDS)
    ranks = table["layers"]["3"]
    assert sorted(e for held in ranks for e in held) == list(range(EXPERTS))
    assert [len(held) for held in ranks] == [EXPERTS // SHARDS] * SHARDS
    assert not any({0, 1} <= set(held) for held in ranks)
    # The table is one TPU_EXPERT_PLACEMENT reads. Imported here so the
    # script's other tests run without the package, as the script does.
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
        [0, 6],
    )
    table = _build(monkeypatch, tmp_path, "0:6", *ROW_COSTS, "--min-gain-us=10")
    assert set(table["layers"]) == {"6"}


def test_the_latest_run_minus_its_skipped_snapshots_is_counted(tmp_path):
    """A restart starts `step` over: only the run from it on counts, and
    `--skip 1` drops that run's first snapshot (its warmup)."""
    warmup = np.full(EXPERTS, 30, dtype=np.int64)
    _snapshot(tmp_path, [NAME], [EVEN], [3], steps=100, seq=0)
    _snapshot(tmp_path, [NAME], [EVEN], [3], steps=200, seq=1)
    _write(tmp_path, [NAME], _counts([warmup], 50), [3], seq=2)
    _write(tmp_path, [NAME], _counts([warmup], 50) + _counts([HOT], 100), [3], seq=3)
    ((index, mean, var),) = moe_expert_placement.load_counts(
        [str(tmp_path)], 1
    ).values()
    assert index == 3
    np.testing.assert_allclose(mean, HOT)
    np.testing.assert_allclose(var, 0)
    with pytest.raises(SystemExit, match="need more than 2"):
        moe_expert_placement.load_counts([str(tmp_path)], 2)


def test_the_ranks_loads_are_summed(tmp_path):
    _snapshot(tmp_path, [NAME], [HOT], [3], rank=0)
    _snapshot(tmp_path, [NAME], [EVEN], [3], rank=1)
    ((_, mean, _),) = moe_expert_placement.load_counts([str(tmp_path)], 0).values()
    np.testing.assert_allclose(mean, HOT + EVEN)


def test_ranks_that_tapped_other_layers_are_refused(tmp_path):
    _snapshot(tmp_path, [NAME], [HOT], [3], rank=0)
    _snapshot(tmp_path, ["model.layers.4.mlp.experts"], [HOT], [4], rank=1)
    with pytest.raises(SystemExit, match="tapped other MoE layers"):
        moe_expert_placement.load_counts([str(tmp_path)], 0)


def test_snapshots_without_table_numbers_are_refused(tmp_path):
    _snapshot(tmp_path, [NAME], [HOT], None)
    with pytest.raises(SystemExit, match="no layer_indices"):
        moe_expert_placement.load_counts([str(tmp_path)], 0)


def test_two_layers_with_one_table_number_are_refused(monkeypatch, tmp_path):
    _snapshot(tmp_path, [NAME, "model.layers.3.extra.experts"], [HOT, HOT], [3, 3])
    with pytest.raises(SystemExit, match="both table layer 3"):
        _build(monkeypatch, tmp_path, "0:10", *ROW_COSTS)


def test_experts_that_do_not_split_evenly_are_refused(monkeypatch, tmp_path):
    _snapshot(tmp_path, [NAME], [HOT], [3])
    with pytest.raises(SystemExit, match="do not split evenly over 3 shards"):
        _build(monkeypatch, tmp_path, "0:10", "--shards=3", *ROW_COSTS)


def test_layers_must_be_first_colon_last(monkeypatch, tmp_path):
    _snapshot(tmp_path, [NAME], [HOT], [3])
    with pytest.raises(SystemExit) as refusal:
        _build(monkeypatch, tmp_path, "3", *ROW_COSTS)
    assert refusal.value.code == 2  # argparse's usage error
