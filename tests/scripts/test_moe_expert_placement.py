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

EXPERTS = 8
# Experts 0 and 1 are hot, so contiguous blocks put both on rank 0.
HOT = np.array([40, 40, 2, 2, 2, 2, 2, 2], dtype=np.int64)
EVEN = np.full(EXPERTS, 12, dtype=np.int64)
# A slot costs its row count: each layer evens out the rows per rank.
ROW_COSTS = ("--floor-us=0", "--visit-us=0", "--tile-us=0", "--short-tile-us=0")
NAME = "model.layers.3.mlp.experts"


def _counts(loads, steps):
    """A run's cumulative counts after `steps` steps at constant loads."""
    counts = np.zeros((len(loads), 3, EXPERTS), dtype=np.int64)
    for row, load in enumerate(loads):
        counts[row, 0], counts[row, 1] = load * steps, load * load * steps
        counts[row, 2, 0] = steps
    return counts


def _write(directory, names, counts, indices, rank=0, seq=0):
    fields = {"counts": counts, "layer_names": np.array(names), "wall": 0.0}
    fields["step"] = int(counts[0, 2, 0])
    if indices is not None:
        fields["layer_indices"] = np.array(indices, dtype=np.int64)
    np.savez(directory / f"counts_ep{rank}_{seq:04d}.npz", **fields)


def _build(monkeypatch, tmp_path, *flags, layers="0:10"):
    out = tmp_path / "placement.json"
    argv = [f"--layers={layers}", "--skip=0", f"--out={out}", "--row-us=1"]
    if not any(flag.startswith("--shards") for flag in flags):
        argv.append("--shards=2")
    monkeypatch.setattr(sys, "argv", ["x", *argv, *ROW_COSTS, *flags, str(tmp_path)])
    moe_expert_placement.main()
    return json.loads(out.read_text())


def test_row_costs_put_the_hot_experts_on_different_ranks(monkeypatch, tmp_path):
    _write(tmp_path, [NAME], _counts([HOT], 100), [3])
    table = _build(monkeypatch, tmp_path)
    ranks = table["layers"]["3"]
    assert (table["experts"], table["shards"]) == (EXPERTS, 2)
    assert sorted(e for held in ranks for e in held) == list(range(EXPERTS))
    assert not any({0, 1} <= set(held) for held in ranks)
    # Imported here so the other tests run without the package, as the script.
    expert_placement = pytest.importorskip(
        "vllm_torchtpu.layers.adapter.expert_placement"
    )
    assert expert_placement.parse(table).slot_order(3) == tuple(sum(ranks, []))


def test_a_drafter_layer_takes_the_table_number_the_tap_recorded(monkeypatch, tmp_path):
    """A `mtp.layers.0` drafter of a 6-layer target is table layer 6, not 0."""
    names = ["model.layers.0.mlp.experts", "mtp.layers.0.mlp.experts"]
    _write(tmp_path, names, _counts([EVEN, HOT], 100), [0, 6])
    table = _build(monkeypatch, tmp_path, "--min-gain-us=10", layers="0:6")
    assert set(table["layers"]) == {"6"}


def test_the_latest_runs_minus_their_skipped_snapshots_are_summed(tmp_path):
    """A restart starts `step` over: only the run from it on counts, minus its
    `--skip`-th snapshot (warmup at another load); ranks are summed."""
    warmup = np.full(EXPERTS, 30, dtype=np.int64)
    _write(tmp_path, [NAME], _counts([EVEN], 200), [3], seq=0)
    _write(tmp_path, [NAME], _counts([warmup], 50), [3], seq=1)
    _write(tmp_path, [NAME], _counts([warmup], 50) + _counts([HOT], 100), [3], seq=2)
    _write(tmp_path, [NAME], _counts([EVEN], 50), [3], rank=1, seq=0)
    _write(tmp_path, [NAME], _counts([EVEN], 150), [3], rank=1, seq=1)
    load = moe_expert_placement.load_counts([str(tmp_path)], 1)
    ((index, mean, var),) = load.values()
    assert index == 3
    np.testing.assert_allclose(mean, HOT + EVEN)
    np.testing.assert_allclose(var, 0)
    with pytest.raises(SystemExit, match="need more than 2"):
        moe_expert_placement.load_counts([str(tmp_path)], 2)


@pytest.mark.parametrize(
    ("snapshots", "flags", "refusal"),
    [
        ([([NAME], [3], 0)], ("--layers=3",), "^2$"),  # argparse's usage error
        ([([NAME], None, 0)], (), "no layer_indices"),
        (
            [([NAME], [3], 0), (["model.layers.4.mlp.experts"], [4], 1)],
            (),
            "tapped other MoE layers",
        ),
        ([([NAME, NAME + "2"], [3, 3], 0)], (), "both table layer 3"),
        ([([NAME], [3], 0)], ("--shards=3",), "split evenly over 3"),
    ],
)
def test_bad_input_is_refused(monkeypatch, tmp_path, snapshots, flags, refusal):
    for names, indices, rank in snapshots:
        _write(tmp_path, names, _counts([HOT] * len(names), 100), indices, rank)
    with pytest.raises(SystemExit, match=refusal):
        _build(monkeypatch, tmp_path, *flags)
