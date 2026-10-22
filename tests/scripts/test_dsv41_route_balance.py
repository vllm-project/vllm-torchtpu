# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_moe_routes import (  # noqa: E402
    aggregate,
    load_summary,
    placement,
    read_replay,
    summarize,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("duplicate", [False, True])
def test_replay_integrity_and_unique_routes(tmp_path, duplicate):
    path = tmp_path / "rank-0-sample-0.npz"
    ids = np.array([[0, 0 if duplicate else 1], [1, 2]], dtype=np.int32)
    np.savez_compressed(path, router0_ids=ids, router0_weights=np.ones((2, 2)))
    metadata = {
        "purpose": "native_tpu_router_replay_of_actual_capture",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rank": 0,
        "sample": 0,
        "row": {"rows": 2, "forward_id": 40},
        "routers": [{"key": "router0", "global_num_experts": 4, "topk": 2}],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    if duplicate:
        with pytest.raises(ValueError, match="Invalid routed"):
            read_replay(path)
    else:
        np.testing.assert_array_equal(read_replay(path)[0]["counts"], [1, 2, 1, 0])
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        read_replay(path)


def records():
    rows = []
    for forward in (40, 48):
        for rank in (0, 1):
            rows.append(
                {
                    "forward_id": forward,
                    "module": "layers.0.mlp",
                    "ep_rank": rank,
                    "rank": rank,
                    "ep_size": 2,
                    "global_num_experts": 4,
                    "local_experts": 2,
                    "topk": 2,
                    "scoring_fn": "sqrtsoftplus",
                    "renormalize": True,
                    "routed_scaling_factor": 1.0,
                    "bias_sha256": "same",
                    "logits_sha256": str(forward) + str(rank),
                    "counts": np.array([128, 128, 1, 1])
                    if forward == 40
                    else np.array([1, 128, 128, 1]),
                }
            )
    return rows


def test_fixed_expert_slots_and_tail_rounding():
    owners = placement(np.array([9, 8, 2, 1]), 2)
    assert np.bincount(owners).tolist() == [2, 2]
    assert np.bincount(owners, weights=[9, 8, 2, 1]).tolist() == [10, 10]
    result = load_summary(
        np.array([0, 1, 64, 129]),
        np.array([0, 2, 64, 130]),
        np.array([0, 0, 1, 1]),
        2,
    )
    assert result["tiles_128"]["per_shard"] == [1, 3]
    assert result["computed_rows_64"]["per_shard"] == [64, 256]
    assert result["active_experts"]["per_shard"] == [1, 2]


def test_source_runs_are_aligned_before_expert_counts_are_summed():
    rows = records()[:2]
    rows[0]["counts"] = np.array([33, 1, 0, 0])
    rows[1]["counts"] = np.array([31, 1, 0, 0])
    step = aggregate(rows)[0]
    np.testing.assert_array_equal(step["counts"], [64, 2, 0, 0])
    np.testing.assert_array_equal(step["aligned_counts"], [66, 4, 0, 0])
    loads = load_summary(
        step["counts"], step["aligned_counts"], np.array([0, 0, 1, 1]), 2
    )
    assert loads["computed_rows_64"]["per_shard"] == [192, 0]


def test_same_forward_is_required_even_if_sample_numbers_match():
    rows = records()
    rows[-1]["forward_id"] = 49
    with pytest.raises(ValueError, match="same-forward"):
        aggregate(rows)


def test_duplicate_ep_owner_and_bias_mismatch_are_rejected():
    rows = records()
    rows[1]["ep_rank"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(rows)
    rows = records()
    rows[1]["bias_sha256"] = "different"
    with pytest.raises(ValueError, match="metadata"):
        aggregate(rows)


def test_holdout_does_not_refit_placement():
    report = summarize(aggregate(records()))
    layer = report["layers"][0]
    assert layer["training_forward"] == 40
    assert [v["split"] for v in layer["steps"]] == ["training", "held_out"]
    train, holdout = layer["steps"]
    assert (
        train["fixed_training_placements"]["routed_rows"]["routed_rows"]["max"]
        < train["contiguous"]["routed_rows"]["max"]
    )
    changed = copy.deepcopy(records())
    changed[-1]["counts"] = np.array([999, 0, 0, 0])
    assert (
        summarize(aggregate(changed))["layers"][0]["placements"] == layer["placements"]
    )
    owners = np.asarray(layer["placements"]["routed_rows"])
    assert holdout["fixed_training_placements"]["routed_rows"] == load_summary(
        np.array([2, 256, 256, 2]), np.array([4, 256, 256, 4]), owners, 2
    )
