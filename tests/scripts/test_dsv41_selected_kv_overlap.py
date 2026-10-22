# SPDX-License-Identifier: Apache-2.0
"""Validate set reuse and reject stale or corrupt decode calibration data."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/dsv41_bench"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_selected_kv import read_capture, request_overlap, summarize  # noqa: E402

pytestmark = pytest.mark.cpu_test


def test_reordered_shared_ids_and_invalid_padding():
    shared = request_overlap(np.array([[0, 1, -1], [1, -1, 0]]))
    assert shared["union_rows"] == 2
    assert shared["redundant_fraction"] == 0.5
    assert shared["adjacent_shared_fractions"] == [1.0]
    assert shared["duplicate_ids_within_queries"] == 0
    distinct = request_overlap(np.array([[0, 1, -1], [2, 3, -1]]))
    assert distinct["redundant_fraction"] == 0
    assert distinct["adjacent_shared_fractions"] == [0.0]
    duplicate = request_overlap(np.array([[0, 0, -1], [0, 1, -1]]))
    assert duplicate["duplicate_ids_within_queries"] == 1


def write_capture(tmp_path, *, stale=False, out_of_range=False):
    path = tmp_path / "rank-0-sample-0.npz"
    selected = np.array([[0, 1], [1, 0], [0, 1], [2, 3]], dtype=np.int32)
    if out_of_range:
        selected[0, 0] = 50
    positions = np.array([99 if stale else 100, 101, 200, 201], dtype=np.int32)
    np.savez_compressed(
        path,
        layer0_selected=selected,
        layer0_positions=positions,
        layer0_requests=np.array([0, 0, 1, 1], dtype=np.int32),
    )
    metadata = {
        "purpose": "calibration_only_not_performance",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rank": 0,
        "sample": 0,
        "queries_per_request": 2,
        "row": {
            "phase": "decode",
            "requests": 2,
            "rows": 4,
            "starts": [100, 200],
            "ends": [102, 202],
            "forward_id": 10,
        },
        "layers": [
            {
                "key": "layer0",
                "module": "layers.0.attn",
                "compress_ratio": 4,
                "compressed_cache_prefix": "cache0",
                "selected_shape": [4, 2],
            }
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path


def test_request_local_ids_are_not_deduplicated_across_requests(tmp_path):
    records = read_capture(write_capture(tmp_path))
    assert [r["union_rows"] for r in records] == [2, 4]
    result = summarize(records)
    assert result["union_rows"] == 6
    assert result["ideal_selected_row_traffic_reduction"] == 0.25


def test_stale_query_positions_are_rejected(tmp_path):
    with pytest.raises(AssertionError):
        read_capture(write_capture(tmp_path, stale=True))


def test_out_of_context_selection_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="outside compressed context"):
        read_capture(write_capture(tmp_path, out_of_range=True))


def test_corrupt_capture_is_rejected_before_loading(tmp_path):
    path = write_capture(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_capture(path)


@pytest.mark.parametrize("offset", [0, -3])
def test_capture_matches_actual_stage_inputs_not_speculative_host_metadata(
    tmp_path, offset
):
    path = write_capture(tmp_path)
    with np.load(path) as payload:
        arrays = dict(payload)
    arrays["encoder_input_positions"] = arrays["layer0_positions"].copy()
    arrays["decoder_input_positions"] = arrays["layer0_positions"].copy()
    arrays["layer0_positions"] += offset
    np.savez_compressed(path, **arrays)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["position_source"] = "actual_stage_input_tensors"
    metadata["host_metadata_row"] = {
        **metadata["row"],
        "starts": [103, 203],
        "ends": [105, 205],
    }
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    if offset:
        with pytest.raises(AssertionError):
            read_capture(path)
    else:
        assert len(read_capture(path)) == 2
