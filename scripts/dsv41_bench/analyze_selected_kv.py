# SPDX-License-Identifier: Apache-2.0
"""Measure within-request KV reuse from calibration-only decode captures.

Each NPZ has a JSON sidecar with its SHA-256, decode boundaries and
indexer metadata. The capture copies selected IDs, query positions and request
IDs together after a real decode forward. IDs are local to each request and
compressed cache; never merge them across requests or indexers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def request_overlap(selected: np.ndarray) -> dict:
    """Count exact set reuse without assuming top-k column order matches."""
    sets = [set(int(v) for v in row if v >= 0) for row in selected]
    valid = int(np.count_nonzero(selected >= 0))
    distinct_per_query = sum(len(values) for values in sets)
    union = len(set().union(*sets))
    adjacent = [
        len(left & right) / max(1, min(len(left), len(right)))
        for left, right in zip(sets, sets[1:])
    ]
    return {
        "valid_selections": valid,
        "duplicate_ids_within_queries": valid - distinct_per_query,
        "union_rows": union,
        "redundant_fraction": 1 - union / max(1, valid),
        "adjacent_shared_fractions": adjacent,
        "union_rows_padded_128": (union + 127) // 128 * 128,
    }


def read_capture(path: Path) -> list[dict]:
    """Validate capture provenance and return one record per request/indexer."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["purpose"] != "calibration_only_not_performance":
        raise ValueError(f"Not a calibration capture: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError(f"Capture checksum mismatch: {path}")
    host = metadata["row"]
    queries = metadata["queries_per_request"]
    count = host["requests"]
    if queries < 2 or count < 1:
        raise ValueError(f"Need multiple queries per active request: {path}")
    if host["phase"] != "decode" or host["rows"] != count * queries:
        raise ValueError(f"Not a complete decode window: {path}")
    if len(host["starts"]) != count or len(host["ends"]) != count:
        raise ValueError(f"Host request boundaries are incomplete: {path}")
    records = []
    with np.load(path, allow_pickle=False) as data:
        input_positions = None
        if metadata.get("position_source") == "actual_stage_input_tensors":
            input_positions = data["encoder_input_positions"]
            if input_positions.shape != (host["rows"],):
                raise ValueError(f"Unexpected stage input shape: {path}")
            np.testing.assert_array_equal(
                input_positions, data["decoder_input_positions"]
            )
        for layer in metadata["layers"]:
            key = layer["key"]
            selected = data[f"{key}_selected"]
            positions = data[f"{key}_positions"]
            requests = data[f"{key}_requests"]
            if (
                selected.ndim != 2
                or selected.shape != tuple(layer["selected_shape"])
                or selected.shape[0] != host["rows"]
                or positions.shape != (host["rows"],)
                or requests.shape != positions.shape
                or not all(
                    np.issubdtype(array.dtype, np.integer)
                    for array in (selected, positions, requests)
                )
            ):
                raise ValueError(f"Invalid capture shapes/dtypes: {path}, {key}")
            if layer["compress_ratio"] <= 0:
                raise ValueError(f"Invalid compression ratio: {path}, {key}")
            np.testing.assert_array_equal(np.unique(requests), np.arange(count))
            if input_positions is not None:
                np.testing.assert_array_equal(positions, input_positions)
            lengths = (positions + 1) // layer["compress_ratio"]
            if np.any(selected < -1) or np.any(
                (selected >= 0) & (selected >= lengths[:, None])
            ):
                raise ValueError(f"Selection outside compressed context: {path}, {key}")
            for request in range(count):
                indices = np.flatnonzero(requests == request)
                expected = np.arange(host["starts"][request], host["ends"][request])
                if len(indices) != queries or len(expected) != queries:
                    raise ValueError(f"Unexpected verification row count: {path}")
                np.testing.assert_array_equal(positions[indices], expected)
                records.append(
                    {
                        "rank": metadata["rank"],
                        "sample": metadata["sample"],
                        "forward_id": host["forward_id"],
                        "module": layer["module"],
                        "compressed_cache_prefix": layer["compressed_cache_prefix"],
                        "request": request,
                        **request_overlap(selected[indices]),
                    }
                )
    if not records:
        raise ValueError(f"No indexer selections in {path}")
    return records


def distribution(values: list[int] | list[float]) -> dict:
    return dict(
        zip(
            ("min", "p50", "p90", "p95", "max"),
            np.percentile(values, [0, 50, 90, 95, 100]).tolist(),
            strict=True,
        )
    )


def summarize(records: list[dict]) -> dict:
    """Report a row-traffic upper bound, without inferring kernel speedup."""
    valid = sum(row["valid_selections"] for row in records)
    union = sum(row["union_rows"] for row in records)
    return {
        "request_indexer_samples": len(records),
        "valid_selections": valid,
        "union_rows": union,
        "duplicate_ids_within_queries": sum(
            row["duplicate_ids_within_queries"] for row in records
        ),
        "ideal_selected_row_traffic_reduction": 1 - union / max(1, valid),
        "union_rows_distribution": distribution([r["union_rows"] for r in records]),
        "redundant_fraction_distribution": distribution(
            [r["redundant_fraction"] for r in records]
        ),
        "adjacent_shared_fraction_distribution": distribution(
            [v for r in records for v in r["adjacent_shared_fractions"]]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.capture_dir.glob("rank-*-sample-*.npz"))
    if not paths:
        parser.error("No rank/sample NPZ captures found")
    records = [record for path in paths for record in read_capture(path)]
    groups = defaultdict(list)
    for record in records:
        groups[record["module"]].append(record)
    report = {
        "scope": "Observed selected rows only; not a kernel speedup estimate",
        "captures": [p.name for p in paths],
        "aggregate": summarize(records),
        "per_indexer": {key: summarize(rows) for key, rows in sorted(groups.items())},
        "records": records,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"captures": len(paths), "aggregate": report["aggregate"]}))


if __name__ == "__main__":
    main()
