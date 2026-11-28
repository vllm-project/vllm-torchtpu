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
"""Unit tests for scripts/vllm/benchmarking/publish_perf_dashboard.py."""

import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "publish_perf_dashboard.py"
)
# The script is run as a script, so it imports its sibling bench_metrics.
# Restore sys.path after module load.
_BENCHMARKING_STR = str(_SCRIPT.parent)
_added_path = _BENCHMARKING_STR not in sys.path
if _added_path:
    sys.path.insert(0, _BENCHMARKING_STR)
try:
    _spec = importlib.util.spec_from_file_location("publish_perf_dashboard", _SCRIPT)
    publish_perf_dashboard = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(publish_perf_dashboard)
finally:
    if _added_path and _BENCHMARKING_STR in sys.path:
        sys.path.remove(_BENCHMARKING_STR)

_CONFIG = {
    "model": "Qwen/Qwen3.5-397B-A17B-FP8",
    "tensor_parallelism": 2,
    "data_parallelism": 4,
    "enable_ep": True,
    "quantization": "fp8",
}
_RAW = {
    "date": "20261020-104845",
    "max_concurrency": 256,
    "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
    "total_token_throughput": 6942.0,
    "output_throughput": 6172.0,
    "mean_ttft_ms": 1434.0,
    "p99_tpot_ms": 40.0,
}


def test_rows_match_the_dashboard_schema(tmp_path):
    """Only measured runs become rows, in seconds and per-chip throughput."""
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    (tmp_path / "isl1024_osl8192_c256.json").write_text(json.dumps(_RAW))
    for skipped in ("warmup1", "profile"):
        (tmp_path / f"isl1024_osl8192_c256.{skipped}.json").write_text("{}")

    (row,) = publish_perf_dashboard.collect_rows(
        tmp_path, device="tpu7x", chips=4, image="img:tag"
    )

    assert row["date"] == "2026-10-20 10:48:45"
    assert (row["isl"], row["osl"], row["conc"]) == (1024, 8192, 256)
    assert (row["tp"], row["dp"], row["ep"]) == (2, 4, 8)
    assert row["tput_per_gpu"] == pytest.approx(6942.0 / 4)
    assert row["input_tput_per_gpu"] == pytest.approx(770.0 / 4)
    assert row["mean_ttft"] == pytest.approx(1.434)
    assert row["p99_intvty"] == pytest.approx(25.0)


@pytest.mark.parametrize("chips", [0, -4, True, None])
def test_a_row_needs_a_positive_chip_count(chips):
    with pytest.raises(ValueError, match="chips must be a positive integer"):
        publish_perf_dashboard.to_row(
            _RAW,
            _CONFIG,
            isl=1024,
            osl=8192,
            conc=256,
            device="tpu7x",
            chips=chips,
            image="img:tag",
        )


def test_a_row_needs_a_finite_throughput():
    raw = {**_RAW, "output_throughput": float("nan")}

    with pytest.raises(ValueError, match="cannot compute throughput per chip"):
        publish_perf_dashboard.to_row(
            raw,
            _CONFIG,
            isl=1024,
            osl=8192,
            conc=256,
            device="tpu7x",
            chips=4,
            image="img:tag",
        )


def test_zero_tpot_publishes_zero_interactivity():
    raw = {**_RAW, "p99_tpot_ms": 0.0}

    row = publish_perf_dashboard.to_row(
        raw,
        _CONFIG,
        isl=1024,
        osl=8192,
        conc=256,
        device="tpu7x",
        chips=4,
        image="img:tag",
    )

    assert (row["p99_tpot"], row["p99_intvty"]) == (0.0, 0.0)
