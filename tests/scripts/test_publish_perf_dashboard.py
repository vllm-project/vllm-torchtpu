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

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "benchmarking"
    / "publish_perf_dashboard.py"
)
_spec = importlib.util.spec_from_file_location("publish_perf_dashboard", _SCRIPT)
publish_perf_dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(publish_perf_dashboard)

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
