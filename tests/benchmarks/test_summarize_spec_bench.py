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

import json
from pathlib import Path

from scripts.vllm.benchmarking.summarize_spec_bench import (
    aggregate_spec_results,
    format_summary_table,
)


def _create_mock_result(
    path: Path,
    duration: float,
    acceptance_rate: float | None = None,
    completed: int = 100,
    output_throughput: float = 250.0,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "duration": duration,
        "completed": completed,
        "output_throughput": output_throughput,
    }
    if acceptance_rate is not None:
        data["spec_decode_acceptance_rate"] = acceptance_rate
    with open(path, "w") as f:
        json.dump(data, f)


def test_summarize_spec_bench_single_rep(tmp_path: Path):
    results_dir = tmp_path / "results"
    # BS=1: K=0, K=1, K=2
    _create_mock_result(results_dir / "bs1" / "k0_rep1.json", duration=100.0)
    _create_mock_result(
        results_dir / "bs1" / "k1_rep1.json", duration=80.0, acceptance_rate=0.75
    )
    _create_mock_result(
        results_dir / "bs1" / "k2_rep1.json", duration=50.0, acceptance_rate=0.60
    )

    summary = aggregate_spec_results(results_dir)
    assert 1 in summary
    bs1_summary = summary[1]

    # K=0 baseline
    assert bs1_summary[0]["e2e_s"] == 100.0
    assert bs1_summary[0]["speedup"] == 1.0
    assert bs1_summary[0]["acceptance_rate"] is None

    # K=1
    assert bs1_summary[1]["e2e_s"] == 80.0
    assert bs1_summary[1]["speedup"] == 1.25
    assert bs1_summary[1]["acceptance_rate"] == 0.75

    # K=2
    assert bs1_summary[2]["e2e_s"] == 50.0
    assert bs1_summary[2]["speedup"] == 2.0
    assert bs1_summary[2]["acceptance_rate"] == 0.60

    table_text = format_summary_table(summary)
    assert "TPU — max_num_seqs = 1" in table_text
    assert "1.25" in table_text
    assert "75.00%" in table_text
    assert "2.00" in table_text
    assert "60.00%" in table_text


def test_summarize_spec_bench_multi_rep(tmp_path: Path):
    results_dir = tmp_path / "results"
    # BS=1: K=0 (reps: 90, 110 -> mean 100), K=1 (reps: 45, 55 -> mean 50)
    _create_mock_result(results_dir / "bs1" / "k0_rep1.json", duration=90.0)
    _create_mock_result(results_dir / "bs1" / "k0_rep2.json", duration=110.0)
    _create_mock_result(
        results_dir / "bs1" / "k1_rep1.json", duration=45.0, acceptance_rate=0.80
    )
    _create_mock_result(
        results_dir / "bs1" / "k1_rep2.json", duration=55.0, acceptance_rate=0.82
    )

    summary = aggregate_spec_results(results_dir)
    bs1_summary = summary[1]

    assert bs1_summary[0]["e2e_s"] == 100.0
    assert bs1_summary[1]["e2e_s"] == 50.0
    assert bs1_summary[1]["speedup"] == 2.0
    assert round(bs1_summary[1]["acceptance_rate"], 4) == 0.81


def test_summarize_spec_bench_multi_batch(tmp_path: Path):
    results_dir = tmp_path / "results"
    # BS=1
    _create_mock_result(results_dir / "bs1" / "k0_rep1.json", duration=100.0)
    _create_mock_result(
        results_dir / "bs1" / "k1_rep1.json", duration=80.0, acceptance_rate=0.70
    )
    # BS=4 (independent baseline)
    _create_mock_result(results_dir / "bs4" / "k0_rep1.json", duration=30.0)
    _create_mock_result(
        results_dir / "bs4" / "k1_rep1.json", duration=20.0, acceptance_rate=0.70
    )

    summary = aggregate_spec_results(results_dir)
    assert 1 in summary
    assert 4 in summary

    # BS=4 speedup relative to BS=4's K=0
    assert summary[4][0]["e2e_s"] == 30.0
    assert summary[4][1]["e2e_s"] == 20.0
    assert summary[4][1]["speedup"] == 1.5

    table_text = format_summary_table(summary)
    assert "TPU — max_num_seqs = 1" in table_text
    assert "TPU — max_num_seqs = 4" in table_text
