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
"""Rows the BigQuery uploaders write for each kind of run, and their helpers.

Covers scripts/vllm/benchmarking/upload_results.py, bq_utils.py and
bench_metrics.py, and .buildkite/scripts/parse_gke_results.py. Uploader tests
run in dry-run mode over fixture files (tests/scripts/data/upload_results) and
read the rows back out of the INSERT statements they print; helper unit tests
call bench_metrics directly. The fixtures have the keys and value types that
the pinned `vllm bench serve` (v0.33.0) saves for each lane's flags; their
values are synthetic. Nothing reaches BigQuery or Spanner: subprocess.run, the
only way either CLI is invoked, raises.
"""

import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import uuid

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
_BENCHMARKING = _REPO / "scripts" / "vllm" / "benchmarking"
_DATA = pathlib.Path(__file__).resolve().parent / "data" / "upload_results"
_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
_TABLE = "cloud-ullm-inference-ci-cd.llm_benchmark_analytics.benchmark_runs"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# upload_results.py imports its sibling bq_utils, so their directory goes on
# sys.path. It stays there: parse_gke_results.py inserts it again for itself.
sys.path.insert(0, str(_BENCHMARKING))
upload_results = _load("upload_results", _BENCHMARKING / "upload_results.py")
parse_gke_results = _load(
    "parse_gke_results", _REPO / ".buildkite" / "scripts" / "parse_gke_results.py"
)
# The module the parser uses, not a second copy of it.
bench_metrics = parse_gke_results.bench_metrics

_ROW_RE = re.compile(
    r"^INSERT INTO `(?P<table>[^`]+)` \( record_id, created_time, repository, "
    r"run_type, model_id, metrics, config \) VALUES \( '(?P<record_id>[^']*)', "
    r"(?P<created_time>.+?), '(?P<repository>[^']*)', '(?P<run_type>[^']*)', "
    r"'(?P<model_id>[^']*)', \[(?P<metrics>.*)\], "
    r"PARSE_JSON\('(?P<config>.*)', wide_number_mode=>'round'\) \);$"
)
_METRICS_RE = re.compile(
    r"STRUCT\('(\w+)', PARSE_JSON\('(.*?)', wide_number_mode=>'round'\)\)"
)


def _sql_json(text: str):
    return json.loads(text.replace("''", "'"))


def _rows(stdout: str) -> list[dict]:
    """The BigQuery rows in an uploader's dry-run output, in insert order."""
    rows = []
    for line in stdout.splitlines():
        match = _ROW_RE.match(line)
        if not match:
            continue
        row = match.groupdict()
        row["metrics"] = {
            metrics_type: _sql_json(values)
            for metrics_type, values in _METRICS_RE.findall(row["metrics"])
        }
        row["config"] = _sql_json(row["config"])
        rows.append(row)
    return rows


def _forbid_subprocess(*args, **kwargs):
    raise AssertionError(f"a test must not run a database CLI: {args}")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """A fixed Buildkite environment, predictable record ids, and no way to
    reach a database."""
    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)
    for name in (
        "BQ_TABLE",
        "CI_RUNNER",
        "CREATED_BY",
        "DOCKER_IMAGE",
        "GCP_INSTANCE_NAME",
        "REPOSITORY",
        "RUN_TYPE",
        "TPU_ACCELERATOR_TYPE",
        "TPU_MULTIHOST_BACKEND",
        "TPU_TOPOLOGY",
        "TPU_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BUILDKITE_COMMIT", "0123abcd")
    monkeypatch.setenv("BUILDKITE_BUILD_NUMBER", "4242")
    monkeypatch.setenv("BUILDKITE_AGENT_NAME", "kube-agent-1")
    monkeypatch.setenv("TPU_NAME", "tpu7x-2x2x1")
    ids = iter(range(1, 1000))
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(ids)))


def _record_id(n: int) -> str:
    return str(uuid.UUID(int=n))


def _results_dir(tmp_path: pathlib.Path, *names: str) -> pathlib.Path:
    """A results directory holding the named fixtures (files or trees)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for name in names:
        source = _DATA / name
        if source.is_dir():
            shutil.copytree(source, results_dir / name)
        else:
            shutil.copy(source, results_dir / name)
    return results_dir


def _upload(monkeypatch, capsys, results_dir, *args) -> tuple[int, list[dict]]:
    monkeypatch.setattr(
        sys,
        "argv",
        ["upload_results.py", "--results-dir", str(results_dir), "--skip-db-upload"]
        + [str(a) for a in args],
    )
    returncode = upload_results.main()
    return returncode, _rows(capsys.readouterr().out)


_BASE_CONFIG = {
    "code_hash": "0123abcd",
    "created_by": "buildkite-agent",
    "device_type": "tpu7x-2x2x1",
    "run_by": "kube-agent-1",
    "job_reference": "4242",
    "profile_gcs_dir": None,
}
_ENGINE_FLAGS = {
    "max_num_seqs": 16,
    "max_num_batched_tokens": 16384,
    "tensor_parallel_size": 2,
    "data_parallel_size": 4,
    "max_model_len": 16384,
    "enable_ep": True,
    "quantization": "fp8",
    "multihost_backend": None,
}
# No p90_* keys: run_benchmarks.sh leaves --metric-percentiles at its default
# of 99.
_BENCH_METRICS = {
    "request_throughput": 0.03125,
    "output_throughput": 256.0,
    "total_token_throughput": 288.0,
    "max_output_tokens_per_s": 352.0,
    "rtfx": 0.0,
    "mean_ttft_ms": 1434.0,
    "median_ttft_ms": 1200.0,
    "std_ttft_ms": 300.0,
    "p99_ttft_ms": 2500.0,
    "mean_tpot_ms": 36.0,
    "median_tpot_ms": 32.0,
    "std_tpot_ms": 4.0,
    "p99_tpot_ms": 50.0,
    "mean_itl_ms": 36.0,
    "median_itl_ms": 32.0,
    "std_itl_ms": 4.0,
    "p99_itl_ms": 50.0,
    "completed": 16,
    "failed": 0,
    "total_input_tokens": 16384,
    "total_output_tokens": 131072,
    "duration": 512.0,
}
_SPEED_GRID_ROW = {
    "table": _TABLE,
    "record_id": _record_id(1),
    "created_time": "TIMESTAMP '2026-10-20T10:48:45Z'",
    "repository": "vllm-torchtpu",
    "run_type": "DAILY",
    "model_id": _MODEL,
    "metrics": {"vllm_bench": _BENCH_METRICS, "lm_eval": {"gsm8k": 0.98}},
    "config": {
        **_BASE_CONFIG,
        "backend": "vllm",
        "engine_flags": _ENGINE_FLAGS,
        "workload": {
            "input_len": 1024,
            "output_len": 8192,
            "num_prompts": 16,
            # What the bench saves for an unlimited rate: a string, not null.
            "request_rate": "inf",
            "burstiness": 1.0,
            "max_concurrency": 256,
        },
    },
}


def test_speed_grid_row_carries_the_cell_and_the_run_accuracy(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(
        tmp_path, "config.json", "isl1024_osl8192_c256.json", "lm_eval"
    )

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [_SPEED_GRID_ROW]


_GKE_DECODE_CELL = "pd_disagg_decode_round1_c8_i8192_o1024"
_GKE_ROW = {
    "table": _TABLE,
    "record_id": f"kube-run-1_inf_c8_{'0' * 6}",
    "created_time": "TIMESTAMP '2026-10-20T10:48:45Z'",
    "repository": "vllm-torchtpu",
    "run_type": "GKE_DISAGG",
    "model_id": _MODEL,
    "metrics": {
        # disagg_benchmark.sh asks for percentiles 50, 90 and 99; the P50 and
        # end-to-end latency keys are not BENCH_METRIC_FIELDS.
        "vllm_bench": {
            "request_throughput": 0.5,
            "output_throughput": 512.0,
            "total_token_throughput": 4608.0,
            "max_output_tokens_per_s": 640.0,
            "rtfx": 0.0,
            "mean_ttft_ms": 500.0,
            "median_ttft_ms": 450.0,
            "std_ttft_ms": 120.0,
            "p90_ttft_ms": 700.0,
            "p99_ttft_ms": 900.0,
            "mean_tpot_ms": 9.0,
            "median_tpot_ms": 8.0,
            "std_tpot_ms": 1.5,
            "p90_tpot_ms": 10.0,
            "p99_tpot_ms": 12.5,
            "mean_itl_ms": 9.0,
            "median_itl_ms": 8.0,
            "std_itl_ms": 1.5,
            "p90_itl_ms": 10.0,
            "p99_itl_ms": 12.5,
            "completed": 8,
            "failed": 0,
            "total_input_tokens": 65536,
            "total_output_tokens": 8192,
            "duration": 16.0,
        },
        # 1000 / the median, P90 and P99 TPOT of 8, 10 and 12.5 ms.
        "derived": {"median_intvty": 125.0, "p90_intvty": 100.0, "p99_intvty": 80.0},
    },
    "config": {
        "code_hash": "0123abcd",
        "job_reference": "4242",
        "image": "latest",
        "created_by": "buildkite",
        "device_type": "tpu7x",
        "backend": "openai",
        "endpoint_type": "openai",
        "workload": {
            "input_len": 8192,
            "output_len": 1024,
            "num_prompts": 8,
            "request_rate": "inf",
            "burstiness": 1.0,
            "max_concurrency": 8,
        },
        "engine_flags": {"tokenizer_id": _MODEL, "label": _GKE_DECODE_CELL},
    },
}


def _parse_gke(capsys, tmp_path, *args, cell=_GKE_DECODE_CELL):
    """Runs the parser's command line over a copy of one GKE fixture cell."""
    path = tmp_path / f"{cell}.json"
    shutil.copy(_DATA / "gke" / path.name, path)
    returncode = parse_gke_results.main([str(path), "kube-run-1", "--dry-run", *args])
    return returncode, _rows(capsys.readouterr().out)


def test_gke_row_carries_the_cell(tmp_path, capsys):
    returncode, rows = _parse_gke(capsys, tmp_path)

    assert returncode == 0
    assert rows == [_GKE_ROW]


def test_gke_row_counts_both_pools_of_the_disaggregated_fleet(tmp_path, capsys):
    returncode, rows = _parse_gke(
        capsys, tmp_path, "--prefill-chips", "6", "--decode-chips=2"
    )

    assert returncode == 0
    assert rows == [
        {
            **_GKE_ROW,
            "metrics": {
                "vllm_bench": _GKE_ROW["metrics"]["vllm_bench"],
                "derived": {
                    # 512 output and 4608 total tok/s on 6 + 2 chips.
                    "num_chips": 8,
                    "output_tput_per_chip": 64.0,
                    "total_tput_per_chip": 576.0,
                    "input_tput_per_chip": 512.0,
                    "median_intvty": 125.0,
                    "p90_intvty": 100.0,
                    "p99_intvty": 80.0,
                },
            },
            "config": {**_GKE_ROW["config"], "prefill_chips": 6, "decode_chips": 2},
        }
    ]


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--prefill-chips", "4"], "--prefill-chips and --decode-chips go together"),
        (["--decode-chips", "4"], "--prefill-chips and --decode-chips go together"),
        (
            ["--prefill-chips", "four", "--decode-chips", "4"],
            "argument --prefill-chips: must be a positive integer, got 'four'",
        ),
        (
            ["--prefill-chips", "4", "--decode-chips=-4"],
            "argument --decode-chips: must be a positive integer, got '-4'",
        ),
        (
            ["--prefill-chips", "4", "--decode-chips", "0"],
            "argument --decode-chips: must be a positive integer, got '0'",
        ),
    ],
)
def test_gke_upload_rejects_a_chip_count_it_cannot_trust(tmp_path, capsys, args, error):
    path = tmp_path / f"{_GKE_DECODE_CELL}.json"
    shutil.copy(_DATA / "gke" / path.name, path)

    with pytest.raises(SystemExit) as exit_info:
        parse_gke_results.main([str(path), "kube-run-1", "--dry-run", *args])
    captured = capsys.readouterr()

    assert exit_info.value.code == 2
    assert error in captured.err
    assert _rows(captured.out) == []


def test_gke_manifest_states_the_chips_its_engines_request():
    manifest = yaml.safe_load(
        (
            _REPO / ".buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml"
        ).read_text()
    )
    containers = {
        job["name"]: job["template"]["spec"]["template"]["spec"]["containers"][0]
        for job in manifest["spec"]["replicatedJobs"]
    }

    assert containers["benchmark"]["command"][2:] == [
        "--prefill-chips=4",
        "--decode-chips=4",
    ]
    assert containers["prefill"]["resources"]["requests"] == {"google.com/tpu": "4"}
    assert containers["decode"]["resources"]["requests"] == {"google.com/tpu": "4"}


def test_gke_ttft_cell_without_tpot_has_no_derived_element(tmp_path, capsys):
    returncode, (row,) = _parse_gke(capsys, tmp_path, cell="pd_disagg_ttft_c1_i8192_o1")

    assert returncode == 0
    assert list(row["metrics"]) == ["vllm_bench"]


def test_gke_cell_with_zero_tpot_has_no_derived_element(tmp_path, capsys):
    # What the bench saves for TPOT when no request decoded a second token.
    cell = json.loads((_DATA / "gke" / f"{_GKE_DECODE_CELL}.json").read_text())
    for key in ("mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms"):
        cell[key] = 0.0
    path = tmp_path / "pd_disagg_prefill_c8_i8192_o1.json"
    path.write_text(json.dumps(cell) + "\n")

    returncode = parse_gke_results.main([str(path), "kube-run-1", "--dry-run"])
    (row,) = _rows(capsys.readouterr().out)

    assert returncode == 0
    assert list(row["metrics"]) == ["vllm_bench"]


@pytest.mark.parametrize(
    "bad_tpot",
    [None, 0, 0.0, -1.0, False, True, "8.0", float("nan"), float("inf")],
)
def test_interactivity_returns_none_when_tpot_is_not_a_positive_finite_number(bad_tpot):
    assert bench_metrics.interactivity(bad_tpot) is None


def test_derived_metrics_ignores_non_finite_values_and_invalid_chips():
    assert (
        bench_metrics.derived_metrics(
            chips=4,
            output_tput=float("nan"),
            total_tput=float("inf"),
            tpot_ms={"median": float("nan")},
        )
        == {}
    )
    assert (
        bench_metrics.derived_metrics(chips=-4, output_tput=100.0, total_tput=200.0)
        == {}
    )
    assert (
        bench_metrics.derived_metrics(chips=True, output_tput=100.0, total_tput=200.0)
        == {}
    )


def test_speculative_decoding_acceptance_is_kept_when_the_bench_reports_it(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "isl1024_osl8192_c256.json")
    cell_path = results_dir / "isl1024_osl8192_c256.json"
    cell = json.loads(cell_path.read_text())
    cell.update(
        spec_decode_acceptance_rate=62.5,
        spec_decode_acceptance_length=2.25,
        spec_decode_num_drafts=1000,
        spec_decode_draft_tokens=2000,
        spec_decode_accepted_tokens=1250,
    )
    cell_path.write_text(json.dumps(cell))

    _, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert row["metrics"]["vllm_bench"] == {
        **_BENCH_METRICS,
        "spec_decode_acceptance_rate": 62.5,
        "spec_decode_acceptance_length": 2.25,
        "spec_decode_num_drafts": 1000,
        "spec_decode_draft_tokens": 2000,
        "spec_decode_accepted_tokens": 1250,
    }
