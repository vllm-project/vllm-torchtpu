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
"""Rows the BigQuery uploaders write for each kind of run and bq_utils helpers.

Covers scripts/vllm/benchmarking/upload_results.py and bq_utils.py, and
.buildkite/scripts/parse_gke_results.py. Uploader tests run in dry-run mode
over fixture files shaped like the real artifacts
(tests/scripts/data/upload_results) and read the rows back out of the INSERT
statements they print; helper unit tests call bq_utils directly. Nothing
reaches BigQuery or Spanner: subprocess.run, the only way either CLI is
invoked, raises.
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


# upload_results.py and parse_gke_results.py import sibling bq_utils at load
# time; scope the sys.path change to loading so the rest of the session stays clean.
_BENCHMARKING_STR = str(_BENCHMARKING)
_added_path = _BENCHMARKING_STR not in sys.path
if _added_path:
    sys.path.insert(0, _BENCHMARKING_STR)
try:
    bq_utils = _load("bq_utils", _BENCHMARKING / "bq_utils.py")
    sys.modules.setdefault("bq_utils", bq_utils)
    upload_results = _load("upload_results", _BENCHMARKING / "upload_results.py")
    parse_gke_results = _load(
        "parse_gke_results", _REPO / ".buildkite" / "scripts" / "parse_gke_results.py"
    )
finally:
    if _added_path and _BENCHMARKING_STR in sys.path:
        sys.path.remove(_BENCHMARKING_STR)

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
        "AGENTX_CHIPS",
        "BQ_TABLE",
        "CI_RUNNER",
        "CREATED_BY",
        "DECODE_CHIPS",
        "DOCKER_IMAGE",
        "GCP_INSTANCE_NAME",
        "NUM_CHIPS",
        "PREFILL_CHIPS",
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
_BENCH_METRICS = {
    "request_throughput": 0.75,
    "output_throughput": 6168.0,
    "total_token_throughput": 6936.0,
    "max_output_tokens_per_s": 7000.0,
    "mean_ttft_ms": 1434.0,
    "median_ttft_ms": 1200.0,
    "std_ttft_ms": 300.0,
    "p90_ttft_ms": 1800.0,
    "p99_ttft_ms": 2500.0,
    "mean_tpot_ms": 36.0,
    "median_tpot_ms": 32.0,
    "std_tpot_ms": 4.0,
    "p90_tpot_ms": 40.0,
    "p99_tpot_ms": 50.0,
    "mean_itl_ms": 36.0,
    "median_itl_ms": 32.0,
    "std_itl_ms": 4.0,
    "p90_itl_ms": 40.0,
    "p99_itl_ms": 50.0,
    "completed": 2560,
    "failed": 0,
    "total_input_tokens": 2621440,
    "total_output_tokens": 20971520,
    "duration": 3400.0,
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
        "backend": "openai-chat",
        "engine_flags": _ENGINE_FLAGS,
        "workload": {
            "input_len": 1024,
            "output_len": 8192,
            "num_prompts": 2560,
            "request_rate": None,
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


_GKE_ROW = {
    "table": _TABLE,
    "record_id": "kube-run-1_inf_c64_000000",
    "created_time": "TIMESTAMP '2026-10-20T10:48:45Z'",
    "repository": "vllm-torchtpu",
    "run_type": "GKE_DISAGG",
    "model_id": _MODEL,
    "metrics": {
        "vllm_bench": {
            "request_throughput": 6.4,
            "output_throughput": 6400.0,
            "total_token_throughput": 12800.0,
            "mean_ttft_ms": 500.0,
            "median_ttft_ms": 450.0,
            "p90_ttft_ms": 700.0,
            "p99_ttft_ms": 900.0,
            "mean_tpot_ms": 9.0,
            "median_tpot_ms": 8.0,
            "p90_tpot_ms": 10.0,
            "p99_tpot_ms": 12.5,
            "completed": 640,
            "failed": 0,
            "total_input_tokens": 655360,
            "total_output_tokens": 655360,
            "duration": 100.0,
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
            "input_len": 1024,
            "output_len": 1024,
            "num_prompts": 640,
            "request_rate": None,
            "burstiness": 1.0,
            "max_concurrency": 64,
        },
        "engine_flags": {"tokenizer_id": _MODEL, "label": None},
    },
}


def _parse_gke(capsys, directory: pathlib.Path) -> tuple[bool, list[dict]]:
    ok = parse_gke_results.parse_and_upload(
        str(directory / "pd_disagg_c64_i1024_o1024.json"), "kube-run-1", dry_run=True
    )
    return ok, _rows(capsys.readouterr().out)


def test_gke_row_carries_the_cell(tmp_path, capsys):
    shutil.copytree(_DATA / "gke", tmp_path / "gke")

    ok, rows = _parse_gke(capsys, tmp_path / "gke")

    assert ok
    assert rows == [_GKE_ROW]


def test_created_time_sql_converts_an_offset_to_utc():
    assert (
        bq_utils.created_time_sql("2026-11-02T09:18:05+08:00")
        == "TIMESTAMP '2026-11-02T01:18:05Z'"
    )
    assert (
        bq_utils.created_time_sql("2026-11-02T01:18:05")
        == "TIMESTAMP '2026-11-02T01:18:05Z'"
    )


@pytest.mark.parametrize(
    "bad_tpot",
    [None, 0, 0.0, -1.0, False, True, "8.0", float("nan"), float("inf")],
)
def test_interactivity_returns_none_when_tpot_is_not_a_positive_finite_number(bad_tpot):
    assert bq_utils.interactivity(bad_tpot) is None


def test_derived_metrics_ignores_non_finite_values_and_invalid_chips():
    assert (
        bq_utils.derived_metrics(
            chips=4,
            output_tput=float("nan"),
            total_tput=float("inf"),
            tpot_ms={"median": float("nan")},
        )
        == {}
    )
    assert bq_utils.derived_metrics(chips=-4, output_tput=100.0, total_tput=200.0) == {}
    assert (
        bq_utils.derived_metrics(chips=True, output_tput=100.0, total_tput=200.0) == {}
    )


@pytest.mark.parametrize("tpot_mode", ["omitted", "zero"])
def test_gke_prefill_cell_without_positive_tpot_omits_derived_when_chips_unset(
    tmp_path, capsys, tpot_mode
):
    gke_dir = tmp_path / "gke"
    gke_dir.mkdir()
    cell = json.loads((_DATA / "gke" / "pd_disagg_c64_i1024_o1024.json").read_text())
    for key in ("mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms"):
        if tpot_mode == "omitted":
            cell.pop(key, None)
        else:
            cell[key] = 0.0
    path = gke_dir / "pd_disagg_prefill_c64_i8192_o1.json"
    path.write_text(json.dumps(cell) + "\n")

    ok = parse_gke_results.parse_and_upload(str(path), "kube-run-1", dry_run=True)
    (row,) = _rows(capsys.readouterr().out)

    assert ok
    assert list(row["metrics"]) == ["vllm_bench"]


def test_malformed_routing_metrics_json_warns_and_is_omitted(tmp_path, capsys):
    shutil.copytree(_DATA / "gke", tmp_path / "gke")
    (tmp_path / "gke" / "routing_metrics.json").write_text("{not valid json")

    ok = parse_gke_results.parse_and_upload(
        str(tmp_path / "gke" / "pd_disagg_c64_i1024_o1024.json"),
        "kube-run-1",
        dry_run=True,
    )
    captured = capsys.readouterr()
    (row,) = _rows(captured.out)

    assert ok
    assert "routing" not in row["metrics"]
    assert "Warning: Failed to load" in captured.err


def test_non_object_routing_metrics_warns_and_is_omitted(capsys):
    assert bq_utils.extract_routing_metrics([0.93, 0.74]) == {}
    assert "expected a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("env", "num_chips"),
    [
        ({"AGENTX_CHIPS": "8"}, 8),
        ({"NUM_CHIPS": "4", "AGENTX_CHIPS": "8"}, 4),
        ({"PREFILL_CHIPS": "16", "DECODE_CHIPS": "8", "NUM_CHIPS": "4"}, 24),
    ],
)
def test_chip_count_falls_back_to_the_launcher_environment(monkeypatch, env, num_chips):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert bq_utils.chips_from_env() == num_chips


@pytest.mark.parametrize(
    ("routing_file", "expected"),
    [
        # A percentage where a fraction belongs would read as a pass.
        (
            {"prefix_cache_hit_rate": 93, "batch_utilization": 0.74},
            {"batch_utilization": 0.74},
        ),
        ({"prefix_cache_hit_rate": "0.93"}, None),
        ({"prefix_cache_hit_rate": True, "batch_utilization": -0.1}, None),
        ({"unrelated": 1.0}, None),
        ([0.93, 0.74], None),
    ],
)
def test_routing_metrics_outside_the_contract_are_left_out(
    tmp_path, capsys, routing_file, expected
):
    shutil.copytree(_DATA / "gke", tmp_path / "gke")
    (tmp_path / "gke" / "routing_metrics.json").write_text(json.dumps(routing_file))

    ok, (row,) = _parse_gke(capsys, tmp_path / "gke")

    assert ok
    assert row["metrics"].get("routing") == expected


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


def test_gke_row_counts_both_pools_of_the_disaggregated_fleet(
    tmp_path, monkeypatch, capsys
):
    shutil.copytree(_DATA / "gke", tmp_path / "gke")
    monkeypatch.setenv("PREFILL_CHIPS", "16")
    monkeypatch.setenv("DECODE_CHIPS", "16")
    (tmp_path / "gke" / "routing_metrics.json").write_text(
        json.dumps({"prefix_cache_hit_rate": 0.93, "batch_utilization": 0.74})
    )

    ok, (row,) = _parse_gke(capsys, tmp_path / "gke")

    assert ok
    assert row["metrics"] == {
        "vllm_bench": _GKE_ROW["metrics"]["vllm_bench"],
        "derived": {
            # 6400 output and 12800 total tok/s on 16 + 16 chips.
            "num_chips": 32,
            "output_tput_per_chip": 200.0,
            "total_tput_per_chip": 400.0,
            "input_tput_per_chip": 200.0,
            "median_intvty": 125.0,
            "p90_intvty": 100.0,
            "p99_intvty": 80.0,
        },
        "routing": {"prefix_cache_hit_rate": 0.93, "batch_utilization": 0.74},
    }
    assert row["config"] == _GKE_ROW["config"]


@pytest.mark.parametrize(
    "env",
    [{"PREFILL_CHIPS": "16"}, {"NUM_CHIPS": "sixteen"}, {"NUM_CHIPS": "-4"}],
)
def test_gke_upload_fails_on_a_chip_count_it_cannot_trust(
    tmp_path, monkeypatch, capsys, env
):
    shutil.copytree(_DATA / "gke", tmp_path / "gke")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    ok, rows = _parse_gke(capsys, tmp_path / "gke")

    assert (ok, rows) == (False, [])
