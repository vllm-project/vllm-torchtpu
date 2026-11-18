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
"""Rows the BigQuery uploaders write for each kind of run.

Covers scripts/vllm/benchmarking/upload_results.py and bq_utils.py, and
.buildkite/scripts/parse_gke_results.py. Every test runs the uploader in its
dry-run mode over fixture files shaped like the real artifacts
(tests/scripts/data/upload_results) and reads the rows back out of the INSERT
statements it prints. Nothing reaches BigQuery or Spanner: subprocess.run,
the only way either CLI is invoked, raises.
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

# upload_results.py is run as a script, so it imports its sibling bq_utils.
sys.path.insert(0, str(_BENCHMARKING))


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upload_results = _load("upload_results", _BENCHMARKING / "upload_results.py")
parse_gke_results = _load(
    "parse_gke_results", _REPO / ".buildkite" / "scripts" / "parse_gke_results.py"
)

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
    "metrics": {
        "vllm_bench": _BENCH_METRICS,
        "lm_eval": {"gsm8k": 0.98},
        # 1000 / the median, P90 and P99 TPOT of 32, 40 and 50 ms.
        "derived": {"median_intvty": 31.25, "p90_intvty": 25.0, "p99_intvty": 20.0},
    },
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


def test_speed_grid_row_divides_throughput_by_the_stated_chips(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "isl1024_osl8192_c256.json")

    returncode, rows = _upload(monkeypatch, capsys, results_dir, "--chips", 4)

    assert returncode == 0
    (row,) = rows
    # 6168 output and 6936 total tok/s on 4 chips, not on TP x DP = 8 ranks.
    assert row["metrics"] == {
        "vllm_bench": _BENCH_METRICS,
        "derived": {
            "num_chips": 4,
            "output_tput_per_chip": 1542.0,
            "total_tput_per_chip": 1734.0,
            "input_tput_per_chip": 192.0,
            "median_intvty": 31.25,
            "p90_intvty": 25.0,
            "p99_intvty": 20.0,
        },
    }
    assert row["config"] == _SPEED_GRID_ROW["config"]


_ACCURACY_CONFIG = {**_BASE_CONFIG, "engine_flags": _ENGINE_FLAGS}


@pytest.mark.parametrize(
    ("fixture", "scores"),
    [
        ("lm_eval", {"gsm8k": 0.98}),
        ("evalscope", {"mmmu_pro": 0.61}),
        ("json_schema", {"json_schema": 0.95}),
    ],
)
def test_accuracy_only_run_writes_one_accuracy_row(
    tmp_path, monkeypatch, capsys, fixture, scores
):
    results_dir = _results_dir(tmp_path, "config.json", fixture)

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [
        {
            "table": _TABLE,
            "record_id": _record_id(1),
            "created_time": "CURRENT_TIMESTAMP()",
            "repository": "vllm-torchtpu",
            "run_type": "DAILY",
            "model_id": _MODEL,
            "metrics": {"lm_eval": scores},
            "config": _ACCURACY_CONFIG,
        }
    ]


def test_accuracy_only_run_merges_every_suite_into_the_one_row(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(
        tmp_path, "config.json", "lm_eval", "evalscope", "json_schema"
    )

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    (row,) = rows
    assert row["metrics"] == {
        "lm_eval": {"gsm8k": 0.98, "mmmu_pro": 0.61, "json_schema": 0.95}
    }


def test_accuracy_row_statement_is_the_documented_insert(tmp_path, monkeypatch, capsys):
    results_dir = _results_dir(tmp_path, "evalscope")
    (results_dir / "config.json").write_text(json.dumps({"model": "org/it's"}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["upload_results.py", "--results-dir", str(results_dir), "--skip-db-upload"],
    )

    assert upload_results.main() == 0

    statements = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("INSERT INTO `")
    ]
    assert statements == [
        "INSERT INTO `" + _TABLE + "` ( record_id, created_time, repository, "
        "run_type, model_id, metrics, config ) VALUES ( "
        "'00000000-0000-0000-0000-000000000001', CURRENT_TIMESTAMP(), "
        "'vllm-torchtpu', 'DAILY', 'org/it''s', "
        """[STRUCT('lm_eval', PARSE_JSON('{"mmmu_pro": 0.61}', """
        "wide_number_mode=>'round'))], "
        """PARSE_JSON('{"code_hash": "0123abcd", "created_by": "buildkite-agent", """
        """"device_type": "tpu7x-2x2x1", "run_by": "kube-agent-1", """
        """"job_reference": "4242", "profile_gcs_dir": null, "engine_flags": """
        """{"max_num_seqs": null, "max_num_batched_tokens": null, """
        """"tensor_parallel_size": null, "data_parallel_size": 1, """
        """"max_model_len": null, "enable_ep": false, "quantization": null, """
        """"multihost_backend": null}}', wide_number_mode=>'round') );"""
    ]


def test_run_with_no_results_uploads_nothing(tmp_path, monkeypatch, capsys):
    results_dir = _results_dir(tmp_path, "config.json")
    monkeypatch.delenv("BUILDKITE_COMMIT")

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert (returncode, rows) == (0, [])


_AGENTX_METRICS = {
    "request_throughput_avg": 0.25,
    "request_count_avg": 500.0,
    "output_token_throughput_avg": 1280.0,
    "input_token_throughput_avg": 11520.0,
    "total_token_throughput_avg": 12800.0,
    "time_to_first_token_avg": 1800.0,
    "time_to_first_token_p50": 900.0,
    "time_to_first_token_p90": 3800.0,
    "time_to_first_token_p99": 25000.0,
    "inter_token_latency_avg": 20.0,
    "inter_token_latency_p50": 12.5,
    "inter_token_latency_p90": 25.0,
    "inter_token_latency_p99": 80.0,
    "request_latency_avg": 9000.0,
    "request_latency_p50": 4000.0,
    "request_latency_p90": 22000.0,
    "request_latency_p99": 58000.0,
    "effective_concurrency_avg": 14.5,
    "benchmark_duration_avg": 1800.0,
    "theoretical_prefix_cache_hit_avg": 91.5,
    "aiperf_version": "0.12.0",
    "was_cancelled": False,
    "scenario": "agentx",
    "submission_valid": True,
}
_AGENTX_CONFIG = {
    **_BASE_CONFIG,
    "engine_flags": _ENGINE_FLAGS,
    "workload": {
        "scenario": "agentx",
        "dataset": "semianalysis_cc_traces_weka_062126_256k",
        "max_context_length": 262144,
        "max_concurrency": 16,
    },
}
# The fixture's two engines: the prefill engine answered 18.4M of 20M queried
# prompt tokens from cache, and 4 and 12 requests ran on average against 16
# batch slots each.
_AGENTX_ROUTING = {"prefix_cache_hit_rate": 0.92, "batch_utilization": 0.5}


def test_created_time_sql_converts_an_offset_to_utc():
    assert (
        upload_results.bq_utils.created_time_sql("2026-11-02T09:18:05+08:00")
        == "TIMESTAMP '2026-11-02T01:18:05Z'"
    )
    assert (
        upload_results.bq_utils.created_time_sql("2026-11-02T01:18:05")
        == "TIMESTAMP '2026-11-02T01:18:05Z'"
    )


def test_agentx_point_row_has_per_chip_throughput_and_p90_interactivity(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "agentx")

    returncode, rows = _upload(
        monkeypatch, capsys, results_dir, "--prefill-chips", 16, "--decode-chips", 16
    )

    assert returncode == 0
    assert rows == [
        {
            "table": _TABLE,
            "record_id": _record_id(1),
            "created_time": "TIMESTAMP '2026-10-28T08:35:19Z'",
            "repository": "vllm-torchtpu",
            "run_type": "DAILY",
            "model_id": _MODEL,
            "metrics": {
                "agentx": _AGENTX_METRICS,
                "derived": {
                    # 16 prefill + 16 decode chips, not either pool alone.
                    "num_chips": 32,
                    "output_tput_per_chip": 40.0,
                    "total_tput_per_chip": 400.0,
                    "input_tput_per_chip": 360.0,
                    "median_intvty": 80.0,
                    "p90_intvty": 40.0,
                    "p99_intvty": 12.5,
                },
                "routing": _AGENTX_ROUTING,
            },
            "config": _AGENTX_CONFIG,
        }
    ]


@pytest.mark.parametrize(
    ("env", "num_chips"),
    [
        ({"AGENTX_CHIPS": "8"}, 8),
        ({"NUM_CHIPS": "4", "AGENTX_CHIPS": "8"}, 4),
        ({"PREFILL_CHIPS": "16", "DECODE_CHIPS": "8", "NUM_CHIPS": "4"}, 24),
    ],
)
def test_chip_count_falls_back_to_the_launcher_environment(
    tmp_path, monkeypatch, capsys, env, num_chips
):
    results_dir = _results_dir(tmp_path, "config.json", "agentx")
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    _, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert row["metrics"]["derived"]["num_chips"] == num_chips
    assert row["metrics"]["derived"]["output_tput_per_chip"] == 1280.0 / num_chips


def test_agentx_row_without_a_chip_count_has_no_per_chip_throughput(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "agentx")

    _, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert row["metrics"]["derived"] == {
        "median_intvty": 80.0,
        "p90_intvty": 40.0,
        "p99_intvty": 12.5,
    }


@pytest.mark.parametrize(
    "args",
    [
        ("--prefill-chips", "16"),
        ("--decode-chips", "16"),
        ("--chips", "0"),
        ("--chips", "four"),
    ],
)
def test_bad_chip_arguments_are_rejected_before_any_row(
    tmp_path, monkeypatch, capsys, args
):
    results_dir = _results_dir(tmp_path, "config.json", "agentx")

    with pytest.raises(SystemExit) as exit_info:
        _upload(monkeypatch, capsys, results_dir, *args)

    assert exit_info.value.code == 2
    assert _rows(capsys.readouterr().out) == []


def test_agentx_run_with_accuracy_writes_a_point_row_and_an_accuracy_row(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "agentx", "lm_eval")
    (results_dir / "agentx" / "conc_16" / "aiperf_artifacts").joinpath(
        "server_metrics_export.json"
    ).unlink()

    returncode, rows = _upload(monkeypatch, capsys, results_dir, "--chips", 32)

    assert returncode == 0
    assert [(row["record_id"], sorted(row["metrics"])) for row in rows] == [
        (_record_id(1), ["agentx", "derived"]),
        (_record_id(2), ["lm_eval"]),
    ]
    assert rows[1]["metrics"] == {"lm_eval": {"gsm8k": 0.98}}


def test_unreadable_agentx_export_fails_the_upload_but_keeps_other_points(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "agentx")
    broken = results_dir / "agentx" / "conc_32" / "aiperf_artifacts"
    broken.mkdir(parents=True)
    (broken / "profile_export_aiperf.json").write_text('{"request_count": {"av')

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert [row["config"]["workload"]["max_concurrency"] for row in rows] == [16]


def test_agentx_export_outside_conc_layout_is_ignored(tmp_path, monkeypatch, capsys):
    results_dir = _results_dir(tmp_path, "config.json", "agentx")
    stray = results_dir / "wrong_location"
    stray.mkdir()
    (stray / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "end_time": "2026-10-28T08:35:19+00:00",
                "input_config": {"models": {"items": [{"name": _MODEL}]}},
                "request_count": {"avg": 999.0},
            }
        )
    )

    _, rows = _upload(monkeypatch, capsys, results_dir)

    assert [row["config"]["workload"]["max_concurrency"] for row in rows] == [16]


def test_run_level_routing_metrics_ride_every_row(tmp_path, monkeypatch, capsys):
    results_dir = _results_dir(
        tmp_path, "config.json", "isl1024_osl8192_c256.json", "lm_eval"
    )
    (results_dir / "routing_metrics.json").write_text(
        json.dumps({"prefix_cache_hit_rate": 0.93, "batch_utilization": 0.74})
    )

    _, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert row["metrics"]["routing"] == {
        "prefix_cache_hit_rate": 0.93,
        "batch_utilization": 0.74,
    }
    assert row["metrics"]["vllm_bench"] == _BENCH_METRICS


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
    tmp_path, monkeypatch, capsys, routing_file, expected
):
    results_dir = _results_dir(tmp_path, "config.json", "evalscope")
    (results_dir / "routing_metrics.json").write_text(json.dumps(routing_file))

    _, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert row["metrics"].get("routing") == expected
    assert row["metrics"]["lm_eval"] == {"mmmu_pro": 0.61}


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
