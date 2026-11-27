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
call bench_metrics directly. Nothing reaches BigQuery or Spanner:
subprocess.run, the only way either CLI is invoked, raises.

The fixture cells come from rows that two nightly builds wrote to the
benchmark_runs table: the speed-grid cell from build 10629 and the two GKE
cells from build 10602. Every field an uploader reads carries the row's
value, and `date` is the row's created_time. The rest of each file follows
what the pinned `vllm bench serve` (v0.33.0) saves for the lane's flags:
request_goodput is null without --goodput, endpoint_type repeats the backend,
a P50 repeats the median, and a speed-grid cell has a null label and the
model as its tokenizer. The keys the writer also saves but the rows do not
hold (latencies, queue_times, max_concurrent_requests and the *_e2el_ms
statistics) are left out; no uploader reads them.

results_mmmu_pro.json is the file run_eval_flow.sh writes for an evalscope
score of 0.61.

The AgentX point (conc_8/) is not a measurement. Its shape follows the public
source of the writer agentx_client.sh pins (aiperf 0.12.0: JsonExportData,
JsonMetricResult and the config it dumps as input_config). Its values are
invented, and it was not checked against a real export. It holds only the
keys and statistics the uploader reads, and the unit every metric block must
carry.
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
_GRID_MODEL = "deepseek-ai/DeepSeek-V2"
_GKE_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
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


def _run_upload(monkeypatch, capsys, results_dir, *args) -> tuple[int, str, str]:
    """The uploader's dry run over a directory: exit status, stdout, stderr."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["upload_results.py", "--results-dir", str(results_dir), "--skip-db-upload"]
        + [str(a) for a in args],
    )
    returncode = upload_results.main()
    captured = capsys.readouterr()
    return returncode, captured.out, captured.err


def _upload(monkeypatch, capsys, results_dir, *args) -> tuple[int, list[dict]]:
    returncode, out, _ = _run_upload(monkeypatch, capsys, results_dir, *args)
    return returncode, _rows(out)


_BASE_CONFIG = {
    "code_hash": "0123abcd",
    "created_by": "buildkite-agent",
    "device_type": "tpu7x-2x2x1",
    "run_by": "kube-agent-1",
    "job_reference": "4242",
    "profile_gcs_dir": None,
}
_ENGINE_FLAGS = {
    "max_num_seqs": 128,
    "max_num_batched_tokens": 8192,
    "tensor_parallel_size": 8,
    "data_parallel_size": 1,
    "max_model_len": 16384,
    "enable_ep": True,
    "quantization": "none",
    "multihost_backend": None,
}
# No p90_* keys: run_benchmarks.sh leaves --metric-percentiles at its default
# of 99.
_BENCH_METRICS = {
    "request_throughput": 1.7395167641836704,
    "output_throughput": 1784.700712133341,
    "total_token_throughput": 3562.5955649268135,
    "max_output_tokens_per_s": 2432.0,
    "rtfx": 0.0,
    "mean_ttft_ms": 699.213593406273,
    "median_ttft_ms": 205.83673850023843,
    "std_ttft_ms": 1183.3377321032601,
    "p99_ttft_ms": 4537.59701739953,
    "mean_tpot_ms": 32.10634284340862,
    "median_tpot_ms": 32.83010131278941,
    "std_tpot_ms": 2.5744545163437387,
    "p99_tpot_ms": 36.14999171315232,
    "mean_itl_ms": 31.92212706080333,
    "median_itl_ms": 27.675446999637643,
    "std_itl_ms": 25.53811036050862,
    "p99_itl_ms": 158.76247312935448,
    "completed": 320,
    "failed": 0,
    "total_input_tokens": 327060,
    "total_output_tokens": 328312,
    "duration": 183.95913542699964,
}
_SPEED_GRID_ROW = {
    "table": _TABLE,
    "record_id": _record_id(1),
    "created_time": "TIMESTAMP '2026-11-21T22:45:57Z'",
    "repository": "vllm-torchtpu",
    "run_type": "DAILY",
    "model_id": _GRID_MODEL,
    "metrics": {"vllm_bench": _BENCH_METRICS, "lm_eval": {"gsm8k": 0.98}},
    "config": {
        **_BASE_CONFIG,
        "backend": "vllm",
        "engine_flags": _ENGINE_FLAGS,
        "workload": {
            "input_len": 1024,
            "output_len": 1024,
            "num_prompts": 320,
            # What the bench saves for an unlimited rate: a string, not null.
            "request_rate": "inf",
            "burstiness": 1.0,
            "max_concurrency": 64,
        },
    },
}


def test_speed_grid_row_carries_the_cell_and_the_run_accuracy(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(
        tmp_path, "config.json", "isl1024_osl1024_c64.json", "lm_eval"
    )

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [_SPEED_GRID_ROW]


_GKE_DECODE_CELL = "pd_disagg_decode_daily_round3_c256_i65536_o1024"
_GKE_ROW = {
    "table": _TABLE,
    "record_id": f"kube-run-1_inf_c256_{'0' * 6}",
    "created_time": "TIMESTAMP '2026-11-21T15:25:05Z'",
    "repository": "vllm-torchtpu",
    "run_type": "GKE_DISAGG",
    "model_id": _GKE_MODEL,
    "metrics": {
        # disagg_benchmark.sh asks for percentiles 50, 90 and 99; the P50
        # keys are not BENCH_METRIC_FIELDS.
        "vllm_bench": {
            "request_throughput": 0.5976054708154231,
            "output_throughput": 611.9480021149933,
            "total_token_throughput": 39776.62013747456,
            "max_output_tokens_per_s": 704.0,
            "rtfx": 0.0,
            "mean_ttft_ms": 203565.2416025781,
            "median_ttft_ms": 203331.61300500022,
            "std_ttft_ms": 116100.16813890262,
            "p90_ttft_ms": 363772.5042280001,
            "p99_ttft_ms": 399683.54623475013,
            "mean_tpot_ms": 24.291133922172836,
            "median_tpot_ms": 24.38053086070386,
            "std_tpot_ms": 0.573331313217146,
            "p90_tpot_ms": 24.70020041642207,
            "p99_tpot_ms": 24.997773420381314,
            "mean_itl_ms": 24.445805767222204,
            "median_itl_ms": 24.102490000132093,
            "std_itl_ms": 18.614944635309854,
            "p90_itl_ms": 25.8508530000654,
            "p99_itl_ms": 31.339102500123776,
            "completed": 256,
            "failed": 0,
            "total_input_tokens": 16777216,
            "total_output_tokens": 262144,
            "duration": 428.37626578399977,
        },
        # 1000 / the median, P90 and P99 TPOT above.
        "derived": {
            "median_intvty": 41.01633412797354,
            "p90_intvty": 40.4855014591357,
            "p99_intvty": 40.00356284470819,
        },
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
            "input_len": 65536,
            "output_len": 1024,
            "num_prompts": 256,
            "request_rate": "inf",
            "burstiness": 1.0,
            "max_concurrency": 256,
        },
        "engine_flags": {"tokenizer_id": _GKE_MODEL, "label": _GKE_DECODE_CELL},
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
                    # Output over decode chips (2), input over prefill chips
                    # (6), and total over both pools (6 + 2 = 8).
                    "num_chips": 8,
                    "output_tput_per_chip": 305.97400105749665,
                    "total_tput_per_chip": 4972.07751718432,
                    "input_tput_per_chip": 6527.445355893261,
                    **_GKE_ROW["metrics"]["derived"],
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


def test_gke_manifest_states_the_chips_its_engines_hold():
    manifest = yaml.safe_load(
        (
            _REPO / ".buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml"
        ).read_text()
    )
    jobs = {job["name"]: job for job in manifest["spec"]["replicatedJobs"]}

    def chips(role: str) -> int:
        """Chips the role's pods hold at once: Jobs x pods per Job x chips per
        pod. `completions` counts pods over the Job's life, not at once."""
        job_spec = jobs[role]["template"]["spec"]
        per_pod = sum(
            int(container["resources"]["requests"]["google.com/tpu"])
            for container in job_spec["template"]["spec"]["containers"]
        )
        return jobs[role]["replicas"] * job_spec["parallelism"] * per_pod

    pod_spec = jobs["benchmark"]["template"]["spec"]["template"]["spec"]
    assert pod_spec["containers"][0]["command"][2:] == [
        f"--prefill-chips={chips('prefill')}",
        f"--decode-chips={chips('decode')}",
    ]


_GKE_TTFT_CELL = "pd_disagg_ttft_c1_i65536_o1"
# --percentile-metrics=ttft,e2el: the cell has no TPOT or ITL keys.
_GKE_TTFT_BENCH = {
    "request_throughput": 0.4628023936055456,
    "output_throughput": 0.4628023936055456,
    "total_token_throughput": 30330.68046972664,
    "max_output_tokens_per_s": 1.0,
    "rtfx": 0.0,
    "mean_ttft_ms": 2160.1349558124953,
    "median_ttft_ms": 2165.3985120000243,
    "std_ttft_ms": 40.305854461149764,
    "p90_ttft_ms": 2205.0965189998806,
    "p99_ttft_ms": 2213.0933955001296,
    "completed": 16,
    "failed": 0,
    "total_input_tokens": 1048576,
    "total_output_tokens": 16,
    "duration": 34.57199059699997,
}
_GKE_TTFT_ROW = {
    **_GKE_ROW,
    "record_id": f"kube-run-1_inf_c1_{'0' * 6}",
    "created_time": "TIMESTAMP '2026-11-21T14:48:38Z'",
    "metrics": {"vllm_bench": _GKE_TTFT_BENCH},
    "config": {
        **_GKE_ROW["config"],
        "workload": {
            "input_len": 65536,
            "output_len": 1,
            "num_prompts": 16,
            "request_rate": "inf",
            "burstiness": 1.0,
            "max_concurrency": 1,
        },
        "engine_flags": {"tokenizer_id": _GKE_MODEL, "label": _GKE_TTFT_CELL},
    },
}


def test_gke_ttft_cell_without_tpot_has_no_derived_element(tmp_path, capsys):
    returncode, rows = _parse_gke(capsys, tmp_path, cell=_GKE_TTFT_CELL)

    assert returncode == 0
    assert rows == [_GKE_TTFT_ROW]


def test_gke_ttft_cell_with_the_manifest_chips_derives_only_per_chip_throughput(
    tmp_path, capsys
):
    returncode, rows = _parse_gke(
        capsys, tmp_path, "--prefill-chips=4", "--decode-chips=4", cell=_GKE_TTFT_CELL
    )

    assert returncode == 0
    assert rows == [
        {
            **_GKE_TTFT_ROW,
            "metrics": {
                "vllm_bench": _GKE_TTFT_BENCH,
                # Output over decode chips (4), input over prefill chips (4),
                # and total over both pools (4 + 4 = 8). No *_intvty: the cell
                # has no TPOT.
                "derived": {
                    "num_chips": 8,
                    "output_tput_per_chip": 0.1157005984013864,
                    "total_tput_per_chip": 3791.33505871583,
                    "input_tput_per_chip": 7582.554416833259,
                },
            },
            "config": {
                **_GKE_TTFT_ROW["config"],
                "prefill_chips": 4,
                "decode_chips": 4,
            },
        }
    ]


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
    assert (
        bench_metrics.derived_metrics(
            prefill_chips=4, output_tput=100.0, total_tput=200.0
        )
        == {}
    )
    assert (
        bench_metrics.derived_metrics(
            prefill_chips=4, decode_chips=0, output_tput=100.0, total_tput=200.0
        )
        == {}
    )
    assert (
        bench_metrics.derived_metrics(
            chips=10,
            prefill_chips=4,
            decode_chips=4,
            output_tput=100.0,
            total_tput=200.0,
        )
        == {}
    )


def test_speculative_decoding_acceptance_is_kept_when_the_bench_reports_it(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "isl1024_osl1024_c64.json")
    cell_path = results_dir / "isl1024_osl1024_c64.json"
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


_ACCURACY_ROW = {
    "table": _TABLE,
    "record_id": _record_id(1),
    # The suites' result files share no measurement time.
    "created_time": "CURRENT_TIMESTAMP()",
    "repository": "vllm-torchtpu",
    "run_type": "DAILY",
    "model_id": _GRID_MODEL,
    "metrics": {"lm_eval": {"gsm8k": 0.98}},
    "config": {**_BASE_CONFIG, "engine_flags": _ENGINE_FLAGS},
}


@pytest.mark.parametrize(
    ("results", "scores"),
    [
        (["lm_eval"], {"gsm8k": 0.98}),
        (["results_mmmu_pro.json"], {"mmmu_pro": 0.61}),
        (["lm_eval", "results_mmmu_pro.json"], {"gsm8k": 0.98, "mmmu_pro": 0.61}),
    ],
)
def test_run_without_a_speed_grid_writes_its_scores_as_one_accuracy_row(
    tmp_path, monkeypatch, capsys, results, scores
):
    results_dir = _results_dir(tmp_path, "config.json", *results)

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [{**_ACCURACY_ROW, "metrics": {"lm_eval": scores}}]


def test_accuracy_row_without_any_model_source_fails_before_writing(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "results_mmmu_pro.json")

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert "Error: Missing required config parameter 'model' in config.json" in err
    assert _rows(out) == []


def test_configless_lm_eval_run_uses_results_model_name_without_engine_flags(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "lm_eval")

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [
        {
            **_ACCURACY_ROW,
            "model_id": _GKE_MODEL,
            "config": _BASE_CONFIG,
        }
    ]


def test_directory_with_nothing_to_upload_is_not_an_error(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json")
    # An empty directory is reported before the commit is asked for.
    monkeypatch.delenv("BUILDKITE_COMMIT")

    returncode, out, _ = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert out == (
        f"Nothing to upload from {results_dir}: no speed-grid cell, AgentX point "
        "or accuracy result.\n"
    )


def test_corrupt_only_accuracy_results_exits_nonzero(tmp_path, monkeypatch, capsys):
    results_dir = _results_dir(tmp_path)
    broken = results_dir / "results_gsm8k.json"
    broken.write_text('{"results": {"gsm8k":')
    monkeypatch.delenv("BUILDKITE_COMMIT")

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert str(broken) in err
    assert _rows(out) == []


def test_non_cell_isl_backup_file_is_ignored(tmp_path, monkeypatch, capsys):
    only_bak = tmp_path / "only_bak"
    only_bak.mkdir()
    empty_dir = _results_dir(only_bak, "config.json")
    shutil.copy(
        _DATA / "isl1024_osl1024_c64.json",
        empty_dir / "isl1024_osl1024_c64_bak.json",
    )
    returncode, out, _ = _run_upload(monkeypatch, capsys, empty_dir)
    assert returncode == 0
    assert out.startswith(f"Nothing to upload from {empty_dir}:")

    with_eval = tmp_path / "with_eval"
    with_eval.mkdir()
    acc_dir = _results_dir(with_eval, "config.json", "lm_eval")
    shutil.copy(
        _DATA / "isl1024_osl1024_c64.json",
        acc_dir / "isl1024_osl1024_c64_bak.json",
    )
    returncode, rows = _upload(monkeypatch, capsys, acc_dir)
    assert returncode == 0
    assert rows == [_ACCURACY_ROW]


def test_corrupt_speed_grid_cell_does_not_emit_orphan_accuracy_row(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "lm_eval")
    broken_cell = results_dir / "isl1024_osl1024_c64.json"
    broken_cell.write_text('{"backend":')

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert f"Error loading {broken_cell}: " in err
    assert _rows(out) == []


_AGENTX_MODEL = "moonshotai/Kimi-K3"
_AGENTX_EXPORT = "conc_8/aiperf_artifacts/profile_export_aiperf.json"
_AGENTX_METRICS = {
    "request_throughput_avg": 0.25,
    "request_count_avg": 300.0,
    "output_token_throughput_avg": 640.0,
    "input_token_throughput_avg": 12160.0,
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
    "effective_concurrency_avg": 7.25,
    "benchmark_duration_avg": 1200.0,
    "theoretical_prefix_cache_hit_avg": 91.5,
    "aiperf_version": "0.12.0",
    "was_cancelled": False,
    "scenario": "agentx",
    "submission_valid": True,
}
_AGENTX_ROW = {
    "table": _TABLE,
    "record_id": _record_id(1),
    # The export's end_time, 2026-10-28T08:35:19.586894, which has no offset.
    "created_time": "TIMESTAMP '2026-10-28T08:35:19Z'",
    "repository": "vllm-torchtpu",
    "run_type": "DAILY",
    "model_id": _AGENTX_MODEL,
    "metrics": {
        "agentx": _AGENTX_METRICS,
        # 1000 / the P50, P90 and P99 inter_token_latency of 12.5, 25 and 80 ms.
        "derived": {"median_intvty": 80.0, "p90_intvty": 40.0, "p99_intvty": 12.5},
    },
    "config": {
        **_BASE_CONFIG,
        # No config.json: engine_flags is omitted instead of fabricating defaults.
        "workload": {
            "scenario": "agentx",
            "dataset": "semianalysis_cc_traces_weka_062126",
            "max_context_length": 131072,
            "max_concurrency": 8,
        },
    },
}


def test_agentx_point_row_carries_the_replay_and_its_interactivity(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "conc_8")

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [_AGENTX_ROW]


def test_agentx_run_with_config_json_and_scores_writes_a_point_and_an_accuracy_row(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "config.json", "conc_8", "lm_eval")

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [
        {
            **_AGENTX_ROW,
            # config.json names the run's model, as on its speed-grid rows; the
            # export's own name is the fallback.
            "model_id": _GRID_MODEL,
            "config": {**_AGENTX_ROW["config"], "engine_flags": _ENGINE_FLAGS},
        },
        {**_ACCURACY_ROW, "record_id": _record_id(2)},
    ]


def test_configless_agentx_with_gsm8k_uploads_both_rows_cleanly(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "conc_8")
    gsm8k_dir = results_dir / "gsm8k_c8"
    gsm8k_dir.mkdir()
    lm_eval_src = _DATA / "lm_eval" / "results_2026-11-02T01-18-05.218848.json"
    lm_eval_data = json.loads(lm_eval_src.read_text())
    lm_eval_data["model_name"] = _AGENTX_MODEL
    (gsm8k_dir / lm_eval_src.name).write_text(json.dumps(lm_eval_data))

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [
        _AGENTX_ROW,
        {
            **_ACCURACY_ROW,
            "record_id": _record_id(2),
            "model_id": _AGENTX_MODEL,
            "config": _BASE_CONFIG,
        },
    ]


def test_configless_accuracy_falls_back_to_agentx_model_when_results_omit_model_name(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "conc_8", "results_mmmu_pro.json")

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [
        _AGENTX_ROW,
        {
            **_ACCURACY_ROW,
            "record_id": _record_id(2),
            "model_id": _AGENTX_MODEL,
            "metrics": {"lm_eval": {"mmmu_pro": 0.61}},
            "config": _BASE_CONFIG,
        },
    ]


def test_conflicting_accuracy_models_abort_before_writing_agentx_rows(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "conc_8")
    (results_dir / "results_a.json").write_text(
        json.dumps(
            {
                "model_name": "model-a",
                "results": {"gsm8k": {"acc,none": 0.9}},
            }
        )
    )
    (results_dir / "results_b.json").write_text(
        json.dumps(
            {
                "model_name": "model-b",
                "results": {"mmlu": {"acc,none": 0.8}},
            }
        )
    )

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert "conflicting 'model_name' values across results_*.json" in err
    assert out == ""


def _agentx_results_dir(tmp_path: pathlib.Path, edit) -> pathlib.Path:
    """A results directory holding the fixture point, its export changed by
    `edit`."""
    results_dir = _results_dir(tmp_path, "conc_8")
    export_path = results_dir / _AGENTX_EXPORT
    export = json.loads(export_path.read_text())
    edit(export)
    export_path.write_text(json.dumps(export))
    return results_dir


def test_agentx_point_without_token_latency_has_no_derived_element(
    tmp_path, monkeypatch, capsys
):
    results_dir = _agentx_results_dir(
        tmp_path, lambda export: export.pop("inter_token_latency")
    )

    _, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert row["metrics"] == {
        "agentx": {
            name: value
            for name, value in _AGENTX_METRICS.items()
            if not name.startswith("inter_token_latency_")
        }
    }


def test_agentx_point_tolerates_json_null_blocks_and_missing_datasets(
    tmp_path, monkeypatch, capsys
):
    def nullify(export):
        export["inter_token_latency"] = None
        export["error_request_count"] = None
        export["metadata"] = None
        export["input_config"]["datasets"] = None

    results_dir = _agentx_results_dir(tmp_path, nullify)

    returncode, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert row["metrics"] == {
        "agentx": {
            name: value
            for name, value in _AGENTX_METRICS.items()
            if not name.startswith("inter_token_latency_")
            and name not in ("scenario", "submission_valid")
        }
    }
    assert row["config"]["workload"] == {
        "scenario": None,
        "dataset": None,
        "max_context_length": None,
        "max_concurrency": 8,
    }


@pytest.mark.parametrize(
    "export",
    [
        {"metadata": None},
        {"metadata": None, "input_config": None},
        {"metadata": None, "input_config": {}},
        {"metadata": None, "input_config": {"datasets": None}},
        {"metadata": None, "input_config": {"datasets": []}},
        {"metadata": None, "input_config": {"datasets": [None]}},
    ],
)
def test_agentx_workload_tolerates_missing_or_null_input_config_and_datasets(export):
    assert upload_results.agentx_workload(export, 8) == {
        "scenario": None,
        "dataset": None,
        "max_context_length": None,
        "max_concurrency": 8,
    }


def test_agentx_point_with_failed_requests_carries_their_count(
    tmp_path, monkeypatch, capsys
):
    results_dir = _agentx_results_dir(
        tmp_path,
        lambda export: export.update(
            error_request_count={"unit": "requests", "avg": 3.0}
        ),
    )

    _, (row,) = _upload(monkeypatch, capsys, results_dir)

    assert row["metrics"]["agentx"] == {
        **_AGENTX_METRICS,
        "error_request_count_avg": 3.0,
    }


def test_unreadable_agentx_export_fails_the_upload_but_keeps_the_other_points(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "conc_8")
    broken = results_dir / "conc_16/aiperf_artifacts/profile_export_aiperf.json"
    broken.parent.mkdir(parents=True)
    # A client killed while it wrote.
    broken.write_text('{"request_count": {"av')

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert f"Error loading {broken}: " in err
    assert _rows(out) == [_AGENTX_ROW]


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        "{}",
        json.dumps({"input_config": None}),
        json.dumps({"input_config": {"models": {"items": [{"name": ""}]}}}),
    ],
)
def test_malformed_agentx_export_fails_gracefully_and_keeps_healthy_points(
    tmp_path, monkeypatch, capsys, payload
):
    results_dir = _results_dir(tmp_path, "conc_8")
    broken = results_dir / "conc_16/aiperf_artifacts/profile_export_aiperf.json"
    broken.parent.mkdir(parents=True)
    broken.write_text(payload)

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert str(broken) in err
    assert _rows(out) == [_AGENTX_ROW]


def _agentx_row(record: int, concurrency: int) -> dict:
    """The fixture point's row, as the `record`th of a run and at another
    concurrency."""
    config = _AGENTX_ROW["config"]
    workload = {**config["workload"], "max_concurrency": concurrency}
    return {
        **_AGENTX_ROW,
        "record_id": _record_id(record),
        "config": {**config, "workload": workload},
    }


def test_agentx_ladder_writes_one_row_per_conc_directory_by_concurrency(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(tmp_path, "conc_8")
    # The same export under each: the concurrency is the directory's, and a
    # point may sit deeper than the top of the results directory.
    for point in ("conc_2", "agentx/conc_16"):
        shutil.copytree(results_dir / "conc_8", results_dir / point)

    returncode, rows = _upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert rows == [_agentx_row(1, 2), _agentx_row(2, 8), _agentx_row(3, 16)]


@pytest.mark.parametrize(
    "not_a_point",
    [
        "profile_export_aiperf.json",
        "wrong_location/profile_export_aiperf.json",
        "conc_abc/profile_export_aiperf.json",
        "conc_/profile_export_aiperf.json",
        "conc_16_old/deep/profile_export_aiperf.json",
        "conc_0/aiperf_artifacts/profile_export_aiperf.json",
        "conc_016/aiperf_artifacts/profile_export_aiperf.json",
    ],
)
def test_aiperf_export_under_no_conc_n_directory_is_named_and_skipped(
    tmp_path, monkeypatch, capsys, not_a_point
):
    results_dir = _results_dir(tmp_path, "conc_8")
    stray = results_dir / not_a_point
    stray.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(results_dir / _AGENTX_EXPORT, stray)

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 0
    assert _rows(out) == [_AGENTX_ROW]
    assert err == (
        f"Skipping {stray}: not under a conc_<N>/ directory, so not an AgentX point.\n"
    )


@pytest.mark.parametrize(
    "duplicate_rel",
    [
        "conc_8/retry/profile_export_aiperf.json",
        "retry/conc_8/aiperf_artifacts/profile_export_aiperf.json",
    ],
)
def test_two_exports_for_same_concurrency_stop_the_upload_before_any_row(
    tmp_path, monkeypatch, capsys, duplicate_rel
):
    results_dir = _results_dir(
        tmp_path, "config.json", "isl1024_osl1024_c64.json", "conc_8"
    )
    first = results_dir / _AGENTX_EXPORT
    second = results_dir / duplicate_rel
    second.parent.mkdir(parents=True)
    shutil.copy(first, second)

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert err == (
        f"Error: two AgentX exports for concurrency 8: {first} and {second}. "
        "One concurrency is one row; remove the export that is not the point's "
        "result.\n"
    )
    # Not the speed-grid cell either: the rerun would write it a second time.
    assert out == ""


def test_nested_conc_directories_are_rejected_as_ambiguous_before_any_row(
    tmp_path, monkeypatch, capsys
):
    results_dir = _results_dir(
        tmp_path, "config.json", "isl1024_osl1024_c64.json", "conc_8"
    )
    nested = (
        results_dir
        / "conc_8/nested/conc_16/aiperf_artifacts/profile_export_aiperf.json"
    )
    nested.parent.mkdir(parents=True)
    shutil.copy(results_dir / _AGENTX_EXPORT, nested)

    returncode, out, err = _run_upload(monkeypatch, capsys, results_dir)

    assert returncode == 1
    assert "ambiguous AgentX export path" in err
    assert str(nested) in err
    assert "conc_8, conc_16" in err
    assert out == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-11-02T09:18:05+08:00", "TIMESTAMP '2026-11-02T01:18:05Z'"),
        ("2026-10-28T08:35:19.586894Z", "TIMESTAMP '2026-10-28T08:35:19Z'"),
        ("2026-10-28T08:35:19Z", "TIMESTAMP '2026-10-28T08:35:19Z'"),
    ],
)
def test_created_time_parses_offsets_and_trailing_z(raw, expected):
    assert upload_results.bq_utils.created_time_sql(raw) == expected


@pytest.mark.parametrize("bad_timestamp", [1793582252.74, 0, 12345, {"ts": 1}])
def test_created_time_falls_back_on_non_string_input(capsys, bad_timestamp):
    assert (
        upload_results.bq_utils.created_time_sql(bad_timestamp) == "CURRENT_TIMESTAMP()"
    )
    assert "Could not parse date string" in capsys.readouterr().err
