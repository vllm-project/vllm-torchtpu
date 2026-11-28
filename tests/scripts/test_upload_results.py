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
