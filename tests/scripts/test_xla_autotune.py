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
"""Unit tests for the general XLA autotune package."""

import json
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from scripts.xla_autotune.config import (
    BenchmarkConfig,
    ModelConfig,
    Trial,
    TrialResult,
    TuningConfig,
    load_tuning_config,
)
from scripts.xla_autotune.local_autotuner import (
    plan_trials,
    run_trial,
    slice_flags,
)
from scripts.xla_autotune.runner import (
    get_benchmark_runner,
    parse_vllm_bench_serve_stdout,
)
from scripts.xla_autotune.server_manager import ServerManager
from scripts.xla_autotune.utils import (
    generate_ranking_report,
    get_cdk_env,
    load_flags,
    parse_cdk_job_id,
    sync_and_parse_cdk_job,
)


def test_load_and_slice_flags(tmp_path):
    flags_file = tmp_path / "test_flags.txt"
    flags_content = """# Comment line
--xla_tpu_flag1=true

--xla_tpu_flag2=false
# Another comment
--xla_tpu_flag3=123
--xla_tpu_flag4=456
--xla_tpu_flag5=789
"""
    flags_file.write_text(flags_content)

    flags = load_flags(str(flags_file))
    assert len(flags) == 5
    assert flags[0] == "--xla_tpu_flag1=true"
    assert flags[-1] == "--xla_tpu_flag5=789"

    # Slice into 2 shards
    s1 = slice_flags(flags, 1, 2)
    s2 = slice_flags(flags, 2, 2)
    assert len(s1) + len(s2) == len(flags)
    assert s1 == [
        "--xla_tpu_flag1=true",
        "--xla_tpu_flag2=false",
        "--xla_tpu_flag3=123",
    ]
    assert s2 == ["--xla_tpu_flag4=456", "--xla_tpu_flag5=789"]

    with pytest.raises(ValueError):
        slice_flags(flags, 0, 2)
    with pytest.raises(ValueError):
        slice_flags(flags, 3, 2)


def test_plan_trials():
    candidates = ["--flagA=1", "--flagB=2", "--flagC=3"]
    trials = plan_trials(candidates, baseline_runs=2, skip_candidates=0)
    assert len(trials) == 5
    assert (
        trials[0].trial_id == "baseline_01"
        and trials[0].kind == "baseline"
        and trials[0].flag is None
    )
    assert (
        trials[1].trial_id == "baseline_02"
        and trials[1].kind == "baseline"
        and trials[1].flag is None
    )
    assert (
        trials[2].trial_id == "cand_001"
        and trials[2].kind == "candidate"
        and trials[2].flag == "--flagA=1"
    )
    assert (
        trials[4].trial_id == "cand_003"
        and trials[4].kind == "candidate"
        and trials[4].flag == "--flagC=3"
    )

    # Resume with skip_candidates = 1
    resumed = plan_trials(candidates, baseline_runs=2, skip_candidates=1)
    assert len(resumed) == 4
    assert resumed[0].trial_id == "baseline_01"
    assert resumed[1].trial_id == "baseline_02"
    assert resumed[2].trial_id == "cand_002"
    assert resumed[3].trial_id == "cand_003"


def test_load_yaml_configs():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    example_path = os.path.join(
        repo_root, "scripts", "xla_autotune", "configs", "example_config.yaml"
    )
    assert os.path.exists(example_path)

    cfg = load_tuning_config(example_path)
    assert cfg.model.name == "meta-llama/Llama-3.1-70B-Instruct"
    assert cfg.model.port == 8000
    assert cfg.benchmark.type == "vllm_bench_serve"
    assert cfg.benchmark.target_metric == "total_token_throughput"
    assert cfg.benchmark.args.get("random_input_len") == 2048
    assert cfg.cdk is not None
    assert cfg.cdk.recipe == "torchtpu-vllm-benchmark"

    flags_path = os.path.join(repo_root, "scripts", "xla_autotune", "flags.txt")
    assert os.path.exists(flags_path)
    flags = load_flags(flags_path)
    assert len(flags) > 500


def test_parse_vllm_bench_serve_stdout():
    sample_table = """
============ Serving Benchmark Result ============
Successful requests:                     2048
Benchmark duration (s):                  163.84
Total input tokens:                      16777216
Total generated tokens:                  2097152
Request throughput (req/s):              12.50
Output tokens per second (tok/s):        1280.00
Total tokens per second (tok/s):         11520.00
Mean TTFT (ms):                          123.45
Median TTFT (ms):                        110.00
P99 TTFT (ms):                           250.00
Mean TPOT (ms):                          5.67
Mean ITL (ms):                           4.32
==================================================
"""
    metrics = parse_vllm_bench_serve_stdout(sample_table)
    assert metrics["request_throughput"] == 12.50
    assert metrics["output_throughput"] == 1280.00
    assert metrics["total_token_throughput"] == 11520.00
    assert metrics["mean_ttft_ms"] == 123.45
    assert metrics["median_ttft_ms"] == 110.00
    assert metrics["p99_ttft_ms"] == 250.00
    assert metrics["mean_tpot_ms"] == 5.67
    assert metrics["mean_itl_ms"] == 4.32


def test_server_manager_env_building():
    model_cfg = ModelConfig(
        name="test-model",
        port=8000,
        base_libtpu_init_args="--base_flag=1",
        server_env={"MY_ENV": "val1"},
    )
    sm = ServerManager(model_cfg, dry_run=True)
    env = sm.build_env(extra_libtpu_init_args=["--candidate_flag=2"])

    assert "--base_flag=1 --candidate_flag=2" in env["LIBTPU_INIT_ARGS"]
    assert env["VLLM_XLA_CHECK_RECOMPILATION"] == "0"
    assert env["MY_ENV"] == "val1"


def test_ranking_report_generation(tmp_path):
    out_file = str(tmp_path / "ranking.md")
    results = [
        TrialResult(
            trial_id="baseline_01",
            kind="baseline",
            flag=None,
            model="m",
            target_metric="total_token_throughput",
            target_value=100.0,
            success=True,
        ),
        TrialResult(
            trial_id="baseline_02",
            kind="baseline",
            flag=None,
            model="m",
            target_metric="total_token_throughput",
            target_value=104.0,
            success=True,
        ),
        TrialResult(
            trial_id="cand_001",
            kind="candidate",
            flag="--flagA=1",
            model="m",
            target_metric="total_token_throughput",
            target_value=112.2,
            success=True,
        ),
        TrialResult(
            trial_id="cand_002",
            kind="candidate",
            flag="--flagB=2",
            model="m",
            target_metric="total_token_throughput",
            target_value=96.9,
            success=True,
        ),
    ]

    generate_ranking_report(results, "total_token_throughput", out_file)
    assert os.path.exists(out_file)

    with open(out_file) as f:
        content = f.read()

    assert "Baseline Performance: `102.00`" in content
    # cand_001 should be +10.00%
    assert "+10.00%" in content
    # cand_002 should be -5.00%
    assert "-5.00%" in content


def test_dry_run_sweep(tmp_path):
    artifact_dir = str(tmp_path / "autotune_output")
    cfg = TuningConfig(
        model=ModelConfig(name="deepseek-ai/DeepSeek-V4-Pro-0813", port=47903),
        benchmark=BenchmarkConfig(target_metric="total_token_throughput"),
        execution_mode="local",
        baseline_runs=1,
        dry_run=True,
    )

    trials = [
        Trial("baseline_01", "baseline", None),
        Trial("cand_001", "candidate", "--xla_tpu_flag=true"),
    ]

    sm = ServerManager(cfg.model, dry_run=True)
    bench_runner = get_benchmark_runner(cfg.model, cfg.benchmark, dry_run=True)
    summary_path = os.path.join(artifact_dir, "summary.jsonl")
    os.makedirs(artifact_dir, exist_ok=True)

    with open(summary_path, "w") as s_fp:
        for t in trials:
            res = run_trial(
                trial=t,
                cfg=cfg,
                server_mgr=sm,
                bench_runner=bench_runner,
                artifact_dir=artifact_dir,
                summary_fp=s_fp,
            )
            assert res.success is True
            assert res.target_value is not None

    assert os.path.exists(summary_path)
    with open(summary_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["trial_id"] == "baseline_01"
    assert lines[1]["trial_id"] == "cand_001"


def test_parse_cdk_job_id():
    text1 = "Created job j-7fb420a0-198f-42df-9d73 on cluster"
    assert parse_cdk_job_id(text1) == "j-7fb420a0-198f-42df-9d73"

    text2 = "Job ID: j-abc123def456-0001"
    assert parse_cdk_job_id(text2) == "j-abc123def456-0001"

    text3 = "No job here"
    assert parse_cdk_job_id(text3) is None


def test_get_cdk_env():
    env = get_cdk_env(cdk_root_dir="/custom/path")
    assert env.get("CDK_ROOT_DIR") == "/custom/path"


def test_sync_and_parse_cdk_job(tmp_path, monkeypatch):
    job_id = "j-test-1234"
    trial_dir = str(tmp_path / "trial")
    mock_data_dir = tmp_path / "cloud-devkit-data" / "job-outputs" / job_id
    mock_data_dir.mkdir(parents=True)

    mock_log = mock_data_dir / "sidecar-bench.log"
    mock_log.write_text("""
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  10.0
Total input tokens:                      1000
Total generated tokens:                  2000
Request throughput (req/s):              10.0
Output token throughput (tok/s):         200.0
Total token throughput (tok/s):          300.0
Mean TTFT (ms):                          50.0
==================================================
""")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: None)

    metrics, err = sync_and_parse_cdk_job(job_id, trial_dir)
    assert err == ""
    assert metrics.get("total_token_throughput") == 300.0
    assert metrics.get("request_throughput") == 10.0
    assert metrics.get("output_throughput") == 200.0
    assert metrics.get("mean_ttft_ms") == 50.0
    assert os.path.exists(os.path.join(trial_dir, f"{job_id}.log"))
