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
"""The default serving mode of run_benchmarks.sh must not see spec_bench knobs."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking"


def _run(config: str, extra_env: dict[str, str], *args: str):
    env = {**os.environ, **extra_env}
    return subprocess.run(
        ["bash", "run_benchmarks.sh", "--config", config, "--dry-run", *args],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _dry_run(config: str, extra_env: dict[str, str]) -> str:
    result = _run(config, extra_env)
    return result.stdout + result.stderr


def test_serving_mode_ignores_served_model_name_env(tmp_path: Path):
    """An exported SERVED_MODEL_NAME must not rename the model in serving mode.

    The serving client sends `--model $MODEL` only, so renaming the server
    would make every request fail.
    """
    results_dir = tmp_path / "results"
    output = _dry_run(
        "qwen3-0.6b-tp1",
        {"SERVED_MODEL_NAME": "renamed-model", "RESULTS_DIR": str(results_dir)},
    )

    assert "vllm serve" in output
    assert "--served-model-name renamed-model" not in output
    config = json.loads((results_dir / "config.json").read_text())
    assert "spec_method" not in config


def test_spec_bench_mode_uses_served_model_name_env(tmp_path: Path):
    output = _dry_run(
        "qwen3-0.6b-ngram",
        {
            "SERVED_MODEL_NAME": "renamed-model",
            "RESULTS_DIR": str(tmp_path / "results"),
        },
    )

    assert "--served-model-name renamed-model" in output


def test_spec_bench_mode_writes_valid_config_json(tmp_path: Path):
    """spec_bench configs set no RANDOM_RANGE_RATIO / EVAL_TOLERANCE; config.json
    must still be valid JSON (the upload scripts parse it) and record the sweep."""
    results_dir = tmp_path / "results"
    result = _run(
        "qwen3-0.6b-ngram",
        {"RESULTS_DIR": str(results_dir)},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    config = json.loads((results_dir / "config.json").read_text())

    assert config["random_range_ratio"] is None
    assert config["eval_tolerance"] is None
    assert config["benchmark_mode"] == "spec_bench"
    assert config["spec_method"] == "ngram"
    assert config["spec_draft_model"] is None
    assert config["spec_draft_tp"] is None
    assert config["spec_k_list"] == "0 1 2 4"
    assert config["spec_bs_list"] == "1"
    assert config["spec_bench_category"] == "summarization"
    assert config["spec_num_prompts"] == 20
    assert config["spec_warmup_prompts"] == 5
    assert config["spec_reps"] == 1


def test_spec_bench_mode_rejects_external_host():
    """--host skips the per-(BS, K) server restart, so every cell would hit the
    same server; spec_bench must refuse it."""
    result = _run("qwen3-0.6b-ngram", {}, "--host", "localhost")

    assert result.returncode != 0
    assert "--host" in result.stdout + result.stderr


def _extra_serve_args_tokens(config: Path) -> list[str]:
    """EXTRA_SERVE_ARGS split the way run_benchmarks.sh expands $server_cmd."""
    script = (
        f"source {shlex.quote(str(config))} >/dev/null 2>&1; "
        'printf "%s\\n" $EXTRA_SERVE_ARGS'
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout
    return [t for t in out.split("\n") if t]


SPEC_CONFIGS = sorted(
    p
    for p in (SCRIPT_DIR / "configs").glob("*.sh")
    if 'BENCHMARK_MODE="spec_bench"' in p.read_text()
)


@pytest.mark.parametrize("config", SPEC_CONFIGS, ids=lambda p: p.stem)
def test_spec_bench_extra_serve_args_survive_word_splitting(config: Path):
    """No quote characters reach vllm serve, and every JSON value parses."""
    for token in _extra_serve_args_tokens(config):
        assert not token.startswith(("'", '"')) and not token.endswith(("'", '"')), (
            token
        )
        if token.startswith("{"):
            json.loads(token)
        elif "={" in token:
            json.loads(token.split("=", 1)[1])
