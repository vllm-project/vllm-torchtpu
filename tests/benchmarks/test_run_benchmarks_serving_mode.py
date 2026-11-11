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


@pytest.mark.parametrize("bad_mode", ["spec-bench", "perf", "eval", "unknown"])
def test_invalid_benchmark_mode_rejected(bad_mode: str, tmp_path: Path):
    """Any BENCHMARK_MODE other than serving or spec_bench must fail fast."""
    result = _run(
        "qwen3-0.6b-tp1",
        {"RESULTS_DIR": str(tmp_path / "results")},
        "--mode",
        bad_mode,
    )

    assert result.returncode != 0
    assert "BENCHMARK_MODE" in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("var_name", "bad_value"),
    [
        ("SPEC_K_LIST", "0 x 2"),
        ("SPEC_K_LIST", "-1"),
        ("SPEC_K_LIST", "   "),
        ("SPEC_BS_LIST", "1 four"),
        ("SPEC_BS_LIST", "0"),
        ("SPEC_BS_LIST", "   "),
        ("SPEC_REPS", "two"),
        ("SPEC_REPS", "0"),
        ("SPEC_NUM_PROMPTS", "many"),
        ("SPEC_NUM_PROMPTS", "0"),
        ("SPEC_WARMUP_PROMPTS", "five"),
        ("SPEC_WARMUP_PROMPTS", "-1"),
    ],
)
def test_spec_bench_validates_numeric_knobs(
    var_name: str, bad_value: str, tmp_path: Path
):
    """Non-integer or out-of-range SPEC_* numeric settings must fail fast."""
    env = {
        "RESULTS_DIR": str(tmp_path / "results"),
        "SPEC_METHOD": "ngram",
        var_name: bad_value,
    }
    result = _run("qwen3-0.6b-tp1", env, "--mode", "spec_bench")

    assert result.returncode != 0
    assert var_name in result.stdout + result.stderr


def test_spec_bench_client_passes_max_concurrency_and_tokenizer(tmp_path: Path):
    """run_spec_bench_client must cap concurrency to $bs and share tokenizer lookup."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    server_flag = tmp_path / "server_up"
    argv_log = tmp_path / "vllm_bench_argv.jsonl"
    dataset = tmp_path / "question.jsonl"
    dataset.write_text(
        '{"question_id": 1, "category": "summarization", "turns": ["hi"]}\n'
    )

    vllm_stub = bin_dir / "vllm"
    vllm_stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        "if sys.argv[1:2] == ['serve']:\n"
        f"    pathlib.Path({str(server_flag)!r}).write_text('up')\n"
        "    time.sleep(60)\n"
        "    sys.exit(0)\n"
        "if sys.argv[1:3] == ['bench', 'serve']:\n"
        "    args = sys.argv[3:]\n"
        f"    with open({str(argv_log)!r}, 'a') as f:\n"
        "        f.write(json.dumps(args) + '\\n')\n"
        "    out = args[args.index('--result-filename') + 1]\n"
        "    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)\n"
        "    pathlib.Path(out).write_text(json.dumps({\n"
        "        'duration': 1.0, 'completed': 1, 'output_throughput': 10.0\n"
        "    }))\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    vllm_stub.chmod(0o755)

    curl_stub = bin_dir / "curl"
    curl_stub.write_text(f"#!/bin/bash\n[ -f {shlex.quote(str(server_flag))} ]\n")
    curl_stub.chmod(0o755)

    pkill_stub = bin_dir / "pkill"
    pkill_stub.write_text(
        f"#!/bin/bash\nrm -f {shlex.quote(str(server_flag))}\nexit 0\n"
    )
    pkill_stub.chmod(0o755)

    sleep_stub = bin_dir / "sleep"
    sleep_stub.write_text("#!/bin/bash\nexit 0\n")
    sleep_stub.chmod(0o755)

    results_dir = tmp_path / "results"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RESULTS_DIR": str(results_dir),
        "SPEC_METHOD": "ngram",
        "SPEC_K_LIST": "0",
        "SPEC_BS_LIST": "4",
        "SPEC_WARMUP_PROMPTS": "0",
        "SPEC_NUM_PROMPTS": "8",
        "SPEC_REPS": "1",
        "SPEC_BENCH_DATASET": str(dataset),
        "TOKENIZER": "/custom/tokenizer",
    }
    result = subprocess.run(
        [
            "bash",
            "run_benchmarks.sh",
            "--config",
            "qwen3-0.6b-tp1",
            "--mode",
            "spec_bench",
        ],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    calls = [json.loads(line) for line in argv_log.read_text().splitlines()]
    assert len(calls) == 1
    bench_args = calls[0]
    assert "--max-concurrency" in bench_args
    assert bench_args[bench_args.index("--max-concurrency") + 1] == "4"
    assert "--tokenizer" in bench_args
    assert bench_args[bench_args.index("--tokenizer") + 1] == "/custom/tokenizer"


def test_qwen35_spec_configs_avoid_ineffective_or_deprecated_flags():
    """Qwen3.5 spec configs use SPEC_METHOD=mtp and omit ineffective template flags."""
    mtp_cfg = (SCRIPT_DIR / "configs" / "qwen3.5-397b-fp8-mtp.sh").read_text()
    dflash_cfg = (SCRIPT_DIR / "configs" / "qwen3.5-35b-fp8-dflash.sh").read_text()

    assert 'SPEC_METHOD="mtp"' in mtp_cfg
    assert "qwen3_next_mtp" not in mtp_cfg
    assert "--default-chat-template-kwargs" not in mtp_cfg
    assert "--default-chat-template-kwargs" not in dflash_cfg


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
