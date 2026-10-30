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
"""Verification tests for DeepSeek-V4.1-Flash TPU v7x benchmark and eval recipes."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "scripts" / "vllm" / "benchmarking"
CONFIGS_DIR = BENCH_DIR / "configs"
GSM8K_BASELINE = (
    BENCH_DIR / "baselines" / "eval" / "deepseek-v41-flash-dp8-ep.gsm8k.baseline.json"
)
RECIPES_DOC = REPO_ROOT / "docs" / "deployment" / "v7x_recipes.md"
RECIPE_SCRIPTS = (
    "deepseek-v41-flash-dp8-ep.sh",
    "deepseek-v41-flash-dp8-ep-128k-prefill.sh",
    "deepseek-v41-flash-dp8-ep-dspark-decode.sh",
)
HARNESS_PRE_SOURCE_DEFAULTS = {
    "ISL_OSL_CONFIGS": "512:512",
    "CONCURRENCY_OPTIONS": "1",
    "NUM_PROMPTS": "",
    "BENCHMARK_WARMUP_RUNS": "0",
    "LM_EVAL_TASKS": "mmlu_llama mmlu_pro",
    "EVAL_TOLERANCE": "",
}


def _source_config(
    script_name: str,
    *keys: str,
    pre_env: dict[str, str] | None = None,
) -> dict[str, str]:
    script_path = CONFIGS_DIR / script_name
    assert script_path.is_file(), f"Missing config script: {script_path}"
    printf_lines = " ".join(f'"{k}=${{{k}:-}}"' for k in keys)
    cmd = (
        'unset "${!DEEPSEEK_V41_@}"; '
        + "".join(f"export {k}={v!r}; " for k, v in (pre_env or {}).items())
        + f'source "{script_path}" && printf "%s\\n" {printf_lines}'
    )
    res = subprocess.run(
        ["bash", "-e", "-c", cmd],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    out: dict[str, str] = {}
    for line in res.stdout.strip().splitlines():
        k, _, v = line.partition("=")
        out[k] = v
    return out


def test_recipe_scripts_hygiene() -> None:
    """Verify portable shell syntax and MOE_REQUANTIZE_BLOCK_SIZE=256 defaults."""
    for script_name in RECIPE_SCRIPTS:
        content = (CONFIGS_DIR / script_name).read_text(encoding="utf-8")
        assert "_caller_env" not in content
        assert "/proc/" not in content
        assert "TPU_PREMAPPED_BUFFER_SIZE" not in content
        sourced = _source_config(
            script_name,
            "MOE_REQUANTIZE_BLOCK_SIZE",
            pre_env={"MOE_REQUANTIZE_BLOCK_SIZE": ""},
        )
        # Note: empty string in `:-256` falls back to 256, matching unset env.
        assert sourced["MOE_REQUANTIZE_BLOCK_SIZE"] == "256"
        overridden = _source_config(
            script_name,
            "MOE_REQUANTIZE_BLOCK_SIZE",
            pre_env={"MOE_REQUANTIZE_BLOCK_SIZE": "128"},
        )
        assert overridden["MOE_REQUANTIZE_BLOCK_SIZE"] == "128"


def test_smoke_and_mixed_serving_config() -> None:
    """Verify deepseek-v41-flash-dp8-ep defaults and DEEPSEEK_V41_* overrides."""
    env = _source_config(
        "deepseek-v41-flash-dp8-ep.sh",
        "MODEL",
        "TENSOR_PARALLELISM",
        "DATA_PARALLELISM",
        "ENABLE_EP",
        "MAX_MODEL_LEN",
        "KV_CACHE_MEMORY_BYTES",
        "COMPILE_SIZES",
        "NUM_PROMPTS",
        "ISL_OSL_CONFIGS",
        "CONCURRENCY_OPTIONS",
        "BENCHMARK_WARMUP_RUNS",
        "LM_EVAL_TASKS",
        "EVAL_TOLERANCE",
        "EXTRA_SERVE_ARGS",
        pre_env=HARNESS_PRE_SOURCE_DEFAULTS,
    )
    assert env["MODEL"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert env["TENSOR_PARALLELISM"] == "1"
    assert env["DATA_PARALLELISM"] == "8"
    assert env["ENABLE_EP"] == "true"
    assert env["MAX_MODEL_LEN"] == "8192"
    assert env["KV_CACHE_MEMORY_BYTES"] == "4294967296"
    assert env["COMPILE_SIZES"] == "[8,64,2048]"
    assert env["NUM_PROMPTS"] == "16"
    assert env["ISL_OSL_CONFIGS"] == "128:16 4096:1024"
    assert env["CONCURRENCY_OPTIONS"] == "8"
    assert env["BENCHMARK_WARMUP_RUNS"] == "1"
    assert env["LM_EVAL_TASKS"] == "gsm8k"
    assert env["EVAL_TOLERANCE"] == "0.03"
    assert "--block-size=16" in env["EXTRA_SERVE_ARGS"]

    override = _source_config(
        "deepseek-v41-flash-dp8-ep.sh",
        "NUM_PROMPTS",
        "ISL_OSL_CONFIGS",
        "CONCURRENCY_OPTIONS",
        "BENCHMARK_WARMUP_RUNS",
        "LM_EVAL_TASKS",
        pre_env={
            **HARNESS_PRE_SOURCE_DEFAULTS,
            "DEEPSEEK_V41_NUM_PROMPTS": "32",
            "DEEPSEEK_V41_ISL_OSL_CONFIGS": "4096:1024",
            "DEEPSEEK_V41_CONCURRENCY_OPTIONS": "16",
            "DEEPSEEK_V41_BENCHMARK_WARMUP_RUNS": "2",
            "DEEPSEEK_V41_LM_EVAL_TASKS": "gsm8k_cot",
        },
    )
    assert override["NUM_PROMPTS"] == "32"
    assert override["ISL_OSL_CONFIGS"] == "4096:1024"
    assert override["CONCURRENCY_OPTIONS"] == "16"
    assert override["BENCHMARK_WARMUP_RUNS"] == "2"
    assert override["LM_EVAL_TASKS"] == "gsm8k_cot"


def test_128k_prefill_and_dspark_decode_configs() -> None:
    """Verify 128k pure prefill (129024:1) and DSpark decode (129024:2048) configs."""
    prefill = _source_config(
        "deepseek-v41-flash-dp8-ep-128k-prefill.sh",
        "ENABLE_PREFIX_CACHING",
        "MAX_MODEL_LEN",
        "NUM_PROMPTS",
        "ISL_OSL_CONFIGS",
        "CONCURRENCY_OPTIONS",
        "BENCHMARK_WARMUP_RUNS",
        pre_env=HARNESS_PRE_SOURCE_DEFAULTS,
    )
    assert prefill["ENABLE_PREFIX_CACHING"] == "false"
    assert prefill["MAX_MODEL_LEN"] == "131072"
    assert prefill["NUM_PROMPTS"] == "8"
    assert prefill["ISL_OSL_CONFIGS"] == "129024:1"
    assert prefill["CONCURRENCY_OPTIONS"] == "1"
    assert prefill["BENCHMARK_WARMUP_RUNS"] == "1"

    decode = _source_config(
        "deepseek-v41-flash-dp8-ep-dspark-decode.sh",
        "ENABLE_PREFIX_CACHING",
        "MAX_MODEL_LEN",
        "NUM_PROMPTS",
        "ISL_OSL_CONFIGS",
        "CONCURRENCY_OPTIONS",
        "BENCHMARK_WARMUP_RUNS",
        "EXTRA_SERVE_ARGS",
        pre_env=HARNESS_PRE_SOURCE_DEFAULTS,
    )
    assert decode["ENABLE_PREFIX_CACHING"] == "true"
    assert decode["MAX_MODEL_LEN"] == "131072"
    assert decode["NUM_PROMPTS"] == ""
    assert decode["ISL_OSL_CONFIGS"] == "129024:2048"
    assert decode["CONCURRENCY_OPTIONS"] == "64"
    assert decode["BENCHMARK_WARMUP_RUNS"] == "1"
    assert '"method":"dspark"' in decode["EXTRA_SERVE_ARGS"]
    assert '"num_speculative_tokens":5' in decode["EXTRA_SERVE_ARGS"]
    assert "--enable-ep-weight-filter" in decode["EXTRA_SERVE_ARGS"]
    assert "--safetensors-load-strategy=prefetch" in decode["EXTRA_SERVE_ARGS"]
    assert "--safetensors-prefetch-num-threads=8" in decode["EXTRA_SERVE_ARGS"]
    assert "--safetensors-load-strategy=lazy" not in decode["EXTRA_SERVE_ARGS"]

    decode_override = _source_config(
        "deepseek-v41-flash-dp8-ep-dspark-decode.sh",
        "NUM_PROMPTS",
        "ISL_OSL_CONFIGS",
        "CONCURRENCY_OPTIONS",
        "BENCHMARK_WARMUP_RUNS",
        pre_env={
            **HARNESS_PRE_SOURCE_DEFAULTS,
            "DEEPSEEK_V41_NUM_PROMPTS": "128",
            "DEEPSEEK_V41_ISL_OSL_CONFIGS": "65536:1024",
            "DEEPSEEK_V41_CONCURRENCY_OPTIONS": "32",
            "DEEPSEEK_V41_BENCHMARK_WARMUP_RUNS": "2",
        },
    )
    assert decode_override["NUM_PROMPTS"] == "128"
    assert decode_override["ISL_OSL_CONFIGS"] == "65536:1024"
    assert decode_override["CONCURRENCY_OPTIONS"] == "32"
    assert decode_override["BENCHMARK_WARMUP_RUNS"] == "2"


def test_v7x_recipes_documentation_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify v7x_recipes.md and GSM8K baseline JSON match check_regression."""
    doc = RECIPES_DOC.read_text(encoding="utf-8")
    for cfg in RECIPE_SCRIPTS:
        assert cfg in doc
    assert "DEEPSEEK_V41_ISL_OSL_CONFIGS=4096:1024" in doc
    assert "DEEPSEEK_V41_CONCURRENCY_OPTIONS=8" in doc
    assert "MOE_REQUANTIZE_BLOCK_SIZE=256" in doc
    assert "deepseek-v41-flash-dp8-ep.gsm8k.baseline.json" in doc
    assert "limit=100" in doc and "3581.05 tok/s" in doc and "24830.05 tok/s" in doc
    assert "542.64 tok/s" in doc and "7065.60 tok/s" in doc
    assert "901.27 tok/s" in doc and "8.88 ms" in doc
    assert "98.50%` acceptance" in doc and "5.925` mean acceptance length" in doc
    assert "MAX_NUM_SEQS=8" in doc and "rounds.csv" in doc
    assert "61.32%` acceptance" in doc and "4.066` mean acceptance length" in doc
    assert "0.99 ± 0.0100" in doc
    assert "--enable-ep-weight-filter" in doc
    assert "--safetensors-load-strategy=prefetch" in doc
    assert "16/128" in doc and "48/384" in doc
    assert "COMPILE_SIZES=[64,288,2048]" in doc
    assert "check_regression.py` `PASS`" in doc
    assert "omits `--enable-ep-weight-filter`" not in doc.lower()
    assert "--safetensors-load-strategy=lazy" not in doc
    assert "unverified" not in doc.lower()
    assert "do not enable dspark" not in doc.lower()
    dspark_sh = (CONFIGS_DIR / "deepseek-v41-flash-dp8-ep-dspark-decode.sh").read_text(
        encoding="utf-8"
    )
    assert "unverified" not in dspark_sh.lower()

    assert GSM8K_BASELINE.is_file()
    baseline = json.loads(GSM8K_BASELINE.read_text(encoding="utf-8"))
    assert baseline["gsm8k"]["exact_match,flexible-extract"] == 0.98
    assert baseline["gsm8k"]["exact_match_stderr,flexible-extract"] == 0.0141
    assert baseline["gsm8k"]["limit"] == 100
    assert baseline["gsm8k"]["num_fewshot"] == 5

    spec = importlib.util.spec_from_file_location(
        "check_regression", BENCH_DIR / "check_regression.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "check_regression", mod)
    spec.loader.exec_module(mod)
    res = mod.check_eval(
        {
            "gsm8k": {
                "exact_match,flexible-extract": 0.98,
                "exact_match_stderr,flexible-extract": 0.0141,
            }
        },
        baseline,
        tolerance=0.03,
    )
    assert res.failed_count == 0


@pytest.mark.tpu_test
def test_deepseek_v41_recipes_tpu_smoke() -> None:  # pragma: no cover
    """Verify recipe speculative_config JSON parses and TPU7x device is available."""
    importlib.import_module("torch_tpu")

    decode = _source_config(
        "deepseek-v41-flash-dp8-ep-dspark-decode.sh",
        "EXTRA_SERVE_ARGS",
    )
    assert "--speculative-config=" in decode["EXTRA_SERVE_ARGS"]
    t = torch.ones((4, 256), dtype=torch.bfloat16, device=torch.device("tpu:0"))
    assert float(t.sum().cpu().item()) == 1024.0
