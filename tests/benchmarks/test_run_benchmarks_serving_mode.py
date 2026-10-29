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

import os
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking"


def _dry_run(config: str, extra_env: dict[str, str]) -> str:
    env = {**os.environ, **extra_env}
    result = subprocess.run(
        ["bash", "run_benchmarks.sh", "--config", config, "--dry-run"],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.stdout + result.stderr


def test_serving_mode_ignores_served_model_name_env():
    """An exported SERVED_MODEL_NAME must not rename the model in serving mode.

    The serving client sends `--model $MODEL` only, so renaming the server
    would make every request fail.
    """
    output = _dry_run("qwen3-0.6b-tp1", {"SERVED_MODEL_NAME": "renamed-model"})

    assert "vllm serve" in output
    assert "--served-model-name renamed-model" not in output


def test_spec_bench_mode_uses_served_model_name_env():
    output = _dry_run(
        "qwen3-0.6b-ngram",
        {"SERVED_MODEL_NAME": "renamed-model", "SPEC_K_LIST": "0"},
    )

    assert "--served-model-name renamed-model" in output
