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
"""Configuration definitions and loaders for general XLA flag autotuning."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ModelConfig:
    """Model server specification."""

    name: str
    model_path: str | None = None  # Local mount path or HuggingFace ID
    port: int = 8000
    health_endpoint: str = "/health"
    startup_timeout_s: int = 7200
    base_libtpu_init_args: str = ""
    server_env: dict[str, str] = field(default_factory=dict)
    serve_args: list[str] = field(default_factory=list)

    def get_effective_model_path(self) -> str:
        return self.model_path if self.model_path else self.name


@dataclass
class BenchmarkConfig:
    """Benchmark runner specification."""

    type: str = "vllm_bench_serve"  # "vllm_bench_serve" | "custom_command"
    target_metric: str = "total_token_throughput"
    warmup_runs: int = 1
    # Arguments for vllm bench serve (e.g. random_input_len, max_concurrency)
    args: dict[str, Any] = field(default_factory=dict)
    # Custom shell command template when type == "custom_command"
    custom_cmd: str | None = None


@dataclass
class CDKConfig:
    """CDK (Cloud DevKit) cluster submission specification."""

    recipe: str  # Recipe name in recipes.yml
    profile: str | None = None
    agent_code: str | None = None
    cdk_root_dir: str | None = None
    extra_recipe_params: dict[str, str] = field(default_factory=dict)
    poll_interval_s: int = 30
    job_timeout_s: int = 14400
    tags: list[str] = field(default_factory=lambda: ["agent-jobs", "xla-autotune"])


@dataclass
class TuningConfig:
    """Root configuration for an autotuning experiment."""

    model: ModelConfig
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    cdk: CDKConfig | None = None
    execution_mode: str = "local"  # "local" | "in_pod" | "cdk"
    flag_list_file: str = "flags.txt"
    baseline_runs: int = 2
    skip_candidates: int = 0
    slice_index: int = 1
    slice_count: int = 1
    artifact_dir: str = "artifacts/xla_autotune"
    dry_run: bool = False


@dataclass
class Trial:
    """One evaluation point in the sweep."""

    trial_id: str
    kind: str  # "baseline" | "candidate"
    flag: str | None = None  # None for baseline


@dataclass
class TrialResult:
    """Result of a single trial evaluation."""

    trial_id: str
    kind: str
    flag: str | None
    model: str
    target_metric: str
    target_value: float | None = None
    success: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    started_utc: str = ""
    finished_utc: str = ""
    error: str = ""
    cdk_job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_tuning_config(path: str) -> TuningConfig:
    """Load TuningConfig from a YAML or JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise ImportError(
                    "PyYAML is required to parse YAML configs. Install it or use JSON."
                )
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return _dict_to_tuning_config(data)


def _dict_to_tuning_config(d: dict[str, Any]) -> TuningConfig:
    model_data = d.get("model", {})
    if "name" not in model_data:
        raise ValueError("Configuration 'model' block must include 'name'")
    model = ModelConfig(**model_data)

    bench_data = d.get("benchmark", {})
    benchmark = BenchmarkConfig(**bench_data)

    cdk_data = d.get("cdk")
    cdk = CDKConfig(**cdk_data) if cdk_data else None

    # Top-level tuning params
    return TuningConfig(
        model=model,
        benchmark=benchmark,
        cdk=cdk,
        execution_mode=d.get("execution_mode", "local"),
        flag_list_file=d.get("flag_list_file", "flags.txt"),
        baseline_runs=int(d.get("baseline_runs", 2)),
        skip_candidates=int(d.get("skip_candidates", 0)),
        slice_index=int(d.get("slice_index", 1)),
        slice_count=int(d.get("slice_count", 1)),
        artifact_dir=d.get("artifact_dir", "artifacts/xla_autotune"),
        dry_run=bool(d.get("dry_run", False)),
    )
