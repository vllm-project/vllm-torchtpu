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
"""Pluggable benchmark runners and metric parsers for XLA autotuning."""

from __future__ import annotations

import abc
import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from typing import Any

from .config import BenchmarkConfig, ModelConfig

logger = logging.getLogger(__name__)


def parse_vllm_bench_serve_stdout(stdout_text: str) -> dict[str, Any]:
    """Parse key metrics from vllm bench serve stdout summary table or text."""
    metrics: dict[str, Any] = {}

    patterns = {
        "request_throughput": [
            r"Request\s+throughput(?:\s+\(req/s\))?:\s+([\d\.]+)",
            r"Requests\s+per\s+second(?:\s+\(req/s\))?:\s+([\d\.]+)",
            r"request_throughput:\s+([\d\.]+)",
        ],
        "output_throughput": [
            r"Output\s+token\s+throughput(?:\s+\(tok/s\))?:\s+([\d\.]+)",
            r"Output\s+tokens\s+per\s+second(?:\s+\(tok/s\))?:\s+([\d\.]+)",
            r"output_throughput:\s+([\d\.]+)",
        ],
        "total_token_throughput": [
            r"Total\s+token\s+throughput(?:\s+\(tok/s\))?:\s+([\d\.]+)",
            r"Total\s+tokens\s+per\s+second(?:\s+\(tok/s\))?:\s+([\d\.]+)",
            r"total_token_throughput:\s+([\d\.]+)",
        ],
        "mean_ttft_ms": [
            r"Mean\s+TTFT(?:\s+\(ms\))?:\s+([\d\.]+)",
            r"mean_ttft_ms:\s+([\d\.]+)",
        ],
        "median_ttft_ms": [
            r"Median\s+TTFT(?:\s+\(ms\))?:\s+([\d\.]+)",
            r"median_ttft_ms:\s+([\d\.]+)",
        ],
        "p99_ttft_ms": [
            r"P99\s+TTFT(?:\s+\(ms\))?:\s+([\d\.]+)",
            r"p99_ttft_ms:\s+([\d\.]+)",
        ],
        "mean_tpot_ms": [
            r"Mean\s+TPOT(?:\s+\(ms\))?:\s+([\d\.]+)",
            r"mean_tpot_ms:\s+([\d\.]+)",
        ],
        "mean_itl_ms": [
            r"Mean\s+ITL(?:\s+\(ms\))?:\s+([\d\.]+)",
            r"mean_itl_ms:\s+([\d\.]+)",
        ],
    }

    for metric_name, regex_list in patterns.items():
        for regex in regex_list:
            match = re.search(regex, stdout_text, re.IGNORECASE)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                    break
                except ValueError:
                    pass

    return metrics


class BaseBenchmarkRunner(abc.ABC):
    """Abstract base benchmark runner."""

    def __init__(
        self, model_cfg: ModelConfig, bench_cfg: BenchmarkConfig, dry_run: bool = False
    ):
        self.model_cfg = model_cfg
        self.bench_cfg = bench_cfg
        self.dry_run = dry_run

    @abc.abstractmethod
    def run_benchmark(self, trial_dir: str, env: dict[str, str]) -> dict[str, Any]:
        """Execute benchmark and return parsed metrics."""
        raise NotImplementedError


class VllmBenchServeRunner(BaseBenchmarkRunner):
    """Runner for standard `vllm bench serve`."""

    def run_benchmark(self, trial_dir: str, env: dict[str, str]) -> dict[str, Any]:
        if self.dry_run:
            logger.info("[dry-run] Simulating vllm bench serve")
            return {
                "request_throughput": 12.5,
                "output_throughput": 250.0,
                "total_token_throughput": 1250.0,
                "mean_ttft_ms": 45.2,
                "mean_tpot_ms": 4.1,
            }

        os.makedirs(trial_dir, exist_ok=True)
        passes = max(0, self.bench_cfg.warmup_runs) + 1
        base_url = f"http://127.0.0.1:{self.model_cfg.port}"
        model_name = self.model_cfg.name

        latest_metrics: dict[str, Any] = {}
        for p in range(passes):
            is_measured = p == passes - 1
            pass_type = "measured" if is_measured else f"warmup_{p + 1}"
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            result_filename = f"bench_{pass_type}_{ts}.json"
            result_path = os.path.join(trial_dir, result_filename)
            log_path = os.path.join(trial_dir, f"bench_{pass_type}_{ts}.log")

            cmd = [
                "vllm",
                "bench",
                "serve",
                "--model",
                model_name,
                "--base-url",
                base_url,
                "--save-result",
                "--result-dir",
                trial_dir,
                "--result-filename",
                result_filename,
            ]

            # Format benchmark arguments from config
            for k, v in self.bench_cfg.args.items():
                flag = k.replace("_", "-")
                flag_str = f"--{flag}" if not flag.startswith("--") else flag
                if isinstance(v, bool):
                    if v:
                        cmd.append(flag_str)
                elif v is not None:
                    cmd.append(f"{flag_str}={v}")

            logger.info(
                "Executing %s pass [%d/%d]: %s", pass_type, p + 1, passes, " ".join(cmd)
            )

            with open(log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                )
                captured_lines = []
                for line in iter(proc.stdout.readline, ""):
                    log_f.write(line)
                    log_f.flush()
                    captured_lines.append(line)
                proc.wait()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"vllm bench serve failed with return code {proc.returncode}. See log: {log_path}"
                )

            # Try loading JSON result file first
            metrics: dict[str, Any] = {}
            if os.path.exists(result_path):
                try:
                    with open(result_path, encoding="utf-8") as jf:
                        metrics = json.load(jf)
                except Exception as e:
                    logger.warning("Could not read result JSON %s: %s", result_path, e)

            # Supplement / fallback with stdout table parsing
            stdout_full = "".join(captured_lines)
            parsed_stdout = parse_vllm_bench_serve_stdout(stdout_full)
            for k, val in parsed_stdout.items():
                if k not in metrics or metrics[k] is None:
                    metrics[k] = val

            if is_measured:
                latest_metrics = metrics

        return latest_metrics


class CustomCommandRunner(BaseBenchmarkRunner):
    """Runner for arbitrary benchmark shell commands."""

    def run_benchmark(self, trial_dir: str, env: dict[str, str]) -> dict[str, Any]:
        if self.dry_run:
            logger.info("[dry-run] Simulating custom benchmark command")
            return {
                "request_throughput": 10.0,
                "total_token_throughput": 1000.0,
            }

        cmd_template = self.bench_cfg.custom_cmd
        if not cmd_template:
            raise ValueError(
                "custom_cmd must be specified in benchmark config for CustomCommandRunner"
            )

        os.makedirs(trial_dir, exist_ok=True)
        base_url = f"http://127.0.0.1:{self.model_cfg.port}"
        model_name = self.model_cfg.name
        port = str(self.model_cfg.port)

        cmd_rendered = cmd_template.format(
            model=model_name,
            base_url=base_url,
            port=port,
            trial_dir=trial_dir,
        )

        log_path = os.path.join(trial_dir, "custom_benchmark.log")
        logger.info("Executing custom benchmark command: %s", cmd_rendered)

        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd_rendered,
                shell=True,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            captured_lines = []
            for line in iter(proc.stdout.readline, ""):
                log_f.write(line)
                log_f.flush()
                captured_lines.append(line)
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Custom benchmark command failed with code {proc.returncode}. Log: {log_path}"
            )

        stdout_full = "".join(captured_lines)
        return parse_vllm_bench_serve_stdout(stdout_full)


def get_benchmark_runner(
    model_cfg: ModelConfig,
    bench_cfg: BenchmarkConfig,
    dry_run: bool = False,
) -> BaseBenchmarkRunner:
    """Factory to create appropriate benchmark runner."""
    if bench_cfg.type == "vllm_bench_serve":
        return VllmBenchServeRunner(model_cfg, bench_cfg, dry_run=dry_run)
    elif bench_cfg.type == "custom_command":
        return CustomCommandRunner(model_cfg, bench_cfg, dry_run=dry_run)
    else:
        raise ValueError(f"Unknown benchmark runner type: {bench_cfg.type}")
