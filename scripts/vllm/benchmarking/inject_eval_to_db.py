#!/usr/bin/env python3
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
"""Stage MMLU-Pro evaluation results and serving performance metrics to GCS staging bucket."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_GCS_BUCKET = "cloud-tpu-inference-eval-staging"
DEFAULT_GCS_PREFIX = "incoming"

MMLU_PRO_DOMAINS = (
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)

ACC_KEYS = (
    "exact_match,custom-extract",
    "exact_match,none",
    "acc,none",
    "exact_match,strict_match",
    "exact_match,flexible-extract",
    "acc_norm,none",
    "exact_match",
    "acc",
)

STDERR_KEYS = (
    "exact_match_stderr,custom-extract",
    "exact_match_stderr,none",
    "acc_stderr,none",
    "exact_match_stderr,strict_match",
    "exact_match_stderr,flexible-extract",
    "acc_norm_stderr,none",
    "exact_match_stderr",
    "acc_stderr",
)


def get_default_username() -> str:
    try:
        return os.environ.get("USER") or getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown_user")


def load_config_json(results_dir: Path) -> dict[str, Any]:
    config_file = results_dir / "config.json"
    if not config_file.exists():
        return {}
    try:
        with config_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load config.json from {config_file}: {e}",
              file=sys.stderr)
        return {}


def find_latest_results_json(results_dir: Path) -> Path | None:
    matched = list(results_dir.glob("results_*.json")) + list(
        results_dir.glob("*/results_*.json"))
    if not matched:
        return None
    matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matched[0]


def find_latest_perf_json(results_dir: Path) -> Path | None:
    matched = list(results_dir.glob("isl*_osl*_c*.json")) + list(
        results_dir.glob("*/isl*_osl*_c*.json"))
    if not matched:
        return None
    matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matched[0]


def extract_metric(d: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (ValueError, TypeError):
                pass
    return None


def extract_mmlu_pro_data(
        results_dir: Path) -> tuple[dict[str, Any], dict[str, float]] | None:
    results_file = find_latest_results_json(results_dir)
    if not results_file:
        return None

    try:
        with results_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load results json from {results_file}: {e}",
              file=sys.stderr)
        return None

    results = data.get("results", {})
    if not results:
        return None

    overall_acc: float | None = None
    overall_stderr: float | None = None

    for target in ("mmlu_pro", "mmlu-pro", "leaderboard_mmlu_pro"):
        if target in results and isinstance(results[target], dict):
            entry = results[target]
            overall_acc = extract_metric(entry, ACC_KEYS)
            overall_stderr = extract_metric(entry, STDERR_KEYS)
            break

    domain_submetrics: dict[str, float] = {}
    for task_name, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue
        for domain in MMLU_PRO_DOMAINS:
            if task_name == f"mmlu_pro_{domain}" or task_name == domain or task_name.endswith(
                    f"_{domain}"):
                val = extract_metric(task_metrics, ACC_KEYS)
                if val is not None:
                    domain_submetrics[domain] = val
                break

    if overall_acc is None:
        if domain_submetrics:
            overall_acc = sum(
                domain_submetrics.values()) / len(domain_submetrics)
        else:
            return None

    overall_accuracy = {
        "acc": round(overall_acc, 4),
        "stderr":
        round(overall_stderr, 4) if overall_stderr is not None else 0.0,
    }
    return overall_accuracy, domain_submetrics


def extract_serving_metrics(results_dir: Path,
                            config: dict[str, Any]) -> dict[str, Any]:
    perf_file = find_latest_perf_json(results_dir)
    tp = int(
        config.get("tensor_parallelism", config.get("tensor_parallel_size",
                                                    1)))
    default_trace_path = f"{results_dir}/traces/trace.json"

    if not perf_file:
        return {
            "time_to_first_token_s": 0.0,
            "latency_decode_per_token_s": 0.0,
            "total_throughput_tok_s": 0.0,
            "qps_per_chip": 0.0,
            "latency_combined_s": 0.0,
            "trace_path": default_trace_path,
        }

    try:
        with perf_file.open("r", encoding="utf-8") as f:
            perf_data = json.load(f)

        ttft_s = float(perf_data.get("median_ttft_ms", 0.0)) / 1000.0
        tpot_s = float(perf_data.get("median_tpot_ms", 0.0)) / 1000.0
        total_throughput = float(
            perf_data.get("output_throughput", 0.0)
            or perf_data.get("total_token_throughput", 0.0))
        qps_per_chip = total_throughput / tp
        latency_combined_s = float(
            perf_data.get("elapsed_time", 0.0)
            or perf_data.get("duration", 0.0))

        return {
            "time_to_first_token_s": round(ttft_s, 6),
            "latency_decode_per_token_s": round(tpot_s, 6),
            "total_throughput_tok_s": round(total_throughput, 4),
            "qps_per_chip": round(qps_per_chip, 4),
            "latency_combined_s": round(latency_combined_s, 4),
            "trace_path": default_trace_path,
        }
    except Exception as e:
        print(f"Warning: Failed to load/parse perf json from {perf_file}: {e}",
              file=sys.stderr)
        return {
            "time_to_first_token_s": 0.0,
            "latency_decode_per_token_s": 0.0,
            "total_throughput_tok_s": 0.0,
            "qps_per_chip": 0.0,
            "latency_combined_s": 0.0,
            "trace_path": default_trace_path,
        }


def get_hardware() -> str:
    if os.getenv("HARDWARE"):
        return os.getenv("HARDWARE")
    tpu_type = os.getenv("TPU_TYPE") or os.getenv("TPU_ACCELERATOR_TYPE")
    if tpu_type:
        return tpu_type if tpu_type.startswith("TPU_") else f"TPU_{tpu_type}"
    tpu_name = os.getenv("TPU_NAME", "")
    queue = os.getenv("BUILDKITE_AGENT_META_DATA_QUEUE", "")
    hostname = socket.gethostname()
    for candidate in (tpu_name, queue, hostname):
        m = re.search(r"v(\d+[a-z]?)[-_](\d+)", candidate, re.IGNORECASE)
        if m:
            return f"TPU_v{m.group(1)}-{m.group(2)}"
    return "TPU_v6e-8"


def get_quantization_mode(config: dict[str, Any]) -> str:
    if os.getenv("QUANTIZATION_MODE"):
        return os.getenv("QUANTIZATION_MODE")
    if "quantization_mode" in config:
        return str(config["quantization_mode"])
    quant = str(config.get("quantization", "none")).lower()
    if quant in ("none", "", "null"):
        return "FP8_KV_BF16_WEIGHTS"
    if "fp8" in quant:
        return "FP8_KV_FP8_WEIGHTS"
    if "nvfp4" in quant or "fp4" in quant:
        return "FP8_KV_NVFP4_WEIGHTS"
    return quant.upper()


def build_payload(
    results_dir: Path,
    config: dict[str, Any],
    serving_metrics: dict[str, Any],
    overall_accuracy: dict[str, Any],
    domain_submetrics: dict[str, float],
    uploader: str,
) -> dict[str, Any]:
    model = config.get("model", "unknown-model")
    sanitized_model = model.replace("/", "_")
    hardware = get_hardware()
    tp = int(
        config.get("tensor_parallelism", config.get("tensor_parallel_size",
                                                    1)))
    dp = int(
        config.get("data_parallelism", config.get("data_parallel_size", 1)))
    quant_mode = get_quantization_mode(config)
    sparsity_mode = config.get("sparsity_mode", "DENSE")
    test_plan_run_id = os.getenv(
        "TEST_PLAN_RUN_ID") or f"TPR_{sanitized_model}_{hardware}_MMLUPRO"
    timestamp = config.get("timestamp") or datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    create_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    vm_raw_path = os.path.abspath(str(results_dir))

    payload = {
        "TestPlanRunID": test_plan_run_id,
        "uploader": uploader,
        "Environment": os.getenv("ENVIRONMENT", "ENV_TPU"),
        "model": model,
        "hardware": hardware,
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "quantization_mode": quant_mode,
        "sparsity_mode": sparsity_mode,
        "create_time": create_time,
        "timestamp": timestamp,
        "vm_raw_path": vm_raw_path,
        "serving_metrics": serving_metrics,
        "overall_accuracy": overall_accuracy,
        "domain_submetrics": {
            k: round(domain_submetrics[k], 4)
            for k in MMLU_PRO_DOMAINS if k in domain_submetrics
        },
    }
    return payload


def upload_payload_to_gcs(
    payload: dict[str, Any],
    gcs_bucket: str,
    gcs_prefix: str,
    uploader: str,
) -> None:
    test_plan_run_id = payload.get("TestPlanRunID", "payload")
    prefix = gcs_prefix.strip("/")
    if prefix and uploader:
        target_dir = f"{prefix}/{uploader}"
    elif prefix:
        target_dir = prefix
    elif uploader:
        target_dir = uploader
    else:
        target_dir = ""

    target_path = f"{target_dir}/{test_plan_run_id}.json" if target_dir else f"{test_plan_run_id}.json"
    payload_str = json.dumps(payload, indent=2)

    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(target_path)
        blob.upload_from_string(payload_str, content_type="application/json")
        print(
            f"✓ Successfully uploaded evaluation payload to: gs://{gcs_bucket}/{target_path}"
        )
        return
    except ImportError:
        pass
    except Exception as e:
        print(
            f"google-cloud-storage upload failed ({e}), falling back to gcloud storage CLI...",
            file=sys.stderr)

    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp.write(payload_str)
        tmp_path = tmp.name

    try:
        cmd = [
            "gcloud", "storage", "cp", tmp_path,
            f"gs://{gcs_bucket}/{target_path}"
        ]
        subprocess.run(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       text=True,
                       check=True)
        print(
            f"✓ Successfully uploaded evaluation payload to: gs://{gcs_bucket}/{target_path}"
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=
        "Stage MMLU-Pro evaluation results and serving performance metrics to GCS staging bucket."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=
        "Directory containing benchmark result files (isl*_osl*_c*.json, results_*.json, and config.json).",
    )
    parser.add_argument(
        "--uploader",
        default=os.getenv("UPLOADER") or get_default_username(),
        help=
        "Engineer username for JSON payload and user-namespaced GCS folder (default: current system user).",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=os.getenv("GCS_STAGING_BUCKET", DEFAULT_GCS_BUCKET),
        help=f"GCS staging bucket name (default: {DEFAULT_GCS_BUCKET}).",
    )
    parser.add_argument(
        "--gcs-prefix",
        default=os.getenv("GCS_STAGING_PREFIX", DEFAULT_GCS_PREFIX),
        help=f"GCS destination prefix/folder (default: {DEFAULT_GCS_PREFIX}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=
        "If set, print the structured JSON mutation payload to stdout without uploading to GCS.",
    )
    parser.add_argument(
        "--skip-db-upload",
        action="store_true",
        help="If set, skip uploading payload to GCS staging bucket.",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Warning: results-dir does not exist: {args.results_dir}",
              file=sys.stderr)
        return 0

    config = load_config_json(args.results_dir)
    extracted = extract_mmlu_pro_data(args.results_dir)
    if extracted is None:
        print(
            f"Warning: No MMLU-Pro results found in {args.results_dir}. Skipping GCS upload.",
            file=sys.stderr)
        return 0

    overall_accuracy, domain_submetrics = extracted
    serving_metrics = extract_serving_metrics(args.results_dir, config)
    payload = build_payload(
        args.results_dir,
        config,
        serving_metrics,
        overall_accuracy,
        domain_submetrics,
        uploader=args.uploader,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    if args.skip_db_upload:
        print("=== Skipping GCS Upload (--skip-db-upload specified) ===")
        return 0

    try:
        upload_payload_to_gcs(
            payload,
            args.gcs_bucket,
            args.gcs_prefix,
            uploader=args.uploader,
        )
    except Exception as e:
        print(
            f"Warning: Failed to upload evaluation payload to gs://{args.gcs_bucket}: {e}",
            file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
