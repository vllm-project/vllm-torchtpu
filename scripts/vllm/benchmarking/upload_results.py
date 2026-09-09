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

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import bq_utils
from bq_utils import sql_escape

# GCP Spanner default coordinates
DEFAULT_PROJECT_ID = "cloud-tpu-inference-test"
DEFAULT_INSTANCE_ID = "vllm-bm-inst"
DEFAULT_DATABASE_ID = "vllm-bm-bk-runs"

RESULT_FILENAME_RE = re.compile(r"^isl(\d+)_osl(\d+)_c(\d+)\.json$")
ACC_KEYS = (
    "exact_match,strict_match",
    "exact_match,custom-extract",
    "exact_match,flexible-extract",
    "acc,none",
    "exact_match,none",
    "acc_norm,none",
    "pass@1,create_test",
    "pass_at_1,create_test",
    "pass_at_1,none",
)


def load_config_json(results_dir: Path) -> dict:
    config_path = results_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}", file=sys.stderr)
    return {}


def load_accuracy_metrics(results_dir: Path) -> dict:
    metrics = {}

    # 1. Parse LM-Eval results
    # Oldest-first so that if a task genuinely is re-run, the newer value wins.
    eval_jsons = sorted(results_dir.rglob("results_*.json"),
                        key=lambda p: p.stat().st_mtime)
    for path in eval_jsons:
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
            for task, task_metrics in (data.get("results") or {}).items():
                if not isinstance(task_metrics, dict):
                    continue
                # Find the primary accuracy metric
                for key in ACC_KEYS:
                    if key in task_metrics and isinstance(
                            task_metrics[key], (int, float)):
                        metrics[task] = float(task_metrics[key])
                        break
        except Exception as e:
            print(f"Warning: Failed to parse lm-eval results from {path}: {e}",
                  file=sys.stderr)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Upload performance benchmark results to Spanner and "
        "BigQuery (dual write during the migration to BigQuery).")
    parser.add_argument("--results-dir",
                        type=Path,
                        required=True,
                        help="Directory containing benchmark result files.")
    parser.add_argument(
        "--skip-db-upload",
        action="store_true",
        help="If set, only print SQL query without executing upload.")
    parser.add_argument("--project",
                        default=os.getenv("GCP_PROJECT_ID",
                                          DEFAULT_PROJECT_ID),
                        help="GCP project ID.")
    parser.add_argument("--instance",
                        default=os.getenv("GCP_INSTANCE_ID",
                                          DEFAULT_INSTANCE_ID),
                        help="Spanner instance ID.")
    parser.add_argument("--database",
                        default=os.getenv("GCP_DATABASE_ID",
                                          DEFAULT_DATABASE_ID),
                        help="Spanner database ID.")
    parser.add_argument("--bq-project",
                        default=None,
                        help="GCP project ID used to run BigQuery jobs "
                        "(default: BQ_PROJECT_ID env or "
                        f"{bq_utils.DEFAULT_BQ_PROJECT_ID}).")
    parser.add_argument("--bq-table",
                        default=None,
                        help="Fully-qualified BigQuery table "
                        "(default: BQ_TABLE env or "
                        f"{bq_utils.DEFAULT_BQ_TABLE}).")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: results-dir does not exist: {args.results_dir}",
              file=sys.stderr)
        return 1

    config = load_config_json(args.results_dir)
    accuracy_metrics = load_accuracy_metrics(args.results_dir)

    # Locate all result json files matching the pattern
    result_files = sorted(args.results_dir.glob("isl*_osl*_c*.json"))
    if not result_files:
        print(
            f"No perf result files (isl*_osl*_c*.json) found in {args.results_dir}."
        )
        return 0

    code_hash = os.getenv("BUILDKITE_COMMIT")
    if not code_hash:
        print("Error: BUILDKITE_COMMIT environment variable is not set.",
              file=sys.stderr)
        return 1

    device = os.getenv("TPU_NAME")
    if not device:
        queue = os.getenv("BUILDKITE_AGENT_META_DATA_QUEUE")
        if queue:
            # map tpu_v7x_8_queue -> tpu7x-8
            m = re.match(r"tpu_v(\d+x)_(\d+)_queue", queue)
            if m:
                device = f"tpu{m.group(1)}-{m.group(2)}"
            else:
                device = queue
        else:
            device = "unknown-device"

    run_by = os.getenv("GCP_INSTANCE_NAME")
    if not run_by:
        run_by = os.getenv("BUILDKITE_AGENT_NAME", "unknown-agent")

    job_ref = os.getenv("BUILDKITE_BUILD_NUMBER",
                        datetime.now().strftime("%Y%m%d_%H%M%S"))

    success = True

    for rf in result_files:
        m = RESULT_FILENAME_RE.match(rf.name)
        if not m:
            continue

        input_len = int(m.group(1))
        output_len = int(m.group(2))
        concurrency = int(m.group(3))

        try:
            with open(rf, "r") as f:
                res = json.load(f)
        except Exception as e:
            print(f"Error loading {rf}: {e}", file=sys.stderr)
            success = False
            continue

        try:
            model = config["model"]
            max_num_seqs = config["max_num_seqs"]
            max_num_batched_tokens = config["max_num_batched_tokens"]
            tensor_parallelism = config["tensor_parallelism"]
            max_model_len = config["max_model_len"]
        except KeyError as e:
            print(
                f"Error: Missing required config parameter {e} in config.json",
                file=sys.stderr)
            success = False
            continue

        try:
            dataset = res["backend"]
        except KeyError as e:
            print(f"Error: Missing required metric {e} in {rf.name}",
                  file=sys.stderr)
            success = False
            continue

        record_id = str(uuid.uuid4())

        # Populate column dictionary with required fields
        columns = {
            "RecordId": f"'{record_id}'",
            # run_in_docker.sh exports these as empty strings when unset
            # upstream; `or` falls back on empty too, unlike os.getenv's
            # unset-only default.
            "RunType": f"'{sql_escape(os.getenv('RUN_TYPE') or 'DAILY')}'",
            "MaxNumSeqs": str(max_num_seqs),
            "MaxNumBatchedTokens": str(max_num_batched_tokens),
            "TensorParallelSize": str(tensor_parallelism),
            "MaxModelLen": str(max_model_len),
            "Dataset": f"'{sql_escape(dataset)}'",
            "CreatedBy":
            f"'{sql_escape(os.getenv('CREATED_BY') or 'buildkite-agent')}'",
            "InputLen": str(input_len),
            "OutputLen": str(output_len),
            "Device": f"'{sql_escape(device)}'",
            "CodeHash": f"'{sql_escape(code_hash)}'",
            "Model": f"'{sql_escape(model)}'",
            "Status": "'COMPLETED'",
            "LastUpdate": "CURRENT_TIMESTAMP()",
            "CreatedTime": "CURRENT_TIMESTAMP()",
        }

        # Optional metrics (omit from columns list if missing, so Spanner defaults to NULL)
        optional_metrics = {
            "Throughput": res.get("request_throughput"),
            "OutputTokenThroughput": res.get("output_throughput"),
            "TotalTokenThroughput": res.get("total_token_throughput"),
            "MedianTTFT": res.get("median_ttft_ms"),
            "P99TTFT": res.get("p99_ttft_ms"),
            "MedianTPOT": res.get("median_tpot_ms"),
            "P99TPOT": res.get("p99_tpot_ms"),
            "MedianITL": res.get("median_itl_ms"),
            "P99ITL": res.get("p99_itl_ms"),
            "NumPrompts": res.get("num_prompts"),
        }

        for col_name, val in optional_metrics.items():
            if val is not None:
                columns[col_name] = str(val)

        if job_ref is not None:
            columns["JobReference"] = f"'{sql_escape(str(job_ref))}'"
        if run_by is not None:
            columns["RunBy"] = f"'{sql_escape(run_by)}'"

        # ExtraArgs construction
        extra_args_parts = []
        extra_args_parts.append(f"tp={tensor_parallelism}")
        dp = config.get("data_parallelism", 1)
        extra_args_parts.append(f"dp={dp}")
        if config.get("enable_ep"):
            extra_args_parts.append("enable_ep")
        if config.get("quantization"):
            extra_args_parts.append(f"quantization={config['quantization']}")
        extra_args_parts.append(f"concurrency={concurrency}")
        extra_args = " ".join(extra_args_parts)
        columns["ExtraArgs"] = f"'{sql_escape(extra_args)}'"

        # Add accuracy metrics if any were parsed
        if accuracy_metrics:
            columns[
                "AccuracyMetrics"] = f"JSON '{json.dumps(accuracy_metrics)}'"

        keys_str = ", ".join(columns.keys())
        vals_str = ", ".join(columns.values())
        sql = f"INSERT INTO RunRecord ({keys_str}) VALUES ({vals_str});"

        print(f"SQL for Spanner ({rf.name}):")
        print(sql)

        if args.skip_db_upload:
            print(
                f"=== Skipping Spanner DB Upload (--skip-db-upload specified) ===. RecordId: {record_id}"
            )
        else:
            cmd = [
                "gcloud", "spanner", "databases", "execute-sql", args.database,
                f"--project={args.project}", f"--instance={args.instance}",
                f"--sql={sql}"
            ]

            print(f"Executing: {' '.join(cmd)}")
            res_proc = subprocess.run(cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True)
            if res_proc.returncode != 0:
                print(f"Failed to update Spanner record for {rf.name}!",
                      file=sys.stderr)
                print(f"Stdout:\n{res_proc.stdout}", file=sys.stderr)
                print(f"Stderr:\n{res_proc.stderr}", file=sys.stderr)
                success = False
            else:
                print(
                    f"Successfully updated Spanner record for {rf.name}. RecordId: {record_id}"
                )

        # Dual write to BigQuery while dashboards migrate off Spanner. The
        # row shares the Spanner RecordId so the two writes can be
        # cross-referenced. Accuracy metrics are per-run but stamped onto
        # every cell row (like Spanner's AccuracyMetrics) — dedupe when
        # aggregating them across rows.
        bq_metrics = [("vllm_bench", bq_utils.extract_bench_metrics(res))]
        if accuracy_metrics:
            bq_metrics.append(("lm_eval", accuracy_metrics))
        bq_config = {
            "code_hash": code_hash,
            "created_by": os.getenv("CREATED_BY") or "buildkite-agent",
            "device_type": device,
            "run_by": run_by,
            "job_reference": str(job_ref),
            "profile_gcs_dir": config.get("profile_gcs_dir"),
            "backend": dataset,
            "engine_flags": {
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "tensor_parallel_size": tensor_parallelism,
                "data_parallel_size": config.get("data_parallelism", 1),
                "max_model_len": max_model_len,
                "enable_ep": bool(config.get("enable_ep")),
                "quantization": config.get("quantization"),
                "multihost_backend": os.getenv("TPU_MULTIHOST_BACKEND")
                or None,
            },
            "workload": {
                "input_len": input_len,
                "output_len": output_len,
                "num_prompts": res.get("num_prompts"),
                "request_rate": res.get("request_rate"),
                "burstiness": res.get("burstiness"),
                "max_concurrency": res.get("max_concurrency", concurrency),
            },
        }
        bq_sql = bq_utils.build_insert_sql(
            record_id=record_id,
            created_time=bq_utils.created_time_sql(res.get("date")),
            run_type=os.getenv("RUN_TYPE") or "DAILY",
            model_id=model,
            metrics=bq_metrics,
            config=bq_config,
            bq_table=args.bq_table)
        if not bq_utils.run_insert(bq_sql,
                                   project=args.bq_project,
                                   label=rf.name,
                                   record_id=record_id,
                                   skip=args.skip_db_upload):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
