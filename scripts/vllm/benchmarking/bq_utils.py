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
"""Shared helpers for inserting benchmark rows into BigQuery.

Table schema: record_id, created_time, repository, run_type, model_id,
metrics ARRAY<STRUCT<type STRING, values JSON>>, config JSON. Each
metrics element's type names the tool that produced its payload (e.g.
'vllm_bench', 'lm_eval'); values keeps that tool's flat leaf keys
verbatim, so generic names (completed, failed, duration) can't clash with
other harnesses writing to the shared table and provenance stays explicit.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

DEFAULT_BQ_PROJECT_ID = "cloud-ullm-inference-ci-cd"
DEFAULT_BQ_TABLE = "cloud-ullm-inference-ci-cd.llm_benchmark_analytics.benchmark_runs"

# Leaf keys exactly as `vllm bench serve` emits them.
BENCH_METRIC_FIELDS = (
    "request_throughput",
    "request_goodput",
    "output_throughput",
    "total_token_throughput",
    "max_output_tokens_per_s",
    "rtfx",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "completed",
    "failed",
    "total_input_tokens",
    "total_output_tokens",
    "duration",
)


def sql_escape(val: str) -> str:
    return val.replace("'", "''")


def extract_bench_metrics(data: dict) -> dict:
    """The `vllm bench serve` payload, with absent metrics omitted so the
    JSON stays compact."""
    return {k: data[k] for k in BENCH_METRIC_FIELDS if data.get(k) is not None}


def created_time_sql(date_str) -> str:
    """TIMESTAMP literal from a `vllm bench serve` `date` stamp (measurement
    time), falling back to insert time."""
    if date_str:
        try:
            dt = datetime.strptime(date_str, "%Y%m%d-%H%M%S")
            return f"TIMESTAMP '{dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        except ValueError:
            print(
                f"Warning: Could not parse date string: {date_str}. "
                "Using CURRENT_TIMESTAMP()",
                file=sys.stderr,
            )
    return "CURRENT_TIMESTAMP()"


def build_insert_sql(
    record_id: str,
    created_time: str,
    run_type: str,
    model_id: str,
    metrics: list,
    config: dict,
    bq_table=None,
) -> str:
    """Render one INSERT statement for the benchmark_runs table.

    `metrics` is a list of (type, payload-dict) pairs, one array element
    per producing tool. `created_time` is a SQL expression (see
    created_time_sql). `bq_table` falls back to the BQ_TABLE env var, then
    the default; pipelines may export BQ_TABLE as an empty string, so `or`
    (empty falls through) is used instead of os.getenv's unset-only
    default. Backticks around an override are accepted and normalized.
    """
    table = (bq_table or os.getenv("BQ_TABLE") or DEFAULT_BQ_TABLE).strip("`")
    repository = os.getenv("REPOSITORY") or "vllm-torchtpu"
    config_json = sql_escape(json.dumps(config))
    metrics_elems = ", ".join(
        f"STRUCT('{metrics_type}', "
        f"PARSE_JSON('{sql_escape(json.dumps(payload))}', "
        "wide_number_mode=>'round'))"
        for metrics_type, payload in metrics
    )
    sql = f"""
    INSERT INTO `{table}` (
        record_id, created_time, repository, run_type, model_id, metrics, config
    ) VALUES (
        '{sql_escape(record_id)}', {created_time},
        '{sql_escape(repository)}', '{sql_escape(run_type)}',
        '{sql_escape(model_id)}',
        [{metrics_elems}],
        PARSE_JSON('{config_json}', wide_number_mode=>'round')
    );
    """
    # Single line so the statement survives shell/CLI round-trips intact.
    return " ".join(sql.split())


def run_insert(
    sql: str, project=None, label: str = "", record_id: str = "", skip: bool = False
) -> bool:
    """Execute one INSERT via the bq CLI. Prints the statement either way;
    with skip=True it is printed but not executed."""
    print(f"SQL for BigQuery ({label}):")
    print(sql)

    if skip:
        print(f"=== Skipping BigQuery upload ===. record_id: {record_id}")
        return True

    project = project or os.getenv("BQ_PROJECT_ID") or DEFAULT_BQ_PROJECT_ID
    cmd = ["bq", "query", "--use_legacy_sql=false", f"--project_id={project}", sql]
    print(f"Executing: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        print(
            f"Failed to insert BigQuery record for {label}: 'bq' CLI not found.",
            file=sys.stderr,
        )
        return False
    if proc.returncode != 0:
        print(f"Failed to insert BigQuery record for {label}!", file=sys.stderr)
        print(f"Stdout:\n{proc.stdout}", file=sys.stderr)
        print(f"Stderr:\n{proc.stderr}", file=sys.stderr)
        return False
    print(f"Successfully inserted BigQuery record for {label}. record_id: {record_id}")
    return True
