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

import json
import os
import re
import sys
import uuid
from datetime import datetime


def parse_and_dump(file_path, record_id):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Extract input_len and output_len from a
    # pd_disagg_c{concurrency}_i{input}_o{output}.json filename.
    base_name = os.path.basename(file_path)
    name_part, _ = os.path.splitext(base_name)
    input_len = None
    output_len = None
    match = re.search(r'_i(\d+)_o(\d+)$', name_part)
    if match:
        input_len = int(match.group(1))
        output_len = int(match.group(2))
    else:
        print(
            f"Warning: Could not parse input/output len from filename: {base_name}. Using None.",
            file=sys.stderr)

    # The pipeline exports BQ_TABLE as empty when no override is given;
    # os.getenv's default only applies when the variable is unset, so fall
    # back on empty too.
    bq_table = os.getenv("BQ_TABLE") or (
        "`cloud-ullm-inference-ci-cd.llm_benchmark_analytics.benchmark_runs`")
    repository = os.getenv("REPOSITORY", "vllm-torchtpu")
    run_type = os.getenv("RUN_TYPE", "GKE_DISAGG")

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from line: {line}",
                      file=sys.stderr)
                continue

            # request_rate is JSON null when the benchmark ran at --request-rate=inf.
            rate = data.get('request_rate')
            rate = 'inf' if rate is None else rate
            concurrency = data.get('max_concurrency', 'unknown')
            short_suffix = uuid.uuid4().hex[:6]

            # Parse date from JSON
            date_str = data.get('date')
            bq_timestamp = 'CURRENT_TIMESTAMP()'
            if date_str:
                try:
                    dt = datetime.strptime(date_str, '%Y%m%d-%H%M%S')
                    bq_timestamp = f"TIMESTAMP '{dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
                except ValueError:
                    print(
                        f"Warning: Could not parse date string: {date_str}. Using CURRENT_TIMESTAMP()",
                        file=sys.stderr)

            unique_record_id = f"{record_id}_{rate}_c{concurrency}_{short_suffix}"
            model_id = data.get('model_id', 'Qwen/Qwen3.5-397B-A17B-FP8')

            # Build performance metrics payload. All fields sit under a
            # single parent named for the tool that produced them, so generic
            # names (completed, failed, duration) can't clash with other
            # harnesses writing to the shared table and provenance is explicit;
            # leaf keys are exactly vLLM's bench-serving output names.
            bench_fields = [
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
            ]
            # Flat payload; the tool that produced it goes in the typed
            # metrics.type field, not a JSON wrapper. Omit None values to
            # keep JSON compact.
            metrics_dict = {
                k: data.get(k)
                for k in bench_fields if data.get(k) is not None
            }

            # Build configuration & workload payload
            config_dict = {
                "code_hash": os.getenv("BUILDKITE_COMMIT", "N/A"),
                "image": os.getenv("DOCKER_IMAGE", "latest"),
                "created_by": os.getenv("CREATED_BY", "buildkite"),
                "device_type": os.getenv("TPU_VERSION", "tpu7x"),
                "backend": data.get("backend", "vllm"),
                "endpoint_type": data.get("endpoint_type", "disagg-proxy"),
                "workload": {
                    "input_len": input_len,
                    "output_len": output_len,
                    "num_prompts": data.get("num_prompts"),
                    "request_rate": data.get("request_rate"),
                    "burstiness": data.get("burstiness"),
                    "max_concurrency": data.get("max_concurrency"),
                },
                "engine_flags": {
                    "tokenizer_id": data.get("tokenizer_id"),
                    "label": data.get("label"),
                }
            }

            metrics_json_str = json.dumps(metrics_dict).replace("'", "''")
            config_json_str = json.dumps(config_dict).replace("'", "''")

            sql = f"""
            INSERT INTO {bq_table} (
                record_id, created_time, repository, run_type, model_id, metrics, config
            ) VALUES (
                '{unique_record_id}', {bq_timestamp}, '{repository}', '{run_type}', '{model_id}',
                STRUCT('vllm_bench',
                       PARSE_JSON('{metrics_json_str}',
                                  wide_number_mode=>'round')),
                PARSE_JSON('{config_json_str}', wide_number_mode=>'round')
            );
            """

            # Print single-line SQL for bash execution
            print(" ".join(sql.split()))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_gke_results.py <file_path> <record_id>",
              file=sys.stderr)
        sys.exit(1)
    parse_and_dump(sys.argv[1], sys.argv[2])
