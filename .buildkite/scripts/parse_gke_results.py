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
"""Parse a GKE disagg benchmark result file and insert its rows into
BigQuery. Exits nonzero if any insert fails; --dry-run prints the SQL
without executing it."""

import json
import os
import re
import sys
import uuid
from pathlib import Path

# The shared BigQuery helpers live with the benchmarking scripts (which also
# ship in the CI image); this script runs on the Buildkite agent host.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking")
)
import bq_utils  # noqa: E402


def parse_and_upload(file_path, record_id, dry_run=False) -> bool:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        return False

    # Extract input_len and output_len from a
    # pd_disagg_c{concurrency}_i{input}_o{output}.json filename.
    base_name = os.path.basename(file_path)
    name_part, _ = os.path.splitext(base_name)
    input_len = None
    output_len = None
    match = re.search(r"_i(\d+)_o(\d+)$", name_part)
    if match:
        input_len = int(match.group(1))
        output_len = int(match.group(2))
    else:
        print(
            f"Warning: Could not parse input/output len from filename: {base_name}. Using None.",
            file=sys.stderr,
        )

    success = True
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Error: Could not decode JSON from line: {line}", file=sys.stderr
                )
                success = False
                continue

            # request_rate is JSON null when the benchmark ran at --request-rate=inf.
            rate = data.get("request_rate")
            rate = "inf" if rate is None else rate
            concurrency = data.get("max_concurrency", "unknown")
            short_suffix = uuid.uuid4().hex[:6]

            unique_record_id = f"{record_id}_{rate}_c{concurrency}_{short_suffix}"
            model_id = data.get("model_id", "Qwen/Qwen3.5-397B-A17B-FP8")

            # Build configuration & workload payload
            config_dict = {
                "code_hash": os.getenv("BUILDKITE_COMMIT") or "N/A",
                "image": os.getenv("DOCKER_IMAGE") or "latest",
                "created_by": os.getenv("CREATED_BY") or "buildkite",
                "device_type": os.getenv("TPU_VERSION") or "tpu7x",
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
                },
            }

            sql = bq_utils.build_insert_sql(
                record_id=unique_record_id,
                created_time=bq_utils.created_time_sql(data.get("date")),
                run_type=os.getenv("RUN_TYPE") or "GKE_DISAGG",
                model_id=model_id,
                metrics=[("vllm_bench", bq_utils.extract_bench_metrics(data))],
                config=config_dict,
            )
            if not bq_utils.run_insert(
                sql, label=base_name, record_id=unique_record_id, skip=dry_run
            ):
                success = False

    return success


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if len(args) < 2:
        print(
            "Usage: python parse_gke_results.py <file_path> <record_id> [--dry-run]",
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(
        0 if parse_and_upload(args[0], args[1], dry_run="--dry-run" in sys.argv) else 1
    )
