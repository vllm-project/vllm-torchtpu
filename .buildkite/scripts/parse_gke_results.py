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
without executing it.

--prefill-chips and --decode-chips state the physical chips of the two
engine pools. With them the row carries both counts in its config and
per-chip throughput (output over decode chips, input over prefill chips,
and total over their sum)."""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

# The shared BigQuery and metric helpers live with the benchmarking scripts
# (which also ship in the CI image); this script runs on the Buildkite agent
# host.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm" / "benchmarking")
)
import bench_metrics
import bq_utils


def parse_and_upload(
    file_path, record_id, dry_run=False, prefill_chips=None, decode_chips=None
) -> bool:
    chips = bench_metrics.physical_chips(
        prefill_chips=prefill_chips, decode_chips=decode_chips
    )
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
            f"Warning: Could not parse input/output len from filename: {base_name}. "
            "Using None.",
            file=sys.stderr,
        )

    success = True
    with open(file_path) as f:
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
                # The build number, the key upload_results.py rows carry too;
                # the launcher forwards BUILDKITE_* into every pod.
                "job_reference": os.getenv("BUILDKITE_BUILD_NUMBER") or "N/A",
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

            if chips is not None:
                # Both pools, since their sum alone cannot tell a 4 + 4 fleet
                # from a 6 + 2 one.
                config_dict["prefill_chips"] = prefill_chips
                config_dict["decode_chips"] = decode_chips

            metrics = [("vllm_bench", bq_utils.extract_bench_metrics(data))]
            derived = bench_metrics.bench_derived_metrics(
                data, prefill_chips=prefill_chips, decode_chips=decode_chips
            )
            if derived:
                metrics.append(("derived", derived))

            sql = bq_utils.build_insert_sql(
                record_id=unique_record_id,
                created_time=bq_utils.created_time_sql(data.get("date")),
                run_type=os.getenv("RUN_TYPE") or "GKE_DISAGG",
                model_id=model_id,
                metrics=metrics,
                config=config_dict,
            )
            if not bq_utils.run_insert(
                sql, label=base_name, record_id=unique_record_id, skip=dry_run
            ):
                success = False

    return success


def _chip_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        count = 0
    if count <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_path", help="Result file of one benchmark cell")
    parser.add_argument("record_id", help="Prefix of each row's record_id")
    parser.add_argument(
        "--prefill-chips", type=_chip_count, help="Chips serving the prefill pool"
    )
    parser.add_argument(
        "--decode-chips", type=_chip_count, help="Chips serving the decode pool"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the SQL without executing it"
    )
    args = parser.parse_args(argv)
    if (args.prefill_chips is None) != (args.decode_chips is None):
        parser.error(
            "--prefill-chips and --decode-chips go together: a disaggregated "
            "run is served by both pools"
        )
    ok = parse_and_upload(
        args.file_path,
        args.record_id,
        dry_run=args.dry_run,
        prefill_chips=args.prefill_chips,
        decode_chips=args.decode_chips,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
