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
"""Publish run_benchmarks.sh results to the Performance tab of ci.vllm.ai.

The tab reads the `vllm_perf_data_ingest` table, which is filled by an
authenticated ingest endpoint shared with the GPU perf pipelines
(vllm-project/perf-eval lib/ingest_perf.py). Each `isl*_osl*_c*.json` under a
results directory becomes one row in that schema: latencies in seconds and
throughput divided by the number of chips, which the dashboard labels
"per GPU".

The dashboard keys a series by device, `tp` and concurrency, so `tp` is the
layout's tensor parallelism and `dp` rides along: the layouts of one model
on one slice stay separate series.
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://vllm-perf-data-ingest-224810116257.us-central1.run.app/"
TOKEN_ENV = "INGEST_BEARER_TOKEN"
RESULT_FILENAME_RE = re.compile(r"^isl(\d+)_osl(\d+)_c(\d+)\.json$")


def to_row(
    raw: dict,
    config: dict,
    *,
    isl: int,
    osl: int,
    conc: int,
    device: str,
    chips: int,
    image: str,
) -> dict:
    tp = int(config["tensor_parallelism"])
    dp = int(config.get("data_parallelism") or 1)
    total_tput = float(raw["total_token_throughput"])
    output_tput = float(raw["output_throughput"])
    row = {
        "date": datetime.datetime.strptime(raw["date"], "%Y%m%d-%H%M%S").strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "device": device,
        "conc": int(raw.get("max_concurrency") or conc),
        "image": image,
        "model": raw.get("model_id") or config["model"],
        "framework": "vllm",
        "precision": config.get("quantization") or "bf16",
        "spec_decoding": "false",
        "disagg": "false",
        "isl": isl,
        "osl": osl,
        "is_multinode": "false",
        "tp": tp,
        "dp": dp,
        "ep": tp * dp if config.get("enable_ep") else 1,
        "dp_attention": "true" if dp > 1 else "false",
        "num_chips": chips,
        "tput_per_gpu": total_tput / chips,
        "output_tput_per_gpu": output_tput / chips,
        "input_tput_per_gpu": (total_tput - output_tput) / chips,
    }
    if os.environ.get("NIGHTLY") == "1":
        row["nightly"] = True
    for key, value in raw.items():
        if not key.endswith("_ms") or not isinstance(value, (int, float)):
            continue
        name = key.removesuffix("_ms")
        row[name] = value / 1000.0
        if "tpot" in name:
            row[name.replace("tpot", "intvty")] = 1000.0 / value if value else 0.0
    return row


def collect_rows(results_dir: Path, **row_args) -> list[dict]:
    config = json.loads((results_dir / "config.json").read_text())
    rows = []
    for path in sorted(results_dir.iterdir()):
        match = RESULT_FILENAME_RE.match(path.name)
        if not match:
            continue
        isl, osl, conc = (int(g) for g in match.groups())
        raw = json.loads(path.read_text())
        rows.append(to_row(raw, config, isl=isl, osl=osl, conc=conc, **row_args))
    return rows


def post(endpoint: str, token: str, row: dict) -> None:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(row).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Source": "vllm-torchtpu-ci",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dirs",
        nargs="+",
        type=Path,
        help="run_benchmarks.sh results directories (each holds a config.json)",
    )
    parser.add_argument("--device", required=True, help="Device tag, e.g. tpu7x")
    parser.add_argument(
        "--chips", type=int, required=True, help="Chips in the slice the run used"
    )
    parser.add_argument("--image", required=True, help="Image the run used")
    parser.add_argument(
        "--endpoint", default=os.environ.get("PERF_INGEST_URL", DEFAULT_ENDPOINT)
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the rows instead of posting"
    )
    args = parser.parse_args()

    rows = []
    for results_dir in args.results_dirs:
        dir_rows = collect_rows(
            results_dir, device=args.device, chips=args.chips, image=args.image
        )
        print(f"{results_dir}: {len(dir_rows)} result(s)")
        rows += dir_rows
    if not rows:
        print("ERROR: no isl*_osl*_c*.json results found", file=sys.stderr)
        return 1
    if args.dry_run:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return 0

    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        print(f"ERROR: {TOKEN_ENV} is not set", file=sys.stderr)
        return 1
    failed = 0
    for row in rows:
        label = f"{row['model']} tp{row['tp']}dp{row['dp']} isl{row['isl']} "
        label += f"osl{row['osl']} c{row['conc']}"
        try:
            post(args.endpoint, token, row)
            print(f"posted {label}")
        except OSError as e:
            print(f"FAILED {label}: {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
