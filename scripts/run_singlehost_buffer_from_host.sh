#!/usr/bin/env bash
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

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
benchmark="${BENCHMARK:-$script_dir/bench_buffer_from_host.py}"
output_root="${1:-$script_dir/../perf_eval_results/buffer_from_host}"
iterations="${2:-100}"
warmup="${3:-10}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python is not executable: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$benchmark" ]]; then
  echo "Benchmark does not exist: $benchmark" >&2
  exit 1
fi

busy_pids="$(fuser /dev/vfio/{0..7} 2>/dev/null | xargs || true)"
if [[ -n "$busy_pids" ]]; then
  echo "TPU devices are busy; owning PIDs: $busy_pids" >&2
  ps -o user,pid,ppid,etimes,cmd -p "${busy_pids// /,}" >&2 || true
  exit 1
fi

mkdir -p "$output_root"
"$python_bin" -m torch_tpu._internal.distributed.launchers.singlehost_wrapper \
  >"$output_root/tpu-env.sh" 2>"$output_root/env-prep.log"
prep_status=$?
if [[ "$prep_status" -ne 0 ]]; then
  echo "Failed to detect TPU topology; see $output_root/env-prep.log" >&2
  exit "$prep_status"
fi

# shellcheck disable=SC1090,SC1091
source "$output_root/tpu-env.sh"
if [[ "${WORLD_SIZE:-0}" -ne 8 ]]; then
  echo "Expected 8 local TPU devices, got WORLD_SIZE=${WORLD_SIZE:-unset}" >&2
  exit 1
fi

master_addr=127.0.0.1
master_port="$("$python_bin" -c \
  'import portpicker; print(portpicker.pick_unused_port())')"
session_id="buffer_singlehost_$(date +%s%N)"

printf '%s\n' \
  "iterations=$iterations" \
  "warmup=$warmup" \
  "WORLD_SIZE=$WORLD_SIZE" \
  "LOCAL_WORLD_SIZE=$WORLD_SIZE" \
  "NNODES=1" \
  "MASTER_ADDR=$master_addr" \
  "MASTER_PORT=$master_port" \
  "TPU_PREMAPPED_BUFFER_SIZE=${TPU_PREMAPPED_BUFFER_SIZE-unset}" \
  "TORCH_TPU_XPROF_SESSION_ID=$session_id" \
  "TORCH_TPU_TOPOLOGY=$TORCH_TPU_TOPOLOGY" \
  "TORCH_TPU_SLICEBUILDER_ADDRESSES=$TORCH_TPU_SLICEBUILDER_ADDRESSES" \
  >"$output_root/run-config.txt"

echo "Starting 8-rank single-host BufferFromHostBuffer benchmark"
echo "Output: $output_root"
echo "Topology: $TORCH_TPU_TOPOLOGY"

ulimit -c 0
pids=()
# shellcheck disable=SC2317
stop_children() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill -TERM "${pids[@]}" 2>/dev/null || true
  fi
}
trap 'stop_children; exit 130' INT TERM

for rank in $(seq 0 7); do
  rank_dir="$output_root/rank_$rank"
  mkdir -p "$rank_dir"
  env \
    PJRT_DEVICE=TPU \
    TPU_SKIP_MDS_QUERY=1 \
    RANK="$rank" \
    LOCAL_RANK="$rank" \
    WORLD_SIZE=8 \
    LOCAL_WORLD_SIZE=8 \
    NNODES=1 \
    NODE_RANK=0 \
    GROUP_RANK=0 \
    TPU_NUM_HOSTS=1 \
    MASTER_ADDR="$master_addr" \
    MASTER_PORT="$master_port" \
    TORCH_TPU_TOPOLOGY="$TORCH_TPU_TOPOLOGY" \
    TORCH_TPU_SLICEBUILDER_ADDRESSES="$TORCH_TPU_SLICEBUILDER_ADDRESSES" \
    TORCH_TPU_XPROF_SESSION_ID="$session_id" \
    TORCH_TPU_SLICE_RANK="$rank" \
    TORCH_TPU_SLICE_LOCAL_RANK="$rank" \
    TORCH_TPU_SLICE_WORLD_SIZE=8 \
    TORCH_TPU_SLICE_LOCAL_WORLD_SIZE=8 \
    timeout --signal=TERM 10m \
    "$python_bin" "$benchmark" \
      --output-dir "$rank_dir" \
      --iterations "$iterations" \
      --warmup "$warmup" \
      >"$rank_dir/stdout.log" 2>&1 &
  pids+=("$!")
done

status=0
: >"$output_root/exit-status.txt"
for rank in $(seq 0 7); do
  if wait "${pids[$rank]}"; then
    rank_status=0
  else
    rank_status=$?
    status=1
  fi
  printf 'rank=%s exit=%s\n' "$rank" "$rank_status" \
    | tee -a "$output_root/exit-status.txt"
done

trap - INT TERM
if [[ "$status" -eq 0 ]]; then
  "$python_bin" - "$output_root" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
columns = ("min_us", "p50_us", "p90_us", "p99_us", "max_us", "mean_us")

print()
print("Wall-time summary for source.to('tpu') (milliseconds)")
print(
    f"{'rank':>4}  {'shape':>10}  {'count':>5}  "
    f"{'min':>9}  {'p50':>9}  {'p90':>9}  {'p99':>9}  {'max':>9}  {'mean':>9}"
)
for rank in range(8):
    summary_path = output_root / f"rank_{rank}" / "summary.json"
    with summary_path.open() as summary_file:
        summary = json.load(summary_file)
    for shape, values in summary["wall_api_time"].items():
        timings_ms = [values[column] / 1000.0 for column in columns]
        print(
            f"{rank:>4}  {shape:>10}  {values['count']:>5}  "
            + "  ".join(f"{value:>9.3f}" for value in timings_ms)
        )
print()
PY
  echo "Benchmark completed successfully: $output_root"
else
  echo "Benchmark failed; inspect rank_*/stdout.log under $output_root" >&2
fi
exit "$status"
