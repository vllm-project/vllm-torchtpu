#!/bin/bash
# Runs scripts/benchmark_prefill_overlap.py once per LIBTPU setting (tpu7x 2x2x1, 8 TensorCores).
# Base = the c48 served flags (recipe defaults + HAR + LIB_R8_48); each variant appends scheduler /
# SparseCore-offload flags that may let the latency-hiding scheduler overlap the SC all-reduce with
# TensorCore work. A flag this libtpu does not know aborts that run only.
# Usage: run_prefill_overlap_bench.sh [out_dir]   (default /cdk-outputs/outputs)
set -u
OUT=${1:-/cdk-outputs/outputs}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/hlo"
BASE="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_dvfs_p_state=7 --xla_tpu_use_enhanced_launch_barrier=true --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=2097152 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1048576"
Q="--xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true"
R="--xla_tpu_rerun_latency_hiding_scheduler_post_sc_assignment=true"
I="--xla_tpu_force_early_inlining_and_flattening=true"
C="--xla_tpu_enable_concurrent_sparse_core_offloading=true"
declare -A SETS=(
  [base]=""
  [q]="$Q"
  [r]="$R"
  [qr]="$Q $R"
  [c]="$C"
  [i]="$I"
)
for tag in base q r qr c i; do
  echo "=== $tag: ${SETS[$tag]}"
  LIBTPU_INIT_ARGS="$BASE ${SETS[$tag]}" timeout 1200 python3 "$HERE/benchmark_prefill_overlap.py" \
    --tokens 8192 4096 --tag "$tag" --json "$OUT/ovl_$tag.json" --dump-hlo "$OUT/hlo" 2>&1 \
    | grep -E "^(DEVICES|LIBTPU|BENCH|SUMMARY)|Error|error" | head -40
done
echo "=== SUMMARY lines"
python3 - "$OUT" <<'PY' || true
import glob, json, sys
for path in sorted(glob.glob(sys.argv[1] + "/ovl_*.json")):
    for row in json.load(open(path)):
        if "summary" in row:
            print(json.dumps(row["summary"]))
PY
