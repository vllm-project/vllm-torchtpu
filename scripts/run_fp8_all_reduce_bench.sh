#!/bin/bash
# Runs scripts/benchmark_fp8_all_reduce.py once per LIBTPU setting (tpu7x 2x2x1, 8 TensorCores).
#   c48     the c48 served flags (recipe defaults + HAR + LIB_R8_48: SC AR >= 2 MiB, SC RS/AG offload on)
#   c48a2a  c48 + SparseCore offload of dense all-to-all
#   c64     the c64 served flags (LIB_48_64: SC AG offload OFF)
#   c64a2a  c64 + SparseCore offload of dense all-to-all
# A flag this libtpu does not know aborts that run only.
# Usage: run_fp8_all_reduce_bench.sh [out_dir] [src_dir]   (defaults /cdk-outputs/outputs, ../src)
set -u
OUT=${1:-/cdk-outputs/outputs}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${2:-$HERE/../src}
mkdir -p "$OUT/hlo"
DEF="--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=12582912 --xla_tpu_dvfs_p_state=7 --xla_tpu_use_enhanced_launch_barrier=true --xla_enable_async_all_gather=true"
C48="$DEF --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=2097152 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=1048576"
C64="$DEF --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=2097152"
A2A="--xla_tpu_enable_sparse_core_collective_offload_dense_all_to_all=true"
declare -A SETS=([c48]="$C48" [c48a2a]="$C48 $A2A" [c64]="$C64" [c64a2a]="$C64 $A2A")
for tag in c48 c48a2a c64 c64a2a; do
  echo "=== $tag: ${SETS[$tag]}"
  LIBTPU_INIT_ARGS="${SETS[$tag]}" timeout 1500 python3 "$HERE/benchmark_fp8_all_reduce.py" --src "$SRC" \
    --tokens 8192 4096 --tag "$tag" --json "$OUT/fp8ar_$tag.json" --dump-hlo "$OUT/hlo" 2>&1 \
    | grep -E "^(DEVICES|LIBTPU|BENCH)|Error|error" | head -60
done
echo "=== TABLE tag T mode ms_per_ar vs_psum snr_db saved_ms_per_step_121"
python3 - "$OUT" <<'PY' || true
import glob, json, sys
for path in sorted(glob.glob(sys.argv[1] + "/fp8ar_*.json")):
    for r in json.load(open(path)):
        f = lambda k, fmt: (fmt % r[k]) if r.get(k) is not None else "-"
        print("TABLE", r["tag"], r["tokens"], r["mode"], f("ms_per_ar", "%.3f"), f("vs_psum", "%.2f"),
              f("snr_db", "%.1f"), f("saved_ms_per_step_121", "%.1f"), r.get("error", ""), r.get("collectives", ""))
PY
