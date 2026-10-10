#!/usr/bin/env bash
# 40-layer real-weights DSv4.1-Flash v2 100k prefill benchmark on TPU7x-8.
# Usage:
#   bench_100k_prefill.sh <label> <src_root> [ENV=val ...]
# <src_root>: path to the repository src/ directory.
# Env knobs: KV_BYTES (unset => auto via gpu-memory-utilization 0.85, logs
# "Memory statistics"), MAX_SEQS (4), CHUNK (2048), COMPILE_SIZES (4,$CHUNK),
# OSL (1 => pure prefill), C1_PROMPTS (2), SAT_PROMPTS (8), SAT_C (4),
# PROFILE=1 (adds a torch-profiler capture of one 100k request after the
# throughput runs; PROF_DELAY=<s> [PROF_SECS=<s>] captures a window mid-request
# instead), SKIP_TPUT=1 (skip the c1/sat runs), ISL_SWEEP="16384 32768 ..."
# (c=1 TTFT per ISL).
set -uo pipefail
label=$1; src=$2; shift 2
# shellcheck disable=SC2163  # args are KEY=VAL pairs
for kv in "$@"; do export "$kv"; done
here=$(cd "$(dirname "$0")" && pwd)
out=${OUT_DIR:-$(cd "$src/.." && pwd)/.dsv41_bench/prefill_100k/${label}}
mkdir -p "$out"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH=$here/hbm_ledger:$src
export DSV41_HBM_LEDGER=1 DSV41_HBM_LEDGER_EVERY=${LEDGER_EVERY:-8}
export OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm
export VLLM_USE_AOT_COMPILE=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-1}" TORCHINDUCTOR_AUTOGRAD_CACHE=0
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export LIBTPU_INIT_ARGS="--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
# Set ENABLE_PREFIX_CACHING=false for uncached baseline measurements.
case "${ENABLE_PREFIX_CACHING:-true}" in
  true) prefix_caching_flag=--enable-prefix-caching ;;
  false) prefix_caching_flag=--no-enable-prefix-caching ;;
  *) echo "ENABLE_PREFIX_CACHING must be true or false" >&2; exit 1 ;;
esac
export TPU_DEEPSEEK_V41_IMPL=v2
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"
CHUNK=${CHUNK:-2048}; MAX_SEQS=${MAX_SEQS:-4}; OSL=${OSL:-1}
prof_args=()
if [ "${PROFILE:-0}" = 1 ]; then
  prof_args=(--profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$out/profile"
    --profiler-config.ignore_frontend true)
fi
port=8012
py="${PYTHON_BIN:-python3}"
$py -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --host 127.0.0.1 --port $port \
  --tensor-parallel-size 8 --data-parallel-size 1 \
  --enable-expert-parallel --enable-ep-weight-filter --language-model-only \
  --async-scheduling "$prefix_caching_flag" \
  --attention-config '{"indexer_kv_dtype":"fp8"}' --kv-cache-dtype fp8 \
  --safetensors-load-strategy prefetch --safetensors-prefetch-num-threads 8 \
  --max-model-len 103424 --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$CHUNK" \
  --gpu-memory-utilization 0.85 --block-size 16 ${KV_BYTES:+--kv-cache-memory-bytes $KV_BYTES} \
  --compilation-config "{\"compile_sizes\":[${COMPILE_SIZES:-$MAX_SEQS,$CHUNK}]}" \
  "${prof_args[@]}" >"$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for i in $(seq 1 720); do
  kill -0 $pid 2>/dev/null || { echo "SERVER_DIED"; tail -40 "$out/server.log"; exit 1; }
  curl -sf http://127.0.0.1:$port/health >/dev/null 2>&1 && { echo "READY after $((i*5))s"; break; }
  sleep 5
done
bench() {  # <tag> <num_prompts> <concurrency> [extra...]
  local tag=$1 n=$2 c=$3; shift 3
  (cd "$out" && $py -m vllm.entrypoints.cli.main bench serve --backend vllm \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --host 127.0.0.1 --port $port --dataset-name random \
    --random-input-len "${ISL:-100000}" --random-output-len "$OSL" --random-range-ratio 0.0 \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el --ignore-eos --temperature 0 \
    --save-result --result-filename "$tag.json" --seed 42 "$@") \
    >"$out/$tag.log" 2>&1
  echo "bench $tag rc=$?"
  grep -E "Successful|Benchmark duration|Total input|Request throughput|Input token|Output token|Mean TTFT|Median TTFT|P99 TTFT" "$out/$tag.log"
}
bench warm 1 1
for isl in ${ISL_SWEEP:-}; do  # TTFT vs ISL => marginal per-chunk cost by context
  ISL=$isl bench "isl$isl" 2 1
done
if [ "${SKIP_TPUT:-0}" != 1 ]; then
  bench c1 "${C1_PROMPTS:-2}" 1
  bench sat "${SAT_PROMPTS:-8}" "${SAT_C:-4}"
fi
if [ "${PROFILE:-0}" = 1 ]; then
  if [ -n "${PROF_DELAY:-}" ]; then
    # Window capture mid-request: whole-request traces truncate after ~8 steps.
    bench prof 1 1 &
    bpid=$!
    sleep "$PROF_DELAY"
    curl -sf -X POST http://127.0.0.1:$port/start_profile && echo "start_profile ok"
    sleep "${PROF_SECS:-1}"
    curl -sf -X POST http://127.0.0.1:$port/stop_profile && echo "stop_profile ok"
    wait $bpid
  else
    bench prof 1 1 --profile
  fi
  du -sh "$out/profile"; find "$out/profile" -name "*.xplane.pb" | head -10
fi
grep -h "Memory statistics\|HBM_LEDGER\] rank=0" "$out/server.log" | sed "s/^.*\] //" | tail -40
grep -c "No space left" "$out/server.log"
