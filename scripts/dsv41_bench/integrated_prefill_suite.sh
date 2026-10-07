#!/usr/bin/env bash
# Integrated 40-Layer Real-Weights Verification & 100k Prefill Suite on TPU7x-8.
# Usage:
#   integrated_prefill_suite.sh <mode> [ENV=val ...]
# Modes:
#   tier2_tier3 : Run bench_kv_compressor.py --verify on 8-rank TPU +
#                 independent smoke_e2e_dsv41_tp8.py --verify-against-baseline.
#   tput_hbm    : Run 100k prefill (c=1, c=4, c=8) with Engram in HBM (offload=0).
#   full_offload: Run with TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD=1:
#                 (1) 100k prefill (c=1, c=4, c=8) + HBM Ledger (-23.90 GiB/rank),
#                 (2) 32k Needle-in-a-Haystack (5/5 + 5/5),
#                 (3) Full GSM8K (all 1319 questions, --eval-batch-size 16).
set -uo pipefail

mode=${1:-full_offload}
shift || true
# shellcheck disable=SC2163
for kv in "$@"; do export "$kv"; done

here=$(cd "$(dirname "$0")" && pwd)
root=${ROOT:-$(cd "$here/../.." && pwd)}
golden_dir=${GOLDEN_DIR:-$root/.dsv41_bench/golden}
py="${PYTHON_BIN:-python3}"
evalscope_bin="${EVALSCOPE_BIN:-evalscope}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm
export VLLM_USE_AOT_COMPILE=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-1}" TORCHINDUCTOR_AUTOGRAD_CACHE=0
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export LIBTPU_INIT_ARGS="--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
export TPU_DEEPSEEK_V41_IMPL=v2
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"

if [ "$mode" = "tier2_tier3" ] || [ "$mode" = "tier3_only" ]; then
  label=${LABEL:-tier2_tier3}
  out="$root/.dsv41_bench/integrated/${label}"
  mkdir -p "$out"
  if [ "$mode" = "tier2_tier3" ]; then
    echo "=== [1/2] Running bench_kv_compressor.py --subop=all --verify on 8-rank TPU ==="
    PYTHONPATH="$root/src:$root" $py "$root/scripts/dsv41_bench/bench_kv_compressor.py" \
      --subop=all --num-tokens 2048 --verify \
      --json-out "$out/kv_compressor_tpu.json" \
      >"$out/kv_compressor_tpu.log" 2>&1
    echo "bench_kv_compressor rc=$?"
    tail -n 30 "$out/kv_compressor_tpu.log"
  fi

  echo "=== Running independent smoke_e2e_dsv41_tp8.py --verify-against-baseline ==="
  PYTHONPATH="$root/src:$root" $py "$root/smoke_e2e_dsv41_tp8.py" \
    --num-layers 4 --max-model-len 92160 --prompt-tokens 90112 --max-num-batched-tokens 2048 --max-tokens 1 \
    --verify-against-baseline "$golden_dir/4l_90k_golden.pt" \
    >"$out/smoke_verify_integrated.log" 2>&1
  echo "smoke_e2e verify rc=$?"
  tail -n 30 "$out/smoke_verify_integrated.log"
  exit 0
fi

label=${LABEL:-integrated_${mode}}
out="$root/.dsv41_bench/integrated/${label}"
mkdir -p "$out"
CHUNK=${CHUNK:-8192}

if [ "$mode" = "all" ] || [ "${RUN_TIER3:-0}" = "1" ]; then
  echo "=== [Step 0] smoke_e2e_dsv41_tp8.py --verify-against-baseline (root=$root) ==="
  PYTHONPATH="$root/src:$root" $py "$root/smoke_e2e_dsv41_tp8.py" \
    --num-layers 4 --max-model-len 92160 --prompt-tokens 90112 --max-num-batched-tokens "${TIER3_CHUNK:-2048}" --max-tokens 1 \
    --verify-against-baseline "$golden_dir/4l_90k_golden.pt" \
    >"$out/smoke_verify_integrated.log" 2>&1
  echo "smoke_e2e verify rc=$?"
  tail -n 30 "$out/smoke_verify_integrated.log"
fi

if [ "$mode" = "tier4_only" ]; then
  export PYTHONPATH="$root/src:$root"
  export DSV41_HBM_LEDGER=0
  ASYNC_FLAG="${ASYNC_FLAG:---async-scheduling}"
  MML="${MML:-36864}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
  COMPILE_SIZES="${COMPILE_SIZES:-16,2048,8192}"
else
  export PYTHONPATH="$root/scripts/dsv41_bench/hbm_ledger:$root/src:$root"
  export DSV41_HBM_LEDGER=1 DSV41_HBM_LEDGER_EVERY=${LEDGER_EVERY:-8}
  ASYNC_FLAG="${ASYNC_FLAG:---async-scheduling}"
  MML="${MML:-103424}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
  COMPILE_SIZES="${COMPILE_SIZES:-16,2048,8192}"
fi
if [ "$mode" = "full_offload" ]; then
  export TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD="${TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD:-1}"
  export TPU_DEEPSEEK_V41_ENGRAM_OFFLOAD="${TPU_DEEPSEEK_V41_ENGRAM_OFFLOAD:-all}"
else
  export TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD="${TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD:-0}"
  # The default "auto" offloads on v2; this mode measures tables in HBM.
  export TPU_DEEPSEEK_V41_ENGRAM_OFFLOAD="${TPU_DEEPSEEK_V41_ENGRAM_OFFLOAD:-off}"
fi

CHUNK=${CHUNK:-8192}
MAX_SEQS=${MAX_SEQS:-16}
OSL=${OSL:-1}
port=${PORT:-8012}
if [ -z "${KV_BYTES:-}" ] && [ "${TPU_DEEPSEEK_V41_SP:-0}" = "1" ]; then
  if [ "$TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD" = "1" ]; then
    KV_BYTES=32212254720
  else
    KV_BYTES=15045270528
  fi
fi
kv_args=(--gpu-memory-utilization "$GPU_MEM_UTIL")
if [ -n "${KV_BYTES:-}" ]; then
  kv_args=(--kv-cache-memory-bytes "$KV_BYTES")
fi

echo "=== Starting vLLM API Server ($mode: SP=${TPU_DEEPSEEK_V41_SP:-0}, OFFLOAD=$TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD, CHUNK=$CHUNK, MAX_SEQS=$MAX_SEQS, COMPILE_SIZES=$COMPILE_SIZES, MML=$MML, ASYNC=$ASYNC_FLAG) ==="
$py -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --host 127.0.0.1 --port "$port" \
  --tensor-parallel-size 8 --data-parallel-size 1 \
  --enable-expert-parallel --enable-ep-weight-filter --language-model-only \
  "$ASYNC_FLAG" --no-enable-prefix-caching \
  --attention-config '{"indexer_kv_dtype":"fp8"}' --kv-cache-dtype fp8 \
  --safetensors-load-strategy prefetch --safetensors-prefetch-num-threads 8 \
  --max-model-len "$MML" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$CHUNK" \
  "${kv_args[@]}" --block-size 16 \
  --compilation-config "{\"compile_sizes\":[${COMPILE_SIZES}]}" \
  >"$out/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT

for i in $(seq 1 720); do
  kill -0 "$pid" 2>/dev/null || { echo "SERVER_DIED"; tail -50 "$out/server.log"; exit 1; }
  curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "READY after $((i*5))s"; break; }
  sleep 5
done

bench() {
  local tag=$1 n=$2 c=$3; shift 3
  (cd "$out" && $py -m vllm.entrypoints.cli.main bench serve --backend vllm \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --host 127.0.0.1 --port "$port" --dataset-name random \
    --random-input-len "${ISL:-100000}" --random-output-len "$OSL" --random-range-ratio 0.0 \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el --ignore-eos --temperature 0 \
    --save-result --result-filename "$tag.json" --seed 42 "$@") \
    >"$out/$tag.log" 2>&1
  echo "bench $tag rc=$?"
  grep -E "Successful|Benchmark duration|Total input|Request throughput|Input token|Output token|Mean TTFT|Median TTFT|P99 TTFT" "$out/$tag.log"
}

if [ "$mode" != "tier4_only" ]; then
  echo "=== [Step 1] 100k Prefill Benchmarks (c=1, c=4, c=8) ==="
  bench warm 1 1
  bench c1 "${C1_PROMPTS:-2}" 1
  bench c4 "${SAT_PROMPTS:-8}" 4
  bench c8 "${C8_PROMPTS:-8}" 8
fi

echo "=== [Step 2] HBM Ledger Summary ==="
grep -h "Memory statistics\|HBM_LEDGER\] rank=0" "$out/server.log" | sed "s/^.*\] //" | tail -40

if [ "$mode" = "full_offload" ] || [ "$mode" = "tier4_only" ] || [ "$mode" = "all" ] || [ "${RUN_TIER4:-0}" = "1" ]; then
  echo "=== [Step 3] 32k Needle-in-a-Haystack (5/5 + 5/5) ==="
  $py "$root/scripts/dsv41_bench/needle_32k.py" \
    --base-url "http://127.0.0.1:$port" \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --target-tokens "${NEEDLE_TOKENS:-32000}" \
    --out "$out/needle_32k.json" \
    >"$out/needle_32k.log" 2>&1
  echo "needle_32k rc=$?"
  cat "$out/needle_32k.log"

  echo "=== [Step 4] Full GSM8K (${GSM8K_LIMIT:-all 1319} questions, batch_size=${GSM8K_BS:-16}) ==="
  limit_args=()
  if [ -n "${GSM8K_LIMIT:-}" ]; then
    limit_args=(--limit "$GSM8K_LIMIT")
  fi
  (cd "$out" && $evalscope_bin eval \
    --model deepseek-ai/DeepSeek-V4.1-Flash --api-url "http://127.0.0.1:$port/v1" --api-key EMPTY \
    --eval-type openai_api --datasets gsm8k \
    --eval-batch-size "${GSM8K_BS:-16}" \
    "${limit_args[@]}" \
    --dataset-args '{"gsm8k":{"few_shot_num":4,"few_shot_random":false,"shuffle":false}}' \
    --generation-config '{"temperature":0,"max_tokens":2048,"extra_body":{"chat_template_kwargs":{"thinking":false}}}' \
    >"$out/gsm8k_eval.log" 2>&1)
  echo "gsm8k eval rc=$?"
  grep -E "flexible|strict|gsm8k|mean_acc" "$out/gsm8k_eval.log" | tail -15
fi
