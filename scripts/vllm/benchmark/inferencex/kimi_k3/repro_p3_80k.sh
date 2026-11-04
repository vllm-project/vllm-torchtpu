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
#
# Reproduce Kimi-K3 TP32 DCP32 EP8 (16 TPU7x chips / 4 hosts, 2x2x4 sub-slice)
# at strict hardware-default P-state 3 (xla_tpu_dvfs_p_state=3) on ISL=80128, OSL=512:
#   - Mode="unified" (default): S=4 decode hot-expert replicas (c=1..64: 51.3%-74.3% of gb300-agg)
#   - Mode="single_stream" (K3_REPLICA_PLAN=none): no decode replicas (c=1: 927.89 tok/s/chip = 77.6% of gb300-agg)
#
# Usage on each of the 4 TPU7x hosts (INDEX=0..3):
#   INDEX=<0..3> HEAD_IP=<host0_ip> POD_IP=<this_host_ip> \
#   MODEL_DIR=/path/to/Kimi-K3 PRESHARDED_DIR=/path/to/Kimi-K3-presharded \
#   bash scripts/vllm/benchmark/inferencex/kimi_k3/repro_p3_80k.sh

set -euo pipefail

INDEX="${INDEX:-${TPU_WORKER_ID:-0}}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
POD_IP="${POD_IP:-${HEAD_IP}}"
PORT="${PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/models/Kimi-K3}"
PRESHARDED_DIR="${PRESHARDED_DIR:-/models/Kimi-K3-presharded}"
RESULT_DIR="${RESULT_DIR:-/dev/shm/results_k3_p3}"
BENCH_CONCS="${BENCH_CONCS:-1 2 4 8 16 32 64}"
RUN_LMEVAL="${RUN_LMEVAL:-0}"
CHIPS=16

mkdir -p "${RESULT_DIR}" /dev/shm/jax_cache /dev/shm/tier3_cache

# Default P3 (TPU_DVFS_P_STATE=3); set TPU_DVFS_P_STATE=7 to enable P-state 7
export TPU_DVFS_P_STATE="${TPU_DVFS_P_STATE:-3}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"
if [[ "${TPU_DVFS_P_STATE}" == "7" ]]; then
  if [[ "${LIBTPU_INIT_ARGS}" != *"xla_tpu_dvfs_p_state=7"* ]]; then
    export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS} --xla_tpu_dvfs_p_state=7"
  fi
elif [[ "${LIBTPU_INIT_ARGS}" == *"xla_tpu_dvfs_p_state=7"* ]]; then
  echo "ERROR: xla_tpu_dvfs_p_state=7 is forbidden in strict default P3 reproduction (set TPU_DVFS_P_STATE=7 to enable P7)." >&2
  exit 1
fi

export VLLM_TARGET_DEVICE=tpu
export VLLM_NO_USAGE_STATS=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_HOST_IP="${POD_IP}"
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,4}"
export TPU_NODE_ID="${INDEX}"
export TPU_SKIP_MDS_QUERY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TCMALLOC_AGGRESSIVE_DECOMMIT=true
export TPU_MULTIHOST_BACKEND=mp
export TPU_MOE_HIERARCHICAL_EP=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export ONEHOT_MOE_PERMUTE_THRESHOLD=32768
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_K3_MOE_REDUCE_DTYPE=float32

if [[ -x /usr/local/bin/vllm-rs ]]; then
  export VLLM_USE_RUST_FRONTEND=1
  export VLLM_RUST_FRONTEND_PATH=/usr/local/bin/vllm-rs
fi

export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/dev/shm/tier3_cache
export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=1
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jax_cache

# Kimi-K3 16K prefill + EP8 real-traffic expert placement & S=4 decode replica plan
export K3_PREFILL_SJF="${K3_PREFILL_SJF:-1}"
export K3_PREFILL_PREFIX_FP8="${K3_PREFILL_PREFIX_FP8:-1}"
export K3_AGGREGATE_EP16="${K3_AGGREGATE_EP16:-0}"
export K3_PREFILL_EXPERT_PLAN="${K3_PREFILL_EXPERT_PLAN:-plan_ep8_real_hotfirst.npz}"
export K3_REPLICA_PLAN="${K3_REPLICA_PLAN:-none}"
export K3_DECODE_FP8_STAGED_ALL="${K3_DECODE_FP8_STAGED_ALL:-1}"
export K3_SERIAL_PREFILL="${K3_SERIAL_PREFILL:-1}"
export K3_MAX_PREFILL_STEP_TOKENS="${K3_MAX_PREFILL_STEP_TOKENS:-16384}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16448}"
KV_BYTES="${KV_BYTES:-18790481920}"

if [[ "${INDEX}" == "0" ]]; then
  ROLE_FLAGS=(--host 0.0.0.0 --port "${PORT}")
else
  sleep 4
  ROLE_FLAGS=(--headless)
fi

SERVER_LOG="${RESULT_DIR}/vllm_serve_h${INDEX}.log"
echo "[h${INDEX}] Launching vllm serve at strict default P3 (log: ${SERVER_LOG})..."
vllm serve "${MODEL_DIR}" \
  --served-model-name moonshotai/Kimi-K3 \
  --tensor-parallel-size 32 \
  --decode-context-parallel-size 32 \
  --enable-expert-parallel \
  --distributed-timeout-seconds 1800 \
  --max-model-len 163840 \
  --kv-cache-dtype fp8 \
  --block-size 512 \
  --cp-kv-cache-interleave-size 512 \
  --trust-remote-code \
  --language-model-only \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --load-format megakernel \
  --model-loader-extra-config "{\"presharded\":\"${PRESHARDED_DIR}\"}" \
  --max-num-seqs 64 \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --kv-cache-memory-bytes "${KV_BYTES}" \
  --async-scheduling \
  --nnodes 4 \
  --node-rank "${INDEX}" \
  --master-addr "${HEAD_IP}" \
  "${ROLE_FLAGS[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

trap 'kill -TERM "${SERVER_PID}" 2>/dev/null || true' EXIT

echo "[h${INDEX}] Waiting for http://${HEAD_IP}:${PORT}/health..."
READY=0
for i in $(seq 1 480); do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://${HEAD_IP}:${PORT}/health" || echo "000")"
  if [[ "${CODE}" == "200" ]]; then
    READY=1
    echo "[h${INDEX}] Server ready after $((i * 5))s."
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[h${INDEX}] ERROR: vllm serve exited prematurely:" >&2
    tail -n 100 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

if [[ "${READY}" != "1" ]]; then
  echo "[h${INDEX}] ERROR: Timed out waiting for server readiness." >&2
  exit 1
fi

if [[ "${INDEX}" != "0" ]]; then
  wait "${SERVER_PID}"
  exit 0
fi

EP="http://127.0.0.1:${PORT}"

# 1. Coherence Smoke Check ("The capital of France is" -> "Paris")
echo "=== [Gate 1] Coherence Smoke Check ==="
curl -s --max-time 600 "${EP}/v1/completions" -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K3","prompt":"The capital of France is","max_tokens":32,"temperature":0.0}' \
  | tee "${RESULT_DIR}/smoke.json"
echo
python3 - "${RESULT_DIR}/smoke.json" <<'PY'
import json, sys
txt = json.load(open(sys.argv[1]))["choices"][0]["text"]
print("[smoke]", repr(txt))
assert "Paris" in txt, f"Smoke check failed: {txt!r}"
PY

# 2. Warmup 80K Prefill + Decode
echo "=== Warming up 80128/512 kernels ==="
curl -sf -X POST "${EP}/reset_prefix_cache" >/dev/null || true
PYTHONHASHSEED=100999 vllm bench serve \
  --backend openai --host 127.0.0.1 --port "${PORT}" --endpoint /v1/completions \
  --model moonshotai/Kimi-K3 --tokenizer "${MODEL_DIR}" --trust-remote-code \
  --dataset-name random --random-input-len 80128 --random-output-len 32 --random-range-ratio 0.0 \
  --ignore-eos --request-rate inf --max-concurrency 1 --num-prompts 1 \
  --num-warmups 0 --ready-check-timeout-sec 0 --seed 99991 --temperature 0.0 \
  >"${RESULT_DIR}/warmup_80128_32.log" 2>&1

# 3. I80128 / O512 Concurrency Sweep (c in BENCH_CONCS)
for c in ${BENCH_CONCS}; do
  n=$(( 2 * c ))
  seed=$(( 50000 + 512 + c * 131 ))
  pyseed=$(( 200000 + 512 + c ))
  tag="k3_p3_80128_512_c${c}"
  echo "=== [I80128/O512] c=${c}, num_prompts=${n}, seed=${seed}, PYTHONHASHSEED=${pyseed} ==="
  curl -sf -X POST "${EP}/reset_prefix_cache" >/dev/null || true
  sleep 1
  PYTHONHASHSEED="${pyseed}" vllm bench serve \
    --backend openai --host 127.0.0.1 --port "${PORT}" --endpoint /v1/completions \
    --model moonshotai/Kimi-K3 --tokenizer "${MODEL_DIR}" --trust-remote-code \
    --dataset-name random --random-input-len 80128 --random-output-len 512 --random-range-ratio 0.0 \
    --ignore-eos --request-rate inf \
    --max-concurrency "${c}" --num-prompts "${n}" \
    --num-warmups 0 --ready-check-timeout-sec 0 \
    --seed "${seed}" --temperature 0.0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 25,50,75,90,95,99 \
    --save-result --save-detailed \
    --result-dir "${RESULT_DIR}" --result-filename "${tag}.json" \
    | tee "${RESULT_DIR}/${tag}.log"

  python3 - "${RESULT_DIR}/${tag}.json" "${c}" "${CHIPS}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
c, chips = int(sys.argv[2]), int(sys.argv[3])
tot_chip = float(d["total_token_throughput"]) / chips
out_chip = float(d["output_throughput"]) / chips
ttft_s = float(d["mean_ttft_ms"]) / 1000.0
tpot_ms = float(d["mean_tpot_ms"])
dur_s = float(d["duration"])
print(f"[RESULT c={c:2d}] total_tok/s/chip={tot_chip:7.2f} | out_tok/s/chip={out_chip:6.2f} | mean_TTFT={ttft_s:6.2f}s | mean_TPOT={tpot_ms:6.2f}ms | duration={dur_s:6.2f}s")
PY
done

# 4. Optional Accuracy Evaluation (GSM8K & MMLU-Pro)
if [[ "${RUN_LMEVAL}" == "1" ]]; then
  echo "=== [Gate 3a] GSM8K (1,319 questions, 5-shot, greedy, thinking=false) ==="
  curl -sf -X POST "${EP}/reset_prefix_cache" >/dev/null || true
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 lm_eval --tasks gsm8k \
    --seed "0,1234,None,1234" \
    --output_path "${RESULT_DIR}/gsm8k" \
    --log_samples \
    --model local-chat-completions \
    --model_args "model=moonshotai/Kimi-K3,base_url=${EP}/v1/chat/completions,num_concurrent=32,max_retries=3,timeout=3600" \
    --apply_chat_template \
    --gen_kwargs '{"chat_template_kwargs": {"thinking": false}, "max_gen_toks": 1024, "temperature": 0.0}' \
    | tee "${RESULT_DIR}/gsm8k.log"

  echo "=== [Gate 3b] MMLU-Pro (5-shot, greedy, thinking=false) ==="
  curl -sf -X POST "${EP}/reset_prefix_cache" >/dev/null || true
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 lm_eval --tasks mmlu_pro \
    --seed "0,1234,None,1234" \
    --output_path "${RESULT_DIR}/mmlu_pro" \
    --log_samples \
    --model local-chat-completions \
    --model_args "model=moonshotai/Kimi-K3,base_url=${EP}/v1/chat/completions,num_concurrent=16,max_retries=3,timeout=3600" \
    --apply_chat_template \
    --gen_kwargs '{"chat_template_kwargs": {"thinking": false}}' \
    | tee "${RESULT_DIR}/mmlu_pro.log"
fi
