#!/usr/bin/env bash
# End-to-end reproduction script for GLM-5.3 FP8 long-context aggregated serving
# on 1x TPU7x 2x2x2 (8 chips / 16 TensorCores across 2 hosts, TP=16 DCP=16 EP=16).
#
# Subcommands:
#   source <script> env             Export all verified runtime & kernel env vars
#   <script> ray-head               Start Ray head on host 0 and wait for 2 TPU hosts
#   <script> ray-worker <HEAD_IP>   Join Ray cluster from host 1
#   <script> stop                   Stop vLLM / EngineCore / Ray processes on current host
#   <script> serve [--daemon]       Launch vllm serve (set MAX_MODEL_LEN=163840 for timed_trace)
#   <script> bench-random           Run random 80128/512 c=1..64 sweep + Prometheus checks
#   <script> bench-timed-trace      Run timed_trace c=1..64 sweep (set SHARD_DIR=...)
#   <script> eval                   Run Passkey (16K/80K serial + 80K concurrent), GSM8K, MMLU-Pro
#   <script> all                    Launch server (--daemon) + run bench-random (+ eval if RUN_EVAL=1)

set -uo pipefail

MODEL_URI="${MODEL_URI:-zai-org/GLM-5.3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3}"
TOKENIZER="${TOKENIZER:-${MODEL_URI}}"
HOST="${HOST:-0.0.0.0}"
BENCH_HOST="${BENCH_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUT_DIR="${OUT_DIR:-/dev/shm/results/glm53_tp16_dcp16_16k}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-80896}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
KV_BYTES="${KV_BYTES:-30064771072}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LOAD_FORMAT="${LOAD_FORMAT:-runai_streamer}"
NUM_CHIPS="${NUM_CHIPS:-8}"

log() {
  echo "[$(date -u +%FT%TZ)] [glm53-repro] $*"
}

export_glm53_env() {
  mkdir -p /dev/shm/tmp/tpu_logs /dev/shm/cache/vllm /dev/shm/cache/inductor /dev/shm/xla_cache/jax "${OUT_DIR}/metrics"

  unset TPU_VISIBLE_CHIPS TPU_VISIBLE_DEVICES TPU_TOPOLOGY TPU_HOST_BOUNDS \
    TPU_PROCESS_ADDRESSES TPU_WORKER_HOSTNAMES TPU_WORKER_ID TPU_ACCELERATOR_TYPE

  if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
    local detected_project
    detected_project="$(curl -sf -m 2 -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || true)"
    if [[ -n "${detected_project}" ]]; then
      export GOOGLE_CLOUD_PROJECT="${detected_project}"
      export GCLOUD_PROJECT="${GCLOUD_PROJECT:-${detected_project}}"
      export CLOUDSDK_CORE_PROJECT="${CLOUDSDK_CORE_PROJECT:-${detected_project}}"
    fi
  fi

  export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
  export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/cache}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/dev/shm/cache/vllm}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/cache/inductor}"
  export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-/dev/shm/xla_cache}"
  export TORCH_TPU_CACHE_DIR="${TORCH_TPU_CACHE_DIR:-/dev/shm/xla_cache}"
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/dev/shm/xla_cache/jax}"

  export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
  export TPU_MULTIHOST_BACKEND="${TPU_MULTIHOST_BACKEND:-ray}"
  export VLLM_TARGET_DEVICE=tpu
  export VLLM_NO_USAGE_STATS=1
  export VLLM_SERVER_DEV_MODE=1
  export VLLM_ENGINE_READY_TIMEOUT_S=14400
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=10800
  export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export VLLM_TPU_FAST_TOKEN_SUBSTITUTION=1
  export VLLM_TPU_SEGMENT_TOKENIZE=1
  export VLLM_DCP_Q_REPLICATE=1

  export TPU_SKIP_MDS_QUERY=1
  export TPU_LOG_DIR=/dev/shm/tmp/tpu_logs
  export TPU_PREMAPPED_BUFFER_SIZE=4294967296
  export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=4294967296
  export KERNEL_PROFILE="${KERNEL_PROFILE:-agentx_opt}"
  export ENABLE_PAW=false

  export TPU_PARALLEL_PRECOMPILE=1
  export TPU_PRECOMPILE_BIGGEST_FIRST=1
  export TPU_PRECOMPILE_PEER_REPLAY=1
  export TPU_PRECOMPILE_SPEC_MIXED=0
  export TPU_PRECOMPILE_TRACE_FIRST=1
  export TPU_TOKEN_BUCKET_EXTRA="${TPU_TOKEN_BUCKET_EXTRA-4,8}"

  local default_libtpu_args
  default_libtpu_args='--xla_tpu_enable_diagnostic_coordinator=false --xla_tpu_sparse_core_reduce_scatter_offload_min_size_in_bytes=1048576 --xla_tpu_msa_sort_order_overrides=overrides{hlo_position_matcher{instruction_regex:".*parameter.*",size_gte:16121856,size_lte:16121856},override_options{assign_first:true}} --xla_tpu_preferred_prefetch_overrides=overrides{hlo_operand_filter{size_gte:16121856,size_lte:16121856,operand_number:1},override_options{prefetch_eagerness:0.5}},overrides{hlo_operand_filter{size_gte:6291456,size_lte:6291456,operand_number:3},override_options{prefetch_eagerness:0.5}},overrides{hlo_operand_filter{instruction_name_regex:"multiply_reduce_fusion.*",size_gte:196608,size_lte:196608,operand_number:1},override_options{prefetch_eagerness:0.5}},overrides{hlo_operand_filter{instruction_name_regex:"fusion.*x_ref",size_gte:12288,size_lte:12288,operand_number:3},override_options{prefetch_eagerness:0.5}} --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=8388608'
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-${default_libtpu_args}}"

  if [[ "${LIBTPU_INIT_ARGS}" == *"dvfs_p_state"* ]]; then
    echo "FATAL: dvfs_p_state is strictly prohibited." >&2
    return 1
  fi

  # MoE flags (TP=2 decode MoE kernel + HBM->Peer VMEM sequence-parallel FP8-wire prefill MoE)
  export FUSED_MOE_DECODE_KERNEL="${FUSED_MOE_DECODE_KERNEL:-0}"
  export FUSED_MOE_PREFILL_KERNEL="${FUSED_MOE_PREFILL_KERNEL:-0}"
  export ONEHOT_MOE_PERMUTE_THRESHOLD=2048
  export USE_MOE_SPARSE_CORE=1
  export USE_MOE_TP_DECODE_KERNEL="${USE_MOE_TP_DECODE_KERNEL:-1}"
  export TPU_MOE_TP_DECODE_FAST="${TPU_MOE_TP_DECODE_FAST:-1}"
  export TPU_MOE_TP_DECODE_HBM_OPERANDS="${TPU_MOE_TP_DECODE_HBM_OPERANDS:-all}"
  export TPU_MOE_EP_TP_SIZE="${TPU_MOE_EP_TP_SIZE:-2}"
  export TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER=0
  export TPU_MOE_ONEHOT_SORTFREE=1
  export TPU_MOE_ROUTER_TOPK_BIAS=1
  export TPU_MOE_SC_COMBINE_TILED=1
  export TPU_MOE_SC_COMBINE_V5=1
  export TPU_MOE_SKIP_PADDED_TOKENS=1
  export TPU_MOE_PREFILL_SEQUENCE_PARALLEL=1
  export TPU_MOE_PREFILL_SP_SHARED_GROUP=1
  export TPU_MOE_PREFILL_SP_FP8_WIRE=1

  # RMSNorm & FP8 linear flags
  export TPU_FUSED_ADD_RMSNORM=1
  export TPU_FP8_LINEAR_PALLAS=1
  export TPU_FP8_SKINNY_SCALE_DMA=1

  # DCP & MLA prefill/decode flags
  export TPU_DCP_PREFILL_TOKEN_PARALLEL=1
  local default_gather_caps="4,8,12,16,20,24"
  if [[ "${MAX_MODEL_LEN}" -gt 80896 ]]; then
    default_gather_caps="4,8,12,16,20,24,32,44"
  fi
  export TPU_DCP_PREFILL_GATHER_CAPS="${TPU_DCP_PREFILL_GATHER_CAPS:-${default_gather_caps}}"
  export TPU_DCP_PREFILL_TP_UV_BEFORE_A2A=1
  export TPU_DCP_PREFILL_INDEXER_QROWS=1
  export TPU_DCP_PREFILL_FUSED_SUBBATCH=1

  export TPU_DCP_INDEXER_A2A_MERGE=1
  export TPU_DCP_LSE_MERGE_A2A=1
  export TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS=64
  export TPU_DCP_LSE_MERGE_A2A_PACK=1
  export TPU_DCP_LSE_MERGE_A2A_REAL_ROWS=1
  export TPU_DCP_MERGE_CHUNKED=8
  export TPU_DCP_MERGE_CHUNKED_LSE_AG=1
  export TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS=2048
  export TPU_DCP_MERGE_CHUNKED_MIN_TOKENS=1024
  export TPU_DCP_MERGE_RS_BF16=1
  export TPU_DCP_MLA_FUSED_PREP=1
  export TPU_DCP_MLA_MASK_IN_KERNEL=1
  export TPU_DCP_MLA_MIXED_BODY=1
  export TPU_DCP_MLA_MIXED_FUSED_PREP=1
  export TPU_DCP_MLA_MIXED_LIVE=1
  export TPU_DCP_MLA_MIXED_LIVE_WIDTHS=256,384
  export TPU_DCP_MLA_MIXED_MASK_IN_KERNEL=1
  export TPU_DCP_MLA_SEND_LAYOUT=1
  export TPU_DCP_MLA_SMALL_CHUNK_BATCH=4
  export TPU_DCP_MLA_SMALL_CHUNK_TOKENS=32
  export TPU_DCP_MLA_TOPK_CAP=512
  export TPU_DCP_MLA_TOPK_CAP_CHUNK=256
  export TPU_DCP_MLA_UV_BEFORE_MERGE=0
  export TPU_DCP_RANK_OPERAND=1
  export TPU_DCP_RS_OUT_PRECISION=1

  export TPU_INDEXER_QB_OVERLAP=1
  export TPU_INDEXER_K_WRITE_COMPACT=1
  export TPU_INDEXER_ROPE_NEOX_PERMUTE=1
  export TPU_INDEXER_TOPK_SHARED_READ=1
  export TPU_MLA_DECODE_PROLOGUE=1
  export TPU_MLA_DECODE_ROW_WIDTHS=128,256,384
  export TPU_MLA_FUSED_SMALL_A2A=1
  export TPU_MLA_FUSED_SMALL_A2A_MERGE=1
  export TPU_MLA_KV_WRITE_IN_GATHER=1
  export TPU_MLA_SPARSE_SPLIT_Q=1
  export TPU_MLA_TC_KV_WRITER=1
  export TPU_SPARSE_MLA_LIVE_GATHER=1
  export TPU_SPARSE_MLA_PACKED_ROPE=1
  export TPU_ROPE_CACHE_TRUNCATE=1

  export TPU_STREAMIDX_CHUNK_TOKENS=256
  export TPU_STREAMIDX_MIXED_KV_PAGES=4
  export TPU_STREAMIDX_SKIP_EMPTY_DECODE=1
  export TPU_STREAMIDX_SKIP_EMPTY_DECODE_CHUNKED=1
  export TPU_STREAMIDX_SMALL_BUCKET_BKVP=32

  export TPU_SAMPLER_FAST_TOP_P=1
  export TPU_SAMPLER_FUSED_CHAIN=1
  export TPU_SAMPLER_VOCAB_PARALLEL=1
  export TPU_GREEDY_TOP_TOKENS=1
  export TPU_SMALL_ALL_REDUCE=1
  export TPU_SMALL_ALL_REDUCE_MAX_ROWS=32
  export TPU_SMALL_ALL_REDUCE_VALID_ROWS=1
  export TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS=0
}

verify_installed_code() {
  python3 - <<'PY'
import sys
import vllm_torchtpu
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.kernels.mla import kv_cache_utils

print(f"[glm53-repro] vllm_torchtpu loaded from: {vllm_torchtpu.__file__}")
if not hasattr(kv_cache_utils, "_compact_dcp_mla_rows"):
    sys.exit("FATAL: vllm_torchtpu is missing _compact_dcp_mla_rows (stale checkout or wrong PYTHONPATH / /workspace/vllm-torchtpu).")
if getattr(tpu_compiler, "PEER_WAIT_S", 0.0) < 7200.0:
    sys.exit(f"FATAL: tpu_compiler.PEER_WAIT_S={getattr(tpu_compiler, 'PEER_WAIT_S', None)} < 7200.0 (stale checkout).")
print(f"[glm53-repro] Verified code revision OK (PEER_WAIT_S={tpu_compiler.PEER_WAIT_S}).")
PY
}

ensure_local_tokenizer() {
  if [[ "${TOKENIZER}" != gs://* ]]; then
    return 0
  fi
  export_glm53_env || return 1
  local local_tok_dir="${LOCAL_TOKENIZER_DIR:-/dev/shm/cache/tokenizer}"
  if [[ ! -f "${local_tok_dir}/tokenizer_config.json" ]]; then
    log "Materializing non-weight tokenizer/config files from ${TOKENIZER} into ${local_tok_dir} ..."
    mkdir -p "${local_tok_dir}"
    python3 - "${TOKENIZER}" "${local_tok_dir}" <<'PY' || return 1
import sys
from runai_model_streamer.safetensors_streamer import safetensors_streamer

src_uri, dst_dir = sys.argv[1:3]
safetensors_streamer.pull_files(
    src_uri,
    dst_dir,
    ignore_pattern=["*.safetensors", "*.safetensors.*", "*.bin", "*.pt"],
)
PY
  fi
  TOKENIZER="${local_tok_dir}"
  log "Using local tokenizer directory: ${TOKENIZER}"
}

stop_glm53() {
  log "Stopping vLLM / EngineCore / Ray processes on current host..."
  pkill -9 -if "[v]llm|[e]nginecore|[r]ayworker" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
  rm -f /tmp/libtpu_lockfile
}

start_ray_head() {
  export_glm53_env || return 1
  verify_installed_code || return 1
  unset RAY_LOG_TO_STDERR
  export RAY_BACKEND_LOG_LEVEL=fatal
  local ray_port="${RAY_PORT:-6379}"
  ray start --head --port="${ray_port}" --num-cpus=64 --resources='{"TPU": 8}'
  log "Waiting for 2 Ray TPU hosts (TPU>=8 each)..."
  for _ in $(seq 1 180); do
    local alive
    alive="$(python3 -c "import ray; ray.init(address='auto', logging_level='error'); print(sum(1 for n in ray.nodes() if n['Alive'] and n.get('Resources', {}).get('TPU', 0) >= 8))" 2>/dev/null | tail -n 1 || echo 0)"
    if [[ "${alive}" -ge 2 ]]; then
      log "Ray cluster ready with ${alive} TPU hosts."
      ray status
      return 0
    fi
    sleep 10
  done
  log "ERROR: Timed out waiting for 2 Ray TPU hosts."
  return 1
}

start_ray_worker() {
  local head_ip="${1:?Usage: $0 ray-worker <HEAD_IP>}"
  local ray_port="${RAY_PORT:-6379}"
  export_glm53_env || return 1
  verify_installed_code || return 1
  unset RAY_LOG_TO_STDERR
  export RAY_BACKEND_LOG_LEVEL=fatal
  until (echo >/dev/tcp/"${head_ip}"/"${ray_port}") 2>/dev/null; do
    sleep 5
  done
  ray start --address="${head_ip}:${ray_port}" --num-cpus=64 --resources='{"TPU": 8}'
  log "Joined Ray head at ${head_ip}:${ray_port}"
}

run_serve() {
  local daemon_mode="${1:-}"
  export_glm53_env || return 1
  verify_installed_code || return 1
  if [[ "${CLEAN_CACHE:-0}" == "1" ]]; then
    log "CLEAN_CACHE=1: removing ${VLLM_CACHE_ROOT} ${TORCHINDUCTOR_CACHE_DIR} ${VLLM_XLA_CACHE_PATH}"
    rm -rf "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_XLA_CACHE_PATH}"
    mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${JAX_COMPILATION_CACHE_DIR}"
  fi
  mkdir -p "${OUT_DIR}"
  env | grep -E "^(TPU_|VLLM_|LIBTPU_|KERNEL_|FUSED_|USE_MOE|ONEHOT|ENABLE_PAW|PYTHONHASHSEED)" | sort >"${OUT_DIR}/server.env"

  local cmd=(
    vllm serve "${MODEL_URI}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tokenizer "${TOKENIZER}"
    --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
    --load-format "${LOAD_FORMAT}"
    --safetensors-load-strategy prefetch
    --data-parallel-size 1
    --tensor-parallel-size 16
    --decode-context-parallel-size 16
    --cp-kv-cache-interleave-size 4
    --enable-expert-parallel
    --enable-ep-weight-filter
    --kv-cache-dtype fp8
    --block-size "${BLOCK_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --kv-cache-memory-bytes "${KV_BYTES}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --no-enable-prefix-caching
    --async-scheduling
    --enable-prompt-tokens-details
    --reasoning-parser glm45
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --compilation-config '{"compile_sizes":[4,8,16,32,64,16384]}'
  )

  if [[ "${daemon_mode}" == "--daemon" ]]; then
    nohup setsid "${cmd[@]}" >"${OUT_DIR}/server.log" 2>&1 </dev/null &
    log "Launched vllm serve in background (pid=$!, log=${OUT_DIR}/server.log)"
  else
    exec "${cmd[@]}"
  fi
}

wait_server_ready() {
  local ep="http://${BENCH_HOST}:${PORT}"
  log "Waiting for ${ep}/health ..."
  for _ in $(seq 1 1080); do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${ep}/health" || echo 000)"
    if [[ "${code}" == "200" ]]; then
      log "Server is healthy at ${ep}"
      return 0
    fi
    sleep 10
  done
  log "ERROR: Server at ${ep} did not become ready within timeout."
  return 1
}

prom_snap() {
  curl -s "http://${BENCH_HOST}:${PORT}/metrics" >"${OUT_DIR}/metrics/$1.prom"
}

prom_delta() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys
metric, before_path, after_path = sys.argv[1:4]
def read_sum(path: str) -> float:
    total = 0.0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(metric + "{") or line.startswith(metric + " "):
                total += float(line.rsplit(" ", 1)[1])
    return total
print(read_sum(after_path) - read_sum(before_path))
PY
}

run_bench_random() {
  mkdir -p "${OUT_DIR}/metrics"
  ensure_local_tokenizer || return 1
  export JAX_PLATFORMS=cpu PJRT_DEVICE=CPU TPU_SKIP_MDS_QUERY=1 TRANSFORMERS_NO_TORCH_TPU=1 VLLM_TARGET_DEVICE=cpu

  local isl="${BENCH_ISL:-80128}"
  local osl="${BENCH_OSL:-512}"
  local concs="${BENCH_CONCS:-1 2 4 8 16 32 64}"
  local arm_name="${ARM:-glm53_tp16_dcp16_16k}"
  local ep="http://${BENCH_HOST}:${PORT}"

  wait_server_ready || return 1

  # 1. Smoke check
  curl -s "${ep}/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0.0}" \
    | tee "${OUT_DIR}/smoke.json"
  echo

  # 2. Warmup (80128 input / 32 output, 2 prompts at c=1)
  if [[ "${BENCH_SKIP_WARMUP:-0}" != "1" ]]; then
    log "Running warmup (${isl}/32, c=1, n=2)..."
    if ! PYTHONHASHSEED=999992 timeout 3600 vllm bench serve \
      --backend openai --host "${BENCH_HOST}" --port "${PORT}" --endpoint /v1/completions \
      --model "${SERVED_MODEL_NAME}" --tokenizer "${TOKENIZER}" --trust-remote-code \
      --dataset-name random --random-input-len "${isl}" --random-output-len 32 --random-range-ratio 0.0 \
      --ignore-eos --request-rate inf --num-prompts 2 --max-concurrency 1 \
      --num-warmups 0 --ready-check-timeout-sec 0 --seed 90031 --temperature 0.0 \
      >"${OUT_DIR}/warmup.log" 2>&1; then
      log "ERROR: Warmup vllm bench serve failed (see ${OUT_DIR}/warmup.log)"
      tail -n 30 "${OUT_DIR}/warmup.log" >&2
      return 1
    fi
    tail -n 5 "${OUT_DIR}/warmup.log"
  fi

  # 3. Full concurrency sweep
  echo "arm,c,n,completed,failed,duration_s,req_s,out_tok_s,total_tok_s,total_tok_s_chip,out_tok_s_chip,ttft_p50,ttft_mean,ttft_p99,tpot_p50,tpot_p99,itl_p50,e2el_p50,prefix_hits,preemptions,prompt_tok,gen_tok" >"${OUT_DIR}/summary.csv"
  local c
  for c in ${concs}; do
    local n=$((2 * c))
    local fname="glm53_${isl}_${osl}_c${c}.json"
    local fpath="${OUT_DIR}/${fname}"
    prom_snap "c${c}.before"
    log "Running random ${isl}/${osl} c=${c} n=${n} ..."
    if ! PYTHONHASHSEED=$((200000 + c)) timeout 7200 vllm bench serve \
      --backend openai --host "${BENCH_HOST}" --port "${PORT}" --endpoint /v1/completions \
      --model "${SERVED_MODEL_NAME}" --tokenizer "${TOKENIZER}" --trust-remote-code \
      --dataset-name random --random-input-len "${isl}" --random-output-len "${osl}" --random-range-ratio 0.0 \
      --ignore-eos --request-rate inf --max-concurrency "${c}" --num-prompts "${n}" \
      --num-warmups 0 --ready-check-timeout-sec 0 \
      --seed "$((50000 + c))" --temperature 0.0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 25,50,75,90,95,99 \
      --save-result --save-detailed --result-dir "${OUT_DIR}" --result-filename "${fname}" \
      >"${OUT_DIR}/c${c}.log" 2>&1 || [[ ! -s "${fpath}" ]]; then
      log "ERROR: Benchmark failed for c=${c} (see ${OUT_DIR}/c${c}.log)"
      tail -n 30 "${OUT_DIR}/c${c}.log" >&2
      return 1
    fi
    prom_snap "c${c}.after"

    local ph pe pt gt
    ph="$(prom_delta vllm:prefix_cache_hits_total "${OUT_DIR}/metrics/c${c}.before.prom" "${OUT_DIR}/metrics/c${c}.after.prom" 2>/dev/null || echo NA)"
    pe="$(prom_delta vllm:num_preemptions_total "${OUT_DIR}/metrics/c${c}.before.prom" "${OUT_DIR}/metrics/c${c}.after.prom" 2>/dev/null || echo NA)"
    pt="$(prom_delta vllm:prompt_tokens_total "${OUT_DIR}/metrics/c${c}.before.prom" "${OUT_DIR}/metrics/c${c}.after.prom" 2>/dev/null || echo NA)"
    gt="$(prom_delta vllm:generation_tokens_total "${OUT_DIR}/metrics/c${c}.before.prom" "${OUT_DIR}/metrics/c${c}.after.prom" 2>/dev/null || echo NA)"

    python3 - "${fpath}" "${arm_name}" "${c}" "${n}" "${NUM_CHIPS}" "${ph}" "${pe}" "${pt}" "${gt}" >>"${OUT_DIR}/summary.csv" <<'PY'
import json
import sys

fpath, arm, c, n, chips, ph, pe, pt, gt = sys.argv[1:10]
chips_f = float(chips)
with open(fpath, encoding="utf-8") as f:
    d = json.load(f)
tt = d["total_token_throughput"]
ot = d["output_throughput"]
row = [
    arm, c, n, d.get("completed", 0), d.get("failed", 0),
    round(d["duration"], 2), round(d["request_throughput"], 5),
    round(ot, 2), round(tt, 1), round(tt / chips_f, 1), round(ot / chips_f, 2),
    d.get("median_ttft_ms", ""), d.get("mean_ttft_ms", ""), d.get("p99_ttft_ms", ""),
    d.get("median_tpot_ms", ""), d.get("p99_tpot_ms", ""),
    d.get("median_itl_ms", ""), d.get("median_e2el_ms", ""),
    ph, pe, pt, gt,
]
print(",".join(str(x) for x in row))
PY
    tail -n 1 "${OUT_DIR}/summary.csv"
  done
  log "Sweep complete: ${OUT_DIR}/summary.csv"
  cat "${OUT_DIR}/summary.csv"
}

run_bench_timed_trace() {
  local shard_dir="${SHARD_DIR:?Set SHARD_DIR to directory containing timed_trace JSONL shards (and ensure server runs with MAX_MODEL_LEN=163840)}"
  local trace_prefix="${TRACE_PREFIX:-a_right_long}"
  local concs="${BENCH_CONCS:-1 2 4 8 16 32 64}"
  local num_prompts="${BENCH_NUM_PROMPTS:-256}"
  local arm_name="${ARM:-glm53_timed_trace_16k}"

  mkdir -p "${OUT_DIR}/metrics"
  ensure_local_tokenizer || return 1
  export JAX_PLATFORMS=cpu PJRT_DEVICE=CPU TPU_SKIP_MDS_QUERY=1 TRANSFORMERS_NO_TORCH_TPU=1 VLLM_TARGET_DEVICE=cpu

  wait_server_ready || return 1

  if [[ "${BENCH_SKIP_WARMUP:-0}" != "1" && -f "${shard_dir}/${trace_prefix}_warmup.jsonl" ]]; then
    log "Running timed_trace warmup (4 prompts, c=2)..."
    if ! PYTHONHASHSEED=999999 timeout 3600 vllm bench serve \
      --backend openai --host "${BENCH_HOST}" --port "${PORT}" --endpoint /v1/completions \
      --model "${SERVED_MODEL_NAME}" --tokenizer "${TOKENIZER}" --trust-remote-code \
      --dataset-name timed_trace --dataset-path "${shard_dir}/${trace_prefix}_warmup.jsonl" \
      --timed-trace-chunk-hash-size 512 --no-self-timed \
      --num-prompts 4 --num-warmups 0 --ready-check-timeout-sec 0 \
      --max-concurrency 2 --request-rate inf --temperature 0.0 >"${OUT_DIR}/warmup_trace.log" 2>&1; then
      log "ERROR: Timed-trace warmup failed (see ${OUT_DIR}/warmup_trace.log)"
      tail -n 30 "${OUT_DIR}/warmup_trace.log" >&2
      return 1
    fi
    tail -n 5 "${OUT_DIR}/warmup_trace.log"
  fi

  echo "arm,c,n,completed,failed,duration_s,req_s,out_tok_s,total_tok_s,total_tok_s_chip,out_tok_s_chip,ttft_p50,ttft_mean,ttft_p99,tpot_p50,tpot_p99,itl_p50,e2el_p50,prefix_hits,preemptions,prompt_tok,gen_tok" >"${OUT_DIR}/summary_timed_trace.csv"
  local c
  for c in ${concs}; do
    local fname="${trace_prefix}_n${num_prompts}_c${c}.json"
    local fpath="${OUT_DIR}/${fname}"
    prom_snap "trace_c${c}.before"
    log "Running timed_trace c=${c} n=${num_prompts} ..."
    if ! PYTHONHASHSEED=$((100000 + c)) timeout 14400 vllm bench serve \
      --backend openai --host "${BENCH_HOST}" --port "${PORT}" --endpoint /v1/completions \
      --model "${SERVED_MODEL_NAME}" --tokenizer "${TOKENIZER}" --trust-remote-code \
      --dataset-name timed_trace --dataset-path "${shard_dir}/${trace_prefix}_c${c}.jsonl" \
      --timed-trace-chunk-hash-size 512 --no-self-timed \
      --num-prompts "${num_prompts}" --num-warmups 0 --ready-check-timeout-sec 0 \
      --max-concurrency "${c}" --request-rate inf --temperature 0.0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 25,50,75,90,95,99 \
      --save-result --save-detailed --result-dir "${OUT_DIR}" --result-filename "${fname}" \
      >"${OUT_DIR}/trace_c${c}.log" 2>&1 || [[ ! -s "${fpath}" ]]; then
      log "ERROR: Timed-trace benchmark failed for c=${c} (see ${OUT_DIR}/trace_c${c}.log)"
      tail -n 30 "${OUT_DIR}/trace_c${c}.log" >&2
      return 1
    fi
    prom_snap "trace_c${c}.after"

    local ph pe pt gt
    ph="$(prom_delta vllm:prefix_cache_hits_total "${OUT_DIR}/metrics/trace_c${c}.before.prom" "${OUT_DIR}/metrics/trace_c${c}.after.prom" 2>/dev/null || echo NA)"
    pe="$(prom_delta vllm:num_preemptions_total "${OUT_DIR}/metrics/trace_c${c}.before.prom" "${OUT_DIR}/metrics/trace_c${c}.after.prom" 2>/dev/null || echo NA)"
    pt="$(prom_delta vllm:prompt_tokens_total "${OUT_DIR}/metrics/trace_c${c}.before.prom" "${OUT_DIR}/metrics/trace_c${c}.after.prom" 2>/dev/null || echo NA)"
    gt="$(prom_delta vllm:generation_tokens_total "${OUT_DIR}/metrics/trace_c${c}.before.prom" "${OUT_DIR}/metrics/trace_c${c}.after.prom" 2>/dev/null || echo NA)"

    python3 - "${fpath}" "${arm_name}" "${c}" "${num_prompts}" "${NUM_CHIPS}" "${ph}" "${pe}" "${pt}" "${gt}" >>"${OUT_DIR}/summary_timed_trace.csv" <<'PY'
import json
import sys

fpath, arm, c, n, chips, ph, pe, pt, gt = sys.argv[1:10]
chips_f = float(chips)
with open(fpath, encoding="utf-8") as f:
    d = json.load(f)
tt = d["total_token_throughput"]
ot = d["output_throughput"]
comp = d.get("completed", 0)
fail = int(n) - int(comp)
print(
    f"{arm},{c},{n},{comp},{fail},{d['duration']:.2f},{d['request_throughput']:.4f},"
    f"{ot:.2f},{tt:.2f},{tt/chips_f:.2f},{ot/chips_f:.2f},"
    f"{d.get('p50_ttft_ms', d.get('median_ttft_ms', 0.0)):.2f},"
    f"{d['mean_ttft_ms']:.2f},{d['p99_ttft_ms']:.2f},"
    f"{d.get('p50_tpot_ms', d.get('median_tpot_ms', 0.0)):.3f},{d['p99_tpot_ms']:.3f},"
    f"{d.get('p50_itl_ms', d.get('median_itl_ms', 0.0)):.3f},"
    f"{d.get('p50_e2el_ms', d.get('median_e2el_ms', 0.0)):.2f},"
    f"{ph},{pe},{pt},{gt}"
)
PY
    tail -n 1 "${OUT_DIR}/summary_timed_trace.csv"
  done
  log "Timed-trace sweep complete: ${OUT_DIR}/summary_timed_trace.csv"
  cat "${OUT_DIR}/summary_timed_trace.csv"
}

run_eval() {
  mkdir -p "${OUT_DIR}/eval"
  ensure_local_tokenizer || return 1
  export JAX_PLATFORMS=cpu PJRT_DEVICE=CPU TPU_SKIP_MDS_QUERY=1 TRANSFORMERS_NO_TORCH_TPU=1 VLLM_TARGET_DEVICE=cpu

  wait_server_ready || return 1

  log "Running Passkey retrieval (serial 16K & 80K x 5 depths + concurrent 10 x 80K)..."
  python3 - "${BENCH_HOST}" "${PORT}" "${SERVED_MODEL_NAME}" "${TOKENIZER}" "${OUT_DIR}/eval/passkey.json" <<'PY' | tee "${OUT_DIR}/eval/passkey.log"
import concurrent.futures
import json
import random
import sys
import urllib.request
from transformers import AutoTokenizer

host, port, model_name, tok_path, out_json = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
filler_sent = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
filler_ids = tok.encode(filler_sent, add_special_tokens=False)

def build_passkey_prompt(target_tokens: int, depth_frac: float, seed: int):
    rng = random.Random(seed)
    code = f"{rng.randint(10000000, 99999999)}"
    needle = f"\nThe pass key is {code}. Remember it. {code} is the pass key.\n"
    suffix = "\nWhat is the pass key? The pass key is"
    needle_ids = tok.encode(needle, add_special_tokens=False)
    suffix_ids = tok.encode(suffix, add_special_tokens=False)
    budget = target_tokens - len(needle_ids) - len(suffix_ids) - 8
    reps = (budget // len(filler_ids)) + 2
    full_filler = (filler_ids * reps)[:budget]
    split = int(len(full_filler) * depth_frac)
    prompt_text = tok.decode(full_filler[:split]) + needle + tok.decode(full_filler[split:]) + suffix
    return prompt_text, code

def query(prompt_text: str):
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        data=json.dumps({
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": 24,
            "temperature": 0.0,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read().decode())
    return d["choices"][0]["text"], d["usage"]["prompt_tokens"]

serial_results = []
for length in [16384, 79872]:
    for idx, depth in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        p, code = build_passkey_prompt(length, depth, seed=length * 10 + idx)
        out, ptoks = query(p)
        ok = code in out
        serial_results.append({"target_len": length, "prompt_tokens": ptoks, "depth": depth, "code": code, "output": out.strip(), "pass": ok})
        print(f"[serial] len={ptoks} depth={depth:.1f} expected={code} got={out.strip()!r} pass={ok}", flush=True)

conc_prompts = [build_passkey_prompt(79872, 0.1 + 0.08 * i, seed=9000 + i) for i in range(10)]
conc_results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    futs = [ex.submit(query, p) for p, _ in conc_prompts]
    for i, (fut, (_, code)) in enumerate(zip(futs, conc_prompts)):
        out, ptoks = fut.result()
        ok = code in out
        conc_results.append({"idx": i, "prompt_tokens": ptoks, "code": code, "output": out.strip(), "pass": ok})
        print(f"[concurrent 80K #{i}] len={ptoks} expected={code} got={out.strip()!r} pass={ok}", flush=True)

summary = {
    "serial_passed": sum(r["pass"] for r in serial_results),
    "serial_total": len(serial_results),
    "concurrent_passed": sum(r["pass"] for r in conc_results),
    "concurrent_total": len(conc_results),
    "serial": serial_results,
    "concurrent": conc_results,
}
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"PASSKEY SUMMARY: serial={summary['serial_passed']}/{summary['serial_total']}, concurrent={summary['concurrent_passed']}/{summary['concurrent_total']}", flush=True)
PY

  local eval_venv="${EVAL_VENV:-/dev/shm/eval_venv}"
  if [[ ! -x "${eval_venv}/bin/lm_eval" ]]; then
    log "Installing lm_eval[api] into ${eval_venv} ..."
    python3 -m venv "${eval_venv}"
    "${eval_venv}/bin/pip" install -q --upgrade pip
    "${eval_venv}/bin/pip" install -q "lm_eval[api]"
  fi

  log "Running GSM8K (5-shot, greedy, reasoning_effort=low)..."
  OPENAI_API_KEY=EMPTY "${eval_venv}/bin/lm_eval" \
    --tasks gsm8k --seed "0,1234,None,1234" \
    --model local-chat-completions --apply_chat_template \
    --model_args "model=${SERVED_MODEL_NAME},base_url=http://${BENCH_HOST}:${PORT}/v1/chat/completions,num_concurrent=64,max_retries=5,timeout=3600,tokenized_requests=False" \
    --gen_kwargs '{"chat_template_kwargs": {"reasoning_effort": "low"}, "max_gen_toks": 1024, "temperature": 0.0}' \
    --output_path "${OUT_DIR}/eval/gsm8k" 2>&1 | tee "${OUT_DIR}/eval/gsm8k.log"

  log "Running MMLU-Pro (--limit 100, 5-shot, greedy, reasoning_effort=low)..."
  OPENAI_API_KEY=EMPTY "${eval_venv}/bin/lm_eval" \
    --tasks mmlu_pro --limit 100 --seed "0,1234,None,1234" \
    --model local-chat-completions --apply_chat_template \
    --model_args "model=${SERVED_MODEL_NAME},base_url=http://${BENCH_HOST}:${PORT}/v1/chat/completions,num_concurrent=64,max_retries=5,timeout=3600,tokenized_requests=False" \
    --gen_kwargs '{"chat_template_kwargs": {"reasoning_effort": "low"}, "max_gen_toks": 1024, "temperature": 0.0}' \
    --output_path "${OUT_DIR}/eval/mmlu_pro" 2>&1 | tee "${OUT_DIR}/eval/mmlu_pro.log"
}

main() {
  local action="${1:-help}"
  shift || true
  case "${action}" in
    env)
      export_glm53_env
      ;;
    ray-head)
      start_ray_head
      ;;
    ray-worker)
      start_ray_worker "$@"
      ;;
    stop)
      stop_glm53
      ;;
    serve)
      run_serve "$@"
      ;;
    bench-random)
      run_bench_random
      ;;
    bench-timed-trace)
      run_bench_timed_trace
      ;;
    eval)
      run_eval
      ;;
    all)
      run_serve --daemon || return 1
      run_bench_random || return 1
      if [[ "${RUN_EVAL:-0}" == "1" ]]; then
        run_eval || return 1
      fi
      ;;
    *)
      cat <<EOF
Usage:
  source $0 env
  $0 ray-head
  $0 ray-worker <HEAD_IP>
  $0 stop
  $0 serve [--daemon]
  $0 bench-random
  SHARD_DIR=/path/to/shards $0 bench-timed-trace
  $0 eval
  $0 all
EOF
      return 1
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  export_glm53_env
else
  main "$@"
fi
