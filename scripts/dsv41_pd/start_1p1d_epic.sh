#!/usr/bin/env bash
# 1P1D（Stage-3 Raiden）启动脚本，跑 cszhz/vllm-torchtpu 的 cszhz/dsv41-1p1d 分支（= 最新 epic/dsv41 + PD 提交）。
#   2z6z: bash start_1p1d_epic.sh prefill   → :8100
#   clsk: bash start_1p1d_epic.sh decode    → :8200
# vllm serve 的参数照 epic/dsv41 的 scripts/dsv41_bench/serve_cached_prefill.sh（server_steps 基准）：
#   --additional-config 选 CED + fused_ep_w4a8（取代 TPU_DEEPSEEK_V41_CED / USE_MOE_FUSED_EP_KERNEL 等环境变量）、
#   --engram-config cpu_offload、KV 自动分配（--gpu-memory-utilization 0.85）、P 的编译桶 [128,256,2048]。
# 和基准不同的地方（PD 需要）：
#   --kv-transfer-config TPURaidenConnector + Stage-3 环境变量；--swa-bounded-replay + TPU_RAIDEN_PREFIX_AWARE_LOAD=1
#   （D 只拉本地前缀命中之后的部分，需要 apply_patches.py 第 4 节的 vLLM 调度器补丁）；
#   ASYNC_SCHED 默认 0（--no-async-scheduling，和原 PD 配方一致；ASYNC_SCHED=1 改成 --async-scheduling）。
# 代码目录用 CODE_DIR 指定；PYTHONPATH 放在前面，盖过 venv 里 editable 安装的旧目录。
set -euo pipefail
ROLE="${1:?usage: start_1p1d_epic.sh <prefill|decode>}"
CODE_DIR="${CODE_DIR:-/zzlfs/temu/src/vllm-torchtpu-cszhz}"

# 新代码要求 torch-tpu 0.2 / jax 0.11.2 / libtpu 0.0.50，用单独的 venv（旧 .venv 留给旧 PD 部署）。
source "${VENV:-/zzlfs/temu/.venv-epic}/bin/activate"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
ulimit -c 0

export MODEL_URI="gs://zzl-bucket/models/DeepSeek-V4.1-Flash"
export TOKENIZER_DIR="/tmp/dsv41-tokenizer"
export POD_IP="${POD_IP:-$(hostname -I | awk '{print $1}')}"
export VLLM_HOST_IP="${POD_IP}"
# --- 单机模式：只用本机 4 颗芯片（这台机器是 tpu7x-32 多机 slice 的一个 worker）---
export TPU_SKIP_MDS_QUERY=1
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1
export TPU_VISIBLE_DEVICES=0,1,2,3
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export JAX_PLATFORMS=tpu,cpu
export PJRT_DEVICE=TPU
export TPU_BACKEND_TYPE=jax
export TPU_ACCELERATOR_TYPE=tpu7x
export TPU_TOPOLOGY=2x2x1
export TPU_HOST_BOUNDS=1,1,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VLLM_XLA_CHECK_RECOMPILATION=0
# --- 与 serve_cached_prefill.sh 相同 ---
# 新代码的 host_memory.py 要求进程里的 malloc 来自 glibc；这套 venv 里不是（有别的分配器先加载），开着会在编译时报错。
export TPU_RECLAIM_COMPILE_MEMORY="${TPU_RECLAIM_COMPILE_MEMORY:-0}"
export OMP_NUM_THREADS=1 NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm VLLM_TARGET_DEVICE=tpu
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_USE_AOT_COMPILE=0
export RUNAI_STREAMER_MEMORY_LIMIT=34359738368 TPU_SHARDED_LOAD_SYNC_EVERY=512
export VLLM_ENGINE_READY_TIMEOUT_S=7200 TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export LIBTPU_INIT_ARGS='--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0 --xla_tpu_dvfs_p_state=7'
export MOE_FUSED_EP_PREPROCESS_BACKEND="${MOE_FUSED_EP_PREPROCESS_BACKEND:-native}"
export VLLM_SERVER_DEV_MODE=1
# 新 DP wave 开始时等几毫秒再走第一步，让几乎同时到达其他 rank 的请求（4P）赶上同一步，
# 否则晚到的请求要多等整整一步（~180 ms）。需要 apply_venv_patches.py 第 6 节。0 = 不等。
# P 默认 40 ms：并发长请求经多个 API server 解析后到达各 rank 仍相差几十 ms。
if [[ "${ROLE}" == "prefill" ]]; then _WAVE_DEFAULT=40; else _WAVE_DEFAULT=20; fi
export VLLM_DP_WAVE_START_DELAY_MS="${VLLM_DP_WAVE_START_DELAY_MS:-${_WAVE_DEFAULT}}"
export PYTHONUNBUFFERED=1
# --- Stage-3 Raiden DSv4 KV 传输 ---
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_RAIDEN_DSV4_ADMISSION=1
export TPU_RAIDEN_DSV4_DEBUG_CHECKSUMS=0
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_ADVERTISE_HOST="${POD_IP}"
export TPU_RAIDEN_TRANSFER_PARALLELISM=1
export TPU_PREMAPPED_BUFFER_SIZE=17179869184
export TPU_P2P_WAIT_PULL_TIMEOUT=600
export TPU_KV_SHM_POOL_GB="${TPU_KV_SHM_POOL_GB:-128}"
export TPU_KV_TRANSFER_PORT=9100
export TPU_SIDE_CHANNEL_PORT=9600
export TPU_RAIDEN_PREFIX_AWARE_LOAD="${TPU_RAIDEN_PREFIX_AWARE_LOAD:-1}"
if [[ "${ROLE}" == "prefill" ]]; then
  export TPU_KV_TRANSFER_NAMESPACE="gke_prefill" TPU_RAIDEN_ENGINE_ID="gke-prefill-engine" TPU_RAIDEN_JOB_NAME="prefill"
  export TPU_RAIDEN_RESHARD_PORT_BASE=27000 TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
  KV_ROLE="kv_producer"; SERVER_PORT=8100
  COMPILE_SIZES="${PREFILL_COMPILE_SIZES:-[128,256,${MAX_BATCHED_TOKENS:-2048}]}"
elif [[ "${ROLE}" == "decode" ]]; then
  export TPU_KV_TRANSFER_NAMESPACE="gke_decode" TPU_RAIDEN_ENGINE_ID="gke-decode-engine" TPU_RAIDEN_JOB_NAME="decode"
  export TPU_RAIDEN_RESHARD_PORT_BASE=28000 TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
  KV_ROLE="kv_consumer"; SERVER_PORT=8200
  # 16：单请求 decode（1 + K5 = 6 行）；96：16 请求的 verify；128/2048：D 本地 prefill / 回退重算
  COMPILE_SIZES="${DECODE_COMPILE_SIZES:-[16,96,128,2048]}"
else
  echo "Usage: $0 <prefill|decode>"; exit 1
fi
# API server 进程数：长 prompt 的请求要在 API server 里解析 ~10^6 个 token id 的 JSON（~0.1 s），
# 1 个进程时并发的 4 条请求排队解析，到达引擎时相差 ~0.3 s，DP lockstep 下晚到的要多等一步。
if [[ "${ASYNC_SCHED:-0}" == "1" ]]; then ASYNC_FLAG="--async-scheduling"; else ASYNC_FLAG="--no-async-scheduling"; fi
if [[ "${KV_BYTES:-auto}" == "auto" ]]; then KV_ARGS=(--gpu-memory-utilization 0.85); else KV_ARGS=(--kv-cache-memory-bytes "${KV_BYTES}"); fi
echo "role=${ROLE} host=$(hostname) ip=${POD_IP} code=${CODE_DIR} commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) tp=${TP_SIZE:-1} dp=${DP_SIZE:-8} compile_sizes=${COMPILE_SIZES} async=${ASYNC_FLAG} kv=${KV_BYTES:-auto} max_model_len=${MAX_MODEL_LEN:-default} extra_args=${EXTRA_ARGS:-}"
MML_ARGS=(); [[ -n "${MAX_MODEL_LEN:-}" ]] && MML_ARGS=(--max-model-len "${MAX_MODEL_LEN}")
# NO_KV_TRANSFER=1：不挂 KV connector，作为独立服务启动（用来单独测 P 的 prefill，比如 TP8：
# TP>1 时 CSA2 关闭，RoPE 附属池的 block 数和主池不同，tpu-sync KVCacheManager 不接受）。
if [[ "${NO_KV_TRANSFER:-0}" == "1" ]]; then
  KV_ARGS_PD=()
else
  KV_ARGS_PD=(--kv-transfer-config "{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"${KV_ROLE}\",\"kv_port\":14579}")
fi
exec vllm serve "${MODEL_URI}" \
  --tokenizer "${TOKENIZER_DIR}" \
  --load-format runai_streamer --served-model-name deepseek-v41 \
  --host 0.0.0.0 --port "${SERVER_PORT}" --api-server-count "${API_SERVERS:-4}" \
  --tensor-parallel-size "${TP_SIZE:-1}" --data-parallel-size "${DP_SIZE:-8}" --data-parallel-size-local "${DP_SIZE:-8}" \
  --enable-expert-parallel --enable-ep-weight-filter --language-model-only \
  "${ASYNC_FLAG}" --enable-prefix-caching --swa-bounded-replay --kv-cache-dtype fp8 \
  --attention-config '{"indexer_kv_dtype":"fp8"}' \
  --engram-config '{"cpu_offload":true}' \
  --additional-config '{"deepseek_v41":{"execution":"ced","moe":"fused_ep_w4a8"}}' \
  --block-size 16 \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}" --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  "${KV_ARGS[@]}" "${MML_ARGS[@]}" \
  --compilation-config "{\"compile_sizes\":${COMPILE_SIZES}}" \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"model\":\"${MODEL_URI}\"}" \
  "${KV_ARGS_PD[@]}" \
  ${EXTRA_ARGS:-}
