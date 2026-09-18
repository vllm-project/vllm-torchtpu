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

# One engine of the 1P1D disaggregated benchmark.
#
#   disagg_engine.sh prefill|decode
#
# The two engines agree on all but a handful of settings, so they share this
# file and differ only in the case blocks below. The manifest passes the role
# and supplies POD_IP, which needs the downward API.
set -euo pipefail

role="${1:-}"
case "$role" in
  prefill|decode) ;;
  *) echo "usage: $0 prefill|decode" >&2; exit 2 ;;
esac

# Core dumps are ~4GiB each and overflow the pod's ephemeral storage.
ulimit -c 0
export VLLM_HOST_IP="${POD_IP}"

# Same for both engines.
export JAX_PLATFORMS=tpu,cpu
export MEGASCALE_COORDINATOR_ADDRESS=''
export MEGASCALE_NUM_SLICES=''
export MEGASCALE_PORT=''
export MEGASCALE_SLICE_ID=''
export PJRT_DEVICE=TPU
export RAY_LOG_TO_STDERR=1
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled
export TPU_ACCELERATOR_TYPE=tpu7x
export TPU_BACKEND_TYPE=jax
export TPU_GDN_CONV_QK_PAIR_LAYOUT=1
export TPU_HOST_BOUNDS=1,1,1
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_KV_SHM_POOL_GB=8
export TPU_KV_TRANSFER_PORT=9100
export TPU_P2P_WAIT_PULL_TIMEOUT=600
export TPU_PREMAPPED_BUFFER_SIZE=17179869184
export TPU_PROCESS_ADDRESSES=''
export TPU_PROCESS_PORT=''
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_TRANSFER_PARALLELISM=8
export TPU_SIDE_CHANNEL_PORT=9600
export TPU_TOPOLOGY=2x2x1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1
export TPU_WORKER_HOSTNAMES=''
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_CACHE_ROOT=/cache/jax/vllm
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_LOGGING_LEVEL=INFO
export VLLM_XLA_CACHE_PATH=/cache/jax/xla
export VLLM_XLA_CHECK_RECOMPILATION=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Identity and ports, which must differ so the two do not collide.
case "$role" in
  prefill)
    export TPU_KV_TRANSFER_NAMESPACE=gke_prefill
    export TPU_RAIDEN_ENGINE_ID=gke-prefill-engine
    export TPU_RAIDEN_JOB_NAME=prefill
    export TPU_RAIDEN_RESHARD_PORT_BASE=27000
    export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
    ;;
  decode)
    export TPU_KV_TRANSFER_NAMESPACE=gke_decode
    export TPU_RAIDEN_ENGINE_ID=gke-decode-engine
    export TPU_RAIDEN_JOB_NAME=decode
    export TPU_RAIDEN_RESHARD_PORT_BASE=28000
    export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
    ;;
esac

serve_args=(
  --model=Qwen/Qwen3.5-397B-A17B-FP8
  --trust-remote-code
  --seed=42
  --max-model-len=66560
  --block-size=8192
  --enable-expert-parallel
  --disable-custom-all-reduce
  --gpu-memory-utilization=0.9
  --kv-cache-dtype=fp8
  --language-model-only
  --no-disable-hybrid-kv-cache-manager
  --attention-backend=CUSTOM
  --mamba-cache-mode=align
  --enable-prompt-tokens-details
  --no-enable-log-requests
  --no-enable-prefix-caching
  --no-async-scheduling
  --tensor-parallel-size=1
)

case "$role" in
  prefill)
    serve_args+=(
      --max-num-batched-tokens=16384
      --long-prefill-token-threshold=16384
      --max-num-seqs=8
      --prefill-context-parallel-size=8
      --cp-kv-cache-interleave-size=256
      --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[1024,2048,4096,5120],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
      --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_port":14579}'
    )
    ;;
  decode)
    serve_args+=(
      --max-num-batched-tokens=4384
      --max-num-seqs=32
      --prefill-context-parallel-size=1
      --data-parallel-size=8
      --data-parallel-size-local=8
      --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[8,16,32,4352,4384],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
      --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer","kv_port":14579}'
    )
    ;;
esac

exec vllm serve "${serve_args[@]}"
