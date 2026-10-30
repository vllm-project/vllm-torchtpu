#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Source in the container: source ./env.sh HOST_RANK
HOST_RANK=${1:?Pass host rank 0, 1, 2 or 3}
: "${NATIVE_HOST_IPS:?Set four comma-separated host addresses in rank order}"
: "${MODEL_PATH:?Set the model configuration and tokenizer directory}"
IFS=',' read -r -a HOST_IPS <<< "$NATIVE_HOST_IPS"
if [[ ${#HOST_IPS[@]} != 4 || ! "$HOST_RANK" =~ ^[0-3]$ ]]; then
  echo 'Expected four host addresses and a host rank in 0..3' >&2
  return 2
fi
export POD_IP=${HOST_IPS[$HOST_RANK]}
export MASTER_ADDRESS=${HOST_IPS[0]}
export VLLM_HOST_IP="$POD_IP"
export TORCH_TPU_DP_MASTER_ADDR="$MASTER_ADDRESS"
export PYTHON_BIN=${PYTHON_BIN:-python3}
export VLLM_BIN=${VLLM_BIN:-$(command -v vllm)}
export DP_START_RANK=$((HOST_RANK * 2))
NATIVE_RUN_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NATIVE_RUN_DIR
# Let the worker launcher assign distributed ranks.
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export LD_PRELOAD=${TCMALLOC_LIBRARY:-/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4}
export TZ=Asia/Shanghai
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=224
export MALLOC_ARENA_MAX=16
export LIBTPU_INIT_ARGS='--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864 --xla_tpu_enable_diagnostic_coordinator=false --tpu_core_dump_directory=/dev/shm/tpu_core_dumps --tpu_core_dump_arch_states=TPU_CORE_DUMP_ARCH_STATE_MINIMAL,TPU_CORE_DUMP_ARCH_STATE_TC,TPU_CORE_DUMP_ARCH_STATE_SCS,TPU_CORE_DUMP_ARCH_STATE_SC_TEC,TPU_CORE_DUMP_ARCH_STATE_SC_SGE --xla_tpu_report_hlo_details_in_core_dump=true --xla_tpu_use_enhanced_launch_barrier=true'
export TPU_ACCELERATOR_TYPE=tpu7x-32
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,4
export TPU_TOPOLOGY=2x2x4
export TPU_TOPOLOGY_WRAP=false,false,false
export TPU_SKIP_MDS_QUERY=true
export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7
export TPU_MULTIHOST_BACKEND=mp
export TPU_PROCESS_PORT=8471
export TPU_SIDE_CHANNEL_PORT=19600
export TPU_PREMAPPED_BUFFER_SIZE=4294967296
export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1
export TPU_GDN_CONV_STATE_TILE_PAD=1
export TPU_GDN_CONV_QK_PAIR_LAYOUT=1
export TPU_PARALLEL_PRECOMPILE=1
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export XLA_FLAGS=--xla_enable_hlo_sharding_v3=false
export USE_BATCHED_RPA_LONGCTX=1
export TPU_ROPE_CACHE_TRUNCATE=1
export VLLM_TARGET_DEVICE=tpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_KV_CACHE_LAYOUT=HND
export VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
export VLLM_PLUGINS=torchtpu,torchtpu_layers
export TORCH_TPU_DP_MASTER_PORT=29750
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_HOST_IP="$POD_IP"
export PYTHONPATH="$NATIVE_RUN_DIR"
# Resolve package locations from the active Python environment.
export TPU_LIBRARY_PATH=${TPU_LIBRARY_PATH:-$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("libtpu").locate_file("libtpu/libtpu.so"))')}
TORCH_LIBRARY_DIR=$("$PYTHON_BIN" -c 'from importlib.metadata import distribution; print(distribution("torch").locate_file("torch/lib"))')
export LD_LIBRARY_PATH="$TORCH_LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$NATIVE_RUN_DIR/compile-cache}
export TORCH_TPU_TIER2_COMPILATION_CACHE="native_decode_${POD_IP}"
