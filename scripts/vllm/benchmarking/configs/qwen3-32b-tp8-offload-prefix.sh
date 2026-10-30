#!/bin/bash
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

# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Dense long-context model with Raiden KV offload under prefix reuse.
# 100 prefixes of 8192 tokens, each request adds a 256-token suffix, so most
# prompt tokens are reloaded from the host pool. The device pool is capped at
# DEVICE_KV_TOKENS so prefixes are evicted and reloaded. KV_PAGE_SIZE=16
# compares against the old default page size at the same token capacity.
# FLASH_ATTN is the backend whose default page size depends on max_model_len.
# CUSTOM only supports 128 and 256 pages, so it cannot run the 16-token arm.
MODEL="Qwen/Qwen3-32B"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=false
MAX_MODEL_LEN=16384
ISL_OSL_CONFIGS="8448:256"
CONCURRENCY_OPTIONS="32"
# get_tpu_min_page_size raises the page to cdiv(MAX_MODEL_LEN, 1M/8/MAX_NUM_SEQS).
# The harness default of 512 seqs would turn the 16-token arm into 64.
MAX_NUM_SEQS=128
NUM_PROMPTS=1000
# prefix_repetition requires num_requests >= num_prefixes (100).
PROFILE_NUM_PROMPTS="${PROFILE_NUM_PROMPTS:-100}"
BENCHMARK_TEMPERATURE="${BENCHMARK_TEMPERATURE:-0}"
# Required by run_benchmarks.sh. prefix_repetition does not use it.
RANDOM_RANGE_RATIO="0.0"
EVAL_TOLERANCE="0.02"
ATTENTION_BACKEND="FLASH_ATTN"
ENABLE_PREFIX_CACHING=true
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"
DATASET_NAME="prefix_repetition"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--prefix-repetition-prefix-len 8192 --prefix-repetition-suffix-len 256 --prefix-repetition-num-prefixes 100 --prefix-repetition-output-len 256"
KV_PAGE_SIZE="${KV_PAGE_SIZE:-128}"
DEVICE_KV_TOKENS="${DEVICE_KV_TOKENS:-262144}"
OFFLOAD_CPU_BYTES="${OFFLOAD_CPU_BYTES:-322122547200}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size ${KV_PAGE_SIZE} --num-gpu-blocks-override $((DEVICE_KV_TOKENS / KV_PAGE_SIZE)) --kv-transfer-config {\"kv_connector\":\"TPURaidenOffloadingConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.offload.raiden_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${OFFLOAD_CPU_BYTES}}}"
