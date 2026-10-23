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
# Benchmark config: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (MoE, TP=2 on v7x-8), multimodal (image input).
MODEL="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
TENSOR_PARALLELISM=2
DATA_PARALLELISM=4
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="512"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
EVAL_TOLERANCE="0.03"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--block-size=256 --max-num-seqs=512 --mm-processor-kwargs {\"size\":{\"longest_edge\":262144,\"shortest_edge\":3136}} --disable-chunked-mm-input --mm-encoder-tp-mode data --compilation-config {\"cudagraph_mm_encoder\":true,\"encoder_cudagraph_max_vision_items_per_batch\":8,\"encoder_cudagraph_max_frames_per_batch\":8}"
