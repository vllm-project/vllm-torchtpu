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
# Benchmark config: SUFE-AIFLM-Lab/Fin-R1 (Qwen2.5-7B-Instruct financial reasoning, TP=8 on v7x-8).
# Uses GQA head padding (28 Q heads -> 32 Q heads, 4 KV heads replicated -> 1 KV head/rank).
MODEL="SUFE-AIFLM-Lab/Fin-R1"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="16 64 128"
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.9
# Uncalibrated BF16 Qwen2.5-7B checkpoint has large Layer 0 k_proj.bias (L2 norm 466.6 > fp8 max 448),
# so use auto (bfloat16) KV cache to preserve CoT reasoning accuracy in lm_eval.
KV_CACHE_DTYPE="auto"
BENCHMARK_TEMPERATURE=0
# mmlu_pro only: LM_EVAL_GEN_KWARGS replaces the harness gen_kwargs, which
# drops the continuation flags mmlu_llama's prefilled prompt needs.
LM_EVAL_TASKS="mmlu_pro"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
EXTRA_LM_EVAL_MODEL_ARGS="max_gen_toks=4096"
LM_EVAL_GEN_KWARGS='{"until": ["<|im_end|>"], "max_gen_toks": 4096}'
EVAL_TOLERANCE="0.02"
