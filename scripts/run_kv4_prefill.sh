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
# One process per config of bench_kv4_prefill.py (the kernels read env at
# import). Usage: run_kv4_prefill.sh <round>
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src:$PYTHONPATH
X=/tmp/kv4pf_xprof; mkdir -p $X
B="python3 scripts/bench_kv4_prefill.py"
case "${1:-r1}" in
r1)
  $B --preset c16 --kv fp8 --label c16-fp8 --xprof $X
  $B --preset c16kv4 --kv int4 --label c16kv4-int4 --xprof $X
  $B --preset c16 --kv int4 --label c16-int4-bq32
  $B --preset c16kv4 --kv int4 --label kv4-lbq1024 --large-bq 1024 --parts score
  $B --preset c16kv4 --kv int4 --label kv4-lbq2048 --large-bq 2048 --parts score
  $B --preset c16 --kv fp8 --label fp8-lbq1024 --large-bq 1024 --parts score
  $B --preset c16kv4 --kv int4 --label kv4-bf16cache --parts attend \
     --env TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16=1
  $B --preset c64 --dcp 4 --kv fp8,int4 --label c64 --xprof $X
  $B --preset c64 --dcp 4 --kv int4 --label c64-lbq1024 --large-bq 1024 --parts score
  $B --preset c64 --dcp 4 --kv int4 --label c64-bf16cache --parts attend \
     --env TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_BF16=1
  ;;
r2)
  # int4fix = int4 caches + TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q=1024 (bitwise check in-process)
  $B --preset c16 --kv fp8,int4,int4fix --label c16
  $B --preset c16kv4 --kv int4,int4fix --label c16kv4
  $B --preset c64 --dcp 4 --kv fp8,int4,int4fix --label c64 --xprof $X
  $B --preset c64 --dcp 4 --kv int4,int4fix --label c64-lbq2048 --parts score \
     --fix-env TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q=2048 --tokens 8192
  ;;
r3)
  # int4 attend env sweep at the c16 env (BQ32 / 32 pages / cache 512 / tile pages 64)
  A="--preset c16 --kv int4 --parts attend"
  $B $A --label c16-int4
  $B $A --label nocache --env TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE=0
  $B $A --label pps16 --env TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP=16
  $B $A --label pps64 --env TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP=64
  $B $A --label tile192 --env TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES=192
  $B $A --label bq16 --env TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q=16
  $B $A --label bq64 --env TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q=64
  $B $A --label rs2 --env TPU_MINIMAX_M3_MSA_PREFILL_ROW_SPLIT=2
  $B $A --label nomaxfree --env TPU_MINIMAX_M3_MSA_MAX_FREE=0
  ;;
*) echo "unknown round $1"; exit 2 ;;
esac
echo "=== run_kv4_prefill done"
