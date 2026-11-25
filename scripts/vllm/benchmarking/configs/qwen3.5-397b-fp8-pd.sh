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

# shellcheck disable=SC2034  # Read by qwen35_pd.sh, its correctness stages and run_benchmarks.sh
#
# Qwen3.5-397B-A17B-FP8 served prefill/decode disaggregated: the model's C3
# config. One engine of the pair per sourcing; PD_ROLE picks which. Sourced by
# .buildkite/kubernetes/qwen35_pd.sh on each engine pod (which serves
# SERVE_ARGS with this file's environment), by
# scripts/vllm/integration/run_qwen35_397b_pd_correctness.sh (which reads the
# thresholds below) and by run_benchmarks.sh when run_eval_flow.sh gates the
# pair through its proxy (which reads the model and the eval settings).
#
# Where the settings come from. Each leg is the leg of a measured record in
# src/vllm_torchtpu/models/qwen3_5/validation/, with that record's serve
# arguments, envs.py environment and --additional-config sections unchanged:
#   PD_DECODE_LAYOUT=dp4tp2   benchmark point p01 (DP4 x TP2 decode, PP8
#                             prefill), as its 1P1D accuracy gate
#                             gate-dp4tp2-1p1d.yaml runs it
#   PD_DECODE_LAYOUT=tp8dcp4  benchmark point p08 (TP8 + DCP4 decode, PP8
#                             prefill), as gate-tp8-dcp4-1p1d.yaml runs it
# tests/models/qwen3_5/test_qwen3_5_recipe.py renders every role, layout and
# trade-off arm and holds them equal to those legs.
#
# Variables the records' runs set that the records cannot carry (they list
# only this package's envs.py variables; these are read by vLLM, libtpu, the
# Raiden library, torch_tpu or the model loader), each explained where it is
# set below:
#   VLLM_KV_CACHE_LAYOUT, VLLM_PP_LAYER_PARTITION, LIBTPU_INIT_ARGS,
#   RAIDEN_EXPECTED_WORKERS_TIMEOUT_S, RAIDEN_NUMA_POLICY,
#   RUNAI_STREAMER_MEMORY_LIMIT, TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS,
#   VLLM_ALLOW_LONG_MAX_MODEL_LEN, VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
#   VLLM_TARGET_DEVICE, VLLM_DP_MASTER_IP, OMP_NUM_THREADS,
#   TPU_PREMAPPED_BUFFER_SIZE, XLA_PYTHON_CLIENT_PREALLOCATE, VLLM_LOGGING_LEVEL
# and TPU_KV_RESHARD_TRANSPORT (an envs.py variable the records leave to the
# deployment with the Raiden ports), plus the one-host pod environment and the
# Raiden ports and identities, which are per pod and so not recorded either.
#
# Departures from the measured runs that CI forces, one per line:
#   VLLM_ENGINE_READY_TIMEOUT_S 7200, not 2700: CI boots both legs from a
#                             cold compile cache.
#   compile caches            the CI fleet's (the manifest), not the measured
#                             runs' own cache directories.
#   front end                 vllm-router (PD_PROXY=router, the default) or
#                             the toy proxy, not the llm-d router the records
#                             ran behind; same two-hop protocol.
#   GSM8K harness             run_eval_flow.sh's lm-eval (5-shot, chat
#                             template, fewshot as multiturn), not InferenceX's
#                             harness that produced the records' GSM8K, so its
#                             baseline is this recipe's own measurement.
#   host setup                no privileged node setup (vm.max_map_count, the
#                             vfio DMA entry limit) and the fleet's memory
#                             request, not 850 GiB.
#
# Parameters (environment, before sourcing):
#   PD_ROLE              prefill | decode (required)
#   PD_DECODE_LAYOUT     dp4tp2 (default) | tp8dcp4; anything else is an error
#   PD_TRADEOFFS         on (default) | off. on runs the Qwen3.5 FP8 trade-off
#                        settings (qwen3_5 native_fp8_attention,
#                        quantize_target_lm_head, exp2_softmax) and the bf16
#                        GDN SSM state on both legs, as every record did; off
#                        is the exact mode, main's precision for all four (the
#                        gates' tradeoffs_off arm). GSM8K measured on vs off:
#                        0.9750 vs 0.9742 at DP4 x TP2, 0.9773 vs 0.9773 at
#                        TP8 + DCP4 (1,319 questions each).
#   PD_MTP_SYNTHETIC_AL  empty (default): real MTP rejection sampling, as the
#                        gates run. A number (the points use 3.39) fixes the
#                        acceptance length; for a benchmark instantiation only.
#   PD_PREFILL_REPLICAS, PD_DECODE_REPLICAS
#                        engines per role (1 each by default); each is one
#                        tpu7x host of 4 chips serving these legs.
#   PD_PROXY             router (default) | toy: what fronts the pair.
#   MODEL_PATH           a local checkpoint to serve instead of the hub id.
#
# Host memory: the prefill leg pins 720 GB of host RAM per host for its KV
# offload tier (cpu_bytes_to_use below); the measured runs found 760 GB too
# close to host OOM. The decode leg's expert placement tables were calibrated
# on the benchmark's traffic.

PD_ROLE="${PD_ROLE:-}"
PD_DECODE_LAYOUT="${PD_DECODE_LAYOUT:-dp4tp2}"
PD_TRADEOFFS="${PD_TRADEOFFS:-on}"
PD_MTP_SYNTHETIC_AL="${PD_MTP_SYNTHETIC_AL:-}"
PD_PREFILL_REPLICAS="${PD_PREFILL_REPLICAS:-1}"
PD_DECODE_REPLICAS="${PD_DECODE_REPLICAS:-1}"
PD_PROXY="${PD_PROXY:-router}"

case "$PD_ROLE" in
    prefill|decode) ;;
    *) echo "ERROR: qwen3.5-397b-fp8-pd.sh: PD_ROLE must be prefill or decode, got '$PD_ROLE'" >&2; return 1 ;;
esac
case "$PD_DECODE_LAYOUT" in
    dp4tp2|tp8dcp4) ;;
    *) echo "ERROR: qwen3.5-397b-fp8-pd.sh: PD_DECODE_LAYOUT must be dp4tp2 or tp8dcp4, got '$PD_DECODE_LAYOUT'" >&2; return 1 ;;
esac
case "$PD_TRADEOFFS" in
    on) _tradeoff=true; _ssm_dtype=bfloat16 ;;
    off) _tradeoff=false; _ssm_dtype=auto ;;
    *) echo "ERROR: qwen3.5-397b-fp8-pd.sh: PD_TRADEOFFS must be on or off, got '$PD_TRADEOFFS'" >&2; return 1 ;;
esac
case "$PD_PROXY" in
    router|toy) ;;
    *) echo "ERROR: qwen3.5-397b-fp8-pd.sh: PD_PROXY must be router or toy, got '$PD_PROXY'" >&2; return 1 ;;
esac
if [[ -n "$PD_MTP_SYNTHETIC_AL" && ! "$PD_MTP_SYNTHETIC_AL" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: qwen3.5-397b-fp8-pd.sh: PD_MTP_SYNTHETIC_AL must be empty or a number, got '$PD_MTP_SYNTHETIC_AL'" >&2
    return 1
fi
if ! [[ "$PD_PREFILL_REPLICAS" =~ ^[1-9][0-9]*$ && "$PD_DECODE_REPLICAS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: qwen3.5-397b-fp8-pd.sh: PD_PREFILL_REPLICAS and PD_DECODE_REPLICAS must be positive integers" >&2
    return 1
fi

MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
SERVED_MODEL="Qwen3.5-397B-A17B-FP8"
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
_validation="$_repo/src/vllm_torchtpu/models/qwen3_5/validation"

# ---- What the correctness stages check ---------------------------------------
# GSM8K on its whole test split (1,319 questions), thinking on, scored by
# strict-match, as the records were measured. 100 samples cannot resolve a
# transfer fault: their stderr is about 1.6 pp, the whole split's about 0.4 pp.
# --require-baselines: a missing baseline file fails the gate rather than
# skipping it. The correctness stages add --run-lm-eval themselves.
PD_EVAL_FLOW_ARGS=(--lm-eval-limit none --require-baselines)
LM_EVAL_TASKS="gsm8k"
# As every Qwen3.5 config. With 1,319 questions the stderr term of the
# allowed drop is about 1.0 pp, so this 2 pp floor decides (check_regression.py).
EVAL_TOLERANCE="0.02"
# Thinking on, explicitly: main's evals turn it off, the records' GSM8K ran
# with it on. The stop strings leave out the task's "Question:", which a
# thinking model can write mid-reasoning; the output budget is the records'
# GSM8K runs' (16,384 tokens). Greedy.
LM_EVAL_GEN_KWARGS='{"chat_template_kwargs":{"enable_thinking":true},"until":["</s>","<|im_end|>"],"max_gen_toks":16384,"temperature":0,"top_p":1,"do_sample":false}'
# 64 requests in flight, the records' GSM8K eval concurrency; long thinking
# answers need more than lm-eval's default request timeout. One attempt per
# request: the proxy runs without retries so that a failed prefill or decode
# hop fails its request, and a client retry would hide it behind a score
# that still passes.
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=64,timeout=1800,max_retries=1"
# The kernel page both legs must run (models/qwen3_5/hooks.py), read from
# every worker's log.
PD_KERNEL_PAGE_TOKENS=128
# MTP acceptance rate (accepted / proposed draft tokens) on decode, as main's
# P/D rigs gate it (MIN_MTP_ACCEPTANCE_RATE): 0.66 is an acceptance length of
# about 3.0 at three draft tokens; the gates measured 3.25 and 3.35 (0.75 and
# 0.78).
PD_MIN_MTP_ACCEPTANCE_RATE=0.66
# The transfer happened: decode computes at most this percent of its prompt
# tokens itself (one token per request, plus nothing it fell back to).
PD_MAX_DECODE_COMPUTE_PCT=5
# run_benchmarks.sh requires it; the correctness run sends no benchmark.
RANDOM_RANGE_RATIO="0.0"
PERF_TOLERANCE="0.05"

# ---- Both legs ----------------------------------------------------------------
# Serve arguments every leg of every entry passes.
SERVE_ARGS=(
    "${MODEL_PATH:-$MODEL}"
    --served-model-name "$SERVED_MODEL" "Qwen/$SERVED_MODEL"
    --default-chat-template-kwargs '{"enable_thinking":true}'
    --tool-call-parser=qwen3_coder
    --reasoning-parser=qwen3
    --api-server-count 1
    --trust-remote-code
    --seed 42
    --max-model-len 262144
    --disable-custom-all-reduce
    --gpu-memory-utilization 0.92
    --kv-cache-dtype fp8
    --quantization fp8
    --language-model-only
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --no-disable-hybrid-kv-cache-manager
    --attention-backend CUSTOM
    --mamba-cache-mode align
    --enable-prompt-tokens-details
    --no-enable-log-requests
    --enable-prefix-caching
    # bf16 GDN SSM state: a trade-off, so the gates' off arm passes auto.
    --mamba-ssm-cache-dtype "$_ssm_dtype"
    --disable-sliding-window
    --host 0.0.0.0
    --prefill-context-parallel-size 1
    --renderer-num-workers 4
    --cpu-distributed-timeout-seconds 10800
)

# Environment every leg of every entry sets.
export MOE_FUSED_EP_KERNEL_MIN_TOKENS=1024
export ONEHOT_MOE_PERMUTE_THRESHOLD=2048
export RAGGED_GATHER_REDUCE_VERSION=v3
export TPU_ENABLE_GDN_DYNAMIC_TILING=1
export TPU_EVICT_WEIGHTS_PAGE_CACHE=1
export TPU_KV_SHM_POOL_GB=8
export TPU_MOE_SINGLE_SORT=1
export TPU_MOE_SKIP_PADDED_TOKENS=1
export TPU_P2P_WAIT_PULL_TIMEOUT=300
export TPU_RAIDEN_MAX_TRANSFER_TOKENS=262144
export TPU_RAIDEN_POOL_TAGS_PER_LAYER=1
export TPU_RAIDEN_PREFIX_AWARE_LOAD=1
export TPU_RAIDEN_QWEN35_ADMISSION=1
export TPU_RAIDEN_RESHARD_IMPL=store
export TPU_RAIDEN_TRANSFER_PARALLELISM=8
export TPU_ROPE_CACHE_TRUNCATE=1
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
export USE_MOE_SPARSE_CORE=1
export VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1
export VLLM_XLA_CHECK_RECOMPILATION=0

# Variables the measured runs' launcher set that the entries cannot carry:
# they list only this package's envs.py variables, and these are read by vLLM,
# libtpu, the Raiden library or the model loader.
# The seq-along-lane KV layout, which the 128-token kernel page both legs
# share applies to (models/qwen3_5/hooks.py).
export VLLM_KV_CACHE_LAYOUT=HND
# --max-model-len 262144 is the checkpoint's whole window.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=300
export VLLM_TARGET_DEVICE=tpu
export OMP_NUM_THREADS=8
# Host memory the runai streamer stages through (the decode leg of entry p01
# loads with it).
export RUNAI_STREAMER_MEMORY_LIMIT=5368709120
# The libtpu flags every measured leg ran, the rematerialization threshold on
# entry p01's prefill leg only.
_libtpu_args="--xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_dvfs_p_state=7 --xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=67108864 --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=8388608"

# One host of four chips per engine, whatever slice the pod's environment
# describes (as .buildkite/kubernetes/disagg_engine.sh states it).
export JAX_PLATFORMS=tpu,cpu
export PJRT_DEVICE=TPU
export TPU_ACCELERATOR_TYPE=tpu7x
export TPU_BACKEND_TYPE=jax
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_TOPOLOGY=2x2x1
export MEGASCALE_COORDINATOR_ADDRESS='' MEGASCALE_NUM_SLICES='' MEGASCALE_PORT='' MEGASCALE_SLICE_ID=''
export TPU_PROCESS_ADDRESSES='' TPU_PROCESS_PORT='' TPU_WORKER_HOSTNAMES=''
# As the records' runs: no JAX client preallocation, INFO logs (the kernel
# page and fingerprint checks read them).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VLLM_LOGGING_LEVEL=INFO
# A cold compile of both legs takes over an hour.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

# Raiden wiring. Per pod, so the entries do not record it; the measured runs'
# values. These two ports are the same on both roles; the identities and port
# bases that must differ so the roles do not collide are set per role below.
export TPU_KV_TRANSFER_PORT=9100
export TPU_SIDE_CHANNEL_PORT=9600
# What the records' runs set beside it: Raiden as the KV reshard transport
# (envs.py's default is zmq; the records leave it to the deployment) and its
# 16 GiB premapped host buffer.
export TPU_KV_RESHARD_TRANSPORT=raiden
export TPU_PREMAPPED_BUFFER_SIZE=17179869184

# ---- Per role and decode layout -----------------------------------------------
# Entry p01's legs are DP4 x TP2's, p08's TP8 + DCP4's; the prefill leg is PP8
# in both but carries a few values of its point.
# PD_DECODE_API_PORTS: the decode leg's API servers, which a proxy in front of
# the pair routes to (DP4 x TP2 runs one per DP rank, ports 7200-7203).
case "$PD_DECODE_LAYOUT" in
    dp4tp2) _raiden_dst_shards=2; _dcp_extra=''; PD_DECODE_API_PORTS="7200 7201 7202 7203" ;;
    tp8dcp4) _raiden_dst_shards=8; _dcp_extra='"destination_decode_context_parallel_size":4,'; PD_DECODE_API_PORTS="7200" ;;
esac
export TPU_RAIDEN_DST_SHARDS="$_raiden_dst_shards"

if [ "$PD_ROLE" = prefill ]; then
    PD_SERVE_PORT=8000
    export TPU_KV_TRANSFER_NAMESPACE=prefill
    export TPU_RAIDEN_ENGINE_ID=prefill-engine
    export TPU_RAIDEN_JOB_NAME=prefill
    export TPU_RAIDEN_RESHARD_PORT_BASE=27000
    export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=27100
    export TPU_MOE_SEQUENCE_PARALLEL=0
    export TPU_PP_CHUNK_SLACK=0.5
    export USE_MOE_FUSED_EP_KERNEL=0
    # 60 layers over 8 stages, the last stage lighter: the measured partition.
    export VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8,8,4
    # Only the offloading connector builds a Raiden store, which waits this
    # long for its workers (the library's default is 120 s); NUMA-local host
    # memory for that tier.
    export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=3600
    export RAIDEN_NUMA_POLICY=preferred
    export VLLM_DP_MASTER_IP=127.0.0.1
    SERVE_ARGS+=(
        --max-num-batched-tokens 8192
        --long-prefill-token-threshold 262144
        --max-num-seqs 32
        # One block holds a request's whole GDN state on this tree: 2048 SSM
        # + 256 conv tokens at TP1 with the bf16 SSM state (validation README).
        --block-size 2304
        --tensor-parallel-size 1
        --port "$PD_SERVE_PORT"
        # No --scheduler-cls: the platform installs the pipeline chunk
        # scheduler for PP > 1 only when no class is named.
        --pipeline-parallel-size 8
        --cp-kv-cache-interleave-size 1
        --data-parallel-size 1
        --data-parallel-size-local 1
        # Raiden to decode, plus a host-RAM offload tier of 720 GB.
        --kv-transfer-config "{\"kv_connector\":\"TPUMultiConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"kv_producer\",\"kv_port\":14579,\"kv_connector_extra_config\":{${_dcp_extra}\"async_complete\":true,\"async_register\":true}},{\"kv_connector\":\"TPURaidenOffloadingConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.offload.raiden_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":720000000000,\"raiden_controller_port\":27800,\"max_async_loads\":16,\"fence_after_draft\":true,\"host_tier_recency\":true}}],\"producer_divergent_hits\":true}}"
        # MTP K=1 on prefill: it fills the draft's KV for decode.
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
    )
    _sections="\"tpu_compile\":{\"drop_global_state_guard\":true,\"fast_compiled_call\":true},\"tpu_frontend\":{\"detok_prompt_tail\":true,\"tokenize_prefix_cache\":true,\"tokenize_prefix_cache_max_tokens\":64000000},\"tpu_pp_scheduler\":{\"paw_aging_tok_per_s\":3000,\"prefix_aware\":true,\"quadratic_cost\":true},\"tpu_worker\":{\"numa_bind\":true}"
    case "$PD_DECODE_LAYOUT" in
        dp4tp2)
            export LIBTPU_INIT_ARGS="$_libtpu_args --xla_tpu_rematerialization_min_size_in_bytes=1099511627776"
            SERVE_ARGS+=(
                --load-format safetensors --safetensors-load-strategy prefetch
                --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[8,16,32,64,128,256,512,1024,1536,1664,2048,2176,3840,4352,6016,6528,8192],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
            )
            _sections="$_sections,\"tpu_runner\":{\"stage_timing\":false}"
            ;;
        tp8dcp4)
            export LIBTPU_INIT_ARGS="$_libtpu_args"
            SERVE_ARGS+=(
                --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[2176,4352,6528,8192],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
            )
            _sections="$_sections,\"tpu_scheduler\":{\"promote_warmest\":false}"
            ;;
    esac
else
    PD_SERVE_PORT=7200
    export TPU_KV_TRANSFER_NAMESPACE=decode
    export TPU_RAIDEN_ENGINE_ID=decode-engine
    export TPU_RAIDEN_JOB_NAME=decode
    export TPU_RAIDEN_RESHARD_PORT_BASE=28000
    export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=28100
    export MOE_FUSED_EP_KERNEL_IMPL=v3
    export TPU_MOE_SEQUENCE_PARALLEL=1
    export TPU_SAMPLER_TOPK_CANDIDATES=40
    export USE_MOE_FUSED_EP_KERNEL=1
    export LIBTPU_INIT_ARGS="$_libtpu_args"
    if [ -n "$PD_MTP_SYNTHETIC_AL" ]; then
        _spec="{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${PD_MTP_SYNTHETIC_AL}}"
    else
        _spec='{"method":"mtp","num_speculative_tokens":3}'
    fi
    SERVE_ARGS+=(
        --enable-expert-parallel
        --enable-ep-weight-filter
        --max-num-batched-tokens 2048
        --async-scheduling
        --scheduler-cls vllm_torchtpu.core.prefix_aware_scheduler.PrefixAwareScheduler
        --port "$PD_SERVE_PORT"
        --kv-transfer-config "{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"kv_consumer\",\"kv_port\":14579,\"kv_connector_extra_config\":{${_dcp_extra}\"late_poll\":true}}"
        --speculative-config "$_spec"
    )
    _sections="\"tpu_collectives\":{\"small_all_reduce\":true},\"tpu_compile\":{\"drop_global_state_guard\":true,\"fast_compiled_call\":true},\"tpu_frontend\":{\"tokenize_prefix_cache\":true},\"tpu_worker\":{\"numa_bind\":true}"
    case "$PD_DECODE_LAYOUT" in
        dp4tp2)
            export TPU_EXPERT_PLACEMENT="$_validation/expert_placement_dp8_ep8.json"
            export TPU_TOKEN_BUCKET_EXTRA=8,48,2048
            SERVE_ARGS+=(
                --load-format runai_streamer
                --max-num-seqs 28
                --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[8,16,24,32,48,64,72,80,96,112,128,256,512,2048],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
                # 1024 SSM + 256 conv tokens at TP2 with the bf16 SSM state.
                --block-size 1280
                # The measured KV budget at this layout.
                --kv-cache-memory-bytes 39500000000
                --tensor-parallel-size 2
                --cp-kv-cache-interleave-size 1
                --data-parallel-size 4
                --data-parallel-size-local 4
                --data-parallel-multi-port-external-lb
            )
            _sections="$_sections,\"tpu_multiport_dp\":{\"enabled\":true},\"tpu_runner\":{\"stage_timing\":false},\"tpu_scheduler\":{\"kv_prefetch\":true}"
            ;;
        tp8dcp4)
            export TPU_EXPERT_PLACEMENT="$_validation/expert_placement_tp8_ep8_agentx_corr_c32.json"
            export TPU_TOKEN_BUCKET_EXTRA=4,8,48,2048
            # Read by torch_tpu, so not in envs.py; the measured TP8 + DCP4
            # decode legs set it off.
            export TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS=0
            SERVE_ARGS+=(
                --max-num-seqs 8
                --compilation-config '{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4,8,16,24,32,48,64,96,128,160,192,256,320,384,448,512,2048],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
                # The record's block; the TP8 decode legs need 320.
                --block-size 640
                --tensor-parallel-size 8
                --decode-context-parallel-size 4
                --cp-kv-cache-interleave-size 128
                --data-parallel-size 1
                --data-parallel-size-local 1
            )
            _sections="$_sections,\"tpu_scheduler\":{\"decode_inflight_throttle\":true,\"kv_prefetch\":true,\"promote_warmest\":false}"
            ;;
    esac
fi
# The Qwen3.5 trade-off settings, on both legs as the gates' arms merge them.
SERVE_ARGS+=(
    --additional-config "{\"qwen3_5\":{\"exp2_softmax\":${_tradeoff},\"native_fp8_attention\":${_tradeoff},\"quantize_target_lm_head\":${_tradeoff}},${_sections}}"
)
unset _tradeoff _ssm_dtype _repo _validation _raiden_dst_shards _dcp_extra _libtpu_args _spec _sections
