// dsv4-agentx-attn-intra-chip-tp: DeepSeek-V4 Pro on two tpu7x 2x2x2 hosts (8 chips,
// 16 TensorCores; TP=1, DP=16, EP=16), AgentX trace replay, with an A/B switch for
// attention-weight intra-chip TP. Derived from infx-deepseek-v4-agentx as run in
// j-bc99ea54 (fused-moe-v2 W4A8 + PR #1557, BLOCK_SIZE=1024 + PR #1518, TpuDpScheduler,
// prefix-aware DP router, Raiden host-DRAM KV offload).
//
// ATTN_INTRA_CHIP_TP=true halves wq_b/wo_a/wo_b per TensorCore across each chip's two
// cores (KV not replicated), growing the KV pool from 2.41M to 3.53M tokens per rank.
//
// Base vs split, c128 (total tok/s per chip: 9,675 base j-34a962a7, 13,889 split
// j-09ee49c7):
//   cdk job create dsv4-agentx-attn-intra-chip-tp $ARGS ATTN_INTRA_CHIP_TP=false
//   cdk job create dsv4-agentx-attn-intra-chip-tp $ARGS ATTN_INTRA_CHIP_TP=true
// with
//   ARGS="MAX_NUM_BATCHED_TOKENS=512 DYNAMIC_BT_ENABLED=true DYNAMIC_BT_LOW_CONC_MAX=8
//     DYNAMIC_BT_LOW_TOKENS=512 DYNAMIC_BT_HIGH_TOKENS=128 BLOCK_SIZE=1024
//     GPU_MEMORY_UTILIZATION=0.96 DP_SCHED_ENABLED=true PREFILL_SCHEDULE_INTERVAL=1
//     DURATION=3600 CLIENT_CONCURRENCIES=128"
// (add ENABLE_XPROF=true PROFILE_DURATION_SECONDS=2 PROFILE_RANKS=0,...,15 for traces).

local agentx = import 'cdk.agentx';
local cdk = import 'cdk.common';

local jobName = cdk.var('JOB_NAME', 'infx-deepseek-v4-agentx');
local serverImage = cdk.var('IMAGE', 'us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-dev:nightly-20261007');
local clientImage = cdk.var('BENCH_IMAGE', 'us-central1-docker.pkg.dev/cloud-tpu-inference-test/inferencex/inferencex-agentx:latest');

local modelPath = cdk.var('MODEL_PATH', 'deepseek-ai/DeepSeek-V4-Pro-0813');
local servedModelName = cdk.var('SERVED_MODEL_NAME', 'deepseek-ai/DeepSeek-V4-Pro-0813');
local gcsCkptDir = cdk.var('GCS_CKPT_DIR', 'gs://cloud-devkit/ckpt/hf/deepseek-ai/DeepSeek-V4-Pro-0813');
local gcsModelSrc = cdk.var('GCS_MODEL_SRC', gcsCkptDir);
local gcsDatasetCache = cdk.var('GCS_DATASET_CACHE', 'gs://cloud-devkit/datasets/hf_cache/semianalysisai/cc-traces-weka-062126');
local sharedCachePvc = cdk.var('SHARED_CACHE_PVC', '');
local hfStorageSize = cdk.var('HF_STORAGE_SIZE', '1500Gi');

// --- Cluster shape -----------------------------------------------------------
local rayReplicas = cdk.numVar('REPLICAS', cdk.numVar('RAY_NODES', 2));
local chipsPerHost = cdk.numVar('CHIPS_PER_HOST', 4);
local accelerator = cdk.var('TPU_ACCELERATOR', cdk.var('TPU_ACCELERATOR_TYPE', 'tpu7x'));
local topology = cdk.var('TPU_TOPOLOGY', '2x2x2');
local nodePool = cdk.var('NODE_POOL', '');
local tpuChips = cdk.numVar('TPU_CHIPS', 8);

// The head blocks until `ray status` reports the whole cluster.
local rayExpectedTpu = cdk.var('EXPECTED_TPU', cdk.var('RAY_EXPECTED_TPU', '16.0 TPU'));

// --- Serving & KV offloading (vllm-torchtpu PR #787 + PR #1518 + PR #1557) ---
local port = cdk.numVar('PORT', 47903);
local dataParallelSize = cdk.numVar('DATA_PARALLEL_SIZE', 16);
local tensorParallelSize = cdk.numVar('TENSOR_PARALLEL_SIZE', 1);
local kvOffload = cdk.boolVar('KV_OFFLOAD', true) || cdk.boolVar('ENABLE_KV_OFFLOAD', true) || cdk.var('OFFLOAD_MODE', 'on') == 'on';
// On TP=1, DP=16, each DP engine has world_size=1 and 8 DP workers/host, so
// KV_OFFLOADING_SIZE=12079595520 (11.25 GiB/rank = 180 GiB cluster-wide) avoids host RAM OOMKilled.
local kvOffloadingSize = cdk.var('KV_OFFLOADING_SIZE', if dataParallelSize > 1 then '12079595520' else '480');
local offloadControllerPort = cdk.numVar('OFFLOAD_CONTROLLER_PORT', 27801);
local globalRegistryPort = cdk.numVar('GLOBAL_REGISTRY_PORT', 27800);
local raidenJobName = cdk.var('RAIDEN_JOB_NAME', 'dsv4-offload-probe');
local enablePrefixRouting = cdk.boolVar('ENABLE_PREFIX_ROUTING', true);

local maxModelLen = cdk.numVar('MAX_MODEL_LEN', 1048576);
local maxNumSeqs = cdk.numVar('MAX_NUM_SEQS', 64);
// Best-performing configuration (j-d5dcc6ae-a442-463f-a8c0 & j-cb710aaa-eed8-4dca-bea1):
// TP=1, DP=16, EP=16, Dynamic BT (512 for C<=8, 128 for C>=16), BLOCK_SIZE=1024, GPU_MEMORY_UTILIZATION=0.96,
// DP_SCHED_ENABLED=true, PR #1518, PR #1557, PR #1728, PR #1729 -> 2.61M HBM KV tokens/rank (41.73M total).
local maxNumBatchedTokens = cdk.numVar('MAX_NUM_BATCHED_TOKENS', 512);
local dynamicBtEnabled = cdk.boolVar('DYNAMIC_BT_ENABLED', true);
local dynamicBtLowConcMax = cdk.numVar('DYNAMIC_BT_LOW_CONC_MAX', 8);
local dynamicBtLowTokens = cdk.numVar('DYNAMIC_BT_LOW_TOKENS', 512);
local dynamicBtHighTokens = cdk.numVar('DYNAMIC_BT_HIGH_TOKENS', 128);
local blockSize = cdk.numVar('BLOCK_SIZE', 1024);
local quantization = cdk.var('QUANTIZATION', 'deepseek_v4_fp8');
local kvCacheDtype = cdk.var('KV_CACHE_DTYPE', 'fp8_e4m3');
local gpuMemoryUtilization = cdk.var('GPU_MEMORY_UTILIZATION', '0.96');
local enablePrefixCaching = cdk.boolVar('ENABLE_PREFIX_CACHING', true);
local dpSchedEnabled = cdk.boolVar('DP_SCHED_ENABLED', true);
local prefillScheduleInterval = cdk.numVar('PREFILL_SCHEDULE_INTERVAL', 1);
local schedulerCls = cdk.var('SCHEDULER_CLS', '');
local loadFormat = cdk.var('LOAD_FORMAT', '');
local toolCallParser = cdk.var('TOOL_CALL_PARSER', '');
local reasoningParser = cdk.var('REASONING_PARSER', 'deepseek_r1');
local chatTemplate = cdk.var('CHAT_TEMPLATE', '/tmp/chat_template.jinja');
local moeRequantizeWeightDtype = cdk.var('MOE_REQUANTIZE_WEIGHT_DTYPE', 'fp4');
local moeRequantizeBlockSize = cdk.numVar('MOE_REQUANTIZE_BLOCK_SIZE', 512);
local extraVllmArgs = cdk.var('EXTRA_VLLM_ARGS', '');
local mostModelLen = cdk.numVar('VLLM_TPU_MOST_MODEL_LEN', 0);

// --- Profiling ----------------------------------------------------------------
local enableXprof = cdk.boolVar('ENABLE_XPROF', false);
local enablePerfetto = cdk.boolVar('ENABLE_PERFETTO', false);
local enableProfiling = enableXprof || enablePerfetto;
local xprofDir = if dataParallelSize > 1 then '/dev/shm/xprof' else '${CDK_OUTPUT_DIR:-/tmp/cdk_output}/xprof';

// --- Attention-weight intra-chip TP -----------------------------------------------
// ATTN_INTRA_CHIP_TP=true shards DSv4 attention wq_b/wo_a/wo_b across each chip's two
// TensorCores while KV and attention stay DP=16 (vllm-torchtpu commit 1fec0d1fc,
// branch experimental/dsv4-attn-intra-chip-tp, flag TPU_DSV4_ATTN_INTRA_CHIP_TP). The
// image predates that commit, so its code is applied at container start: write
// deepseek_v4/attn_intra_chip.py and add the flag and class choice to envs.py and model.py.
local attnIntraChipTp = cdk.boolVar('ATTN_INTRA_CHIP_TP', false);
local attnIntraChipPatch = if !attnIntraChipTp then 'echo "ATTN_INTRA_CHIP_TP off"' else |||
  echo "aW1wb3J0IGJhc2U2NCwgcGF0aGxpYgpyb290ID0gcGF0aGxpYi5QYXRoKCIvd29ya3NwYWNlL3ZsbG0tdG9yY2h0cHUvc3JjL3ZsbG1fdG9yY2h0cHUiKQoocm9vdCAvICJtb2RlbHMvdmxsbS9kZWVwc2Vla192NC9hdHRuX2ludHJhX2NoaXAucHkiKS53cml0ZV9ieXRlcyhiYXNlNjQuYjY0ZGVjb2RlKCJJeUJEYjNCNWNtbG5hSFFnTWpBeU5pQkhiMjluYkdVZ1RFeERDaU1LSXlCTWFXTmxibk5sWkNCMWJtUmxjaUIwYUdVZ1FYQmhZMmhsSUV4cFkyVnVjMlVzSUZabGNuTnBiMjRnTWk0d0lDaDBhR1VnSWt4cFkyVnVjMlVpS1RzS0l5QjViM1VnYldGNUlHNXZkQ0IxYzJVZ2RHaHBjeUJtYVd4bElHVjRZMlZ3ZENCcGJpQmpiMjF3YkdsaGJtTmxJSGRwZEdnZ2RHaGxJRXhwWTJWdWMyVXVDaU1nV1c5MUlHMWhlU0J2WW5SaGFXNGdZU0JqYjNCNUlHOW1JSFJvWlNCTWFXTmxibk5sSUdGMENpTUtJeUFnSUNBZ2FIUjBjRG92TDNkM2R5NWhjR0ZqYUdVdWIzSm5MMnhwWTJWdWMyVnpMMHhKUTBWT1UwVXRNaTR3Q2lNS0l5QlZibXhsYzNNZ2NtVnhkV2x5WldRZ1lua2dZWEJ3YkdsallXSnNaU0JzWVhjZ2IzSWdZV2R5WldWa0lIUnZJR2x1SUhkeWFYUnBibWNzSUhOdlpuUjNZWEpsQ2lNZ1pHbHpkSEpwWW5WMFpXUWdkVzVrWlhJZ2RHaGxJRXhwWTJWdWMyVWdhWE1nWkdsemRISnBZblYwWldRZ2IyNGdZVzRnSWtGVElFbFRJaUJDUVZOSlV5d0tJeUJYU1ZSSVQxVlVJRmRCVWxKQlRsUkpSVk1nVDFJZ1EwOU9SRWxVU1U5T1V5QlBSaUJCVGxrZ1MwbE9SQ3dnWldsMGFHVnlJR1Y0Y0hKbGMzTWdiM0lnYVcxd2JHbGxaQzRLSXlCVFpXVWdkR2hsSUV4cFkyVnVjMlVnWm05eUlIUm9aU0J6Y0dWamFXWnBZeUJzWVc1bmRXRm5aU0JuYjNabGNtNXBibWNnY0dWeWJXbHpjMmx2Ym5NZ1lXNWtDaU1nYkdsdGFYUmhkR2x2Ym5NZ2RXNWtaWElnZEdobElFeHBZMlZ1YzJVdUNpSWlJa2x1ZEhKaExXTm9hWEFnVkZBZ1ptOXlJRVJUZGpRZ1lYUjBaVzUwYVc5dUlIZGxhV2RvZEhNc0lIZHBkR2dnUzFZZ1lXNWtJR0YwZEdWdWRHbHZiaUJyWlhCMElFUlFMZ29LVlc1a1pYSWdSRkE5TVRZdlZGQTlNU0JsZG1WeWVTQmpiM0psSUdodmJHUnpJR0ZzYkNCdlppQmdkM0ZmWW1Bc0lHQjNiMTloWUNCaGJtUWdZSGR2WDJKZ0xDQnliM1ZuYUd4NUNqSTROU0JOUWlCd1pYSWdiR0Y1WlhJdUlHQldiR3h0UkdWbGNITmxaV3RXTkUxTVFVRjBkR1Z1ZEdsdmJrbHVkSEpoUTJocGNHQWdjM0JzYVhSeklIUm9iM05sSUhSb2NtVmxDbUZqY205emN5QjBhR1VnZEhkdklHTnZjbVZ6SUc5bUlHRWdZMmhwY0RvZ1pXRmphQ0JqYjNKbElHdGxaWEJ6SUdoaGJHWWdkR2hsSUdobFlXUnpJRzltSUdCM2NWOWlZQ0JoYm1RS2FHRnNaaUIwYUdVZ2J5MW5jbTkxY0hNZ2IyWWdZSGR2WDJGZ0wyQjNiMTlpWUM0Z2RreE1UU0J6ZEdsc2JDQnpaV1Z6SUVSUVBURTJMMVJRUFRFc0lITnZJSFJvWlFwelkyaGxaSFZzWlhJc0lISnZkWFJsY2l3Z1MxWWdjRzl2YkNCaGJtUWdkR2hsSUhOd1lYSnpaUzFoZEhSbGJuUnBiMjRnYTJWeWJtVnNjeUJoY21VZ2RXNWphR0Z1WjJWa0lHRnVaQXB5ZFc0Z2IzWmxjaUJoYkd3Z01USTRJR2hsWVdSeklHWnZjaUIwYUdseklISmhibXNuY3lCdmQyNGdjMlZ4ZFdWdVkyVnpMZ29LVUdWeUlHeGhlV1Z5TENCcGJuTnBaR1VnZEdobElHTm9hWEFnY0dGcGNpQW9kRzlyWlc1eklHRnlaU0J3WVdSa1pXUWdkRzhnZEdobElITmhiV1VnWW5WamEyVjBJRzl1SUdWMlpYSjVDa1JRSUhKaGJtc3NJSE52SUhOb1lYQmxjeUJ0WVhSamFDazZDaUFnTVM0Z1lXeHNMV2RoZEdobGNpQmdjWEpnSUMwK0lIRWdabTl5SUcxNUlEWTBJR2hsWVdSeklHOTJaWElnWW05MGFDQmpiM0psY3ljZ2RHOXJaVzV6Q2lBZ01pNGdjM2RoY0NCb1lXeDJaWE1nTFQ0Z2NTQm1iM0lnWVd4c0lERXlPQ0JvWldGa2N5QnZkbVZ5SUcxNUlIUnZhMlZ1Y3pzZ1lYUjBaVzUwYVc5dUlISjFibk1nYkc5allXeHNlUW9nSURNdUlITjNZWEFnYUdGc2RtVnpJR0poWTJzZ0xUNGdieUJtYjNJZ2JYa2dOalFnYUdWaFpITWdiM1psY2lCaWIzUm9JR052Y21Wekp5QjBiMnRsYm5NS0lDQTBMaUJtZFhObFpDQnBiblpsY25ObExWSnZVRVVnS3lCZ2QyOWZZV0FnYjI0Z2JYa2dPQ0JuY205MWNITXNJR0IzYjE5aVlDQnZiaUJ0ZVNBNE1Ua3lJSEp2ZDNNS0lDQTFMaUJ5WldSMVkyVXRjMk5oZEhSbGNpQXRQaUJ0ZVNCMGIydGxibk1uSUdaMWJHd2diM1YwY0hWMENncEZkbVZ5ZVNCamIyeHNaV04wYVhabElHbHpJR2x6YzNWbFpDQm9aWEpsTENCdmRtVnlJSFJvWlNCamFHbHdMWEJoYVhJZ1ozSnZkWEE3SUc1dmJtVWdZWEpsSUd4bFpuUWdkRzhLZEdobElHTnZiWEJwYkdWeUxpQkZlSEJsY21sdFpXNTBZV3c2SUdCRVpXVndjMlZsYTFZMFJHVmpiMlJsY2t4aGVXVnlZQ0IxYzJWeklIUm9hWE1nWTJ4aGMzTWdkMmhsYmdwVVVGVmZSRk5XTkY5QlZGUk9YMGxPVkZKQlgwTklTVkJmVkZBOU1TNEtJaUlpQ2dwbWNtOXRJRjlmWm5WMGRYSmxYMThnYVcxd2IzSjBJR0Z1Ym05MFlYUnBiMjV6Q2dwcGJYQnZjblFnZEc5eVkyZ0tabkp2YlNCMmJHeHRMbVJwYzNSeWFXSjFkR1ZrSUdsdGNHOXlkQ0J3WVhKaGJHeGxiRjl6ZEdGMFpRcG1jbTl0SUhac2JHMHViVzlrWld4ZlpYaGxZM1YwYjNJdWJHRjVaWEp6TG14cGJtVmhjaUJwYlhCdmNuUWdRMjlzZFcxdVVHRnlZV3hzWld4TWFXNWxZWElzSUZKdmQxQmhjbUZzYkdWc1RHbHVaV0Z5Q2dwbWNtOXRJSFpzYkcxZmRHOXlZMmgwY0hVdWJHOW5aMlZ5SUdsdGNHOXlkQ0JwYm1sMFgyeHZaMmRsY2dwbWNtOXRJSFpzYkcxZmRHOXlZMmgwY0hVdWJXOWtaV3h6TG5ac2JHMHVaR1ZsY0hObFpXdGZkalF1WVhSMFpXNTBhVzl1SUdsdGNHOXlkQ0JXYkd4dFJHVmxjSE5sWld0V05FMU1RVUYwZEdWdWRHbHZiZ29LYkc5bloyVnlJRDBnYVc1cGRGOXNiMmRuWlhJb1gxOXVZVzFsWDE4cENncGZaM0p2ZFhBNklIQmhjbUZzYkdWc1gzTjBZWFJsTGtkeWIzVndRMjl2Y21ScGJtRjBiM0lnZkNCT2IyNWxJRDBnVG05dVpRb0tDbVJsWmlCZmNHRnBjbDluY205MWNITW9LU0F0UGlCc2FYTjBXMnhwYzNSYmFXNTBYVjA2Q2lBZ0lDQWlJaUpIYkc5aVlXd2djbUZ1YTNNZ1ozSnZkWEJsWkNCaWVTQndhSGx6YVdOaGJDQmphR2x3TENCbWNtOXRJR1ZoWTJnZ2NtRnVheWR6SUdSbGRtbGpaU0JwWkM0aUlpSUtJQ0FnSUdaeWIyMGdkbXhzYlY5MGIzSmphSFJ3ZFM1a2FYTjBjbWxpZFhSbFpDNWphR2x3WDNSdmNHOXNiMmQ1SUdsdGNHOXlkQ0JuWlhSZlkyaHBjRjkwYjNCdmJHOW5lUW9nSUNBZ1puSnZiU0IyYkd4dFgzUnZjbU5vZEhCMUxtUnBjM1J5YVdKMWRHVmtMbTFsYzJoZmRYUnBiSE1nYVcxd2IzSjBJQ2dLSUNBZ0lDQWdJQ0JmWTI5c2JHVmpkRjl5WVc1clgzUnZYMlJsZG1salpWOXBaQ3dLSUNBZ0lDQWdJQ0JmWjJWMFgyTjFjbkpsYm5SZloyeHZZbUZzWDNKaGJtc3NDaUFnSUNBZ0lDQWdYMmRsZEY5MGNIVmZaMnh2WW1Gc1gyUmxkbWxqWlY5cFpDd0tJQ0FnSUNrS0NpQWdJQ0IzYjNKc1pDQTlJSEJoY21Gc2JHVnNYM04wWVhSbExtZGxkRjkzYjNKc1pGOW5jbTkxY0NncENpQWdJQ0J5WVc1clgzUnZYMlJsZG1salpTQTlJRjlqYjJ4c1pXTjBYM0poYm10ZmRHOWZaR1YyYVdObFgybGtLQW9nSUNBZ0lDQWdJSGR2Y214a0xDQmZaMlYwWDJOMWNuSmxiblJmWjJ4dlltRnNYM0poYm1zb0tTd2dYMmRsZEY5MGNIVmZaMnh2WW1Gc1gyUmxkbWxqWlY5cFpDZ3BDaUFnSUNBcENpQWdJQ0IwYjNCdmJHOW5lU0E5SUU1dmJtVUtJQ0FnSUhSeWVUb0tJQ0FnSUNBZ0lDQjBiM0J2Ykc5bmVTQTlJR2RsZEY5amFHbHdYM1J2Y0c5c2IyZDVLQ2tLSUNBZ0lHVjRZMlZ3ZENCRmVHTmxjSFJwYjI0Z1lYTWdaWEp5YjNJNklDQWpJRzV2Y1dFNklFSk1SVEF3TVNBdElHOXdkR2x2Ym1Gc0lHTmhjR0ZpYVd4cGRIa2djSEp2WW1VS0lDQWdJQ0FnSUNCc2IyZG5aWEl1ZDJGeWJtbHVaeWdpUVZSVVRsOUpUbFJTUVY5RFNFbFFYMVJRT2lCamFHbHdJSFJ2Y0c5c2IyZDVJSFZ1WVhaaGFXeGhZbXhsT2lBbGN5SXNJR1Z5Y205eUtRb2dJQ0FnYVdZZ2RHOXdiMnh2WjNrZ2FYTWdibTkwSUU1dmJtVWdZVzVrSUdGc2JDZ0tJQ0FnSUNBZ0lDQmhibmtvWkNCcGJpQmphR2x3SUdadmNpQmphR2x3SUdsdUlIUnZjRzlzYjJkNUxtTm9hWEJ6S1NCbWIzSWdaQ0JwYmlCeVlXNXJYM1J2WDJSbGRtbGpaUzUyWVd4MVpYTW9LUW9nSUNBZ0tUb0tJQ0FnSUNBZ0lDQmphR2x3WDI5bVgyUmxkbWxqWlNBOUlIdGtPaUJwSUdadmNpQnBMQ0JqYUdsd0lHbHVJR1Z1ZFcxbGNtRjBaU2gwYjNCdmJHOW5lUzVqYUdsd2N5a2dabTl5SUdRZ2FXNGdZMmhwY0gwS0lDQWdJQ0FnSUNCemIzVnlZMlVnUFNBaVkyOXZjbVJ6SWdvZ0lDQWdaV3h6WlRvS0lDQWdJQ0FnSUNBaklIUndkVGQ0SUc1MWJXSmxjbk1nWVNCamFHbHdKM01nZEhkdklGUmxibk52Y2tOdmNtVnpJREpySUdGdVpDQXlheXN4TGdvZ0lDQWdJQ0FnSUdOb2FYQmZiMlpmWkdWMmFXTmxJRDBnZTJRNklHUWdMeThnTWlCbWIzSWdaQ0JwYmlCeVlXNXJYM1J2WDJSbGRtbGpaUzUyWVd4MVpYTW9LWDBLSUNBZ0lDQWdJQ0J6YjNWeVkyVWdQU0FpWkdWMmFXTmxYMmxrSUM4dklESWlDaUFnSUNCaWVWOWphR2x3T2lCa2FXTjBXMmx1ZEN3Z2JHbHpkRnRwYm5SZFhTQTlJSHQ5Q2lBZ0lDQm1iM0lnY21GdWF5d2daR1YyYVdObElHbHVJSEpoYm10ZmRHOWZaR1YyYVdObExtbDBaVzF6S0NrNkNpQWdJQ0FnSUNBZ1lubGZZMmhwY0M1elpYUmtaV1poZFd4MEtHTm9hWEJmYjJaZlpHVjJhV05sVzJSbGRtbGpaVjBzSUZ0ZEtTNWhjSEJsYm1Rb2NtRnVheWtLSUNBZ0lHZHliM1Z3Y3lBOUlGdHpiM0owWldRb1lubGZZMmhwY0Z0alhTa2dabTl5SUdNZ2FXNGdjMjl5ZEdWa0tHSjVYMk5vYVhBcFhRb2dJQ0FnYkc5bloyVnlMbWx1Wm04b0NpQWdJQ0FnSUNBZ0lrRlVWRTVmU1U1VVVrRmZRMGhKVUY5VVVEb2dZMmhwY0NCd1lXbHljeUJtY205dElDVnpPaUFsY3lBb2NtRnVheTArWkdWMmFXTmxJQ1Z6S1NJc0NpQWdJQ0FnSUNBZ2MyOTFjbU5sTEFvZ0lDQWdJQ0FnSUdkeWIzVndjeXdLSUNBZ0lDQWdJQ0JrYVdOMEtITnZjblJsWkNoeVlXNXJYM1J2WDJSbGRtbGpaUzVwZEdWdGN5Z3BLU2tzQ2lBZ0lDQXBDaUFnSUNCcFppQmhibmtvYkdWdUtHY3BJQ0U5SURJZ1ptOXlJR2NnYVc0Z1ozSnZkWEJ6S1RvS0lDQWdJQ0FnSUNCeVlXbHpaU0JTZFc1MGFXMWxSWEp5YjNJb1ppSkJWRlJPWDBsT1ZGSkJYME5JU1ZCZlZGQWdibVZsWkhNZ01pQmpiM0psY3lCd1pYSWdZMmhwY0N3Z1oyOTBJSHRuY205MWNITjlJaWtLSUNBZ0lISmxkSFZ5YmlCbmNtOTFjSE1LQ2dwa1pXWWdaMlYwWDJOb2FYQmZjR0ZwY2w5bmNtOTFjQ2dwSUMwK0lIQmhjbUZzYkdWc1gzTjBZWFJsTGtkeWIzVndRMjl2Y21ScGJtRjBiM0k2Q2lBZ0lDQWlJaUpVYUdVZ1IzSnZkWEJEYjI5eVpHbHVZWFJ2Y2lCdmRtVnlJSFJvYVhNZ2NtRnVheWR6SUdOb2FYQWdjR0ZwY2k0Z1EyOXNiR1ZqZEdsMlpTQnZiaUJtYVhKemRDQmpZV3hzTGlJaUlnb2dJQ0FnWjJ4dlltRnNJRjluY205MWNBb2dJQ0FnYVdZZ1gyZHliM1Z3SUdseklFNXZibVU2Q2lBZ0lDQWdJQ0FnYVcxd2IzSjBJSFJ2Y21Ob0xtUnBjM1J5YVdKMWRHVmtJR0Z6SUdScGMzUUtDaUFnSUNBZ0lDQWdkMjl5YkdRZ1BTQndZWEpoYkd4bGJGOXpkR0YwWlM1blpYUmZkMjl5YkdSZlozSnZkWEFvS1FvZ0lDQWdJQ0FnSUY5bmNtOTFjQ0E5SUhCaGNtRnNiR1ZzWDNOMFlYUmxMbWx1YVhSZmJXOWtaV3hmY0dGeVlXeHNaV3hmWjNKdmRYQW9DaUFnSUNBZ0lDQWdJQ0FnSUY5d1lXbHlYMmR5YjNWd2N5Z3BMQW9nSUNBZ0lDQWdJQ0FnSUNCM2IzSnNaQzVzYjJOaGJGOXlZVzVyTEFvZ0lDQWdJQ0FnSUNBZ0lDQmthWE4wTG1kbGRGOWlZV05yWlc1a0tIZHZjbXhrTG1SbGRtbGpaVjluY205MWNDa3NDaUFnSUNBZ0lDQWdJQ0FnSUdkeWIzVndYMjVoYldVOUltRjBkRzVmYVc1MGNtRmZZMmhwY0NJc0NpQWdJQ0FnSUNBZ0tRb2dJQ0FnY21WMGRYSnVJRjluY205MWNBb0tDbVJsWmlCZmRYTmxYMjE1WDNOc2FXTmxLR3hoZVdWeU9pQjBiM0pqYUM1dWJpNU5iMlIxYkdVc0lISmhibXM2SUdsdWRDd2djMmw2WlRvZ2FXNTBLU0F0UGlCT2IyNWxPZ29nSUNBZ0lpSWlUV0ZyWlNCZ2JHRjVaWEpnSUd4dllXUWdjMnhwWTJVZ1lISmhibXRnSUc5bUlHQnphWHBsWUNCbWNtOXRJSFJvWlNCbWRXeHNJR05vWldOcmNHOXBiblFnZEdWdWMyOXljeTRLQ2lBZ0lDQjJURXhOSjNNZ2JHOWhaR1Z5Y3lCemJHbGpaU0JpZVNCZ2RIQmZjbUZ1YTJBdllIUndYM05wZW1WZ0xDQnJaWEIwSUc5dUlIUm9aU0JzWVhsbGNpQW9kVzV4ZFdGdWRHbDZaV1FLSUNBZ0lIZGxhV2RvZEhNcElHRnVaQ0J2YmlCbFlXTm9JSEYxWVc1MGFYcGxaQ0J3WVhKaGJXVjBaWElnS0dKc2IyTnJMV1p3T0NCM1pXbG5hSFJ6SUdGdVpDQnpZMkZzWlhNcExnb2dJQ0FnUW05MGFDQmhjbVVnY21WaFpDQm1jbTl0SUhSb1pTQm5iRzlpWVd3Z1ZGQWdaM0p2ZFhBZ1lYUWdZMjl1YzNSeWRXTjBhVzl1TENCM2FHbGphQ0JwY3lCemFYcGxJREVLSUNBZ0lHaGxjbVVzSUhOdklIUm9aWGtnWVhKbElITmxkQ0IwYnlCMGFHbHpJR052Y21VbmN5QndiR0ZqWlNCcGJpQjBhR1VnWTJocGNDQndZV2x5SUdsdWMzUmxZV1F1Q2lBZ0lDQWlJaUlLSUNBZ0lHeGhlV1Z5TG5Sd1gzSmhibXNzSUd4aGVXVnlMblJ3WDNOcGVtVWdQU0J5WVc1ckxDQnphWHBsQ2lBZ0lDQm1iM0lnY0dGeVlXMGdhVzRnYkdGNVpYSXVjR0Z5WVcxbGRHVnljeWdwT2dvZ0lDQWdJQ0FnSUdsbUlHaGhjMkYwZEhJb2NHRnlZVzBzSUNKMGNGOXlZVzVySWlrNkNpQWdJQ0FnSUNBZ0lDQWdJSEJoY21GdExuUndYM0poYm1zc0lIQmhjbUZ0TG5Sd1gzTnBlbVVnUFNCeVlXNXJMQ0J6YVhwbENnb0taR1ZtSUY5emQyRndYM2RwZEdoZmNHVmxjaWhuY205MWNDd2diV2x1WlRvZ2RHOXlZMmd1VkdWdWMyOXlMQ0IwYUdWcGNuTTZJSFJ2Y21Ob0xsUmxibk52Y2lrNkNpQWdJQ0FpSWlKVFpXNWtJR0IwYUdWcGNuTmdJSFJ2SUhSb1pTQndaV1Z5SUdOdmNtVTdJSEpsZEhWeWJpQjNhR0YwSUhSb1pTQndaV1Z5SUhObGJuUWdkWE11Q2dvZ0lDQWdRU0F5TFhKaGJtc2dZV3hzTFdkaGRHaGxjaUIwYUdGMElHdGxaWEJ6SUhSb1pTQndaV1Z5SjNNZ2FHRnNaaTRLSUNBZ0lDSWlJZ29nSUNBZ2NtRnVheUE5SUdkeWIzVndMbkpoYm10ZmFXNWZaM0p2ZFhBS0lDQWdJR2RoZEdobGNtVmtJRDBnWjNKdmRYQXVZV3hzWDJkaGRHaGxjaWgwYUdWcGNuTXVZMjl1ZEdsbmRXOTFjeWdwTENCa2FXMDlNQ2tLSUNBZ0lISmxkSFZ5YmlCbllYUm9aWEpsWkM1MmFXVjNLRElzSUNwMGFHVnBjbk11YzJoaGNHVXBXekVnTFNCeVlXNXJYUW9LQ21Oc1lYTnpJRlpzYkcxRVpXVndjMlZsYTFZMFRVeEJRWFIwWlc1MGFXOXVTVzUwY21GRGFHbHdLRlpzYkcxRVpXVndjMlZsYTFZMFRVeEJRWFIwWlc1MGFXOXVLVG9LSUNBZ0lDSWlJa1JUZGpRZ1lYUjBaVzUwYVc5dUlIZHBkR2dnWUhkeFgySmdMMkIzYjE5aFlDOWdkMjlmWW1BZ2MzQnNhWFFnWVdOeWIzTnpJR0VnWTJocGNDZHpJSFIzYnlCamIzSmxjeTRpSWlJS0NpQWdJQ0JrWldZZ1gxOXBibWwwWDE4b2MyVnNaaXdnZG14c2JWOWpiMjVtYVdjc0lIQnlaV1pwZURvZ2MzUnlJRDBnSWlJc0lDcGhjbWR6TENBcUttdDNZWEpuY3lrZ0xUNGdUbTl1WlRvS0lDQWdJQ0FnSUNBaklGUm9aU0J6Y0d4cGRDQm9ZV3gyWlhNZ1lTQnNZWGxsY2lCMGFHRjBJR2h2YkdSeklHRnNiQ0JvWldGa2N6c2dkVzVrWlhJZ2RreE1UU0JVVUQ0eElIUm9aUW9nSUNBZ0lDQWdJQ01nYUdWaFpITWdZWEpsSUdGc2NtVmhaSGtnYzNCc2FYUXNJSE52SUhSb1pTQm9ZV3gyWlhNZ2QyOTFiR1FnWW1VZ2IyWWdZU0J6YUdGeVpDNEtJQ0FnSUNBZ0lDQjBjRjl6YVhwbElEMGdjR0Z5WVd4c1pXeGZjM1JoZEdVdVoyVjBYM1JsYm5OdmNsOXRiMlJsYkY5d1lYSmhiR3hsYkY5M2IzSnNaRjl6YVhwbEtDa0tJQ0FnSUNBZ0lDQnBaaUIwY0Y5emFYcGxJQ0U5SURFNkNpQWdJQ0FnSUNBZ0lDQWdJSEpoYVhObElGSjFiblJwYldWRmNuSnZjaWdLSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdJQ0pVVUZWZlJGTldORjlCVkZST1gwbE9WRkpCWDBOSVNWQmZWRkFnY21WeGRXbHlaWE1nZEdWdWMyOXlJSEJoY21Gc2JHVnNJSE5wZW1VZ01Td2dJZ29nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdaaUpuYjNRZ2UzUndYM05wZW1WOUlnb2dJQ0FnSUNBZ0lDQWdJQ0FwQ2lBZ0lDQWdJQ0FnYzNWd1pYSW9LUzVmWDJsdWFYUmZYeWgyYkd4dFgyTnZibVpwWnl3Z2NISmxabWw0TENBcVlYSm5jeXdnS2lwcmQyRnlaM01wQ2lBZ0lDQWdJQ0FnWjNKdmRYQWdQU0JuWlhSZlkyaHBjRjl3WVdseVgyZHliM1Z3S0NrS0lDQWdJQ0FnSUNCeVlXNXJMQ0J6YVhwbElEMGdaM0p2ZFhBdWNtRnVhMTlwYmw5bmNtOTFjQ3dnWjNKdmRYQXVkMjl5YkdSZmMybDZaUW9nSUNBZ0lDQWdJSEYxWVc1MFgyTnZibVpwWnlBOUlIWnNiRzFmWTI5dVptbG5MbkYxWVc1MFgyTnZibVpwWndvZ0lDQWdJQ0FnSUNNZ1VtVmlkV2xzZENCb1lXeG1MWE5wZW1VN0lIUm9aU0J5WlhCc2FXTmhkR1ZrSUc5dVpYTWdabkp2YlNCMGFHVWdZbUZ6WlNCaGNtVWdaSEp2Y0hCbFpDNEtJQ0FnSUNBZ0lDQnpaV3htTG5keFgySWdQU0JEYjJ4MWJXNVFZWEpoYkd4bGJFeHBibVZoY2lnS0lDQWdJQ0FnSUNBZ0lDQWdjMlZzWmk1eFgyeHZjbUZmY21GdWF5d0tJQ0FnSUNBZ0lDQWdJQ0FnYzJWc1ppNXVYMmhsWVdSeklDb2djMlZzWmk1b1pXRmtYMlJwYlN3S0lDQWdJQ0FnSUNBZ0lDQWdZbWxoY3oxR1lXeHpaU3dLSUNBZ0lDQWdJQ0FnSUNBZ2NYVmhiblJmWTI5dVptbG5QWEYxWVc1MFgyTnZibVpwWnl3S0lDQWdJQ0FnSUNBZ0lDQWdjbVYwZFhKdVgySnBZWE05Um1Gc2MyVXNDaUFnSUNBZ0lDQWdJQ0FnSUhCeVpXWnBlRDFtSW50d2NtVm1hWGg5TG5keFgySWlMQW9nSUNBZ0lDQWdJQ0FnSUNCMGNGOXlZVzVyUFhKaGJtc3NDaUFnSUNBZ0lDQWdJQ0FnSUhSd1gzTnBlbVU5YzJsNlpTd0tJQ0FnSUNBZ0lDQXBDaUFnSUNBZ0lDQWdjMlZzWmk1M2IxOWhJRDBnUTI5c2RXMXVVR0Z5WVd4c1pXeE1hVzVsWVhJb0NpQWdJQ0FnSUNBZ0lDQWdJSE5sYkdZdWJsOW9aV0ZrY3lBcUlITmxiR1l1YUdWaFpGOWthVzBnTHk4Z2MyVnNaaTV1WDJkeWIzVndjeXdLSUNBZ0lDQWdJQ0FnSUNBZ2MyVnNaaTV1WDJkeWIzVndjeUFxSUhObGJHWXViMTlzYjNKaFgzSmhibXNzQ2lBZ0lDQWdJQ0FnSUNBZ0lHSnBZWE05Um1Gc2MyVXNDaUFnSUNBZ0lDQWdJQ0FnSUhGMVlXNTBYMk52Ym1acFp6MXhkV0Z1ZEY5amIyNW1hV2NzQ2lBZ0lDQWdJQ0FnSUNBZ0lISmxkSFZ5Ymw5aWFXRnpQVVpoYkhObExBb2dJQ0FnSUNBZ0lDQWdJQ0J3Y21WbWFYZzlaaUo3Y0hKbFptbDRmUzUzYjE5aElpd0tJQ0FnSUNBZ0lDQWdJQ0FnZEhCZmNtRnVhejF5WVc1ckxBb2dJQ0FnSUNBZ0lDQWdJQ0IwY0Y5emFYcGxQWE5wZW1Vc0NpQWdJQ0FnSUNBZ0tRb2dJQ0FnSUNBZ0lITmxiR1l1ZDI5ZllTNXBjMTlpYlcwZ1BTQlVjblZsQ2lBZ0lDQWdJQ0FnYzJWc1ppNTNiMTloTG1KdGJWOWlZWFJqYUY5emFYcGxJRDBnYzJWc1ppNXVYMmR5YjNWd2N5QXZMeUJ6YVhwbENpQWdJQ0FnSUNBZ0l5QlNiM2RRWVhKaGJHeGxiRXhwYm1WaGNpQjBZV3RsY3lCdWJ5QjBjRjl5WVc1ckwzUndYM05wZW1Vc0lITnZJR2wwSUdseklHSjFhV3gwSUdGMElHMTVDaUFnSUNBZ0lDQWdJeUJvWVd4bUlHOW1JSFJvWlNCcGJuQjFkQ0J5YjNkeklIZHBkR2dnVkZBZ2IyWm1JR0Z1WkNCd2IybHVkR1ZrSUdGMElHbDBjeUJ6YkdsalpTQmlaV3h2ZHk0S0lDQWdJQ0FnSUNCelpXeG1MbmR2WDJJZ1BTQlNiM2RRWVhKaGJHeGxiRXhwYm1WaGNpZ0tJQ0FnSUNBZ0lDQWdJQ0FnYzJWc1ppNXVYMmR5YjNWd2N5QXFJSE5sYkdZdWIxOXNiM0poWDNKaGJtc2dMeThnYzJsNlpTd0tJQ0FnSUNBZ0lDQWdJQ0FnYzJWc1ppNW9hV1JrWlc1ZmMybDZaU3dLSUNBZ0lDQWdJQ0FnSUNBZ1ltbGhjejFHWVd4elpTd0tJQ0FnSUNBZ0lDQWdJQ0FnY1hWaGJuUmZZMjl1Wm1sblBYRjFZVzUwWDJOdmJtWnBaeXdLSUNBZ0lDQWdJQ0FnSUNBZ2NtVmtkV05sWDNKbGMzVnNkSE05Um1Gc2MyVXNDaUFnSUNBZ0lDQWdJQ0FnSUhKbGRIVnlibDlpYVdGelBVWmhiSE5sTEFvZ0lDQWdJQ0FnSUNBZ0lDQndjbVZtYVhnOVppSjdjSEpsWm1sNGZTNTNiMTlpSWl3S0lDQWdJQ0FnSUNBZ0lDQWdaR2x6WVdKc1pWOTBjRDFVY25WbExBb2dJQ0FnSUNBZ0lDa0tJQ0FnSUNBZ0lDQm1iM0lnYkdGNVpYSWdhVzRnS0hObGJHWXVkM0ZmWWl3Z2MyVnNaaTUzYjE5aExDQnpaV3htTG5kdlgySXBPZ29nSUNBZ0lDQWdJQ0FnSUNCZmRYTmxYMjE1WDNOc2FXTmxLR3hoZVdWeUxDQnlZVzVyTENCemFYcGxLUW9nSUNBZ0lDQWdJRzlpYW1WamRDNWZYM05sZEdGMGRISmZYeWh6Wld4bUxDQWlYMk5vYVhCZmNHRnBjaUlzSUdkeWIzVndLUW9nSUNBZ0lDQWdJR3h2WjJkbGNpNXBibVp2WDI5dVkyVW9DaUFnSUNBZ0lDQWdJQ0FnSUNKQlZGUk9YMGxPVkZKQlgwTklTVkJmVkZBNklIZHhYMkl2ZDI5ZllTOTNiMTlpSUhOb1lYSmtaV1FnWVdOeWIzTnpJSFJvWlNCamFHbHdJSEJoYVhJaUNpQWdJQ0FnSUNBZ0tRb0tJQ0FnSUdSbFppQndjbTlqWlhOelgzZGxhV2RvZEhOZllXWjBaWEpmYkc5aFpHbHVaeWh6Wld4bUxDQmhZM1JmWkhSNWNHVTZJSFJ2Y21Ob0xtUjBlWEJsS1NBdFBpQk9iMjVsT2dvZ0lDQWdJQ0FnSUNNZ1ZHaGxJR0poYzJVZ2NtVnphR0Z3WlhNZ2QyOWZZU0JpZVNCMGFHVWdiRzlqWVd3Z2FHVmhaQ0JoYm1RZ1ozSnZkWEFnWTI5MWJuUnpMQ0IzYUdsamFDQmhjbVVLSUNBZ0lDQWdJQ0FqSUdoaGJIWmxaQ0JtYjNJZ2RHaGxjMlVnZDJWcFoyaDBjeUJpZFhRZ2JtOTBJR1p2Y2lCaGRIUmxiblJwYjI0c0lITnZJSE4zWVhBZ2RHaGxiU0JwYmlCb1pYSmxMZ29nSUNBZ0lDQWdJSE5wZW1VZ1BTQnpaV3htTGw5amFHbHdYM0JoYVhJdWQyOXliR1JmYzJsNlpRb2dJQ0FnSUNBZ0lHaGxZV1J6TENCbmNtOTFjSE1nUFNCelpXeG1MbTVmYkc5allXeGZhR1ZoWkhNc0lITmxiR1l1Ymw5c2IyTmhiRjluY205MWNITUtJQ0FnSUNBZ0lDQnpaV3htTG01ZmJHOWpZV3hmYUdWaFpITXNJSE5sYkdZdWJsOXNiMk5oYkY5bmNtOTFjSE1nUFNCb1pXRmtjeUF2THlCemFYcGxMQ0JuY205MWNITWdMeThnYzJsNlpRb2dJQ0FnSUNBZ0lIUnllVG9LSUNBZ0lDQWdJQ0FnSUNBZ2MzVndaWElvS1M1d2NtOWpaWE56WDNkbGFXZG9kSE5mWVdaMFpYSmZiRzloWkdsdVp5aGhZM1JmWkhSNWNHVXBDaUFnSUNBZ0lDQWdabWx1WVd4c2VUb0tJQ0FnSUNBZ0lDQWdJQ0FnYzJWc1ppNXVYMnh2WTJGc1gyaGxZV1J6TENCelpXeG1MbTVmYkc5allXeGZaM0p2ZFhCeklEMGdhR1ZoWkhNc0lHZHliM1Z3Y3dvS0lDQWdJR1JsWmlCaGRIUmxiblJwYjI1ZmFXMXdiQ2h6Wld4bUxDQm9hV1JrWlc1ZmMzUmhkR1Z6TENCeGNpd2dhM1lzSUdsdVpHVjRaWEpmZDJWcFoyaDBjeXdnY0c5emFYUnBiMjV6TENCdmRYUXBPZ29nSUNBZ0lDQWdJR2R5YjNWd0lEMGdjMlZzWmk1ZlkyaHBjRjl3WVdseUNpQWdJQ0FnSUNBZ2NtRnVheUE5SUdkeWIzVndMbkpoYm10ZmFXNWZaM0p2ZFhBS0lDQWdJQ0FnSUNCdWRXMWZkRzlyWlc1eklEMGdjWEl1YzJoaGNHVmJNRjBLSUNBZ0lDQWdJQ0FqSUhFZ1ptOXlJRzE1SUdobFlXUnpJRzkyWlhJZ1cyTnZjbVV3SUhSdmEyVnVjenNnWTI5eVpURWdkRzlyWlc1elhTNEtJQ0FnSUNBZ0lDQnhYMmhoYkdZZ1BTQnpaV3htTG5keFgySW9aM0p2ZFhBdVlXeHNYMmRoZEdobGNpaHhjaTVqYjI1MGFXZDFiM1Z6S0Nrc0lHUnBiVDB3S1NrS0lDQWdJQ0FnSUNCeFgyaGhiR1lnUFNCeFgyaGhiR1l1ZG1sbGR5Z3lMQ0J1ZFcxZmRHOXJaVzV6TENBdE1Ta0tJQ0FnSUNBZ0lDQWpJRTE1SUhSdmEyVnVjeUIzYVhSb0lHMTVJR2hsWVdSeklITjBZWGs3SUcxNUlIUnZhMlZ1Y3lCM2FYUm9JSFJvWlNCd1pXVnlKM01nYUdWaFpITWdZWEp5YVhabExnb2dJQ0FnSUNBZ0lIQmxaWElnUFNCZmMzZGhjRjkzYVhSb1gzQmxaWElvWjNKdmRYQXNJSEZmYUdGc1psdHlZVzVyWFN3Z2NWOW9ZV3htV3pFZ0xTQnlZVzVyWFNrS0lDQWdJQ0FnSUNCd2FXVmpaWE1nUFNCYmNWOW9ZV3htVzNKaGJtdGRMQ0J3WldWeVhTQnBaaUJ5WVc1cklEMDlJREFnWld4elpTQmJjR1ZsY2l3Z2NWOW9ZV3htVzNKaGJtdGRYUW9nSUNBZ0lDQWdJSEVnUFNCMGIzSmphQzVqWVhRb2NHbGxZMlZ6TENCa2FXMDlMVEVwTG5acFpYY29DaUFnSUNBZ0lDQWdJQ0FnSUc1MWJWOTBiMnRsYm5Nc0lITmxiR1l1Yld4aFgyRjBkRzR1Ymw5c2IyTmhiRjlvWldGa2N5d2djMlZzWmk1dGJHRmZZWFIwYmk1b1pXRmtYMlJwYlFvZ0lDQWdJQ0FnSUNrS0lDQWdJQ0FnSUNCeElEMGdjMlZzWmk1eGJtOXliVjl5YjNCbEtIRXNJSEJ2YzJsMGFXOXVjeWtLSUNBZ0lDQWdJQ0JyZGlBOUlITmxiR1l1YTNaZmNtOXdaU2hyZGl3Z2NHOXphWFJwYjI1ektRb0tJQ0FnSUNBZ0lDQjBiM0JyWDJsdVpHbGpaWE1nUFNCT2IyNWxDaUFnSUNBZ0lDQWdZMjl0Y0hKbGMzTnZjaUE5SUdkbGRHRjBkSElvYzJWc1ppNXRiR0ZmWVhSMGJpd2dJbU52YlhCeVpYTnpiM0lpTENCT2IyNWxLUW9nSUNBZ0lDQWdJR2xtSUhObGJHWXVhVzVrWlhobGNpQnBjeUJ1YjNRZ1RtOXVaVG9LSUNBZ0lDQWdJQ0FnSUNBZ2FXNWtaWGhsY2w5bGJXSWdQU0JuWlhSaGRIUnlLQW9nSUNBZ0lDQWdJQ0FnSUNBZ0lDQWdjMlZzWmk1dGJHRmZZWFIwYml3S0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUNKcGJtUmxlR1Z5WDNKdmRHRnllVjlsYldJaUxBb2dJQ0FnSUNBZ0lDQWdJQ0FnSUNBZ1oyVjBZWFIwY2loelpXeG1MbTFzWVY5aGRIUnVMQ0FpY205MFlYSjVYMlZ0WWlJc0lFNXZibVVwTEFvZ0lDQWdJQ0FnSUNBZ0lDQXBDaUFnSUNBZ0lDQWdJQ0FnSUhSdmNHdGZhVzVrYVdObGN5QTlJSE5sYkdZdWFXNWtaWGhsY2lnS0lDQWdJQ0FnSUNBZ0lDQWdJQ0FnSUdocFpHUmxibDl6ZEdGMFpYTXNJSEZ5TENCcGJtUmxlR1Z5WDNkbGFXZG9kSE1zSUhCdmMybDBhVzl1Y3l3Z2FXNWtaWGhsY2w5bGJXSUtJQ0FnSUNBZ0lDQWdJQ0FnS1FvZ0lDQWdJQ0FnSUdsbUlHTnZiWEJ5WlhOemIzSWdhWE1nYm05MElFNXZibVU2Q2lBZ0lDQWdJQ0FnSUNBZ0lHTnZiWEJ5WlhOemIzSW9hR2xrWkdWdVgzTjBZWFJsY3l3Z2NHOXphWFJwYjI1ekxDQnpaV3htTG0xc1lWOWhkSFJ1TG5KdmRHRnllVjlsYldJcENnb2dJQ0FnSUNBZ0lISmxkSFZ5YmlCelpXeG1MbVp2Y25kaGNtUmZiWEZoS0hFc0lHdDJMQ0J3YjNOcGRHbHZibk1zSUU1dmJtVXNJSFJ2Y0d0ZmFXNWthV05sY3oxMGIzQnJYMmx1WkdsalpYTXBDZ29nSUNBZ1pHVm1JRjl2WDNCeWIyb29jMlZzWml3Z2J5d2djRzl6YVhScGIyNXpLVG9LSUNBZ0lDQWdJQ0JuY205MWNDQTlJSE5sYkdZdVgyTm9hWEJmY0dGcGNnb2dJQ0FnSUNBZ0lISmhibXNnUFNCbmNtOTFjQzV5WVc1clgybHVYMmR5YjNWd0NpQWdJQ0FnSUNBZ2JuVnRYM1J2YTJWdWN5QTlJRzh1YzJoaGNHVmJNRjBLSUNBZ0lDQWdJQ0JvWVd4bUlEMGdjMlZzWmk1dVgyeHZZMkZzWDJobFlXUnpJQzh2SURJS0lDQWdJQ0FnSUNCdklEMGdieTV5WlhOb1lYQmxLRzUxYlY5MGIydGxibk1zSURJc0lHaGhiR1lzSUhObGJHWXVhR1ZoWkY5a2FXMHBDaUFnSUNBZ0lDQWdJeUJRWldWeUozTWdkRzlyWlc1eklIZHBkR2dnYlhrZ2FHVmhaSE1nWVhKeWFYWmxPeUJ0ZVNCMGIydGxibk1nZDJsMGFDQnRlU0JvWldGa2N5QnpkR0Y1TGdvZ0lDQWdJQ0FnSUhCbFpYSWdQU0JmYzNkaGNGOTNhWFJvWDNCbFpYSW9aM0p2ZFhBc0lHOWJPaXdnY21GdWExMHNJRzliT2l3Z01TQXRJSEpoYm10ZEtRb2dJQ0FnSUNBZ0lIQnBaV05sY3lBOUlGdHZXem9zSUhKaGJtdGRMQ0J3WldWeVhTQnBaaUJ5WVc1cklEMDlJREFnWld4elpTQmJjR1ZsY2l3Z2IxczZMQ0J5WVc1clhWMEtJQ0FnSUNBZ0lDQnZYMmhoYkdZZ1BTQjBiM0pqYUM1allYUW9jR2xsWTJWekxDQmthVzA5TUNrdVkyOXVkR2xuZFc5MWN5Z3BDaUFnSUNBZ0lDQWdjRzl6YVhScGIyNXpYMkZzYkNBOUlHZHliM1Z3TG1Gc2JGOW5ZWFJvWlhJb0NpQWdJQ0FnSUNBZ0lDQWdJSEJ2YzJsMGFXOXVjeTUwYnloMGIzSmphQzVwYm5Rek1pa3VZMjl1ZEdsbmRXOTFjeWdwTENCa2FXMDlNQW9nSUNBZ0lDQWdJQ2t1ZEc4b2NHOXphWFJwYjI1ekxtUjBlWEJsS1FvZ0lDQWdJQ0FnSUhvZ1BTQnpaV3htTG05ZmNISnZhbDl2Y0NnS0lDQWdJQ0FnSUNBZ0lDQWdiMTlvWVd4bUxBb2dJQ0FnSUNBZ0lDQWdJQ0J3YjNOcGRHbHZibk5mWVd4c0xBb2dJQ0FnSUNBZ0lDQWdJQ0J6Wld4bUxuSnZkR0Z5ZVY5bGJXSXVZMjl6WDNOcGJsOWpZV05vWlN3S0lDQWdJQ0FnSUNBZ0lDQWdjMlZzWmk1M2IxOWhMbmRsYVdkb2RDd0tJQ0FnSUNBZ0lDQWdJQ0FnYzJWc1ppNTNiMTloTG5kbGFXZG9kRjl6WTJGc1pTd0tJQ0FnSUNBZ0lDQXBDaUFnSUNBZ0lDQWdjR0Z5ZEdsaGJDQTlJSE5sYkdZdWQyOWZZaWg2S1FvZ0lDQWdJQ0FnSUdsbUlHbHphVzV6ZEdGdVkyVW9jR0Z5ZEdsaGJDd2dkSFZ3YkdVcE9nb2dJQ0FnSUNBZ0lDQWdJQ0J3WVhKMGFXRnNJRDBnY0dGeWRHbGhiRnN3WFFvZ0lDQWdJQ0FnSUhKbGRIVnliaUJuY205MWNDNXlaV1IxWTJWZmMyTmhkSFJsY2lod1lYSjBhV0ZzTG1OdmJuUnBaM1Z2ZFhNb0tTd2daR2x0UFRBcENnPT0iKSkKZSA9IHJvb3QgLyAiZW52cy5weSI7IHMgPSBlLnJlYWRfdGV4dCgpCmlmICJUUFVfRFNWNF9BVFROX0lOVFJBX0NISVBfVFAiIG5vdCBpbiBzOgogICAgYSA9ICIgICAgVFBVX01PRV9TS0lQX1BBRERFRF9UT0tFTlM6IGJvb2wgPSBGYWxzZVxuIgogICAgYiA9ICcgICAgIlRQVV9NT0VfU0tJUF9QQURERURfVE9LRU5TIjogZW52X2Jvb2woIlRQVV9NT0VfU0tJUF9QQURERURfVE9LRU5TIiksXG4nCiAgICBhc3NlcnQgcy5jb3VudChhKSA9PSAxIGFuZCBzLmNvdW50KGIpID09IDEsICJlbnZzLnB5IGFuY2hvcnMiCiAgICBzID0gcy5yZXBsYWNlKGEsIGEgKyAiICAgIFRQVV9EU1Y0X0FUVE5fSU5UUkFfQ0hJUF9UUDogYm9vbCA9IEZhbHNlXG4iKQogICAgcyA9IHMucmVwbGFjZShiLCBiICsgJyAgICAiVFBVX0RTVjRfQVRUTl9JTlRSQV9DSElQX1RQIjogZW52X2Jvb2woIlRQVV9EU1Y0X0FUVE5fSU5UUkFfQ0hJUF9UUCIpLFxuJykKICAgIGUud3JpdGVfdGV4dChzKQptID0gcm9vdCAvICJtb2RlbHMvdmxsbS9kZWVwc2Vla192NC9tb2RlbC5weSI7IHQgPSBtLnJlYWRfdGV4dCgpCmlmICJWbGxtRGVlcHNlZWtWNE1MQUF0dGVudGlvbkludHJhQ2hpcCIgbm90IGluIHQ6CiAgICBvbGQgPSAiICAgICAgICBzZWxmLmF0dG4gPSBWbGxtRGVlcHNlZWtWNE1MQUF0dGVudGlvbihcbiIKICAgIG5ldyA9ICgKICAgICAgICAiICAgICAgICBhdHRuX2NscyA9IFZsbG1EZWVwc2Vla1Y0TUxBQXR0ZW50aW9uXG4iCiAgICAgICAgIiAgICAgICAgaWYgZW52cy5UUFVfRFNWNF9BVFROX0lOVFJBX0NISVBfVFA6XG4iCiAgICAgICAgIiAgICAgICAgICAgIGZyb20gdmxsbV90b3JjaHRwdS5tb2RlbHMudmxsbS5kZWVwc2Vla192NC5hdHRuX2ludHJhX2NoaXAgaW1wb3J0IChcbiIKICAgICAgICAiICAgICAgICAgICAgICAgIFZsbG1EZWVwc2Vla1Y0TUxBQXR0ZW50aW9uSW50cmFDaGlwIGFzIGF0dG5fY2xzLFxuIgogICAgICAgICIgICAgICAgICAgICApXG4iCiAgICAgICAgIiAgICAgICAgc2VsZi5hdHRuID0gYXR0bl9jbHMoXG4iCiAgICApCiAgICBhc3NlcnQgdC5jb3VudChvbGQpID09IDEsICJtb2RlbC5weSBhbmNob3IiCiAgICB0ID0gdC5yZXBsYWNlKG9sZCwgbmV3KQogICAgaWYgImZyb20gdmxsbV90b3JjaHRwdSBpbXBvcnQgZW52c1xuIiBub3QgaW4gdDoKICAgICAgICBpID0gdC5pbmRleCgiXG5mcm9tIHZsbG1fdG9yY2h0cHUiKQogICAgICAgIHQgPSB0WzppXSArICJcbmZyb20gdmxsbV90b3JjaHRwdSBpbXBvcnQgZW52cyIgKyB0W2k6XQogICAgbS53cml0ZV90ZXh0KHQpCnByaW50KCJBcHBsaWVkIEFUVE5fSU5UUkFfQ0hJUF9UUCAodmxsbS10b3JjaHRwdSBjb21taXQgMWZlYzBkMWZjKSIpCg==" | base64 -d > /tmp/aic_apply.py
  python3 /tmp/aic_apply.py && python3 -m py_compile /workspace/vllm-torchtpu/src/vllm_torchtpu/models/vllm/deepseek_v4/attn_intra_chip.py /workspace/vllm-torchtpu/src/vllm_torchtpu/models/vllm/deepseek_v4/model.py /workspace/vllm-torchtpu/src/vllm_torchtpu/envs.py
|||;

// --- Benchmark client --------------------------------------------------------
local fastMode = cdk.boolVar('FAST', false) || cdk.boolVar('FAST_MODE', false);
local customDuration = cdk.rawVar('DURATION', '') != '';
local duration =
  if customDuration then cdk.numVar('DURATION')
  else if fastMode then 1200
  else 3600;

local rawWarmup = cdk.var('AIPERF_WARMUP_REQUESTS_PER_LANE', cdk.var('WARMUP_REQUESTS_PER_LANE', cdk.var('WARMUP', '10')));
local warmupReqs = if rawWarmup == '0' || rawWarmup == '' then '10' else rawWarmup;

local concurrencies = agentx.parseConcurrencies(cdk.var('CLIENT_CONCURRENCIES', '4 8 16 32 64'));
local benchName = cdk.var('BENCH_NAME', 'deepseek-v4-agentx-multihost-nondisagg');

local serverJobName = 'ray-node';
local benchJobName = 'benchmark';

// ---------------------------------------------------------------------------
// The Ray cluster
// ---------------------------------------------------------------------------

local tpuProcessAddresses = if cdk.rawVar('TPU_PROCESS_ADDRESSES', '') != '' then
  cdk.var('TPU_PROCESS_ADDRESSES')
else
  std.join(',', [
    '$(JS_NAME)-%s-%d-0.$(JS_NAME).$(POD_NAMESPACE).svc.cluster.local:8476' % [serverJobName, i]
    for i in std.range(0, rayReplicas - 1)
  ]);

local offloadSetup = if !kvOffload then 'echo "Starting vLLM server (offload=OFF)..."\n' else agentx.fill(
  |||
    # A value small enough to be a GiB count is read as one; anything larger is bytes.
    KV_OFFLOAD_SIZE="@SIZE@"
    if [ "$KV_OFFLOAD_SIZE" -le 1000000 ]; then
      CPU_BYTES=$(( KV_OFFLOAD_SIZE * 1024 * 1024 * 1024 ))
    else
      CPU_BYTES="$KV_OFFLOAD_SIZE"
    fi
    RAIDEN_CONTROLLER_PORT="@CONTROLLER_PORT@"
    export PYTHONHASHSEED=0
    KVT='{"kv_connector":"TPURaidenOffloadingConnector","kv_connector_module_path":"vllm_torchtpu.offload.raiden_connector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":'"${CPU_BYTES}"',"raiden_controller_port":'"${RAIDEN_CONTROLLER_PORT}"',"store_server_ip":"'"${HEAD_ADDR}"'","raiden_job_name":"@RAIDEN_JOB_NAME@"}}'
    OFFLOAD_ARGS=(
      --enable-prompt-tokens-details
      --kv-transfer-config "$KVT"
    )
    echo "Starting vLLM server with TPURaidenOffloadingConnector enabled (cpu_bytes_to_use=${CPU_BYTES} per DP rank, controller_port=${RAIDEN_CONTROLLER_PORT}, head=${HEAD_ADDR})..."
  |||,
  { SIZE: kvOffloadingSize, CONTROLLER_PORT: offloadControllerPort, GLOBAL_REGISTRY_PORT: globalRegistryPort, RAIDEN_JOB_NAME: raidenJobName }
);

local serveArgs = [
  'vllm serve "$MODEL_PATH"',
  '--served-model-name "$SERVED_MODEL_NAME"',
  '--quantization %s' % quantization,
  '--trust-remote-code',
  '--data-parallel-size=%d' % dataParallelSize,
  '--tensor-parallel-size=%d' % tensorParallelSize,
]
+ (if loadFormat != '' then ['--load-format=%s' % loadFormat] else [])
+ [
  '--max-model-len=%d' % maxModelLen,
  '--max-num-seqs=%d' % maxNumSeqs,
  '--max-num-batched-tokens=%d' % maxNumBatchedTokens,
]
+ (if blockSize != 0 then ['--block-size=%d' % blockSize] else [])
+ [
  '--port %d' % port,
  '--api-server-count=1',
  '--async-scheduling',
  if enablePrefixCaching then '--enable-prefix-caching' else '--no-enable-prefix-caching',
  '--enable-prompt-tokens-details',
  '--kv-cache-dtype=%s' % kvCacheDtype,
  '--safetensors-load-strategy=prefetch',
  '--enable-expert-parallel',
  '--enable-ep-weight-filter',
  '--gpu-memory-utilization=%s' % gpuMemoryUtilization,
  '"${OFFLOAD_ARGS[@]}"',
]
+ (if schedulerCls != '' then ['--scheduler-cls %s' % schedulerCls] else [])
+ (if prefillScheduleInterval > 1 then ['--prefill-schedule-interval=%d' % prefillScheduleInterval] else [])
+ (if toolCallParser != '' then ['--tool-call-parser=%s' % toolCallParser] else [])
+ [
  '--reasoning-parser=%s' % reasoningParser,
  '--chat-template=%s' % chatTemplate,
]
+ (if enableProfiling then agentx.profilerArgs('"%s"' % xprofDir) else []);

local serverScript = agentx.fill(
  |||
    mkdir -p /dev/shm/{jax_cache,vllm_cache,torchinductor}
    echo "JOBSET_JOB_INDEX=$JOBSET_JOB_INDEX"
    if [ -n "$TPU_LIBRARY_PATH" ]; then
      echo "TPU_LIBRARY_PATH is set to $TPU_LIBRARY_PATH. Copying libtpu.so from GCS..."
      mkdir -p "$(dirname "$TPU_LIBRARY_PATH")"
      gcloud storage cp gs://cloud-devkit/libs/libtpu/20260521/libtpu.so $TPU_LIBRARY_PATH 2>/dev/null || true
    fi

    export TPU_NODE_ID=$JOBSET_JOB_INDEX
    export TPU_WORKER_ID=$JOBSET_JOB_INDEX
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
    export RAIDEN_EXPECTED_WORKERS_TIMEOUT_S=3600
    export TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL="${TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL:-30m}"

    HEAD_ADDR="@HEAD_ADDR@"
    echo "Resolved Ray head address (HEAD_ADDR) as: ${HEAD_ADDR}"

    # 1. Raiden KV offload fixes: sort pool_tensors by descending page_size_bytes (HCA 128 KiB first),
    # use get_tp_group().rank_in_group for DP>1, and support reset_cache() via key_namespace rotation.
    python3 -c 'import re; nl = chr(10); p = "/workspace/vllm-torchtpu/src/vllm_torchtpu/offload/raiden_connector.py"; s = open(p).read(); t = "        self._pool_sync_tensors = ["; patch = "        pool_tensors.sort(key=lambda x: -x.page_size_bytes)" + nl + t; ("pool_tensors.sort" not in s and t in s) and (open(p, "w").write(s.replace(t, patch, 1)), print("Patched raiden_connector.py: sorted pool_tensors by descending page_size_bytes")); s = open(p).read(); pat = r"        logger\.warning\(\s*\"TPURaidenOffloadingConnector: external cache reset is not \"\s*\"supported by the Raiden offload store; reporting failure\.\"\s*\)\n        return False"; rr = "        if self.connector_scheduler is not None:" + nl + "            self.connector_scheduler._fence_pending.clear()" + nl + "            self.connector_scheduler.reset_cache()" + nl + "        return True"; re.search(pat, s) and (open(p, "w").write(re.sub(pat, rr, s, count=1)), print("Patched raiden_connector.py: enabled reset_cache"))'
    python3 -c 'import re; nl = chr(10); p = "/workspace/vllm-torchtpu/src/vllm_torchtpu/offload/raiden_store.py"; s = open(p).read(); t = "rank = parallel_state.get_world_group().rank_in_group"; r = "rank = parallel_state.get_tp_group().rank_in_group"; (t in s) and (open(p, "w").write(s.replace(t, r)), print("Patched raiden_store.py: rank = get_tp_group().rank_in_group")); s = open(p).read(); pat = r"        raise NotImplementedError\(\s*\"TPURaidenOffloadingConnector: external cache reset is not supported \"\s*\"by the Raiden offload store\.\"\s*\)"; rr = "        if self._admissions:" + nl + "            self.drain_jobs(set(self._admissions.keys()))" + nl + "        self._key_namespace = os.urandom(8)" + nl + "        self._inflight_store_keys.clear()" + nl + "        self._remote_slices.clear()" + nl + "        self._pending_remote_slices.clear()" + nl + "        self._finished.clear()" + nl + "        logger.info(\"TPURaidenOffloadingConnector: rotated Raiden key_namespace=%s on reset_cache\", self._key_namespace.hex())"; re.search(pat, s) and (open(p, "w").write(re.sub(pat, rr, s, count=1)), print("Patched raiden_store.py: enabled reset_cache via key_namespace rotation"))'

    # 2. Apply vllm-torchtpu PR #1518 (https://github.com/vllm-project/vllm-torchtpu/pull/1518:
    # "[Kernel][Core] DSv4: keep long-context page tables out of SMEM"):
    #   - compress_norm_rope_store (buffered_ref.py, kernel.py): look up 2-5 state pages per token before pallas_call.
    #   - mla_swa.py: narrow_page_indices keeps only each sequence's sliding-window pages and shifts kv_lens.
    #   - kv_cache_dsv4.py: _refresh_ds_v4_request_caps sets num_reqs_max_model_len=64 from compressed-KV tables only.
    echo "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" | base64 -d | gunzip > /tmp/pr1518.patch
    if git -C /workspace/vllm-torchtpu apply --check /tmp/pr1518.patch 2>/dev/null; then
      git -C /workspace/vllm-torchtpu apply /tmp/pr1518.patch
      echo "Applied vllm-torchtpu PR #1518 (DSv4: keep long-context page tables out of SMEM)"
    else
      echo "vllm-torchtpu PR #1518 already applied or not applicable"
    fi
    python3 -m py_compile \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/compress_and_store/buffered_ref.py \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/compress_and_store/kernel.py \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/core_attention/mla_swa.py \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/runner/kv_cache_dsv4.py && echo "Verified PR #1518 py_compile"

    # 3. Apply vllm-torchtpu PR #1557 (https://github.com/vllm-project/vllm-torchtpu/pull/1557:
    # "[Kernel] Fused EP MoE v2: skip padded tokens when TPU_MOE_SKIP_PADDED_TOKENS=1"):
    #   - token_padding.py + fused_moe_ep.py: mask_router_logits_for_padding masks padded rows of router_logits to NaN when TPU_MOE_SKIP_PADDED_TOKENS=1.
    #   - fused_moe/v2/layer.py: _row_routes checks both scores and gate_l for finiteness, routes padded rows to sentinel expert e_total, and guards coef / slot_coef when weight == 0.
    #   - fused_moe/v2/host.py: build_routing_tables redirects sentinel e_total pairs to off_shard_slab_row(shard_stride, ep).
    #   - fused_moe/v2/kernel.py: reduce_token guards uninitialized arrival rows/scales when coef == 0.
    echo "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" | base64 -d | gunzip > /tmp/pr1557.patch
    if git -C /workspace/vllm-torchtpu apply --check /tmp/pr1557.patch 2>/dev/null; then
      git -C /workspace/vllm-torchtpu apply /tmp/pr1557.patch
      echo "Applied vllm-torchtpu PR #1557 (Fused EP MoE v2: skip padded tokens when TPU_MOE_SKIP_PADDED_TOKENS=1)"
    else
      echo "vllm-torchtpu PR #1557 already applied or applying inline fallback"
      python3 -c 'import glob; nl = chr(10); t1 = "    extra = tuple(t for t in (score_bias, hash_ids) if t is not None)"; r1 = "    try:" + nl + "        from vllm_torchtpu.layers.adapter import token_padding as _tp" + nl + "    except ImportError:" + nl + "        from vllm_torchtpu.layers.vllm import token_padding as _tp" + nl + "    if _tp._padding_state is not None:" + nl + "        _is_pad = _tp._padding_state.get_local_padding_mask(router_logits.shape[0])" + nl + "        router_logits = torch.where(_is_pad[:, None], float(\"nan\"), router_logits)" + nl + t1; [open(p, "w").write(s.replace(t1, r1, 1)) and print("Patched fused_moe_ep.py in", p) for p in glob.glob("/workspace/**/fused_moe_ep.py", recursive=True) + glob.glob("/usr/local/**/fused_moe_ep.py", recursive=True) for s in [open(p).read()] if t1 in s and "_is_pad" not in s and "mask_router_logits_for_padding" not in s]; t2r = "        row_routes = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)"; r2r = "        row_routes = jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True) & jnp.any(jnp.isfinite(gate_l), axis=-1, keepdims=True)"; t2a = "        topk_weights = jnp.where(row_routes, topk_weights, 0.0)"; r2a = t2a + nl + "        topk_idx = jnp.where(row_routes, topk_idx, jnp.int32(e_total))"; t2b = "    coef = row_scales * topk_weights.astype(jnp.float32)"; r2b = "    coef = jnp.where(topk_weights != 0, row_scales * topk_weights.astype(jnp.float32), 0.0)"; t2c = "        terms.append(slot_rows.astype(jnp.float32) * slot_coef[:, None, None])"; r2c = "        terms.append(jnp.where(slot_coef[:, None, None] != 0, slot_rows.astype(jnp.float32) * slot_coef[:, None, None], 0.0))"; [open(p, "w").write(s.replace(t2r, r2r, 1).replace(t2a, r2a, 1).replace(t2b, r2b, 1).replace(t2c, r2c, 1)) and print("Patched fused_moe/v2/layer.py in", p) for p in glob.glob("/workspace/**/fused_moe/v2/layer.py", recursive=True) + glob.glob("/usr/local/**/fused_moe/v2/layer.py", recursive=True) for s in [open(p).read()] if t2a in s and "jnp.int32(e_total)" not in s]; t3 = "    slab_row = base_of_pair + slot  # always < total"; r3 = "    slab_row = jnp.where(expert_blocks.reshape(-1) < e_total, base_of_pair + slot, jnp.int32(off_shard_slab_row(shard_stride, ep)))"; [open(p, "w").write(s.replace(t3, r3, 1)) and print("Patched fused_moe/v2/host.py in", p) for p in glob.glob("/workspace/**/fused_moe/v2/host.py", recursive=True) + glob.glob("/usr/local/**/fused_moe/v2/host.py", recursive=True) for s in [open(p).read()] if t3 in s]; t4 = "                term = (" + nl + "                    rows_vm[parity, k * tb + i].astype(jnp.float32)" + nl + "                    * coef_sm[(base + i) * topk + k]" + nl + "                )"; r4 = "                _c = coef_sm[(base + i) * topk + k]" + nl + "                term = jnp.where(" + nl + "                    _c != 0," + nl + "                    rows_vm[parity, k * tb + i].astype(jnp.float32) * _c," + nl + "                    0.0," + nl + "                )"; [open(p, "w").write(s.replace(t4, r4, 1)) and print("Patched fused_moe/v2/kernel.py in", p) for p in glob.glob("/workspace/**/fused_moe/v2/kernel.py", recursive=True) + glob.glob("/usr/local/**/fused_moe/v2/kernel.py", recursive=True) for s in [open(p).read()] if t4 in s]'
    fi
    python3 -m py_compile \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/fused_moe/v2/host.py \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/fused_moe/v2/kernel.py \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/fused_moe/v2/layer.py \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/layers/adapter/fused_moe_ep.py \
      /workspace/vllm-torchtpu/src/vllm_torchtpu/layers/adapter/token_padding.py && echo "Verified PR #1557 py_compile"

    # 3b. Attention-weight intra-chip TP (ATTN_INTRA_CHIP_TP).
    @AIC_PATCH@

    # 4. Suppress unhandled cancelled RuntimeError in shm_broadcast.py during phase transitions
    python3 -c 'import glob; [open(p, "w").write(s.replace("raise RuntimeError(\"cancelled\")", "return None")) for p in glob.glob("/workspace/**/shm_broadcast.py", recursive=True) + glob.glob("/usr/local/**/shm_broadcast.py", recursive=True) for s in [open(p).read()] if "raise RuntimeError(\"cancelled\")" in s]'

    # 5. Apply PR #1728 (guard solo-prefill async-scheduling flush in TPUModelRunner),
    # PR #1729 (break .t().contiguous() view lineage & avoid per-step clones in VllmDeepseekCompressor),
    # and Dynamic BT in TpuDpScheduler (BT=512 for C<=8, BT=128 for C>=16).
    python3 -c 'import glob, py_compile; nl = chr(10); t1 = "        previous_req_ids = self._pre_async_results.req_id_to_index_copy" + nl + "        if any(req_id in current_req_ids for req_id in previous_req_ids):" + nl + "            return"; r1 = "        previous_req_ids = self._pre_async_results.req_id_to_index_copy" + nl + "        if not previous_req_ids:" + nl + "            self._pre_async_results = None" + nl + "            return" + nl + "        if any(req_id in current_req_ids for req_id in previous_req_ids):" + nl + "            return"; t2 = "        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()"; r2 = "        if not pre_request_seq_lens:" + nl + "            return" + nl + "        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()"; [open(p, "w").write(s.replace(t1, r1, 1).replace(t2, r2, 1)) and py_compile.compile(p, doraise=True) and print("Patched solo-prefill async guard in", p) for p in glob.glob("/workspace/**/tpu_runner.py", recursive=True) + glob.glob("/usr/local/**/tpu_runner.py", recursive=True) for s in [open(p).read()] if t1 in s and "if not previous_req_ids:" not in s]; tc1 = "        weight = cast(torch.Tensor, self.fused_wkv_wgate.weight).data.t().contiguous()" + nl + "        self.fused_wkv_wgate.weight = torch.nn.Parameter(weight, requires_grad=False)"; rc1 = "        w_t = cast(torch.Tensor, self.fused_wkv_wgate.weight).data.t().contiguous()" + nl + "        weight = torch.empty(w_t.shape, dtype=w_t.dtype, device=w_t.device).copy_(w_t)" + nl + "        self.fused_wkv_wgate.weight = torch.nn.Parameter(weight, requires_grad=False)" + nl + "        if self.fused_wkv_wgate.weight.device.type == \"tpu\":" + nl + "            from vllm_torchtpu.utils import synchronize_tensors" + nl + "            synchronize_tensors([self.fused_wkv_wgate.weight])"; tc2 = "            self.ape.clone()," + nl + "            self.norm.weight.clone(),"; rc2 = "            self.ape," + nl + "            self.norm.weight,"; [open(p, "w").write(s.replace(tc1, rc1, 1).replace(tc2, rc2, 1)) and py_compile.compile(p, doraise=True) and print("Patched VllmDeepseekCompressor in", p) for p in glob.glob("/workspace/**/deepseek_v4_compressor.py", recursive=True) + glob.glob("/usr/local/**/deepseek_v4_compressor.py", recursive=True) for s in [open(p).read()] if tc1 in s and "torch.empty(w_t.shape" not in s]' && echo "Verified tpu_runner & deepseek_v4_compressor py_compile"
    python3 -c 'import base64, glob, py_compile; code = base64.b64decode("IyBDb3B5cmlnaHQgMjAyNiBHb29nbGUgTExDCiMKIyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKIyB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuCiMgWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0CiMKIyAgICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wCiMKIyBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlCiMgZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gIkFTIElTIiBCQVNJUywKIyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KIyBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kCiMgbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuCmltcG9ydCBtbWFwCmltcG9ydCBvcwppbXBvcnQgc3RydWN0Cgpmcm9tIHZsbG0ubG9nZ2VyIGltcG9ydCBpbml0X2xvZ2dlcgpmcm9tIHZsbG0udjEuY29yZS5zY2hlZC5hc3luY19zY2hlZHVsZXIgaW1wb3J0IEFzeW5jU2NoZWR1bGVyCmZyb20gdmxsbS52MS5jb3JlLnNjaGVkLmludGVyZmFjZSBpbXBvcnQgUGF1c2VTdGF0ZQpmcm9tIHZsbG0udjEuY29yZS5zY2hlZC5vdXRwdXQgaW1wb3J0IFNjaGVkdWxlck91dHB1dAoKbG9nZ2VyID0gaW5pdF9sb2dnZXIoX19uYW1lX18pCgpfU0hNX0NPTkNfUEFUSCA9ICIvZGV2L3NobS9kc3Y0X2RwX2NvbmN1cnJlbmN5LmJpbiIKX1NITV9CWVRFX1NJWkUgPSA3MiAgIyAxOCBpbnQzMiBzbG90czogWzAuLjE1XT1wZXItcmFuayBhY3RpdmUsIFsxNl09cm91dGVyIGluZmxpZ2h0LCBbMTddPXN0YWdlIHBlYWsKCgpkZWYgX29wZW5fY29uY19zaG0oKToKICAgIHRyeToKICAgICAgICBmZCA9IG9zLm9wZW4oX1NITV9DT05DX1BBVEgsIG9zLk9fQ1JFQVQgfCBvcy5PX1JEV1IsIDBvNjY2KQogICAgICAgIHRyeToKICAgICAgICAgICAgaWYgb3MuZnN0YXQoZmQpLnN0X3NpemUgPCBfU0hNX0JZVEVfU0laRToKICAgICAgICAgICAgICAgIG9zLmZ0cnVuY2F0ZShmZCwgX1NITV9CWVRFX1NJWkUpCiAgICAgICAgICAgIHJldHVybiBtbWFwLm1tYXAoZmQsIF9TSE1fQllURV9TSVpFLCBtbWFwLk1BUF9TSEFSRUQsIG1tYXAuUFJPVF9SRUFEIHwgbW1hcC5QUk9UX1dSSVRFKQogICAgICAgIGZpbmFsbHk6CiAgICAgICAgICAgIG9zLmNsb3NlKGZkKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICByZXR1cm4gTm9uZQoKCmNsYXNzIFRwdURwU2NoZWR1bGVyKEFzeW5jU2NoZWR1bGVyKToKICAgICIiIlNjaGVkdWxlciB0aGF0IGFsaWducyBwcmVmaWxsIGFkbWlzc2lvbiBhY3Jvc3MgRFAgcmFua3MgYW5kIGR5bmFtaWNhbGx5CiAgICBzY2FsZXMgbWF4X251bV9zY2hlZHVsZWRfdG9rZW5zIGJ5IGFjdGl2ZSBjbHVzdGVyIGNvbmN1cnJlbmN5IChlLmcuIEJUPTUxMgogICAgZm9yIEM8PTggYW5kIEJUPTEyOCBmb3IgQz49MTYpLgogICAgIiIiCgogICAgZGVmIF9faW5pdF9fKHNlbGYsICphcmdzLCAqKmt3YXJncykgLT4gTm9uZToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCphcmdzLCAqKmt3YXJncykKICAgICAgICBzZWxmLl9kcmFpbl9wcmVmaWxsID0gRmFsc2UKICAgICAgICBzZWxmLl9zaG1fbW1hcCA9IF9vcGVuX2NvbmNfc2htKCkKICAgICAgICBwX2NmZyA9IGdldGF0dHIoc2VsZiwgInBhcmFsbGVsX2NvbmZpZyIsIE5vbmUpIG9yIGdldGF0dHIoCiAgICAgICAgICAgIGdldGF0dHIoc2VsZiwgInZsbG1fY29uZmlnIiwgTm9uZSksICJwYXJhbGxlbF9jb25maWciLCBOb25lCiAgICAgICAgKQogICAgICAgIGRwX2lkeCA9IGdldGF0dHIocF9jZmcsICJkYXRhX3BhcmFsbGVsX2luZGV4IiwgTm9uZSkKICAgICAgICBpZiBkcF9pZHggaXMgTm9uZToKICAgICAgICAgICAgZHBfaWR4ID0gZ2V0YXR0cihwX2NmZywgImRhdGFfcGFyYWxsZWxfcmFuayIsIDApCiAgICAgICAgc2VsZi5fZHBfcmFuayA9IGludChkcF9pZHggb3IgMCkgJSAxNgogICAgICAgIHNlbGYuX2NlaWxpbmdfYnQgPSBpbnQoc2VsZi5tYXhfbnVtX3NjaGVkdWxlZF90b2tlbnMpCiAgICAgICAgc2VsZi5fZHluYW1pY19idF9lbmFibGVkID0gb3MuZW52aXJvbi5nZXQoIkRZTkFNSUNfQlRfRU5BQkxFRCIsICIxIikgaW4gKAogICAgICAgICAgICAiMSIsCiAgICAgICAgICAgICJ0cnVlIiwKICAgICAgICAgICAgIlRydWUiLAogICAgICAgICkKICAgICAgICBzZWxmLl9keW5hbWljX2J0X2xvd19jb25jX21heCA9IGludCgKICAgICAgICAgICAgb3MuZW52aXJvbi5nZXQoIkRZTkFNSUNfQlRfTE9XX0NPTkNfTUFYIiwgIjgiKQogICAgICAgICkKICAgICAgICBzZWxmLl9keW5hbWljX2J0X2xvd190b2tlbnMgPSBtaW4oCiAgICAgICAgICAgIGludChvcy5lbnZpcm9uLmdldCgiRFlOQU1JQ19CVF9MT1dfVE9LRU5TIiwgIjUxMiIpKSwgc2VsZi5fY2VpbGluZ19idAogICAgICAgICkKICAgICAgICBzZWxmLl9keW5hbWljX2J0X2hpZ2hfdG9rZW5zID0gbWluKAogICAgICAgICAgICBpbnQob3MuZW52aXJvbi5nZXQoIkRZTkFNSUNfQlRfSElHSF9UT0tFTlMiLCAiMTI4IikpLCBzZWxmLl9jZWlsaW5nX2J0CiAgICAgICAgKQogICAgICAgIGxvZ2dlci5pbmZvKAogICAgICAgICAgICAiVHB1RHBTY2hlZHVsZXIgYWN0aXZlIChkcF9yYW5rPSVkLCBkeW5hbWljX2J0PSVzLCBDPD0lZCAtPiBCVD0lZCwgQz4lZCAtPiBCVD0lZCwgY2VpbGluZz0lZCkiLAogICAgICAgICAgICBzZWxmLl9kcF9yYW5rLAogICAgICAgICAgICBzZWxmLl9keW5hbWljX2J0X2VuYWJsZWQsCiAgICAgICAgICAgIHNlbGYuX2R5bmFtaWNfYnRfbG93X2NvbmNfbWF4LAogICAgICAgICAgICBzZWxmLl9keW5hbWljX2J0X2xvd190b2tlbnMsCiAgICAgICAgICAgIHNlbGYuX2R5bmFtaWNfYnRfbG93X2NvbmNfbWF4LAogICAgICAgICAgICBzZWxmLl9keW5hbWljX2J0X2hpZ2hfdG9rZW5zLAogICAgICAgICAgICBzZWxmLl9jZWlsaW5nX2J0LAogICAgICAgICkKCiAgICBkZWYgcmVzZXRfcHJlZml4X2NhY2hlKHNlbGYsICphcmdzLCAqKmt3YXJncykgLT4gYm9vbDoKICAgICAgICBpZiBzZWxmLl9zaG1fbW1hcCBpcyBub3QgTm9uZToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc2VsZi5fc2htX21tYXAuc2VlaygwKQogICAgICAgICAgICAgICAgc2VsZi5fc2htX21tYXAud3JpdGUoYiJceDAwIiAqIF9TSE1fQllURV9TSVpFKQogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9uOgogICAgICAgICAgICAgICAgcGFzcwogICAgICAgIGlmIHNlbGYuX2R5bmFtaWNfYnRfZW5hYmxlZDoKICAgICAgICAgICAgc2VsZi5tYXhfbnVtX3NjaGVkdWxlZF90b2tlbnMgPSBzZWxmLl9keW5hbWljX2J0X2xvd190b2tlbnMKICAgICAgICAgICAgc2VsZi5zY2hlZHVsZXJfY29uZmlnLm1heF9udW1fYmF0Y2hlZF90b2tlbnMgPSBzZWxmLl9keW5hbWljX2J0X2xvd190b2tlbnMKICAgICAgICByZXR1cm4gc3VwZXIoKS5yZXNldF9wcmVmaXhfY2FjaGUoKmFyZ3MsICoqa3dhcmdzKQoKICAgIGRlZiBfdXBkYXRlX2R5bmFtaWNfYnQoc2VsZikgLT4gTm9uZToKICAgICAgICBpZiBub3Qgc2VsZi5fZHluYW1pY19idF9lbmFibGVkOgogICAgICAgICAgICByZXR1cm4KICAgICAgICBsb2NhbF9hY3RpdmUgPSAoCiAgICAgICAgICAgIGxlbihzZWxmLnJ1bm5pbmcpICsgbGVuKHNlbGYud2FpdGluZykgKyBsZW4oc2VsZi5za2lwcGVkX3dhaXRpbmcpCiAgICAgICAgKQogICAgICAgIGNsdXN0ZXJfY29uYyA9IGxvY2FsX2FjdGl2ZQogICAgICAgIHN0YWdlX3BlYWsgPSBsb2NhbF9hY3RpdmUKICAgICAgICBpZiBzZWxmLl9zaG1fbW1hcCBpcyBub3QgTm9uZToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgc3RydWN0LnBhY2tfaW50bygiaSIsIHNlbGYuX3NobV9tbWFwLCBzZWxmLl9kcF9yYW5rICogNCwgbG9jYWxfYWN0aXZlKQogICAgICAgICAgICAgICAgdmFscyA9IHN0cnVjdC51bnBhY2tfZnJvbSgiMThpIiwgc2VsZi5fc2htX21tYXAsIDApCiAgICAgICAgICAgICAgICBjbHVzdGVyX2NvbmMgPSBtYXgoc3VtKHZhbHNbOjE2XSksIHZhbHNbMTZdLCBsb2NhbF9hY3RpdmUpCiAgICAgICAgICAgICAgICBzdGFnZV9wZWFrID0gbWF4KHZhbHNbMTddLCBjbHVzdGVyX2NvbmMpCiAgICAgICAgICAgICAgICBpZiBzdGFnZV9wZWFrID4gdmFsc1sxN106CiAgICAgICAgICAgICAgICAgICAgc3RydWN0LnBhY2tfaW50bygiaSIsIHNlbGYuX3NobV9tbWFwLCA2OCwgc3RhZ2VfcGVhaykKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICAgICAgICAgIHBhc3MKICAgICAgICB0YXJnZXRfYnQgPSAoCiAgICAgICAgICAgIHNlbGYuX2R5bmFtaWNfYnRfbG93X3Rva2VucwogICAgICAgICAgICBpZiBzdGFnZV9wZWFrIDw9IHNlbGYuX2R5bmFtaWNfYnRfbG93X2NvbmNfbWF4CiAgICAgICAgICAgIGVsc2Ugc2VsZi5fZHluYW1pY19idF9oaWdoX3Rva2VucwogICAgICAgICkKICAgICAgICBpZiBzZWxmLm1heF9udW1fc2NoZWR1bGVkX3Rva2VucyAhPSB0YXJnZXRfYnQ6CiAgICAgICAgICAgIGxvZ2dlci5pbmZvKAogICAgICAgICAgICAgICAgIlRwdURwU2NoZWR1bGVyIFtkcF9yYW5rPSVkXTogZHluYW1pYyBCVCBzd2l0Y2hlZCBtYXhfbnVtX3NjaGVkdWxlZF90b2tlbnMgJWQgLT4gJWQgKGNsdXN0ZXJfYWN0aXZlPSVkLCBzdGFnZV9wZWFrPSVkKSIsCiAgICAgICAgICAgICAgICBzZWxmLl9kcF9yYW5rLAogICAgICAgICAgICAgICAgc2VsZi5tYXhfbnVtX3NjaGVkdWxlZF90b2tlbnMsCiAgICAgICAgICAgICAgICB0YXJnZXRfYnQsCiAgICAgICAgICAgICAgICBjbHVzdGVyX2NvbmMsCiAgICAgICAgICAgICAgICBzdGFnZV9wZWFrLAogICAgICAgICAgICApCiAgICAgICAgICAgIHNlbGYubWF4X251bV9zY2hlZHVsZWRfdG9rZW5zID0gdGFyZ2V0X2J0CiAgICAgICAgICAgIHNlbGYuc2NoZWR1bGVyX2NvbmZpZy5tYXhfbnVtX2JhdGNoZWRfdG9rZW5zID0gdGFyZ2V0X2J0CgogICAgZGVmIF9zY2hlZHVsZV9wYXVzZWRfbmV3KHNlbGYpIC0+IFNjaGVkdWxlck91dHB1dDoKICAgICAgICAiIiJSdW4gb25lIHN0ZXAgd2hpbGUga2VlcGluZyBuZXdseSBwcmVmaWxscyBidWZmZXJlZC4iIiIKICAgICAgICBwcmV2X3BhdXNlX3N0YXRlID0gc2VsZi5fcGF1c2Vfc3RhdGUKICAgICAgICBpZiBwcmV2X3BhdXNlX3N0YXRlID09IFBhdXNlU3RhdGUuVU5QQVVTRUQ6CiAgICAgICAgICAgIHNlbGYuX3BhdXNlX3N0YXRlID0gUGF1c2VTdGF0ZS5QQVVTRURfTkVXCiAgICAgICAgdHJ5OgogICAgICAgICAgICByZXR1cm4gc3VwZXIoKS5zY2hlZHVsZShGYWxzZSkKICAgICAgICBmaW5hbGx5OgogICAgICAgICAgICBzZWxmLl9wYXVzZV9zdGF0ZSA9IHByZXZfcGF1c2Vfc3RhdGUKCiAgICBkZWYgc2NoZWR1bGUoc2VsZiwgdGhyb3R0bGVfcHJlZmlsbHM6IGJvb2wgPSBGYWxzZSkgLT4gU2NoZWR1bGVyT3V0cHV0OgogICAgICAgIHNlbGYuX3VwZGF0ZV9keW5hbWljX2J0KCkKICAgICAgICBzZWxmLnByZWZpbGxfY2FwYWNpdHlfYm91bmQgPSBUcnVlCiAgICAgICAgaWYgbm90IHRocm90dGxlX3ByZWZpbGxzOgogICAgICAgICAgICAjIE9wZW4gd2luZG93IGZvciBkcmFpbmluZyBhY2N1bXVsYXRlZCBwcmVmaWxscy4KICAgICAgICAgICAgc2VsZi5fZHJhaW5fcHJlZmlsbCA9IFRydWUKICAgICAgICBpZiBzZWxmLl9kcmFpbl9wcmVmaWxsIGFuZCBub3QgKHNlbGYud2FpdGluZyBvciBzZWxmLnNraXBwZWRfd2FpdGluZyk6CiAgICAgICAgICAgICMgSW4gZHJhaW4gbW9kZSB1bnRpbCBhbGwgYWNjdW11bGF0ZWQgcHJlZmlsbHMgYXJlIHNjaGVkdWxlZC4KICAgICAgICAgICAgc2VsZi5fZHJhaW5fcHJlZmlsbCA9IEZhbHNlCiAgICAgICAgaWYgc2VsZi5fZHJhaW5fcHJlZmlsbDoKICAgICAgICAgICAgcmV0dXJuIHN1cGVyKCkuc2NoZWR1bGUoRmFsc2UpCiAgICAgICAgcmV0dXJuIHNlbGYuX3NjaGVkdWxlX3BhdXNlZF9uZXcoKQo=").decode(); [open(p, "w").write(code) and py_compile.compile(p, doraise=True) and print("Patched dynamic BT in", p) for p in glob.glob("/workspace/**/tpu_scheduler.py", recursive=True) + glob.glob("/usr/local/**/tpu_scheduler.py", recursive=True)]' && echo "Verified tpu_scheduler py_compile"

    if [ -d "/vllm-cache/deepseek-ai/DeepSeek-V4-Pro-0813" ]; then
      echo "Found DeepSeek-V4 on shared PVC at /vllm-cache/deepseek-ai/DeepSeek-V4-Pro-0813"
      export MODEL_PATH="/vllm-cache/deepseek-ai/DeepSeek-V4-Pro-0813"
      export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro-0813}"
      export HF_HUB_OFFLINE=1
    elif [ -d "/vllm-cache/hub/models--deepseek-ai--DeepSeek-V4-Pro-0813" ]; then
      echo "Found DeepSeek-V4 on shared PVC at /vllm-cache/hub/models--deepseek-ai--DeepSeek-V4-Pro-0813"
      export MODEL_PATH="deepseek-ai/DeepSeek-V4-Pro-0813"
      export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro-0813}"
      export HF_HUB_OFFLINE=1
    elif [ -n "${GCS_CKPT_DIR:-}" ]; then
      LOCAL_CKPT_DIR="/root/.cache/huggingface/deepseek-ai/DeepSeek-V4-Pro-0813"
      echo "Syncing DeepSeek-V4 weights from GCS ($GCS_CKPT_DIR) to $LOCAL_CKPT_DIR..."
      mkdir -p "$LOCAL_CKPT_DIR"
      gcloud storage rsync -r "$GCS_CKPT_DIR" "$LOCAL_CKPT_DIR"
      echo "Evicting 831 GiB synced checkpoint files from Linux page cache (POSIX_FADV_DONTNEED)..."
      python3 -c 'import os, glob; [os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED) or os.close(fd) for f in glob.glob(os.path.join("/root/.cache/huggingface", "**", "*"), recursive=True) if os.path.isfile(f) for fd in [os.open(f, os.O_RDONLY)]]' 2>/dev/null || true
      export MODEL_PATH="$LOCAL_CKPT_DIR"
      export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Pro-0813}"
      export HF_HUB_OFFLINE=1
    fi

    if [ "$JOBSET_JOB_INDEX" = "0" ]; then
      ray stop --force 2>/dev/null || true
      echo "Starting Ray head on $HOSTNAME..."
      ray start --head --port=6379

      sleep 10
      echo "Waiting for Ray workers to join..."
      EXPECTED_TPU="@EXPECTED_TPU@"
      until ray status 2>/dev/null | grep -q "$EXPECTED_TPU"; do
        sleep 2
      done
      echo "Worker joined. Ray cluster status:"
      ray status

      if [ ! -f /tmp/chat_template.jinja ]; then
        echo "eyUgaWYgbm90IGFkZF9nZW5lcmF0aW9uX3Byb21wdCBpcyBkZWZpbmVkICV9CiAgeyUgc2V0IGFkZF9nZW5lcmF0aW9uX3Byb21wdCA9IGZhbHNlICV9CnslIGVuZGlmICV9CnslIGZvciBtZXNzYWdlIGluIG1lc3NhZ2VzICV9CiAgeyUgaWYgbWVzc2FnZVsicm9sZSJdID09ICJzeXN0ZW0iICV9CiAgICB7eyBtZXNzYWdlWyJjb250ZW50Il0gKyAiXG4iIH19CiAgeyUgZWxpZiBtZXNzYWdlWyJyb2xlIl0gPT0gInVzZXIiICV9CiAgICB7eyAiPO+9nFVzZXLvvZw+IiArIG1lc3NhZ2VbImNvbnRlbnQiXSB9fQogIHslIGVsaWYgbWVzc2FnZVsicm9sZSJdID09ICJhc3Npc3RhbnQiICV9CiAgICB7eyAiPO+9nEFzc2lzdGFudO+9nD4iICsgbWVzc2FnZVsiY29udGVudCJdICsgIjzvvZxlbmQgb2Ygc2VudGVuY2XvvZw+IiB9fQogIHslIGVuZGlmICV9CnslIGVuZGZvciAlfQp7JSBpZiBhZGRfZ2VuZXJhdGlvbl9wcm9tcHQgJX0KICB7eyAiPO+9nEFzc2lzdGFudO+9nD48dGhpbms+XG4iIH19CnslIGVuZGlmICV9Cg==" | base64 -d > /tmp/chat_template.jinja
      fi

      mkdir -p "${CDK_OUTPUT_DIR:-/tmp/cdk_output}"
      mkdir -p "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/xprof"
      mkdir -p "@XPROF_DIR@"
      if [ "@XPROF_DIR@" != "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/xprof" ]; then
        (
          { set +x; } 2>/dev/null
          while true; do
            sleep 10
            if [ -d "@XPROF_DIR@/plugins/profile" ]; then
              for ts_dir in "@XPROF_DIR@/plugins/profile"/*; do
                [ -d "$ts_dir" ] || continue
                ts_name="$(basename "$ts_dir")"
                dst_dir="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/xprof/plugins/profile/${ts_name}"
                mkdir -p "$dst_dir"
                for r in 0 1 2 3; do
                  for f in "$ts_dir"/rank${r}_*.xplane.pb "$ts_dir"/rank${r}_*.pt.trace.json.gz; do
                    [ -f "$f" ] || continue
                    bn="$(basename "$f")"
                    [ -f "$dst_dir/$bn" ] || cp -f "$f" "$dst_dir/$bn" 2>/dev/null || true
                  done
                done
                rm -f "$ts_dir"/rank[4-9]_*.xplane.pb "$ts_dir"/rank1[0-9]_*.xplane.pb "$ts_dir"/rank[4-9]_*.pt.trace.json.gz "$ts_dir"/rank1[0-9]_*.pt.trace.json.gz 2>/dev/null || true
              done
            fi
          done
        ) &
      fi

      # 6. Install Prefix-Aware Frontend DP Router (Session-ID affinity + 512-token block-radix prefix match + load balancing + DynamicBT shm)
      if [ "@ENABLE_PREFIX_ROUTING@" = "true" ]; then
        CORE_CLIENT_PATH="/usr/local/lib/python3.12/site-packages/vllm/v1/engine/core_client.py"
        if [ ! -f "$CORE_CLIENT_PATH" ]; then
          CORE_CLIENT_PATH=$(python3 -c "import importlib.util; print(importlib.util.find_spec('vllm.v1.engine.core_client').origin)" 2>/dev/null | tail -n 1)
        fi
        if [ -f "$CORE_CLIENT_PATH" ] && ! grep -q "_install_prefix_aware_routing" "$CORE_CLIENT_PATH"; then
          echo "Installing Prefix-Aware DP Routing Hook into $CORE_CLIENT_PATH..."
          echo "ZGVmIF9pbnN0YWxsX3ByZWZpeF9hd2FyZV9yb3V0aW5nKCk6CiAgICBpbXBvcnQgb3MsIHRpbWUsIGFycmF5LCB6bGliLCBtbWFwLCBzdHJ1Y3QKICAgIGlmIG9zLmVudmlyb24uZ2V0KCJFTkFCTEVfUFJFRklYX1JPVVRJTkciLCAiMSIpIG5vdCBpbiAoIjEiLCAidHJ1ZSIsICJUcnVlIik6CiAgICAgICAgcmV0dXJuCgogICAgZnJvbSB2bGxtLnYxLmVuZ2luZS5jb3JlX2NsaWVudCBpbXBvcnQgRFBMQkFzeW5jTVBDbGllbnQKICAgIG9yaWdfZ2V0ID0gRFBMQkFzeW5jTVBDbGllbnQuZ2V0X2NvcmVfZW5naW5lX2Zvcl9yZXF1ZXN0CgogICAgX2Jsb2NrX21hcCA9IHt9CiAgICBfZW5naW5lX2hhc2hlcyA9IHt9CiAgICBfcm91dGVfY291bnRzID0ge30KICAgIF9zZXNzaW9uX21hcCA9IHt9CiAgICBfZW5naW5lX2xhc3RfYWN0aXZlID0ge30KICAgIF9CTE9DS19TSVpFID0gNTEyCiAgICBfTUlOX01BVENIX0JMT0NLUyA9IDEyCiAgICBfSEJNX0NBUEFDSVRZX0JMT0NLUyA9IDk2MAoKICAgIF9TSE1fQ09OQ19QQVRIID0gIi9kZXYvc2htL2RzdjRfZHBfY29uY3VycmVuY3kuYmluIgogICAgX1NITV9CWVRFX1NJWkUgPSA3MgogICAgX2NvbmNfc2htID0gTm9uZQogICAgdHJ5OgogICAgICAgIF9mZCA9IG9zLm9wZW4oX1NITV9DT05DX1BBVEgsIG9zLk9fQ1JFQVQgfCBvcy5PX1JEV1IsIDBvNjY2KQogICAgICAgIHRyeToKICAgICAgICAgICAgaWYgb3MuZnN0YXQoX2ZkKS5zdF9zaXplIDwgX1NITV9CWVRFX1NJWkU6CiAgICAgICAgICAgICAgICBvcy5mdHJ1bmNhdGUoX2ZkLCBfU0hNX0JZVEVfU0laRSkKICAgICAgICAgICAgX2NvbmNfc2htID0gbW1hcC5tbWFwKF9mZCwgX1NITV9CWVRFX1NJWkUsIG1tYXAuTUFQX1NIQVJFRCwgbW1hcC5QUk9UX1JFQUQgfCBtbWFwLlBST1RfV1JJVEUpCiAgICAgICAgZmluYWxseToKICAgICAgICAgICAgb3MuY2xvc2UoX2ZkKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICBfY29uY19zaG0gPSBOb25lCgogICAgZGVmIF9jb21wdXRlX2Jsb2NrX2hhc2hlcyh0b2tlbnMpOgogICAgICAgIGlmIG5vdCB0b2tlbnMgb3IgbGVuKHRva2VucykgPCAoX0JMT0NLX1NJWkUgKiBfTUlOX01BVENIX0JMT0NLUyk6CiAgICAgICAgICAgIHJldHVybiBOb25lCiAgICAgICAgbl90b2sgPSBtaW4obGVuKHRva2VucyksIDc4NjQzMikKICAgICAgICBudW1fYmxvY2tzID0gbl90b2sgLy8gX0JMT0NLX1NJWkUKICAgICAgICBhcnIgPSBhcnJheS5hcnJheSgiSSIsIHRva2Vuc1s6bnVtX2Jsb2NrcyAqIF9CTE9DS19TSVpFXSkKICAgICAgICByYXcgPSBhcnIudG9ieXRlcygpCiAgICAgICAgYl9oYXNoZXMgPSBbXQogICAgICAgIGggPSAwCiAgICAgICAgYmxvY2tfYnl0ZXMgPSBfQkxPQ0tfU0laRSAqIDQKICAgICAgICBmb3IgaSBpbiByYW5nZShudW1fYmxvY2tzKToKICAgICAgICAgICAgaCA9IHpsaWIuY3JjMzIocmF3W2kgKiBibG9ja19ieXRlcyA6IChpICsgMSkgKiBibG9ja19ieXRlc10sIGgpCiAgICAgICAgICAgIGJfaGFzaGVzLmFwcGVuZChoKQogICAgICAgIHJldHVybiBiX2hhc2hlcwoKICAgIGRlZiBwcmVmaXhfYXdhcmVfZ2V0X2NvcmVfZW5naW5lKHNlbGYsIHJlcXVlc3QpOgogICAgICAgIG5vbmxvY2FsIF9ibG9ja19tYXAsIF9lbmdpbmVfaGFzaGVzLCBfcm91dGVfY291bnRzLCBfc2Vzc2lvbl9tYXAsIF9lbmdpbmVfbGFzdF9hY3RpdmUKICAgICAgICBpZiBnZXRhdHRyKHJlcXVlc3QsICJkYXRhX3BhcmFsbGVsX3JhbmsiLCBOb25lKSBpcyBub3QgTm9uZToKICAgICAgICAgICAgcmV0dXJuIG9yaWdfZ2V0KHNlbGYsIHJlcXVlc3QpCgogICAgICAgIG51bV9lbmdpbmVzID0gbGVuKHNlbGYuY29yZV9lbmdpbmVzKQogICAgICAgIGlmIG51bV9lbmdpbmVzIDw9IDE6CiAgICAgICAgICAgIHJldHVybiBvcmlnX2dldChzZWxmLCByZXF1ZXN0KQoKICAgICAgICBub3cgPSB0aW1lLm1vbm90b25pYygpCiAgICAgICAgc2lkID0gZ2V0YXR0cihyZXF1ZXN0LCAic2Vzc2lvbl9pZCIsIE5vbmUpCiAgICAgICAgdG9rZW5zID0gZ2V0YXR0cihyZXF1ZXN0LCAicHJvbXB0X3Rva2VuX2lkcyIsIE5vbmUpCiAgICAgICAgYl9oYXNoZXMgPSBfY29tcHV0ZV9ibG9ja19oYXNoZXModG9rZW5zKQogICAgICAgIHJlcV9ibG9ja3MgPSBsZW4oYl9oYXNoZXMpIGlmIGJfaGFzaGVzIGVsc2UgKChsZW4odG9rZW5zKSAvLyBfQkxPQ0tfU0laRSkgaWYgdG9rZW5zIGVsc2UgMCkKCiAgICAgICAgY2FuZGlkYXRlX2lkeCA9IE5vbmUKCiAgICAgICAgIyBTdHJhdGVneSAxOiBFeGFjdCBTZXNzaW9uIElEIGFmZmluaXR5IGZvciBhY3RpdmUgc2Vzc2lvbnMgaW4gY3VycmVudCBzdGFnZQogICAgICAgIGlmIHNpZCBhbmQgc2lkIGluIF9zZXNzaW9uX21hcDoKICAgICAgICAgICAgY2FuZGlkYXRlX2lkeCA9IF9zZXNzaW9uX21hcFtzaWRdICUgbnVtX2VuZ2luZXMKCiAgICAgICAgIyBTdHJhdGVneSAyOiBDdW11bGF0aXZlIEZpeGVkLUJsb2NrIFJhZGl4IFByZWZpeCBSb3V0aW5nIChjYXRjaGVzIHN1YmFnZW50IGZvcmtzICYgc2Vzc2lvbmxlc3MgdHVybnMpCiAgICAgICAgaWYgY2FuZGlkYXRlX2lkeCBpcyBOb25lIGFuZCBiX2hhc2hlczoKICAgICAgICAgICAgZm9yIGkgaW4gcmFuZ2UobGVuKGJfaGFzaGVzKSAtIDEsIF9NSU5fTUFUQ0hfQkxPQ0tTIC0gMiwgLTEpOgogICAgICAgICAgICAgICAgYmggPSBiX2hhc2hlc1tpXQogICAgICAgICAgICAgICAgaWYgYmggaW4gX2Jsb2NrX21hcDoKICAgICAgICAgICAgICAgICAgICBjYW5kaWRhdGVfaWR4ID0gX2Jsb2NrX21hcFtiaF0gJSBudW1fZW5naW5lcwogICAgICAgICAgICAgICAgICAgIGJyZWFrCgogICAgICAgICMgU3RyYXRlZ3kgMzogVG9rZW4tQ2FwYWNpdHkgJiBMb2FkLUF3Rm9yIG5ldyByb290IHNlc3Npb25zIC8gc25hcHNob3QgcHJpbWVycwogICAgICAgIGlmIGNhbmRpZGF0ZV9pZHggaXMgTm9uZToKICAgICAgICAgICAgZGVmIF9lbmdpbmVfa2V5KGkpOgogICAgICAgICAgICAgICAgZW5nID0gc2VsZi5jb3JlX2VuZ2luZXNbaV0KICAgICAgICAgICAgICAgIGluZmxpZ2h0ID0gc2VsZi5lbmdpbmVfaW5mbGlnaHRbZW5nXQogICAgICAgICAgICAgICAgd2FpdGluZywgcnVubmluZywga3ZfdXNhZ2UgPSBzZWxmLmxiX2VuZ2luZXNbaV0KICAgICAgICAgICAgICAgIGFjdGl2ZV9yZXFzID0gbWF4KHNlbGYuY2xpZW50X2NvdW50ICogaW5mbGlnaHQsIHdhaXRpbmcgKyBydW5uaW5nKQogICAgICAgICAgICAgICAgaXNfYnVzeSA9IDEgaWYgYWN0aXZlX3JlcXMgPiAwIGVsc2UgMAogICAgICAgICAgICAgICAgaGFzX2JlZW5fdXNlZCA9IDEgaWYgX3JvdXRlX2NvdW50cy5nZXQoaSwgMCkgPiAwIGVsc2UgMAogICAgICAgICAgICAgICAgYXNzaWduZWRfYmxvY2tzID0gbGVuKF9lbmdpbmVfaGFzaGVzLmdldChpLCAoKSkpCiAgICAgICAgICAgICAgICBlc3RfYmxvY2tzID0gbWF4KGFzc2lnbmVkX2Jsb2NrcywgaW50KGt2X3VzYWdlICogMTAwOCkpCiAgICAgICAgICAgICAgICBvdmVyZmxvd19ibG9ja3MgPSBtYXgoMCwgZXN0X2Jsb2NrcyArIHJlcV9ibG9ja3MgLSBfSEJNX0NBUEFDSVRZX0JMT0NLUykgaWYgZXN0X2Jsb2NrcyA+IDAgZWxzZSAwCiAgICAgICAgICAgICAgICBlZmZlY3RpdmVfbG9hZCA9IG1heChhY3RpdmVfcmVxcywgZXN0X2Jsb2NrcyAvLyAyNTYpCiAgICAgICAgICAgICAgICBsYXN0X3QgPSBfZW5naW5lX2xhc3RfYWN0aXZlLmdldChpLCAwLjApCiAgICAgICAgICAgICAgICByZXR1cm4gKGlzX2J1c3ksIGhhc19iZWVuX3VzZWQsIG92ZXJmbG93X2Jsb2NrcywgZWZmZWN0aXZlX2xvYWQsIGVzdF9ibG9ja3MsIGFjdGl2ZV9yZXFzLCBsYXN0X3QsIGkpCgogICAgICAgICAgICBjYW5kaWRhdGVfaWR4ID0gbWluKHJhbmdlKG51bV9lbmdpbmVzKSwga2V5PV9lbmdpbmVfa2V5KQogICAgICAgICAgICBfcm91dGVfY291bnRzW2NhbmRpZGF0ZV9pZHhdID0gX3JvdXRlX2NvdW50cy5nZXQoY2FuZGlkYXRlX2lkeCwgMCkgKyAxCgogICAgICAgIGlmIHNpZDoKICAgICAgICAgICAgX3Nlc3Npb25fbWFwW3NpZF0gPSBjYW5kaWRhdGVfaWR4CgogICAgICAgIF9lbmdpbmVfbGFzdF9hY3RpdmVbY2FuZGlkYXRlX2lkeF0gPSBub3cKICAgICAgICBpZiBiX2hhc2hlczoKICAgICAgICAgICAgZW5nX3NldCA9IF9lbmdpbmVfaGFzaGVzLnNldGRlZmF1bHQoY2FuZGlkYXRlX2lkeCwgc2V0KCkpCiAgICAgICAgICAgIGZvciBiaCBpbiBiX2hhc2hlczoKICAgICAgICAgICAgICAgIF9ibG9ja19tYXBbYmhdID0gY2FuZGlkYXRlX2lkeAogICAgICAgICAgICAgICAgZW5nX3NldC5hZGQoYmgpCgogICAgICAgIGNob3NlbiA9IHNlbGYuY29yZV9lbmdpbmVzW2NhbmRpZGF0ZV9pZHhdCiAgICAgICAgc2VsZi5sYl9lbmdpbmVzW2NhbmRpZGF0ZV9pZHhdWzBdICs9IHNlbGYuY2xpZW50X2NvdW50CiAgICAgICAgc2VsZi5yZXFzX2luX2ZsaWdodFtyZXF1ZXN0LnJlcXVlc3RfaWRdID0gY2hvc2VuCiAgICAgICAgc2VsZi5lbmdpbmVfaW5mbGlnaHRbY2hvc2VuXSArPSAxCiAgICAgICAgaWYgX2NvbmNfc2htIGlzIG5vdCBOb25lOgogICAgICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBpbmZsaWdodF9ub3cgPSBsZW4oc2VsZi5yZXFzX2luX2ZsaWdodCkKICAgICAgICAgICAgICAgIHN0cnVjdC5wYWNrX2ludG8oImkiLCBfY29uY19zaG0sIDY0LCBpbmZsaWdodF9ub3cpCiAgICAgICAgICAgICAgICBwcmV2X3BlYWsgPSBzdHJ1Y3QudW5wYWNrX2Zyb20oImkiLCBfY29uY19zaG0sIDY4KVswXQogICAgICAgICAgICAgICAgaWYgaW5mbGlnaHRfbm93ID4gcHJldl9wZWFrOgogICAgICAgICAgICAgICAgICAgIHN0cnVjdC5wYWNrX2ludG8oImkiLCBfY29uY19zaG0sIDY4LCBpbmZsaWdodF9ub3cpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgcmV0dXJuIGNob3NlbgogICAgZGVmIF9jbGVhcl9yb3V0ZXJfc3RhdGUoY2xpZW50KToKICAgICAgICBfYmxvY2tfbWFwLmNsZWFyKCkKICAgICAgICBfZW5naW5lX2hhc2hlcy5jbGVhcigpCiAgICAgICAgX3JvdXRlX2NvdW50cy5jbGVhcigpCiAgICAgICAgX3Nlc3Npb25fbWFwLmNsZWFyKCkKICAgICAgICBfZW5naW5lX2xhc3RfYWN0aXZlLmNsZWFyKCkKICAgICAgICBmb3IgbGIgaW4gZ2V0YXR0cihjbGllbnQsICJsYl9lbmdpbmVzIiwgKCkpOgogICAgICAgICAgICBsYlswXSA9IDAKICAgICAgICAgICAgbGJbMV0gPSAwCiAgICAgICAgICAgIGxiWzJdID0gMC4wCiAgICAgICAgaWYgaGFzYXR0cihjbGllbnQsICJyZXFzX2luX2ZsaWdodCIpOgogICAgICAgICAgICBjbGllbnQucmVxc19pbl9mbGlnaHQuY2xlYXIoKQogICAgICAgIGlmIGhhc2F0dHIoY2xpZW50LCAiZW5naW5lX2luZmxpZ2h0Iik6CiAgICAgICAgICAgIGNsaWVudC5lbmdpbmVfaW5mbGlnaHQuY2xlYXIoKQogICAgICAgIGlmIF9jb25jX3NobSBpcyBub3QgTm9uZToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgX2NvbmNfc2htLnNlZWsoMCkKICAgICAgICAgICAgICAgIF9jb25jX3NobS53cml0ZShiIlx4MDAiICogX1NITV9CWVRFX1NJWkUpCiAgICAgICAgICAgIGV4Y2VwdCBFeGNlcHRpb246CiAgICAgICAgICAgICAgICBwYXNzCiAgICAgICAgcHJpbnQoIltwcmVmaXhfcm91dGluZ10gQ2xlYXJlZCBEUCByb3V0ZXIgc3RhdGUgKF9ibG9ja19tYXAsIF9zZXNzaW9uX21hcCwgX2VuZ2luZV9oYXNoZXMsIGxiX2VuZ2luZXMsIF9jb25jX3NobSkiLCBmbHVzaD1UcnVlKQoKICAgIG9yaWdfcmVzZXRfYXN5bmMgPSBnZXRhdHRyKERQTEJBc3luY01QQ2xpZW50LCAicmVzZXRfcHJlZml4X2NhY2hlX2FzeW5jIiwgTm9uZSkKICAgIGlmIG9yaWdfcmVzZXRfYXN5bmMgaXMgbm90IE5vbmU6CiAgICAgICAgYXN5bmMgZGVmIHByZWZpeF9hd2FyZV9yZXNldF9wcmVmaXhfY2FjaGVfYXN5bmMoc2VsZiwgKmFyZ3MsICoqa3dhcmdzKToKICAgICAgICAgICAgX2NsZWFyX3JvdXRlcl9zdGF0ZShzZWxmKQogICAgICAgICAgICByZXR1cm4gYXdhaXQgb3JpZ19yZXNldF9hc3luYyhzZWxmLCAqYXJncywgKiprd2FyZ3MpCiAgICAgICAgRFBMQkFzeW5jTVBDbGllbnQucmVzZXRfcHJlZml4X2NhY2hlX2FzeW5jID0gcHJlZml4X2F3YXJlX3Jlc2V0X3ByZWZpeF9jYWNoZV9hc3luYwoKICAgIG9yaWdfcmVzZXQgPSBnZXRhdHRyKERQTEJBc3luY01QQ2xpZW50LCAicmVzZXRfcHJlZml4X2NhY2hlIiwgTm9uZSkKICAgIGlmIG9yaWdfcmVzZXQgaXMgbm90IE5vbmU6CiAgICAgICAgZGVmIHByZWZpeF9hd2FyZV9yZXNldF9wcmVmaXhfY2FjaGUoc2VsZiwgKmFyZ3MsICoqa3dhcmdzKToKICAgICAgICAgICAgX2NsZWFyX3JvdXRlcl9zdGF0ZShzZWxmKQogICAgICAgICAgICByZXR1cm4gb3JpZ19yZXNldChzZWxmLCAqYXJncywgKiprd2FyZ3MpCiAgICAgICAgRFBMQkFzeW5jTVBDbGllbnQucmVzZXRfcHJlZml4X2NhY2hlID0gcHJlZml4X2F3YXJlX3Jlc2V0X3ByZWZpeF9jYWNoZQoKICAgIERQTEJBc3luY01QQ2xpZW50LmdldF9jb3JlX2VuZ2luZV9mb3JfcmVxdWVzdCA9IHByZWZpeF9hd2FyZV9nZXRfY29yZV9lbmdpbmUKICAgIHByaW50KCJbcHJlZml4X3JvdXRpbmddIEVuYWJsZWQgVHJhamVjdG9yeSAmIFByZWZpeC1Bd2FyZSBEUCBSb3V0ZXIgKFNlc3Npb25BZmZpbml0eSArIEJsb2NrLVJhZGl4ICsgVG9rZW5DYXAgKyBBc3luY1Jlc2V0ICsgRHluYW1pY0JUKSIsIGZsdXNoPVRydWUpCgpfaW5zdGFsbF9wcmVmaXhfYXdhcmVfcm91dGluZygpCg==" | base64 -d >> "$CORE_CLIENT_PATH"
          python3 -m py_compile "$CORE_CLIENT_PATH" && echo "Prefix-Aware DP Routing hook compiled successfully"
        fi
      fi

      OFFLOAD_ARGS=()
      @OFFLOAD_SETUP@

      VLLM_ARGS=(
        @SERVE_ARGS@
      )

      echo "Starting vLLM serve for DeepSeek-V4 AgentX..."
      echo "trace-start-event: name=vllm-server ts=$(date +'%s.%N') pid=vllm tid=vllm"
      echo "trace-start-event: name=model-weight-load ts=$(date +'%s.%N') pid=vllm tid=vllm"

      (
        { set +x; } 2>/dev/null
        vllm_log="${CDK_OUTPUT_DIR:-/tmp/cdk_output}/vllm.log"
        weight_loaded=0
        for _ in {1..30}; do
          [ -f "$vllm_log" ] && break
          sleep 1
        done
        if [ -f "$vllm_log" ]; then
          stdbuf -oL tail -F -n +1 "$vllm_log" 2>/dev/null | while read -r line; do
            if [ $weight_loaded -eq 0 ] && [[ "$line" =~ "Loading weights took" || "$line" =~ "Model weights loaded" ]]; then
              weight_loaded=1
              echo "trace-end-event: name=model-weight-load ts=$(date +%s.%N) pid=vllm tid=vllm"
              echo "trace-start-event: name=server-upfront-compile ts=$(date +%s.%N) pid=vllm tid=vllm"
            elif [[ "$line" =~ "Application startup complete" || "$line" =~ "Uvicorn running on" ]]; then
              if [ $weight_loaded -eq 0 ]; then
                echo "trace-end-event: name=model-weight-load ts=$(date +%s.%N) pid=vllm tid=vllm"
              else
                echo "trace-end-event: name=server-upfront-compile ts=$(date +%s.%N) pid=vllm tid=vllm"
              fi
              break
            fi
          done
        fi
      ) &

      "${VLLM_ARGS[@]}" @EXTRA_ARGS@ 2>&1 | tee "${CDK_OUTPUT_DIR:-/tmp/cdk_output}/vllm.log"
      echo "trace-end-event: name=vllm-server ts=$(date +'%s.%N') pid=vllm tid=vllm"
    else
      mkdir -p "@XPROF_DIR@"
      if [ "@XPROF_DIR@" = "/dev/shm/xprof" ]; then
        (
          { set +x; } 2>/dev/null
          while true; do
            sleep 15
            rm -rf /dev/shm/xprof/plugins/profile/* 2>/dev/null || true
          done
        ) &
      fi
      ray stop --force 2>/dev/null || true
      echo "Starting Ray worker on $HOSTNAME..."

      while ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/${HEAD_ADDR}/6379" 2>/dev/null; do
        echo "Ray head at ${HEAD_ADDR} is not reachable yet, retrying in 2 seconds..."
        sleep 2
      done
      echo "Ray head at ${HEAD_ADDR} is reachable! Starting Ray worker..."

      sleep 5
      ray start --address=${HEAD_ADDR}:6379 --block
    fi
  |||,
  {
    HEAD_ADDR: cdk.host(serverJobName),
    EXPECTED_TPU: rayExpectedTpu,
    XPROF_DIR: xprofDir,
    ENABLE_PREFIX_ROUTING: if enablePrefixRouting then 'true' else 'false',
    OFFLOAD_SETUP: std.lstripChars(agentx.indent(offloadSetup, 6), ' '),
    SERVE_ARGS: std.lstripChars(agentx.indent(std.join('\n', serveArgs), 8), ' '),
    EXTRA_ARGS: extraVllmArgs,
    AIC_PATCH: attnIntraChipPatch,
  }
);

local hfTokenVolume = {
  name: 'hf-token-volume',
  secret: {
    secretName: 'vllm-hf-token',
    defaultMode: 420,
    optional: true,
    items: [{ key: 'HF_TOKEN', path: 'token' }],
  },
};

local serverVolumes = [
  hfTokenVolume,
  agentx.memoryVolume('dshm', '300Gi'),
] + (
  if sharedCachePvc != '' then [
    agentx.sharedCacheVolume(sharedCachePvc, name='vllm-shared-cache'),
  ] else [
    {
      name: 'huggingface-storage',
      ephemeral: {
        volumeClaimTemplate: {
          spec: {
            storageClassName: 'premium-rwo',
            accessModes: ['ReadWriteOnce'],
            resources: {
              requests: {
                storage: hfStorageSize,
              },
            },
          },
        },
      },
    },
  ]
);

local serverVolumeMounts = [
  agentx.hfTokenMount(),
  agentx.mount('dshm', '/dev/shm'),
] + (
  if sharedCachePvc != '' then [
    agentx.sharedCacheMount('vllm-shared-cache', '/vllm-cache'),
  ] else [
    agentx.mount('huggingface-storage', '/root/.cache/huggingface/'),
  ]
);

local serverJob = cdk.job(
  name=serverJobName,
  replicas=rayReplicas,
  podSpec=cdk.tpuNodeSelector(accelerator, topology)
          + (if nodePool != '' then { nodeSelector+: { 'cloud.google.com/gke-nodepool': nodePool } } else {}),
  volumes=serverVolumes,
  containers=[
    cdk.container(
      name='vllm-node',
      image=serverImage,
      script=serverScript,
      env={
        RECIPE_NAME: 'vllm-dsv4-agentx-v7x-16',
        MODEL_PATH: modelPath,
        SERVED_MODEL_NAME: servedModelName,
        GCS_CKPT_DIR: gcsCkptDir,
        HF_HUB_OFFLINE: cdk.var('HF_HUB_OFFLINE', '1'),
        JOBSET_JOB_INDEX: { valueFrom: { fieldRef: { fieldPath: "metadata.labels['jobset.sigs.k8s.io/job-index']" } } },
        POD_NAMESPACE: { valueFrom: { fieldRef: { fieldPath: 'metadata.namespace' } } },

        VLLM_DISABLE_COMPILE_CACHE: cdk.var('VLLM_DISABLE_COMPILE_CACHE', if dataParallelSize > 1 then '1' else '0'),
        JAX_COMPILER_CACHE_DIR: if dataParallelSize > 1 then '/tmp/jax_cache' else '/dev/shm/jax_cache',
        TORCHINDUCTOR_CACHE_DIR: if dataParallelSize > 1 then '/tmp/torchinductor' else '/dev/shm/torchinductor',
        VLLM_WORKER_MULTIPROC_METHOD: 'spawn',
        TORCHINDUCTOR_AUTOGRAD_CACHE: '0',
        VLLM_ENGINE_READY_TIMEOUT_S: cdk.numVar('VLLM_ENGINE_READY_TIMEOUT_S', 10800),
        VLLM_LOGGING_LEVEL: cdk.var('VLLM_LOGGING_LEVEL', 'INFO'),
        VLLM_TARGET_DEVICE: 'tpu',
        TPU_MULTIHOST_BACKEND: 'ray',
        RAY_ADDRESS: '127.0.0.1:6379',
        PJRT_DEVICE: 'TPU',
        TPU_PROCESS_ADDRESSES: tpuProcessAddresses,
        TPU_PROCESS_PORT: '8476',
        RAY_DEDUP_LOGS: '0',
        TPU_STDERR_LOG_LEVEL: '0',
        TPU_VMODULE: cdk.var('TPU_VMODULE', 'tpu_pjrt_client=1,pjrt_stream_executor_client=1,tpu_pjrt_compiler_utils=1'),

        MOE_REQUANTIZE_WEIGHT_DTYPE: moeRequantizeWeightDtype,
        MOE_REQUANTIZE_BLOCK_SIZE: moeRequantizeBlockSize,
        NEW_MODEL_DESIGN: '1',
        MODEL_IMPL_TYPE: 'vllm',
        VLLM_PLUGINS: 'torchtpu,torchtpu_layers',
        TORCH_DIST_TIMEOUT: '1200',
        VLLM_RPC_TIMEOUT: '1200000',
        VLLM_SHM_BROADCAST_TIMEOUT_S: '1200',
        VLLM_ENGINE_ITERATION_TIMEOUT_S: '1200',
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: cdk.var('VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS', '1800'),
        VLLM_USE_BREAKABLE_CUDAGRAPH: '0',
        VLLM_USE_AOT_COMPILE: '0',
        VLLM_ENFORCE_EAGER: '0',
        VLLM_NO_USAGE_STATS: '1',
        TPU_ROPE_CACHE_ROW_MAJOR: '1',
        TPU_MOE_HASH_TABLE_ROW_MAJOR: '1',
        // Fused MoE v2 (W4A8) + vllm-torchtpu PR #1557 (TPU_MOE_SKIP_PADDED_TOKENS=1 + MOE_FUSED_EP_V2_SHARDED_PLAN=1)
        USE_MOE_FUSED_EP_KERNEL: cdk.var('USE_MOE_FUSED_EP_KERNEL', '1'),
        MOE_FUSED_EP_ENABLE_W4A8: cdk.var('MOE_FUSED_EP_ENABLE_W4A8', '1'),
        MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE: cdk.var('MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE', '2'),
        MOE_FUSED_EP_V2_SHARDED_PLAN: cdk.var('MOE_FUSED_EP_V2_SHARDED_PLAN', '1'),
        TPU_MOE_SKIP_PADDED_TOKENS: cdk.var('TPU_MOE_SKIP_PADDED_TOKENS', '1'),
        TPU_DSV4_ATTN_INTRA_CHIP_TP: if attnIntraChipTp then '1' else '0',
        DP_SCHED_ENABLED: if dpSchedEnabled then '1' else '0',
        DYNAMIC_BT_ENABLED: if dynamicBtEnabled then '1' else '0',
        DYNAMIC_BT_LOW_CONC_MAX: '%d' % dynamicBtLowConcMax,
        DYNAMIC_BT_LOW_TOKENS: '%d' % dynamicBtLowTokens,
        DYNAMIC_BT_HIGH_TOKENS: '%d' % dynamicBtHighTokens,
        RAGGED_GATHER_REDUCE_VERSION: cdk.var('RAGGED_GATHER_REDUCE_VERSION', 'v3'),
        TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL: cdk.var('TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL', '30m'),
        LIBTPU_INIT_ARGS: cdk.var('LIBTPU_INIT_ARGS', '--xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_enable_async_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=262144 --deepsea_slice_builder_heartbeat_interval=30m'),
        VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: 'NEW_MODEL_DESIGN,MODEL_IMPL_TYPE,MOE_REQUANTIZE_WEIGHT_DTYPE,MOE_REQUANTIZE_BLOCK_SIZE,TPU_ROPE_CACHE_ROW_MAJOR,TPU_MOE_HASH_TABLE_ROW_MAJOR,TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE,TORCH_DIST_TIMEOUT,USE_MOE_FUSED_EP_KERNEL,MOE_FUSED_EP_ENABLE_W4A8,MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE,MOE_FUSED_EP_V2_SHARDED_PLAN,TPU_MOE_SKIP_PADDED_TOKENS,TPU_DSV4_ATTN_INTRA_CHIP_TP,DP_SCHED_ENABLED,DYNAMIC_BT_ENABLED,DYNAMIC_BT_LOW_CONC_MAX,DYNAMIC_BT_LOW_TOKENS,DYNAMIC_BT_HIGH_TOKENS,RAGGED_GATHER_REDUCE_VERSION,TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL,LIBTPU_INIT_ARGS,TPU_PJRT_EVENT_TRACE_ENABLE,USE_PHASED_PROFILER,VLLM_TPU_MOST_MODEL_LEN,VLLM_SERVER_DEV_MODE',
        [if mostModelLen > 0 then 'VLLM_TPU_MOST_MODEL_LEN']: '%d' % mostModelLen,
        // DSv4 multi-shape KV tensors require VLLM_TPU_BLOCK_MAJOR_KV=0 (PR #787)
        VLLM_TPU_BLOCK_MAJOR_KV: '0',
        [if kvOffload then 'RAIDEN_EXPECTED_WORKERS_TIMEOUT_S']: cdk.var('RAIDEN_EXPECTED_WORKERS_TIMEOUT_S', '900'),
        [if kvOffload then 'RAIDEN_NUMA_POLICY']: cdk.var('RAIDEN_NUMA_POLICY', 'preferred'),
        [if kvOffload then 'TPU_EVICT_WEIGHTS_PAGE_CACHE']: cdk.var('TPU_EVICT_WEIGHTS_PAGE_CACHE', 'true'),
        [if enableProfiling then 'TPU_PJRT_EVENT_TRACE_ENABLE']: '1',
        [if enableProfiling then 'USE_PHASED_PROFILER']: cdk.var('USE_PHASED_PROFILER', if dataParallelSize > 1 then 'false' else 'true'),
        TPU_PARALLEL_PRECOMPILE: cdk.var('TPU_PARALLEL_PRECOMPILE', '1'),
        VLLM_CACHE_ROOT: cdk.var('VLLM_CACHE_ROOT', if dataParallelSize > 1 then '/root/vllm-torchtpu-cache/cache' else '/dev/shm/vllm_cache'),
        VLLM_ALLOW_LONG_MAX_MODEL_LEN: '1',
        VLLM_SERVER_DEV_MODE: '1',
        RAY_memory_usage_threshold: '0.99',
        RAY_memory_monitor_refresh_ms: '0',
      } + cdk.jobSetNameEnv + agentx.megascaleOff + {
        TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL: cdk.var('TPU_SLICE_BUILDER_HEARTBEAT_INTERVAL', '30m'),
      },
      resources=cdk.tpuResources(chipsPerHost, cpu='32', memory='900Gi'),
      volumeMounts=serverVolumeMounts,
    ),
  ],
) + {
  template+: {
    metadata+: {
      annotations+: {
        'gke-gcsfuse/volumes': 'true',
      },
      labels+: {
        app: 'vllm-dsv4-agentx-v7x-16',
      },
    },
  },
};

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

local prelude = |||
  if [ -d "/root/aiperf-venv" ]; then
    echo "Activating /root/aiperf-venv..."
    source /root/aiperf-venv/bin/activate
  fi

  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

  if [ -d "/root/.cache/huggingface/token" ]; then
    echo "Warning: /root/.cache/huggingface/token is a directory, unmounting and removing..."
    umount "/root/.cache/huggingface/token" 2>/dev/null || true
    rm -rf "/root/.cache/huggingface/token" 2>/dev/null || true
  fi

  if [ -f "/root/cloud-devkit/InferenceX/benchmarks/runtime_settings.sh" ]; then
    source "/root/cloud-devkit/InferenceX/benchmarks/runtime_settings.sh"
  fi

  export INFMAX_CONTAINER_WORKSPACE="/root/cloud-devkit/InferenceX"
  export AIPERF_DIR="/root/cloud-devkit/InferenceX/utils/aiperf"
  export AIPERF_VENV="/root/aiperf-venv"
  export MODEL="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Pro-0813}"

  source "${INFMAX_CONTAINER_WORKSPACE}/benchmarks/benchmark_lib.sh" 2>/dev/null || echo "Loading benchmark_lib..."
  export AIPERF_HF_CLI="$(which hf 2>/dev/null || echo '/usr/local/bin/hf')"

  if [ -f /root/.cache/huggingface/token ]; then
    export HF_TOKEN="$(cat /root/.cache/huggingface/token | tr -d '\r\n')"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  fi

  if [[ "$MODEL" == gs://* ]] || [[ "$MODEL" == /vllm-cache* ]] || [[ -n "${GCS_MODEL_SRC:-}" ]]; then
    echo "Fast-copying tokenizer and config from GCS or shared cache..."
    mkdir -p /tmp/tokenizer
    GCS_SRC="${GCS_MODEL_SRC:-}"
    if [[ -n "$GCS_SRC" ]]; then
      gcloud storage cp "${GCS_SRC}/tokenizer*" /tmp/tokenizer/ || true
      gcloud storage cp "${GCS_SRC}/config.json" /tmp/tokenizer/ || true
      export MODEL="/tmp/tokenizer"
    elif [ -d "$MODEL" ]; then
      cp "$MODEL"/tokenizer* /tmp/tokenizer/ 2>/dev/null || true
      cp "$MODEL"/config.json /tmp/tokenizer/ 2>/dev/null || true
      export MODEL="/tmp/tokenizer"
    fi
  else
    echo "Pre-downloading tokenizer for ${MODEL}..."
    ${AIPERF_HF_CLI} download "${MODEL}" --include "tokenizer*" || true
    ${AIPERF_HF_CLI} download "${MODEL}" config.json || true
  fi
|||;

local datasetInit = agentx.fill(
  |||
    if [ -n "${GCS_DATASET_CACHE:-}" ]; then
      echo "Fast-syncing HuggingFace AgentX dataset cache from ${GCS_DATASET_CACHE}..."
      emit_trace_event start "dataset-sync"
      mkdir -p /root/.cache/huggingface
      gcloud storage cp -r "${GCS_DATASET_CACHE}/*" /root/.cache/huggingface/ 2>/dev/null || true
      emit_trace_event end "dataset-sync"
    fi

    export WEKA_LOADER_OVERRIDE="@WEKA_LOADER@"
    resolve_trace_source 2>/dev/null || true
  |||,
  {
    WEKA_LOADER: cdk.var('WEKA_LOADER', 'semianalysis_cc_traces_weka_062126'),
  }
);

local replayRewrites = [
  |||
    echo "Waiting for vLLM in-flight requests to drain before resetting prefix cache..."
    for _drain_i in $(seq 1 60); do
      _running=$(vllm_metric num_requests_running 2>/dev/null || echo 0)
      _waiting=$(vllm_metric num_requests_waiting 2>/dev/null || echo 0)
      _inflight=$(awk -v r="${_running:-0}" -v w="${_waiting:-0}" 'BEGIN { printf "%.0f", r + w }')
      if [ "${_inflight:-0}" -le 0 ]; then
        echo "vLLM server completely idle (running=${_running:-0}, waiting=${_waiting:-0})."
        break
      fi
      echo "  [drain ${_drain_i}/60] vLLM still processing running=${_running}, waiting=${_waiting}; sleeping 2s..."
      sleep 2
    done
    echo "Resetting vLLM prefix cache (HBM + external Raiden store + DP router) before starting concurrency benchmark..."
    curl -sf -X POST "${VLLM_URL}/reset_prefix_cache?reset_external=true" || true
    echo ""
    pc_h0=$(vllm_metric prefix_cache_hits_total)
    pc_q0=$(vllm_metric prefix_cache_queries_total)
    ext_h0=$(vllm_metric external_prefix_cache_hits_total)
    ext_q0=$(vllm_metric external_prefix_cache_queries_total)
    qt_s0=$(vllm_metric_ms request_queue_time_seconds_sum)
    qt_c0=$(vllm_metric request_queue_time_seconds_count)
    pt_s0=$(vllm_metric_ms request_prefill_time_seconds_sum)
    pt_c0=$(vllm_metric request_prefill_time_seconds_count)
    it_s0=$(vllm_metric_ms request_inference_time_seconds_sum)
    it_c0=$(vllm_metric request_inference_time_seconds_count)
    REPLAY_CMD="${REPLAY_CMD/--trajectory-start-min-ratio 0.25/--trajectory-start-min-ratio ${TRAJECTORY_START_MIN_RATIO:-0.25}}"
    REPLAY_CMD="${REPLAY_CMD/--trajectory-start-max-ratio 0.75/--trajectory-start-max-ratio ${TRAJECTORY_START_MAX_RATIO:-0.75}}"
    REPLAY_CMD="$(echo "$REPLAY_CMD" | sed -E "s/--warmup-requests-per-lane [0-9]+/--warmup-requests-per-lane ${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}/")"
    REPLAY_CMD="$(echo "$REPLAY_CMD" | sed -E "s/--benchmark-duration [0-9]+/--benchmark-duration ${DURATION}/")"
    if [ "${DURATION:-3600}" -lt 900 ] || [ "${AIPERF_UNSAFE_OVERRIDE:-false}" = "true" ]; then
      if [[ "$REPLAY_CMD" != *"--unsafe-override"* ]]; then
        REPLAY_CMD="$REPLAY_CMD --unsafe-override"
      fi
    fi
  |||,
];

local benchJob = agentx.benchJob(
  name=benchJobName,
  containerName='sidecar-bench',
  ttlSeconds=null,
  host=cdk.host(serverJobName),
  port=port,
  image=clientImage,
  concurrencies=concurrencies,
  resultFilePrefix='deepseek_v4_fp8_tpu_v7x_ix_conc_',
  benchName=benchName,
  tpuChips=tpuChips,
  prelude=prelude,
  datasetInit=datasetInit,
  exports={
    AIPERF_SERVER_URL: '${VLLM_URL}',
    AIPERF_UNSAFE_OVERRIDE: '${AIPERF_UNSAFE_OVERRIDE:-false}',
    MAX_MODEL_LEN: '${MAX_MODEL_LEN:-1048576}',
    KV_OFFLOADING: if kvOffload then 'dram' else 'none',
    PRECISION: 'fp8',
    FRAMEWORK: 'vllm',
    IS_MULTINODE: 'true',
    TP: tensorParallelSize,
    EP_SIZE: dataParallelSize,
    AIPERF_WARMUP_REQUESTS_PER_LANE: '${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}',
    AIPERF_TRACE_IDLE_GAP_CAP_SECONDS: '${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}',
  } + (if !kvOffload then {} else {
    KV_OFFLOAD_BACKEND: 'raiden',
    KV_OFFLOAD_BACKEND_METADATA: '{"name": "raiden"}',
    TOTAL_CPU_DRAM_GB: '${TOTAL_CPU_DRAM_GB:-400}',
  }),
  scrapePrefixCache=true,
  scrapeRequestTiming=true,
  enableXprof=enableXprof,
  profileDirs=[xprofDir],
  replayRewrites=replayRewrites,
  env={
    MAX_MODEL_LEN: maxModelLen,
    MODEL_PATH: modelPath,
    GCS_MODEL_SRC: gcsModelSrc,
    GCS_DATASET_CACHE: gcsDatasetCache,
    SERVED_MODEL_NAME: servedModelName,
    CLIENT_CONCURRENCIES: std.join(',', [std.toString(c) for c in concurrencies]),
    DURATION: duration,
    AIPERF_EXPERIMENTAL_FAST: if fastMode then '1' else '0',
    AIPERF_UNSAFE_OVERRIDE: cdk.var('AIPERF_UNSAFE_OVERRIDE', 'false'),
    AIPERF_WARMUP_REQUESTS_PER_LANE: warmupReqs,
    PROFILE_RANKS: cdk.var('PROFILE_RANKS', '0,1,2,3,4,5,6,7'),
    PROFILE_DURATION_SECONDS: cdk.numVar('PROFILE_DURATION_SECONDS', 10),
    TRAJECTORY_START_MIN_RATIO: cdk.var('TRAJECTORY_START_MIN_RATIO', '0.25'),
    TRAJECTORY_START_MAX_RATIO: cdk.var('TRAJECTORY_START_MAX_RATIO', '0.75'),
    AIPERF_TRACE_IDLE_GAP_CAP_SECONDS: cdk.numVar('AIPERF_TRACE_IDLE_GAP_CAP_SECONDS', 300),
  },
  waitTimeoutSeconds=cdk.numVar('WAIT_TIMEOUT_SECONDS', 10800),
  volumes=[agentx.memoryVolume('dshm'), hfTokenVolume],
  volumeMounts=[agentx.mount('dshm', '/dev/shm'), agentx.hfTokenMount()],
) + {
  template+: {
    spec+: {
      template+: {
        metadata+: {
          annotations+: {
            'gke-gcsfuse/volumes': 'true',
          },
        },
      },
    },
  },
};

local republishPayload = cdk.var('REPUBLISH_PAYLOAD', '');
local republishJob = cdk.job(
  name=benchJobName,
  containers=[
    cdk.container(
      name='sidecar-bench',
      image=clientImage,
      script='result-updater "$REPUBLISH_PAYLOAD"',
      env={ REPUBLISH_PAYLOAD: republishPayload },
      resources={ requests: { cpu: '1', memory: '2Gi' }, limits: { cpu: '2', memory: '4Gi' } },
    ),
  ],
);

if republishPayload != '' then
  cdk.jobSet(
    jobName,
    successPolicy=cdk.succeedWhen([benchJobName]),
    jobs=[republishJob],
  )
else
  cdk.jobSet(
    jobName,
    successPolicy=cdk.succeedWhen([benchJobName]),
    network={ enableDNSHostnames: true },
    jobs=[serverJob, benchJob],
  )
