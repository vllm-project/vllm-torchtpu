# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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

import functools
import math
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    TPU_ATTN_MOE_OVERLAP: bool = False
    TPU_K3_SP_PREFILL: bool = False
    TPU_DCP_MLA_TOPK_CAP: int = 0
    TPU_DCP_MLA_TOPK_CAP_CHUNK: int = 256
    TPU_DCP_LSE_MERGE_A2A: bool = False
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS: int = 64
    TPU_DCP_RS_OUT_PRECISION: bool = True
    TPU_MLA_SPARSE_SPLIT_Q: bool = False
    JAX_PLATFORMS: str = ""
    TPU_ACCELERATOR_TYPE: str | None = None
    TPU_NAME: str | None = None
    TPU_LIBRARY_PATH: str | None = None
    TPU_WORKER_ID: str | None = None
    TPU_MULTIHOST_BACKEND: str = ""
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    PYTHON_TRACER_LEVEL: int = 1
    HOST_TRACER_LEVEL: int = 2
    DEVICE_TRACER_LEVEL: int = 1
    TPU_TRACE_MODE: str = "TRACE_COMPUTE"
    TPU_NUM_SPARSE_CORES_TO_TRACE: int = 1
    TPU_NUM_SPARSE_CORE_TILES_TO_TRACE: int = 1
    RAY_USAGE_STATS_ENABLED: str = "0"
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "shm"
    ENABLE_QUANTIZED_MATMUL_KERNEL: bool = False
    REQUANTIZE_BLOCK_SIZE: int | None = None
    REQUANTIZE_WEIGHT_DTYPE: str = "float8_e4m3fn"
    TPU_FP8_LINEAR_W8A16_MAX_ROWS: int = 0
    MOE_REQUANTIZE_WEIGHT_DTYPE: str = "float8_e4m3fn"
    MOE_REQUANTIZE_BLOCK_SIZE: int | None = None
    TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL: bool | None = None
    TPU_USE_RAIDEN_KV_CACHE_MANAGER: bool = False
    TPU_RAIDEN_QWEN35_ADMISSION: bool = False
    TPU_RAIDEN_KIMIK3_ADMISSION: bool = False
    TPU_RAIDEN_GLM_ADMISSION: bool = False
    TPU_RAIDEN_DSV4_ADMISSION: bool = False
    TPU_KV_RESHARD_TRANSPORT: str = "zmq"
    TPU_RAIDEN_CONTROLLER_ADDRESS: str = ""
    TPU_RAIDEN_JOB_NAME: str = ""
    TPU_RAIDEN_ENGINE_ID: str = "0"
    TPU_RAIDEN_TRANSFER_PARALLELISM: int = 8
    TPU_RAIDEN_DST_SHARDS: int | None = None
    TPU_RAIDEN_POOL_TAGS_PER_LAYER: bool = False
    TPU_RAIDEN_MAX_TRANSFER_TOKENS: int | None = None
    TPU_RAIDEN_RESHARD_IMPL: str = "controller"
    TPU_RAIDEN_ADVERTISE_HOST: str = ""
    TPU_RAIDEN_RESHARD_PORT_BASE: int = 27000
    TPU_RAIDEN_STORE_DISPATCH_PORT_BASE: int = 27100
    TPU_RAIDEN_PREFIX_AWARE_LOAD: bool = False
    USE_MOE_SPARSE_CORE: bool = True
    ONEHOT_MOE_PERMUTE_THRESHOLD: int | None = None
    TPU_MOE_SEQUENCE_PARALLEL: bool = False
    TPU_GMM_FP8_W8A8: bool = False
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    TPU_MOE_DENSE_EXPERT_THRESHOLD: int = 0
    TPU_MINIMAX_M3_FP8_LINEARS: bool = True
    TPU_MINIMAX_M3_FP8_W8A8_MAX_ROWS: int = 0
    TPU_MINIMAX_M3_FP8_W8A8_MIN_ROWS: int = 0
    TPU_MINIMAX_M3_DSPARK_ONE_KV_WRITE: bool = False
    TPU_MINIMAX_M3_DSPARK_CONTEXT_KV_CAT: bool = False
    TPU_DRAFT_DCP_PAIR_MERGE: bool = False
    TPU_DRAFT_DCP_PAIR_MERGE_BLOCK: bool = False
    TPU_MINIMAX_M3_FP8_SPLIT_MAX_ROWS: int = 0
    TPU_MINIMAX_M3_FP8_SPLIT_MIN_ROWS: int = 0
    TPU_MINIMAX_M3_SP_MODE: str = "off"
    TPU_MINIMAX_M3_SP_MIN_TOKENS: int = 4096
    TPU_MINIMAX_M3_SP_FP8_GATHER: bool = False
    TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER: bool = False
    TPU_MINIMAX_M3_SP_FUSED_NORM: bool = False
    TPU_MINIMAX_M3_SP_E4M3_QKV: bool = False
    TPU_MINIMAX_M3_SP_E4M3_SHARED: bool = False
    TPU_MINIMAX_M3_SP_A2A_OPROJ: str = "off"
    TPU_MINIMAX_M3_SP_A2A_E4M3: bool = False
    TPU_MINIMAX_M3_SP_QRS: bool = False
    TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS: int = 2048
    TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS: int = 0
    TPU_MINIMAX_M3_FP8_CAST_SKIP: str = "shared_experts.gate_up_proj"
    TPU_MINIMAX_M3_FP8_CAST_IN_TORCH: bool = False
    TPU_MINIMAX_M3_FP8_CAST_NORM: bool = False
    TPU_MINIMAX_M3_SP_HIER_RS: bool = False
    TPU_MINIMAX_M3_SP_HIER_RS_MIN_TOKENS: int = 8192
    TPU_MINIMAX_M3_SHARED_IN_COMBINE: bool = False
    TPU_MINIMAX_M3_SHARED_IN_COMBINE_MIN_TOKENS: int = 4096
    TPU_MINIMAX_M3_SP_FFN_PIECES: int = 1
    TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS: int = 4096
    TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES: bool = True
    TPU_MINIMAX_M3_SP_DENSE_FFN_PER_PIECE: bool = False
    TPU_MINIMAX_M3_SP_ATTN_PIECES: bool = False
    TPU_MOE_DENSE_FP4_MAX_TOKENS: int = 16
    TPU_MOE_DENSE_FP4_SPARSE_MAX_TOKENS: int = 0
    FUSED_MOE_DECODE_KERNEL: bool = False
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: int = 256
    TPU_MOE_SINGLE_SORT: bool = True
    TPU_MOE_COUNTING_SORT: bool = True
    TPU_MOE_VALID_COUNT_FROM_SORT: bool = False
    TPU_MOE_ONEHOT_COMBINE_FLAT: bool = False
    TPU_MOE_ONEHOT_COMBINE_FLAT_MAX_ROUTES: int = 0
    TPU_MOE_BOUNDED_ONEHOT: bool = True
    USE_MOE_TP_DECODE_KERNEL: bool = False
    TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES: int = 0
    TPU_TP_PALLAS_SMALL_ALL_GATHER: bool = False
    TPU_TP_PALLAS_ALL_REDUCE_DIRECT_ROWS: int = 0
    TPU_TP_AR_QDQ_ROWS: int = 0
    TPU_TP_QAR_MIN_TOKENS: int = 0
    TPU_TP_AR_QDQ_MIN_TOKENS: int = 0
    TPU_TP_PALLAS_ALL_REDUCE_ONESHOT_ROWS: int = 0
    TPU_TP_PALLAS_ALL_REDUCE_HIER_ROWS: int = 0
    TPU_TP_PALLAS_ALL_REDUCE_HIER_MODE: str = "ag"
    TPU_TP_PALLAS_ALL_REDUCE_HIER_MIN_ROWS: int = 0
    TPU_MOE_TC_COMBINE: bool = False
    TPU_MOE_TC_COMBINE_PIECES: int = 1
    RAGGED_GATHER_REDUCE_VERSION: str = "v2"
    TPU_MOE_TC_COMBINE_MIN_TOKENS: int = 0
    TPU_MOE_V3_PAYLOAD_SORT: bool = False
    TPU_MOE_FLAT_ROUTING_MIN_TOKENS: int = 0
    TPU_MOE_ROUTE_TRACE: int = 0
    TPU_MOE_ROUTE_TRACE_EVERY: int = 8192
    VLLM_TPU_MOE_REDUNDANT_EXPERTS: int = 0
    TPU_MOE_REPLICA_ASSIGN_REFINE: int = 0
    TPU_MOE_REPLICA_ASSIGN_FAST: bool = False
    TPU_MOE_NVFP4_SKIP_PADDED: bool = False
    TPU_MINIMAX_M3_INDEXER_DECODE_OPT: str = ""
    TPU_MINIMAX_M3_INDEXER_PREFILL_OPT: str = ""
    TPU_MOE_SKIP_PADDED_MIN_ROUTES: int = 0
    VLLM_TPU_MOE_REDUNDANT_PLACEMENT: str = ""
    USE_PHASED_PROFILER: bool = False
    TPU_KERNEL_ITER_MODE: bool = False
    TPU_KERNEL_RELOAD_MODULES: str = ""
    TPU_GREEDY_TOP_TOKENS: bool = False
    DP_SCHED_ENABLED: bool = False
    MLA_XPOSE_N_TILE_SIZE: int = 160
    MIXED_Q_SPLIT: int | None = None
    MIXED_NUM_QUERIES_PER_BLOCK: int | None = None
    MIXED_NUM_KV_PAGES_PER_BLOCK: int | None = None
    TPU_MLA_MASKED_DENSE_ENABLED: bool = False
    TPU_MLA_MASKED_DENSE_ANALYTIC_MAX_KV_LEN: int | None = None
    TPU_MLA_MASKED_DENSE_MAX_KV_LEN: int | None = None
    VLLM_TPU_MOST_MODEL_LEN: int | None = None
    TPU_GDN_CONV_QK_PAIR_LAYOUT: bool = False
    TPU_GDN_QKVZ_KERNEL_LAYOUT: bool = False
    TPU_ENABLE_GDN_DYNAMIC_TILING: bool = False
    TPU_MOE_SKIP_PADDED_TOKENS: bool = False
    TPU_MOE_HIERARCHICAL_EP: bool = False
    TPU_MOE_ROUTER_TOPK: str = "rowmax"
    TPU_MOE_ROUTER_TOPK_VALUES: bool = False
    TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS: int = 0
    TPU_ROUTER_TOPK_FUSED_BIAS_MIN_ROWS: int = 0
    TPU_MINIMAX_M3_QK_NORM_ROPE_BLOCK_TOKENS: int = 0
    TPU_MOE_TOPK_WEIGHTS_FLOAT32: bool = False
    USE_MOE_FUSED_EP_KERNEL: bool = False
    MOE_FUSED_EP_KERNEL_IMPL: str | None = None
    MOE_FUSED_EP_ENABLE_W4A8: bool = False
    MOE_FUSED_EP_KERNEL_MIN_TOKENS: int = 1024
    MOE_FUSED_EP_V2_SHARDED_PLAN: bool = False
    MOE_FUSED_EP_SORTED_GATHER: bool = False
    MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE: int = 0
    TPU_LATENT_PROJ_INTRA_CHIP_TP: bool = False
    TPU_DSV4_ATTN_INTRA_CHIP_TP: bool = False
    TPU_TOKEN_BUCKET_EXTRA: list[int] = []
    TPU_TOKEN_BUCKET_LINEAR: bool = False
    TPU_PP_DYNAMIC_CHUNKS: bool = True
    TPU_PP_CHUNK_SLACK: float = 0.1
    TPU_PP_LAZY_PUSH: bool = False
    TPU_PP_HANDOFF: str = "wave"
    TPU_PP_RING_DEPTH: int = 2
    TPU_ROPE_CACHE_TRUNCATE: bool = False
    TPU_ROPE_CACHE_ROW_MAJOR: bool = False
    TPU_MOE_HASH_TABLE_ROW_MAJOR: bool = False
    TPU_PARALLEL_PRECOMPILE: bool = False
    TPU_PRECOMPILE_BIGGEST_FIRST: bool = False
    TPU_PRECOMPILE_PEER_REPLAY: bool = False
    TPU_PRECOMPILE_PEER_WAIT_S: float = 600.0
    TPU_PRECOMPILE_TRACE_FIRST: bool = False
    TPU_STREAMIDX_CHUNK_TOKENS: int | None = None
    TPU_SPEC_STATIC_VERIFY: bool = True
    TPU_SPEC_SYNTHETIC_ONE_ROW_VERIFY: bool = True
    TPU_DRAFT_BUCKET_STEP: int = 32
    TPU_SPEC_SYNTHETIC_ROW_SPLIT: bool = True
    TPU_SPEC_GLUE_FUSION: bool = False
    TPU_SPEC_GATHER_IN_BOUNDS: bool = False
    TPU_SPEC_SYNTHETIC_INGRAPH_RNG: bool = False
    TPU_SPEC_SYNTHETIC_HEAD_FUSION: bool = False
    TPU_SPEC_SUBST_EXTRACT_FUSION: bool = False
    TPU_HOST_STAGE_TIMING_HOST_ONLY: bool = False
    TPU_HOST_TOKEN_PADDING_REUSE: bool = False
    TPU_HOST_SAMPLING_PARAMS_REUSE: bool = False
    TPU_HOST_SKIP_GUARD_EVAL_AFTER_STEPS: int = 0
    TPU_HOST_KEEP_INPUT_MUTATIONS: int = 0
    TPU_DSPARK_MARKOV_KERNEL: bool = False
    TPU_DCP_INDEXER_A2A_MERGE: bool = False
    TPU_DCP_INDEXER_A2A_MIN_TOKENS: int = 64
    TPU_STREAMIDX_SMALL_BUCKET_BQ: int | None = None
    TPU_STREAMIDX_SMALL_BUCKET_BKVP: int | None = None
    TPU_STREAMIDX_SMALL_BUCKET_TOKENS: int = 64
    VLLM_TPU_INDEXER_KV_CACHE_MODE: str = "sal"
    TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER: bool = True
    TPU_SPARSE_MLA_CHUNK_TOKENS: int | None = None
    TPU_MOE_COLLECTION_CHUNK_SIZE: int = 0
    TPU_MOE_FP8_ALL_GATHER: bool = False
    TPU_PCP_TOPOLOGY_AWARE_MESH: bool = True
    USE_RPA_PIPELINED_DMA_STAGING: bool = False
    USE_PIPELINED_DMA_STAGING: bool = False
    VLLM_TPU_BLOCK_MAJOR_KV: bool = False
    TPU_EVICT_WEIGHTS_PAGE_CACHE: bool = False
    TPU_SPARSE_MLA_NOPE_LAYOUT: str = "sparsecore"
    TPU_SPARSE_MLA_ROPE_LAYOUT: str = "sparsecore"
    TPU_KV_TRANSFER_PORT: str = "9100"
    TPU_SIDE_CHANNEL_PORT: str = "9600"
    TPU_NODE_ID: int = 0
    TPU_KV_TRANSFER_CHANNEL_NUMBER: int = 0
    TPU_KV_COORD_EXECUTOR_MAX_WORKERS: int = 0
    TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS: int = 0
    TPU_KV_STAGE_WAITER_POOL_SIZE: int = 0
    TPU_P2P_WAIT_PULL_TIMEOUT: int = 120
    TPU_RAIDEN_STAGE3_STATUS_PROBE_S: float = 1.0
    TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT: bool = True
    TPU_RAIDEN_STAGE3_REGISTER_AHEAD: bool = True
    TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S: float = 30.0
    TPU_RAIDEN_TEST_REGISTRATION_DELAY_S: float = 0.0
    TPU_KV_STAGE_WAIT_TIMEOUT_SECS: float = 30.0
    TPU_KV_SHM_POOL_GB: float = 128.0
    TPU_KV_TRANSFER_NAMESPACE: str = ""
    TPU_IPC_SOCKET_DIR: str = "/tmp"
    TPU_KV_WARMUP_ENABLED: bool = True
    TPU_KV_LATENCY_LOG_INTERVAL: float = 30.0
    TPU_KV_PIN_SHM: bool = False
    TPU_USE_RAIDEN_CONNECTOR: bool = False
    TPU_RAIDEN_TRANSFER_NUM_SLOTS: int = 0
    TPU_RAIDEN_POOL_STAGING_LEASES: int = 8
    TPU_RAIDEN_INLINE_LOAD: bool = False
    QUANTIZE_ON_LOAD_PREFIXES: list[str] = []
    KDA_MANUAL_STATE_DMA: bool | None = None
    KDA_MANUAL_H0_DMA: bool | None = None
    KDA_MANUAL_HT_DMA: bool | None = None
    KDA_OVERLAP_H0_DMA: bool = True
    KDA_OVERLAP_HT_DMA: bool = True
    KDA_PACK_HEAD_INV: bool = True
    KDA_PACKED_METADATA: bool = True
    KDA_FWD_MB: int | None = None
    TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT: str = ""
    SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES: int | None = None
    SPEC_WARMUP: bool = True
    RAIDEN_DISABLE_SINGLETON_WORKER: bool = True
    RAIDEN_SHM_KEY: str = ""
    RAIDEN_STORAGE_PATH: str = ""
    RAIDEN_STORAGE_TYPE: str = ""
    RAIDEN_STORAGE_CAPACITY_GB: str = ""
    RAIDEN_STORAGE_PROPERTIES: str = ""
    EXPERIMENTAL_RAIDEN_STORAGE_SKIP_PARALLELISM_CHECK: bool = False
    VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S: float = 30.0
    VLLM_TPU_OFFLOAD_SAVE_RETRIES: int = 1
    VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_MIB: int = 128
    VLLM_TPU_OFFLOAD_MAX_INFLIGHT_STORE_MIB: int = 64
    VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS: int = 0
    VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH: bool = False
    TPU_RAIDEN_LRU_REFRESH: int = 0
    TPU_HOST_HEAP_REPORT: int = 0
    TPU_HOST_HEAP_USAGE_AFTER_S: int = 900
    TPU_RELEASE_JAX_CACHES: bool = False
    TPU_SPEC_PRECOMPILE_SYNTHETIC_VERIFY_ONLY: bool = False
    TPU_SPEC_PRECOMPILE_SAMPLER_VARIANTS: str = ""
    TPU_RAIDEN_OFFLOAD_STATS: bool = False
    TPU_RAIDEN_FLUSH_REAL_REUSE: bool = False
    TPU_RAIDEN_LOSS_TRACE: bool = False
    TPU_RAIDEN_WB_DROP_COVERED: bool = False
    VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: bool = False
    VLLM_TORCHTPU_IPC_KEY: str = ""
    TPU_SHARDED_LOAD_SYNC_EVERY: int = 512
    VLLM_TPU_MXFP4_DEVICE_DEQUANT: bool = True
    VLLM_TPU_DEBUG_PCP_LAYOUT: bool = False
    TPU_LOCAL_RANK_OFFSET: int = 0
    VLLM_TPU_DEVICE_ORDER: str = ""
    DEBUG_TPU_LOCAL_RANK_OFFSET: int = 0
    TORCH_TPU_BASE_PORT: int = 8070
    TORCH_TPU_MP_RENDEZVOUS_PORT: int | None = None
    VLLM_TPU_FAST_TOKEN_SUBSTITUTION: bool = False
    VLLM_TPU_FUSED_DRAFT_LOOP: bool = False
    TPU_SAMPLER_TOPK_CANDIDATES: int = 0
    TPU_SAMPLER_TOPP_CANDIDATES: str = ""
    TPU_LM_HEAD_E4M3: bool = False
    TPU_LM_HEAD_E4M3_SPLIT_MAX_ROWS: int = 0
    TPU_LM_HEAD_E4M3_DUMP: int = 0
    MM_ENCODER_FRAME_CHUNK_PATCH_SIZE: int = 0
    VLLM_TPU_STATIC_POOLER: bool = True
    VLLM_TPU_POOLING_KV_CACHE_TOKENS: int = -1
    VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS: int = 8
    VLLM_TPU_SRPT_ATTN_WORK_CAP: int = 0
    VLLM_TPU_SRPT_ATTN_TILE_TOKENS: int = 32
    VLLM_TPU_PREFILL_SPACING_FRAC: float = 0.5
    VLLM_TPU_EVICT_DECODE_FIRST: bool = False
    VLLM_TPU_EVICT_PROTECT_HITS: int = 0
    VLLM_TPU_EVICT_PROTECTED_FRAC: float = 0.3
    VLLM_TPU_PREFILL_SPACING_MIN_TOKENS: int = 8192
    VLLM_TPU_PREFILL_TPOT_TARGET_MS: float = 0.0
    VLLM_TPU_PREFILL_TPOT_MAX_HOLD_S: float = 10.0
    VLLM_TPU_PREFILL_TPOT_PRICED: bool = False
    TPU_FP8_LM_HEAD: bool = False
    TPU_MINIMAX_M3_FP8_DRAFT_LINEARS: bool = False
    VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON: float = 1.0
    VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S: float = 0.0
    VLLM_TPU_PREFILL_SRPT: bool = False
    VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S: float = 256.0
    VLLM_TPU_PREFILL_DECODE_CADENCE: int = 0
    VLLM_TPU_PREFILL_CADENCE_MAX_WAIT_S: float = 10.0
    VLLM_TPU_PREFILL_COALESCE_TOKENS: int = 0
    VLLM_TPU_PREFILL_COALESCE_MAX_WAIT_S: float = 0.5
    VLLM_TPU_SIBLING_HOLD: bool = False
    VLLM_TPU_SIBLING_HOLD_MAX_S: float = 4.0
    TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS: int = 0
    TPU_MINIMAX_M3_DCP_QCAND_GATHER_MAX_TOKENS: int = 0
    TPU_MINIMAX_M3_DCP_FUSED_DECODE_MAX_TOKENS: int = 0
    TPU_MINIMAX_M3_MSA_WRITE_RUNS_IN_KERNEL_MAX_TOKENS: int = 0
    TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MAX_TOKENS: int = 0
    TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MIN_TOKENS: int = 0
    TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE: str = ""
    TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT: bool = False
    TPU_MINIMAX_M3_DCP_LSE_PLANES: bool = False
    TPU_MINIMAX_M3_NORM_WEIGHT_HOIST: bool = False
    TPU_MINIMAX_M3_DCP_MERGE_KERNEL_MIN_CANDIDATES: int = 64
    TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG: str = ""
    TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR: int = 1
    TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT: bool = False
    TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR: bool = False
    TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER: str = ""
    TPU_MINIMAX_M3_DCP_HEADS_MERGE_A2A_MAX_TOKENS: int = 0
    TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A: bool = False
    TPU_MINIMAX_M3_PT16: bool = False
    TPU_MINIMAX_M3_INDEX_FOLD_OPT: str = ""
    TPU_MINIMAX_M3_DCP_EXCHANGE_IN_FLIGHT: int = 8
    TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT: bool = False
    TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC: bool = False
    TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES: bool = False
    TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE: str = ""
    TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT: str = ""
    TPU_MINIMAX_M3_MSA_PAGE_CACHE_WIDTHS: str = ""
    TPU_MINIMAX_M3_MSA_PREFILL_PRECISION: str = ""
    TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE: str = ""
    TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT: str = ""
    TPU_MINIMAX_M3_MSA_MAX_FREE: bool = False
    TPU_MINIMAX_M3_MSA_MXU_MASK: bool = False
    TPU_MINIMAX_M3_KV4_SHARED_SCALES: bool = False
    TPU_SPEC_VERIFY_COARSE_PAD: bool = False
    TPU_SAMPLING_METADATA_FIXED_SHAPE_COPIES: bool = False
    TPU_SKIP_DEVICE_SAMPLING_METADATA: bool = False
    TPU_REUSE_UNCHANGED_BLOCK_TABLES: bool = False
    TPU_COMPACT_NEW_REQ_TOKENS: bool = False
    TPU_COMPACT_NEW_REQ_TOKENS_MIN: int = 4096
    TPU_SAMPLER_TOP_P_KERNEL: bool = False
    TPU_MOE_ONEHOT_BOUNDED_SUBLANE_TOKENS: bool = False
    TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q: int = 0


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """
    Create a lambda that validates environment variable against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict
    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}. "
                f"Valid options: {actual_choices}."
            )

        return value

    return _get_validated_env


_TRUE_STRINGS = ("true", "1", "yes", "on")
_FALSE_STRINGS = ("false", "0", "no", "off")
DEFAULT_INDEXER_KV_CACHE_MODE = "sal"


def env_bool(env_name: str, default: bool = False) -> Callable[[], bool]:
    """
    Accepts numeric strings ("0", "1"), boolean strings ("true", "false"),
    and the yes/no and on/off spellings, in any case. Anything else raises:
    a misspelled flag is a launch-script bug, and silently falling back to
    the default hides it until someone wonders why the knob did nothing.
    """

    def _get_bool_env() -> bool:
        value = os.getenv(env_name)
        if value is None or value.strip() == "":
            return default

        value_lower = value.strip().lower()
        if value_lower in _TRUE_STRINGS:
            return True
        if value_lower in _FALSE_STRINGS:
            return False
        raise ValueError(
            f"Invalid boolean value '{value}' for {env_name}. "
            f"Valid options: {_TRUE_STRINGS + _FALSE_STRINGS} "
            f"(any case)."
        )

    return _get_bool_env


def env_int(env_name: str, default: int) -> Callable[[], int]:
    """Integer knob with a default. An unparseable value raises rather than
    falling back, for the same reason as ``env_bool``."""

    def _get_int_env() -> int:
        value = os.getenv(env_name)
        if value is None or value.strip() == "":
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}: must be an integer."
            ) from exc

    return _get_int_env


def env_nonnegative_int(env_name: str, default: int) -> Callable[[], int]:
    """Integer knob that must not be negative.

    Pool sizes and slot counts use this. A negative value is rejected rather
    than clamped: it reaches ``ThreadPoolExecutor(max_workers=...)`` and
    similar, where silently substituting the default hides the typo and the
    process runs with a pool size nobody asked for.
    """
    return _nonnegative(env_int(env_name, default), env_name)


def _nonnegative(parse: Callable[[], Any], env_name: str) -> Callable[[], Any]:
    """Wrap a number parser so a negative value raises instead of passing."""

    def _get_nonnegative_env() -> Any:
        value = parse()
        if value < 0:
            raise ValueError(f"Invalid value '{value}' for {env_name}: must be >= 0.")
        return value

    return _get_nonnegative_env


def env_float(env_name: str, default: float) -> Callable[[], float]:
    """Float knob with a default. An unparseable value raises."""

    def _get_float_env() -> float:
        value = os.getenv(env_name)
        if value is None or value.strip() == "":
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}: must be a number."
            ) from exc

    return _get_float_env


def env_nonnegative_float(env_name: str, default: float) -> Callable[[], float]:
    """Float knob that must be finite and not negative.

    `nan` and `inf` parse as floats but are rejected: as a timeout, `nan`
    compares false against everything and `inf` never expires.
    """
    parse = _nonnegative(env_float(env_name, default), env_name)

    def _get_nonnegative_float_env() -> float:
        value = parse()
        if not math.isfinite(value):
            raise ValueError(f"Invalid value '{value}' for {env_name}: must be finite.")
        return value

    return _get_nonnegative_float_env


def env_str(env_name: str, default: str) -> Callable[[], str]:
    """String knob with a default, surrounding whitespace stripped."""

    def _get_str_env() -> str:
        value = os.getenv(env_name)
        if value is None:
            return default
        return value.strip()

    return _get_str_env


def env_optional_bool(env_name: str) -> Callable[[], bool | None]:
    """
    Same parsing as ``env_bool``, but an unset variable reads as ``None`` so
    callers can tell "explicitly disabled" from "not configured".
    """
    parse_bool = env_bool(env_name)

    def _get_optional_bool_env() -> bool | None:
        if not os.getenv(env_name):
            return None
        return parse_bool()

    return _get_optional_bool_env


def env_nonnegative_int_or_auto(env_name: str) -> Callable[[], int | None]:
    """Size knob where unset, empty or ``auto`` (any case) reads as ``None``,
    meaning the caller picks the value itself. Anything else must parse as
    an integer that is not negative."""
    parse_int = env_nonnegative_int(env_name, 0)

    def _get_nonnegative_int_or_auto_env() -> int | None:
        value = os.getenv(env_name)
        if value is None or value.strip().lower() in ("", "auto"):
            return None
        return parse_int()

    return _get_nonnegative_int_or_auto_env


def env_optional_int(env_name: str) -> Callable[[], int | None]:
    """
    Integer parsing where, like ``env_optional_bool``, an unset or empty
    variable reads as ``None`` so callers can tell "explicitly set" from
    "not configured".
    """

    def _get_optional_int_env() -> int | None:
        value = os.getenv(env_name)
        if value is None or value == "":
            return None
        return int(value)

    return _get_optional_int_env


def env_positive_int_or_none(env_name: str) -> Callable[[], int | None]:
    """Optional size knob that must be greater than zero when it is set.

    Unset or empty reads as ``None``, meaning the feature is off.
    """
    parse_int = env_optional_int(env_name)

    def _get_positive_int_or_none_env() -> int | None:
        value = parse_int()
        if value is not None and value <= 0:
            raise ValueError(f"Invalid value '{value}' for {env_name}: must be > 0.")
        return value

    return _get_positive_int_or_none_env


environment_variables: dict[str, Callable[[], Any]] = {
    # Qwen3.5 MoE at TP > 1: reduce-scatter the attention output, run the
    # post-attention norm and router gate on the token shard, and fold the
    # attention partial sum into the MoE all-reduce. Default MoE path only.
    "TPU_ATTN_MOE_OVERLAP": env_bool("TPU_ATTN_MOE_OVERLAP", default=False),
    # Opt in to K3 TP32/EP32 sequence-parallel prefill.
    "TPU_K3_SP_PREFILL": env_bool("TPU_K3_SP_PREFILL", default=False),
    # DCP sparse MLA (SparseCore cache layout): gather and attend this many
    # top-k slots per token instead of all `topk`; tokens owning more are
    # recomputed exactly in-kernel over all their windows
    # (`sparse_ragged_paged_attention(topk_cap=...)`). Opt-in; 0 = off.
    "TPU_DCP_MLA_TOPK_CAP": env_int("TPU_DCP_MLA_TOPK_CAP", 0),
    # Gather/attention chunk (query tokens) of the capped path: fewer, larger
    # chunks amortise the per-chunk calls; each chunk's static gather buffer
    # is chunk * topk rows (~335 MB at 256), of which only the live windows
    # are written.
    "TPU_DCP_MLA_TOPK_CAP_CHUNK": env_int("TPU_DCP_MLA_TOPK_CAP_CHUNK", 256),
    # Opt in to the all-to-all DCP MLA lse merge for decode-sized steps.
    "TPU_DCP_LSE_MERGE_A2A": env_bool("TPU_DCP_LSE_MERGE_A2A", default=False),
    # Largest step (tokens) that uses it; above this, the reduce-scatter is
    # faster. Only the backbone uses the all-to-all; the MTP drafter, traced
    # once for every bucket, always takes the reduce-scatter.
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": env_int("TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS", 64),
    # Backbone MLA merge: reduce-scatter the weighted outputs in the output's
    # precision (bf16), half f32's bytes. MTP drafter stays f32; 0 = f32.
    "TPU_DCP_RS_OUT_PRECISION": env_bool("TPU_DCP_RS_OUT_PRECISION", default=True),
    # Sparse MLA split q (`sparse_ragged_paged_attention(q_pe=...)`): hand
    # q_nope and q_pe to the kernel as separate operands. The kernel joins
    # them in VMEM and its output aliases q_nope (512 wide), instead of the
    # caller materialising a concat the wrapper pads to [T, H, 640]. Where the
    # caller also needs the log-sum-exp (DCP), it is exported lane-dense as
    # f32[T, H] (`lse_lane_dense=True`) instead of f32[T, H, 1], whose
    # lane-padded tiles cost a broadcast init, a relayout copy and padded
    # copies through the LSE merge. Bit-identical outputs.
    "TPU_MLA_SPARSE_SPLIT_Q": env_bool("TPU_MLA_SPARSE_SPLIT_Q", default=False),
    # QK pair-blocked GDN conv layout for Raiden transfers.  Not supported
    # with the unified GDN state pool: its kernel reads segment-major
    # [Q | K | V] conv rows, so the pool manifest rejects this flag.  Must
    # match across a disagg pair; the value rides the raiden layout
    # fingerprint.
    "TPU_GDN_CONV_QK_PAIR_LAYOUT": env_bool(
        "TPU_GDN_CONV_QK_PAIR_LAYOUT", default=False
    ),
    # Store GDN's in_proj_qkvz in the matmul's (k, n) layout, like every other
    # linear, instead of vLLM's [n_out, n_in]. On the per-block FP8 path (gmm_v2)
    # the kept layout costs a full weight copy every step. Ignored with prefill
    # context parallelism, whose pooled op reads the weight raw as
    # [n_out, n_in]. The experimental AOT export rejects it at TP > 1: its
    # weight mapper assumes the vLLM layout for this weight. Opt-in while the
    # (k, n) layout is verified only on the per-block FP8 path; once the BF16,
    # checkpoint-FP8 and per-channel paths are verified and the export mapper
    # reads the layer's layout marker, make it the default and delete the flag.
    "TPU_GDN_QKVZ_KERNEL_LAYOUT": env_bool("TPU_GDN_QKVZ_KERNEL_LAYOUT", default=False),
    # Size GDN v3 decode tiles and prefill chunks from per-rank heads, active
    # sequences and the VMEM estimate (kernels/gdn/v3/tiling.py) instead of
    # the fixed defaults.
    "TPU_ENABLE_GDN_DYNAMIC_TILING": env_bool(
        "TPU_ENABLE_GDN_DYNAMIC_TILING", default=False
    ),
    # Register the DeepSeek-V4 tokenizer that honors continue_final_message.
    # Off by default so tokenizer behavior matches stock vLLM; the DSv4
    # eval config turns it on, since a prefilled assistant turn needs it.
    "TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE": env_bool(
        "TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", default=False
    ),
    # NoPE and RoPE KV cache layouts allocated for sparse MLA.
    "TPU_SPARSE_MLA_NOPE_LAYOUT": env_with_choices(
        "TPU_SPARSE_MLA_NOPE_LAYOUT", "sparsecore", ["sparsecore", "tensorcore"]
    ),
    "TPU_SPARSE_MLA_ROPE_LAYOUT": env_with_choices(
        "TPU_SPARSE_MLA_ROPE_LAYOUT", "sparsecore", ["sparsecore", "tensorcore"]
    ),
    # JAX platform selection (e.g., "tpu", "cpu", "proxy")
    "JAX_PLATFORMS": lambda: os.getenv("JAX_PLATFORMS", "").lower(),
    # TPU accelerator type (e.g., "v5litepod-16", "v4-8")
    "TPU_ACCELERATOR_TYPE": lambda: os.getenv("TPU_ACCELERATOR_TYPE", None),
    # Name of the TPU resource
    "TPU_NAME": lambda: os.getenv("TPU_NAME", None),
    # libtpu.so loaded by the TPU runtime instead of the pip libtpu package
    # (e.g. container images that ship a prebuilt library). A path to a file
    # other than the pip package's own libtpu.so takes precedence over an
    # installed libtpu package, so it is used to identify the loaded library.
    "TPU_LIBRARY_PATH": lambda: os.getenv("TPU_LIBRARY_PATH", None),
    # Worker ID for multi-host TPU setups
    "TPU_WORKER_ID": lambda: os.getenv("TPU_WORKER_ID", None),
    # Backend for multi-host communication on TPU. "" auto-detects: "mp" is
    # used whenever --nnodes > 1 and this isn't set to "ray"; set explicitly
    # to "mp" to force it (e.g. to silence the auto-detection log line).
    "TPU_MULTIHOST_BACKEND": env_with_choices(
        "TPU_MULTIHOST_BACKEND", "", ["ray", "mp"]
    ),
    # Check for XLA recompilation during execution
    "VLLM_XLA_CHECK_RECOMPILATION": lambda: bool(
        int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION") or "0")
    ),
    # Python tracer level for profiling
    "PYTHON_TRACER_LEVEL": lambda: int(os.getenv("PYTHON_TRACER_LEVEL") or "1"),
    # Host tracer level for profiling
    "HOST_TRACER_LEVEL": lambda: int(os.getenv("HOST_TRACER_LEVEL") or "2"),
    # Device tracer level for profiling
    "DEVICE_TRACER_LEVEL": lambda: int(os.getenv("DEVICE_TRACER_LEVEL") or "1"),
    # TPU trace mode (e.g. TRACE_COMPUTE, TRACE_COMPUTE_AND_SYNC)
    "TPU_TRACE_MODE": lambda: os.getenv("TPU_TRACE_MODE", "TRACE_COMPUTE"),
    # Number of sparse cores to trace
    "TPU_NUM_SPARSE_CORES_TO_TRACE": lambda: int(
        os.getenv("TPU_NUM_SPARSE_CORES_TO_TRACE") or "1"
    ),
    # Number of sparse core tiles to trace
    "TPU_NUM_SPARSE_CORE_TILES_TO_TRACE": lambda: int(
        os.getenv("TPU_NUM_SPARSE_CORE_TILES_TO_TRACE") or "1"
    ),
    # Enable/disable Ray usage statistics collection
    "RAY_USAGE_STATS_ENABLED": lambda: os.getenv("RAY_USAGE_STATS_ENABLED", "0"),
    # Ray compiled DAG channel type for TPU
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE": env_with_choices(
        "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "shm", ["shm"]
    ),
    # Enable the blockwise Pallas quantized matmul kernel for dense linears.
    "ENABLE_QUANTIZED_MATMUL_KERNEL": env_bool("ENABLE_QUANTIZED_MATMUL_KERNEL"),
    # Runtime block size for dense linear weight requantization.
    "REQUANTIZE_BLOCK_SIZE": lambda: int(block_size)
    if (block_size := os.getenv("REQUANTIZE_BLOCK_SIZE")) is not None
    else None,
    # Runtime dtype for dense linear weight requantization.
    "REQUANTIZE_WEIGHT_DTYPE": lambda: os.getenv(
        "REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn"
    ),
    # Opt-in W8A16 for small FP8 dense-linear steps. A step whose padded token
    # bucket (from compile_sizes, not the number of live requests) has at most
    # this many rows keeps bf16 activations against the FP8 weights instead of
    # quantizing them per call: small decode steps are weight-bandwidth bound,
    # so the per-call quantization (an abs-max reduction and a cast) costs
    # more than FP8 activations save. Measured on v7x only. Applies only to
    # the per-channel path with FP8 weights; the blockwise kernel path
    # (ENABLE_QUANTIZED_MATMUL_KERNEL=1) and int8 weights always quantize.
    # With compile_sizes [256] and a threshold below 256, no bucket qualifies.
    # 0 (default) keeps W8A8 everywhere.
    "TPU_FP8_LINEAR_W8A16_MAX_ROWS": env_nonnegative_int(
        "TPU_FP8_LINEAR_W8A16_MAX_ROWS", 0
    ),
    # Specify dtype for quantized MoE weights
    "MOE_REQUANTIZE_WEIGHT_DTYPE": lambda: os.getenv(
        "MOE_REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn"
    ),
    # Specify requantization block size for MoE weights. When unset, NVFP4
    # expert-parallel layers admitted with USE_MOE_FUSED_EP_KERNEL and
    # MOE_FUSED_EP_ENABLE_W4A8 use the kernel's smallest supported block.
    # Selecting MOE_FUSED_EP_KERNEL_IMPL=adaptive also does this automatically.
    # Other MoE paths retain their default, including explicit requantization.
    "MOE_REQUANTIZE_BLOCK_SIZE": lambda: int(block_size)
    if (block_size := os.getenv("MOE_REQUANTIZE_BLOCK_SIZE")) is not None
    else None,
    # TPU unified block-pool cache layout. Tri-state: an explicit 0/1 always
    # wins; unset defers to
    # platforms.tpu_block_size_utils.unified_kv_layout_enabled, which selects
    # the pool for the hybrid models that ship a pooled state path and keeps
    # everything else on the per-layer KV caches.
    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": env_optional_bool(
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL"
    ),
    # Destination (decode-side) page geometry in tokens, required on the
    # producer: the byte-span lowering splits declarations at destination
    # page boundaries at registration time. 0 means unset.
    # Raiden admission gates: construct a Raiden KVCacheManager and
    # register the pool manifest derived from the live typed KV caches.
    # This does not switch the V2 strided transfer transport.
    "TPU_USE_RAIDEN_KV_CACHE_MANAGER": env_bool("TPU_USE_RAIDEN_KV_CACHE_MANAGER"),
    "TPU_RAIDEN_QWEN35_ADMISSION": env_bool("TPU_RAIDEN_QWEN35_ADMISSION"),
    # Kimi K3 entry point to shared hybrid-pool admission.
    "TPU_RAIDEN_KIMIK3_ADMISSION": env_bool("TPU_RAIDEN_KIMIK3_ADMISSION"),
    # Tag every raiden pool with its layer index so pools pair up by layer
    # across a pipeline-parallel producer and its decode peer. Both peers
    # must set it identically.
    "TPU_RAIDEN_POOL_TAGS_PER_LAYER": env_bool("TPU_RAIDEN_POOL_TAGS_PER_LAYER"),
    "TPU_RAIDEN_GLM_ADMISSION": env_bool("TPU_RAIDEN_GLM_ADMISSION"),
    # DeepSeek-V4 admits several attention KV cache groups at once: CSA,
    # indexer, HCA, SWA and the compressor states. Every other model here has
    # exactly one.
    "TPU_RAIDEN_DSV4_ADMISSION": env_bool("TPU_RAIDEN_DSV4_ADMISSION"),
    # Controller-driven PCP->DP pool resharding. The default preserves
    # the existing v1 connector protocol; selecting ``raiden`` enables
    # the fail-closed controller path.
    "TPU_KV_RESHARD_TRANSPORT": env_with_choices(
        "TPU_KV_RESHARD_TRANSPORT", "zmq", ["zmq", "raiden"], case_sensitive=False
    ),
    "TPU_RAIDEN_CONTROLLER_ADDRESS": lambda: os.getenv(
        "TPU_RAIDEN_CONTROLLER_ADDRESS", ""
    ).strip(),
    "TPU_RAIDEN_JOB_NAME": lambda: os.getenv("TPU_RAIDEN_JOB_NAME", "").strip(),
    "TPU_RAIDEN_ENGINE_ID": lambda: os.getenv("TPU_RAIDEN_ENGINE_ID", "0").strip(),
    "TPU_RAIDEN_TRANSFER_PARALLELISM": lambda: int(
        os.getenv("TPU_RAIDEN_TRANSFER_PARALLELISM") or "8"
    ),
    # TP shards in one decode engine, not its DP replica count. Configure
    # both roles alike; the producer uses this to route head-sharded state.
    "TPU_RAIDEN_DST_SHARDS": env_optional_int("TPU_RAIDEN_DST_SHARDS"),
    "TPU_RAIDEN_MAX_TRANSFER_TOKENS": env_optional_int(
        "TPU_RAIDEN_MAX_TRANSFER_TOKENS"
    ),
    # Reshard control-plane hosting (zero sidecar processes): "store" hosts the
    # reshard service inside each engine's rank-0 worker via an in-process
    # KVCacheStore; "controller" keeps the external (sidecar) controller.
    "TPU_RAIDEN_RESHARD_IMPL": env_with_choices(
        "TPU_RAIDEN_RESHARD_IMPL",
        "controller",
        ["controller", "store"],
        case_sensitive=False,
    ),
    # Host advertised for in-process reshard/dispatch services (store mode).
    "TPU_RAIDEN_ADVERTISE_HOST": lambda: os.getenv(
        "TPU_RAIDEN_ADVERTISE_HOST", ""
    ).strip(),
    # Per-engine deterministic ports (store mode): engine dp_rank i binds
    # reshard service at RESHARD_PORT_BASE+i and its dispatch controller at
    # STORE_DISPATCH_PORT_BASE+i; peers derive the same addresses.
    "TPU_RAIDEN_RESHARD_PORT_BASE": lambda: int(
        os.getenv("TPU_RAIDEN_RESHARD_PORT_BASE") or "27000"
    ),
    "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE": lambda: int(
        os.getenv("TPU_RAIDEN_STORE_DISPATCH_PORT_BASE") or "27100"
    ),
    # Decode-side prefix-cache-aware reshard loads: when true, the consumer
    # subtracts the local prefix-cache hit and transfers only the missing
    # FA suffix (page-aligned dst_skip_bytes clip planned by the source
    # store); GDN state always transfers whole. When false, or when the
    # installed client lacks dst_skip_bytes support, local hits are ignored
    # and the full payload is pulled into every destination page.
    "TPU_RAIDEN_PREFIX_AWARE_LOAD": env_bool("TPU_RAIDEN_PREFIX_AWARE_LOAD", False),
    # Selects the #193 SparseCore MoE token-movement path. When 0, the EP
    # ragged gather + gather-reduce fall back to the pre-#193 plain-JAX
    # path (functionally equivalent; valid-mask gating unchanged). Mirrors
    # upstream vLLM's VLLM_USE_FLASHINFER_MOE_* style of env-gated MoE kernel
    # selection (vLLM 0.19.0 has no --moe-backend; see fused_moe.py TODO).
    "USE_MOE_SPARSE_CORE": lambda: bool(int(os.getenv("USE_MOE_SPARSE_CORE") or "1")),
    # Use one-hot matmuls for MoE permute and unpermute when the routed row
    # count (num_tokens * topk) is <= this threshold. Unset or empty means
    # auto: fused_moe_gmm.resolve_onehot_permute_threshold derives the value
    # from SparseCore geometry (0 on hosts without SparseCore). 0 keeps every
    # size on the SparseCore path; a positive value forces that
    # threshold. The knob and its row-count semantics come from
    # tpu-inference PR #2674.
    "ONEHOT_MOE_PERMUTE_THRESHOLD": env_optional_int("ONEHOT_MOE_PERMUTE_THRESHOLD"),
    # Shard tokens over TP between attention and MoE: attention reduce-scatters
    # instead of all-reducing, the router, shared expert, and experts run on
    # T/tp tokens, and the next layer all-gathers. Qwen3.5 MoE with
    # EP and TP > 1 only.
    "TPU_MOE_SEQUENCE_PARALLEL": env_bool("TPU_MOE_SEQUENCE_PARALLEL", default=False),
    # Prefill-sized gmm_v2 calls with fp4 experts (per-128-K-block scales) run
    # fp8 x fp8: the rhs tile requantized to e4m3 with per-column scales, the
    # lhs rows to e4m3 with per-row scales (kernels/megablox/gmm_v2.py
    # uses_fp8_w8a8). ~1.6x faster at 8k-token chunks, ~6% output error vs 0.4%.
    "TPU_GMM_FP8_W8A8": env_bool("TPU_GMM_FP8_W8A8", default=False),
    # Select the blockwise owner-output kernel after the outer one-hot MoE
    # threshold has selected the TensorCore path. "off" keeps the dense
    # one-hot combine; "on" enables the kernel for every shape that passes
    # its dtype, alignment, and VMEM safety checks.
    "TPU_MOE_OWNER_OUTPUT_MODE": env_with_choices(
        "TPU_MOE_OWNER_OUTPUT_MODE", "off", ["off", "on"], case_sensitive=False
    ),
    # Maximum hidden-state row count (after any EP all-gather) for dense
    # expert MoE. 0 disables it; larger shapes and unsupported layouts use
    # the original GMM path. This counts tokens, not tokens * topk.
    "TPU_MOE_DENSE_EXPERT_THRESHOLD": env_nonnegative_int(
        "TPU_MOE_DENSE_EXPERT_THRESHOLD", 0
    ),
    # MiniMax-M3 target-model MXFP8 linears run on e4m3 weights with a
    # per-channel power-of-two scale (models/minimax_m3/quant.py); 0 restores
    # the exact bf16 dequantization.
    "TPU_MINIMAX_M3_FP8_LINEARS": env_bool("TPU_MINIMAX_M3_FP8_LINEARS", default=True),
    # MiniMax-M3 e4m3 linears (TPU_MINIMAX_M3_FP8_LINEARS): token buckets of at
    # most this many rows quantize the activations to e4m3 per token (dynamic
    # abs-max scale) and run the matmul fp8 x fp8 on the MXU instead of
    # upcasting the weights to bf16 (decode buckets: faster on v7x); larger
    # buckets keep bf16 activations, where XLA's fp8 dot plus the quantization
    # measured slower. Changes numerics. 0 (default) keeps bf16 activations.
    "TPU_MINIMAX_M3_FP8_W8A8_MAX_ROWS": env_nonnegative_int(
        "TPU_MINIMAX_M3_FP8_W8A8_MAX_ROWS", 0
    ),
    # Smallest token bucket TPU_MINIMAX_M3_FP8_W8A8_MAX_ROWS applies to.
    "TPU_MINIMAX_M3_FP8_W8A8_MIN_ROWS": env_nonnegative_int(
        "TPU_MINIMAX_M3_FP8_W8A8_MIN_ROWS", 0
    ),
    # MiniMax-M3 DSpark under DCP: write every draft layer's context K/V into
    # its sharded (head-split) cache with one op instead of one op per layer,
    # so the shared token layout, slots and runs are computed once
    # (`cp_draft_attention.write_layers_op`). Applies only when all draft
    # layers are DCP-sharded with one sharding and share one attention
    # metadata; otherwise the per-layer writes run. Default off.
    "TPU_MINIMAX_M3_DSPARK_ONE_KV_WRITE": env_bool(
        "TPU_MINIMAX_M3_DSPARK_ONE_KV_WRITE", default=False
    ),
    # With TPU_MINIMAX_M3_DSPARK_ONE_KV_WRITE armed: compute every draft
    # layer's context K/V with one matmul against their qkv_proj K/V columns
    # concatenated at load (instead of one matmul per layer). Same products.
    "TPU_MINIMAX_M3_DSPARK_CONTEXT_KV_CAT": env_bool(
        "TPU_MINIMAX_M3_DSPARK_CONTEXT_KV_CAT", default=False
    ),
    # A DCP-sharded draft attention layer whose KV head spans two ranks
    # merges the two partials in one Pallas exchange
    # (kernels/collectives/pair_lse_merge.py) instead of cp_lse_ag_out_rs's
    # all-gather + reduce-scatter. Needs the Pallas TP all-reduce armed
    # (TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES > 0) in rank order and an async
    # drafter; armed after a self-test, else the usual merge. Default off.
    "TPU_DRAFT_DCP_PAIR_MERGE": env_bool("TPU_DRAFT_DCP_PAIR_MERGE", default=False),
    # With TPU_DRAFT_DCP_PAIR_MERGE armed: the draft block's own attention and
    # the final merge run in the pair exchange kernel
    # (kernels/collectives/pair_lse_merge.py `exchange_block`, uniform draft
    # blocks) instead of the XLA block op after it. Armed per block
    # configuration after a self-test; else the unfused pair merge. Default off.
    "TPU_DRAFT_DCP_PAIR_MERGE_BLOCK": env_bool(
        "TPU_DRAFT_DCP_PAIR_MERGE_BLOCK", default=False
    ),
    # MiniMax-M3 e4m3 linears: token buckets of at most this many rows split
    # the bf16 activations into two e4m3 terms (x ~ s * (hi + lo / 16), one
    # power-of-two scale s per row) and run one fp8 x fp8 MXU matmul over the
    # stacked [hi; lo] rows, instead of upcasting the whole e4m3 weight to bf16
    # inside the dot fusion. The two terms keep ~8 mantissa bits, about bf16's
    # rounding of the activations. 0 (default) keeps the upcast path.
    "TPU_MINIMAX_M3_FP8_SPLIT_MAX_ROWS": env_nonnegative_int(
        "TPU_MINIMAX_M3_FP8_SPLIT_MAX_ROWS", 0
    ),
    # Smallest token bucket TPU_MINIMAX_M3_FP8_SPLIT_MAX_ROWS applies to.
    "TPU_MINIMAX_M3_FP8_SPLIT_MIN_ROWS": env_nonnegative_int(
        "TPU_MINIMAX_M3_FP8_SPLIT_MIN_ROWS", 0
    ),
    # MiniMax-M3 sequence parallelism on steps of at least
    # TPU_MINIMAX_M3_SP_MIN_TOKENS tokens (and a multiple of TP; 0 = never)
    # (models/minimax_m3/model.py). "attn": MoE layers reduce-scatter the
    # attention output with the residual, normalize their token shard,
    # all-gather it as the MoE input, and fold the attention partial into the
    # MoE all-reduce. "full": the residual stream stays a token shard through
    # every layer (norms on the shard, all-gather before attention and FFN,
    # reduce-scatter after). "off": two all-reduces per layer on every step.
    "TPU_MINIMAX_M3_SP_MODE": env_with_choices(
        "TPU_MINIMAX_M3_SP_MODE", "off", ["off", "attn", "full"]
    ),
    "TPU_MINIMAX_M3_SP_MIN_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_SP_MIN_TOKENS", 4096
    ),
    # The FFN-input (and, for "full", the attention-input) all-gather sends the
    # normalized shard as e4m3 with a per-row f32 scale: half the bytes, and a
    # change to that input's precision.
    "TPU_MINIMAX_M3_SP_FP8_GATHER": env_bool(
        "TPU_MINIMAX_M3_SP_FP8_GATHER", default=False
    ),
    "TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER": env_bool(
        "TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER", default=False
    ),
    # "full" SP: the residual add, norm and e4m3 quantization before each
    # e4m3 all-gather run as one Pallas pass over the shard
    # (models/minimax_m3/kernels/sp_norm.py) instead of three XLA fusions.
    "TPU_MINIMAX_M3_SP_FUSED_NORM": env_bool(
        "TPU_MINIMAX_M3_SP_FUSED_NORM", default=False
    ),
    # "full" SP with the e4m3 gathers: qkv_proj and a dense layer's MLP gate_up
    # contract the gathered e4m3 rows against their e4m3 weight (e4m3 x e4m3,
    # f32 accumulation) instead of dequantizing both to bf16.
    "TPU_MINIMAX_M3_SP_E4M3_QKV": env_bool("TPU_MINIMAX_M3_SP_E4M3_QKV", default=False),
    # With the e4m3 gathers: the MoE layers' shared expert gate_up contracts
    # the gathered e4m3 rows (e4m3 x e4m3, f32 accumulation) as qkv does,
    # instead of the bf16 rows dequantized from them (bf16 x bf16).
    "TPU_MINIMAX_M3_SP_E4M3_SHARED": env_bool(
        "TPU_MINIMAX_M3_SP_E4M3_SHARED", default=False
    ),
    # "full" SP steps whose MoE combine runs on the TensorCore
    # (TPU_MOE_TC_COMBINE): the shared expert's down_proj runs inside the
    # combine's output pass (kernels/megablox/tc_combine.tc_combine_shared),
    # so its [T, hidden] partial never round-trips HBM. Same roundings as the
    # unfused graph; only the K=384 dot's accumulation order may differ.
    "TPU_MINIMAX_M3_SHARED_IN_COMBINE": env_bool(
        "TPU_MINIMAX_M3_SHARED_IN_COMBINE", default=False
    ),
    # Smallest step it applies to (exos microbench: -69 us/layer at 8192,
    # -30 at 4096, +6..+15 at 1024-2048).
    "TPU_MINIMAX_M3_SHARED_IN_COMBINE_MIN_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_SHARED_IN_COMBINE_MIN_TOKENS", 4096
    ),
    # "full" SP steps of at least TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS
    # tokens norm and all-gather each layer's FFN input per row piece of the
    # token shard (the FFN's entry work, e.g. the router, starts on a piece as
    # it lands), run the FFN once on the pieces' rows, and reduce-scatter its
    # output per piece (through the SP row reduce-scatter, so QRS / HIER_RS
    # still apply). Row-local math only: same bits as one piece. 1 = off.
    "TPU_MINIMAX_M3_SP_FFN_PIECES": env_nonnegative_int(
        "TPU_MINIMAX_M3_SP_FFN_PIECES", 1
    ),
    "TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS", 4096
    ),
    # With TPU_MINIMAX_M3_SP_FFN_PIECES: gather the FFN input in pieces (off:
    # only the attention side, TPU_MINIMAX_M3_SP_ATTN_PIECES, runs per piece).
    "TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES": env_bool(
        "TPU_MINIMAX_M3_SP_FFN_GATHER_PIECES", default=True
    ),
    # With TPU_MINIMAX_M3_SP_FFN_PIECES: run a dense layer's FFN once per
    # gathered piece instead of once over all pieces, so a piece's
    # reduce-scatter overlaps the next piece's FFN. MoE layers keep one pass.
    "TPU_MINIMAX_M3_SP_DENSE_FFN_PER_PIECE": env_bool(
        "TPU_MINIMAX_M3_SP_DENSE_FFN_PER_PIECE", default=False
    ),
    # With TPU_MINIMAX_M3_SP_FFN_PIECES: the attention input is gathered in the
    # same row pieces and qkv_proj runs on each piece as it lands; o_proj's
    # e4m3 all-to-all (TPU_MINIMAX_M3_SP_A2A_OPROJ + _A2A_E4M3) and
    # projection run per piece. Row-local math only: same bits.
    "TPU_MINIMAX_M3_SP_ATTN_PIECES": env_bool(
        "TPU_MINIMAX_M3_SP_ATTN_PIECES", default=False
    ),
    # "full" SP: instead of reduce-scattering o_proj's per-rank partial sums
    # (bf16 [T, hidden]), all-to-all the attention output (bf16 [T, heads/TP x
    # head_dim], 1/6 of the bytes) so each rank holds every head of its token
    # shard, and run o_proj over the full input dim on the shard. The full
    # e4m3 weight is all-gathered per step ("gather") or kept on every rank
    # from load ("replicate": +7/8 of o_proj's weight bytes per rank, held
    # by decode steps too; gathered through the host at load, one layer at a
    # time). Either way the shards are rescaled at load onto
    # one per-channel scale, which every path (decode included) then uses:
    # a small change to o_proj's numerics. Needs the armed Pallas TP
    # all-reduce (TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES > 0) for its rank index.
    "TPU_MINIMAX_M3_SP_A2A_OPROJ": env_with_choices(
        "TPU_MINIMAX_M3_SP_A2A_OPROJ", "off", ["off", "gather", "replicate"]
    ),
    # With TPU_MINIMAX_M3_SP_A2A_OPROJ: the attention heads cross the
    # all-to-all as e4m3 rows with power-of-two row scales, and o_proj
    # contracts e4m3 x e4m3 (a change to o_proj's input precision).
    "TPU_MINIMAX_M3_SP_A2A_E4M3": env_bool("TPU_MINIMAX_M3_SP_A2A_E4M3", default=False),
    # Sequence-parallel row reduce-scatters (TPU_MINIMAX_M3_SP_MODE) of at
    # least TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS rows run qar_reduce_scatter (int8
    # payload on the ICI links, kernels/collectives/qar_reduce_scatter.py)
    # instead of XLA's reduce-scatter; smaller ones keep XLA's. Needs the
    # Pallas all-reduce armed (TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES > 0) on a
    # rank-ordered tpu7x 2x2x1 TP8 group. Changes numerics (~0.7% rel L2 per
    # reduce-scatter). Off: the reduce-scatters are unchanged. exos
    # microbench (hidden 6144): 0.34 vs 0.90 ms at 8192 rows, 0.15 vs 0.28 at
    # 2048, slower than XLA at 1024: hence the 2048 default minimum.
    "TPU_MINIMAX_M3_SP_QRS": env_bool("TPU_MINIMAX_M3_SP_QRS", default=False),
    "TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS", 2048
    ),
    # MiniMax-M3 e4m3 linears: token buckets of at least this many rows cast
    # the bf16 activations to e4m3 unscaled (clamped to +-448; no abs-max
    # pass) and run the matmul fp8 x fp8 on the MXU with f32 accumulation.
    # The cast fuses into the activation's producer (input RMSNorm for
    # qkv), so it costs no extra pass; the per-token abs-max of
    # TPU_MINIMAX_M3_FP8_W8A8_* costs as much as the fp8 MXU saves at
    # prefill sizes. Changes numerics. 0 (default) keeps bf16 activations.
    "TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS": env_nonnegative_int(
        "TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS", 0
    ),
    # Comma-separated linear-name suffixes TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS
    # leaves on bf16 activations: the shared expert's gate_up input also
    # feeds the bf16 routed experts, so its cast is an extra pass.
    "TPU_MINIMAX_M3_FP8_CAST_SKIP": env_str(
        "TPU_MINIMAX_M3_FP8_CAST_SKIP", "shared_experts.gate_up_proj"
    ),
    # TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS: cast the activations in torch before
    # the fp8 matmul op instead of inside it.
    "TPU_MINIMAX_M3_FP8_CAST_IN_TORCH": env_bool(
        "TPU_MINIMAX_M3_FP8_CAST_IN_TORCH", default=False
    ),
    # TPU_MINIMAX_M3_FP8_CAST_MIN_ROWS: the input RMSNorm of a layer whose
    # qkv projection casts emits the e4m3 activation itself (one write
    # instead of a bf16 copy and a separate cast pass).
    "TPU_MINIMAX_M3_FP8_CAST_NORM": env_bool(
        "TPU_MINIMAX_M3_FP8_CAST_NORM", default=False
    ),
    # Sequence-parallel row reduce-scatters (TPU_MINIMAX_M3_SP_MODE) of at
    # least TPU_MINIMAX_M3_SP_HIER_RS_MIN_TOKENS rows run the hierarchical
    # TensorCore reduce-scatter (kernels/collectives/hierrs_tc: on-chip d2d
    # leg, then a 2-step chip hypercube with both ICI links busy) instead of
    # XLA's SparseCore one; bf16 on the wire, bf16 sums in another order (not
    # bitwise, ~0.31% vs 0.37% rel L2 against an f32 sum). Needs the Pallas
    # all-reduce armed (TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES > 0) on a
    # rank-ordered tpu7x 2x2x1 TP8 group; exclusive with TPU_MINIMAX_M3_SP_QRS.
    # exos microbench (hidden 6144): standalone 0.442 vs 0.489 ms at 8192
    # rows, 0.061 vs 0.069 at 1024; inside an SP layer composition only 8192
    # rows gain (-0.01..-0.066 ms per layer; 6144 rows +0.05..+0.085, 2048-4096
    # flat), hence the 8192 default minimum. Off: the reduce-scatters are
    # unchanged.
    "TPU_MINIMAX_M3_SP_HIER_RS": env_bool("TPU_MINIMAX_M3_SP_HIER_RS", default=False),
    "TPU_MINIMAX_M3_SP_HIER_RS_MIN_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_SP_HIER_RS_MIN_TOKENS", 8192
    ),
    # EP steps of up to this many tokens with fp4 (NVFP4) experts run the
    # permutation-free dense-expert kernel (kernels/experimental/
    # moe_dense_expert_fp4.py) instead of sort + grouped GEMM + combine.
    # 0 disables it.
    "TPU_MOE_DENSE_FP4_MAX_TOKENS": env_nonnegative_int(
        "TPU_MOE_DENSE_FP4_MAX_TOKENS", 16
    ),
    # Steps above TPU_MOE_DENSE_FP4_MAX_TOKENS and up to this many tokens run
    # the same kernel on each expert's own routes only (rows gathered into
    # 16-row tiles) instead of the routing sort, permute, GMM and combine.
    # Set (> 0), TP-MoE ranks (every expert's slice) take the dense fp4
    # kernel too, at both sizes. 0 (default) disables it.
    "TPU_MOE_DENSE_FP4_SPARSE_MAX_TOKENS": env_nonnegative_int(
        "TPU_MOE_DENSE_FP4_SPARSE_MAX_TOKENS", 0
    ),
    # Serve eligible MoE layers (TP experts, or EP with TP attention) with the
    # fused MoE decode kernel: routed experts, shared expert and all-reduce in
    # one Pallas call (layers/adapter/fused_moe_decode.py). Armed once after
    # load. Steps above FUSED_MOE_DECODE_KERNEL_MAX_TOKENS tokens run the GMM
    # MoE path; arming refuses a cap whose buffers do not fit in VMEM.
    "FUSED_MOE_DECODE_KERNEL": env_bool("FUSED_MOE_DECODE_KERNEL"),
    "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS": env_int(
        "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", 256
    ),
    # Build the MoE permutation, its inverse, and the sorted token ids from
    # one packed sort instead of three (up to 2,048 routed rows on non-ragged
    # paths). Within that cap the counting sort takes precedence unless it
    # declines (fewer routed rows than local experts + 1, or local experts >=
    # 512) or TPU_MOE_COUNTING_SORT is off; above 2,048 rows on non-ragged
    # paths three sorts run.
    "TPU_MOE_SINGLE_SORT": env_bool("TPU_MOE_SINGLE_SORT", True),
    # Derive the MoE routing positions and group sizes on non-ragged paths
    # (EP one-hot permute and non-EP TP, up to 2,048 routed rows) from the MXU
    # counting sort instead of comparison sorts. Off, those paths fall back to
    # the packed single sort (TPU_MOE_SINGLE_SORT) or the three-sort form, as
    # before the counting sort reached them; the ragged permute path keeps the
    # counting sort either way. A stand-down knob in case a libtpu release
    # miscompiles the rank contraction again.
    "TPU_MOE_COUNTING_SORT": env_bool("TPU_MOE_COUNTING_SORT", True),
    # With the counting sort, the number of valid local routes (the computed
    # prefix of the grouped matmul's rows) is read off the sort's own bucket
    # base contraction (the sentinel bucket's base) instead of a separate
    # sum over group_sizes; the one-hot combine's row mask uses it. Exact.
    "TPU_MOE_VALID_COUNT_FROM_SORT": env_bool("TPU_MOE_VALID_COUNT_FROM_SORT"),
    # The dense one-hot MoE combine (`fused_moe_gmm._flat_onehot_combine`)
    # builds its [T, T * topk] matrix from the flat revert indices with one
    # small contraction instead of reshaping them to [T, topk] (a lane
    # relayout). Same values.
    "TPU_MOE_ONEHOT_COMBINE_FLAT": env_bool("TPU_MOE_ONEHOT_COMBINE_FLAT"),
    # TPU_MOE_ONEHOT_COMBINE_FLAT only on calls of at most this many routes
    # (T * topk); 0 = no cap. Its [routes, routes] permutation operand grows
    # quadratically (c64 decode: 1536-1792 routes).
    "TPU_MOE_ONEHOT_COMBINE_FLAT_MAX_ROUTES": env_int(
        "TPU_MOE_ONEHOT_COMBINE_FLAT_MAX_ROUTES", 0
    ),
    # Run the EP one-hot MoE permute and combine as Pallas kernels that visit
    # only the 128-slot blocks of the valid local prefix (about 1/EP of the
    # routed slots) instead of dense one-hot matmuls over every slot, where
    # the shapes allow (kernels/megablox/moe_onehot_bounded.py). A stand-down
    # knob; the owner-output combine (TPU_MOE_OWNER_OUTPUT_MODE) takes
    # precedence over this combine.
    "TPU_MOE_BOUNDED_ONEHOT": env_bool("TPU_MOE_BOUNDED_ONEHOT", True),
    # Serve eligible tensor-parallel MoE layers (sigmoid + correction-bias
    # top-k with renormalization, one shared expert, FP8 per-channel weights)
    # with one op that also covers the shared expert and the TP all-reduce:
    # at 16, 32 and 64 tokens a single Pallas kernel per rank, at other token
    # counts the usual path plus a psum (layers/adapter/moe_tp_decode.py).
    # Armed once after load; needs TPU_MOE_SKIP_PADDED_TOKENS=1 (the op reads
    # the real token count from its padding mask), otherwise every layer keeps
    # the usual path. Arms nothing while FUSED_MOE_DECODE_KERNEL is set. Remove
    # once the path is the default for these layers or is replaced by another
    # fused TP MoE path.
    "USE_MOE_TP_DECODE_KERNEL": env_bool("USE_MOE_TP_DECODE_KERNEL"),
    # All-reduce the TP group's bf16 [tokens, hidden] activations with one
    # Pallas call (reduce-scatter + all-gather, distributed/pallas_all_reduce.py)
    # when the kernel's view of the tensor is at most this many bytes; XLA's
    # all-reduce above it. Needs the EP group to span the TP group and the
    # world. 0 (default) keeps XLA's all-reduce everywhere; capped at 4 MiB
    # (VMEM). At 8 x 6144 bf16 on tpu7x TP8 the kernel takes 8.3 us against
    # 14.6 us; 524288 (the bytes of 42.7 rows of 6144) admits the decode
    # buckets up to 40 rows of 6144, where the two still differ.
    "TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES": env_int(
        "TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES", 0
    ),
    # With the Pallas TP all-reduce armed (rank-ordered mesh): all-gather
    # small 32-bit tensors (an SP step's e4m3 row scales, the DSpark draft's
    # context fc columns, the row-split synthetic sampler's tokens) with one
    # Pallas call (kernels/collectives/tp_all_reduce.py `all_gather_small`)
    # instead of XLA's TensorCore all-gather. Same bits.
    "TPU_TP_PALLAS_SMALL_ALL_GATHER": env_bool(
        "TPU_TP_PALLAS_SMALL_ALL_GATHER", default=False
    ),
    # Pallas TP all-reduce on the activation's own [rows, hidden] layout when
    # rows is a multiple of this value (0, default: off). Off, the kernel runs
    # on a [16, w] view, and the relayout into the view (and back) is a copy
    # of the whole activation on each side of every decode all-reduce. Same
    # bits either way (each element is summed in rank order).
    "TPU_TP_PALLAS_ALL_REDUCE_DIRECT_ROWS": env_int(
        "TPU_TP_PALLAS_ALL_REDUCE_DIRECT_ROWS", 0
    ),
    # Accuracy emulation of an int8-payload TP all-reduce: quantize each
    # rank's bf16 partial and the reduced result to int8 with one absmax scale
    # per (lane, ROWS-row block), dequantize, and all-reduce as usual. No
    # speedup; measures what an int8 wire would cost in accuracy. 0 = off.
    # Applies to all-reduces of at least TPU_TP_AR_QDQ_MIN_TOKENS rows.
    "TPU_TP_AR_QDQ_ROWS": env_int("TPU_TP_AR_QDQ_ROWS", 0),
    # TP all-reduces of at least this many rows run qar_all_reduce (int8
    # payload on the ICI links, kernels/collectives/qar_all_reduce.py) instead
    # of XLA's; needs the Pallas all-reduce armed (TPU_TP_PALLAS_ALL_REDUCE_
    # MAX_BYTES > 0) on a tpu7x 2x2x1 TP8 group. Changes numerics (~0.9% rel
    # L2 per all-reduce). 0 = off.
    "TPU_TP_QAR_MIN_TOKENS": env_int("TPU_TP_QAR_MIN_TOKENS", 0),
    "TPU_TP_AR_QDQ_MIN_TOKENS": env_int("TPU_TP_AR_QDQ_MIN_TOKENS", 0),
    # Pallas TP all-reduces of at most this many rows (a multiple of 8) run
    # the one-shot kernel: each rank pushes its whole activation to every
    # peer and sums all n copies in rank order itself, one exchange round
    # instead of reduce-scatter + all-gather. Same bits. 0 (default) = off.
    "TPU_TP_PALLAS_ALL_REDUCE_ONESHOT_ROWS": env_int(
        "TPU_TP_PALLAS_ALL_REDUCE_ONESHOT_ROWS", 0
    ),
    # Pallas TP all-reduces (TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES) of at most
    # this many rows run the two-level kernel on a tpu7x 2x2x1 TP8 group
    # (kernels/collectives/tp_all_reduce_hier.py), which sends each chip's
    # data over ICI once instead of once per core. 0 (default) = off.
    "TPU_TP_PALLAS_ALL_REDUCE_HIER_ROWS": env_int(
        "TPU_TP_PALLAS_ALL_REDUCE_HIER_ROWS", 0
    ),
    # ... and at least this many rows (0 = no minimum). On tpu7x TP8 the
    # flat kernel is as fast or faster below 32 rows of 6144 (k7 microbench).
    "TPU_TP_PALLAS_ALL_REDUCE_HIER_MIN_ROWS": env_int(
        "TPU_TP_PALLAS_ALL_REDUCE_HIER_MIN_ROWS", 0
    ),
    # Which legs of that kernel are two-level. "ag" (default): only the
    # all-gather (each reduced chunk crosses ICI once per chip, the sibling
    # core gets a forwarded copy); the reduce-scatter is the flat kernel's,
    # so the bits are the flat kernel's. "full": the reduce-scatter too (the
    # two cores of a chip add their halves first): half the ICI bytes in
    # both legs, but the pair sums are rounded to bf16 before the ICI leg
    # (two roundings per element instead of one), so not bitwise.
    "TPU_TP_PALLAS_ALL_REDUCE_HIER_MODE": env_with_choices(
        "TPU_TP_PALLAS_ALL_REDUCE_HIER_MODE", "ag", ["ag", "full"]
    ),
    # Expert-parallel MoE combine of large batches on the TensorCore
    # (kernels/megablox/tc_combine.py) instead of the SparseCore
    # gather-reduce, where its shape admits it.
    "TPU_MOE_TC_COMBINE": env_bool("TPU_MOE_TC_COMBINE"),
    # TPU_MOE_TC_COMBINE on steps of at least 4096 tokens: combine the tokens
    # in this many consecutive row pieces (one kernel call each, without the
    # out-of-order gather fallback), so a consumer of the first piece (an SP
    # piece's reduce-scatter, TPU_MINIMAX_M3_SP_FFN_PIECES) can start while
    # the next is combined. Per-token math unchanged: same bits. 1 = off.
    "TPU_MOE_TC_COMBINE_PIECES": env_nonnegative_int("TPU_MOE_TC_COMBINE_PIECES", 1),
    # SparseCore MoE gather-reduce (combine) kernel version used by
    # fused_moe_gmm. "v2" is the default; "v1" selects the legacy kernel and
    # "v3" selects the destination-major prototype.
    "RAGGED_GATHER_REDUCE_VERSION": env_with_choices(
        "RAGGED_GATHER_REDUCE_VERSION", "v2", ["v1", "v2", "v3"]
    ),
    # EP MoE steps of at least this many tokens combine the expert outputs on
    # the TensorCore (kernels/megablox/moe_tc_combine.tc_slab_combine: aligned
    # 16-row slab DMAs plus a 0/1 selection matmul, then the f32 top-k-order sum
    # of the SparseCore v3 combine) instead of the SparseCore gather-reduce.
    # 0 (default) keeps the SparseCore combine. Needs tokens divisible by 256.
    "TPU_MOE_TC_COMBINE_MIN_TOKENS": env_nonnegative_int(
        "TPU_MOE_TC_COMBINE_MIN_TOKENS", 0
    ),
    # Build the v3 combine's per-block route metadata with one multi-operand
    # TensorCore sort instead of an argsort plus four gathers (which XLA
    # offloads to the SparseCore ahead of the dispatch). Same values.
    "TPU_MOE_V3_PAYLOAD_SORT": env_bool("TPU_MOE_V3_PAYLOAD_SORT"),
    # EP MoE steps of at least this many tokens flatten the [tokens, top_k]
    # routing ids and weights once (behind an optimization barrier) before the
    # routing prep, so the [T, 4] arrays (32x lane-padded in HBM) are read once
    # instead of being relaid out for every consumer shape. Same values. 0 = off.
    "TPU_MOE_FLAT_ROUTING_MIN_TOKENS": env_nonnegative_int(
        "TPU_MOE_FLAT_ROUTING_MIN_TOKENS", 0
    ),
    # Diagnostics (layers/adapter/moe_route_trace.py): keep the per-expert
    # routed-row counts of the last N forward steps of every MoE layer in a
    # device ring buffer and log them from TP rank 0 every
    # TPU_MOE_ROUTE_TRACE_EVERY model-runner steps. Adds an index_copy per MoE
    # layer to the graph, so it is compile-hashed here. 0 = off.
    "TPU_MOE_ROUTE_TRACE": env_nonnegative_int("TPU_MOE_ROUTE_TRACE", 0),
    "TPU_MOE_ROUTE_TRACE_EVERY": env_nonnegative_int(
        "TPU_MOE_ROUTE_TRACE_EVERY", 8192
    ),
    # NVFP4 GMM MoE: give padded rows of a token bucket zero routing weights
    # (token_padding mask), so TPU_MOE_SKIP_PADDED_TOKENS drops them from the
    # grouped matmul as the FP8 path already does. Their outputs are discarded,
    # but today they route like real tokens and every expert they touch costs
    # its weight read. Needs TPU_MOE_SKIP_PADDED_TOKENS=1. Default off.
    "TPU_MOE_NVFP4_SKIP_PADDED": env_bool("TPU_MOE_NVFP4_SKIP_PADDED"),
    # NVFP4 decode MoE: run gmm_v2_mlp calls of at most <max size_m> rows as
    # one streamed kernel (kernels/megablox/gmm_decode_stream.py) that feeds
    # every hit expert's gate_up then down weights as <chunk_rows>-row k-slices
    # through <slots> VMEM slots, so one chunk (not one whole expert block per
    # gmm) is exposed per call. "<chunk_rows>:<slots>[@<max size_m>]", "on" =
    # 768:4@512. Same numerics as the fp8 rhs feed. Empty = off.
    "TPU_GMM_V2_DECODE_STREAM": env_str("TPU_GMM_V2_DECODE_STREAM", ""),
    # MSA indexer decode (transposed block_q class, folded 4-bit pages)
    # options, comma-separated: "issue" = a full page group starts its page
    # copies without a per-page live check or block-table clamp; "slots<N>" =
    # N page groups in the DMA ring (default 3); "quad" = fp8 queries and four
    # pages per MXU row (contraction 4 x 128) instead of two. "issue" and
    # "slots" are bit-identical; "quad" rounds the queries to fp8 (scores keep
    # their order up to that rounding). Empty = off.
    "TPU_MINIMAX_M3_INDEXER_DECODE_OPT": env_str(
        "TPU_MINIMAX_M3_INDEXER_DECODE_OPT", ""
    ),
    # MSA indexer prefill (token-major-reduced kernel v3 tiles, e.g. the
    # large class of 4-bit folded pages) options: "unpair" = one page per MXU
    # row against the plain query tile (contraction D) instead of two pages
    # against the zero-padded paired query (2 D): half the matmul pushes per
    # page, same products and f32 sums; "chunk<N>" = N pages per dot instead
    # of the default (fewer query weight loads per page). Empty = off.
    "TPU_MINIMAX_M3_INDEXER_PREFILL_OPT": env_str(
        "TPU_MINIMAX_M3_INDEXER_PREFILL_OPT", ""
    ),
    # TPU_MOE_SKIP_PADDED_TOKENS drops padded rows from the grouped matmul only
    # in batches of at least this many routes (T * top_k); smaller batches run
    # them as with the skip off (bitwise for real rows). 0 = every size.
    "TPU_MOE_SKIP_PADDED_MIN_ROUTES": env_nonnegative_int(
        "TPU_MOE_SKIP_PADDED_MIN_ROUTES", 0
    ),
    # Redundant MoE experts on the grouped-matmul EP path
    # (layers/adapter/moe_redundant.py): extra expert slots per EP rank per
    # MoE layer, each holding a copy of another rank's expert, and a per-step
    # router remap that moves work onto the replicas. 0 = off.
    "VLLM_TPU_MOE_REDUNDANT_EXPERTS": env_nonnegative_int(
        "VLLM_TPU_MOE_REDUNDANT_EXPERTS", 0
    ),
    # With redundant experts: after the per-step greedy copy assignment, up to
    # this many rounds of moves (one replicated expert to another copy, or a
    # two-step chain) that lower the busiest rank's expert count
    # (kernels/moe_replica_assign.py _refine). Changes which rank computes
    # an expert, not the math. 0 (default) = greedy only.
    "TPU_MOE_REPLICA_ASSIGN_REFINE": env_nonnegative_int(
        "TPU_MOE_REPLICA_ASSIGN_REFINE", 0
    ),
    # With redundant experts: the same copy assignment from a leaner kernel
    # (copy table flattened with each copy's rank at load, no per-step init
    # input or table reshape, branch-free scalar loops). Same output.
    "TPU_MOE_REPLICA_ASSIGN_FAST": env_bool("TPU_MOE_REPLICA_ASSIGN_FAST"),
    # Which experts get the redundant slots: a JSON file path, inline JSON
    # ('{"L3": [17, ...]}', hottest first), or "builtin:<name>". Empty = the
    # built-in round-robin fill. Placement reaches the compiled graph as
    # buffers, never as constants. Route-trace keys ("moe<i>", the i-th MoE
    # layer) come from `python -m vllm_torchtpu.layers.adapter.moe_redundant LOG`.
    "VLLM_TPU_MOE_REDUNDANT_PLACEMENT": env_str("VLLM_TPU_MOE_REDUNDANT_PLACEMENT", ""),
    # Capture one trace per inference phase instead of one continuous trace.
    # The trace directory and iteration limits come from `profiler_config`;
    # this only selects which profiler consumes them.
    "USE_PHASED_PROFILER": env_bool("USE_PHASED_PROFILER"),
    # Kernel-iteration mode: split the compiled graph at the Pallas custom
    # ops, keep kernel sources out of the compile-cache key, and enable the
    # /reload_kernel endpoint so kernel edits apply to a running server
    # without restarting it. Development aid; off by default.
    "TPU_KERNEL_ITER_MODE": lambda: bool(int(os.getenv("TPU_KERNEL_ITER_MODE") or "0")),
    # Comma-separated module names re-imported by /reload_kernel (and, in
    # kernel-iteration mode, excluded from the compile-cache key). Defaults
    # to the in-tree experimental RPA kernel modules; see
    # compilation/kernel_reload.py.
    "TPU_KERNEL_RELOAD_MODULES": lambda: os.getenv("TPU_KERNEL_RELOAD_MODULES", ""),
    # Greedy steps without logprobs or grammar take the vocabulary-parallel
    # argmax (runner/greedy_top_tokens.py) instead of gathering the logits.
    "TPU_GREEDY_TOP_TOKENS": env_bool("TPU_GREEDY_TOP_TOKENS"),
    # Enable TpuDpScheduler.
    "DP_SCHED_ENABLED": lambda: bool(int(os.getenv("DP_SCHED_ENABLED") or "0")),
    # Tile size for MLA transpose pipeline.
    "MLA_XPOSE_N_TILE_SIZE": lambda: int(os.getenv("MLA_XPOSE_N_TILE_SIZE", "160")),
    "MIXED_Q_SPLIT": env_optional_int("MIXED_Q_SPLIT"),
    "MIXED_NUM_QUERIES_PER_BLOCK": env_optional_int("MIXED_NUM_QUERIES_PER_BLOCK"),
    "MIXED_NUM_KV_PAGES_PER_BLOCK": env_optional_int("MIXED_NUM_KV_PAGES_PER_BLOCK"),
    # Opt in to GLM-5.2's measured masked-dense sparse-MLA prefill routing.
    # Off by default until the model and hardware scope of the cost model is
    # broadened or selected automatically from an explicit model profile.
    "TPU_MLA_MASKED_DENSE_ENABLED": env_bool(
        "TPU_MLA_MASKED_DENSE_ENABLED", default=False
    ),
    # Longest per-sequence KV length for which an enabled sparse-MLA step runs
    # the masked-dense kernel instead of the gather kernel. ANALYTIC is the
    # cheaper masked-dense tier, which computes the causal mask in registers
    # and needs its limit at or under the indexer's topk; the other is the
    # tier that materializes the CSA bitmap on SparseCore. Unset uses the
    # measured default; 0 disables that tier; any other value must be a
    # multiple of the streamed KV block size. These controls take effect only
    # when TPU_MLA_MASKED_DENSE_ENABLED=1 and only for eligible prefill; a
    # batch containing any decode always uses sparse gather attention.
    "TPU_MLA_MASKED_DENSE_ANALYTIC_MAX_KV_LEN": env_optional_int(
        "TPU_MLA_MASKED_DENSE_ANALYTIC_MAX_KV_LEN"
    ),
    "TPU_MLA_MASKED_DENSE_MAX_KV_LEN": env_optional_int(
        "TPU_MLA_MASKED_DENSE_MAX_KV_LEN"
    ),
    # Most model length cap for TPU compile bucketing
    "VLLM_TPU_MOST_MODEL_LEN": lambda: int(val)
    if (val := os.getenv("VLLM_TPU_MOST_MODEL_LEN")) is not None
    else None,
    # Skip padded tokens from MoE, otherwise they will route like
    # real tokens and activate experts whose output is thrown away.
    "TPU_MOE_SKIP_PADDED_TOKENS": env_bool("TPU_MOE_SKIP_PADDED_TOKENS"),
    # Hierarchical MoE parallelism: expert-parallel between chips,
    # tensor-parallel between the two chiplets of a chip. Off by
    # default; it helps only at low concurrency, where the MoE is
    # weight-bandwidth bound and the per-device expert load is badly
    # imbalanced. See K3_HIERARCHICAL_EP_PLAN.md.
    "TPU_MOE_HIERARCHICAL_EP": env_bool("TPU_MOE_HIERARCHICAL_EP"),
    # MoE router top-k implementation: "rowmax" (default) is the sort-free
    # k-pass selection, "sort" is torch.topk, kept as an escape hatch.
    # Router top-k kernel (`kernels/router_topk.select`): calls of at least
    # this many rows (a multiple of 128) run the transposed kernel (experts
    # on sublanes, rows on lanes: VPU reductions instead of cross-lane XLU
    # ones). 0 (default) keeps the row kernel. Same picks bit for bit.
    # MiniMax M3 fused qk norm + rope kernel: token block of calls with at
    # least this many tokens (0, default: the kernel's 256). Fewer grid
    # steps on large prefill chunks; same bits.
    "TPU_MINIMAX_M3_QK_NORM_ROPE_BLOCK_TOKENS": env_int(
        "TPU_MINIMAX_M3_QK_NORM_ROPE_BLOCK_TOKENS", 0
    ),
    # MoE router with a score-correction bias: calls of at least this many
    # rows (a multiple of 128) pick the top-k of scores + bias and gather the
    # unbiased scores in one transposed kernel pass instead of an add, the
    # top-k kernel and an XLA gather. 0 (default) keeps the unfused path.
    # Same weights and ids bit for bit.
    "TPU_ROUTER_TOPK_FUSED_BIAS_MIN_ROWS": env_int(
        "TPU_ROUTER_TOPK_FUSED_BIAS_MIN_ROWS", 0
    ),
    "TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS": env_int(
        "TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS", 0
    ),
    "TPU_MOE_ROUTER_TOPK": env_with_choices(
        "TPU_MOE_ROUTER_TOPK", "rowmax", ["sort", "rowmax"]
    ),
    # With a score-correction bias and the "rowmax" router top-k, the top-k
    # kernel also returns the unbiased scores of the picked experts, so the
    # routing weights need no separate XLA gather (a ~1.1 us op per MoE layer
    # at decode). Same values. Off by default.
    "TPU_MOE_ROUTER_TOPK_VALUES": env_bool("TPU_MOE_ROUTER_TOPK_VALUES"),
    # Keep FusedMoE top-k routing weights in float32 precision instead of casting
    # to the activation dtype (e.g. bfloat16) in fused_moe_gmm. Improves numerical
    # parity with training frameworks (e.g. MaxText) in RL and distillation workloads.
    "TPU_MOE_TOPK_WEIGHTS_FLOAT32": env_bool(
        "TPU_MOE_TOPK_WEIGHTS_FLOAT32", default=False
    ),
    # Run the whole expert-parallel MoE layer -- top-k, dispatch, both expert
    # matmuls, combine, and the cross-rank transport -- as one SPMD Pallas
    # program, instead of bracketing the per-rank compute with vLLM's
    # all-gather and reduce-scatter. The transport then overlaps the matmuls;
    # the reduce-scatter it replaces is otherwise fully exposed.
    "USE_MOE_FUSED_EP_KERNEL": lambda: (
        environment_variables["MOE_FUSED_EP_KERNEL_IMPL"]() is not None
        or env_bool("USE_MOE_FUSED_EP_KERNEL")()
    ),
    # Explicit implementation selection also enables fused EP. When unset,
    # preserve the legacy enable flag and v2 behavior. Adaptive owns its
    # admission, automatically supports FP4 and never falls back to GMM.
    "MOE_FUSED_EP_KERNEL_IMPL": env_with_choices(
        "MOE_FUSED_EP_KERNEL_IMPL", None, ["v2", "adaptive"]
    ),
    # Opt in to NVFP4 W4A8 in fused EP MoE v2. Requires the fused EP switch
    # above; defaults off so enabling fused EP alone preserves NVFP4 W4A16.
    "MOE_FUSED_EP_ENABLE_W4A8": env_bool("MOE_FUSED_EP_ENABLE_W4A8"),
    # Largest node-wide token count, out of the scheduler's cap, at or above
    # which the fused kernel is armed for the whole deployment. Decided once at
    # weight load, not per step: a per-step choice would need the token count
    # inside the traced graph, and on Qwen3.5-397B it also measured worse than
    # arming everything. Below the threshold the grouped-matmul path serves.
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS": lambda: int(
        os.getenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "1024")
    ),
    # Build each rank's two consumed routing-plan slices instead of
    # replicating the full T * topk * experts plan arithmetic on every rank.
    "MOE_FUSED_EP_V2_SHARDED_PLAN": env_bool("MOE_FUSED_EP_V2_SHARDED_PLAN"),
    # Keep the fused EP MoE mesh in EP-rank order and do the activation
    # all-gather outside the kernel's program, over a process group in
    # ascending device-id order, so the SparseCore gather gets its
    # N-dimensional algorithm without moving the program's DeviceAssignment.
    "MOE_FUSED_EP_SORTED_GATHER": env_bool("MOE_FUSED_EP_SORTED_GATHER"),
    # How many copies of the FFN body fused EP MoE v2 emits, standing in for
    # kernel.HEIGHT_RUNGS. A tile runs the shortest of them that covers its
    # live rows, so one is the ladder off. Zero leaves the choice to the
    # kernel.
    "MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE": lambda: int(
        os.getenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", "0") or 0
    ),
    # Shard Kimi K3's latent-MoE down/up projections across the cores of a
    # chip instead of replicating them. They are unquantized [7168, 3584] and
    # [3584, 7168] matmuls run at full HBM bandwidth, ~22% of decode device
    # time at TP=32; halving the bytes each core reads costs two on-package
    # all-gathers per layer. Independent of TPU_MOE_HIERARCHICAL_EP.
    "TPU_LATENT_PROJ_INTRA_CHIP_TP": env_bool("TPU_LATENT_PROJ_INTRA_CHIP_TP"),
    # Experimental: shard DSv4 attention wq_b/wo_a/wo_b across a chip's two
    # cores while KV and attention stay DP (deepseek_v4/attn_intra_chip.py).
    "TPU_DSV4_ATTN_INTRA_CHIP_TP": env_bool("TPU_DSV4_ATTN_INTRA_CHIP_TP"),
    # Slice the rotary cos_sin caches to max model len at load to
    # minimize xla layout data copy overhead. Text-only.
    "TPU_ROPE_CACHE_TRUNCATE": env_bool("TPU_ROPE_CACHE_TRUNCATE"),
    # Materialize the rotary cos_sin caches with a row-major ({1,0}) device
    # layout at load, May cost more HBM due to padding.
    "TPU_ROPE_CACHE_ROW_MAJOR": env_bool("TPU_ROPE_CACHE_ROW_MAJOR", default=False),
    "TPU_MOE_HASH_TABLE_ROW_MAJOR": env_bool(
        "TPU_MOE_HASH_TABLE_ROW_MAJOR", default=False
    ),
    # Extra token buckets added to the default list of buckets.
    "TPU_TOKEN_BUCKET_EXTRA": lambda: [
        int(v) for v in os.getenv("TPU_TOKEN_BUCKET_EXTRA", "").split(",") if v.strip()
    ],
    # Add linear token buckets above a quarter of the token budget, for
    # prompt-sized steps (platforms/tpu_platform.py _get_linear_token_paddings).
    "TPU_TOKEN_BUCKET_LINEAR": env_bool("TPU_TOKEN_BUCKET_LINEAR"),
    # Pipeline parallelism: size prefill chunks by a per-step time budget
    # instead of a token budget (core/pp_chunks.py). 0 keeps token budgets.
    "TPU_PP_DYNAMIC_CHUNKS": env_bool("TPU_PP_DYNAMIC_CHUNKS", default=True),
    # Fraction above the time of a full prefix-free step that a step may
    # take before its chunks are shortened.
    "TPU_PP_CHUNK_SLACK": lambda: float(os.getenv("TPU_PP_CHUNK_SLACK", "0.1")),
    # Pipeline parallelism over more than two stages: the engine core pushes
    # the pipeline only when stage 0 would otherwise idle and drops empty
    # steps, so requests share the pipeline (core/pp_lazy_step.py).
    "TPU_PP_LAZY_PUSH": env_bool("TPU_PP_LAZY_PUSH", default=False),
    # How pipeline stages hand activations on: "wave" runs one collective
    # launch on every stage per forward (distributed/pp_wave.py), "ring"
    # sends each forward's output point to point (distributed/pp_ring.py).
    # Goes once the ring replaces the wave.
    "TPU_PP_HANDOFF": env_with_choices("TPU_PP_HANDOFF", "wave", ["wave", "ring"]),
    # With TPU_PP_HANDOFF=ring: the forwards a stage may queue for sending
    # to the next stage before end_forward waits for room.
    "TPU_PP_RING_DEPTH": env_int("TPU_PP_RING_DEPTH", 2),
    # Enable pre-compile rotation to speed up the startup time.
    "TPU_PARALLEL_PRECOMPILE": env_bool("TPU_PARALLEL_PRECOMPILE", default=False),
    # Under TPU_PARALLEL_PRECOMPILE, take a subgraph's ranges biggest first,
    # so local rank 0 starts on the biggest one. Off keeps vLLM's order.
    "TPU_PRECOMPILE_BIGGEST_FIRST": env_bool(
        "TPU_PRECOMPILE_BIGGEST_FIRST", default=False
    ),
    # Replay a bucket another rank of the host compiled instead of compiling
    # it again: under TPU_PARALLEL_PRECOMPILE each bucket gets one owner rank
    # per host and the others wait for its artifact. Off compiles every bucket
    # on every rank, sharing only through the Tier-2 cache.
    "TPU_PRECOMPILE_PEER_REPLAY": env_bool("TPU_PRECOMPILE_PEER_REPLAY", default=False),
    # Under TPU_PRECOMPILE_PEER_REPLAY, how long a rank waits for a bucket's
    # owner to save it before compiling the bucket itself. The default fits
    # well over a single bucket compile with every rank of the host compiling
    # at once (~165 s for the biggest Qwen3.5-397B bucket at DP8); raise it for
    # a model whose biggest bucket takes longer, since a waiter that times out
    # compiles concurrently with the owner. 0 checks once, then compiles.
    "TPU_PRECOMPILE_PEER_WAIT_S": env_nonnegative_float(
        "TPU_PRECOMPILE_PEER_WAIT_S", 600.0
    ),
    # With shape-specialized backbone traces (one per token-bucket group, see
    # compilation/shape_variants.py), trace every group before compiling any
    # bucket, then compile all buckets at once across the host's ranks
    # (owners compile, peers replay). Without it each group's buckets compile
    # before the next group is even traced, so the groups run one after
    # another and most ranks wait. Needs TPU_PARALLEL_PRECOMPILE.
    "TPU_PRECOMPILE_TRACE_FIRST": env_bool("TPU_PRECOMPILE_TRACE_FIRST", default=False),
    # Token-axis chunk size for the GLM-5.2 / DSv4 StreamIndex top-k kernel.
    # Cutting the flat token axis into `chunk_tokens`-sized pieces lets each
    # chunk's SparseCore top-k overlap the next chunk's TensorCore scoring.
    # Unset means unchunked: one single pass, no pipelining.
    "TPU_STREAMIDX_CHUNK_TOKENS": env_positive_int_or_none(
        "TPU_STREAMIDX_CHUNK_TOKENS"
    ),
    # Score full speculative verify windows (1 + K tokens) in the attention
    # kernels' static query-length pass instead of the mixed pass. Only takes
    # effect for models whose ops implement that pass (DeepSeek-V4); 0 keeps
    # every verify window on the mixed pass.
    "TPU_SPEC_STATIC_VERIFY": env_bool("TPU_SPEC_STATIC_VERIFY", default=True),
    # Under synthetic rejection sampling, decide acceptance before the LM head
    # and compute logits and sample on one row per request (its first
    # rejected slot or its bonus slot) instead of every verify row.
    "TPU_SPEC_SYNTHETIC_ONE_ROW_VERIFY": env_bool(
        "TPU_SPEC_SYNTHETIC_ONE_ROW_VERIFY", default=True
    ),
    # Extra DFlash/DSpark draft-forward buckets above 64 tokens, this many
    # apart (`DFlashProposer._fine_draft_buckets`; 0: the target's token
    # buckets only). A draft is num_reqs * block tokens, which the target's
    # coarse buckets pad by up to ~45 %.
    "TPU_DRAFT_BUCKET_STEP": env_nonnegative_int("TPU_DRAFT_BUCKET_STEP", 32),
    # Under the synthetic one-row verify, split the requests' rows over the TP
    # ranks: an all-to-all of the LM head shards instead of an all-gather of
    # the full logits, each rank samples its block, the tokens are gathered.
    # Needs the model's compute_logits_shard; same tokens.
    "TPU_SPEC_SYNTHETIC_ROW_SPLIT": env_bool(
        "TPU_SPEC_SYNTHETIC_ROW_SPLIT", default=True
    ),
    # Run the async spec-decode glue between the draft and the next forward
    # (substitution source, rejected count, the next step's seq_lens and
    # positions correction) as two compiled programs instead of about eight
    # eager ones. Integer ops only; same values.
    "TPU_SPEC_GLUE_FUSION": env_bool("TPU_SPEC_GLUE_FUSION", default=False),
    # Gather the non-greedy verify's bonus and target logits rows with one
    # JAX op (in-bounds gathers) instead of torch index_select, which lowers
    # to an out-of-bounds-filling gather that reassembles the [T, vocab]
    # rows in vocab chunks (pad + maximum): 54 -> 9 us at T=48. Same values:
    # the indices are always in bounds (padding rows use 0).
    "TPU_SPEC_GATHER_IN_BOUNDS": env_bool("TPU_SPEC_GATHER_IN_BOUNDS", default=False),
    # Under the synthetic one-row verify, draw the accept and recovery
    # uniforms inside the two compiled sampler programs from a device counter
    # (keyed counter-based hash, 24-bit uniforms like torch.rand) instead of
    # two eager torch.rand calls on the sampling generator. Same distribution,
    # different random stream.
    "TPU_SPEC_SYNTHETIC_INGRAPH_RNG": env_bool(
        "TPU_SPEC_SYNTHETIC_INGRAPH_RNG", default=False
    ),
    # With TPU_SPEC_SYNTHETIC_INGRAPH_RNG, run a decode-only step's synthetic
    # one-row verify (accept decisions, LM head on the chosen rows, sampling
    # and assembly) as one compiled program instead of three. Same functions
    # in the same order; the step's logits stay inside the program.
    "TPU_SPEC_SYNTHETIC_HEAD_FUSION": env_bool(
        "TPU_SPEC_SYNTHETIC_HEAD_FUSION", default=False
    ),
    # Async spec, steady decode: when the next step's input_ids are the first
    # rows of the parked substitution source (fast token substitution's
    # identity case), build them and re-source the verify's draft ids in one
    # compiled program instead of an eager slice plus an eager gather.
    "TPU_SPEC_SUBST_EXTRACT_FUSION": env_bool(
        "TPU_SPEC_SUBST_EXTRACT_FUSION", default=False
    ),
    # Host critical path of the decode loop (all default off, same tokens).
    # Runner stage timings record host time only: no TpuEvent pair per stage
    # (each record is ~30 us of host time; about 8 stages per decode cycle).
    # The device-side stage samples (KV connector stats only) are dropped.
    "TPU_HOST_STAGE_TIMING_HOST_ONLY": env_bool(
        "TPU_HOST_STAGE_TIMING_HOST_ONLY", default=False
    ),
    # Skip the token-padding mask upload (a host alloc plus a synchronous
    # H2D copy, twice per spec-decode cycle: target forward and drafter)
    # when the mask already holds the same (valid, padded) rows.
    "TPU_HOST_TOKEN_PADDING_REUSE": env_bool(
        "TPU_HOST_TOKEN_PADDING_REUSE", default=False
    ),
    # Reuse the previous upload of the padded sampling params (temperature,
    # top-k, top-p) when the batch's values are unchanged.
    "TPU_HOST_SAMPLING_PARAMS_REUSE": env_bool(
        "TPU_HOST_SAMPLING_PARAMS_REUSE", default=False
    ),
    # After this many consecutive execute_model steps with no new Dynamo
    # graph, set torch.compiler stance skip_guard_eval_unsafe: Dynamo then
    # runs only the guards that ever told its cache entries apart. A later
    # recompile raises instead of compiling, so this needs every program
    # warm. 0 = off.
    "TPU_HOST_SKIP_GUARD_EVAL_AFTER_STEPS": env_nonnegative_int(
        "TPU_HOST_SKIP_GUARD_EVAL_AFTER_STEPS", 0
    ),
    # Keep compiled graphs' input mutations (kv/index cache writes) in the
    # graph instead of AOTAutograd's eager copy_ epilogue (120 host copy_ per
    # MiniMax forward). 1 = keep_inference_input_mutations; 2 = also donate
    # the written placeholders to torch_tpu's StaticCompiler. 0 = off. Changes
    # every compiled program (and the compile cache key). BROKEN on the
    # pinned torch_tpu: exos j-d93796de lost every in-graph cache write in
    # both modes (caches_equal 0/120). Kept only for a future torch_tpu.
    "TPU_HOST_KEEP_INPUT_MUTATIONS": env_nonnegative_int(
        "TPU_HOST_KEEP_INPUT_MUTATIONS", 0
    ),
    # Run the DSpark greedy Markov chain over the vocab-parallel draft logits
    # as one Pallas call (kernels/collectives/markov_argmax.py) instead of K
    # steps of XLA matvec + argmax + all-gather. Same ids.
    "TPU_DSPARK_MARKOV_KERNEL": env_bool("TPU_DSPARK_MARKOV_KERNEL", default=False),
    # Merge the DCP indexer candidates with an all-to-all over the token axis
    # instead of an all-gather over the candidate axis. Same result.
    "TPU_DCP_INDEXER_A2A_MERGE": env_bool("TPU_DCP_INDEXER_A2A_MERGE", default=False),
    # Smallest padded token bucket that takes the all-to-all merge when
    # TPU_DCP_INDEXER_A2A_MERGE is on. The bucket is static, so each compiled
    # variant picks one merge with no runtime cost.
    "TPU_DCP_INDEXER_A2A_MIN_TOKENS": env_nonnegative_int(
        "TPU_DCP_INDEXER_A2A_MIN_TOKENS", 64
    ),
    # MIXED-pass query block for StreamIndex top-k token buckets of at most
    # TPU_STREAMIDX_SMALL_BUCKET_TOKENS. Under MTP every decode request has
    # q_len = gamma + 1 and runs in the MIXED pass, whose default 128-row block
    # scores ~124 padding rows per request. 4 is the recommended value. Unset
    # keeps 128 everywhere.
    "TPU_STREAMIDX_SMALL_BUCKET_BQ": env_positive_int_or_none(
        "TPU_STREAMIDX_SMALL_BUCKET_BQ"
    ),
    # MIXED-pass KV pages per block for those buckets; unset takes 8. Read only
    # when TPU_STREAMIDX_SMALL_BUCKET_BQ is set.
    "TPU_STREAMIDX_SMALL_BUCKET_BKVP": env_positive_int_or_none(
        "TPU_STREAMIDX_SMALL_BUCKET_BKVP"
    ),
    # Largest padded token bucket that takes the small-bucket block sizes. A
    # small query block re-reads a sequence's keys once per block, so one long
    # q_len row in the bucket erodes the win: at 64 a lone 60-token tail is at
    # parity with the 128-row block.
    "TPU_STREAMIDX_SMALL_BUCKET_TOKENS": env_int(
        "TPU_STREAMIDX_SMALL_BUCKET_TOKENS", 64
    ),
    # Indexer K-cache writer and layout mode for GLM-5.2 / DSv4 StreamIndex:
    #   "sal" (default): Pallas coalesced_kv_append(_dcp) in SEQ_ALONG_LANE layout
    #     (falls back to "has" when block_size < 128 or when Raiden GLM admission
    #     is active with the default mode; an explicit "sal" with Raiden GLM
    #     admission raises ValueError).
    #   "has": Pallas coalesced_kv_append(_dcp) in HEAD_ALONG_SUBLANE layout.
    #   "xla": XLA .at[].set scatter in HEAD_ALONG_SUBLANE layout.
    "VLLM_TPU_INDEXER_KV_CACHE_MODE": lambda: (
        env_with_choices(
            "VLLM_TPU_INDEXER_KV_CACHE_MODE",
            DEFAULT_INDEXER_KV_CACHE_MODE,
            ["xla", "has", "sal"],
            case_sensitive=False,
        )()
        or DEFAULT_INDEXER_KV_CACHE_MODE
    ).lower(),
    "TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER": env_bool(
        "TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER", True
    ),
    # Query-token chunk size for sparse MLA attention (GLM-5.2 / DSv3.2-style):
    # each chunk gathers its tokens' top-k KV rows into one buffer and then
    # attends over it. The chunk loop is unrolled at trace time, so a graph
    # holds `num_tokens / chunk` gather + attention kernel pairs per layer;
    # bigger chunks compile faster but need a bigger gather buffer
    # (chunk * top-k rows) and pipeline gather with attention more coarsely.
    # Unset keeps the kernel default (64).
    "TPU_SPARSE_MLA_CHUNK_TOKENS": env_positive_int_or_none(
        "TPU_SPARSE_MLA_CHUNK_TOKENS"
    ),
    # Post-gather MoE token chunk size for communication-computation pipelining.
    # When set to 0, chunking and communication pipelining are disabled.
    "TPU_MOE_COLLECTION_CHUNK_SIZE": lambda: int(
        os.getenv("TPU_MOE_COLLECTION_CHUNK_SIZE") or "0"
    ),
    # Row-quantize MoE activations to fp8 before the DP/PCP all-gather and
    # restore them on arrival, so the gather moves one byte per element
    # instead of two while everything downstream of it is unchanged. Costs one
    # rounding of the activations plus a second collective for the row scales;
    # only takes effect on the grouped-matmul MoE path (the fused EP kernel
    # already carries fp8 on its own wire).
    "TPU_MOE_FP8_ALL_GATHER": env_bool("TPU_MOE_FP8_ALL_GATHER"),
    # 0 = the arithmetic device-order layout (the one every disaggregated
    # Raiden pair was verified with before #587; also the only option on a
    # torch_tpu older than 2026-08-13, which lacks the API).
    "TPU_PCP_TOPOLOGY_AWARE_MESH": lambda: os.getenv("TPU_PCP_TOPOLOGY_AWARE_MESH", "1")
    .strip()
    .lower()
    not in ("0", "false", "off", "no"),
    # Pipelined double-buffered DMA staging in ragged_kv_cache_update and fused
    # query DMA layout in RPA v3.
    "USE_RPA_PIPELINED_DMA_STAGING": lambda: env_bool("USE_RPA_PIPELINED_DMA_STAGING")()
    or env_bool("USE_PIPELINED_DMA_STAGING")(),
    "USE_PIPELINED_DMA_STAGING": lambda: env_bool("USE_PIPELINED_DMA_STAGING")()
    or env_bool("USE_RPA_PIPELINED_DMA_STAGING")(),
    # Enables the block-major KV cache layout for Raiden offloading. Bundles all
    # attention layer fragments into a single contiguous array in HBM, collapsing
    # per-block transfers from F independent DMAs into a single hardware DMA.
    # Refer to `vllm_torchtpu.offload.block_major_layout` for contract details.
    "VLLM_TPU_BLOCK_MAJOR_KV": env_bool("VLLM_TPU_BLOCK_MAJOR_KV"),
    # Evict host page cache for checkpoint weight files via posix_fadvise
    # after loading into TPU HBM.
    "TPU_EVICT_WEIGHTS_PAGE_CACHE": env_bool(
        "TPU_EVICT_WEIGHTS_PAGE_CACHE", default=False
    ),
    # KV transfer / Raiden knobs. Each getter in distributed/utils.py
    # documents what its knob does; this is the parsing and the default.
    # TCP port the KV wire transfer listens on.
    "TPU_KV_TRANSFER_PORT": env_str("TPU_KV_TRANSFER_PORT", "9100"),
    # TCP port for the out-of-band side channel.
    "TPU_SIDE_CHANNEL_PORT": env_str("TPU_SIDE_CHANNEL_PORT", "9600"),
    # Host ordinal within a multi-host slice.
    "TPU_NODE_ID": env_int("TPU_NODE_ID", 0),
    # 0 means auto; a non-zero override is clamped to tp_size downstream.
    "TPU_KV_TRANSFER_CHANNEL_NUMBER": env_int("TPU_KV_TRANSFER_CHANNEL_NUMBER", 0),
    # Request-level pull executor size; 0 means auto.
    "TPU_KV_COORD_EXECUTOR_MAX_WORKERS": env_nonnegative_int(
        "TPU_KV_COORD_EXECUTOR_MAX_WORKERS", 0
    ),
    # Channel-level pull executor size; 0 means auto.
    "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS": env_nonnegative_int(
        "TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS", 0
    ),
    # Worker count for the stage waiter pool; 0 means auto (4).
    "TPU_KV_STAGE_WAITER_POOL_SIZE": env_nonnegative_int(
        "TPU_KV_STAGE_WAITER_POOL_SIZE", 0
    ),
    # KV-cache transfer timeout, in seconds.
    "TPU_P2P_WAIT_PULL_TIMEOUT": env_int("TPU_P2P_WAIT_PULL_TIMEOUT", 120),
    # Seconds; 0 disables the probe.
    "TPU_RAIDEN_STAGE3_STATUS_PROBE_S": env_float(
        "TPU_RAIDEN_STAGE3_STATUS_PROBE_S", 1.0
    ),
    # On by default; inline-load mode overrides it back to off.
    "TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT": env_bool(
        "TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT", default=True
    ),
    # On by default: a PCP, TP or DP Stage-3 producer registers a request's
    # KV blocks in the step that completes its prompt rather than in the step
    # after its finish response (pipeline producers always do).
    "TPU_RAIDEN_STAGE3_REGISTER_AHEAD": env_bool(
        "TPU_RAIDEN_STAGE3_REGISTER_AHEAD", default=True
    ),
    # Seconds, bounded by TPU_P2P_WAIT_PULL_TIMEOUT downstream.
    "TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S": env_float(
        "TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S", 30.0
    ),
    # TEST-ONLY fault injection, in seconds; 0 disables it.
    "TPU_RAIDEN_TEST_REGISTRATION_DELAY_S": env_float(
        "TPU_RAIDEN_TEST_REGISTRATION_DELAY_S", 0.0
    ),
    # Seconds; non-positive falls back to 30 downstream.
    "TPU_KV_STAGE_WAIT_TIMEOUT_SECS": env_float("TPU_KV_STAGE_WAIT_TIMEOUT_SECS", 30.0),
    # Total GB budget for the per-host shared-memory KV staging pool.
    "TPU_KV_SHM_POOL_GB": env_float("TPU_KV_SHM_POOL_GB", 128.0),
    # Optional engine namespace; sanitized at the call site.
    "TPU_KV_TRANSFER_NAMESPACE": env_str("TPU_KV_TRANSFER_NAMESPACE", ""),
    # Directory holding the per-host intra-process IPC socket.
    "TPU_IPC_SOCKET_DIR": env_str("TPU_IPC_SOCKET_DIR", "/tmp"),
    # Pre-compile KV gather/scatter executables at startup.
    "TPU_KV_WARMUP_ENABLED": env_bool("TPU_KV_WARMUP_ENABLED", default=True),
    # Seconds; 0 disables the periodic summaries.
    "TPU_KV_LATENCY_LOG_INTERVAL": env_float("TPU_KV_LATENCY_LOG_INTERVAL", 30.0),
    # mlock(2) the KV shm pool at startup.
    "TPU_KV_PIN_SHM": env_bool("TPU_KV_PIN_SHM", default=False),
    # kv_connector_extra_config.use_raiden_connector takes precedence.
    "TPU_USE_RAIDEN_CONNECTOR": env_bool("TPU_USE_RAIDEN_CONNECTOR", default=False),
    # 0 means auto-size from TPU_KV_SHM_POOL_GB, split across TP ranks.
    "TPU_RAIDEN_TRANSFER_NUM_SLOTS": env_nonnegative_int(
        "TPU_RAIDEN_TRANSFER_NUM_SLOTS", 0
    ),
    # Concurrent pool-reshard transfers to provision bounded host staging for.
    "TPU_RAIDEN_POOL_STAGING_LEASES": env_nonnegative_int(
        "TPU_RAIDEN_POOL_STAGING_LEASES", 8
    ),
    # Load remote KV before the first forward, not in a no-forward step.
    "TPU_RAIDEN_INLINE_LOAD": env_bool("TPU_RAIDEN_INLINE_LOAD", default=False),
    # Comma-separated dot-bounded module prefix names to quantize to FP8 on load.
    # Examples: QUANTIZE_ON_LOAD_PREFIXES="self_attn,shared_experts,layers.0.mlp"
    "QUANTIZE_ON_LOAD_PREFIXES": lambda: [
        p.strip()
        for p in os.getenv("QUANTIZE_ON_LOAD_PREFIXES", "").split(",")
        if p.strip()
    ],
    # Kimi K3 KDA kernel DMA and performance tuning knobs.
    "KDA_MANUAL_STATE_DMA": env_optional_bool("KDA_MANUAL_STATE_DMA"),
    "KDA_MANUAL_H0_DMA": env_optional_bool("KDA_MANUAL_H0_DMA"),
    "KDA_MANUAL_HT_DMA": env_optional_bool("KDA_MANUAL_HT_DMA"),
    "KDA_OVERLAP_H0_DMA": env_bool("KDA_OVERLAP_H0_DMA", default=True),
    "KDA_OVERLAP_HT_DMA": env_bool("KDA_OVERLAP_HT_DMA", default=True),
    "KDA_PACK_HEAD_INV": env_bool("KDA_PACK_HEAD_INV", default=True),
    "KDA_PACKED_METADATA": env_bool("KDA_PACKED_METADATA", default=True),
    "KDA_FWD_MB": env_optional_int("KDA_FWD_MB"),
    # Startup and host-side knobs. None of them change what gets compiled;
    # each is also listed in _TPU_COMPILE_ENV_IGNORED so it stays out of the
    # compile-cache key.
    # Run the real spec-decode dispatch once at startup so the first request
    # does not pay for those compiles. Off skips that warmup.
    "SPEC_WARMUP": env_bool("SPEC_WARMUP", default=True),
    # Raiden library switch, on by default. The TPU offloading connector
    # cannot run next to the raiden singleton worker, so it rejects an
    # explicit off and exports the variable as on for the raiden library.
    "RAIDEN_DISABLE_SINGLETON_WORKER": env_bool(
        "RAIDEN_DISABLE_SINGLETON_WORKER", default=True
    ),
    # The raiden library's shared-memory key, a string naming the segment
    # behind its shm-backed host pools. The TPU offloading connector cannot
    # serve those pools, so it refuses to start when the key is set at all.
    # Read raw, not through env_str: the check must see the same bytes the
    # raiden library sees, and a whitespace-only key still counts as set.
    "RAIDEN_SHM_KEY": lambda: os.getenv("RAIDEN_SHM_KEY", ""),
    # Directory for the TPU offloading connector's storage tier, a third
    # tier behind host DRAM. Every block the connector offloads to DRAM is
    # also written under this directory. "" (the default) turns the tier off.
    # The raiden_storage_path key in kv_connector_extra_config wins over it.
    "RAIDEN_STORAGE_PATH": env_str("RAIDEN_STORAGE_PATH", ""),
    # Storage backend type for RAIDEN_STORAGE_PATH; "" means "posix", the
    # only type supported today.
    "RAIDEN_STORAGE_TYPE": env_str("RAIDEN_STORAGE_TYPE", ""),
    # Storage capacity in GiB, reported to the raiden store; "" leaves it
    # unset. Kept as a string here; raiden_store.py parses and validates it.
    "RAIDEN_STORAGE_CAPACITY_GB": env_str("RAIDEN_STORAGE_CAPACITY_GB", ""),
    # Further storage backend properties, passed to raiden unread as
    # comma-separated key=value pairs, e.g.
    # "direct_io=true,metadata_cache_max_entries=4096". The store's metadata
    # cache is tuned this way too: metadata_cache_max_entries (raiden's
    # default, 0, turns it off; -1 removes the limit) and
    # metadata_cache_ttl_secs (60 by default; -1 for no expiry). The
    # raiden_storage_properties key in kv_connector_extra_config wins key by
    # key.
    "RAIDEN_STORAGE_PROPERTIES": env_str("RAIDEN_STORAGE_PROPERTIES", ""),
    # Experimental, for testing only: lets the storage tier start under
    # pipeline parallelism, which is refused by default until end-to-end
    # validation completes. The
    # experimental_raiden_storage_skip_parallelism_check key in
    # kv_connector_extra_config wins over it.
    "EXPERIMENTAL_RAIDEN_STORAGE_SKIP_PARALLELISM_CHECK": env_bool(
        "EXPERIMENTAL_RAIDEN_STORAGE_SKIP_PARALLELISM_CHECK", default=False
    ),
    # Seconds; upper bound on a scheduler-side drain of in-flight offload
    # jobs whose blocks are about to be reused.
    "VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S": env_float(
        "VLLM_TPU_OFFLOAD_WAIT_TIMEOUT_S", 30.0
    ),
    # How many times an offload save whose transfer failed is retried.
    "VLLM_TPU_OFFLOAD_SAVE_RETRIES": env_int("VLLM_TPU_OFFLOAD_SAVE_RETRIES", 1),
    # MiB of one worker's KV that offload loads may have in flight at once
    # (0: no bound); the step's input uploads queue behind them.
    "VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_MIB": env_int(
        "VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_MIB", 128
    ),
    # MiB of one worker's KV that offload saves may have in flight at once
    # (0: no bound); the next forward's KV write waits for them.
    "VLLM_TPU_OFFLOAD_MAX_INFLIGHT_STORE_MIB": env_int(
        "VLLM_TPU_OFFLOAD_MAX_INFLIGHT_STORE_MIB", 64
    ),
    # Raiden copies (runs of host- and HBM-consecutive sub-blocks) offload
    # loads may have in flight at once (0: no bound); each holds the device
    # queue the step's programs wait in. Applies only with the MiB bound on.
    "VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS": env_int(
        "VLLM_TPU_OFFLOAD_MAX_INFLIGHT_LOAD_RUNS", 0
    ),
    # Raiden KV offload: refresh a request's host-resident chain head last
    # (`RaidenOffloadingManager.refresh_resident`), so a live conversation's
    # oldest chunks are not evicted first. Write-through refreshes when the
    # request finishes; write-back when it flushes the held request (its host
    # copy only matters once HBM evicts it). Scheduler-side, host-only.
    # Same refresh as TPU_RAIDEN_LRU_REFRESH=2, but timed by the offload mode.
    "VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH": env_bool("VLLM_TPU_OFFLOAD_TOUCH_ON_FINISH"),
    # Raiden host-pool LRU refresh of chains a request already finds resident.
    # The store skips a resident key without touching it, and a key served
    # from HBM is never looked up on the host, so the head of a long-lived
    # prefix ages out of the host LRU first and makes the whole chain
    # unreachable. 0: off. 1: refresh resident keys a store skips (the
    # flush that runs right before HBM evicts the chain). 2: also refresh
    # the finished request's resident leading chain (vLLM's CPU offload
    # manager touches at request finish). Scheduler-side, host-only.
    "TPU_RAIDEN_LRU_REFRESH": env_int("TPU_RAIDEN_LRU_REFRESH", 0),
    # Host-heap diagnostics at worker start-up (worker/host_heap_report.py):
    # 1 = RSS / tcmalloc / anon mappings, 2 = + Python heap census,
    # 3 = + measure dropping start-up-only references. Host-only.
    # 4 = + per-precompile-pass heap and executable counts, and which of the
    # compiled executables serving calls (logged TPU_HOST_HEAP_USAGE_AFTER_S
    # seconds after warm-up, then every 10 min). Host-only.
    "TPU_HOST_HEAP_REPORT": env_int("TPU_HOST_HEAP_REPORT", 0),
    "TPU_HOST_HEAP_USAGE_AFTER_S": env_int("TPU_HOST_HEAP_USAGE_AFTER_S", 900),
    # Spec decode with synthetic acceptance (SYNTHETIC_ACCEPTANCE_LENGTH, perf
    # runs): warm only the synthetic one-row verify at each rejection-sampler
    # bucket, not the full greedy / non-greedy rejection-sampler graphs that
    # such steps never run. Saves their compile time and host memory; a step
    # that needs the full path (logprobs, grammar, min_tokens) compiles it on
    # first use. No effect on real-acceptance runs.
    "TPU_SPEC_PRECOMPILE_SYNTHETIC_VERIFY_ONLY": env_bool(
        "TPU_SPEC_PRECOMPILE_SYNTHETIC_VERIFY_ONLY", default=False
    ),
    # Which sampler variants of the synthetic one-row verify
    # (synthetic_sample_and_assemble[_rng], synthetic_verify_head) warm-up
    # compiles at each rejection-sampler bucket. Each step picks one of
    # `full` (top-k and top-p), `no_topk_topp` (no request sets either),
    # `top_p` (top-p set, no top-k) and `topk_candidates` from its requests'
    # sampling params. Empty (default): all of them. `auto`: the one a batch
    # of requests on the model's default sampling params (generation config)
    # picks. Otherwise a comma list of names. A step that picks a variant not
    # warmed compiles it on first use (logged once). Also warms the verify
    # row selection on its runtime inputs. Host memory only; outputs
    # unchanged.
    "TPU_SPEC_PRECOMPILE_SAMPLER_VARIANTS": env_str(
        "TPU_SPEC_PRECOMPILE_SAMPLER_VARIANTS", ""
    ),
    # Drop JAX's tracing / lowering caches once warm-up has compiled every
    # bucket. The Pallas kernels are already lowered into the compiled
    # programs; the cached jaxprs only serve a re-trace. Frees ~0.9 GiB host
    # per worker on MiniMax-M3.1 c48 (~7 GiB per host). Host-only.
    "TPU_RELEASE_JAX_CACHES": env_bool("TPU_RELEASE_JAX_CACHES", default=False),
    # Periodic Raiden offload counters (stores offered/skipped/stored,
    # refreshed entries, cancelled stores, write-back drops by cause, host
    # lookups that miss a chain head while a later key is resident).
    "TPU_RAIDEN_OFFLOAD_STATS": env_bool("TPU_RAIDEN_OFFLOAD_STATS", default=False),
    # Flush (cancel a fence-pending / drain a launched) offload store only for
    # blocks the block pool really handed out fresh this step. vLLM's offload
    # scheduler counts a new request's prefix-HIT blocks as allocated too, so a
    # prefix hit on a chain whose store is still parked cancels the whole store
    # (the chain never reaches the host) or drains it synchronously. A hit
    # block keeps its contents, so reading it later is safe. Scheduler-side.
    "TPU_RAIDEN_FLUSH_REAL_REUSE": env_bool(
        "TPU_RAIDEN_FLUSH_REAL_REUSE", default=False
    ),
    # Diagnostics: remember each offload key's last store/cancel/evict event
    # and, for each request whose host lookup stops >= 8k tokens short of its
    # prompt, log why the first missing key is missing; time store drains.
    "TPU_RAIDEN_LOSS_TRACE": env_bool("TPU_RAIDEN_LOSS_TRACE", default=False),
    # Write-back: a held (finished, not yet stored) request is dropped
    # unstored only when the pool reallocates one of its blocks or one step's
    # prefix hits cover all of its blocks (the conversation's next turn now
    # holds the chain), not on any prefix hit. A subagent hitting a parent's
    # shared prefix no longer drops the parent's chain. Scheduler-side.
    "TPU_RAIDEN_WB_DROP_COVERED": env_bool(
        "TPU_RAIDEN_WB_DROP_COVERED", default=False
    ),
    # Scheduler: keep the trailing prefix-cache block that EAGLE/MTP normally
    # drops (controls vLLM's disable_eagle_block_drop). Host-only.
    "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP": env_bool(
        "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", default=False
    ),
    # Secret for the KV-transfer IPC socket; "" derives one from the host.
    # Read raw, not through env_str: a secret's bytes must not be trimmed.
    "VLLM_TORCHTPU_IPC_KEY": lambda: os.getenv("VLLM_TORCHTPU_IPC_KEY", ""),
    # Sync the device every N tensors during sharded EP weight loading so
    # queued host copies are released; 0 disables the periodic sync.
    "TPU_SHARDED_LOAD_SYNC_EVERY": env_int("TPU_SHARDED_LOAD_SYNC_EVERY", 512),
    # Dequantize and pad MXFP4 expert weights on the TPU, where they already
    # sit after loading, instead of round-tripping them through host memory.
    "VLLM_TPU_MXFP4_DEVICE_DEQUANT": env_bool("VLLM_TPU_MXFP4_DEVICE_DEQUANT", True),
    # Log the PCP sequence layout window on every step.
    "VLLM_TPU_DEBUG_PCP_LAYOUT": env_bool("VLLM_TPU_DEBUG_PCP_LAYOUT"),
    # Added to every worker's local rank when it binds to a physical chip.
    # worker/tpu_rank_binding.py reads it from the env mapping it is given.
    "TPU_LOCAL_RANK_OFFSET": env_int("TPU_LOCAL_RANK_OFFSET", 0),
    # Physical device index per local rank ("0,2,1,3,4,6,5,7"); empty = native.
    "VLLM_TPU_DEVICE_ORDER": lambda: os.getenv("VLLM_TPU_DEVICE_ORDER", ""),
    # Same offset for the legacy worker init path; debug only.
    "DEBUG_TPU_LOCAL_RANK_OFFSET": env_int("DEBUG_TPU_LOCAL_RANK_OFFSET", 0),
    # First slicebuilder port; each chip on a host takes base + local rank.
    "TORCH_TPU_BASE_PORT": env_int("TORCH_TPU_BASE_PORT", 8070),
    # TCPStore port for the mp multihost rendezvous; unset means the vLLM
    # coordination port plus one.
    "TORCH_TPU_MP_RENDEZVOUS_PORT": env_optional_int("TORCH_TPU_MP_RENDEZVOUS_PORT"),
    # Directory for the on-disk TorchTPU compilation cache; read raw because
    # env_override.py copies it verbatim into the TORCH_TPU_INTERNAL_* vars.
    "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": lambda: os.getenv(
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", ""
    ),
    # SparseCore offload threshold in bytes; None means "auto" and
    # env_override.py resolves it from the chip family. 0 sets no flag.
    "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES": env_nonnegative_int_or_auto(
        "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES"
    ),
    # Enables steady-state fast path token substitution (direct device slicing
    # and persistent index caching) in TPUModelRunner._apply_async_token_substitution.
    "VLLM_TPU_FAST_TOKEN_SUBSTITUTION": env_bool("VLLM_TPU_FAST_TOKEN_SUBSTITUTION"),
    # Run an MTP draft's loop steps 1..K-1 as one compiled program
    # (spec_decode/mtp_fused_loop.py). Off by default: one program per step.
    # TODO: support the two cases that still take the per-step loop:
    #   - GLM-5.2 / Gemma-4 MTP: IndexShare's per-chunk top-k restore, and a
    #     KV-sharing draft's fixed positions and seq_lens.
    #   - EP-DP lockstep: run_dp_dummy_draft only replays the per-step
    #     collective trace.
    "VLLM_TPU_FUSED_DRAFT_LOOP": env_bool("VLLM_TPU_FUSED_DRAFT_LOOP"),
    # Non-greedy sampling filters the vocab to this many 128-logit blocks,
    # ranked by block max, before running top-k (and any subsequent top-p).
    # 0 searches the full vocab. Used only when every active request has
    # 1 <= top_k <= value // 2 and no grammar mask applies.
    "TPU_SAMPLER_TOPK_CANDIDATES": env_nonnegative_int(
        "TPU_SAMPLER_TOPK_CANDIDATES", 0
    ),
    # Top-p sampling finds its probability threshold from candidate blocks
    # (layers/adapter/sample/topp_candidates.py): comma list of tiers in
    # 128-logit blocks, e.g. "64,512". Each tier is used only when its blocks
    # provably hold the whole top-p set of every row, else the next tier, and
    # finally the full-vocabulary bisection. Same threshold up to the float32
    # summation order. Empty (default): the full bisection only.
    "TPU_SAMPLER_TOPP_CANDIDATES": env_str("TPU_SAMPLER_TOPP_CANDIDATES", ""),
    # Store the LM head (shared with the draft) as e4m3 with a per-row scale,
    # halving the bytes every logits call reads (`layers.adapter.lm_head_e4m3`).
    # Not exact: logits carry the e4m3 rounding of the weight.
    "TPU_LM_HEAD_E4M3": env_bool("TPU_LM_HEAD_E4M3", False),
    # With TPU_LM_HEAD_E4M3: logits calls of at most this many rows split the
    # activations into two e4m3 terms and run one fp8 x fp8 matmul (the weight
    # is read once, in e4m3, by the fp8 MXU); larger calls keep the bf16
    # matmul on the upcast weight. 0 = off.
    "TPU_LM_HEAD_E4M3_SPLIT_MAX_ROWS": env_nonnegative_int(
        "TPU_LM_HEAD_E4M3_SPLIT_MAX_ROWS", 0
    ),
    # Debug only, with TPU_LM_HEAD_E4M3: rank 0 saves its bf16 head shard and,
    # for this many decode steps, the LM head inputs under
    # $CDK_OUTPUT_DIR/lm_head_dump. 0 = off.
    "TPU_LM_HEAD_E4M3_DUMP": env_nonnegative_int("TPU_LM_HEAD_E4M3_DUMP", 0),
    # Chunk the vision encoder over video frames to bound peak VMEM: encode at
    # most this many patches (t*h*w) per vision forward, then concatenate the
    # per-chunk outputs. 0 disables (encode the whole video in one batch).
    "MM_ENCODER_FRAME_CHUNK_PATCH_SIZE": env_nonnegative_int(
        "MM_ENCODER_FRAME_CHUNK_PATCH_SIZE", 0
    ),
    # Pooling models: run LAST/CLS/MEAN embed pooling through the AOT-compiled
    # static pooler over the (token, request) bucket grid. 0 falls back to the
    # eager vLLM pooler (finished-mask gating still applies).
    "VLLM_TPU_STATIC_POOLER": env_bool("VLLM_TPU_STATIC_POOLER", True),
    # Pooling models: KV cache token budget. -1 (default) derives a
    # deterministic cap from the scheduler config, 0 disables the cap (all
    # post-profile HBM goes to KV), N > 0 caps KV at N tokens.
    "VLLM_TPU_POOLING_KV_CACHE_TOKENS": env_int("VLLM_TPU_POOLING_KV_CACHE_TOKENS", -1),
    # SrptPrefillScheduler: consecutive steps a running prefill chunk may go
    # without getting one KV block of tokens before it moves ahead of the
    # other prefill chunks and is no longer held (0 disables hold-back).
    # Prefixed with VLLM_ so vLLM's Ray worker env propagation
    # copies it automatically to DP/Ray engine-core actors.
    "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS": env_nonnegative_int(
        "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS", 8
    ),
    # SrptPrefillScheduler: cap on a step's attention work, sum of
    # n * (context + n / 2) over its scheduled chunks; while some DP rank
    # decodes, prefill chunks are shrunk to stay under it (0 disables).
    "VLLM_TPU_SRPT_ATTN_WORK_CAP": env_nonnegative_int(
        "VLLM_TPU_SRPT_ATTN_WORK_CAP", 0
    ),
    # SrptPrefillScheduler: a chunk shrunk by the attention work cap ends on a
    # multiple of this many tokens, rounded down (the attention kernel's query
    # tile).
    "VLLM_TPU_SRPT_ATTN_TILE_TOKENS": env_nonnegative_int(
        "VLLM_TPU_SRPT_ATTN_TILE_TOKENS", 32
    ),
    # PrefixAwareScheduler (and subclasses), reuse-aware eviction
    # (core/reuse_aware_eviction.py): a finished request's decode-output
    # blocks (past its last full prompt block) go to the head of the free
    # queue, reused before any prompt block. For clients that never send our
    # output back verbatim (AgentX replay, thinking stripped).
    "VLLM_TPU_EVICT_DECODE_FIRST": env_bool("VLLM_TPU_EVICT_DECODE_FIRST"),
    # Reuse-aware eviction: free blocks with at least this many prefix hits
    # since they were filled go to a protected LRU segment evicted after the
    # probationary one (2-segment LRU; 0 disables).
    "VLLM_TPU_EVICT_PROTECT_HITS": env_nonnegative_int(
        "VLLM_TPU_EVICT_PROTECT_HITS", 0
    ),
    # Reuse-aware eviction: cap of the protected segment, as a fraction of the
    # pool's blocks; past it the oldest protected blocks become probationary.
    "VLLM_TPU_EVICT_PROTECTED_FRAC": env_nonnegative_float(
        "VLLM_TPU_EVICT_PROTECTED_FRAC", 0.3
    ),
    # PrefillSpacingScheduler: decode-only time owed after a step that runs a
    # chunk of a long prompt, as a fraction of that step's time (0 disables
    # spacing; prefill chunks are still ordered shortest first).
    "VLLM_TPU_PREFILL_SPACING_FRAC": env_nonnegative_float(
        "VLLM_TPU_PREFILL_SPACING_FRAC", 0.5
    ),
    # PrefillSpacingScheduler: a prompt is long, and its chunks are spaced,
    # while at least this many of its prompt tokens are left to compute.
    "VLLM_TPU_PREFILL_SPACING_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_PREFILL_SPACING_MIN_TOKENS", 8192
    ),
    # PrefillSpacingScheduler: TPOT target in ms. Prefill is deferred while a
    # long-chunk step would push a decoding request that can still meet it past
    # it (TPOT projected from max_tokens: only acts when clients set max_tokens
    # with ignore_eos; 0 disables).
    "VLLM_TPU_PREFILL_TPOT_TARGET_MS": env_nonnegative_float(
        "VLLM_TPU_PREFILL_TPOT_TARGET_MS", 0.0
    ),
    # PrefillSpacingScheduler: longest run of steps, in seconds, that the TPOT
    # target alone may defer prefill for; the next step then runs prefill.
    "VLLM_TPU_PREFILL_TPOT_MAX_HOLD_S": env_nonnegative_float(
        "VLLM_TPU_PREFILL_TPOT_MAX_HOLD_S", 10.0
    ),
    # PrefillSpacingScheduler: while the TPOT target defers prefill, still run
    # whole remaining prefills up to the largest prefill-token bin (512-4096)
    # whose measured step time fits every decoding request's slack.
    "VLLM_TPU_PREFILL_TPOT_PRICED": env_bool("VLLM_TPU_PREFILL_TPOT_PRICED"),
    # MiniMax-M3's LM head (the target's, and the DSpark draft's that shares it)
    # runs on an e4m3 copy of its weight with one scale per vocabulary row
    # (kernels/fp8_lm_head.py): half the bytes per decode step. Changes the
    # logits by the weight's e4m3 rounding, so it is opt-in.
    "TPU_FP8_LM_HEAD": env_bool("TPU_FP8_LM_HEAD", default=False),
    # The DSpark draft's MXFP8 MLP linears (gate_up/down; its qkv/o_proj stay
    # bf16) follow TPU_MINIMAX_M3_FP8_LINEARS and stay e4m3 with per-channel
    # power-of-two scales, halving the bytes the drafter streams every draft
    # step. 0 (default) dequantizes every draft linear to bf16.
    "TPU_MINIMAX_M3_FP8_DRAFT_LINEARS": env_bool(
        "TPU_MINIMAX_M3_FP8_DRAFT_LINEARS", default=False
    ),
    # PrefillSpacingScheduler: the TPOT target defers prefill while a decoding
    # request's slack is below this many long-chunk step prices (long minus
    # decode step time). Above 1 it also covers the step async scheduling has
    # already queued when the decision is made.
    "VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON": env_nonnegative_float(
        "VLLM_TPU_PREFILL_TPOT_HOLD_HORIZON", 1.0
    ),
    # PrefillSpacingScheduler: the TPOT target does not defer prefill while a
    # pending prefill (waiting or chunking) arrived more than this many seconds
    # ago, which bounds how long holds starve a prompt's TTFT. 0 turns it off.
    "VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S": env_nonnegative_float(
        "VLLM_TPU_PREFILL_TPOT_MAX_WAIT_S", 0.0
    ),
    # PrefillSpacingScheduler: order the waiting window and the running prefill
    # chunks by one aged shortest-remaining key, uncached prompt tokens left
    # minus VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S per second since arrival,
    # so a long prompt eventually goes ahead of every newer short one.
    # Off (default): uncached tokens alone, no aging.
    "VLLM_TPU_PREFILL_SRPT": env_bool("VLLM_TPU_PREFILL_SRPT"),
    # PrefillSpacingScheduler with VLLM_TPU_PREFILL_SRPT: prompt tokens one
    # second of waiting is worth in the SRPT key. A prompt with R uncached
    # tokens left outranks any newer request after R / this many seconds
    # (0: pure shortest-remaining first, no starvation bound).
    "VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S": env_nonnegative_float(
        "VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S", 256.0
    ),
    # PrefillSpacingScheduler decode-priority cadence: after a step that runs
    # prefill while requests decode, run this many decode-only steps per full
    # step of prefill tokens (max_num_batched_tokens; a partial prefill step
    # owes its share, rounded up) before the next prefill step. 0 (default)
    # turns it off.
    "VLLM_TPU_PREFILL_DECODE_CADENCE": env_nonnegative_int(
        "VLLM_TPU_PREFILL_DECODE_CADENCE", 0
    ),
    # With VLLM_TPU_PREFILL_DECODE_CADENCE or VLLM_TPU_PREFILL_COALESCE_TOKENS:
    # neither defers prefill while a waiting prompt not started yet arrived
    # more than this many seconds ago (TTFT starvation bound; running chunks
    # keep the cadence). 0 turns the bound off.
    "VLLM_TPU_PREFILL_CADENCE_MAX_WAIT_S": env_nonnegative_float(
        "VLLM_TPU_PREFILL_CADENCE_MAX_WAIT_S", 10.0
    ),
    # PrefillSpacingScheduler: while requests decode, hold prefill until the
    # pending prefill tokens (admissible waiting prompts, uncached, plus the
    # running chunks' tokens left) reach this many, so short prompts run
    # together as one full chunk. 0 (default) turns it off.
    "VLLM_TPU_PREFILL_COALESCE_TOKENS": env_nonnegative_int(
        "VLLM_TPU_PREFILL_COALESCE_TOKENS", 0
    ),
    # With VLLM_TPU_PREFILL_COALESCE_TOKENS: stop holding once the oldest
    # pending prompt arrived this many seconds ago.
    "VLLM_TPU_PREFILL_COALESCE_MAX_WAIT_S": env_nonnegative_float(
        "VLLM_TPU_PREFILL_COALESCE_MAX_WAIT_S", 0.5
    ),
    # PrefixAwareScheduler (and PrefillSpacingScheduler): in-batch shared-prefix
    # dedupe. A fresh waiting request whose next uncached KV block is a prompt
    # block that a running request has still to prefill is held in the queue
    # until that block is scheduled (and so cached), instead of prefilling the
    # shared prefix a second time. Off (default): no hold.
    "VLLM_TPU_SIBLING_HOLD": env_bool("VLLM_TPU_SIBLING_HOLD"),
    # VLLM_TPU_SIBLING_HOLD: seconds a request may be held in total; after that
    # it is admitted as usual (bounds the TTFT added by the hold).
    "VLLM_TPU_SIBLING_HOLD_MAX_S": env_nonnegative_float(
        "VLLM_TPU_SIBLING_HOLD_MAX_S", 4.0
    ),
    # MiniMax M3 DCP steps of at most this many tokens merge the attention
    # partials with one packed all-to-all instead of an LSE all-gather plus an
    # output reduce-scatter (`models.minimax_m3.dcp_pack`). 0 keeps the
    # reduce-scatter. Exact on a DCP 2 pair.
    "TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_DCP_MERGE_A2A_MAX_TOKENS", 0
    ),
    # MiniMax M3 DCP steps of at most this many tokens gather the query heads
    # and the top-k candidates in one all-gather, after the scoring op,
    # instead of two (`models.minimax_m3.dcp_pack`). 0 keeps two gathers.
    "TPU_MINIMAX_M3_DCP_QCAND_GATHER_MAX_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_DCP_QCAND_GATHER_MAX_TOKENS", 0
    ),
    # MiniMax M3 decode steps of at most this many tokens (DCP 2, one KV head
    # per pair: c4 / c16 with fp8 caches, or 4-bit caches) run the attention
    # half of the DCP op (query and candidate exchange, candidate merge, work
    # list, sparse attention, partial exchange, LSE merge) as one Pallas call
    # with remote DMAs between the pair
    # (`models.minimax_m3.kernels.msa_dcp_fused`; bitwise equal). At most 256;
    # 0 (default) keeps the unfused program.
    "TPU_MINIMAX_M3_DCP_FUSED_DECODE_MAX_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_DCP_FUSED_DECODE_MAX_TOKENS", 0
    ),
    # MiniMax M3 MSA cache writes of at most this many tokens find their token
    # runs inside the write kernel (a scalar pass over the tokens' pages and
    # slots, `kernels.msa_write._runs_kernel`) instead of XLA's compare /
    # cumsum / gather glue (`token_runs`); the same cache bytes. 0 (default) =
    # off.
    "TPU_MINIMAX_M3_MSA_WRITE_RUNS_IN_KERNEL_MAX_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_MSA_WRITE_RUNS_IN_KERNEL_MAX_TOKENS", 0
    ),
    # MiniMax M3 4-bit DCP decode steps of at most this many tokens score the
    # index keys split over the DCP groups (`kernels.msa_dcp.split_candidates`):
    # the groups hold the same index keys, so member r of each of the
    # TP / DCP groups scores 1 / (TP / DCP) of member r's pages for every
    # index head, and the parts' local top-k lists meet by a TP all-gather.
    # The candidates are bitwise those of the default scoring. 0 (default) =
    # off.
    "TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MAX_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MAX_TOKENS", 0
    ),
    # With TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MAX_TOKENS: split only steps of at
    # least this many tokens (the split pays two TP all-gathers per layer, a
    # loss on small steps); smaller steps keep the default scoring. 0
    # (default) = no lower bound.
    "TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MIN_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_MSA_INDEX_SPLIT_MIN_TOKENS", 0
    ),
    # MiniMax M3 4-bit MSA prefill attention with a page cache:
    # "<threshold>:<block_q>:<tile_pages>" runs a call whose selections hold
    # more than <threshold> distinct pages per query tile on average with
    # <block_q>-token tiles and <tile_pages> cached union pages, other calls
    # with the configured tiles (`msa_attention_lse.adaptive_prefill`). Empty
    # keeps one configuration. More tiers, ";" between them, each with its
    # own threshold (prefix "x": the widest sampled tile's distinct pages
    # instead of the mean) and tiles (`msa_attention_lse.adaptive_tiers`).
    "TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE": env_str(
        "TPU_MINIMAX_M3_MSA_PREFILL_ADAPTIVE", ""
    ),
    # MiniMax M3 DCP candidate merge (`msa_dcp.merge_candidates`, the dense
    # ranking used below 64 candidates per token, i.e. DCP 2) picks each
    # output slot with a max over the candidate (lane) axis instead of over
    # a middle axis. Same values; avoids a slow XLA layout at some row counts.
    "TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT": env_bool(
        "TPU_MINIMAX_M3_DCP_MERGE_LANE_SELECT", default=False
    ),
    # MiniMax M3 DCP 2 decode partial exchange (`msa_dcp._exchange_partials`):
    # each f32 LSE rides as a low-half lane plane and a high-half lane plane
    # instead of interleaved uint16 pairs, so the receiver rebuilds it from
    # two contiguous lane slices instead of a stride-2 lane slice (~5 us per
    # layer at c48 T=96). Same bits.
    "TPU_MINIMAX_M3_DCP_LSE_PLANES": env_bool(
        "TPU_MINIMAX_M3_DCP_LSE_PLANES", default=False
    ),
    # MiniMax M3: the Gemma RMSNorm ``1 + w`` FP32 weights and the q/k norm
    # tables are computed once after loading (`norm_hoist`), not per forward.
    # Same values.
    "TPU_MINIMAX_M3_NORM_WEIGHT_HOIST": env_bool(
        "TPU_MINIMAX_M3_NORM_WEIGHT_HOIST", default=False
    ),
    # MiniMax M3 DCP candidate merges of at least this many candidates per
    # token (DCP * top-k) run as the Pallas merge kernel, smaller ones as the
    # dense XLA ranking (`msa_dcp.merge_head_candidates`). 32 moves DCP 2
    # onto the kernel too. Same selection either way.
    "TPU_MINIMAX_M3_DCP_MERGE_KERNEL_MIN_CANDIDATES": env_nonnegative_int(
        "TPU_MINIMAX_M3_DCP_MERGE_KERNEL_MIN_CANDIDATES", 64
    ),
    # MiniMax M3 head-parallel DCP prefill attention (`msa_dcp_heads`, steps of
    # at least 2,048 tokens at DCP 8) kernel config, "<block_q>:<pages per
    # step>:<step widths, '/'-separated>:<page cache slots>", e.g.
    # "16:16:2/4/6/8/12:512". Empty keeps msa_dcp_heads' defaults (block_q
    # 8, 64 pages per step, the kernel's widths, no page cache); the
    # TPU_MINIMAX_M3_MSA_PREFILL_* flags apply only to `msa_attend_dcp`.
    "TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG": env_str(
        "TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG", ""
    ),
    # MiniMax M3 head-parallel MSA prefill: page buffer of each attention
    # round as this many max-length requests' blocks (default 1). A long
    # step whose prefill requests' blocks exceed one buffer attends them in
    # several rounds (an exchange, sorts and kernel calls each); 2 holds two
    # (c64 8k chunks of ~140K-token contexts) in one round.
    "TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR": env_nonnegative_int(
        "TPU_MINIMAX_M3_DCP_HP_ROUND_FACTOR", 1
    ),
    # MiniMax M3 head-parallel MSA prefill: smaller round buffers, as
    # "<tokens>/<tokens>/..." of summed prefill-request context. A step whose
    # prefill requests fit the smallest such rung attends them in one round
    # of that buffer instead of the max_model_len-sized one (a lax.switch in
    # the same program, so no extra compiled programs); the scales' static
    # gather / all-to-all / all-gather / padding shrink with the buffer.
    # Same outputs bit for bit. Empty = off. Not with EXCHANGE_ASYNC.
    "TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER": env_str(
        "TPU_MINIMAX_M3_DCP_HP_ROUND_LADDER", ""
    ),
    # MiniMax M3 head-parallel MSA prefill: the replicas of a KV head split
    # each kernel call's rows (alternate query tiles) instead of its query
    # heads, so each kernel tile carries all the KV head's query heads over
    # the same selected-page union (twice the rows per page at 2 replicas).
    # The query heads and outputs move by an all-to-all inside the replica
    # group (one chip's cores). Same outputs bit for bit.
    "TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT": env_bool(
        "TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT", default=False
    ),
    # MiniMax M3 MSA DCP over several KV heads: gather the group's K/V/index
    # queries lane-dense ([T, 3 * D] rows) and hand the scoring op the
    # gathered rows head-major (a free view) instead of three token-major
    # [T, G, D] copies; the op splits them. Same values bit for bit; drops
    # the relayout copies at the op boundary (export.py copy_/reshape,
    # ~0.18 ms per layer at c64 DCP8 8K steps). Default off.
    "TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR": env_bool(
        "TPU_MINIMAX_M3_DCP_KV_GATHER_HEAD_MAJOR", default=False
    ),
    # MiniMax M3 DCP attention over several KV heads (`msa_dcp_heads`): steps
    # of at most this many rows merge the ranks' partials by one all-to-all
    # of the head blocks and their LSEs plus the f32 merge kernel
    # (`msa_dcp.merge_partials`) instead of an LSE all-gather and a bf16
    # reduce-scatter of the weighted partials. 0 = off.
    "TPU_MINIMAX_M3_DCP_HEADS_MERGE_A2A_MAX_TOKENS": env_nonnegative_int(
        "TPU_MINIMAX_M3_DCP_HEADS_MERGE_A2A_MAX_TOKENS", 0
    ),
    # MiniMax M3 head-parallel prefill steps: merge the decode rows' DCP
    # partials (static max_num_seqs * decode_rows rows) by the all-to-all +
    # f32 merge kernel of TPU_MINIMAX_M3_DCP_HEADS_MERGE_A2A_MAX_TOKENS, on
    # those steps only (decode-only steps keep their merge). At c64 DCP4 the
    # SparseCore reduce-scatter of the bf16[dcp * Hl, 512, 128] partials
    # shares the SparseCores with the output projection's weight all-gather
    # and waits behind it (~187 us per layer served, ~45 us alone); the
    # all-to-all stays off the SparseCores. f32 merge: more precise than the
    # bf16 reduce-scatter, not bit-identical to it.
    "TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A": env_bool(
        "TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A", default=False
    ),
    # MiniMax M3 MSA kernels read the block table from SMEM as 16-bit page
    # ids, two per 32-bit word (`models/minimax_m3/kernels/page_table.py`):
    # half the SMEM per request, so max_num_seqs at a 1M max_model_len
    # doubles (DCP2 32 -> 64, DCP4 64 -> 128). Tables up to 768 KiB (DCP2 48,
    # DCP4 96 at 1M) stay int32, the kernels unchanged; larger ones need at
    # most 65536 KV pages per rank (else they stay int32).
    "TPU_MINIMAX_M3_PT16": env_bool("TPU_MINIMAX_M3_PT16", default=False),
    # MiniMax M3 sharded index scores (`msa.partial_block_scores`): "gather"
    # folds the index heads into query rows by a gather instead of a
    # scatter; "compose" unfolds the scores inside `block_scores`' layout
    # gather instead of a second row gather; "+"-joined. Bit-identical.
    "TPU_MINIMAX_M3_INDEX_FOLD_OPT": lambda: os.getenv(
        "TPU_MINIMAX_M3_INDEX_FOLD_OPT", ""
    ),
    # MiniMax M3 head-parallel MSA prefill page exchange
    # (`msa_dcp_heads.exchange_head_pages`): pages whose copies a rank keeps
    # in flight (default 8), and DIRECT=1 copies every page straight to every
    # replica of its KV head instead of forwarding inside the replica group.
    "TPU_MINIMAX_M3_DCP_EXCHANGE_IN_FLIGHT": env_nonnegative_int(
        "TPU_MINIMAX_M3_DCP_EXCHANGE_IN_FLIGHT", 8
    ),
    "TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT": env_bool(
        "TPU_MINIMAX_M3_DCP_EXCHANGE_DIRECT", default=False
    ),
    # ASYNC=1: the first round's page copies start ahead of the decode rows'
    # DCP attention and the selection merge and are retired after them
    # (a start / wait pair of kernels), so the exchange overlaps that work.
    "TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC": env_bool(
        "TPU_MINIMAX_M3_DCP_EXCHANGE_ASYNC", default=False
    ),
    # SCALES=1: the (synchronous) page exchange also copies each live page's
    # K/V scale rows of the destination's KV head by remote DMA, in place of
    # the XLA gather + all-to-all + all-gather + pad of the whole page
    # buffer's scales. Same values (bit-exact); ignored with ASYNC=1.
    "TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES": env_bool(
        "TPU_MINIMAX_M3_DCP_EXCHANGE_SCALES", default=False
    ),
    # COALESCE (synchronous exchange, not with DIRECT): "1" = 32-slot
    # forwarding chunks; else ':'-separated "g" (each source copies its own KV
    # head's pages of every part straight to every replica of its own
    # replica group, so only the other groups' pages are forwarded) and
    # "c<slots>" (forward each source's received part in copies of that
    # many consecutive slots instead of one copy per page). Same buffer
    # contents in every live slot (bit-exact). Empty = off.
    "TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE": env_str(
        "TPU_MINIMAX_M3_DCP_EXCHANGE_COALESCE", ""
    ),
    # MiniMax M3 MSA prefill page cache table (`msa_attention_lse._page_cache`):
    # how its sorts run. Empty keeps three stable sorts (two of two operands);
    # "unstable" drops the stability; "packed" sorts one packed int32 operand
    # per sort, unstable; "packed_search" also finds the distinct keys by a
    # binary search instead of the second sort; "packed_batched" runs the
    # second and third packed sorts as one two-row sort (they are
    # independent). Same table bit for bit.
    "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT": env_str(
        "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT", ""
    ),
    # MiniMax M3 MSA prefill page cache: narrower table widths below
    # TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES, '/'-separated (e.g.
    # "64/128"). Each layer's calls sort the narrowest width that holds their
    # widest tile union (one branch of the same program). Same table bit for
    # bit. Empty always sorts TILE_PAGES entries per tile.
    "TPU_MINIMAX_M3_MSA_PAGE_CACHE_WIDTHS": env_str(
        "TPU_MINIMAX_M3_MSA_PAGE_CACHE_WIDTHS", ""
    ),
    # MiniMax M3 MSA prefill attention on 4-bit caches (calls with the page
    # cache): ','-separated precision levers, empty = exact incumbent.
    # "qk8": fp8 queries x fp8 keys (int4 keys are exact in fp8) on the MXU;
    # "pv8": fp8 probabilities (carrying the V scales) x fp8 int4 values.
    # Not bit-identical.
    # MiniMax M3 MSA prefill attention on fp8 KV caches: "exp16" runs the
    # softmax exp in bf16 (p enters p @ v in bf16 anyway; not bitwise); "bias"
    # masks whole pages by a lane-row add instead of a per-score select
    # (bitwise); "lanemask" compares the mask once per 128 lanes (bitwise);
    # "vt" transposes V pages as they unpack for an untransposed p @ v dot;
    # "lmxu" takes the softmax row sums from the p @ v dot (not bitwise).
    # Comma-separated; empty
    # (default) = off.
    "TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT": env_str(
        "TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT", ""
    ),
    "TPU_MINIMAX_M3_MSA_PREFILL_PRECISION": env_str(
        "TPU_MINIMAX_M3_MSA_PREFILL_PRECISION", ""
    ),
    # MiniMax M3 MSA sparse attention softmax with no running max: at load the
    # model bounds |q . k| * head_dim**-0.5 from its q / k RMS-norm weights
    # (times the KV cache's key rounding); when every layer's bound is <= 75,
    # every MSA attention kernel (prefill and decode, DCP or not, all cache
    # dtypes) exponentiates scores against 0 instead of a running max, with
    # no max reduction or rescale. Not bitwise. Default off.
    "TPU_MINIMAX_M3_MSA_MAX_FREE": env_bool(
        "TPU_MINIMAX_M3_MSA_MAX_FREE", default=False
    ),
    # With TPU_MINIMAX_M3_MSA_MAX_FREE in effect: the 8/16-bit transposed
    # (prefill) MSA attention path takes each token's key mask from the QK
    # matmul (an extra head_dim-wide key column block of 0 / -1e30, a one-hot
    # of the token beside the queries) instead of a select per score.
    # Ignored without max-free. Default off.
    "TPU_MINIMAX_M3_MSA_MXU_MASK": env_bool(
        "TPU_MINIMAX_M3_MSA_MXU_MASK", default=False
    ),
    # MiniMax M3 MSA cross-layer block-selection reuse (IndexCache) on
    # decode-sized steps under DCP (`models.minimax_m3.index_reuse`): the
    # pattern of Full layers, which run the indexer; the other (Shared) layers
    # attend over the nearest preceding Full layer's selection. "N" or
    # "stride:N" (Full layers 0, N, 2N, ...), "full:0,a,b,..." (explicit Full
    # list, must hold 0) or one F/S letter per layer. Empty (or "1") = off.
    # Lossy. Prefill steps are unchanged.
    "TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE": env_str(
        "TPU_MINIMAX_M3_MSA_INDEX_REUSE_DECODE", ""
    ),
    # MiniMax M3 4-bit MSA caches (--kv-cache-dtype int4_per_token_head)
    # under DCP: consecutive layers share one f32 scale page (`msa_fp4.
    # layers_per_scale_page`: 8 // (2 * KV heads per rank), so 4 layers at
    # DCP 2 and 2 at DCP 4), each layer's K/V scale rows at its own offset.
    # A page of 8 f32 rows padded every layer's 2 or 4 scale rows; sharing
    # cuts 24 (DCP 2) or 16 (DCP 4) bytes per token per layer from the KV
    # pool (+9.7% / +4.2% tokens). Same stored values and kernel reads (the
    # same page DMA, other rows): bitwise. Default off.
    "TPU_MINIMAX_M3_KV4_SHARED_SCALES": env_bool(
        "TPU_MINIMAX_M3_KV4_SHARED_SCALES", default=False
    ),
    # Spec decode: pad a verify chunk's logits rows (padded_logits_length,
    # the length of every verify-row tensor) to a coarse ladder (powers of
    # two from 8 up to the token bucket of a full batch of verify windows)
    # instead of the token buckets, so the verify / sampler programs do not
    # multiply with TPU_TOKEN_BUCKET_EXTRA. The backbone keeps the token
    # buckets; a full-logits verify runs the LM head at the token-bucket rows
    # and pads the logits; eager uniforms are drawn at the token-bucket rows.
    # Same tokens.
    "TPU_SPEC_VERIFY_COARSE_PAD": env_bool("TPU_SPEC_VERIFY_COARSE_PAD", default=False),
    # Upload upstream InputBatch's sampling metadata (temperature, top_p,
    # top_k, penalties) at full length instead of [:num_reqs]: on TPU each
    # new live request count compiled an eager copy_ + as_strided_inverse
    # program mid-serving. Rows past num_reqs are never read; same values.
    "TPU_SAMPLING_METADATA_FIXED_SHAPE_COPIES": env_bool(
        "TPU_SAMPLING_METADATA_FIXED_SHAPE_COPIES", default=False
    ),
    # Never rebuild upstream InputBatch.sampling_metadata after the first
    # batch: the TPU samplers read the CPU sampling tensors, and each rebuild
    # uploads temperature/top_p/top_k (an online compile per request count,
    # H2D copies queued behind KV-offload loads). Supersedes
    # TPU_SAMPLING_METADATA_FIXED_SHAPE_COPIES. Exact (same tokens).
    "TPU_SKIP_DEVICE_SAMPLING_METADATA": env_bool(
        "TPU_SKIP_DEVICE_SAMPLING_METADATA", default=False
    ),
    # The batched block-table upload of a metadata walk reuses the previous
    # step's device copy while the packed tables are unchanged (they change
    # only when a request gains or loses a block): one H2D copy fewer per
    # walk and step, none queued behind KV-offload loads. Exact.
    "TPU_REUSE_UNCHANGED_BLOCK_TABLES": env_bool(
        "TPU_REUSE_UNCHANGED_BLOCK_TABLES", default=False
    ),
    # A new request's prompt token ids go to the workers as one int32 ndarray
    # instead of a Python list (core/compact_prompt_tokens.py). The engine
    # core's input thread converts the list once per request, off the busy
    # loop; the scheduler output then pickles the ids as one out-of-band
    # buffer, and the worker's unpickle and InputBatch row copy become
    # memcpys. At c48 with 100K-680K-token agentic prompts, the list unpickle
    # (~22 ms) and the list-to-row copy (~14 ms) ran on the worker main
    # thread before every prefill chunk, with the device idle. Exact (same
    # token values). Only prompts with at least
    # TPU_COMPACT_NEW_REQ_TOKENS_MIN tokens are converted.
    "TPU_COMPACT_NEW_REQ_TOKENS": env_bool(
        "TPU_COMPACT_NEW_REQ_TOKENS", default=False
    ),
    "TPU_COMPACT_NEW_REQ_TOKENS_MIN": env_nonnegative_int(
        "TPU_COMPACT_NEW_REQ_TOKENS_MIN", 4096
    ),
    # Token recovery without top-k or draft probabilities (the synthetic
    # verify's sampler) runs on a Pallas kernel holding each row as a dense
    # [vocab / 128, 128] slab (kernels/top_p_sample.py), 8 rows per call,
    # instead of XLA's ~37 passes over [rows, vocab]. Same steps; the row
    # sums run in another order. Takes precedence over
    # TPU_SAMPLER_TOPP_CANDIDATES on the steps it serves.
    "TPU_SAMPLER_TOP_P_KERNEL": env_bool("TPU_SAMPLER_TOP_P_KERNEL", default=False),
    # The bounded one-hot MoE permute/unpermute kernels also take token counts
    # past 256 that are whole bf16 sublane tiles (multiples of 16), not only
    # multiples of 128: the 320/352/416/448/480-token decode buckets. Exact.
    "TPU_MOE_ONEHOT_BOUNDED_SUBLANE_TOKENS": env_bool(
        "TPU_MOE_ONEHOT_BOUNDED_SUBLANE_TOKENS", default=False
    ),
    # MiniMax M3 4-bit index caches (--kv-cache-dtype int4_per_token_head):
    # query rows per tile of the MSA indexer's large (prefill) class, instead
    # of its default 512, on steps of at least twice that many query rows.
    # A 4-bit page's keys are unpacked and scaled to bf16 once per tile that
    # scores it, so wider tiles unpack each page fewer times (the fp8 cache
    # converts its keys cheaper). Each page maximum is still one dot of the
    # same keys and query: bitwise. 0 (default) = off.
    "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q": env_nonnegative_int(
        "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q", 0
    ),
}


def __getattr__(name: str) -> Any:
    """
    Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def enable_envs_cache() -> None:
    """
    Enables caching of environment variables by wrapping the module's __getattr__
    function with functools.cache(). This improves performance by avoiding
    repeated re-evaluation of environment variables.

    NOTE: This should be called after service initialization. Once enabled,
    environment variable values are cached and will not reflect changes to
    os.environ until the process is restarted.
    """
    # Tag __getattr__ with functools.cache
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    # Cache all environment variables
    for key in environment_variables:
        __getattr__(key)


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def use_raiden_glm_admission() -> bool:
    """True when Raiden KV-cache manager and GLM Stage-3 admission are both enabled."""
    mod = sys.modules[__name__]
    return bool(mod.TPU_USE_RAIDEN_KV_CACHE_MANAGER and mod.TPU_RAIDEN_GLM_ADMISSION)


def is_indexer_kv_cache_mode_explicit() -> bool:
    """True when VLLM_TPU_INDEXER_KV_CACHE_MODE was explicitly set."""
    return "VLLM_TPU_INDEXER_KV_CACHE_MODE" in os.environ
