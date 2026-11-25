# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import functools
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    TPU_RAIDEN_HANDOFF_ALIGN_TOKENS: int = 0
    USE_MOE_SPARSE_CORE: bool = True
    ONEHOT_MOE_PERMUTE_THRESHOLD: int | None = None
    TPU_MOE_SEQUENCE_PARALLEL: bool = False
    TPU_QWEN4_HC_DECODE_FUSED: bool = False
    TPU_QWEN4_HC_DECODE_UP_T: bool = False
    # q38 PORT-BL HC (HC fold + decode row tile, ported from Benliang Wang); program-changing, hashed.
    TPU_QWEN4_HC_DECODE_FOLD: bool = False
    TPU_QWEN4_HC_DECODE_TILE_ROWS: int = 0
    # q38 PORT-BL HC end
    VLLM_TPU_HC_DECODE_COUNT: int = 0
    VLLM_TPU_Q38_STACK_ROUTE_COUNT: int = 0
    TPU_QWEN4_HC_PREFILL_ODD_TILES: bool = False
    TPU_QWEN4_HC_PREFILL_FUSED_MIX: bool = False
    TPU_QWEN4_HC_PREFILL_PARTIAL_TILE: bool = False
    TPU_QWEN4_HC_PREFILL_NORM: str = "vpu"
    TPU_QWEN4_HC_PREFILL_FOLD: bool = False
    TPU_QWEN4_HC_PREFILL_DEFER_COMBINE: bool = False
    TPU_QWEN4_HC_PREFILL_DEFER_WHOLE: bool = False
    VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT: bool = False
    QWEN4_EXP_PREFILL_SP: bool = False
    QWEN4_EXP_PD_STAGING_BYTES: int = 8 << 30
    QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS: int = 3
    QWEN4_EXP_PD_STAGING_LEASES: int = 0
    QWEN4_EXP_PD_DRAFT_KV: bool = False
    QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP: float = 0.0
    QWEN4_EXP_PD_DRAFT_HOLD: bool = False
    QWEN4_EXP_PD_ROUTED_DST: str = ""
    QWEN4_EXP_PD_SYNC_REPACK: bool = False
    QWEN4_EXP_PD_SYNC_REPACK_AB_S: int = 0
    QWEN4_EXP_PD_LANDING_LOG: int = 0
    QWEN4_EXP_PD_REPACK_V2: int = 0
    QWEN4_EXP_PD_REPACK_L2: bool = False
    QWEN4_EXP_PD_REPACK_PLAN_AT_LAND: bool = False
    QWEN4_EXP_PD_SHARED_COMPRESSED: bool = False
    QWEN4_EXP_PD_BATCHED_ADMISSION: bool = False
    QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES: int = 0
    QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK: int = 0
    QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR: int = 1
    TPU_KV_CACHE_GROUP_SIZE_RULE: str = "gcd"
    QWEN4_EXP_PREFILL_SP_MIN_TOKENS: int = 1024
    QWEN4_EXP_PREFILL_SP_MOE: bool = False
    QWEN4_EXP_PREFILL_SP_SHARED_LOCAL: bool = False
    QWEN4_EXP_PREFILL_SP_PAD_HC: bool = False
    QWEN4_EXP_PACK_MAIN_KV: bool = False
    QWEN4_EXP_PACK_COMPRESSED_K: bool = False
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    TPU_MOE_DENSE_EXPERT_THRESHOLD: int = 0
    FUSED_MOE_DECODE_KERNEL: bool = False
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: int = 256
    FUSED_MOE_DECODE_KERNEL_FOLD: str = "off"
    FUSED_MOE_DECODE_COMPACT_ROWS: int = 0
    FUSED_MOE_DECODE_COMPACT_MIN_TOKENS: int = 256
    FUSED_MOE_DECODE_COMPACT_PIPE: int = 0
    FUSED_MOE_DECODE_COMPACT_PIPE_ROWS: int = 32
    FUSED_MOE_DECODE_KERNEL_SCOPES: int = 0
    # q38 PORT-BL MOE (exact width, ported from Benliang Wang's b0c1453ba)
    FUSED_MOE_DECODE_EXACT_WIDTH: bool = False
    TPU_MOE_SINGLE_SORT: bool = True
    TPU_MOE_NONEP_GATHER: str = "xla"
    TPU_MOE_EP_GATHER: str = "sc"
    TPU_MOE_EP_GMM_STACK: bool = False
    TPU_MOE_EP_COMBINE: str = "served"
    TPU_MOE_NONEP_DISPATCH_CHUNKS: str = "1"
    TPU_MOE_EP_PACKED_SORTS: str = "off"
    TPU_MOE_COMBINE_PRESORTED: bool = False
    TPU_MOE_NONEP_GATHER_COLS: int = 0
    TPU_MOE_NONEP_GATHER_CHUNK: int = 16384
    TPU_MOE_NONEP_PERMUTE: str = "sorts"
    TPU_MOE_NONEP_COMBINE_SPLIT: str = "1"
    TPU_MOE_NONEP_GMM_TILE_M: str = "0"
    TPU_MOE_NONEP_GMM_HOIST_QUANT: bool = False
    TPU_MOE_NONEP_GMM2_RHS_SPLIT: str = "1"
    TPU_MOE_NONEP_GMM_EXPERTS_PER_VISIT: str = "1"
    TPU_MOE_NONEP_GMM2_EXPERTS_PER_VISIT: str = "0"
    TPU_MOE_NONEP_FUSED_MLP: bool = False
    TPU_SC_COMBINE_DENSE: str = ""
    TPU_MOE_NONEP_COMBINE_PACKED: bool = False
    TPU_MOE_NONEP_GMM_BUCKET: str = "0"
    USE_MOE_TP_DECODE_KERNEL: bool = False
    RAGGED_GATHER_REDUCE_VERSION: str = "v2"
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
    VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR: bool = False
    VLLM_TPU_GDN_PREFILL_BF16_QKV_2D: bool = False
    VLLM_TPU_GDN_PREFILL_DENSE_CONV: bool = False
    VLLM_TPU_GDN_WINDOW_TILE: bool = False
    VLLM_TPU_GDN_PREFILL_QKV_FROM_PROJ: bool = False
    TPU_ENABLE_GDN_DYNAMIC_TILING: bool = False
    VLLM_TPU_QSA_DECODE_QUERY_BLOCK: int = 0
    VLLM_TPU_QSA_TOPK_STATS: int = 0
    VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK: int = 0
    VLLM_TPU_QSA_PREFILL_INDEXER_BQ: int = 128
    VLLM_TPU_QSA_PREFILL_INDEXER_BKV: int = 256
    VLLM_TPU_QSA_DECODE_KV_BLOCK: int = 1024
    VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK: int = 256
    VLLM_TPU_QSA_RING_SELECT: bool = False
    VLLM_TPU_QSA_RING_DMA_BATCH: int = 0
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE: bool = False
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS: int = 512
    VLLM_TPU_GDN_HOIST_READ_OFFSETS: bool = False
    VLLM_TPU_QSA_BF16_REDUCE_PRECISION: bool = False
    VLLM_TPU_QSA_INDEXER_HOIST_META: bool = False
    VLLM_TPU_QSA_INDEXER_MIXED_ONLY: bool = False
    VLLM_TPU_QSA_DRAFT_SHARED_SELECTION: bool = False
    VLLM_TPU_DRAFT_EMBED_H8: bool = False
    VLLM_TPU_DRAFT_EMBED_REPL: bool = False
    VLLM_TPU_TARGET_EMBED_REPL: bool = False
    VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE: bool = False
    VLLM_TPU_QWEN4_NGRAM_WINDOW: str = ""
    VLLM_TPU_QWEN4_FUSED_PREP: bool = False
    VLLM_TPU_QWEN4_FUSED_PREP_CHECK: str = ""
    VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL: bool = False
    VLLM_TPU_QSA_FUSED_SPARSE_DECODE: bool = False
    VLLM_TPU_QSA_PREFIX_SLICE: bool = False
    VLLM_TPU_QSA_ZERO_PREFIX_FAST: bool = False
    VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT: bool = False
    VLLM_TPU_QSA_PREFILL_BITSET_KERNEL: bool = True
    VLLM_TPU_QSA_PREFILL_BF16_MXU: bool = False
    VLLM_TPU_QSA_PREFILL_LEAN: bool = False
    VLLM_TPU_QSA_PREFILL_QUERY_TILE: int = 0
    VLLM_TPU_QSA_PREFILL_Q128_FALLBACK: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_WALK: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT: int = 32768
    VLLM_TPU_QSA_PREFILL_UNION_CAP: int = 12288
    VLLM_TPU_QSA_PREFILL_UNION_KV_TILE: int = 1024
    VLLM_TPU_QSA_PREFILL_UNION_COUNT: int = 0
    VLLM_TPU_PDENSE_COUNT: int = 0
    VLLM_TPU_PDENSE_TRACE_RECORDS: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_LISTED: str = ""
    VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_ALPHA: float = 0.0
    VLLM_TPU_QSA_PREFILL_UNION_QK16: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_IN_PLACE: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_SC_BITS_PLAN: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_GATHERED_LEAN: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_MASK_VIEW: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_VGUARD_LEAN: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_ALIGN_GATHER: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_WALK_PIPE: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_DOT16: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_VGUARD_NONE: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_BF16_SOFTMAX: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_FOS: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_LEAN_TILE: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_GATHERED: str = ""
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS: int = 4
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH: float = 1.0
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY: str = "vmem"
    VLLM_TPU_QSA_MAIN_KV_DCP: int = 0
    VLLM_TPU_QSA_DCP_ONE_GATHER: bool = False
    VLLM_TPU_QSA_MTP_KV_DCP: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP: bool = False
    VLLM_TPU_QSA_DCP_HEAD_A2A: str = ""
    VLLM_TPU_QSA_DCP_Q_ONLY_GATHER: bool = False
    VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_LEAN_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_SLOT_READ: bool = False
    VLLM_TPU_QSA_DECODE_FAST_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_RPA_FETCH: str = ""
    VLLM_TPU_QSA_DECODE_RPA_BKV: int = 256
    VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE: bool = False
    VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA: bool = False
    VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_INDEXER_STRIDED_KEYS: bool = False
    VLLM_TPU_QSA_INDEXER_KEY_LANES: int = 0
    # q38 PORT-BL IDX (ported from Benliang Wang, 2ce02bb1a): decode indexer
    # dense step and tail skip.
    VLLM_TPU_QSA_INDEXER_DENSE_STEP: bool = False
    VLLM_TPU_QSA_INDEXER_TAIL_SKIP: int = 0
    VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_DECODE_RPA_LANES: int = 8
    VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES: int = 0
    VLLM_TPU_QSA_FUSED_KV_WRITE: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS: int = 0
    VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP: bool = False
    VLLM_TPU_QSA_INDEXER_TOKEN_SHARD: bool = False
    VLLM_TPU_QSA_DECODE_REQUEST_SHARD: bool = False
    VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS: int = 0
    VLLM_TPU_QSA_SHARED_RANK_OPERANDS: bool = False
    VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT: bool = False
    VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS: int = 8
    # q38 PORT-BL MOE (PLE lookup chunk, stock buckets only)
    VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS: int = 0
    TPU_MOE_SKIP_PADDED_TOKENS: bool = False
    TPU_MOE_HIERARCHICAL_EP: bool = False
    TPU_MOE_ROUTER_TOPK: str = "rowmax"
    USE_MOE_FUSED_EP_KERNEL: bool = False
    MOE_FUSED_EP_KERNEL_IMPL: str | None = None
    MOE_FUSED_EP_ENABLE_W4A8: bool = False
    MOE_FUSED_EP_KERNEL_MIN_TOKENS: int = 1024
    MOE_FUSED_EP_V2_SHARDED_PLAN: bool = False
    MOE_FUSED_EP_SORTED_GATHER: bool = False
    MOE_ADAPTIVE_2D_CHANNEL_SCALES: bool = False
    MOE_ADAPTIVE_AG_LEAD_AXIS: bool = False
    MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE: int = 0
    TPU_LATENT_PROJ_INTRA_CHIP_TP: bool = False
    TPU_TOKEN_BUCKET_EXTRA: list[int] = []
    TPU_TOKEN_BUCKET_LINEAR: bool = False
    TPU_PP_DYNAMIC_CHUNKS: bool = True
    TPU_PP_CHUNK_SLACK: float = 0.1
    TPU_PP_LAZY_PUSH: bool = False
    TPU_PP_HANDOFF: str = "wave"
    TPU_PP_RING_DEPTH: int = 2
    TPU_PP_EXTRA_STEPS_IN_FLIGHT: int = -1
    TPU_PP_RING_VERIFY: int = 0
    TPU_PP_RING_VERIFY_MUTANT: int = -1
    TPU_PP_WARMUP_BARRIER: bool = False
    TPU_PP_MTP_WARMUP_BARRIER: bool = False
    TPU_PP_SPREAD_MAX_PREFIX: int = 0
    TPU_PP_WARM_FIRST: bool = False
    TPU_PP_WARM_FIRST_AGING: float = 10000.0
    TPU_PP_WARM_FIRST_TIER: bool = False
    TPU_ROPE_CACHE_TRUNCATE: bool = False
    TPU_ROPE_CACHE_ROW_MAJOR: bool = False
    TPU_MOE_HASH_TABLE_ROW_MAJOR: bool = False
    TPU_PARALLEL_PRECOMPILE: bool = False
    TPU_PRECOMPILE_BIGGEST_FIRST: bool = False
    TPU_PRECOMPILE_PEER_REPLAY: bool = False
    TPU_STREAMIDX_CHUNK_TOKENS: int | None = None
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
    VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: bool = False
    VLLM_TPU_MAMBA_CHECKPOINT_RETENTION: bool = True
    VLLM_TPU_MAMBA_DECODE_CHECKPOINT: bool = True
    VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT: bool = False
    VLLM_TPU_PREFIX_TAIL_EVICT_FIRST: bool = False
    VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER: bool = False
    VLLM_TPU_PREFIX_REFRESH_WAITING: int = 0
    VLLM_TPU_PREFIX_REFRESH_INTERVAL_S: float = 2.0
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
    VLLM_TPU_SEGMENT_TOKENIZE: bool = False
    VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS: int = 33554432
    VLLM_TPU_SEGMENT_TOKENIZE_VERIFY: bool = False
    VLLM_TPU_DP_AFFINITY: bool = False
    VLLM_TPU_DP_AFFINITY_CHUNK: int = 1024
    VLLM_TPU_DP_AFFINITY_SLACK: float = 0.25
    VLLM_TPU_DP_AFFINITY_MAX_ENTRIES: int = 1000000
    VLLM_TPU_DP_AFFINITY_SLOT_GUARD: bool = False
    VLLM_TPU_DP_AFFINITY_SHARED: bool = False
    VLLM_TPU_DP_AFFINITY_SHARED_SLOTS: int = 2097152
    VLLM_TPU_FUSED_DRAFT_LOOP: bool = False
    VLLM_TPU_WARM_SPEC_LOGITS_LENGTHS: bool = True
    VLLM_TPU_PLE_HOST_FAST_LOOKUP: bool = False
    VLLM_TPU_PLE_HOST_TIMING: bool = False
    VLLM_TPU_PLE_HOST_OVERLAP: bool = False
    VLLM_TPU_PLE_HOST_OVERLAP_CHECK: int = 0
    VLLM_TPU_PLE_FUSED: bool = False
    VLLM_TPU_PLE_FUSED_TAP_MODE: str = "slice"
    VLLM_TPU_PLE_FUSED_TILE: int = 128
    VLLM_TPU_PLE_FUSED_NORM: str = "sum"
    VLLM_TPU_PLE_FUSED_PRECISION: str = "torch"
    VLLM_TPU_PLE_CONV_ONEHOT: bool = False
    Q38_PROBE_PLE_ONEHOT_MUTANT: bool = False
    VLLM_TPU_PLE_DECODE_BOUNDED: str = "0"
    VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS: int = 0
    VLLM_TPU_PLE_D14_FOLD: bool = False
    VLLM_TPU_PLE_WRDEFER: str = "0"
    VLLM_TPU_FUSED_ASYNC_DRAFT_SEED: bool = True
    VLLM_TPU_DEFER_PREV_RESULTS_WAIT: bool = False
    VLLM_TPU_LOG_POST_WARMUP_COMPILES: bool = False
    VLLM_TPU_CPU_PIN: str = "off"
    VLLM_TPU_HOST_PHASES: int = 0
    VLLM_TPU_AOT_SKIP_GUARD_CHECK: bool = False
    # q38 L4 D98 / X-H14-D98R (compilation/fast_compiled_call.py); host-only, compile-ignored.
    VLLM_TPU_FAST_COMPILED_CALL: bool = False
    VLLM_TPU_FAST_CALL_GUARD_MEMO: str = ""
    VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE: bool = False
    VLLM_TPU_FAST_CALL_LOG_EVERY: int = 200
    VLLM_TPU_ENGINE_PHASES: int = 0
    # q38 L1-C48 DECODE-TRIM (core/decode_trim.py); engine-core host-only, compile-ignored.
    VLLM_TPU_DECODE_TRIM: str = ""
    VLLM_TPU_DECODE_TRIM_LOG: int = 2000
    # q38 L1-DSTASH (runner/draft_stash.py); worker host-only, compile-ignored.
    VLLM_TPU_DECODE_TRIM_DRAFT_STASH: bool = False
    VLLM_TPU_DSTASH_STATE_LOG: bool = False  # q38 S3A-ENG T4: the per-rank DSTASH state line; host-only, compile-ignored
    VLLM_TPU_MEMSTAT_INTERVAL_S: int = 0
    VLLM_TPU_DECODE_HOST_TIER: bool = False
    VLLM_TPU_DECODE_HOST_TIER_VERIFY: int = 0
    VLLM_TPU_PD_PROMPT_TAIL: int = 0
    VLLM_TPU_HOST_PHASES_SLOW_MS: float = 0.0
    VLLM_TPU_HOST_PHASES_SUBS: bool = False
    VLLM_TPU_HOST_PHASES_GC: bool = True
    VLLM_TPU_KV_TWIN_GROUPS: bool = False
    VLLM_TPU_HOST_SWITCH_US: float = 0.0
    VLLM_TPU_HOST_GC: str = ""
    VLLM_TPU_HOST_GC_AFTER: int = 500
    VLLM_TPU_HOST_GC_GEN0: int = 50000
    VLLM_TPU_HOST_AB_PERIOD_MS: int = 0
    VLLM_TPU_L6_AB_PERIOD_MS: int = 0
    VLLM_TPU_L6_AB_CLASSES: int = 2
    VLLM_TPU_L6_AB_STEPS: int = 0
    VLLM_TPU_L6_AB_CONFIRM: int = 16
    VLLM_TPU_L6_AB_DIR: str = ""
    VLLM_TPU_L6_CHECK_STEPS: int = 0
    VLLM_TPU_L6_ENVS_CACHE: str = ""
    VLLM_TPU_L6_ENVS_CACHE_CHECK: int = 0
    VLLM_TPU_L6_ENVS_CENSUS: int = 0
    VLLM_TPU_L6_FAST_FWD_CTX: str = ""
    VLLM_TPU_L6_GC_FREEZE: str = ""
    VLLM_TPU_L6_SWITCH: str = ""
    VLLM_TPU_L6_SWITCH_US: int = 200
    VLLM_TPU_L6_MAIN_PIN: str = ""
    VLLM_TPU_L6_MAIN_PIN_CPU: int = -1
    VLLM_TPU_L6_MAIN_PIN2: str = ""
    VLLM_TPU_L6_NUMA_BIND: bool = False
    VLLM_TPU_L6_SYNC_GC: str = ""
    VLLM_TPU_L6_SYNC_GC_EVERY: int = 100
    VLLM_TPU_L6_SYNC_GC_VALVE: int = 100000
    VLLM_TPU_L6_DEFER_PREV: str = ""
    VLLM_TPU_L6_STEADY_STEP: str = ""
    VLLM_TPU_L6_STEADY_STEP_CHECK: int = 0
    VLLM_TPU_L6_DECODE_CARRY: str = ""
    VLLM_TPU_L6_DECODE_CARRY_CHECK: int = 0
    VLLM_TPU_L6_CARRY_MIN_REQS: int = 0
    VLLM_TPU_L6_STEADY_MIN_REQS: int = 0
    VLLM_TPU_L6_ADAPT_CARRY: str = ""
    VLLM_TPU_L6_ADAPT_CARRY_CHECK: int = 0
    VLLM_TPU_L6_ENGINE_ALLOC_FAST: str = ""
    VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK: int = 0
    VLLM_TPU_L6_SETUP_FREE: str = ""
    VLLM_TPU_L6_SETUP_FREE_CHECK: int = 0
    VLLM_TPU_L6_FENCE_SKIP: str = ""
    VLLM_TPU_L6_TOKEN_LISTS: str = ""
    VLLM_TPU_L6_TOKEN_LISTS_CHECK: int = 0
    VLLM_TPU_L6_AOT_SKIP: str = ""
    VLLM_TPU_L6_AOT_SKIP_CHECK: int = 0
    VLLM_TPU_L6_GC_ADMIT_FREEZE: str = ""
    VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP: int = 1000000
    VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK: int = 64
    VLLM_TPU_L6_GC_CENSUS_TYPES: bool = False
    VLLM_TPU_L6_GC_CENSUS_MS: int = 1000
    VLLM_TPU_L6_GC_CENSUS_MAX: int = 4
    VLLM_TPU_L6_EOS_CANARY: str = ""
    VLLM_TPU_L6_PREP_TRIMS: str = ""
    VLLM_TPU_L6_PREP_TRIMS_CHECK: int = 0
    VLLM_TPU_L6_ADMIT_INC: str = ""
    VLLM_TPU_L6_ADMIT_INC_CHECK: int = 0
    VLLM_TPU_L6_MK_CTX: str = ""
    VLLM_TPU_L6_MK_CTX_CHECK: int = 0
    VLLM_TPU_L6_MK_THIN_TT: str = ""
    VLLM_TPU_L6_MK_THIN_TT_CHECK: int = 0
    VLLM_TPU_L6_MK_DIRECT: str = ""
    VLLM_TPU_L6_MK_DIRECT_CHECK: int = 0
    VLLM_TPU_L6_FUSE_RPC: str = ""
    VLLM_TPU_L6_OUT_KVONLY: str = ""
    VLLM_TPU_L6_OUT_KVONLY_CHECK: int = 0
    VLLM_TPU_L6_CARRY_LEAN: str = ""
    VLLM_TPU_L6_CARRY_LEAN_CHECK: int = 0
    VLLM_TPU_L6_PREP_SHORT: str = ""
    VLLM_TPU_L6_PREP_SHORT_CHECK: int = 0
    VLLM_TPU_L6_PMETER: str = ""
    VLLM_TPU_L6_PMETER_WINDOW: int = 200
    VLLM_TPU_L6_INJECT: str = ""
    VLLM_TPU_L6_INJECT_US: int = 250
    VLLM_TPU_L6_INJECT_RANK: int = 0
    VLLM_TPU_L6_INJECT_ALL: str = ""
    VLLM_TPU_L6_INJECT_ALL_US: int = 1500
    VLLM_TPU_L6_NODEFER_CENSUS: str = ""
    VLLM_TPU_L6_NODEFER_EVERY: int = 64
    VLLM_TPU_S3_ASYNC_SUBMIT: str = ""
    VLLM_TPU_S3_ASYNC_LAG: int = 2
    VLLM_TPU_S3_ASYNC_JOIN_TIMEOUT_S: int = 120
    VLLM_TPU_S3_ASYNC_STATS_EVERY: int = 200
    VLLM_TPU_S3_SUBMARKS: bool = False
    VLLM_TPU_S3_DEFER_AR: str = ""
    VLLM_TPU_S3_DEFER_GATE_MIN_REQS: int = 0
    VLLM_TPU_GDN_SEED_COPY_PACKED: bool = False
    VLLM_TPU_GDN_SEED_COPY_PACKED_CHECK: int = 0
    VLLM_TPU_D1N_AB_PERIOD_MS: int = 0
    VLLM_TPU_LOG_STEP_BUCKETS: int = 0
    VLLM_TPU_LOG_STEP_BUCKETS_PERIOD: bool = False
    VLLM_TPU_SERVED_AB: str = ""
    VLLM_TPU_SERIALIZE_COPY_WAIT: bool = False
    VLLM_TPU_PINNED_TOKEN_COPY: bool = False
    VLLM_TPU_SAMPLING_PARAMS_REUSE: bool = False
    # q38 L4 SEEDFIX: seeded requests sample from their per-request generators.
    VLLM_TPU_SEEDED_SAMPLING: bool = True
    # PD DEC-FB-2 vocabulary-parallel verify sampler (runner/vps_wiring.py).
    VLLM_TPU_VPS_VERIFY: bool = False
    VLLM_TPU_VPS_FALLBACK_ROWS: int = 0
    VLLM_TPU_VPS_TIEGAP_SLOTS: int = 0
    VLLM_TPU_VPS_DRAWS: str = "full"
    VLLM_TPU_VPS_CHECK: int = 0
    # q38 DEC-FB-3 lever 2: MM DSTOCK's draft top-tokens sites (6dcff75dc).
    VLLM_TPU_DRAFT_TOP_TOKENS: bool = False
    VLLM_TPU_DRAFT_TOP_TOKENS_CHECK: int = 0
    VLLM_TPU_VPS_CHECK_FAULT: int = 0
    VLLM_TPU_VPS_STATS_EVERY: int = 1000
    # q38 L4 D83: the VPS verify also computes the mamba read offsets.
    VLLM_TPU_VPS_FUSED_READ_OFFSETS: bool = False
    VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK: int = 0
    VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT: int = 0
    # q38 L4 VFY-ONE: one VPS verify program (T-3), + the draft seed (DH-C) at 2.
    VLLM_TPU_VPS_VFY_ONE: bool = False
    VLLM_TPU_VPS_VFY_ONE_CHECK: int = 0
    VLLM_TPU_VPS_VFY_ONE_AB: str = ""
    VLLM_TPU_VPS_VFY_ONE_KEYLITE: bool = False
    # q38 STALL-HUNT: VFY-ONE warm-up traces the seed with serving's argument aliasing.
    VLLM_TPU_VFY_WARM_ALIAS: bool = False
    VLLM_TPU_PY_SAMPLER_S: float = 0.0
    VLLM_TPU_STAGE_ENTRIES_CACHE: bool = False
    VLLM_TPU_SPEC_TAIL_COMPILED: bool = False
    VLLM_TPU_SPEC_TAIL_REJ_PAD: bool = False
    VLLM_TPU_DRAFT_ASM_FOLD: bool = False
    VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT: bool = False
    VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY: bool = False
    VLLM_TPU_DRAFT_ASM_CHECK_STEPS: int = 0
    VLLM_TPU_DRAFT_ASM_AB: str = ""
    VLLM_TPU_DRAFT_ASM_CARRY_LOOP: bool = False
    VLLM_TPU_DRAFT_ASM_ONE: bool = False
    VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB: str = ""
    VLLM_TPU_DRAFT_ASM_ONE_AB: str = ""
    VLLM_TPU_DRAFT_ASM_CAP: bool = False
    VLLM_TPU_K1FUSE: bool = False
    VLLM_TPU_K1FUSE_AB: str = ""
    VLLM_TPU_K1FUSE_CHECK: int = 256
    VLLM_TPU_SAMPLING_COPY_FULL: bool = False
    VLLM_TPU_EAGER_SHAPE_LOG: int = 0
    VLLM_TPU_VPS_WARM_DRAWS: int = 0
    VLLM_TPU_L4_SAMP_FB_ORDER: str = ""
    VLLM_TPU_L4_SAMP_SNAP: str = ""
    VLLM_TPU_L4_SAMP_SNAP_CHECK: int = 0
    VLLM_TPU_L4_VPS_STATS_ASYNC: str = ""
    VLLM_TPU_VPS_TOPK_KEYS: bool = False
    VLLM_TPU_VPS_TOPK_CUTOFF: bool = False
    VLLM_TPU_VPS_INGRAPH_RNG: bool = False
    VLLM_TPU_VPS_INGRAPH_RNG_AB: str = ""
    VLLM_TPU_STALL_WATCHDOG_S: float = 0.0
    VLLM_TPU_VERIFY_FUSED: bool = False
    VLLM_TPU_VERIFY_FUSED_CHECK: int = 0
    VLLM_TPU_VERIFY_FUSED_CHECK_FAULT: int = 0
    VLLM_TPU_DRAFT_SEED_FUSED: bool = False
    VLLM_TPU_DRAFT_CARRY_FUSED: bool = False
    VLLM_TPU_VERIFY_FUSED_READ_OFFSETS: bool = False
    VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS: int = 0
    VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY: bool = False
    VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS: int = 0
    VLLM_TPU_DRAFT_ROWS_FRESH: bool = False
    VLLM_TPU_DRAFT_CARRY_ROWS_FRESH: bool = False
    VLLM_TPU_DRAFT_SEED_EXTRAS: bool = False
    VLLM_TPU_CPU_PIN_RESERVE: int = 2
    VLLM_TPU_CPU_PROBE_S: float = 0.0
    VLLM_TPU_FUSED_READ_OFFSET_UPDATE: bool = True
    VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE: bool = False
    VLLM_TPU_BLOCK_TABLE_STAGE_INC: bool = False
    VLLM_TPU_BLOCK_TABLE_STAGE_INC_CHECK: int = 0
    VLLM_TPU_TOKEN_PADDING_REUSE: bool = False
    VLLM_TPU_FAST_METADATA_BUILD: bool = False
    VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK: bool = False
    VLLM_TPU_STAGE_TIMING_OFF: bool = False
    VLLM_TPU_STAGED_WALK_CLASS_GATHER: bool = False
    VLLM_TPU_WARMED_SUBSTITUTE_SOURCE: bool = False
    VLLM_TPU_PROPOSE_DEVICE_CONSTANTS: bool = False
    VLLM_TPU_MTP_DRAFT_ROWS: bool = False
    TPU_SAMPLER_TOPK_CANDIDATES: int = 0
    VLLM_TPU_POSTFWD_FUSED: bool = False
    MM_ENCODER_FRAME_CHUNK_PATCH_SIZE: int = 0
    VLLM_TPU_STATIC_POOLER: bool = True
    VLLM_TPU_POOLING_KV_CACHE_TOKENS: int = -1
    VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS: int = 8
    VLLM_TPU_MEGAKERNEL: str = ""
    VLLM_TPU_MK_OWN_OPERANDS: bool = True
    VLLM_TPU_MK_HOST_PLE: bool = False
    VLLM_TPU_MK_OPERAND_DUMP: bool = False
    VLLM_TPU_MK_STEP_MIX_INTERVAL_S: float = 10.0
    VLLM_TPU_Q38_CTX_CENSUS: int = 0
    CDK_OUTPUT_DIR: str = "/tmp/cdk_output"
    VLLM_TPU_MK_GDN_CHUNK_GATHER: str = "auto"
    VLLM_TPU_MK_GDN_HM: str = "1"
    VLLM_TPU_MK_GDN_REC_F32: str = "0"
    VLLM_TPU_MK_HIER: str = ""
    VLLM_TPU_MK_MOE_ABLATE: str = ""
    VLLM_TPU_MK_RINGS: str = ""
    VLLM_TPU_HIX_PREGATE: int = 0
    VLLM_TPU_HIX_PREGATE_MAX_ROWS: int = 16
    VLLM_TPU_MK_SCOPES: str = "0"
    VLLM_TPU_MK_UNION_STACK: str = "1"
    VLLM_TPU_MK_UNION_BOUNDED: bool = False
    VLLM_TPU_MK_UNION_LEAN: bool = False
    VLLM_TPU_MK_SLOT_BATCH: bool = False
    VLLM_TPU_MK_UNION_ISSUE: int = 0
    VLLM_TPU_MK_UNION_UNPACK: bool = False
    VLLM_TPU_MK_UNION_PIPE: int = 0
    VLLM_TPU_MK_GDN_META_DEDUP: bool = False
    VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS: int = 0
    VLLM_TPU_MK_SCORE_CHUNK: str = "0"
    VLLM_TPU_MK_SCORE_LANES: str = "0"
    VLLM_TPU_MK_SCORE_MP: str = "0"
    VLLM_TPU_MK_SELECT_RADIX: str = "1"; VLLM_TPU_MK_SCAN_BOUND: str = "0"  # q38 HXSCAN (same line)
    VLLM_TPU_MK_UNION_QSPLIT: str = "0"
    # q38 HXSMALL (the five bitwise S2 smalls; each default "0", strict "0" / "1", registered so it is hashed)
    VLLM_TPU_HXS_GLUE_TRIM: str = "0"
    VLLM_TPU_HXS_LEAN_DRAFT: str = "0"
    VLLM_TPU_HXS_VERIFY_FUSE: str = "0"
    VLLM_TPU_HXS_AR0ST: str = "0"
    VLLM_TPU_HXS_RUNNER_HOIST: str = "0"
    # q38 HIX-GLUE (HIX-LEAD-8 lane; each default "0", strict "0" / "1", registered so it is hashed)
    VLLM_TPU_HXG_SINK_LAYOUT: str = "0"
    VLLM_TPU_HXG_GDN_META: str = "0"
    VLLM_TPU_HXG_PLE_ONE: str = "0"
    VLLM_TPU_MK_TABLE_SLIM: str = "0"
    VLLM_TPU_MK_RING_S1: str = ""
    VLLM_TPU_MK_LIVE_ROWS: bool = False
    VLLM_TPU_MK_LIVE_GDN_COMPUTE: bool = False
    VLLM_TPU_MK_KERN_REC_SKIP: bool = False  # q38 HIX-KERN K8 (default off; hashed)
    VLLM_TPU_MK_GDN_CKPT_INK: bool = False
    VLLM_TPU_MK_GDN_SIDECAR: bool = False
    VLLM_TPU_MK_GDN_COMMIT1: bool = False
    VLLM_TPU_MK_DCP_QX_EARLY: str = "0"
    VLLM_TPU_MK_L47_FOLD: bool = False
    VLLM_TPU_MK_DRAFT_ATTN: bool = False
    VLLM_TPU_MK_GDN_INK: str = "0"
    VLLM_TPU_MK_SELECT_OWNED: str = "0"; VLLM_TPU_MK_IDX_TWIN: str = "0"  # q38 HXIDX (same line)
    VLLM_TPU_MK_QSA_WRITE_ALL: bool = False
    VLLM_TPU_MK_QSA_WRITE_KVAL: bool = False
    VLLM_TPU_MK_KERN_SIDE_PREFETCH: bool = False; VLLM_TPU_MK_KERN_ROUTE_VEC: bool = False; VLLM_TPU_MK_KERN_STOCK_KV: bool = False  # q38 HIX-KERN (same line)
    # q38 PORT-BL GDN (begin)
    VLLM_TPU_GDN_LEAN_GLUE3: bool = False
    VLLM_TPU_GDN_LEAN_GLUE3_COUNT: int = 0
    # q38 PORT-BL GDN (end)
    # q38 PORT-BL UNION (ported from Benliang Wang, 2ce02bb1a)
    VLLM_TPU_QSA_DECODE_RPA_UNION_O1S: int = 0
    VLLM_TPU_RPA_SCHEDULE_HBM_O1S: bool = False


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


def vfy_one_value() -> int:
    """q38 L4 VFY-ONE: the value of VLLM_TPU_VPS_VFY_ONE, 0 (off), 1 (T-3 VFY-ONE: one
    VPS verify program) or 2 (T-3 + DH-C: the program also emits the draft seed).
    Anything else raises. The registered getter below exposes only on / off, so the
    compile key is the same at 1 and 2 (both variants are precompiled when it is on);
    the runner reads the value from here."""
    raw = (os.getenv("VLLM_TPU_VPS_VFY_ONE") or "").strip()
    if raw in ("", "0"):
        return 0
    if raw in ("1", "2"):
        return int(raw)
    raise ValueError(
        f"Invalid value '{raw}' for VLLM_TPU_VPS_VFY_ONE. Valid options: 0, 1, 2."
    )


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
    parse_int = env_int(env_name, default)

    def _get_nonnegative_int_env() -> int:
        value = parse_int()
        if value < 0:
            raise ValueError(f"Invalid value '{value}' for {env_name}: must be >= 0.")
        return value

    return _get_nonnegative_int_env


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
    # QK pair-blocked pooled GDN conv layout: full-width states interleave
    # Q and K row-pairs per tap so a TP-rank shard's first conv token maps
    # to one whole destination pool token (2 contiguous transfer spans per
    # rank per state group).  Identity for head-shard states.  Must match
    # across a disagg pair; the value rides the raiden layout fingerprint.
    # Resolution is layered (see env_override.py): an explicitly set
    # value always wins; otherwise raiden runs derive it from
    # TPU_RAIDEN_TRANSFER_PARALLELISM (enabled iff >= 8, and 8 is that
    # variable's default — so it is on by default under raiden).
    "TPU_GDN_CONV_QK_PAIR_LAYOUT": env_bool(
        "TPU_GDN_CONV_QK_PAIR_LAYOUT", default=False
    ),
    # Hand the pooled GDN kernel a load-time float32 copy of the conv weight
    # in its `[kernel_size, 1, dim]` order. Otherwise every GDN layer swaps
    # and casts the bf16 `[dim, 1, kernel_size]` weight on every forward.
    "VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR": env_bool("VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR"),
    # GDN v3 prefill: DMA bf16 qkv rows into PER_SEQ and cast in VMEM, not in HBM.
    "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D": env_bool("VLLM_TPU_GDN_PREFILL_BF16_QKV_2D"),
    # With VLLM_TPU_GDN_PREFILL_BF16_QKV_2D: the PER_SEQ chunked tile unpacks
    # its bf16 rows token-dense ([rows, dim] f32) and runs the causal conv,
    # SiLU and head split there (a tap is a sublane shift of dense rows),
    # instead of on the one-row-per-vreg compact layout (Qwen3.5's
    # TPU_GDN_PREFILL_DENSE_CONV, translated to q38's pooled GDN v3).
    "VLLM_TPU_GDN_PREFILL_DENSE_CONV": env_bool("VLLM_TPU_GDN_PREFILL_DENSE_CONV"),
    # GDN v3 pooled verify windows (num_spec_tokens > 0, BATCHED tiles): run
    # them in kernels/gdn/v3/window_tile.py, one window position and one value
    # head at a time on dense activations (PR 2821 d8340b973), instead of
    # wrapper.inner_kernel's whole-tile arrays. Bitwise the same outputs and
    # checkpoint bytes; configurations window_tile.supported refuses keep
    # inner_kernel. Default off: the kernel is today's.
    "VLLM_TPU_GDN_WINDOW_TILE": env_bool("VLLM_TPU_GDN_WINDOW_TILE"),
    # With VLLM_TPU_GDN_PREFILL_BF16_QKV_2D, at TP1, for models that opt in
    # (qwen4_exp): hand the pooled GDN op the whole in_proj_qkvz output and let
    # the PER_SEQ kernel DMA the q|k|v columns from it, instead of copying the
    # [tokens, qkv] slice out at the op boundary (as_strided.qkv_ref).
    "VLLM_TPU_GDN_PREFILL_QKV_FROM_PROJ": env_bool("VLLM_TPU_GDN_PREFILL_QKV_FROM_PROJ"),
    # Size GDN v3 decode tiles and prefill chunks from per-rank heads, active
    # sequences and the VMEM estimate (kernels/gdn/v3/tiling.py) instead of
    # the fixed defaults.
    "TPU_ENABLE_GDN_DYNAMIC_TILING": env_bool(
        "TPU_ENABLE_GDN_DYNAMIC_TILING", default=False
    ),
    # Qwen4-exp QSA decode/verify indexer: score all query tokens of a
    # request (e.g. the K+1 MTP verify tokens) in one query block of this many
    # tokens, sharing one stream over the request's compressed keys instead of
    # re-streaming it once per token. Each token keeps its per-token scoring
    # expression, so top-k selections match the per-token path. 0 (default)
    # keeps the per-token path; 4 or 8 covers MTP K <= 3 or K <= 7 per block.
    "VLLM_TPU_QSA_DECODE_QUERY_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_QUERY_BLOCK", 0
    ),
    # Debug only: every N QSA selections, print top-k statistics (run lengths
    # of selected groups, overlap of a request's verify rows) from a host
    # callback. 0 (default) adds nothing to the program.
    "VLLM_TPU_QSA_TOPK_STATS": env_nonnegative_int("VLLM_TPU_QSA_TOPK_STATS", 0),
    # Qwen4-exp ragged prefill indexer: score/select in chunks of this many
    # tokens, overlapping chunk g's SparseCore top-k with chunk g+1's scoring.
    # 0 (default) keeps 512-token chunks for steps that are a multiple of 512
    # and one unchunked call otherwise. A value that does not divide the step
    # (or is not a multiple of 8) falls back to the default. Per-row scores
    # and winners do not depend on the chunking.
    "VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK", 0
    ),
    # Qwen4-exp ragged prefill indexer tiles: queries per block of the prefill
    # passes and compressed keys per KV block (a multiple of 256). Defaults
    # 128 / 256 are the served tiles. Each score keeps its own expression, so
    # the winners do not depend on the tiling.
    "VLLM_TPU_QSA_PREFILL_INDEXER_BQ": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_BQ", 128
    ),
    "VLLM_TPU_QSA_PREFILL_INDEXER_BKV": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_BKV", 256
    ),
    # Compressed keys per KV block of the shared-stream pass above (a multiple
    # of 128). Only read when VLLM_TPU_QSA_DECODE_QUERY_BLOCK > 0.
    "VLLM_TPU_QSA_DECODE_KV_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_KV_BLOCK", 1024
    ),
    # Compressed keys per KV block for the one-token requests of that pass on
    # steps of at most one token per request (MTP draft rows, plain decode);
    # verify steps keep 256. Exact selection at any block size. Only read
    # when VLLM_TPU_QSA_DECODE_QUERY_BLOCK > 0.
    "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK", 256
    ),
    # Qwen4-exp pooled QSA: request row r owns ring row r, so the indexer
    # update's whole-row ring/tag gathers and final scatter become selects.
    "VLLM_TPU_QSA_RING_SELECT": env_bool("VLLM_TPU_QSA_RING_SELECT", default=False),
    # Qwen4-exp QSA ring/tag state I/O: concurrent row DMAs per tile. 0 uses
    # the measured table (pool pages without an entry, e.g. 512-token FP8
    # pages, fall back to one serial DMA per request); N > 0 uses the largest
    # power of two <= N that divides the request rows.
    "VLLM_TPU_QSA_RING_DMA_BATCH": env_nonnegative_int(
        "VLLM_TPU_QSA_RING_DMA_BATCH", 0
    ),
    # Qwen4-exp pooled QSA: write compressed keys with the run-based writer
    # (kernels/qwen4_exp/compressed_write.py): one DMA per run of consecutive
    # pair rows, all in flight, instead of per-tile / per-row serial writes.
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE": env_bool(
        "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", default=False
    ),
    # With VLLM_TPU_QSA_COMPRESSED_RUN_WRITE: steps with fewer key rows keep the
    # tiled writer (the run writer is slower on decode-sized steps).
    "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS", 512
    ),
    # Qwen4-exp QSA: round to BF16 with lax.reduce_precision instead of the
    # integer bit trick. Same RNE; the boundary stays observable to XLA and
    # fuses into neighbouring ops instead of running as its own fusion.
    "VLLM_TPU_QSA_BF16_REDUCE_PRECISION": env_bool(
        "VLLM_TPU_QSA_BF16_REDUCE_PRECISION", default=False
    ),
    # Qwen4-exp QSA indexer: build StreamIndex's step-level metadata (and the
    # pass schedules of decode / verify calls) outside the early-exit cond, so
    # XLA computes them once per step instead of once per QSA layer.
    "VLLM_TPU_QSA_INDEXER_HOIST_META": env_bool(
        "VLLM_TPU_QSA_INDEXER_HOIST_META", default=False
    ),
    # q38 L2-META (D49): GDN v3 PER_SEQ metadata indexes the layer's state,
    # read-offset and checkpoint ids by absolute request index instead of
    # rolling them by distribution[0] in every GDN layer (36 per step on
    # Qwen3.8; each GDN layer is its own KV-cache group, so the ids differ per
    # layer and XLA cannot share the rolls). The kernel reads the same values
    # at every tile; the remaining step metadata is layer-invariant and XLA
    # keeps one copy per step. Changes the traced graph: keys the compile cache.
    "VLLM_TPU_GDN_HOIST_READ_OFFSETS": env_bool(
        "VLLM_TPU_GDN_HOIST_READ_OFFSETS", default=False
    ),
    # Qwen4-exp QSA indexer on decode / verify steps: score every request in the
    # mixed pass and drop the two one-token decode passes (empty on verify
    # steps). Bypasses VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK on one-token steps.
    # Qwen4-exp MTP: the draft's QSA layer attends the target's last QSA layer
    # selection (the request's last verified row) instead of scoring its own
    # indexer: a selection-sharing approximation, gated on real AL.
    # Qwen4Exp n-gram context: "1" = H2D only each row's window of the host token store that the
    # device context can address ([host_ctx - K - ngram + 1, host_ctx), K = spec tokens) instead of
    # the whole [max_num_seqs, max_model_len] table every step (16 MiB at c1, ~2 ms of transfer);
    # "check" = both paths and log mismatches for the first 200 steps, then as "1"
    # (models/vllm/qwen4_exp/inputs.py).
    "VLLM_TPU_QWEN4_NGRAM_WINDOW": lambda: os.getenv("VLLM_TPU_QWEN4_NGRAM_WINDOW", ""),
    # Qwen4Exp: run the adapter's per-step input preparation (n-gram context,
    # speculative record, slots) as one compiled program instead of ~20 eager
    # launches; models/vllm/qwen4_exp/inputs.py fused_step_inputs.
    "VLLM_TPU_QWEN4_FUSED_PREP": env_bool("VLLM_TPU_QWEN4_FUSED_PREP"),
    # "check": compare the fused input preparation with the eager path on every
    # live step (counts logged every 500 checks); "fault": its positive control.
    "VLLM_TPU_QWEN4_FUSED_PREP_CHECK": lambda: os.getenv(
        "VLLM_TPU_QWEN4_FUSED_PREP_CHECK", ""
    ),
    # The fused program returns the updated speculative record instead of
    # mutating its inputs (no eager copy-back per record tensor after it).
    "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL": env_bool(
        "VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL"
    ),
    # q38 L4 X-D14-DRAFT-2 (a): the MTP draft's vocab-parallel embedding all-reduce
    # goes through the armed H8 decode all-reduce (VLLM_TPU_DECODE_TP_AR) instead of
    # XLA's; a no-op unless H8 is armed. Changes the traced draft graph (hashed).
    "VLLM_TPU_DRAFT_EMBED_H8": env_bool("VLLM_TPU_DRAFT_EMBED_H8", default=False),
    # q38 L4 X-D14-DRAFT-2 (c): the MTP draft embeds from a replicated copy of the shared
    # vocab-parallel embedding (assembled once at load), so its per-pass embedding all-reduce
    # is gone; takes precedence over VLLM_TPU_DRAFT_EMBED_H8. Every TP rank must agree (load-time
    # fail-stop). +1.27 GB HBM per rank for Qwen3.8. Changes the traced draft graph (hashed).
    "VLLM_TPU_DRAFT_EMBED_REPL": env_bool("VLLM_TPU_DRAFT_EMBED_REPL", default=False),
    # q38 L4 EMB-REP-TGT: the target's embedding (the module the MTP draft borrows) also looks up
    # DRAFT-2 (c)'s replicated table, removing one embedding all-reduce per target forward (stock,
    # segment and D14 paths alike: every site calls the module). Requires VLLM_TPU_DRAFT_EMBED_REPL=1;
    # all TP ranks must agree (load-time fail-stop). No HBM beyond DRAFT-2 (c)'s table. Hashed.
    "VLLM_TPU_TARGET_EMBED_REPL": env_bool("VLLM_TPU_TARGET_EMBED_REPL", default=False),
    "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION": env_bool(
        "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION"
    ),
    # q38 X-D14-DRAFT-1. With VLLM_TPU_QSA_DRAFT_SHARED_SELECTION: the MTP draft's
    # shared-selection ops write its main K/V only. No index_qk_proj, no raw-ring /
    # tag read, no _indexer_updates, no compressed-key / ring / tag write: once bound,
    # the draft attends the target's carried selection on every pass, so nothing reads
    # them. Changes the draft's traced graph (hashed in the compile key).
    "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE": env_bool("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE"),
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": env_bool(
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY", default=False
    ),
    # Qwen4-exp QSA decode/verify attention: run the decode rows (and the
    # decode prefix of mixed steps) on the fused sparse GQA decode kernel
    # (kernels/sparse_gqa_decode) instead of batched RPA. Inputs outside the
    # kernel's static envelope (non-TPU7x, unsupported shapes) still use RPA.
    "VLLM_TPU_QSA_FUSED_SPARSE_DECODE": env_bool(
        "VLLM_TPU_QSA_FUSED_SPARSE_DECODE", default=False
    ),
    # Mixed QSA steps: move the decode prefix off the prefill rows with contiguous dynamic
    # slices instead of row gathers (same values; prefill chunks spent ~3 ms/8K on the gathers).
    "VLLM_TPU_QSA_PREFIX_SLICE": env_bool("VLLM_TPU_QSA_PREFIX_SLICE", default=False),
    # Mixed QSA steps without a decode prefix (prefix 0: every step of a disaggregated prefill engine but a tiny
    # leading request): the prefill kernel reads the step's rows in place, with no row gathers in and no gather /
    # merge out; steps with a prefix route as before (with PREFIX_SLICE's slices if set). Same values (bitwise).
    "VLLM_TPU_QSA_ZERO_PREFIX_FAST": env_bool("VLLM_TPU_QSA_ZERO_PREFIX_FAST", default=False),
    # QSA indexer cache updates: build each token's compression-group members from static
    # shifts and the ring / tag rows from a one-hot product instead of per-token row gathers
    # (same values; ~3 ms of gathers per 8K prefill chunk).
    "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT": env_bool(
        "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT", default=False
    ),
    # Qwen4-exp QSA decode/verify attention: present every head slot of the
    # main KV row (zero queries for the other layer's head) so all QSA layers
    # share one decode kernel configuration and XLA computes the decode
    # schedule once per step; RPA also skips its empty mixed pass.
    "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE": env_bool(
        "VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE", default=False
    ),
    # Qwen4-exp QSA decode/verify attention on RPA: fetch each KV page with
    # one scalar load instead of three (the decode KV fetch issues one DMA
    # per 4-token group and is scalar-issue bound). Same DMAs and results.
    "VLLM_TPU_QSA_DECODE_LEAN_FETCH": env_bool(
        "VLLM_TPU_QSA_DECODE_LEAN_FETCH", default=False
    ),
    # Qwen4-exp QSA decode/verify attention on RPA with FP8 main KV rows:
    # each layer attends its own head slot of the packed row (one KV head,
    # only its K/V pair unpacked) instead of unpacking every slot and, for a
    # paired page's second layer, presenting both heads with zero queries.
    "VLLM_TPU_QSA_DECODE_SLOT_READ": env_bool(
        "VLLM_TPU_QSA_DECODE_SLOT_READ", default=False
    ),
    # Qwen4-exp QSA decode/verify attention on RPA (implies the lean fetch):
    # full KV tiles fetch with static page sizes and page ids read at static
    # offsets, cutting the scalar work per DMA that bounds the decode kernel.
    "VLLM_TPU_QSA_DECODE_FAST_FETCH": env_bool(
        "VLLM_TPU_QSA_DECODE_FAST_FETCH", default=False
    ),
    # Qwen4-exp QSA decode/verify attention on RPA: compact KV schedule (one
    # entry per lane-tile instead of one per 4-token page), which lets the
    # decode KV block grow past 256 tokens within the SMEM schedule budget.
    # The RPA body has a fixed cost per step, so fewer, larger steps help.
    "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE": env_bool(
        "VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE", default=False
    ),
    # q38mm DSTOCK D12 (ported fast only): with the compact schedule and the
    # lean fetch, "fast" issues the decode RPA KV fetch on the clamp-free row
    # path (batched_rpa/configs.py ServingConfigs.kv_fetch_mode); "" (default)
    # keeps the per-page path. The same bytes to the same VMEM offsets (bitwise).
    # Changes the kernel (name suffix -f), so it keys the compile cache.
    "VLLM_TPU_QSA_DECODE_RPA_FETCH": lambda: os.getenv(
        "VLLM_TPU_QSA_DECODE_RPA_FETCH", ""
    ),
    # Decode KV tokens per RPA step and lanes (requests) per step with the
    # compact schedule; only read when it is on.
    "VLLM_TPU_QSA_DECODE_RPA_BKV": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_BKV", 256
    ),
    # Qwen4-exp shared-stream decode indexer: pass StreamIndex the pool-page
    # table and expand each entry into its kernel pages inside the KV reader.
    # The scalar-prefetched table shrinks by the kernel pages per pool page
    # (13 at 1664-token pages), so a full-batch verify step scores in one
    # query window instead of several serial ones. Same pages, same order.
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE": env_bool(
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE", default=False
    ),
    # With the pool-page table: when the decode KV block holds whole pool
    # pages (VLLM_TPU_QSA_DECODE_KV_BLOCK a multiple of 13 kernel pages x 32
    # keys, e.g. 3328), DMA each pool page's consecutive kernel pages at once
    # instead of one 16 KiB descriptor per kernel page. Same bytes and order.
    "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA": env_bool(
        "VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA", default=False
    ),
    # Qwen4-exp shared-stream decode indexer: SparseCore top-k copies only
    # each score row's live prefix (one subcore per row) instead of every
    # subcore copying its whole slice of a row sized for max_model_len.
    # Same winners; selection time follows the context, not the table width.
    "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY": env_bool(
        "VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY", default=False
    ),
    # q38 L2 D24: StreamIndex reads the QSA index keys (BF16 values in the 8-bit
    # pool carrier) as 32-bit words with a sublane-strided load and a word ->
    # BF16 vreg bitcast, instead of a BF16 view whose reshape and lane slice
    # repack sublanes (vrot / vcombine). The same key bits, so the same scores
    # and selections. Changes the scoring kernel (name suffix -sk): compile key.
    "VLLM_TPU_QSA_INDEXER_STRIDED_KEYS": env_bool(
        "VLLM_TPU_QSA_INDEXER_STRIDED_KEYS", default=False
    ),
    # Qwen4-exp decode / verify indexer (DEC-FB lever 11): StreamIndex copies
    # only the first N lanes of each compressed-key row. The BF16 index keys
    # (indexer_head_dim 128) sit in 256-lane pool rows whose upper lanes are
    # never scored, so 128 halves the key bytes with the same scores. 0 (the
    # default) copies whole rows. A multiple of 128 between the index query
    # width and the pool row width. Prefill keeps whole rows.
    "VLLM_TPU_QSA_INDEXER_KEY_LANES": env_nonnegative_int(
        "VLLM_TPU_QSA_INDEXER_KEY_LANES", 0
    ),
    # q38 PORT-BL IDX (ported from Benliang Wang <benliangw@google.com>, commit
    # 2ce02bb1a on benliangw/q38fh-integ-pin-5b44702aea; his names and meaning).
    # Qwen4-exp decode indexer (StreamIdxTC mixed pass with per-token scores):
    # a 4-row query block is scored head-major on whole vregs (the head sums
    # in the per-token expression's (h0 + h2) + (h1 + h3) order) and the
    # block's page DMAs take their table rows without per-page carries or end
    # clamps. The keys come from the pass's own reader (on v8 the D24 strided
    # read, VLLM_TPU_QSA_INDEXER_STRIDED_KEYS). Kernel name suffix -ds.
    # Changes the scoring kernel: compile key (not ignored).
    "VLLM_TPU_QSA_INDEXER_DENSE_STEP": env_bool(
        "VLLM_TPU_QSA_INDEXER_DENSE_STEP", default=False
    ),
    # q38 PORT-BL IDX (ported from Benliang Wang, as above): a decode KV
    # block's page DMAs stop at the last group of this many kernel pages that
    # holds keys of its sequence (0: whole blocks). The skipped keys lie past
    # the sequence and are masked; their VMEM rows keep stale bytes that no
    # score reads. Kernel name suffix -ts<N>. Compile key (not ignored).
    "VLLM_TPU_QSA_INDEXER_TAIL_SKIP": env_nonnegative_int(
        "VLLM_TPU_QSA_INDEXER_TAIL_SKIP", 0
    ),
    # Qwen4-exp prefill indexer: the same bounded SparseCore score copy for
    # prefill chunks (rows early in a prompt see a short context).
    "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY": env_bool(
        "VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY", default=False
    ),
    "VLLM_TPU_QSA_DECODE_RPA_LANES": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_LANES", 8
    ),
    # Qwen4-exp QSA decode/verify attention with the compact schedule: query
    # rows per RPA call (0 keeps the 64-row SMEM bound of the per-page
    # schedule). Larger steps otherwise run as a scan of 64-row calls.
    "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES", 0
    ),
    # Qwen4-exp QSA prefill: build the per-query selection bitset with a
    # compare-and-accumulate Pallas kernel. Off falls back to the vmapped
    # scatter-add, which XLA:TPU lowers to a global sort of every selection.
    "VLLM_TPU_QSA_PREFILL_BITSET_KERNEL": env_bool(
        "VLLM_TPU_QSA_PREFILL_BITSET_KERNEL", default=True
    ),
    # Qwen4-exp QSA prefill attention: feed the MXU BF16 operands (exact
    # queries and keys, probabilities split into two BF16 terms) instead of
    # FP32 operands, which Mosaic multiplies at full FP32 precision.
    "VLLM_TPU_QSA_PREFILL_BF16_MXU": env_bool(
        "VLLM_TPU_QSA_PREFILL_BF16_MXU", default=False
    ),
    # qwen4_exp QSA sparse prefill: skip the FP8 key finite/written guard,
    # which the score mask select already covers (bit-identical output).
    "VLLM_TPU_QSA_PREFILL_LEAN": env_bool("VLLM_TPU_QSA_PREFILL_LEAN"),
    # Qwen4-exp QSA prefill: query subtile (rows that share one decoded KV tile)
    # for paired-page tiles whose table entry leaves it to the kernel default
    # (the query block). 64 measured bitwise and 6-9% faster at Q256/KV512 on
    # a TP1 stage's 3200-token pages (P-ATTN i1f, j-03e68c32). 0 = default.
    "VLLM_TPU_QSA_PREFILL_QUERY_TILE": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_QUERY_TILE", 0
    ),
    # Qwen4-exp QSA prefill on a TP1 stage's 3200-token pages: run token counts
    # that are a multiple of 128 but not of 256 (the 3200-token bucket) on the
    # paired Q128/KV512 kernel instead of the unpaired Q256/KV128 fallback.
    # Not bitwise against the fallback (the online softmax regroups at 512
    # keys); identity/accuracy gated (reports/prereg/q38_attn_tile3200_plan_DRAFT.md).
    "VLLM_TPU_QSA_PREFILL_Q128_FALLBACK": env_bool(
        "VLLM_TPU_QSA_PREFILL_Q128_FALLBACK", default=False
    ),
    # Qwen4-exp QSA prefill: the D2 union-walk attention kernel
    # (qsa_union_dense.union_walk_prefill) with query groups of this many rows
    # (a divisor of 128; 0, the default, keeps the served block-sparse kernel).
    # Only Q24/KV2/D256 steps on an FP8 pool of 128-aligned pages, at least
    # 1024 tokens and a multiple of the group, without DCP, a KV head offset or
    # LSE output. Each step falls back to the served kernel when its longest
    # row is below VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT tokens or a group's
    # union exceeds VLLM_TPU_QSA_PREFILL_UNION_CAP 4-token groups. Not bitwise
    # against the served kernel (union-order accumulation); accuracy gated.
    "VLLM_TPU_QSA_PREFILL_UNION_WALK": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_WALK", 0
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", 32768
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_CAP": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_CAP", 12288
    ),
    # KV tile of the union walk: union groups per tile times 4 (256-2048).
    "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_KV_TILE", 1024
    ),
    # Debug: log the D2 union walk's eligible prefill steps every N steps on each stage process
    # (host-side count, log-only, no graph change; 0 = off). See qsa_union_dense.count_union_walk_step.
    # P-DENSE engagement counters (models/vllm/qwen4_exp/pdense_engagement.py): log every N
    # prefill steps which P-DENSE paths ran (host-side, log-only, no graph change; 0 = off).
    "VLLM_TPU_PDENSE_COUNT": env_nonnegative_int("VLLM_TPU_PDENSE_COUNT", 0),
    # P-DENSE trace-time records (pdense_engagement): off by default, because recording while vLLM traces with a
    # dynamic token dimension specializes it (engine init failed: q38 p75 j-8eb772a8). Host-only, debug.
    "VLLM_TPU_PDENSE_TRACE_RECORDS": env_bool("VLLM_TPU_PDENSE_TRACE_RECORDS"),
    "VLLM_TPU_QSA_PREFILL_UNION_COUNT": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_COUNT", 0
    ),
    # Union walk options (all off by default; bitwise unless noted):
    # LISTED: issue the walk's copies from a union id list instead of scanning
    # the union bitset; "sparsecore" (union_sc_list, TPU) or "reference" (XLA).
    # SHARE_BITS: a cap fallback reuses the walk plan's selection bitset.
    # ALPHA: also fall back when a group's union exceeds this fraction of its
    # visible groups (0 = off). QK16: QK as one BF16 MXU pass (exact operands,
    # power-of-two score scales only; bitwise on TPU in q38-attn-exact-i1k2).
    "VLLM_TPU_QSA_PREFILL_UNION_LISTED": env_str("VLLM_TPU_QSA_PREFILL_UNION_LISTED", ""),
    "VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS": env_bool("VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS"),
    "VLLM_TPU_QSA_PREFILL_UNION_ALPHA": env_float("VLLM_TPU_QSA_PREFILL_UNION_ALPHA", 0.0),
    "VLLM_TPU_QSA_PREFILL_UNION_QK16": env_bool("VLLM_TPU_QSA_PREFILL_UNION_QK16"),
    # RUN_TABLES (with LISTED; parked, default off): the walk issued from SparseCore run tables.
    "VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES": env_bool("VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES"),
    # LAZY_SHARE (with SHARE_BITS): walk on group-aligned rows; only a fallback builds the served layout.
    "VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE": env_bool("VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE"),
    # SC_BITSET: the union plan's per-row bitset built on the SparseCore (the same bits), off the TensorCore.
    "VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET": env_bool("VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET"),
    # IN_PLACE (with GATHERED=sparsecore): gather/walk chunks read the plan's bits, queries and lengths in place
    # (static chunk offsets) instead of from sliced copies, and the walk stages no unused bitset block. Bitwise.
    "VLLM_TPU_QSA_PREFILL_UNION_IN_PLACE": env_bool("VLLM_TPU_QSA_PREFILL_UNION_IN_PLACE"),
    # SC_BITS_PLAN (with SC_BITSET; ignored without it): the SparseCore bitset kernel writes the plan layout
    # [G * words, 128] itself (each 16-row slab into its 16 lanes of its group's rows), so the TensorCore regroup
    # (reshape, copy and pad) is gone. The group's lanes are the same bits; lanes >= group are left unwritten (no reader
    # uses them).
    "VLLM_TPU_QSA_PREFILL_UNION_SC_BITS_PLAN": env_bool("VLLM_TPU_QSA_PREFILL_UNION_SC_BITS_PLAN"),
    # GATHERED_LEAN (with IN_PLACE; ignored without it): the in-place gathered walk takes no union words and no page
    # rows, which it never reads (no per-chunk union-word slices / padding / staging, no per-word pool bases). Bitwise.
    "VLLM_TPU_QSA_PREFILL_UNION_GATHERED_LEAN": env_bool("VLLM_TPU_QSA_PREFILL_UNION_GATHERED_LEAN"),
    # q38 QA-1 (PREFILL-10 / QA1BUILD; all default off; bitwise on finite pools; gathered walk only):
    # MASK_VIEW: the walk tile applies its [rows, keys] visibility on a [q heads per KV head, rows, keys] view of the
    # scores instead of materializing the per-head broadcast (the same select, so the same scores).
    "VLLM_TPU_QSA_PREFILL_UNION_MASK_VIEW": env_bool("VLLM_TPU_QSA_PREFILL_UNION_MASK_VIEW"),
    # VGUARD_LEAN: the walk tile's value guard keeps only its `written` term (no isfinite test). Every row the
    # gathered walk reads is a written pool row (union ranks, or the padding row = word 0 of the group's page
    # row), so on a finite pool the isfinite term is always true and the select is the same.
    "VLLM_TPU_QSA_PREFILL_UNION_VGUARD_LEAN": env_bool("VLLM_TPU_QSA_PREFILL_UNION_VGUARD_LEAN"),
    # ALIGN_GATHER (PA-0): union_walk_prefill builds its group-aligned row layout by an inverse row gather (each
    # aligned slot computes its source row) instead of a fill + scatter. Pure data movement: bit-identical.
    "VLLM_TPU_QSA_PREFILL_UNION_ALIGN_GATHER": env_bool("VLLM_TPU_QSA_PREFILL_UNION_ALIGN_GATHER"),
    # q38 UNION-BUILD (default off; gathered walk only): WALK_PIPE: the gathered walk's tile loop software-pipelined (stage A
    # = mask / decode / guard / masked QK of tile t + 1 beside stage B = softmax / PV / state of tile t, in one block, the
    # staged scores and values in VMEM). The served per-tile ops on the same operands, the state in tile order: bitwise
    # the served walk (a TPU measurement). 2 = the two stages interleaved per KV head in program order. Kernel name gains
    # _pipe / _pipe2; one engagement line per trace. 0 / unset = off.
    "VLLM_TPU_QSA_PREFILL_UNION_WALK_PIPE": env_int("VLLM_TPU_QSA_PREFILL_UNION_WALK_PIPE", 0),
    # q38 UNION-BUILD (default off; gathered walk only): DOT16: both walk dots take BF16 operands (queries once per block, keys and
    # values decoded to BF16, probabilities cast at the PV dot). Mosaic's f32 dot there is one BF16 pass with round-to-nearest
    # operands (M, T83m precision probe), so the dots' results are the served ones; bitwise checked on TPU. Kernel name _d16.
    "VLLM_TPU_QSA_PREFILL_UNION_DOT16": env_bool("VLLM_TPU_QSA_PREFILL_UNION_DOT16"),
    # q38 UNION-BUILD (default off; gathered walk only): VGUARD_NONE: no value guard (QA-1's VGUARD_LEAN keeps `written`, which is
    # true on every rank whose probability can be non-zero). Bitwise on finite read rows (QA-1's precondition). Kernel name _vn.
    "VLLM_TPU_QSA_PREFILL_UNION_VGUARD_NONE": env_bool("VLLM_TPU_QSA_PREFILL_UNION_VGUARD_NONE"),
    # q38 PSTEP-QSA WALK-FOS (default off; NOT bitwise, accuracy-gated): the gathered union walk's softmax in packed BF16
    # (MM P17 R-B port), FOS 1 (lag-1 offset) / 2 (static offset 0 with the exact re-walk guard) on it (MM L-A port) and
    # the lean FP8 tile (MM L-B port). FOS and LEAN_TILE need BF16_SOFTMAX.
    "VLLM_TPU_QSA_PREFILL_UNION_BF16_SOFTMAX": env_bool("VLLM_TPU_QSA_PREFILL_UNION_BF16_SOFTMAX", default=False),
    "VLLM_TPU_QSA_PREFILL_UNION_FOS": env_int("VLLM_TPU_QSA_PREFILL_UNION_FOS", 0),
    "VLLM_TPU_QSA_PREFILL_UNION_LEAN_TILE": env_bool("VLLM_TPU_QSA_PREFILL_UNION_LEAN_TILE", default=False),
    # GATHERED ("", "reference", "sparsecore"; with LISTED): the walk-free kernel on chunked gathered tiles.
    "VLLM_TPU_QSA_PREFILL_UNION_GATHERED": env_str("VLLM_TPU_QSA_PREFILL_UNION_GATHERED", ""),
    "VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS": env_int("VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS", 4),
    # GATHER_GROWTH (>= 1; 1 = even chunks): chunk k of the gathered walk holds about growth**k as many query groups
    # as chunk 0, so the first gather (the one the TensorCore waits on) is small. Same outputs (bitwise).
    "VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH": env_float("VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH", 1.0),
    # COMPACT_REQS (N > 0; with GATHERED): the SparseCore takes at most 16 GiB per tensor, so the gather reads a
    # per-step compact pool of the first N requests' pages (whole-page copies); a step with more requests takes the
    # listed walk on the full pool (bitwise the gathered walk). 0 (default): the gather reads the pool itself.
    "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS": env_nonnegative_int("VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS", 0),
    # COMPACT_COPY: how the compact pool is copied ("vmem": whole pages staged through VMEM by DMA; "pipe": a Pallas
    # pipeline copying 32-bit words; "dma": HBM-to-HBM DMAs, slow on TPU per i2b j-950efc5d). All byte-exact.
    "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY": env_str("VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY", "vmem"),
    # Main KV of the target QSA layers split over TP rank pairs that share a
    # KV head (2; 0/1 = off): each rank stores and attends half the tokens,
    # the pair's partial attention is LSE-combined.
    "VLLM_TPU_QSA_MAIN_KV_DCP": env_nonnegative_int("VLLM_TPU_QSA_MAIN_KV_DCP", 0),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: the MTP draft layers' main KV split the
    # same way (otherwise the draft keeps whole pages).
    "VLLM_TPU_QSA_MTP_KV_DCP": env_bool("VLLM_TPU_QSA_MTP_KV_DCP"),
    # Main-KV DCP decode attention: gather the pair's partial outputs and LSEs
    # in one all-gather instead of two (pure data movement; q38 D-MK).
    "VLLM_TPU_QSA_DCP_ONE_GATHER": env_bool("VLLM_TPU_QSA_DCP_ONE_GATHER"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP and VLLM_TPU_QSA_MTP_KV_DCP: compressed keys
    # split the same way; each rank scores its owned groups and the pair merges
    # the local top-k lists into the exact global selection.
    "VLLM_TPU_QSA_COMPRESSED_DCP": env_bool("VLLM_TPU_QSA_COMPRESSED_DCP"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: "xla" merges the DCP group's partial attention with one all-to-all of head
    # slices (output + LSE packed per destination) instead of two all-gathers of every head; each rank receives
    # only its own heads (a quarter of the bytes at DCP4) and runs the same merge, so the output is bitwise
    # today's (kernels/collectives/dcp_head_a2a.py, DEC-COLL). "" (default) keeps the all-gathers.
    "VLLM_TPU_QSA_DCP_HEAD_A2A": env_str("VLLM_TPU_QSA_DCP_HEAD_A2A", ""),
    # q38 L2 D27a: with VLLM_TPU_QSA_MAIN_KV_DCP, the decode DCP ops all-gather only the q halves of qgate over the
    # pair (the DCP attention returns ungated partials and _dcp_finish gates with the local qgate, so the gathered
    # gate halves were dead data); half the bytes of that all-gather. Bitwise by construction. Default off.
    "VLLM_TPU_QSA_DCP_Q_ONLY_GATHER": env_bool("VLLM_TPU_QSA_DCP_Q_ONLY_GATHER"),
    # Qwen4-exp pooled QSA decode/verify steps: write main KV, compressed
    # indexer keys and the raw ring/tags of a layer with one Pallas kernel
    # whose row DMAs are all in flight together, and read the ring with every
    # request's DMA outstanding. Prefill-sized steps keep the tiled writers.
    "VLLM_TPU_QSA_FUSED_KV_WRITE": env_bool(
        "VLLM_TPU_QSA_FUSED_KV_WRITE", default=False
    ),
    # Qwen4-exp pooled QSA: split the lightning indexer across the TP group.
    # Each rank scores an interleaved 1/TP of every request's compressed keys,
    # the local top-k candidates are gathered, and every rank merges the exact
    # global top-k. Off: every rank scores every key (replicated indexer).
    "VLLM_TPU_QSA_INDEXER_SHARD": env_bool("VLLM_TPU_QSA_INDEXER_SHARD", default=False),
    # With VLLM_TPU_QSA_INDEXER_SHARD: steps of at least this many tokens merge
    # a 1/TP token block per rank after an all-to-all and all-gather the merged
    # blocks, instead of all-gathering every rank's candidates. 0: never.
    "VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS", 0
    ),
    # With VLLM_TPU_QSA_INDEXER_SHARD: the prefill sequence-parallel backbone
    # (QWEN4_EXP_PREFILL_SP, steps of at least its minimum token count) keeps
    # the replicated indexer; only the smaller steps shard it. The prefill
    # candidate exchange costs more than the sharded scoring saves. The two
    # backbones are separate graphs with disjoint buckets, so the choice is
    # static per graph.
    "VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP": env_bool(
        "VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP", default=False
    ),
    # Qwen4-exp pooled QSA, prefill sequence-parallel backbone only: each TP
    # rank runs the indexer on one contiguous 1/TP of the gathered step's
    # tokens and the selected blocks are all-gathered, instead of every rank
    # indexing every token. The selection is the replicated indexer's. Takes
    # precedence over VLLM_TPU_QSA_INDEXER_SHARD in that graph.
    "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD": env_bool(
        "VLLM_TPU_QSA_INDEXER_TOKEN_SHARD", default=False
    ),
    # q38 O1s: request-sharded decode selection under main-KV DCP
    # (VLLM_TPU_QSA_MAIN_KV_DCP > 1, TP > 1). On decode-shaped steps each TP
    # rank runs the QSA indexer for its 1/TP share of the step's requests
    # instead of all of them; the ranks all-gather the unfiltered selected
    # group ids (int32) and each attends its owned main-KV rows of the merged
    # selection. Same selection per row as the replicated indexer. Off: the
    # replicated indexer, unchanged.
    "VLLM_TPU_QSA_DECODE_REQUEST_SHARD": env_bool(
        "VLLM_TPU_QSA_DECODE_REQUEST_SHARD", default=False
    ),
    # q38 L2 (D21 min-rows gate): with VLLM_TPU_QSA_DECODE_REQUEST_SHARD on,
    # only graphs of at least this many rows (the padded bucket's token count)
    # take the request-sharded selection; smaller decode buckets keep every
    # row on every rank (the request-shard module's plain DCP step, which is
    # pooled_qsa_dcp plus an unused operand). O1s carries a near-fixed cost
    # (one all-gather per QSA layer plus the re-base / reassembly glue) that
    # its indexer saving does not cover at small row counts. 0 = no gate.
    "VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_REQUEST_SHARD_MIN_ROWS", 0
    ),
    # q38 L2 O1s-HOIST (port of MM DSTOCK-B HOIST H-A, q38mm-dstockb 581cfffd3 /
    # int-5 97b9b6e17): every QSA layer on a rank registers the SAME int32 [1]
    # rank tensors (dcp_rank, request_shard_index, token_shard_index;
    # models/vllm/qwen4_exp/qsa_rank_operands.py), so the compiled step takes
    # one rank input instead of one per layer and XLA can merge the
    # layer-invariant O1s selection metadata (request_shard_step and the
    # shared-stream _request_window) across the 12 QSA layers. Values are
    # unchanged and stay operands. Off: a fresh tensor per layer, as before.
    "VLLM_TPU_QSA_SHARED_RANK_OPERANDS": env_bool(
        "VLLM_TPU_QSA_SHARED_RANK_OPERANDS", default=False
    ),
    # q38 L2 O1s-HOIST port 2 (MM's compressed-DCP decode indexer runs without
    # it: int-5 _dcp_topk pops enable_early_exit): the decode shared-stream
    # indexer calls StreamIndex without its per-layer early-exit lax.cond
    # (max(seq_lens) // ratio <= k picks a fast path that lists every visible
    # group). At served contexts the cond always takes the kernel path; the
    # kernel path selects the same set at short contexts (order may differ;
    # the attention consumes the set). Off: the cond, unchanged.
    "VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT": env_bool(
        "VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT", default=False
    ),
    # Qwen4-exp resident PLE table: tokens per gather in the raw embedding
    # lookup (a multiple of 8). 8 is the originally validated chunk; an
    # 8192-token prefill step then runs 1024 sequential gathers.
    "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS": env_nonnegative_int(
        "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS", 8
    ),
    # q38 PORT-BL MOE (PLE lookup chunk; Benliang Wang <benliangw@google.com> serves
    # VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS=512 on benliangw/q38fh-integ-pin-5b44702aea):
    # with N > 0, VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS applies only to PLE lookups of at
    # least N tokens; smaller lookups keep the default 8-token gathers, so the
    # megakernel (D14) buckets keep today's program (N = 80 with D14 at <= 64
    # rows). 0 (default): the chunk applies to every lookup, as before. Bitwise the
    # same rows either way (the gather is per token). Changes the traced program,
    # so it is a compile-key flag.
    "VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS": env_nonnegative_int(
        "VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS", 0
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
    # Stage-3 producers of state-bearing (Mamba / request-state) models: end
    # the handoff on a multiple of this many tokens instead of one token short
    # of the prompt, and let the decode peer recompute the rest. Set it to the
    # decode peer's prefix-hit alignment (its largest token page, e.g. 512-row
    # pages x main-KV DCP 4 = 2048): the transferred running state then sits on
    # a decode block boundary, which is the decode's replay checkpoint, so the
    # decode can cache it and the next turn can hit there. 0 keeps the
    # one-token handoff. Prompts shorter than one alignment unit also keep it.
    "TPU_RAIDEN_HANDOFF_ALIGN_TOKENS": env_int("TPU_RAIDEN_HANDOFF_ALIGN_TOKENS", 0),
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
    # qwen4_exp: run each hyper-connection mix (per-branch norm, low-rank
    # gate, branch mean and the combine's injection gate) at <= 512 tokens as
    # one Pallas call, with the attn combine fused ahead of the mlp mix.
    "TPU_QWEN4_HC_DECODE_FUSED": env_bool("TPU_QWEN4_HC_DECODE_FUSED"),
    # With TPU_QWEN4_HC_DECODE_FUSED: the fused decode HC kernel reads the up
    # weight as up.T ([lowrank, hc * hidden]), a bitcast of the {0,1} layout
    # the served programs keep it in, instead of taking [hc * hidden, lowrank]
    # through a 13 MB XLA relayout copy before every call (copy.weight_ref).
    # Same kernel inputs; only the MXU accumulation order may differ. Default
    # off. DEC-HC (plan v4 lane, DEC-FB lever 1).
    "TPU_QWEN4_HC_DECODE_UP_T": env_bool("TPU_QWEN4_HC_DECODE_UP_T"),
    # q38 PORT-BL HC (ported from Benliang Wang <benliangw@google.com>, commits d523c2ce0 / 28419c4a5 on
    # benliangw/q38fh-integ-pin-5b44702aea). Both change traced programs: hashed (not compile-ignored).
    # With TPU_QWEN4_HC_DECODE_FUSED: each decoder layer hands its mlp HC combine to the next layer's
    # attn HC (never into a PLE layer) and the last layer's to the final mixer, where it runs as the
    # prologue of that HC's fused decode mix (one Pallas call instead of an XLA combine + the mix).
    # Deferred only at row counts the fused decode prologue covers (with VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS
    # = 128: 80-128 stock rows); every other row count keeps today's trace. Off under sequence
    # parallelism; D14 / forward_segments and the MTP draft never read it. Default off.
    "TPU_QWEN4_HC_DECODE_FOLD": env_bool("TPU_QWEN4_HC_DECODE_FOLD"),
    # With TPU_QWEN4_HC_DECODE_FUSED: steps of >= 128 rows run the fused decode HC kernel as the fewest
    # equal row tiles of at most N rows that are multiples of 16 (192 rows: one tile, not three of 64).
    # 0 (default) = today's tiles; else a multiple of 16 in [128, 512]. Changes only the kernel's tile,
    # never which rows take the fused path (decode_token_tile / HCGATE decide that).
    "TPU_QWEN4_HC_DECODE_TILE_ROWS": env_nonnegative_int("TPU_QWEN4_HC_DECODE_TILE_ROWS", 0),
    # q38 PORT-BL HC end
    # DEC-HC engagement (debug, log-only; in _TPU_COMPILE_ENV_IGNORED): every N
    # target forwards, log how many HC modules run the fused decode mix and how
    # many forwards had a fused-decode row count (models/vllm/qwen4_exp/
    # hc_engagement.py). 0 (default) = off.
    "VLLM_TPU_HC_DECODE_COUNT": env_nonnegative_int("VLLM_TPU_HC_DECODE_COUNT", 0),
    # q38 L2 STACK-MARK (debug, log-only): every N target forwards each rank logs per-bucket route counts
    # (O1s request shard, fused decode HC, GDN window-tile flag) from the code's own predicates
    # (models/vllm/qwen4_exp/stack_route.py). Host only; compile-ignored. 0 (default) = off.
    "VLLM_TPU_Q38_STACK_ROUTE_COUNT": env_nonnegative_int("VLLM_TPU_Q38_STACK_ROUTE_COUNT", 0),
    # qwen4_exp prefill: let the fused HC mix (hc_full) and the gated_residual
    # combine take any whole number (>= 2) of hc_prefill_full_tile tiles, not
    # only even counts, so 6,400- and 3,200-token steps leave the unfused
    # fallback (norm kernel + XLA down + up epilogue + XLA combine).
    "TPU_QWEN4_HC_PREFILL_ODD_TILES": env_bool("TPU_QWEN4_HC_PREFILL_ODD_TILES"),
    # qwen4_exp prefill: one Pallas pass per HC mix that also produces the
    # combine's injection gate (norm never reaches HBM), with the attn combine
    # fused ahead of the mlp mix; whole hc_prefill_full_tile tiles only.
    "TPU_QWEN4_HC_PREFILL_FUSED_MIX": env_bool("TPU_QWEN4_HC_PREFILL_FUSED_MIX"),
    # qwen4_exp prefill: the fused HC kernels (hc_full, the fused mix, the
    # gated_residual combine) also take a partial last hc_prefill_full_tile
    # tile (any multiple of 32 rows, at least two tiles), so every served step
    # size runs 256-row tiles (3,200 = 12 x 256 + 128). Implies odd counts.
    "TPU_QWEN4_HC_PREFILL_PARTIAL_TILE": env_bool("TPU_QWEN4_HC_PREFILL_PARTIAL_TILE"),
    # qwen4_exp prefill fused HC mixes (hc_full, the fused mix): the per-branch
    # RMS norm's sum of squares and loop step. vpu (default, today); rows64
    # (64-row steps, bitwise); gram / gram64 (sum of squares as the diagonal of
    # x x^T on the MXU; not bitwise, only the f32 summation order changes);
    # bf16ep / bf16ep64 (the epilogue n * w, n + . in bf16 arithmetic; bitwise).
    "TPU_QWEN4_HC_PREFILL_NORM": env_with_choices(
        "TPU_QWEN4_HC_PREFILL_NORM",
        "vpu",
        ["vpu", "rows64", "gram", "gram64", "bf16ep", "bf16ep64"],
        case_sensitive=False,
    ),
    # The fused prefill HC mix with the norm weight folded into the down/inject
    # rows (hc_prefill_fused._fold_kernel, P-DENSE D4). Not bitwise (float64
    # rule); needs TPU_QWEN4_HC_PREFILL_FUSED_MIX. Default off.
    "TPU_QWEN4_HC_PREFILL_FOLD": env_bool("TPU_QWEN4_HC_PREFILL_FOLD"),
    # Each decoder layer hands its mlp HC combine to the next layer of the
    # stage, which runs it as the prologue of its attn mix (combine_mix; the
    # fused prologue kernel where the tile gate allows). Bitwise where the
    # prologue is; never into the PLE layer or across a stage boundary; off
    # under sequence parallelism (P-DENSE). Default off.
    "TPU_QWEN4_HC_PREFILL_DEFER_COMBINE": env_bool("TPU_QWEN4_HC_PREFILL_DEFER_COMBINE"),
    # The deferral only on whole-tile prefill steps (rows a multiple of the HC
    # prefill tile), with the deferred prologue reading its combined rows back
    # (hc_prefill_mix(reread=True)); other steps keep the direct combine + mix.
    # Bitwise on TPU at 6,400 rows (P-DENSE mb5c j-095837ed). Default off.
    "TPU_QWEN4_HC_PREFILL_DEFER_WHOLE": env_bool("TPU_QWEN4_HC_PREFILL_DEFER_WHOLE"),
    # Pooled GDN: skip zero-filling the [T, n_v, d_v] output before the
    # decode (BATCHED) and prefill (PER_SEQ) kernels write it. Rows no request
    # owns (bucket padding) are then left unwritten; the decoder layer masks
    # them before the MoE.
    "VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT": env_bool("VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT"),
    # qwen4_exp: shard prefill-sized steps across TP ranks between attention
    # blocks (reduce-scatter instead of all-reduce), so the hyper-connection
    # streams, PLE gate and residual math run on 1/TP of the tokens. Steps
    # below QWEN4_EXP_PREFILL_SP_MIN_TOKENS keep the replicated TP path.
    "QWEN4_EXP_PREFILL_SP": env_bool("QWEN4_EXP_PREFILL_SP"),
    # A TP>1 Qwen4Exp decode behind TP1 pipeline stages lands each load in a
    # staging carrier of the producer's page geometry, then repacks it
    # (models/vllm/qwen4_exp/pd_staging.py). Bytes of that carrier per rank;
    # a load that does not fit recomputes on the decode.
    "QWEN4_EXP_PD_STAGING_BYTES": env_int("QWEN4_EXP_PD_STAGING_BYTES", 8 << 30),
    # Engine steps a staged decode holds a load's staging pages after its
    # repack is enqueued, before another load may reuse them. The repack (and
    # S4's landing gather) run asynchronously on the device, while the next
    # load's bytes land through tpu_sync's raw host-to-device copies, which do
    # not wait for queued device reads: freeing the pages at once lets a new
    # load overwrite them before the old repack reads them (one request then
    # repacks another's KV). Pages released in step N return to the free map
    # at the start of step N + steps; 3 covers a repack enqueued after step
    # N's forward with step results read one step late
    # (VLLM_TPU_DEFER_PREV_RESULTS_WAIT). 0 frees at once (the old behaviour).
    "QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS": env_nonnegative_int(
        "QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS", 3
    ),
    # Bounded host staging on a Qwen4Exp PD producer: the native manager keeps
    # a host arena of N transfers' source pages per storage instead of a full
    # host mirror of every pool (about 8 x QWEN4_EXP_MAX_POOL_BYTES per prefill
    # host). Transfers lease arena slots and wait (bounded by tpu_sync) when it
    # is full. 0 (the default) keeps the full mirror. Deliberately not
    # TPU_RAIDEN_POOL_STAGING_LEASES, whose default of 8 would turn this on.
    "QWEN4_EXP_PD_STAGING_LEASES": env_nonnegative_int("QWEN4_EXP_PD_STAGING_LEASES", 0),
    # BOTH legs of a TP1 pipeline producer -> staged TP>1 Qwen4Exp decode
    # (q38 F2): the producer's last stage runs the MTP draft over its prefill
    # chunks (prefill-only MTP, num_speculative_tokens=1) and ships the draft
    # layer's caches (main KV, compressed keys, raw ring) like any QSA layer's;
    # the decode lands them in its draft's owners, so the draft no longer
    # attends unwritten prompt KV. A main-KV DCP decode also needs
    # VLLM_TPU_QSA_MTP_KV_DCP=1 (the draft is then one more DCP-split owner).
    # Part of the PD identity, so a one-sided flag fails the handshake.
    # Host-only (admission and the staging plan): in the compile-key ignore set.
    "QWEN4_EXP_PD_DRAFT_KV": env_bool("QWEN4_EXP_PD_DRAFT_KV"),
    # With QWEN4_EXP_PD_DRAFT_KV on a staged decode (q38 F2-3's forced-gap
    # arm): also zero-fill the MTP draft's rows of each load's first
    # <fraction> of tokens (down to a compressed pair), through the path that
    # zero-fills the rows the producer never computed (above the decode's
    # local hit, below the producer's). In [0, 1); 0 is off. Host-only.
    "QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP": env_float("QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP", 0.0),
    # BOTH legs, with QWEN4_EXP_PD_DRAFT_KV (q38 F2-HOLD): the producer holds
    # MTP draft blocks only above each request's prefix hit Hp. Its draft KV is
    # never prefix-cached (F2 option (a)), so it never writes draft rows below
    # Hp and the decode zero-fills them; the producer's draft main-KV and
    # compressed-key tables open with null blocks there instead of fresh pool
    # IDs (core/producer_draft_hold), and its send registers only the live
    # pages, the null positions' spans reading a live page's rows (the decode
    # zero-fills those rows anyway). The decode's staging layout is unchanged.
    # Part of the PD identity. Host-only: in the compile-key ignore set.
    "QWEN4_EXP_PD_DRAFT_HOLD": env_bool("QWEN4_EXP_PD_DRAFT_HOLD"),
    # "<decode tp>,<decode dcp>,<decode block>" on BOTH legs of a TP1 pipeline
    # producer -> staged TP>1 Qwen4Exp decode: the producer routes each decode
    # rank only its slice (main-KV DCP runs, its GDN heads; compressed keys,
    # ring and PLE to all) into a dense compact staging space per rank, and
    # the decode stages in 256-row pages (models/vllm/qwen4_exp/pd_route.py).
    # Empty (the default) broadcasts every page to every rank. Changes the
    # decode's staging carrier and repack program, so it joins the compile key.
    "QWEN4_EXP_PD_ROUTED_DST": env_str("QWEN4_EXP_PD_ROUTED_DST", ""),
    # A staged Qwen4Exp decode runs each load's repack in the step after the
    # scheduler sees the load complete on EVERY rank, on all ranks together,
    # instead of on each rank in the step its own bytes land. Ranks land a
    # load up to ~0.7 s apart, and every rank's repack holds the whole TP
    # group at the next collective, so one load stalled 2-3.5 decode steps.
    # Host-side scheduling only: the repack program and its inputs are
    # unchanged.
    "QWEN4_EXP_PD_SYNC_REPACK": env_bool("QWEN4_EXP_PD_SYNC_REPACK"),
    # > 0: alternate QWEN4_EXP_PD_SYNC_REPACK by wall clock, off in even and
    # on in odd periods of this many seconds, so one serving job measures
    # both arms on the same engines (an in-job A/B). Overrides the switch.
    "QWEN4_EXP_PD_SYNC_REPACK_AB_S": env_int("QWEN4_EXP_PD_SYNC_REPACK_AB_S", 0),
    # q38 XFER-4: a staged decode's per-load landing timeline (pd_landing_log):
    # scheduler stamps (landed on all ranks / named / scheduled), TP rank 0's
    # repack host time per load, and every N steps the split of HOST_PHASES'
    # kv_connector phase. Log-only (host clocks, no device sync); 0 = off.
    "QWEN4_EXP_PD_LANDING_LOG": env_nonnegative_int("QWEN4_EXP_PD_LANDING_LOG", 0),
    # q38 XFER-4 F1-F3: a staged decode lands each load in fewer repack
    # programs. N > 0: GDN owners go in programs of N owners each, main /
    # copy rows in bucketed chunks (pd_staging.repack_programs_v2), each
    # program's indices in one host-to-device copy, and the plan is made when
    # the load lands on the rank. Same rows written; 0 (default) = today's
    # repack_chunks programs.
    "QWEN4_EXP_PD_REPACK_V2": env_nonnegative_int("QWEN4_EXP_PD_REPACK_V2", 0),
    # q38 L2 D74 (the "L2" rung): land each routed load with ONE descriptor-driven Pallas program (run DMAs, staging
    # to pool, head select in VMEM) instead of v2's 2..39 per-row programs. Takes precedence over
    # QWEN4_EXP_PD_REPACK_V2, which stays the per-load fallback. Same bytes on every non-null page. Adds its own
    # program (compiled at startup by warm_repack_l2); changes none of the model's, so compile-ignored.
    "QWEN4_EXP_PD_REPACK_L2": env_bool("QWEN4_EXP_PD_REPACK_L2"),
    # q38 XFER-4 F3, with QWEN4_EXP_PD_REPACK_V2 only: plan each repack when the
    # load lands on the rank instead of in the named step. Its own switch,
    # because that moves numpy work onto each rank's own (unsynchronized)
    # landing step; host-only, same programs.
    "QWEN4_EXP_PD_REPACK_PLAN_AT_LAND": env_bool("QWEN4_EXP_PD_REPACK_PLAN_AT_LAND"),
    # Both legs (q38 XFER S4): a staged decode's compressed-key pairs cross
    # once per decode host instead of once per rank. Page b of a tag's pair
    # space goes to rank b % TP; the ranks all-gather them in the L1-named
    # step (models/vllm/qwen4_exp/pd_shared_compressed.py; needs
    # QWEN4_EXP_PD_SYNC_REPACK=1). Ported from the Qwen3.5 head-unique
    # transfer (3eeddf339).
    "QWEN4_EXP_PD_SHARED_COMPRESSED": env_bool("QWEN4_EXP_PD_SHARED_COMPRESSED"),
    # Stage-3 destination admission in ONE TP collective per connector pass
    # (distributed/kv_transfer/stage3_admission.py) instead of a free-map
    # all-reduce per load plus an expiry all-reduce per load that does not fit:
    # the same admission and expiry decisions, in the same order. Host-side
    # only (q38 D-MK; FR1b c48 spent ~14 ms per step in 16 such all-reduces).
    "QWEN4_EXP_PD_BATCHED_ADMISSION": env_bool("QWEN4_EXP_PD_BATCHED_ADMISSION"),
    # > 0: alternate the batched admission off / on every this many connector
    # passes, off first; every rank counts the same passes, so all flip
    # together (an in-job A/B on the same engines). Overrides the switch.
    "QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES": env_int(
        "QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES", 0
    ),
    # With the batched admission on: on the first N passes that admit a load,
    # also run today's per-load decisions on copies of the state and log the
    # mismatch count (an inline exactness check; costs today's collectives).
    "QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK": env_int(
        "QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK", 0
    ),
    # A Qwen4Exp pipeline engine with prefix caching gives the scheduler this
    # many block IDs per physical pool page. Every stage shares the block-ID
    # space but writes only its own groups' pages (2-4 of a block's 24), so
    # each stage maps a block ID to a page of its pool only while one of its
    # own groups holds it (models/vllm/qwen4_exp/stage_pages.py). 1 is off.
    "QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR": env_int(
        "QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR", 1
    ),
    # KV cache group size for uniform-page hybrid models (patched
    # kv_cache_utils._get_kv_cache_groups_uniform_page_size): "gcd" of the
    # buckets with more than one layer, small buckets padded (a lone PLE
    # layer no longer forces one group per layer); "min" is upstream vLLM.
    "TPU_KV_CACHE_GROUP_SIZE_RULE": env_with_choices(
        "TPU_KV_CACHE_GROUP_SIZE_RULE", "gcd", ["gcd", "min"]
    ),
    "QWEN4_EXP_PREFILL_SP_MIN_TOKENS": env_int(
        "QWEN4_EXP_PREFILL_SP_MIN_TOKENS", 1024
    ),
    # With QWEN4_EXP_PREFILL_SP=1: the adaptive fused EP MoE takes each
    # rank's own rows on sequence-parallel steps (no replica split/restore)
    # and the shared expert's TP sum is reduce-scattered, not all-reduced.
    "QWEN4_EXP_PREFILL_SP_MOE": env_bool("QWEN4_EXP_PREFILL_SP_MOE"),
    # With QWEN4_EXP_PREFILL_SP_MOE=1: the shared expert of sequence-parallel steps runs on
    # this rank's own rows with a replicated copy of its weights (loaded next to the TP slices,
    # which decode keeps), so it needs no all-gather or reduce-scatter. Costs one full shared
    # expert of HBM per rank.
    "QWEN4_EXP_PREFILL_SP_SHARED_LOCAL": env_bool("QWEN4_EXP_PREFILL_SP_SHARED_LOCAL"),
    # With QWEN4_EXP_PREFILL_SP=1: the sequence-parallel graph pads each
    # rank's shard to the HC kernels' row tile, so buckets whose shard does
    # not divide it (1024 and 3072 at TP8) take sequence parallelism too.
    "QWEN4_EXP_PREFILL_SP_PAD_HC": env_bool("QWEN4_EXP_PREFILL_SP_PAD_HC"),
    # q38 HASH-PATH R2: pairs two QSA layers' main KV in one pool page. The reader stays
    # models/vllm/qwen4_exp/pool_layout.py pack_main_kv_enabled(); the whole-step megakernel body
    # that consumes the paired layout (megakernel/wholestep/union_unpack.py,
    # megakernel_runtime/qwen38_wholestep_call.py) is outside the hashed sources, so the value
    # keys the AOT and piecewise caches here. Same parse and default as the reader, and it never
    # raises: enable_envs_cache() calls every getter, and the factor loop drops a getter that raises.
    "QWEN4_EXP_PACK_MAIN_KV": lambda: (
        os.environ.get("QWEN4_EXP_PACK_MAIN_KV", "0").strip().lower()
        in ("1", "true", "yes", "on")
    ),
    # q38 HASH-PATH R2 (lead follow-on): packs several QSA layers' compressed keys in one pool
    # page. The only env reader is models/vllm/qwen4_exp/pool_layout.py pack_compressed_k_enabled()
    # (called by pooled_cache_specs and by the PP check in qwen4_exp/config.py). pool_layout.py is
    # a hashed source but the value it reads is not, so the value keys the AOT and piecewise
    # caches here. Same parse and default as the reader, and it never raises (as above).
    "QWEN4_EXP_PACK_COMPRESSED_K": lambda: (
        os.environ.get("QWEN4_EXP_PACK_COMPRESSED_K", "0").strip().lower()
        in ("1", "true", "yes", "on")
    ),
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
    # Serve eligible MoE layers (TP experts, or EP with TP attention) with the
    # fused MoE decode kernel: routed experts, shared expert and all-reduce in
    # one Pallas call (layers/adapter/fused_moe_decode.py). Armed once after
    # load. Steps above FUSED_MOE_DECODE_KERNEL_MAX_TOKENS tokens run the GMM
    # MoE path; arming refuses a cap whose buffers do not fit in VMEM.
    "FUSED_MOE_DECODE_KERNEL": env_bool("FUSED_MOE_DECODE_KERNEL"),
    "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS": env_int(
        "FUSED_MOE_DECODE_KERNEL_MAX_TOKENS", 256
    ),
    # Run the fused MoE decode kernel's routing prologue (row padding, route
    # weights to per-expert coefficients, active experts, shared-gate
    # broadcast) inside the kernel instead of as ~8 XLA ops per layer (fold=True
    # in fused_moe_decode_kernel.py). Bitwise the same output. "xla" keeps the
    # shared gate's dot + sigmoid in XLA; "kernel" also moves it into the
    # kernel. "off": the unfolded call.
    "FUSED_MOE_DECODE_KERNEL_FOLD": env_with_choices(
        "FUSED_MOE_DECODE_KERNEL_FOLD", "off", ["off", "xla", "kernel"]
    ),
    # q38 lever 5b (DEC-MOEKV; kernel by DEC-FB-2): routed-row compaction in the
    # fused MoE decode kernel (compact_rows=R in fused_moe_decode_kernel.py).
    # 0 (default): the dense expert loop, every active expert on all rows.
    # R > 0, a multiple of 16: token buckets of at least
    # FUSED_MOE_DECODE_COMPACT_MIN_TOKENS run each local expert on its own
    # routed rows only (up to R; an expert with more takes the dense body).
    # Bitwise the dense kernel's output. Folded calls only
    # (FUSED_MOE_DECODE_KERNEL_FOLD != off) in whole-expert mode; other
    # buckets keep the dense loop.
    "FUSED_MOE_DECODE_COMPACT_ROWS": env_nonnegative_int(
        "FUSED_MOE_DECODE_COMPACT_ROWS", 0
    ),
    "FUSED_MOE_DECODE_COMPACT_MIN_TOKENS": env_int(
        "FUSED_MOE_DECODE_COMPACT_MIN_TOKENS", 256
    ),
    # q38 L3 D60 (lever 5c, "pipelined loop"): compacted experts scheduled for
    # the weight stream (compact_pipe in fused_moe_decode_kernel.py: weight ring
    # primed before the compaction prologue, slot-major tables, software-
    # pipelined expert loop). 0 (default): off, today's programs. N > 0: folded
    # whole-expert token buckets of at least N rows run it with
    # FUSED_MOE_DECODE_COMPACT_PIPE_ROWS rows per expert (a positive multiple
    # of 16; an expert with more takes the dense body), taking precedence over
    # FUSED_MOE_DECODE_COMPACT_ROWS there; smaller buckets are unchanged.
    # Bitwise the dense loop. Changes the program, so both are compile-key
    # flags (and part of the op's closure key when on).
    "FUSED_MOE_DECODE_COMPACT_PIPE": env_nonnegative_int(
        "FUSED_MOE_DECODE_COMPACT_PIPE", 0
    ),
    "FUSED_MOE_DECODE_COMPACT_PIPE_ROWS": env_nonnegative_int(
        "FUSED_MOE_DECODE_COMPACT_PIPE_ROWS", 32
    ),
    # q38 IQD #10 measurement (DEC-MOEKV): named phase scopes inside the fused
    # MoE decode kernel (prologue, shared expert, routed experts, all-reduce
    # barrier wait / all-to-all / all-gather), visible on the "XLA TraceMe"
    # line with LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true.
    # 0 (default): no scopes, the same program. Changes the program when on,
    # so it is part of the compile key. For annotated traces only.
    "FUSED_MOE_DECODE_KERNEL_SCOPES": env_nonnegative_int(
        "FUSED_MOE_DECODE_KERNEL_SCOPES", 0
    ),
    # q38 PORT-BL MOE (ported from Benliang Wang <benliangw@google.com>, b0c1453ba on
    # benliangw/q38fh-integ-pin-5b44702aea; exact width only, not AR2D / routed /
    # padding mask): the target model's fused MoE decode kernel calls exchange and
    # write [m, H] with uneven per-rank lane tiles (Qwen3.8-Flash: 3, 3, 3, 3, 2,
    # 2, 2, 2 tiles of 128) instead of [m, H padded to 8 lanes of 128] followed by
    # an XLA slice. Bitwise the same values. The draft model keeps the padded
    # split. 0 (default): today's programs. Changes the traced program, so it is
    # a compile-key flag (and part of the op's closure key when on).
    "FUSED_MOE_DECODE_EXACT_WIDTH": env_bool("FUSED_MOE_DECODE_EXACT_WIDTH"),
    # Build the MoE permutation, its inverse, and the sorted token ids from
    # one packed sort instead of three. The ragged (counting sort) path ignores it.
    "TPU_MOE_SINGLE_SORT": env_bool("TPU_MOE_SINGLE_SORT", True),
    # Token movement of the whole-expert (non-EP) grouped MoE path, as the
    # PP8xTP1 prefill stages run it (fused_moe_gmm.prepare_routed_gmm_inputs).
    # Gather of the routed rows before the expert GMM:
    #   "xla"          hidden_states[token_indices_sorted]; XLA picks the lowering.
    #                  A [3200, 2560] bf16 source table sits in VMEM and gathers
    #                  at ~4 ns per row; a [6400, 2560] one stays in HBM and
    #                  gathers at ~36 ns per row (2.31 ms per call, aa1 xprof).
    #   "xla_colchunk" the same gather per block of TPU_MOE_NONEP_GATHER_COLS
    #                  columns, concatenated, so each block's source table is
    #                  small enough for VMEM. Bitwise the same rows.
    #   "xla_chunked"  the same gather in pieces of TPU_MOE_NONEP_GATHER_CHUNK
    #                  rows (the source table is unchanged; a control).
    #   "sc"           the SparseCore ragged_gather_v2 kernel.
    #   "sc_padded"    the same, keeping the kernel's padded row blocks (no
    #                  trimming copy; the extra rows are never read).
    #   "sc_fp8"       sc_padded on rows quantized once to fp8 by gmm_v2's own
    #                  lhs quantizer and packed with their scales (3,072 bytes
    #                  per routed row instead of 5,120), gathered DMA-only;
    #                  the GMM reads them pre-quantized. Same output bits.
    #                  Falls back to sc_padded where it does not apply.
    "TPU_MOE_NONEP_GATHER": env_with_choices(
        "TPU_MOE_NONEP_GATHER",
        "xla",
        ["xla", "xla_colchunk", "xla_chunked", "sc", "sc_padded", "sc_fp8"],
        case_sensitive=False,
    ),
    # EP path's SparseCore dispatch gather (above the one-hot permute
    # threshold): "sc" (served: trims the padded row blocks), "sc_padded"
    # (keeps them; the same rows are read), "sc_fp8" (sc_padded on packed fp8
    # rows, as TPU_MOE_NONEP_GATHER=sc_fp8; falls back to sc_padded where it
    # does not apply). Same output bits.
    "TPU_MOE_EP_GATHER": env_with_choices(
        "TPU_MOE_EP_GATHER", "sc", ["sc", "sc_padded", "sc_fp8"], case_sensitive=False
    ),
    # The non-EP GMM knobs (TPU_MOE_NONEP_GMM_EXPERTS_PER_VISIT, _GMM2_..., _GMM_BUCKET,
    # _FUSED_MLP) also on the EP path. Same output bits.
    "TPU_MOE_EP_GMM_STACK": env_bool("TPU_MOE_EP_GMM_STACK"),
    # Non-EP dispatch in this many expert-range chunks (1 = off; 2 or 4): each
    # chunk's SparseCore gather overlaps the previous chunk's fused expert MLP;
    # outputs land in one buffer. Needs TPU_MOE_NONEP_GATHER=sc_padded|sc_fp8
    # and TPU_MOE_NONEP_FUSED_MLP=1. Same output bits.
    "TPU_MOE_NONEP_DISPATCH_CHUNKS": env_with_choices(
        "TPU_MOE_NONEP_DISPATCH_CHUNKS", "1", ["1", "2", "4"], case_sensitive=False
    ),
    # EP combine: "served" (valid-mask SparseCore combine) or "dense" (all-valid
    # routes, non-local ones on a zeroed row with weight 0: the dense run-table
    # path, packed with TPU_MOE_NONEP_COMBINE_PACKED). Same output bits.
    "TPU_MOE_EP_COMBINE": env_with_choices(
        "TPU_MOE_EP_COMBINE", "served", ["served", "dense"], case_sensitive=False
    ),
    # EP path D (TPU_MOE_EP_ARGSORT_PERMUTE): build the routing permutation from
    # two packed single-operand sorts (the inverse without a scatter) and the
    # group sizes from the sorted keys, instead of argsort + inverse argsort +
    # a packed token sort + a one-hot reduce. Same integer arrays, same output
    # bits. Falls back where the packing does not fit int32 (> 65,536 routes).
    # "onehot" keeps the served one-hot group-size reduce (it does not wait for
    # the sort) and changes only the sorts.
    "TPU_MOE_EP_PACKED_SORTS": env_with_choices(
        "TPU_MOE_EP_PACKED_SORTS", "off", ["off", "on", "onehot"], case_sensitive=False
    ),
    # SparseCore combine (ragged_gather_reduce_v3): skip the per-block stable
    # partition sort when every block's valid routes already form a prefix;
    # statically on the EP all-valid combine (TPU_MOE_EP_COMBINE=dense), by a
    # runtime check on the non-EP combine. Same metadata, same output bits.
    "TPU_MOE_COMBINE_PRESORTED": env_bool("TPU_MOE_COMBINE_PRESORTED"),
    # Columns per block for "xla_colchunk". 0 picks the widest block (a
    # multiple of 128 dividing the hidden size) whose source table is at most
    # 3200 x 2560 bf16 bytes, the largest size measured in VMEM; a step whose
    # whole table fits keeps the single unchunked gather.
    "TPU_MOE_NONEP_GATHER_COLS": env_nonnegative_int("TPU_MOE_NONEP_GATHER_COLS", 0),
    "TPU_MOE_NONEP_GATHER_CHUNK": env_int("TPU_MOE_NONEP_GATHER_CHUNK", 16384),
    # Permutation metadata of the non-EP path above TPU_MOE_SINGLE_SORT's row
    # cap: "sorts" runs three sorts (argsort, its inverse, the packed token
    # ids); "two_sorts" takes the sorted token ids from the argsort by
    # division; "counting" takes the inverse permutation and the group sizes
    # from the one-hot counting sort (block 1024) and keeps one argsort;
    # "packed" runs two packed single-operand sorts (the inverse without a
    # scatter) and takes the group sizes from the sorted keys; "packed_onehot"
    # keeps the one-hot group sizes. All of them give the same permutation.
    "TPU_MOE_NONEP_PERMUTE": env_with_choices(
        "TPU_MOE_NONEP_PERMUTE",
        "sorts",
        ["sorts", "two_sorts", "counting", "packed", "packed_onehot"],
        case_sensitive=False,
    ),
    # Blocks of output columns for the non-EP second GMM and its SparseCore
    # combine: with N > 1 each block's combine (an async SparseCore call) can
    # run while the TensorCore computes the next block's GMM, instead of the
    # TensorCore waiting for one whole combine. 1 keeps one GMM and one
    # combine. Bitwise the same output.
    # (4 blocks was measured NOT bitwise on TPU in q38-pmoe-mb1, j-49e001c7, so
    # it is not offered.)
    "TPU_MOE_NONEP_COMBINE_SPLIT": env_with_choices(
        "TPU_MOE_NONEP_COMBINE_SPLIT", "1", ["1", "2"]
    ),
    # Rows per m-tile of the non-EP GMMs (gmm_v2 tile_m); "0" keeps gmm_v2's
    # default (512 at prefill sizes). Each tile boundary inside an expert's
    # rows streams that expert's weights again, so larger tiles stream fewer
    # weight bytes. Bitwise the same output.
    "TPU_MOE_NONEP_GMM_TILE_M": env_with_choices(
        "TPU_MOE_NONEP_GMM_TILE_M", "0", ["0", "256", "512", "1024", "2048"]
    ),
    # The non-EP GMMs quantize each lhs k-block once per tile before the
    # column loop (gmm_v2 hoist_lhs_quant) instead of once per mxu-wide column
    # chunk. Same ops on the same values.
    "TPU_MOE_NONEP_GMM_HOIST_QUANT": env_bool("TPU_MOE_NONEP_GMM_HOIST_QUANT"),
    # The non-EP second GMM receives each expert's weight tile as this many
    # column parts, each its own pipeline input (a concurrent DMA) (gmm_v2
    # rhs_split). Its visits are paced by the weight DMA at prefill sizes.
    # Bitwise the same output.
    "TPU_MOE_NONEP_GMM2_RHS_SPLIT": env_with_choices(
        "TPU_MOE_NONEP_GMM2_RHS_SPLIT", "1", ["1", "2", "5"]
    ),
    # Both non-EP GMMs visit this many consecutive experts per pipeline step
    # (gmm_v2 experts_per_visit): one E-expert weight block and one pass over
    # their rows, amortizing the fixed per-step cost. Bitwise the same output.
    "TPU_MOE_NONEP_GMM_EXPERTS_PER_VISIT": env_with_choices(
        "TPU_MOE_NONEP_GMM_EXPERTS_PER_VISIT", "1", ["1", "2", "4"]
    ),
    # The non-EP second GMM's own experts per pipeline step; "0" = the first
    # GMM's (TPU_MOE_NONEP_GMM_EXPERTS_PER_VISIT). gmm2's per-expert weight
    # block is half gmm1's and streams slower. Bitwise the same output.
    "TPU_MOE_NONEP_GMM2_EXPERTS_PER_VISIT": env_with_choices(
        "TPU_MOE_NONEP_GMM2_EXPERTS_PER_VISIT", "0", ["0", "1", "2", "4"]
    ),
    # Both non-EP expert GMMs as one fused expert-MLP kernel (gmm_v2
    # gmm_mlp_v2): one weight step of w1 + w2 per expert visit, gmm1's output
    # kept in VMEM. Needs gmm2's experts per visit equal to gmm1's and no
    # combine split; falls back to two kernels elsewhere. Bitwise the same output.
    "TPU_MOE_NONEP_FUSED_MLP": env_bool("TPU_MOE_NONEP_FUSED_MLP"),
    # SparseCore combine accumulation for all-valid blocks: "reg" (run table,
    # register sums), "addupdate" (run table, contiguous memory adds) or unset
    # (indexed scatter-add). Read at trace time by ragged_gather_reduce_v3;
    # registered here so the compile-cache key sees it. Bitwise the same output.
    "TPU_SC_COMBINE_DENSE": lambda: os.getenv("TPU_SC_COMBINE_DENSE", "").strip(),
    # The non-EP second GMM stores each row's bf16 column pairs (j, j + N/2)
    # as whole 32-bit words (gmm_v2 pack_out), and the SparseCore combine
    # gathers single rows instead of row pairs: half the gather bytes.
    # Bitwise the same output.
    "TPU_MOE_NONEP_COMBINE_PACKED": env_bool("TPU_MOE_NONEP_COMBINE_PACKED"),
    # Row-bucket base of both non-EP GMMs (gmm_v2 tile bucket_base): a visit
    # computes its rows rounded up to a bucket; 0 = default (128 at prefill).
    # Rows are independent, so the output is bitwise the same.
    "TPU_MOE_NONEP_GMM_BUCKET": env_with_choices(
        "TPU_MOE_NONEP_GMM_BUCKET", "0", ["0", "32", "64"]
    ),
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
    # SparseCore MoE gather-reduce (combine) kernel version used by
    # fused_moe_gmm. "v2" is the default; "v1" selects the legacy kernel and
    # "v3" selects the destination-major prototype.
    "RAGGED_GATHER_REDUCE_VERSION": env_with_choices(
        "RAGGED_GATHER_REDUCE_VERSION", "v2", ["v1", "v2", "v3"]
    ),
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
    "TPU_MOE_ROUTER_TOPK": env_with_choices(
        "TPU_MOE_ROUTER_TOPK", "rowmax", ["sort", "rowmax"]
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
    # Give the adaptive fused MoE per-channel weight scales [E, N] buffers of
    # their own at load time. Squeezing the GMM [E, 1, 1, N] scales in the hot
    # path relayouts both scales of every MoE layer on every forward.
    "MOE_ADAPTIVE_2D_CHANNEL_SCALES": env_bool("MOE_ADAPTIVE_2D_CHANNEL_SCALES"),
    # All-gather the adaptive fused MoE's FP8 activations over a new leading
    # axis instead of tiled on the token axis. Same bytes; XLA then gathers
    # straight into the row-major layout the kernel's token-row DMAs read,
    # instead of gathering in {2,0,1} and relayouting the whole [T, H/128,
    # 128] block (~140 us per layer at 8K tokens on tpu7x).
    "MOE_ADAPTIVE_AG_LEAD_AXIS": env_bool("MOE_ADAPTIVE_AG_LEAD_AXIS"),
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
    # Slice the rotary cos_sin caches to max model len at load to
    # minimize xla layout data copy overhead. Text-only.
    "TPU_ROPE_CACHE_TRUNCATE": env_bool("TPU_ROPE_CACHE_TRUNCATE"),
    # Materialize the rotary cos_sin caches with a row-major ({1,0}) device
    # layout at load, May cost more HBM due to padding.
    "TPU_ROPE_CACHE_ROW_MAJOR": env_bool("TPU_ROPE_CACHE_ROW_MAJOR", default=False),
    "TPU_MOE_HASH_TABLE_ROW_MAJOR": env_bool(
        "TPU_MOE_HASH_TABLE_ROW_MAJOR", default=False
    ),
    # Extra token buckets added to the default list of buckets. Values are separated
    # by "," or ":" (or ";"): recipes pass server env as a comma-separated KEY=VALUE
    # list, so a list value has to use ":" there, e.g. 48:80:96:112.
    "TPU_TOKEN_BUCKET_EXTRA": lambda: [
        int(v)
        for v in os.getenv("TPU_TOKEN_BUCKET_EXTRA", "")
        .replace(":", ",")
        .replace(";", ",")
        .split(",")
        if v.strip()
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
    # Steps the engine keeps in flight beyond the pipeline's stage count, for
    # either hand-off (distributed/pp_push.py extra_steps_in_flight). -1 keeps
    # each hand-off's default: 0 for the wave, stages // 2 for the ring.
    "TPU_PP_EXTRA_STEPS_IN_FLIGHT": env_int("TPU_PP_EXTRA_STEPS_IN_FLIGHT", -1),
    # With TPU_PP_HANDOFF=ring: check the first N sampled transfers byte for
    # byte. The sender and the receiver each log an integer digest of the
    # activations of every 64th transfer (distributed/pp_ring.py); equal
    # digests on both ends show the hand-off moved the bytes unchanged.
    # Each check syncs the host with the device once on both ends. 0: off.
    "TPU_PP_RING_VERIFY": env_int("TPU_PP_RING_VERIFY", 0),
    # Test control for TPU_PP_RING_VERIFY: the receiving stage of this rank
    # digests a copy of what it received with one bit flipped, so its link
    # must report mismatches; the forward reads the unchanged tensor. -1: off.
    "TPU_PP_RING_VERIFY_MUTANT": env_int("TPU_PP_RING_VERIFY_MUTANT", -1),
    # Pipeline stages meet at a host barrier (the world group's CPU group)
    # after each has compiled its standalone forwards and before the
    # hand-off warmup runs every bucket through the stages in step. Without
    # it a stage that finishes compiling more than 300 s after its upstream
    # neighbour trips the ring's cross-host descriptor wait (c10d store
    # timeout; q38 b1 lever P j-8b02d3ec, IX l5p1), which uneven layer
    # partitions (VLLM_PP_LAYER_PARTITION) and cold compiles make likely.
    # Host-only, startup-only: in the compile-key ignore set.
    "TPU_PP_WARMUP_BARRIER": env_bool("TPU_PP_WARMUP_BARRIER"),
    # The same host barrier before each prompt length of the pipeline MTP
    # prefill warmup (runner _warmup_mtp_prefill). The last stage
    # cold-compiles its draft inside the first prompt, after posting that
    # step's receive, so the stages ahead of it send the next prompt's
    # hand-off and wait on the ring's cross-host descriptor (c10d store,
    # 300 s) for as long as the compile takes (q38 x1c r j-ab73d6ff: PP3-PP6
    # timed out while PP7 compiled 12 draft programs). With it, that skew
    # waits in the barrier, under the CPU group's own timeout (PyTorch's
    # gloo default, 1800 s). Host-only, startup-only: in the compile-key
    # ignore set.
    "TPU_PP_MTP_WARMUP_BARRIER": env_bool("TPU_PP_MTP_WARMUP_BARRIER"),
    # Pipeline chunk scheduler: pending prefill of requests that have at least
    # this many tokens computed is left out of the spread over the stages, so
    # their chunks are cut by the step's time budget alone (full steps at long
    # context instead of halves). 0: every request spreads (core/pp_chunk_
    # scheduler.py _pending_step_limit).
    "TPU_PP_SPREAD_MAX_PREFIX": env_int("TPU_PP_SPREAD_MAX_PREFIX", 0),
    # q38 HIGHC-WARMFIRST (port of benliangw d523c2ce0): pipeline chunk
    # scheduler: the head of the waiting queue is ordered by uncached prompt
    # tokens, fewest first, less TPU_PP_WARM_FIRST_AGING tokens for every
    # second a request has waited, then by arrival (core/pp_chunk_scheduler.py
    # _order_waiting). Off: first come, first served. Host-only: in the
    # compile-key ignore set.
    "TPU_PP_WARM_FIRST": env_bool("TPU_PP_WARM_FIRST", default=False),
    # Aging rate of TPU_PP_WARM_FIRST in prompt tokens per second waited (the
    # starvation guard). A request with U uncached tokens is passed only by
    # requests that arrive less than U / rate seconds after it. 0: no aging
    # (pure shortest-uncached-first; can starve long prompts). Host-only.
    "TPU_PP_WARM_FIRST_AGING": env_float("TPU_PP_WARM_FIRST_AGING", 10000.0),
    # With TPU_PP_WARM_FIRST: a prompt's tokens that a KV connector's host pool
    # holds count as cached too (a connector method host_tier_hit_tokens). On
    # this tree no connector implements it, so setting it only logs a warning
    # and the local prefix cache alone counts. Off: only the local prefix
    # cache counts. Host-only.
    "TPU_PP_WARM_FIRST_TIER": env_bool("TPU_PP_WARM_FIRST_TIER", default=False),
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
    # Token-axis chunk size for the GLM-5.2 / DSv4 StreamIndex top-k kernel.
    # Cutting the flat token axis into `chunk_tokens`-sized pieces lets each
    # chunk's SparseCore top-k overlap the next chunk's TensorCore scoring.
    # Unset means unchunked: one single pass, no pipelining.
    "TPU_STREAMIDX_CHUNK_TOKENS": env_positive_int_or_none(
        "TPU_STREAMIDX_CHUNK_TOKENS"
    ),
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
    # Scheduler: keep the trailing prefix-cache block that EAGLE/MTP normally
    # drops (controls vLLM's disable_eagle_block_drop). Host-only.
    "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP": env_bool(
        "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP", default=False
    ),
    # Scheduler: pin Mamba "align" prefix-cache checkpoints until their
    # request is freed and free them after its KV
    # (core/mamba_checkpoint_retention.py). Host-only.
    "VLLM_TPU_MAMBA_CHECKPOINT_RETENTION": env_bool(
        "VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", default=True
    ),
    # Scheduler: also checkpoint the latest decode block boundary so the
    # next turn can resume past this turn's output. Only without speculative
    # decoding, where that state is exact. Host-only.
    "VLLM_TPU_MAMBA_DECODE_CHECKPOINT": env_bool(
        "VLLM_TPU_MAMBA_DECODE_CHECKPOINT", default=True
    ),
    # Scheduler: a checkpoint a turn hit on (published by its previous turn)
    # is freed to the evict-first end of the cached blocks instead of the MRU
    # end (core/prefix_cache_policy.py). Host-only; outputs unchanged.
    "VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT": env_bool(
        "VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT", default=False
    ),
    # Scheduler: a freed block whose prefix-cache key is a partial entry (the
    # --prefix-match-unit tail: a state checkpoint or KV page that ends inside
    # its block) joins an evict-first segment of the cached blocks, oldest
    # first, instead of the MRU end (core/prefix_cache_policy.py). Host-only;
    # outputs unchanged.
    "VLLM_TPU_PREFIX_TAIL_EVICT_FIRST": env_bool(
        "VLLM_TPU_PREFIX_TAIL_EVICT_FIRST", default=False
    ),
    # Admit --prefix-match-unit on q38 F2's prefill-only MTP producer (mtp, K = 1;
    # its draft caches are non-prefix-cacheable, so partial-tail hits never touch
    # draft pages). Config validation only (qwen4_exp/prefix_granularity.py).
    "VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER": env_bool(
        "VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER", default=False
    ),
    # Scheduler: before each step, move the cached prefix blocks of the first
    # N waiting requests to the MRU end of the free queue (not pinned), so a
    # queued request's prefix is not evicted while it waits. 0 is off.
    # Host-only; outputs unchanged (core/prefix_cache_policy.py).
    "VLLM_TPU_PREFIX_REFRESH_WAITING": env_int(
        "VLLM_TPU_PREFIX_REFRESH_WAITING", 0
    ),
    # Seconds between refreshes of one waiting request.
    "VLLM_TPU_PREFIX_REFRESH_INTERVAL_S": env_float(
        "VLLM_TPU_PREFIX_REFRESH_INTERVAL_S", 2.0
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
    # Exact segment-cached prompt tokenization in the API server: message
    # bodies between added tokens are cached and reused across turns
    # (segment_tokenize.py). Ineligible tokenizers fall back automatically.
    "VLLM_TPU_SEGMENT_TOKENIZE": env_bool("VLLM_TPU_SEGMENT_TOKENIZE"),
    # Token budget of that cache (~12 B/token: id pointers plus piece text).
    "VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS": env_int(
        "VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS", 33554432
    ),
    # Shadow-check segment tokenization against the full path on every
    # request (logs mismatches, serves the full result).
    "VLLM_TPU_SEGMENT_TOKENIZE_VERIFY": env_bool("VLLM_TPU_SEGMENT_TOKENIZE_VERIFY"),
    # Prefix-affine DP routing in the API server (core/dp_affinity.py): a
    # request goes to the DP engine that served its longest prompt prefix
    # (chained hashes of CHUNK-token chunks, an LRU of MAX_ENTRIES), unless
    # that engine's stock load score exceeds (1 + SLACK) x mean + 1 or, with
    # SLOT_GUARD (off by default: the score also counts requests still waiting
    # for their P->D load, so "full" over-fires), its score is >= max_num_seqs
    # while another engine's is below.
    "VLLM_TPU_DP_AFFINITY": env_bool("VLLM_TPU_DP_AFFINITY"),
    "VLLM_TPU_DP_AFFINITY_CHUNK": env_int("VLLM_TPU_DP_AFFINITY_CHUNK", 1024),
    "VLLM_TPU_DP_AFFINITY_SLACK": env_float("VLLM_TPU_DP_AFFINITY_SLACK", 0.25),
    "VLLM_TPU_DP_AFFINITY_MAX_ENTRIES": env_int("VLLM_TPU_DP_AFFINITY_MAX_ENTRIES", 1000000),
    "VLLM_TPU_DP_AFFINITY_SLOT_GUARD": env_bool("VLLM_TPU_DP_AFFINITY_SLOT_GUARD"),
    # One prefix table shared by all API-server processes of the decode server
    # (vLLM defaults api_server_count to the DP size, and each process has its
    # own DP client, so per-process tables split a conversation's turns across
    # 4 tables on O4). POSIX shared memory, lock-free, epoch-named.
    "VLLM_TPU_DP_AFFINITY_SHARED": env_bool("VLLM_TPU_DP_AFFINITY_SHARED"),
    "VLLM_TPU_DP_AFFINITY_SHARED_SLOTS": env_int("VLLM_TPU_DP_AFFINITY_SHARED_SLOTS", 2097152),
    # Run an MTP draft's loop steps 1..K-1 as one compiled program
    # (spec_decode/mtp_fused_loop.py). Off by default: one program per step.
    # TODO: support the two cases that still take the per-step loop:
    #   - GLM-5.2 / Gemma-4 MTP: IndexShare's per-chunk top-k restore, and a
    #     KV-sharing draft's fixed positions and seq_lens.
    #   - EP-DP lockstep: run_dp_dummy_draft only replays the per-step
    #     collective trace.
    "VLLM_TPU_FUSED_DRAFT_LOOP": env_bool("VLLM_TPU_FUSED_DRAFT_LOOP"),
    # Precompile compute_selected_logits under no_grad (as served) and at the
    # spec-verify logits lengths (num_tokens_paddings up to
    # max_num_reqs * (K + 1)). 0 restores the old warm-up.
    "VLLM_TPU_WARM_SPEC_LOGITS_LENGTHS": env_bool(
        "VLLM_TPU_WARM_SPEC_LOGITS_LENGTHS", default=True
    ),
    # Host PLE (CPU offload): gather into two reused, pinned-if-possible
    # buffers and copy them to the device without staging (ple_storage.py).
    "VLLM_TPU_PLE_HOST_FAST_LOOKUP": env_bool("VLLM_TPU_PLE_HOST_FAST_LOOKUP"),
    # Log the host PLE lookup's D2H / gather / H2D host time every 1000 calls.
    "VLLM_TPU_PLE_HOST_TIMING": env_bool("VLLM_TPU_PLE_HOST_TIMING"),
    # Host PLE: start the ids' copy before the decoder layers that precede the
    # PLE layer and finish the lookup just before it (ple_offload.py).
    "VLLM_TPU_PLE_HOST_OVERLAP": env_bool("VLLM_TPU_PLE_HOST_OVERLAP"),
    # Verify the first N overlapped ids copies against a blocking copy.
    "VLLM_TPU_PLE_FUSED": env_bool("VLLM_TPU_PLE_FUSED"),
    # How the fused conv pass reads earlier rows: slice, roll or mxu (two
    # passes + fix-up); single / single_mxu run the whole layer as one pass
    # (ple_fused.ple_single_after_lookup, offset-load or MXU taps).
    "VLLM_TPU_PLE_FUSED_TAP_MODE": env_with_choices(
        "VLLM_TPU_PLE_FUSED_TAP_MODE",
        "slice",
        ["slice", "roll", "mxu", "single", "single_mxu"],
    ),
    "VLLM_TPU_PLE_FUSED_TILE": env_int("VLLM_TPU_PLE_FUSED_TILE", 128),
    # The fused PLE kernels' RMS-norm arithmetic (ple_fused._norm): sum, mean,
    # recip or newton (mb6 localizes mb1's window error and picks one).
    "VLLM_TPU_PLE_FUSED_NORM": env_with_choices(
        "VLLM_TPU_PLE_FUSED_NORM", "sum", ["sum", "mean", "recip", "newton"]
    ),
    # The fused PLE kernels' rounding: "torch" (bf16 value projection, the conv
    # norm reads bf16(sig * value)) or "xla" (f32 value, the conv norm reads the
    # f32 product; how XLA computes the served-math port on TPU, P-DENSE mb6b).
    "VLLM_TPU_PLE_FUSED_PRECISION": env_with_choices(
        "VLLM_TPU_PLE_FUSED_PRECISION", "torch", ["torch", "xla"]
    ),
    # PLE short conv: read history taps with a one-hot matmul instead of a
    # [T, C] gather per tap (bitwise equal; pooled path only).
    "VLLM_TPU_PLE_CONV_ONEHOT": env_bool("VLLM_TPU_PLE_CONV_ONEHOT"),
    # Probe-only control for the identity probe: the one-hot conv history read
    # takes the next history row (must change outputs). Never set in serving.
    "Q38_PROBE_PLE_ONEHOT_MUTANT": env_bool("Q38_PROBE_PLE_ONEHOT_MUTANT"),
    # q38 L1 D07-b2: PLE speculative conv on megakernel decode steps (uniform
    # 4-token verify windows, at most rows / 4 live requests). "1" reads the
    # history for rows / 4 request rows instead of all max-num-seqs rows; "2"
    # also takes each token's taps and checkpoint window by static slices
    # instead of gathers. Bitwise equal; stock steps unchanged. Hashed.
    "VLLM_TPU_PLE_DECODE_BOUNDED": env_with_choices(
        "VLLM_TPU_PLE_DECODE_BOUNDED", "0", ["0", "1", "2"]
    ),
    # q38 L1-GLUEFIX: with VLLM_TPU_PLE_DECODE_BOUNDED set, the bounded conv only on
    # megakernel buckets of at least this many tokens; smaller buckets trace the
    # default (bounded=0) computation inside the bounded op. 0 = every bucket (the
    # D07-b2 behaviour). Changes the traced program, so it is hashed.
    "VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS": env_nonnegative_int(
        "VLLM_TPU_PLE_DECODE_BOUNDED_MIN_ROWS", 0
    ),
    # q38 L1-PLE (PLE-INTO-D14): with VLLM_TPU_PLE_DECODE_BOUNDED >= 1, a D14
    # bucket of exactly rows * 4 tokens runs the PLE conv and its checkpoint
    # writes as one Pallas program (kernels/qwen4_exp/ple_d14_fold.py) instead
    # of the history read, gathers / slices and the per-token checkpoint writer.
    # Bitwise equal on CPU; inert without the bounded op. Hashed.
    "VLLM_TPU_PLE_D14_FOLD": env_bool("VLLM_TPU_PLE_D14_FOLD"),
    # q38 L1 PLE-WRDEFER: K > 0 keeps K checkpoint DMAs in flight in the PLE pool
    # writer (kernels/qwen4_exp/pool.write_region_ring: one grid step, a ring of K
    # VMEM slots / DMA semaphores) instead of one grid step and one waited DMA per
    # row. Bitwise (unique positive ids; null ids skipped). 0 = stock writer. Hashed.
    "VLLM_TPU_PLE_WRDEFER": env_with_choices(
        "VLLM_TPU_PLE_WRDEFER", "0", ["0", "1", "2", "4", "8"]
    ),
    "VLLM_TPU_PLE_HOST_OVERLAP_CHECK": env_nonnegative_int(
        "VLLM_TPU_PLE_HOST_OVERLAP_CHECK", 0
    ),
    # Log one WARNING per torch.compile compile after warmup (runner/compile_watch.py).
    "VLLM_TPU_LOG_POST_WARMUP_COMPILES": env_bool(
        "VLLM_TPU_LOG_POST_WARMUP_COMPILES", default=False
    ),
    # After warmup, pin each worker's threads near its chip: "numa" (its
    # chip's NUMA node) or "slice" (own physical cores on that node, keeping
    # VLLM_TPU_CPU_PIN_RESERVE cores per node free). See worker/cpu_pinning.py.
    "VLLM_TPU_CPU_PIN": env_str("VLLM_TPU_CPU_PIN", "off"),
    # Log each rank's per-step host time by runner phase every N steps
    # (0 = off); runner/host_phases.py.
    "VLLM_TPU_HOST_PHASES": env_nonnegative_int("VLLM_TPU_HOST_PHASES", 0),
    # q38 L4 D98 / X-H14-AOTSKIP: port of MM's VLLM_TPU_AOT_SKIP_GUARD_CHECK
    # (compilation/aot_guard_skip.py): a freshly AOT-compiled vLLM model skips the
    # per-call guard check when vLLM does not evaluate guards (as vLLM already does
    # for an artifact loaded from the cache). Hashed in the compile key on purpose,
    # so an arm that sets it compiles its own AOT forwards.
    "VLLM_TPU_AOT_SKIP_GUARD_CHECK": env_bool(
        "VLLM_TPU_AOT_SKIP_GUARD_CHECK", default=False
    ),
    # q38 L4 D98 / X-H14-D98R (compilation/fast_compiled_call.py, port of IX
    # QX_FAST_COMPILED_CALL / VLLM_TPU_T3_GUARD_MEMO / QX_TPU_DROP_GLOBAL_STATE_GUARD and
    # MM FD-H option D): eligible torch.compile(backend="tpu", fullgraph=True,
    # dynamic=False) helpers dispatch their guard-checked cache entry's transformed code
    # directly, skipping Dynamo's frame; any miss takes the normal call. Never compiles.
    "VLLM_TPU_FAST_COMPILED_CALL": env_bool("VLLM_TPU_FAST_COMPILED_CALL", default=False),
    # On top of the fast call: "1" memoizes the entry per argument signature (incl.
    # storage_offset) and skips the guards on a memo hit; "check" also runs the guards
    # on every hit and counts mismatches; "" / "0" off.
    "VLLM_TPU_FAST_CALL_GUARD_MEMO": env_str("VLLM_TPU_FAST_CALL_GUARD_MEMO", ""),
    # Compile those helpers without Dynamo's GLOBAL_STATE guard (same graphs).
    "VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE": env_bool(
        "VLLM_TPU_FAST_CALL_DROP_GLOBAL_STATE", default=False
    ),
    # With the fast call on: a per-rank D98FAST engagement line every N runner steps
    # (0 = none).
    "VLLM_TPU_FAST_CALL_LOG_EVERY": env_nonnegative_int(
        "VLLM_TPU_FAST_CALL_LOG_EVERY", 200
    ),
    # Log the engine core's per-step phase times every N engine steps, plus
    # each step slower than 40 ms (0 = off); core/engine_phases.py.
    "VLLM_TPU_ENGINE_PHASES": env_nonnegative_int("VLLM_TPU_ENGINE_PHASES", 0),
    # q38 L1-C48 DECODE-TRIM: row-aware decode step trimming on the engine core's scheduler
    # ('' = off; 'ROWS:MS/...[/margin:F][/al:F][/reentry:F]' per-bucket cost table); core/decode_trim.py.
    "VLLM_TPU_DECODE_TRIM": env_str("VLLM_TPU_DECODE_TRIM", ""),
    # Engine steps per 'q38 DECODE-TRIM' window line (0 = install line only).
    "VLLM_TPU_DECODE_TRIM_LOG": env_nonnegative_int("VLLM_TPU_DECODE_TRIM_LOG", 2000),
    # q38 L1-DSTASH: the TPU runner keeps the [bonus, draft_1..K] source row of a request DECODE_TRIM
    # held back for one step and substitutes it at its re-entry, so it verifies its real drafts
    # (a no-op unless VLLM_TPU_DECODE_TRIM is also set); runner/draft_stash.py.
    "VLLM_TPU_DECODE_TRIM_DRAFT_STASH": env_bool("VLLM_TPU_DECODE_TRIM_DRAFT_STASH"),
    # q38 S3A-ENG T4: one line per rank at its first _update_states, "q38 DSTASH state rank=<r>: enabled=<0|1>
    # draft_stash=<0|1> decode_trim=<set|unset>" (runner/tpu_runner.py); log-only, compile-ignored. Default off.
    "VLLM_TPU_DSTASH_STATE_LOG": env_bool("VLLM_TPU_DSTASH_STATE_LOG"),
    # q38 HOSTTIER-D: log the pod cgroup's memory.stat, tmpfs usage and per-process
    # RSS every N seconds from local DP rank 0's engine core (0 = off); core/memstat.py.
    "VLLM_TPU_MEMSTAT_INTERVAL_S": env_nonnegative_int("VLLM_TPU_MEMSTAT_INTERVAL_S", 0),
    # q38 HOSTTIER-D decode host-DRAM prefix tier (default off; port of IX DSYS A1h).
    # With a TPUMultiConnector of a consumer Qwen4ExpRaidenConnector and a
    # TPURaidenOffloadingConnector, a decode request's load is split: the
    # offloading child reloads the attention pages (main KV, compressed keys,
    # MTP draft KV) it holds in host DRAM for [L, L+H) and Raiden transfers
    # [L+H, R) with the final GDN/PLE state and the raw ring. Host-side only.
    "VLLM_TPU_DECODE_HOST_TIER": env_bool("VLLM_TPU_DECODE_HOST_TIER", default=False),
    # Decode host tier page check (0 = off): digest one sampled page key in N
    # on device when the tier stores it and again when a reload lands; log
    # mismatches and running counts. Debug only; host-side.
    "VLLM_TPU_DECODE_HOST_TIER_VERIFY": env_nonnegative_int(
        "VLLM_TPU_DECODE_HOST_TIER_VERIFY", 0
    ),
    # q38 ADM-P: on a kv_consumer engine, ship new requests' prompt tokens to the
    # workers only from num_computed_tokens - N (0 = whole prompt); core/pd_prompt_tail.py.
    "VLLM_TPU_PD_PROMPT_TAIL": env_nonnegative_int("VLLM_TPU_PD_PROMPT_TAIL", 0),
    # q38 DEC-HOST-B: with VLLM_TPU_HOST_PHASES > 0, log every step whose host
    # time is at least this many ms on its own HOSTSLOW line (0 = off).
    "VLLM_TPU_HOST_PHASES_SLOW_MS": env_float("VLLM_TPU_HOST_PHASES_SLOW_MS", 0.0),
    # q38 L4 host sub-mark instrument: with VLLM_TPU_HOST_PHASES > 0, sum the
    # host_phases sub-phases (~pi_*, ~am_*, ~us_*, ~pf_*, ~st_*, ...) per step and
    # log them every N steps on a separate HOSTSUBS line (runner/host_phases.py).
    "VLLM_TPU_HOST_PHASES_SUBS": env_bool("VLLM_TPU_HOST_PHASES_SUBS", default=False),
    # q38 L6-INSTR (H1 rung 1): with VLLM_TPU_HOST_PHASES > 0 the worker times its
    # cyclic GC pauses (gc.callbacks) and logs them per window on a GCCENSUS line
    # (runner/host_phases.py); 0 turns the census off. Log-only.
    "VLLM_TPU_HOST_PHASES_GC": env_bool("VLLM_TPU_HOST_PHASES_GC", default=True),
    # q38 D1k: KV cache groups of one class (same manager type and allocation-relevant
    # spec) skip their per-request scheduler work when the class's first group did
    # nothing; core/kv_twin_groups.py.
    "VLLM_TPU_KV_TWIN_GROUPS": env_bool("VLLM_TPU_KV_TWIN_GROUPS", default=False),
    # q38 D1e: interpreter knobs for the decode loop (core/host_knobs.py): the GIL
    # switch interval in µs (0 = Python's default) and "freeze" = gc.freeze() after
    # VLLM_TPU_HOST_GC_AFTER steps with gen-0 threshold VLLM_TPU_HOST_GC_GEN0.
    "VLLM_TPU_HOST_SWITCH_US": env_float("VLLM_TPU_HOST_SWITCH_US", 0.0),
    "VLLM_TPU_HOST_GC": env_str("VLLM_TPU_HOST_GC", ""),
    "VLLM_TPU_HOST_GC_AFTER": env_nonnegative_int("VLLM_TPU_HOST_GC_AFTER", 500),
    "VLLM_TPU_HOST_GC_GEN0": env_nonnegative_int("VLLM_TPU_HOST_GC_GEN0", 50000),
    # q38 D1e in-process A/B: knobs on in even wall-clock windows of N ms, interpreter defaults in odd ones
    # (0 = off); VLLM_TPU_HOST_GC may be "threshold" (windowable) but not "freeze" (one-shot).
    "VLLM_TPU_HOST_AB_PERIOD_MS": env_nonnegative_int("VLLM_TPU_HOST_AB_PERIOD_MS", 0),
    # q38 L6-HOST step-time host levers (core/l6_ab.py; read per step from os.environ, never
    # at trace time): the windowed A/B clock and class count, and the levers. A lever value is
    # 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_AB_PERIOD_MS": env_nonnegative_int("VLLM_TPU_L6_AB_PERIOD_MS", 0),
    "VLLM_TPU_L6_AB_CLASSES": env_nonnegative_int("VLLM_TPU_L6_AB_CLASSES", 2),
    # Windows by step count instead (k = j // S; set S = VLLM_TPU_HOST_PHASES to align them
    # with the HOSTPHASES windows); exclusive with the period.
    "VLLM_TPU_L6_AB_STEPS": env_nonnegative_int("VLLM_TPU_L6_AB_STEPS", 0),
    # Lockstep of the step windows across ranks (SAB's fingerprint protocol): fingerprints are
    # written D steps before each window and checked on its last step; the store directory.
    "VLLM_TPU_L6_AB_CONFIRM": env_nonnegative_int("VLLM_TPU_L6_AB_CONFIRM", 16),
    "VLLM_TPU_L6_AB_DIR": env_str("VLLM_TPU_L6_AB_DIR", ""),
    # Bound every L6 lever's CHECK mode to the first N worker steps (0 = unbounded).
    "VLLM_TPU_L6_CHECK_STEPS": env_nonnegative_int("VLLM_TPU_L6_CHECK_STEPS", 0),
    # Lever envcache (core/l6_envcache.py): memoise this module's reads in the worker.
    "VLLM_TPU_L6_ENVS_CACHE": env_str("VLLM_TPU_L6_ENVS_CACHE", ""),
    # Re-read every cached name every N cached steps; drop the cache on a mismatch (0 = off).
    "VLLM_TPU_L6_ENVS_CACHE_CHECK": env_nonnegative_int("VLLM_TPU_L6_ENVS_CACHE_CHECK", 0),
    # Count this module's reads in the worker and log them every N steps (0 = off).
    "VLLM_TPU_L6_ENVS_CENSUS": env_nonnegative_int("VLLM_TPU_L6_ENVS_CENSUS", 0),
    # Lever fctx (runner/fast_forward_context.py, port of the Qwen3.5 lane's cached forward
    # context): reuse one ForwardContext per (config, num_tokens, DP counts) on the step path.
    "VLLM_TPU_L6_FAST_FWD_CTX": env_str("VLLM_TPU_L6_FAST_FWD_CTX", ""),
    # JITTER levers (core/l6_jitter.py): windowable gc.freeze / gc.unfreeze, and
    # sys.setswitchinterval(VLLM_TPU_L6_SWITCH_US µs) while on.
    "VLLM_TPU_L6_GC_FREEZE": env_str("VLLM_TPU_L6_GC_FREEZE", ""),
    "VLLM_TPU_L6_SWITCH": env_str("VLLM_TPU_L6_SWITCH", ""),
    "VLLM_TPU_L6_SWITCH_US": env_nonnegative_int("VLLM_TPU_L6_SWITCH_US", 200),
    # JITTER lever mainpin: pin the worker's main thread to one CPU and move every other
    # thread off it while on (CPU, default -1 = the highest allowed CPU).
    "VLLM_TPU_L6_MAIN_PIN": env_str("VLLM_TPU_L6_MAIN_PIN", ""),
    "VLLM_TPU_L6_MAIN_PIN_CPU": env_int("VLLM_TPU_L6_MAIN_PIN_CPU", -1),
    # Lever mainpin2 (core/l6_jitter.py, MAINPIN-2): each rank's main thread on its own physical core
    # (idle SMT sibling) of the node's reserved cores; every other worker thread excluded from them once
    # after warm-up; only the main thread's mask toggles per window. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_MAIN_PIN2": env_str("VLLM_TPU_L6_MAIN_PIN2", ""),
    # Bind each worker to its chip's NUMA node before the runtime starts (worker/numa_bind.py, Jordan
    # Dotzel 4cfb870d7; same as additional_config tpu_worker.numa_bind, Shrihari Murali 96bfba489).
    "VLLM_TPU_L6_NUMA_BIND": env_bool("VLLM_TPU_L6_NUMA_BIND", default=False),
    # Lever syncgc (core/l6_jitter.py, H3-L3 SYNC-GC): automatic GC off, and every rank collects at the
    # same worker step (every N-th, at the draft propose); a gen-0 count above the valve collects at once.
    "VLLM_TPU_L6_SYNC_GC": env_str("VLLM_TPU_L6_SYNC_GC", ""),
    "VLLM_TPU_L6_SYNC_GC_EVERY": env_nonnegative_int("VLLM_TPU_L6_SYNC_GC_EVERY", 100),
    "VLLM_TPU_L6_SYNC_GC_VALVE": env_nonnegative_int("VLLM_TPU_L6_SYNC_GC_VALVE", 100000),
    # Lever defer (runner/tpu_runner.py l6_defer_ok): on a pure KV consumer, wait for the
    # previous step's sampled tokens after the draft launch instead of before it.
    "VLLM_TPU_L6_DEFER_PREV": env_str("VLLM_TPU_L6_DEFER_PREV", ""),
    # Lever steady (runner/l6_steady.py): steady-step fast path. When the batch, its order,
    # the block tables and every mamba state column are unchanged since the last full walk,
    # reuse that walk's staged tables / mamba products / metadata (only the per-step tensors
    # are refreshed), the reorder + chunk count, and the adapter's block-table views; skip
    # upstream's batch reorder when it is provably a no-op. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_STEADY_STEP": env_str("VLLM_TPU_L6_STEADY_STEP", ""),
    # Every N fast steps also run the full path and compare on the host (block-table
    # bytes, mamba products, metadata fields, reorder results); a mismatch logs an ERROR
    # and turns the fast path off for the process (0 = off).
    "VLLM_TPU_L6_STEADY_STEP_CHECK": env_nonnegative_int("VLLM_TPU_L6_STEADY_STEP_CHECK", 0),
    # Lever carry (runner/decode_carry.py, port of IX's PR14 decode carry): in steady async-spec
    # decode, skip the input_ids / positions / seq_lens uploads and feed the previous step's
    # corrected device tensors (+ 1 + K) to the fused subtract / substitute / extract program;
    # also reuse unchanged mamba-index and request-distribution uploads. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_DECODE_CARRY": env_str("VLLM_TPU_L6_DECODE_CARRY", ""),
    # Every N carried steps also run the upload path and compare on the host (0 = off; syncs).
    "VLLM_TPU_L6_DECODE_CARRY_CHECK": env_nonnegative_int("VLLM_TPU_L6_DECODE_CARRY_CHECK", 0),
    # q38 L6-GATE batch-size gates (core/l6_ab.py GATES): with N > 0, a worker step whose scheduled batch holds fewer
    # than N requests runs CARRY / STEADY off for that step (their lever-off paths; STEADY drops its caches), counted as
    # carry.why_minreqs / steady.minreqs on the L6AB line. 0 = no gate (the default).
    "VLLM_TPU_L6_CARRY_MIN_REQS": env_nonnegative_int("VLLM_TPU_L6_CARRY_MIN_REQS", 0),
    "VLLM_TPU_L6_STEADY_MIN_REQS": env_nonnegative_int("VLLM_TPU_L6_STEADY_MIN_REQS", 0),
    # Lever adcarry (runner/adapt_carry.py, H0-1 ADAPT-CARRY + POSTFWD-MERGE): run the Qwen3.8 adapter's
    # input program inside the post-forward / carry program (both merged programs precompiled at warm-up,
    # never compiled while serving), reuse the previous row map, and on carried steps the previous device
    # n-gram window. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_ADAPT_CARRY": env_str("VLLM_TPU_L6_ADAPT_CARRY", ""),
    # Every N engaged steps also run today's path and compare on the host (0 = off; syncs).
    "VLLM_TPU_L6_ADAPT_CARRY_CHECK": env_nonnegative_int("VLLM_TPU_L6_ADAPT_CARRY_CHECK", 0),
    # Lever engalloc (core/l6_engine_alloc.py): quiet decode calls of KVCacheManager.allocate_slots /
    # cache_blocks skip the per-group walk in the engine core (0 / 1 / ab:<classes>, read per schedule).
    "VLLM_TPU_L6_ENGINE_ALLOC_FAST": env_str("VLLM_TPU_L6_ENGINE_ALLOC_FAST", ""),
    # Run the stock path instead of every N-th fast call and compare; a mismatch turns the fast path off (0 = off).
    "VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK": env_nonnegative_int("VLLM_TPU_L6_ENGINE_ALLOC_FAST_CHECK", 0),
    # Lever setupfree (runner/l6_setupfree.py, q38 T-5(b)): STEADY's fast path rewrites one of two kept
    # AttentionMetadata sets in place (when nothing else holds it) instead of cloning a fresh set
    # that dies at the next sample_setup; same values, host-only. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_SETUP_FREE": env_str("VLLM_TPU_L6_SETUP_FREE", ""),
    # Every N rewritten sets also build STEADY's clone and compare on the host; a mismatch logs an
    # ERROR and turns the lever off (0 = off; bounded by VLLM_TPU_L6_CHECK_STEPS).
    "VLLM_TPU_L6_SETUP_FREE_CHECK": env_nonnegative_int("VLLM_TPU_L6_SETUP_FREE_CHECK", 0),
    # q38 L6-H2 FENCE-SKIP (runner/tpu_runner.py _record_input_staging_fence): record no input
    # staging fence on a chunk when no upload from a runner-held host tensor was issued since the
    # previous fence point (0 / 1 / ab:<classes>; host-only).
    "VLLM_TPU_L6_FENCE_SKIP": env_str("VLLM_TPU_L6_FENCE_SKIP", ""),
    # q38 L6-H2 TOKEN-LISTS-ONCE (port of IX a46d91d20's async-output part, Jordan Dotzel): the
    # sampled-token copy converts its rows to Python lists once; get_output and the main thread's
    # rollback filter those lists instead of masking tensors (0 / 1 / ab:<classes>; host-only).
    "VLLM_TPU_L6_TOKEN_LISTS": env_str("VLLM_TPU_L6_TOKEN_LISTS", ""),
    # Every N-th lists call also runs the tensor path and compares (0 = off).
    "VLLM_TPU_L6_TOKEN_LISTS_CHECK": env_nonnegative_int("VLLM_TPU_L6_TOKEN_LISTS_CHECK", 0),
    # q38 L6-H2 CF-2 AOTSKIP-AB (compilation/l6_aotskip.py): toggle the AOT artifacts' per-call
    # guard check per l6_ab window (0 / 1 / ab:<classes>; no recompile, compile-ignored).
    "VLLM_TPU_L6_AOT_SKIP": env_str("VLLM_TPU_L6_AOT_SKIP", ""),
    # Every N-th lever-on step evaluates the guards as the stock path does (0 = off).
    "VLLM_TPU_L6_AOT_SKIP_CHECK": env_nonnegative_int("VLLM_TPU_L6_AOT_SKIP_CHECK", 0),
    # Lever gcadmit (core/l6_gcadmit.py, GC-ADMIT-FREEZE): after each worker step that admitted requests,
    # gc.collect() then gc.freeze() once (never unfrozen). 0 / 1 / ab:<classes>. Host-only.
    "VLLM_TPU_L6_GC_ADMIT_FREEZE": env_str("VLLM_TPU_L6_GC_ADMIT_FREEZE", ""),
    # Stop freezing once the lever's freezes moved more than this many objects in total (0 = no cap).
    "VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP": env_nonnegative_int("VLLM_TPU_L6_GC_ADMIT_FREEZE_CAP", 1000000),
    # Log a GCADMIT line at the first three freezes and every N-th freeze after them.
    "VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK": env_nonnegative_int("VLLM_TPU_L6_GC_ADMIT_FREEZE_CHECK", 64),
    # q38 L6 gen-2 heap type census (core/l6_gccensus.py, log-only): top-15 types of gc generation 2 after warm-up
    # and after any gen-2 collection >= VLLM_TPU_L6_GC_CENSUS_MS ms; at most VLLM_TPU_L6_GC_CENSUS_MAX per process.
    "VLLM_TPU_L6_GC_CENSUS_TYPES": env_bool("VLLM_TPU_L6_GC_CENSUS_TYPES"),
    "VLLM_TPU_L6_GC_CENSUS_MS": env_nonnegative_int("VLLM_TPU_L6_GC_CENSUS_MS", 1000),
    "VLLM_TPU_L6_GC_CENSUS_MAX": env_nonnegative_int("VLLM_TPU_L6_GC_CENSUS_MAX", 4),
    # L6 end-token canary (core/l6_eos_canary.py, log-only): end-token ids counted per L6 window on rank 0.
    "VLLM_TPU_L6_EOS_CANARY": env_str("VLLM_TPU_L6_EOS_CANARY", ""),
    # Lever ptrims (runner/l6_ptrims.py, q38 H0-2 PREP-TRIMS): Python / numpy trims of the worker's
    # input prep (_prepare_inputs) that produce the same host arrays, net of CARRY and STEADY;
    # host-only, no compiled program changes. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_PREP_TRIMS": env_str("VLLM_TPU_L6_PREP_TRIMS", ""),
    # Every N trimmed steps also run the stock code and compare on the host; a mismatch logs
    # an ERROR and turns the trims off for the process (0 = off; bounded by VLLM_TPU_L6_CHECK_STEPS).
    "VLLM_TPU_L6_PREP_TRIMS_CHECK": env_nonnegative_int("VLLM_TPU_L6_PREP_TRIMS_CHECK", 0),
    # Lever admitinc (layers/core/attention_metadata.py, q38 L6 ADMIT-INC): with BT-INC on, a staged
    # pack whose only geometry change is the request count copies the dirty rows and the rows that
    # entered the window and zeroes the rows that left it, instead of the full pack; and
    # MultiGroupBlockTable's row writes run as one loop over the KV groups with one version bump.
    # Same staged bytes, same block tables, same programs. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_ADMIT_INC": env_str("VLLM_TPU_L6_ADMIT_INC", ""),
    # Every N admit packs (and every N batched row writes) also run the stock path and compare;
    # a mismatch logs an ERROR, keeps the stock result and turns the lever off (0 = off).
    "VLLM_TPU_L6_ADMIT_INC_CHECK": env_nonnegative_int("VLLM_TPU_L6_ADMIT_INC_CHECK", 0),
    # q38 L6-MK levers (runner/l6_mk.py, core/l6_fuse_rpc.py; 0 / 1 / ab:<classes>, read per step on the
    # host, never at trace time). mkctx: the D14 driver's whole-step call trims (cached contexts, one
    # eligibility read, lazy check_real). mkthin: torch_tpu single-partition executables call their
    # fast runner directly. mkdirect: persistent-argument direct call of the MK-type programs.
    # fuserpc: the engine core sends execute_model + sample_tokens as one RPC when no grammar applies.
    # e2kvonly: with a KV connector, ranks other than the output rank reply with the KV output only.
    # Each *_CHECK=N runs every N-th on-call through the stock path as well and compares (0 = off).
    "VLLM_TPU_L6_MK_CTX": env_str("VLLM_TPU_L6_MK_CTX", ""),
    "VLLM_TPU_L6_MK_CTX_CHECK": env_nonnegative_int("VLLM_TPU_L6_MK_CTX_CHECK", 0),
    "VLLM_TPU_L6_MK_THIN_TT": env_str("VLLM_TPU_L6_MK_THIN_TT", ""),
    "VLLM_TPU_L6_MK_THIN_TT_CHECK": env_nonnegative_int("VLLM_TPU_L6_MK_THIN_TT_CHECK", 0),
    "VLLM_TPU_L6_MK_DIRECT": env_str("VLLM_TPU_L6_MK_DIRECT", ""),
    "VLLM_TPU_L6_MK_DIRECT_CHECK": env_nonnegative_int("VLLM_TPU_L6_MK_DIRECT_CHECK", 0),
    "VLLM_TPU_L6_FUSE_RPC": env_str("VLLM_TPU_L6_FUSE_RPC", ""),
    "VLLM_TPU_L6_OUT_KVONLY": env_str("VLLM_TPU_L6_OUT_KVONLY", ""),
    "VLLM_TPU_L6_OUT_KVONLY_CHECK": env_nonnegative_int("VLLM_TPU_L6_OUT_KVONLY_CHECK", 0),
    # Lever clean (runner/l6_carry_lean.py, q38 H2 CARRY-LEAN): CARRY's host check reduced to one
    # vector test on steps whose STEADY reorder verdict was ro_fast against the recorded epoch, and
    # an in-place record; every other step runs CARRY's stock check. Host-only, no compiled program
    # changes. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_CARRY_LEAN": env_str("VLLM_TPU_L6_CARRY_LEAN", ""),
    # Every N lean accepts also run the stock check and compare; a mismatch logs an ERROR and turns the
    # lean check off for the process (0 = off; bounded by VLLM_TPU_L6_CHECK_STEPS).
    "VLLM_TPU_L6_CARRY_LEAN_CHECK": env_nonnegative_int("VLLM_TPU_L6_CARRY_LEAN_CHECK", 0),
    # Lever pshort (runner/l6_prep_short.py, q38 H2 PREP-SHORT): a carried steady step's
    # _prepare_inputs reuses the recorded batch's plan, spec metadata and device tensors and calls only
    # the carry program; host-only, no compiled program changes. 0 / 1 / ab:<classes>.
    "VLLM_TPU_L6_PREP_SHORT": env_str("VLLM_TPU_L6_PREP_SHORT", ""),
    # Every N short-path steps run the full path instead and compare it with the short path's
    # predictions; a mismatch logs an ERROR and turns the short path off for the process (0 = off;
    # bounded by VLLM_TPU_L6_CHECK_STEPS).
    "VLLM_TPU_L6_PREP_SHORT_CHECK": env_nonnegative_int("VLLM_TPU_L6_PREP_SHORT_CHECK", 0),
    # q38 L6 P-METER (runner/l6_pmeter.py; log-only, read per step through core/l6_ab.py):
    # 0 / 1 / ab:<classes>. PMETER: per-step wall-clock stamps (dequeue, forward enqueue, the
    # previous-step copy wait, dp_loop enqueue) logged per window of PMETER_WINDOW steps on a
    # PMETER line, plus a THREADSTAT line. INJECT: rank INJECT_RANK busy-spins INJECT_US us
    # (GIL held) before the step's first forward; INJECT_ALL: every rank spins INJECT_ALL_US us.
    "VLLM_TPU_L6_PMETER": env_str("VLLM_TPU_L6_PMETER", ""),
    "VLLM_TPU_L6_PMETER_WINDOW": env_nonnegative_int("VLLM_TPU_L6_PMETER_WINDOW", 200),
    "VLLM_TPU_L6_INJECT": env_str("VLLM_TPU_L6_INJECT", ""),
    "VLLM_TPU_L6_INJECT_US": env_nonnegative_int("VLLM_TPU_L6_INJECT_US", 250),
    "VLLM_TPU_L6_INJECT_RANK": env_nonnegative_int("VLLM_TPU_L6_INJECT_RANK", 0),
    "VLLM_TPU_L6_INJECT_ALL": env_str("VLLM_TPU_L6_INJECT_ALL", ""),
    "VLLM_TPU_L6_INJECT_ALL_US": env_nonnegative_int("VLLM_TPU_L6_INJECT_ALL_US", 1500),
    # q38 L6 NODEFER census (runner/l6_nodefer.py; log-only, read per step through core/l6_ab.py):
    # 0 / 1 / ab:<classes>. Every NODEFER_EVERY-th VFY call counts its arguments torch_tpu has not
    # started materializing (is_materializing False) on the L6AB line (nodefer.*).
    "VLLM_TPU_L6_NODEFER_CENSUS": env_str("VLLM_TPU_L6_NODEFER_CENSUS", ""),
    "VLLM_TPU_L6_NODEFER_EVERY": env_nonnegative_int("VLLM_TPU_L6_NODEFER_EVERY", 64),
    # q38 L6-S3ASYNC (distributed/kv_transfer/stage3_async.py; host-only, read per submit pass through
    # core/l6_ab.py lever s3async): 0 / 1 / ab:<classes>. On, a sharded Stage-3 consumer hands each
    # pass's loads to a per-rank coordinator thread that runs the admission collective, the
    # reservation, the staging and the all-rank ready agreement on its own gloo group (never the
    # model-runner thread's groups); every rank applies a job's decisions to its scheduler-visible
    # tables at the same submit pass, LAG passes after the hand-off (blocking only if the job is not
    # done by then; JOIN_TIMEOUT_S bounds that wait). Needs QWEN4_EXP_PD_BATCHED_ADMISSION=1 with its
    # A/B and CHECK off. An S3ASYNC line every STATS_EVERY passes per rank is the engagement marker.
    "VLLM_TPU_S3_ASYNC_SUBMIT": env_str("VLLM_TPU_S3_ASYNC_SUBMIT", ""),
    "VLLM_TPU_S3_ASYNC_LAG": env_nonnegative_int("VLLM_TPU_S3_ASYNC_LAG", 2),
    "VLLM_TPU_S3_ASYNC_JOIN_TIMEOUT_S": env_nonnegative_int("VLLM_TPU_S3_ASYNC_JOIN_TIMEOUT_S", 120),
    "VLLM_TPU_S3_ASYNC_STATS_EVERY": env_nonnegative_int("VLLM_TPU_S3_ASYNC_STATS_EVERY", 200),
    # q38 L6-S3ASYNC sub-marks (log-only): with VLLM_TPU_HOST_PHASES (SUBS or SLOW_MS) on, the
    # Stage-3 submit pass charges its parts as host_phases sub-phases ~s3_pass / ~s3_adm / ~s3_res /
    # ~s3_exp / ~s3_stage / ~s3_ag / ~s3_disp / ~s3_after, the TP submit-state broadcast ~s3_sync and
    # the Qwen sync repack ~s3_repack; with lever s3async on, the pass is ~s3a_pass and a blocking
    # apply ~s3a_join. charge() moves no boundary, so ~kc_start_load is unchanged.
    "VLLM_TPU_S3_SUBMARKS": env_bool("VLLM_TPU_S3_SUBMARKS", default=False),
    # q38 L6-S3ASYNC S3-DEFER-AR (core/l6_ab.py lever s3defer; host-only): 0 / 1 / ab:<classes>. On, a
    # sharded Stage-3 consumer ISSUES the pass's admission MIN all-reduce (gloo CPU group, async_op) in
    # start_load_kv and runs the rest of the submit pass after the step's forward and sampling dispatch,
    # on the model-runner thread (the deferral is Benliang Wang's QWEN4_EXP_PD_DEFER_LOAD_SUBMIT,
    # benliangw/q38fh-integ-pin-5b44702aea 09b6990cb, credited), with stage3_async.fast_reserve and his
    # one-int32 all_staged. Needs QWEN4_EXP_PD_BATCHED_ADMISSION=1 (A/B and CHECK off) and lever s3async
    # unset. An S3DEFER line every VLLM_TPU_S3_ASYNC_STATS_EVERY passes per rank is the engagement marker.
    "VLLM_TPU_S3_DEFER_AR": env_str("VLLM_TPU_S3_DEFER_AR", ""),
    # q38 L1-S3DGATE S3D load gate (core/l6_ab.py S3D_GATE_ENV; host-only, HUNT-11 §5.1 form b): T > 0 runs lever
    # s3defer in L6 step window k only if window k-1's mean scheduled-request count (its steps before the lockstep
    # write step) is at least T; the ab:<classes> pattern stays unchanged inside those windows, so its off classes are
    # an off share not confounded with batch size. Needs VLLM_TPU_S3_DEFER_AR=ab:<classes> and VLLM_TPU_L6_AB_STEPS > 0
    # (else l6_ab refuses at import). The decision rides the lockstep fingerprint (every rank must agree). An
    # 'S3DEFER gate rank<r>:' line per window per worker is the engagement marker. 0 = no gate (the default).
    "VLLM_TPU_S3_DEFER_GATE_MIN_REQS": env_nonnegative_int("VLLM_TPU_S3_DEFER_GATE_MIN_REQS", 0),
    # q38 D1-norm L-GDNCOPY (runner/tpu_runner.py _collect_mamba_state_seed_copies):
    # upload a step's mamba seed-copy index tensors in one H2D per dtype and take
    # device views, instead of one upload per tensor. The index values, and so
    # every compiled program, are unchanged (host-only, compile-ignored).
    "VLLM_TPU_GDN_SEED_COPY_PACKED": env_bool("VLLM_TPU_GDN_SEED_COPY_PACKED"),
    # Compare the packed device views with the per-tensor host arrays for the
    # first N packed crossing steps and raise on any difference (0 = off).
    "VLLM_TPU_GDN_SEED_COPY_PACKED_CHECK": env_nonnegative_int(
        "VLLM_TPU_GDN_SEED_COPY_PACKED_CHECK", 0
    ),
    # q38 D1-norm windowed A/B (core/d1n_ab.py): period P ms of the wall-clock
    # window classes (k mod 4: 0 L-PREP on, 1 off_a, 2 L-GDNCOPY on, 3 off_b);
    # 0 = off (an armed lever is uniformly on). Refused with D1e's period.
    "VLLM_TPU_D1N_AB_PERIOD_MS": env_nonnegative_int("VLLM_TPU_D1N_AB_PERIOD_MS", 0),
    # Log each rank's cumulative step counts per padded token bucket every N
    # steps (0 = off); runner/step_buckets.py.
    "VLLM_TPU_LOG_STEP_BUCKETS": env_nonnegative_int("VLLM_TPU_LOG_STEP_BUCKETS", 0),
    # q38 L5 STEPBUCKETS2: with VLLM_TPU_LOG_STEP_BUCKETS=N, also log each rank's
    # cumulative host step period per padded token bucket (host clock; log-only).
    "VLLM_TPU_LOG_STEP_BUCKETS_PERIOD": env_bool(
        "VLLM_TPU_LOG_STEP_BUCKETS_PERIOD", default=False
    ),
    # q38 L5 SAB: 'whole:P[:D]' alternates the whole-step megakernel ON / OFF (OFF = the
    # wholedry stock route) in windows of P serving calls, inside one served run;
    # megakernel_runtime/served_ab.py. Host-side routing between existing programs.
    "VLLM_TPU_SERVED_AB": env_str("VLLM_TPU_SERVED_AB", ""),
    # Let only one thread synchronize a sampled-token copy at a time (the
    # async-output thread and the deferred previous-results wait share one).
    "VLLM_TPU_SERIALIZE_COPY_WAIT": env_bool("VLLM_TPU_SERIALIZE_COPY_WAIT"),
    # Copy sampled tokens into reused pinned host buffers instead of a fresh
    # pageable tensor every step (runner/tpu_runner_async_output.py).
    "VLLM_TPU_PINNED_TOKEN_COPY": env_bool("VLLM_TPU_PINNED_TOKEN_COPY"),
    # Reuse the padded per-request temperature / top-k / top-p uploads while
    # their values repeat (tpu_runner._build_padded_sampling_params).
    "VLLM_TPU_SAMPLING_PARAMS_REUSE": env_bool("VLLM_TPU_SAMPLING_PARAMS_REUSE"),
    # q38 L4 SEEDFIX (runner/seedfix.py; default on): a request with its own
    # SamplingParams.seed is admitted (tpu_platform.validate_request) and its rows
    # draw their uniforms from its per-request generator (input_batch.generators),
    # overwriting their slices of the shared draws, so its output is reproducible
    # by seed; unseeded rows read the same uniforms as without it. Off: seeded
    # requests are rejected at admission, as before. Host-only (eager overlay; no
    # compiled program changes): in _TPU_COMPILE_ENV_IGNORED.
    "VLLM_TPU_SEEDED_SAMPLING": env_bool("VLLM_TPU_SEEDED_SAMPLING", True),
    # PD DEC-FB-2 VPS (runner/vps_wiring.py; layers/adapter/sample/vp_sampler.py):
    # spec-verify chunks keep the rank's vocab shard of the target logits and
    # sample from per-shard top-32 candidates (token-identical to the stock
    # verify with the default draws). Default off: today's verify.
    "VLLM_TPU_VPS_VERIFY": env_bool("VLLM_TPU_VPS_VERIFY"),
    # Per-row full-vocabulary fallback, at most this many rows per chunk (0 off).
    "VLLM_TPU_VPS_FALLBACK_ROWS": env_nonnegative_int("VLLM_TPU_VPS_FALLBACK_ROWS", 0),
    # Fallback rows reserved on every VPS chunk for exact-tie rows.
    "VLLM_TPU_VPS_TIEGAP_SLOTS": env_nonnegative_int("VLLM_TPU_VPS_TIEGAP_SLOTS", 0),
    # "full" (token-identical, VPBX) or "candidate" (one uniform per candidate,
    # VPB: same distribution, different tokens).
    "VLLM_TPU_VPS_DRAWS": env_str("VLLM_TPU_VPS_DRAWS", "full"),
    # Diagnostics (host-only, in _TPU_COMPILE_ENV_IGNORED): check the first N VPS
    # chunks against the stock verify; perturb the first M checks; log period.
    "VLLM_TPU_VPS_CHECK": env_nonnegative_int("VLLM_TPU_VPS_CHECK", 0),
    "VLLM_TPU_VPS_CHECK_FAULT": env_nonnegative_int("VLLM_TPU_VPS_CHECK_FAULT", 0),
    "VLLM_TPU_VPS_STATS_EVERY": env_nonnegative_int("VLLM_TPU_VPS_STATS_EVERY", 1000),
    # q38 L4 D83 (P3): under VPS the mamba read offsets take their own program
    # (`_update_read_offsets_compiled`) after the verify. With this flag the VPS
    # verify program also returns the scatter operands (padded, cast, one per
    # group) and the runner applies them with one eager index_put_. Same
    # offsets; the verify program and its warm-up change, so it is hashed.
    "VLLM_TPU_VPS_FUSED_READ_OFFSETS": env_bool("VLLM_TPU_VPS_FUSED_READ_OFFSETS"),
    # Diagnostics (host-only, in _TPU_COMPILE_ENV_IGNORED): for the first N folded
    # chunks also run the separate `_update_read_offsets_impl` on a copy and count
    # mismatching slots (VPS_RO_FOLD_CHECK lines); the first _FAULT checks perturb
    # the compared copy (a positive control that must be caught).
    "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK": env_nonnegative_int(
        "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK", 0
    ),
    "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT": env_nonnegative_int(
        "VLLM_TPU_VPS_FUSED_READ_OFFSETS_CHECK_FAULT", 0
    ),
    # q38 L4 VFY-ONE (H1 cards T-3 VFY-ONE + DH-C SEED-INTO-VERIFY; runner/vfy_one.py):
    # on a single-chunk VPS fallback step with D83's fold on, the local LM-head logits,
    # the verify, D83's read-offset operands and the fallback-stats accumulate run as ONE
    # compiled program (the three uniform draws stay eager, so the tokens and the random
    # stream are the D83 chain's). 2 also emits the draft seed, so the drafter's fused
    # seed program does not run. Values 0 / 1 / 2 (envs.vfy_one_value); the compile key
    # sees on / off only, because both variants are precompiled whenever it is on, so
    # the seed is switched on by value with no new compile. Adds programs: hashed.
    "VLLM_TPU_VPS_VFY_ONE": lambda: vfy_one_value() > 0,
    # Diagnostic (host-only, in _TPU_COMPILE_ENV_IGNORED): the first N VFY-eligible
    # serving steps run the D83 chain for the results and the VFY program beside it,
    # and count mismatching tokens / stats / read offsets / seed entries; any mismatch
    # on any TP rank keeps every rank on the D83 chain for the run (agreed at the next
    # window boundary).
    "VLLM_TPU_VPS_VFY_ONE_CHECK": env_nonnegative_int("VLLM_TPU_VPS_VFY_ONE_CHECK", 0),
    # Window switch (host-only, in _TPU_COMPILE_ENV_IGNORED; core/l6_ab.py lever
    # "vfyone"): "" or "1" = VFY-ONE on every window, "0" = the D83 chain, "ab:<c>[+<c>]"
    # = VFY-ONE in those l6_ab step-window classes only (needs VLLM_TPU_L6_AB_STEPS).
    # Picks between precompiled programs; the path changes only at window boundaries,
    # by an all-rank agreement (runner/vfy_one.py).
    "VLLM_TPU_VPS_VFY_ONE_AB": env_str("VLLM_TPU_VPS_VFY_ONE_AB", ""),
    # q38 L4 KEYLITE (host-only, in _TPU_COMPILE_ENV_IGNORED): VFY-ONE's warm-key test from the
    # scheduler integers that fix every shape, memoized per lite key, with a full-key audit every
    # 256 VFY calls and on CHECK steps (runner/vfy_one.py). Picks no program of its own.
    "VLLM_TPU_VPS_VFY_ONE_KEYLITE": env_bool("VLLM_TPU_VPS_VFY_ONE_KEYLITE", False),
    # q38 STALL-HUNT (host-only warm-up, in _TPU_COMPILE_ENV_IGNORED; reports/v8_stall_hunt_1116.md): at
    # value 2, serving's VFY call passes the SAME tensor as the verify's draft_lengths and as the seed's
    # num_draft (DraftChunkInputs.draft_lengths is the spec metadata's draft_lengths), while the warm-up
    # passed two distinct zero tensors. Dynamo guards the warm-up entry on no tensor aliasing, so every
    # token bucket's first serving step recompiled its VFY program (~9 s per bucket on v8 T94u). With this
    # on, the warm-up seed takes the warm-up call's own draft_lengths tensor, so the warm entry is the one
    # serving hits. The program keys (runner/vfy_one.key) and the served programs are unchanged.
    "VLLM_TPU_VFY_WARM_ALIAS": env_bool("VLLM_TPU_VFY_WARM_ALIAS", False),
    # MM DSTOCK 6dcff75dc (runner/vps_wiring.py), ported by q38 DEC-FB-3 lever 2:
    # the MTP draft's greedy token from LogitsProcessor.get_top_tokens (each rank
    # argmaxes its vocab shard, one (value, id) exchange) at the propose, carry
    # and fused-loop sites, instead of argmax over all-gathered logits. The same
    # token, ties included. Only for drafts with plain_compute_logits.
    "VLLM_TPU_DRAFT_TOP_TOKENS": env_bool("VLLM_TPU_DRAFT_TOP_TOKENS"),
    # Diagnostic (host-only, in _TPU_COMPILE_ENV_IGNORED; q38 D-MK's inline check):
    # the first N step-0 proposals also run the full-logits argmax and log the
    # mismatch count (DRAFT_TOP_TOKENS_CHECK lines).
    "VLLM_TPU_DRAFT_TOP_TOKENS_CHECK": env_nonnegative_int("VLLM_TPU_DRAFT_TOP_TOKENS_CHECK", 0),
    # Sample the step thread's Python stack for N s (runner/py_sampler.py).
    "VLLM_TPU_PY_SAMPLER_S": env_float("VLLM_TPU_PY_SAMPLER_S", 0.0),
    # Keep the per-group (block table, width) list the staged metadata walk
    # builds, instead of re-walking every KV cache group each step.
    "VLLM_TPU_STAGE_ENTRIES_CACHE": env_bool("VLLM_TPU_STAGE_ENTRIES_CACHE"),
    # Async spec decode, one chunk: the next step's substitution source and the
    # rejected count (with their row fits and the source pad) as one program.
    "VLLM_TPU_SPEC_TAIL_COMPILED": env_bool("VLLM_TPU_SPEC_TAIL_COMPILED"),
    # q38 L4 EAGER-FIX (E-a): the async-spec rejected count is padded to max_num_reqs
    # rows (zeros past the live requests) inside the compiled spec tail and on the
    # other two producers, so the next step's POSTFWD_FUSED shape key no longer
    # carries the previous step's request count and served steps stop falling to
    # the eager subtract / extract path. Same values on every row read. Changes the
    # spec tail program's output shape: compile-keyed (hashed).
    "VLLM_TPU_SPEC_TAIL_REJ_PAD": env_bool("VLLM_TPU_SPEC_TAIL_REJ_PAD"),
    # q38 L4 DRAFT-ASM (DH-A NODEFER-LOOP + T-4 SPEC-TAIL-FOLD; spec_decode/draft_asm.py):
    # on a steady async decode verify step the fused seed program also emits step 1's
    # seq_lens and the spec tail's bonus / rejected rows, the fresh carry program also
    # emits step 1's positions, and a second fused draft-loop program also emits the
    # stride-(1+K) [bonus, drafts] source. The draft path then runs no eager op (no
    # seq_lens + 1 - rejected, positions + 1, stack / cat), passes no deferred argument
    # to the loop, and sample_tail stores the source instead of calling the spec tail.
    # The base programs stay precompiled beside the folded ones (SAB dual-path), so a
    # lockstep window can switch per step; both paths give the same integers. Adds
    # programs: compile-keyed (hashed). On-value "1".
    "VLLM_TPU_DRAFT_ASM_FOLD": env_bool("VLLM_TPU_DRAFT_ASM_FOLD"),
    # q38 L4 LMH-STREAM stage 0 (port of PR 2821's TPU_LM_HEAD_HIDDEN_MAJOR, credit Yin Lin;
    # runner/utils.py arm_lm_head_hidden_major): the MTP draft's LM head is stored
    # [hidden, vocab shard] row-major, so its projection is a plain (m,k)@(k,n) matmul.
    # The draft shares the target's head module; with this flag alone the draft gets
    # its own hidden-major copy (one extra vocab shard x hidden table per rank) and the
    # target's head is untouched. Same weights, but the hidden dimension may be summed
    # in a different order (last-bit logits differences): draft AL only. Read once at
    # load; changes the draft programs' weight operand: compile-keyed (hashed).
    "VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT": env_bool("VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT"),
    # q38 L4 LMH-STREAM stage 0, verify head (PR 2821's TPU_LM_HEAD_HIDDEN_MAJOR as
    # written): the target's LM head is relayouted in place, so the target logits (and
    # with them the verify) may change in the last bit: accuracy class B. Needs
    # VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT=1 when a draft shares the head (both heads then
    # use the one hidden-major table, no extra HBM). Read once at load; hashed.
    "VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY": env_bool("VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY"),
    # q38 L4 DRAFT-ASM CHECK: on the first N fold-eligible serving steps run the base
    # path for the step's results and the folded programs beside it, and compare the
    # drafts, step 1's seq_lens / positions, the source and the rejected count (one
    # D2H each). The first mismatch logs one ERROR and turns the fold off for the rest
    # of the run. 0 = off. Host-side (ignored).
    "VLLM_TPU_DRAFT_ASM_CHECK_STEPS": env_nonnegative_int(
        "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", 0
    ),
    # q38 L4 DRAFT-ASM window switch, read only where core/l6_ab.py registers the lever
    # "draftasm" to this variable: "" = follow VLLM_TPU_DRAFT_ASM_FOLD on every step,
    # "0" = base path, "1" = fold, "ab:<c>[+<c>]" = fold in those l6_ab step-window
    # classes only. Picks between precompiled programs: host-side (ignored).
    "VLLM_TPU_DRAFT_ASM_AB": env_str("VLLM_TPU_DRAFT_ASM_AB", ""),
    # q38 L4 DRAFT1-B stage A (H1 card DH-B CARRY-INTO-LOOP; spec_decode/draft_one.py): with
    # VLLM_TPU_DRAFT_ASM_FOLD on, the fold's step-0 carry (position carry, first draft token,
    # hidden carry) becomes the prologue of the fused draft-loop program, so a folded step runs
    # three compiled draft programs (seed, step-0 forward, carry + loop) instead of four. The
    # folded seed also emits step 1's positions. Adds programs: compile-keyed (hashed). On-value
    # "1"; needs VLLM_TPU_DRAFT_ASM_FOLD.
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP": env_bool("VLLM_TPU_DRAFT_ASM_CARRY_LOOP"),
    # q38 L4 DRAFT1-B DH-D (H1 card DH-D DRAFT1-B, with X-D14-DRAFT-3's int32 draft metadata):
    # with VLLM_TPU_DRAFT_ASM_FOLD on, draft step 0, the carry and the K-1 loop steps run as one
    # compiled program per (token bucket, loop bucket); the loop's attention metadata is derived
    # on device from step 0's (seq_lens, query_start_loc and request_distribution are switched
    # in place inside the program). Implies stage A's seed. Adds programs: compile-keyed
    # (hashed). On-value "1"; needs VLLM_TPU_DRAFT_ASM_FOLD.
    "VLLM_TPU_DRAFT_ASM_ONE": env_bool("VLLM_TPU_DRAFT_ASM_ONE"),
    # q38 L4 DRAFT1-B window switches, read only where core/l6_ab.py registers the lever
    # "draftasmcl" (stage A) / "draftasmone" (DH-D) to these variables, with the values of
    # VLLM_TPU_DRAFT_ASM_AB: "" = follow the stage flag, "0" = off, "1" = on, "ab:<c>[+<c>]" =
    # on in those l6_ab step-window classes only. A stage that is off runs the next lower fold
    # (DH-D -> stage A -> DRAFT-ASM). Picks between precompiled programs: host-side (ignored).
    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB": env_str("VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB", ""),
    "VLLM_TPU_DRAFT_ASM_ONE_AB": env_str("VLLM_TPU_DRAFT_ASM_ONE_AB", ""),
    # q38 L4 DRAFT-CAP inside DH-D (card DRAFT-CAP; spec_decode/draft_cap.py): each DH-D program
    # (VLLM_TPU_DRAFT_ASM_ONE) runs its draft passes over R_d = min(p, capacity) request slots, p
    # its loop bucket, instead of the request capacity (max-num-seqs): persistent per-p buffers
    # take the first R_d entries of seq_lens, query_start_loc and every per-group block table by
    # in-program copies, and step 0's draft_rows are the seed's [p] gather indices. Lossless;
    # same program count at smaller shapes. Changes the programs: compile-keyed (hashed).
    # On-value "1"; needs VLLM_TPU_DRAFT_ASM_ONE (ignored without it).
    "VLLM_TPU_DRAFT_ASM_CAP": env_bool("VLLM_TPU_DRAFT_ASM_CAP"),
    # q38 L4 T1-K1FUSE (spec_decode/k1fuse.py; K-step R4 at K = 1): the VFY-ONE verify and the
    # draft (draft_one.draft_body: superset seed + DH-D's body) as ONE guard-free compiled program
    # per static shape signature (per DH-D loop bucket, split by request bucket / fallback rows),
    # so a steady decode step runs one compiled call for verify + draft instead of two. Adds
    # programs: compile-keyed (hashed). On-value "1"; needs VLLM_TPU_VPS_VFY_ONE=2 (with the D83
    # fold), VLLM_TPU_DRAFT_ASM_FOLD and VLLM_TPU_DRAFT_ASM_ONE, else it stays off (logged once).
    "VLLM_TPU_K1FUSE": env_bool("VLLM_TPU_K1FUSE"),
    # q38 L4 T1-K1FUSE window switch, read only where core/l6_ab.py registers the lever "k1fuse"
    # to it, with the values of VLLM_TPU_DRAFT_ASM_AB: "" = follow the flag, "0" = off, "1" = on,
    # "ab:<c>[+<c>]" = on in those l6_ab step-window classes only. Off runs today's VFY + DH-D
    # chain. Picks between precompiled programs: host-side (ignored).
    "VLLM_TPU_K1FUSE_AB": env_str("VLLM_TPU_K1FUSE_AB", ""),
    # q38 L4 T1-K1FUSE CHECK: the first N K1-eligible serving steps run the fused program FIRST and
    # then today's VFY + DH-D chain for the step's results (its KV writes stand), and compare the
    # tokens, stats, read offsets, drafts, source and rejected count (one D2H each). K1 serves only
    # after the window closes with 0 mismatches over the TP group; a mismatch turns K1 off on every
    # rank at the next agreement (not a fail-stop). 0 = no CHECK. Host-side (ignored).
    "VLLM_TPU_K1FUSE_CHECK": env_nonnegative_int("VLLM_TPU_K1FUSE_CHECK", 256),
    # q38 L4 EAGER-FIX (E-c): vLLM InputBatch host->device sampling copies write the
    # whole fixed-length 1-D buffer instead of the [:num_reqs] slice (no per-batch-size
    # eager copy__as_strided_inverse program); same values on rows < num_reqs.
    # Host-side only (runner/sampling_copy_full.py).
    "VLLM_TPU_SAMPLING_COPY_FULL": env_bool("VLLM_TPU_SAMPLING_COPY_FULL"),
    # q38 L4 EAGER-FIX: log each eager layout first seen after warm-up at the suspect
    # call sites (EAGERSHAPE lines), at most N per process; 0 = off. Log-only
    # (runner/eager_shape_log.py).
    "VLLM_TPU_EAGER_SHAPE_LOG": env_nonnegative_int("VLLM_TPU_EAGER_SHAPE_LOG", 0),
    # q38 L4 EAGER-FIX (E-b): at the VPS verify precompile, run the served steps'
    # eager uniform draws (bonus / accept / recover, and fb_u for served fallback
    # widths) at each precompiled bucket N times with a scratch generator, so their
    # eager programs exist before serving; the sampling generator is never touched.
    # 0 = off. Host-side warm-up only.
    "VLLM_TPU_VPS_WARM_DRAWS": env_nonnegative_int("VLLM_TPU_VPS_WARM_DRAWS", 0),
    # q38 L4 VPSHOST host levers (l6_ab classes, core/l6_ab.py LEVERS; values 0 / 1 / ab:<c>[+<c>]),
    # read once per worker step, never at trace time (compile-ignored).
    # T-2 stage 0 FB-ORDER: on the VPS fallback path, skip the top-k candidate scan whose result the
    # fallback overwrites (runner/tpu_runner.py execute_model; bitwise by construction).
    "VLLM_TPU_L4_SAMP_FB_ORDER": env_str("VLLM_TPU_L4_SAMP_FB_ORDER", ""),
    # T-2 stage 1 SAMP-SNAP: the VPS verify's fallback row count F, padded sampling-parameter uploads and
    # num_active upload from a per-batch snapshot keyed by the chunk's request ids (runner/l4_vpshost.py).
    "VLLM_TPU_L4_SAMP_SNAP": env_str("VLLM_TPU_L4_SAMP_SNAP", ""),
    # Its check mode: every N-th snapshot hit compares the rows' bytes with the snapshot's (host only); 0 = off.
    "VLLM_TPU_L4_SAMP_SNAP_CHECK": env_nonnegative_int("VLLM_TPU_L4_SAMP_SNAP_CHECK", 0),
    # T-1(i) STATS-ASYNC: the VPS_FALLBACK counts every VLLM_TPU_VPS_STATS_EVERY chunks are copied to the host
    # without blocking and logged (the same line) once the copy lands (runner/l4_vpshost.py AsyncCounts).
    "VLLM_TPU_L4_VPS_STATS_ASYNC": env_str("VLLM_TPU_L4_VPS_STATS_ASYNC", ""),
    # q38 L4 S1: VPS local_candidates takes its two "stable descending sort, first kc" steps as an exact
    # top-kc over tie-free int32 keys (bf16 logits; same indices, same order). Changes the traced graph (hashed).
    "VLLM_TPU_VPS_TOPK_KEYS": env_bool("VLLM_TPU_VPS_TOPK_KEYS", default=False),
    # q38 L4 S2: the VPS candidate mask's top-k cutoff is the k-th value of one top-k over the candidate row instead of
    # topk_mask's 32-step bisection (bitwise equal). Read at import in vp_sampler.py; changes the traced graph (hashed).
    "VLLM_TPU_VPS_TOPK_CUTOFF": env_bool("VLLM_TPU_VPS_TOPK_CUTOFF", default=False),
    # q38 L4 VRNG-VPS: the VPS verify's uniforms drawn inside a second compiled program from a device int32 counter
    # (layers/adapter/sample/vps_rng.py; runner/l4_samp2.py), warmed beside the base programs; with
    # VLLM_TPU_VPS_VFY_ONE the VFY program also gets a counter variant. Same distribution, different random stream
    # (class R). Adds programs (hashed).
    "VLLM_TPU_VPS_INGRAPH_RNG": env_bool("VLLM_TPU_VPS_INGRAPH_RNG", default=False),
    # Its path switch (host only): "" / 1 = VRNG on every agreed call, 0 = base path, alt:P = alternating windows of P
    # VPS serving calls (even windows VRNG); the TP ranks agree on the path at every window start.
    "VLLM_TPU_VPS_INGRAPH_RNG_AB": env_str("VLLM_TPU_VPS_INGRAPH_RNG_AB", ""),
    # Dump all processes' stacks when no step starts for N s with live
    # requests (stall_watchdog.py; read directly there, registered here).
    "VLLM_TPU_STALL_WATCHDOG_S": env_float("VLLM_TPU_STALL_WATCHDOG_S", 0.0),
    # Non-greedy spec verify as one program (tpu_runner._verify_fused_impl).
    "VLLM_TPU_VERIFY_FUSED": env_bool("VLLM_TPU_VERIFY_FUSED"),
    # Compare the fused verify and draft seed with the separate programs for
    # the first N chunks; the first _FAULT checks perturb the compared copy
    # (a positive control that must be caught).
    "VLLM_TPU_VERIFY_FUSED_CHECK": env_nonnegative_int("VLLM_TPU_VERIFY_FUSED_CHECK", 0),
    "VLLM_TPU_VERIFY_FUSED_CHECK_FAULT": env_nonnegative_int(
        "VLLM_TPU_VERIFY_FUSED_CHECK_FAULT", 0
    ),
    # Async draft seed + input shift/scatter + gather pad as one program
    # (eagle3._async_seed_inputs_impl).
    "VLLM_TPU_DRAFT_SEED_FUSED": env_bool("VLLM_TPU_DRAFT_SEED_FUSED"),
    # Row-selected draft step 0: position carry + first draft token as one
    # program (eagle3._draft_rows_carry_and_token).
    "VLLM_TPU_DRAFT_CARRY_FUSED": env_bool("VLLM_TPU_DRAFT_CARRY_FUSED"),
    # With VLLM_TPU_MTP_DRAFT_ROWS: draft step 0's first token and hidden carry
    # from one program over the untrimmed [capacity] outputs, instead of eager
    # [:rows] views that torch_tpu copies with an execute_as_strided program
    # before the propose and the fused loop (same values; off with
    # VLLM_TPU_DRAFT_CARRY_FUSED, which takes precedence).
    "VLLM_TPU_DRAFT_ROWS_FRESH": env_bool("VLLM_TPU_DRAFT_ROWS_FRESH"),
    # With VLLM_TPU_DRAFT_CARRY_FUSED and MTP_DRAFT_ROWS: the carry program
    # takes step 0's untrimmed [capacity] outputs, trims inside and returns
    # the hidden carry as a buffer of its own (no execute_as_strided views
    # before the carry and the fused loop). Same values.
    "VLLM_TPU_DRAFT_CARRY_ROWS_FRESH": env_bool("VLLM_TPU_DRAFT_CARRY_ROWS_FRESH"),
    # With VLLM_TPU_VERIFY_FUSED: compute the mamba read offsets inside the
    # fused verify program and apply them with one eager scatter.
    "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS": env_bool("VLLM_TPU_VERIFY_FUSED_READ_OFFSETS"),
    # Warm-up only: cap the spec-verify precompile sweep (rejection sampler,
    # fused verify, VPS verify) at N requests (0 = off). max_num_seqs is
    # unchanged; larger request counts compile lazily. Host-only (ignored in
    # the compile key): it picks which programs are warmed, not their contents.
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS": env_nonnegative_int(
        "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_MAX_SEQS", 0
    ),
    # Warm-up only: with a VPS head or VLLM_TPU_VERIFY_FUSED, precompile only
    # the spec-verify program a non-greedy, no-logprobs, no-grammar step runs
    # (VPS at its in-range fallback width, else the fused verify); the others
    # compile lazily if reached. Host-only (ignored in the compile key).
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY": env_bool(
        "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_SERVED_ONLY"
    ),
    # q38 L4 EAGER-FIX (E-d): with SERVED_ONLY and a VPS head, also warm the served VPS
    # verify at the (logits bucket, request bucket) pairs a full-draft step of
    # MAX_SEQS+1 .. N live requests lands in (the cap drops them; at c48 two pairs
    # recompiled in steady state). 0 = off. Warm-up only: same programs serving compiles.
    "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS": env_nonnegative_int(
        "VLLM_TPU_SPEC_VERIFY_PRECOMPILE_FULL_DRAFT_MAX_SEQS", 0
    ),
    # With VLLM_TPU_DRAFT_SEED_FUSED: the fused seed also emits step 0's
    # draft_rows and the loop's padded rejected counts (no eager pads).
    "VLLM_TPU_DRAFT_SEED_EXTRAS": env_bool("VLLM_TPU_DRAFT_SEED_EXTRAS"),
    "VLLM_TPU_CPU_PIN_RESERVE": env_nonnegative_int("VLLM_TPU_CPU_PIN_RESERVE", 2),
    # Log each worker's thread CPU placement every N seconds (0 = off).
    "VLLM_TPU_CPU_PROBE_S": env_float("VLLM_TPU_CPU_PROBE_S", 0.0),
    # Async scheduling: wait for the previous step's sampled-token copy after
    # launching this step's draft rather than before it (sample_tokens).
    "VLLM_TPU_DEFER_PREV_RESULTS_WAIT": env_bool(
        "VLLM_TPU_DEFER_PREV_RESULTS_WAIT", default=False
    ),
    # Async spec decode: derive the draft seed (token, position, rejected
    # count) from the rejection-sampler output in one compiled program per
    # request-padding bucket instead of ~15 eager ones (eagle3.py
    # _async_draft_seed). Partial-prefill overrides keep the eager path.
    "VLLM_TPU_FUSED_ASYNC_DRAFT_SEED": env_bool(
        "VLLM_TPU_FUSED_ASYNC_DRAFT_SEED", default=True
    ),
    # Spec decode on hybrid models: scatter the post-verify mamba read
    # offsets of all groups in one compiled program instead of ~10 eager ones
    # (tpu_runner.py _update_read_offsets_impl).
    "VLLM_TPU_FUSED_READ_OFFSET_UPDATE": env_bool(
        "VLLM_TPU_FUSED_READ_OFFSET_UPDATE", default=True
    ),
    # Metadata walk: reuse the previous step's staged block-table upload when
    # its geometry and contents are unchanged (layers/core/attention_metadata.py
    # stage_block_table_uploads).
    "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": env_bool(
        "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE", default=False
    ),
    # q38 DEC-HOST-B BT-INC: when the staged walk's geometry is unchanged, repack
    # only the block-table rows BlockTable mutations marked dirty instead of the
    # full zero + per-group copy + compare + clone (attention_metadata.py). The
    # staged bytes are identical, so the compiled programs do not change.
    "VLLM_TPU_BLOCK_TABLE_STAGE_INC": env_bool(
        "VLLM_TPU_BLOCK_TABLE_STAGE_INC", default=False
    ),
    # Compare each of the first N incremental packs with a full pack and raise on
    # any difference (a served guard for BT-INC; 0 = off).
    "VLLM_TPU_BLOCK_TABLE_STAGE_INC_CHECK": env_nonnegative_int(
        "VLLM_TPU_BLOCK_TABLE_STAGE_INC_CHECK", 0
    ),
    # Skip the token-padding mask's synchronous H2D when its rows are the same
    # as the last write (layers/adapter/token_padding.py).
    "VLLM_TPU_TOKEN_PADDING_REUSE": env_bool("VLLM_TPU_TOKEN_PADDING_REUSE"),
    # Staged metadata walk: build every group's AttentionMetadata inline instead
    # of one builder call per group (tpu_runner._build_attention_metadata_staged).
    "VLLM_TPU_FAST_METADATA_BUILD": env_bool("VLLM_TPU_FAST_METADATA_BUILD"),
    # Host-path pieces (q38f-int: default off, toggled per arm).
    # Record stage-timing device events only when a KV connector consumes them.
    "VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK": env_bool(
        "VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK", default=False
    ),
    # q38 D1b: no runner stage timing at all (no TpuEvent records, no collector
    # thread, no vllm:tpu_runner_* samples), even with a KV connector. Host
    # side only; set it per role (the PD decode) through the role's env.
    "VLLM_TPU_STAGE_TIMING_OFF": env_bool("VLLM_TPU_STAGE_TIMING_OFF", default=False),
    # Staged walk: one gather per mamba geometry class instead of per group.
    "VLLM_TPU_STAGED_WALK_CLASS_GATHER": env_bool(
        "VLLM_TPU_STAGED_WALK_CLASS_GATHER", default=False
    ),
    # Async substitution: pad next_tokens to a precompiled source length.
    "VLLM_TPU_WARMED_SUBSTITUTE_SOURCE": env_bool(
        "VLLM_TPU_WARMED_SUBSTITUTE_SOURCE", default=False
    ),
    # MTP propose: upload per-request-count constants once and reuse them.
    "VLLM_TPU_PROPOSE_DEVICE_CONSTANTS": env_bool(
        "VLLM_TPU_PROPOSE_DEVICE_CONSTANTS", default=False
    ),
    # MTP draft step 0 on a draft declaring `supports_draft_rows` (Qwen4Exp):
    # write the draft layer's caches for every token of the target's step but
    # run attention, MoE and hyper-connections only at each request's last
    # token, the rows that produce a draft. Off by default until validated on
    # TPU. Not used with IndexShare, context parallelism or EP-DP lockstep.
    "VLLM_TPU_MTP_DRAFT_ROWS": env_bool("VLLM_TPU_MTP_DRAFT_ROWS"),
    # Non-greedy sampling filters the vocab to this many 128-logit blocks,
    # ranked by block max, before running top-k (and any subsequent top-p).
    # 0 searches the full vocab. Used only when every active request has
    # 1 <= top_k <= value // 2 and no grammar mask applies.
    "TPU_SAMPLER_TOPK_CANDIDATES": env_nonnegative_int(
        "TPU_SAMPLER_TOPK_CANDIDATES", 0
    ),
    # Async spec decode: the next step's rejected-draft subtraction, placeholder
    # token substitution and draft-token extraction as one compiled program
    # instead of ~5 eager ones plus the substitution program (q38 DEC-FB-2 G2,
    # after Qwen3.5 E4c 1784e91de; the same integer values).
    "VLLM_TPU_POSTFWD_FUSED": env_bool("VLLM_TPU_POSTFWD_FUSED"),
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
    # Enable and configure the fused Pallas megakernel ("1"/"true" for defaults,
    # "ep"/"tp" for mode, or "mode=tp,buckets=8:16:32,verify_buckets=4:8:16,ab=1").
    "VLLM_TPU_MEGAKERNEL": env_str("VLLM_TPU_MEGAKERNEL", ""),
    # Qwen3.8 megakernel: boundary-friendly kernel operands (flat smalls, fresh
    # tensors, small packed operands round-tripped through the host). torch_tpu
    # re-materializes a buffer with an in-place slice write or a dtype view on
    # every call (~22 as_strided/step, ~1 ms); exact, so on by default.
    "VLLM_TPU_MK_OWN_OPERANDS": env_bool("VLLM_TPU_MK_OWN_OPERANDS", True),
    # q38 L5 CTXCENSUS (debug, log-only): every N real whole-step calls (and every N steps sent to
    # stock by `wholecap`) TP rank 0 logs the slots' host-side computed-token counts in slot order
    # (megakernel_runtime/qwen38_driver.py `_ctx_census`). Host only; compile-ignored. 0 (default) = off.
    "VLLM_TPU_Q38_CTX_CENSUS": env_nonnegative_int("VLLM_TPU_Q38_CTX_CENSUS", 0),
    # Knobs of the megakernel tree imported from dongruizi / MM 37b960ea9 (c32 DCP-2 path). The first four are
    # MM's own registrations (envs.py at 37b960ea9). The rest are read straight from os.environ at trace time by
    # the imported code (same defaults as there); they are registered here only so that they key the TPU compile
    # cache (_tpu_compile_env_factors hashes every registered name not in _TPU_COMPILE_ENV_IGNORED): a raw read
    # outside this registry changes the traced program without changing the cache key.
    "VLLM_TPU_MK_HOST_PLE": env_bool("VLLM_TPU_MK_HOST_PLE"),
    "VLLM_TPU_MK_OPERAND_DUMP": env_bool("VLLM_TPU_MK_OPERAND_DUMP", False),
    "VLLM_TPU_MK_STEP_MIX_INTERVAL_S": env_float("VLLM_TPU_MK_STEP_MIX_INTERVAL_S", 10.0),
    "CDK_OUTPUT_DIR": env_str("CDK_OUTPUT_DIR", "/tmp/cdk_output"),
    "VLLM_TPU_MK_GDN_CHUNK_GATHER": env_str("VLLM_TPU_MK_GDN_CHUNK_GATHER", "auto"),
    "VLLM_TPU_MK_GDN_HM": env_str("VLLM_TPU_MK_GDN_HM", "1"),
    "VLLM_TPU_MK_GDN_REC_F32": env_str("VLLM_TPU_MK_GDN_REC_F32", "0"),
    "VLLM_TPU_MK_HIER": env_str("VLLM_TPU_MK_HIER", ""),
    "VLLM_TPU_MK_MOE_ABLATE": env_str("VLLM_TPU_MK_MOE_ABLATE", ""),
    "VLLM_TPU_MK_RINGS": env_str("VLLM_TPU_MK_RINGS", ""),
    # q38 HIX700 PREGATE: P > 0 prefetches P predicted routed experts per rank into the whole step's expert ring
    # before routing (MoETunables.pregate, in-kernel predictor, issued before the mixer all-reduce; bit-identical
    # side-buffer replay) for whole-step buckets of at most VLLM_TPU_HIX_PREGATE_MAX_ROWS rows
    # (megakernel_runtime/hix700_pregate.py). Changes the kernel, so both are hashed. Default 0 (off).
    "VLLM_TPU_HIX_PREGATE": env_int("VLLM_TPU_HIX_PREGATE", 0),
    "VLLM_TPU_HIX_PREGATE_MAX_ROWS": env_int("VLLM_TPU_HIX_PREGATE_MAX_ROWS", 16),
    "VLLM_TPU_MK_SCOPES": env_str("VLLM_TPU_MK_SCOPES", "0"),
    "VLLM_TPU_MK_UNION_STACK": env_str("VLLM_TPU_MK_UNION_STACK", "1"),
    # q38 L1-D08S1: D08 S1 "UNION-LEAN" stages of the whole step's in-kernel indexer (wholestep/qsa_full_lean.py), each
    # default off and bitwise exact; read at trace time, so hashed (not in _TPU_COMPILE_ENV_IGNORED).
    "VLLM_TPU_MK_UNION_BOUNDED": env_bool("VLLM_TPU_MK_UNION_BOUNDED"),
    "VLLM_TPU_MK_UNION_LEAN": env_bool("VLLM_TPU_MK_UNION_LEAN"),
    "VLLM_TPU_MK_SLOT_BATCH": env_bool("VLLM_TPU_MK_SLOT_BATCH"),
    # q38 L2-UAISSUE (X-UA-ISSUE): the whole step's union-walk DMA descriptors (wholestep/union_issue.py): 0 off,
    # 1 stage (a) slab-bounded physical ids, 2 stages (a) + (b) MM D12 F0a's issue recipe; bitwise exact, read at trace
    # time, so hashed (not in _TPU_COMPILE_ENV_IGNORED).
    "VLLM_TPU_MK_UNION_ISSUE": env_int("VLLM_TPU_MK_UNION_ISSUE", 0),
    # q38 L2-UNPACK (X-UA-UNPACK): the whole step's union tile body reads its ring slot with full-vreg strided loads
    # (wholestep/union_unpack.py), head pair 0 only; bitwise exact, read at trace time, so hashed (not in
    # _TPU_COMPILE_ENV_IGNORED).
    "VLLM_TPU_MK_UNION_UNPACK": env_bool("VLLM_TPU_MK_UNION_UNPACK"),
    # q38 L2-UAPIPE: X-UA-PIPE on the whole step's union attention (bit mask 1 co-issue / 2 cross-request / 4 starts
    # after the unpack; wholestep/union_attn.py), default 0 = off, bitwise exact; read at trace time, so hashed.
    "VLLM_TPU_MK_UNION_PIPE": env_nonnegative_int("VLLM_TPU_MK_UNION_PIPE", 0),
    # q38 L1-D07B1: the whole-step call takes the GDN read offsets, query starts and sequence lengths once
    # (they are the runner's step-level tensors, shared by every GDN layer's metadata) and derives all 33
    # layers' read / write / has-initial block ids in one vectorised pass instead of 33 per-layer chains
    # (qwen38_wholestep_call.call_shared). Changes the traced program and its operand list, so it is hashed.
    "VLLM_TPU_MK_GDN_META_DEDUP": env_bool("VLLM_TPU_MK_GDN_META_DEDUP", False),
    # q38 L1-GLUEFIX: with VLLM_TPU_MK_GDN_META_DEDUP=1, the shared-metadata whole-step program only for buckets of at
    # least this many rows; a smaller bucket runs the flag-off program (`call`, per-layer GDN ids), its operands expanded
    # from the shared list on the host (qwen38_wholestep._WholeKind.callable_for). 0 = every bucket (the D07-b1
    # behaviour). Changes which program a bucket compiles, so it is hashed.
    "VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS": env_nonnegative_int("VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS", 0),
    # q38 L1-IDX: the in-kernel indexer package on the whole step (D14 path; qwen38_wholestep_call._idx_options /
    # _idx_pps / _ring_s1, wholestep/qsa_full_idx.py). Every knob is default off, read at build / trace time and hashed
    # (not in _TPU_COMPILE_ENV_IGNORED).
    # S2P (= D08 S2 = D18): the key scorer's chunk pages ("0" = the kernel's 32), carrier lanes read per row ("0" = all
    # 256; "128" = the real lanes only) and manager-page DMA run ("0" = one DMA per kernel page; "4" = one per 512-token
    # pool block, with the compressed-key table `pages[:, :n_ck:4] // 4` and trace-time layout asserts).
    "VLLM_TPU_MK_SCORE_CHUNK": env_with_choices("VLLM_TPU_MK_SCORE_CHUNK", "0", ["0", "32", "64", "128"]),
    "VLLM_TPU_MK_SCORE_LANES": env_with_choices("VLLM_TPU_MK_SCORE_LANES", "0", ["0", "128", "256"]),
    "VLLM_TPU_MK_SCORE_MP": env_with_choices("VLLM_TPU_MK_SCORE_MP", "0", ["0", "2", "4"]),
    # S2P radix-2: the selection's thresholds decide two bits per counting pass (identical (thr, tmax)).
    "VLLM_TPU_MK_SELECT_RADIX": env_with_choices("VLLM_TPU_MK_SELECT_RADIX", "1", ["1", "2"]), "VLLM_TPU_MK_SCAN_BOUND": env_with_choices("VLLM_TPU_MK_SCAN_BOUND", "0", ["0", "1"]),  # q38 HXSCAN
    # QSPLIT: under main-KV DCP each DCP rank's union is compacted from its own rows of the code array (512-token pool
    # blocks: group g is on rank (g // 128) % D = its sublane row mod D): "1" by a strided sublane load, "2" by a
    # one-hot row-gather matmul (the fallback if Mosaic refuses the strided load). Bitwise.
    "VLLM_TPU_MK_UNION_QSPLIT": env_with_choices("VLLM_TPU_MK_UNION_QSPLIT", "0", ["0", "1", "2"]),
    # q38 HXSMALL (DSTEP lane, 2026-11-22): the five bitwise S2 smalls of the ox 16-row decode step, each default "0"
    # and strict ("0" / "1"; anything else raises). Each is read at trace time and changes the traced program, so each
    # is registered here and hashed (not in _TPU_COMPILE_ENV_IGNORED). Each prints one engagement line when it engages.
    # GLUE_TRIM (P1d): forward glue outside the whole-step kernel: one batched ring-tag read for every whole-step QSA
    # layer, the writer's ring read / write id vectors padded to whole DMA batches, one merged rope gather.
    "VLLM_TPU_HXS_GLUE_TRIM": env_with_choices("VLLM_TPU_HXS_GLUE_TRIM", "0", ["0", "1"]),
    # LEAN_DRAFT (1a): the MTP draft program's lean form (pad rows skip the draft MoE, one metadata schedule for the
    # draft steps, in-place selection sink); live rows bitwise.
    "VLLM_TPU_HXS_LEAN_DRAFT": env_with_choices("VLLM_TPU_HXS_LEAN_DRAFT", "0", ["0", "1"]),
    # VERIFY_FUSE (6, XLA part): the verify program's sampler / acceptance glue fused; bitwise incl. the RNG order.
    "VLLM_TPU_HXS_VERIFY_FUSE": env_with_choices("VLLM_TPU_HXS_VERIFY_FUSE", "0", ["0", "1"]),
    # AR0ST (B AR0S-T): the whole step's last routed expert finishes the twin's column half first and starts the hier=2
    # all-reduce round 0 before its own half (in-kernel); bitwise.
    "VLLM_TPU_HXS_AR0ST": env_with_choices("VLLM_TPU_HXS_AR0ST", "0", ["0", "1"]),
    # RUNNER_HOIST (R, device part): fewer runner input programs per decode step; the forward's inputs bitwise.
    "VLLM_TPU_HXS_RUNNER_HOIST": env_with_choices("VLLM_TPU_HXS_RUNNER_HOIST", "0", ["0", "1"]),
    # TSLIM (table half of D08 S3): "1" sizes the main page table operand and its SMEM window to
    # max_groups // (groups per pool block * DCP) columns (128 at DCP 4) instead of max_groups * 4 // 128 (2048);
    # "2" also double-buffers the per-layer SMEM windows and prefetches the next QSA layer's rows.
    "VLLM_TPU_MK_TABLE_SLIM": env_with_choices("VLLM_TPU_MK_TABLE_SLIM", "0", ["0", "1", "2"]),
    # RING-S1: per-bucket whole-step ring depths "B=nbe:nbd[,B=nbe:nbd...]" (e.g. "32=7:10,64=6:4"), applied only
    # when the D08 S1 envs VLLM_TPU_MK_UNION_LEAN and VLLM_TPU_MK_UNION_BOUNDED are on (S1 frees the VMEM) and main-KV
    # DCP > 2; VLLM_TPU_MK_RINGS still overrides it. "" = off.
    "VLLM_TPU_MK_RING_S1": env_str("VLLM_TPU_MK_RING_S1", ""),
    # q38 X-K-LIVE (L1 GDN half, L3 MoE half): the whole-step call passes a live-row mask (SMEM int32 [B], from the
    # query start locations) and the kernel skips pad rows' GDN / MoE work (megakernel/live_rows.py). Changes the
    # traced program and its operand list, so it is hashed. Default off.
    "VLLM_TPU_MK_LIVE_ROWS": env_bool("VLLM_TPU_MK_LIVE_ROWS", False),
    # q38 X-K-LIVE GDN compute sub-mode (with VLLM_TPU_MK_LIVE_ROWS=1): pad requests also skip the GDN recurrence and their
    # state copies (ExecSpec.live_rows = 2). Without it the GDN half skips only pad requests' pool-gather DMAs (bitwise vs
    # the flag off by construction). Changes the traced program, so it is hashed. Default off.
    "VLLM_TPU_MK_LIVE_GDN_COMPUTE": env_bool("VLLM_TPU_MK_LIVE_GDN_COMPUTE", False),
    "VLLM_TPU_MK_KERN_REC_SKIP": env_bool("VLLM_TPU_MK_KERN_REC_SKIP", False),  # q38 HIX-KERN K8
    # q38 X-GLUE-1-D14: the pool GDN block writes its verify checkpoints (f32 state + conv window) into the pool
    # in-kernel (megakernel/models/qwen38_gdn_ckpt_ink.py); the whole-step / segment calls drop the rec_ckpt / conv_ckpt
    # outputs, the conv-window prep and the scatter. Changes the traced program, so it is hashed. Default off.
    "VLLM_TPU_MK_GDN_CKPT_INK": env_bool("VLLM_TPU_MK_GDN_CKPT_INK", False),
    # q38 HIX700 GDN-SIDECAR: the whole step keeps its GDN verify checkpoints in a slot-indexed sidecar and the next step reads the
    # accepted one from it (megakernel/models/qwen38_gdn_sidecar.py); the scatter runs only when the runtime flushes. Changes the
    # traced program, so it is hashed. Default off.
    "VLLM_TPU_MK_GDN_SIDECAR": env_bool("VLLM_TPU_MK_GDN_SIDECAR", False),
    # q38 L1-G1 GDN-COMMIT1: the whole-step call skips the GDN checkpoint scatter and returns the checkpoints; the runner
    # writes only the accepted one per request and layer after the read-offset update (megakernel_runtime/
    # qwen38_gdn_commit.py). Changes the traced whole-step program (outputs, no scatter), so it is hashed. Default off.
    "VLLM_TPU_MK_GDN_COMMIT1": env_bool("VLLM_TPU_MK_GDN_COMMIT1", False),
    # q38 L1-QX (X-DCP-QX-EARLY, cycle 6): "1" = under main-KV DCP 2 / 4 the whole step starts the DCP q exchange right after
    # the QSA projection (`_project`) and waits for it just before the union attention, so the transfer hides under the
    # indexer chain (`_select`); same copies, same values (wholestep/qsa_full_idx.py `_QxEarlyMixin`, bitwise). Read at build /
    # trace time by qwen38_wholestep_call._qx_options; changes the traced kernel, so it is hashed. Default "0" (off).
    "VLLM_TPU_MK_DCP_QX_EARLY": env_with_choices("VLLM_TPU_MK_DCP_QX_EARLY", "0", ["0", "1"]),
    # q38 L1-L47F (L47-FOLD-S1): the whole step covers model layers 3..47 (L 45) and the stock layer-47 path is skipped;
    # the MTP draft's shared-selection sink is written from the kernel's last QSA layer (megakernel_runtime/qwen38_l47fold.py).
    # Changes the traced programs and the kernel, so it is hashed. Needs the D08 S1 lean block. Default off.
    "VLLM_TPU_MK_L47_FOLD": env_bool("VLLM_TPU_MK_L47_FOLD", False),
    # q38 L1-H3 (H3-DRAFT-ATTN-D14): the MTP draft's fused-loop QSA attention runs as one L = 1 attention-only
    # megakernel call per pass (megakernel/wholestep/qsa_draft.py, megakernel_runtime/qwen38_draft_attn.py) instead of
    # the stock shared-selection step + DCP collectives. Read when the draft is built / traced (it changes the draft's
    # traced program and operand list), so it is hashed. Needs the attached MTP tail (`mtp=1`), shared selection and
    # VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE. Default off.
    "VLLM_TPU_MK_DRAFT_ATTN": env_bool("VLLM_TPU_MK_DRAFT_ATTN", False),
    # q38 L1-GDNINK: in-kernel GDN variants of the megakernel (read at trace time by megakernel/blocks/gdn/recurrence.py
    # `gdnink_mode`; megakernel_runtime/qwen38_wholestep_call.py `_gdnink_rel_fetch`), a bit mask: "0" / unset = off; bit 1 =
    # GDN-REC-WSHM (head-major state update on the whole-state path, <= 16 rows; same f32 operations per element); bit 2 =
    # GDN-REL-FETCH (the whole step built with Tiles.rel_fetch: dense refills decoded at trace time; scheduling only).
    # Changes the traced program, so it is hashed. Default off.
    "VLLM_TPU_MK_GDN_INK": env_str("VLLM_TPU_MK_GDN_INK", "0"),
    # q38 L1-SELLOOP: the whole step's in-kernel selection loops only over this rank's owned request slots
    # (q = run * npr + i, i < npr) instead of over every slot with the owned work under `pl.when` (wholestep/
    # qsa_full_lean.py `_owned_slots`; needs wholeloop=1, wholeshard=1 and VLLM_TPU_MK_SLOT_BATCH=1). "1": one
    # straight-line iteration at one request per rank, a fori_loop over the owned count above; "2": every owned slot
    # unrolled. Bitwise; changes the traced program, so it is hashed. Default "0" (off).
    "VLLM_TPU_MK_SELECT_OWNED": env_with_choices("VLLM_TPU_MK_SELECT_OWNED", "0", ["0", "1", "2"]), "VLLM_TPU_MK_IDX_TWIN": env_with_choices("VLLM_TPU_MK_IDX_TWIN", "0", ["0", "1"]),  # q38 HXIDX: the whole step's selection split over a twin core pair at <= R / 2 request slots (wholestep/qsa_full_twin.py); bitwise, changes the traced program, hashed. Default "0" (off). Same line.
    # q38 L2 KVW-ALL: the whole step's post-kernel QSA cache writes as one all-layer Pallas call
    # (megakernel_runtime/qwen38_qsa_write_all.batched_writes_all; MM's lean-writer pattern over the layer axis).
    # Read at trace time by the whole-step write dispatch, so hashed (not in _TPU_COMPILE_ENV_IGNORED).
    "VLLM_TPU_MK_QSA_WRITE_ALL": env_bool("VLLM_TPU_MK_QSA_WRITE_ALL", False),
    # q38 L2 WFOLD (QSA-WRITE-FOLD stage A): with VLLM_TPU_MK_QSA_WRITE_ALL also on, the all-layer writer takes the
    # whole-step kernel's own QSA cache values (KV_NEW keys, CK_NEW, RING_NEW) instead of recomputing them in XLA.
    # Read when the whole-step call is built (extra kernel outputs, writer operands), so hashed.
    "VLLM_TPU_MK_QSA_WRITE_KVAL": env_bool("VLLM_TPU_MK_QSA_WRITE_KVAL", False),
    # q38 HIX-KERN (megakernel_runtime/qwen38_kern.py): K1 SIDE_PREFETCH = the GDN-SIDECAR block starts ordinal o+1's sidecar copies at the end
    # of ordinal o (same copies, same bytes); K3 ROUTE_VEC = the MoE hit list from an order-preserving vector pass (same elist / n_hit).
    # Both change the traced kernel, so both are hashed (compile key). Default off.
    # K6 STOCK_KV = the union walk feeds FP8 K / V to the dots, KV scales after them (wholestep/union_attn.py).
    "VLLM_TPU_MK_KERN_SIDE_PREFETCH": env_bool("VLLM_TPU_MK_KERN_SIDE_PREFETCH", False), "VLLM_TPU_MK_KERN_ROUTE_VEC": env_bool("VLLM_TPU_MK_KERN_ROUTE_VEC", False),
    "VLLM_TPU_MK_KERN_STOCK_KV": env_bool("VLLM_TPU_MK_KERN_STOCK_KV", False),
    # q38 PORT-BL GDN (begin): Benliang Wang's GDN lean glue (aa8e6b237, VLLM_TPU_GDN_LEAN_GLUE) less its
    # window-log piece ("no_ckpt", which needs VLLM_TPU_GDN_WINDOW_LOG and so refuses D14). On the per-region unified
    # pool, every pooled GDN call with read offsets (speculative decoding) reads the step's stacked slot rows (one
    # read-offset gather per step instead of one per layer, PER_SEQ metadata by absolute request index as
    # VLLM_TPU_GDN_HOIST_READ_OFFSETS does, which it subsumes); every pooled call's BATCHED kernel DMA-writes the
    # output's zero rows (no XLA zeros_like per layer), and b / a reach the kernels as one operand. Bitwise (outputs,
    # GDN state and checkpoint pool). Decode engine env only. Changes the traced programs, so it is hashed. Default off.
    "VLLM_TPU_GDN_LEAN_GLUE3": env_bool("VLLM_TPU_GDN_LEAN_GLUE3", False),
    # With VLLM_TPU_GDN_LEAN_GLUE3: log the engagement line (layers/adapter/custom_ops/gdn_lean_engagement.py) every N
    # target forwards on each rank; 0 = off. Host-side, log-only (in _TPU_COMPILE_ENV_IGNORED).
    "VLLM_TPU_GDN_LEAN_GLUE3_COUNT": env_nonnegative_int("VLLM_TPU_GDN_LEAN_GLUE3_COUNT", 0),
    # q38 PORT-BL GDN (end)
    # q38 PORT-BL UNION (ported from Benliang Wang <benliangw@google.com>, commit 2ce02bb1a on
    # benliangw/q38fh-integ-pin-5b44702aea; his VLLM_TPU_QSA_DECODE_RPA_UNION, whose only call site is his
    # compressed-DCP attend). Q38 O1s target verify steps (pooled_qsa_request_attend_dcp_step, stock path, main-KV
    # DCP decode layout, compact schedule and slot reads, 1 < rows a request <= 4): attend each request's rows as ONE
    # RPA sequence over the union of their owned pages (SparseCore metadata kernel qsa_union_meta) with a per-token
    # row mask, instead of one sequence per row. The value is the KV tile in tokens (a multiple of 128; his 640
    # measured fastest). NOT bitwise against the per-row path (same keys, the sums run in another order): needs the
    # stacked accuracy gate. Drafts, single-token programs, prefill and D14 keep their paths. Read at trace time, so
    # hashed (not in _TPU_COMPILE_ENV_IGNORED). 0 (default) off.
    "VLLM_TPU_QSA_DECODE_RPA_UNION_O1S": env_nonnegative_int("VLLM_TPU_QSA_DECODE_RPA_UNION_O1S", 0),
    # q38 PORT-BL UNION (Benliang Wang's VLLM_TPU_RPA_SCHEDULE_HBM, scoped to the stock path): the batched-RPA calls of
    # the O1s target attend (pooled_qsa_request_attend_dcp_step: its per-row decode attention and the union calls)
    # type their schedule arrays as HBM outputs of the schedule kernel. Untyped, XLA may place them in VMEM and then
    # copy each one to HBM before the attention kernel reads it. Same schedule values (placement only). His key types
    # every QSA decode RPA call; this one leaves the MTP draft, D14 and prefill programs alone (PORT-BL rule 3).
    # Changes the traced program, so hashed. Default off.
    "VLLM_TPU_RPA_SCHEDULE_HBM_O1S": env_bool("VLLM_TPU_RPA_SCHEDULE_HBM_O1S"),
    # q38 HIX-GLUE (HIX-LEAD-8 lane, 2026-11-24; vllm_torchtpu/hixglue.py): the HIX 16-row outside-K glue levers, each
    # default "0" and strict ("0" / "1"). Each changes the traced programs, so each is hashed (not in
    # _TPU_COMPILE_ENV_IGNORED). Each prints a per-rank bind line when it binds.
    # SINK_LAYOUT (SL): the MTP draft shared-selection buffers padded to a 128-lane width (row-major on device; no
    # full-buffer layout copies in the draft program); readers slice the logical width. Bitwise.
    "VLLM_TPU_HXG_SINK_LAYOUT": env_with_choices("VLLM_TPU_HXG_SINK_LAYOUT", "0", ["0", "1"]),
    # GDN_META (GM): the whole step's shared-GDN-metadata program (D07-b1) at every bucket, b16 included (overrides
    # VLLM_TPU_MK_GDN_META_DEDUP_MIN_ROWS; needs VLLM_TPU_MK_GDN_META_DEDUP=1), plus the GDN-SIDECAR accepted index in
    # one vectorised gather. Bitwise (integer ops).
    "VLLM_TPU_HXG_GDN_META": env_with_choices("VLLM_TPU_HXG_GDN_META", "0", ["0", "1"]),
    # PLE_ONE (PF0): the resident PLE lookup of a megakernel bucket in one gather (VLLM_TPU_PLE_LOOKUP_CHUNK_MIN_ROWS
    # treated as 0; needs VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS >= 64). Bitwise (per-token gather).
    "VLLM_TPU_HXG_PLE_ONE": env_with_choices("VLLM_TPU_HXG_PLE_ONE", "0", ["0", "1"]),
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
