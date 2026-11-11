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
    TPU_QWEN4_HC_DECODE_FOLD: bool = False
    TPU_QWEN4_HC_DECODE_TILE_ROWS: int = 0
    TPU_QWEN4_HC_DECODE_COMPARE: bool = False
    TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID: bool = False
    TPU_QWEN4_HC_DECODE_WIDE: int = 0
    TPU_QWEN4_HC_PREFILL_ODD_TILES: bool = False
    TPU_QWEN4_HC_PREFILL_FUSED_MIX: bool = False
    TPU_QWEN4_HC_PREFILL_PARTIAL_TILE: bool = False
    TPU_QWEN4_HC_PREFILL_NORM: str = "vpu"
    TPU_QWEN4_HC_PREFILL_COMPARE: bool = False
    VLLM_TPU_GDN_PREFILL_NO_ZERO_INIT: bool = False
    QWEN4_EXP_PREFILL_SP: bool = False
    QWEN4_EXP_PD_STAGING_BYTES: int = 8 << 30
    QWEN4_EXP_PD_STAGING_QUARANTINE_STEPS: int = 3
    QWEN4_EXP_PD_VERIFY_SPLIT_KEYS: bool = False
    QWEN4_EXP_PD_STAGING_LEASES: int = 0
    QWEN4_EXP_PD_DRAFT_KV: bool = False
    QWEN4_EXP_PD_DRAFT_KV_FORCE_GAP: float = 0.0
    QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS: bool = False
    QWEN4_EXP_PD_ROUTED_DST: str = ""
    QWEN4_EXP_PD_REPACK_SHAPE: str = ""
    QWEN4_EXP_PD_SYNC_REPACK: bool = False
    QWEN4_EXP_PD_SYNC_REPACK_AB_S: int = 0
    QWEN4_EXP_PD_DEFER_LOAD_SUBMIT: bool = False
    QWEN4_EXP_PD_LANDX_DEFER_REPACK: bool = False
    QWEN4_EXP_PD_EARLY_POLL: bool = False
    QWEN4_EXP_PD_SHARED_COMPRESSED: bool = False
    QWEN4_EXP_PD_BATCHED_ADMISSION: bool = False
    QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES: int = 0
    QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK: int = 0
    QWEN4_EXP_PD_FAST_ADMIT: bool = False
    QWEN4_EXP_PP_LOGICAL_BLOCKS_FACTOR: int = 1
    TPU_KV_CACHE_GROUP_SIZE_RULE: str = "gcd"
    QWEN4_EXP_PREFILL_SP_MIN_TOKENS: int = 1024
    QWEN4_EXP_PREFILL_SP_MOE: bool = False
    QWEN4_EXP_PREFILL_SP_SHARED_LOCAL: bool = False
    QWEN4_EXP_PREFILL_SP_PAD_HC: bool = False
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    TPU_MOE_DENSE_EXPERT_THRESHOLD: int = 0
    FUSED_MOE_DECODE_KERNEL: bool = False
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: int = 256
    FUSED_MOE_DECODE_KERNEL_FOLD: str = "off"
    FUSED_MOE_DECODE_EXACT_WIDTH: bool = False
    FUSED_MOE_DECODE_KERNEL_ROUTED_MIN_TOKENS: int = 0
    FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS: int = 0
    FUSED_MOE_DECODE_KERNEL_PADDING_MASK: bool = False
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
    VLLM_TPU_GDN_WINDOW_LOG: bool = False
    VLLM_TPU_L6_SEED_SPEC: str = ""
    VLLM_TPU_GDN_WINDOW_FACTORED: bool = False
    VLLM_TPU_GDN_LEAN_GLUE: bool = False
    VLLM_TPU_GDN_PREFILL_BF16_QKV_2D: bool = False
    VLLM_TPU_GDN_PREFILL_DENSE_CONV: bool = False
    VLLM_TPU_GDN_PREFILL_QKV_FROM_PROJ: bool = False
    TPU_ENABLE_GDN_DYNAMIC_TILING: bool = False
    VLLM_TPU_QSA_DECODE_QUERY_BLOCK: int = 0
    VLLM_TPU_QSA_TOPK_STATS: int = 0
    VLLM_TPU_QSA_SEL_DUMP: int = 0
    VLLM_TPU_QSA_SEL_DUMP_AT: str = "240:60:6"
    VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY: int = 0
    VLLM_TPU_QSA_DECODE_RPA_PT_PREFIX: bool = False
    VLLM_TPU_QSA_DECODE_RPA_UNION: int = 0
    VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK: int = 0
    VLLM_TPU_QSA_PREFILL_INDEXER_BQ: int = 128
    VLLM_TPU_QSA_PREFILL_INDEXER_BKV: int = 256
    VLLM_TPU_QSA_DECODE_KV_BLOCK: int = 1024
    VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK: int = 256
    VLLM_TPU_QSA_RING_SELECT: bool = False
    VLLM_TPU_QSA_RING_DMA_BATCH: int = 0
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE: bool = False
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS: int = 512
    VLLM_TPU_QSA_COMPRESSED_WRITE: str = ""
    VLLM_TPU_GDN_HOIST_READ_OFFSETS: bool = False
    VLLM_TPU_QSA_BF16_REDUCE_PRECISION: bool = False
    VLLM_TPU_QSA_INDEXER_HOIST_META: bool = False
    VLLM_TPU_QSA_INDEXER_MIXED_ONLY: bool = False
    VLLM_TPU_QSA_INDEXER_DENSE_STEP: bool = False
    VLLM_TPU_QSA_INDEXER_TAIL_SKIP: int = 0
    VLLM_TPU_QSA_DRAFT_SHARED_SELECTION: bool = False
    VLLM_TPU_QWEN4_NGRAM_WINDOW: str = ""
    VLLM_TPU_QWEN4_FUSED_PREP: bool = False
    VLLM_TPU_QWEN4_FUSED_PREP_CHECK: str = ""
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
    VLLM_TPU_QSA_PREFILL_UNION_GATHERED: str = ""
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS: int = 4
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH: float = 1.0
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY: str = "vmem"
    VLLM_TPU_QSA_PREFILL_COMPARE_CORE: bool = False
    VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP: int = 4096
    VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH: int = 8
    VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES: str = "sparsecore"
    VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS: int = 0
    VLLM_TPU_QSA_PREFILL_COMPARE_CORE_HELD: int = 0
    VLLM_TPU_QSA_MAIN_KV_DCP: int = 0
    VLLM_TPU_QSA_MTP_KV_DCP: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD_MIN_ROWS: int = 0
    VLLM_TPU_QSA_DCP_MERGE_SCORES: bool = False
    VLLM_TPU_RPA_SCHEDULE_HBM: bool = False
    VLLM_TPU_QSA_UNION_LEAN_GLUE: bool = False
    VLLM_TPU_QSA_UNION_CHAINED_CALLS: bool = False
    VLLM_TPU_QSA_DECODE_RPA_UNION_WIDE: int = 0
    VLLM_TPU_DECODE_ROWS_GLUE: bool = False
    VLLM_TPU_QSA_DCP_SORT_ROWS: int = 0
    VLLM_TPU_QSA_INDEXER_META_SELECT: bool = False
    VLLM_TPU_QSA_DCP_MERGE: str = ""
    VLLM_TPU_QSA_DCP_MERGE_CHECK: bool = False
    VLLM_TPU_QSA_DCP_LOCAL_QGATE: bool = False
    VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_LEAN_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_SLOT_READ: bool = False
    VLLM_TPU_QSA_DECODE_FAST_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_RPA_BKV: int = 256
    VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE: bool = False
    VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE: bool = False
    VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA: bool = False
    VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_DECODE_SC_OVERLAP: int = 0
    VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_DECODE_RPA_LANES: int = 8
    VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES: int = 0
    VLLM_TPU_QSA_FUSED_KV_WRITE: bool = False
    VLLM_TPU_QSA_PAIRED_MAIN_PAD_TILE: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS: int = 0
    VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP: bool = False
    VLLM_TPU_QSA_INDEXER_TOKEN_SHARD: bool = False
    VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS: int = 8
    VLLM_TPU_PLE_SKIP_IDLE_ROWS: bool = False
    VLLM_TPU_PLE_HISTORY_ROW_KERNEL: bool = False
    VLLM_TPU_PLE_WRDEFER: str = "0"
    VLLM_TPU_PLE_UNIFORM_COND: bool = False
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
    TPU_QWEN4_HC_UP_ROW_MAJOR: bool = False
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
    VLLM_TPU_OFFLOAD_TIER_CLASSES: bool = False
    VLLM_TPU_OFFLOAD_TIER_CLASS_RATIO: float = 4.7
    VLLM_TPU_OFFLOAD_TIER_KEY_ROWS: bool = False
    VLLM_TPU_OFFLOAD_TIER_CHECK: bool = False
    VLLM_TPU_OFFLOAD_TIER_EXCLUSIVE: bool = False
    VLLM_TPU_OFFLOAD_TIER_X2: bool = False
    VLLM_TPU_OFFLOAD_TIER_X2_MISS_TTL_MS: int = 1000
    VLLM_TPU_OFFLOAD_DIVERGENT_HITS: bool = False
    VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: bool = False
    VLLM_TPU_MAMBA_CHECKPOINT_RETENTION: bool = True
    VLLM_TPU_MAMBA_DECODE_CHECKPOINT: bool = True
    VLLM_TPU_MAMBA_PIN_HIT_CHECKPOINT: bool = False
    VLLM_TPU_PREFIX_LOOKUP_LOG: bool = False
    VLLM_TPU_KV_DECODE_FAST_PATH: bool = False
    VLLM_TPU_KV_LOAD_STARTS_PER_STEP: int = 0
    VLLM_TPU_PROMPT_TOKEN_ARRAY: bool = False
    VLLM_TPU_GC_AT_WAITS: bool = False
    VLLM_TPU_GC_FREEZE: bool = False
    VLLM_TPU_GC_UNTRACK_TOKEN_LISTS: bool = False
    VLLM_TPU_GC_GEN2_THRESHOLD: int = 0
    VLLM_TPU_GC_GEN2_PERIOD_S: float = 0.0
    VLLM_TPU_GC_CENSUS: str = ""
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
    VLLM_TPU_REASONING_COUNT_FAST: bool = False
    VLLM_TPU_FUSED_DRAFT_LOOP: bool = False
    VLLM_TPU_DRAFT_TOP_TOKENS: bool = False
    VLLM_TPU_WARM_SPEC_LOGITS_LENGTHS: bool = True
    VLLM_TPU_PLE_HOST_FAST_LOOKUP: bool = False
    VLLM_TPU_PLE_HOST_TIMING: bool = False
    VLLM_TPU_PLE_HOST_OVERLAP: bool = False
    VLLM_TPU_PLE_HOST_OVERLAP_CHECK: int = 0
    VLLM_TPU_FUSED_ASYNC_DRAFT_SEED: bool = True
    VLLM_TPU_DEFER_PREV_RESULTS_WAIT: bool = False
    VLLM_TPU_PP_LAST_ASYNC_OUTPUT: bool = False
    VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER: bool = False
    VLLM_TPU_LOG_POST_WARMUP_COMPILES: bool = False
    VLLM_TPU_CPU_PIN: str = "off"
    VLLM_TPU_WEIGHT_DIGEST_RANKS: str = ""
    VLLM_TPU_HOST_COPY_CHECK: bool = False
    VLLM_TPU_HOST_PHASES: int = 0
    VLLM_TPU_HOSTADMIT_PROBE: str = ""
    VLLM_TPU_HOSTADMIT_PYSPY: str = ""
    VLLM_TPU_HOSTADMIT_PYSPY_RANKS: str = "0"
    VLLM_TPU_HOSTADMIT_EVENTS: int = 4_000_000
    VLLM_TPU_LOG_STEP_BUCKETS: int = 0
    VLLM_TPU_SERIALIZE_COPY_WAIT: bool = False
    VLLM_TPU_PINNED_TOKEN_COPY: bool = False
    VLLM_TPU_SAMPLING_PARAMS_REUSE: bool = False
    VLLM_TPU_PY_SAMPLER_S: float = 0.0
    VLLM_TPU_STAGE_ENTRIES_CACHE: bool = False
    VLLM_TPU_STALL_WATCHDOG_S: float = 0.0
    VLLM_TPU_VERIFY_FUSED: bool = False
    VLLM_TPU_VERIFY_FUSED_CHECK: int = 0
    VLLM_TPU_VERIFY_FUSED_CHECK_FAULT: int = 0
    VLLM_TPU_VERIFY_SHARD_TOPK: int = 0
    VLLM_TPU_VERIFY_SHARD_CHECK: int = 0
    VLLM_TPU_DRAFT_SEED_FUSED: bool = False
    VLLM_TPU_CPU_PIN_RESERVE: int = 2
    VLLM_TPU_CPU_PROBE_S: float = 0.0
    VLLM_TPU_FUSED_READ_OFFSET_UPDATE: bool = True
    VLLM_TPU_GLUE_FOLD: bool = False
    VLLM_TPU_GLUE_FOLD_CHECK: int = 0
    VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE: bool = False
    VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK: bool = False
    VLLM_TPU_STAGED_WALK_CLASS_GATHER: bool = False
    VLLM_TPU_WARMED_SUBSTITUTE_SOURCE: bool = False
    VLLM_TPU_PROPOSE_DEVICE_CONSTANTS: bool = False
    VLLM_TPU_MTP_DRAFT_ROWS: bool = False
    VLLM_TPU_MTP_DRAFT_DECODE_ROWS: bool = False
    VLLM_TPU_MTP_DRAFT_SKIP_INDEXER_WRITES: bool = False
    VLLM_TPU_MTP_DRAFT_CHECK: int = 0
    VLLM_TPU_MTP_DRAFT_PADDING_MASK: bool = False
    TPU_SAMPLER_TOPK_CANDIDATES: int = 0
    MM_ENCODER_FRAME_CHUNK_PATCH_SIZE: int = 0
    VLLM_TPU_STATIC_POOLER: bool = True
    VLLM_TPU_POOLING_KV_CACHE_TOKENS: int = -1
    VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS: int = 8
    VLLM_TPU_MEGAKERNEL: str = ""
    VLLM_TPU_MK_OWN_OPERANDS: bool = True


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
    # Qwen4Exp staged decode with MTP: the pooled GDN kernel keeps the state
    # after verify window position 0 in checkpoint block 0 and the operands
    # of draft positions 1..K (gate, key, update vector, conv input) in that
    # block's spare rows, instead of one full state per window position. The
    # next step applies the first read-offset logged positions on load.
    # Refused with the megakernel and on any other leg.
    "VLLM_TPU_GDN_WINDOW_LOG": env_bool("VLLM_TPU_GDN_WINDOW_LOG"),
    # q38 SEED-SPEC (HCDEC-HOSTADM, ported by BLSTACK-SEED; runner/l6_seedspec.py,
    # core/l6_ab.py shim; host-only): unset / 0 off, 1 on; "ab:" is refused.
    # On, a request whose last token of the step sits within the previous
    # step's draft count past a GDN state-column edge no longer makes
    # _collect_mamba_state_seed_copies wait for the previous step's sampled
    # copy: in the canonical case (recorded column = the low side, all
    # accepted = low + 1, speculative checkpoint seed) the seed is staged as
    # for a crossing and the column is resolved at the next step from that
    # copy. Same programs and shapes; never part of the compile key.
    "VLLM_TPU_L6_SEED_SPEC": env_str("VLLM_TPU_L6_SEED_SPEC", ""),
    # With VLLM_TPU_GDN_WINDOW_LOG: the BATCHED GDN kernel runs the verify
    # window in factored form (one product of the window's keys and queries
    # with the stored state, the logged and window updates as vector work, one
    # update pass for the stored state). Not bit-identical to the recurrence.
    # Refused without VLLM_TPU_GDN_WINDOW_LOG.
    "VLLM_TPU_GDN_WINDOW_FACTORED": env_bool("VLLM_TPU_GDN_WINDOW_FACTORED"),
    # GDN v3 decode glue, bit-identical: in LOG mode (VLLM_TPU_GDN_WINDOW_LOG)
    # the kernels take no checkpoint-index operand (they only address
    # checkpoint 0, which is the state block), and on zero-filled buckets the
    # BATCHED kernel writes zeros to the rows past its sequences instead of
    # reading an XLA zero buffer.
    "VLLM_TPU_GDN_LEAN_GLUE": env_bool("VLLM_TPU_GDN_LEAN_GLUE"),
    # GDN v3 prefill: DMA bf16 qkv rows into PER_SEQ and cast in VMEM, not in HBM.
    "VLLM_TPU_GDN_PREFILL_BF16_QKV_2D": env_bool("VLLM_TPU_GDN_PREFILL_BF16_QKV_2D"),
    # With VLLM_TPU_GDN_PREFILL_BF16_QKV_2D: the PER_SEQ chunked tile unpacks
    # its bf16 rows token-dense ([rows, dim] f32) and runs the causal conv,
    # SiLU and head split there (a tap is a sublane shift of dense rows),
    # instead of on the one-row-per-vreg compact layout (Qwen3.5's
    # TPU_GDN_PREFILL_DENSE_CONV, translated to q38's pooled GDN v3).
    "VLLM_TPU_GDN_PREFILL_DENSE_CONV": env_bool("VLLM_TPU_GDN_PREFILL_DENSE_CONV"),
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
    # Debug only: every target QSA layer keeps its first N selected-token rows
    # in a buffer, and the runner saves them on a few sampled verify steps
    # (models/vllm/qwen4_exp/selection_dump.py). 0 (default) adds nothing to
    # the program.
    "VLLM_TPU_QSA_SEL_DUMP": env_nonnegative_int("VLLM_TPU_QSA_SEL_DUMP", 0),
    # Debug only, "T:N": every target decoder layer keeps digests of its mixer
    # input, mixer output, MoE output and layer output in a buffer, and the
    # runner logs them on every rank after each of the first N steps of exactly
    # T tokens (models/vllm/qwen4_exp/layer_digest.py). Unset (default) adds
    # nothing to the program.
    "VLLM_TPU_LAYER_DIGEST": lambda: os.getenv("VLLM_TPU_LAYER_DIGEST", ""),
    # With VLLM_TPU_QSA_SEL_DUMP: "t0:dt:n" seconds after the first verify step
    # with at least 16 requests: the first sample, the spacing, the count.
    "VLLM_TPU_QSA_SEL_DUMP_AT": lambda: os.getenv(
        "VLLM_TPU_QSA_SEL_DUMP_AT", "240:60:6"
    ),
    # Qwen4-exp QSA decode/verify attention on RPA with the compact schedule:
    # issue each lane-tile's page DMAs in groups of this many pages, each group
    # only when the tile has tokens in it, instead of issuing zero-size DMAs
    # past the tile's last page. Same DMAs with data, same results. 0 (default)
    # issues every page slot.
    "VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_SKIP_EMPTY", 0
    ),
    # q38 RPAD-HIGHROWS F4: the union-route RPA decode kernel takes its page
    # table in HBM instead of as a scalar prefetch, and copies into SMEM only
    # each sequence's used prefix (its KV tiles below kv_lens, whole tiles,
    # plus the table's first tile, which idle lanes index). The scalar prefetch
    # copies the whole worst-case [requests, 2080] table every call. Every page
    # id the fetch reads holds the value it holds today, so the same DMAs run
    # and the results are the same. Read inside the compiled forward, so it is
    # a compile key (not in _TPU_COMPILE_ENV_IGNORED). False (default) off.
    "VLLM_TPU_QSA_DECODE_RPA_PT_PREFIX": env_bool("VLLM_TPU_QSA_DECODE_RPA_PT_PREFIX"),
    # Qwen4-exp QSA verify steps under main-KV DCP (decode layout) with the
    # compact schedule and slot reads: attend each request's rows as one RPA
    # sequence over the union of their owned pages, with a per-token row mask,
    # instead of one sequence per row. The value is the KV tile in tokens (a
    # multiple of 128; 640 measured fastest). The union lists and masks come
    # from a SparseCore kernel (qsa_union_meta). Drafts, single-token steps,
    # prefill and every other program keep the per-row path. 0 (default) off.
    "VLLM_TPU_QSA_DECODE_RPA_UNION": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_UNION", 0
    ),
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
    # Qwen4-exp pooled QSA, compressed-key writer of decode/verify-sized steps
    # under VLLM_TPU_QSA_COMPRESSED_DCP. "scatter": write_compressed's per-pair
    # kernel at every row count (no run compaction, no 128-row tiles).
    # "inflight": compressed_write.write_pairs, one program with every read and
    # store DMA in flight. Any other value keeps the run / tiled / per-pair
    # choice by row count.
    "VLLM_TPU_QSA_COMPRESSED_WRITE": env_str("VLLM_TPU_QSA_COMPRESSED_WRITE", ""),
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
    "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION": env_bool(
        "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION"
    ),
    "VLLM_TPU_QSA_INDEXER_MIXED_ONLY": env_bool(
        "VLLM_TPU_QSA_INDEXER_MIXED_ONLY", default=False
    ),
    # Qwen4-exp decode indexer (StreamIdxTC mixed pass with per-token scores):
    # each key block is read with one sublane-strided load per 16 keys, a
    # 4-row query block is scored head-major on whole vregs, and the block's
    # page DMAs take their table rows without per-page carries or end clamps.
    # Same scores, bit for bit, as the per-token expression.
    "VLLM_TPU_QSA_INDEXER_DENSE_STEP": env_bool(
        "VLLM_TPU_QSA_INDEXER_DENSE_STEP", default=False
    ),
    # Qwen4-exp decode indexer: a KV block's page DMAs stop at the last group
    # of this many kernel pages that holds keys of its sequence (0: whole
    # blocks). The rest are masked; vLLM's null-block tail entries otherwise
    # make every rank read pool page 0.
    "VLLM_TPU_QSA_INDEXER_TAIL_SKIP": env_int(
        "VLLM_TPU_QSA_INDEXER_TAIL_SKIP", 0
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
    # The same pool-page table for the owned-group decode indexer of
    # VLLM_TPU_QSA_COMPRESSED_DCP: its table holds a rank's owned blocks (130
    # entries at 262144 tokens, 4 kernel pages each), so a 256-row verify step
    # scores in one query window instead of 128-row ones. Same pages, same
    # order.
    "VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE": env_bool(
        "VLLM_TPU_QSA_DCP_POOL_PAGE_TABLE", default=False
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
    # Qwen4-exp shared-stream decode indexer (VLLM_TPU_QSA_DECODE_QUERY_BLOCK):
    # keep the TensorCore busy while the SparseCore selects. The layer's
    # main-KV write runs after the indexer's scoring, the indexer's early-exit
    # lax.cond goes and its window loop is unrolled (XLA cannot schedule ops
    # of the enclosing computation inside either). 1: the main-KV write shares
    # an XLA scheduling group with the last window's SparseCore top-k, and
    # window g's top-k one with window g + 1's scoring. 2: the same program
    # without the group annotations; XLA's latency-hiding scheduler fills the
    # top-k windows. 0 (default): off. Same selections and cache bytes. Mode 1
    # compiles only where each QSA layer's step carries its own group ids; the
    # QSA layers of a forward share one cached jax_op export, so their equal
    # ids form one group that XLA rejects ("annotation groups with gaps").
    "VLLM_TPU_QSA_DECODE_SC_OVERLAP": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_SC_OVERLAP", 0
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
    # COMPARE_CORE: the QSA prefill attention of the union walk's eligible steps through
    # qsa_compare_core (16-row union groups, one flat tile list over all of them, 512-token tiles of
    # 4-token copies from the served pool; default off). It takes the union walk's static guard
    # (Q24/KV2/D256, FP8 pool of 128-aligned pages, no DCP, offset or LSE, at least 1024 rows) with
    # rows a multiple of 16, and falls back to the call the flag-off path makes when the longest row is
    # below VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT or a union group's union passes CAP 4-token groups
    # (a multiple of 128 whose 128-slot tiles are a power of two). DEPTH: tiles in flight (a power of
    # two). TABLES: the union list and row codes on the "sparsecore" (with the bitset there under
    # VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET) or by the XLA "reference". Not bitwise against the served
    # kernels (bf16 probabilities, union-order accumulation).
    "VLLM_TPU_QSA_PREFILL_COMPARE_CORE": env_bool("VLLM_TPU_QSA_PREFILL_COMPARE_CORE"),
    "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP": env_nonnegative_int("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", 4096),
    "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH": env_nonnegative_int("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH", 8),
    "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES": env_str("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES", "sparsecore"),
    # Debug only (untimed runs), N seconds: each pooled QSA layer folds its compare-core calls' union groups,
    # tiles, largest union and max relative error against the flag-off call on the same inputs (which then runs
    # too) into a device buffer, by range of the step's longest row; every N seconds the runner logs and zeroes
    # the buffers (QSA_CMP_STATS lines, compare_core_stats). 0 (default) adds nothing to the program.
    "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS": env_nonnegative_int("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS", 0),
    # HELD (tokens, 0 = off): the compare core's held read (qsa_compare_held): each sequence's cached KV of one KV
    # head held in VMEM for that head's pass, union slots gathered from it by vector loads; a step with a sequence
    # longer than this many tokens takes the flag-off call.
    "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_HELD": env_nonnegative_int("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_HELD", 0),
    # Main KV of the target QSA layers split over TP rank pairs that share a
    # KV head (2; 0/1 = off): each rank stores and attends half the tokens,
    # the pair's partial attention is LSE-combined.
    "VLLM_TPU_QSA_MAIN_KV_DCP": env_nonnegative_int("VLLM_TPU_QSA_MAIN_KV_DCP", 0),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: the MTP draft layers' main KV split the
    # same way (otherwise the draft keeps whole pages).
    "VLLM_TPU_QSA_MTP_KV_DCP": env_bool("VLLM_TPU_QSA_MTP_KV_DCP"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP and VLLM_TPU_QSA_MTP_KV_DCP: compressed keys
    # split the same way; each rank scores its owned groups and the pair merges
    # the local top-k lists into the exact global selection.
    "VLLM_TPU_QSA_COMPRESSED_DCP": env_bool("VLLM_TPU_QSA_COMPRESSED_DCP"),
    # With VLLM_TPU_QSA_COMPRESSED_DCP: each rank keeps only its own entries
    # of the merged selection (dcp_select.keep_mask on its local list against
    # the pair's boundary) instead of building the whole selection, and the
    # owned-group indexer's score buffer rows are padded to whole sublane
    # tiles. Same attended groups. The layer whose selection the MTP draft
    # reads (bind_shared_selection) keeps the whole selection.
    "VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE": env_bool(
        "VLLM_TPU_QSA_COMPRESSED_DCP_LOCAL_MERGE"
    ),
    # With VLLM_TPU_QSA_COMPRESSED_DCP: the local top-k lists travel in one
    # all-gather, the values' raw f32 bits and the group ids side by side in
    # one int32 buffer [T, 2k], in place of one all-gather each. Byte-exact.
    "VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER": env_bool(
        "VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER"
    ),
    # q38 BLSTACK-IDX. With VLLM_TPU_QSA_COMPRESSED_DCP and TP larger than
    # VLLM_TPU_QSA_MAIN_KV_DCP: the target layers' decode/verify-shaped graphs
    # split the owned-group indexer's rows over the TP / DCP groups as well
    # (ranks r and r + DCP own the same compressed groups). Each group scores
    # one contiguous token shard of the step with the decode indexer, one TP
    # all-gather brings every group's local lists, and every rank merges and
    # attends exactly as without it (the prefill token-sharded op of
    # VLLM_TPU_QSA_INDEXER_TOKEN_SHARD, routed on decode steps). Same
    # selection, same attended groups; per-rank scoring rows drop by TP / DCP.
    # Off: every rank scores every row of the step over its owned groups.
    "VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD": env_bool(
        "VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD", default=False
    ),
    # With VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD: graphs of fewer token rows
    # keep the unsharded owned-group indexer (0 = every decode-shaped graph).
    "VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD_MIN_ROWS": env_nonnegative_int(
        "VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD_MIN_ROWS", 0
    ),
    # With VLLM_TPU_QSA_COMPRESSED_DCP: the pair merge's boundary key comes
    # from the bits the merge top-k returns with its slots, instead of a
    # gather of the candidate keys at those slots. Same integers.
    "VLLM_TPU_QSA_DCP_MERGE_SCORES": env_bool("VLLM_TPU_QSA_DCP_MERGE_SCORES"),
    # QSA decode's batched-RPA calls type their schedule arrays as HBM outputs
    # of the schedule kernel. Untyped, XLA may place them in VMEM and then
    # copy each one to HBM before the attention kernel reads it.
    "VLLM_TPU_RPA_SCHEDULE_HBM": env_bool("VLLM_TPU_RPA_SCHEDULE_HBM"),
    # With VLLM_TPU_QSA_DECODE_RPA_UNION: the union RPA call's inputs and
    # outputs are built at the step's size when one call covers the step (up to
    # UNION_REQUESTS * max_query_len rows), so the f32 query, LSE buffer and
    # placeholder KV are not padded to that many rows; and the call reads the
    # selection words as [tiles * bkv / 128, 1, 128], the union kernel's flat
    # output without a relayout. Same results.
    "VLLM_TPU_QSA_UNION_LEAN_GLUE": env_bool("VLLM_TPU_QSA_UNION_LEAN_GLUE"),
    # With VLLM_TPU_QSA_DECODE_RPA_UNION: a step of several union RPA calls
    # runs each over the step's rows and threads one output and one LSE buffer
    # through them in the kernel's layout, in place of a padded query sliced
    # per call and each call's output scattered back. Same results.
    "VLLM_TPU_QSA_UNION_CHAINED_CALLS": env_bool("VLLM_TPU_QSA_UNION_CHAINED_CALLS"),
    # q38 DEC-STEP-3 (port of DECODE-QSA WIDE v2, 3772ac01ab): requests per union
    # RPA call for verify steps of more than UNION_REQUESTS * max_query_len rows.
    # 0 (default) keeps 32-request calls (5 per QSA layer at 448 / 480 rows); 64
    # runs ceil(requests / 64). Only 64: a [64, 2080] int32 page table is 532 KB
    # of the 1 MiB scalar memory (128 does not fit: DECODE-QSA aot2). Each
    # request stays its own RPA sequence, so every row's sums run in the same
    # order. Compile-hashed: it is read inside the model forward, behind
    # custom-op boundaries the FX graph fingerprint does not see, so a cache
    # restored across a flag flip would run the other programs.
    "VLLM_TPU_QSA_DECODE_RPA_UNION_WIDE": env_nonnegative_int(
        "VLLM_TPU_QSA_DECODE_RPA_UNION_WIDE", 0
    ),
    # q38 DEC-STEP-3: data-movement glue of large decode steps without the
    # layout copies / scatters the profile shows above 384 rows: the QSA
    # selected-token tail fill as selects (no scatter) and GDN's packed gate
    # input built by one concatenate (no stack + relayout). Same values.
    # Compile-hashed (read inside the model forward, as WIDE).
    "VLLM_TPU_DECODE_ROWS_GLUE": env_bool("VLLM_TPU_DECODE_ROWS_GLUE"),
    # q38 DEC-STEP-3: the compressed-DCP global selection's per-row sort runs
    # over row blocks of at most this many rows when a step has more (0 = one
    # sort over every row). Each row is sorted alone either way: same values.
    # Compile-hashed (read inside the model forward, as WIDE).
    "VLLM_TPU_QSA_DCP_SORT_ROWS": env_nonnegative_int("VLLM_TPU_QSA_DCP_SORT_ROWS", 0),
    # The indexer kernel's step metadata is selected (empty pass: zeros) rather
    # than computed in a conditional, whose small outputs XLA places in VMEM
    # and then copies to HBM before every StreamIdxTC call. Same integers.
    "VLLM_TPU_QSA_INDEXER_META_SELECT": env_bool("VLLM_TPU_QSA_INDEXER_META_SELECT"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: how the group merges its partial attention
    # outputs. Unset: all-gather every rank's f32 partial and LSE, each rank
    # combines its heads. "rs" / "rs_bf16": all-gather the LSE only, scale the
    # partial by its rank's weight and reduce-scatter it over the heads (f32 /
    # bf16 addends). "proj": each rank projects its share of the gated merge of
    # all the group's heads with the output projection's columns of those heads
    # (held per rank, 4x the o_proj weight under DCP 4), and the projection's TP
    # all-reduce adds the shares; the output collective goes away.
    "VLLM_TPU_QSA_DCP_MERGE": env_with_choices(
        "VLLM_TPU_QSA_DCP_MERGE", "", ["", "rs", "rs_bf16", "proj"]
    ),
    # Debug: the DCP layers keep the all-gather merge and also run the rs,
    # rs_bf16 and proj merges, recording their disagreement with it per layer;
    # every rank logs DCPMERGE_CHECK lines (dcp_merge_check.py). Overrides
    # VLLM_TPU_QSA_DCP_MERGE.
    "VLLM_TPU_QSA_DCP_MERGE_CHECK": env_bool("VLLM_TPU_QSA_DCP_MERGE_CHECK"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: each DCP layer projects the query and gate
    # of all its group's heads itself (the q_proj rows of the group's TP ranks,
    # held per rank: 4x the q_proj weight under DCP 4) instead of all-gathering
    # the group's q_proj outputs.
    "VLLM_TPU_QSA_DCP_LOCAL_QGATE": env_bool("VLLM_TPU_QSA_DCP_LOCAL_QGATE"),
    # Qwen4-exp pooled QSA decode/verify steps: write main KV, compressed
    # indexer keys and the raw ring/tags of a layer with one Pallas kernel
    # whose row DMAs are all in flight together, and read the ring with every
    # request's DMA outstanding. Prefill-sized steps keep the tiled writers.
    "VLLM_TPU_QSA_FUSED_KV_WRITE": env_bool(
        "VLLM_TPU_QSA_FUSED_KV_WRITE", default=False
    ),
    # Qwen4-exp paired main KV (QWEN4_EXP_PACK_MAIN_KV): steps whose row count
    # is not a multiple of 128 pad the rows to whole 128-row tiles (padded rows
    # invalid) and take the paired-slot writer kernel instead of the XLA
    # gather and scatter.
    "VLLM_TPU_QSA_PAIRED_MAIN_PAD_TILE": env_bool(
        "VLLM_TPU_QSA_PAIRED_MAIN_PAD_TILE", default=False
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
    # Qwen4-exp resident PLE table: tokens per gather in the raw embedding
    # lookup (a multiple of 8). 8 is the originally validated chunk; an
    # 8192-token prefill step then runs 1024 sequential gathers.
    "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS": env_nonnegative_int(
        "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS", 8
    ),
    # Qwen4-exp pooled PLE with speculative checkpoints: the history read moves
    # bytes only for request rows with tokens, and the checkpoint write fetches
    # values only for rows it writes. Skipped rows of the read are left
    # unwritten; every consumer masks them.
    "VLLM_TPU_PLE_SKIP_IDLE_ROWS": env_bool("VLLM_TPU_PLE_SKIP_IDLE_ROWS"),
    # Qwen4-exp PLE: every gather of the short-conv history (conv taps, final
    # window, checkpoint windows) runs as a Pallas DMA row copy with the
    # history pinned to HBM, so no XLA gather reads it from VMEM (XLA's TPU
    # gather halts some programs when MSA places that operand in VMEM).
    # Bitwise the XLA gathers.
    "VLLM_TPU_PLE_HISTORY_ROW_KERNEL": env_bool("VLLM_TPU_PLE_HISTORY_ROW_KERNEL"),
    # q38 PLE-WRDEFER (port of L1's PLE-WRDEFER onto the R form): K > 0 keeps K checkpoint
    # DMAs in flight in the PLE pool writer (kernels/qwen4_exp/pool.write_region_ring: one
    # grid step, a ring of K VMEM slots / DMA semaphores) instead of one grid step and one
    # waited DMA per row. Bitwise (unique positive ids; null ids skipped). 0 = stock writer.
    # VLLM_TPU_PLE_SKIP_IDLE_ROWS keeps the stock writer. Hashed (changes device programs).
    "VLLM_TPU_PLE_WRDEFER": env_with_choices(
        "VLLM_TPU_PLE_WRDEFER", "0", ["0", "1", "2", "4", "8"]
    ),
    # q38 PLE-UNIFORM-COND (layer U; port of L1's D07-b2 uniform_window_conv onto the R form):
    # a token-path PLE bucket of T % window == 0 chooses on the device (lax.cond) between the
    # uniform-window path (history for the T // window leading rows, static slices) and the
    # stock path; bitwise the stock op. Off = the stock op, no cond. Hashed (changes programs).
    "VLLM_TPU_PLE_UNIFORM_COND": env_bool("VLLM_TPU_PLE_UNIFORM_COND"),
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
    # gate, branch mean and the combine's injection gate) below 512 tokens as
    # one Pallas call, with the attn combine fused ahead of the mlp mix.
    "TPU_QWEN4_HC_DECODE_FUSED": env_bool("TPU_QWEN4_HC_DECODE_FUSED"),
    # qwen4_exp, with TPU_QWEN4_HC_DECODE_FUSED: below 512 tokens, run each
    # layer's mlp combine as the prologue of the next layer's fused attn mix
    # (and the last layer's ahead of the final mixer), as the attn combine runs
    # ahead of the mlp mix. Layers with a per-layer embedding keep the
    # separate combine, since the embedding is added between the two.
    "TPU_QWEN4_HC_DECODE_FOLD": env_bool("TPU_QWEN4_HC_DECODE_FOLD"),
    # qwen4_exp, with TPU_QWEN4_HC_DECODE_FUSED: the fused decode HC kernels
    # split a step of 128 or more rows into the fewest equal row tiles of at
    # most this many rows (a multiple of 16), not 128-row or smaller tiles
    # (192 rows: one tile instead of three of 64). Each tile streams the
    # layer's HC weights through the MXU again. 0 keeps the 128-row tiling.
    "TPU_QWEN4_HC_DECODE_TILE_ROWS": env_int("TPU_QWEN4_HC_DECODE_TILE_ROWS", 0),
    # qwen4_exp, with TPU_QWEN4_HC_DECODE_FUSED: steps of 16 to 256 rows (a
    # multiple of 16) run the fused decode HC mix as hc_decode_cmp.py, which
    # issues every row and weight copy at entry, takes the injection logits
    # from the down projection's MXU pass and writes the combined rows back
    # stream by stream. Other row counts keep hc_decode.py.
    "TPU_QWEN4_HC_DECODE_COMPARE": env_bool("TPU_QWEN4_HC_DECODE_COMPARE"),
    # q38 HC-HIGHROWS-2 (CMP-S), only with TPU_QWEN4_HC_DECODE_COMPARE: the cmp
    # kernel's mix term takes the served kernel's jax.nn.sigmoid of bf16(u)
    # instead of 1 + tanh(u / 2); the op names gain _cmps. Set without
    # TPU_QWEN4_HC_DECODE_COMPARE it is refused.
    "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID": env_bool(
        "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID"
    ),
    # q38 HC-HIGHROWS, qwen4_exp with TPU_QWEN4_HC_DECODE_FUSED: 1 or 2 runs
    # the fused decode HC mix at 160-511 rows (a multiple of 16) as
    # hc_decode_wide.py (row blocks copied by hand under a byte budget, the
    # combined and mixed rows written back block by block, f32 intermediates
    # bounded to 128-row chunks). 1 keeps hc_decode.py's math and order
    # (bitwise target); 2 takes hc_decode_cmp.py's math (injection logits on
    # the MXU, tanh sigmoid; ulp-level). 3 (q38 HC-HIGHROWS-2, W3) keeps mode
    # 1's math on a row-block software pipeline (bitwise target). 0 (default)
    # keeps hc_decode.py.
    # Exclusive with TPU_QWEN4_HC_DECODE_COMPARE.
    "TPU_QWEN4_HC_DECODE_WIDE": env_int("TPU_QWEN4_HC_DECODE_WIDE", 0),
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
    # qwen4_exp prefill, at the fused-mix row counts (whole or partial
    # hc_prefill_full_tile tiles, at least two): the HC mixes run as
    # hc_prefill_cmp.py (row tile r's combine, norm and down dot beside row
    # tile r - 1's up dots and mean; weights copied into VMEM from their
    # checkpoint layouts), and each layer's mlp combine runs as the prologue of
    # the next layer's attn mix inside a pipeline stage (as
    # TPU_QWEN4_HC_DECODE_FOLD at decode sizes). Layers with a per-layer
    # embedding and a stage's last layer keep the separate combine. Implies the
    # fused mix's tile counts.
    "TPU_QWEN4_HC_PREFILL_COMPARE": env_bool("TPU_QWEN4_HC_PREFILL_COMPARE"),
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
    # A staged Qwen PD decode with VLLM_TPU_QSA_COMPRESSED_DCP=1: after some
    # loads land (the first 8 without and the first 8 with a local prefix,
    # then one in 64), each rank reads its owned compressed pairs of three
    # layers back from the pool and the same pairs from the staging pages, and
    # logs both digests and the mismatched pairs (pd_staging.StagedConsumer.
    # verify_split_keys). With QWEN4_EXP_PACK_MAIN_KV=1 the same loads also
    # check both members' head-slot pairs of one paired main-KV owner against
    # the rank's head pair in their staging rows (verify_main_pairs). Each
    # check syncs the host with the device once.
    "QWEN4_EXP_PD_VERIFY_SPLIT_KEYS": env_bool("QWEN4_EXP_PD_VERIFY_SPLIT_KEYS"),
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
    # With QWEN4_EXP_PD_DRAFT_KV on a staged decode that splits compressed
    # keys over its main-KV DCP ranks (VLLM_TPU_QSA_COMPRESSED_DCP=1): admit
    # it (q38 BLSTACK-R). The landing planners split every compressed owner,
    # the F2 draft's included, and zero-fill its draft gap through the same
    # split pair tokens. Off keeps the refusal. Decode-side, host-only
    # (admission of the staging plan): in the compile-key ignore set.
    "QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS": env_bool("QWEN4_EXP_PD_DRAFT_KV_SPLIT_KEYS"),
    # "<decode tp>,<decode dcp>,<decode block>" on BOTH legs of a TP1 pipeline
    # producer -> staged TP>1 Qwen4Exp decode: the producer routes each decode
    # rank only its slice (main-KV DCP runs, its GDN heads; compressed keys,
    # ring and PLE to all) into a dense compact staging space per rank, and
    # the decode stages in 256-row pages (models/vllm/qwen4_exp/pd_route.py).
    # Empty (the default) broadcasts every page to every rank. Changes the
    # decode's staging carrier and repack program, so it joins the compile key.
    "QWEN4_EXP_PD_ROUTED_DST": env_str("QWEN4_EXP_PD_ROUTED_DST", ""),
    # "<main rows>,<copied rows>,<GDN pages>" (":" separates too) per staged
    # repack program on a staged Qwen4Exp decode (models/vllm/qwen4_exp/
    # pd_staging.py parse_repack_shape). Empty (the default) is
    # 16384,16384,4. When the GDN pages cover every GDN owner, a load's first
    # program lands all GDN state and its later programs run a repack built
    # without it. Changes the repack programs, so it joins the compile key.
    "QWEN4_EXP_PD_REPACK_SHAPE": env_str("QWEN4_EXP_PD_REPACK_SHAPE", ""),
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
    # Consumer: a step's Stage-3 load submit runs after its forward and
    # sampling dispatch instead of before the forward; the staged repack stays
    # before it (qwen4_exp pd_connector.submit_deferred_loads). Host-only.
    "QWEN4_EXP_PD_DEFER_LOAD_SUBMIT": env_bool("QWEN4_EXP_PD_DEFER_LOAD_SUBMIT"),
    # Consumer (q38 HCDEC-LANDX / BLSTACK-LANDX, LAND-OVERLAP): with
    # QWEN4_EXP_PD_SYNC_REPACK the scheduler holds each load that landed on
    # every rank out of vLLM's finished_recving for one step and names it in
    # qwen_landx_repack_req_ids; every rank runs that load's staged repack in
    # submit_deferred_loads, after the step's forward and sampling dispatch,
    # instead of in start_load_kv before the forward, and the scheduler
    # releases it at its next update, so the request joins one step later.
    # Same repack programs and bytes; host-side scheduling only. Refused at
    # build without SYNC_REPACK, with SYNC_REPACK_AB_S or SHARED_COMPRESSED.
    "QWEN4_EXP_PD_LANDX_DEFER_REPACK": env_bool("QWEN4_EXP_PD_LANDX_DEFER_REPACK"),
    # Consumer (q38 DEC-STEP-2): poll the KV connector (finished loads, worker
    # meta, stats, metadata clear) right after the step's forward and sampling
    # dispatch and the deferred submit, before the wait for the previous step's
    # sampled tokens, instead of after the drafter. On a decode host the poll
    # then runs while the device is still busy, off the serial host chain that
    # ends at the next forward's dispatch. Same programs, same inputs: the
    # poll's results still reach this step's output; a load that lands in the
    # moved window is reported one step later. Refused at the first step on a
    # KV producer and on a connector other than a Raiden PD consumer (the
    # producer's save fence and an offloading save need the drafter first).
    "QWEN4_EXP_PD_EARLY_POLL": env_bool("QWEN4_EXP_PD_EARLY_POLL"),
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
    # A staged Qwen4Exp decode's admission and landing host work, faster with
    # identical results (models/vllm/qwen4_exp/pd_staging.py): staging page
    # counts computed once per (tokens, prefix) and per cache kind, reserve
    # scanning the free map once in numpy, the repack's row indices built from
    # per-load arrays shared by every layer, each repack program's index lists
    # sent to the device in one upload, and the per-load "every rank staged
    # its submit" check as one int32 MIN all-reduce instead of an
    # all_gather_object. The same pages, rows, writes and decisions.
    "QWEN4_EXP_PD_FAST_ADMIT": env_bool("QWEN4_EXP_PD_FAST_ADMIT"),
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
    # The fused MoE decode kernel's TP all-reduce output is [tokens, hidden]:
    # each rank's all-gather sends only the columns of its reduce chunk below
    # hidden, instead of the whole chunk into a buffer padded to n_ranks lanes
    # of 128 that a slice then cuts back (a copy whenever the consumer is a
    # Pallas kernel).
    "FUSED_MOE_DECODE_EXACT_WIDTH": env_bool("FUSED_MOE_DECODE_EXACT_WIDTH"),
    # Fused MoE decode kernel calls of at least this many (padded) rows run
    # each active expert only over the row blocks that hold its routed rows
    # (routed mode, fused_moe_decode_kernel.py `_kernel`), bitwise the dense
    # output; smaller calls stay dense. Needs FUSED_MOE_DECODE_KERNEL_FOLD.
    # 0: dense at every row count.
    "FUSED_MOE_DECODE_KERNEL_ROUTED_MIN_TOKENS": env_nonnegative_int(
        "FUSED_MOE_DECODE_KERNEL_ROUTED_MIN_TOKENS", 0
    ),
    # Fused MoE decode kernel calls of at least this many (padded) rows run the
    # TP all-reduce as a reduction over the chip mate and the two link
    # directions (AR2D, fused_moe_decode_kernel.py `_ar2d_exchange`) instead of
    # the all-to-all + all-gather, when the layer has 8 ranks, hidden 2560 and
    # whole experts and the row count is a multiple of 64. Not bitwise the
    # all-to-all's output (other summation order, ar2d_reference.py); every
    # rank holds the same bits. 0: never.
    "FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS": env_nonnegative_int(
        "FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS", 0
    ),
    # The fused MoE decode kernel zeroes the target model's padded rows itself
    # (hidden states and route weights from the real-row count, read from the
    # TPU_MOE_SKIP_PADDED_TOKENS mask), so the decoder layer's mask of the MoE
    # input and zero_routing_weights_for_padding are skipped for armed layers.
    # Bitwise the same output. Needs TPU_MOE_SKIP_PADDED_TOKENS, DP 1 and PCP 1;
    # the drafter's layers keep both masks.
    "FUSED_MOE_DECODE_KERNEL_PADDING_MASK": env_bool(
        "FUSED_MOE_DECODE_KERNEL_PADDING_MASK"
    ),
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
    # qwen4_exp: materialize each hyper-connection's input_mix_weight_up
    # (bf16 [hc * hidden, lowrank]) with a row-major ({1,0}) device layout at
    # load, the layout its Pallas consumers (hc_up_epilogue, hc_full, the
    # fused decode and prefill mixes) take. The default layout puts the
    # lowrank dim major, so every compiled step copies the weight before each
    # such call. Costs the lowrank dim's lane padding (320 -> 384 for Flash).
    "TPU_QWEN4_HC_UP_ROW_MAJOR": env_bool("TPU_QWEN4_HC_UP_ROW_MAJOR"),
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
    # Pipeline chunk scheduler: the head of the waiting queue is ordered by
    # uncached prompt tokens, fewest first, less TPU_PP_WARM_FIRST_AGING tokens
    # for every second a request has waited, then by arrival (core/
    # pp_chunk_scheduler.py _order_waiting). Off: first come, first served.
    # Host-only.
    "TPU_PP_WARM_FIRST": env_bool("TPU_PP_WARM_FIRST", default=False),
    # Aging rate of TPU_PP_WARM_FIRST in prompt tokens per second waited. A
    # request with U uncached tokens is passed only by requests that arrive
    # less than U / rate seconds after it. 0: no aging. Host-only.
    "TPU_PP_WARM_FIRST_AGING": env_float("TPU_PP_WARM_FIRST_AGING", 10000.0),
    # With TPU_PP_WARM_FIRST: a prompt's tokens that the raiden offload
    # connector's host pool holds count as cached too, as a tier load computes
    # nothing (offload/raiden_connector.py host_tier_hit_tokens). Off: only the
    # local prefix cache counts. Host-only.
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
    # Host tier in size classes: one Raiden store for each set of pool regions
    # a group of offloaded caches uses, so an entry moves only the rows its
    # group occupies (offload/tier_classes.py). Local host DRAM tier only.
    # Host-only.
    "VLLM_TPU_OFFLOAD_TIER_CLASSES": env_bool(
        "VLLM_TPU_OFFLOAD_TIER_CLASSES", default=False
    ),
    # Token-blocks held per checkpoint when the tier classes split the byte
    # budget: attention classes hold this many entries per state-class entry.
    # Host-only.
    "VLLM_TPU_OFFLOAD_TIER_CLASS_RATIO": env_float(
        "VLLM_TPU_OFFLOAD_TIER_CLASS_RATIO", 4.7
    ),
    # With the tier classes: the compressed-key class keeps only the keys'
    # leading bytes of each page, registering each region as a view of
    # finer rows. Needs a Raiden build that accepts that view. Host-only.
    "VLLM_TPU_OFFLOAD_TIER_KEY_ROWS": env_bool(
        "VLLM_TPU_OFFLOAD_TIER_KEY_ROWS", default=False
    ),
    # With the tier classes, smokes only: every worker hashes the device rows
    # its class owns of each entry a store saved, and compares after a load
    # of that key; counts go on the offload-tier telemetry line. Reads rows
    # back to the host on every store and load. Host-only.
    "VLLM_TPU_OFFLOAD_TIER_CHECK": env_bool(
        "VLLM_TPU_OFFLOAD_TIER_CHECK", default=False
    ),
    # With QWEN4_EXP_HOST_TIER_POLICY=victim: exclusive restore. Once a load from
    # the host tier lands, its host copy is evicted (KVCacheStore.evict, which
    # erases only unpinned host entries), so the tier never holds a block that
    # HBM holds; HBM's later eviction stores it again. Needs a Raiden build
    # with KVCacheStore.evict. Host-only.
    "VLLM_TPU_OFFLOAD_TIER_EXCLUSIVE": env_bool(
        "VLLM_TPU_OFFLOAD_TIER_EXCLUSIVE", default=False
    ),
    # With VLLM_TPU_OFFLOAD_TIER_CLASSES: a second host tier in another host's DRAM
    # (TIER-X2). Each class store joins the global registry
    # (kv_connector_extra_config global_registry_address / store_server_ip /
    # kv_pool_group), Raiden's evict sweep (RAIDEN_ENABLE_STORE_MONITOR /
    # RAIDEN_ENABLE_EVICT_SWEEP) demotes cold host blocks to the store nodes of the
    # class's pool group, and a host-tier miss reads them back from there with
    # read_remote. Classes are split into shards of equal per-worker geometry, the
    # only form Raiden's transfer spec describes. The bytes a load lands are the
    # bytes the block was saved with (cache placement only). Host-only.
    "VLLM_TPU_OFFLOAD_TIER_X2": env_bool("VLLM_TPU_OFFLOAD_TIER_X2", default=False),
    # TIER-X2: how long a key that missed both the host tier and the registry is
    # answered as a miss without asking the registry again (milliseconds; 0 asks
    # every time). A key stored or demoted meanwhile is at worst found late.
    "VLLM_TPU_OFFLOAD_TIER_X2_MISS_TTL_MS": env_int(
        "VLLM_TPU_OFFLOAD_TIER_X2_MISS_TTL_MS", 1000
    ),
    # Prefill with the host tier (TPURaidenOffloadingConnector) on a hybrid model:
    # report "divergent local hybrid hits", so the local prefix hit is the
    # attention groups' cached run (core/divergent_hybrid_hits.py) even when HBM
    # keeps no recurrent-state checkpoint at its end, and the host tier's load
    # supplies the state at the boundary it restores. Without it a hot shared
    # head that lives only in HBM (the victim policy never stores a block HBM
    # keeps) cuts every tier hit to 0: the local hit falls to the last HBM
    # checkpoint, often 0, and the tier lookup starts there and misses block 0
    # (hole_diag_1120.md). A pure PD producer child of the multi connector
    # never loads and does not veto. Same bytes as a tier hit; scheduler-only.
    "VLLM_TPU_OFFLOAD_DIVERGENT_HITS": env_bool(
        "VLLM_TPU_OFFLOAD_DIVERGENT_HITS", default=False
    ),
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
    # Scheduler: the Mamba checkpoint a request resumed from stays pinned with
    # its own checkpoints until the request is freed, then leaves after its KV
    # (core/mamba_checkpoint_retention.py). Host-only.
    "VLLM_TPU_MAMBA_PIN_HIT_CHECKPOINT": env_bool(
        "VLLM_TPU_MAMBA_PIN_HIT_CHECKPOINT", default=False
    ),
    # Offload connector: one line per request at its first lookup with the HBM
    # and tier hits and what each level held of its blocks
    # (offload/lookup_log.py). Host-only.
    "VLLM_TPU_PREFIX_LOOKUP_LOG": env_bool("VLLM_TPU_PREFIX_LOOKUP_LOG", default=False),
    # Scheduler: a steady decode step skips the per-group KV cache
    # bookkeeping when no group's state can change; the same blocks are
    # allocated, cached and freed (core/kv_decode_fast_path.py). Host-only.
    "VLLM_TPU_KV_DECODE_FAST_PATH": env_bool(
        "VLLM_TPU_KV_DECODE_FAST_PATH", default=False
    ),
    # Scheduler: at most this many remote KV loads start in one schedule
    # pass of a KV-consumer engine; the other waiting requests keep their
    # queue order for the next pass (core/kv_load_pacing.py). 0 = no cap.
    # Host-only.
    "VLLM_TPU_KV_LOAD_STARTS_PER_STEP": env_nonnegative_int(
        "VLLM_TPU_KV_LOAD_STARTS_PER_STEP", 0
    ),
    # Engine core: a new request's prompt reaches the workers as an int32
    # array instead of a list (core/prompt_token_array.py). Host-only.
    "VLLM_TPU_PROMPT_TOKEN_ARRAY": env_bool(
        "VLLM_TPU_PROMPT_TOKEN_ARRAY", default=False
    ),
    # Engine core and workers: due garbage collections run while the main
    # thread waits on a message queue or a token copy (runner/gc_at_waits.py).
    # Host-only.
    "VLLM_TPU_GC_AT_WAITS": env_bool("VLLM_TPU_GC_AT_WAITS", default=False),
    # Engine core: gc.freeze() once the engine core is built (after warmup and
    # KV-cache init), before the busy loop (runner/gc_at_waits.py). Host-only.
    "VLLM_TPU_GC_FREEZE": env_bool("VLLM_TPU_GC_FREEZE", default=False),
    # Engine core: a request's prompt and all-token lists leave the collector
    # (they hold only ints) when the Request is built. Host-only.
    "VLLM_TPU_GC_UNTRACK_TOKEN_LISTS": env_bool(
        "VLLM_TPU_GC_UNTRACK_TOKEN_LISTS", default=False
    ),
    # Engine core: generation-2 threshold set with the freeze; 0 keeps
    # CPython's. Host-only.
    "VLLM_TPU_GC_GEN2_THRESHOLD": env_nonnegative_int(
        "VLLM_TPU_GC_GEN2_THRESHOLD", 0
    ),
    # Engine core: seconds between generation-2 collections run at a
    # message-queue wait; 0 leaves generation 2 to CPython. Host-only.
    "VLLM_TPU_GC_GEN2_PERIOD_S": env_float("VLLM_TPU_GC_GEN2_PERIOD_S", 0.0),
    # Engine core: held-request counts separated by colons or commas
    # ("0:60:90:120"); the first wait at or above each logs a census of the
    # tracked heap and the length of one generation-2 pass. Also logs every
    # generation-2 pass. "" is off. Host-only.
    "VLLM_TPU_GC_CENSUS": env_str("VLLM_TPU_GC_CENSUS", ""),
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
    # Count a finished stream's reasoning tokens in the API server with the
    # parser engine's feed alone, without rebuilding its deltas and tool calls
    # (reasoning_count.py). Same count.
    "VLLM_TPU_REASONING_COUNT_FAST": env_bool("VLLM_TPU_REASONING_COUNT_FAST"),
    # Run an MTP draft's loop steps 1..K-1 as one compiled program
    # (spec_decode/mtp_fused_loop.py). Off by default: one program per step.
    # TODO: support the two cases that still take the per-step loop:
    #   - GLM-5.2 / Gemma-4 MTP: IndexShare's per-chunk top-k restore, and a
    #     KV-sharing draft's fixed positions and seq_lens.
    #   - EP-DP lockstep: run_dp_dummy_draft only replays the per-step
    #     collective trace.
    "VLLM_TPU_FUSED_DRAFT_LOOP": env_bool("VLLM_TPU_FUSED_DRAFT_LOOP"),
    # The greedy draft's tokens come from the vocabulary-parallel argmax
    # (runner/greedy_top_tokens.py): each rank argmaxes its LM-head slice and
    # only (value, id) pairs are gathered, never the draft logits. Same tokens
    # as compute_logits(...).argmax(-1).
    "VLLM_TPU_DRAFT_TOP_TOKENS": env_bool("VLLM_TPU_DRAFT_TOP_TOKENS"),
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
    # Comma-separated global ranks (or "all") that log a digest of every loaded
    # parameter and buffer at the end of model loading (runner/weight_digest.py).
    "VLLM_TPU_WEIGHT_DIGEST_RANKS": env_str("VLLM_TPU_WEIGHT_DIGEST_RANKS", ""),
    # Compare every parameter and buffer built by a host-to-device copy during
    # model loading with the host bytes, on every rank (runner/host_copy_check.py).
    "VLLM_TPU_HOST_COPY_CHECK": env_bool("VLLM_TPU_HOST_COPY_CHECK", default=False),
    # Log each rank's per-step host time by runner phase every N steps
    # (0 = off); runner/host_phases.py.
    "VLLM_TPU_HOST_PHASES": env_nonnegative_int("VLLM_TPU_HOST_PHASES", 0),
    # Host timeline of the admission path in profile windows, as
    # "<start_s>:<duration_s>[+...]" after the first Stage-3 load each process
    # sees ("" = off); runner/host_admit_probe.py.
    "VLLM_TPU_HOSTADMIT_PROBE": env_str("VLLM_TPU_HOSTADMIT_PROBE", ""),
    # py-spy binary run on the engine core and the rank-0 worker in those
    # windows ("" = timers only).
    "VLLM_TPU_HOSTADMIT_PYSPY": env_str("VLLM_TPU_HOSTADMIT_PYSPY", ""),
    # Worker ranks py-spy samples besides the engine core ("0+3", or "all").
    "VLLM_TPU_HOSTADMIT_PYSPY_RANKS": env_str("VLLM_TPU_HOSTADMIT_PYSPY_RANKS", "0"),
    # Events each process's probe buffer holds from the first Stage-3 load
    # until the last window ends (29 bytes an event).
    "VLLM_TPU_HOSTADMIT_EVENTS": env_nonnegative_int("VLLM_TPU_HOSTADMIT_EVENTS", 4_000_000),
    # Log each rank's cumulative step counts per padded token bucket every N
    # steps (0 = off); runner/step_buckets.py.
    "VLLM_TPU_LOG_STEP_BUCKETS": env_nonnegative_int("VLLM_TPU_LOG_STEP_BUCKETS", 0),
    # Let only one thread synchronize a sampled-token copy at a time (the
    # async-output thread and the deferred previous-results wait share one).
    "VLLM_TPU_SERIALIZE_COPY_WAIT": env_bool("VLLM_TPU_SERIALIZE_COPY_WAIT"),
    # Copy sampled tokens into reused pinned host buffers instead of a fresh
    # pageable tensor every step (runner/tpu_runner_async_output.py).
    "VLLM_TPU_PINNED_TOKEN_COPY": env_bool("VLLM_TPU_PINNED_TOKEN_COPY"),
    # Reuse the padded per-request temperature / top-k / top-p uploads while
    # their values repeat (tpu_runner._build_padded_sampling_params).
    "VLLM_TPU_SAMPLING_PARAMS_REUSE": env_bool("VLLM_TPU_SAMPLING_PARAMS_REUSE"),
    # Sample the step thread's Python stack for N s (runner/py_sampler.py).
    "VLLM_TPU_PY_SAMPLER_S": env_float("VLLM_TPU_PY_SAMPLER_S", 0.0),
    # Keep the per-group (block table, width) list the staged metadata walk
    # builds, instead of re-walking every KV cache group each step.
    "VLLM_TPU_STAGE_ENTRIES_CACHE": env_bool("VLLM_TPU_STAGE_ENTRIES_CACHE"),
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
    # Spec verify on vocabulary-shard candidates (runner/vocab_shard_verify.py):
    # each rank takes the top N of its LM-head slice and only those are
    # gathered. Steps whose requests all sample greedily or with
    # 1 <= top_k < N, ask for no logprobs and carry no grammar take it; other
    # steps gather the full logits. 0 = off.
    "VLLM_TPU_VERIFY_SHARD_TOPK": env_nonnegative_int("VLLM_TPU_VERIFY_SHARD_TOPK", 0),
    # With VLLM_TPU_VERIFY_SHARD_TOPK: for the first N candidate chunks also
    # gather the full logits and compare, per row, the ids top-k/top-p keep and
    # the argmax id with the candidates' (logged per mismatch and in total).
    "VLLM_TPU_VERIFY_SHARD_CHECK": env_nonnegative_int("VLLM_TPU_VERIFY_SHARD_CHECK", 0),
    # Async draft seed + input shift/scatter + gather pad as one program
    # (eagle3._async_seed_inputs_impl).
    "VLLM_TPU_DRAFT_SEED_FUSED": env_bool("VLLM_TPU_DRAFT_SEED_FUSED"),
    "VLLM_TPU_CPU_PIN_RESERVE": env_nonnegative_int("VLLM_TPU_CPU_PIN_RESERVE", 2),
    # Log each worker's thread CPU placement every N seconds (0 = off).
    "VLLM_TPU_CPU_PROBE_S": env_float("VLLM_TPU_CPU_PROBE_S", 0.0),
    # Async scheduling: wait for the previous step's sampled-token copy after
    # launching this step's draft rather than before it (sample_tokens).
    "VLLM_TPU_DEFER_PREV_RESULTS_WAIT": env_bool(
        "VLLM_TPU_DEFER_PREV_RESULTS_WAIT", default=False
    ),
    # With async scheduling off, the last pipeline stage returns its sampled
    # tokens unwaited and replies from the worker's output thread, so its next
    # step's host work overlaps the forward (runner/pp_last_async_output.py).
    "VLLM_TPU_PP_LAST_ASYNC_OUTPUT": env_bool(
        "VLLM_TPU_PP_LAST_ASYNC_OUTPUT", default=False
    ),
    # q38 PPARF: PP last async output also admits a prefill-only MTP producer;
    # its drafts stay on device and the copy gate is re-armed behind them
    # (runner/pp_last_async_drafter.py). Host only.
    "VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER": env_bool(
        "VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER", default=False
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
    # q38 DEC-STEP-3: the async spec step's integer glue (post-verify read
    # offsets + rejected count + bonus; the draft source assembly + pad; the
    # cross-step subtract; the substitution + draft extraction) as four
    # compiled programs, op for op the stock ones (tpu_runner._glue_*). The
    # uniform draws stay eager. Single-chunk async spec steps only.
    "VLLM_TPU_GLUE_FOLD": env_bool("VLLM_TPU_GLUE_FOLD", default=False),
    # With VLLM_TPU_GLUE_FOLD: run the stock ops on copies for the first N
    # folded steps and log the mismatch counts once (must be 0).
    "VLLM_TPU_GLUE_FOLD_CHECK": env_nonnegative_int("VLLM_TPU_GLUE_FOLD_CHECK", 0),
    # Metadata walk: reuse the previous step's staged block-table upload when
    # its geometry and contents are unchanged (layers/core/attention_metadata.py
    # stage_block_table_uploads).
    "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE": env_bool(
        "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE", default=False
    ),
    # Host uploads issued while the runner's input staging fence is pending
    # (after the forward was dispatched, e.g. the drafter's metadata walk) copy
    # from a private host tensor instead of the shared staging buffer: a
    # non-blocking upload reads its host memory until the transfer completes,
    # and nothing orders such an upload against the step's own upload from the
    # same buffer or against the next step's rewrite of it. Covers the
    # block-table staging scratch and mamba_state_indices_cpu. Host-only.
    "VLLM_TPU_UNFENCED_UPLOAD_PRIVATE": env_bool(
        "VLLM_TPU_UNFENCED_UPLOAD_PRIVATE", default=False
    ),
    # Debug: every staging walk reads the previous block-table upload back and
    # counts entries that differ from the host copy it was made from, and counts
    # walks that packed while the input staging fence was pending; one log line
    # every N checks (0: off). Each check waits for the previous upload.
    # Host-only.
    "VLLM_TPU_BLOCK_TABLE_STAGE_CHECK": env_nonnegative_int(
        "VLLM_TPU_BLOCK_TABLE_STAGE_CHECK", 0
    ),
    # Qwen4Exp n-gram token store without VLLM_TPU_QWEN4_NGRAM_WINDOW=1 (the
    # prefill engine): upload input_batch.token_ids_cpu_tensor from one of two
    # host copies used in turn instead of the tensor itself, which the next
    # step's _update_states rewrites before the input staging fence orders the
    # upload (models/vllm/qwen4_exp/inputs.py begin_step). Host-only.
    "VLLM_TPU_TOKEN_STORE_UPLOAD_COPY": env_bool(
        "VLLM_TPU_TOKEN_STORE_UPLOAD_COPY", default=False
    ),
    # Debug: for the first N token-store uploads, keep a host copy and read the
    # device copy back at the next step, counting entries that differ; logs
    # each of the first 20 bad uploads and a summary after N checks (0: off).
    # Host-only.
    "VLLM_TPU_TOKEN_STORE_CHECK": env_nonnegative_int(
        "VLLM_TPU_TOKEN_STORE_CHECK", 0
    ),
    # Host-path pieces (q38f-int: default off, toggled per arm).
    # Record stage-timing device events only when a KV connector consumes them.
    "VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK": env_bool(
        "VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK", default=False
    ),
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
    # With VLLM_TPU_MTP_DRAFT_ROWS: draft step 0 of a step with at most
    # 16 * ceil(T / 64) requests (T the token bucket; a pure K=3 decode step
    # has at most T / 4) runs a second compiled program that evaluates that
    # many rows instead of the chunk's request capacity. Steps with more
    # requests keep the capacity-row program.
    "VLLM_TPU_MTP_DRAFT_DECODE_ROWS": env_bool("VLLM_TPU_MTP_DRAFT_DECODE_ROWS"),
    # A Qwen4Exp MTP draft attending the target's selection
    # (VLLM_TPU_QSA_DRAFT_SHARED_SELECTION) under VLLM_TPU_QSA_MTP_KV_DCP writes
    # only its main KV: its compressed keys, ring rows and ring tags feed only
    # its own indexer, which never runs.
    "VLLM_TPU_MTP_DRAFT_SKIP_INDEXER_WRITES": env_bool(
        "VLLM_TPU_MTP_DRAFT_SKIP_INDEXER_WRITES"
    ),
    # Check mode for the two flags above: the first N spec proposes also run the
    # draft through reference programs (capacity rows, indexer writes on) and
    # count requests whose K draft tokens differ, logged as MTP_DRAFT_CHECK lines.
    "VLLM_TPU_MTP_DRAFT_CHECK": env_nonnegative_int("VLLM_TPU_MTP_DRAFT_CHECK", 0),
    # With TPU_MOE_SKIP_PADDED_TOKENS: the MTP draft's row-evaluated forwards
    # (step 0 at draft rows, the fused loop) write the MoE padding mask for
    # their own rows first. Otherwise they read the target step's mask, and a
    # padded draft row below the target's valid count keeps router weight: a
    # zero row ties to the lowest expert ids, all on EP rank 0.
    "VLLM_TPU_MTP_DRAFT_PADDING_MASK": env_bool("VLLM_TPU_MTP_DRAFT_PADDING_MASK"),
    # Non-greedy sampling filters the vocab to this many 128-logit blocks,
    # ranked by block max, before running top-k (and any subsequent top-p).
    # 0 searches the full vocab. Used only when every active request has
    # 1 <= top_k <= value // 2 and no grammar mask applies.
    "TPU_SAMPLER_TOPK_CANDIDATES": env_nonnegative_int(
        "TPU_SAMPLER_TOPK_CANDIDATES", 0
    ),
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
