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
    USE_MOE_SPARSE_CORE: bool = True
    ONEHOT_MOE_PERMUTE_THRESHOLD: int | None = None
    TPU_MOE_SEQUENCE_PARALLEL: bool = False
    TPU_QWEN4_HC_DECODE_FUSED: bool = False
    QWEN4_EXP_PREFILL_SP: bool = False
    QWEN4_EXP_PREFILL_SP_MIN_TOKENS: int = 1024
    QWEN4_EXP_PREFILL_SP_MOE: bool = False
    VLLM_TPU_MK_HOST_PLE: bool = False
    VLLM_TPU_MK_WHOLE_BATCHED_INPUTS: bool = False
    VLLM_TPU_MK_PLE_TRIM: bool = False
    QWEN4_EXP_PREFILL_SP_SHARED_LOCAL: bool = False
    QWEN4_EXP_PREFILL_SP_PAD_HC: bool = False
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    TPU_MOE_DENSE_EXPERT_THRESHOLD: int = 0
    FUSED_MOE_DECODE_KERNEL: bool = False
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: int = 256
    FUSED_MOE_DECODE_KERNEL_FOLD: str = "off"
    TPU_MOE_SINGLE_SORT: bool = True
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
    VLLM_TPU_GDN_HOIST_METADATA: bool = False
    TPU_ENABLE_GDN_DYNAMIC_TILING: bool = False
    TPU_GDN_CHUNK_PAIR: bool = False
    TPU_GDN_RAW_QKV: bool = False
    TPU_GDN_RAW_BA: bool = False
    TPU_GDN_KERNEL_ZERO_OUT: bool = False
    VLLM_TPU_QSA_DECODE_QUERY_BLOCK: int = 0
    VLLM_TPU_QSA_TOPK_STATS: int = 0
    VLLM_TPU_QSA_DECODE_KV_BLOCK: int = 1024
    VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK: int = 256
    VLLM_TPU_QSA_RING_SELECT: bool = False
    VLLM_TPU_QSA_RING_DMA_BATCH: int = 0
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE: bool = False
    VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS: int = 512
    VLLM_TPU_QSA_MAIN_RUN_WRITE: bool = False
    VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS: int = 512
    VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE: int = 512
    VLLM_TPU_GDN_HOIST_READ_OFFSETS: bool = False
    VLLM_TPU_QSA_BF16_REDUCE_PRECISION: bool = False
    VLLM_TPU_QSA_INDEXER_HOIST_META: bool = False
    VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES: bool = False
    VLLM_TPU_QSA_INDEXER_MIXED_ONLY: bool = False
    VLLM_TPU_QSA_DRAFT_SHARED_SELECTION: bool = False
    VLLM_TPU_QWEN4_NGRAM_WINDOW: str = ""
    VLLM_TPU_QWEN4_FUSED_PREP: bool = False
    VLLM_TPU_QWEN4_FUSED_PREP_CHECK: str = ""
    VLLM_TPU_QWEN4_FUSED_PREP_FUNCTIONAL: bool = False
    VLLM_TPU_QSA_FUSED_SPARSE_DECODE: bool = False
    VLLM_TPU_QSA_PREFIX_SLICE: bool = False
    VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT: bool = False
    VLLM_TPU_QSA_PREFILL_BITSET_KERNEL: bool = True
    VLLM_TPU_QSA_PREFILL_BITSET_RANGED: bool = False
    VLLM_TPU_QSA_PREFILL_BF16_MXU: bool = False
    VLLM_TPU_QSA_DCP_FINE_PREFIX: bool = False
    VLLM_TPU_QSA_DCP_FINE_PREFIX_SEQ_TAIL: bool = False
    VLLM_TPU_QSA_DCP_FINE_PREFIX_CONNECTOR: bool = False
    VLLM_TPU_QSA_PREFILL_LEAN: bool = False
    VLLM_TPU_QSA_PREFILL_VPU: str = ""
    VLLM_TPU_QSA_PREFILL_PAGE_COPIES: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_WALK: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT: int = 32768
    VLLM_TPU_QSA_PREFILL_UNION_CAP: int = 12288
    VLLM_TPU_QSA_PREFILL_UNION_KV_TILE: int = 1024
    VLLM_TPU_QSA_PREFILL_UNION_LISTED: str = ""
    VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_ALPHA: float = 0.0
    VLLM_TPU_QSA_PREFILL_UNION_QK16: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET: bool = False
    VLLM_TPU_QSA_PREFILL_UNION_GATHERED: str = ""
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS: int = 4
    VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH: float = 1.0
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS: int = 0
    VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY: str = "vmem"
    VLLM_TPU_QSA_PREFILL_UNION_DCP: bool = False
    VLLM_TPU_QSA_PREFILL_PREFIX_PAGES: bool = False
    VLLM_TPU_QSA_MAIN_KV_DCP: int = 0
    VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK: int = 1024
    VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE: int = 256
    VLLM_TPU_QSA_MTP_KV_DCP: bool = False
    VLLM_TPU_QSA_COMPRESSED_DCP: bool = False
    VLLM_TPU_QSA_DCP_SHARD_MERGE: bool = False
    VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT: bool = False
    VLLM_TPU_MK_STACKED_QSA_WRITES: bool = False
    VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE: bool = False
    VLLM_TPU_MK_HC_FP8: bool = False
    VLLM_TPU_MK_ALLOW_FALLBACK: bool = False
    VLLM_TPU_MK_SCATTER_DEEP: bool = False
    VLLM_TPU_MK_STACKED_MAMBA_OPERANDS: bool = False
    VLLM_TPU_MK_BUCKET_TUNE: bool = False
    VLLM_TPU_MK_ROW_BUCKET: bool = False
    VLLM_TPU_MK_DRAFT: bool = False
    VLLM_TPU_MK_DRAFT_LOOP: bool = False
    VLLM_TPU_MK_QSEL_ASYNC: bool = False
    VLLM_TPU_MK_WHOLE_L47: bool = False
    VLLM_TPU_MK_WHOLE_L45: bool = False
    VLLM_TPU_HC_PREFILL_UP_T: bool = False
    VLLM_TPU_HC_PREFILL_COMBINE_MIX: bool = False
    VLLM_TPU_HC_PREFILL_WIDE_TILE: int = 0
    VLLM_TPU_HC_PREFILL_INJECT: bool = False
    TPU_QWEN4_HC_PREFILL_NORM: str = "vpu"
    VLLM_TPU_QSA_FUSED_DCP_BLOCKS: bool = False
    VLLM_TPU_QSA_DCP_HEAD_EXCHANGE: bool = False
    VLLM_TPU_QSA_DCP_Q_GATHER: bool = False
    VLLM_TPU_QSA_DECODE_SHARED_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_LEAN_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_SLOT_READ: bool = False
    VLLM_TPU_QSA_DECODE_FAST_FETCH: bool = False
    VLLM_TPU_QSA_DECODE_COMPACT_SCHEDULE: bool = False
    VLLM_TPU_QSA_DECODE_RPA_BKV: int = 256
    VLLM_TPU_QSA_DECODE_POOL_PAGE_TABLE: bool = False
    VLLM_TPU_QSA_DECODE_POOL_PAGE_DMA: bool = False
    VLLM_TPU_QSA_INDEXER_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_PREFILL_SC_BOUNDED_COPY: bool = False
    VLLM_TPU_QSA_PREFILL_INDEXER_CHUNK: int = 0
    VLLM_TPU_QSA_PREFILL_INDEXER_BQ: int = 128
    VLLM_TPU_QSA_PREFILL_INDEXER_BKV: int = 256
    VLLM_TPU_QSA_DECODE_RPA_LANES: int = 8
    VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES: int = 0
    VLLM_TPU_SHAPE_TUNED: bool = False
    VLLM_TPU_QSA_FUSED_KV_WRITE: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD: bool = False
    VLLM_TPU_QSA_INDEXER_SHARD_A2A_MIN_TOKENS: int = 0
    VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP: bool = False
    VLLM_TPU_QSA_INDEXER_TOKEN_SHARD: bool = False
    VLLM_TPU_QSA_DECODE_TOKEN_SHARD: bool = False
    VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD: bool = False
    VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD_MIN_TOKENS: int = 256
    VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS: int = 8
    VLLM_TPU_PLE_FUSED_GATE: bool = False
    VLLM_TPU_PLE_CONV_HISTORY_FAST: bool = False
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
    VLLM_TPU_MAMBA_TAIL_FOLD: bool = False
    VLLM_TPU_MAMBA_TAIL_FOLD_UNIT: int = 0
    VLLM_TPU_MAMBA_TAIL_FOLD_DCP: bool = False
    VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY: bool = False
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
    VLLM_TPU_FUSED_DRAFT_LOOP: bool = False
    VLLM_TPU_WARM_SPEC_LOGITS_LENGTHS: bool = True
    VLLM_TPU_PLE_HOST_FAST_LOOKUP: bool = False
    VLLM_TPU_PLE_HOST_GATHER_WORKERS: int = 0
    VLLM_TPU_QSA_VISIBLE_BYPASS: bool = False
    VLLM_TPU_PLE_HOST_TIMING: bool = False
    VLLM_TPU_PLE_HOST_OVERLAP: bool = False
    VLLM_TPU_PLE_HOST_OVERLAP_CHECK: int = 0
    VLLM_TPU_FUSED_ASYNC_DRAFT_SEED: bool = True
    VLLM_TPU_DEFER_PREV_RESULTS_WAIT: bool = False
    VLLM_TPU_LOG_POST_WARMUP_COMPILES: bool = False
    VLLM_TPU_CPU_PIN: str = "off"
    VLLM_TPU_HOST_PHASES: int = 0
    VLLM_TPU_PROMPT_IDS_NDARRAY: bool = False
    VLLM_TPU_PAW_SKIP_FULL: bool = False
    VLLM_TPU_SERIALIZE_COPY_WAIT: bool = False
    VLLM_TPU_PINNED_TOKEN_COPY: bool = False
    VLLM_TPU_SAMPLING_PARAMS_REUSE: bool = False
    VLLM_TPU_PY_SAMPLER_S: float = 0.0
    VLLM_TPU_PY_SAMPLER_INTERVAL_MS: float = 1.0
    VLLM_TPU_PY_SAMPLER_DELAY_S: float = 600.0
    VLLM_TPU_STAGE_ENTRIES_CACHE: bool = False
    VLLM_TPU_SUBST_STATIC_ARGS: bool = False
    VLLM_TPU_SUBTRACT_REJECTED_COMPILED: bool = False
    VLLM_TPU_SPEC_TAIL_COMPILED: bool = False
    VLLM_TPU_STALL_WATCHDOG_S: float = 0.0
    VLLM_TPU_LOOP_LAG_WATCHDOG_S: float = 30.0
    VLLM_TPU_FREEZE_PYSPY_S: float = 120.0
    VLLM_TPU_REASONING_COUNT_SKIP_TOOLS: bool = False
    VLLM_TPU_QSA_PREFILL_BITSET_OWNED: bool = False
    VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR: bool = False
    VLLM_TPU_QSA_DCP_GATE_IN_COMBINE: bool = False
    VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED: bool = False
    VLLM_TPU_QSA_DCP_SPLIT_LSE: bool = False
    VLLM_TPU_QSA_DCP_PACK_LISTS: bool = False
    VLLM_TPU_QSA_DCP_META_PARTITION: bool = False
    VLLM_TPU_TP_ONESHOT_AR_MAX_ROWS: int = 0
    VLLM_TPU_VERIFY_FUSED: bool = False
    VLLM_TPU_VERIFY_FUSED_CHECK: int = 0
    VLLM_TPU_VERIFY_FUSED_CHECK_FAULT: int = 0
    VLLM_TPU_DRAFT_SEED_FUSED: bool = False
    VLLM_TPU_DRAFT_CARRY_FUSED: bool = False
    VLLM_TPU_VERIFY_FUSED_READ_OFFSETS: bool = False
    VLLM_TPU_VERIFY_ONE_DRAW: bool = False
    VLLM_TPU_SUBST_FRESH_IDS: bool = False
    VLLM_TPU_ZERO_PADDED_SEQ_LENS: bool = False
    VLLM_TPU_DRAFT_ROWS_FRESH: bool = False
    VLLM_TPU_DRAFT_CARRY_ROWS_FRESH: bool = False
    VLLM_TPU_SUBST_EXTRACT_FUSED: bool = False
    VLLM_TPU_VERIFY_RO_INGRAPH: bool = False
    VLLM_TPU_DRAFT_SEED_EXTRAS: bool = False
    VLLM_TPU_DRAFT_SEED_BOOTSTRAP_PAD: bool = False
    VLLM_TPU_CPU_PIN_RESERVE: int = 2
    VLLM_TPU_CPU_PROBE_S: float = 0.0
    VLLM_TPU_CPU_PIN_MIGRATE: bool = False
    VLLM_TPU_SAMPLING_PARAMS_MEMO: bool = False
    VLLM_TPU_MAMBA_INDICES_REUSE: bool = False
    VLLM_TPU_GC_THRESHOLD: str = ""
    VLLM_TPU_FUSED_READ_OFFSET_UPDATE: bool = True
    VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE: bool = False
    VLLM_TPU_BLOCK_TABLE_PACK_DELTA: bool = False
    VLLM_TPU_TOKEN_PADDING_REUSE: bool = False
    VLLM_TPU_PREFLAT_FWD_ARGS: bool = False
    VLLM_TPU_MK_TRYFWD_LEAN: bool = False
    VLLM_TPU_FAST_METADATA_BUILD: bool = False
    VLLM_TPU_STEADY_DECODE_METADATA: bool = False
    VLLM_TPU_AOT_SKIP_GUARD_CHECK: bool = False
    VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK: bool = False
    VLLM_TPU_STAGED_WALK_CLASS_GATHER: bool = False
    VLLM_TPU_STAGED_WALK_STACKED: bool = False
    VLLM_TPU_WARMED_SUBSTITUTE_SOURCE: bool = False
    VLLM_TPU_PROPOSE_DEVICE_CONSTANTS: bool = False
    VLLM_TPU_MTP_DRAFT_ROWS: bool = False
    TPU_SAMPLER_TOPK_CANDIDATES: int = 0
    VLLM_TPU_SAMPLER_CANDIDATES_ONLY: bool = False
    VLLM_TPU_GDN_MIXED_TILE_POW2: bool = False
    VLLM_TPU_SPEC_BOUNDARY_STATE_COPY: bool = False
    VLLM_TPU_GDN_V3_WINDOW_VERIFY: bool = False
    VLLM_TPU_GDN_V3_SINGLE_LAUNCH: bool = False
    VLLM_TPU_CANDIDATE_SAMPLING: bool = False
    MM_ENCODER_FRAME_CHUNK_PATCH_SIZE: int = 0
    VLLM_TPU_STATIC_POOLER: bool = True
    VLLM_TPU_POOLING_KV_CACHE_TOKENS: int = -1
    VLLM_TPU_KV_BUDGET_PIN_TOLERANCE: float = 0.01
    VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS: int = 8
    VLLM_TPU_MEGAKERNEL: str = ""
    VLLM_TPU_MK_OWN_OPERANDS: bool = True
    VLLM_TPU_MK_OPERAND_DUMP: bool = False
    VLLM_TPU_MK_STEP_MIX_INTERVAL_S: float = 10.0
    CDK_OUTPUT_DIR: str = "/tmp/cdk_output"
    QWEN4_EXP_MAX_POOL_BYTES: int | None = None
    QWEN4_EXP_POOL_SIZING: str = "hbm"
    QWEN4_EXP_POOL_GRAPH_COPIES: int = 0
    QWEN4_EXP_POOL_HEADROOM_BYTES: int = 6 * 1024**3
    QWEN4_EXP_PACK_COMPRESSED_K: bool = False
    QWEN4_EXP_PACK_MAIN_KV: bool = False


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
    # Compute the pooled GDN step metadata (read-offset gather, int32 casts,
    # PER_SEQ roll) once per step and KV-cache group instead of in every GDN
    # layer: each layer's op is a separate program, so XLA cannot share them.
    # Same integers; spec decoding on the plain unified pool only.
    "VLLM_TPU_GDN_HOIST_METADATA": env_bool("VLLM_TPU_GDN_HOIST_METADATA"),
    # Size GDN v3 decode tiles and prefill chunks from per-rank heads, active
    # sequences and the VMEM estimate (kernels/gdn/v3/tiling.py) instead of
    # the fixed defaults.
    "TPU_ENABLE_GDN_DYNAMIC_TILING": env_bool(
        "TPU_ENABLE_GDN_DYNAMIC_TILING", default=False
    ),
    # GDN chunked prefill: each grid step takes two chunks of the same
    # sequence. The state-independent intra-chunk work (gates, K K^T, the
    # triangular inverse, U/W, masked Q K^T) runs once for both chunks batched
    # over heads, then the two state updates run in order. Same ops per chunk,
    # so bit-identical. A sequence's odd last chunk still costs a full pair
    # (its empty half is masked).
    "TPU_GDN_CHUNK_PAIR": env_bool("TPU_GDN_CHUNK_PAIR", default=False),
    # GDN v3: hand the kernel the projection's qkv [tokens, dim] as is (BF16)
    # instead of an FP32 [tokens, 1, dim] copy. The kernel DMAs a tile-aligned
    # row block and casts / places the rows in VMEM. Same values (BF16 -> FP32
    # is exact), so bit-identical; removes an XLA convert + relayout copy per
    # layer.
    "TPU_GDN_RAW_QKV": env_bool("TPU_GDN_RAW_QKV", default=False),
    # GDN v3: the same for the b / a gate inputs ([tokens, n_v] as is, no FP32
    # cast and 128-lane pad in XLA). GDN only (not KDA).
    "TPU_GDN_RAW_BA": env_bool("TPU_GDN_RAW_BA", default=False),
    # GDN v3: the first kernel of a call zero-fills the output rows no kernel
    # writes (bucket padding) instead of XLA zero-initializing the whole
    # output buffer first. Bit-identical.
    "TPU_GDN_KERNEL_ZERO_OUT": env_bool("TPU_GDN_KERNEL_ZERO_OUT", default=False),
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
    # Qwen4-exp QSA: run-based main KV writer for prefill-sized steps
    # (kernels/qwen4_exp/main_write.py): one DMA per run of consecutive rows
    # (power-of-two pieces), all in flight, instead of per-row DMAs in every
    # 128-token tile that is not page aligned and fully valid.
    "VLLM_TPU_QSA_MAIN_RUN_WRITE": env_bool("VLLM_TPU_QSA_MAIN_RUN_WRITE", default=False),
    # With VLLM_TPU_QSA_MAIN_RUN_WRITE: steps with fewer rows keep the tiled writers.
    "VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS", 512
    ),
    # Rows per grid step of the run writer (a power of two); runs never cross a
    # tile, and a tile's DMAs all complete before the next tile starts.
    "VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE": env_nonnegative_int(
        "VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE", 512
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
    # Qwen4-exp decode / verify indexer: StreamIndex pads each token's score
    # rows to whole (8, 128) int32 tiles, so its HBM scores reach the
    # SparseCore top-k as a bitcast instead of a relayout copy (~8 us per QSA
    # layer at c48). The padding lies past every row length (same selection).
    "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES": env_bool(
        "VLLM_TPU_QSA_INDEXER_TILE_ALIGNED_SCORES", default=False
    ),
    # Qwen4-exp QSA indexer on decode / verify steps: score every request in the
    # mixed pass and drop the two one-token decode passes (empty on verify
    # steps). Bypasses VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK on one-token steps.
    # Qwen4-exp MTP: the draft's QSA layer attends the target's last QSA layer
    # selection (the request's last verified row) instead of scoring its own
    # indexer: a selection-sharing approximation, gated on real AL.
    # Qwen4Exp n-gram context: "1" = H2D only each row's window of the host
    # token store that the device context can address
    # ([host_ctx - K - ngram + 1, host_ctx), K = spec tokens) instead of the
    # whole [max_num_seqs, max_model_len] table every step (16 MiB at c1, ~2 ms
    # of transfer); "check" = both paths and log mismatches for the first 200
    # steps, then as "1" (models/vllm/qwen4_exp/inputs.py).
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
    "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION": env_bool(
        "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION"
    ),
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
    # Mixed QSA steps: move the decode prefix off the prefill rows with
    # contiguous dynamic slices instead of row gathers (same values; prefill
    # chunks spent ~3 ms/8K on the gathers).
    "VLLM_TPU_QSA_PREFIX_SLICE": env_bool("VLLM_TPU_QSA_PREFIX_SLICE", default=False),
    # QSA indexer cache updates: build each token's compression-group members
    # from static shifts and the ring / tag rows from a one-hot product instead
    # of per-token row gathers (same values; ~3 ms of gathers per 8K prefill).
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
    # Qwen4-exp prefill indexer: the same bounded SparseCore score copy for
    # prefill chunks (rows early in a prompt see a short context).
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
    # the winners do not depend on the tiling. BKV=1024 also aligns the score
    # row width to whole sublane tiles, so the SparseCore's 1-D view of the
    # scores needs no relayout copy before the top-k.
    "VLLM_TPU_QSA_PREFILL_INDEXER_BQ": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_BQ", 128
    ),
    "VLLM_TPU_QSA_PREFILL_INDEXER_BKV": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_INDEXER_BKV", 256
    ),
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
    # Per-shape tile tables (kernels/qwen4_exp/shape_tuned.py): kernels look up
    # measured tiles by their static shape bucket (decode rows, prefill chunk
    # tokens) and override the env-var tiles. Off keeps every env tile as set.
    "VLLM_TPU_SHAPE_TUNED": env_bool("VLLM_TPU_SHAPE_TUNED", default=False),
    # Qwen4-exp QSA prefill: build the per-query selection bitset with a
    # compare-and-accumulate Pallas kernel. Off falls back to the vmapped
    # scatter-add, which XLA:TPU lowers to a global sort of every selection.
    "VLLM_TPU_QSA_PREFILL_BITSET_KERNEL": env_bool(
        "VLLM_TPU_QSA_PREFILL_BITSET_KERNEL", default=True
    ),
    # Qwen4-exp QSA prefill, with the bitset kernel: per eight-word group
    # compare only the selection chunks whose word range can reach it,
    # O(selections) per query block on the SparseCore top-k's ascending ids
    # instead of every word group against every selection (bit-identical).
    "VLLM_TPU_QSA_PREFILL_BITSET_RANGED": env_bool(
        "VLLM_TPU_QSA_PREFILL_BITSET_RANGED", default=False
    ),
    # Qwen4-exp QSA prefill attention: feed the MXU BF16 operands (exact
    # queries and keys, probabilities split into two BF16 terms) instead of
    # FP32 operands, which Mosaic multiplies at full FP32 precision.
    "VLLM_TPU_QSA_PREFILL_BF16_MXU": env_bool(
        "VLLM_TPU_QSA_PREFILL_BF16_MXU", default=False
    ),
    # Under VLLM_TPU_QSA_MAIN_KV_DCP > 1: prefix-cache hits at the hash unit
    # (B) instead of the D*B scheduler block, with copy-on-write of a partially
    # hit block (vllm_torchtpu/__init__.py _patch_vllm_qsa_dcp_fine_prefix_hits).
    "VLLM_TPU_QSA_DCP_FINE_PREFIX": env_bool("VLLM_TPU_QSA_DCP_FINE_PREFIX"),
    # With it: at request free, also register partial entries at the hash
    # boundaries inside the request's last block (next turns resuming past the
    # previous output), not only at the prompt's last boundary.
    "VLLM_TPU_QSA_DCP_FINE_PREFIX_SEQ_TAIL": env_bool(
        "VLLM_TPU_QSA_DCP_FINE_PREFIX_SEQ_TAIL"
    ),
    # With it: keep VLLM_TPU_QSA_DCP_FINE_PREFIX on when the KV connector is a
    # CPU offloading one (the Raiden host tier). A host lookup runs only after
    # a chunk-aligned HBM hit (_patch_vllm_offload_fine_prefix_alignment).
    # Off: any KV connector turns the fine hits off (hits round to D*B).
    "VLLM_TPU_QSA_DCP_FINE_PREFIX_CONNECTOR": env_bool(
        "VLLM_TPU_QSA_DCP_FINE_PREFIX_CONNECTOR"
    ),
    # qwen4_exp QSA sparse prefill: skip the FP8 key finite/written guard,
    # which the score mask select already covers (bit-identical output).
    "VLLM_TPU_QSA_PREFILL_LEAN": env_bool("VLLM_TPU_QSA_PREFILL_LEAN"),
    # qwen4_exp QSA sparse prefill: cut the per-score vector (VPU) work of the
    # block-sparse kernel's inner loop (block_sparse_gqa_prefill, with or
    # without DCP). "" / "0" = off; "1" = the default set (mask,qk16,pv16); or
    # a comma list of: mask (always on: per-segment selection bits, causal and
    # written guards only on the diagonal KV tile; exact), qk16 (one BF16 QK
    # MXU pass; exact when the score scale is a power of two), pv16 (BF16
    # probabilities into the PV dot; bit-identical on v7x, where Mosaic
    # already runs the FP32 dot as one BF16 pass).
    # Precedence: VLLM_TPU_QSA_PREFILL_UNION_WALK replaces the block-sparse
    # kernel where it applies; this flag acts on every remaining call (DCP,
    # kv_head_offset layers, short contexts, small steps, the union walk's
    # shared fallback).
    "VLLM_TPU_QSA_PREFILL_VPU": env_str("VLLM_TPU_QSA_PREFILL_VPU", ""),
    # qwen4_exp QSA sparse prefill: fetch each KV tile with one DMA per
    # page-sized run instead of one per 128-row selection segment, when KV
    # tiles and pages nest (TP8: 1024-row tiles over 512-row pages; DCP: one
    # page per tile). Same rows, bit-identical.
    "VLLM_TPU_QSA_PREFILL_PAGE_COPIES": env_bool("VLLM_TPU_QSA_PREFILL_PAGE_COPIES"),
    # Qwen4-exp QSA prefill: union-walk attention (qsa_union_dense.
    # union_walk_prefill) with query groups of this many rows (a divisor of 128,
    # at least 8; 0, the default, keeps the block-sparse kernel). Each group of
    # rows attends only the union of its rows' selected 4-token groups, DMA'd
    # straight from the paged pool, instead of every KV tile any row of a
    # 256-row query block selected. Taken for D256 steps (24/2 or a TP rank's
    # 3/1, 6/1, 12/1 query/KV heads) on an FP8 pool of 128-aligned (1, 4, 256)
    # pages, at least 1024 tokens and a multiple of the group, without DCP, a KV
    # head offset or LSE output. Each step falls back to the block-sparse kernel
    # when its longest row is below VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT
    # tokens or a group's union exceeds VLLM_TPU_QSA_PREFILL_UNION_CAP 4-token
    # groups. Not bitwise against the block-sparse kernel (union-order online
    # softmax, equal precision); accuracy gated.
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
    # Union walk options (all off by default; bitwise unless noted):
    # LISTED: issue the walk's copies from a union id list instead of scanning
    # the union bitset; "sparsecore" (union_sc_list, TPU) or "reference" (XLA).
    # SHARE_BITS: a cap fallback reuses the walk plan's selection bitset.
    # ALPHA: also fall back when a group's union exceeds this fraction of its
    # visible groups (0 = off). QK16: QK as one BF16 MXU pass (exact operands,
    # power-of-two score scales only).
    "VLLM_TPU_QSA_PREFILL_UNION_LISTED": env_str("VLLM_TPU_QSA_PREFILL_UNION_LISTED", ""),
    "VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_SHARE_BITS"
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_ALPHA": env_float("VLLM_TPU_QSA_PREFILL_UNION_ALPHA", 0.0),
    "VLLM_TPU_QSA_PREFILL_UNION_QK16": env_bool("VLLM_TPU_QSA_PREFILL_UNION_QK16"),
    # RUN_TABLES (with LISTED; parked): the walk issued from SparseCore run tables.
    "VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_RUN_TABLES"
    ),
    # LAZY_SHARE (with SHARE_BITS): walk on group-aligned rows; only a fallback
    # builds the block-sparse layout.
    "VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_LAZY_SHARE"
    ),
    # SC_BITSET: the union plan's per-row bitset built on the SparseCore (the
    # same bits), off the TensorCore.
    "VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET": env_bool(
        "VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET"
    ),
    # GATHERED ("", "reference", "sparsecore"; with LISTED): the walk-free
    # kernel on chunked SparseCore-gathered union tiles.
    "VLLM_TPU_QSA_PREFILL_UNION_GATHERED": env_str(
        "VLLM_TPU_QSA_PREFILL_UNION_GATHERED", ""
    ),
    "VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS": env_int(
        "VLLM_TPU_QSA_PREFILL_UNION_GATHER_CHUNKS", 4
    ),
    # GATHER_GROWTH (>= 1; 1 = even chunks): chunk k of the gathered walk holds
    # about growth**k as many query groups as chunk 0. Bitwise.
    "VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH": env_float(
        "VLLM_TPU_QSA_PREFILL_UNION_GATHER_GROWTH", 1.0
    ),
    # COMPACT_REQS (N > 0; with GATHERED): the SparseCore takes at most 16 GiB
    # per tensor, so the gather reads a per-step compact pool of the first N
    # requests' pages (whole-page copies); a step with more requests takes the
    # listed walk on the full pool (bitwise). 0: the gather reads the pool.
    "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_REQS", 0
    ),
    # COMPACT_COPY: how the compact pool is copied ("vmem": whole pages staged
    # through VMEM; "pipe": a Pallas pipeline of 32-bit words; "dma": HBM-to-HBM
    # DMAs, slow). All byte-exact.
    "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY": env_str(
        "VLLM_TPU_QSA_PREFILL_UNION_COMPACT_COPY", "vmem"
    ),
    # DCP (with VLLM_TPU_QSA_PREFILL_UNION_WALK; default off): under main-KV
    # DCP (VLLM_TPU_QSA_MAIN_KV_DCP) the union walk also serves the prefill's
    # (out, lse) partials over this rank's owned groups, on either paired main
    # KV slot, instead of the DCP block-sparse kernel (which stays its
    # short-context / cap / alpha fallback). Bit-scan walk only: with LISTED,
    # RUN_TABLES, GATHERED, COMPACT_REQS or SHARE_BITS set, DCP keeps the
    # block-sparse kernel. Not bitwise against it (union-order online
    # softmax, equal precision).
    "VLLM_TPU_QSA_PREFILL_UNION_DCP": env_bool("VLLM_TPU_QSA_PREFILL_UNION_DCP"),
    # qwen4_exp QSA (unsharded indexer path): when every row's completed groups
    # fit the top-k budget, take all visible groups and skip the StreamIndex
    # scoring call (the sharded path already does). Same selected set; the
    # order of the selected groups may differ from the kernel's.
    "VLLM_TPU_QSA_VISIBLE_BYPASS": env_bool("VLLM_TPU_QSA_VISIBLE_BYPASS"),
    # qwen4_exp QSA sparse prefill: when a step's selections and causal lengths
    # fit a page-table prefix, build the selection metadata (bitset, active
    # tiles) over that prefix instead of max_model_len (bit-identical).
    "VLLM_TPU_QSA_PREFILL_PREFIX_PAGES": env_bool("VLLM_TPU_QSA_PREFILL_PREFIX_PAGES"),
    # Main KV of the target QSA layers split over TP rank pairs that share a
    # KV head (2; 0/1 = off): each rank stores and attends half the tokens,
    # the pair's partial attention is LSE-combined.
    "VLLM_TPU_QSA_MAIN_KV_DCP": env_nonnegative_int("VLLM_TPU_QSA_MAIN_KV_DCP", 0),
    # Main-KV DCP prefill (owned-page walk): query block / subtile; 0 keeps the tile table's.
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK": env_nonnegative_int(
        "VLLM_TPU_QSA_DCP_PREFILL_QUERY_BLOCK", 1024
    ),
    "VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE": env_nonnegative_int(
        "VLLM_TPU_QSA_DCP_PREFILL_QUERY_TILE", 256
    ),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: the MTP draft layers' main KV split the
    # same way (otherwise the draft keeps whole pages).
    "VLLM_TPU_QSA_MTP_KV_DCP": env_bool("VLLM_TPU_QSA_MTP_KV_DCP"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP and VLLM_TPU_QSA_MTP_KV_DCP: compressed keys
    # split the same way; each rank scores its owned groups and the pair merges
    # the local top-k lists into the exact global selection.
    "VLLM_TPU_QSA_COMPRESSED_DCP": env_bool("VLLM_TPU_QSA_COMPRESSED_DCP"),
    # With VLLM_TPU_QSA_COMPRESSED_DCP on the prefill token-shard path: each
    # rank merges its own 1/TP of the rows from its pair's lists and one TP
    # all-gather shares the merged selection, instead of every rank merging
    # every row of every pair's raw lists (bit-identical).
    "VLLM_TPU_QSA_DCP_SHARD_MERGE": env_bool("VLLM_TPU_QSA_DCP_SHARD_MERGE"),
    # Compressed-key writes: take each pair's two key rows by selecting over
    # statically shifted copies of the keys (offsets within two compression
    # groups of the token) instead of a gather; a lax.cond falls back to the
    # gather when a written row's offset is out of range (bit-identical pool).
    "VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT": env_bool("VLLM_TPU_QSA_COMPRESSED_PAIR_SHIFT"),
    # Whole-step megakernel decode: write all QSA layers' step tokens with
    # layer-stacked ops (one main K/V scatter, one compressed-key writer call)
    # instead of one writer sequence per layer. Steps that are not a multiple of
    # 128 tokens only; same pool bytes.
    "VLLM_TPU_MK_STACKED_QSA_WRITES": env_bool("VLLM_TPU_MK_STACKED_QSA_WRITES"),
    # With the stacked writes: the compressed keys of all layers go out in one
    # kernel step with concurrent DMAs (`write_compressed_pairs_unique`) instead
    # of one ordered grid step per row. Same pool bytes.
    "VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE": env_bool("VLLM_TPU_MK_COMPRESSED_UNIQUE_WRITE"),
    # Whole-step megakernel: the target model's HC mix weights (layers 3..46, HC of
    # the mixer and the MoE) stream as FP8 E4M3 with per-output-column scales. The
    # draft (MTP) and the layers outside the whole-step kernel keep BF16 HC.
    "VLLM_TPU_MK_HC_FP8": env_bool("VLLM_TPU_MK_HC_FP8"),
    # Megakernel failures (setup, self-check, a failed call, MTP tail / MK draft setup)
    # raise by default; =1 restores the old stock fallback (draft kernels detached).
    "VLLM_TPU_MK_ALLOW_FALLBACK": env_bool("VLLM_TPU_MK_ALLOW_FALLBACK"),
    # Whole-step GDN checkpoint scatter: 16 staging slots / 12 loads in flight from 64
    # checkpoints (the default deep ring starts at 144). Same pool bytes.
    "VLLM_TPU_MK_SCATTER_DEEP": env_bool("VLLM_TPU_MK_SCATTER_DEEP"),
    # Whole-step megakernel with VLLM_TPU_STAGED_WALK_STACKED: the GDN layers' metadata go
    # to the call as the walk's two stacks + a row index + the shared read offsets /
    # query_start_loc / seq_lens (6 operands instead of 5 per GDN layer). Same values.
    "VLLM_TPU_MK_STACKED_MAMBA_OPERANDS": env_bool("VLLM_TPU_MK_STACKED_MAMBA_OPERANDS"),
    # Whole-step megakernel: per-row-bucket tile / tunable overrides (BUCKET_TUNE in
    # qwen38_wholestep_call). Layout-only entries keep the bytes; see its comments.
    "VLLM_TPU_MK_BUCKET_TUNE": env_bool("VLLM_TPU_MK_BUCKET_TUNE"),
    # Whole-step megakernel: routed-row bucketing in the PIPE_EP expert loop (each
    # expert's e13 / e2 on its gathered routed rows; MoETunables.row_bucket).
    "VLLM_TPU_MK_ROW_BUCKET": env_bool("VLLM_TPU_MK_ROW_BUCKET"),
    # Whole-step megakernel: MTP draft step 0 (draft layer, final mixer, lm_head,
    # greedy pick, carry) as one kernel call (megakernel_runtime.qwen38_draft). Draft
    # weights at checkpoint precision (FP8 routed experts, BF16 elsewhere; no HC FP8).
    # Needs the whole-step MK and the fused loop.
    "VLLM_TPU_MK_DRAFT": env_bool("VLLM_TPU_MK_DRAFT"),
    # With VLLM_TPU_MK_DRAFT: the fused loop's steps 1..K-1 through the MK draft call
    # too (else only draft step 0). Off: at T=1 the call is slower than stock
    # (16 request slots per kernel tile; j-1767564a).
    "VLLM_TPU_MK_DRAFT_LOOP": env_bool("VLLM_TPU_MK_DRAFT_LOOP"),
    # Whole-step QSA selection: the per-row id copies into SMEM and the new group's
    # loads / stores are issued together and waited once instead of in start-wait
    # pairs (distinct destinations; same bytes).
    "VLLM_TPU_MK_QSEL_ASYNC": env_bool("VLLM_TPU_MK_QSEL_ASYNC"),
    # Whole-step megakernel: also run the last QSA layer (model layer 47) in the whole-step
    # call, as a one-layer kernel after layers 3..46, its step tokens written with theirs;
    # the stock L47 attention core and last segment kernel are skipped.
    "VLLM_TPU_MK_WHOLE_L47": env_bool("VLLM_TPU_MK_WHOLE_L47"),
    # Whole-step megakernel: layers 3..47 in ONE kernel (L = 45, a partial last cycle of
    # its QSA layer) instead of VLLM_TPU_MK_WHOLE_L47's second one-layer kernel; implies
    # the L47 plumbing (cache writes, stock L47 skipped).
    "VLLM_TPU_MK_WHOLE_L45": env_bool("VLLM_TPU_MK_WHOLE_L45"),
    # Prefill HC full mix: take the up projection as its [low rank, width]
    # transpose. The checkpoint weight is held column-major, so the transpose
    # is a bitcast and the per-call relayout copy before the kernel goes away
    # (bit-identical).
    "VLLM_TPU_HC_PREFILL_UP_T": env_bool("VLLM_TPU_HC_PREFILL_UP_T"),
    # Prefill HC combine_mix (attn -> mlp): the gated residual runs as the
    # prologue of the next HC's full-mix kernel, which writes the combined hyper
    # input once instead of a separate merge kernel writing it and the mix
    # reading it back (bit-identical).
    "VLLM_TPU_HC_PREFILL_COMBINE_MIX": env_bool("VLLM_TPU_HC_PREFILL_COMBINE_MIX"),
    # Prefill HC full mix: token tile for calls whose rows divide two such tiles
    # (0 = always hc_prefill_full_tile). Rows are independent, so the tile only
    # changes the schedule (bit-identical); 256 is ~25 us/call faster than 128
    # at 1024 rows.
    "VLLM_TPU_HC_PREFILL_WIDE_TILE": env_nonnegative_int(
        "VLLM_TPU_HC_PREFILL_WIDE_TILE", 0
    ),
    # Prefill HC full mix: project the normalized input onto the combine's gate
    # weight in the kernel and return that [T, hc] projection instead of writing
    # the normalized [T, hc * H] input for an XLA projection to read back.
    "VLLM_TPU_HC_PREFILL_INJECT": env_bool("VLLM_TPU_HC_PREFILL_INJECT"),
    # Prefill HC full mix: RMS-norm variant of the kernel's norm loop. vpu (the
    # default, unchanged program); rows64 (64-row loop steps, bitwise); gram /
    # gram64 (each branch's sum of squares as the diagonal of x x^T on the MXU;
    # not bitwise, only the f32 summation order of the variance changes);
    # bf16ep / bf16ep64 (the epilogue n * w, n + . in bf16 arithmetic, bitwise:
    # a bf16 x bf16 product is exact in f32 and the bf16 sum rounds the same).
    "TPU_QWEN4_HC_PREFILL_NORM": env_with_choices(
        "TPU_QWEN4_HC_PREFILL_NORM",
        "vpu",
        ["vpu", "rows64", "gram", "gram64", "bf16ep", "bf16ep64"],
        case_sensitive=False,
    ),
    # With VLLM_TPU_QSA_COMPRESSED_DCP: merge a pair's local top-k lists into the
    # global selection with one TensorCore Pallas kernel (radix select, tie
    # cut, bitonic sort) instead of the SparseCore top-k + keep mask + sort
    # (bit-identical).
    "VLLM_TPU_QSA_FUSED_DCP_BLOCKS": env_bool("VLLM_TPU_QSA_FUSED_DCP_BLOCKS"),
    # With VLLM_TPU_QSA_MAIN_KV_DCP: the pair exchanges each rank's heads of
    # the partial attention (all-to-all) instead of all-gathering every
    # partial; half the bytes, bit-identical.
    "VLLM_TPU_QSA_DCP_HEAD_EXCHANGE": env_bool("VLLM_TPU_QSA_DCP_HEAD_EXCHANGE"),
    # Qwen4-exp token-shard DCP prefill: all-gather only the pair's q heads,
    # not their gates (the DCP attention is ungated; each rank gates its own
    # heads); half the bytes, bit-identical.
    "VLLM_TPU_QSA_DCP_Q_GATHER": env_bool("VLLM_TPU_QSA_DCP_Q_GATHER"),
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
    # Qwen4-exp pooled QSA with VLLM_TPU_QSA_COMPRESSED_DCP, the replicated
    # (non-SP) backbone, i.e. decode / verify and the small prefill buckets:
    # each DCP pair indexes one contiguous 1/(TP / DCP) of the step's rows
    # (token_shard_step) and one TP all-gather brings every pair's local lists
    # for the pair merge, instead of every pair indexing every row. The
    # selection is the replicated indexer's.
    "VLLM_TPU_QSA_DECODE_TOKEN_SHARD": env_bool(
        "VLLM_TPU_QSA_DECODE_TOKEN_SHARD", default=False
    ),
    # Qwen4-exp pooled QSA, prefill buckets that do not take the
    # sequence-parallel graph (e.g. 256 / 512 / 3072 at TP8): route them to a
    # third compiled graph (Qwen4ExpBackboneRP), the replicated backbone whose
    # QSA layers run the token-sharded indexer of
    # VLLM_TPU_QSA_INDEXER_TOKEN_SHARD: each TP rank scores and selects one
    # contiguous 1/TP of the step's rows and the selected blocks are
    # all-gathered. The selection is the replicated indexer's. A bucket takes
    # it when it has at least ..._MIN_TOKENS rows, divides by TP and leaves
    # each rank more rows than request slots. Not with
    # VLLM_TPU_QSA_COMPRESSED_DCP (there VLLM_TPU_QSA_DECODE_TOKEN_SHARD
    # shards the replicated backbone's indexer over the DCP pairs).
    "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD": env_bool(
        "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD", default=False
    ),
    "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD_MIN_TOKENS": env_nonnegative_int(
        "VLLM_TPU_QSA_PREFILL_REPLICATED_TOKEN_SHARD_MIN_TOKENS", 256
    ),
    # Qwen4-exp resident PLE table: tokens per gather in the raw embedding
    # lookup (a multiple of 8). 8 is the originally validated chunk; an
    # 8192-token prefill step then runs 1024 sequential gathers.
    # qwen4_exp PLE: one Pallas kernel for the per-stream normalized key .
    # query gate dot (norm weights, BF16 products, FP32 reduction) instead of
    # two grouped RMS norms and a Torch product/sum. Reduction order differs.
    "VLLM_TPU_PLE_FUSED_GATE": env_bool("VLLM_TPU_PLE_FUSED_GATE"),
    # qwen4_exp PLE short conv (request-owned history): skip the history read
    # for the zero-distance tap and gather only each request's first
    # `distance` history rows for the others, instead of one history row per
    # token per tap (bit-identical).
    "VLLM_TPU_PLE_CONV_HISTORY_FAST": env_bool("VLLM_TPU_PLE_CONV_HISTORY_FAST"),
    "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS": env_nonnegative_int(
        "VLLM_TPU_PLE_LOOKUP_CHUNK_TOKENS", 8
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
    # qwen4_exp: shard prefill-sized steps across TP ranks between attention
    # blocks (reduce-scatter instead of all-reduce), so the hyper-connection
    # streams, PLE gate and residual math run on 1/TP of the tokens. Steps
    # below QWEN4_EXP_PREFILL_SP_MIN_TOKENS keep the replicated TP path.
    "QWEN4_EXP_PREFILL_SP": env_bool("QWEN4_EXP_PREFILL_SP"),
    "QWEN4_EXP_PREFILL_SP_MIN_TOKENS": env_int("QWEN4_EXP_PREFILL_SP_MIN_TOKENS", 1024),
    # With QWEN4_EXP_PREFILL_SP=1: the adaptive fused EP MoE takes each
    # rank's own rows on sequence-parallel steps (no replica split/restore)
    # and the shared expert's TP sum is reduce-scattered, not all-reduced.
    "QWEN4_EXP_PREFILL_SP_MOE": env_bool("QWEN4_EXP_PREFILL_SP_MOE"),
    # Qwen3.8 megakernel with PLE CPU offload: the segment / whole-step programs take the
    # PLE layer's rows from the host lookup instead of refusing to run (the kernels never
    # read the PLE table; only apply_ple before the PLE layer's head group does).
    "VLLM_TPU_MK_HOST_PLE": env_bool("VLLM_TPU_MK_HOST_PLE"),
    # Qwen3.8 whole-step megakernel: build the QSA layers' page tables and read their rings
    # for all layers at once (one ring-read call) instead of per layer; same values.
    "VLLM_TPU_MK_WHOLE_BATCHED_INPUTS": env_bool("VLLM_TPU_MK_WHOLE_BATCHED_INPUTS"),
    # Qwen4Exp PLE decode trims, same bytes: on steps of <= 8 tokens the head
    # groups' n-gram rows ride the token-embedding all-reduce as integer-valued
    # BF16 lanes (one collective instead of two), and the n-gram hash constants
    # are trace-time Python ints instead of int64 buffers.
    "VLLM_TPU_MK_PLE_TRIM": env_bool("VLLM_TPU_MK_PLE_TRIM"),
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
    # Build the MoE permutation, its inverse, and the sorted token ids from
    # one packed sort instead of three. The ragged (counting sort) path ignores it.
    "TPU_MOE_SINGLE_SORT": env_bool("TPU_MOE_SINGLE_SORT", True),
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
    # Qwen4Exp unified pool: finish a prefill in one chunk instead of
    # splitting off the tokens past its last Mamba block boundary, and write
    # the GDN / conv / PLE state at that boundary from inside the step
    # (core/mamba_tail_fold.py). Engine core and workers.
    "VLLM_TPU_MAMBA_TAIL_FOLD": env_bool("VLLM_TPU_MAMBA_TAIL_FOLD", default=False),
    # With VLLM_TPU_MAMBA_TAIL_FOLD: the granularity (tokens) of the internal
    # prefill checkpoint and of prefix-cache hits. 0 = the Mamba block (512,
    # unchanged). A finer unit (e.g. 256) that divides the Mamba block and is a
    # multiple of the largest GDN prefill tile (128, 256 with TPU_GDN_CHUNK_PAIR)
    # checkpoints the prompt nearer its end, so the next turn's prefix hit
    # stops at most unit - 1 tokens short of the prior prompt instead of up
    # to 511. Also sets --prefix-match-unit to the unit when it is unset.
    "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT": env_int("VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", 0),
    # Ignored (hard-off): the fold under the Qwen4Exp main-KV DCP gave wrong
    # outputs (za6 GSM8K c32 0.26, c48 0.008). The fold turns itself off under
    # any context parallelism.
    "VLLM_TPU_MAMBA_TAIL_FOLD_DCP": env_bool(
        "VLLM_TPU_MAMBA_TAIL_FOLD_DCP", default=False
    ),
    # With VLLM_TPU_MAMBA_TAIL_FOLD (and speculative decoding): also fold a
    # prompt's last chunk when it holds no Mamba block boundary, e.g. a prefix
    # hit at 82944 and a 473-token rest that the fine unit would split 256 +
    # 217. The checkpoint goes to the request's first speculative scratch
    # block and is moved to its own block before the next forward
    # (core/mamba_tail_fold.py).
    "VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY": env_bool(
        "VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY", default=False
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
    # Host-offloaded PLE table: split lookups of at least 2 * 16384 rows over
    # up to this many threads (disjoint output ranges; 0/1 = serial). Prefill
    # chunks only; the gathered rows are identical.
    "VLLM_TPU_PLE_HOST_GATHER_WORKERS": env_nonnegative_int(
        "VLLM_TPU_PLE_HOST_GATHER_WORKERS", 0
    ),
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
    # chip's NUMA node), "slice" (own physical cores on that node, keeping
    # VLLM_TPU_CPU_PIN_RESERVE cores per node free) or "core" (the step thread
    # alone on one physical core, sibling idle; every other thread of every
    # vLLM process kept off those cores). See worker/cpu_pinning.py.
    "VLLM_TPU_CPU_PIN": env_str("VLLM_TPU_CPU_PIN", "off"),
    # Log each rank's per-step host time by runner phase every N steps
    # (0 = off); runner/host_phases.py.
    "VLLM_TPU_HOST_PHASES": env_nonnegative_int("VLLM_TPU_HOST_PHASES", 0),
    # Ship a new request's prompt token ids to the workers as one int32
    # ndarray (converted once when the request is added) instead of a Python
    # list. A ~100K-token agentic prompt otherwise costs every rank a list
    # unpickle plus a list->numpy copy into token_ids_cpu on the admit step's
    # critical path (core/prefix_aware_scheduler.py). Same token values.
    "VLLM_TPU_PROMPT_IDS_NDARRAY": env_bool("VLLM_TPU_PROMPT_IDS_NDARRAY"),
    # PrefixAwareScheduler: skip the warmth scoring (one prefix-cache lookup
    # per waiting request, every step) when no seat is free and the whole
    # queue fits the window; the admission order is unchanged.
    "VLLM_TPU_PAW_SKIP_FULL": env_bool("VLLM_TPU_PAW_SKIP_FULL"),
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
    "VLLM_TPU_PY_SAMPLER_INTERVAL_MS": env_float(
        "VLLM_TPU_PY_SAMPLER_INTERVAL_MS", 1.0
    ),
    "VLLM_TPU_PY_SAMPLER_DELAY_S": env_float("VLLM_TPU_PY_SAMPLER_DELAY_S", 600.0),
    # Keep the per-group (block table, width) list the staged metadata walk
    # builds, instead of re-walking every KV cache group each step.
    "VLLM_TPU_STAGE_ENTRIES_CACHE": env_bool("VLLM_TPU_STAGE_ENTRIES_CACHE"),
    # Strip dynamic-dim stamps from the async token substitution's inputs so it
    # does not recompile mid-run (tpu_runner._apply_async_token_substitution).
    "VLLM_TPU_SUBST_STATIC_ARGS": env_bool("VLLM_TPU_SUBST_STATIC_ARGS"),
    # With VLLM_TPU_FAST_TOKEN_SUBSTITUTION: the 1:1 steady-decode input ids
    # come from one compiled slice of the previous step's source (a buffer of
    # their own) instead of an eager view of it, which torch_tpu copies with an
    # execute_as_strided program before every compiled consumer (same values).
    "VLLM_TPU_SUBST_FRESH_IDS": env_bool("VLLM_TPU_SUBST_FRESH_IDS"),
    # Runner: zero the padded request slots' sequence lengths each step (they
    # otherwise keep earlier steps' lengths). Kernels mask padded slots by their
    # zero query length; batch-wide maxima (the QSA visible-group bypass) do not.
    "VLLM_TPU_ZERO_PADDED_SEQ_LENS": env_bool("VLLM_TPU_ZERO_PADDED_SEQ_LENS"),
    # Async spec decode: the per-step rejected-count correction of seq_lens and
    # positions as one compiled program (precompiled per request bucket).
    "VLLM_TPU_SUBTRACT_REJECTED_COMPILED": env_bool(
        "VLLM_TPU_SUBTRACT_REJECTED_COMPILED"
    ),
    # Async spec decode, one chunk: the next step's substitution source and the
    # rejected count (with their row fits and the source pad) as one program.
    "VLLM_TPU_SPEC_TAIL_COMPILED": env_bool("VLLM_TPU_SPEC_TAIL_COMPILED"),
    # Dump all processes' stacks when no step starts for N s with live
    # requests (stall_watchdog.py; read directly there, registered here).
    "VLLM_TPU_STALL_WATCHDOG_S": env_float("VLLM_TPU_STALL_WATCHDOG_S", 0.0),
    # With the stall watchdog on: dump the API server's stacks when its event
    # loop blocks for N s (stall_watchdog.py; read directly there).
    "VLLM_TPU_LOOP_LAG_WATCHDOG_S": env_float("VLLM_TPU_LOOP_LAG_WATCHDOG_S", 30.0),
    # With the stall watchdog on: an out-of-process monitor runs py-spy dump
    # --native on the API server and its children when a heartbeat or progress
    # stops for N s (freeze_spy.py; read directly there; 0 disables).
    "VLLM_TPU_FREEZE_PYSPY_S": env_float("VLLM_TPU_FREEZE_PYSPY_S", 120.0),
    # API server: count a streamed response's reasoning tokens without
    # re-parsing its tool calls (core/reasoning_count_patch.py). The full
    # re-parse is quadratic in tool-argument length and freezes the server.
    "VLLM_TPU_REASONING_COUNT_SKIP_TOOLS": env_bool("VLLM_TPU_REASONING_COUNT_SKIP_TOOLS"),
    # Main-KV DCP sparse prefill: the selection bitset kernel emits this rank's
    # words only, instead of a global bitset then an owned-word gather
    # (bit-identical).
    "VLLM_TPU_QSA_PREFILL_BITSET_OWNED": env_bool("VLLM_TPU_QSA_PREFILL_BITSET_OWNED"),
    # Compressed main-KV DCP ops: all-gather the pair's q/gate rank-major
    # (dim 0) and join the heads inside the step, instead of a dim -1 gather
    # whose head join runs as a standalone copy (bit-identical).
    "VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR": env_bool("VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR"),
    # Main-KV DCP: pass the whole q/gate row to the pair combine, which slices
    # the gate half itself, instead of slicing it on the Torch side
    # (a standalone strided slice plus relayout; bit-identical).
    "VLLM_TPU_QSA_DCP_GATE_IN_COMBINE": env_bool("VLLM_TPU_QSA_DCP_GATE_IN_COMBINE"),
    # Sparse prefill with lse (main-KV DCP): mask dead rows only in the padded
    # layouts; the aligned layout's kernel rows are already 0 / -inf there
    # (bit-identical, drops an f32 [T, heads, D] select per call).
    "VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED": env_bool("VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED"),
    # Main-KV DCP attention: route the partial output and its lse as two arrays
    # instead of one [.., D + 1] channel (a padded f32 concat plus the slices
    # that undo it at every 8K prefill step); same values.
    "VLLM_TPU_QSA_DCP_SPLIT_LSE": env_bool("VLLM_TPU_QSA_DCP_SPLIT_LSE"),
    # Compressed main-KV DCP: all-gather the local top-k values (f32, as int32
    # bits) and group ids (int32) as one stacked tensor instead of two
    # gathers; same bits.
    "VLLM_TPU_QSA_DCP_PACK_LISTS": env_bool("VLLM_TPU_QSA_DCP_PACK_LISTS"),
    # Decode DCP metadata: the owned-first order of an already ascending
    # selection by a stable partition instead of a [rows, budget] sort.
    "VLLM_TPU_QSA_DCP_META_PARTITION": env_bool("VLLM_TPU_QSA_DCP_META_PARTITION"),
    # TP all-reduce of BF16 tensors with at most this many rows (0 = off): one-shot
    # all-gather + local f32 sum (tpu_communicator.TpuDeviceCommunicator.all_reduce).
    # Faster for decode-sized tensors; not bit-identical to psum.
    "VLLM_TPU_TP_ONESHOT_AR_MAX_ROWS": env_nonnegative_int("VLLM_TPU_TP_ONESHOT_AR_MAX_ROWS", 0),
    # Non-greedy spec verify as one program (tpu_runner._verify_fused_impl).
    "VLLM_TPU_VERIFY_FUSED": env_bool("VLLM_TPU_VERIFY_FUSED"),
    # Compare the fused verify and draft seed with the separate programs for
    # the first N chunks; the first _FAULT checks perturb the compared copy
    # (a positive control that must be caught).
    "VLLM_TPU_VERIFY_FUSED_CHECK": env_nonnegative_int(
        "VLLM_TPU_VERIFY_FUSED_CHECK", 0
    ),
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
    # With VLLM_TPU_SUBST_FRESH_IDS: the steady-decode substitution program
    # also gathers the verified draft tokens (extract_draft_token_ids), one
    # program instead of two. Same values.
    "VLLM_TPU_SUBST_EXTRACT_FUSED": env_bool("VLLM_TPU_SUBST_EXTRACT_FUSED"),
    # With VLLM_TPU_VERIFY_ONE_DRAW (and VERIFY_FUSED_READ_OFFSETS off): the
    # post-verify mamba read-offset scatter runs inside the verify program
    # (one program instead of two). Same values.
    "VLLM_TPU_VERIFY_RO_INGRAPH": env_bool("VLLM_TPU_VERIFY_RO_INGRAPH"),
    # With VLLM_TPU_VERIFY_FUSED: compute the mamba read offsets inside the
    # fused verify program and apply them with one eager scatter.
    "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS": env_bool("VLLM_TPU_VERIFY_FUSED_READ_OFFSETS"),
    # With VLLM_TPU_VERIFY_FUSED: draw the bonus, accept and recover uniforms
    # as one flat torch.rand split inside the fused verify program (one eager
    # uniform program per step instead of three). Same order of values; the
    # warmup logs whether it bit-equals the three draws on this backend
    # ("VERIFY_ONE_DRAW stream check"); if not, same distribution, other stream.
    "VLLM_TPU_VERIFY_ONE_DRAW": env_bool("VLLM_TPU_VERIFY_ONE_DRAW"),
    # With VLLM_TPU_DRAFT_SEED_FUSED: the fused seed also emits step 0's
    # draft_rows and the loop's padded rejected counts (no eager pads).
    "VLLM_TPU_DRAFT_SEED_EXTRAS": env_bool("VLLM_TPU_DRAFT_SEED_EXTRAS"),
    # Prefill-bootstrap draft seed scatter at the loop bucket width (a
    # precompiled key) instead of the real request count (mid-serve compiles).
    "VLLM_TPU_DRAFT_SEED_BOOTSTRAP_PAD": env_bool("VLLM_TPU_DRAFT_SEED_BOOTSTRAP_PAD"),
    "VLLM_TPU_CPU_PIN_RESERVE": env_nonnegative_int("VLLM_TPU_CPU_PIN_RESERVE", 2),
    # Log each worker's thread CPU placement every N seconds (0 = off).
    "VLLM_TPU_CPU_PROBE_S": env_float("VLLM_TPU_CPU_PROBE_S", 0.0),
    # With VLLM_TPU_CPU_PIN on: also move the worker's private pages to its
    # chip's NUMA node (migrate_pages) and prefer that node for the step
    # thread's new allocations. Pinning happens after warmup, so the heap
    # built during load/warmup can sit on the far socket.
    "VLLM_TPU_CPU_PIN_MIGRATE": env_bool("VLLM_TPU_CPU_PIN_MIGRATE"),
    # Memoize the padded per-request sampling tensors (temperature/top-k/
    # top-p uploads) and the top-k candidate check on the requests' host
    # sampling values; steady decode skips ~10 CPU tensor ops per verify.
    "VLLM_TPU_SAMPLING_PARAMS_MEMO": env_bool("VLLM_TPU_SAMPLING_PARAMS_MEMO"),
    # Reuse the compact mamba_state_indices upload while the chunk's slot
    # list repeats (target prep and each draft forward rebuild it).
    "VLLM_TPU_MAMBA_INDICES_REUSE": env_bool("VLLM_TPU_MAMBA_INDICES_REUSE"),
    # "g0,g1[,g2]": the worker's gc.set_threshold after warmup (empty = keep
    # CPython's 700,10,10). A step allocates thousands of container objects,
    # so the default runs several young-generation passes per step;
    # HOST_PHASES reports them as x_gc0/1/2.
    "VLLM_TPU_GC_THRESHOLD": env_str("VLLM_TPU_GC_THRESHOLD", ""),
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
    # With UPLOAD_REUSE: re-pack only the block tables that changed since the
    # previous pack of the same layout (per-table versions).
    "VLLM_TPU_BLOCK_TABLE_PACK_DELTA": env_bool("VLLM_TPU_BLOCK_TABLE_PACK_DELTA"),
    # Skip the token-padding mask's synchronous H2D when its rows are the same
    # as the last write (layers/adapter/token_padding.py).
    "VLLM_TPU_TOKEN_PADDING_REUSE": env_bool("VLLM_TPU_TOKEN_PADDING_REUSE"),
    # Steady-decode forward plumbing per padded bucket: the token-padding mask
    # becomes one owned device buffer per (valid, padded) rows key, swapped in
    # instead of rebuilt and copied synchronously each step, and the whole-step
    # megakernel call's contexts are resolved once (token_padding.py,
    # qwen38_driver.py). Same values, same graph inputs.
    "VLLM_TPU_PREFLAT_FWD_ARGS": env_bool("VLLM_TPU_PREFLAT_FWD_ARGS"),
    # Qwen3.8 megakernel try_forward host bookkeeping per step: the real-traffic
    # gate is read only when a shadow check can use it, the whole-step context
    # check runs once per step instead of twice, and the step-mix total is a
    # running count. Same decisions, counters, logs and outputs
    # (qwen38_driver.py).
    "VLLM_TPU_MK_TRYFWD_LEAN": env_bool("VLLM_TPU_MK_TRYFWD_LEAN"),
    # Staged metadata walk: build every group's AttentionMetadata inline instead
    # of one builder call per group (tpu_runner._build_attention_metadata_staged).
    "VLLM_TPU_FAST_METADATA_BUILD": env_bool("VLLM_TPU_FAST_METADATA_BUILD"),
    # Steady decode (same requests, order and per-request token counts as the
    # previous step, no prefill piece): reuse the previous step's batch-shape
    # host arrays and index uploads, build the staged walk inline, and keep the
    # walk's mamba indices while no request can cross a mamba block
    # (runner/steady_decode_metadata.py). Any other step rebuilds as before.
    "VLLM_TPU_STEADY_DECODE_METADATA": env_bool("VLLM_TPU_STEADY_DECODE_METADATA"),
    # Freshly AOT-compiled models skip the per-call Dynamo guard check when
    # vLLM does not evaluate guards (as it already does for cached artifacts).
    "VLLM_TPU_AOT_SKIP_GUARD_CHECK": env_bool("VLLM_TPU_AOT_SKIP_GUARD_CHECK"),
    # Host-path pieces (q38f-int: default off, toggled per arm).
    # Record stage-timing device events only when a KV connector consumes them.
    "VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK": env_bool(
        "VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK", default=False
    ),
    # Staged walk: one gather per mamba geometry class instead of per group.
    "VLLM_TPU_STAGED_WALK_CLASS_GATHER": env_bool(
        "VLLM_TPU_STAGED_WALK_CLASS_GATHER", default=False
    ),
    # With VLLM_TPU_STAGED_WALK_CLASS_GATHER: the staged walk returns each
    # block-geometry class's mamba state / checkpoint ids as one [G, n] /
    # [G, n, W] stack (2 outputs per class instead of 2 per mamba group);
    # each group's metadata reads its row in-graph
    # (StackedMambaAttentionMetadata). Same values.
    "VLLM_TPU_STAGED_WALK_STACKED": env_bool("VLLM_TPU_STAGED_WALK_STACKED"),
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
    # With TPU_SAMPLER_TOPK_CANDIDATES in use: run the non-greedy rejection
    # sampler on the candidate logits only, instead of masking, normalizing
    # and arg-maxing full-vocab rows. Same uniforms; probabilities can differ
    # in the last bit (softmax normalizer summed in another order).
    "VLLM_TPU_SAMPLER_CANDIDATES_ONLY": env_bool("VLLM_TPU_SAMPLER_CANDIDATES_ONLY"),
    # GDN v3 static tiling: round the mixed tile down to a power of two so a
    # 24/40-row bucket stays divisible by the triangular solve's block, and
    # assert that divisibility (PR #2821 ece3d83b5). Dynamic tiling
    # (TPU_ENABLE_GDN_DYNAMIC_TILING) already picks powers of two.
    "VLLM_TPU_GDN_MIXED_TILE_POW2": env_bool("VLLM_TPU_GDN_MIXED_TILE_POW2"),
    # MTP + prefix caching (mamba align mode): after sampling, copy the GDN
    # state at a mamba block boundary that a verify window reached into the
    # block the boundary completes. Without it that block keeps checkpoint 0
    # of the last window that ended inside it, a few tokens short of the
    # boundary, and a later prefix hit resumes from that state (PR #2821
    # c5b1c7b67). Only on steps with blocks hashed (prefix caching or a KV
    # connector).
    "VLLM_TPU_SPEC_BOUNDARY_STATE_COPY": env_bool(
        "VLLM_TPU_SPEC_BOUNDARY_STATE_COPY"
    ),
    # GDN v3 pooled verify windows (MTP, BATCHED tiles of 2+ tokens): run
    # kernels/gdn/v3/window_tile.py, one window position and one value head
    # at a time on dense [dim / 128, 128] activations, instead of
    # wrapper.inner_kernel's whole-state arrays (PR #2821 d8340b973). Same
    # outputs and checkpoint bytes; pools window_tile does not support keep
    # inner_kernel.
    "VLLM_TPU_GDN_V3_WINDOW_VERIFY": env_bool("VLLM_TPU_GDN_V3_WINDOW_VERIFY"),
    # GDN v3 pooled calls: run the BATCHED and the PER_SEQ segments in one
    # kernel instead of two, so a decode/verify-only step launches no empty
    # PER_SEQ kernel (PR #2821 d8340b973). Same outputs and pool bytes.
    "VLLM_TPU_GDN_V3_SINGLE_LAUNCH": env_bool("VLLM_TPU_GDN_V3_SINGLE_LAUNCH"),
    # Non-greedy MTP verify steps whose requests all keep top_k <= 48 (or are
    # greedy) keep each rank's LM-head vocab shard instead of all-gathering
    # the logits: one program takes each shard's 64 largest logits per row,
    # all-gathers those (64 x TP per row) and runs temperature, top-k/top-p,
    # the accept test and recovery on them with one uniform per candidate; a
    # row whose candidates miss a kept token falls back to the gathered
    # vocabulary in the same program. Same distributions, different draws
    # (PR #2821 049bb029b). Takes precedence over VLLM_TPU_VERIFY_FUSED and
    # TPU_SAMPLER_TOPK_CANDIDATES on the steps it serves; logprob, grammar
    # and all-greedy steps keep their paths.
    "VLLM_TPU_CANDIDATE_SAMPLING": env_bool("VLLM_TPU_CANDIDATE_SAMPLING"),
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
    # Reuse the KV byte budget an earlier start with the same compile hash
    # recorded (under VLLM_CACHE_ROOT) when it fits and is within this
    # fraction of the measured budget, so the KV block count, and with it the
    # backbone compile hash, is stable across restarts and a restored compile
    # cache is actually read. 0 disables (see worker/kv_budget_pin.py).
    "VLLM_TPU_KV_BUDGET_PIN_TOLERANCE": env_float(
        "VLLM_TPU_KV_BUDGET_PIN_TOLERANCE", 0.01
    ),
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
    "VLLM_TPU_MK_OPERAND_DUMP": env_bool("VLLM_TPU_MK_OPERAND_DUMP", False),
    "VLLM_TPU_MK_STEP_MIX_INTERVAL_S": env_float(
        "VLLM_TPU_MK_STEP_MIX_INTERVAL_S", 10.0
    ),
    "CDK_OUTPUT_DIR": env_str("CDK_OUTPUT_DIR", "/tmp/cdk_output"),
    "QWEN4_EXP_MAX_POOL_BYTES": env_optional_int("QWEN4_EXP_MAX_POOL_BYTES"),
    "QWEN4_EXP_POOL_SIZING": env_str("QWEN4_EXP_POOL_SIZING", "hbm"),
    "QWEN4_EXP_POOL_GRAPH_COPIES": env_int("QWEN4_EXP_POOL_GRAPH_COPIES", 0),
    "QWEN4_EXP_POOL_HEADROOM_BYTES": env_int(
        "QWEN4_EXP_POOL_HEADROOM_BYTES", 6 * 1024**3
    ),
    "QWEN4_EXP_PACK_COMPRESSED_K": env_bool(
        "QWEN4_EXP_PACK_COMPRESSED_K", default=False
    ),
    "QWEN4_EXP_PACK_MAIN_KV": env_bool("QWEN4_EXP_PACK_MAIN_KV", default=False),
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
