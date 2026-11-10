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
    TPU_RUNNER_STAGE_TIMING: bool = False
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
    TPU_RAIDEN_STORE_DISPATCH_WAIT_S: float = 600.0
    TPU_RAIDEN_PREFIX_AWARE_LOAD: bool = False
    USE_MOE_SPARSE_CORE: bool = True
    ONEHOT_MOE_PERMUTE_THRESHOLD: int | None = None
    TPU_MOE_SEQUENCE_PARALLEL: bool = False
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    TPU_MOE_DENSE_EXPERT_THRESHOLD: int = 0
    FUSED_MOE_DECODE_KERNEL: bool = False
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: int = 256
    TPU_MOE_SINGLE_SORT: bool = True
    TPU_MOE_LOWTOK: bool = True
    TPU_MOE_LOWTOK_FROM_SCORES: bool = True
    TPU_MOE_LOWTOK_IDS: bool = True
    TPU_MOE_LOWTOK_MAX_TOKENS: int = 256
    TPU_MOE_LOWTOK_FUSED: bool = False
    TPU_MOE_LOWTOK_FUSED_DISPATCH_FORM: str = "hier"
    TPU_MOE_LOWTOK_FUSED_COMBINE_FORM: str = "hier"
    TPU_MOE_LOWTOK_FUSED_FP8_DISPATCH: bool = False
    TPU_MOE_LOWTOK_RING_DEPTH: int = 1
    TPU_MOE_LOWTOK_GATHER: bool = False
    TPU_MOE_NONEP_GATHER: str = "xla"
    TPU_MOE_NONEP_GATHER_CHUNK: int = 16384
    TPU_MOE_NONEP_GATHER_COLS: int = 1024
    TPU_MOE_NONEP_COMBINE: str = "sc"
    TPU_MOE_NONEP_PERMUTE: str = "sorts"
    TPU_SC_COMBINE_BENCH: str = ""
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
    TPU_ENABLE_GDN_DYNAMIC_TILING: bool = False
    TPU_GDN_PREFILL_DOT_PRECISION: str = ""
    TPU_GDN_PREFILL_CHUNK: int = 0
    TPU_GDN_PREFILL_HEAD_GROUP: int = 0
    TPU_GDN_PREFILL_TINV_BLOCK: int = 0
    TPU_GDN_PREFILL_DENSE_CONV: bool = False
    TPU_GDN_PREFILL_VMEM_FRACTION: float = 0.0
    TPU_GDN_PREFILL_TINV_MODE: str = ""
    TPU_GDN_PREFILL_SHARED_KKT: bool = False
    TPU_GDN_PREFILL_LAZY_CONV_STATE: bool = False
    TPU_MOE_SKIP_PADDED_TOKENS: bool = False
    TPU_RPA_NATIVE_FP8: str = ""
    TPU_RPA_NATIVE_FP8_PREFILL: str = ""
    TPU_RPA_PREFILL_BLOCKS: str = ""
    TPU_PCP_HISTORY_RPA: bool = False
    TPU_PCP_CURRENT_RPA: bool = False
    TPU_PCP_ENTRY_BARRIER: bool = False
    TPU_MOE_HIERARCHICAL_EP: bool = False
    TPU_MOE_ROUTER_TOPK: str = "rowmax"
    USE_MOE_FUSED_EP_KERNEL: bool = False
    MOE_FUSED_EP_KERNEL_IMPL: str | None = None
    MOE_FUSED_EP_ENABLE_W4A8: bool = False
    MOE_FUSED_EP_KERNEL_MIN_TOKENS: int = 1024
    MOE_FUSED_EP_V2_SHARDED_PLAN: bool = False
    MOE_FUSED_EP_SORTED_GATHER: bool = False
    MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE: int = 0
    MOE_FUSED_EP_V2_TILE_M: int = 0
    MOE_FUSED_EP_V2_FORCE_ONE_HOP: bool = False
    MOE_FUSED_EP_V2_SCOPES: int = 0
    MOE_FUSED_EP_V2_TEST_MUTANT: str = ""
    MOE_FUSED_EP_V2_INTERPRET: bool = False
    MOE_FUSED_EP_V2_GATHER_LOOKAHEAD: bool = False
    MOE_FUSED_EP_V2_SORTED_PLAN: bool = False
    MOE_FUSED_EP_V2_PLAN_KERNEL: bool = False
    MOE_FUSED_EP_V2_QUANT_KERNEL: bool = False
    MOE_FUSED_EP_V2_ROUTE_KERNEL: bool = False
    MOE_FUSED_EP_V2_COMBINE_TOKENS: int = 0
    MOE_FUSED_EP_V2_COMBINE_LEAD: int = 0
    MOE_FUSED_EP_V2_COMBINE_PARITIES: int = 0
    MOE_FUSED_EP_V2_FLAT_HANDOFF: int = 0
    MOE_FUSED_EP_V2_AG_IN_PLAN: bool = False
    MOE_FUSED_EP_V2_PAIR_SCALE: bool = False
    MOE_FUSED_EP_V2_PLAN_CUMSUM_DOT: bool = False
    MOE_FUSED_EP_V2_PREQUANT: bool = False
    MOE_FUSED_EP_V2_STAGE_SCALES: bool = False
    MOE_FUSED_EP_V2_SLAB_BASE_T: bool = False
    TPU_LATENT_PROJ_INTRA_CHIP_TP: bool = False
    TPU_TOKEN_BUCKET_EXTRA: list[int] = []
    TPU_TOKEN_BUCKET_HALF_STEPS: bool = False
    TPU_OFFLOAD_MIN_HBM_USAGE: float = 0.0
    TPU_EXPERT_PLACEMENT_MTP: bool = True
    TPU_TOKEN_BUCKET_LINEAR: bool = False
    TPU_PP_DYNAMIC_CHUNKS: bool = True
    TPU_PP_CHUNK_SLACK: float = 0.1
    TPU_PP_LAZY_PUSH: bool = False
    TPU_PP_EXTRA_STEPS_IN_FLIGHT: int = 0
    TPU_PREFILL_ONLY_STATIC_GUARDS: bool = False
    TPU_PP_HANDOFF: str = "wave"
    TPU_PP_RING_DEPTH: int = 2
    TPU_ROPE_CACHE_TRUNCATE: bool = False
    TPU_ROPE_CACHE_ROW_MAJOR: bool = False
    TPU_MOE_HASH_TABLE_ROW_MAJOR: bool = False
    TPU_GDN_FUSED_NORM_QUANT: bool = True
    TPU_FUSED_SE_GATE_FRESH_BUFFER: bool = False
    TPU_SAMPLER_TOPK_PALLAS: bool = False
    TPU_SAMPLER_TOPK_MAX_K: int = 0
    TPU_FUSE_PRE_ATTN_NORM_QUANT: bool = True
    TPU_FUSE_POST_ATTN_NORM_MOE: bool = True
    TPU_SE_PREQUANT_KERNEL: int = 0
    TPU_SE_PREQUANT_VMEM_MIB: int = 0
    TPU_SE_PREQUANT_MUSTFUSE: bool = False
    TPU_GDN_FUSED_NORM_QUANT_MIN_ROWS: int = 16
    TPU_FUSED_QK_NORM_ROPE: bool = True
    TPU_FUSED_QK_NORM_ROPE_MIN_ROWS: int = 0
    TPU_QK_NORM_ROPE_INPLACE_TILE: int = 0
    TPU_QK_NORM_ROPE_Q_PACKED: bool = False
    TPU_QMM_ACT_F32_AMAX: bool = False
    TPU_PCP_Q_PACK_OUTSIDE: bool = False
    TPU_RPA_DECODE_BATCHED: bool = False
    TPU_RPA_DECODE_LANES: int = 0
    TPU_RPA_DECODE_V2: bool = False
    TPU_RPA_KV_SPLIT: bool = False
    TPU_PARALLEL_PRECOMPILE: bool = False
    TPU_PRECOMPILE_BIGGEST_FIRST: bool = False
    TPU_PRECOMPILE_PEER_REPLAY: bool = False
    TPU_WORKER_NUMA_BIND: bool = True
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
    TPU_EXPERT_PLACEMENT: str = ""
    TPU_PCP_TOPOLOGY_AWARE_MESH: bool = True
    USE_RPA_PIPELINED_DMA_STAGING: bool = False
    USE_PIPELINED_DMA_STAGING: bool = False
    VLLM_TPU_BLOCK_MAJOR_KV: bool = False
    TPU_EVICT_WEIGHTS_PAGE_CACHE: bool = False
    TPU_TCMALLOC_RELEASE_FREE_MEMORY: bool = False
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
    TPU_RAIDEN_ASYNC_COMPLETE: bool = False
    TPU_RAIDEN_COMPLETE_SLOW_MS: float = 100.0
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
    TPU_RAIDEN_STAGE3_SUBMIT_WORKERS: int = 1
    TPU_RAIDEN_STAGE3_MAX_INFLIGHT_LOADS: int = 0
    TPU_RAIDEN_STAGE3_HEAD_UNIQUE: bool = False
    TPU_RAIDEN_INLINE_LOAD: bool = False
    TPU_RAIDEN_STAGE3_LATE_POLL: bool = False
    TPU_RAIDEN_STAGE3_ASYNC_REGISTER: bool = False
    TPU_RAIDEN_STAGE3_ASYNC_REGISTER_DELAY_MS: float = 10.0
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
    TPU_DRAFT_LOAD_BARRIER: bool = True
    TPU_DRAFT_LOAD_BARRIER_TIMEOUT_S: int | None = None
    TPU_DIST_CPU_TIMEOUT_S: int | None = None
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
    TPU_HYBRID_STATE_RETENTION: bool | None = None
    TPU_HYBRID_RETENTION_MAX_CONVS: int = 96
    TPU_HYBRID_RETENTION_MAX_PINNED_BLOCKS: int = 360
    TPU_HYBRID_RETENTION_KEEP_PRIOR_TURNS: int = 1
    TPU_HYBRID_RETENTION_SMART_FORK: bool = True
    TPU_HYBRID_RETENTION_SAFE_TAIL_OFFSET: int = 4
    TPU_RAIDEN_INLINE_LOAD_MAX_CHUNKS: int = 2
    VLLM_TORCHTPU_IPC_KEY: str = ""
    TPU_SHARDED_LOAD_SYNC_EVERY: int = 512
    VLLM_TPU_MXFP4_DEVICE_DEQUANT: bool = True
    VLLM_TPU_DEBUG_PCP_LAYOUT: bool = False
    TPU_LOCAL_RANK_OFFSET: int = 0
    VLLM_TPU_DEVICE_ORDER: str = ""
    DEBUG_TPU_LOCAL_RANK_OFFSET: int = 0
    TORCH_TPU_BASE_PORT: int = 8070
    TORCH_TPU_MP_RENDEZVOUS_PORT: int | None = None
    TPU_MULTIPORT_DP_LOCAL_SIZE: int = 0
    VLLM_TPU_FAST_TOKEN_SUBSTITUTION: bool = False
    VLLM_TPU_FUSED_DRAFT_LOOP: bool = False
    TPU_MTP_STEP0_POST_ATTN_GATHER: bool = True
    TPU_PREFILL_SKIP_VERIFY_WARMUP: bool = False
    TPU_SAMPLER_TOPK_CANDIDATES: int = 0
    MM_ENCODER_FRAME_CHUNK_PATCH_SIZE: int = 0
    VLLM_TPU_STATIC_POOLER: bool = True
    VLLM_TPU_POOLING_KV_CACHE_TOKENS: int = -1
    TPU_SMALL_ALL_REDUCE: bool = True
    TPU_SMALL_ALL_REDUCE_TP: bool = False
    TPU_SMALL_ALL_REDUCE_TP_FORM: str = ""
    TPU_DP_PACKED_ROUTER_GATHER: bool = False
    TPU_SMALL_ALL_REDUCE_MAX_ROWS: int = 128
    TPU_ATTN_MOE_OVERLAP: bool = False
    TPU_ATTN_MOE_OVERLAP_MIN_TOKENS: int = 64
    TPU_MOE_FUSED_EP_STEP_MIN_TOKENS: int = 64
    TPU_MOE_FUSED_EP_V3_CONFIG: str = ""
    TPU_MOE_FUSED_EP_V3_TILE_ROWS: int = 128
    TPU_SP_MIN_ROWS: int = 512
    TPU_SP_FP8_GATHER: bool = True
    TPU_SCHED_SRPF_RUNNING: bool = True
    TPU_SCHED_COLD_CHUNK_CAP: int = 2048
    TPU_SCHED_COLD_THRESHOLD: int = 8192
    TPU_SCHED_COLD_MIN_DECODERS: int = 12
    TPU_SCHED_STEP_TOTAL_CAP: int = 4096
    TPU_SCHED_WARM_STREAK: int = 0
    TPU_SCHED_WARM_MIN_DECODERS: int = 24
    TPU_SCHED_MK_MAX_RUNNING: int = 0
    TPU_SCHED_KV_PREFETCH_WAITING: int = 0
    TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS: int = 512
    TPU_SCHED_PAW_SORT_WAITING: bool = False
    TPU_SCHED_PAW_SORT_WINDOW: int = 128
    TPU_SCHED_SKIP_MIDBLOCK_ALIGN_STOP: bool = False
    TPU_SCHED_SINGLE_STEP_PREFILL: bool = True
    TPU_SCHED_FINISH_DECODE_STEPS: int = 64
    TPU_SCHED_FINISH_DECODE_MAX_DEFERRAL_STEPS: int = 64
    TPU_SCHED_FINISH_DECODE_PREFILL_MIN_TOKENS: int = 1024
    QX_TTFT_STAGE_LOG_PERIOD_S: float = 0.0
    VLLM_TPU_LIGHTWEIGHT_STEP_STATS: bool = False
    VLLM_TPU_LIGHTWEIGHT_STEP_STATS_INTERVAL: int = 500
    VLLM_TPU_RAIDEN_LOAD_WAIT_MS: float = 1.5
    VLLM_TPU_QWEN35_MEGAKERNEL: bool = False
    VLLM_TPU_QWEN35_MK_WIDE_MB: int = 0
    VLLM_TPU_QWEN35_MK_WIDE_MIN_ROWS: int = 160
    VLLM_TPU_QWEN35_MK_DCP_KV_HEADROOM: bool = False
    TPU_RAIDEN_PP_MAX_CPU_BYTES: int | None = None


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
    # Size GDN v3 decode tiles and prefill chunks from per-rank heads, active
    # sequences and the VMEM estimate (kernels/gdn/v3/tiling.py) instead of
    # the fixed defaults.
    "TPU_ENABLE_GDN_DYNAMIC_TILING": env_bool(
        "TPU_ENABLE_GDN_DYNAMIC_TILING", default=False
    ),
    # GDN v3 prefill (PER_SEQ chunked) kernel knobs, all default off.
    # Operand precision of the chunked path's MXU dots: "" keeps the f32
    # operands at Mosaic's default contraction, "bf16" rounds the operands to
    # bf16 (f32 accumulation), "fp32" asks for Precision.HIGHEST.
    "TPU_GDN_PREFILL_DOT_PRECISION": env_str("TPU_GDN_PREFILL_DOT_PRECISION", ""),
    # Tokens per PER_SEQ chunk, bypassing the VMEM-fit cap when positive.
    "TPU_GDN_PREFILL_CHUNK": env_nonnegative_int("TPU_GDN_PREFILL_CHUNK", 0),
    # Value heads per chunked-GDN head group (0: all heads in one group).
    "TPU_GDN_PREFILL_HEAD_GROUP": env_nonnegative_int("TPU_GDN_PREFILL_HEAD_GROUP", 0),
    # Sub-block size of the chunked GDN triangular solve (0: the config's
    # head-count-derived default).
    "TPU_GDN_PREFILL_TINV_BLOCK": env_nonnegative_int("TPU_GDN_PREFILL_TINV_BLOCK", 0),
    # PER_SEQ tiles with 2D bf16 qkv run the unpack, conv, SiLU and head
    # split on token-dense [chunk, dim] rows instead of the one-row-per-vreg
    # compact layout.
    "TPU_GDN_PREFILL_DENSE_CONV": env_bool("TPU_GDN_PREFILL_DENSE_CONV", default=False),
    # Scoped-VMEM fraction of the PER_SEQ pallas_call (0: GDNConfig default).
    "TPU_GDN_PREFILL_VMEM_FRACTION": env_float("TPU_GDN_PREFILL_VMEM_FRACTION", 0.0),
    # Triangular solve of the chunked GDN: "" is the blocked forward
    # substitution, "doubling" the nilpotent-series product
    # (I - L)(I + L^2)(I + L^4)... on the MXU.
    "TPU_GDN_PREFILL_TINV_MODE": env_str("TPU_GDN_PREFILL_TINV_MODE", ""),
    # Chunked GDN computes k k^T once per kq head and scales the rows by
    # beta per value head, instead of one (beta k) k^T dot per value head.
    "TPU_GDN_PREFILL_SHARED_KKT": env_bool("TPU_GDN_PREFILL_SHARED_KKT", default=False),
    # PER_SEQ chunked tiles take the conv state from the static tail rows and
    # run the per-row masked select only on a partial last tile.
    "TPU_GDN_PREFILL_LAZY_CONV_STATE": env_bool(
        "TPU_GDN_PREFILL_LAZY_CONV_STATE", default=False
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
    "PYTHON_TRACER_LEVEL": lambda: int(
        os.getenv("PYTHON_TRACER_LEVEL") if os.getenv("PYTHON_TRACER_LEVEL") is not None else "1"
    ),
    "TPU_RUNNER_STAGE_TIMING": lambda: os.getenv(
        "TPU_RUNNER_STAGE_TIMING", "0"
    ).strip().lower() not in ("0", "false", "off", "no"),
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
    # How long a worker waits for rank 0's reshard store dispatch endpoint
    # at register_runner. The ranks of one engine reach that step up to a
    # minute apart on a loaded host, and only rank 0 hosts the store.
    "TPU_RAIDEN_STORE_DISPATCH_WAIT_S": lambda: float(
        os.getenv("TPU_RAIDEN_STORE_DISPATCH_WAIT_S") or "600"
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
    # Build the MoE permutation, its inverse, and the sorted token ids from
    # one packed sort instead of three. The ragged (counting sort) path ignores it.
    "TPU_MOE_SINGLE_SORT": env_bool("TPU_MOE_SINGLE_SORT", True),
    # Experimental small-batch MoE (kernels/experimental/moe_lowtok) for EP
    # calls of at most TPU_MOE_LOWTOK_MAX_TOKENS rows; the FP8 method also moves
    # a plain softmax top-k router into it. Can be tiered with dense expert MoE
    # when TPU_MOE_DENSE_EXPERT_THRESHOLD > TPU_MOE_LOWTOK_MAX_TOKENS.
    "TPU_MOE_LOWTOK": env_bool("TPU_MOE_LOWTOK", default=True),
    "TPU_MOE_LOWTOK_FROM_SCORES": env_bool("TPU_MOE_LOWTOK_FROM_SCORES", default=True),
    "TPU_MOE_LOWTOK_IDS": env_bool("TPU_MOE_LOWTOK_IDS", default=True),
    "TPU_MOE_LOWTOK_MAX_TOKENS": env_nonnegative_int("TPU_MOE_LOWTOK_MAX_TOKENS", 256),
    # Run the DP dispatch and the EP combine of a replicated few-token MoE step
    # inside the moe_lowtok kernel as remote DMAs (kernels/experimental/
    # moe_lowtok/fused_collectives.py), on a DP x TP layout whose TP pairs are
    # the mesh's consecutive cores. The forms are "flat" or "hier"; the fp8
    # dispatch pushes rows quantized per row with their scale; the ring depth
    # is the number of experts the weight ring holds.
    "TPU_MOE_LOWTOK_FUSED": env_bool("TPU_MOE_LOWTOK_FUSED", default=False),
    "TPU_MOE_LOWTOK_FUSED_DISPATCH_FORM": env_str("TPU_MOE_LOWTOK_FUSED_DISPATCH_FORM", "hier"),
    "TPU_MOE_LOWTOK_FUSED_COMBINE_FORM": env_str("TPU_MOE_LOWTOK_FUSED_COMBINE_FORM", "hier"),
    "TPU_MOE_LOWTOK_FUSED_FP8_DISPATCH": env_bool(
        "TPU_MOE_LOWTOK_FUSED_FP8_DISPATCH", default=False
    ),
    "TPU_MOE_LOWTOK_RING_DEPTH": env_nonnegative_int("TPU_MOE_LOWTOK_RING_DEPTH", 1),
    # With TPU_MOE_LOWTOK_FUSED, each expert runs on its routed rows only
    # (gathered and scattered back by one-hot matmuls inside the kernel)
    # instead of on every gathered row.
    "TPU_MOE_LOWTOK_GATHER": env_bool("TPU_MOE_LOWTOK_GATHER", default=False),
    # Token movement of the whole-expert (non-EP) grouped MoE path, as the
    # PP8xTP1 prefill stages run it. Gather of the routed rows before the
    # expert GMM: "xla" is hidden_states[token_indices_sorted] (XLA decides
    # between a TensorCore gather fusion and the SparseCore gather offload),
    # "xla_chunked" the same gather issued in pieces of
    # TPU_MOE_NONEP_GATHER_CHUNK rows (each piece small enough for XLA's
    # TensorCore gather fusion), "xla_colchunk" the gather done per block of
    # TPU_MOE_NONEP_GATHER_COLS columns (each block's source table small
    # enough to sit in VMEM for XLA's TensorCore gather fusion), "sc" the
    # sc_ragged_gather_v2 kernel, "tc" a TensorCore Pallas kernel that copies
    # one row per DMA (kernels/experimental/moe_tc_permute.py).
    "TPU_MOE_NONEP_GATHER": env_with_choices(
        "TPU_MOE_NONEP_GATHER",
        "xla",
        ["xla", "xla_chunked", "xla_colchunk", "sc", "tc"],
        case_sensitive=False,
    ),
    "TPU_MOE_NONEP_GATHER_CHUNK": env_int("TPU_MOE_NONEP_GATHER_CHUNK", 16384),
    "TPU_MOE_NONEP_GATHER_COLS": env_int("TPU_MOE_NONEP_GATHER_COLS", 1024),
    # Weighted reduction of the expert outputs back to tokens: "sc" is the
    # destination-major SparseCore kernel sc_ragged_gather_reduce_v3, "sc_v2"
    # the v2 kernel, "xla" a gather and a weighted sum in XLA (f32 products),
    # "xla_kmajor" one gather of num_tokens rows per route slot k summed in
    # f32 in slot order, "xla_scatter" an f32 scatter-add indexed by the
    # sorted token ids, "tc" the TensorCore Pallas kernel of
    # kernels/experimental/moe_tc_permute.py.
    "TPU_MOE_NONEP_COMBINE": env_with_choices(
        "TPU_MOE_NONEP_COMBINE",
        "sc",
        ["sc", "sc_v2", "xla", "xla_kmajor", "xla_scatter", "tc"],
        case_sensitive=False,
    ),
    # How the permutation metadata is built above TPU_MOE_SINGLE_SORT's row
    # cap: "sorts" runs three sorts (argsort, its inverse, the packed token
    # ids), "two_sorts" derives the sorted token ids from the argsort by
    # division, "counting" takes the inverse permutation and the group sizes
    # from the one-hot prefix-sum counting sort (block 1024) and keeps one
    # argsort for the forward permutation.
    "TPU_MOE_NONEP_PERMUTE": env_with_choices(
        "TPU_MOE_NONEP_PERMUTE",
        "sorts",
        ["sorts", "two_sorts", "counting"],
        case_sensitive=False,
    ),
    # Bench-only knob of sc_ragged_gather_reduce_v3: "skip_math" issues the
    # row gathers and skips the scatter-adds, "half_cols" runs the scatter-adds
    # on the first half of each column partition. Must stay unset in serving.
    "TPU_SC_COMBINE_BENCH": env_str("TPU_SC_COMBINE_BENCH", ""),
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
    "TPU_GREEDY_TOP_TOKENS": env_bool("TPU_GREEDY_TOP_TOKENS", default=True),
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
    "TPU_MOE_SKIP_PADDED_TOKENS": env_bool("TPU_MOE_SKIP_PADDED_TOKENS", default=True),
    # Runs the batched RPA prefill / mixed kernel's QK and/or PV matmul with
    # fp8 operands against the fp8 KV cache: "qk", "pv" or "qkpv", each
    # operand optionally suffixed "s" for its static scale (e.g. "qkpvs").
    # Empty: bf16 matmuls. See kernels/experimental/batched_rpa/configs.py.
    # Native FP8 QK/PV in the batched RPA decode kernel ("qk", "pv", "pvs", ...);
    # the wrapper reads it directly. Target model only (native_fp8_scope).
    "TPU_RPA_NATIVE_FP8": lambda: os.getenv("TPU_RPA_NATIVE_FP8", ""),
    "TPU_RPA_NATIVE_FP8_PREFILL": lambda: os.getenv("TPU_RPA_NATIVE_FP8_PREFILL", ""),
    # "bq,bq_c,bkv,batch": replaces the batched RPA prefill/mixed tile sizes that
    # calculate_block_sizes picks (empty: the heuristic). See wrapper.prefill_block_override.
    "TPU_RPA_PREFILL_BLOCKS": lambda: os.getenv("TPU_RPA_PREFILL_BLOCKS", ""),
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
    # "v3" selects the fused expert-parallel kernel v3; with
    # TPU_MOE_LOWTOK_MAX_TOKENS=256 and TPU_SP_MIN_ROWS=512, steps <= 256 run
    # moe_lowtok and steps >= 512 run SP + fused EP v3.
    "MOE_FUSED_EP_KERNEL_IMPL": env_with_choices(
        "MOE_FUSED_EP_KERNEL_IMPL", None, ["v2", "adaptive", "v3"]
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
    # The tile height (and so the ragged-stride unit) the serving adapter
    # passes fused EP MoE v2 as its capacity. Zero keeps the adapter's 128.
    # The tail-tile height rungs and push_from_vmem are built only while the
    # shard slab bound fits ep * local experts * tile height, which a larger
    # tile keeps true at larger steps (256 covers 8,704-token PCP steps).
    # A power of two from 128 to 512; anything else refuses at op build.
    "MOE_FUSED_EP_V2_TILE_M": lambda: int(
        os.getenv("MOE_FUSED_EP_V2_TILE_M", "0") or 0
    ),
    # Bench-only: build fused EP MoE v2 with the height rungs and
    # push_from_vmem whatever the slab bound says (both are exact for any
    # tile count; the bound is a performance choice). Read at BUILD time.
    "MOE_FUSED_EP_V2_FORCE_ONE_HOP": env_bool("MOE_FUSED_EP_V2_FORCE_ONE_HOP"),
    # Named device scopes inside the fused EP MoE v2 kernel body (0 = off,
    # 1 = phases). Visible on the XLA TraceMe line only with
    # --xla_enable_custom_call_region_trace=true. Read at BUILD time; a
    # scoped build is not a numerics reference.
    "MOE_FUSED_EP_V2_SCOPES": lambda: int(
        os.getenv("MOE_FUSED_EP_V2_SCOPES", "0") or 0
    ),
    # Test-only fused EP MoE v2 build mutants (default empty = none). Read at
    # BUILD time. "mirror_stride": a tile above 128 rows addresses its scale
    # mirror rows with stride 1 instead of tile_m / 128, so scales land on
    # the wrong rows; numerics must fail. "la_scale" / "la_window" (with
    # MOE_FUSED_EP_V2_GATHER_LOOKAHEAD): the next expert's first tile takes
    # the current expert's activation-scale window / reads its token rows
    # through the wrong index window. (Same bytes, so no semaphore hangs.)
    "MOE_FUSED_EP_V2_TEST_MUTANT": env_str("MOE_FUSED_EP_V2_TEST_MUTANT", ""),
    # Test-only: build the fused EP MoE v2 kernel in Pallas' TPU interpret mode
    # (runs on the host). Read at BUILD time.
    "MOE_FUSED_EP_V2_INTERPRET": env_bool("MOE_FUSED_EP_V2_INTERPRET"),
    # Fused EP MoE v2: issue each expert's first-tile row gather at the head
    # of the previous expert's last tile (under its matmuls) instead of after
    # it. Read at BUILD time; default off.
    "MOE_FUSED_EP_V2_GATHER_LOOKAHEAD": env_bool("MOE_FUSED_EP_V2_GATHER_LOOKAHEAD"),
    # With MOE_FUSED_EP_V2_SHARDED_PLAN: build the shard's routing tables from
    # two stable sorts of the expert ids instead of one-hot rank passes
    # (host.build_routing_tables_sorted; same tables on every live pair).
    "MOE_FUSED_EP_V2_SORTED_PLAN": env_bool("MOE_FUSED_EP_V2_SORTED_PLAN"),
    # With MOE_FUSED_EP_V2_SHARDED_PLAN: build the routing plan's per-pair
    # tables (pos, mirror_pos, slab_row) with the Pallas plan kernel
    # (kernels/fused_moe/v2/plan_kernel.py), reading the gathered exchange in
    # place, instead of the XLA one-hot passes. Same tables on every live
    # pair. Read at TRACE time; default off; shapes it does not take fall back.
    "MOE_FUSED_EP_V2_PLAN_KERNEL": env_bool("MOE_FUSED_EP_V2_PLAN_KERNEL"),
    # Fused EP MoE v2: quantize the layer's activation rows with the Pallas
    # row-quant kernel (kernels/fused_moe/v2/quant_kernel.py: one pass, fp8
    # rows written in the all-gather's per-token tiles) instead of XLA's
    # abs-max / quantize / re-tile chain. Same bits. Read at TRACE time.
    "MOE_FUSED_EP_V2_QUANT_KERNEL": env_bool("MOE_FUSED_EP_V2_QUANT_KERNEL"),
    # Fused EP MoE v2: the router epilogue (top-k, row routes and the routing
    # exchange's per-expert count) as one Pallas pass over the softmax scores
    # (kernels/fused_moe/v2/route_kernel.py) instead of the row-major selector,
    # XLA's finite guard and the pair histogram. Same bits. Softmax scoring and
    # the plan kernel path only. Read at TRACE time; default off.
    "MOE_FUSED_EP_V2_ROUTE_KERNEL": env_bool("MOE_FUSED_EP_V2_ROUTE_KERNEL"),
    # Bench-only tuning of the fused EP MoE v2 arrival combine (0 = the kernel's
    # COMBINE_TOKENS / COMBINE_LEAD / COMBINE_PARITIES). Read at BUILD time.
    "MOE_FUSED_EP_V2_COMBINE_TOKENS": lambda: int(os.getenv("MOE_FUSED_EP_V2_COMBINE_TOKENS", "0") or 0),
    "MOE_FUSED_EP_V2_COMBINE_LEAD": lambda: int(os.getenv("MOE_FUSED_EP_V2_COMBINE_LEAD", "0") or 0),
    "MOE_FUSED_EP_V2_COMBINE_PARITIES": lambda: int(os.getenv("MOE_FUSED_EP_V2_COMBINE_PARITIES", "0") or 0),
    # Fused EP MoE v2 (plan-kernel path, fp8 wire, combine kernel): build the
    # routing hand-offs on flat, lane-dense pair tables instead of [tokens, topk]
    # ones. 1: the combine coefficient from a 1-D gather of the scale mirror.
    # 2: also the activation scale slab from the token-gather scatter (token + 1)
    # and a gather, instead of a repeat and a second scatter. Same bits on every
    # table the kernels read. Read at TRACE time; default 0 (off).
    "MOE_FUSED_EP_V2_FLAT_HANDOFF": lambda: int(os.getenv("MOE_FUSED_EP_V2_FLAT_HANDOFF", "0") or 0),
    # Fused EP MoE v2 (plan-kernel path, whole fused layer): the activation all-gather as DMAs at the
    # plan kernel's head, waited at its tail, instead of XLA's (synchronous) all-gather, so the rows
    # move under the plan's own compute. Same bits. Read at TRACE time; default off.
    "MOE_FUSED_EP_V2_AG_IN_PLAN": env_bool("MOE_FUSED_EP_V2_AG_IN_PLAN"),
    # Fused EP MoE v2 (plan-kernel path, fp8 wire): each routed pair's activation-scale bits for the
    # scale-slab scatter come from one exact one-hot matmul over the exchange's scale rows (byte planes),
    # already in the pairs' lane-dense order, instead of a repeat through a [tokens, topk] array and a
    # relayout. Same bits. Read at TRACE time; default off.
    "MOE_FUSED_EP_V2_PAIR_SCALE": env_bool("MOE_FUSED_EP_V2_PAIR_SCALE"),
    # Fused EP MoE v2 plan bases: the expert-axis prefix sums as exact HIGHEST-precision triangular
    # matmuls instead of XLA cumsums (reduce-window). Integer-exact below 2**24. Read at TRACE time; default off.
    "MOE_FUSED_EP_V2_PLAN_CUMSUM_DOT": env_bool("MOE_FUSED_EP_V2_PLAN_CUMSUM_DOT"),
    # Fused EP MoE v2 (MO3b, whole fused layer, fp8 wire): prebuild also builds the op variant that
    # takes the activation rows already quantized (q, row scale) from the post-attention norm/SE op
    # (fused_moe_ep.stage_fused_moe_ep_prequant), so the layer skips its own quant. The producer must
    # reproduce the in-program quant bit for bit. Read at WEIGHT LOAD; default off.
    "MOE_FUSED_EP_V2_PREQUANT": env_bool("MOE_FUSED_EP_V2_PREQUANT"),
    # Fused EP MoE v2, fp8 weights: prebuild stages each per-channel weight scale once as an [E, N] base
    # buffer (the 4-D [E, 1, 1, N] parameter becomes a view of it) and the call hands the op the base, so
    # the step no longer relayouts the padded 4-D scales on every call. Same values. Read at WEIGHT LOAD.
    "MOE_FUSED_EP_V2_STAGE_SCALES": env_bool("MOE_FUSED_EP_V2_STAGE_SCALES"),
    # Fused EP MoE v2 plan bases: the per-source slab bases from the exchanged count table in its own
    # [ep, e_total] orientation instead of transposing the [e_total, ep] run starts. Integer-exact, same
    # bits. Read at TRACE time; default off.
    "MOE_FUSED_EP_V2_SLAB_BASE_T": env_bool("MOE_FUSED_EP_V2_SLAB_BASE_T"),
    # Shard Kimi K3's latent-MoE down/up projections across the cores of a
    # chip instead of replicating them. They are unquantized [7168, 3584] and
    # [3584, 7168] matmuls run at full HBM bandwidth, ~22% of decode device
    # time at TP=32; halving the bytes each core reads costs two on-package
    # all-gathers per layer. Independent of TPU_MOE_HIERARCHICAL_EP.
    "TPU_LATENT_PROJ_INTRA_CHIP_TP": env_bool("TPU_LATENT_PROJ_INTRA_CHIP_TP"),
    # Slice the rotary cos_sin caches to max model len at load to
    # minimize xla layout data copy overhead. Text-only.
    "TPU_ROPE_CACHE_TRUNCATE": env_bool("TPU_ROPE_CACHE_TRUNCATE", default=True),
    # Materialize the rotary cos_sin caches with a row-major ({1,0}) device
    # layout at load, May cost more HBM due to padding.
    "TPU_ROPE_CACHE_ROW_MAJOR": env_bool("TPU_ROPE_CACHE_ROW_MAJOR", default=True),
    "TPU_MOE_HASH_TABLE_ROW_MAJOR": env_bool(
        "TPU_MOE_HASH_TABLE_ROW_MAJOR", default=False
    ),
    # Small fused kernels (kernels/experimental), each off by default.
    # Run a GDN layer's output gated RMS norm and its out_proj's per-token fp8
    # quantization as one kernel. Needs per-channel fp8 out_proj weights.
    "TPU_GDN_FUSED_NORM_QUANT": env_bool("TPU_GDN_FUSED_NORM_QUANT", default=True),
    # Gives every fused shared-expert gate weight its own [hidden] device
    # buffer. Built from a reshape of the [1, hidden] gate weight, the
    # parameter otherwise keeps that buffer and torch_tpu extracts it with one
    # tt_jit_execute_as_strided program per layer on every compiled call.
    "TPU_FUSED_SE_GATE_FRESH_BUFFER": env_bool(
        "TPU_FUSED_SE_GATE_FRESH_BUFFER", default=False
    ),
    # Sampler top-k masking over the candidate logits takes the k-th value
    # from the repeated max-and-mask Pallas kernel (kernels/router_topk.py)
    # instead of torch.topk.
    "TPU_SAMPLER_TOPK_PALLAS": env_bool("TPU_SAMPLER_TOPK_PALLAS", default=False),
    # Caps the sampler's candidate top-k width at this many values (0: the
    # candidate block count). The runner keeps the candidate path only while
    # every active request's top_k fits under the cap.
    "TPU_SAMPLER_TOPK_MAX_K": env_nonnegative_int("TPU_SAMPLER_TOPK_MAX_K", 0),
    # Qwen3.5: compute the decoder input RMSNorm and the per-row FP8
    # quantization feeding attention as one op (needs TPU_SP_FP8_GATHER; see
    # models/vllm/qwen3_next_patch.py).
    "TPU_FUSE_PRE_ATTN_NORM_QUANT": env_bool(
        "TPU_FUSE_PRE_ATTN_NORM_QUANT", default=True
    ),
    # Qwen3.5: compute the post-attention RMSNorm, the FP8 router gate and the
    # shared expert as one op ahead of the routed experts.
    "TPU_FUSE_POST_ATTN_NORM_MOE": env_bool(
        "TPU_FUSE_POST_ATTN_NORM_MOE", default=True
    ),
    # Qwen3.5 fused post-attention norm + router + shared expert (R1 push DN1):
    # 1 runs the x absmax + fp8 convert as one Pallas pass over the residual
    # sum; 2 runs norm, x quant, the router/gate_up matmul, silu*u and the
    # inter quant as one kernel with the weight resident in VMEM; 3 is 2 with
    # the shared-expert gate GEMV in the kernel too. All are meant
    # bit-identical to 0 (kernels/experimental/se_prequant).
    "TPU_SE_PREQUANT_KERNEL": env_nonnegative_int("TPU_SE_PREQUANT_KERNEL", 0),
    # Scoped VMEM limit (MiB) of the TPU_SE_PREQUANT_KERNEL >= 2 kernel; 0 keeps the
    # kernel's 56 MiB. The kernel needs ~14.5 MiB at 256-row tiles (AOT, tpu7x);
    # a smaller limit leaves VMEM to neighbouring ops (r1pdnskb5: 40 MiB -5.7
    # us/call, bit-identical).
    "TPU_SE_PREQUANT_VMEM_MIB": env_nonnegative_int("TPU_SE_PREQUANT_VMEM_MIB", 0),
    # TPU_SE_PREQUANT_KERNEL=4 with per-layer ids: the kernel reads x as the
    # must-fuse group of its convert and call (the out_proj producer's tiling, no
    # relayout copy), with MUST_FUSE pushed onto the callee's ops so it survives
    # torch_tpu's embedding. Off: mode 4's bf16 x input (a relayout copy).
    "TPU_SE_PREQUANT_MUSTFUSE": env_bool("TPU_SE_PREQUANT_MUSTFUSE", default=False),
    "TPU_GDN_FUSED_NORM_QUANT_MIN_ROWS": env_nonnegative_int(
        "TPU_GDN_FUSED_NORM_QUANT_MIN_ROWS", 16
    ),
    # Run the gated q split, q/k RMS norms and NeoX rope of Qwen3NextAttention
    # as one kernel on token buckets of up to 64 rows.
    "TPU_FUSED_QK_NORM_ROPE": env_bool("TPU_FUSED_QK_NORM_ROPE", default=True),
    "TPU_FUSED_QK_NORM_ROPE_MIN_ROWS": env_nonnegative_int(
        "TPU_FUSED_QK_NORM_ROPE_MIN_ROWS", 0
    ),
    # R1 push DN3: N > 0 runs the fused q/k norm + rope kernel on the whole
    # qkv projection output read in place (no sliced copy of q_gate at the
    # jax_op boundary) with row tiles of up to N (VMEM limit raised above 128).
    # Bit-identical to 0 (kernels/experimental/qk_norm_rope_gate v2).
    "TPU_QK_NORM_ROPE_INPLACE_TILE": env_nonnegative_int(
        "TPU_QK_NORM_ROPE_INPLACE_TILE", 0
    ),
    # ATS (R1 push): the in-place q/k norm kernel writes q in RPAm's packed kv-major order
    # [Hk, T, G/2, 2, D]; the adapter hands torch the same values token-major.
    "TPU_QK_NORM_ROPE_Q_PACKED": lambda: os.getenv("TPU_QK_NORM_ROPE_Q_PACKED", "0") == "1",
    # xla_quantized_matmul's per-row activation quant takes its row max on f32
    # instead of bf16 for bf16 rows: the same values bit for bit, a faster TPU
    # fusion (the prefill FA projection's bf16 row max is ~15.8 us/call).
    "TPU_QMM_ACT_F32_AMAX": env_bool("TPU_QMM_ACT_F32_AMAX", default=False),
    # ATS (R1 push): the PCP all-RPA entry packs q (the RPA's T<->kv swapaxes) outside its
    # shard_map and exchanges the packed q, so XLA folds the q permutations.
    "TPU_PCP_Q_PACK_OUTSIDE": lambda: os.getenv("TPU_PCP_Q_PACK_OUTSIDE", "0") == "1",
    # Run batched RPA in token buckets of up to 64 rows with half the chip's
    # VMEM and a decode batch of min(8, max_num_seqs), at max_model_len <= 16k.
    "TPU_RPA_DECODE_BATCHED": env_bool("TPU_RPA_DECODE_BATCHED"),
    # Sequences a batched RPA decode step works on concurrently (the kernel's
    # decode_batch_size; 0 keeps the wrapper default of 8). Fewer lanes leave
    # VMEM for a larger KV block: at DP4xTP2 (1 KV head per core, ~98K ctx)
    # 4 lanes take 4096-token blocks instead of 2816 and run ~15% faster.
    "TPU_RPA_DECODE_LANES": env_nonnegative_int("TPU_RPA_DECODE_LANES", 0),
    # The DCP CACHE_ONLY pass of a decode step runs the streaming kernel
    # batched_rpa/decode_v2.py (f32 softmax and LSE); steps with prefill or
    # mixed sequences keep the batched RPA kernel.
    "TPU_RPA_DECODE_V2": env_bool("TPU_RPA_DECODE_V2", default=False),
    # Decode steps split the KV read over the tensor-parallel ranks that share
    # a KV head (layers/adapter/kv_split_attention.py).
    "TPU_RPA_KV_SPLIT": env_bool("TPU_RPA_KV_SPLIT"),
    # PCP prefill: attend the cached history with the batched RPA kernel
    # (CACHE_ONLY over each rank's pages, f32 LSE merge) instead of the
    # streaming kernel's history pass. See pcp_streaming_rpa/history_rpa.py.
    "TPU_PCP_HISTORY_RPA": env_bool("TPU_PCP_HISTORY_RPA"),
    # With TPU_PCP_HISTORY_RPA: also run the current chunk (causal, plus the
    # history-boundary page) on batched RPA instead of the streaming current
    # pass. See history_rpa.pcp_attention_all_rpa.
    "TPU_PCP_CURRENT_RPA": env_bool("TPU_PCP_CURRENT_RPA"),
    # PCP kernels that copy into the peer at entry (GDN fused projection +
    # exchange, streaming attention current/history) first barrier the PCP
    # ring on the runtime barrier semaphore (collective ids 13/14/15). The
    # kernels read os.environ directly; listed here for discoverability.
    "TPU_PCP_ENTRY_BARRIER": env_bool("TPU_PCP_ENTRY_BARRIER"),
    # Extra token buckets added to the default list of buckets.
    "TPU_TOKEN_BUCKET_EXTRA": lambda: [
        int(v) for v in os.getenv("TPU_TOKEN_BUCKET_EXTRA", "").split(",") if v.strip()
    ],
    # Generate {2^k, 3 * 2^k} half-step token buckets from 8 up to budget.
    "TPU_TOKEN_BUCKET_HALF_STEPS": env_bool("TPU_TOKEN_BUCKET_HALF_STEPS"),
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
    # Pipeline parallelism: scheduler steps the engine keeps in flight beyond
    # the pipeline depth (distributed/pp_push.py). With the depth alone, the
    # step after the one on the last stage is dispatched only once that
    # stage has sampled, and stage 0 waits for the engine core's update,
    # schedule and RPC and for the worker's input preparation; each extra
    # step moves that chain one wave earlier. The lazy-push loop keeps the
    # depth alone.
    "TPU_PP_EXTRA_STEPS_IN_FLIGHT": env_nonnegative_int(
        "TPU_PP_EXTRA_STEPS_IN_FLIGHT", 0
    ),
    # A prefill-only runner (an MTP producer under context or pipeline
    # parallelism) never runs a step without prefill sequences, so the
    # attention and GDN wrappers call their per-sequence kernels directly
    # instead of under a lax.cond on the request distribution
    # (kernels/static_guards.py).
    "TPU_PREFILL_ONLY_STATIC_GUARDS": env_bool(
        "TPU_PREFILL_ONLY_STATIC_GUARDS", default=False
    ),
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
    # Bind each worker to its TPU chip's NUMA node (CPU set and preferred
    # memory) before the runtime starts; see worker/numa_bind.py. On by
    # default (qx-ovl-s2 shipped it off but every measured run set it:
    # c16 int90 99.6 vs 96.6). It only narrows the inherited CPU set and is a
    # logged no-op when ranks do not map onto chips or sysfs has no node.
    "TPU_WORKER_NUMA_BIND": env_bool("TPU_WORKER_NUMA_BIND", default=True),
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
    # A JSON table giving, per backbone MoE layer, the global experts each
    # expert-parallel rank holds (expert_placement.py); empty keeps the
    # checkpoint's contiguous placement.
    "TPU_EXPERT_PLACEMENT": lambda: os.getenv(
        "TPU_EXPERT_PLACEMENT",
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "scripts",
                "vllm",
                "benchmark",
                "inferencex",
                "qwen3.5",
                "expert_placement_dp8_ep8_opt_copies128.json",
            )
        ),
    ),
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
    # Return freed host heap through tcmalloc's MallocExtension_ReleaseFreeMemory
    # when a gperftools tcmalloc is preloaded (the serving image does that),
    # under which glibc's malloc_trim(0) releases nothing. Off, the loader
    # calls malloc_trim(0) only.
    "TPU_TCMALLOC_RELEASE_FREE_MEMORY": env_bool(
        "TPU_TCMALLOC_RELEASE_FREE_MEMORY", default=False
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
    # Issue the producer's post-transfer complete_request_blocks RPC on a
    # background thread instead of the engine thread.
    "TPU_RAIDEN_ASYNC_COMPLETE": env_bool("TPU_RAIDEN_ASYNC_COMPLETE", default=False),
    # Milliseconds; complete_request_blocks calls at least this slow are logged.
    "TPU_RAIDEN_COMPLETE_SLOW_MS": env_float("TPU_RAIDEN_COMPLETE_SLOW_MS", 100.0),
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
    # Concurrent Stage-3 coordinations of a head-sharded decode engine, capped
    # at its TPU_RAIDEN_POOL_STAGING_LEASES.
    "TPU_RAIDEN_STAGE3_SUBMIT_WORKERS": env_nonnegative_int(
        "TPU_RAIDEN_STAGE3_SUBMIT_WORKERS", 1
    ),
    # Decode-side cap on Stage-3 loads that hold allocated blocks and have
    # not landed; a request over the cap is skipped without allocation and
    # asked again on the next scheduling pass. 0 = unbounded.
    "TPU_RAIDEN_STAGE3_MAX_INFLIGHT_LOADS": env_nonnegative_int(
        "TPU_RAIDEN_STAGE3_MAX_INFLIGHT_LOADS", 0
    ),
    # Decode-side head-unique Stage-3 transfer: each FA page is sent once per
    # KV head (from the source replica the page index names) and the decode
    # ranks of the head group copy it to each other over ICI after landing.
    # Needs a head-sharded TP decode (tp > num_key_value_heads, DCP 1) and a
    # PCP1 x TP producer of the same TP size.
    "TPU_RAIDEN_STAGE3_HEAD_UNIQUE": env_bool(
        "TPU_RAIDEN_STAGE3_HEAD_UNIQUE", default=False
    ),
    # Load remote KV before the first forward, not in a no-forward step.
    "TPU_RAIDEN_INLINE_LOAD": env_bool("TPU_RAIDEN_INLINE_LOAD", default=False),
    # Consumer only: poll Stage-3 load completion again when a step's async
    # output is materialized, so a load that lands during the step is
    # scheduled one step earlier.
    "TPU_RAIDEN_STAGE3_LATE_POLL": env_bool(
        "TPU_RAIDEN_STAGE3_LATE_POLL", default=False
    ),
    # Producer only: run the Stage-3 request-block registration (span
    # lowering and controller RPCs) on a side thread so it overlaps the
    # step's forward instead of delaying its dispatch.
    "TPU_RAIDEN_STAGE3_ASYNC_REGISTER": env_bool(
        "TPU_RAIDEN_STAGE3_ASYNC_REGISTER", default=False
    ),
    # Longest the side thread waits for the step's forward dispatch before
    # it starts the registration anyway.
    "TPU_RAIDEN_STAGE3_ASYNC_REGISTER_DELAY_MS": env_float(
        "TPU_RAIDEN_STAGE3_ASYNC_REGISTER_DELAY_MS", 10.0
    ),
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
    # Barrier on the world CPU group right before the draft model load, so the
    # ranks enter the load's process-group creation together. torch_tpu's
    # new_group waits a fixed 300 s for every rank's store key, and the ranks
    # can finish streaming the target weights further apart than that.
    "TPU_DRAFT_LOAD_BARRIER": env_bool("TPU_DRAFT_LOAD_BARRIER", default=True),
    # Seconds the draft load barrier waits. Set, the barrier is a
    # monitored_barrier with this timeout instead of the CPU group's own (30
    # min under gloo), so a rank still streaming weights under model-store
    # contention does not take the others down.
    "TPU_DRAFT_LOAD_BARRIER_TIMEOUT_S": env_optional_int(
        "TPU_DRAFT_LOAD_BARRIER_TIMEOUT_S"
    ),
    # Seconds every CPU (gloo) collective waits. Set, it is passed to
    # init_distributed_environment and fills the parallel config's
    # cpu_distributed_timeout_seconds when that is unset; unset keeps
    # gloo's 30 min default.
    "TPU_DIST_CPU_TIMEOUT_S": env_optional_int("TPU_DIST_CPU_TIMEOUT_S"),
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
    # Number of DP ranks of a `vllm serve --data-parallel-multi-port-external-lb`
    # server that share this host (must equal --data-parallel-size; one host
    # only). 0 = off. See distributed/tpu_multiport_dp.py.
    "TPU_MULTIPORT_DP_LOCAL_SIZE": env_nonnegative_int(
        "TPU_MULTIPORT_DP_LOCAL_SIZE", 0
    ),
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
    "VLLM_TPU_FAST_TOKEN_SUBSTITUTION": env_bool(
        "VLLM_TPU_FAST_TOKEN_SUBSTITUTION", default=True
    ),
    # Run an MTP draft's loop steps 1..K-1 as one compiled program
    # (spec_decode/mtp_fused_loop.py). Off by default: one program per step.
    # TODO: support the two cases that still take the per-step loop:
    #   - GLM-5.2 / Gemma-4 MTP: IndexShare's per-chunk top-k restore, and a
    #     KV-sharing draft's fixed positions and seq_lens.
    #   - EP-DP lockstep: run_dp_dummy_draft only replays the per-step
    #     collective trace.
    "VLLM_TPU_FUSED_DRAFT_LOOP": env_bool("VLLM_TPU_FUSED_DRAFT_LOOP", default=True),
    # Gather last-token rows immediately after self_attn.attn(q, k, v) on MTP
    # draft step 0 so gate, o_proj, post-attention norm, MoE, and final RMSNorm
    # execute on p = pad(num_reqs) rows instead of all T verify rows.
    "TPU_MTP_STEP0_POST_ATTN_GATHER": env_bool(
        "TPU_MTP_STEP0_POST_ATTN_GATHER", default=True
    ),
    "TPU_PREFILL_SKIP_VERIFY_WARMUP": env_bool(
        "TPU_PREFILL_SKIP_VERIFY_WARMUP", default=False
    ),
    # Non-greedy sampling filters the vocab to this many 128-logit blocks,
    # ranked by block max, before running top-k (and any subsequent top-p).
    # 0 searches the full vocab. Used only when every active request has
    # 1 <= top_k <= value // 2 and no grammar mask applies.
    "TPU_SAMPLER_TOPK_CANDIDATES": env_nonnegative_int(
        "TPU_SAMPLER_TOPK_CANDIDATES", 64
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
    # --- Small collectives ---
    # Run small decode-step TP all-reduces and DP MoE combines as Pallas
    # kernels (kernels/experimental/small_collectives) instead of XLA's.
    "TPU_SMALL_ALL_REDUCE": env_bool("TPU_SMALL_ALL_REDUCE", default=True),
    # With TPU_SMALL_ALL_REDUCE, also run the TP all-reduces of a layout whose
    # TP group is a sub-group of the EP mesh (DP x TP engines) as the Pallas
    # kernel over the TP pair; TPU_SMALL_ALL_REDUCE_TP_FORM picks the kernel
    # form ("oneshot", "twoshot"; empty: the kernel's size rule).
    "TPU_SMALL_ALL_REDUCE_TP": env_bool("TPU_SMALL_ALL_REDUCE_TP", default=False),
    "TPU_SMALL_ALL_REDUCE_TP_FORM": env_str("TPU_SMALL_ALL_REDUCE_TP_FORM", ""),
    # A replicated sequence-parallel MoE step under DP>1 gathers the hidden rows
    # and the router logits in one DP all-gather (packed along the last dim)
    # instead of two. Lossless: the split returns the same values.
    "TPU_DP_PACKED_ROUTER_GATHER": env_bool(
        "TPU_DP_PACKED_ROUTER_GATHER", default=False
    ),
    "TPU_EXPERT_PLACEMENT_MTP": env_bool("TPU_EXPERT_PLACEMENT_MTP", default=True),
    "TPU_OFFLOAD_MIN_HBM_USAGE": env_float("TPU_OFFLOAD_MIN_HBM_USAGE", 0.0),
    "TPU_SMALL_ALL_REDUCE_MAX_ROWS": env_nonnegative_int(
        "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 128
    ),
    "TPU_ATTN_MOE_OVERLAP": env_bool("TPU_ATTN_MOE_OVERLAP", default=False),
    "TPU_ATTN_MOE_OVERLAP_MIN_TOKENS": env_nonnegative_int(
        "TPU_ATTN_MOE_OVERLAP_MIN_TOKENS", 64
    ),
    "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS": env_int("TPU_MOE_FUSED_EP_STEP_MIN_TOKENS", 64),
    # Fused EP v3: JSON object of kernels.fused_moe.v3.Config field overrides on top of
    # Config.for_shape (e.g. '{"sorted_rows": false}'); empty = the shape's defaults.
    "TPU_MOE_FUSED_EP_V3_CONFIG": env_str("TPU_MOE_FUSED_EP_V3_CONFIG", ""),
    # Fused EP v3 FFN tile height (the kernel's tile_rows), a multiple of 8.
    "TPU_MOE_FUSED_EP_V3_TILE_ROWS": env_int("TPU_MOE_FUSED_EP_V3_TILE_ROWS", 128),
    "TPU_SP_MIN_ROWS": lambda: int(os.getenv("TPU_SP_MIN_ROWS", "512")),
    "TPU_SP_FP8_GATHER": env_bool("TPU_SP_FP8_GATHER", default=True),
    # --- PrefixAwareScheduler SRPF & cold-prefill chunk capping ---
    # Sort prefilling requests in `running` by fewest remaining prompt tokens.
    "TPU_SCHED_SRPF_RUNNING": env_bool("TPU_SCHED_SRPF_RUNNING", default=True),
    # Max prefill tokens per step for a cold prefill (remaining > threshold)
    # when num_decoding >= TPU_SCHED_COLD_MIN_DECODERS or (in `running`) when
    # `waiting` is non-empty. 0 disables capping.
    "TPU_SCHED_COLD_CHUNK_CAP": env_nonnegative_int("TPU_SCHED_COLD_CHUNK_CAP", 2048),
    # Remaining uncached prompt tokens threshold defining a cold prefill.
    "TPU_SCHED_COLD_THRESHOLD": env_nonnegative_int("TPU_SCHED_COLD_THRESHOLD", 8192),
    # Minimum active decoding requests before cold-prefill chunk capping
    # activates when `waiting` is empty.
    "TPU_SCHED_COLD_MIN_DECODERS": env_nonnegative_int(
        "TPU_SCHED_COLD_MIN_DECODERS", 12
    ),
    "TPU_SCHED_STEP_TOTAL_CAP": env_nonnegative_int("TPU_SCHED_STEP_TOTAL_CAP", 4096),
    "TPU_SCHED_WARM_STREAK": env_nonnegative_int("TPU_SCHED_WARM_STREAK", 0),
    "TPU_SCHED_WARM_MIN_DECODERS": env_nonnegative_int(
        "TPU_SCHED_WARM_MIN_DECODERS", 24
    ),
    # Cap active running sequences in PrefixAwareScheduler to the megakernel's
    # verify capacity: 0 = disabled (default); -1 = auto,
    # mk_max_verify_rows // (1 + num_spec_tokens) when
    # VLLM_TPU_QWEN35_MEGAKERNEL=1 and dcp == 1 (128 rows under
    # VLLM_TPU_QWEN35_MK_SPLIT_WIDE, so 32 at MTP=3); >0 = explicit cap.
    "TPU_SCHED_MK_MAX_RUNNING": env_int("TPU_SCHED_MK_MAX_RUNNING", 0),
    # KV consumer: while the running set is full, start the remote KV pull
    # of up to this many waiting requests so a freed slot is refilled on the
    # same step (PrefixAwareScheduler._kv_prefetch_begin_step). 0 = off.
    "TPU_SCHED_KV_PREFETCH_WAITING": env_nonnegative_int(
        "TPU_SCHED_KV_PREFETCH_WAITING", 0
    ),
    # Free blocks kept for the running set when a prefetched pull is
    # admitted at the cap, on top of the stock async-load reservation.
    "TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS": env_nonnegative_int(
        "TPU_SCHED_KV_PREFETCH_RESERVE_BLOCKS", 512
    ),
    # Sort the head of the waiting queue by estimated uncached prompt tokens
    # (minus TPU_SCHED_PAW_AGING_TOK_PER_S per second waited) at the start of
    # every schedule() (PrefixAwareScheduler._sort_waiting_queue).
    "TPU_SCHED_PAW_SORT_WAITING": env_bool("TPU_SCHED_PAW_SORT_WAITING", default=False),
    # Waiting requests the sort covers, from the head of the queue.
    "TPU_SCHED_PAW_SORT_WINDOW": env_nonnegative_int("TPU_SCHED_PAW_SORT_WINDOW", 128),
    # Mamba "align" prefill split: let a chunk that starts mid-block (a
    # 128-token partial-tail prefix hit) run past the next block boundary.
    # Under sparse retention (prefix_cache_retention_interval=0) the state at
    # that boundary is never hashed, so the stop only costs a step.
    "TPU_SCHED_SKIP_MIDBLOCK_ALIGN_STOP": env_bool(
        "TPU_SCHED_SKIP_MIDBLOCK_ALIGN_STOP", default=False
    ),
    # Single-step warm prefill (core/single_step_prefill.py): one prefill
    # step writes the last-full-block and partial-tail GDN checkpoints inside
    # the chunk instead of ending a chunk at each. Changes the stock GDN
    # program signature, so it is part of the compile hash. Must be set for
    # the engine core and every worker alike.
    "TPU_SCHED_SINGLE_STEP_PREFILL": env_bool(
        "TPU_SCHED_SINGLE_STEP_PREFILL", default=True
    ),
    # Near-finish decode protection: when >=1 decoding request has remaining
    # output tokens <= TPU_SCHED_FINISH_DECODE_STEPS * (1 + num_spec_tokens) and
    # scheduling a prefill would add >= TPU_SCHED_FINISH_DECODE_PREFILL_MIN_TOKENS
    # tokens, defer that prefill for up to TPU_SCHED_FINISH_DECODE_MAX_DEFERRAL_STEPS
    # consecutive steps. 0 disables.
    "TPU_SCHED_FINISH_DECODE_STEPS": env_nonnegative_int(
        "TPU_SCHED_FINISH_DECODE_STEPS", 64
    ),
    "TPU_SCHED_FINISH_DECODE_MAX_DEFERRAL_STEPS": env_nonnegative_int(
        "TPU_SCHED_FINISH_DECODE_MAX_DEFERRAL_STEPS", 64
    ),
    "TPU_SCHED_FINISH_DECODE_PREFILL_MIN_TOKENS": env_nonnegative_int(
        "TPU_SCHED_FINISH_DECODE_PREFILL_MIN_TOKENS", 1024
    ),
    # Host-side 5-point TTFT stage telemetry periodic summary interval in
    # seconds; 0 disables (default, zero overhead).
    "QX_TTFT_STAGE_LOG_PERIOD_S": env_float("QX_TTFT_STAGE_LOG_PERIOD_S", 0.0),
    # Hybrid-state retention (core/hybrid_retention.py): keeps the GDN state
    # snapshots a later turn hits on. Unset: on, except on a KV consumer.
    "TPU_HYBRID_STATE_RETENTION": env_optional_bool("TPU_HYBRID_STATE_RETENTION"),
    "TPU_HYBRID_RETENTION_MAX_CONVS": env_nonnegative_int(
        "TPU_HYBRID_RETENTION_MAX_CONVS", 96
    ),
    "TPU_HYBRID_RETENTION_MAX_PINNED_BLOCKS": env_nonnegative_int(
        "TPU_HYBRID_RETENTION_MAX_PINNED_BLOCKS", 360
    ),
    "TPU_HYBRID_RETENTION_KEEP_PRIOR_TURNS": env_nonnegative_int(
        "TPU_HYBRID_RETENTION_KEEP_PRIOR_TURNS", 1
    ),
    "TPU_HYBRID_RETENTION_SMART_FORK": env_bool(
        "TPU_HYBRID_RETENTION_SMART_FORK", default=True
    ),
    "TPU_HYBRID_RETENTION_SAFE_TAIL_OFFSET": env_nonnegative_int(
        "TPU_HYBRID_RETENTION_SAFE_TAIL_OFFSET", 4
    ),
    "TPU_RAIDEN_INLINE_LOAD_MAX_CHUNKS": env_nonnegative_int(
        "TPU_RAIDEN_INLINE_LOAD_MAX_CHUNKS", 2
    ),
    "VLLM_TPU_LIGHTWEIGHT_STEP_STATS": env_bool(
        "VLLM_TPU_LIGHTWEIGHT_STEP_STATS", default=False
    ),
    "VLLM_TPU_LIGHTWEIGHT_STEP_STATS_INTERVAL": env_nonnegative_int(
        "VLLM_TPU_LIGHTWEIGHT_STEP_STATS_INTERVAL", 500
    ),
    "VLLM_TPU_RAIDEN_LOAD_WAIT_MS": env_float(
        "VLLM_TPU_RAIDEN_LOAD_WAIT_MS", 1.5
    ),
    "VLLM_TPU_QWEN35_MEGAKERNEL": env_bool("VLLM_TPU_QWEN35_MEGAKERNEL", default=False),
    # Rows per microbatch of the Qwen3.5 megakernel's wide verify step, 0 for
    # off; parse_mk_wide_mb checks the value.
    "VLLM_TPU_QWEN35_MK_WIDE_MB": env_nonnegative_int("VLLM_TPU_QWEN35_MK_WIDE_MB", 0),
    # Smallest verify bucket (rows) the wide step applies to when it is on; smaller
    # buckets above 80 rows keep the split plan of 80-row calls.
    "VLLM_TPU_QWEN35_MK_WIDE_MIN_ROWS": env_nonnegative_int("VLLM_TPU_QWEN35_MK_WIDE_MIN_ROWS", 160),
    # Qwen3.5-397B with the megakernel under DCP > 1 and more than 16
    # sequences: size the auto KV budget with the DCP 1 headroom (1 GiB less
    # KV than the DCP > 1 budget) for the program-load scratch of wide steps.
    "VLLM_TPU_QWEN35_MK_DCP_KV_HEADROOM": env_bool(
        "VLLM_TPU_QWEN35_MK_DCP_KV_HEADROOM", default=False
    ),
    # Host bytes of the Raiden offload tier on a pipeline-parallel engine,
    # replacing the connector's cpu_bytes_to_use (which TPU_RAIDEN_MAX_CPU_BYTES
    # only caps). Unset: the cap applies as on a TP engine.
    "TPU_RAIDEN_PP_MAX_CPU_BYTES": env_optional_int("TPU_RAIDEN_PP_MAX_CPU_BYTES"),
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
