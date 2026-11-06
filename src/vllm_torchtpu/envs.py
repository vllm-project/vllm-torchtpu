# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import functools
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    TPU_K3_SP_PREFILL: bool = False
    TPU_DCP_MLA_TOPK_CAP: int = -1
    TPU_DCP_MLA_TOPK_CAP_CHUNK: int = 512
    TPU_SPARSE_MLA_LIVE_GATHER: bool = True
    TPU_DCP_LSE_MERGE_A2A: bool = True
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS: int = 32
    TPU_DCP_RS_OUT_PRECISION: bool = True
    TPU_DCP_MLA_SMALL_CHUNK_BATCH: int = 4
    TPU_DCP_MLA_UV_BEFORE_MERGE: bool = False
    TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS: int = 0
    TPU_DCP_MERGE_VSPACE_FUSED: bool = False
    TPU_DCP_MLA_OPROJ_ALL_REDUCE: bool = False
    TPU_DCP_MLA_OPROJ_ALL_REDUCE_CHUNK_TOKENS: int = 512
    TPU_MLA_SPARSE_SPLIT_Q: bool = True
    TPU_DCP_LSE_MERGE_A2A_PACK: bool = False
    TPU_DCP_LSE_MERGE_A2A_REAL_ROWS: bool = False
    TPU_MTP_DCP_MERGE_SIZED: bool = False
    TPU_DCP_MERGE_CHUNKED: int = 0
    TPU_DCP_MERGE_CHUNKED_MIN_TOKENS: int = 1024
    TPU_DCP_MERGE_CHUNKED_LSE_AG: bool = False
    TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS: int = 0
    TPU_DCP_MERGE_RS_BF16: int = 0
    TPU_DCP_MLA_SMALL_CHUNK_TOKENS: int = 16
    TPU_DCP_MLA_MASK_IN_KERNEL: bool = False
    TPU_DCP_MLA_FUSED_PREP: bool = False
    TPU_DCP_MLA_MIXED_LIVE: bool = False
    TPU_DCP_MLA_MIXED_LIVE_WIDTHS: str = ""
    TPU_MLA_DECODE_ROW_WIDTHS: str = ""
    TPU_DCP_MLA_MIXED_BODY: str = ""
    TPU_DCP_MLA_MIXED_MASK_IN_KERNEL: bool = False
    TPU_DCP_MLA_MIXED_FUSED_PREP: bool = False
    TPU_MTP_DCP_MERGE_SMALL_A2A: bool = True
    TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS: int = 8
    TPU_VOCAB_EMBED_I32_BOUNDS: bool = True
    TPU_MOE_TP_DECODE_HBM_OPERANDS: str = "drafter"
    TPU_DCP_MLA_SEND_LAYOUT: bool = True
    TPU_DCP_MLA_SEND_LAYOUT_UNPACK: bool = True
    TPU_DCP_MLA_SEND_LAYOUT_MIN_TOKENS: int = 0
    TPU_DCP_MLA_SEND_GROUP_ROWS: int = 8
    TPU_DCP_MLA_SEND_SMALL_A2A: bool = True
    TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS: int = 8
    TPU_MLA_FUSED_SMALL_A2A: bool = True
    TPU_MLA_FUSED_SMALL_A2A_VMEM_MIB: int = 16
    TPU_MLA_FUSED_SMALL_A2A_PIN_MIN_TOKENS: int = 8
    TPU_MLA_FUSED_SMALL_A2A_PIN_VMEM_MIB: int = 32
    TPU_MLA_FUSED_SMALL_A2A_BUCKETS: str = "4=kv@16"
    TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS: int = 1
    TPU_MLA_FUSED_SMALL_A2A_MERGE: bool = True
    TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS: int = 4
    TPU_SMALL_COLL_HBM_OPERAND: bool = True
    TPU_DCP_MERGE_CHUNKED_SMALL: int = 2
    TPU_DCP_MLA_MERGE_PIECES: int = 2
    TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS: int = 2048
    TPU_DCP_MERGE_WEIGHTED_HM: bool = True
    TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS: int = 2048
    TPU_MLA_DECODE_BODY_LEVERS: str = "1"
    TPU_DCP_MLA_MIXED_KERNEL_V2: bool = True
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
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    TPU_MOE_DENSE_EXPERT_THRESHOLD: int = 0
    FUSED_MOE_DECODE_KERNEL: bool = False
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: int = 256
    FUSED_MOE_PREFILL_KERNEL: bool = True
    FUSED_MOE_PREFILL_KERNEL_V2: bool = False
    TPU_MOE_SINGLE_SORT: bool = True
    USE_MOE_TP_DECODE_KERNEL: bool = False
    TPU_MOE_TP_DECODE_FAST: bool = True
    TPU_MOE_TP_DECODE_SPLIT_META: bool = True
    TPU_MOE_TP_DECODE_SHARED_META: bool = True
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
    TPU_MOE_SKIP_PADDED_TOKENS: bool = False
    TPU_TOKEN_PADDING_SKIP_UNCHANGED: bool = True
    TPU_STAGE_TIMING_ONLY_WITH_CONSUMER: bool = True
    TPU_WORKER_CPU_PIN: bool = True
    TPU_WORKER_CPU_PIN_SKIP: int = 2
    TPU_AOT_SKIP_GUARD_CHECK: bool = True
    TPU_FOREACH_MUTATION_EPILOGUE: bool = True
    TPU_H2D_REUSE_STEADY: bool = True
    TPU_MOE_ONEHOT_SORTFREE: bool = True
    TPU_MOE_SC_COMBINE_TILED: bool = False
    TPU_MOE_SC_COMBINE_V5: bool = False
    TPU_MOE_SC_COMBINE_V6: bool = False
    TPU_MOE_HIERARCHICAL_EP: bool = False
    TPU_MOE_EP_TP_SIZE: int = 1
    TPU_MOE_ROUTER_TOPK: str = "rowmax"
    TPU_MOE_ROUTER_TOPK_BIAS: bool = True
    USE_MOE_FUSED_EP_KERNEL: bool = False
    MOE_FUSED_EP_KERNEL_IMPL: str | None = None
    MOE_FUSED_EP_ENABLE_W4A8: bool = False
    MOE_FUSED_EP_KERNEL_MIN_TOKENS: int = 1024
    MOE_FUSED_EP_V2_SHARDED_PLAN: bool = False
    MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE: int = 0
    TPU_LATENT_PROJ_INTRA_CHIP_TP: bool = False
    TPU_TOKEN_BUCKET_EXTRA: list[int] = []
    TPU_TOKEN_BUCKET_LINEAR: bool = False
    TPU_PP_DYNAMIC_CHUNKS: bool = True
    TPU_PP_CHUNK_SLACK: float = 0.1
    TPU_PP_LAZY_PUSH: bool = False
    TPU_ROPE_CACHE_TRUNCATE: bool = False
    TPU_ROPE_CACHE_ROW_MAJOR: bool = True
    TPU_MLA_ROPE_NEOX_PERMUTE: bool = True
    TPU_INDEXER_ROPE_NEOX_PERMUTE: bool = True
    TPU_INDEXER_QPREP_FUSED: bool = True
    TPU_INDEXER_QPREP_FUSED_MIN_TOKENS: int = 128
    TPU_INDEXER_K_WRITE_COMPACT: bool = True
    TPU_MOE_HASH_TABLE_ROW_MAJOR: bool = False
    TPU_PARALLEL_PRECOMPILE: bool = False
    TPU_PRECOMPILE_BIGGEST_FIRST: bool = True
    TPU_PRECOMPILE_PEER_REPLAY: bool = True
    TPU_PRECOMPILE_TRACE_FIRST: bool = False
    TPU_STREAMIDX_CHUNK_TOKENS: int | None = None
    TPU_DCP_INDEXER_A2A_MERGE: bool = True
    TPU_DCP_INDEXER_HIST_MERGE: bool = True
    TPU_DCP_INDEXER_A2A_MIN_TOKENS: int = 64
    TPU_STREAMIDX_SMALL_BUCKET_BQ: int | None = 4
    TPU_STREAMIDX_SMALL_BUCKET_BKVP: int | None = None
    TPU_STREAMIDX_SMALL_BUCKET_TOKENS: int = 64
    TPU_IDX_TOPK_SCHED_GROUPS: bool = False
    TPU_DCP_RANK_OPERAND: bool = False
    TPU_SC_TOPK_COST_BYTES: int = 0
    VLLM_TPU_INDEXER_KV_CACHE_MODE: str = "sal"
    TPU_SPARSE_MLA_DCP_FUSED_Q_GATHER: bool = True
    TPU_STREAMIDX_SKIP_EMPTY_DECODE: bool = True
    TPU_STREAMIDX_DECODE_PACK: bool = True
    TPU_STREAMIDX_SHARED_META: bool = True
    TPU_SC_TOPK_DCP_MERGE_FLAT: bool = True
    TPU_SC_TOPK_COOP_V2: bool = True
    TPU_SC_TOPK_COOP_MERGE: bool = True
    TPU_DCP_INDEXER_SMALL_AG: bool = True
    TPU_INDEXER_TOPK_SHARED_READ: bool = True
    TPU_SPARSE_MLA_CHUNK_TOKENS: int | None = None
    TPU_STREAMIDX_SKIP_EMPTY_DECODE_CHUNKED: bool = False
    TPU_STREAMIDX_MIXED_KV_PAGES: int | None = None
    TPU_MOE_COLLECTION_CHUNK_SIZE: int = 0
    TPU_MOE_FP8_ALL_GATHER: bool = False
    TPU_PCP_TOPOLOGY_AWARE_MESH: bool = True
    USE_RPA_PIPELINED_DMA_STAGING: bool = False
    USE_PIPELINED_DMA_STAGING: bool = False
    VLLM_TPU_BLOCK_MAJOR_KV: bool = False
    TPU_EVICT_WEIGHTS_PAGE_CACHE: bool = False
    TPU_SPARSE_MLA_NOPE_LAYOUT: str = "sparsecore"
    TPU_SPARSE_MLA_ROPE_LAYOUT: str = "sparsecore"
    TPU_MLA_TC_KV_WRITER: bool = False
    TPU_MLA_DECODE_PROLOGUE: bool = False
    TPU_MLA_KV_WRITE_IN_GATHER: bool = False
    TPU_INDEXER_QB_OVERLAP: bool = False
    TPU_SPARSE_MLA_PACKED_ROPE: bool = False
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
    TPU_PRECOMPILE_SPEC_MIXED: bool = False
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
    VLLM_TPU_FUSED_DRAFT_LOOP: bool = False
    TPU_SAMPLER_TOPK_CANDIDATES: int = 0
    VLLM_TPU_MTP_FUSED_INDEXSHARE: bool = True
    TPU_MTP_TOPK_INPLACE: bool = True
    MM_ENCODER_FRAME_CHUNK_PATCH_SIZE: int = 0
    VLLM_TPU_STATIC_POOLER: bool = True
    VLLM_TPU_POOLING_KV_CACHE_TOKENS: int = -1
    TPU_SMALL_ALL_REDUCE: bool = True
    TPU_SMALL_ALL_REDUCE_MAX_ROWS: int = 32
    TPU_SMALL_ALL_REDUCE_DRAFT: bool = True
    TPU_FP8_LINEAR_PALLAS: bool = True
    TPU_SAMPLER_VOCAB_PARALLEL: bool = True
    TPU_SMALL_ALL_REDUCE_VALID_ROWS: bool = False
    TPU_FUSED_OPROJ_AR: bool = True
    TPU_FUSED_OPROJ_AR_WAVES: bool = True
    TPU_FP8_SKINNY_SCALE_DMA: bool = False
    TPU_FP8_SKINNY_HBM_PIN_SITES: str = "self_attn.o_proj,q_b_proj"
    TPU_LM_HEAD_HBM: str = "drafter"
    TPU_FUSED_ADD_RMSNORM: bool = False
    TPU_FUSED_ADD_RMSNORM_MIN_TOKENS: int = 512
    TPU_FUSED_ADD_RMSNORM_MIN_HIDDEN: int = 4096
    TPU_FUSED_ADD_RMSNORM_QUANT: bool = True
    TPU_FUSED_ADD_RMSNORM_QUANT_MOE: bool = False
    TPU_MTP_EH_PROJ_TP: bool = False
    TPU_MTP_ONEHOT_AR: bool = True
    TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS: int = 0
    TPU_SAMPLER_BSEARCH_RADIX_BITS: int = 0
    TPU_SAMPLER_FAST_TOP_P: bool = True
    TPU_SAMPLER_FUSED_CHAIN: bool = True
    TPU_SAMPLER_GRAPH_RNG: bool = True
    TPU_FP8_LINEAR_PALLAS_SITES: str = "q_b_proj,self_attn.o_proj,mlp.gate_up_proj"
    TPU_SAMPLER_VERIFY_TRIM: bool = False


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


def env_positive_int_or_none(
    env_name: str, default: int | None = None
) -> Callable[[], int | None]:
    """Optional size knob that must be greater than zero when it is set.

    Unset or empty reads as ``default``; ``None`` means the feature is off.
    """
    parse_int = env_optional_int(env_name)

    def _get_positive_int_or_none_env() -> int | None:
        value = parse_int()
        if value is None:
            value = default
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
    # (`sparse_ragged_paged_attention(topk_cap=...)`). 0 = off; -1 (default)
    # = 512 at DCP >= 8 (GLM-5.2 TP16 x DCP8 served it, trial 4) and off
    # below, where a rank owns most of the top-k and the cap makes large
    # prefill steps ~30% slower (DCP2, measured).
    "TPU_DCP_MLA_TOPK_CAP": env_int("TPU_DCP_MLA_TOPK_CAP", -1),
    # Gather/attention chunk (query tokens) of the capped path: fewer, larger
    # chunks amortise the per-chunk calls; each chunk's static gather buffer
    # is chunk * topk rows (~335 MB at 256), of which only the live windows
    # are written. Default 512 (trial 6 served it; bit-identical to 256 on the
    # served mixed config).
    "TPU_DCP_MLA_TOPK_CAP_CHUNK": env_int("TPU_DCP_MLA_TOPK_CAP_CHUNK", 512),
    # Capped path: the native gather skips each window row's atoms past its live
    # top-k count (outputs unchanged). On by default.
    "TPU_SPARSE_MLA_LIVE_GATHER": env_bool("TPU_SPARSE_MLA_LIVE_GATHER", default=True),
    # Opt in to the all-to-all DCP MLA lse merge for decode-sized steps.
    "TPU_DCP_LSE_MERGE_A2A": env_bool("TPU_DCP_LSE_MERGE_A2A", default=True),
    # Largest step (tokens) that uses it; above this, the reduce-scatter is
    # faster. Only the backbone uses the all-to-all; the MTP drafter, traced
    # once for every bucket, always takes the reduce-scatter.
    "TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS": env_int("TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS", 32),
    # Backbone MLA merge: reduce-scatter the weighted outputs in the output's
    # precision (bf16), half f32's bytes. MTP drafter stays f32; 0 = f32.
    "TPU_DCP_RS_OUT_PRECISION": env_bool("TPU_DCP_RS_OUT_PRECISION", default=True),
    # With TPU_DCP_LSE_MERGE_A2A: carry each destination's lse bits as extra
    # rows of its partial-output send block, so the merge runs ONE all-to-all
    # instead of two (the f32 lse exchange is a latency-bound collective of a
    # few KB). Bits are moved, not converted: the merge is unchanged.
    "TPU_DCP_LSE_MERGE_A2A_PACK": env_bool("TPU_DCP_LSE_MERGE_A2A_PACK", default=False),
    # With TPU_DCP_LSE_MERGE_A2A_PACK: send only the real tokens' rows (a
    # ragged all-to-all sized by the step's token count) instead of every
    # padded row of the bucket. Real rows unchanged; padding rows merge to 0.
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": env_bool(
        "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS", default=False
    ),
    # The MTP drafter's DCP lse merge as one JAX op over the DCP mesh that picks
    # its exchange per exported bucket: steps of at most
    # TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS tokens take one packed all-to-all (the
    # partial's bf16 words plus the lse bits) and merge locally, larger ones
    # the lse all-gather + f32 reduce-scatter. The drafter's FX graph serves
    # every bucket and cannot branch on the token count in torch; JAX exports
    # each bucket at its concrete shape, so the choice leaves no guard. Same
    # per-element weights as the reduce-scatter merge; only the order the
    # ranks' weighted partials are summed in differs.
    "TPU_MTP_DCP_MERGE_SIZED": env_bool("TPU_MTP_DCP_MERGE_SIZED", default=False),
    # With TPU_MTP_DCP_MERGE_SIZED, on drafter buckets of
    # T <= TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS tokens (a power of 2): move
    # the sized merge's packed words with one Pallas round of remote DMAs
    # (small_collectives.all_to_all) instead of XLA's all-to-all. Each
    # destination block is padded to whole 8-row tiles; every row the unpack
    # reads arrives bit for bit, so the merge is unchanged. Peers are
    # addressed by (lane, rank) as data (the op takes the DCP rank and lane
    # operands on every bucket). Needs TPU_DCP_RANK_OPERAND.
    "TPU_MTP_DCP_MERGE_SMALL_A2A": env_bool(
        "TPU_MTP_DCP_MERGE_SMALL_A2A", default=True
    ),
    "TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS": env_int(
        "TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS", 8
    ),
    # VocabParallelEmbedding's five shard bounds as int32 scalar buffers
    # instead of int64. An int64 scalar program input costs one X64SplitLow
    # custom call (~1.15 us, serial, at every program start that embeds: the
    # backbone once per step, the MTP drafter once per draft pass). The
    # bounds are < 2**31, so the masks and the shifted ids are the same
    # values; the ids are widened to int64 for the lookup as before.
    "TPU_VOCAB_EMBED_I32_BOUNDS": env_bool("TPU_VOCAB_EMBED_I32_BOUNDS", default=True),
    # With TPU_DCP_LSE_MERGE_A2A_REAL_ROWS and TPU_MLA_DECODE_PROLOGUE: on
    # decode buckets the DCP MLA kernel writes the packed real-rows exchange's
    # send buffer itself (partials destination-major, lse words in place), so
    # no XLA retile / concat runs between the kernel and the all-to-all. Same
    # bits sent, same merge.
    "TPU_DCP_MLA_SEND_LAYOUT": env_bool("TPU_DCP_MLA_SEND_LAYOUT", default=True),
    # With TPU_DCP_MLA_SEND_LAYOUT: the exchange and the unpack of the
    # received buffer (slices + bitcasts, no arithmetic) as one JAX op, the
    # merge arithmetic unchanged in torch. Same bits.
    "TPU_DCP_MLA_SEND_LAYOUT_UNPACK": env_bool(
        "TPU_DCP_MLA_SEND_LAYOUT_UNPACK", default=True
    ),
    # TPU_DCP_MLA_SEND_LAYOUT only on decode buckets of at least this many
    # tokens (0 = every bucket it supports). Below ~8 tokens the send buffer's
    # SEND_PAD_ROWS lse row block is as large as the partial rows, so the
    # exchange carries ~2x the packed real-rows bytes (mirror: bucket 4 +1.4 %,
    # buckets 8 / 16 -1.8 / -2.8 %). Smaller buckets keep the packed a2a merge.
    "TPU_DCP_MLA_SEND_LAYOUT_MIN_TOKENS": env_int(
        "TPU_DCP_MLA_SEND_LAYOUT_MIN_TOKENS", 0
    ),
    # With TPU_DCP_MLA_SEND_LAYOUT: run the
    # ragged exchange over groups of this many rows (8 or 16: whole (8,128)
    # tiles of the int16 buffer, so XLA keeps the kernel's tiled layout
    # instead of copying it to and from [rows, 2, 256]), from the group
    # holding the lse word row to the one holding the last real partial row;
    # received partial rows past the real ones are zeroed as the row-ragged
    # exchange leaves them. 0 = the row-ragged exchange. Same bits on every
    # row the merge reads, same merge. Applies where the block splits into
    # whole groups (fixed per bucket at trace time).
    "TPU_DCP_MLA_SEND_GROUP_ROWS": env_int("TPU_DCP_MLA_SEND_GROUP_ROWS", 8),
    # With TPU_DCP_MLA_SEND_LAYOUT, on backbone buckets of
    # T <= TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS tokens (a power of 2): run the
    # exchange as one Pallas round of remote DMAs
    # (small_collectives.all_to_all) instead of XLA's ragged all-to-all --
    # every destination block from the (8,128) tile row holding the lse word
    # row to the block's end, received partial rows past the real ones zeroed
    # as the ragged exchange leaves them. Same bits on every row the merge
    # reads, same merge. Takes precedence over TPU_DCP_MLA_SEND_GROUP_ROWS on
    # those buckets. Peers are addressed by
    # (lane, rank) as data, so it needs the lane operand the indexer's
    # TPU_DCP_INDEXER_SMALL_AG already lifts on every bucket (with
    # TPU_SC_TOPK_DCP_MERGE_FLAT and TPU_DCP_RANK_OPERAND); otherwise off.
    "TPU_DCP_MLA_SEND_SMALL_A2A": env_bool("TPU_DCP_MLA_SEND_SMALL_A2A", default=True),
    "TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS": env_int(
        "TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS", 8
    ),
    # On the buckets TPU_DCP_MLA_SEND_SMALL_A2A takes (T <= its cap, a power of
    # 2) whose attention grid is one step (decode body lever "step8"): the
    # decode MLA kernel runs that exchange itself -- barrier signalled at
    # kernel entry, each destination's block built in a VMEM slab and pushed
    # once to the peer -- and returns the receive buffer, so the send buffer
    # and the separate small all-to-all call are gone. Same received bits,
    # same unpack and merge. Other buckets keep their exchange.
    "TPU_MLA_FUSED_SMALL_A2A": env_bool("TPU_MLA_FUSED_SMALL_A2A", default=True),
    # With TPU_MLA_FUSED_SMALL_A2A: the fused call's scoped VMEM limit (MiB). It
    # decides what XLA may keep in VMEM across the call -- the receive buffer
    # the unpack reads, and the prefetches of the next ops' weights.
    "TPU_MLA_FUSED_SMALL_A2A_VMEM_MIB": env_int("TPU_MLA_FUSED_SMALL_A2A_VMEM_MIB", 16),
    # Fused calls of buckets of at least this many tokens pin their array
    # operands to HBM (so XLA cannot stage the gathered KV into VMEM ahead of
    # the call) and take TPU_MLA_FUSED_SMALL_A2A_PIN_VMEM_MIB of scoped VMEM;
    # smaller buckets take neither. Decided on the bucket's static shape in the
    # JAX op. Full-depth decode (78 layers): T=4 best unpinned at 16 MiB, T=8
    # best pinned at 32 MiB.
    "TPU_MLA_FUSED_SMALL_A2A_PIN_MIN_TOKENS": env_int(
        "TPU_MLA_FUSED_SMALL_A2A_PIN_MIN_TOKENS", 8
    ),
    "TPU_MLA_FUSED_SMALL_A2A_PIN_VMEM_MIB": env_int(
        "TPU_MLA_FUSED_SMALL_A2A_PIN_VMEM_MIB", 32
    ),
    # Per bucket, overriding the three knobs above for the buckets it lists:
    # "T=<pins>@<MiB>,..." with <pins> all | none | operand roles joined by
    # "+" (kv, q, q_pe, tab, recv), e.g. "4=kv@16,8=kv@32" pins only the
    # gathered KV and leaves q (and its relayout copy) where XLA puts it.
    # Default "4=kv@16" (trial 6): T=4 pins only the gathered KV at 16 MiB;
    # other buckets take the knobs above.
    "TPU_MLA_FUSED_SMALL_A2A_BUCKETS": lambda: os.getenv(
        "TPU_MLA_FUSED_SMALL_A2A_BUCKETS", "4=kv@16"
    ),
    # Buckets below this many tokens keep the separate small all-to-all inside
    # the same op (kernel send buffer -> small_all_to_all -> unpack: the
    # unfused pair's kernels and bits); only buckets >= it run the exchange in
    # the kernel. Decided on the JAX op's static shape: no second token-count
    # gate in the traced graph. 1 = every bucket the flag takes.
    "TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS": env_int(
        "TPU_MLA_FUSED_SMALL_A2A_MIN_TOKENS", 1
    ),
    # With TPU_MLA_FUSED_SMALL_A2A: peers push into a VMEM receive scratch and
    # the kernel also runs the DCP merge (the unpack + `_merge_received`, term
    # by term), so the op returns the merged [T, L, V] -- no receive buffer, no
    # unpack or merge ops after the call. Not bit-identical to torch's merge
    # (exp / summation order). Needs MIN_TOKENS = 1.
    "TPU_MLA_FUSED_SMALL_A2A_MERGE": env_bool(
        "TPU_MLA_FUSED_SMALL_A2A_MERGE", default=True
    ),
    # With _MERGE: buckets above this many tokens keep the in-kernel exchange
    # but merge in the op (cp_mla_attention.merge_received_jax), returning the
    # same merged [T, L, V]. Static per bucket in the JAX op.
    "TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS": env_int(
        "TPU_MLA_FUSED_SMALL_A2A_MERGE_MAX_TOKENS", 4
    ),
    # The DCP small all-gather (TPU_DCP_INDEXER_SMALL_AG) and small all-to-all
    # (TPU_DCP_MLA_SEND_SMALL_A2A) take their data operand constrained to HBM.
    # Unconstrained (pl.ANY), XLA stages it into VMEM with a copy-start/-done
    # right before the call, exposed (~0.9-1.1 us per call at decode); the
    # kernels only DMA the operand, so the bytes and the results are the same.
    "TPU_SMALL_COLL_HBM_OPERAND": env_bool("TPU_SMALL_COLL_HBM_OPERAND", default=True),
    # The reduce-scatter merge of a backbone bucket of at least
    # TPU_DCP_MERGE_CHUNKED_MIN_TOKENS tokens, split into this many equal token
    # chunks (0/1 = unsplit; a bucket that does not split into multiples of 8
    # rows stays unsplit). Per-row arithmetic is unchanged; chunk k's
    # SparseCore reduce-scatter can overlap chunk k+1's TensorCore lse gather
    # and weight multiply. The MTP drafter (one trace for every bucket) never
    # splits.
    "TPU_DCP_MERGE_CHUNKED": env_int("TPU_DCP_MERGE_CHUNKED", 0),
    "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": env_int(
        "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS", 1024
    ),
    # The chunked merge (TPU_DCP_MERGE_CHUNKED) gathers every chunk's lse in
    # one all-gather up front instead of one synchronous all-gather per chunk
    # (the scheduler issues most of those ahead of chunk 0's weights, so each
    # one's fixed cost delays the first SparseCore reduce-scatter). Bitwise
    # the same merge. Only for buckets of at most
    # TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS tokens (0 = every chunked
    # bucket): the one gather moves all the bytes the per-chunk gathers did,
    # so on wide DCP groups and large buckets it is bandwidth-bound and the
    # fixed costs it saves no longer pay for it.
    "TPU_DCP_MERGE_CHUNKED_LSE_AG": env_bool(
        "TPU_DCP_MERGE_CHUNKED_LSE_AG", default=False
    ),
    "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": env_int(
        "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS", 0
    ),
    # The chunked merge's chunk count for the buckets that gather their lse
    # once (TPU_DCP_MERGE_CHUNKED_LSE_AG, <= ..._LSE_AG_MAX_TOKENS: c1k, c2k)
    # instead of TPU_DCP_MERGE_CHUNKED (kept for the larger buckets). Their
    # V-space reduce-scatters compile synchronous, so each chunk pays the
    # collective's fixed cost on the critical path; fewer, larger chunks pay it
    # fewer times. Per-row arithmetic is unchanged (bit-identical). 0 = off;
    # values below 2 are ignored (an unsplit bucket lifts another input set).
    "TPU_DCP_MERGE_CHUNKED_SMALL": env_int("TPU_DCP_MERGE_CHUNKED_SMALL", 2),
    # NOT bitwise (needs the accuracy gate): the chunked merge
    # (TPU_DCP_MERGE_CHUNKED) sends its LSE-weighted partial in bf16 instead
    # of f32, halving the bytes of the SparseCore-bound reduce-scatter chain.
    # 1 = bf16 reduce-scatter (the collective sums in its operand dtype);
    # 2 = bf16 all-to-all of the same bytes, then a local f32 sum over the
    # ranks (bf16 wire, f32 accumulation). 0 = f32 reduce-scatter (today).
    "TPU_DCP_MERGE_RS_BF16": env_int("TPU_DCP_MERGE_RS_BF16", 0),
    # With TPU_DCP_MERGE_CHUNKED, mixed / prefill buckets the merge chunks: the
    # DCP sparse MLA kernel writes each merge chunk's rows (output and lse)
    # into that chunk's own buffer instead of one q-aliased buffer threaded
    # through all its attention chunks (`out_pieces`), so merge chunk k's lse
    # gather, weights and SparseCore reduce-scatter depend only on the
    # attention chunks that produce its rows and can run under the later
    # ones. Every merged row is bit-identical. 2 also splits the one lse
    # gather of TPU_DCP_MERGE_CHUNKED_LSE_AG into chunk 0's rows and the rest,
    # so chunk 0's reduce-scatter need not wait for the last attention chunk
    # (same gathered rows). 3 also issues every attention chunk's SparseCore
    # gather first and ties the later chunks' gathers to piece 0's buffers
    # (`out_pieces_gather_first`), so the schedule closes the gathers before
    # piece 0's reduce-scatter instead of holding them open under the
    # attention calls. 0 = off.
    "TPU_DCP_MLA_MERGE_PIECES": env_int("TPU_DCP_MLA_MERGE_PIECES", 2),
    # The kernel pieces of TPU_DCP_MLA_MERGE_PIECES only for buckets of at
    # least this many tokens (0 = every chunked bucket). The lse-gather head
    # split (2) is torch-side and does not follow it (a token-count test there
    # specializes range-compiled graphs).
    "TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS": env_int(
        "TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS", 2048
    ),
    # The chunked reduce-scatter merge (TPU_DCP_MERGE_CHUNKED) builds each
    # chunk's weighted operand heads-major in one TensorCore Pallas kernel
    # (`kernels/lse_rescale.py`, as `cp_lse_ag_out_rs`) instead of a
    # tokens-major weight
    # multiply plus the relayout copy XLA adds for the head-axis
    # reduce-scatter (which it may offload to the SparseCore and complete
    # late). Same products, same reduce-scatter: bit-identical.
    "TPU_DCP_MERGE_WEIGHTED_HM": env_bool("TPU_DCP_MERGE_WEIGHTED_HM", default=True),
    # With TPU_DCP_MERGE_WEIGHTED_HM: the Pallas kernel only for buckets of at
    # least this many tokens, from the chunk's static row count times
    # TPU_DCP_MERGE_CHUNKED (0 = every chunked bucket). Smaller buckets take
    # the tokens-major weight multiply.
    "TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS": env_nonnegative_int(
        "TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS", 2048
    ),
    # DCP sparse MLA attention grid step for decode-sized chunks
    # (`sparse_ragged_paged_attention(small_chunk_batch_size=...)`): attend
    # this many query tokens per kernel grid step instead of one, so the
    # fixed per-step cost is paid once per N tokens (MTP decode has q=4 per
    # request). The kernel uses the largest divisor of the chunk <= N and
    # grids `cdiv(real tokens, step)` steps, so up to N-1 padding rows are
    # attended (outputs don't-care). 1 = one token per step.
    "TPU_DCP_MLA_SMALL_CHUNK_BATCH": env_int("TPU_DCP_MLA_SMALL_CHUNK_BATCH", 4),
    # Under qrep, apply W_UV before the DCP merge, so the reduce-scatter carries
    # v_head_dim (256) instead of kv_lora_rank (512). Replicates W_UV per rank.
    "TPU_DCP_MLA_UV_BEFORE_MERGE": env_bool(
        "TPU_DCP_MLA_UV_BEFORE_MERGE", default=False
    ),
    # With TPU_DCP_MLA_UV_BEFORE_MERGE: apply W_UV before the merge only on
    # the backbone's buckets of at least this many tokens (`T > bucket_gate_edge`;
    # the prefill-sized ones, as #1935's TPU_DCP_MERGE_VSPACE did from 1024).
    # The smaller buckets keep the latent merge and with it the decode send
    # layout / fused small all-to-all, which carry the latent partial. The MTP
    # drafter, traced once for every bucket, then keeps the latent merge. 0 =
    # every bucket (and the drafter, as before).
    "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS": env_nonnegative_int(
        "TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS", 0
    ),
    # With TPU_DCP_MLA_UV_BEFORE_MERGE: on the buckets that take the chunked
    # reduce-scatter merge, build its operand in V space in one computation
    # (#1935 trial 6's TPU_DCP_MERGE_VSPACE: head-batched W_UV matmul, weight,
    # mask, one rounding; layers/adapter/dcp_vspace.py) instead of a pre-merge
    # projection followed by the latent merge path. GLM-5.2 AgentX c4 TP16 x
    # DCP8: P90 TTFT -10..-12% (the separate W_UV bmm, concat and relayouts
    # cost ~36 ms per 4096-token step).
    "TPU_DCP_MERGE_VSPACE_FUSED": env_bool("TPU_DCP_MERGE_VSPACE_FUSED"),
    # Where W_UV is replicated over the DCP group (TPU_DCP_MLA_UV_BEFORE_MERGE),
    # replicate o_proj too, so the backbone buckets that apply W_UV before the
    # merge (TPU_DCP_MLA_UV_BEFORE_MERGE_MIN_TOKENS) and would reduce-scatter
    # it (above TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS) skip the reduce-scatter: each
    # rank runs o_proj on its weighted partial of every group head, and
    # o_proj's all-reduce sums the ranks. Every other step keeps the merge and
    # o_proj itself. Off by default: the replicated o_proj does not fit some
    # served configs' HBM. A no-op without TPU_DCP_MLA_UV_BEFORE_MERGE,
    # unless o_proj is FP8 from an FP8 checkpoint (requantized to a float8),
    # and with an output gate. Costs one o_proj per layer of HBM.
    "TPU_DCP_MLA_OPROJ_ALL_REDUCE": env_bool(
        "TPU_DCP_MLA_OPROJ_ALL_REDUCE", default=False
    ),
    # Its o_proj and all-reduce run in chunks of this many tokens (a multiple
    # of 128), pipelined: each chunk's all-reduce under the next chunk's
    # o_proj. 0 = one chunk. A 512-token chunk's all-reduce is 6 MiB, over the
    # 2 MiB SparseCore offload threshold of the GLM-5.2 AgentX recipe; there
    # 512 beat 256, 768 and 1024 below 4096 tokens and tied at 4096.
    "TPU_DCP_MLA_OPROJ_ALL_REDUCE_CHUNK_TOKENS": env_int(
        "TPU_DCP_MLA_OPROJ_ALL_REDUCE_CHUNK_TOKENS", 512
    ),
    # Sparse MLA split q (`sparse_ragged_paged_attention(q_pe=...)`): hand
    # q_nope and q_pe to the kernel as separate operands. The kernel joins
    # them in VMEM and its output aliases q_nope (512 wide), instead of the
    # caller materialising a concat the wrapper pads to [T, H, 640]. Where the
    # caller also needs the log-sum-exp (DCP), it is exported lane-dense as
    # f32[T, H] (`lse_lane_dense=True`) instead of f32[T, H, 1], whose
    # lane-padded tiles cost a broadcast init, a relayout copy and padded
    # copies through the LSE merge. Bit-identical outputs. On by default on
    # glm-agentx (it replaces the branch's TPU_DCP_MLA_SPLIT_Q_LSE).
    "TPU_MLA_SPARSE_SPLIT_Q": env_bool("TPU_MLA_SPARSE_SPLIT_Q", default=True),
    # DCP sparse MLA: chunks of at most this many query tokens take the
    # decode-sized (small-chunk) path -- TPU_DCP_MLA_SMALL_CHUNK_BATCH tokens
    # per grid step and the one-atom live-bounded SparseCore gather
    # (`sparse_ragged_paged_attention(small_chunk_tokens=...)`). 16 = the
    # kernel default (unchanged); 32 also puts the 32-token MTP decode bucket
    # (5-8 running requests x 4 tokens) on that path. 0 disables it.
    "TPU_DCP_MLA_SMALL_CHUNK_TOKENS": env_int("TPU_DCP_MLA_SMALL_CHUNK_TOKENS", 16),
    # DCP sparse MLA, decode buckets (<= 32 tokens): drop the per-layer
    # owns mask / dummy key / lse select glue. The kernel takes each token's
    # true kv_len (0 = this rank owns none of its top-k), attends nothing for
    # such a token (finite out = 0) and writes its lse as -inf itself.
    # Same merge result; also removes the NaN a dummy key read from an
    # unwritten slot could put into an empty row's partial output.
    "TPU_DCP_MLA_MASK_IN_KERNEL": env_bool("TPU_DCP_MLA_MASK_IN_KERNEL", default=False),
    # With TPU_DCP_MLA_MASK_IN_KERNEL, decode buckets: build the capped
    # window indices, the live-window bound and the SparseCore gather's
    # live-count vector in ONE TensorCore Pallas call
    # (`prepare_capped_window_fused`), handing the indices to the gather
    # flat (a bitcast, no relayout copies). Same gathered rows.
    "TPU_DCP_MLA_FUSED_PREP": env_bool("TPU_DCP_MLA_FUSED_PREP", default=False),
    # DCP sparse MLA, mixed / prefill buckets (> 32 tokens, capped path):
    # gather and attend only each 16-token grid step's live 64-row atoms of
    # the first top-k window (cdiv(max kv_len, 64) of 8 at cap 512) instead
    # of all 512 slots per token. Same outputs; decode buckets unaffected.
    "TPU_DCP_MLA_MIXED_LIVE": env_bool("TPU_DCP_MLA_MIXED_LIVE", default=False),
    # With TPU_DCP_MLA_MIXED_LIVE: comma-separated ascending row widths
    # (multiples of 128 below the cap, e.g. "256"); a 16-token grid step whose
    # live atoms fit a width runs its attention body on that many rows per
    # token instead of the full 512. Same outputs. Empty = off.
    "TPU_DCP_MLA_MIXED_LIVE_WIDTHS": lambda: os.getenv(
        "TPU_DCP_MLA_MIXED_LIVE_WIDTHS", ""
    ),
    # DCP sparse MLA, capped path outside the live-group prep (decode): comma
    # list of ascending row widths (multiples of 128 below TPU_DCP_MLA_TOPK_CAP);
    # each grid step fetches and attends only the first w rows of its tokens'
    # tiles, w the smallest width covering the batch's largest per-rank kv_len.
    # Rows past kv_len are masked either way: same results. "" = off.
    "TPU_MLA_DECODE_ROW_WIDTHS": lambda: os.getenv("TPU_MLA_DECODE_ROW_WIDTHS", ""),
    # With TPU_MLA_DECODE_ROW_WIDTHS (fused-prep decode buckets): attention-body
    # levers on the decode body -- "1" = the bitwise default set (byte-transpose
    # NOPE unpack, per-row zeroing on the packed words, NOPE-only PV, no forced
    # MXU/VPU dependency, one stacked softmax per grid step, fix-up only on
    # overflowing steps, live-row KV fetches, eight tokens per attention grid
    # step), or a comma list of lever names (kernels/mla/sparse/kernel.py
    # DECODE_BODY_LEVERS). Empty = off.
    "TPU_MLA_DECODE_BODY_LEVERS": lambda: os.getenv("TPU_MLA_DECODE_BODY_LEVERS", "1"),
    # With TPU_DCP_MLA_MIXED_LIVE: attention-body rewrites of the live grid
    # steps -- "1" = the bitwise-identical default set (byte-transpose NOPE
    # unpack, per-row zeroing on the packed words, NOPE-only PV, no forced
    # MXU/VPU dependency), or a comma list of lever names
    # (kernels/mla/sparse/kernel.py MIXED_BODY_LEVERS). Empty = off.
    "TPU_DCP_MLA_MIXED_BODY": lambda: os.getenv("TPU_DCP_MLA_MIXED_BODY", ""),
    # With TPU_DCP_MLA_MIXED_BODY set: add the stage-A body levers
    # (kernels/mla/sparse/kernel.py MIXED_KERNEL_V2_LEVERS) -- the capped
    # fix-up only on steps that overflow, and a two-token stacked softmax with
    # the unpacked KV staged in VMEM. Bitwise identical to the body levers
    # alone.
    "TPU_DCP_MLA_MIXED_KERNEL_V2": env_bool(
        "TPU_DCP_MLA_MIXED_KERNEL_V2", default=True
    ),
    # DCP sparse MLA, mixed / prefill buckets (> 32 tokens): the mixed-step
    # counterpart of TPU_DCP_MLA_MASK_IN_KERNEL. Drop the per-layer owns mask,
    # dummy-key update of the top-k (and the copy of the top-k it forces when
    # a later layer reuses it) and the lse select; the kernel takes each
    # token's true kv_len (0 = this rank owns none of its top-k), attends
    # nothing for such a token (out = 0) and writes its lse as -inf itself.
    # Every owning row is bit-identical; the merge result is unchanged.
    "TPU_DCP_MLA_MIXED_MASK_IN_KERNEL": env_bool(
        "TPU_DCP_MLA_MIXED_MASK_IN_KERNEL", default=False
    ),
    # With TPU_DCP_MLA_MIXED_LIVE, mixed / prefill buckets: build kv_lens, the
    # live-group table and the window rows in ONE TensorCore Pallas call
    # (`prepare_capped_window_live_fused`) that reads the top-k as 128-lane
    # rows (a bitcast of the SparseCore top-k output: no [T, 2048] relayout)
    # and writes one window array per chunk (no per-chunk slice copy). Same
    # rows, kv_lens and table; decode buckets unaffected.
    "TPU_DCP_MLA_MIXED_FUSED_PREP": env_bool(
        "TPU_DCP_MLA_MIXED_FUSED_PREP", default=False
    ),
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
    # Write decode-sized steps into both SparseCore-layout sparse-MLA caches
    # with one TensorCore kernel instead of two SparseCore offloads.
    "TPU_MLA_TC_KV_WRITER": env_bool("TPU_MLA_TC_KV_WRITER", default=False),
    # DCP sparse MLA with query replication: rope q_pe inside the attention
    # kernel from the raw q head (no torch-side q rope / q_pe concat), and on
    # decode buckets let the kernel lay down the lse sentinel itself (no
    # XLA-initialised aliased lse). Needs TPU_MLA_SPARSE_SPLIT_Q and
    # TPU_MLA_ROPE_NEOX_PERMUTE.
    "TPU_MLA_DECODE_PROLOGUE": env_bool("TPU_MLA_DECODE_PROLOGUE", default=False),
    # DCP sparse MLA, fused-prep decode buckets (TPU_DCP_MLA_FUSED_PREP): the
    # SparseCore gather feeding the attention kernel writes the step's KV rows
    # into both caches itself (then fences), instead of a separate cache
    # writer kernel (tc_scatter / scatter_rows). Same bytes, same owner mask.
    "TPU_MLA_KV_WRITE_IN_GATHER": env_bool("TPU_MLA_KV_WRITE_IN_GATHER", default=False),
    # Run the MLA q_b projection inside the DCP indexer op, in the local
    # SparseCore top-k's XLA scheduling group, so decode issues it while the TC
    # would otherwise idle on the top-k's call-done. Same kernel, same inputs:
    # bit-identical q. Needs the fp8 Pallas q_b site (TPU_FP8_LINEAR_PALLAS).
    "TPU_INDEXER_QB_OVERLAP": env_bool("TPU_INDEXER_QB_OVERLAP", default=False),
    # Store each token's 64-byte fp8 RoPE key in 64 bytes instead of a padded
    # 128-byte row (eight tokens per 128-word row), cutting the sparse MLA
    # latent cache from 640 to 576 B/token. SparseCore layouts only; the
    # gather expands the rows back to the padded format the attention kernel
    # reads. Changes the cache layout: a disagg pair must agree on it.
    "TPU_SPARSE_MLA_PACKED_ROPE": env_bool("TPU_SPARSE_MLA_PACKED_ROPE", default=False),
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
    # With FUSED_MOE_DECODE_KERNEL=1 (or USE_MOE_TP_DECODE_KERNEL=1), serve the
    # armed TP-expert layers' steps above FUSED_MOE_DECODE_KERNEL_MAX_TOKENS
    # tokens with the fused MoE prefill kernel (routed and shared experts,
    # combine and all-reduce) instead of the GMM MoE path, in the same
    # exported program. EP layers keep the GMM path for now.
    "FUSED_MOE_PREFILL_KERNEL": env_bool("FUSED_MOE_PREFILL_KERNEL", default=True),
    # With FUSED_MOE_PREFILL_KERNEL: run the stage-A build of the prefill kernel
    # (kernels/experimental/fused_moe_prefill/fused_moe_prefill_kernel_v2.py:
    # four-deep weight ring, both matmuls on the explicit MXU with the down
    # epilogue per 256-column N-tile, 256-row tiles from 4096 tokens, flat
    # combine). It quantizes activations per row (one scale per token row),
    # not per 512 columns as the default build does, so it is not bit-identical
    # to it. Whole experts only: layers the default build serves in blocked
    # mode keep it.
    "FUSED_MOE_PREFILL_KERNEL_V2": env_bool("FUSED_MOE_PREFILL_KERNEL_V2"),
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
    # the usual path; fused_moe_decode then arms only the layers it refused.
    # Remove once the path is the default for these layers or is replaced by
    # another fused TP MoE path.
    "USE_MOE_TP_DECODE_KERNEL": env_bool("USE_MOE_TP_DECODE_KERNEL"),
    # With USE_MOE_TP_DECODE_KERNEL: the kernel's fast schedule (same arithmetic,
    # bit-identical output): top-k without the per-pick index reduction (exact
    # fallback on ties), the expert compaction on the vector units, ring primes
    # issued after the compaction instead of inside it, and a ring loop without
    # conditional DMAs or a stale matmul. Default on.
    "TPU_MOE_TP_DECODE_FAST": env_bool("TPU_MOE_TP_DECODE_FAST", True),
    # moe_tp_decode: pass n_valid, rank and the expert offset as three [1]
    # scalar operands instead of one stacked int32[3] (one XLA fusion per MoE
    # layer, ~0.6 us in situ). Exact.
    "TPU_MOE_TP_DECODE_SPLIT_META": env_bool(
        "TPU_MOE_TP_DECODE_SPLIT_META", default=True
    ),
    # moe_tp_decode: every armed layer reads ONE experts_start buffer (per
    # value) instead of its own, so the per-layer scalar meta stack
    # (n_valid, rank, experts_start) is the same HLO value in every layer and
    # XLA CSE keeps one per step (was one HBM->HBM op per layer). Exact.
    "TPU_MOE_TP_DECODE_SHARED_META": env_bool(
        "TPU_MOE_TP_DECODE_SHARED_META", default=True
    ),
    # moe_tp_decode's 8 weight operands (routed w13/w2 and scales, shared
    # expert and scales) constrained to HBM: "drafter" = the MTP drafter's
    # MoE layer only, "all" = every armed layer, "" = off. The kernel DMAs
    # what it reads from HBM itself; unconstrained, XLA can stage whole
    # weights into VMEM with a copy-start/-done right before the call (the
    # drafter's one-layer program: ~7 us/pass exposed). Same output bits.
    "TPU_MOE_TP_DECODE_HBM_OPERANDS": lambda: os.getenv(
        "TPU_MOE_TP_DECODE_HBM_OPERANDS", "drafter"
    )
    .strip()
    .lower(),
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
    # Skip the per-forward synchronous upload of the token padding mask when
    # its padding rows are unchanged since the last write (steady decode at a
    # fixed bucket). The device mask is then already what the upload would
    # write.
    "TPU_TOKEN_PADDING_SKIP_UNCHANGED": env_bool(
        "TPU_TOKEN_PADDING_SKIP_UNCHANGED", default=True
    ),
    # Record runner stage timings (TpuEvent pairs + a collector thread) only
    # when a KV connector can consume them; without a KV transfer config the
    # samples are dropped every step, so the stages become no-ops.
    "TPU_STAGE_TIMING_ONLY_WITH_CONSUMER": env_bool(
        "TPU_STAGE_TIMING_ONLY_WITH_CONSUMER", default=True
    ),
    # Give each worker's serving thread a whole physical core on the NUMA node
    # of its TPU chip (core TPU_WORKER_CPU_PIN_SKIP + LOCAL_RANK of that node)
    # and move the process's other threads to the rest of the node
    # (worker/cpu_pin.py). Placement only.
    "TPU_WORKER_CPU_PIN": env_bool("TPU_WORKER_CPU_PIN", default=True),
    "TPU_WORKER_CPU_PIN_SKIP": env_int("TPU_WORKER_CPU_PIN_SKIP", 2),
    # After warmup, skip the per-call Dynamo guard check of AOT-compiled model
    # forwards when vLLM compiled them without guards (dynamic_shapes_config
    # .evaluate_guards False). vLLM already does this for an AOT artifact
    # loaded from disk; a freshly compiled one keeps checking an empty guard
    # set on every call.
    "TPU_AOT_SKIP_GUARD_CHECK": env_bool("TPU_AOT_SKIP_GUARD_CHECK", default=True),
    # Write the inputs a compiled graph mutates (the KV caches) back with one
    # torch._foreach_copy_ instead of one copy_ per input in AOTAutograd's
    # epilogue (compilation/foreach_epilogue.py). Same pairs, same order.
    "TPU_FOREACH_MUTATION_EPILOGUE": env_bool(
        "TPU_FOREACH_MUTATION_EPILOGUE", default=True
    ),
    # Reuse the previous device copy of per-step host tensors whose values
    # repeat in steady decode -- the padded sampling params (temperature,
    # top_k, top_p) and the drafter loop's query_start_loc / request
    # distribution -- instead of a fresh H2D each step (value-keyed, like
    # _upload_unless_unchanged).
    "TPU_H2D_REUSE_STEADY": env_bool("TPU_H2D_REUSE_STEADY", default=True),
    # Under the EP one-hot MoE permute (routed rows <= the one-hot threshold,
    # at most 1024), derive the stable-sort positions, permuted token ids and
    # group sizes by int32 compare-and-count instead of three comparison
    # sorts. Outputs are bit-identical to the sorted form.
    "TPU_MOE_ONEHOT_SORTFREE": env_bool("TPU_MOE_ONEHOT_SORTFREE", True),
    # EP MoE combine on SparseCore: run the destination-tiled bf16 kernel
    # (ragged_gather_reduce_v4) instead of v2. v2 scatters a float32 partial
    # per source row and leaves the bf16 cast to a TensorCore pass; v4 owns
    # token tiles, accumulates in VMEM and writes bf16 rows. Bit-identical to
    # v2. Falls back to v2 where v4 does not apply.
    "TPU_MOE_SC_COMBINE_TILED": env_bool("TPU_MOE_SC_COMBINE_TILED"),
    # EP MoE combine on SparseCore: ragged_gather_reduce_v5, v4 with tokens
    # grouped by route count so each token's rows and its single output write
    # sit at static positions (no per-row accumulator store, no float32
    # accumulator). Bit-identical to v4 / v2. Takes precedence over
    # TPU_MOE_SC_COMBINE_TILED; falls back to it (or v2) where v5 does not apply.
    "TPU_MOE_SC_COMBINE_V5": env_bool("TPU_MOE_SC_COMBINE_V5"),
    # With TPU_MOE_SC_COMBINE_V5: run v5's combine as ragged_gather_reduce_v6,
    # the same steps without the emit_pipeline grid (fori_loops over column
    # chunks and groups, manually double-buffered gather and output DMAs).
    # Bit-identical to v5. Only where v5 applies.
    "TPU_MOE_SC_COMBINE_V6": env_bool("TPU_MOE_SC_COMBINE_V6"),
    # Hierarchical MoE parallelism: expert-parallel between chips,
    # tensor-parallel between the two chiplets of a chip. Off by
    # default; it helps only at low concurrency, where the MoE is
    # weight-bandwidth bound and the per-device expert load is badly
    # imbalanced. See K3_HIERARCHICAL_EP_PLAN.md.
    "TPU_MOE_HIERARCHICAL_EP": env_bool("TPU_MOE_HIERARCHICAL_EP"),
    # Tensor-parallel degree of the routed experts inside each expert-parallel
    # group: with --enable-expert-parallel on W ranks, the experts split into
    # W/N groups of N consecutive ranks, and each rank holds 1/N of the
    # intermediate width of every expert in its group (EP W/N x TP N) instead
    # of whole experts (EP W). Same weight bytes per rank and the same final
    # all-reduce over TP. At decode the MoE is weight-bandwidth bound and the
    # step waits for the rank with the most active experts; splitting each
    # expert N ways evens that out. 0 or 1 = off (vLLM's flat EP). Default 1
    # (flat EP): the GLM-5.2 AgentX trial-4 configuration (TP16 x DCP8,
    # --enable-expert-parallel) was served and accuracy-gated with flat EP16.
    # A default above 1 would apply only to GlmMoeDsaForCausalLM with EP on,
    # DP=1, PCP=1, no EPLB or sequence-parallel MoE
    # (moe_hierarchical.ep_tp_size); an explicit value applies as given and
    # keeps refusing those deployments.
    "TPU_MOE_EP_TP_SIZE": env_nonnegative_int("TPU_MOE_EP_TP_SIZE", 1),
    # MoE router top-k implementation: "rowmax" (default) is the sort-free
    # k-pass selection, "sort" is torch.topk, kept as an escape hatch.
    "TPU_MOE_ROUTER_TOPK": env_with_choices(
        "TPU_MOE_ROUTER_TOPK", "rowmax", ["sort", "rowmax"]
    ),
    # With the rowmax router and a score-correction bias, add the bias inside
    # the top-k kernel and return the unbiased scores at the chosen ids, so
    # the separate scores.gather (one gather fusion per MoE layer) goes away.
    # Ids and weights are bit-identical to the gather path.
    "TPU_MOE_ROUTER_TOPK_BIAS": env_bool("TPU_MOE_ROUTER_TOPK_BIAS", True),
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
    "TPU_ROPE_CACHE_ROW_MAJOR": env_bool("TPU_ROPE_CACHE_ROW_MAJOR", default=True),
    # Permute each MLA head's interleaved (GPT-J) rope columns of the q and
    # k_pe projections to half-split order at load and run the MLA rope
    # NeoX-style. q.k is unchanged; the `x[..., ::2]` split and its relayout
    # copies disappear.
    "TPU_MLA_ROPE_NEOX_PERMUTE": env_bool("TPU_MLA_ROPE_NEOX_PERMUTE", default=True),
    # The same for the sparse-attention indexer: permute each indexer head's
    # interleaved rope columns of wq_b, the k rope columns of wk_weights_proj
    # and the matching k_norm weight/bias entries at load, and run the indexer
    # rope NeoX-style. Indexer scores are unchanged; independent of the MLA
    # flag so each can be measured alone.
    "TPU_INDEXER_ROPE_NEOX_PERMUTE": env_bool(
        "TPU_INDEXER_ROPE_NEOX_PERMUTE", default=True
    ),
    # Indexer q prep (rope + per-(token, head) amax + fp8 cast) as one Pallas
    # pass over the wq_b output, written in the top-k op's [T, H, D] layout
    # (kernels/indexer_qprep.py). Needs TPU_INDEXER_ROPE_NEOX_PERMUTE.
    # Bit-identical to the unfused path.
    "TPU_INDEXER_QPREP_FUSED": env_bool("TPU_INDEXER_QPREP_FUSED", True),
    # Token counts below this keep the same expression in plain XLA (decided on
    # the static JAX shape inside the op): at decode buckets the kernel's fixed
    # cost exceeds the whole XLA chain (twin j-58e2f29b: +0.2 ms per decode step).
    "TPU_INDEXER_QPREP_FUSED_MIN_TOKENS": env_int(
        "TPU_INDEXER_QPREP_FUSED_MIN_TOKENS", 128
    ),
    # DCP indexer K write: compact this rank's owned rows (about T / dcp) to the
    # front before the Pallas writer, so its scalar walk skips the other
    # ranks' rows. Same rows written, same order.
    "TPU_INDEXER_K_WRITE_COMPACT": env_bool("TPU_INDEXER_K_WRITE_COMPACT", True),
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
        "TPU_PRECOMPILE_BIGGEST_FIRST", default=True
    ),
    # Replay a bucket another rank of the host compiled instead of compiling
    # it again: under TPU_PARALLEL_PRECOMPILE each bucket gets one owner rank
    # per host and the others wait for its artifact. Off compiles every bucket
    # on every rank, sharing only through the Tier-2 cache.
    "TPU_PRECOMPILE_PEER_REPLAY": env_bool("TPU_PRECOMPILE_PEER_REPLAY", default=True),
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
    # Merge the DCP indexer candidates with an all-to-all over the token axis
    # instead of an all-gather over the candidate axis. Same result.
    "TPU_DCP_INDEXER_A2A_MERGE": env_bool("TPU_DCP_INDEXER_A2A_MERGE", default=True),
    # Find the DCP indexer's global top-k in each chunk's SparseCore pass by
    # summing its radix histograms over the ranks, instead of a local top-k,
    # a candidate exchange and a merge, on the buckets that would take the
    # all-to-all merge (TPU_DCP_INDEXER_A2A_MIN_TOKENS). Decode-sized buckets
    # keep the all-gather merge: there the exchange rounds' latency outweighs
    # the merge they save.
    "TPU_DCP_INDEXER_HIST_MERGE": env_bool("TPU_DCP_INDEXER_HIST_MERGE", default=True),
    # Smallest padded token bucket that takes the all-to-all merge when
    # TPU_DCP_INDEXER_A2A_MERGE is on. The bucket is static, so each compiled
    # variant picks one merge with no runtime cost.
    "TPU_DCP_INDEXER_A2A_MIN_TOKENS": env_nonnegative_int(
        "TPU_DCP_INDEXER_A2A_MIN_TOKENS", 64
    ),
    # MIXED-pass query block for StreamIndex top-k token buckets of at most
    # TPU_STREAMIDX_SMALL_BUCKET_TOKENS. Under MTP every decode request has
    # q_len = gamma + 1 and runs in the MIXED pass, whose default 128-row block
    # scores ~124 padding rows per request. 4 (the default on glm-agentx) is
    # the recommended value; TPU_STREAMIDX_SMALL_BUCKET_TOKENS=0 keeps 128
    # everywhere.
    "TPU_STREAMIDX_SMALL_BUCKET_BQ": env_positive_int_or_none(
        "TPU_STREAMIDX_SMALL_BUCKET_BQ", default=4
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
    # Per DCP indexer layer, put the cross-rank SparseCore top-k and q_b_proj in
    # one XLA scheduling group (fixed per-layer id), so the scheduler can run
    # q_b_proj while the TensorCore waits on the top-k. Same values.
    "TPU_IDX_TOPK_SCHED_GROUPS": env_bool("TPU_IDX_TOPK_SCHED_GROUPS"),
    # DCP Pallas ops (indexer K write + top-k, sparse MLA) take this rank's
    # DCP index as a sharded int32 operand instead of learning it from an
    # all_to_all of an iota once per program (~8 us on 2x2x2 TP16xDCP16,
    # backbone and every drafter pass). Same values.
    "TPU_DCP_RANK_OPERAND": env_bool("TPU_DCP_RANK_OPERAND"),
    # bytes_accessed cost hint on decode-sized (b <= 32) SparseCore top-k
    # kernels, for XLA's latency-hiding scheduler; 0 = none. Same values.
    "TPU_SC_TOPK_COST_BYTES": env_nonnegative_int("TPU_SC_TOPK_COST_BYTES", 0),
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
    # Branch at runtime around StreamIndex top-k's two DECODE (q_len 1)
    # passes when the batch has no q_len-1 sequence -- every step under MTP,
    # where they launch over an empty range. Unchunked calls only. Exact.
    "TPU_STREAMIDX_SKIP_EMPTY_DECODE": env_bool(
        "TPU_STREAMIDX_SKIP_EMPTY_DECODE", default=True
    ),
    # Chunked StreamIndex top-k: drop the two DECODE (q_len 1) passes from
    # every chunk that starts at or past `max_num_seqs` tokens (decode rows sit
    # first on the token axis, one token each, so those chunks hold none), and
    # branch around them in chunk 0 when the batch has no q_len-1 sequence.
    # Exact.
    "TPU_STREAMIDX_SKIP_EMPTY_DECODE_CHUNKED": env_bool(
        "TPU_STREAMIDX_SKIP_EMPTY_DECODE_CHUNKED", default=False
    ),
    # Score the small-bucket (q block 4) MIXED pass -- every decode step under
    # MTP -- with the packed body: the query block goes to the MXU as a
    # block-diagonal [8H, 2D] matrix against the KV block's two halves stacked
    # on the contraction, so each head's scores fill whole f32 tiles (half the
    # head-reduction vector ops, no odd-head relayout), interior KV blocks skip
    # the mask, scores leave by strided stores, and the KV block is gathered
    # page-major (one contiguous copy per page) with a 6-deep ring.
    # Bit-identical indices and scores. SEQ_ALONG_LANE index cache only; other
    # passes are unchanged. Pairs with TPU_STREAMIDX_SMALL_BUCKET_BKVP=32
    # (GLM-5.2 DCP8, 9k-67k keys per rank: -10...-48 % per call).
    "TPU_STREAMIDX_DECODE_PACK": env_bool("TPU_STREAMIDX_DECODE_PACK", default=True),
    # StreamIdx per-step inputs built once per step instead of once per
    # indexer layer: the MIXED pass schedule without its empty-pass cond (so
    # XLA CSEs the layers' identical copies), its scalar [start, end, chunk]
    # operand outside the skip-empty conditional, and the DECODE schedules
    # only inside the branch that runs them (never, under MTP). GLM-5.2 full
    # depth, 1 decoder: 21 layers x (3 schedule conds + 1 operand fusion)
    # = 190 us per step. Same operands, same result.
    "TPU_STREAMIDX_SHARED_META": env_bool("TPU_STREAMIDX_SHARED_META", default=True),
    # KV pages per block of the StreamIndex top-k MIXED (multi-token) pass for
    # every bucket that does not take the small-bucket sizes (with
    # TPU_STREAMIDX_SMALL_BUCKET_BQ set, buckets above
    # TPU_STREAMIDX_SMALL_BUCKET_TOKENS; with it unset, all buckets). Unset
    # keeps the default 2 (a 512-position kv block at page 256). 4 halves the
    # grid steps per pass: GLM-5.2 DCP16 per rank, 4096-token bucket, 30.7k
    # local KV: -23..-25% per scoring call, 6.9k: -2..-3%, 1.8k: ~0; scores
    # bit-identical (same per-element math).
    "TPU_STREAMIDX_MIXED_KV_PAGES": env_positive_int_or_none(
        "TPU_STREAMIDX_MIXED_KV_PAGES"
    ),
    # DCP indexer top-k, all-gather merge, T <= 32 tokens: all-gather each
    # rank's candidate scores flat ([T * k] -> [dcp * T * k], rank-major) and
    # let the SparseCore merge read each row's dcp pieces in place, instead of
    # gathering [T, dcp * k] and padding it to 32 rows (an XLA pad + relayout
    # copy of 32 * dcp * k words, plus relayouts of the local slots and the
    # output slice). T > 32 keeps the padded path. Bitwise identical output.
    "TPU_SC_TOPK_DCP_MERGE_FLAT": env_bool("TPU_SC_TOPK_DCP_MERGE_FLAT", default=True),
    # `sparsecore_topk` at P > 1: the refinement levels as one runtime loop
    # (the SparseCore call's startup grows with its program size), each
    # level's peer reduce + scan + bucket search as one
    # loop, and no serial winner gather -- P >= 4 (T <= 8 at the DCP8
    # decode shapes): each subcore scatters its winners into the group's
    # staging slot at its prefix offset; P = 2: all peer copies in flight
    # before the first wait. Single-wave calls only (T <= 16 at DCP8); larger
    # buckets keep the default path. Bitwise identical output.
    "TPU_SC_TOPK_COOP_V2": env_bool("TPU_SC_TOPK_COOP_V2", default=True),
    # With TPU_SC_TOPK_DCP_MERGE_FLAT's flat merge and T <= 16 rows: split each
    # merge row over p = min(dcp, 32 / T) subcores by rank segment
    # (`sparsecore_topk.sparsecore_topk_dcp_merge_coop`) instead of one
    # subcore walking all dcp * k scores; only the subcore holding this rank's
    # segment emits. T > 16 keeps the one-subcore merge. Bitwise identical.
    "TPU_SC_TOPK_COOP_MERGE": env_bool("TPU_SC_TOPK_COOP_MERGE", default=True),
    # With TPU_SC_TOPK_DCP_MERGE_FLAT's flat path, buckets T <= 8
    # (streamindex_topk.SMALL_AG_MAX_TOKENS): the DCP
    # indexer's candidate all-gather (dcp x T * k int32 words) as one Pallas
    # remote-DMA kernel (kernels/experimental/small_collectives/all_gather.py)
    # instead of XLA's all-gather, which runs serially on the TensorCore at
    # 13-17 us per idx layer. Needs TPU_DCP_RANK_OPERAND (the kernel's peer
    # addresses take the rank and lane as data). Same bytes, same result.
    "TPU_DCP_INDEXER_SMALL_AG": env_bool("TPU_DCP_INDEXER_SMALL_AG", default=True),
    # The indexed layer's own attention reads its rows back from the updated
    # shared table, the same `table[:num_tokens]` expression the following
    # skip_topk layers read, instead of the fresh rows. One expression lets XLA
    # CSE the attention's top-k-only index prep (physical rows, live counts,
    # window metadata) across the indexed layer and its skip_topk layers; with
    # the fresh rows the first skip_topk layer recomputes it whenever
    # num_tokens is short of the table height (decode). Same values either way.
    "TPU_INDEXER_TOPK_SHARED_READ": env_bool(
        "TPU_INDEXER_TOPK_SHARED_READ", default=True
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
    # Startup precompile of the spec-decode mixed-step shapes the sweeps above
    # miss: compute_selected_logits at (token bucket, padded_logits_length) and
    # the async substitution at the int32 stride-(1+K) source. Without it a
    # prefill chunk sharing a step with drafted decodes compiles them on the
    # serving path (GLM-5.2: ~1.5 s per LM-head shape). Off by default.
    "TPU_PRECOMPILE_SPEC_MIXED": env_bool("TPU_PRECOMPILE_SPEC_MIXED", default=False),
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
    # API server: tokenize chat prompts piecewise between added tokens and
    # cache message-body pieces across turns (segment_tokenize.py). Exact;
    # falls back to full tokenization for ineligible tokenizers/requests.
    "VLLM_TPU_SEGMENT_TOKENIZE": env_bool("VLLM_TPU_SEGMENT_TOKENIZE"),
    # Token budget of that cache (~12 B/token: id pointers plus piece text).
    "VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS": env_int(
        "VLLM_TPU_SEGMENT_TOKENIZE_CACHE_TOKENS", 33554432
    ),
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
    # Support IndexShare (GLM-5.2 / DeepSeek-V3.2) in _MtpFusedLoop by taking
    # saved step-0 top-k rows and installing them in-graph before each unrolled step.
    "VLLM_TPU_MTP_FUSED_INDEXSHARE": env_bool(
        "VLLM_TPU_MTP_FUSED_INDEXSHARE", default=True
    ),
    # IndexShare's per-propose top-k bookkeeping without pool-sized copies.
    # The step-0 rows are snapshotted with one gather (no compaction write
    # into the shared [max_num_tokens, index_topk] table: every loop step
    # restores them before it reads), and each restore writes its rows in
    # place (kernels/partial_kv_update.pallas_index_copy_). Eager slice
    # assignment into the table is functional on TPU: each one copied the
    # whole 32 MiB table (~26 us on tpu7x), three times per propose.
    "TPU_MTP_TOPK_INPLACE": env_bool("TPU_MTP_TOPK_INPLACE", default=True),
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
    # kernels (kernels/experimental/small_collectives) instead of XLA's. On by
    # default; skipped where there is no expert-parallel mesh unless set.
    "TPU_SMALL_ALL_REDUCE": env_bool("TPU_SMALL_ALL_REDUCE", True),
    # Largest TP all-reduce, in rows, that TPU_SMALL_ALL_REDUCE hands to the
    # kernel (default 32); 0 keeps small_all_reduce_op.MAX_ROWS. The crossover
    # moves with the core count and hidden size, so a slice other than the one MAX_ROWS
    # was measured on sets its own; mixed-step blocks stay on XLA's above it.
    "TPU_SMALL_ALL_REDUCE_MAX_ROWS": env_nonnegative_int(
        "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 32
    ),
    # With TPU_SMALL_ALL_REDUCE, also give the small kernel the TP all-reduces
    # of traces compiled outside the backbone warmup -- the MTP drafter's step-0
    # and loop-step programs. The row choice is taken per compiled bucket inside
    # the JAX op (XLA's all-reduce above the cap), so no trace forks.
    "TPU_SMALL_ALL_REDUCE_DRAFT": env_bool("TPU_SMALL_ALL_REDUCE_DRAFT", True),
    # With TPU_SMALL_ALL_REDUCE and the token-padding mask (TPU_MOE_SKIP_PADDED_
    # TOKENS), backbone TP all-reduces hand the two-shot kernel the step's
    # valid-row count: up to fold_rows valid rows (5 at H=6144 on 16 cores) are
    # folded into one row tile and only that moves, padding rows coming out +0;
    # more valid rows run the plain two-shot. A batch-1 MTP decode step pads 4
    # rows to 16. Valid rows are bitwise TPU_SMALL_ALL_REDUCE's (same f32 sum
    # in core order).
    "TPU_SMALL_ALL_REDUCE_VALID_ROWS": env_bool("TPU_SMALL_ALL_REDUCE_VALID_ROWS"),
    # With TPU_SMALL_ALL_REDUCE and TPU_FP8_LINEAR_PALLAS: attention's o_proj at
    # decode (<= 16 rows, a measured fp8_skinny shape, TP group = the mesh) runs its
    # matmul and its TP all-reduce as one Pallas kernel
    # (kernels/quantized_matmul/fp8_oproj_all_reduce.py): the all-reduce's barrier
    # round hides under the weight stream. Bitwise the two-call path.
    "TPU_FUSED_OPROJ_AR": env_bool("TPU_FUSED_OPROJ_AR", True),
    # With TPU_FUSED_OPROJ_AR: push each owner's reduce-scatter chunk as soon as the
    # weight tile holding its last column is done (waits the barrier after tile 0),
    # instead of all after the last tile. 0.3-0.7 us/call faster on 2x2x2 tpu7x
    # (t5-oproj-ar probe 02b); 0 keeps every push after the last tile.
    "TPU_FUSED_OPROJ_AR_WAVES": env_bool("TPU_FUSED_OPROJ_AR_WAVES", True),
    # Per-channel FP8 dense linears whose decode weight XLA reads from HBM: route
    # M <= 16 through the weight-streaming Pallas kernel
    # (kernels/quantized_matmul/fp8_skinny.py), XLA otherwise. Outputs differ from
    # the XLA path in the f32 accumulation (exact here; XLA's fp8 x fp8 dot is not).
    # A selected layer whose width is not a multiple of 128 (q_a + kv_a, 2624) also
    # keeps a zero-padded copy of its weight for the kernel: +16.5 MB per layer on
    # GLM-5.2-FP8, i.e. about 1.3 GB per rank over 79 layers.
    # --- Sampler ---
    # Lighter non-greedy sampler / spec-verify (rejection) programs. Skips the
    # 32-pass top-k bisection when no active request sets top_k, and the
    # 33-pass top-p bisection when none sets top_p < 1 (both decided on the
    # host per step; a skipped mask is the identity, so tokens are exact). The
    # greedy draft argmax all-gathers a (max, id) pair per row instead of the
    # full logits (spec_decode/utils.vocab_parallel_argmax).
    "TPU_SAMPLER_VOCAB_PARALLEL": env_bool("TPU_SAMPLER_VOCAB_PARALLEL", True),
    # MTP draft (DeepSeek / GLM-5.2): split the replicated eh_proj (2H -> H,
    # nn.Linear upstream) by output column across TP and all-gather the
    # result, so each draft pass reads H*2H/tp weight bytes instead of H*2H
    # (spec_decode/utils.ColumnParallelEhProj). Same per-element dot product.
    "TPU_MTP_EH_PROJ_TP": env_bool("TPU_MTP_EH_PROJ_TP"),
    # MTP draft: the two tiny TP all-gathers per draft pass -- the column-
    # parallel eh_proj output (TPU_MTP_EH_PROJ_TP) and the vocab-parallel
    # argmax (max, id) pairs (TPU_SAMPLER_VOCAB_PARALLEL) -- become a TP
    # all-reduce of a zero buffer holding this rank's block in its own slot,
    # which TPU_SMALL_ALL_REDUCE_DRAFT routes to the small Pallas all-reduce
    # (~8 us) instead of XLA's all-gather (~14 us at 16 cores). Bitwise the
    # gather up to the sign of a zero: each element sums one block and 15 +0.
    "TPU_MTP_ONEHOT_AR": env_bool("TPU_MTP_ONEHOT_AR", default=True),
    # MTP draft (DeepSeek-V3.2 / GLM-5.2 DSA): draft step 0 of a chunk with at
    # least this many (padded) tokens runs in two passes. Pass 1 only writes the
    # MTP layer's KV and indexer-K caches for every row; pass 2 runs the full
    # layer on one row per request, the only rows a draft token is read from.
    # Same draft tokens and caches, without the indexer, sparse attention, MoE
    # and DCP merge of every other prefill row (spec_decode/mtp_kv_only.py).
    # 0 (default) keeps the single full pass.
    "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS": env_nonnegative_int(
        "TPU_MTP_PREFILL_KV_ONLY_MIN_TOKENS", 0
    ),
    # Bits of the float32 threshold bisection resolved per vocab pass in
    # top-k/top-p masking (2**bits - 1 candidate thresholds per pass, same
    # result as 1 bit/pass since the predicate is monotonic). 0 = 1. Measured
    # slower than 1 on tpu7x at decode shapes; kept for other vocab/row counts.
    "TPU_SAMPLER_BSEARCH_RADIX_BITS": env_nonnegative_int(
        "TPU_SAMPLER_BSEARCH_RADIX_BITS", 0
    ),
    # Non-greedy sampler + verify: run the top-p threshold bisection without
    # the steps proven never to set a bit for softmax probs (bit 30; bits 15..0
    # on the bf16 bonus rows): 15 vocab passes instead of 32 there, 31 on f32
    # verify rows. Same threshold bit for bit (top_k_top_p.topp_mask). A top-K
    # candidate prefilter was measured 10x slower (lax.top_k sorts the vocab).
    "TPU_SAMPLER_FAST_TOP_P": env_bool("TPU_SAMPLER_FAST_TOP_P", True),
    # Non-greedy spec verify: logits gathers, bonus sampling, the per-token
    # param gathers and the rejection sampler as ONE compiled program (the
    # three uniform draws stay eager: same generator, shapes and order).
    "TPU_SAMPLER_FUSED_CHAIN": env_bool("TPU_SAMPLER_FUSED_CHAIN", True),
    "TPU_FP8_LINEAR_PALLAS": env_bool("TPU_FP8_LINEAR_PALLAS", True),
    # With TPU_SAMPLER_FUSED_CHAIN: a verify chunk whose drafts fit in 8 rows
    # runs the rejection sampler's full-vocab target work (softmax, top-p
    # bisection, recovery argmax) on 8 target rows instead of the padded
    # logits length (16 at 1-2 requests with MTP k=3). The dropped rows are
    # padding slots; the uniforms are drawn at the full shape and sliced, so
    # tokens and accept decisions are unchanged.
    "TPU_SAMPLER_VERIFY_TRIM": env_bool("TPU_SAMPLER_VERIFY_TRIM"),
    # With TPU_SAMPLER_FUSED_CHAIN: the fused verify chain draws its three
    # uniforms (bonus_u, accept_u, recover_u) inside the compiled program
    # instead of as six eager programs before it. Dynamo cannot take a
    # torch.Generator, so the sampling generator becomes the device's default
    # generator, seeded once from model_config.seed exactly like the dedicated
    # one; a compiled torch.rand on it yields the same bits as the eager draw
    # with the same state, so the stream (and every token) is unchanged as long
    # as nothing else draws from the default generator while serving.
    "TPU_SAMPLER_GRAPH_RNG": env_bool("TPU_SAMPLER_GRAPH_RNG", default=True),
    # With TPU_FP8_LINEAR_PALLAS: the skinny kernel takes its per-channel weight
    # scale as an HBM operand and DMAs it to VMEM itself, issued with the weight
    # ring. Without it XLA stages the scale (o_proj f32[6144]) into VMEM with a copy
    # placed right before the call, whose ~0.8 us issue latency is exposed each
    # layer at decode. Applies to the sites in fp8_skinny.SCALE_DMA_SITES (o_proj).
    # Outputs are bit-identical.
    "TPU_FP8_SKINNY_SCALE_DMA": env_bool("TPU_FP8_SKINNY_SCALE_DMA"),
    # With TPU_FP8_LINEAR_PALLAS: comma-separated layer-prefix suffixes ("."-bounded,
    # as TPU_FP8_LINEAR_PALLAS_SITES) whose skinny kernel takes its weight operand
    # constrained to HBM (pltpu.with_memory_space_constraint). The kernel streams the
    # weight from HBM itself; without the constraint XLA's memory-space assignment
    # may stage it into VMEM with a copy the call then waits on (o_proj 6.29 MB, q_b
    # 16.8 MB at TP16 x DCP8 decode), exposed whenever too little runs between the
    # copy's start and the call. Per site, not global: a site whose staging copy is
    # hidden (qkv_a under its prefetch override) keeps it. Bit-identical outputs.
    "TPU_FP8_SKINNY_HBM_PIN_SITES": env_str(
        "TPU_FP8_SKINNY_HBM_PIN_SITES", "self_attn.o_proj,q_b_proj"
    ),
    # The LM head (vocab-parallel `F.linear(x, W)`, W bf16 [vocab/tp, hidden] N-major)
    # at <= 64 logits rows runs as a Pallas kernel that streams W from HBM through a
    # 3-slot VMEM ring (kernels/lm_head_hbm.py; GLM-5.2 TP16 [9680, 6144] on tpu7x:
    # ~36.6-37.3 us vs XLA's 44-47 us per call). Other row counts keep XLA's dot.
    # "drafter": only an MTP/eagle draft's passes (the draft gets its own head object
    # sharing the target's weight) -- drafts only, lossless for the target
    # distribution. "all": also the target sampler's logits. NOT bit-identical: ~1-2
    # in 10^4 logits differ from XLA's dot by 1-4 bf16 ulp (MXU accumulation order),
    # so "all" needs an accuracy check. "" = off. Default "drafter" (trial 6).
    "TPU_LM_HEAD_HBM": env_str("TPU_LM_HEAD_HBM", "drafter"),
    # RMSNorm (and fused add + RMSNorm) on >= MIN_TOKENS rows of a >= MIN_HIDDEN
    # residual stream runs as one Pallas pass (layers/adapter/rmsnorm.py) instead of
    # XLA's reduce + re-read apply. Decode-sized batches are unchanged. Residual
    # output is bit-identical; the normalized output can differ in the last bf16 ulp
    # (f32 summation order of the mean square).
    "TPU_FUSED_ADD_RMSNORM": env_bool("TPU_FUSED_ADD_RMSNORM"),
    "TPU_FUSED_ADD_RMSNORM_MIN_TOKENS": env_int(
        "TPU_FUSED_ADD_RMSNORM_MIN_TOKENS", 512
    ),
    "TPU_FUSED_ADD_RMSNORM_MIN_HIDDEN": env_int(
        "TPU_FUSED_ADD_RMSNORM_MIN_HIDDEN", 4096
    ),
    # With TPU_FUSED_ADD_RMSNORM: a residual-stream norm whose output feeds a
    # per-channel FP8 linear (the MLA fused_qkv_a_proj; a dense layer's
    # gate_up_proj) also emits that linear's per-row fp8 activation + scale from
    # the same Pallas pass, and the linear skips its amax + convert passes over
    # the bf16 output. Bit-identical. Only where the Pallas norm already runs
    # (the same rows predicate); decode buckets are untouched.
    "TPU_FUSED_ADD_RMSNORM_QUANT": env_bool("TPU_FUSED_ADD_RMSNORM_QUANT", True),
    # With TPU_FUSED_ADD_RMSNORM, USE_MOE_TP_DECODE_KERNEL, FUSED_MOE_PREFILL_KERNEL
    # and FUSED_MOE_PREFILL_KERNEL_V2: a MoE layer's post-attention norm also
    # emits the stage-A prefill kernel's per-row fp8 activation rows and scales
    # from its Pallas pass, and the kernel skips its own amax / relayout /
    # convert passes (#1935 trial 4). Bit-identical to the kernel's own
    # quantization. Only where the Pallas norm already runs and its rows
    # threshold is at or above the layer's prefill cut-over; buckets where the
    # stage-A build does not serve (blocked mode) quantize as before.
    "TPU_FUSED_ADD_RMSNORM_QUANT_MOE": env_bool("TPU_FUSED_ADD_RMSNORM_QUANT_MOE"),
    # Comma-separated layer-prefix suffixes that take that path, matched on a "."
    # boundary. The defaults are the sites XLA does not prefetch on GLM-5.2-FP8
    # decode; a VMEM-prefetched weight (indexer wq_b, shared experts, dense
    # down_proj) is faster under XLA than any HBM stream.
    "TPU_FP8_LINEAR_PALLAS_SITES": env_str(
        "TPU_FP8_LINEAR_PALLAS_SITES",
        "q_b_proj,self_attn.o_proj,mlp.gate_up_proj",
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


def is_set(env_name: str) -> bool:
    """True when ``env_name`` is set to a non-empty value, i.e. not defaulted.

    Defaults that only make sense on some deployments (the GLM-5.2 AgentX
    flags baked in from #1510) step aside quietly where they do not apply,
    but an explicit setting still fails loudly there.
    """
    value = os.getenv(env_name)
    return value is not None and value.strip() != ""


def is_indexer_kv_cache_mode_explicit() -> bool:
    """True when VLLM_TPU_INDEXER_KV_CACHE_MODE was explicitly set."""
    return "VLLM_TPU_INDEXER_KV_CACHE_MODE" in os.environ


def bucket_gate_edge(min_tokens: int) -> int:
    """The cap `c` a lower token gate "buckets of at least `min_tokens`" is
    written against: callers test `T > c`, never `T >= min_tokens`.

    Token-count gates that pick a code path are `T <= cap` / `T > cap` with
    `cap` a power of 2 (a compile-bucket edge), so a lower gate and an upper
    gate at the same edge are disjoint instead of both holding at one bucket
    (two gates true at a single value pin the traced token count to it: the
    trial-4 bucket-64 ShapeSpecializationError). For a power-of-2
    `min_tokens` the edge is the bucket below it, `min_tokens // 2`: on the
    power-of-2 bucket ladder `T > min_tokens // 2` picks exactly the buckets
    `T >= min_tokens` did. A `min_tokens` of 0 or 1 is every bucket (`T > 0`);
    a non-power-of-2 one keeps its exact meaning (`T > min_tokens - 1`)."""
    m = int(min_tokens)
    if m <= 1:
        return 0
    if m & (m - 1) == 0:
        return m // 2
    return m - 1
