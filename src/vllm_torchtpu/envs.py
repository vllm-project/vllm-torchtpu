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
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    TPU_MOE_DENSE_EXPERT_THRESHOLD: int = 0
    FUSED_MOE_DECODE_KERNEL: bool = False
    FUSED_MOE_DECODE_KERNEL_MAX_TOKENS: int = 256
    TPU_MOE_SINGLE_SORT: bool = True
    USE_MOE_TP_DECODE_KERNEL: bool = False
    RAGGED_GATHER_REDUCE_VERSION: str = "v2"
    USE_PHASED_PROFILER: bool = False
    TPU_KERNEL_ITER_MODE: bool = False
    TPU_KERNEL_RELOAD_MODULES: str = ""
    TPU_GREEDY_TOP_TOKENS: bool = False
    TPU_SAMPLER_FUSED_CHAIN: bool = True
    TPU_SAMPLER_GRAPH_RNG: bool = False
    DP_SCHED_ENABLED: bool = False
    TPU_KV_HOLD_REPLAY_CHECKPOINT: bool = False
    TPU_FINE_PREFIX_HIT_TOKENS: int = 0
    TPU_STEP_TIMING_LOG_STEPS: int = 0
    TPU_ASYNC_DRAFT_BEFORE_RESULT_WAIT: bool = False
    TPU_KV_DROP_HCA_STATE_CHECKPOINT: bool = False
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
    TPU_MOE_SKIP_PADDED_TOKENS: bool = False
    TPU_MOE_HIERARCHICAL_EP: bool = False
    TPU_MOE_ROUTER_TOPK: str = "rowmax"
    USE_MOE_FUSED_EP_KERNEL: bool = False
    MOE_FUSED_EP_KERNEL_IMPL: str | None = None
    MOE_FUSED_EP_ENABLE_W4A8: bool = False
    MOE_FUSED_EP_KERNEL_MIN_TOKENS: int = 1024
    MOE_FUSED_EP_V2_SHARDED_PLAN: bool = False
    MOE_FUSED_EP_SORTED_GATHER: bool = False
    MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE: int = 0
    MOE_FUSED_EP_V2_DECODE_TILE_M: int = 0
    MOE_FUSED_EP_V2_DECODE_TILE_MAX_LOCAL_TOKENS: int = 16
    MOE_FUSED_EP_V2_MERGED_EXCHANGE: bool = True
    MOE_FUSED_EP_V2_ONEHOT_SCATTER: bool = True
    MOE_FUSED_EP_V2_PALLAS_PLAN: bool = True
    MOE_FUSED_EP_V2_DECODE_COMBINE: bool = True
    MOE_FUSED_EP_V2_PALLAS_RESTORE: bool = True
    MOE_FUSED_EP_V2_FUSED_SELECT: bool = True
    MOE_FUSED_EP_V2_LANE_QUANT: bool = True
    MOE_FUSED_EP_V2_ONESHOT_GATHER: bool = True
    MOE_FUSED_EP_V2_COMBINE_RESTORE: bool = False
    MOE_FUSED_EP_V2_FOLD_RESTORE_INTO_SHARED_AR: bool = False
    MOE_FUSED_EP_V2_TAIL_BARRIER: bool = False
    MOE_FUSED_EP_V2_FILL_SCHEDULE: bool = True
    TPU_EXPERT_PLACEMENT: str = ""
    MOE_EXPERT_COUNT_TAP_DIR: str = ""
    MOE_EXPERT_COUNT_TAP_EVERY: int = 1000
    MOE_FUSED_EP_SHARED_BEFORE_KERNEL: bool = False
    MOE_FUSED_EP_SHARED_IN_OP: bool = True
    MOE_FUSED_EP_SHARED_RS_MIN_ROWS: int = -1
    TPU_LATENT_PROJ_INTRA_CHIP_TP: bool = False
    TPU_DSV4_ATTN_INTRA_CHIP_TP: bool = False
    TPU_DSV4_PAIR_SWAP: bool = False
    TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK: tuple[int, int, int] = (256, 128, 128)
    TPU_DSV4_HCA_NUM_QUERIES_PER_BLOCK: tuple[int, int, int] = (1, 32, 32)
    TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK: tuple[int, int, int] = (64, 128, 64)
    TPU_DSV4_INDEXER_NUM_QUERIES_PER_BLOCK: tuple[int, int, int] = (1, 32, 32)
    TPU_DSV4_CSA_GATHER_AND_ATTENTION_CHUNK_SIZE: int = 64
    TPU_DSV4_COMPRESSOR_PROJ_RUNS_MIN_TOKENS: int = 128
    TPU_DSV4_COMPRESSOR_PROJ_RUNS_TILE_N: int = 512
    TPU_DSV4_COMPRESSOR_PROJ_RUNS_INDEXER: bool = False
    TPU_DSV4_COMPRESSOR_PREFILL_BLOCK: int = 128
    TPU_DSV4_COMPRESSOR_PREFILL_BLOCK_MIN_TOKENS: int = 256
    TPU_DSV4_COMPRESSOR_PROJ_STREAM_MAX_TOKENS: int = 0
    TPU_DSV4_COMPRESSOR_PROJ_STREAM_CFG: tuple[int, int, int] = (3584, 2, 1)
    TPU_DSV4_COMPRESSOR_FP8: bool = False
    TPU_DSV4_COMPRESSOR_PAIR: bool = False
    TPU_DSV4_COMPRESSOR_PROJ_STREAM_MIN_WIDTH: int = 0
    TPU_TOKEN_BUCKET_EXTRA: list[int] = []
    TPU_SMALL_ALL_REDUCE: bool = True
    TPU_SMALL_ALL_REDUCE_MAX_ROWS: int = 0
    TPU_SMALL_ALL_REDUCE_PAIRS: int = 0
    TPU_SMALL_COLLECTIVES_SKIP_DEVICE_BARRIER: bool = False
    MOE_FUSED_EP_V2_FOLD_ANY_DECODE: bool = False
    TPU_DSV4_EP_MIXED_PLAN_KERNEL: bool = False
    TPU_DSV4_EP_MIXED_EXCHANGE_ONESHOT: bool = False
    TPU_DSV4_EP_MIXED_COMBINE_KERNEL: bool = False
    MOE_FUSED_EP_V2_ONESHOT_GATHER_MAX_LOCAL_TOKENS: int = 4
    TPU_FP8_LINEAR_PALLAS: bool = True
    TPU_FP8_LINEAR_PALLAS_SITES: str = ""
    TPU_FP8_SKINNY_HBM_PIN_SITES: str = FP8_LINEAR_PALLAS_DEFAULT_SITES
    TPU_FP8_SKINNY_SCALE_DMA: bool = True
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
    TPU_STREAMIDX_VERIFY_BD: bool = False
    TPU_DSV4_SC_TOPK_FLAT_SCORES: bool = True
    TPU_DSV4_SC_TOPK_COOP_MAX_ROWS: int = 16
    TPU_DSV4_SWA_NUM_QUERIES_PER_BLOCK: tuple[int, int, int] = (1, 32, 32)
    TPU_DSV4_SWA_Q_COMPUTE_BLOCK_SIZE: int = 16
    TPU_DSV4_SWA_KV_BLOCK_TOKENS: int = 256
    TPU_DSV4_HCA_PREFETCH_EARLY: bool = False
    TPU_DSV4_FUSED_HCA_DECODE: bool = False
    TPU_DSV4_FUSED_HCA_BLK_PAGES: int = 128
    TPU_DSV4_FUSED_HCA_ROPE: bool = False
    TPU_DSV4_FUSED_HCA_MIXED: bool = False
    TPU_DSV4_FUSED_HCA_MIXED_KT: int = 2048
    TPU_DSV4_FUSED_HCA_MIXED_KT_TAIL: int = 512
    TPU_DSV4_FUSED_HCA_MIXED_Q_ROWS: int = 1024
    TPU_DSV4_SWA_SINGLE_PASS: bool = False
    TPU_DSV4_SWA_SINGLE_PASS_MAX_TOKENS: int = 32
    TPU_DSV4_SWA_SINGLE_PASS_BQ: int = 8
    TPU_DSV4_INDEXER_KV_SPLIT_MIN_TOKENS: int = 0
    TPU_STREAMIDX_STRICT_CAUSAL: bool = True
    TPU_DCP_INDEXER_A2A_MERGE: bool = False
    TPU_DCP_INDEXER_A2A_MIN_TOKENS: int = 64
    TPU_STREAMIDX_SMALL_BUCKET_BQ: int | None = None
    TPU_STREAMIDX_SMALL_BUCKET_BKVP: int | None = None
    TPU_STREAMIDX_SMALL_BUCKET_TOKENS: int = 64
    TPU_DSV4_CSA_GATHER_NO_PAD: bool = True
    TPU_DSV4_CSA_UNION_DENSE: bool = False
    TPU_DSV4_SMLA_UNION_VERIFY: bool = False
    TPU_DSV4_SMLA_UNION_VERIFY_MAX_TOKENS: int = 128
    TPU_DSV4_SMLA_UNION_VERIFY_ABOVE: int = 0
    TPU_DSV4_SMLA_UNION_VERIFY_EARLY: bool = True
    TPU_DSV4_SMLA_UNION_VERIFY_FENCE: int = 0
    TPU_DSV4_QNORM_ROPE_FLAT: bool = False
    TPU_DSV4_ROPE_QUANT_FLAT: bool = False
    TPU_DSV4_CSA_UNION_DENSE_MIN_TOKENS: int = 512
    TPU_DSV4_CSA_WINDOW_MIXED: bool = False
    TPU_DSV4_CSA_WINDOW_MIXED_ABOVE: int = 256
    TPU_DSV4_CSA_WINDOW_MIXED_MIN_HEADS: int = 32
    TPU_DSV4_CSA_WINDOW_MIXED_GROUPS: int = 1
    TPU_DSV4_CSA_WINDOW_MIXED_MAX_TOKENS: int = 2048
    TPU_DSV4_CSA_WINDOW_MIXED_WQ: int = 8
    TPU_DSV4_CSA_UNION_DENSE_ROWS_PER_TOKEN: int = 160
    TPU_DSV4_CSA_UNION_DENSE_MAX_HEADS: int = 32
    TPU_DSV4_CSA_UNION_DENSE_MAX_KV_TP4: int = 0
    TPU_DSV4_CSA_TOPK_AFTER_Q_ABOVE: int = 0
    TPU_DSV4_CSA_TOPK_FENCE_Q: bool = False
    TPU_DSV4_CSA_TOPK_AFTER_KV: bool = False
    TPU_DSV4_SEQ_PARALLEL: bool = False
    TPU_NUM_GPU_BLOCKS_OVERRIDE: int = 0
    TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS: int = 0
    TPU_DSV4_SP_SHARED_TP: int = 0
    TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE: int = 0
    TPU_DSV4_INDEXER_TOKEN_SPLIT_PIECES: int = 1
    TPU_DSV4_INDEXER_GROUP_MAX: bool = False
    TPU_DSV4_INDEXER_GROUP_MAX_MIN_KV: int = 65536
    TPU_STREAMIDX_CHUNK_SCHED_GROUPS: bool = True
    TPU_DSPARK_VOCAB_PARALLEL_MARKOV: bool = True
    TPU_DSPARK_SHARDED_CONTEXT_KV: bool = True
    VLLM_TPU_INDEXER_KV_CACHE_MODE: str = "sal"
    VLLM_TPU_DSV4_MHC_GATES_IN_KERNEL: bool = True
    VLLM_TPU_DSV4_MHC_FUSE_NORM: bool = True
    VLLM_TPU_DSV4_MHC_STACKED_FN: bool = True
    VLLM_TPU_DSV4_MHC_TOKEN_BLOCK: int = 128
    VLLM_TPU_DSV4_MHC_GATES_MAX_TOKENS: int = 0
    VLLM_TPU_DSV4_MHC_SMALL_BLOCK: int = 16
    VLLM_TPU_DSV4_MHC_FN_CHUNKS: int = 3
    VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS: int = 0
    VLLM_TPU_DSV4_MHC_SEAM_FN_HBM: str = ""
    TPU_DSV4_MHC_AR_SEAM: bool = False
    TPU_DSV4_MHC_MOE_AR_SEAM: int = 0
    VLLM_TPU_DSV4_MHC_POST_HEAD: bool = True
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
    MM_ENCODER_FRAME_CHUNK_PATCH_SIZE: int = 0
    VLLM_TPU_STATIC_POOLER: bool = True
    VLLM_TPU_POOLING_KV_CACHE_TOKENS: int = -1
    VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS: int = 8
    VLLM_TPU_SRPT_ATTN_WORK_CAP: int = 0


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
# DeepSeek-V4 per-core decode linears at TP8 (attention TP, shared expert TP).
# DSv4 decode linears measured faster on the kernel (close-gaps fp8-linears);
# the shared expert is left out (no gain at c16, already prefetched at c4).
FP8_LINEAR_PALLAS_DEFAULT_SITES = (
    "attn.fused_wqa_wkv,attn.wq_b,attn.wo_b,indexer.wq_b"
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


def env_positive_int(env_name: str, default: int) -> Callable[[], int]:
    """Integer knob that must be greater than zero, such as a chunk size
    that is divided by."""
    parse_int = env_int(env_name, default)

    def _get_positive_int_env() -> int:
        value = parse_int()
        if value <= 0:
            raise ValueError(f"Invalid value '{value}' for {env_name}: must be > 0.")
        return value

    return _get_positive_int_env


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


def env_positive_int_triple(
    env_name: str, default: tuple[int, int, int]
) -> Callable[[], tuple[int, int, int]]:
    """Per-pass kernel block size written as ``decode,prefill,mixed``.

    Unset or empty reads as the default. Anything else must be exactly three
    comma-separated integers greater than zero.
    """

    def _get_positive_int_triple_env() -> tuple[int, int, int]:
        value = os.getenv(env_name)
        if value is None or value.strip() == "":
            return default
        parts = value.split(",")
        try:
            ints = tuple(int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}: must be three "
                "comma-separated integers (decode,prefill,mixed)."
            ) from exc
        if len(ints) != 3 or any(i <= 0 for i in ints):
            raise ValueError(
                f"Invalid value '{value}' for {env_name}: must be three "
                "comma-separated integers > 0 (decode,prefill,mixed)."
            )
        return ints[0], ints[1], ints[2]

    return _get_positive_int_triple_env


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
    # Non-greedy, grammar-free spec-verify steps run the logits gathers, bonus
    # sampling, per-token parameter gathers and the rejection sampler as one
    # compiled program instead of about ten. Same uniforms, same tokens.
    "TPU_SAMPLER_FUSED_CHAIN": env_bool("TPU_SAMPLER_FUSED_CHAIN", True),
    # With TPU_SAMPLER_FUSED_CHAIN: draw the chain's uniforms inside that
    # program instead of as eager programs before it. Dynamo cannot take a
    # torch.Generator, so the sampling generator becomes the device's default
    # generator, seeded once from model_config.seed like the dedicated one; a
    # compiled torch.rand on it yields the eager draw's bits, so the stream is
    # unchanged as long as nothing else draws from the default generator.
    "TPU_SAMPLER_GRAPH_RNG": env_bool("TPU_SAMPLER_GRAPH_RNG"),
    # Enable TpuDpScheduler.
    "DP_SCHED_ENABLED": lambda: bool(int(os.getenv("DP_SCHED_ENABLED") or "0")),
    # Engine core: sliding-window groups keep each request's replay-boundary
    # prefix-cache checkpoint until the request is freed, and the freed request
    # queues it with the prefix it serves (core/checkpoint_free_order.py).
    "TPU_KV_HOLD_REPLAY_CHECKPOINT": env_bool("TPU_KV_HOLD_REPLAY_CHECKPOINT"),
    # Engine core: prefix-cache hits on full-attention + sliding-window hybrids
    # land on multiples of this many tokens instead of the scheduler block; the
    # full-attention block holding the hit point is copied on write
    # (core/fine_prefix_hits.py). 0 = off.
    "TPU_FINE_PREFIX_HIT_TOKENS": env_nonnegative_int("TPU_FINE_PREFIX_HIT_TOKENS", 0),
    # Steps between per-rank "step timing" log lines: the step period grouped
    # by DP token bucket (runner/step_timing_log.py). 0 disables it.
    "TPU_STEP_TIMING_LOG_STEPS": env_nonnegative_int("TPU_STEP_TIMING_LOG_STEPS", 0),
    # Async scheduling + DFlash/DSpark: dispatch the drafter before the host
    # waits on the previous step's sampled-token copy (runner/tpu_runner.py).
    "TPU_ASYNC_DRAFT_BEFORE_RESULT_WAIT": env_bool("TPU_ASYNC_DRAFT_BEFORE_RESULT_WAIT"),
    # Model workers: DeepSeek-V4's 128-ratio compressor state cache is declared
    # not prefix-cacheable (no checkpoint cached, held or host-offloaded); see
    # core/hca_state_checkpoint.py.
    "TPU_KV_DROP_HCA_STATE_CHECKPOINT": env_bool("TPU_KV_DROP_HCA_STATE_CHECKPOINT"),
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
    # Decode buckets of fused EP MoE v2 (kernels/fused_moe/v2/layer.py
    # SMALL_BATCH_LOCAL_TOKENS, fused stage only), all bit-identical and ON
    # by default (set =0 to fall back to the XLA glue):
    # carry the routing exchange inside the activation all-gather (needs the
    # sharded plan); build the slab tables by TensorCore one-hot rather than
    # SparseCore scatter; build the whole routing plan in one Pallas call
    # (needs the sharded plan); sum the arrivals in one Pallas call; restore
    # the TP replicas' rows with a small Pallas all-gather; fold the score
    # bias and row-finiteness test into the top-k selector.
    "MOE_FUSED_EP_V2_MERGED_EXCHANGE": env_bool("MOE_FUSED_EP_V2_MERGED_EXCHANGE", True),
    "MOE_FUSED_EP_V2_ONEHOT_SCATTER": env_bool("MOE_FUSED_EP_V2_ONEHOT_SCATTER", True),
    "MOE_FUSED_EP_V2_PALLAS_PLAN": env_bool("MOE_FUSED_EP_V2_PALLAS_PLAN", True),
    "MOE_FUSED_EP_V2_DECODE_COMBINE": env_bool("MOE_FUSED_EP_V2_DECODE_COMBINE", True),
    "MOE_FUSED_EP_V2_PALLAS_RESTORE": env_bool("MOE_FUSED_EP_V2_PALLAS_RESTORE", True),
    "MOE_FUSED_EP_V2_FUSED_SELECT": env_bool("MOE_FUSED_EP_V2_FUSED_SELECT", True),
    # Decode buckets: quantize the wire rows straight into lane blocks.
    "MOE_FUSED_EP_V2_LANE_QUANT": env_bool("MOE_FUSED_EP_V2_LANE_QUANT", True),
    # Decode buckets with MERGED_EXCHANGE: move the merged payload with the
    # one-shot Pallas all-gather (barrier + direct remote DMAs).
    "MOE_FUSED_EP_V2_ONESHOT_GATHER": env_bool("MOE_FUSED_EP_V2_ONESHOT_GATHER", True),
    # Decode buckets with DECODE_COMBINE and PALLAS_RESTORE in effect: one
    # Pallas call sums the arrivals and pushes them to every TP member.
    "MOE_FUSED_EP_V2_COMBINE_RESTORE": env_bool("MOE_FUSED_EP_V2_COMBINE_RESTORE"),
    # Fused EP MoE v2, in-op shared expert at decode buckets (Pallas restore +
    # small all-reduce in effect): add each core's routed rows into its slot of
    # the shared partial and let the one small all-reduce sum both, instead of
    # a restore all-gather followed by the shared all-reduce. Not bit-exact
    # (one extra bf16 rounding moves before the reduce).
    "MOE_FUSED_EP_V2_FOLD_RESTORE_INTO_SHARED_AR": env_bool(
        "MOE_FUSED_EP_V2_FOLD_RESTORE_INTO_SHARED_AR"
    ),
    # With FOLD_RESTORE_INTO_SHARED_AR: fold at every decode-glue bucket the
    # small all-reduce takes (e.g. T32 at TP4, t_local 8, with
    # TPU_SMALL_ALL_REDUCE_MAX_ROWS=32), not only where the Pallas restore is
    # in effect (t_local <= 4).
    "MOE_FUSED_EP_V2_FOLD_ANY_DECODE": env_bool("MOE_FUSED_EP_V2_FOLD_ANY_DECODE"),
    # Mixed buckets (t_local > 16, sharded plan, quantized rows): build the
    # fused-EP routing plan in one Pallas call (mixed_plan_kernel) instead of
    # ~100 XLA ops and two SparseCore scatters. Same bits.
    "TPU_DSV4_EP_MIXED_PLAN_KERNEL": env_bool("TPU_DSV4_EP_MIXED_PLAN_KERNEL"),
    # With the mixed plan kernel on: move the routing exchange with the
    # one-shot Pallas all-gather on the TensorCore instead of XLA's all-gather
    # (SparseCore-offloaded ahead of the activation gather at c4/c8, a
    # synchronous TensorCore collective at c16). Same bits.
    "TPU_DSV4_EP_MIXED_EXCHANGE_ONESHOT": env_bool("TPU_DSV4_EP_MIXED_EXCHANGE_ONESHOT"),
    # Mixed buckets (t_local > 16, quantized rows): the combine (scale gather,
    # coefficient, row gather, k-sum, row_routes select) as one Pallas call
    # (mixed_combine_kernel) instead of a SparseCore gather + XLA/Pallas sum.
    "TPU_DSV4_EP_MIXED_COMBINE_KERNEL": env_bool("TPU_DSV4_EP_MIXED_COMBINE_KERNEL"),
    # Largest local token count the one-shot Pallas EP gather takes (smallest
    # stays 2).
    "MOE_FUSED_EP_V2_ONESHOT_GATHER_MAX_LOCAL_TOKENS": env_nonnegative_int(
        "MOE_FUSED_EP_V2_ONESHOT_GATHER_MAX_LOCAL_TOKENS", 4
    ),
    # A JSON file naming, per MoE layer, which global experts each EP rank
    # holds and in what slot order (expert_placement.py), or the name of one
    # shipped in vllm_torchtpu/expert_placements/. Empty: the checkpoint's
    # contiguous blocks.
    "TPU_EXPERT_PLACEMENT": env_str("TPU_EXPERT_PLACEMENT", ""),
    # Directory the per-layer routed-expert counts are written to
    # (layers/adapter/expert_count_tap.py), every MOE_EXPERT_COUNT_TAP_EVERY
    # steps. Empty disables the tap and leaves the program unchanged.
    "MOE_EXPERT_COUNT_TAP_DIR": env_str("MOE_EXPERT_COUNT_TAP_DIR", ""),
    "MOE_EXPERT_COUNT_TAP_EVERY": env_int("MOE_EXPERT_COUNT_TAP_EVERY", 1000),
    # Finish the shared expert before the fused EP kernel starts, so it runs
    # while the kernel's token all-gather is in flight instead of after the
    # kernel; see VllmMoERunner._apply_quant_method.
    # With MOE_FUSED_EP_SHARED_BEFORE_KERNEL: the fused EP op takes the shared
    # expert's TP-partial output and returns shared + routed, reduced. From
    # MOE_FUSED_EP_SHARED_RS_MIN_ROWS rows per TP replica up, the shared partial
    # is reduce-scattered onto each core's own rows before the routed rows'
    # replica all-gather (one reduce-scatter instead of an all-reduce); below,
    # it is all-reduced in the op (the runner's own arithmetic). -1 (default)
    # picks by TP size: 1024 rows at TP >= 8, 256 below (DSv4 twin: the
    # reduce-scatter wins at T2048 TP8 and T512 TP4); 0 never reduce-scatters.
    "MOE_FUSED_EP_SHARED_IN_OP": env_bool("MOE_FUSED_EP_SHARED_IN_OP", True),
    "MOE_FUSED_EP_SHARED_RS_MIN_ROWS": env_int("MOE_FUSED_EP_SHARED_RS_MIN_ROWS", -1),
    "MOE_FUSED_EP_SHARED_BEFORE_KERNEL": env_bool("MOE_FUSED_EP_SHARED_BEFORE_KERNEL"),
    # How many copies of the FFN body fused EP MoE v2 emits, standing in for
    # kernel.HEIGHT_RUNGS. A tile runs the shortest of them that covers its
    # live rows, so one is the ladder off. Zero leaves the choice to the
    # kernel.
    "MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE": lambda: int(
        os.getenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", "0") or 0
    ),
    # Tile height (= capacity) of fused EP MoE v2's decode builds: a fused
    # call with t_local <= MOE_FUSED_EP_V2_DECODE_TILE_MAX_LOCAL_TOKENS builds
    # at this height instead of the adapter's 128, with a single rung. At
    # decode every expert holds a few rows, so the 32-row tile computes what
    # the ladder's short rung computes today, while the program text drops
    # the 128-row FFN body copies; in a 61-layer program that text evicted
    # the next op's code from the TensorCore's instruction overlay (the
    # decode combine paid a ~4.6 us overlay load per layer). Per-row
    # arithmetic is unchanged, so the output bits are. Zero is off.
    "MOE_FUSED_EP_V2_DECODE_TILE_M": env_nonnegative_int("MOE_FUSED_EP_V2_DECODE_TILE_M", 0),
    "MOE_FUSED_EP_V2_DECODE_TILE_MAX_LOCAL_TOKENS": env_nonnegative_int(
        "MOE_FUSED_EP_V2_DECODE_TILE_MAX_LOCAL_TOKENS", 16
    ),
    # Restore fused EP MoE v2's all-pairs barrier after the transport drain.
    # The kernel's double head barrier is sufficient on its own; the trailing
    # one only waits for the slowest core (~4.7 us a call at EP16).
    "MOE_FUSED_EP_V2_TAIL_BARRIER": env_bool("MOE_FUSED_EP_V2_TAIL_BARRIER"),
    # Fused EP MoE v2's decode fill schedule: on chunk-streamed builds with at
    # most kernel.FILL_MAX_ROWS_PER_EXPERT routed rows per expert, stream each
    # weight slab in two chunks and wait the gate/up slab chunk by chunk, so
    # the first visit computes before its whole slab lands. 0 turns it off.
    "MOE_FUSED_EP_V2_FILL_SCHEDULE": env_bool("MOE_FUSED_EP_V2_FILL_SCHEDULE", True),
    # Shard Kimi K3's latent-MoE down/up projections across the cores of a
    # chip instead of replicating them. They are unquantized [7168, 3584] and
    # [3584, 7168] matmuls run at full HBM bandwidth, ~22% of decode device
    # time at TP=32; halving the bytes each core reads costs two on-package
    # all-gathers per layer. Independent of TPU_MOE_HIERARCHICAL_EP.
    "TPU_LATENT_PROJ_INTRA_CHIP_TP": env_bool("TPU_LATENT_PROJ_INTRA_CHIP_TP"),
    # Experimental: shard DSv4 attention wq_b/wo_a/wo_b across a chip's two
    # cores while KV and attention stay DP (deepseek_v4/attn_intra_chip.py).
    "TPU_DSV4_ATTN_INTRA_CHIP_TP": env_bool("TPU_DSV4_ATTN_INTRA_CHIP_TP"),
    # Under TPU_DSV4_ATTN_INTRA_CHIP_TP: wo_a fetches the peer's attention-output
    # rows by remote DMA inside the kernel, overlapped with its own rows
    # (o_projection.fused_reverse_rope_wo_a_projection_pair_swap), instead of a
    # strided copy + all-gather before it.
    "TPU_DSV4_PAIR_SWAP": env_bool("TPU_DSV4_PAIR_SWAP"),
    # DSv4 kernel block sizes, each "decode,prefill,mixed". Defaults are the
    # microbenchmark-tuned values, tuned at a 1024-token KV block; the KV page
    # counts are doubled for the default 512-token block so a kernel block
    # spans as many tokens. With --block-size 1024, halve them (128,64,64 and
    # 32,64,32) to get the tuned blocks back. HCA: compressed-KV pass
    # (mla_ragged_paged_attention).
    "TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK": env_positive_int_triple(
        "TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK", (256, 128, 128)
    ),
    "TPU_DSV4_HCA_NUM_QUERIES_PER_BLOCK": env_positive_int_triple(
        "TPU_DSV4_HCA_NUM_QUERIES_PER_BLOCK", (1, 32, 32)
    ),
    # Indexer: StreamIndex top-k scoring (streamindex_topk).
    "TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK": env_positive_int_triple(
        "TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK", (64, 128, 64)
    ),
    "TPU_DSV4_INDEXER_NUM_QUERIES_PER_BLOCK": env_positive_int_triple(
        "TPU_DSV4_INDEXER_NUM_QUERIES_PER_BLOCK", (1, 32, 32)
    ),
    # DSv4 compressor state projection (proj_and_save_state): calls of at least
    # this many tokens write one DMA per page-contiguous run of token states
    # instead of one per token and half-state (same bytes). 0 disables.
    "TPU_DSV4_COMPRESSOR_PROJ_RUNS_MIN_TOKENS": env_nonnegative_int(
        "TPU_DSV4_COMPRESSOR_PROJ_RUNS_MIN_TOKENS", 128
    ),
    "TPU_DSV4_COMPRESSOR_PROJ_RUNS_TILE_N": env_positive_int(
        "TPU_DSV4_COMPRESSOR_PROJ_RUNS_TILE_N", 512
    ),
    # Also the 128-dim indexer compressor (not bit-exact: last-bit differences
    # in the score half of the state).
    "TPU_DSV4_COMPRESSOR_PROJ_RUNS_INDEXER": env_bool(
        "TPU_DSV4_COMPRESSOR_PROJ_RUNS_INDEXER"
    ),
    # DSv4 compress-norm-rope-store of a step's prefill boundaries in blocks of
    # this many compressed tokens (prefill_block.py), CSA and the SAL indexer,
    # for calls of at least ..._MIN_TOKENS tokens (256 keeps it off c16's
    # 128-token decode bucket, which has no prefill boundaries). 0 disables.
    "TPU_DSV4_COMPRESSOR_PREFILL_BLOCK": env_nonnegative_int(
        "TPU_DSV4_COMPRESSOR_PREFILL_BLOCK", 128
    ),
    "TPU_DSV4_COMPRESSOR_PREFILL_BLOCK_MIN_TOKENS": env_nonnegative_int(
        "TPU_DSV4_COMPRESSOR_PREFILL_BLOCK_MIN_TOKENS", 256
    ),
    # DSv4 compressor projection: calls of at most this many tokens stream the
    # weight from HBM through a VMEM ring (proj_and_save_state.kernel_stream)
    # instead of the 4-step emit_pipeline kernel. 0 disables.
    "TPU_DSV4_COMPRESSOR_PROJ_STREAM_MAX_TOKENS": env_nonnegative_int(
        "TPU_DSV4_COMPRESSOR_PROJ_STREAM_MAX_TOKENS", 0
    ),
    # ...its ring as "k_tile,slots,dmas_per_tile". k_tile 3584 keeps the bf16
    # stored bytes identical to the emit_pipeline kernel.
    "TPU_DSV4_COMPRESSOR_PROJ_STREAM_CFG": env_positive_int_triple(
        "TPU_DSV4_COMPRESSOR_PROJ_STREAM_CFG", (3584, 2, 1)
    ),
    # Store the compressor projections (fused_wkv_wgate; bf16 in the
    # checkpoint) as per-output-channel e4m3 with an f32 scale, and run every
    # projection through the stream kernel (bf16 x fp8 dot, f32 accumulation).
    # Numerics change: needs accuracy evidence.
    "TPU_DSV4_COMPRESSOR_FP8": env_bool("TPU_DSV4_COMPRESSOR_FP8"),
    # CSA layers: the KV compressor's and the indexer compressor's projections
    # (same tokens) in one weight-stream call, for steps the stream serves
    # (TPU_DSV4_COMPRESSOR_PROJ_STREAM_MAX_TOKENS).
    "TPU_DSV4_COMPRESSOR_PAIR": env_bool("TPU_DSV4_COMPRESSOR_PAIR"),
    # Stream only compressor projections whose state is at least this wide
    # (CSA KV 2048, HCA 1024, indexer 512); narrower ones keep the pipeline
    # kernel. 0 = all.
    "TPU_DSV4_COMPRESSOR_PROJ_STREAM_MIN_WIDTH": env_nonnegative_int(
        "TPU_DSV4_COMPRESSOR_PROJ_STREAM_MIN_WIDTH", 0
    ),
    # CSA: query tokens per gather + attention chunk of the sparse attention
    # (sparse_ragged_paged_attention). Steps of at most 128 tokens keep 32.
    "TPU_DSV4_CSA_GATHER_AND_ATTENTION_CHUNK_SIZE": env_positive_int(
        "TPU_DSV4_CSA_GATHER_AND_ATTENTION_CHUNK_SIZE", 64
    ),
    # Slice the rotary cos_sin caches to max model len at load to
    # minimize xla layout data copy overhead. Text-only.
    "TPU_ROPE_CACHE_TRUNCATE": env_bool("TPU_ROPE_CACHE_TRUNCATE"),
    # Materialize the rotary cos_sin caches with a row-major ({1,0}) device
    # layout at load, May cost more HBM due to padding.
    "TPU_ROPE_CACHE_ROW_MAJOR": env_bool("TPU_ROPE_CACHE_ROW_MAJOR", default=False),
    "TPU_MOE_HASH_TABLE_ROW_MAJOR": env_bool(
        "TPU_MOE_HASH_TABLE_ROW_MAJOR", default=False
    ),
    # Sum small bf16 TP all-reduces with the Pallas two-shot kernel
    # (kernels/experimental/small_collectives) over the TP group's cores of the
    # expert-parallel mesh, instead of XLA's all-reduce. Needs an EP mesh.
    "TPU_SMALL_ALL_REDUCE": env_bool("TPU_SMALL_ALL_REDUCE", True),
    # Largest block, in rows, the kernel takes; XLA sums larger ones inside the
    # same op. 0 picks by TP group size (small_all_reduce_op.max_rows: 32 rows
    # at 8 cores, 16 below, where the kernel measured faster than XLA under the
    # DSv4 served LIBTPU flags).
    "TPU_SMALL_ALL_REDUCE_MAX_ROWS": env_nonnegative_int(
        "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 0
    ),
    # Four-member TP groups laid out as two chips (group positions 0,1 and 2,3
    # each the two cores of one chip, as DP4 x TP4 on 2x2x2 lists them): sum
    # with the small all-reduce's chip-aware "pairs" form (on-chip
    # reduce-scatter, cross-chip swap of pair sums, on-chip all-gather: half the
    # two-shot's chip-to-chip bytes) pipelined over this many column chunks.
    # Rounds twice, like XLA's psum. 0 = off (two-shot / one-shot as before).
    "TPU_SMALL_ALL_REDUCE_PAIRS": env_nonnegative_int(
        "TPU_SMALL_ALL_REDUCE_PAIRS", 0
    ),
    # Small Pallas collectives (all-reduce, one-shot gather): skip Mosaic's
    # default entry device barrier; their own group barrier already orders
    # every push after every member's entry.
    "TPU_SMALL_COLLECTIVES_SKIP_DEVICE_BARRIER": env_bool(
        "TPU_SMALL_COLLECTIVES_SKIP_DEVICE_BARRIER"
    ),
    # Per-channel FP8 dense linears at decode token counts (<= fp8_skinny.MAX_M rows)
    # run as the weight-streaming Pallas kernel (kernels/quantized_matmul/
    # fp8_skinny.py) instead of XLA's quantize + fp8 dot, on the layers named by
    # TPU_FP8_LINEAR_PALLAS_SITES whose (K, N) has a measured tiling. Outputs differ
    # from the XLA path (exact f32 accumulation): tolerance-tier numerics.
    "TPU_FP8_LINEAR_PALLAS": env_bool("TPU_FP8_LINEAR_PALLAS", True),
    # Comma-separated layer-prefix suffixes, "."-bounded, e.g. "attn.wo_b".
    "TPU_FP8_LINEAR_PALLAS_SITES": env_str(
        "TPU_FP8_LINEAR_PALLAS_SITES", FP8_LINEAR_PALLAS_DEFAULT_SITES
    ),
    # Of the TPU_FP8_LINEAR_PALLAS sites, those whose kernel weight operand is
    # constrained to HBM, so XLA cannot stage it into VMEM ahead of the call.
    # Same arithmetic, bit-identical outputs; placement only.
    "TPU_FP8_SKINNY_HBM_PIN_SITES": env_str(
        "TPU_FP8_SKINNY_HBM_PIN_SITES", FP8_LINEAR_PALLAS_DEFAULT_SITES
    ),
    # HBM-pinned fp8_skinny sites with N % 1024 == 0: the per-channel weight scale
    # is an HBM operand DMA'd in-kernel with the weight ring, instead of a VMEM
    # operand XLA stages with an exposed copy just before the call. Bit-identical.
    "TPU_FP8_SKINNY_SCALE_DMA": env_bool("TPU_FP8_SKINNY_SCALE_DMA", True),
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
    # DSv4 sliding-window attention kernel blocks (decode, prefill, mixed)
    # query tokens per block, q rows per compute block, KV tokens per block.
    "TPU_DSV4_SWA_NUM_QUERIES_PER_BLOCK": env_positive_int_triple(
        "TPU_DSV4_SWA_NUM_QUERIES_PER_BLOCK", (1, 32, 32)
    ),
    # 16 (was a hardcoded 4): bit-exact, mixed T2048 SWA pass 0.497 -> 0.359 ms
    # per layer on one core; decode unchanged.
    "TPU_DSV4_SWA_Q_COMPUTE_BLOCK_SIZE": env_positive_int(
        "TPU_DSV4_SWA_Q_COMPUTE_BLOCK_SIZE", 16
    ),
    "TPU_DSV4_SWA_KV_BLOCK_TOKENS": env_positive_int(
        "TPU_DSV4_SWA_KV_BLOCK_TOKENS", 256
    ),
    # DSv4 sliding-window attention: run every sequence in one mixed-case kernel
    # launch instead of three (decode / verify / mixed), two of which are empty
    # on a speculative-decoding decode step (~1-1.5 us per empty launch).
    "TPU_DSV4_SWA_SINGLE_PASS": env_bool("TPU_DSV4_SWA_SINGLE_PASS"),
    # ... only for buckets of T <= this many tokens (decode steps), with this
    # mixed-case q block (a verify window, not a prefill chunk).
    "TPU_DSV4_SWA_SINGLE_PASS_MAX_TOKENS": env_int(
        "TPU_DSV4_SWA_SINGLE_PASS_MAX_TOKENS", 32
    ),
    "TPU_DSV4_SWA_SINGLE_PASS_BQ": env_positive_int("TPU_DSV4_SWA_SINGLE_PASS_BQ", 8),
    # DSv4 HCA attention: issue the next KV block's page DMAs before the current
    # block's compute instead of after it, so the copy overlaps the compute.
    "TPU_DSV4_HCA_PREFETCH_EARLY": env_bool("TPU_DSV4_HCA_PREFETCH_EARLY"),
    # DSv4 HCA layers, steps of T <= 32 tokens: one fused kernel for the SWA window
    # + compressed rows (one online softmax + sink, new rows stored in-kernel)
    # instead of the SWA kernel, the HCA MLA kernels and their hand-off glue.
    "TPU_DSV4_FUSED_HCA_DECODE": env_bool("TPU_DSV4_FUSED_HCA_DECODE"),
    # ... HCA pages (8 compressed rows each) per KV block of that kernel.
    "TPU_DSV4_FUSED_HCA_BLK_PAGES": env_positive_int("TPU_DSV4_FUSED_HCA_BLK_PAGES", 128),
    # ... and (with TPU_DSV4_FUSED_HCA_DECODE) the HCA layers' q RMSNorm + RoPE and kv RoPE move
    # into the attention op: in the fused kernel's prologue at T <= 32, the stock rope kernels otherwise.
    "TPU_DSV4_FUSED_HCA_ROPE": env_bool("TPU_DSV4_FUSED_HCA_ROPE"),
    # DSv4 HCA layers, steps of T > 32 tokens (mixed / prefill-chunk steps): one fused kernel
    # (fused_hca_mixed) per layer instead of the SWA kernel + MLA-p + MLA-m + hand-off glue; each
    # request's compressed rows are read once into VMEM and reused by all its query blocks.
    "TPU_DSV4_FUSED_HCA_MIXED": env_bool("TPU_DSV4_FUSED_HCA_MIXED"),
    # ... compressed rows per wide compute tile (where every query row of the block sees them), per
    # narrow tail / visibility-edge tile, and query rows (tokens x heads) per prefill block.
    "TPU_DSV4_FUSED_HCA_MIXED_KT": env_positive_int("TPU_DSV4_FUSED_HCA_MIXED_KT", 2048),
    "TPU_DSV4_FUSED_HCA_MIXED_KT_TAIL": env_positive_int("TPU_DSV4_FUSED_HCA_MIXED_KT_TAIL", 512),
    "TPU_DSV4_FUSED_HCA_MIXED_Q_ROWS": env_positive_int("TPU_DSV4_FUSED_HCA_MIXED_Q_ROWS", 1024),
    # DSv4 indexer: from this many tokens per bucket on, each TP rank scores
    # only its contiguous 1/TP of every sequence's compressed positions; the
    # ranks exchange their top-k candidates (all-to-all) and merge them (same
    # indices, same order as the replicated indexer). 0 = replicated indexer.
    # Ported from patemotter fdcc70a0d/cccc1c5b7/ec1c92c5f/9fb77a965.
    "TPU_DSV4_INDEXER_KV_SPLIT_MIN_TOKENS": env_int(
        "TPU_DSV4_INDEXER_KV_SPLIT_MIN_TOKENS", 0
    ),
    "TPU_STREAMIDX_CHUNK_TOKENS": env_positive_int_or_none(
        "TPU_STREAMIDX_CHUNK_TOKENS"
    ),
    # StreamIndex verify pass (speculative-decoding windows of <= 8 tokens):
    # score two lane tiles per 256-deep MXU pass with a block-diagonal q
    # (16 rows per head) and reduce the heads in packed bf16 (relu(q.k) and the
    # weights rounded to bf16, products accumulated in f32), the numerics the
    # mixed/prefill pass already uses. Halves the VPU work of the reduction.
    "TPU_STREAMIDX_VERIFY_BD": env_bool("TPU_STREAMIDX_VERIFY_BD"),
    # DSv4 q-RMSNorm + RoPE / indexer rope_quant: read the wq_b projection in its
    # flat [T, heads * dim] (8, 128) tiles inside the kernel instead of letting
    # XLA relay it out to [T, heads, dim] first (a TC copy per layer at decode,
    # a SparseCore data-format call at prefill). Bit-exact. Ported from cuiq
    # forge/cmp-c90c0ac7 af9afe1f1 / a2c62f3b6.
    "TPU_DSV4_QNORM_ROPE_FLAT": env_bool("TPU_DSV4_QNORM_ROPE_FLAT"),
    "TPU_DSV4_ROPE_QUANT_FLAT": env_bool("TPU_DSV4_ROPE_QUANT_FLAT"),
    # DSv4 indexer: hand the SparseCore top-k the scoring kernel's [T, S, 128]
    # buffer as is, without padding rows, instead of reshaping it to [T, S*128]
    # and padding the rows to the top-k's wave size. The bytes are identical, so
    # XLA passes it as a bitcast; the padded form costs a relayout, a pad and a
    # SparseCore data-format conversion (~48 us per CSA layer at T8/T16, 1M
    # max-model-len). Same selection. Set 0 to restore the padded form.
    "TPU_DSV4_SC_TOPK_FLAT_SCORES": env_bool(
        "TPU_DSV4_SC_TOPK_FLAT_SCORES", default=True
    ),
    # DSv4 indexer: token buckets up to which the SparseCore top-k splits every
    # row over a group of subcores (the cooperative pass) instead of one
    # subcore per row. A decode bucket holds a few long verify rows, which
    # otherwise leave most of the 32 subcores idle. Same selection. 0 = off.
    # (port of patemotter 591c2370b; 32-row buckets measured no better)
    "TPU_DSV4_SC_TOPK_COOP_MAX_ROWS": env_int("TPU_DSV4_SC_TOPK_COOP_MAX_ROWS", 16),
    # DSv4 indexer rows see only the compressed groups complete at their own
    # position, (pos + 1) // ratio, in multi-token forwards (chunked prefill,
    # speculative verify). 0 restores the old bound, under which a row can
    # select a group that a later token of the same forward completes.
    "TPU_STREAMIDX_STRICT_CAUSAL": env_bool("TPU_STREAMIDX_STRICT_CAUSAL", default=True),
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
    # DSv4 CSA: size the SparseCore KV gather's per-subcore chunk so its output
    # needs no padding rows, and hence no slice copy after it.
    "TPU_DSV4_CSA_GATHER_NO_PAD": env_bool("TPU_DSV4_CSA_GATHER_NO_PAD", True),
    # CSA prefill (buckets of at least ..._MIN_TOKENS tokens): attend each query
    # block's (64 tokens at <= 16 local heads, else 32) union of top-k rows once,
    # gathered on SparseCore, with a per-token selection mask
    # (`csa_union_dense`), instead of gathering every token's own 1024 rows.
    # Same keys per token. A step whose union rows exceed ROWS_PER_TOKEN x
    # tokens (where it would not pay) falls back to the sparse path.
    "TPU_DSV4_CSA_UNION_DENSE": env_bool("TPU_DSV4_CSA_UNION_DENSE"),
    # CSA decode buckets (T <= ..._MAX_TOKENS, a power of 2): cut every request
    # into windows of <= 8 tokens (a speculative verify window is one) and
    # attend each window's union of top-k rows once (`csa_window_union`: one
    # SparseCore call dedups, ranks and gathers the union rows and writes each
    # token's keep mask; one TensorCore call attends all windows), instead of
    # gathering and pushing every token's own 1024 rows. Same keys per token;
    # no context cap, no runtime fallback.
    "TPU_DSV4_SMLA_UNION_VERIFY": env_bool("TPU_DSV4_SMLA_UNION_VERIFY"),
    "TPU_DSV4_SMLA_UNION_VERIFY_MAX_TOKENS": env_int(
        "TPU_DSV4_SMLA_UNION_VERIFY_MAX_TOKENS", 128
    ),
    # ... and only buckets of T > this many tokens (a power of 2; 0 = all).
    "TPU_DSV4_SMLA_UNION_VERIFY_ABOVE": env_int("TPU_DSV4_SMLA_UNION_VERIFY_ABOVE", 0),
    # Issue the union SparseCore call as soon as the top-k exists (it then runs
    # behind the SWA pass on the TensorCore) instead of after the SWA pass.
    "TPU_DSV4_SMLA_UNION_VERIFY_EARLY": env_bool(
        "TPU_DSV4_SMLA_UNION_VERIFY_EARLY", True
    ),
    # ... and issue that SparseCore call only once the attention op's q (1) or
    # q and roped kv (2) exist. Unfenced, XLA starts it -- and so waits on the
    # indexer top-k -- before the q projection: served c16 T32 exposed
    # ~16.5 us/CSA layer of top-k wait that the base arm hid behind SWA.
    # Scheduling only. 0 = unfenced.
    "TPU_DSV4_SMLA_UNION_VERIFY_FENCE": env_int("TPU_DSV4_SMLA_UNION_VERIFY_FENCE", 0),
    # CSA buckets of T > ..._ABOVE tokens (a power of 2, above the decode
    # buckets) with at least ..._MIN_HEADS local heads (32 = TP4): attend each
    # request's 8-token windows' unions (`csa_window_union`, the decode route's
    # kernels run in SparseCore waves) instead of union-dense blocks. Same keys
    # per token; no context cap, no runtime fallback.
    "TPU_DSV4_CSA_WINDOW_MIXED": env_bool("TPU_DSV4_CSA_WINDOW_MIXED"),
    "TPU_DSV4_CSA_WINDOW_MIXED_ABOVE": env_int("TPU_DSV4_CSA_WINDOW_MIXED_ABOVE", 256),
    "TPU_DSV4_CSA_WINDOW_MIXED_MIN_HEADS": env_int("TPU_DSV4_CSA_WINDOW_MIXED_MIN_HEADS", 32),
    # Window groups per step: one SparseCore call per group, so group g + 1's
    # dedup/gather runs under group g's TensorCore attention (1 = one call).
    "TPU_DSV4_CSA_WINDOW_MIXED_GROUPS": env_int("TPU_DSV4_CSA_WINDOW_MIXED_GROUPS", 1),
    # ... and buckets of T <= this many tokens (a power of 2; at TP8 the
    # union-dense bq-64 kernel stays ahead at T2048 below ~100k-token prefixes).
    "TPU_DSV4_CSA_WINDOW_MIXED_MAX_TOKENS": env_int("TPU_DSV4_CSA_WINDOW_MIXED_MAX_TOKENS", 2048),
    # Tokens per window on the mixed route: 8, or 16 (M = 16 x 16 heads = 256
    # rows per pushed tile at TP8; half the gathered rows of 8-token windows).
    "TPU_DSV4_CSA_WINDOW_MIXED_WQ": env_int("TPU_DSV4_CSA_WINDOW_MIXED_WQ", 8),
    "TPU_DSV4_CSA_UNION_DENSE_MIN_TOKENS": env_int(
        "TPU_DSV4_CSA_UNION_DENSE_MIN_TOKENS", 512
    ),
    "TPU_DSV4_CSA_UNION_DENSE_ROWS_PER_TOKEN": env_int(
        "TPU_DSV4_CSA_UNION_DENSE_ROWS_PER_TOKEN", 160
    ),
    # Only at most this many local query heads (16 = TP8, 32 = TP4). At TP4 (T512
    # buckets) the plan runs as one pass on two subcores per block.
    "TPU_DSV4_CSA_UNION_DENSE_MAX_HEADS": env_int(
        "TPU_DSV4_CSA_UNION_DENSE_MAX_HEADS", 32
    ),
    # At more than 16 local heads (TP4): route a sequence to union-dense only
    # when its compressed KV (seq_len // 4, + 1) is <= this many rows; longer
    # sequences stay on the sparse path, and a step with none routed skips the
    # SparseCore plan. At TP4 the dense pass costs ~heads x union, so it pays
    # only at short contexts (-17% @22.5k, -7% @51k, 0% @100k tokens per call),
    # and a step that falls back pays the plan for nothing. 0 = no limit.
    "TPU_DSV4_CSA_UNION_DENSE_MAX_KV_TP4": env_int(
        "TPU_DSV4_CSA_UNION_DENSE_MAX_KV_TP4", 0
    ),
    # CSA buckets above this many tokens fence the top-k with the projected
    # q before the attention op uses it, so the indexer's top-k all-gather
    # (token split) completes under the q projection instead of right behind
    # its start. Scheduling only. 0 = off.
    "TPU_DSV4_CSA_TOPK_AFTER_Q_ABOVE": env_int("TPU_DSV4_CSA_TOPK_AFTER_Q_ABOVE", 0),
    # Indexer token-split path: fence this rank's SparseCore top-k with the
    # projected + roped q before the top-k all-gather, so the q projection
    # runs while the top-k is on the SparseCore instead of after its done.
    # Scheduling only.
    "TPU_DSV4_CSA_TOPK_FENCE_Q": env_bool("TPU_DSV4_CSA_TOPK_FENCE_Q"),
    # With TPU_DSV4_CSA_TOPK_AFTER_Q_ABOVE: fence the top-k with the roped kv
    # too (the union plan starts after kv_rope). Scheduling only.
    "TPU_DSV4_CSA_TOPK_AFTER_KV": env_bool("TPU_DSV4_CSA_TOPK_AFTER_KV"),
    # DeepSeek-V4 sequence parallelism over the TP group: the mHC residual
    # stream, the mHC seams and the MoE (router, fused EP, a replicated shared
    # expert) run on this rank's 1/tp of the tokens. Attention all-gathers its
    # input and reduce-scatters its output (instead of one all-reduce), and the
    # fused EP needs no restore all-gather or shared reduce-scatter.
    "TPU_DSV4_SEQ_PARALLEL": env_bool("TPU_DSV4_SEQ_PARALLEL"),
    # Pin the KV pool to exactly this many blocks per engine, as
    # --num-gpu-blocks-override does, for launchers whose server command fixes
    # --gpu-memory-utilization (a served-config invariant: resident-weight
    # changes must not resize the pool). Ignored when the CLI flag is set.
    # 0 = off (pool sized from --gpu-memory-utilization).
    "TPU_NUM_GPU_BLOCKS_OVERRIDE": env_int("TPU_NUM_GPU_BLOCKS_OVERRIDE", 0),
    # Per-bucket gate on TPU_DSV4_SEQ_PARALLEL: steps of T <= this many tokens
    # (a power of 2) run the replicated layout -- one TP all-reduce after
    # attention, the TP-sharded shared expert, the fused EP's TP-replica op --
    # and only T > it run sequence parallel. 0: every bucket sequence parallel.
    "TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS": env_int("TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS", 0),
    # Under TPU_DSV4_SEQ_PARALLEL: keep only this rank's TP slice of the shared
    # expert (the full replicated copy, +3.75 GB/rank at TP8, is dropped after
    # load). The sequence-parallel side all-gathers its rows over the TP group,
    # runs the TP-sliced gate/up/down and reduce-scatters back onto its rows
    # (the replicated side of the gate already ran on the slice).
    # 0 = off (full weight on the SP side); 1 = bf16 row all-gather;
    # 2 = the rows quantized to fp8 before the all-gather (per-channel weight
    # scales only; the same per-row quantization the matmul applies after it).
    "TPU_DSV4_SP_SHARED_TP": env_int("TPU_DSV4_SP_SHARED_TP", 0),
    # Under TPU_DSV4_SEQ_PARALLEL (sequence-parallel side of the gate): for
    # T > this (a power of 2), each TP rank scores and selects the CSA indexer
    # top-k of its own contiguous T / tp rows only (the whole batch's metadata
    # and row lengths, so the same selection), and the indices are all-gathered.
    # Needs max_num_seqs x (spec + 1) <= 2 x this / tp (no verify window cut).
    # 0 = off.
    "TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE": env_int(
        "TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE", 0
    ),
    # Token-split indexer windows (above): cut each rank's window into this
    # many row pieces (a power of 2), each scored into its own buffer with its
    # own SparseCore top-k, so piece i's top-k can overlap piece i+1's scoring.
    # Same selection. 1 = one unpipelined pass.
    "TPU_DSV4_INDEXER_TOKEN_SPLIT_PIECES": env_int(
        "TPU_DSV4_INDEXER_TOKEN_SPLIT_PIECES", 1
    ),
    # Token-split indexer windows: the mixed scoring pass also writes each
    # row's 16-column group maxima, and the SparseCore top-k of long rows reads
    # those and gathers only the candidate groups (sparsecore_topk_pf) instead
    # of the whole score row. Same selection, same column order.
    "TPU_DSV4_INDEXER_GROUP_MAX": env_bool("TPU_DSV4_INDEXER_GROUP_MAX"),
    # ... only for sequences with at least this many compressed rows (both the
    # scorer's group-max work and the SparseCore prefilter; shorter rows keep
    # the full-row top-k, which is faster there). 0 = every eligible row.
    "TPU_DSV4_INDEXER_GROUP_MAX_MIN_KV": env_int(
        "TPU_DSV4_INDEXER_GROUP_MAX_MIN_KV", 65536
    ),
    # StreamIndex chunked pipeline: annotate consecutive chunks' SparseCore
    # top-k and TensorCore scoring with XLA scheduling groups (1, default), or
    # leave the overlap to the scheduler (0; the annotated form failed to
    # compile in served graphs: "annotation groups with gaps").
    "TPU_STREAMIDX_CHUNK_SCHED_GROUPS": env_bool(
        "TPU_STREAMIDX_CHUNK_SCHED_GROUPS", default=True
    ),
    # DSpark greedy drafting: shard the Markov head over the vocabulary and
    # exchange per-rank (max, argmax) pairs instead of reading the replicated
    # full-vocabulary weight at every draft position.
    "TPU_DSPARK_VOCAB_PARALLEL_MARKOV": env_bool("TPU_DSPARK_VOCAB_PARALLEL_MARKOV", True),
    # DSpark context-KV step: each tensor-parallel rank computes its 1/tp of
    # the replicated main_proj columns and all-gathers them, and one matmul
    # computes only the KV columns of every draft layer's fused_wqa_wkv.
    "TPU_DSPARK_SHARDED_CONTEXT_KV": env_bool("TPU_DSPARK_SHARDED_CONTEXT_KV", True),
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
    # DeepSeek-V4 mHC: compute the post/comb gates (sigmoid, softmax and
    # Sinkhorn) inside the Pallas pre / fused seam kernels instead of as an
    # XLA epilogue of ~130 tiny ops per seam.
    "VLLM_TPU_DSV4_MHC_GATES_IN_KERNEL": env_bool(
        "VLLM_TPU_DSV4_MHC_GATES_IN_KERNEL", True
    ),
    # DeepSeek-V4 mHC: the pre / fused seam kernels also apply the
    # sublayer's attn_norm / ffn_norm RMSNorm to the layer input they emit.
    "VLLM_TPU_DSV4_MHC_FUSE_NORM": env_bool("VLLM_TPU_DSV4_MHC_FUSE_NORM", True),
    # DeepSeek-V4 mHC: keep fn as its three bf16 chunks stacked (3 * 24, K)
    # and run the mix GEMM as one MXU pass (the MXU pads the 24-wide output
    # anyway). Not bit-identical to the three-pass path: the partial sums
    # round differently (measured closer to an f64 reference).
    "VLLM_TPU_DSV4_MHC_STACKED_FN": env_bool("VLLM_TPU_DSV4_MHC_STACKED_FN", True),
    # DeepSeek-V4 mHC: fused seam token block (0 = kernel default 32). 128
    # amortizes the per-grid-step cost at prefill sizes.
    "VLLM_TPU_DSV4_MHC_TOKEN_BLOCK": env_int("VLLM_TPU_DSV4_MHC_TOKEN_BLOCK", 128),
    # DeepSeek-V4 mHC: in-kernel gates only up to this many tokens
    # (-1 = kernel default 512, 0 = no cap; with a 128-token block the
    # token-major Sinkhorn fills every lane, so no cap is needed).
    "VLLM_TPU_DSV4_MHC_GATES_MAX_TOKENS": env_int(
        "VLLM_TPU_DSV4_MHC_GATES_MAX_TOKENS", 0
    ),
    # DeepSeek-V4 mHC: fused seam batches of at most this many tokens (with
    # in-kernel gates) run as one block of exactly their rows. 16 = off (the
    # bf16 sublane tile); 8 halves the f32 VPU work of a T<=8 decode seam
    # (6.42 -> 5.78 us at T=8; post/comb mixes move by f32 ulps).
    "VLLM_TPU_DSV4_MHC_SMALL_BLOCK": env_int("VLLM_TPU_DSV4_MHC_SMALL_BLOCK", 16),
    # DeepSeek-V4 mHC: bf16 chunks of the stacked fn the fused seam reads.
    # 3 = exact f32 mix GEMM (default); 2 = hi + mid (16 mantissa bits): the
    # lo third of the fn bytes and MXU pushes is skipped, seam -0.9 us at
    # T=8..32; mix logits move ~3e-6 relative (f64 oracle; 3-chunk 1.2e-7).
    "VLLM_TPU_DSV4_MHC_FN_CHUNKS": env_int("VLLM_TPU_DSV4_MHC_FN_CHUNKS", 3),
    # DeepSeek-V4 mHC: a fused seam of at most this many tokens reserves
    # 32 MiB of scoped VMEM instead of the 100 MiB default (clamped to 64 MiB),
    # which made XLA evict the AR output and the neighbours' prefetched
    # operands around every seam. 0 = off. A power of 2 (gate `T <= cap`).
    "VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS": env_int(
        "VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS", 0
    ),
    # DeepSeek-V4 mHC: which fused seams read their fn from HBM in the
    # kernel's own pipeline instead of an XLA VMEM prefetch ahead of the call:
    # "attn" (the seam after the MoE, gating the attention input), "ffn" (the
    # seam after attention), "both", or "" (none). With a 32 MiB seam
    # (VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS) XLA prefetches fn, and behind
    # the fused MoE the copy is exposed (~2.2 us). Applies only where the
    # 32 MiB reservation does (T <= the cap above). Bit-identical.
    "VLLM_TPU_DSV4_MHC_SEAM_FN_HBM": lambda: os.getenv(
        "VLLM_TPU_DSV4_MHC_SEAM_FN_HBM", ""
    ),
    # DeepSeek-V4 mHC: attention side -- wo_b's TP all-reduce and the ffn seam
    # as one Pallas call (mhc/ar_seam_kernel.py). Bit-identical; measured
    # parity with the low-VMEM unfused pair, so off.
    "TPU_DSV4_MHC_AR_SEAM": env_bool("TPU_DSV4_MHC_AR_SEAM"),
    # DeepSeek-V4 mHC: MoE side -- the fused EP op returns its pre-all-reduce
    # partial (fold path) and the next layer's attention seam sums it in the
    # same Pallas call as the seam, with hc_attn_fn DMA'd from HBM under the
    # all-reduce's ICI rounds (behind the fused MoE, XLA's fn prefetch is
    # exposed). 1 = fn by entry DMA (HBM-pinned), 2 = fn as an XLA-placed VMEM
    # block (removal arm for the fn overlap). 0 = off. Bit-identical.
    "TPU_DSV4_MHC_MOE_AR_SEAM": env_int("TPU_DSV4_MHC_MOE_AR_SEAM", 0),
    # DeepSeek-V4 mHC: fuse the posts that only feed the hc head collapse or
    # DSpark's aux stream mean into one kernel each, so the (T, 4, H) streams
    # are never written (one to three posts per forward).
    "VLLM_TPU_DSV4_MHC_POST_HEAD": env_bool("VLLM_TPU_DSV4_MHC_POST_HEAD", True),
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
    # SrptPrefillScheduler: cap on a step's attention work, the sum over its
    # scheduled tokens of the context each attends to, enforced by clipping
    # prefill chunks while some DP rank decodes (0 disables it).
    "VLLM_TPU_SRPT_ATTN_WORK_CAP": env_nonnegative_int(
        "VLLM_TPU_SRPT_ATTN_WORK_CAP", 0
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
