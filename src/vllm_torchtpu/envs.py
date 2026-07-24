# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    JAX_PLATFORMS: str = ""
    TPU_ACCELERATOR_TYPE: str | None = None
    TPU_NAME: str | None = None
    TPU_WORKER_ID: str | None = None
    TPU_MULTIHOST_BACKEND: str = ""
    SKIP_JAX_PRECOMPILE: bool = False
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    MODEL_IMPL_TYPE: str = "vllm"
    PHASED_PROFILING_DIR: str = ""
    PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR: int = 15
    PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP: int = 0
    PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD: int = -1
    PYTHON_TRACER_LEVEL: int = 1
    USE_MOE_EP_KERNEL: bool = False
    NUM_SLICES: int = 1
    RAY_USAGE_STATS_ENABLED: str = "0"
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "shm"
    ENABLE_QUANTIZED_MATMUL_KERNEL: bool = False
    REQUANTIZE_BLOCK_SIZE: int | None = None
    REQUANTIZE_WEIGHT_DTYPE: str = "float8_e4m3fn"
    MOE_REQUANTIZE_WEIGHT_DTYPE: str = "float8_e4m3fn"
    MOE_REQUANTIZE_BLOCK_SIZE: int | None = None
    TPU_KV_CACHE_HEADROOM_MIB: int = 5120
    TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL: bool = False
    TPU_KV_RESHARD_DST_PAGE_TOKENS: int = 0
    TPU_USE_RAIDEN_KV_CACHE_MANAGER: bool = False
    TPU_RAIDEN_QWEN35_ADMISSION: bool = False
    TPU_KV_RESHARD_TRANSPORT: str = "zmq"
    TPU_RAIDEN_CONTROLLER_ADDRESS: str = ""
    TPU_RAIDEN_JOB_NAME: str = ""
    TPU_RAIDEN_ENGINE_ID: str = "0"
    TPU_RAIDEN_TRANSFER_PARALLELISM: int = 8
    RAGGED_GATED_DELTA_RULE_IMPL: str = "chunked_kernel_v3_pd"
    USE_MOE_SPARSE_CORE: bool = True
    ONEHOT_MOE_PERMUTE_THRESHOLD: int = 0
    RAGGED_GATHER_VERSION: str = "v2"
    RAGGED_GATHER_REDUCE_VERSION: str = "v2"
    TPU_KERNEL_ITER_MODE: bool = False
    TPU_KERNEL_RELOAD_MODULES: str = ""
    DP_SCHED_ENABLED: bool = False
    DP_SCHED_BUFFER_PREFILL: bool = False
    DP_SCHED_BUFFER_PREFILL_TIMEOUT_MS: int = 10000


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
            raise ValueError(f"Invalid value '{value}' for {env_name}. "
                             f"Valid options: {actual_choices}.")

        return value

    return _get_validated_env


def env_bool(env_name: str, default: bool = False) -> Callable[[], bool]:
    """
    Accepts both numeric strings ("0", "1") and boolean strings
    ("true", "false", "True", "False").
    """

    def _get_bool_env() -> bool:
        value = os.getenv(env_name)
        if value is None or value == "":
            return default

        value_lower = value.lower()
        if value_lower in ("true", "1"):
            return True
        if value_lower in ("false", "0"):
            return False
        raise ValueError(
            f"Invalid boolean value '{value}' for {env_name}. "
            f"Valid options: '0', '1', 'true', 'false', 'True', 'False'.")

    return _get_bool_env


environment_variables: dict[str, Callable[[], Any]] = {
    # JAX platform selection (e.g., "tpu", "cpu", "proxy")
    "JAX_PLATFORMS":
    lambda: os.getenv("JAX_PLATFORMS", "").lower(),
    # TPU accelerator type (e.g., "v5litepod-16", "v4-8")
    "TPU_ACCELERATOR_TYPE":
    lambda: os.getenv("TPU_ACCELERATOR_TYPE", None),
    # Name of the TPU resource
    "TPU_NAME":
    lambda: os.getenv("TPU_NAME", None),
    # Worker ID for multi-host TPU setups
    "TPU_WORKER_ID":
    lambda: os.getenv("TPU_WORKER_ID", None),
    # Backend for multi-host communication on TPU
    "TPU_MULTIHOST_BACKEND":
    env_with_choices("TPU_MULTIHOST_BACKEND", "", ["ray"]),
    # Skip JAX precompilation step during initialization
    "SKIP_JAX_PRECOMPILE":
    lambda: bool(int(os.getenv("SKIP_JAX_PRECOMPILE") or "0")),
    # Check for XLA recompilation during execution
    "VLLM_XLA_CHECK_RECOMPILATION":
    lambda: bool(int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION") or "0")),
    # Model implementation type (e.g., "flax_nnx")
    "MODEL_IMPL_TYPE":
    env_with_choices("MODEL_IMPL_TYPE", "vllm",
                     ["vllm", "flax_nnx", "jetpack"]),
    # Directory to store phased profiling output
    "PHASED_PROFILING_DIR":
    lambda: os.getenv("PHASED_PROFILING_DIR", ""),
    "PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR":
    lambda: int(os.getenv("PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR", "15")),
    "PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP":
    lambda: int(os.getenv("PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP", "0")),
    "PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD":
    lambda: int(os.getenv("PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD", "-1")
                ),
    # Python tracer level for profiling
    "PYTHON_TRACER_LEVEL":
    lambda: int(os.getenv("PYTHON_TRACER_LEVEL") or "1"),
    # Use custom expert-parallel kernel for MoE (Mixture of Experts)
    "USE_MOE_EP_KERNEL":
    lambda: bool(int(os.getenv("USE_MOE_EP_KERNEL") or "0")),
    # Number of TPU slices for multi-slice mesh
    "NUM_SLICES":
    lambda: int(os.getenv("NUM_SLICES") or "1"),
    # Enable/disable Ray usage statistics collection
    "RAY_USAGE_STATS_ENABLED":
    lambda: os.getenv("RAY_USAGE_STATS_ENABLED", "0"),
    # Ray compiled DAG channel type for TPU
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":
    env_with_choices("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "shm", ["shm"]),
    # Enable the blockwise Pallas quantized matmul kernel for dense linears.
    "ENABLE_QUANTIZED_MATMUL_KERNEL":
    env_bool("ENABLE_QUANTIZED_MATMUL_KERNEL"),
    # Runtime block size for dense linear weight requantization.
    "REQUANTIZE_BLOCK_SIZE":
    lambda: int(block_size) if
    (block_size := os.getenv("REQUANTIZE_BLOCK_SIZE")) is not None else None,
    # Runtime dtype for dense linear weight requantization.
    "REQUANTIZE_WEIGHT_DTYPE":
    lambda: os.getenv("REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn"),
    # Specify dtype for quantized MoE weights
    "MOE_REQUANTIZE_WEIGHT_DTYPE":
    lambda: os.getenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn"),
    # Specify requantization block size for MoE weights
    "MOE_REQUANTIZE_BLOCK_SIZE":
    lambda: int(block_size) if (block_size := os.getenv(
        "MOE_REQUANTIZE_BLOCK_SIZE")) is not None else None,
    # HBM reserve kept out of vLLM KV-cache sizing for TPU runtime programs and
    # post-profiling activation allocations.
    "TPU_KV_CACHE_HEADROOM_MIB":
    lambda: int(os.getenv("TPU_KV_CACHE_HEADROOM_MIB") or "5120"),
    # Experimental TPU unified block-pool cache layout. Disabled by default to
    # preserve the compact-mamba allocation/indexing path.
    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL":
    env_bool("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL"),
    # Destination (decode-side) page geometry in tokens, required on the
    # producer: the byte-span lowering splits declarations at destination
    # page boundaries at registration time. 0 means unset.
    "TPU_KV_RESHARD_DST_PAGE_TOKENS":
    lambda: int(os.getenv("TPU_KV_RESHARD_DST_PAGE_TOKENS") or "0"),
    # Raiden admission gates: construct a Raiden KVCacheManager and
    # register the pool manifest derived from the live typed KV caches.
    # This does not switch the V2 strided transfer transport.
    "TPU_USE_RAIDEN_KV_CACHE_MANAGER":
    env_bool("TPU_USE_RAIDEN_KV_CACHE_MANAGER"),
    "TPU_RAIDEN_QWEN35_ADMISSION":
    env_bool("TPU_RAIDEN_QWEN35_ADMISSION"),
    # Controller-driven PCP->DP pool resharding. The default preserves
    # the existing v1 connector protocol; selecting ``raiden`` enables
    # the fail-closed controller path.
    "TPU_KV_RESHARD_TRANSPORT":
    env_with_choices("TPU_KV_RESHARD_TRANSPORT",
                     "zmq", ["zmq", "raiden"],
                     case_sensitive=False),
    "TPU_RAIDEN_CONTROLLER_ADDRESS":
    lambda: os.getenv("TPU_RAIDEN_CONTROLLER_ADDRESS", "").strip(),
    "TPU_RAIDEN_JOB_NAME":
    lambda: os.getenv("TPU_RAIDEN_JOB_NAME", "").strip(),
    "TPU_RAIDEN_ENGINE_ID":
    lambda: os.getenv("TPU_RAIDEN_ENGINE_ID", "0").strip(),
    "TPU_RAIDEN_TRANSFER_PARALLELISM":
    lambda: int(os.getenv("TPU_RAIDEN_TRANSFER_PARALLELISM") or "8"),
    # Gated Delta Rule implementation. Default is the fused Conv1D+GDN v3
    # kernel; the older `chunked_kernel_pd` (v1) path is intentionally not a
    # valid choice — v1's ConcatBitcast conv-state assembly corrupts persistent
    # I/O-aliased state under libtpu 0.0.42.1 (mmlu_pro collapse on Qwen3.5),
    # and v3 supersedes it in every config we run.
    "RAGGED_GATED_DELTA_RULE_IMPL":
    env_with_choices("RAGGED_GATED_DELTA_RULE_IMPL", "chunked_kernel_v3_pd", [
        "ref", "chunked_jax_pd", "chunked_kernel_p_jax_d",
        "chunked_kernel_p_recurrent_kernel_d", "recurrent_kernel_pd",
        "chunked_kernel_v3_pd"
    ]),
    # Selects the #193 SparseCore MoE token-movement path. When 0, the EP
    # ragged gather + gather-reduce fall back to the pre-#193 plain-JAX
    # path (functionally equivalent; valid-mask gating unchanged). Mirrors
    # upstream vLLM's VLLM_USE_FLASHINFER_MOE_* style of env-gated MoE kernel
    # selection (vLLM 0.19.0 has no --moe-backend; see fused_moe.py TODO).
    "USE_MOE_SPARSE_CORE":
    lambda: bool(int(os.getenv("USE_MOE_SPARSE_CORE") or "1")),
    # Use Onehot+Matmul for permute and unpermute before and after moe
    # when the batch size <= this threshold. When set to 0, this feature
    # is effectively disabled.
    "ONEHOT_MOE_PERMUTE_THRESHOLD":
    lambda: int(os.getenv("ONEHOT_MOE_PERMUTE_THRESHOLD") or "0"),
    # SparseCore MoE gather kernel version used by fused_moe_gmm.
    # "v2" (default) = ragged_gather_v2; "v1" = legacy ragged_gather.
    "RAGGED_GATHER_VERSION":
    env_with_choices("RAGGED_GATHER_VERSION", "v2", ["v1", "v2"]),
    # SparseCore MoE gather-reduce (combine) kernel version used by
    # fused_moe_gmm. "v2" (default) = ragged_gather_reduce_v2; "v1" = legacy
    # ragged_gather_reduce.
    "RAGGED_GATHER_REDUCE_VERSION":
    env_with_choices("RAGGED_GATHER_REDUCE_VERSION", "v2", ["v1", "v2"]),
    # Kernel-iteration mode: split the compiled graph at the Pallas custom
    # ops, keep kernel sources out of the compile-cache key, and enable the
    # /reload_kernel endpoint so kernel edits apply to a running server
    # without restarting it. Development aid; off by default.
    "TPU_KERNEL_ITER_MODE":
    lambda: bool(int(os.getenv("TPU_KERNEL_ITER_MODE") or "0")),
    # Comma-separated module names re-imported by /reload_kernel (and, in
    # kernel-iteration mode, excluded from the compile-cache key). Defaults
    # to the in-tree experimental RPA kernel modules; see
    # compilation/kernel_reload.py.
    "TPU_KERNEL_RELOAD_MODULES":
    lambda: os.getenv("TPU_KERNEL_RELOAD_MODULES", ""),
    # Enable TpuDpScheduler.
    "DP_SCHED_ENABLED":
    lambda: bool(int(os.getenv("DP_SCHED_ENABLED") or "0")),
    # TpuDpScheduler: buffer new prefills and flush them in batches.
    "DP_SCHED_BUFFER_PREFILL":
    lambda: bool(int(os.getenv("DP_SCHED_BUFFER_PREFILL") or "0")),
    # TpuDpScheduler: flush buffered prefills once the oldest pending request
    # has waited this long (bounds TTFT).
    "DP_SCHED_BUFFER_PREFILL_TIMEOUT_MS":
    lambda: int(os.getenv("DP_SCHED_BUFFER_PREFILL_TIMEOUT_MS") or "10000"),
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
