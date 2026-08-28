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
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
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
    TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL: bool | None = None
    TPU_USE_RAIDEN_KV_CACHE_MANAGER: bool = False
    TPU_RAIDEN_QWEN35_ADMISSION: bool = False
    TPU_RAIDEN_GLM_ADMISSION: bool = False
    TPU_KV_RESHARD_TRANSPORT: str = "zmq"
    TPU_RAIDEN_CONTROLLER_ADDRESS: str = ""
    TPU_RAIDEN_JOB_NAME: str = ""
    TPU_RAIDEN_ENGINE_ID: str = "0"
    TPU_RAIDEN_TRANSFER_PARALLELISM: int = 8
    TPU_RAIDEN_MAX_TRANSFER_TOKENS: int | None = None
    TPU_RAIDEN_RESHARD_IMPL: str = "controller"
    TPU_RAIDEN_CLIENT_IMPL: str = "cpp"
    TPU_RAIDEN_ADVERTISE_HOST: str = ""
    TPU_RAIDEN_RESHARD_PORT_BASE: int = 27000
    TPU_RAIDEN_STORE_DISPATCH_PORT_BASE: int = 27100
    TPU_RAIDEN_PREFIX_AWARE_LOAD: bool = False
    USE_MOE_SPARSE_CORE: bool = True
    ONEHOT_MOE_PERMUTE_THRESHOLD: int | None = None
    TPU_MOE_OWNER_OUTPUT_MODE: str = "off"
    RAGGED_GATHER_REDUCE_VERSION: str = "v2"
    USE_PHASED_PROFILER: bool = False
    TPU_KERNEL_ITER_MODE: bool = False
    TPU_KERNEL_RELOAD_MODULES: str = ""
    DP_SCHED_ENABLED: bool = False
    MLA_XPOSE_N_TILE_SIZE: int = 160
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_TPU_MOST_MODEL_LEN: int | None = None
    TPU_GDN_CONV_QK_PAIR_LAYOUT: bool = False
    TPU_MOE_SKIP_PADDED_TOKENS: bool = False
    TPU_MOE_HIERARCHICAL_EP: bool = False
    TPU_MOE_ROUTER_TOPK: str = "rowmax"
    USE_MOE_FUSED_EP_KERNEL: bool = False
    MOE_FUSED_EP_KERNEL_MIN_TOKENS: int = 1024
    MOE_FUSED_EP_V2_SHARDED_PLAN: bool = False
    TPU_TOKEN_BUCKET_EXTRA: list[int] = []
    TPU_ROPE_CACHE_TRUNCATE: bool = False
    TPU_ROPE_CACHE_ROW_MAJOR: bool = False
    TPU_MOE_HASH_TABLE_ROW_MAJOR: bool = False
    TPU_PARALLEL_PRECOMPILE: bool = False
    TPU_MOE_COLLECTION_CHUNK_SIZE: int = 0
    USE_BATCHED_RPA_SEQ_ON_LANE: bool = False
    USE_BATCHED_RPA_LONGCTX: bool = False
    VLLM_TPU_BLOCK_MAJOR_KV: bool = False
    TPU_SPARSE_MLA_NOPE_LAYOUT: str = "sparsecore"
    TPU_SPARSE_MLA_ROPE_LAYOUT: str = "tensorcore"


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


environment_variables: dict[str, Callable[[], Any]] = {
    # QK pair-blocked pooled GDN conv layout: full-width states interleave
    # Q and K row-pairs per tap so a TP-rank shard's first conv token maps
    # to one whole destination pool token (2 contiguous transfer spans per
    # rank per state group).  Identity for head-shard states.  Must match
    # across a disagg pair; the value rides the raiden layout fingerprint.
    # Resolution is layered (see env_override.py): an explicitly set
    # value always wins; otherwise raiden runs derive it from
    # TPU_RAIDEN_TRANSFER_PARALLELISM (enabled iff >= 8, and 8 is that
    # variable's default — so it is on by default under raiden).
    "TPU_GDN_CONV_QK_PAIR_LAYOUT":
    env_bool("TPU_GDN_CONV_QK_PAIR_LAYOUT", default=False),
    # Register the DeepSeek-V4 tokenizer that honors continue_final_message.
    # Off by default so tokenizer behavior matches stock vLLM; the DSv4
    # eval config turns it on, since a prefilled assistant turn needs it.
    "TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE":
    env_bool("TPU_DSV4_HONOR_CONTINUE_FINAL_MESSAGE", default=False),
    # NoPE and RoPE KV cache layouts allocated for sparse MLA.
    "TPU_SPARSE_MLA_NOPE_LAYOUT":
    env_with_choices("TPU_SPARSE_MLA_NOPE_LAYOUT", "sparsecore",
                     ["sparsecore", "tensorcore"]),
    "TPU_SPARSE_MLA_ROPE_LAYOUT":
    env_with_choices("TPU_SPARSE_MLA_ROPE_LAYOUT", "tensorcore",
                     ["sparsecore", "tensorcore"]),
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
    # Backend for multi-host communication on TPU. "" auto-detects: "mp" is
    # used whenever --nnodes > 1 and this isn't set to "ray"; set explicitly
    # to "mp" to force it (e.g. to silence the auto-detection log line).
    "TPU_MULTIHOST_BACKEND":
    env_with_choices("TPU_MULTIHOST_BACKEND", "", ["ray", "mp"]),
    # Check for XLA recompilation during execution
    "VLLM_XLA_CHECK_RECOMPILATION":
    lambda: bool(int(os.getenv("VLLM_XLA_CHECK_RECOMPILATION") or "0")),
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
    # TPU unified block-pool cache layout. Tri-state: an explicit 0/1 always
    # wins; unset defers to
    # platforms.tpu_block_size_utils.unified_kv_layout_enabled, which selects
    # the pool for the hybrid models that ship a pooled state path and keeps
    # everything else on the per-layer KV caches.
    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL":
    env_optional_bool("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL"),
    # Destination (decode-side) page geometry in tokens, required on the
    # producer: the byte-span lowering splits declarations at destination
    # page boundaries at registration time. 0 means unset.
    # Raiden admission gates: construct a Raiden KVCacheManager and
    # register the pool manifest derived from the live typed KV caches.
    # This does not switch the V2 strided transfer transport.
    "TPU_USE_RAIDEN_KV_CACHE_MANAGER":
    env_bool("TPU_USE_RAIDEN_KV_CACHE_MANAGER"),
    "TPU_RAIDEN_QWEN35_ADMISSION":
    env_bool("TPU_RAIDEN_QWEN35_ADMISSION"),
    "TPU_RAIDEN_GLM_ADMISSION":
    env_bool("TPU_RAIDEN_GLM_ADMISSION"),
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
    "TPU_RAIDEN_MAX_TRANSFER_TOKENS":
    env_optional_int("TPU_RAIDEN_MAX_TRANSFER_TOKENS"),
    # Reshard control-plane hosting (zero sidecar processes): "store" hosts the
    # reshard service inside each engine's rank-0 worker via an in-process
    # KVCacheStore; "controller" keeps the external (sidecar) controller.
    "TPU_RAIDEN_RESHARD_IMPL":
    env_with_choices("TPU_RAIDEN_RESHARD_IMPL",
                     "controller", ["controller", "store"],
                     case_sensitive=False),
    # Reshard client ownership: "cpp" backs the facade surface
    # with the C++ ReshardClient (encodings, transport, and error text in
    # C++, byte-compatible); "python" keeps the legacy Python facade as
    # the one-release rollback.
    "TPU_RAIDEN_CLIENT_IMPL":
    env_with_choices("TPU_RAIDEN_CLIENT_IMPL",
                     "cpp", ["cpp", "python"],
                     case_sensitive=False),
    # Host advertised for in-process reshard/dispatch services (store mode).
    "TPU_RAIDEN_ADVERTISE_HOST":
    lambda: os.getenv("TPU_RAIDEN_ADVERTISE_HOST", "").strip(),
    # Per-engine deterministic ports (store mode): engine dp_rank i binds
    # reshard service at RESHARD_PORT_BASE+i and its dispatch controller at
    # STORE_DISPATCH_PORT_BASE+i; peers derive the same addresses.
    "TPU_RAIDEN_RESHARD_PORT_BASE":
    lambda: int(os.getenv("TPU_RAIDEN_RESHARD_PORT_BASE") or "27000"),
    "TPU_RAIDEN_STORE_DISPATCH_PORT_BASE":
    lambda: int(os.getenv("TPU_RAIDEN_STORE_DISPATCH_PORT_BASE") or "27100"),
    # Decode-side prefix-cache-aware reshard loads: when true, the consumer
    # subtracts the local prefix-cache hit and transfers only the missing
    # FA suffix (page-aligned dst_skip_bytes clip planned by the source
    # store); GDN state always transfers whole. When false, or when the
    # installed client lacks dst_skip_bytes support, partial local hits skip
    # the remote transfer and compute their missing suffix on decode.
    "TPU_RAIDEN_PREFIX_AWARE_LOAD":
    env_bool("TPU_RAIDEN_PREFIX_AWARE_LOAD", False),
    # Selects the #193 SparseCore MoE token-movement path. When 0, the EP
    # ragged gather + gather-reduce fall back to the pre-#193 plain-JAX
    # path (functionally equivalent; valid-mask gating unchanged). Mirrors
    # upstream vLLM's VLLM_USE_FLASHINFER_MOE_* style of env-gated MoE kernel
    # selection (vLLM 0.19.0 has no --moe-backend; see fused_moe.py TODO).
    "USE_MOE_SPARSE_CORE":
    lambda: bool(int(os.getenv("USE_MOE_SPARSE_CORE") or "1")),
    # Use one-hot matmuls for MoE permute and unpermute when the routed row
    # count (num_tokens * topk) is <= this threshold. Unset or empty means
    # auto: fused_moe_gmm.resolve_onehot_permute_threshold derives the value
    # from SparseCore geometry (0 on hosts without SparseCore). 0 keeps every
    # size on the SparseCore path; a positive value forces that
    # threshold. The knob and its row-count semantics come from
    # tpu-inference PR #2674.
    "ONEHOT_MOE_PERMUTE_THRESHOLD":
    env_optional_int("ONEHOT_MOE_PERMUTE_THRESHOLD"),
    # Select the blockwise owner-output kernel after the outer one-hot MoE
    # threshold has selected the TensorCore path. "off" keeps the dense
    # one-hot combine; "on" enables the kernel for every shape that passes
    # its dtype, alignment, and VMEM safety checks.
    "TPU_MOE_OWNER_OUTPUT_MODE":
    env_with_choices("TPU_MOE_OWNER_OUTPUT_MODE",
                     "off", ["off", "on"],
                     case_sensitive=False),
    # SparseCore MoE gather kernel version used by fused_moe_gmm.
    # "v2" (default) = ragged_gather_v2; "v1" = legacy ragged_gather.
    "RAGGED_GATHER_VERSION":
    env_with_choices("RAGGED_GATHER_VERSION", "v2", ["v1", "v2"]),
    # SparseCore MoE gather-reduce (combine) kernel version used by
    # fused_moe_gmm. "v2" is the default; "v1" selects the legacy kernel and
    # "v3" selects the destination-major prototype.
    "RAGGED_GATHER_REDUCE_VERSION":
    env_with_choices("RAGGED_GATHER_REDUCE_VERSION", "v2", ["v1", "v2", "v3"]),
    # Capture one trace per inference phase instead of one continuous trace.
    # The trace directory and iteration limits come from `profiler_config`;
    # this only selects which profiler consumes them.
    "USE_PHASED_PROFILER":
    env_bool("USE_PHASED_PROFILER"),
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
    # Tile size for MLA transpose pipeline.
    "MLA_XPOSE_N_TILE_SIZE":
    lambda: int(os.getenv("MLA_XPOSE_N_TILE_SIZE", "160")),
    # Bucket padding gap for TPU compile sizes
    "VLLM_TPU_BUCKET_PADDING_GAP":
    lambda: int(os.getenv("VLLM_TPU_BUCKET_PADDING_GAP", "0")),
    # Most model length cap for TPU compile bucketing
    "VLLM_TPU_MOST_MODEL_LEN":
    lambda: int(val)
    if (val := os.getenv("VLLM_TPU_MOST_MODEL_LEN")) is not None else None,
    # Skip padded tokens from MoE, otherwise they will route like
    # real tokens and activate experts whose output is thrown away.
    "TPU_MOE_SKIP_PADDED_TOKENS":
    env_bool("TPU_MOE_SKIP_PADDED_TOKENS"),
    # Hierarchical MoE parallelism: expert-parallel between chips,
    # tensor-parallel between the two chiplets of a chip. Off by
    # default; it helps only at low concurrency, where the MoE is
    # weight-bandwidth bound and the per-device expert load is badly
    # imbalanced. See K3_HIERARCHICAL_EP_PLAN.md.
    "TPU_MOE_HIERARCHICAL_EP":
    env_bool("TPU_MOE_HIERARCHICAL_EP"),
    # MoE router top-k implementation: "rowmax" (default) is the sort-free
    # k-pass selection, "sort" is torch.topk, kept as an escape hatch.
    "TPU_MOE_ROUTER_TOPK":
    env_with_choices("TPU_MOE_ROUTER_TOPK", "rowmax", ["sort", "rowmax"]),
    # Run the whole expert-parallel MoE layer -- top-k, dispatch, both expert
    # matmuls, combine, and the cross-rank transport -- as one SPMD Pallas
    # program, instead of bracketing the per-rank compute with vLLM's
    # all-gather and reduce-scatter. The transport then overlaps the matmuls;
    # the reduce-scatter it replaces is otherwise fully exposed.
    "USE_MOE_FUSED_EP_KERNEL":
    env_bool("USE_MOE_FUSED_EP_KERNEL"),
    # Largest node-wide token count, out of the scheduler's cap, at or above
    # which the fused kernel is armed for the whole deployment. Decided once at
    # weight load, not per step: a per-step choice would need the token count
    # inside the traced graph, and on Qwen3.5-397B it also measured worse than
    # arming everything. Below the threshold the grouped-matmul path serves.
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS":
    lambda: int(os.getenv("MOE_FUSED_EP_KERNEL_MIN_TOKENS", "1024")),
    # Build each rank's two consumed routing-plan slices instead of
    # replicating the full T * topk * experts plan arithmetic on every rank.
    "MOE_FUSED_EP_V2_SHARDED_PLAN":
    env_bool("MOE_FUSED_EP_V2_SHARDED_PLAN"),

    # Slice the rotary cos_sin caches to max model len at load to
    # minimize xla layout data copy overhead. Text-only.
    "TPU_ROPE_CACHE_TRUNCATE":
    env_bool("TPU_ROPE_CACHE_TRUNCATE"),

    # Materialize the rotary cos_sin caches with a row-major ({1,0}) device
    # layout at load, May cost more HBM due to padding.
    "TPU_ROPE_CACHE_ROW_MAJOR":
    env_bool("TPU_ROPE_CACHE_ROW_MAJOR", default=False),
    "TPU_MOE_HASH_TABLE_ROW_MAJOR":
    env_bool("TPU_MOE_HASH_TABLE_ROW_MAJOR", default=False),
    # Extra token buckets added to the default list of buckets.
    "TPU_TOKEN_BUCKET_EXTRA":
    lambda: [
        int(v) for v in os.getenv("TPU_TOKEN_BUCKET_EXTRA", "").split(",")
        if v.strip()
    ],
    # Enable pre-compile rotation to speed up the startup time.
    "TPU_PARALLEL_PRECOMPILE":
    env_bool("TPU_PARALLEL_PRECOMPILE", default=False),
    # Post-gather MoE token chunk size for communication-computation pipelining.
    # When set to 0, chunking and communication pipelining are disabled.
    "TPU_MOE_COLLECTION_CHUNK_SIZE":
    lambda: int(os.getenv("TPU_MOE_COLLECTION_CHUNK_SIZE") or "0"),
    "USE_BATCHED_RPA_SEQ_ON_LANE":
    env_bool("USE_BATCHED_RPA_SEQ_ON_LANE"),
    # Temporary: selects the batched_rpa_longctx fork over mainline batched_rpa
    "USE_BATCHED_RPA_LONGCTX":
    env_bool("USE_BATCHED_RPA_LONGCTX"),
    # Enables the block-major KV cache layout for Raiden offloading. Bundles all
    # attention layer fragments into a single contiguous array in HBM, collapsing
    # per-block transfers from F independent DMAs into a single hardware DMA.
    # Refer to `vllm_torchtpu.offload.block_major_layout` for contract details.
    "VLLM_TPU_BLOCK_MAJOR_KV":
    env_bool("VLLM_TPU_BLOCK_MAJOR_KV"),
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
