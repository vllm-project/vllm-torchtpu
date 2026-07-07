# SPDX-License-Identifier: Apache-2.0

# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import vllm_torchtpu.env_override  # noqa: F401  # isort: skip

import os
import time
from typing import TYPE_CHECKING, Optional, Tuple, Union

import portpicker
import torch
import vllm.envs as vllm_envs
# Ensure the "tpu" torch.compile backend is registered before vllm
# tries to use it (e.g. in @torch.compile decorators at import time).
from torch_tpu._internal import compile as _register_tpu_backend  # noqa: F401
from torch_tpu._internal.utils import hardware
from vllm.platforms.interface import Platform, PlatformEnum

from vllm_torchtpu import envs
from vllm_torchtpu.layers.common.sharding import ShardingConfigManager
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.platforms.tpu_block_size_utils import \
    update_tpu_block_size_and_slot_config
from vllm_torchtpu.worker.tpu_rank_binding import ensure_pcp_local_rank_remap

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.inputs import ProcessorInputs
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams, SamplingType
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    ModelConfig = None
    VllmConfig = None
    ProcessorInputs = None
    PoolingParams = None
    AttentionBackendEnum = None
    SamplingParams = None
    SamplingType = None

logger = init_logger(__name__)

# TODO(ranlihao): add more flexible topology map
TPU_MULTIHOST_TOPOLOGY_MAP = {
    16: "2,2,2,2",
}

# ---------------------------------------------------------------------------
# Modules/attrs whose @torch.compile(dynamic=True) wrappers must be removed.
# Each entry is (module_path, function_name).
# ---------------------------------------------------------------------------
_DYNAMIC_COMPILE_TARGETS: list[tuple[str, str]] = [
    ("vllm.model_executor.layers.vocab_parallel_embedding",
     "get_masked_input_and_mask"),
    ("vllm.v1.sample.ops.logprobs", "batched_count_greater_than"),
    ("vllm.v1.sample.ops.topk_topp_sampler", "compiled_random_sample"),
    ("vllm.utils.deep_gemm", "per_block_cast_to_fp8"),
]

_dynamic_compile_unwrapped = False


def _is_language_model_only_config(model_config: "ModelConfig") -> bool:
    multimodal_config = model_config.multimodal_config
    if multimodal_config is None:
        return False
    return bool(multimodal_config.language_model_only)


def _configure_torchtpu_eager_mode() -> None:
    from torch_tpu._internal import execution_mode
    previous_mode = execution_mode.eager_mode
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_AND_FUSE
    logger.info("TorchTPU eager mode configured: %s (previous=%s)",
                execution_mode.eager_mode.name, previous_mode.name)


def _unwrap_dynamic_compile_fns() -> None:
    """Unwrap all known ``@torch.compile(dynamic=True)`` decorators in vllm.

    Safe to call multiple times; only the first call per process does work.
    Failures to import a module are silently ignored (the module may not be
    used at all in this configuration).
    """
    global _dynamic_compile_unwrapped
    if _dynamic_compile_unwrapped:
        return
    _dynamic_compile_unwrapped = True

    import importlib

    for module_path, fn_name in _DYNAMIC_COMPILE_TARGETS:
        try:
            mod = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError):
            continue
        fn = getattr(mod, fn_name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(mod, fn_name, fn.__wrapped__)
            logger.debug("Unwrapped @torch.compile(dynamic=True) from %s.%s",
                         module_path, fn_name)


def _patch_scheduler_mamba_external_kv() -> None:
    """Allow vLLM's mamba-align scheduler path to coexist with TPU PD loads.

    vLLM 0.22.1 asserts when `num_external_computed_tokens > 0` reaches
    `_mamba_block_aligned_split`, and then calls that split even for async KV
    load scheduling where `num_new_tokens == 0`. TPU PD decode uses external
    computed tokens for producer-restored KV, so mirror the local vLLM patch:
    count those tokens in the split boundary and skip the split while
    `load_kv_async` is scheduling the remote load.
    """
    import inspect
    import textwrap

    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_tpu_mamba_external_kv_patch", False):
        return

    def _mamba_block_aligned_split(
        self,
        request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_local_computed_tokens +
                               num_external_computed_tokens)
        if num_computed_tokens < max(request.num_prompt_tokens,
                                     request.num_tokens - 1):
            block_size = self.cache_config.block_size
            last_cache_position = (request.num_tokens -
                                   request.num_tokens % block_size)
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = (num_computed_tokens +
                                               num_new_tokens)
            if num_computed_tokens_after_sched < last_cache_position:
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (num_computed_tokens < last_cache_position <
                  num_computed_tokens_after_sched):
                num_new_tokens = last_cache_position - num_computed_tokens
        return num_new_tokens

    schedule_source = inspect.getsource(Scheduler.schedule)
    old_guard = "                if self.need_mamba_block_aligned_split:\n"
    new_guard = ("                if self.need_mamba_block_aligned_split "
                 "and not load_kv_async:\n")
    if old_guard in schedule_source:
        schedule_source = schedule_source.replace(old_guard, new_guard, 1)
        patched_namespace = {}
        exec(
            compile(
                textwrap.dedent(schedule_source),
                inspect.getsourcefile(Scheduler.schedule)
                or "<vllm_scheduler_patch>", "exec"),
            Scheduler.schedule.__globals__,
            patched_namespace,
        )
        Scheduler.schedule = patched_namespace["schedule"]
    elif new_guard not in schedule_source:
        raise RuntimeError(
            "Unsupported vLLM Scheduler.schedule mamba split guard")

    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
    Scheduler._tpu_mamba_external_kv_patch = True
    logger.info(
        "Applied TPU patch: allow mamba-align scheduler with external KV.")


def apply_tpu_patches() -> None:
    """Apply all module-level patches required for TorchTPU.

    Must run in **every** process (including spawned workers) because
    module-level state is re-initialized on import in spawned processes.
    All patches are idempotent.
    """
    from vllm_torchtpu import (_patch_default_moe_runner_select_forward,
                               _patch_disable_sequence_parallel_moe,
                               _patch_moe_no_ep_tp_scope,
                               _patch_rowparallel_defer_bias,
                               _patch_vllm_disable_compile_ranges,
                               _patch_vllm_tpu_group_custom_ops)
    from vllm_torchtpu.layers.vllm.custom_ops import _register_custom_ops
    _register_custom_ops()
    _patch_vllm_tpu_group_custom_ops()
    _patch_default_moe_runner_select_forward()
    _patch_vllm_disable_compile_ranges()
    _patch_disable_sequence_parallel_moe()
    _patch_moe_no_ep_tp_scope()
    _patch_rowparallel_defer_bias()
    from vllm_torchtpu import (_patch_disable_dp_ubatch,
                               _patch_multiproc_worker_global_rank_env)
    _patch_disable_dp_ubatch()
    _patch_multiproc_worker_global_rank_env()
    # Register the out-of-tree TPU vision-attention CustomOp by importing the
    # module: its @CustomOp.register_oot makes vLLM instantiate our
    # MMEncoderAttention subclass (Pallas flash kernel on forward_oot).
    import vllm_torchtpu.layers.vllm.vision_attention  # noqa: F401
    _configure_torchtpu_eager_mode()
    _unwrap_dynamic_compile_fns()


class TpuPlatform(Platform):
    # Registered via the `vllm.platform_plugins` entry point in pyproject.toml.
    _enum = PlatformEnum.OOT
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "PrivateUse1"
    ray_device_key: str = "TPU"
    dist_backend: str = "tpu_dist"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "tpu"

    supported_quantization: list[str] = [
        "tpu_int8", "compressed-tensors", "awq", "fp8", "mxfp4"
    ]

    additional_env_vars: list[str] = [
        "PHASED_PROFILING_DIR", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS",
        "TPU_MULTIHOST_BACKEND", "VLLM_MLA_DISABLE", "TPU_BACKEND_TYPE",
        "NEW_MODEL_DESIGN", "ENABLE_QUANTIZED_MATMUL_KERNEL",
        "REQUANTIZE_BLOCK_SIZE", "REQUANTIZE_WEIGHT_DTYPE",
        "MOE_REQUANTIZE_BLOCK_SIZE", "MOE_REQUANTIZE_WEIGHT_DTYPE",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
        "TORCHINDUCTOR_AUTOGRAD_CACHE", "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY"
    ]

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        del parser
        from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                         register_backend)

        register_backend(
            AttentionBackendEnum.FLASH_ATTN,
            "vllm_torchtpu.layers.vllm.attention.PallasAttentionBackend",
        )
        # Experimental batched RPA — opt in via `--attention-backend CUSTOM`.
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "vllm_torchtpu.layers.vllm.attention."
            "PallasBatchedRPAAttentionBackend",
        )

    @classmethod
    def get_worker_distributed_backend(cls, world_size: int) -> str:
        """Pick torch.distributed backend used by worker bootstrap.

        TPU collectives are not needed when world_size==1, and single-rank
        `tpu_dist` bootstrap is unstable in CI. Use `gloo` in that case while
        keeping `tpu_dist` for multi-rank runs.
        """
        if world_size == 1 and cls.dist_backend == "tpu_dist":
            return "gloo"
        return cls.dist_backend

    @classmethod
    def _prepare_singlehost_tpu_env(cls, world_size: int) -> None:
        """Set TORCH_TPU_* env vars needed by PjRt initialization.

        TPUWorker.init_device() always sets WORLD_SIZE in the env, which
        causes PjRt to require TORCH_TPU_SLICEBUILDER_ADDRESSES and
        TORCH_TPU_TOPOLOGY. For world_size > 1, topology is looked up
        via PCI scan using world_size (not auto-detected chip count) so
        slicebuilder and topology match the actual number of workers.
        """
        os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID",
                              str(time.time_ns()))

        sb_addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
        sb_count = len(sb_addresses.split(",")) if sb_addresses else 0
        if sb_count != world_size:
            sb_ports = [
                portpicker.pick_unused_port() for _ in range(world_size)
            ]
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
                f"localhost:{p}" for p in sb_ports)

        if world_size == 1:
            os.environ["TORCH_TPU_TOPOLOGY"] = "1,1,1"
        else:
            os.environ["TORCH_TPU_TOPOLOGY"] = \
                cls._get_tpu_topology(world_size)

    @classmethod
    def _get_tpu_topology(cls, world_size: int) -> str:
        """Return the torch_tpu-reported topology for the requested world size."""
        topology = hardware.get_tpu_topology(world_size)
        if topology is None:
            raise ValueError("No TPU devices found.")
        return topology

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: "AttentionBackendEnum",
                             attn_selector_config: "AttentionSelectorConfig",
                             **kwargs) -> str:
        backend_name = getattr(selected_backend, "name", None)
        if backend_name == "CUSTOM":
            logger.info("Using TPU Pallas attention backend (batched-RPA "
                        "variant via CUSTOM; requires block_size=256).")
            return ("vllm_torchtpu.layers.vllm.attention."
                    "PallasBatchedRPAAttentionBackend")

        if backend_name not in (None, "FLASH_ATTN"):
            logger.info("Cannot use %s backend on TPU.", selected_backend)

        logger.info("Using TPU Pallas attention backend via FLASH_ATTN.")
        return "vllm_torchtpu.layers.vllm.attention.PallasAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return hardware.get_tpu_device_name()

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.get_device_name().lower() == "tpu v6e":
            logger.info(
                "Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.")
            return torch.float8_e5m2
        return torch.float8_e4m3fn

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_torchtpu.lora.torch_punica_tpu.PunicaWrapperTPU"

    @classmethod
    def get_infinity_values(cls, dtype) -> Tuple[float, float]:
        return torch.finfo(dtype).min, torch.finfo(dtype).max

    @classmethod
    def can_update_inplace(cls):
        return False

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        return 1

    @classmethod
    def inference_mode(cls):
        return True

    @classmethod
    def get_compile_backend(cls) -> str:
        return "vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor"

    @classmethod
    def set_additional_forward_context(cls, *args, **kwargs) -> dict:
        """Move DP metadata onto the TPU for DP+EP.

        vLLM builds ``num_tokens_across_dp_cpu`` on CPU, but TPU compiled
        forwards require tensor arguments to be on device.
        """
        dp_metadata = kwargs.get("dp_metadata")
        if dp_metadata is not None:
            t = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
            if t is not None and getattr(t, "device", None) is not None \
                    and t.device.type == "cpu":
                tpu_t = t.to("tpu")
                try:
                    dp_metadata.num_tokens_across_dp_cpu = tpu_t
                except Exception:
                    # Frozen dataclass fallback.
                    object.__setattr__(dp_metadata, "num_tokens_across_dp_cpu",
                                       tpu_t)
        return {}

    @classmethod
    def _initialize_sharding_config(cls, vllm_config: VllmConfig) -> None:
        # Store the parsed sharding config keyed by vllm_config (not as an
        # attribute on it) so vLLM's replace()/compute_hash() machinery never
        # sees it. Read it back via ShardingConfigManager.get(vllm_config).
        sharding_config = ShardingConfigManager.initialize(vllm_config)
        logger.info(f"Initialized sharding configuration: {sharding_config}")

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        apply_tpu_patches()

        if vllm_envs.VLLM_TPU_USING_PATHWAYS:
            raise NotImplementedError(
                "Pathways is not supported by torchtpu-vllm. "
                "Unset VLLM_TPU_USING_PATHWAYS.")
        cls._initialize_sharding_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        pcp_config = PcpStaticSupportValidator.validate_platform_config(
            vllm_config,
            multihost_backend=envs.TPU_MULTIHOST_BACKEND,
        )
        pcp_size = pcp_config.pcp_size
        if pcp_config.enabled:
            logger.info("Using vLLM native multiprocess PCP world; PCP is not "
                        "represented as a JAX mesh axis.")

        from vllm.config import CompilationMode
        compilation_config = vllm_config.compilation_config
        if compilation_config.mode == CompilationMode.NONE:
            # --enforce-eager is set
            pass
        else:
            compilation_config.mode = CompilationMode.VLLM_COMPILE

        # No graph splitting — TPU handles the full graph including
        compilation_config.splitting_ops = []

        # Disable inductor-specific fusion passes (only available on CUDA).
        compilation_config.pass_config.fuse_norm_quant = False
        compilation_config.pass_config.fuse_act_quant = False
        compilation_config.pass_config.fuse_attn_quant = False
        compilation_config.pass_config.eliminate_noops = False

        # Set compile_sizes to match the token padding buckets used by
        # TPUModelRunner. These are the exact shapes PiecewiseBackend
        # will compile for.
        if compilation_config.compile_sizes is None:
            scheduler_config = vllm_config.scheduler_config
            compilation_config.compile_sizes = _get_token_paddings(
                min_token_size=16,
                max_token_size=scheduler_config.max_num_batched_tokens,
                padding_gap=vllm_envs.VLLM_TPU_BUCKET_PADDING_GAP,
            )
        else:
            compilation_config.compile_sizes = sorted(
                compilation_config.compile_sizes)
        # Clear compile_ranges_split_points — TPU always pads to exact
        # compile_sizes so catch-all ranges are never used.
        compilation_config.compile_ranges_split_points = []

        model_config = vllm_config.model_config
        if model_config is not None and model_config.dtype in (
                torch.float16,
                torch.float32,
        ):
            logger.warning(
                "The TPU backend currently does not support %s. "
                "Using bfloat16 instead.",
                model_config.dtype,
            )
            model_config.dtype = torch.bfloat16

        cache_config = vllm_config.cache_config
        # vLLM's CacheConfig._apply_block_size_default has already populated
        # block_size with DEFAULT_BLOCK_SIZE if the user didn't pass one, so
        # `block_size is None` is never true by this point. The authoritative
        # signal is the `user_specified_block_size` flag pydantic sets.
        block_size_was_unspecified = not getattr(
            cache_config, "user_specified_block_size", cache_config.block_size
            is None)

        from vllm_torchtpu.layers.vllm.attention import (
            PallasAttentionBackend, PallasBatchedRPAAttentionBackend)
        attn_backend = getattr(getattr(vllm_config, "attention_config", None),
                               "backend", None)
        selected_name = getattr(attn_backend, "name", None)
        backend_cls = (PallasBatchedRPAAttentionBackend if selected_name
                       == "CUSTOM" else PallasAttentionBackend)
        is_hybrid = getattr(vllm_config.model_config, "is_hybrid", False)
        cls._is_hybrid = is_hybrid
        cls._speculative_enabled = vllm_config.speculative_config is not None
        if cls._speculative_enabled and \
                vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError(
                "Speculative decoding with async scheduling is not yet "
                "supported on TPU; run with async_scheduling=False.")
        if not is_hybrid and block_size_was_unspecified:
            default = backend_cls.get_page_size(vllm_config)
            cache_config.block_size = (  # type: ignore[assignment]
                backend_cls.get_preferred_block_size(default))
        if envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL:
            update_tpu_block_size_and_slot_config(vllm_config, backend_cls)

        min_page_size = backend_cls.get_min_page_size(vllm_config)
        if min_page_size > cache_config.block_size:
            logger.warning(
                "Increase the page size from %s to %s to make sure there's"
                "no SMEM OOM",
                cache_config.block_size,
                min_page_size,
            )
            cache_config.block_size = min_page_size  # type: ignore[assignment]
        logger.info("Using KV cache block size: %s", cache_config.block_size)

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        parallel_config.worker_cls = \
                        "vllm_torchtpu.worker.tpu_worker.TPUWorker"

        multihost_backend = envs.TPU_MULTIHOST_BACKEND
        if not multihost_backend:  # Single host
            dp_size = parallel_config.data_parallel_size
            if dp_size > 1:
                if pcp_size > 1:
                    ensure_pcp_local_rank_remap(
                        parallel_config.world_size_across_dp,
                        get_topology=cls._get_tpu_topology)
                # Single-host DP uses one torch_tpu slice across all DP*TP
                # workers; the worker spawn shim exposes a DP-adjusted chip
                # ordinal to TorchTPU for physical binding.
                cls.device_control_env_var = \
                    "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"
                # vLLM may pass per-engine ParallelConfig objects to workers
                # with data_parallel_size collapsed to 1. Preserve the original
                # single-host DP size for TorchTPU rank/env setup.
                # TODO: Remove TORCH_TPU_DP_SIZE once TorchTPU physical
                # chip/PjRt binding no longer depends on torchrun-style
                # RANK/LOCAL_RANK/WORLD_SIZE before vLLM distributed init.
                os.environ["TORCH_TPU_DP_SIZE"] = str(dp_size)
                os.environ.setdefault(
                    "TORCH_TPU_DP_MASTER_ADDR",
                    parallel_config.data_parallel_master_ip or "localhost")
                os.environ.setdefault("TORCH_TPU_DP_MASTER_PORT",
                                      str(portpicker.pick_unused_port()))
                # Full-slice slicebuilder list + topology (TP*PP*DP), once.
                cls._prepare_singlehost_tpu_env(
                    parallel_config.world_size_across_dp)
            else:
                os.environ.pop("TORCH_TPU_DP_SIZE", None)
                torch_tpu_world_size = parallel_config.world_size
                if pcp_size > 1:
                    logger.info(
                        "Preparing TorchTPU bootstrap env for native PCP "
                        "multiprocess world_size=%d.", torch_tpu_world_size)
                    ensure_pcp_local_rank_remap(
                        torch_tpu_world_size,
                        get_topology=cls._get_tpu_topology)
                cls._prepare_singlehost_tpu_env(torch_tpu_world_size)
            if (pcp_size <= 1 and parallel_config.data_parallel_size == 1
                    and parallel_config.pipeline_parallel_size == 1
                    and parallel_config.tensor_parallel_size == 1):
                logger.info("Force using UniProcExecutor for TPU on \
                        single host without tensor/pipeline parallelism.")
                parallel_config.distributed_executor_backend = "uni"
            else:
                logger.info(
                    "Force using TpuMultiprocExecutor for TPU on single host "
                    "with tensor/pipeline/PCP parallelism.")
                from vllm_torchtpu.executors.tpu_multiproc_executor import \
                    TpuMultiprocExecutor
                parallel_config.distributed_executor_backend = TpuMultiprocExecutor
        elif multihost_backend == "ray":
            # TODO(ranlihao): Use the vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND to determine which executor to use.
            # use_ray_v2 = vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND
            use_ray_v2 = False
            if use_ray_v2:
                from vllm_torchtpu.executors.ray_distributed_executor_v2 import \
                    RayDistributedExecutorV2
                parallel_config.distributed_executor_backend = RayDistributedExecutorV2
                logger.info(
                    "Force using RayDistributedExecutorV2 for TPU on multihost."
                )
            else:
                from vllm_torchtpu.executors.ray_distributed_executor import \
                    RayDistributedExecutor
                parallel_config.distributed_executor_backend = RayDistributedExecutor
                logger.info(
                    "Force using RayDistributedExecutor for TPU on multihost.")
        else:
            logger.warning(
                f"Unknown TPU multihost backend: {multihost_backend}. "
                "Using uniproc_executor.")
            parallel_config.distributed_executor_backend = "uni"

        if (scheduler_config.is_multimodal_model
                and not _is_language_model_only_config(model_config)
                and not scheduler_config.disable_chunked_mm_input):
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            _TPU_SUPPORTED_KV_CONNECTORS = {
                "TPUConnector",
                "TPUConnectorV2",
                "TPURaidenConnector",
                "TPUConnectorHMA",
                "OffloadingConnector",
            }
            assert kv_transfer_config.kv_connector in \
                _TPU_SUPPORTED_KV_CONNECTORS, (
                f"TPU only supports the following KV connectors: "
                f"{_TPU_SUPPORTED_KV_CONNECTORS}, but got "
                f"'{kv_transfer_config.kv_connector}'."
            )
            if (kv_transfer_config.kv_connector == "TPUConnectorV2"
                    and not envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL):
                raise ValueError("TPUConnectorV2 requires "
                                 "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1")
            if kv_transfer_config.kv_connector == "TPUConnectorV2":
                _patch_scheduler_mamba_external_kv()

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        # TODO: TPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        pass

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_torchtpu.distributed.tpu_communicator.TpuDeviceCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on TPU is experimental
        return True

    @classmethod
    def validate_request(
        cls,
        processed_inputs: ProcessorInputs,
        params: Union["SamplingParams", PoolingParams],
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        from vllm.sampling_params import SamplingParams, SamplingType

        del processed_inputs
        if isinstance(params, SamplingParams):
            if params.sampling_type == SamplingType.RANDOM_SEED:
                raise ValueError("JAX does not support per-request seed.")
            if params.sampling_type not in (SamplingType.GREEDY,
                                            SamplingType.RANDOM):
                raise ValueError(
                    f"Sampling type {params.sampling_type} is not supported on TPU."
                )
            if getattr(cls, "_speculative_enabled",
                       False) and params.sampling_type != SamplingType.GREEDY:
                raise NotImplementedError(
                    "Speculative decoding currently only supports greedy "
                    "sampling (temperature=0) on TPU.")
            if params.top_k != 0 or params.top_p != 1.0:
                logger.warning(
                    "Top-K and Top-P are not yet supported on TPU and will be ignored."
                )

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    _model_config: ModelConfig) -> bool:
        supported = {"auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"}
        return kv_cache_dtype in supported

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        """
        Returns if the current platform needs to sync weight loader.
        """
        # TODO: Fix this
        return False

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return getattr(cls, "_is_hybrid", False)


def _get_token_paddings(min_token_size: int, max_token_size: int,
                        padding_gap: int) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size

    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice,
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        logger.info("Using exponential token paddings:")
        while True:
            logger.info("    %d", num)
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        logger.info("Using incremental token paddings:")
        while num <= padding_gap:
            logger.info("    %d", num)
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            logger.info("    %d", num)
            paddings.append(num)

    return paddings
