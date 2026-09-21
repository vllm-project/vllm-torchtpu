# SPDX-License-Identifier: Apache-2.0

# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import vllm_torchtpu.env_override  # noqa: F401  # isort: skip

import os
import time
from typing import TYPE_CHECKING, Union

import portpicker
import torch
import vllm.envs as vllm_envs
# Ensure the "tpu" torch.compile backend is registered before vllm
# tries to use it (e.g. in @torch.compile decorators at import time).
from torch_tpu._internal import compile as _register_tpu_backend  # noqa: F401
from torch_tpu._internal.utils import hardware
from vllm.platforms.interface import Platform, PlatformEnum

from vllm_torchtpu import envs, patch_registry
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.platforms.pp_validation import \
    validate_pipeline_parallel_config
from vllm_torchtpu.platforms.tpu_block_size_utils import (
    unified_kv_layout_enabled, update_tpu_block_size_and_slot_config)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.multimodal.processing import ProcessorInputs
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
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

# Every map below is keyed on the number of devices in the slice, which is
# what world_size counts and what the slice builder sizes its worker address
# list against. How that key relates to the topology string differs by
# generation, so the maps cannot be merged:
#
#   - 2D Torus (v5e, v6e): one core per chip, T=1, so key == X*Y.
#   - 3D Torus (v4, v5p): two cores per chip, but megacore presents the pair
#     as a single device, so key == X*Y*Z and the trailing T=2 is not counted.
#   - 3D Torus (v7x / Ironwood): two cores per chip exposed as two separate
#     devices, so key == X*Y*Z*T, twice the chip count.
#   - Hierarchical BoardFly (TPU v8i): 4 chips per board,
#     up to 8 boards per group, up to 32 groups per slice, so key == C*B*G.
#
# Multi-host TPU slice mesh topologies for 2D Torus architectures.
TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP = {
    4: "2,2,1",
    8: "2,4,1",
    16: "4,4,1",
    32: "4,8,1",
    64: "8,8,1",
    128: "8,16,1",
    256: "16,16,1",
}

# Multi-host TPU slice mesh topologies for TPU v4 / v5p (4 devices per host).
TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP = {
    8: "2,2,2,2",
    16: "2,2,4,2",
    32: "2,4,4,2",
    64: "4,4,4,2",
    128: "4,4,8,2",
    256: "4,8,8,2",
}

# Multi-host TPU slice mesh topologies for TPU v7x / Ironwood, which exposes
# both cores of a chip as separate devices (8 devices per host). Same meshes
# as above, addressed by twice as many devices.
TPU_3D_TORUS_DUAL_DEVICE_TOPOLOGY_MAP = {
    16: "2,2,2,2",
    32: "2,2,4,2",
    64: "2,4,4,2",
    128: "4,4,4,2",
    256: "4,4,8,2",
    512: "4,8,8,2",
}

# Multi-host TPU slice mesh topologies for TPU v8i, which uses a hierarchical
# BoardFly topology: (Chips per Board) x (Boards per Group) x (Groups per Slice),
# noted as C x B x G (4 chips per board, up to 8 boards per group, up to 32 groups per slice).
TPU_8I_MULTIHOST_TOPOLOGY_MAP = {
    4: "4,1,1",
    8: "4,2,1",
    12: "4,3,1",
    16: "4,4,1",
    20: "4,5,1",
    24: "4,6,1",
    28: "4,7,1",
    32: "4,8,1",
    64: "4,8,2",
    128: "4,8,4",
    256: "4,8,8",
    512: "4,8,16",
    1024: "4,8,32",
}

# TPU generations with a 2D Torus interconnect (3-tuple mesh geometry: X,Y,T).
# TPU v4, v5p, and v7 (Ironwood) use 3D Torus interconnects (4-tuple mesh geometry: X,Y,Z,T).
_TPU_2D_TORUS_GENERATIONS = ("v5e", "v6e")

# 3D Torus generations that expose two devices per chip. The device name
# surfaces as "TPU v7" from get_tpu_device_name() and as "TPU7x" from the
# device-kind path, so both spellings have to match.
_TPU_DUAL_DEVICE_GENERATIONS = ("v7", "tpu7x")

# TPU v8i / TPU8i generations.
_TPU_8I_GENERATIONS = ("v8i", "tpu8i")


def get_tpu_multihost_topology(
    world_size: int,
    device_name: str | None = None,
) -> str:
    """Return the multi-host mesh topology string for a TPU cluster.

    `world_size` is the number of devices in the slice (one per worker), which
    is what the fallback maps are keyed on. The map is selected by generation
    because devices-per-chip varies; see the maps above.

    Resolution order:
    1. `TORCH_TPU_TOPOLOGY` environment variable (set by GKE).
    2. Fallback map for environments without TORCH_TPU_TOPOLOGY set.
    """
    env_val = os.environ.get("TORCH_TPU_TOPOLOGY")
    if env_val:
        # Automatically normalize standard 'AxBxC' syntax to PyTorch/XLA 'A,B,C'
        return env_val.replace("x", ",")

    if device_name is None:
        try:
            device_name = TpuPlatform.get_device_name()
        except Exception:
            device_name = ""

    device_name = device_name.lower()
    if any(gen in device_name for gen in _TPU_2D_TORUS_GENERATIONS):
        topo_map = TPU_2D_TORUS_MULTIHOST_TOPOLOGY_MAP
    elif any(gen in device_name for gen in _TPU_8I_GENERATIONS):
        topo_map = TPU_8I_MULTIHOST_TOPOLOGY_MAP
    elif any(gen in device_name for gen in _TPU_DUAL_DEVICE_GENERATIONS):
        topo_map = TPU_3D_TORUS_DUAL_DEVICE_TOPOLOGY_MAP
    else:
        topo_map = TPU_3D_TORUS_MULTIHOST_TOPOLOGY_MAP

    topo = topo_map.get(world_size)
    if topo is None:
        raise ValueError(
            f"Cannot find topology for {world_size} devices in {topo_map}. "
            "Please export TORCH_TPU_TOPOLOGY in your environment.")
    return topo


# ---------------------------------------------------------------------------
# Modules/attrs whose @torch.compile(dynamic=True) wrappers must be removed.
# Each entry is (module_path, function_name).
# ---------------------------------------------------------------------------
_DYNAMIC_COMPILE_TARGETS: list[tuple[str, str]] = [
    ("vllm.model_executor.layers.vocab_parallel_embedding",
     "get_masked_input_and_mask"),
    ("vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
     "grouped_topk"),
    ("vllm.v1.sample.ops.logprobs", "batched_count_greater_than"),
    ("vllm.v1.sample.ops.topk_topp_sampler", "compiled_random_sample"),
    ("vllm.utils.deep_gemm", "per_block_cast_to_fp8"),
]

_dynamic_compile_unwrapped = False
_tpu_kv_connectors_registered = False


def _configure_torchtpu_eager_mode() -> None:
    from torch_tpu._internal import execution_mode
    previous_mode = execution_mode.eager_mode
    # Kernel-iteration mode keeps DEFER_NEVER: DEFER_AND_FUSE fuses the eager
    # region around the split-out Pallas ops into per-context device programs,
    # each of which recompiles after a kernel hot-swap. With DEFER_NEVER the
    # kernel program is context-free, so a swap costs one kernel compile.
    if envs.TPU_KERNEL_ITER_MODE:
        logger.info("TorchTPU eager mode left at %s (TPU_KERNEL_ITER_MODE).",
                    previous_mode.name)
        return
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


def _patch_api_server_kernel_reload_endpoint() -> None:
    """Add ``POST /reload_kernel`` to the OpenAI API server.

    Kernel-iteration mode only. Wraps ``api_server.build_app`` so the route
    is present on the served app; the request body may carry an optional
    ``{"modules": [...]}`` override of the reloadable module list. Only takes
    effect in the API-server process (the module is already imported there
    when the platform activates); worker processes never import it.
    """
    if not envs.TPU_KERNEL_ITER_MODE:
        return
    import sys
    api_server = sys.modules.get("vllm.entrypoints.openai.api_server")
    if api_server is None or getattr(api_server, "_tpu_kernel_reload_patch",
                                     False):
        return

    orig_build_app = api_server.build_app

    def build_app_with_reload(*args, **kwargs):
        app = orig_build_app(*args, **kwargs)
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.post("/reload_kernel")
        async def reload_kernel(raw_request: Request):
            modules = None
            body = await raw_request.body()
            if body:
                import json
                modules = json.loads(body).get("modules")
            client = raw_request.app.state.engine_client
            # The next request after the swap pays the kernel compile; make
            # sure no batch is in flight while the workers swap.
            if hasattr(client, "wait_for_requests_to_drain"):
                await client.wait_for_requests_to_drain()
            results = await client.collective_rpc(
                "reload_kernels",
                timeout=1800,
                kwargs={"modules": modules},
            )
            return JSONResponse({"results": results})

        return app

    api_server.build_app = build_app_with_reload
    api_server._tpu_kernel_reload_patch = True
    logger.info("Applied TPU patch: /reload_kernel dev endpoint "
                "(TPU_KERNEL_ITER_MODE).")


def _validate_phased_profiling_config(vllm_config: "VllmConfig") -> None:
    """Fail fast on phased-profiling settings that would silently do nothing.

    Both mistakes below produce a run that looks healthy and writes no
    traces, which is only discovered once the benchmark has finished.
    """
    assert "phased_profiling_dir" not in vllm_config.additional_config, (
        "Legacy additional_config['phased_profiling_dir'] is no longer "
        "supported. Set USE_PHASED_PROFILER=true and "
        "--profiler-config.torch_profiler_dir=<dir> instead.")
    if envs.USE_PHASED_PROFILER:
        # torch_profiler_dir truthy already implies profiler == "torch";
        # ProfilerConfig's own validator rejects the dir without it.
        assert vllm_config.profiler_config.torch_profiler_dir, (
            "USE_PHASED_PROFILER is set but there is nowhere to write traces. "
            "Add --profiler-config.profiler=torch with "
            "--profiler-config.torch_profiler_dir=<dir>.")


def _register_tpu_kv_connectors() -> None:
    """Register TPU-specific KV connectors in vLLM's factory name registry.

    Ensures connectors loaded dynamically via module paths (e.g., TPURaidenConnector,
    TPUMultiConnector, TPURaidenOffloadingConnector) are discoverable by name.

    Required for the API server metrics pipeline: TPUMultiConnector aggregates child
    connector statistics by looking up classes via get_connector_class_by_name().
    Without registry registration, the first metrics collection triggers an unhandled
    exception in the AsyncLLM output handler, causing HTTP 500 errors on subsequent requests.

    Invoked from check_and_update_config() in every process that builds a VllmConfig
    (including the API server). Defers imports lazily until connector names are accessed.
    """
    global _tpu_kv_connectors_registered
    if _tpu_kv_connectors_registered:
        return
    _tpu_kv_connectors_registered = True

    from vllm.distributed.kv_transfer.kv_connector.factory import \
        KVConnectorFactory

    for name, module_path in (
        ("TPURaidenConnector",
         "vllm_torchtpu.distributed.kv_transfer.tpu_connector"),
        ("TPUMultiConnector",
         "vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector"),
        ("TPURaidenOffloadingConnector",
         "vllm_torchtpu.offload.raiden_connector"),
    ):
        KVConnectorFactory.register_connector(name, module_path, name)


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
    _is_hybrid: bool = False

    supported_quantization: list[str] = [
        "tpu_int8", "compressed-tensors", "awq", "fp8", "mxfp4",
        "modelopt_fp4", "deepseek_v4_fp8"
    ]

    additional_env_vars: list[str] = [
        "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS",
        "TPU_MULTIHOST_BACKEND", "VLLM_MLA_DISABLE", "TPU_BACKEND_TYPE",
        "MIXED_Q_SPLIT", "MIXED_NUM_QUERIES_PER_BLOCK",
        "MIXED_NUM_KV_PAGES_PER_BLOCK", "ENABLE_QUANTIZED_MATMUL_KERNEL",
        "REQUANTIZE_BLOCK_SIZE", "REQUANTIZE_WEIGHT_DTYPE",
        "MOE_REQUANTIZE_BLOCK_SIZE", "MOE_REQUANTIZE_WEIGHT_DTYPE",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
        "TORCHINDUCTOR_AUTOGRAD_CACHE", "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY", "TPU_KERNEL_ITER_MODE",
        "TPU_KERNEL_RELOAD_MODULES", "TPU_MOE_ROUTER_TOPK",
        "MOE_FUSED_EP_V2_SHARDED_PLAN", "MOE_FUSED_EP_ENABLE_W4A8"
    ]

    # The "Platform" base class has import_kernels() that tries to import
    # vllm._C, which is not available in the TPU Platform setup.
    # Override it to do nothing so we won't cause confusing warning logs.
    @classmethod
    def import_kernels(cls) -> None:
        return

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int) -> int:
        """Map logical DP/TP device id to a local physical TPU device.

        Under multi-core TPU (e.g. v7x with 2 cores per chip), the logical
        rank/device_id may exceed the physical chip count in TPU_VISIBLE_CHIPS.
        Wrap with modulo so every rank resolves to a valid local physical chip.
        """
        device_control_env = os.environ.get(cls.device_control_env_var, "")
        if device_control_env:
            device_ids = [
                s.strip() for s in device_control_env.split(",") if s.strip()
            ]
            if device_ids:
                return int(device_ids[device_id % len(device_ids)])
        return int(device_id % max(1, cls.device_count()))

    @classmethod
    def device_count(cls) -> int:
        """Local physical chip count on this host, stable across contexts.

        Without this override, `device_count()` resolves dynamically to
        `torch.tpu.device_count()` (see torch_tpu's device module), which
        returns the *local* chip count outside any torch.distributed group
        but drops to 1 (chips bound to this rank) once a real multi-host
        `tpu_dist` process group is active — vLLM workers are one-process-
        per-chip, so every rank observes exactly 1 there. Callers like
        MessageQueue.create_from_process_group_single_reader use this value
        as "how many ranks share this host" (the CUDA convention, where
        device_count() stays host-wide regardless of the process's own
        device binding), so the dynamic proxy silently breaks cross-host
        same-node detection for every rank but the reader itself. Scanning
        /dev/vfio (or /dev/accel*) directly sidesteps the distributed
        context entirely and always returns the true local chip count.
        """
        from vllm_torchtpu.tpu_info import get_num_chips
        return get_num_chips()

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

        For world_size > 1, topology is looked up via PCI scan using
        world_size (not auto-detected chip count) so slicebuilder and topology
        match the actual number of workers. A single TPU does not need the
        distributed PjRt bootstrap.
        """
        os.environ.setdefault("TORCH_TPU_XPROF_SESSION_ID",
                              str(time.time_ns()))

        if world_size == 1:
            os.environ.pop("WORLD_SIZE", None)
            return

        # PjRt's distributed bootstrap requires a native endpoint even when
        # vLLM uses a c10d FileStore. Select it once before spawning workers,
        # alongside the shared slice topology, so every rank inherits it.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(portpicker.pick_unused_port())

        sb_addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
        sb_count = len(sb_addresses.split(",")) if sb_addresses else 0
        if sb_count != world_size:
            sb_ports = [
                portpicker.pick_unused_port() for _ in range(world_size)
            ]
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
                f"localhost:{p}" for p in sb_ports)

        # A per-engine DP config copy (data_parallel_size collapsed to 1,
        # see the caller) re-enters this function in a child process that
        # inherits TORCH_TPU_TOPOLOGY already computed for the *real* multi-
        # host slice by prepare_mp_multihost_dp_env. Only the single-host
        # PCI-scan lookup below needs a fresh call, and re-running it here
        # would fail: this process only has world_size's local share of
        # chips, not the whole cross-host slice.
        if "TORCH_TPU_TOPOLOGY" not in os.environ:
            os.environ["TORCH_TPU_TOPOLOGY"] = cls._get_tpu_topology(
                world_size)

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
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if attn_selector_config.use_mla:
            selected_backend = AttentionBackendEnum.FLASH_ATTN_MLA

        supported_backends = [
            AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.CUSTOM,
            AttentionBackendEnum.FLASH_ATTN_MLA
        ]
        if selected_backend not in supported_backends:
            logger.info("Cannot use %s backend on TPU. Setting to FLASH_ATTN.",
                        selected_backend)
            selected_backend = AttentionBackendEnum.FLASH_ATTN
        logger.info("Using %s backend.", selected_backend.name)
        return selected_backend.get_path()

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
    def get_current_memory_usage(cls,
                                 device: torch.device | None = None) -> float:
        if not torch.tpu.is_initialized():
            return 0.0
        free, total = torch.accelerator.get_memory_info(device)
        return float(total - free)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm_torchtpu.lora.torch_punica_tpu.PunicaWrapperTPU"

    @classmethod
    def get_infinity_values(cls, dtype) -> tuple[float, float]:
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
            t = dp_metadata.num_tokens_across_dp_cpu
            if t.device.type == "cpu":
                tpu_t = t.to("tpu")
                try:
                    dp_metadata.num_tokens_across_dp_cpu = tpu_t
                except Exception:
                    # Frozen dataclass fallback.
                    object.__setattr__(dp_metadata, "num_tokens_across_dp_cpu",
                                       tpu_t)
        return {}

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        assert "sharding" not in vllm_config.additional_config, (
            "Legacy additional_config['sharding'] is no longer supported. "
            "Use --data-parallel-size and --enable-expert-parallel instead.")
        _validate_phased_profiling_config(vllm_config)
        if "USE_MOE_EP_KERNEL" in os.environ:
            raise ValueError(
                "USE_MOE_EP_KERNEL is no longer supported and enables "
                "nothing. Unset it; the fused expert-parallel MoE kernel is "
                "USE_MOE_FUSED_EP_KERNEL=1.")
        patch_registry.apply("platform_activation",
                             model_config=vllm_config.model_config)
        _register_tpu_kv_connectors()

        if vllm_envs.VLLM_TPU_USING_PATHWAYS:
            raise NotImplementedError(
                "Pathways is not supported by vllm-torchtpu. "
                "Unset VLLM_TPU_USING_PATHWAYS.")
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        # TPU's page geometry and block-size selection depend on layout while
        # loading the model, before engine-core's post-load resolution. Resolve
        # explicitly here through vLLM's API and publish the same CacheConfig
        # value to workers, rather than maintaining a process-global getter.
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        from vllm.v1.attention.backends.utils import (
            get_flashinfer_layout_string, resolve_kv_cache_layout)

        from vllm_torchtpu.layers.adapter.attention import (
            PallasAttentionBackend, PallasBatchedRPAAttentionBackend)
        backend = (PallasBatchedRPAAttentionBackend
                   if vllm_config.attention_config.backend
                   == AttentionBackendEnum.CUSTOM else PallasAttentionBackend)
        supported = [
            layout.name for layout in backend.supported_kv_cache_layouts()
        ]
        if (vllm_config.model_config is not None and "DeepseekV4ForCausalLM"
                in vllm_config.model_config.architectures):
            supported = ["BLHNC"]
        layout = resolve_kv_cache_layout(vllm_config, [supported])
        kv_cache_layout = get_flashinfer_layout_string(layout)
        pcp_config = PcpStaticSupportValidator.validate_platform_config(
            vllm_config,
            kv_cache_layout=kv_cache_layout,
        )
        pcp_size = pcp_config.pcp_size
        if pcp_config.enabled:
            logger.info("Using vLLM native multiprocess PCP world; PCP is not "
                        "represented as a JAX mesh axis.")
        validate_pipeline_parallel_config(vllm_config)
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            # Engines built in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
            # get the hand-off push here; engine-core processes install it
            # again on their side.
            from vllm_torchtpu.distributed.pp_push import \
                patch_executor_for_pp_wave
            patch_executor_for_pp_wave(vllm_config)
            _configure_pipeline_chunks(vllm_config)

        from vllm.config import CompilationMode
        compilation_config = vllm_config.compilation_config
        if compilation_config.mode == CompilationMode.NONE:
            # --enforce-eager is set
            pass
        else:
            compilation_config.mode = CompilationMode.VLLM_COMPILE

        # No graph splitting — TPU handles the full graph including
        compilation_config.splitting_ops = []

        # Kernel-iteration mode splits the compiled graph at the Pallas RPA
        # custom ops so they execute eagerly outside the compiled pieces.
        # Together with the cache-key scoping in tpu_compiler.py this lets a
        # kernel-source edit reuse the cached backbone executables, and lets
        # /reload_kernel swap the kernel in a running server. Op instance ids
        # are allocated at model build, so list every id the registry could
        # plausibly allocate (should_split does exact name matching).
        if envs.TPU_KERNEL_ITER_MODE:
            compilation_config.splitting_ops = [
                f"pallas::{prefix}_{i}"
                for prefix in ("rpa_kernel", "rpa_kernel_batched")
                for i in range(64)
            ]

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
                extra_bucket_sizes=envs.TPU_TOKEN_BUCKET_EXTRA,
            )
        else:
            compilation_config.compile_sizes = sorted(
                compilation_config.compile_sizes)
        # Clear compile_ranges_split_points — TPU always pads to exact
        # compile_sizes so catch-all ranges are never used.
        compilation_config.compile_ranges_split_points = []

        # Backed dynamic shapes 0/1-specialize a size-1 compile into a static
        # retrace that trips PiecewiseBackend's single-entry assert;
        # size-oblivious backed shapes keep the graph dynamic at size 1.
        from vllm.config.compilation import DynamicShapesType
        dynamic_shapes_config = compilation_config.dynamic_shapes_config
        if (1 in compilation_config.compile_sizes
                and dynamic_shapes_config.type == DynamicShapesType.BACKED):
            dynamic_shapes_config.type = (
                DynamicShapesType.BACKED_SIZE_OBLIVIOUS)
            logger.info(
                "compile_sizes contains 1, which backed dynamic shapes "
                "cannot compile (0/1 specialization); switching to "
                "backed_size_oblivious dynamic shapes.")

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

        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config

        is_hybrid = model_config.is_hybrid if model_config is not None else False
        if vllm_config.speculative_config is not None and scheduler_config.async_scheduling:
            method = vllm_config.speculative_config.method
            if not vllm_config.speculative_config.use_eagle():
                # Ngram needs the sampled tokens on the host, which async defers.
                raise NotImplementedError(
                    f"Async scheduling with speculative method '{method}' is "
                    "not supported on TPU; Run with async_scheduling=False.")
        # Real hybrid prefix hits need pooled Mamba state seed copies.
        # DecodeBench installs synthetic state for each request instead.
        transfer_config = vllm_config.kv_transfer_config
        is_decode_bench = (transfer_config is not None
                           and transfer_config.kv_connector
                           == "DecodeBenchConnector")
        if (is_hybrid and cache_config.enable_prefix_caching
                and not is_decode_bench):
            if cache_config.mamba_cache_mode != "align":
                raise NotImplementedError(
                    "Prefix caching on hybrid Mamba models requires "
                    "mamba_cache_mode='align' (got "
                    f"{cache_config.mamba_cache_mode!r}).")
            if not unified_kv_layout_enabled(vllm_config):
                # Seed copies live on the pooled path only, so the per-layer
                # layout restores no Mamba state on a prefix-cache hit and
                # would silently generate from an unrelated slot.
                raise NotImplementedError(
                    "Prefix caching on hybrid Mamba models requires the "
                    "unified KV pool; remove "
                    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0.")

        parallel_config = vllm_config.parallel_config
        parallel_config.worker_cls = \
                        "vllm_torchtpu.worker.tpu_worker.TPUWorker"

        multihost_backend = envs.TPU_MULTIHOST_BACKEND
        if multihost_backend != "ray" and parallel_config.nnodes > 1:
            # Genuine multi-host run without Ray: each host runs its own
            # `vllm serve --nnodes N --node-rank R [--headless]` process and
            # spawns its local TP-slice workers via TpuMultiprocExecutor.
            # Unlike the Ray path there's no actor system to discover peer
            # hosts, so a short TCPStore rendezvous fills in the TORCH_TPU_*
            # env vars (slicebuilder addresses spanning every host, full
            # ICI topology, TPU_NUM_HOSTS/NODE_RANK) before workers are
            # forked. This must be checked before the "not multihost_backend"
            # single-host branch below: an unset TPU_MULTIHOST_BACKEND still
            # means "mp" whenever --nnodes > 1 was actually passed.
            from vllm_torchtpu.distributed.tpu_mp_multihost import \
                prepare_mp_multihost_env
            prepare_mp_multihost_env(parallel_config)
            logger.info(
                "Force using TpuMultiprocExecutor for TPU multi-host "
                "(nnodes=%d, node_rank=%d).", parallel_config.nnodes,
                parallel_config.node_rank)
            from vllm_torchtpu.executors.tpu_multiproc_executor import \
                TpuMultiprocExecutor
            parallel_config.distributed_executor_backend = TpuMultiprocExecutor
        elif multihost_backend == "mp" or not multihost_backend:
            # Single host, or genuine multi-host DP without --nnodes (this
            # host owns only a slice of data_parallel_size via
            # --data-parallel-size-local / --data-parallel-start-rank).
            dp_size = parallel_config.data_parallel_size
            if dp_size > 1:
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
                if parallel_config.enable_expert_parallel:
                    # EP's expert-combine collective genuinely spans every
                    # DP*TP worker, so torch_tpu needs one shared slice
                    # across all of them; the worker spawn shim exposes a
                    # DP-adjusted chip ordinal to TorchTPU for physical
                    # binding.
                    cls.device_control_env_var = \
                        "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"
                # Genuine multi-host DP (this host only owns a slice of the
                # DP ranks) needs a TCPStore rendezvous across hosts instead
                # of the single-host localhost/PCI-scan bootstrap below.
                from vllm_torchtpu.distributed.tpu_mp_multihost import \
                    prepare_mp_multihost_dp_env
                if not prepare_mp_multihost_dp_env(
                        parallel_config, parallel_config.world_size_across_dp):
                    cls._prepare_singlehost_tpu_env(
                        parallel_config.world_size_across_dp)
            else:
                # vLLM hands each DP engine a ParallelConfig with
                # data_parallel_size collapsed to 1, so the inherited
                # TORCH_TPU_DP_SIZE is the only record of how wide the slice
                # really is. Keep sizing the bootstrap by the whole slice.
                dp_slice_size = int(
                    os.environ.pop("TORCH_TPU_DP_SIZE", "1") or 1)
                torch_tpu_world_size = (parallel_config.world_size *
                                        dp_slice_size)
                if pcp_size > 1:
                    logger.info(
                        "Preparing TorchTPU bootstrap env for native PCP "
                        "multiprocess world_size=%d.", torch_tpu_world_size)
                cls._prepare_singlehost_tpu_env(torch_tpu_world_size)
            if (pcp_size <= 1 and parallel_config.data_parallel_size == 1
                    and parallel_config.pipeline_parallel_size == 1
                    and parallel_config.tensor_parallel_size == 1):
                logger.info("Force using UniProcExecutor for TPU on "
                            "single host without tensor/pipeline parallelism.")
                parallel_config.distributed_executor_backend = "uni"
            else:
                logger.info(
                    "Force using TpuMultiprocExecutor for TPU on single host "
                    "with tensor/pipeline/PCP parallelism.")
                from vllm_torchtpu.executors.tpu_multiproc_executor import \
                    TpuMultiprocExecutor
                parallel_config.distributed_executor_backend = TpuMultiprocExecutor
        elif multihost_backend == "ray":
            if parallel_config.data_parallel_size > 1:
                if pcp_size > 1:
                    raise NotImplementedError(
                        "Prefill context parallelism is not supported "
                        "together with multihost data parallelism.")
                if not vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                    raise NotImplementedError(
                        "Multihost data parallelism requires the Ray V2 "
                        "executor. Set VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1.")
            if vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
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

        if envs.DP_SCHED_ENABLED and parallel_config.data_parallel_size > 1:
            dp_sched_cls = "vllm_torchtpu.core.tpu_scheduler.TpuDpScheduler"
            if scheduler_config.scheduler_cls != dp_sched_cls:
                assert scheduler_config.scheduler_cls is None, (
                    "Cannot have DP_SCHED_ENABLED enabled and also a custom "
                    "scheduler being provided.")
                scheduler_config.scheduler_cls = dp_sched_cls
                logger.info(
                    "Enabled TpuDpScheduler (DP_SCHED_ENABLED=1) for DP=%d.",
                    parallel_config.data_parallel_size)

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            _TPU_SUPPORTED_KV_CONNECTORS = {
                "DecodeBenchConnector",
                "TPUConnector",
                "TPURaidenConnector",
                "TPUMultiConnector",
                "TPURaidenOffloadingConnector",
            }
            assert kv_transfer_config.kv_connector in \
                _TPU_SUPPORTED_KV_CONNECTORS, (
                f"TPU only supports the following KV connectors: "
                f"{_TPU_SUPPORTED_KV_CONNECTORS}, but got "
                f"'{kv_transfer_config.kv_connector}'."
            )
            is_hybrid_offloading = (kv_transfer_config.kv_connector
                                    == "TPURaidenOffloadingConnector"
                                    and is_hybrid)
            if (is_hybrid_offloading
                    and not unified_kv_layout_enabled(vllm_config)):
                # Hybrid model offloading transfers uniform pool rows; the non-unified
                # typed-view layout lacks a common per-block stride for DMA copies.
                raise ValueError(
                    f"CPU offloading ({kv_transfer_config.kv_connector}) "
                    "on hybrid attention+Mamba models requires the unified "
                    "block pool; set TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1")

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        cache_config = vllm_config.cache_config
        backend_cls = cls._find_non_ssm_backend(vllm_config)

        if backend_cls is None:
            return

        # `get_supported_kernel_block_sizes` and
        # `get_kv_cache_shape` resolve the KV layout, which falls through
        # to the KV connector and so needs a current config. Upstream's
        # `update_block_size_for_backend` wraps the same calls.
        from vllm.config.vllm import set_current_vllm_config
        with set_current_vllm_config(vllm_config):
            architectures = getattr(
                getattr(vllm_config.model_config, "hf_config", None),
                "architectures", None) or []
            is_ds_v4 = any("DeepseekV4ForCausalLM" in a for a in architectures)

            if is_ds_v4 and not cache_config.user_specified_block_size:
                # DSv4 pages hold compressed rows. Its backend's
                # get_preferred_block_size is a hardcoded 256, which does not fit
                # the packed latent record, so take the MLA page size directly.
                from vllm_torchtpu.layers.adapter.attention import \
                    PallasMLAttentionBackend
                cache_config.block_size = (  # type: ignore[assignment]
                    PallasMLAttentionBackend.get_page_size(vllm_config))
            else:
                is_hybrid = vllm_config.model_config.is_hybrid
                if not is_hybrid and not cache_config.user_specified_block_size:
                    default = backend_cls.get_page_size(vllm_config)
                    cache_config.block_size = (  # type: ignore[assignment]
                        backend_cls.get_preferred_block_size(default))
            if unified_kv_layout_enabled(vllm_config):
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
            logger.info("Using KV cache block size: %s",
                        cache_config.block_size)

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        from vllm_torchtpu.distributed.tpu_communicator import \
            TpuDeviceCommunicator
        return f"{TpuDeviceCommunicator.__module__}.{TpuDeviceCommunicator.__qualname__}"

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
        return True


def _get_exponential_token_paddings(min_token_size: int,
                                    max_token_size: int) -> list[int]:
    """Sizes doubling from min_token_size, capped exactly at max_token_size.

    The last bucket is max_token_size itself rather than the next power of
    two above it: the scheduler never schedules more than max_token_size
    tokens in one step, so a bucket larger than that only wastes compile
    time and HBM (each bucket's compiled program stays resident) without
    ever being used. (e.g. max_token_size=5000 will produce buckets
    [..., 4096, 5000] rather than [..., 4096, 8192]).
    """
    paddings = []
    num = min_token_size
    while num < max_token_size:
        paddings.append(num)
        num *= 2
    paddings.append(max_token_size)
    return paddings


def _configure_pipeline_chunks(vllm_config: VllmConfig) -> None:
    """Install the time-budget scheduler for a pipeline and the engine-core
    hook that times the stages. Off when the env disables it, when the DP
    scheduler or another scheduler is configured, or when chunked prefill is
    disabled."""
    from vllm_torchtpu.core.pp_chunk_scheduler import (
        SCHEDULER_CLS, patch_engine_core_for_pp_chunks)
    scheduler_config = vllm_config.scheduler_config
    if not envs.TPU_PP_DYNAMIC_CHUNKS:
        return
    if (envs.DP_SCHED_ENABLED
            and vllm_config.parallel_config.data_parallel_size > 1):
        logger.warning(
            "Pipeline chunk sizing is off: the DP scheduler is enabled.")
        return
    if scheduler_config.scheduler_cls not in (None, SCHEDULER_CLS):
        logger.warning(
            "Pipeline chunk sizing is off: scheduler %s is configured.",
            scheduler_config.scheduler_cls)
        return
    if not scheduler_config.enable_chunked_prefill:
        logger.warning(
            "Pipeline chunk sizing is off: chunked prefill is disabled.")
        return
    if envs.TPU_PP_CHUNK_SLACK < 0:
        raise ValueError("TPU_PP_CHUNK_SLACK must not be negative: "
                         f"{envs.TPU_PP_CHUNK_SLACK}")
    scheduler_config.scheduler_cls = SCHEDULER_CLS
    patch_engine_core_for_pp_chunks(vllm_config)


def _get_token_paddings(
        min_token_size: int,
        max_token_size: int,
        extra_bucket_sizes: list[int] | None = None) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size.

    ``extra_bucket_sizes`` are merged into the list.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0

    paddings = _get_exponential_token_paddings(min_token_size, max_token_size)
    if extra_bucket_sizes:
        paddings = sorted(
            set(paddings)
            | {
                size
                for size in extra_bucket_sizes if 0 < size <= max_token_size
            })
    logger.info("Using token paddings: %s", paddings)
    return paddings
