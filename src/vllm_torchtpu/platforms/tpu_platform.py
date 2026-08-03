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
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.platforms.tpu_block_size_utils import (
    unified_kv_layout_enabled, update_tpu_block_size_and_slot_config)
from vllm_torchtpu.worker.tpu_rank_binding import ensure_pcp_local_rank_remap

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

# TODO(ranlihao): add more flexible topology map
TPU_MULTIHOST_TOPOLOGY_MAP = {
    16: "2,2,2,2",
    32: "2,2,4,2",
}

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
_tpu_patches_applied = False


def _is_language_model_only_config(model_config: "ModelConfig") -> bool:
    multimodal_config = model_config.multimodal_config
    if multimodal_config is None:
        return False
    return bool(multimodal_config.language_model_only)


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


def apply_tpu_patches() -> None:
    """Apply all module-level patches required for TorchTPU.

    Must run in **every** process (including spawned workers) because
    module-level state is re-initialized on import in spawned processes.
    All patches are idempotent.
    """
    global _tpu_patches_applied
    if _tpu_patches_applied:
        return
    _tpu_patches_applied = True

    import vllm_torchtpu as tpu_plugin
    from vllm_torchtpu import (_patch_default_moe_runner_select_forward,
                               _patch_disable_sequence_parallel_moe,
                               _patch_moe_no_ep_tp_scope,
                               _patch_rowparallel_defer_bias,
                               _patch_vllm_disable_compile_ranges,
                               _patch_vllm_hybrid_producer_prefix_hits,
                               _patch_vllm_tpu_group_custom_ops)
    from vllm_torchtpu.layers.vllm.custom_ops import _register_custom_ops

    from vllm_torchtpu import _patch_vllm_hybrid_pcp_block_sizes  # isort: skip
    from vllm_torchtpu import _patch_vllm_kimi_kda_layer_counts  # isort: skip
    from vllm_torchtpu import _patch_expert_map_host_lookup  # isort: skip
    _register_custom_ops()
    tpu_plugin._patch_vllm_aot_compile_cache_key()
    tpu_plugin._patch_vllm_config_hash_ignore_diagnostics()
    _patch_vllm_tpu_group_custom_ops()
    _patch_default_moe_runner_select_forward()
    _patch_vllm_disable_compile_ranges()
    _patch_disable_sequence_parallel_moe()
    _patch_moe_no_ep_tp_scope()
    _patch_rowparallel_defer_bias()
    _patch_expert_map_host_lookup()
    _patch_vllm_hybrid_pcp_block_sizes()
    _patch_vllm_kimi_kda_layer_counts()
    _patch_vllm_hybrid_producer_prefix_hits()
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
    _patch_api_server_kernel_reload_endpoint()


def _is_qwen3_vl_model(model_config: Optional["ModelConfig"]) -> bool:
    if model_config is None:
        return False
    hf_config = model_config.hf_config
    if hf_config is not None:
        model_type = getattr(hf_config, "model_type", "")
        if model_type and "qwen3_vl" in str(model_type).lower():
            return True
        architectures = getattr(hf_config, "architectures", [])
        if any("qwen3vl" in str(arch).lower() for arch in architectures):
            return True
    model_name = model_config.model
    if "qwen3-vl" in str(model_name).lower() or "qwen3vl" in str(
            model_name).lower():
        return True
    return False


def _apply_model_specific_patches(
        model_config: Optional["ModelConfig"] = None) -> None:
    if not _is_qwen3_vl_model(model_config):
        return
    """Apply model-specific and PyTorch XLA op-level patches for Qwen3-VL on TPU.

    This function applies the following patches:
    1. Qwen3VLModel.get_rope_index:
       - Grid Metadata Fix (`fix_grid`): Unsqueezes 1D `image_grid_thw`/`video_grid_thw`
         tensors (shape [3] -> [1, 3]). HF's `get_rope_index` assumes 2D grid tensors
         and attempts `grid[:, 0]`, which raises a fatal IndexError on 1D inputs.
       - CPU RoPE Evaluation (`to_cpu`): Evaluates 3D RoPE (Time, Height, Width) position
         ID calculation on host CPU and returns results back to TPU, preventing XLA
         lowering and device sync errors during 3D RoPE index construction.
    2. Qwen3VLVisionAttention.forward:
       - Moves `cu_seqlens` (cumulative visual patch sequence lengths) to CPU to avoid
         device mismatch in vision attention kernels.
    3. torch.masked_scatter / masked_scatter_:
       - Replaces native TPU `masked_scatter` with a strict 1D sequential indexing
         decomposition (`indices = torch.nonzero(flat_mask)[0]`; `res[indices] = flat_source`),
         eliminating unlowered XLA `scan` HLO nodes while preserving exact PyTorch
         sequential source consumption semantics.
    4. torch.repeat_interleave & torch.cumsum:
       - Evaluates `repeat_interleave` and `cumsum` on CPU strictly for integer and
         boolean TPU tensors (`int64`, `int32`, `int16`, `bool`) and returns tensors
         on the original TPU device (`res_cpu.to(input.device)`), preventing unlowered
         XLA `scan` HLO nodes without affecting standard floating-point TPU ops.
    """
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
        _orig_get_rope_index = modeling.Qwen3VLModel.get_rope_index

        def patched_get_rope_index(self, *args, **kwargs):

            def fix_grid(g):
                if g is not None and isinstance(g, torch.Tensor):
                    if g.numel() == 0:
                        return None
                    if g.dim() == 1:
                        return g.unsqueeze(0)
                return g

            target_device = None

            def to_cpu(t):
                nonlocal target_device
                if isinstance(t, torch.Tensor):
                    if target_device is None:
                        target_device = t.device
                    return t.cpu()
                return t

            new_args = [to_cpu(a) for a in args]
            new_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}

            if 'image_grid_thw' in new_kwargs:
                new_kwargs['image_grid_thw'] = fix_grid(
                    new_kwargs['image_grid_thw'])
            if 'video_grid_thw' in new_kwargs:
                new_kwargs['video_grid_thw'] = fix_grid(
                    new_kwargs['video_grid_thw'])
            if len(new_args) > 2:
                new_args[2] = fix_grid(new_args[2])
            if len(new_args) > 3:
                new_args[3] = fix_grid(new_args[3])

            if ('input_ids' not in new_kwargs or new_kwargs['input_ids'] is None) and \
                len(new_args) == 0:
                new_kwargs['input_ids'] = torch.zeros((1, 1),
                                                      dtype=torch.long,
                                                      device='cpu')

            res = _orig_get_rope_index(self, *new_args, **new_kwargs)
            if target_device is not None:

                def to_device(r):
                    if isinstance(r, torch.Tensor):
                        return r.to(target_device)
                    if isinstance(r, (tuple, list)):
                        return type(r)(to_device(x) for x in r)
                    return r

                res = to_device(res)
            return res

        modeling.Qwen3VLModel.get_rope_index = patched_get_rope_index
        logger.info("Applied TPU patch: Qwen3VLModel.get_rope_index.")

        _orig_vision_attn_forward = modeling.Qwen3VLVisionAttention.forward

        def patched_vision_attn_forward(self, hidden_states, cu_seqlens, *args,
                                        **kwargs):
            if isinstance(cu_seqlens, torch.Tensor):
                try:
                    cu_seqlens = cu_seqlens.cpu()
                except Exception:
                    cu_seqlens = torch.tensor([0, hidden_states.shape[0]],
                                              dtype=torch.int32)
            return _orig_vision_attn_forward(self, hidden_states, cu_seqlens,
                                             *args, **kwargs)

        modeling.Qwen3VLVisionAttention.forward = patched_vision_attn_forward
        logger.info("Applied TPU patch: Qwen3VLVisionAttention.forward.")

        orig_masked_scatter_ = torch.Tensor.masked_scatter_

        def patched_masked_scatter_(self, mask, source):
            if self.device.type == "tpu":
                mask_bool = mask.bool()
                flat_self = self.reshape(-1)
                flat_mask = mask_bool.reshape(-1)
                flat_source = source.reshape(-1)
                indices = torch.nonzero(flat_mask, as_tuple=True)[0]
                res = flat_self.clone()
                res[indices] = flat_source[:indices.numel()].to(
                    dtype=self.dtype)
                self.copy_(res.reshape(self.shape))
                return self
            return orig_masked_scatter_(self, mask, source)

        torch.Tensor.masked_scatter_ = patched_masked_scatter_

        orig_masked_scatter = torch.masked_scatter

        def patched_masked_scatter(input, mask, source):
            if isinstance(input, torch.Tensor) and input.device.type == "tpu":
                mask_bool = mask.bool()
                flat_input = input.reshape(-1)
                flat_mask = mask_bool.reshape(-1)
                flat_source = source.reshape(-1)
                indices = torch.nonzero(flat_mask, as_tuple=True)[0]
                res = flat_input.clone()
                res[indices] = flat_source[:indices.numel()].to(
                    dtype=input.dtype)
                return res.reshape(input.shape)
            return orig_masked_scatter(input, mask, source)

        torch.Tensor.masked_scatter = patched_masked_scatter
        torch.masked_scatter = patched_masked_scatter

        try:
            torch._C._TensorBase.masked_scatter_ = patched_masked_scatter_
            torch._C._TensorBase.masked_scatter = patched_masked_scatter
        except Exception:
            pass
        logger.info(
            "Applied TPU patch: torch.masked_scatter and masked_scatter_.")

        orig_repeat_interleave = torch.repeat_interleave

        def patched_repeat_interleave(input,
                                      repeats,
                                      dim=None,
                                      output_size=None):
            if (isinstance(input, torch.Tensor) and input.device.type == "tpu"
                    and input.dtype
                    in (torch.int64, torch.int32, torch.int16, torch.bool)):
                input_cpu = input.cpu()
                repeats_cpu = repeats.cpu() if isinstance(
                    repeats, torch.Tensor) else repeats
                res_cpu = orig_repeat_interleave(input_cpu,
                                                 repeats_cpu,
                                                 dim=dim,
                                                 output_size=output_size)
                return res_cpu.to(input.device)
            return orig_repeat_interleave(input,
                                          repeats,
                                          dim=dim,
                                          output_size=output_size)

        torch.repeat_interleave = patched_repeat_interleave
        logger.info("Applied TPU patch: torch.repeat_interleave.")

        orig_cumsum = torch.cumsum

        def patched_cumsum(input, *args, **kwargs):
            if (isinstance(input, torch.Tensor) and input.device.type == "tpu"
                    and input.dtype
                    in (torch.int64, torch.int32, torch.int16, torch.bool)):
                return orig_cumsum(input.cpu(), *args,
                                   **kwargs).to(input.device)
            return orig_cumsum(input, *args, **kwargs)

        torch.cumsum = patched_cumsum

        orig_tensor_cumsum = torch.Tensor.cumsum

        def patched_tensor_cumsum(self, *args, **kwargs):
            if (self.device.type == "tpu" and self.dtype
                    in (torch.int64, torch.int32, torch.int16, torch.bool)):
                return orig_tensor_cumsum(self.cpu(), *args,
                                          **kwargs).to(self.device)
            return orig_tensor_cumsum(self, *args, **kwargs)

        torch.Tensor.cumsum = patched_tensor_cumsum
        logger.info("Applied TPU patch: torch.cumsum.")
    except Exception:
        pass


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
        "tpu_int8", "compressed-tensors", "awq", "fp8", "mxfp4", "modelopt_fp4"
    ]

    additional_env_vars: list[str] = [
        "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS",
        "TPU_MULTIHOST_BACKEND", "VLLM_MLA_DISABLE", "TPU_BACKEND_TYPE",
        "ENABLE_QUANTIZED_MATMUL_KERNEL", "REQUANTIZE_BLOCK_SIZE",
        "REQUANTIZE_WEIGHT_DTYPE", "MOE_REQUANTIZE_BLOCK_SIZE",
        "MOE_REQUANTIZE_WEIGHT_DTYPE",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
        "TORCHINDUCTOR_AUTOGRAD_CACHE", "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY", "TPU_KERNEL_ITER_MODE",
        "TPU_KERNEL_RELOAD_MODULES"
    ]

    # The "Platform" base class has import_kernels() that tries to import
    # vllm._C, which is not available in the TPU Platform setup.
    # Override it to do nothing so we won't cause confusing warning logs.
    @classmethod
    def import_kernels(cls) -> None:
        return

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

        sb_addresses = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
        sb_count = len(sb_addresses.split(",")) if sb_addresses else 0
        if sb_count != world_size:
            sb_ports = [
                portpicker.pick_unused_port() for _ in range(world_size)
            ]
            os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
                f"localhost:{p}" for p in sb_ports)

        os.environ["TORCH_TPU_TOPOLOGY"] = cls._get_tpu_topology(world_size)

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

        if getattr(attn_selector_config, "use_mla", False):
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
        apply_tpu_patches()
        _apply_model_specific_patches(vllm_config.model_config)

        if vllm_envs.VLLM_TPU_USING_PATHWAYS:
            raise NotImplementedError(
                "Pathways is not supported by torchtpu-vllm. "
                "Unset VLLM_TPU_USING_PATHWAYS.")
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
                linear_until=envs.TPU_TOKEN_BUCKET_LINEAR_UNTIL,
                linear_interval=envs.TPU_TOKEN_BUCKET_LINEAR_INTERVAL,
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

        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config

        is_hybrid = model_config.is_hybrid
        if vllm_config.speculative_config is not None and scheduler_config.async_scheduling:
            method = vllm_config.speculative_config.method
            if not vllm_config.speculative_config.use_eagle():
                # Ngram needs the sampled tokens on the host, which async defers.
                raise NotImplementedError(
                    f"Async scheduling with speculative method '{method}' is "
                    "not supported on TPU; Run with async_scheduling=False.")
        # Hybrid (attention + Mamba) models with prefix caching enabled need
        # align-mode mamba state seeding (MambaApcStateCopier on the
        # typed-view layout, or the pool's seed copies); other cache modes
        # and speculative decoding must be rejected up front instead of
        # failing partway through warmup.
        if is_hybrid and cache_config.enable_prefix_caching:
            if cache_config.mamba_cache_mode != "align":
                raise NotImplementedError(
                    "Prefix caching on hybrid Mamba models requires "
                    "mamba_cache_mode='align' (got "
                    f"{cache_config.mamba_cache_mode!r}).")
            if vllm_config.speculative_config is not None:
                raise NotImplementedError(
                    "Speculative decoding is not yet supported with hybrid "
                    "Mamba prefix caching (mamba_cache_mode='align').")

        parallel_config = vllm_config.parallel_config
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
                    ensure_pcp_local_rank_remap(
                        torch_tpu_world_size,
                        get_topology=cls._get_tpu_topology)
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

        if (scheduler_config.is_multimodal_model
                and not _is_language_model_only_config(model_config)
                and not scheduler_config.disable_chunked_mm_input):
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

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
                "TPUConnector",
                "TPUConnectorV2",
                "TPURaidenConnector",
                "TPUMultiConnector",
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
            is_hybrid_offloading = (kv_transfer_config.kv_connector
                                    == "OffloadingConnector" and is_hybrid)
            if (is_hybrid_offloading
                    and not envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL):
                # Hybrid CPU offloading transfers whole pool rows; the
                # typed-view layout has no uniform per-block row to copy.
                raise ValueError(
                    "CPU offloading (OffloadingConnector) on hybrid "
                    "attention+Mamba models requires the unified block "
                    "pool; set TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1")

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        cache_config = vllm_config.cache_config
        backend_cls = cls._find_non_ssm_backend(vllm_config)

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
        logger.info("Using KV cache block size: %s", cache_config.block_size)

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
    """Sizes doubling from min_token_size until max_token_size is covered."""
    paddings = []
    num = min_token_size
    while True:
        paddings.append(num)
        if num >= max_token_size:
            return paddings
        num *= 2


def _get_linear_then_exponential_token_paddings(
        min_token_size: int, max_token_size: int, linear_until: int,
        linear_interval: int) -> list[int]:
    """Sizes stepping by ``linear_interval`` to ``linear_until``, doubling after."""
    linear_end = min(linear_until, max_token_size)
    paddings = sorted(
        {min_token_size}
        | set(range(linear_interval, linear_end + 1, linear_interval)))
    while paddings[-1] < max_token_size:
        paddings.append(paddings[-1] * 2)
    return paddings


def _get_token_paddings(min_token_size: int,
                        max_token_size: int,
                        linear_until: int = 0,
                        linear_interval: int = 16) -> list[int]:
    """Generate a list of padding size, starting from min_token_size,
    ending with a number that can cover max_token_size.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0

    if linear_until:
        # Linear padding sizes up to linear_until, then exponential doubling after that.
        paddings = _get_linear_then_exponential_token_paddings(
            min_token_size, max_token_size, linear_until, linear_interval)
    else:
        # Double sizes from min_token_size until max_token_size is covered.
        paddings = _get_exponential_token_paddings(min_token_size,
                                                   max_token_size)
    logger.info("Using token paddings: %s", paddings)
    return paddings
