# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm_torchtpu.env_override  # noqa: F401
from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _patch_vllm_aot_compile_cache_key() -> None:
    """Include the TPU compiler hash in vLLM's outer AOT cache key."""
    from vllm.compilation import caching

    original = caching.aot_compile_hash_factors
    if getattr(original, "_tpu_compiler_hash_patch", False):
        return

    def tpu_aot_compile_hash_factors(vllm_config):
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash
        return [
            *original(vllm_config),
            compute_tpu_compilation_hash(vllm_config),
        ]

    tpu_aot_compile_hash_factors._tpu_compiler_hash_patch = True
    caching.aot_compile_hash_factors = tpu_aot_compile_hash_factors
    logger.info("Applied TPU patch: include compiler hash in AOT cache key.")


def _patch_vllm_config_hash_ignore_diagnostics() -> None:
    """Keep diagnostics-only ``additional_config`` keys out of cache keys.

    ``VllmConfig.compute_hash()`` unconditionally folds
    ``json.dumps(additional_config)`` into its result, and that hash feeds both
    the AOT compile cache key (``caching.aot_compile_hash_factors``) and the
    piecewise cache dir (``backends.py``). The phased profiler is configured
    through ``additional_config``, so a per-run trace directory would force a
    full recompile on every server start. Upstream takes the same care in
    ``ProfilerConfig.compute_hash()`` ("this config will not affect the
    computation graph"); this is the equivalent carve-out for the plugin-owned
    keys, mirroring ``_TPU_COMPILE_ENV_IGNORED`` for env vars.
    """
    import copy

    from vllm.config import VllmConfig

    from vllm_torchtpu.runner.utils import HASH_IGNORED_ADDITIONAL_CONFIG_KEYS

    if getattr(VllmConfig, "_tpu_additional_config_hash_patch", False):
        return

    original = VllmConfig.compute_hash

    def compute_hash(self) -> str:
        additional_config = self.additional_config
        if not isinstance(additional_config, dict):
            return original(self)
        filtered = {
            key: value
            for key, value in additional_config.items()
            if key not in HASH_IGNORED_ADDITIONAL_CONFIG_KEYS
        }
        if filtered == additional_config:
            return original(self)
        # Hash a shallow copy with the filtered dict rather than mutating the
        # live config: compute_hash() may run while other code holds the same
        # VllmConfig, and a shallow copy shares every sub-config by reference
        # (cheap, no re-validation -- __post_init__ is not re-run) while giving
        # compute_hash its own additional_config to read.
        proxy = copy.copy(self)
        proxy.additional_config = filtered
        return original(proxy)

    VllmConfig._tpu_upstream_compute_hash = original
    VllmConfig.compute_hash = compute_hash
    VllmConfig._tpu_additional_config_hash_patch = True
    logger.info(
        "Applied TPU patch: exclude diagnostics-only additional_config keys "
        "%s from the compile cache key.",
        sorted(HASH_IGNORED_ADDITIONAL_CONFIG_KEYS))


def _patch_vllm_tpu_group_custom_ops() -> None:
    """Disable vLLM custom collective ops for TPU in this TorchTPU integration.

    vLLM's `torch.ops.vllm.all_reduce/all_gather` custom ops are not available
    on the TorchTPU backend in this repo. Force GroupCoordinator to use
    communicator-backed collectives instead.

    TODO (geyuhao): Do we actually need to support torch.ops.vllm.* on TPU? Currently
    this patch will use torch.distributed.all_reduce/all_gather.
    """

    from vllm.distributed.parallel_state import GroupCoordinator

    if getattr(GroupCoordinator, "_tpu_no_custom_collective_patch", False):
        return

    original_init = GroupCoordinator.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.use_custom_op_call = False

    GroupCoordinator.__init__ = patched_init
    GroupCoordinator._tpu_no_custom_collective_patch = True
    logger.info("Applied TPU patch: disable vLLM custom collective ops.")


def _patch_default_moe_runner_select_forward() -> None:
    """Keep DefaultMoERunner on the direct ``_moe_forward`` path under OOT.

    Upstream gates this on ``is_tpu()``; under our OOT plugin that's False
    and we'd take ``torch.ops.vllm.moe_forward``, which captures weight
    tensors as closures and trips torch_tpu's MLIR builder.
    """
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as _dmr

    if getattr(_dmr.MoERunner, "_tpu_select_forward_patch", False):
        return

    def patched_select(self):
        return (_dmr._moe_forward
                if self.shared_experts is None else _dmr._moe_forward_shared)

    _dmr.MoERunner._select_forward = patched_select
    _dmr.MoERunner._tpu_select_forward_patch = True
    logger.info(
        "Applied TPU patch: DefaultMoERunner uses direct _moe_forward.")


def _patch_expert_map_host_lookup() -> None:
    """Serve expert-map lookups from a host-side copy of the expert map.

    ``ExpertMapManager.map_global_to_local`` reads the device-resident
    expert map with ``.item()`` once per (expert, shard) tensor during MoE
    weight loading. On TPU every ``.item()`` is a blocking sync that
    flushes the deferred graph, so loading dispatches one tiny device
    program per expert copy (~200k per worker on Qwen3.5-397B). Index a
    cached host copy in Python instead; the cache follows reassignment
    and in-place rebalance via the tensor's version counter.
    """
    from vllm.model_executor.layers.fused_moe import expert_map_manager as _em

    cls = _em.ExpertMapManager
    if getattr(cls, "_tpu_host_expert_map_patch", False):
        return

    def map_global_to_local(self, global_id: int) -> int:
        expert_map = self._expert_map
        if expert_map is None:
            return global_id
        if (self._tpu_expert_map_source is not expert_map
                or self._tpu_expert_map_version != expert_map._version):
            self._tpu_expert_map_source = expert_map
            self._tpu_expert_map_version = expert_map._version
            self._tpu_expert_map_host = expert_map.tolist()
        return self._tpu_expert_map_host[global_id]

    cls.map_global_to_local = map_global_to_local
    cls._tpu_expert_map_source = None
    cls._tpu_expert_map_version = None
    cls._tpu_expert_map_host = None
    cls._tpu_host_expert_map_patch = True
    logger.info("Applied TPU patch: host-side expert-map lookup.")


def _patch_vllm_disable_compile_ranges() -> None:
    """Force TPU to compile only fixed bucketed sizes, never dynamic ranges.

    ``VllmConfig._set_compile_ranges()`` always appends
    ``scheduler_config.max_num_batched_tokens`` to ``compile_ranges_endpoints``,
    so ``PiecewiseBackend`` builds a dynamic ``Range(1, max)`` and compiles it
    with backed ``SymInt`` shapes. The torch_tpu backend rejects SymInts ("does
    not support dynamic shape" / "No shape env"). The TPU runner pads every shape
    to an exact ``compile_size``, so dynamic ranges are never needed.

    Patch the consumption point: ``CompilationConfig.get_compile_ranges() -> []``.
    Clearing the field (e.g. in the worker) does not stick --
    ``_set_compile_ranges`` rewrites it on every config reconstruction
    (Qwen3-VL ``with_hf_config`` -> ``replace``), and the compile-time
    compilation_config is not the instance a platform/worker hook can reach --
    so a class-level override is the robust fix.
    """
    from vllm.config.compilation import CompilationConfig

    def get_compile_ranges(self):
        return []

    CompilationConfig.get_compile_ranges = get_compile_ranges
    logger.info("Applied TPU patch: disable dynamic compile_ranges (static "
                "compile_sizes only).")


def _patch_rowparallel_defer_bias() -> None:
    """Defer RowParallelLinear bias past the TP all-reduce on TPU.

    Upstream ``RowParallelLinear.forward`` fuses bias into the GEMM **only on
    rank 0** (``bias_ = None if tp_rank > 0 ... else self.bias``). Under TP>1
    this makes the per-rank FX graphs structurally different, so the torch_tpu
    XLA backend assigns mismatched all_reduce channel IDs and the collective is
    broken.
    """
    from vllm.distributed import (split_tensor_along_last_dim,
                                  tensor_model_parallel_all_reduce)
    from vllm.model_executor.layers.linear import RowParallelLinear

    # Preserve the genuine upstream forward once so re-application (e.g. a
    # test that re-applies after mocking the all-reduce) never captures an
    # already-patched forward as the trunk reference.
    if not hasattr(RowParallelLinear, "_tpu_upstream_forward"):
        RowParallelLinear._tpu_upstream_forward = RowParallelLinear.forward

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        do_all_reduce = self.reduce_results and self.tp_size > 1
        # Defer the (replicated) bias past the all-reduce so every rank runs an
        # identical GEMM graph -> matching XLA all_reduce channel IDs.
        defer_tpu_bias = (do_all_reduce and not self.skip_bias_add
                          and self.bias is not None)
        if defer_tpu_bias:
            bias_ = None
        else:
            bias_ = (None if
                     (self.tp_rank > 0 or self.skip_bias_add) else self.bias)
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if do_all_reduce:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if defer_tpu_bias:
            output = output + self.bias

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    RowParallelLinear.forward = forward
    logger.info(
        "Applied TPU patch: defer RowParallelLinear bias past TP all_reduce.")


def _patch_disable_sequence_parallel_moe() -> None:
    """Disable vLLM's sequence-parallel MoE on TPU.

    The sequence-parallel path is tied to vLLM all2all MoE backends. The TPU
    communicator handles EP with unsplit token buffers.
    """
    from vllm.config.parallel import ParallelConfig

    if getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        return

    ParallelConfig.use_sequence_parallel_moe = property(lambda self: False)
    ParallelConfig._tpu_no_sp_moe_patch = True
    logger.info("Applied TPU patch: disable sequence-parallel MoE.")


def _patch_disable_dp_ubatch() -> None:
    """Disable DP microbatching (DBO) on TPU.

    TPU does not support vLLM DP microbatching yet. Force the post-sync ubatch
    decision off so DP uses one forward per engine step.
    """
    from vllm.v1.worker import dp_utils

    if getattr(dp_utils, "_tpu_no_ubatch_patch", False):
        return

    dp_utils._post_process_ubatch = lambda tensor, num_ubatches: False
    dp_utils._tpu_no_ubatch_patch = True
    logger.info("Applied TPU patch: disable DP microbatching.")


def _patch_moe_no_ep_tp_scope() -> None:
    """Keep non-EP MoE tensor parallelism scoped to each DP engine.

    Upstream vLLM flattens TP across DP for MoE when expert parallelism is
    disabled. This temporary TPU patch avoids that path until the TPU dummy
    batch path can keep no-EP MoE ranks in the same lockstep as upstream
    model-internal DP.
    """
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.fused_moe import FusedMoEParallelConfig

    if getattr(FusedMoEParallelConfig, "_tpu_no_ep_tp_scope_patch", False):
        return

    original_make = FusedMoEParallelConfig.make

    # TODO: Remove this patch in a follow-up PR. TorchTPU should match vLLM's
    # no-EP MoE convention by flattening DP*TP and extending the coordinated
    # dummy/lockstep execution used for EP to no-EP MoE as well.
    def patched_make(tp_size_, pcp_size_, dp_size_, sp_size_,
                     vllm_parallel_config):
        use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1
                  and vllm_parallel_config.enable_expert_parallel)
        if use_ep:
            return original_make(tp_size_, pcp_size_, dp_size_, sp_size_,
                                 vllm_parallel_config)

        dp_rank = (parallel_state.get_dp_group().rank_in_group
                   if dp_size_ > 1 else 0)
        pcp_rank = (parallel_state.get_pcp_group().rank_in_group
                    if pcp_size_ > 1 else 0)
        tp_rank = (0 if tp_size_ == 1 else
                   parallel_state.get_tensor_model_parallel_rank())
        return FusedMoEParallelConfig(
            tp_size=tp_size_,
            tp_rank=tp_rank,
            pcp_size=pcp_size_,
            pcp_rank=pcp_rank,
            dp_size=dp_size_,
            dp_rank=dp_rank,
            ep_size=1,
            ep_rank=0,
            sp_size=sp_size_,
            use_ep=False,
            all2all_backend=vllm_parallel_config.all2all_backend,
            enable_eplb=vllm_parallel_config.enable_eplb,
        )

    FusedMoEParallelConfig.make = staticmethod(patched_make)
    FusedMoEParallelConfig._tpu_no_ep_tp_scope_patch = True
    logger.info(
        "Applied TPU patch: scope non-EP MoE TP inside each DP engine.")


def _patch_multiproc_worker_global_rank_env() -> None:
    """Set TorchTPU binding env on spawned workers before torch_tpu import.

    Native vLLM DP passes per-engine rank/local_rank to workers and computes
    global torch.distributed rank later from ParallelConfig. TorchTPU currently
    reads RANK/LOCAL_RANK/WORLD_SIZE earlier for physical chip/PjRt binding, so
    this shim exposes the unified single-host DP*TP slice to TorchTPU without
    changing the worker rank arguments passed through vLLM.
    """
    import os as _os

    from vllm.v1.executor.multiproc_executor import WorkerProc

    from vllm_torchtpu.worker.tpu_rank_binding import get_tpu_worker_binding

    if getattr(WorkerProc, "_tpu_global_rank_env_patch", False):
        return
    _orig = WorkerProc.make_worker_process

    def _wrapped(vllm_config, local_rank, rank, *args, **kwargs):
        pc = vllm_config.parallel_config
        binding = get_tpu_worker_binding(pc, rank, local_rank, env=_os.environ)
        _os.environ.update(binding.as_env())
        logger.info(
            "Applied TPU patch: worker spawn env RANK=%d LOCAL_RANK=%d "
            "WORLD_SIZE=%d LOCAL_WORLD_SIZE=%d "
            "(rank=%d local_rank=%d dp_rank=%d dp_size=%d offset=%d "
            "init_local_rank=%d)", binding.rank, binding.local_rank,
            binding.world_size, binding.local_world_size, rank, local_rank,
            binding.dp_rank, binding.dp_size, binding.local_rank_offset,
            binding.init_local_rank)
        if binding.pcp_local_rank_remap is not None:
            logger.info(
                "Applied TPU patch: PCP native-rank worker spawn binding "
                "native_local_rank=%d local_rank_env=%d "
                "tpu_local_rank_env=%d local_world=%d tpu_local_world=%d "
                "remap=%s source=%s",
                binding.native_local_rank,
                binding.init_local_rank,
                binding.local_rank,
                binding.world_size,
                binding.local_world_size,
                binding.pcp_local_rank_remap,
                binding.pcp_remap_source,
            )
        return _orig(vllm_config, local_rank, rank, *args, **kwargs)

    WorkerProc.make_worker_process = staticmethod(_wrapped)
    WorkerProc._tpu_global_rank_env_patch = True
    logger.info("Applied TPU patch: MultiprocExecutor worker binding env.")


def _run_engine_core_with_tpu_patches(*args, **kwargs):
    _patch_vllm_hybrid_pcp_block_sizes()
    _patch_vllm_kimi_kda_layer_counts()

    from vllm.v1.engine.core import EngineCoreProc

    original_run = EngineCoreProc._tpu_original_run_engine_core
    return original_run(*args, **kwargs)


def _patch_vllm_kimi_kda_layer_counts() -> None:
    """Patch ModelConfig.get_num_layers_by_block_type to report accurate counts
    for Kimi models with KDA using linear_attn_config (e.g. Kimi-K3).
    """
    from vllm.config.model import ModelConfig

    if getattr(ModelConfig, "_tpu_kimi_kda_layer_counts_patched", False):
        return

    original_get_num_layers = ModelConfig.get_num_layers_by_block_type

    def patched_get_num_layers_by_block_type(self,
                                             parallel_config,
                                             block_type="attention") -> int:
        try:
            return original_get_num_layers(self, parallel_config, block_type)
        except ValueError:
            pass

        start, end = self.get_layers_start_end_indices(parallel_config)
        linear_attn_config = getattr(self.hf_text_config, "linear_attn_config",
                                     None)
        if linear_attn_config is not None and block_type == "attention":
            kda_layers = set(linear_attn_config.get("kda_layers", []))
            return sum(
                (idx + 1) not in kda_layers for idx in range(start, end))

        return original_get_num_layers(self, parallel_config, block_type)

    ModelConfig.get_num_layers_by_block_type = (
        patched_get_num_layers_by_block_type)
    ModelConfig._tpu_kimi_kda_layer_counts_patched = True
    logger.info(
        "Applied TPU patch: accurate layer counts for Kimi-K3 / KDA hybrid"
        " models.")


def _patch_vllm_hybrid_pcp_block_sizes() -> None:
    """Resolve full-attention + Mamba block sizes under TPU PCP.

    Upstream vLLM rejects all multi-group KV cache configs when
    ``pcp_world_size > 1`` because it cannot infer the scheduler block size for
    mixed cache types. TorchTPU's Qwen3.5 path has a narrower, well-defined
    layout: each physical cache page is rank-local, while the scheduler and
    cache coordinator operate on logical pages spanning all PCP ranks. Present
    those logical page sizes to the generic hybrid coordinator instead of
    rejecting the config.
    """
    import math
    import sys
    from dataclasses import replace

    from vllm.v1.core import kv_cache_coordinator, kv_cache_utils
    from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheSpec,
                                            MambaSpec, UniformTypeKVCacheSpecs)

    already_patched = getattr(kv_cache_utils,
                              "_tpu_hybrid_pcp_block_sizes_patch", False)
    original_resolve = getattr(
        kv_cache_utils,
        "_tpu_original_resolve_kv_cache_block_sizes",
        kv_cache_utils.resolve_kv_cache_block_sizes,
    )

    def _iter_leaf_specs(spec: KVCacheSpec):
        if isinstance(spec, UniformTypeKVCacheSpecs):
            yield from spec.kv_cache_specs.values()
        else:
            yield spec

    def _has_spec_type(spec: KVCacheSpec, spec_type: type) -> bool:
        return any(
            isinstance(leaf, spec_type) for leaf in _iter_leaf_specs(spec))

    def _is_attention_mamba_hybrid(groups) -> bool:
        return any(
            _has_spec_type(group.kv_cache_spec, MambaSpec)
            for group in groups) and any(
                _has_spec_type(group.kv_cache_spec, AttentionSpec)
                for group in groups)

    def _with_effective_block_size(spec: KVCacheSpec, pcp: int) -> KVCacheSpec:
        """Restate a rank-local spec in the logical (all-ranks) page size."""
        if isinstance(spec, UniformTypeKVCacheSpecs):
            effective_specs = {
                name: _with_effective_block_size(leaf, pcp)
                for name, leaf in spec.kv_cache_specs.items()
            }
            effective_block_size = math.lcm(
                *(leaf.block_size for leaf in effective_specs.values()))
            return replace(spec,
                           block_size=effective_block_size,
                           kv_cache_specs=effective_specs)
        if isinstance(spec, (AttentionSpec, MambaSpec)):
            return replace(spec, block_size=spec.block_size * pcp)
        return spec

    def _effective_block_size(spec: KVCacheSpec, pcp: int) -> int:
        return _with_effective_block_size(spec, pcp).block_size

    def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config):
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        dcp = parallel_config.decode_context_parallel_size
        pcp = parallel_config.prefill_context_parallel_size
        groups = kv_cache_config.kv_cache_groups

        if len(groups) <= 1 or dcp != 1 or pcp == 1:
            return original_resolve(kv_cache_config, vllm_config)

        if not _is_attention_mamba_hybrid(groups):
            return original_resolve(kv_cache_config, vllm_config)

        group_block_sizes = [
            _effective_block_size(group.kv_cache_spec, pcp) for group in groups
        ]
        scheduler_block_size = math.lcm(*group_block_sizes)

        connector_enabled = vllm_config.kv_transfer_config is not None
        if not (cache_config.enable_prefix_caching or connector_enabled):
            hash_block_size = scheduler_block_size
        elif (not hasattr(cache_config, "prefix_match_unit")
              or cache_config.prefix_match_unit is None):
            hash_block_size = math.gcd(*group_block_sizes)
        else:
            hash_block_size = cache_config.prefix_match_unit
            if any(bs % hash_block_size != 0 for bs in group_block_sizes):
                raise ValueError(
                    f"Invalid prefix_match_unit={hash_block_size}; all logical "
                    "PCP KV cache group block sizes must be divisible by "
                    "prefix_match_unit. Got "
                    f"block sizes={group_block_sizes}.")

        logger.info(
            "Applied TPU PCP hybrid KV cache block-size resolution: "
            "pcp=%d group_effective_block_sizes=%s scheduler_block_size=%d "
            "hash_block_size=%d", pcp, group_block_sizes, scheduler_block_size,
            hash_block_size)
        return scheduler_block_size, hash_block_size

    patched_resolve = (kv_cache_utils.resolve_kv_cache_block_sizes
                       if already_patched else resolve_kv_cache_block_sizes)
    if not already_patched:
        kv_cache_utils._tpu_original_resolve_kv_cache_block_sizes = (
            original_resolve)
        kv_cache_utils.resolve_kv_cache_block_sizes = patched_resolve
        kv_cache_utils._tpu_hybrid_pcp_block_sizes_patch = True

    hybrid_coordinator = kv_cache_coordinator.HybridKVCacheCoordinator
    if not getattr(hybrid_coordinator, "_tpu_hybrid_pcp_coordinator_patch",
                   False):
        original_hybrid_init = hybrid_coordinator.__init__

        def _hybrid_init_with_logical_pcp_blocks(self,
                                                 kv_cache_config,
                                                 max_model_len,
                                                 max_num_batched_tokens,
                                                 use_eagle,
                                                 enable_caching,
                                                 enable_kv_cache_events,
                                                 dcp_world_size,
                                                 pcp_world_size,
                                                 scheduler_block_size,
                                                 hash_block_size,
                                                 metrics_collector=None):
            groups = kv_cache_config.kv_cache_groups
            if (pcp_world_size > 1 and _is_attention_mamba_hybrid(groups)):
                if dcp_world_size != 1:
                    raise ValueError("TPU hybrid PCP cache coordination does "
                                     "not support DCP.")
                # The generic coordinator only needs logical token
                # granularity. Encoding PCP into each group spec and then
                # passing pcp_world_size=1 avoids multiplying block sizes
                # twice and keeps its prefix-cache lookup, hashing, and
                # allocation invariants internally consistent.
                kv_cache_config = replace(
                    kv_cache_config,
                    kv_cache_groups=[
                        replace(group,
                                kv_cache_spec=_with_effective_block_size(
                                    group.kv_cache_spec, pcp_world_size))
                        for group in kv_cache_config.kv_cache_groups
                    ])
                pcp_world_size = 1

            original_hybrid_init(
                self,
                kv_cache_config,
                max_model_len,
                max_num_batched_tokens,
                use_eagle,
                enable_caching,
                enable_kv_cache_events,
                dcp_world_size,
                pcp_world_size,
                scheduler_block_size,
                hash_block_size,
                metrics_collector,
            )

        hybrid_coordinator.__init__ = _hybrid_init_with_logical_pcp_blocks
        hybrid_coordinator._tpu_hybrid_pcp_coordinator_patch = True

    from vllm.v1.engine.core import EngineCoreProc

    # EngineCore imports the function directly, so patch module bindings as
    # well as the source module.
    for module_name in ("vllm.v1.engine.core", "vllm.v1.kv_offload.base"):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "resolve_kv_cache_block_sizes", patched_resolve)

    if not getattr(EngineCoreProc, "_tpu_engine_core_patch_wrapper", False):
        EngineCoreProc._tpu_original_run_engine_core = (
            EngineCoreProc.run_engine_core)
        EngineCoreProc.run_engine_core = staticmethod(
            _run_engine_core_with_tpu_patches)
        EngineCoreProc._tpu_engine_core_patch_wrapper = True

    if not already_patched:
        logger.info("Applied TPU patch: hybrid full-attention + Mamba PCP "
                    "block sizes.")


if "proxy" in envs.JAX_PLATFORMS:
    logger.info("Running vLLM on TPU via Pathways proxy.")
    # Must run pathwaysutils.initialize() before any JAX operations
    try:
        import traceback

        import pathwaysutils
        import vllm
        from vllm.platforms import (resolve_current_platform_cls_qualname,
                                    resolve_obj_by_qualname)
        pathwaysutils.initialize()
        logger.info("Module pathwaysutils is imported.")

        # Pathways requires eager resolution of vllm.current_platform instead of
        # lazy resolution in the normal code path. Since this part involves
        # global topology discovery across multiple hosts, the platform
        # resolution must happen before other components are loaded.
        logger.info("Eagerly resolving vLLM current_platform for Pathways.")
        platform_cls_qualname = resolve_current_platform_cls_qualname()
        resolved_platform_instance = resolve_obj_by_qualname(
            platform_cls_qualname)()
        vllm.platforms._current_platform = resolved_platform_instance
        vllm.platforms._init_trace = "".join(traceback.format_stack())
        logger.info(
            f"vLLM platform resolved to: {resolved_platform_instance.__class__.__name__}"
        )

    except Exception as e:
        logger.error(
            f"Error occurred while importing pathwaysutils or logging TPU info: {e}"
        )
