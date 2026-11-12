# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm_torchtpu.env_override  # noqa: F401
from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _reconcile_hybrid_producer_prefix_hits(scheduler) -> None:
    """Require a common local prefix hit on a hybrid KV producer.

    vLLM 0.26.1rc0 lets a hybrid model with any KV connector resume from the
    deepest per-group local hit. That is valid on a consumer when the connector
    transfers the missing groups, but a producer has no external state to fill
    the gap. In particular, FA can have a cached page where Mamba has no state
    checkpoint, and resuming from the FA-only boundary corrupts generation.

    Keep it local to the coordinator instance so
    consumer-side divergent lookup and remote suffix transfer are unchanged.
    """
    from types import MethodType

    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    kv_transfer_config = scheduler.vllm_config.kv_transfer_config
    coordinator = scheduler.kv_cache_manager.coordinator
    if (kv_transfer_config is None or not kv_transfer_config.is_kv_producer
            or not scheduler.has_mamba_layers
            or not isinstance(coordinator, HybridKVCacheCoordinator)):
        return

    def find_common_prefix_hit(self, block_hashes, max_cache_hit_length):
        blocks, hit_length, _ = self.find_longest_cache_hit(
            block_hashes, max_cache_hit_length)
        per_group_hits = (hit_length, ) * len(
            self.kv_cache_config.kv_cache_groups)
        return blocks, per_group_hits

    coordinator.find_longest_cache_hit_per_group = MethodType(
        find_common_prefix_hit, coordinator)
    logger.info("Reconciled hybrid producer prefix hits to a common boundary.")


def _patch_vllm_hybrid_producer_prefix_hits() -> None:
    """Reconcile hybrid connector cache hits for vLLM 0.26.0."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if Scheduler.__dict__.get("_tpu_hybrid_producer_prefix_hit_patch", False):
        return

    original_init = Scheduler.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _reconcile_hybrid_producer_prefix_hits(self)

    Scheduler.__init__ = patched_init
    Scheduler._tpu_hybrid_producer_prefix_hit_patch = True
    logger.info("Applied TPU patch: reconcile hybrid producer prefix hits.")


def _patch_vllm_mamba_split_scheduler_block_size() -> None:
    """Align Mamba prefill splits to the scheduler block size.

    vLLM 0.27.0 uses the rank-local physical ``CacheConfig.block_size`` in
    ``Scheduler._mamba_block_aligned_split``.  Under hybrid PCP, prefix hashes
    and reusable Mamba states instead advance at ``Scheduler.block_size``, the
    logical block spanning all PCP ranks.  Calling the upstream method with a
    scheduler-local cache-config view supplies the logical block size
    without mutating the shared ``VllmConfig.cache_config`` or copying the
    scheduling algorithm into this plugin.
    """
    import copy
    from functools import wraps

    from vllm.v1.core.sched.scheduler import Scheduler

    original_split = Scheduler._mamba_block_aligned_split
    if getattr(original_split, "_tpu_scheduler_block_size_patch", False):
        return

    @wraps(original_split)
    def split_with_scheduler_block_size(self, *args, **kwargs):
        physical_cache_config = self.cache_config
        if physical_cache_config.block_size == self.block_size:
            return original_split(self, *args, **kwargs)

        scheduler_cache_config = copy.copy(physical_cache_config)
        scheduler_cache_config.block_size = self.block_size
        # schedule() calls this method synchronously on the EngineCore thread,
        # and the upstream implementation contains no callback or await point.
        self.cache_config = scheduler_cache_config
        try:
            return original_split(self, *args, **kwargs)
        finally:
            self.cache_config = physical_cache_config

    split_with_scheduler_block_size._tpu_scheduler_block_size_patch = True
    Scheduler._mamba_block_aligned_split = split_with_scheduler_block_size
    logger.info(
        "Applied TPU patch: align Mamba prefill splits to scheduler blocks.")


def _patch_vllm_aot_compile_cache_key() -> None:
    """Include the TPU compiler hash, and the trace ordinal, in vLLM's outer
    AOT cache key."""
    from vllm.compilation import caching

    original = caching.aot_compile_hash_factors
    if getattr(original, "_tpu_compiler_hash_patch", False):
        return

    def tpu_aot_compile_hash_factors(vllm_config):
        from vllm_torchtpu.compilation import shape_variants
        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash
        return [
            *original(vllm_config),
            compute_tpu_compilation_hash(vllm_config),
            *filter(None, [shape_variants.aot_cache_tag()]),
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


def _patch_default_moe_runner_select_forward() -> None:
    """Keep DefaultMoERunner on the direct ``_moe_forward`` path under OOT.

    Upstream gates this on ``is_tpu()``, which is False for this OOT plugin.
    The alternate ``torch.ops.vllm.moe_forward`` path captures weight tensors
    as closures that torch_tpu's MLIR builder cannot handle.
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


def _patch_moe_runner_fused_output_is_reduced() -> None:
    """Tell vLLM the fused EP MoE kernel already reduced its output.

    `fused_moe_ep` returns this rank's tokens combined over ALL experts -- the
    kernel pushes each result row back to the chip that owns the token, so
    there is nothing left to reduce. vLLM does not know that: it decides from
    `MoERunner._fused_output_is_reduced`, which reads
    `quant_method.moe_kernel.output_is_reduced()`, and the TPU fp8 method never
    sets `moe_kernel` -- it declares itself through `supports_internal_mk`
    instead. So `moe_runner.py` all-reduces the already-combined output
    whenever `tp_size > 1 or ep_size > 1`.

    Under data parallelism the tensor-parallel group has world size one and
    all three all-reduces this flag gates are `return input_`, so the patch
    changes no answer there. Under tensor parallelism the kernel arms for the
    sequence-parallel steps and returns each rank's own rows; the patch keeps
    the unpatched path, which would multiply every routed MoE output by
    tp_size, from ever running.

    Reporting True here is the correct signal rather than deleting the
    all-reduce, because the SHARED expert genuinely is TP-partial
    (`Qwen3NextMLP(..., reduce_results=False)`): the same flag routes it to
    `_maybe_reduce_shared_expert_output`, which reduces it on its own.
    """
    # Install only when the fused kernel is enabled, matching the gate in
    # `prebuild_fused_moe_ep`. Any additional enablement path must update
    # both gates so reduced-output reporting follows kernel eligibility.
    if not envs.USE_MOE_FUSED_EP_KERNEL:
        return

    from vllm.model_executor.layers.fused_moe.runner import moe_runner as _mr

    if getattr(_mr.MoERunner, "_tpu_fused_output_reduced_patch", False):
        return

    original = _mr.MoERunner._fused_output_is_reduced

    @property
    def patched(self):
        # `skip_final_all_reduce` asserts the fused output is NOT pre-reduced,
        # and in that mode the model reduces externally anyway, so leave it be.
        if getattr(self.moe_config, "skip_final_all_reduce", False):
            return original.fget(self)
        try:
            from vllm_torchtpu.layers.vllm.fused_moe_ep import \
                fused_moe_ep_supported as fused_moe_ep_v2_supported
            from vllm_torchtpu.layers.vllm.fused_moe_ep_v3 import \
                fused_moe_ep_supported
        except ImportError:
            return original.fget(self)
        # Asked of the layer's own quant method, because arming is per layer:
        # a model whose layers do not all qualify must not have one armed layer
        # answer for the rest.
        owner = getattr(self, "_quant_method", None)
        return (fused_moe_ep_supported(owner)
                or fused_moe_ep_v2_supported(owner) or original.fget(self))

    _mr.MoERunner._fused_output_is_reduced = patched
    _mr.MoERunner._tpu_fused_output_reduced_patch = True
    logger.info("Applied TPU patch: the fused EP MoE output is pre-reduced.")


def _patch_moe_fused_ep_threshold_contract() -> None:
    """Use external DP dispatch below the fused-EP threshold."""
    from torch.nn import functional as F
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    threshold = envs.TPU_MOE_FUSED_EP_STEP_MIN_TOKENS
    if not envs.USE_MOE_FUSED_EP_KERNEL or threshold <= 0:
        return

    runner_cls = moe_runner.MoERunner
    if getattr(runner_cls, "_tpu_fused_ep_threshold_patch", False):
        return
    original_forward_impl = runner_cls._forward_impl

    def forward_impl(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
    ):
        config = self.moe_config
        # Prefill context parallelism adds its own collectives to vLLM's
        # dispatch and combine; this copy of the small-shape path carries
        # none of them, so it stays out of that case.
        if (config.dp_size <= 1 or getattr(config, "pcp_size", 1) > 1
                or hidden_states.shape[0] >= threshold):
            return original_forward_impl(
                self,
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )

        quant_method = self.routed_experts.quant_method
        external_apply = getattr(quant_method,
                                 "apply_monolithic_external_dispatch", None)
        if external_apply is None:
            return original_forward_impl(
                self,
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )

        # Run the baseline external-dispatch path for small shapes.
        self.routed_experts._ensure_moe_quant_config_init()
        self._maybe_sync_shared_experts_stream(shared_experts_input)
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states,
                                         self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            ep_group = moe_runner.get_ep_group()
            result = ep_group.dispatch_router_logits(
                hidden_states,
                router_logits,
                config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

            self._maybe_apply_shared_experts(
                shared_experts_input,
                moe_runner.SharedExpertsOrder.NO_OVERLAP,
            )
            fused_output = external_apply(
                layer=self.routed_experts,
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
            self._maybe_apply_shared_experts(
                shared_experts_input,
                moe_runner.SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
            )

            fused_output = ep_group.combine(
                fused_output,
                config.is_sequence_parallel,
            )
            if self._shared_experts is None:
                return fused_output
            return self._shared_experts.output, fused_output

    runner_cls._forward_impl = forward_impl
    runner_cls._tpu_fused_ep_threshold_patch = True
    logger.info(
        "Applied TPU patch: use external MoE dispatch below %d tokens.",
        threshold)


def _patch_moe_explicit_pcp_collectives() -> None:
    """Give each PCP MoE implementation exactly one collective owner.

    vLLM treats PCP+EP as an all-to-all-kernel configuration and therefore
    skips the explicit PCP all-gather/reduce-scatter in ``MoERunner``.  The
    TorchTPU monolithic MoE kernels only compute local expert contributions;
    they only implement internal PCP dispatch/combine when MoE collective
    chunking is enabled.  Otherwise keep dispatch/combine explicit.

    Chunk pipelining is a process-wide configuration, so it can select the
    contract through ``FusedMoEParallelConfig.use_all2all_kernels``.  The fused
    EP kernel is different: it is armed per layer after checking that layer's
    weights, routing and device resources.  Patch ``MoERunner`` as well so an
    armed layer suppresses the outer PCP pair, while a refused layer in the
    same model still gets the explicit fallback collectives.
    """
    from vllm.model_executor.layers.fused_moe import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.runner import \
        moe_runner as _moe_runner

    if not getattr(FusedMoEParallelConfig,
                   "_tpu_explicit_pcp_collectives_patch", False):
        upstream_property = FusedMoEParallelConfig.use_all2all_kernels
        upstream_getter = upstream_property.fget
        assert upstream_getter is not None

        def use_all2all_kernels(self):
            if (self.use_ep and self.pcp_size > 1 and self.dp_size == 1
                    and not self.is_sequence_parallel
                    and envs.TPU_MOE_COLLECTION_CHUNK_SIZE <= 0):
                return False
            return upstream_getter(self)

        FusedMoEParallelConfig.use_all2all_kernels = property(
            use_all2all_kernels, doc=upstream_property.__doc__)
        FusedMoEParallelConfig._tpu_explicit_pcp_collectives_patch = True

    runner_cls = _moe_runner.MoERunner
    if not getattr(runner_cls, "_tpu_fused_ep_pcp_collectives_patch", False):
        upstream_dispatch = runner_cls._maybe_dispatch
        upstream_combine = runner_cls._maybe_combine

        def fused_ep_owns_pcp_collectives(self) -> bool:
            if self.moe_config.pcp_size <= 1:
                return False
            from vllm_torchtpu.layers.vllm.fused_moe_ep import \
                fused_moe_ep_supported as fused_moe_ep_v2_supported
            from vllm_torchtpu.layers.vllm.fused_moe_ep_v3 import \
                fused_moe_ep_supported
            owner = getattr(self, "_quant_method", None)
            return (fused_moe_ep_supported(owner)
                    or fused_moe_ep_v2_supported(owner))

        def maybe_dispatch(self, hidden_states, router_logits):
            if fused_ep_owns_pcp_collectives(self):
                # An armed fused method also reports supports_internal_mk, so
                # upstream's generic DP/EP dispatch is a no-op.  The only work
                # being bypassed here is its explicit PCP all-gather pair.
                return hidden_states, router_logits
            return upstream_dispatch(self, hidden_states, router_logits)

        def maybe_combine(self, shared_output, hidden_states):
            if fused_ep_owns_pcp_collectives(self):
                # The fused program has already pushed and summed routed rows
                # back onto their token-owning PCP rank.  Preserve upstream's
                # return shape for a separately-computed shared expert.
                if self.shared_experts is not None:
                    return shared_output, hidden_states
                return hidden_states
            return upstream_combine(self, shared_output, hidden_states)

        runner_cls._maybe_dispatch = maybe_dispatch
        runner_cls._maybe_combine = maybe_combine
        runner_cls._tpu_fused_ep_pcp_collectives_patch = True

    logger.info("Applied TPU patch: use explicit PCP collectives unless MoE "
                "chunk pipelining or an armed fused EP layer owns them.")


def _patch_qwen3_next_fused_qk_norm_rope_gate() -> None:
    """Use the TPU split, QK-RMSNorm, RoPE, and gate kernel for decode buckets.

    TPU_FUSED_QK_NORM_ROPE enables the fusion for Qwen3-Next attention.
    The eager path serves rows above the cap because the kernel holds
    its complete row block in VMEM.
    """
    from vllm_torchtpu import envs as _envs
    if not _envs.TPU_FUSED_QK_NORM_ROPE:
        return
    from vllm.model_executor.models import qwen3_next as _qn

    from vllm_torchtpu.layers.vllm import fused_qk_norm_rope_tpu as _tpu
    if getattr(_qn, "_tpu_fused_qk_norm_rope_patch", False):
        return
    row_cap = 64
    cls = _qn.Qwen3NextAttention
    original_init = cls.__init__
    original_project = cls._project_qkv_gate

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        model_config = kwargs.get("model_config",
                                  args[1] if len(args) > 1 else None)
        mm_config = getattr(model_config, "multimodal_config",
                            None) if model_config else None
        text_only = mm_config is None or getattr(mm_config,
                                                 "language_model_only", False)
        self._tpu_fused_qk = bool(
            getattr(self, "attn_output_gate", False)
            and getattr(self.rotary_emb, "is_neox_style", False) and text_only)
        if self._tpu_fused_qk:
            _tpu.prebuild(self.num_heads, self.num_kv_heads, self.head_dim,
                          self.rotary_emb.rotary_dim,
                          self.q_norm.variance_epsilon)

    def _norm_weight_plus_one(norm):
        hoisted = getattr(norm, "_tpu_weight_plus_one", None)
        return hoisted if hoisted is not None else norm.weight.float() + 1.0

    def _project_qkv_gate(self, qkv, positions):
        if getattr(self, "_tpu_fused_qk", False) and qkv.shape[0] <= row_cap:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            pos = positions[0] if positions.ndim == 2 else positions
            # the norms' (1 + w): the hoisted f32 buffer when TPU_NORM_WEIGHT_HOIST installed one (the row
            # buckets above the cap run the norm modules, which read the same buffer, so every trace lifts the
            # same inputs), else computed here as upstream does
            q, k, gate = _tpu.fused_qk_rmsnorm_rope_gate(
                q_gate, k, _norm_weight_plus_one(self.q_norm),
                _norm_weight_plus_one(self.k_norm),
                self.rotary_emb.cos_sin_cache, pos,
                self.q_norm.variance_epsilon, self.num_heads,
                self.num_kv_heads, self.head_dim, self.rotary_emb.rotary_dim)
            return q, k, v, gate
        return original_project(self, qkv, positions)

    cls.__init__ = __init__
    cls._project_qkv_gate = _project_qkv_gate
    _qn._tpu_fused_qk_norm_rope_patch = True


def _patch_expert_map_host_lookup() -> None:
    """Serve expert-map lookups from a host-side copy of the expert map.

    ``ExpertMapManager.map_global_to_local`` reads the device-resident
    expert map with ``.item()`` once per (expert, shard) tensor during MoE
    weight loading. On TPU every ``.item()`` is a blocking sync that
    flushes the deferred graph, so loading dispatches one tiny device
    program per expert copy. Index a
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


def _patch_vllm_compile_all_ranges() -> None:
    """Patch vllm compile all ranges to support compilation rotation.

    Compilation rotation allows vllm to compile multiple sizes at the same time,
    which greatly reduce the vllm startup time.
    """
    import os
    from collections import deque

    from vllm.compilation.piecewise_backend import (PiecewiseBackend,
                                                    create_concrete_args,
                                                    get_fake_args_from_graph)

    def compile_all_ranges(self) -> None:
        """Compile all range entries for this piecewise subgraph up front."""
        assert self.graph is not None, (
            "Cannot compile without a graph. "
            "When loading from cache/AOT artifacts, "
            "compile_all_ranges should not be called.")

        local_rank = int(os.environ["LOCAL_RANK"])
        range_entry_list = list(self.range_entries.values())

        range_entry_len = len(range_entry_list)

        expanded_indexes = list(range((range_entry_len + 7) // 8 * 8))
        d = deque(sorted(expanded_indexes, key=lambda x: (x % 8, x)))
        d.rotate(-local_rank * (len(d) // 8))
        indexes = list(d)

        for i in indexes:
            if i >= len(range_entry_list):
                continue

            range_entry = range_entry_list[i]

            if range_entry.compiled:
                continue

            self._log_compile_start(range_entry.compile_range)

            if range_entry.compile_range.is_single_size():
                args_list = create_concrete_args(
                    self.graph, range_entry.compile_range.start)
            else:
                args_list = get_fake_args_from_graph(self.graph)

            range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args_list,
                self.vllm_backend.inductor_config,
                self.compilation_config,
                compile_range=range_entry.compile_range,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                is_encoder=self.vllm_backend.is_encoder,
            )

            range_entry.compiled = True

    if envs.TPU_PARALLEL_PRECOMPILE:
        PiecewiseBackend.compile_all_ranges = compile_all_ranges
        logger.info("Applied TPU patch: compile_all_ranges.")


def _patch_disable_sequence_parallel_moe() -> None:
    """Disable vLLM's sequence-parallel MoE on TPU.

    The sequence-parallel path is tied to vLLM all2all MoE backends. The TPU
    communicator handles EP with unsplit token buffers.
    """
    from vllm.config.parallel import ParallelConfig

    if getattr(ParallelConfig, "_tpu_no_sp_moe_patch", False):
        return

    if envs.TPU_SP_MIN_ROWS > 0:
        # The row-gated form (layers/vllm/sequence_parallel,
        # _patch_qwen3_next_sequence_parallel_by_rows) combines expert and
        # tensor parallelism without data parallelism. It partitions prompt
        # rows to avoid repeating norms, routing, and shared-expert work.
        def use_sequence_parallel_moe(self) -> bool:
            return bool(self.enable_expert_parallel
                        and self.tensor_parallel_size > 1
                        and self.data_parallel_size == 1)

        ParallelConfig.use_sequence_parallel_moe = property(
            use_sequence_parallel_moe)
        ParallelConfig._tpu_no_sp_moe_patch = True
        logger.info(
            "Applied TPU patch: sequence-parallel MoE for steps of at "
            "least %d rows.", envs.TPU_SP_MIN_ROWS)
        return

    ParallelConfig.use_sequence_parallel_moe = property(lambda self: False)
    ParallelConfig._tpu_no_sp_moe_patch = True
    logger.info("Applied TPU patch: disable sequence-parallel MoE.")


def _patch_qwen3_next_sequence_parallel_by_rows() -> None:
    """Sequence parallelism for the tensor-parallel prefill, by row count.

    Upstream's ``use_attn_reduce_scatter_for_moe`` form of the Qwen3-Next
    decoder layer, taken only by the steps with at least TPU_SP_MIN_ROWS rows;
    the smaller steps keep the all-reduce form (two collectives per layer,
    latency-bound at decode). The residual's chunk is an index-select through
    the layer's rank buffer so every rank compiles one program, and the steps
    above the threshold are the token buckets, multiples of the tensor-parallel
    width, so no row padding is needed.
    """
    from vllm_torchtpu.layers.vllm import sequence_parallel as _sp
    if not _sp.enabled():
        return
    import torch

    from vllm_torchtpu.layers.vllm import linear_common as _lc
    if _sp.fp8_gather():
        _lc.prebuild_prequant_ops()
    from vllm.distributed import (get_tensor_model_parallel_rank,
                                  get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_gather,
                                  tensor_model_parallel_reduce_scatter)
    from vllm.model_executor.models import qwen3_next as _qn

    from vllm_torchtpu.layers.vllm import small_all_reduce_op as _sar

    cls = _qn.Qwen3NextDecoderLayer
    if getattr(cls, "_tpu_sp_by_rows_patch", False):
        return

    attention_cls = _qn.Qwen3NextAttention
    original_attention_forward = attention_cls.forward

    def attention_forward(self, positions, hidden_states, prequant=None):
        # Sequence parallelism supplies this interface independently of the
        # optional Q/K fusion. Ordinary rows retain the upstream forward.
        if prequant is None:
            return original_attention_forward(self, positions, hidden_states)
        proj = self.qkv_proj
        bias = None if proj.skip_bias_add else proj.bias
        qkv = proj.quant_method.apply_prequant(proj, prequant[0], prequant[1],
                                               bias)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    attention_cls.forward = attention_forward

    def with_rank_buffer(original_init):

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # this rank's index as device data, so the residual chunk's
            # offset is an operand and not a per-rank constant in the
            # compiled program
            device = self.input_layernorm.weight.device
            self._tpu_sp_rank = torch.tensor(
                [get_tensor_model_parallel_rank()],
                dtype=torch.int32,
                device=device)

        __init__._tpu_sp_rank_init = True
        return __init__

    # Qwen3.5's layer defines its own __init__ and skips this class's, so
    # every layer class with an __init__ of its own gets the buffer.
    from vllm.model_executor.models import qwen3_5 as _q35
    layer_classes = [cls, _q35.Qwen3_5DecoderLayer]

    def forward(self, hidden_states, residual, positions=None, **kwargs):
        # The decision comes from a tensor shape, which leaves a shape guard
        # behind: the compile flow traces the model once with the token count
        # dynamic and re-traces only the buckets a trace's guards refuse
        # (compilation/shape_variants); a Python value would be frozen at
        # the first trace. The runner set the same decision for the
        # communicator before calling the model. Within one trace every
        # layer takes the same form: the first layer's input is the full
        # rows (no residual yet), every later layer's input is this rank's
        # slice.
        sp = False
        if (self.use_attn_reduce_scatter_for_moe
                and positions.shape[-1] >= _sp.min_rows()):
            sp = True
        input_is_sequence_parallel = sp and residual is not None
        # Every trace group lifts the rank buffer in every layer: the compile
        # flow hands a bucket's executable between traces and refuses a
        # differing set of lifted inputs (the residual slice below uses it
        # only in the first layer of the sequence-parallel trace). A zero
        # rides on the positions, which the compiler folds.
        positions = positions + (self._tpu_sp_rank *
                                 0)[0] + _sar.zero_from_rank()

        prequant = None
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        if input_is_sequence_parallel and _sp.fp8_gather():
            # the rows reach the projections quantised: this rank quantises
            # its own rows once (the rows and scales the projections would
            # compute), the all-gather carries fp8 rows and float32 scales
            # instead of bf16 rows, and the projections take them as they
            # are. The gathered fp8 rows stand in for hidden_states, of
            # which the attention modules read only the row count.
            x_q, x_s = _lc.quantize_rows(hidden_states)
            x_q = tensor_model_parallel_all_gather(x_q, 0)
            x_s = tensor_model_parallel_all_gather(x_s, 0)
            prequant = (x_q, x_s)
            hidden_states = x_q
        elif input_is_sequence_parallel:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states,
                                             prequant=prequant)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(hidden_states=hidden_states,
                                           positions=positions,
                                           prequant=prequant)
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1)

        if self.use_attn_reduce_scatter_for_moe:
            if sp:
                # the steps above the row threshold are token buckets, whole
                # multiples of the tensor-parallel width
                hidden_states = tensor_model_parallel_reduce_scatter(
                    hidden_states, 0)
                if not input_is_sequence_parallel:
                    # this rank's slice of the residual: the rank rides in as
                    # data, so every rank compiles one program
                    tp = get_tensor_model_parallel_world_size()
                    residual = residual.reshape(
                        tp, -1, residual.shape[-1]).index_select(
                            0, self._tpu_sp_rank).squeeze(0)
            else:
                # the projections were built without their all-reduce
                hidden_states = _sar.all_reduce_tp(hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        if self.use_attn_reduce_scatter_for_moe:
            # own rows above the threshold, replicated rows below: the
            # communicator gathers and combines accordingly, and the block
            # neither chunks nor gathers on its own
            hidden_states = self.mlp(hidden_states,
                                     already_sequence_parallel=True)
        else:
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
            else:
                assert len(hidden_states.shape) == len(
                    self.ffn_layer_scale.shape)
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1)

        return hidden_states, residual

    for layer_cls in layer_classes:
        if "__init__" in vars(layer_cls):
            layer_cls.__init__ = with_rank_buffer(layer_cls.__init__)
    cls.forward = forward
    cls._tpu_sp_by_rows_patch = True
    logger.info("Applied TPU patch: sequence parallelism by rows (>= %d).",
                _sp.min_rows())


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
        return _orig(vllm_config, local_rank, rank, *args, **kwargs)

    WorkerProc.make_worker_process = staticmethod(_wrapped)
    WorkerProc._tpu_global_rank_env_patch = True
    logger.info("Applied TPU patch: MultiprocExecutor worker binding env.")


def _run_engine_core_with_tpu_patches(*args, **kwargs):
    from vllm_torchtpu import patch_registry

    patch_registry.apply("engine_core")
    from vllm_torchtpu.core.pp_chunk_scheduler import \
        patch_engine_core_for_pp_chunks
    from vllm_torchtpu.distributed.pp_push import patch_executor_for_pp_wave
    patch_executor_for_pp_wave(kwargs.get("vllm_config"))
    patch_engine_core_for_pp_chunks(kwargs.get("vllm_config"))

    from vllm.v1.engine.core import EngineCoreProc

    original_run = EngineCoreProc._tpu_original_run_engine_core
    return original_run(*args, **kwargs)


def _patch_vllm_offloading_config_build() -> None:
    """Adapt vLLM's OffloadingConfig construction to TPU PCP.

    ``build_offloading_config`` scales each group's ``tokens_per_block`` by
    DCP only (it records ``pcp_size`` in the parallel config but never
    folds PCP into block spans). TorchTPU PCP scheduler blocks
    span all PCP ranks (see ``_patch_vllm_hybrid_pcp_block_sizes``), so
    ``tokens_per_block`` must scale by PCP as well and ``tokens_per_hash``
    must come from the runtime (PCP-patched) resolve; otherwise the
    offloading manager would track sub-scheduler-block chunks, and on PCP
    hybrid configs ``build_offloading_config`` trips its own
    ``tokens_per_block % tokens_per_hash`` assert (the PCP-patched resolve
    returns logical hash sizes while the group sizes stay rank-local).
    """
    import sys
    from dataclasses import replace

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import \
        config as offloading_config_module

    if getattr(offloading_config_module, "_tpu_offloading_config_patch",
               False):
        return

    original_build = offloading_config_module.build_offloading_config

    def build_offloading_config_tpu(vllm_config, kv_cache_config):
        pcp = vllm_config.parallel_config.prefill_context_parallel_size
        if pcp <= 1:
            return original_build(vllm_config, kv_cache_config)

        # Run the original against rank-local block sizes (the
        # unpatched resolve, so its tokens_per_block % tokens_per_hash
        # assert compares rank-local to rank-local), then scale each
        # group's tokens_per_block to the logical all-PCP-rank span.
        from vllm.v1.core import kv_cache_utils
        unpatched_resolve = getattr(
            kv_cache_utils, "_tpu_original_resolve_kv_cache_block_sizes",
            kv_cache_utils.resolve_kv_cache_block_sizes)
        saved_resolve = (offloading_config_module.resolve_kv_cache_block_sizes)
        offloading_config_module.resolve_kv_cache_block_sizes = (
            unpatched_resolve)
        try:
            config = original_build(vllm_config, kv_cache_config)
        finally:
            offloading_config_module.resolve_kv_cache_block_sizes = (
                saved_resolve)
        # tokens_per_hash must match the granularity the scheduler
        # actually hashes Request.block_hashes at -- the runtime
        # (PCP-patched) resolve: logical under hybrid PCP, rank-local
        # otherwise.
        _, tokens_per_hash = kv_cache_utils.resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config)
        config = replace(
            config,
            groups=tuple(
                replace(group, tokens_per_block=group.tokens_per_block * pcp)
                for group in config.groups),
            cache=replace(config.cache, tokens_per_hash=tokens_per_hash))
        for group in config.groups:
            assert group.tokens_per_block % tokens_per_hash == 0, (
                f"tokens_per_block={group.tokens_per_block} not "
                f"divisible by tokens_per_hash={tokens_per_hash} after "
                f"PCP scaling (pcp={pcp})")
        return config

    offloading_config_module.build_offloading_config = (
        build_offloading_config_tpu)
    offloading_config_module._tpu_offloading_config_patch = True

    # OffloadingConnector imports the function directly; rebind if the
    # module is already loaded.
    connector_module = sys.modules.get(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector")
    if connector_module is not None:
        connector_module.build_offloading_config = build_offloading_config_tpu

    logger.info("Applied TPU patch: PCP-aware OffloadingConfig build.")


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
        elif cache_config.prefix_match_unit is None:
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
                                                 max_in_flight_tokens,
                                                 use_eagle,
                                                 enable_caching,
                                                 enable_kv_cache_events,
                                                 dcp_world_size,
                                                 pcp_world_size,
                                                 scheduler_block_size,
                                                 hash_block_size,
                                                 metrics_collector=None):
            groups = kv_cache_config.kv_cache_groups
            logical_pcp_world_size = pcp_world_size
            if (pcp_world_size == 1 and _is_attention_mamba_hybrid(groups)):
                # vLLM 0.26's Scheduler already folds PCP into
                # scheduler_block_size, but hard-codes pcp_world_size=1 when
                # constructing KVCacheManager. Recover the folded PCP factor
                # from the logical scheduler block and the rank-local specs.
                physical_scheduler_block_size = math.lcm(
                    *(_effective_block_size(group.kv_cache_spec, 1)
                      for group in groups))
                if scheduler_block_size % physical_scheduler_block_size != 0:
                    raise ValueError(
                        "Logical scheduler block size must be divisible by "
                        "the physical hybrid KV cache block size. Got "
                        f"scheduler_block_size={scheduler_block_size}, "
                        "physical_scheduler_block_size="
                        f"{physical_scheduler_block_size}.")
                logical_pcp_world_size = (scheduler_block_size //
                                          physical_scheduler_block_size)
                if logical_pcp_world_size > 1:
                    logger.info(
                        "Inferred PCP world size %d from logical scheduler "
                        "block size %d and physical block size %d.",
                        logical_pcp_world_size, scheduler_block_size,
                        physical_scheduler_block_size)

            if (logical_pcp_world_size > 1
                    and _is_attention_mamba_hybrid(groups)):
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
                                    group.kv_cache_spec,
                                    logical_pcp_world_size))
                        for group in kv_cache_config.kv_cache_groups
                    ])
                pcp_world_size = 1

            original_hybrid_init(
                self,
                kv_cache_config,
                max_model_len,
                max_in_flight_tokens,
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

    # Upstream _promote_local_kv_cache_specs asserts that all workers share a
    # uniform KV cache spec across groups, raising ValueError when encountering
    # heterogeneous layer configurations (e.g. hybrid attention/Mamba models,
    # pure linear state layers, or disaggregated PCP configurations with
    # per-group block sizing). We patch it to catch ValueError and return an
    # empty dict so initialization falls back cleanly to the TPU hybrid/PCP
    # cache coordinator without crashing.
    original_promote = getattr(kv_cache_utils, "_promote_local_kv_cache_specs",
                               None)
    if original_promote is not None and not getattr(
            original_promote, "_tpu_hybrid_promote_patch", False):

        def patched_promote(specs):
            try:
                return original_promote(specs)
            except ValueError:
                return {}

        patched_promote._tpu_hybrid_promote_patch = True
        kv_cache_utils._promote_local_kv_cache_specs = patched_promote
        for module_name in ("vllm.v1.engine.core",
                            "vllm.v1.core.kv_cache_utils"):
            mod = sys.modules.get(module_name)
            if mod is not None and hasattr(mod,
                                           "_promote_local_kv_cache_specs"):
                setattr(mod, "_promote_local_kv_cache_specs", patched_promote)

    if not getattr(EngineCoreProc, "_tpu_engine_core_patch_wrapper", False):
        EngineCoreProc._tpu_original_run_engine_core = (
            EngineCoreProc.run_engine_core)
        EngineCoreProc.run_engine_core = staticmethod(
            _run_engine_core_with_tpu_patches)
        EngineCoreProc._tpu_engine_core_patch_wrapper = True

    if not already_patched:
        logger.info("Applied TPU patch: hybrid full-attention + Mamba PCP "
                    "block sizes.")


def _patch_vllm_merge_multimodal_embeddings() -> None:
    """Patch vLLM's _merge_multimodal_embeddings to use static indexing on TPU.

    PyTorch boolean indexing (inputs[mask] = values) internally lowers to
    `nonzero()` + `index_put_`, which creates dynamic coordinate shapes
    parameterized by the number of True entries. On TPU, this causes
    `tt_jit_index_put__nonzero` to recompile on every request with a different
    image token count. Static prefix-sum indexing and `torch.where` avoid
    the dynamic `nonzero` shapes and the resulting recompilation.
    """
    import sys

    import torch
    import vllm.model_executor.models.utils as vllm_utils
    from vllm.multimodal import NestedTensors

    if not getattr(vllm_utils, "_tpu_static_merge_mm_patch", False):
        orig_merge = vllm_utils._merge_multimodal_embeddings

        def static_merge_multimodal_embeddings(
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: NestedTensors,
            is_multimodal: torch.Tensor,
        ) -> torch.Tensor:
            if len(multimodal_embeddings) == 0:
                return inputs_embeds

            mm_embeds_flat = vllm_utils._flatten_embeddings(
                multimodal_embeddings)
            if mm_embeds_flat.numel() == 0:
                return inputs_embeds

            mm_embeds_flat = mm_embeds_flat.to(dtype=inputs_embeds.dtype,
                                               device=inputs_embeds.device)

            idx = torch.cumsum(is_multimodal, dim=0, dtype=torch.int32) - 1
            idx = torch.clamp(idx, 0, mm_embeds_flat.shape[0] - 1)
            gathered = mm_embeds_flat[idx]
            condition = is_multimodal.unsqueeze(-1)
            return torch.where(condition, gathered, inputs_embeds)

        vllm_utils._original_merge_multimodal_embeddings = orig_merge
        vllm_utils._merge_multimodal_embeddings = static_merge_multimodal_embeddings
        vllm_utils._tpu_static_merge_mm_patch = True
        logger.info("Applied TPU patch: static _merge_multimodal_embeddings")

    # Re-run the sweep across imported model modules on every invocation so that
    # models imported after the first call also have their bindings updated.
    patched_fn = vllm_utils._merge_multimodal_embeddings
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("vllm.model_executor.models") and hasattr(
                mod, "_merge_multimodal_embeddings"):
            if getattr(mod, "_merge_multimodal_embeddings") is not patched_fn:
                setattr(mod, "_merge_multimodal_embeddings", patched_fn)


def _patch_vllm_reset_compile_wrapper() -> None:
    """Reset the compile wrapper on every decoder layer, not just the model.

    `support_torch_compile` makes each decorated module its own compile
    wrapper, so compiling per layer puts a wrapper on all 40+ layers. vLLM's
    `reset_compile_wrapper` only looks at the object it is handed and that
    object's `.model` attribute; finding no wrapper there, it returns and
    resets nothing.
    """
    import torch
    import vllm.compilation.wrapper as wrapper_mod
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    original_reset = wrapper_mod.reset_compile_wrapper
    if getattr(original_reset, "_tpu_reset_compile_wrapper_patch", False):
        return

    def reset_compile_wrapper_tpu(model: torch.nn.Module) -> None:
        if model is None:
            return
        if isinstance(model, TorchCompileWithNoGuardsWrapper):
            original_reset(model)
            return

        found_wrapper = False
        for m in model.modules():
            if isinstance(m, TorchCompileWithNoGuardsWrapper):
                found_wrapper = True
                original_reset(m)

        if not found_wrapper:
            original_reset(model)

    reset_compile_wrapper_tpu._tpu_reset_compile_wrapper_patch = True
    wrapper_mod.reset_compile_wrapper = reset_compile_wrapper_tpu
    logger.info(
        "Applied TPU patch: recursive reset_compile_wrapper for per-layer compilation."
    )


def _patch_vllm_piecewise_backend() -> None:
    """Pick the compiled token-count bucket from a tensor's leading dimension.

    A graph is compiled once per bucket of token counts, and vLLM decides
    which one a call needs by reading an argument that is a `SymInt`. A layer
    whose arguments are all tensors has no such argument, so vLLM assumes only
    one bucket was compiled and asserts when several were.
    """
    import torch
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    original_call = PiecewiseBackend.__call__
    if getattr(original_call, "_tpu_piecewise_backend_patch", False):
        return

    def patched_call(self, *args, **kwargs):
        if self.sym_shape_indices:
            runtime_shape = args[self.sym_shape_indices[0]]
            range_entry = self._find_range_for_shape(runtime_shape)
            assert range_entry is not None, (
                f"Shape: {runtime_shape} out of considered ranges: "
                f"{self.compile_ranges}")
        elif len(self.range_entries) > 1 or (self.compile_sizes
                                             and len(self.compile_sizes) > 1):
            # Dynamic token count from tensor dimension 0
            first_tensor = next(
                (x for x in args if isinstance(x, torch.Tensor)), None)
            if first_tensor is not None:
                runtime_shape = first_tensor.shape[0]
                range_entry = self._find_range_for_shape(runtime_shape)
                assert range_entry is not None, (
                    f"Shape: {runtime_shape} out of considered ranges: "
                    f"{self.compile_ranges}")
            else:
                compiled_entries = [
                    re for re in self.range_entries.values() if re.compiled
                ]
                assert len(compiled_entries) == 1, (
                    f"Expected exactly one compiled range_entry for static shape "
                    f"compilation, but found {len(compiled_entries)}")
                range_entry = compiled_entries[0]
        else:
            compiled_entries = [
                re for re in self.range_entries.values() if re.compiled
            ]
            assert len(compiled_entries) == 1, (
                f"Expected exactly one compiled range_entry for static shape "
                f"compilation, but found {len(compiled_entries)}")
            range_entry = compiled_entries[0]

        assert range_entry.compiled, (
            "All ranges should be compiled or loaded up front in "
            "PiecewiseBackend.__init__. "
            f"range_entry={range_entry.compile_range}")
        return range_entry.runnable(*args, **kwargs)

    patched_call._tpu_piecewise_backend_patch = True
    PiecewiseBackend.__call__ = patched_call
    logger.info(
        "Applied TPU patch: PiecewiseBackend tensor-dim-0 bucket dispatch.")


def _patch_vllm_compile_prefix_isolation() -> None:
    """Give each support_torch_compile'd instance its own cache identity.

    vLLM's shared `compile_prefix=""` is only safe when all instances of a
    class close over identically-shaped buffers; DSv4's KV cache overlay
    breaks that, and one layer's executable can be reused by another (XLA
    e0102). Keying on self.prefix is a no-op for shape-uniform models.
    """
    import itertools

    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    original_init = TorchCompileWithNoGuardsWrapper.__init__
    if getattr(original_init, "_tpu_compile_prefix_isolation_patch", False):
        return

    counter = itertools.count()

    def patched_init(self,
                     compile_prefix: str = "",
                     is_encoder: bool = False) -> None:
        if not compile_prefix and not is_encoder:
            prefix = getattr(self, "prefix", None)
            if prefix:
                compile_prefix = prefix
            else:
                # No self.prefix: fall back to a counter. Stable within a
                # process but not across restarts, so the cache won't persist.
                compile_prefix = f"_tpu_instance_{next(counter)}"
                logger.warning(
                    "compile_prefix isolation: %s has no self.prefix; using "
                    "non-deterministic fallback %s (its on-disk compile "
                    "cache will not persist across restarts).",
                    type(self).__name__, compile_prefix)
        original_init(self,
                      compile_prefix=compile_prefix,
                      is_encoder=is_encoder)

    patched_init._tpu_compile_prefix_isolation_patch = True
    TorchCompileWithNoGuardsWrapper.__init__ = patched_init
    logger.info(
        "Applied TPU patch: per-instance compile-artifact cache isolation "
        "(compile_prefix keyed off self.prefix).")


def _patch_dflash_bypass_v2_runner_check() -> None:
    """Bypass V2 model runner check in DFlash for TPU execution.

    Upstream ``qwen3_dflash._resolve_layer_attention`` enforces
    ``use_v2_model_runner`` for mixed SWA/full attention DFlash drafters.
    TPU uses TPUModelRunner which handles mixed attention without Triton V2 runner.
    """
    try:
        from vllm.model_executor.models import qwen3_dflash
    except ImportError:
        return

    if getattr(qwen3_dflash, "_tpu_bypass_v2_runner_patch", False):
        return

    original_resolve = qwen3_dflash._resolve_layer_attention
    _FLAG = "use_v2_model_runner"

    def patched_resolve(config, layer_idx: int):
        # Set the flag only for the upstream call, then restore the class
        # attribute. Explicit lookup raises immediately if the upstream
        # attribute moves or is renamed.
        from vllm.config import get_current_vllm_config
        cfg_cls = type(get_current_vllm_config())
        if not hasattr(cfg_cls, _FLAG):
            raise RuntimeError(
                f"VllmConfig no longer defines {_FLAG!r}; the TPU DFlash "
                "bypass needs updating for this vLLM version.")

        # Restore exactly what was there: if the flag is inherited rather than
        # defined on this class, putting it back with setattr would leave a
        # permanent shadow on the subclass.
        had_own = _FLAG in cfg_cls.__dict__
        original_flag = cfg_cls.__dict__.get(_FLAG)
        setattr(cfg_cls, _FLAG, property(lambda self: True))
        try:
            return original_resolve(config, layer_idx)
        finally:
            if had_own:
                setattr(cfg_cls, _FLAG, original_flag)
            else:
                delattr(cfg_cls, _FLAG)

    qwen3_dflash._resolve_layer_attention = patched_resolve
    qwen3_dflash._tpu_bypass_v2_runner_patch = True
    logger.info("Applied TPU patch: bypass DFlash V2 runner check on TPU.")


def _patch_vllm_config_triton_tpu() -> None:
    """Bypass V2 model runner Triton check on TPU.

    The config forces use_v2_model_runner=True for hybrid DFlash drafters.
    When it validates V2, it crashes on TPU because HAS_TRITON is False.
    TPU uses TPUModelRunner instead, so we can safely bypass this.
    """
    from vllm.config.vllm import VllmConfig

    if getattr(VllmConfig, "_tpu_triton_patch_applied", False):
        return

    original_validate = VllmConfig._validate_v2_model_runner

    def patched_validate(self):
        if getattr(self, "device_config",
                   None) and self.device_config.device_type == "tpu":
            return
        return original_validate(self)

    VllmConfig._validate_v2_model_runner = patched_validate
    VllmConfig._tpu_triton_patch_applied = True
    logger.info("Applied TPU patch: bypass V2 model runner Triton check.")


def _patch_vllm_force_v1_runner_tpu() -> None:
    """Keep ``use_v2_model_runner`` False for DSpark on TPU.

    Upstream forces the flag True for dspark (and for hybrid DFlash drafts
    via ``_dflash_needs_multi_kv_group``), because those methods are only
    implemented by the V2 *GPU* model runner. The flag is not just a runner
    selector: the scheduler, async scheduler, and input processor all branch
    on it and emit V2-shaped request data (resumed requests folded into
    scheduled_new_reqs, no prev_step_scheduled_req_ids, ...). TPUModelRunner
    subclasses the V1 GPUModelRunner and expects V1-shaped output, so the
    flag must stay False for DSpark. An explicit VLLM_USE_V2_MODEL_RUNNER env
    setting still wins, matching upstream's own precedence.
    """
    from vllm.config.vllm import VllmConfig

    if getattr(VllmConfig, "_tpu_force_v1_runner_patch", False):
        return

    original_prop = VllmConfig.use_v2_model_runner
    if not isinstance(original_prop, property):
        logger.warning(
            "VllmConfig.use_v2_model_runner is no longer a property; skipping "
            "the TPU V1-runner patch. DSpark on TPU may misbehave until this "
            "is updated for the current vLLM version.")
        return

    def patched_use_v2(self):
        import vllm.envs as envs
        if envs.VLLM_USE_V2_MODEL_RUNNER is not None:
            return original_prop.fget(self)
        device_config = getattr(self, "device_config", None)
        spec_config = getattr(self, "speculative_config", None)
        if (device_config is not None and device_config.device_type == "tpu"
                and spec_config is not None
                and getattr(spec_config, "method", None) == "dspark"):
            return False
        return original_prop.fget(self)

    VllmConfig.use_v2_model_runner = property(patched_use_v2)
    VllmConfig._tpu_force_v1_runner_patch = True
    logger.info("Applied TPU patch: force V1 model runner semantics on TPU.")


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


def _patch_vllm_block_pool_lifo_free() -> None:
    """Reuse freed blocks in LIFO order when prefix caching is disabled.

    FIFO reuse cycles through the full KV block-ID space under sustained
    load. On TPU v7x, indexed SparseCore operations on the paged-KV path
    can halt with E0200 RuntimeUnexpectedCoreHalt under that access pattern.
    LIFO reuse keeps a smaller set of block IDs active. Prefix caching
    retains its own reuse order.
    """
    from vllm.v1.core.block_pool import BlockPool

    if BlockPool.__dict__.get("_tpu_lifo_free_patch", False):
        return

    def free_blocks(self, ordered_blocks) -> None:
        blocks_with_hash = []
        blocks_without_hash = []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                if block.block_hash is None:
                    blocks_without_hash.append(block)
                else:
                    blocks_with_hash.append(block)
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    BlockPool.free_blocks = free_blocks
    BlockPool._tpu_lifo_free_patch = True
    logger.info("Applied LIFO free-block patch to BlockPool")


def _patch_vllm_vocab_parallel_embedding() -> None:
    """Register shard index bounds as module buffers in VocabParallelEmbedding.

    Prevents TorchTPU from baking rank-specific integer literals into the MLIR
    graph as scalar constants, ensuring identical CompilationCacheKeys across
    all TP ranks.
    """
    import torch
    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    if getattr(vpe.VocabParallelEmbedding,
               "_tpu_vocab_parallel_embedding_patch", False):
        return

    original_init = vpe.VocabParallelEmbedding.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "org_vocab_start_index"):
            self.register_buffer(
                "org_vocab_start_index",
                torch.tensor(self.shard_indices.org_vocab_start_index,
                             dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "org_vocab_end_index",
                torch.tensor(self.shard_indices.org_vocab_end_index,
                             dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "num_org_vocab_padding",
                torch.tensor(self.shard_indices.num_org_vocab_padding,
                             dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "added_vocab_start_index",
                torch.tensor(self.shard_indices.added_vocab_start_index,
                             dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "added_vocab_end_index",
                torch.tensor(self.shard_indices.added_vocab_end_index,
                             dtype=torch.int64),
                persistent=False,
            )

    def patched_forward(self, input_):
        if self.tp_size > 1:
            masked_input, input_mask = vpe.get_masked_input_and_mask(
                input_,
                getattr(self, "org_vocab_start_index",
                        self.shard_indices.org_vocab_start_index),
                getattr(self, "org_vocab_end_index",
                        self.shard_indices.org_vocab_end_index),
                getattr(self, "num_org_vocab_padding",
                        self.shard_indices.num_org_vocab_padding),
                getattr(self, "added_vocab_start_index",
                        self.shard_indices.added_vocab_start_index),
                getattr(self, "added_vocab_end_index",
                        self.shard_indices.added_vocab_end_index),
            )
        else:
            masked_input = input_
        output_parallel = self.quant_method.embedding(self,
                                                      masked_input.long())
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            from vllm.distributed import tensor_model_parallel_all_reduce
            return tensor_model_parallel_all_reduce(output_parallel)
        return output_parallel

    def patched_get_masked_input_and_mask(
        input_: torch.Tensor,
        org_vocab_start_index: int | torch.Tensor,
        org_vocab_end_index: int | torch.Tensor,
        num_org_vocab_padding: int | torch.Tensor,
        added_vocab_start_index: int | torch.Tensor,
        added_vocab_end_index: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = input_.device
        dtype = input_.dtype

        def _to_tensor(val: int | torch.Tensor) -> torch.Tensor:
            if isinstance(val, torch.Tensor):
                return val.to(device=device)
            return torch.tensor(val, dtype=dtype, device=device)

        start = _to_tensor(org_vocab_start_index)
        end = _to_tensor(org_vocab_end_index)
        num_padding = _to_tensor(num_org_vocab_padding)
        added_start = _to_tensor(added_vocab_start_index)
        added_end = _to_tensor(added_vocab_end_index)

        org_vocab_mask = (input_ >= start) & (input_ < end)
        added_vocab_mask = (input_ >= added_start) & (input_ < added_end)
        added_offset = (added_start - (end - start) - num_padding)
        valid_offset = (start * org_vocab_mask) + (added_offset *
                                                   added_vocab_mask)
        vocab_mask = org_vocab_mask | added_vocab_mask
        input_ = vocab_mask * (input_ - valid_offset)
        return input_, ~vocab_mask

    vpe.get_masked_input_and_mask = patched_get_masked_input_and_mask
    vpe.VocabParallelEmbedding.__init__ = patched_init
    vpe.VocabParallelEmbedding.forward = patched_forward
    vpe.VocabParallelEmbedding._tpu_vocab_parallel_embedding_patch = True
    logger.info(
        "Applied TPU patch: register shard index bounds as buffers in VocabParallelEmbedding"
    )
