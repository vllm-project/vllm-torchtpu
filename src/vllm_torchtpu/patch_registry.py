# SPDX-License-Identifier: Apache-2.0
"""Applies ordered, process-local patches at TPU lifecycle boundaries.

Targets are imported lazily so the import stage can run after environment
setup without loading TorchTPU or resolving vLLM's platform prematurely.
Callbacks normally return None, including for permanent no-ops. Only callbacks
that need another attempt at a later lifecycle boundary return False. Completed
callbacks are remembered across stages. Spawned processes start with an empty
registry; forked processes inherit patches and their application records.
"""

import importlib
import logging
import os
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm.config import ModelConfig

Stage = Literal[
    "import", "platform_activation", "engine_core", "worker_init", "model_load"
]


@dataclass(frozen=True)
class Patch:
    """Declares a patch's lifecycle and optional binding refresh."""

    target: str
    stages: tuple[Stage, ...]
    model_config: bool = False
    refresh: str | None = None


# Processes can enter through different lifecycle boundaries. _ACTIVE covers
# platform activation and worker initialization, since spawned workers do not
# inherit the parent's patches.
_ACTIVE: tuple[Stage, ...] = ("platform_activation", "worker_init")

# _ENGINE includes both stages in _ACTIVE and adds engine-core startup.
# These patches must also be installed in the engine-core process before
# scheduler and KV connector construction, because patches installed in workers
# do not affect that process.
_ENGINE: tuple[Stage, ...] = (*_ACTIVE, "engine_core")

# Tuple order is application order, including when several stages share a patch.
PATCHES = (
    # Environment setup and custom-op registration.
    Patch("vllm_torchtpu.env_override:_patch_jax_pallas_fori_lowering", ("import",)),
    # TODO(https://github.com/vllm-project/vllm-torchtpu/pull/1182): Remove
    # this patch once the pinned TorchTPU supports tuple-axis PartitionSpecs
    # in both get_global_shape and get_local_shape.
    Patch(
        "vllm_torchtpu.distributed.pallas_shapes:install_pallas_shape_conversion",
        _ACTIVE,
    ),
    Patch("vllm_torchtpu.layers.adapter.custom_ops:_register_custom_ops", _ACTIVE),
    # Compilation and cache keys.
    Patch("vllm_torchtpu:_patch_vllm_aot_compile_cache_key", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_compile_all_ranges", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_compile_prefix_isolation", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_config_hash_ignore_diagnostics", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_disable_compile_ranges", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_piecewise_backend", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_reset_compile_wrapper", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_bucket_filter", _ACTIVE),
    # MoE routing and collectives.
    Patch("vllm_torchtpu:_patch_sequence_parallel_moe_admission", _ACTIVE),
    Patch("vllm_torchtpu:_patch_expert_map_host_lookup", _ACTIVE),
    Patch("vllm_torchtpu:_patch_moe_explicit_pcp_collectives", _ACTIVE),
    # KV cache, scheduling, and offloading.
    Patch("vllm_torchtpu:_patch_vllm_compact_mamba_streaming_input", _ACTIVE),
    Patch(
        "vllm_torchtpu.platforms.tpu_platform:_register_tpu_kv_connectors",
        _ENGINE,
    ),
    Patch(
        "vllm_torchtpu:_patch_vllm_hybrid_pcp_block_sizes",
        _ENGINE,
        refresh="vllm_torchtpu:_patch_vllm_hybrid_pcp_block_sizes",
    ),
    # After hybrid PCP: wraps the resolve it installed and rebinds its refs.
    Patch(
        "vllm_torchtpu:_patch_vllm_dcp_fine_prefix_hits",
        _ENGINE,
        refresh="vllm_torchtpu:_patch_vllm_dcp_fine_prefix_hits",
    ),
    Patch("vllm_torchtpu:_patch_vllm_mamba_split_scheduler_block_size", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_offloading_config_build", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_offload_all_ranks_kv_bytes", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_offload_fine_prefix_alignment", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_hybrid_kv_load_failure_recovery", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_block_pool_lifo_free", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_kv_cache_group_size", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_same_step_prefix_hits", _ENGINE),
    Patch(
        "vllm_torchtpu.core.eagle_block_drop:patch_disable_eagle_block_drop",
        _ENGINE,
    ),
    Patch(
        "vllm_torchtpu.core.mamba_checkpoint_retention:"
        "patch_mamba_checkpoint_retention",
        _ENGINE,
    ),
    Patch(
        "vllm_torchtpu.core.divergent_hybrid_hits:patch_divergent_hybrid_hits",
        _ENGINE,
    ),
    # Admission-path host timeline (inert unless VLLM_TPU_HOSTADMIT_PROBE).
    Patch("vllm_torchtpu.runner.host_admit_probe:patch_engine_core", ("engine_core",)),
    # Victim host-tier policy hooks (inert unless the policy is on), after the
    # LIFO free patch and retention so they wrap the final free paths.
    Patch("vllm_torchtpu.offload.victim:patch_block_pool_victims", _ENGINE),
    # Per-request prefix lookup line (inert unless VLLM_TPU_PREFIX_LOOKUP_LOG).
    Patch("vllm_torchtpu.offload.lookup_log:patch_prefix_lookup_log", _ENGINE),
    # After every patch of the KV cache manager, its groups and the block
    # pool: it checks the functions it encodes as they are installed.
    Patch("vllm_torchtpu.core.kv_decode_fast_path:patch_kv_decode_fast_path", _ENGINE),
    Patch("vllm_torchtpu.core.kv_load_pacing:patch_kv_load_pacing", _ENGINE),
    # Arrival-time prefix pins and host-tier refresh (inert unless
    # VLLM_TPU_ARRIVAL_TOUCH), around the final add_request / schedule.
    Patch("vllm_torchtpu.core.arrival_touch:patch_arrival_touch", _ENGINE),
    # Host admission costs (inert unless their flags are on).
    Patch(
        "vllm_torchtpu.core.prompt_token_array:patch_prompt_token_array",
        ("engine_core",),
    ),
    Patch("vllm_torchtpu.runner.gc_at_waits:patch_gc_at_waits", _ENGINE),
    Patch("vllm_torchtpu.runner.gc_at_waits:patch_engine_core_gc", ("engine_core",)),
    # New requests built in the engine core's waits (inert unless
    # VLLM_TPU_PREPROC_IN_WAIT): outermost preprocess_add_request wrapper.
    Patch(
        "vllm_torchtpu.core.preproc_in_wait:patch_preproc_in_wait", ("engine_core",)
    ),
    # Finished requests freed in the engine core's waits (inert unless
    # VLLM_TPU_DEFER_FIN): outermost Scheduler free / update / schedule wrappers.
    Patch("vllm_torchtpu.core.defer_fin:patch_defer_fin", ("engine_core",)),
    # Non-output ranks' step replies read one call late (inert unless
    # VLLM_TPU_FANIN_LATE): replaces MultiprocExecutor.collective_rpc.
    Patch("vllm_torchtpu.core.fanin_late:patch_fanin_late", ("engine_core",)),
    Patch("vllm_torchtpu:_patch_vllm_kimi_kda_layer_counts", _ACTIVE),
    # Process setup and runner selection.
    Patch("vllm_torchtpu:_patch_multiproc_worker_global_rank_env", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_force_v1_runner_tpu", _ACTIVE),
    Patch("vllm_torchtpu:_patch_dflash_bypass_v2_runner_check", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_config_triton_tpu", _ACTIVE),
    # Weight loading and embeddings.
    Patch(
        "vllm_torchtpu.model_loader_patches:patch_default_loader_ep_weight_filter",
        _ACTIVE,
    ),
    Patch(
        "vllm_torchtpu.model_loader_patches:patch_runai_sharded_expert_streaming",
        _ACTIVE,
    ),
    Patch(
        "vllm_torchtpu.model_loader_patches:patch_default_model_loader_page_cache",
        _ACTIVE,
    ),
    Patch("vllm_torchtpu.model_loader_patches:patch_moe_expert_write_staging", _ACTIVE),
    Patch("vllm_torchtpu.model_loader_patches:patch_expert_mapping_cache", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_vocab_parallel_embedding", _ACTIVE),
    Patch(
        "vllm_torchtpu:_patch_vllm_merge_multimodal_embeddings",
        _ENGINE,
        refresh="vllm_torchtpu:_patch_vllm_merge_multimodal_embeddings",
    ),
    Patch("vllm_torchtpu.layers.adapter.vision_attention", _ACTIVE),
    # Platform runtime configuration.
    Patch(
        "vllm_torchtpu.platforms.tpu_platform:_configure_torchtpu_eager_mode", _ACTIVE
    ),
    Patch("vllm_torchtpu.platforms.tpu_platform:_unwrap_dynamic_compile_fns", _ACTIVE),
    Patch(
        "vllm_torchtpu.platforms.tpu_platform:_patch_api_server_kernel_reload_endpoint",
        _ACTIVE,
    ),
    # API-server prompt tokenization (VLLM_TPU_SEGMENT_TOKENIZE).
    Patch("vllm_torchtpu.segment_tokenize:patch_renderer_segment_tokenize", _ACTIVE),
    # API-server reasoning-token count (VLLM_TPU_REASONING_COUNT_FAST).
    Patch("vllm_torchtpu.reasoning_count:patch_reasoning_count_fast", _ACTIVE),
    # EngineCore / API-server CPU affinity (q38 SV-PAIR; inert unless
    # VLLM_TPU_CPU_PIN_ENGINE is set, which needs VLLM_TPU_CPU_PIN=slice).
    Patch("vllm_torchtpu.worker.cpu_pinning:patch_engine_core_cpu_pin", ("engine_core",)),
    Patch("vllm_torchtpu.worker.cpu_pinning:patch_api_server_cpu_pin", _ACTIVE),
    # Model-specific patches.
    # Append patches for new models here with
    # stages=("platform_activation", "model_load") and model_config=True.
    # See docs/developers_guide/tpu_patch_mechanism.md for details.
    Patch(
        "vllm_torchtpu.models.vllm.qwen3_vl_patch:maybe_patch_qwen3_vl",
        ("platform_activation", "model_load"),
        model_config=True,
    ),
    Patch(
        "vllm_torchtpu.models.vllm.qwen3_next_patch:maybe_patch_qwen3_next",
        ("platform_activation", "model_load"),
    ),
)

_applied: set[str] = set()
_applying: set[str] = set()
_active_stages: set[Stage] = set()
_lock = RLock()
_logger = logging.getLogger("vllm." + __name__)


def _reset_after_fork() -> None:
    """Retains installed patches and resets process-local execution state."""
    global _lock
    _lock = RLock()
    # A fork inside a callback also clears the enclosing application state.
    _applying.clear()
    _active_stages.clear()


os.register_at_fork(after_in_child=_reset_after_fork)


def _invoke(target: str, **kwargs):
    module_name, _, name = target.partition(":")
    module = importlib.import_module(module_name)
    if name:
        return getattr(module, name)(**kwargs)
    return None


def apply(stage: Stage, *, model_config: "ModelConfig | None" = None) -> None:
    """Applies a stage once per patch and refreshes late module bindings.

    Import-triggered re-entry into the same stage returns immediately: the
    outer invocation already owns that ordered pass. Failures propagate and
    leave the failing patch eligible for retry, retaining completed patches.
    """
    if stage not in (
        "import",
        "platform_activation",
        "engine_core",
        "worker_init",
        "model_load",
    ):
        raise ValueError(f"Unknown TPU patch stage: {stage}")
    with _lock:
        if stage in _active_stages:
            return
        _active_stages.add(stage)
        applied = []
        try:
            for patch in PATCHES:
                if stage not in patch.stages or patch.target in _applying:
                    continue
                if patch.target in _applied:
                    if patch.refresh is not None:
                        _invoke(patch.refresh)
                    continue
                _applying.add(patch.target)
                try:
                    kwargs = (
                        {"model_config": model_config} if patch.model_config else {}
                    )
                    result = _invoke(patch.target, **kwargs)
                finally:
                    _applying.discard(patch.target)
                if result is False:
                    continue
                _applied.add(patch.target)
                applied.append(patch.target)
        finally:
            _active_stages.discard(stage)
            if applied:
                _logger.info(
                    "Completed TPU patch callbacks at %s: %s", stage, ", ".join(applied)
                )
