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

Stage = Literal["import", "platform_activation", "engine_core", "worker_init",
                "model_load"]


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
    Patch("vllm_torchtpu.env_override:_patch_jax_pallas_fori_lowering",
          ("import", )),
    Patch("vllm_torchtpu.layers.adapter.custom_ops:_register_custom_ops",
          _ACTIVE),

    # Compilation and cache keys.
    Patch("vllm_torchtpu:_patch_vllm_aot_compile_cache_key", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_compile_all_ranges", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_compile_prefix_isolation", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_config_hash_ignore_diagnostics", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_disable_compile_ranges", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_piecewise_backend", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_reset_compile_wrapper", _ACTIVE),

    # MoE routing and collectives.
    Patch("vllm_torchtpu:_patch_default_moe_runner_select_forward", _ACTIVE),
    Patch("vllm_torchtpu:_patch_disable_sequence_parallel_moe", _ACTIVE),
    Patch("vllm_torchtpu:_patch_expert_map_host_lookup", _ACTIVE),
    Patch("vllm_torchtpu:_patch_moe_explicit_pcp_collectives", _ACTIVE),
    Patch("vllm_torchtpu:_patch_moe_runner_fused_output_is_reduced", _ACTIVE),

    # KV cache, scheduling, and offloading.
    Patch("vllm_torchtpu:_patch_vllm_hybrid_pcp_block_sizes",
          _ENGINE,
          refresh="vllm_torchtpu:_patch_vllm_hybrid_pcp_block_sizes"),
    Patch("vllm_torchtpu:_patch_vllm_mamba_split_scheduler_block_size",
          _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_offloading_config_build", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_hybrid_producer_prefix_hits", _ENGINE),
    Patch("vllm_torchtpu:_patch_vllm_block_pool_lifo_free", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_kimi_kda_layer_counts", _ACTIVE),

    # Process setup and runner selection.
    Patch("vllm_torchtpu:_patch_multiproc_worker_global_rank_env", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_force_v1_runner_tpu", _ACTIVE),
    Patch("vllm_torchtpu:_patch_dflash_bypass_v2_runner_check", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_config_triton_tpu", _ACTIVE),

    # Weight loading and embeddings.
    Patch(
        "vllm_torchtpu.model_loader_patches:patch_default_loader_ep_weight_filter",
        _ACTIVE),
    Patch(
        "vllm_torchtpu.model_loader_patches:patch_runai_sharded_expert_streaming",
        _ACTIVE),
    Patch(
        "vllm_torchtpu.model_loader_patches:patch_default_model_loader_page_cache",
        _ACTIVE),
    Patch("vllm_torchtpu.model_loader_patches:patch_moe_expert_write_staging",
          _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_vocab_parallel_embedding", _ACTIVE),
    Patch("vllm_torchtpu:_patch_vllm_merge_multimodal_embeddings",
          _ENGINE,
          refresh="vllm_torchtpu:_patch_vllm_merge_multimodal_embeddings"),
    Patch("vllm_torchtpu.layers.adapter.vision_attention", _ACTIVE),

    # Platform runtime configuration.
    Patch(
        "vllm_torchtpu.platforms.tpu_platform:_configure_torchtpu_eager_mode",
        _ACTIVE),
    Patch("vllm_torchtpu.platforms.tpu_platform:_unwrap_dynamic_compile_fns",
          _ACTIVE),
    Patch(
        "vllm_torchtpu.platforms.tpu_platform:_patch_api_server_kernel_reload_endpoint",
        _ACTIVE),

    # Model-specific patches.
    # Append patches for new models here with stages=("platform_activation", "model_load")
    # and model_config=True. See docs/tpu_patch_mechanism.md for details.
    Patch("vllm_torchtpu.models.vllm.qwen3_vl_patch:maybe_patch_qwen3_vl",
          ("platform_activation", "model_load"),
          model_config=True),
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
    if stage not in ("import", "platform_activation", "engine_core",
                     "worker_init", "model_load"):
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
                    kwargs = {
                        "model_config": model_config
                    } if patch.model_config else {}
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
                _logger.info("Completed TPU patch callbacks at %s: %s", stage,
                             ", ".join(applied))
