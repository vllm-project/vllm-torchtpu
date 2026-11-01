# Copyright 2026 Google LLC
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
"""Opt-in retention of trailing prefix-cache blocks with EAGLE/MTP (env-gated).

With EAGLE/MTP, vLLM drops the last matched block from every prefix-cache
hit. In the draft layers, the KV slot of a chunk's final token was written
together with the *next* token. A later request with a different next token
would read a draft-KV slot "polluted" by the previous request. The pinned
scheduler also backs off every Mamba "align" split by one block to stay
consistent with that drop (``Scheduler._mamba_block_aligned_split``).

The drop only protects draft-model KV, so it only affects the acceptance
rate. The target model's KV and Mamba state are exact either way. vLLM main
makes it optional (``SpeculativeConfig.disable_eagle_block_drop``;
``Scheduler.use_eagle_block_drop`` is passed to ``KVCacheManager`` and gates
the Mamba back-off). This module installs the same behavior on the pinned
release, in the engine-core process:

- The ``KVCacheManager`` is built with ``use_eagle=False``, and any
  annotated eagle groups are cleared afterwards. No group drops a block,
  and ``cache_blocks`` caches whole aligned blocks.
- ``_mamba_block_aligned_split`` skips the one-block back-off. Mamba state is
  then checkpointed at the last full block, which the hit can now reach.
- ``Scheduler.use_eagle`` itself stays True. The prefill lookahead and the
  encoder shift are unchanged.

For a multi-turn replay with a 1152-token scheduler block, this gives up to
one extra block of prefix hit per turn, which means ~1152 fewer prefill
tokens. Enable it with ``VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1``. Check real
(non-synthetic) MTP acceptance when A/B testing.
"""

from functools import wraps
from typing import Any, cast

from vllm.logger import init_logger

from vllm_torchtpu import envs

logger = init_logger(__name__)


def clear_eagle_block_drop(kv_cache_manager) -> None:
    """Clear every EAGLE drop flag in a built KV cache manager.

    ``use_eagle=False`` only turns off the coordinator's "flag every group"
    fallback. Groups that ``_annotate_eagle_groups`` marked (DSpark,
    DeepseekV4) would still drop a block, so this clears them too.
    """
    kv_cache_manager.use_eagle = False
    coordinator = kv_cache_manager.coordinator
    if hasattr(coordinator, "eagle_group_ids"):
        coordinator.eagle_group_ids = set()
    for manager in coordinator.single_type_managers:
        manager.use_eagle = False
        if hasattr(manager, "drop_eagle_checkpoint_block"):
            manager.drop_eagle_checkpoint_block = False
    groups = getattr(coordinator, "attention_groups", None)
    if groups:
        coordinator.attention_groups = [g._replace(use_eagle=False) for g in groups]


def patch_offload_without_eagle_groups() -> None:
    """Build the KV offloading scheduler config without EAGLE groups.

    The offloading connector applies its own copy of the drop: each EAGLE
    group's trailing chunk is skipped on store while decoding and popped from
    every host lookup. With no annotated group it flags *every* group, the
    Mamba groups included, so a host hit must end one chunk before the
    retained Mamba boundary. Under sparse retention
    (``--prefix-cache-retention-interval 0``) no state exists there and the
    host cache almost never hits. Like the device-side drop this only
    protects draft-model KV, so it goes with it. Recomputes
    ``supports_partial_tail`` now that ``is_eagle_group`` is cleared so
    128-token fine-grained partial-tail chunks are offloaded and reloaded.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
        scheduler as offload_scheduler,
    )

    config_cls = offload_scheduler.SchedulerOffloadConfig
    original_from_spec = config_cls.from_spec
    if getattr(original_from_spec, "_tpu_disable_eagle_block_drop_patch", False):
        return

    @wraps(original_from_spec)
    def from_spec(spec, vllm_config, kv_cache_config):
        config = original_from_spec(spec, vllm_config, kv_cache_config)
        clear_retention = getattr(config, "retention_interval", None) == 0
        if clear_retention:
            config = config._replace(retention_interval=None)
            logger.info(
                "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: cleared "
                "SchedulerOffloadConfig.retention_interval=0 -> None."
            )
        if not any(group.is_eagle_group for group in config.kv_group_configs):
            return config
        kv_group_configs = tuple(
            group._replace(is_eagle_group=False) for group in config.kv_group_configs
        )
        group_block_sizes = {g.tokens_per_block for g in kv_group_configs}
        has_partial_recurrent_group = any(
            g.requires_cow_source and g.tokens_per_block > spec.tokens_per_hash
            for g in kv_group_configs
        )
        dcp_size = getattr(
            getattr(vllm_config, "parallel_config", None),
            "decode_context_parallel_size",
            1,
        )
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and has_partial_recurrent_group
            and all(
                g.sliding_window_size_in_chunks is None or g.requires_cow_source
                for g in kv_group_configs
            )
            and dcp_size == 1
        )
        logger.info(
            "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: KV offloading keeps the "
            "trailing chunk of every group (supports_partial_tail=%s).",
            supports_partial_tail,
        )
        return config._replace(
            kv_group_configs=kv_group_configs,
            supports_partial_tail=supports_partial_tail,
        )

    from_spec._tpu_disable_eagle_block_drop_patch = True
    config_cls.from_spec = staticmethod(from_spec)


def patch_disable_eagle_block_drop() -> None:
    if not envs.VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP:
        return
    patch_offload_without_eagle_groups()
    import vllm.v1.core.sched.scheduler as scheduler_module
    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__
    if getattr(original_init, "_tpu_disable_eagle_block_drop_patch", False):
        return
    original_split = Scheduler._mamba_block_aligned_split

    @wraps(original_init)
    def __init__(self, *args, **kwargs):
        manager_cls = scheduler_module.KVCacheManager

        def kv_cache_manager_without_eagle_drop(*m_args, **m_kwargs):
            m_kwargs["use_eagle"] = False
            return manager_cls(*m_args, **m_kwargs)

        # Engine-core startup is single threaded; restore right away.
        scheduler_module.KVCacheManager = cast(Any, kv_cache_manager_without_eagle_drop)
        try:
            original_init(self, *args, **kwargs)
        finally:
            scheduler_module.KVCacheManager = manager_cls
        clear_eagle_block_drop(self.kv_cache_manager)
        self.use_eagle_block_drop = False
        if self.use_eagle:
            logger.warning(
                "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: EAGLE/MTP trailing "
                "prefix-cache block drop is disabled (use_eagle=True, "
                "scheduler block %d). Target KV is unaffected; speculative "
                "acceptance may change.",
                self.block_size,
            )

    @wraps(original_split)
    def split(self, *args, **kwargs):
        old_use_eagle = getattr(self, "use_eagle", False)
        old_use_eagle_block_drop = getattr(self, "use_eagle_block_drop", False)
        if not old_use_eagle and not old_use_eagle_block_drop:
            return original_split(self, *args, **kwargs)
        # The upstream split reads use_eagle or use_eagle_block_drop for the
        # one-block back-off. schedule() calls it synchronously on EngineCore.
        self.use_eagle = False
        self.use_eagle_block_drop = False
        try:
            return original_split(self, *args, **kwargs)
        finally:
            self.use_eagle = old_use_eagle
            self.use_eagle_block_drop = old_use_eagle_block_drop

    setattr(__init__, "_tpu_disable_eagle_block_drop_patch", True)  # noqa: B010
    setattr(split, "_tpu_disable_eagle_block_drop_patch", True)  # noqa: B010
    Scheduler.__init__ = __init__
    Scheduler._mamba_block_aligned_split = split
    logger.info("VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP: patch installed.")
