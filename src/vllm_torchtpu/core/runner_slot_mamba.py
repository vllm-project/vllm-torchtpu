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
"""GDN groups that take no scheduler blocks on the compact-mamba layout.

With prefix caching off, vLLM's MambaManager gives every request one block per
mamba group from the pool it shares with attention. On the compact-mamba
layout (``TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0``) the runner never reads those
blocks for GDN layers: it keeps GDN state in its own ``max_num_reqs + 1`` slot
arrays, indexed by per-request slot ids it assigns itself
(``TPUModelRunner._build_mamba_state_indices``), and gives the attention pool
everything else. For Qwen3.8-2.4T (3 GDN groups) at 8k input / 1k output that
is 3 of the 75 blocks the scheduler charges each request.

Where the runner owns the slots, it hands vLLM its GDN specs as
``RunnerSlotMambaSpec``. vLLM picks each group's manager from the spec's class,
and TpuPlatform registers ``RunnerSlotMambaManager`` for this one: a
MambaManager that allocates nothing. The coordinator, block pool and the rest
of the scheduler stay on vLLM's stock caching-off path.

That holds on the compact layout (the unified pool stores GDN state in those
very blocks), for GDN layers, with mamba cache mode "none", no speculative
decoding and no KV connector. It is derived from the config, so it is on by
default; everything else keeps vLLM's MambaSpec and MambaManager.
TPU_SKIP_MAMBA_SCHEDULER_BLOCKS=0 is a debug escape hatch back to the stock
accounting, which charges blocks the runner never reads.

``check_runner_slot_groups`` guards the invariant this creates: a
RunnerSlotMambaSpec group has empty block-table rows, so its state must come
from the runner's slot ids.

Port of vllm-project/tpu-inference#3736.
"""

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, MambaSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class RunnerSlotMambaSpec(MambaSpec):
    """A GDN MambaSpec whose state slots the TPU runner assigns itself.

    Registered by register_runner_slot_mamba_spec, not @register_kv_cache_spec:
    a registration at import time would run before vLLM's built-in specs and
    keep them from loading.
    """


class RunnerSlotMambaManager(MambaManager):
    """A MambaManager that allocates no blocks.

    The overrides keep vLLM's MambaManager signatures, so the type checker
    flags any signature change in a vLLM upgrade. Freeing and skipped-block
    removal need no override: they walk the request's block list, which stays
    empty. KV-connector allocation is never reached: the spec is not swapped
    when a connector is configured.
    """

    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        logger.info_once(
            "[runner_slot_mamba] GDN groups take no scheduler blocks: the "
            "runner owns their state slots. TPU_SKIP_MAMBA_SCHEDULER_BLOCKS=0 "
            "restores vLLM's accounting."
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        return 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        return []


def _ineligible_reason(
    spec: MambaSpec, vllm_config: "VllmConfig", runner_owns_slots: bool
) -> str | None:
    if not runner_owns_slots:
        return (
            "the runner does not own the GDN slots (the unified KV pool, or "
            "compact-mamba sizing did not apply)"
        )
    # Exact class only: a subclass may carry fields RunnerSlotMambaSpec cannot
    # take, or be registered to a manager of its own.
    if type(spec) is not MambaSpec:
        return f"its spec is {type(spec).__name__}, not MambaSpec"
    if spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN:
        return f"it applies to GDN layers only, not {spec.mamba_type.name}"
    # The spec's own mamba_cache_mode is copied from this config. The mode
    # tracks prefix caching: vLLM sets "align" for hybrid models whenever
    # caching is on and "none" when it is off (MambaModelConfig), and
    # TpuPlatform rejects caching on hybrid models off the unified pool.
    mode = vllm_config.cache_config.mamba_cache_mode
    if mode != "none":
        return f"mamba cache mode is {mode!r}, not 'none' (prefix caching)"
    if vllm_config.speculative_config is not None:
        return "speculative decoding is on"
    if vllm_config.kv_transfer_config is not None:
        return "a KV connector is configured"
    return None


def maybe_runner_slot_mamba_spec(
    spec: KVCacheSpec, vllm_config: "VllmConfig", *, runner_owns_slots: bool
) -> KVCacheSpec:
    """Returns `spec` as a RunnerSlotMambaSpec when the runner owns its state
    slots, otherwise `spec` unchanged. TPU_SKIP_MAMBA_SCHEDULER_BLOCKS=0 keeps
    every spec unchanged.

    `runner_owns_slots`: the runner indexes GDN state by its own per-request
    slot ids (the compact-mamba layout), so the layer's scheduler blocks are
    never read.
    """
    if not envs.TPU_SKIP_MAMBA_SCHEDULER_BLOCKS or not isinstance(spec, MambaSpec):
        return spec
    reason = _ineligible_reason(spec, vllm_config, runner_owns_slots)
    if reason is not None:
        # Debug only: on by default, so most hybrid deployments (the unified
        # pool included) land here by design.
        logger.debug_once(
            "[runner_slot_mamba] Mamba layer keeps its scheduler blocks: %s.",
            reason,
        )
        return spec
    return RunnerSlotMambaSpec(
        **{f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.init}
    )


def check_runner_slot_groups(
    kv_cache_config: KVCacheConfig, *, runner_owns_slots: bool
) -> None:
    """Raises if a RunnerSlotMambaSpec group reaches a runner without its own
    mamba slot pool.

    The scheduler gives such a group no blocks, so its block-table rows stay
    empty. Without the runner's slot ids the GDN op falls back to
    `block_tables[:, 0]`, block 0 for every request: all requests would share
    one recurrent state, with no error.

    `runner_owns_slots`: the runner initialized its compact slot pool, which
    feeds `mamba_state_indices`.
    """
    if runner_owns_slots:
        return
    group_ids = [
        gid
        for gid, group in enumerate(kv_cache_config.kv_cache_groups)
        if isinstance(group.kv_cache_spec, RunnerSlotMambaSpec)
    ]
    if group_ids:
        raise RuntimeError(
            f"KV cache groups {group_ids} take no scheduler blocks "
            "(RunnerSlotMambaSpec), but this runner has no compact mamba slot "
            "pool to index their state. The spec gate in get_kv_cache_spec and "
            "the slot-pool setup in initialize_kv_cache disagree; set "
            "TPU_SKIP_MAMBA_SCHEDULER_BLOCKS=0 to work around it."
        )


def register_runner_slot_mamba_spec() -> None:
    """Maps RunnerSlotMambaSpec to RunnerSlotMambaManager in vLLM's registry.

    Called from TpuPlatform.register_custom_kv_cache_specs, which vLLM runs
    right after registering its built-in specs, in each process that looks a
    manager up. Registering the same pair again is a no-op.
    """
    KVCacheSpecRegistry.register(
        RunnerSlotMambaSpec, RunnerSlotMambaManager, uniform_type_base_spec=MambaSpec
    )
