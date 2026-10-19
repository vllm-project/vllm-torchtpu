# SPDX-License-Identifier: Apache-2.0
"""Keeps unwritten Mamba scratch blocks out of the prefix cache on vLLM < 0.31.1rc0.

TODO(vLLM #59759): delete this module once the pin includes the upstream fix.
"""

from functools import wraps

from packaging.version import Version
from vllm import __version__ as vllm_version
from vllm.v1.core.single_type_kv_cache_manager import MambaManager

_MARKER = "_tpu_mamba_scratch_hash_patch"
_VLLM_VERSION = Version(vllm_version)
_SCRATCH_RELOCATION_VERSION = Version("0.31.1rc0")


def _drop_scratch_block_hashes(manager: MambaManager, request_id: str) -> None:
    """Unhashes unwritten scratch blocks below the reserved checkpoint.

    vLLM v0.31.0 leaves speculative blocks between the incoming state and
    the internal checkpoint, then hashes them in its full-block pass. Only
    the checkpoint and final state are written by this prefill step.
    """
    checkpoint = manager._checkpoints.get(request_id)
    state_idx = manager.last_state_block_idx.get(request_id)
    if checkpoint is None or state_idx is None:
        return
    _, checkpoint_idx = checkpoint
    pool = manager.block_pool
    dropped: set[int] = set()
    for block in manager.req_to_blocks[request_id][state_idx + 1 : checkpoint_idx]:
        if block.is_null or block.ref_cnt != 1:
            continue
        removed = pool._remove_cached_block_hashes(block)
        if removed:
            # Emit BlockRemoved only for hashes with no remaining cached blocks.
            evicted = [
                key
                for key in removed
                if pool.cached_block_hash_to_block.get_one_block(key) is None
            ]
            pool._emit_block_removed_events(evicted)
            dropped.add(block.block_id)
    if dropped:
        manager._pending_boundary_state_offloads = [
            entry
            for entry in manager._pending_boundary_state_offloads
            if entry[0] != request_id or entry[2].block_id not in dropped
        ]


def patch_mamba_checkpoint_hashes() -> None:
    """Preserves upstream allocation while excluding unwritten cache entries."""
    if _VLLM_VERSION >= _SCRATCH_RELOCATION_VERSION:
        return
    original = MambaManager.cache_blocks
    if getattr(original, _MARKER, False):
        return

    @wraps(original)
    def cache_blocks(self, request, *args, **kwargs):
        original(self, request, *args, **kwargs)
        if self.mamba_cache_mode == "align":
            _drop_scratch_block_hashes(self, request.request_id)

    setattr(cache_blocks, _MARKER, True)
    MambaManager.cache_blocks = cache_blocks
