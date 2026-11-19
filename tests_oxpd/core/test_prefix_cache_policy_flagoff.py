# SPDX-License-Identifier: Apache-2.0
"""Flags-off inertness of the q38 prefix-cache policy patches (CACHE-3 tail + demote, STACK-1 intake 1).

With every VLLM_TPU_PREFIX_* flag unset (their defaults), each patch entry point must leave the vLLM
objects it would wrap untouched, so a served run that does not set the flags runs vLLM's own code
paths. This is the property the STACK-1 intake relies on; the intake's CPU proof mutation-checks it
(a demote patch that ignores its flag must fail here).
"""

import pytest

vllm = pytest.importorskip("vllm")

from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.core.single_type_kv_cache_manager import MambaManager  # noqa: E402

from vllm_torchtpu.core import mamba_checkpoint_retention as mcr  # noqa: E402
from vllm_torchtpu.core import prefix_cache_policy as pcp  # noqa: E402

FLAGS = (
    "VLLM_TPU_PREFIX_DEMOTE_STALE_CHECKPOINT",
    "VLLM_TPU_PREFIX_TAIL_EVICT_FIRST",
    "VLLM_TPU_PREFIX_UNIT_ALLOW_F2_PRODUCER",
    "VLLM_TPU_PREFIX_REFRESH_WAITING",
    "VLLM_TPU_PREFIX_REFRESH_INTERVAL_S",
)


def _targets():
    return {
        "MambaManager.cache_blocks": MambaManager.cache_blocks,
        "MambaManager.pop_blocks_for_free": MambaManager.pop_blocks_for_free,
        "mamba_checkpoint_retention._release": mcr._release,
        "BlockPool.free_blocks": BlockPool.free_blocks,
        "KVCacheCoordinator.__init__": KVCacheCoordinator.__init__,
        "Scheduler.schedule": Scheduler.schedule,
    }


@pytest.fixture
def flags_off(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)
    # Retention is on by default in the served base; keep it on so the demote patch's only guard is its own flag.
    monkeypatch.setenv("VLLM_TPU_MAMBA_CHECKPOINT_RETENTION", "1")


@pytest.mark.parametrize(
    "entry",
    ["patch_demote_stale_checkpoints", "patch_tail_evict_first", "patch_refresh_waiting_prefixes"],
)
def test_patch_is_inert_with_flags_off(flags_off, entry):
    before = _targets()
    getattr(pcp, entry)()
    after = _targets()
    changed = [k for k in before if after[k] is not before[k]]
    assert not changed, f"{entry} changed {changed} with every VLLM_TPU_PREFIX_* flag unset"


def test_all_three_patches_together_are_inert_with_flags_off(flags_off):
    before = _targets()
    pcp.patch_demote_stale_checkpoints()
    pcp.patch_tail_evict_first()
    pcp.patch_refresh_waiting_prefixes()
    after = _targets()
    assert all(after[k] is before[k] for k in before)
