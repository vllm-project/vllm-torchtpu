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
"""The prefix-cache epoch and change log are registry patches.

Importing the scheduler must leave vLLM's block pool, KV cache manager and
scheduler exactly as main's same-step patch left them. The patch registry
installs the epoch counters and then the change log, both after the same-step
patch, and each entry is idempotent.
"""

import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import vllm_torchtpu
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_torchtpu import patch_registry

    vllm_torchtpu._patch_vllm_same_step_prefix_hits()

    def methods():
        return (
            BlockPool._insert_block_hash,
            BlockPool._remove_cached_block_hashes,
            BlockPool.reset_prefix_cache,
            BlockPool.get_cached_block,
            KVCacheManager.new_step_starts,
            Scheduler.schedule,
        )

    before = methods()
    import vllm_torchtpu.core.prefix_aware_scheduler  # noqa: F401
    from vllm_torchtpu.core import prefix_change_log

    assert methods() == before, "importing the scheduler patched vLLM"
    assert not prefix_change_log.epoch_installed()
    assert not prefix_change_log.install(), "change log without epoch counters"
    assert methods() == before

    targets = [
        p.target
        for p in patch_registry.PATCHES
        if p.target.startswith("vllm_torchtpu.core.prefix_change_log:")
    ]
    assert targets == [
        "vllm_torchtpu.core.prefix_change_log:install_epoch",
        "vllm_torchtpu.core.prefix_change_log:install",
    ], targets
    for target in targets:
        assert patch_registry._invoke(target) is not False, target
    after = methods()
    assert prefix_change_log.epoch_installed() and prefix_change_log.installed()
    assert after[3] is before[3] and after[5] is before[5]
    assert all(a is not b for a, b in zip(after[:3], before[:3]))
    assert after[4] is not before[4]
    for target in targets:
        patch_registry._invoke(target)
    assert methods() == after, "the registry entries are not idempotent"
    print("REGISTRY-INSTALL-OK")
    """
)


def test_import_leaves_vllm_unpatched_and_the_registry_installs():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "REGISTRY-INSTALL-OK" in result.stdout
