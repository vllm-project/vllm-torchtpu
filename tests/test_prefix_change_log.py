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
"""The prefix-cache epoch and change log install lazily.

Importing the scheduler must leave vLLM's block pool, KV cache manager and
scheduler exactly as main's same-step patch left them, so a configuration
that does not select PrefixAwareScheduler runs unchanged. The counters
install when a scheduler that memoises lookups is built.
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

    prefix_change_log.install_epoch()
    after = methods()
    assert prefix_change_log.epoch_installed()
    assert after[3] is before[3] and after[5] is before[5]
    assert all(a is not b for a, b in zip(after[:3], before[:3]))
    assert after[4] is not before[4]
    prefix_change_log.install_epoch()
    assert methods() == after, "install_epoch is not idempotent"
    assert prefix_change_log.install()
    print("LAZY-INSTALL-OK")
    """
)


def test_import_leaves_vllm_unpatched_and_install_is_lazy():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "LAZY-INSTALL-OK" in result.stdout
