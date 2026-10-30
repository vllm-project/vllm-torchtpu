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
"""v17c PAW cross-step memo x v17e single-step prefill.

The single-step MambaManager patches create cache entries the stock path
never does: a partial tail on a speculative block (re-pointed to a private
copy one step later), a speculative block kept in place as a full-block
checkpoint, and an internal-checkpoint block for chunks that start mid-block.
Runs the multi-turn replay of test_single_step_prefill (production geometry,
real HybridStateRetention hooks) with the plugin's same-step prefix-hit patch
and the prefix change log installed, and after every step start and every
allocation compares the memoized `find_longest_cache_hit` of a set of waiting
probe requests (future turns, rollbacks to earlier checkpoints) with a fresh
lookup, block ids included.
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import test_single_step_prefill as T
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler

import vllm_torchtpu
from vllm_torchtpu.core import single_step_prefill as ssp
from vllm_torchtpu.core.prefix_aware_scheduler import (
    PrefixAwareScheduler,
    _hit_signature,
)


@pytest.fixture
def patched_pool():
    names = (
        "_insert_block_hash",
        "_remove_cached_block_hashes",
        "reset_prefix_cache",
        "get_cached_block",
    )
    saved_pool = {k: BlockPool.__dict__[k] for k in names}
    saved_sched = Scheduler.__dict__["schedule"]
    vllm_torchtpu._patch_vllm_same_step_prefix_hits()
    init_none_hash(sha256)
    ssp.install_mamba_manager_patches()
    try:
        yield
    finally:
        for k, v in saved_pool.items():
            setattr(BlockPool, k, v)
        Scheduler.schedule = saved_sched


def _schedule_prologue(kvm):
    # The real (patched) Scheduler.schedule prologue: change-log wrapper and
    # the unwritten-block clear; the upstream body then fails on the stub.
    with contextlib.suppress(AttributeError):
        Scheduler.schedule(SimpleNamespace(kv_cache_manager=kvm))


class MemoArm(T.Arm):
    def __init__(self, single_step):
        super().__init__(single_step)
        paw = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
        paw.kv_cache_manager = self.kvm
        paw._paw_scoring_broken = False
        paw._enable_memo_xstep()
        assert paw._memo_xstep
        paw._memo_verify_left = 0
        self.paw = paw
        self.probes = {}
        self.checks = self.nontrivial = 0
        alloc = self.kvm.allocate_slots

        def allocate_slots(*args, **kwargs):
            out = alloc(*args, **kwargs)
            self.check()
            return out

        self.kvm.allocate_slots = allocate_slots
        new_step = self.kvm.new_step_starts

        def new_step_starts():
            _schedule_prologue(self.kvm)
            new_step()
            self.check()

        self.kvm.new_step_starts = new_step_starts

    def check(self):
        paw = self.paw
        for req in self.probes.values():
            epoch = paw._get_prefix_cache_epoch()
            max_len = req.num_tokens - 1
            c = paw._memo_get(req, epoch, max_len)
            if c is None:
                c = paw._memo_compute(req, epoch, max_len)
            memo = _hit_signature(c[2])
            fresh = _hit_signature(
                self.kvm.coordinator.find_longest_cache_hit(req.block_hashes, max_len)
            )
            assert memo == fresh, (req.request_id, memo[:2], fresh[:2])
            self.checks += 1
            self.nontrivial += memo[0] > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_memo_exact_under_single_step(patched_pool, seed):
    rng = np.random.default_rng(seed)
    stock, single = MemoArm(False), MemoArm(True)
    for conv in range(3):
        prompts = T._conversation(rng, turns=6)
        probes = {}
        for t, p in enumerate(prompts[1:], start=1):
            probes[f"p{conv}_{t}"] = p
            # Rollback to the previous prompt's last full block + new tail.
            prev = prompts[t - 1]
            cut = len(prev) // T.BS * T.BS
            probes[f"r{conv}_{t}"] = prev[:cut] + list(rng.integers(1, 30000, size=300))
        for arm in (stock, single):
            arm.probes = {k: T._request(f"{k}_{id(arm)}", v) for k, v in probes.items()}
        for t, p in enumerate(prompts):
            rid = f"c{seed}x{conv}_{t}"
            h0 = stock.run(T._request(rid, p))
            h1 = single.run(T._request(rid, p))
            assert h0 == h1, (conv, t, h0, h1)
            for arm in (stock, single):
                arm.probes.pop(f"p{conv}_{t}", None)
    for arm in (stock, single):
        assert arm.paw._memo_xstep
        assert arm.checks > 500 and arm.nontrivial > 100
        assert arm.paw._memo_hits > arm.checks // 4
    assert single.cut_log
    print(
        f"single: checks={single.checks} memo_hits={single.paw._memo_hits} "
        f"nontrivial={single.nontrivial} cuts={len(single.cut_log)} "
        f"steps={single.steps}"
    )
