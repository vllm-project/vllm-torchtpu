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
"""Host-only tests for TPU_SCHED_SINGLE_STEP_PREFILL (scheduler + KV manager).

The real vLLM `KVCacheManager` (hybrid: full attention + Mamba "align" with
3 speculative blocks, 640-token blocks, 128-token prefix_match_unit, sparse
retention) runs multi-turn conversations two ways: the stock align split
(chunks end at every cacheable boundary) and the single-step planner with
the MambaManager patches. A model of the GDN kernel tracks which state each
physical Mamba block holds (running block <- chunk end; in-chunk cuts <- the
cut position; CoW copies propagate). Checks:

* every hashed Mamba block holds exactly the state its hash claims (no
  poisoning), in both arms, across turns and evictions;
* the next turn's prefix-cache hit is identical in both arms;
* a warm turn takes one prefill step with the planner;
* the worker-side arrays name the blocks the KV manager confirmed.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core import single_step_prefill as ssp
from vllm_torchtpu.core.hybrid_retention import HybridStateRetention

BS = 640
HB = 128
NUM_SPEC = 3
MAMBA_GID = 1
BUDGET = 8192

_PREFIX_CACHE_METHODS = (
    "_insert_block_hash",
    "_remove_cached_block_hashes",
    "reset_prefix_cache",
    "get_cached_block",
)


@pytest.fixture(scope="module", autouse=True)
def _patches():
    # Upstream prefix-cache semantics: an earlier test that built a TPU
    # VllmConfig installs the plugin's same-step prefix-hit patch for the whole
    # session, and only a real Scheduler.schedule prologue (never run here)
    # clears its unwritten set, so every later hit would miss.
    saved = {k: BlockPool.__dict__[k] for k in _PREFIX_CACHE_METHODS}
    for k, fn in saved.items():
        setattr(BlockPool, k, inspect.unwrap(fn))
    init_none_hash(sha256)
    ssp.install_mamba_manager_patches()
    try:
        yield
    finally:
        for k, fn in saved.items():
            setattr(BlockPool, k, fn)


def _manager(num_blocks=6000):
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BS,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
        ],
        prefix_cache_retention_interval=0,
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=BS,
        hash_block_size=HB,
        enable_caching=True,
        use_eagle=False,  # VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1
    )


def _scheduler_stub(kvm):
    s = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=BS),
        block_size=BS,
        hash_block_size=HB,
        use_eagle=False,
        use_eagle_block_drop=False,
        need_mamba_block_aligned_split=True,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=True,
        mamba_shared_prefix_checkpoint=False,
        max_num_scheduled_tokens=BUDGET,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        kv_cache_manager=kvm,
    )
    return s


def _retention(kvm, smart_fork):
    # The real object: installs the MambaManager partial-tail / CoW hooks
    # (after the single-step class patches, as PrefixAwareScheduler does).
    return HybridStateRetention(
        kvm,
        max_conversations=0,
        max_pinned_blocks=0,
        smart_fork=smart_fork,
        safe_tail_offset=4,
    )


_HASHER = None


def _request(rid, tokens):
    global _HASHER
    if _HASHER is None:
        _HASHER = get_request_block_hasher(HB, sha256)
    sp = SamplingParams(max_tokens=64)
    sp.update_from_generation_config({}, 0)
    return Request(
        request_id=rid,
        prompt_token_ids=list(tokens),
        sampling_params=sp,
        pooling_params=None,
        block_hasher=_HASHER,
    )


class Arm:
    """One serving arm: the KV manager plus a model of the GDN kernel."""

    def __init__(self, single_step, smart_fork=True):
        self.kvm = _manager()
        self.sched = _scheduler_stub(self.kvm)
        self.ret = _retention(self.kvm, smart_fork)
        self.single_step = single_step
        self.mm = self.kvm.coordinator.single_type_managers[MAMBA_GID]
        self.state_at: dict[int, int] = {}  # block_id -> token offset
        self.steps = []
        self.cut_log = []
        self.checked = 0

    def _apply_copies(self):
        copies, retained = self.kvm.take_kv_cache_block_copies()
        for c in copies:
            if c.src_block_id in self.state_at:
                self.state_at[c.dst_block_id] = self.state_at[c.src_block_id]
            else:
                self.state_at.pop(c.dst_block_id, None)
        # The step ran: release the copy endpoints (scheduler deferred free).
        if retained:
            self.kvm.block_pool.free_blocks(retained[::-1])

    def _split(self, req, start, n, local):
        self.ret.prepare_prefill_split(req, start, n)
        if self.single_step:
            num, cuts = ssp.plan_single_step_split(self.sched, req, start, n, self.ret)
            ssp.set_plan(self.kvm, req.request_id, (start, *cuts) if cuts else None)
            return num
        num = Scheduler._mamba_block_aligned_split(self.sched, req, n, local)
        return self.ret.adjust_prefill_split(req, start, num)

    def run(self, req, decode_steps=3):
        blocks, hit = self.kvm.get_computed_blocks(req)[:2]
        first = True
        nsteps = 0
        while req.num_computed_tokens < req.num_prompt_tokens:
            self.kvm.new_step_starts()
            ssp.begin_step(self.kvm)
            local = hit if first else 0
            start = req.num_computed_tokens + local
            n = min(BUDGET, req.num_prompt_tokens - start)
            num = self._split(req, start, n, local) if n else 0
            assert num > 0
            out = self.kvm.allocate_slots(
                req,
                num,
                num_new_computed_tokens=local,
                new_computed_blocks=blocks if first else None,
                num_lookahead_tokens=NUM_SPEC,
                has_scheduled_reqs=False,
            )
            assert out is not None
            self._apply_copies()
            cuts = ssp.confirmed_cuts(self.kvm, {req.request_id: num}).get(
                req.request_id
            )
            # Worker: seed the running block (crossing), then the kernel.
            end = start + num
            rb = self.mm.req_to_blocks[req.request_id]
            run_col = (end - 1) // BS
            if cuts:
                full_cut, partial_cut = cuts
                bt = np.asarray([[b.block_id for b in rb]], np.int32)
                bounds, per_gid = ssp.build_seg_arrays(
                    {req.request_id: cuts},
                    [req.request_id],
                    0,
                    1,
                    1,
                    np.asarray([start]),
                    {req.request_id: num},
                    {MAMBA_GID: bt},
                    BS,
                )
                if full_cut:
                    assert bounds[0, 0] == full_cut - start
                    self.state_at[per_gid[MAMBA_GID][0, 0]] = full_cut
                if partial_cut:
                    assert bounds[0, 1] == partial_cut - start
                    assert per_gid[MAMBA_GID][0, 1] == rb[run_col + 1].block_id
                    self.state_at[per_gid[MAMBA_GID][0, 1]] = partial_cut
                self.cut_log.append(cuts)
            self.state_at[rb[run_col].block_id] = end
            req.num_computed_tokens = end
            first = False
            nsteps += 1
        self.steps.append(nsteps)
        # Decode a few verify steps (state block advances like the kernel's
        # checkpoint 0 + accepted).
        for _ in range(decode_steps):
            self.kvm.new_step_starts()
            ssp.begin_step(self.kvm)
            req.append_output_token_ids([7])
            out = self.kvm.allocate_slots(
                req, 1, num_lookahead_tokens=NUM_SPEC, has_scheduled_reqs=False
            )
            assert out is not None
            self._apply_copies()
            rb = self.mm.req_to_blocks[req.request_id]
            col = (req.num_computed_tokens) // BS
            # The verify window writes checkpoints at the state block and the
            # speculative blocks after it.
            for t in range(NUM_SPEC + 1):
                if col + t < len(rb) and not rb[col + t].is_null:
                    self.state_at[rb[col + t].block_id] = -1  # decode garbage
            self.state_at[rb[col].block_id] = req.num_computed_tokens + 1
            req.num_computed_tokens += 1
        self.checked += self.check_no_poison()
        self.kvm.free(req)
        return hit

    def check_no_poison(self):
        pool = self.kvm.block_pool
        checked = 0
        for block in pool.blocks:
            h = block.block_hash
            if h is None or block.is_null:
                continue
            if h.group_id if hasattr(h, "group_id") else None:
                pass
            gid = getattr(h, "group_id", None)
            if gid is None:
                from vllm.v1.core.kv_cache_utils import get_group_id

                gid = get_group_id(h)
            if gid != MAMBA_GID:
                continue
            claimed = block.block_hash_num_tokens
            assert self.state_at.get(block.block_id) == claimed, (
                f"block {block.block_id} hashed as state@{claimed} holds "
                f"state@{self.state_at.get(block.block_id)}"
            )
            checked += 1
        return checked


def _conversation(rng, turns=6, base=None):
    """Prompt i+1 extends floor128(prompt i) with fresh tokens (AgentX shape:
    the assistant output and template tail are rewritten next turn)."""
    tokens = list(base or rng.integers(1, 30000, size=rng.integers(3000, 12000)))
    prompts = [tokens]
    for _ in range(turns - 1):
        keep = len(tokens) // HB * HB if rng.random() < 0.85 else len(tokens) // BS * BS
        tokens = tokens[:keep] + list(
            rng.integers(1, 30000, size=rng.integers(150, 2600))
        )
        prompts.append(tokens)
    return prompts


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_hits_match_and_no_poisoning(seed):
    rng = np.random.default_rng(seed)
    stock, single = Arm(False), Arm(True)
    for conv in range(3):
        prompts = _conversation(rng)
        for t, p in enumerate(prompts):
            rid = f"c{seed}x{conv}_{t}"
            h0 = stock.run(_request(rid, p))
            h1 = single.run(_request(rid, p))
            assert h0 == h1, (conv, t, h0, h1)
    warm0 = [s for s in stock.steps[1:]]
    warm1 = [s for s in single.steps[1:]]
    assert sum(warm1) < sum(warm0)
    assert single.cut_log, "planner never cut"
    assert stock.checked > 50 and single.checked > 50
    print("stock steps", stock.steps, "single steps", single.steps)


def test_representative_warm_turn_single_step():
    rng = np.random.default_rng(11)
    base = list(rng.integers(1, 30000, size=94_050))
    turn2 = base[:93_952] + list(rng.integers(1, 30000, size=96_300 - 93_952))
    stock, single = Arm(False, smart_fork=False), Arm(True, smart_fork=False)
    for arm in (stock, single):
        arm.run(_request("conv_1", base), decode_steps=1)
        assert arm.run(_request("conv_2", turn2), decode_steps=2) == 93_952
    # Stock: 93,952 -> 94,080 (mid-block) -> 96,000 -> 96,256 -> 96,300.
    assert stock.steps[1] == 4
    assert single.steps[1] == 1
    assert single.cut_log[-1] == (96_000, 96_256)
    # Turn 3 hits the f128 checkpoint the single step registered.
    turn3 = turn2[:96_256] + list(rng.integers(1, 30000, size=700))
    for arm in (stock, single):
        assert arm.run(_request("conv_3", turn3), decode_steps=1) == 96_256
    # A rollback to turn 2's f640 hits the in-chunk full checkpoint.
    turn3b = turn2[:96_000] + list(rng.integers(1, 30000, size=500))
    for arm in (stock, single):
        assert arm.run(_request("conv_3b", turn3b), decode_steps=1) == 96_000


def _paw(arm):
    from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    for k, v in vars(arm.sched).items():
        setattr(s, k, v)
    s._cold_chunk_cap, s._cold_threshold, s._cold_min_decoders = 2048, 8192, 12
    s._step_total_cap, s._warm_streak, s._warm_min_decoders = 4096, 0, 24
    s._srpf_running, s._step_num_decoding, s._step_decode_tokens = True, 0, 0
    s._step_prefill_tokens_by_req = None
    s._skip_midblock_align_stop = False
    s._retention = arm.ret
    s._single_step = True
    return s


def test_supported_and_paw_split_records_plan():
    arm = Arm(True, smart_fork=False)
    ok, why = ssp.supported(arm.sched)
    assert ok, why
    arm.sched.kv_cache_manager.coordinator.retention_interval = None
    assert not ssp.supported(arm.sched)[0]
    arm.sched.kv_cache_manager.coordinator.retention_interval = 0
    arm.sched.use_eagle, arm.sched.use_eagle_block_drop = True, True
    assert not ssp.supported(arm.sched)[0]
    arm.sched.use_eagle = False

    rng = np.random.default_rng(5)
    base = list(rng.integers(1, 30000, size=94_050))
    arm.run(_request("conv_1", base), decode_steps=1)
    turn2 = base[:93_952] + list(rng.integers(1, 30000, size=96_300 - 93_952))
    req = _request("conv_2", turn2)
    paw = _paw(arm)
    ssp.begin_step(arm.kvm)
    got = paw._mamba_block_aligned_split(req, 96_300 - 93_952, 93_952, 0)
    assert got == 96_300 - 93_952
    assert arm.mm._tpu_seg_plan["conv_2"] == (93_952, 96_000, 96_256)


def test_supported_mirrors_worker_pool_preconditions(monkeypatch):
    """The scheduler refuses the feature whenever the workers cannot write
    in-chunk checkpoints (TPUModelRunner._mamba_seg_enabled raises on the
    first cut): no unified KV pool, the block-major unified pool, or a pooled
    hybrid whose op ignores seg_bounds (Kimi's KDA)."""
    from vllm_torchtpu import block_major_pool
    from vllm_torchtpu.platforms import tpu_block_size_utils

    layout = {"unified": True, "block_major": False}
    monkeypatch.setattr(
        tpu_block_size_utils, "unified_kv_layout_enabled", lambda _: layout["unified"]
    )
    monkeypatch.setattr(
        block_major_pool, "unified_block_major_enabled", lambda _: layout["block_major"]
    )
    arm = Arm(True, smart_fork=False)
    model_config = SimpleNamespace(architecture="Qwen3_5MoeForConditionalGeneration")
    arm.sched.vllm_config = SimpleNamespace(model_config=model_config)
    assert ssp.supported(arm.sched) == (True, "")
    model_config.architecture = "KimiK3ForConditionalGeneration"
    ok, why = ssp.supported(arm.sched)
    assert not ok and "GDN pooled op" in why
    model_config.architecture = "Qwen3NextForCausalLM"
    layout["block_major"] = True
    ok, why = ssp.supported(arm.sched)
    assert not ok and "VLLM_TPU_BLOCK_MAJOR_KV" in why
    layout.update(unified=False, block_major=False)
    ok, why = ssp.supported(arm.sched)
    assert not ok and "unified KV block pool" in why
