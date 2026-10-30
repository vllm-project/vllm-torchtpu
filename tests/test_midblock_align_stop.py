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
"""Host-only tests for TPU_SCHED_SKIP_MIDBLOCK_ALIGN_STOP.

They drive the real PrefixAwareScheduler._mamba_block_aligned_split, which
calls the pinned upstream Scheduler._mamba_block_aligned_split and the real
HybridStateRetention prepare/adjust hooks, chunk by chunk over a warm
prefill, with the Qwen3.5 AgentX geometry (640-token Mamba/scheduler block,
128-token prefix_match_unit, sparse retention, no EAGLE back-off).
"""

import random
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core.hybrid_retention import HybridStateRetention, _ConvLRUEntry
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BS = 640
HB = 128
BUDGET = 8192


def _retention(bs=BS, hb=HB, smart_fork=True, safe_tail=4):
    r = HybridStateRetention.__new__(HybridStateRetention)
    r.enabled = True
    r.smart_fork = smart_fork
    r.safe_tail_offset = safe_tail
    r.mamba_groups = [SimpleNamespace(group_ids=[0])]
    r.coordinator = SimpleNamespace(
        single_type_managers=[SimpleNamespace(block_size=bs)],
        _cache_hit_alignment_tokens=hb,
    )
    r._fork_decision = {}
    r._conv_lru = OrderedDict()
    return r


def _scheduler(skip, bs=BS, hb=HB, retention_interval=0, retention=True):
    s = PrefixAwareScheduler.__new__(PrefixAwareScheduler)
    # Upstream split inputs (vLLM v0.29.0 Scheduler.__init__).
    s.cache_config = SimpleNamespace(block_size=bs)
    s.block_size = bs
    s.hash_block_size = hb
    # VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP=1 (default) runs the split with
    # use_eagle=False, i.e. without the one-block back-off.
    s.use_eagle = False
    s.use_eagle_block_drop = False
    s.need_mamba_block_aligned_split = True
    s.mamba_has_prefill_checkpoint_blocks = False
    s.mamba_partial_cache_hit = hb < bs
    s.mamba_shared_prefix_checkpoint = False
    s.max_num_scheduled_tokens = BUDGET
    s.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
    s.kv_cache_manager = SimpleNamespace(
        enable_caching=True,
        coordinator=SimpleNamespace(retention_interval=retention_interval),
    )
    # PrefixAwareScheduler knobs (production defaults; caps idle at c<=16).
    s._cold_chunk_cap = 2048
    s._cold_threshold = 8192
    s._cold_min_decoders = 12
    s._step_total_cap = 4096
    s._warm_streak = 0
    s._warm_min_decoders = 24
    s._srpf_running = True
    s._step_num_decoding = 0
    s._step_decode_tokens = 0
    s._step_prefill_tokens_by_req = None
    s._skip_midblock_align_stop = skip
    s._n_midblock_stops_skipped = 0
    s._retention = _retention(bs, hb) if retention else None
    return s


def _request(prompt, rid="conv7_3"):
    return SimpleNamespace(
        request_id=rid,
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        num_computed_tokens=0,
        shared_prefix_boundary=0,
        block_hashes=list(range(prompt // HB)),
    )


def _retain_previous_turn(s, hit):
    """Retained conversation tail at ``hit`` that the request continues (a
    warm turn), so it is not a smart-fork candidate."""
    r = s._retention
    if r is None or hit < HB:
        return
    idx = hit // HB - 1
    r._conv_lru[("prev", hit)] = _ConvLRUEntry(
        conv_key=("prev", hit),
        req_id="prev",
        first_hash=0,
        boundary_idx=idx,
        boundary_hash=idx,
        blocks=[],
        hit=hit,
    )


def chunk_ends(s, prompt, hit, rid="conv7_3", budget=BUDGET, continuation=True):
    """Chunk ends of a prefill from a ``hit``-token prefix-cache hit."""
    req = _request(prompt, rid)
    if continuation:
        _retain_previous_turn(s, hit)
    ends = []
    local = hit  # first schedule passes the hit as new local computed tokens
    while req.num_computed_tokens + local < prompt:
        start = req.num_computed_tokens + local
        n = min(budget, prompt - start)
        got = s._mamba_block_aligned_split(req, n, local, 0)
        assert got > 0, (prompt, hit, ends)
        req.num_computed_tokens = start + got
        local = 0
        ends.append(req.num_computed_tokens)
    return ends


def test_representative_warm_turn_94k_cached_2k_new():
    # Previous turn's prompt 94,050 left its partial-tail state at 93,952
    # (floor128); this turn's prompt is 96,300 (2,348 new tokens).
    off = chunk_ends(_scheduler(False), 96_300, 93_952)
    on = chunk_ends(_scheduler(True), 96_300, 93_952)
    # Stops: ceil640(hit)=94,080 (mid-block), f640(P)=96,000 (last cacheable
    # block), f128(P)=96,256 (partial tail), P.
    assert off == [94_080, 96_000, 96_256, 96_300]
    assert on == [96_000, 96_256, 96_300]


def test_block_aligned_hit_is_unchanged():
    for skip in (False, True):
        assert chunk_ends(_scheduler(skip), 96_300, 93_440) == [
            96_000,
            96_256,
            96_300,
        ]


def test_refused_when_last_cache_position_is_the_boundary():
    # f640(94,700) = 94,080 = ceil640(93,952): the boundary is a checkpoint.
    for skip in (False, True):
        assert chunk_ends(_scheduler(skip), 94_700, 93_952) == [
            94_080,
            94_592,
            94_700,
        ]


def test_refused_under_dense_retention():
    s = _scheduler(True, retention_interval=None)
    assert chunk_ends(s, 96_300, 93_952) == [94_080, 96_000, 96_256, 96_300]
    assert s._n_midblock_stops_skipped == 0


def test_safe_tail_prompt_keeps_both_tail_stops():
    # 96,258 % 128 = 2 < 4: retention checkpoints floor128(P - 4) = 96,128;
    # upstream still stops at the raw tail 96,256.
    off = chunk_ends(_scheduler(False), 96_258, 93_952)
    on = chunk_ends(_scheduler(True), 96_258, 93_952)
    assert off == [94_080, 96_000, 96_128, 96_256, 96_258]
    assert on == off[1:]


def test_budget_limited_chunks_stay_block_aligned():
    off = chunk_ends(_scheduler(False), 40_000, 10_112)
    on = chunk_ends(_scheduler(True), 40_000, 10_112)
    assert off[0] == 10_240
    assert on == off[1:]
    assert all(e % BS == 0 for e in on[:-3])


def test_flag_off_matches_upstream_without_retention():
    s = _scheduler(False, retention=False)
    assert chunk_ends(s, 96_300, 93_952) == [94_080, 96_000, 96_256, 96_300]
    s = _scheduler(True, retention=False)
    assert chunk_ends(s, 96_300, 93_952) == [96_000, 96_256, 96_300]


def _checkpoints(prompt, safe_tail=4):
    """Positions whose Mamba state is cached under sparse retention."""
    eff = (
        prompt - safe_tail
        if (prompt % HB < safe_tail or prompt % BS < safe_tail)
        else prompt
    )
    fine = eff // HB * HB
    full = fine // BS * BS
    return {
        full,
        fine,
        full - BS,  # smart-fork pre-final block (fork candidates)
        prompt // BS * BS,  # upstream last_cache_position
        prompt // HB * HB,  # upstream partial-tail boundary
    }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_turns_drop_only_the_midblock_stop(seed):
    rng = random.Random(seed)
    merged = 0
    for _ in range(1500):
        hit = rng.randrange(1, 1600) * HB
        prompt = hit + rng.randrange(1, 12_000)
        rid = rng.choice(["conv1_2", "conv1::sa:002_0", "conv9_5"])
        off = chunk_ends(_scheduler(False), prompt, hit, rid)
        on = chunk_ends(_scheduler(True), prompt, hit, rid)
        ckpt = _checkpoints(prompt)
        # Every cacheable stop of the stock split is still a chunk end.
        assert {e for e in off if e in ckpt} <= set(on), (hit, prompt, off, on)
        boundary = (hit // BS + 1) * BS
        if on != off:
            # The only change: the mid-block chunk is merged into the next.
            assert hit % BS and off[0] == boundary and boundary not in ckpt
            assert on == off[1:], (hit, prompt, off, on)
            merged += 1
        assert len(on) <= len(off)
    assert merged > 500
