# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

import vllm_torchtpu
from vllm_torchtpu import _patch_vllm_mamba_split_scheduler_block_size

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _restore_scheduler_split_method():
    original_split = Scheduler._mamba_block_aligned_split
    try:
        yield
    finally:
        Scheduler._mamba_block_aligned_split = original_split


def _request(prompt_len: int):
    return SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        shared_prefix_boundary=0,
    )


def _scheduler(page: int = 768, block: int = 3072, budget: int = 16384):
    physical_cache_config = SimpleNamespace(block_size=page)
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        cache_config=physical_cache_config,
        block_size=block,
        use_eagle=True,
        max_num_scheduled_tokens=budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        hash_block_size=block,
    )
    return scheduler, physical_cache_config


@pytest.fixture(autouse=True)
def _no_page_aligned_program(monkeypatch):
    monkeypatch.setattr(
        vllm_torchtpu, "_model_needs_page_aligned_chunks", lambda vllm_config: False
    )


def test_mamba_split_uses_scheduler_block_size_and_restores_cache_config():
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler, physical_cache_config = _scheduler()

    split = Scheduler._mamba_block_aligned_split(
        scheduler,
        _request(prompt_len=8755),
        num_new_tokens=8755,
    )

    assert split == 3072
    assert scheduler.cache_config is physical_cache_config
    assert scheduler.cache_config.block_size == 768


def test_mamba_split_restores_cache_config_when_upstream_raises(monkeypatch):
    def raising_split(self, *args, **kwargs):
        raise RuntimeError("upstream split failed")

    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", raising_split)
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler, physical_cache_config = _scheduler()

    with pytest.raises(RuntimeError, match="upstream split failed"):
        Scheduler._mamba_block_aligned_split(
            scheduler,
            _request(prompt_len=8755),
            num_new_tokens=8755,
        )

    assert scheduler.cache_config is physical_cache_config
    assert scheduler.cache_config.block_size == 768


def test_mamba_split_patch_is_idempotent():
    _patch_vllm_mamba_split_scheduler_block_size()
    patched_split = Scheduler._mamba_block_aligned_split

    _patch_vllm_mamba_split_scheduler_block_size()

    assert Scheduler._mamba_block_aligned_split is patched_split


@pytest.mark.parametrize("pages", [True, False])
def test_sub_block_chunk_ends_stay_on_pages_for_programs_that_need_it(
    monkeypatch, pages
):
    # DCP 32: 8192-token scheduler blocks, 256-token pages, a 4096-token
    # prefill budget. A shared budget cut at 942 tokens ends mid-page; a
    # program that writes whole pages (Capabilities.page_aligned_chunks) gets
    # the chunk cut back to 768.
    monkeypatch.setattr(
        vllm_torchtpu, "_model_needs_page_aligned_chunks", lambda vllm_config: pages
    )
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler, _ = _scheduler(page=256, block=8192, budget=4096)

    split = Scheduler._mamba_block_aligned_split(
        scheduler,
        _request(prompt_len=9000),
        num_new_tokens=942,
    )

    assert split == (768 if pages else 942)


def test_serial_prefill_two_slot_tail_packing_and_skip_guard(monkeypatch):
    monkeypatch.setattr(
        vllm_torchtpu, "_model_needs_page_aligned_chunks", lambda vllm_config: True
    )
    monkeypatch.setattr(vllm_torchtpu.envs, "K3_SERIAL_PREFILL", True)
    monkeypatch.setattr(vllm_torchtpu.envs, "K3_MAX_PREFILL_STEP_TOKENS", 16384)
    _patch_vllm_mamba_split_scheduler_block_size()

    scheduler, _ = _scheduler(page=512, block=16384, budget=16448)
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=163840)
    )
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0, max_num_seqs=64
    )
    scheduler._tpu_unfinished_prefill_ids = {"r0"}
    scheduler._tpu_step_prefill_info = {}
    scheduler._tpu_effective_start = {"r0": 65536}

    r0 = _request(prompt_len=80128)
    r0.request_id = "r0"
    r0.num_computed_tokens = 65536
    s0 = Scheduler._mamba_block_aligned_split(scheduler, r0, num_new_tokens=14592)
    assert s0 == 14592

    # Short 12K prompt would trigger skips_step if partially scheduled into 1536 rows:
    # must return 0 as the 2nd prefill so it runs cleanly in one 12288 window next step.
    r_short = _request(prompt_len=12000)
    r_short.request_id = "r_short"
    assert (
        Scheduler._mamba_block_aligned_split(scheduler, r_short, num_new_tokens=12000)
        == 0
    )

    # Long 80128 prompt packs cleanly into remaining 1536 rows (14848 + 1536 = 16384).
    r1 = _request(prompt_len=80128)
    r1.request_id = "r1"
    s1 = Scheduler._mamba_block_aligned_split(scheduler, r1, num_new_tokens=1792)
    assert s1 == 1536
    assert scheduler.max_num_scheduled_tokens == 16448

    # Third prefill request in the same step is rejected (max 2 slots).
    r2 = _request(prompt_len=80128)
    r2.request_id = "r2"
    assert Scheduler._mamba_block_aligned_split(scheduler, r2, num_new_tokens=512) == 0

    # Next step: r1 continues from 1536 and lands cleanly on 16384 block boundary
    # (14848 tokens), allowing r2 to pack 0->1536 in the same 16384 window.
    scheduler._tpu_unfinished_prefill_ids = {"r1"}
    scheduler._tpu_step_prefill_info = {}
    r1.num_computed_tokens = 1536
    s1_next = Scheduler._mamba_block_aligned_split(scheduler, r1, num_new_tokens=16448)
    assert s1_next == 14848
    s2_next = Scheduler._mamba_block_aligned_split(scheduler, r2, num_new_tokens=1599)
    assert s2_next == 1536


def test_serial_prefill_aggregate_target_and_step_cap(monkeypatch):
    monkeypatch.setattr(
        vllm_torchtpu, "_model_needs_page_aligned_chunks", lambda vllm_config: True
    )
    monkeypatch.setattr(vllm_torchtpu.envs, "K3_SERIAL_PREFILL", True)
    _patch_vllm_mamba_split_scheduler_block_size()

    scheduler, _ = _scheduler(page=512, block=16384, budget=16448)
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=163840)
    )
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0, max_num_seqs=64
    )
    scheduler._tpu_is_aggregate = True
    scheduler._tpu_decoding_ids = set()

    # 1) Default K3_MAX_PREFILL_STEP_TOKENS=8192 caps each request at 8192 while
    #    allowing two 8192 chunks to pack into largest_window=16384.
    monkeypatch.setattr(vllm_torchtpu.envs, "K3_MAX_PREFILL_STEP_TOKENS", 8192)
    scheduler._tpu_unfinished_prefill_ids = set()
    scheduler._tpu_step_prefill_info = {}
    scheduler._tpu_effective_start = {}

    r_seed = _request(prompt_len=80128)
    r_seed.request_id = "r_seed"
    r_seed.num_computed_tokens = 80127
    # Aggregate prefill ends at num_tokens - 1 = 80127; the 1-token decode seed
    # must not occupy a prefill slot.
    assert (
        Scheduler._mamba_block_aligned_split(scheduler, r_seed, num_new_tokens=1) == 1
    )
    assert "r_seed" in scheduler._tpu_decoding_ids
    assert scheduler._tpu_step_prefill_info == {}

    r_one = _request(prompt_len=1)
    r_one.request_id = "r_one"
    assert Scheduler._mamba_block_aligned_split(scheduler, r_one, num_new_tokens=1) == 1
    assert scheduler._tpu_step_prefill_info == {}

    ra = _request(prompt_len=80128)
    ra.request_id = "ra"
    rb = _request(prompt_len=80128)
    rb.request_id = "rb"
    assert (
        Scheduler._mamba_block_aligned_split(scheduler, ra, num_new_tokens=16448)
        == 8192
    )
    assert (
        Scheduler._mamba_block_aligned_split(scheduler, rb, num_new_tokens=16448)
        == 8192
    )

    # 2) K3_MAX_PREFILL_STEP_TOKENS=0 derives largest_window=16384 from prefill_windows
    #    rather than raw max_num_scheduled_tokens=16448.
    monkeypatch.setattr(vllm_torchtpu.envs, "K3_MAX_PREFILL_STEP_TOKENS", 0)
    scheduler._tpu_step_prefill_info = {}
    scheduler._tpu_effective_start = {"rc": 65536}
    scheduler._tpu_unfinished_prefill_ids = {"rc"}
    rc = _request(prompt_len=80128)
    rc.request_id = "rc"
    rc.num_computed_tokens = 65536
    # Aggregate target is 80127, so remaining prefill tokens are 14591
    # (+ 1 decode token = 14592); rows_used = ceil(14591 / 512) * 512 = 14848,
    # leaving 16384 - 14848 = 1536 (not 16448 - 14848 = 1600).
    assert (
        Scheduler._mamba_block_aligned_split(scheduler, rc, num_new_tokens=14592)
        == 14592
    )
    assert scheduler._tpu_step_prefill_info["rc"] == (14848, 65536)
    rd = _request(prompt_len=80128)
    rd.request_id = "rd"
    assert (
        Scheduler._mamba_block_aligned_split(scheduler, rd, num_new_tokens=2048) == 1536
    )
