# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import (
    _patch_vllm_mamba_split_scheduler_block_size,
)


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


def _scheduler():
    physical_cache_config = SimpleNamespace(block_size=768)
    scheduler = SimpleNamespace(
        cache_config=physical_cache_config,
        block_size=3072,
        use_eagle=True,
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_shared_prefix_checkpoint=False,
        hash_block_size=3072,
    )
    return scheduler, physical_cache_config


def test_mamba_split_uses_scheduler_block_size_and_restores_cache_config():
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler, physical_cache_config = _scheduler()

    split = Scheduler._mamba_block_aligned_split(
        scheduler,
        _request(prompt_len=8755),
        num_new_tokens=8755,
    )

    assert split in (3072, 6144)
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


def _pp4pcp8_scheduler(checkpoint_blocks: bool, drop_eagle_block: bool):
    """The engine-core scheduler state under PP4xPCP8 with --block-size 384."""
    scheduler, _ = _scheduler()
    scheduler.cache_config.block_size = 384
    scheduler.use_eagle = drop_eagle_block
    scheduler.use_eagle_block_drop = drop_eagle_block
    scheduler.scheduler_config.long_prefill_token_threshold = 16384
    scheduler.mamba_has_prefill_checkpoint_blocks = checkpoint_blocks
    scheduler.mamba_prefill_checkpoint_alignment = 64 if checkpoint_blocks else None
    return scheduler


def _schedule_prompt(scheduler, prompt_len: int, budget: int = 16384) -> list[int]:
    request = _request(prompt_len)
    chunks = []
    while request.num_computed_tokens < prompt_len:
        remaining = prompt_len - request.num_computed_tokens
        chunk = Scheduler._mamba_block_aligned_split(
            scheduler, request, num_new_tokens=min(budget, remaining)
        )
        chunks.append(chunk)
        request.num_computed_tokens += chunk
    return chunks


@pytest.mark.parametrize(
    ("checkpoint_blocks", "chunks"), [(True, [8191]), (False, [3072, 5119])]
)
def test_internal_checkpoint_keeps_an_mtp_prompt_in_one_chunk(
    checkpoint_blocks, chunks
):
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler = _pp4pcp8_scheduler(checkpoint_blocks, drop_eagle_block=True)

    assert _schedule_prompt(scheduler, 8191) == chunks
    assert scheduler.cache_config.block_size == 384


@pytest.mark.parametrize(
    ("checkpoint_blocks", "chunks"),
    [(True, [15360, 4096]), (False, [15360, 3072, 1024])],
)
def test_internal_checkpoint_drops_the_last_block_stop_of_a_long_prompt(
    checkpoint_blocks, chunks
):
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler = _pp4pcp8_scheduler(checkpoint_blocks, drop_eagle_block=False)

    assert _schedule_prompt(scheduler, 19456) == chunks
