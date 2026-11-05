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

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

import vllm_torchtpu
from vllm_torchtpu import _patch_vllm_mamba_split_scheduler_block_size


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
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_shared_prefix_checkpoint=False,
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
