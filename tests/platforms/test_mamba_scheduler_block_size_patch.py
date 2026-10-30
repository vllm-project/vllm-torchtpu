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
    original_init = Scheduler.__init__
    try:
        yield
    finally:
        Scheduler._mamba_block_aligned_split = original_split
        Scheduler.__init__ = original_init


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


@pytest.fixture
def _page_split_only(monkeypatch):
    """The patched split for a page-aligned program outside align mode, with
    an upstream split that must not run."""

    def upstream_split(self, *args, **kwargs):
        raise AssertionError("the upstream split runs only in align mode")

    monkeypatch.setattr(Scheduler, "_mamba_block_aligned_split", upstream_split)
    monkeypatch.setattr(
        vllm_torchtpu, "_model_needs_page_aligned_chunks", lambda vllm_config: True
    )
    _patch_vllm_mamba_split_scheduler_block_size()


# Prefix caching off: vLLM sets mamba_cache_mode "none", so only the page clip
# runs. 512-token pages, an 8192-token budget, 16384-token (DCP 32) or 512-token
# scheduler blocks.
@pytest.mark.parametrize("block", [16384, 512])
@pytest.mark.parametrize(
    ("prompt", "computed", "output", "new", "expected"),
    [
        # A 9000-token prompt gets 7192 tokens of a shared budget: cut to 7168.
        (9000, 0, 0, 7192, 7168),
        # Its last chunk is not clipped.
        (9000, 7168, 0, 1832, 1832),
        # Less than a page of budget at a page start: wait for the next step.
        (9000, 7168, 0, 300, 0),
        # A resumed request replays its output: clipped against num_tokens - 1.
        (9000, 0, 600, 9100, 8704),
    ],
)
def test_page_alignment_without_mamba_align_mode(
    _page_split_only, block, prompt, computed, output, new, expected
):
    scheduler, _ = _scheduler(page=512, block=block, budget=8192)
    scheduler._tpu_mamba_align_split = False
    request = _request(prompt_len=prompt)
    request.num_computed_tokens = computed
    request.num_tokens = prompt + output

    split = Scheduler._mamba_block_aligned_split(scheduler, request, num_new_tokens=new)

    assert split == expected


def _patched_init(monkeypatch, *, pages, align, budget=8192, threshold=0):
    def init(self, *args, **kwargs):
        self.vllm_config = SimpleNamespace()
        self.need_mamba_block_aligned_split = align
        self.cache_config = SimpleNamespace(block_size=512)
        self.max_num_scheduled_tokens = budget
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=threshold)

    monkeypatch.setattr(Scheduler, "__init__", init)
    # An unpatched split, so the patch installs its init wrapper even when the
    # registry already patched the scheduler in this process.
    monkeypatch.setattr(
        Scheduler, "_mamba_block_aligned_split", lambda self, *args: args[0]
    )
    monkeypatch.setattr(
        vllm_torchtpu, "_model_needs_page_aligned_chunks", lambda vllm_config: pages
    )
    _patch_vllm_mamba_split_scheduler_block_size()
    scheduler = Scheduler.__new__(Scheduler)
    Scheduler.__init__(scheduler)
    return scheduler


@pytest.mark.parametrize("pages", [True, False])
@pytest.mark.parametrize("align", [True, False])
def test_scheduler_init_enables_the_split_for_page_aligned_programs(
    monkeypatch, pages, align
):
    scheduler = _patched_init(monkeypatch, pages=pages, align=align)

    assert scheduler._tpu_mamba_align_split is align
    assert scheduler._tpu_page_aligned_chunks is pages
    assert scheduler.need_mamba_block_aligned_split is (pages or align)


@pytest.mark.parametrize(("budget", "threshold"), [(256, 0), (8192, 256)])
def test_page_aligned_programs_reject_chunk_limits_under_a_page(
    monkeypatch, budget, threshold
):
    with pytest.raises(ValueError, match="whole 512-token pages"):
        _patched_init(
            monkeypatch, pages=True, align=True, budget=budget, threshold=threshold
        )
