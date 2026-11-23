# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_ATTN_METADATA_REUSE: on steady-decode target walks the staged
walk takes the previous walk's tables / mamba products without re-deriving
them, and the inline build reuses every group's per-group fields. Both must
hand the model the same metadata as the full walk and build, every step,
through request add / finish / reorder, prefill pieces, block-table growth and
mamba block crossings."""

import dataclasses
import os
from unittest.mock import patch

import pytest
import torch

from tests.runner.steady_decode_fakes import Workload, env, make_runner, run_prepare
from tests.runner.test_steady_decode_metadata import _assert_metadata_equal
from vllm_torchtpu import envs
from vllm_torchtpu.layers.core import attention_metadata as am
from vllm_torchtpu.runner import tpu_runner as tr

REUSE = {"VLLM_TPU_ATTN_METADATA_REUSE": "1"}
FAST_BUILD = {"VLLM_TPU_FAST_METADATA_BUILD": "1"}
STACKED = {"VLLM_TPU_STAGED_WALK_STACKED": "1"}


class _Counts:
    def __init__(self):
        self.stage_fast = 0
        self.md_reuse = 0


@pytest.fixture
def counts():
    c = _Counts()
    fast = am._steady_stage_fast_path
    reuse = tr.TPUModelRunner._reuse_attention_metadata

    def count_fast(*args, **kwargs):
        hit = fast(*args, **kwargs)
        c.stage_fast += bool(hit)
        return hit

    def count_reuse(self, *args, **kwargs):
        c.md_reuse += 1
        return reuse(self, *args, **kwargs)

    with (
        patch.object(am, "_steady_stage_fast_path", count_fast),
        patch.object(tr.TPUModelRunner, "_reuse_attention_metadata", count_reuse),
    ):
        yield c


def _assert_rebuild_identical(runner, out):
    """The flag-on metadata is, field for field, the very objects a full
    inline build from the same builder context returns."""
    ctx = out["ctx"]
    fresh = dataclasses.replace(
        ctx,
        unified_mamba_state_indices=(
            [] if ctx.unified_mamba_state_indices is not None else None
        ),
        mamba_row_plans={},
    )
    with patch.dict(os.environ, {"VLLM_TPU_ATTN_METADATA_REUSE": "0"}):
        rebuilt = runner._build_attention_metadata_staged(fresh)
    assert rebuilt is not None
    assert list(rebuilt) == list(out["metadata"])
    seen = {}
    for layer, md in out["metadata"].items():
        ref = rebuilt[layer]
        assert type(md) is type(ref), layer
        assert list(md.__dict__) == list(ref.__dict__), layer
        for name, value in ref.__dict__.items():
            assert md.__dict__[name] is value or (
                isinstance(value, (str, int)) and md.__dict__[name] == value
            ), (layer, name)
        # Layers of one group share one object, as in the build.
        seen.setdefault(id(ref), id(md))
        assert seen[id(ref)] == id(md), layer
    if ctx.unified_mamba_state_indices is not None:
        a = ctx.unified_mamba_state_indices
        b = fresh.unified_mamba_state_indices
        assert len(a) == len(b) and all(x is y for x, y in zip(a, b))


def _run(
    seed,
    num_reqs,
    extra,
    counts,
    steps=300,
    mamba_groups=3,
    max_model_len=8192,
    attn_block=64,
    mamba_block=64,
    **wl_kw,
):
    blocks = dict(attn_block=attn_block, mamba_block=mamba_block)
    wl = Workload(seed, num_reqs, max_model_len=max_model_len, **blocks, **wl_kw)
    off, on = (
        make_runner(
            max_model_len=max_model_len, num_mamba_groups=mamba_groups, **blocks
        )
        for _ in range(2)
    )
    for step in range(steps):
        if step:
            wl.step()
        num_decode, num_windowed = wl.segments()
        wl.apply(off)
        wl.apply(on)
        so = wl.scheduler_output()
        with env(steady=True, extra=extra):
            out_off = run_prepare(off, so, num_decode, num_windowed)
        with env(steady=True, extra={**extra, **REUSE}):
            out_on = run_prepare(on, so, num_decode, num_windowed)
            _assert_rebuild_identical(on, out_on)
        _assert_metadata_equal(out_off, out_on, step)
    # The steady bookkeeping (hits, walk skips) is the same with the flag.
    a, b = off._steady_decode_cache, on._steady_decode_cache
    assert (a.hits, a.misses, a.walks, a.walk_skips) == (
        b.hits,
        b.misses,
        b.walks,
        b.walk_skips,
    )
    return on


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("num_reqs", [1, 4])
def test_reuse_matches_full_walk(seed, num_reqs, counts):
    """Random acceptance, block crossings, churn and prefill pieces: equal to
    the flag-off runner every step, identical to a full rebuild from the same
    ctx, and both reuse paths ran on most steady steps."""
    on = _run(seed, num_reqs, FAST_BUILD, counts, churn=0.06, prefill=0.04)
    cache = on._steady_decode_cache
    assert counts.stage_fast > 0.5 * cache.walk_skips, (counts.stage_fast, cache)
    assert counts.md_reuse >= counts.stage_fast > 20
    assert counts.stage_fast < cache.walks


def test_reuse_with_stacked_walk(counts):
    """Stacked mamba products (MambaStackRef rows) reuse the same way."""
    _run(3, 4, {**FAST_BUILD, **STACKED}, counts, churn=0.06, prefill=0.04)
    assert counts.stage_fast > 20 and counts.md_reuse > 20


def test_reuse_attention_page_smaller_than_mamba_block(counts):
    """Attention pages fill several times per mamba block, so the attention
    table grows while every mamba offset stays put: only the block-table
    version tells the fast path the upload changed."""
    _run(7, 4, FAST_BUILD, counts, churn=0.0, prefill=0.0, attn_block=16)
    assert counts.stage_fast > 20


def test_reuse_without_fast_build(counts):
    """FAST_METADATA_BUILD off: steady hits still take the inline build."""
    _run(4, 3, {}, counts, churn=0.05, prefill=0.05)
    assert counts.md_reuse > 20


def test_reuse_many_groups_long_context(counts):
    """Served-like geometry: many mamba groups, 512-token pages, long
    contexts, no churn: nearly every step is a reuse."""
    wl_kw = dict(context=20000, churn=0.0, prefill=0.0, max_model_len=32768)
    blocks = dict(attn_block=512, mamba_block=512)
    on = _run(5, 4, FAST_BUILD, counts, steps=120, mamba_groups=12, **blocks, **wl_kw)
    assert counts.stage_fast >= on._steady_decode_cache.walk_skips - 2


def test_reuse_invalidates_on_every_change(counts):
    """Each kind of change forces the full walk and build on that step."""
    wl = Workload(6, 3, max_model_len=8192, churn=0.0, prefill=0.0)
    runner = make_runner(max_model_len=8192)
    log = []

    def step(**change):
        before = (counts.stage_fast, counts.md_reuse)
        wl.step()
        if change.get("add"):
            wl._add()
        if change.get("finish"):
            wl.finished.add(wl.reqs.pop(0)["id"])
        if change.get("reorder"):
            wl.reqs.reverse()
        if change.get("prefill"):
            wl.prefill_id = wl.reqs[0]["id"]
        if change.get("cross"):
            # Push every request past its next attention / mamba block.
            for req in wl.reqs:
                req["computed"] += 64
        num_decode, num_windowed = wl.segments()
        wl.apply(runner)
        with env(steady=True, extra={**FAST_BUILD, **REUSE}):
            out = run_prepare(runner, wl.scheduler_output(), num_decode, num_windowed)
            _assert_rebuild_identical(runner, out)
        log.append((counts.stage_fast - before[0], counts.md_reuse - before[1]))
        return log[-1]

    # Warm up into steady state (acceptance draws keep blocks stable for a
    # few steps at a time; take any step that reused).
    for _ in range(40):
        step()
    assert any(s == (1, 1) for s in log), log
    for change in ("add", "finish", "reorder", "prefill", "cross"):
        # Re-enter steady state first.
        for _ in range(60):
            if step() == (1, 1):
                break
        else:
            pytest.fail(f"no steady reuse before {change}: {log[-10:]}")
        assert step(**{change: True}) == (0, 0), change


def test_draft_walk_in_between_is_caught(counts):
    """A partial (draft) walk that replaces the upload record makes the next
    target walk take the full path."""
    wl = Workload(8, 2, max_model_len=8192, churn=0.0, prefill=0.0)
    runner = make_runner(max_model_len=8192)
    with env(steady=True, extra={**FAST_BUILD, **REUSE}):
        for _ in range(30):
            wl.step()
            nd, nw = wl.segments()
            wl.apply(runner)
            run_prepare(runner, wl.scheduler_output(), nd, nw)
        assert runner._steady_stage_record is not None
        # A walk over a subset of groups (as a drafter's would) re-packs and
        # replaces `_block_table_stage_prev`.
        ctx = dataclasses.replace(
            runner._attn_metadata_builder_ctx,
            staged_block_tables=None,
            staged_mamba_products=None,
            steady_walk_bounds=None,
            steady_hit=False,
            unified_mamba_state_indices=[],
            mamba_row_plans={},
        )
        am.stage_block_table_uploads(runner, ctx, ctx.num_reqs + 3, gids={0})
        before = counts.stage_fast
        wl.step()
        nd, nw = wl.segments()
        wl.apply(runner)
        run_prepare(runner, wl.scheduler_output(), nd, nw)
        assert counts.stage_fast == before


def test_flag_off_is_a_no_op():
    assert envs.VLLM_TPU_ATTN_METADATA_REUSE is False
    wl = Workload(9, 2, max_model_len=8192)
    runner = make_runner(max_model_len=8192)
    with env(steady=True, extra=FAST_BUILD):
        for step in range(30):
            if step:
                wl.step()
            nd, nw = wl.segments()
            wl.apply(runner)
            run_prepare(runner, wl.scheduler_output(), nd, nw)
    assert "_steady_stage_record" not in runner.__dict__
    assert "_attn_metadata_reuse" not in runner.__dict__


def test_reused_metadata_is_a_new_object_each_step():
    """Reuse never hands out an earlier step's object (its holders, e.g. the
    drafter, keep what they had)."""
    wl = Workload(10, 2, max_model_len=8192, churn=0.0, prefill=0.0)
    runner = make_runner(max_model_len=8192)
    prev = None
    with env(steady=True, extra={**FAST_BUILD, **REUSE}):
        for _ in range(40):
            wl.step()
            nd, nw = wl.segments()
            wl.apply(runner)
            out = run_prepare(runner, wl.scheduler_output(), nd, nw)
            if prev is not None:
                cur_ids = {id(md) for md in out["metadata"].values()}
                assert not cur_ids & {id(md) for md in prev.values()}
                for md in prev.values():
                    assert isinstance(md.seq_lens, torch.Tensor)
            prev = out["metadata"]


def test_hit_counters_match_reuse_and_log(counts, monkeypatch, caplog):
    """The self-check counters count exactly the reuses taken, and the rank-0
    line reports them (counting only: outputs are checked by `_run`)."""
    import logging

    monkeypatch.setattr(am, "_reuse_counts", {"stage": [0, 0], "build": [0, 0]})
    monkeypatch.setattr(am, "REUSE_LOG_FIRST", 50)
    monkeypatch.setattr(am, "REUSE_LOG_EVERY", 100)
    with caplog.at_level(logging.INFO):
        _run(0, 4, {}, counts, steps=200)
    stage_hit, stage_miss = am._reuse_counts["stage"]
    build_hit, build_miss = am._reuse_counts["build"]
    assert stage_hit == counts.stage_fast and stage_hit > 0
    assert build_hit == counts.md_reuse and build_hit > 0
    assert stage_miss > 0 and build_miss >= 0
    lines = [r.getMessage() for r in caplog.records if "METADATA_REUSE hits" in r.getMessage()]
    assert lines and lines[-1].startswith("VLLM_TPU_ATTN_METADATA_REUSE hits: stage ")
    assert am.metadata_reuse_line().endswith("%)")
