# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PREFILL_TIME_BUDGET_MS: per-step prefill chunk from a time budget."""

import types

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import prefill_time_budget as tb


def test_parse_off_and_errors():
    assert tb.parse("") is None
    cfg = tb.parse("131:ref=16:alpha=0.5:min=4096:max=16384")
    assert (cfg.budget_ms, cfg.ref, cfg.alpha, cfg.min_tokens, cfg.max_tokens) == (
        131, 16, 0.5, 4096, 16384)
    for bad in ("0", "abc", "100:mode=x", "100:foo=1", "100:min=0", "100:align=0"):
        with pytest.raises(ValueError):
            tb.parse(bad)


def test_stall_mode_shrinks_with_decodes():
    cfg = tb.parse("131.072:min=2048")  # 8192 tokens at 16 decodes
    sizes = [cfg.chunk_tokens(n, 16384) for n in (0, 4, 8, 16, 32, 64)]
    assert sizes == [16384, 16384, 16384, 8192, 4096, 2048]
    assert all(s % 512 == 0 for s in sizes)
    # The engine cap and max= both bound the chunk.
    assert cfg.chunk_tokens(1, 8192) == 8192
    assert tb.parse("131.072:max=12288").chunk_tokens(1, 16384) == 12288


def test_step_mode():
    cfg = tb.parse("200:mode=step:fixed=36:dec=0.5:us=16")
    # (200 - 36 - 0.5 * 16) / 0.016 = 9750 -> 9728 aligned
    assert cfg.chunk_tokens(16, 16384) == 9728
    assert cfg.chunk_tokens(300, 16384) == 2048  # floor


def _sched(running, mnbt=16384, spec=3):
    return types.SimpleNamespace(
        running=running,
        max_num_scheduled_tokens=mnbt,
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=mnbt),
        vllm_config=types.SimpleNamespace(
            speculative_config=types.SimpleNamespace(num_speculative_tokens=spec)),
    )


def _req(decoding):
    return types.SimpleNamespace(num_prompt_tokens=100,
                                 num_computed_tokens=100 if decoding else 10)


def test_patch_sets_and_restores_budget(monkeypatch):
    seen = []

    def fake(self, throttle_prefills=False):
        seen.append((self.max_num_scheduled_tokens, throttle_prefills))
        return types.SimpleNamespace(num_scheduled_tokens={})

    monkeypatch.setattr(Scheduler, "schedule", fake)
    monkeypatch.setenv("VLLM_TPU_PREFILL_TIME_BUDGET_MS", "131.072")
    tb.patch_prefill_time_budget()
    tb.patch_prefill_time_budget()  # idempotent
    s32 = _sched([_req(True)] * 32 + [_req(False)])
    Scheduler.schedule(s32, True)
    s0 = _sched([_req(False)])
    Scheduler.schedule(s0)
    # 32 decodes: 4096-token chunk + 32 * 4 decode tokens; none: engine max.
    assert seen == [(4096 + 128, True), (16384, False)]
    assert s32.max_num_scheduled_tokens == 16384 and s0.max_num_scheduled_tokens == 16384


def test_off_leaves_schedule(monkeypatch):
    def fake(self, throttle_prefills=False):
        return None

    monkeypatch.setattr(Scheduler, "schedule", fake)
    monkeypatch.delenv("VLLM_TPU_PREFILL_TIME_BUDGET_MS", raising=False)
    tb.patch_prefill_time_budget()
    assert Scheduler.schedule is fake
