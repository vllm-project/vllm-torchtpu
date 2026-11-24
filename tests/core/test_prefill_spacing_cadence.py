# SPDX-License-Identifier: Apache-2.0
"""PrefillSpacingScheduler decode-priority cadence and prefill coalescing
(VLLM_TPU_PREFILL_DECODE_CADENCE / _COALESCE_TOKENS / _CADENCE_MAX_WAIT_S /
_COALESCE_MAX_WAIT_S) on a real async vLLM scheduler over a CPU KV cache.

Budget 64 tokens per step; a decoding request takes 1 of them. Steps are
scheduled back to back without outputs (async scheduling runs ahead), and the
scheduler's wall clock is a fake, so waits are deterministic.
"""

import pytest
from vllm import SamplingParams
from vllm.v1.request import Request

from tests.core.test_prefill_spacing_srpt import (  # noqa: F401
    T0,
    _build,
    clock,
    model_config,
)
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler

_KEYS = (
    "VLLM_TPU_PREFILL_DECODE_CADENCE",
    "VLLM_TPU_PREFILL_CADENCE_MAX_WAIT_S",
    "VLLM_TPU_PREFILL_COALESCE_TOKENS",
    "VLLM_TPU_PREFILL_COALESCE_MAX_WAIT_S",
)


def _sched(
    model_config,
    monkeypatch,
    *,
    cadence=0,
    coalesce=0,
    coalesce_wait=0.5,
    max_wait=10.0,
    srpt=False,
    aging=10.0,
    spacing_frac=0.0,
    **kw,
):
    monkeypatch.setenv("VLLM_TPU_PREFILL_DECODE_CADENCE", str(cadence))
    monkeypatch.setenv("VLLM_TPU_PREFILL_COALESCE_TOKENS", str(coalesce))
    monkeypatch.setenv("VLLM_TPU_PREFILL_COALESCE_MAX_WAIT_S", str(coalesce_wait))
    monkeypatch.setenv("VLLM_TPU_PREFILL_CADENCE_MAX_WAIT_S", str(max_wait))
    kw.setdefault("max_num_batched_tokens", 64)
    return _build(
        model_config,
        monkeypatch,
        srpt=srpt,
        aging=aging,
        spacing_frac=spacing_frac,
        **kw,
    )


def _req(rid, num_tokens, hasher, seed, arrival, max_tokens=200):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [seed * 100_000 + i for i in range(num_tokens)]
    r = Request(rid, prompt, params, None, block_hasher=hasher)
    r.arrival_time = arrival
    return r


def _step(s):
    return dict(s.schedule().num_scheduled_tokens)


def _with_decoder(model_config, monkeypatch, clock, **kw):
    """Scheduler with one decoding request ("dec") and nothing owed."""
    s, h = _sched(model_config, monkeypatch, **kw)
    s.add_request(_req("dec", 16, h, 1, clock.t))
    assert _step(s) == {"dec": 16}
    s._cadence_left = 0
    return s, h


def test_envs_default_off_and_skip_compile_cache(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
        assert k in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        assert k not in tpu_compiler._tpu_compile_env_factors()
    assert envs.VLLM_TPU_PREFILL_DECODE_CADENCE == 0
    assert envs.VLLM_TPU_PREFILL_COALESCE_TOKENS == 0
    assert envs.VLLM_TPU_PREFILL_CADENCE_MAX_WAIT_S == 10.0
    assert envs.VLLM_TPU_PREFILL_COALESCE_MAX_WAIT_S == 0.5


def test_env_vars_are_read(model_config, monkeypatch):
    s, _ = _sched(
        model_config, monkeypatch, cadence=24, coalesce=6144, coalesce_wait=0.3,
        max_wait=7.0,
    )
    assert (s.cadence, s.coalesce_tokens) == (24, 6144)
    assert (s.coalesce_max_wait_s, s.cadence_max_wait_s) == (0.3, 7.0)


def _mixed_replay(model_config, monkeypatch, clock, hold_raises=False, **kw):
    """Decoders, short and long arrivals, spacing on: every step's tokens."""
    s, h = _sched(
        model_config,
        monkeypatch,
        spacing_frac=0.7,
        max_num_batched_tokens=24,
        long_prefill_token_threshold=16,
        **kw,
    )
    if hold_raises:
        def boom(queue):
            raise AssertionError("decode-priority hold consulted while off")

        monkeypatch.setattr(s, "_decode_priority_hold", boom)
    steps = []
    s.add_request(_req("a", 64, h, 1, T0 - 3))
    s.add_request(_req("b", 16, h, 2, T0 - 2))
    for i in range(12):
        if i in (2, 5):
            s.add_request(_req(f"c{i}", 16, h, 3 + i, clock.t))
            s.add_request(_req(f"d{i}", 40, h, 4 + i, clock.t))
        steps.append(_step(s))
        clock.t += 0.1
    return steps


def test_key_off_identity(model_config, monkeypatch, clock):
    """Off (cadence 0, coalescing 0): the hold is never consulted and the other
    knobs do not matter; the schedule equals the defaults' schedule."""
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    s, h = _build(model_config, monkeypatch, srpt=False, spacing_frac=0.7,
                  max_num_batched_tokens=24, long_prefill_token_threshold=16)
    ref = []
    s.add_request(_req("a", 64, h, 1, T0 - 3))
    s.add_request(_req("b", 16, h, 2, T0 - 2))
    for i in range(12):
        if i in (2, 5):
            s.add_request(_req(f"c{i}", 16, h, 3 + i, clock.t))
            s.add_request(_req(f"d{i}", 40, h, 4 + i, clock.t))
        ref.append(_step(s))
        clock.t += 0.1
    clock.t = T0
    off = _mixed_replay(model_config, monkeypatch, clock, hold_raises=True,
                        coalesce_wait=0.01, max_wait=0.01)
    assert off == ref
    assert s._cadence_left == 0


def test_cadence_decode_steps_after_prefill(model_config, monkeypatch, clock):
    """N=3: a 48-token prefill step owes ceil(3 * 48 / 64) = 3 decode-only
    steps; a later arrival waits them out, then runs."""
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=3)
    s.add_request(_req("a", 48, h, 2, clock.t))
    assert _step(s) == {"dec": 1, "a": 48}
    assert s._cadence_left == 3
    s.add_request(_req("b", 16, h, 3, clock.t))
    for _ in range(3):
        assert _step(s) == {"dec": 1, "a": 1}  # a decodes now too
    assert _step(s) == {"dec": 1, "a": 1, "b": 16}
    assert s._cadence_left == 1  # ceil(3 * 16 / 64)
    assert s._cadence_steps == 3


def test_cadence_full_chunk_owes_n(model_config, monkeypatch, clock):
    """A long prompt alternates one full chunk (63 tokens) with N decode-only
    steps while a request decodes."""
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=2)
    s.add_request(_req("long", 200, h, 2, clock.t))
    pattern = [set(_step(s)) for _ in range(9)]
    assert pattern == [{"dec", "long"}, {"dec"}, {"dec"}] * 3


def test_cadence_skipped_while_nothing_decodes(model_config, monkeypatch, clock):
    s, h = _sched(model_config, monkeypatch, cadence=8)
    s.add_request(_req("long", 192, h, 1, clock.t))
    assert [_step(s) for _ in range(3)] == [{"long": 64}] * 3


@pytest.mark.parametrize(("waited", "runs"), [(5.0, False), (11.0, True)])
def test_ttft_bound_lets_old_prompt_through(
    model_config, monkeypatch, clock, waited, runs
):
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=8, max_wait=10.0)
    s._cadence_left = 8
    s.add_request(_req("old", 16, h, 2, clock.t - waited))
    assert ("old" in _step(s)) is runs
    assert s._bound_steps == int(runs)


def test_ttft_bound_off(model_config, monkeypatch, clock):
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=8, max_wait=0.0)
    s._cadence_left = 8
    s.add_request(_req("old", 16, h, 2, clock.t - 1000.0))
    assert "old" not in _step(s)


def test_coalesce_holds_until_full(model_config, monkeypatch, clock):
    """Coalesce 48: short prompts wait until 48 pending tokens, then run in one
    step."""
    s, h = _with_decoder(model_config, monkeypatch, clock, coalesce=48)
    for i in range(3):
        s.add_request(_req(f"s{i}", 16, h, 10 + i, clock.t))
        out = _step(s)
        clock.t += 0.05
        if i < 2:
            assert out == {"dec": 1}
    assert out == {"dec": 1, "s0": 16, "s1": 16, "s2": 16}
    assert s._coalesce_steps == 2


def test_coalesce_max_wait(model_config, monkeypatch, clock):
    s, h = _with_decoder(model_config, monkeypatch, clock, coalesce=48,
                         coalesce_wait=0.5)
    s.add_request(_req("s", 16, h, 2, clock.t))
    assert _step(s) == {"dec": 1}
    clock.t += 0.6
    assert _step(s) == {"dec": 1, "s": 16}


def test_coalesce_counts_running_chunk(model_config, monkeypatch, clock):
    """A running chunk's tokens left count toward the fill: a long prompt keeps
    chunking without coalescing holds."""
    s, h = _with_decoder(model_config, monkeypatch, clock, coalesce=48)
    s.add_request(_req("long", 200, h, 2, clock.t))
    for _ in range(3):
        assert set(_step(s)) == {"dec", "long"}
    assert s._coalesce_steps == 0


def test_coalesce_idle_runs_at_once(model_config, monkeypatch, clock):
    s, h = _sched(model_config, monkeypatch, coalesce=48)
    s.add_request(_req("s", 16, h, 2, clock.t))
    assert _step(s) == {"s": 16}


def test_cadence_then_coalesce(model_config, monkeypatch, clock):
    """After the cadence is paid, coalescing still holds a short batch."""
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=1, coalesce=32)
    s._cadence_left = 1
    s.add_request(_req("s0", 16, h, 2, clock.t))
    assert _step(s) == {"dec": 1}  # cadence
    assert _step(s) == {"dec": 1}  # coalesce: 16 < 32
    s.add_request(_req("s1", 16, h, 3, clock.t))
    assert _step(s) == {"dec": 1, "s0": 16, "s1": 16}
    assert (s._cadence_steps, s._coalesce_steps) == (1, 1)


@pytest.mark.parametrize(("srpt", "first"), [(False, "short"), (True, "long")])
def test_srpt_picks_which_after_cadence(
    model_config, monkeypatch, clock, srpt, first
):
    """The cadence decides whether a step runs prefill, SRPT which prefill:
    the aged long prompt (48 - 10 * 5 = -2) goes ahead of a fresh 16."""
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=1, srpt=srpt)
    s._cadence_left = 1
    s.add_request(_req("long", 48, h, 2, clock.t - 5))
    s.add_request(_req("short", 16, h, 3, clock.t))
    assert _step(s) == {"dec": 1}
    out = _step(s)
    prefills = [r for r in out if r != "dec"]
    assert prefills[0] == first


def test_spacing_debt_and_cadence_both_hold(model_config, monkeypatch, clock):
    """Spacing debt defers on its own; once it is paid the cadence still owes
    steps; decode-only steps pay both."""
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=2,
                         spacing_frac=0.7)
    s._debt_s = 1.0
    s._step_s = {"decode": 0.6}
    s._cadence_left = 3
    s.add_request(_req("p", 16, h, 2, clock.t))
    assert _step(s) == {"dec": 1}  # spacing (cadence not consulted)
    assert s._cadence_left == 2 and s._cadence_steps == 0
    assert _step(s) == {"dec": 1}  # spacing, debt 0.4 -> 0
    assert s._debt_s == 0.0
    assert _step(s) == {"dec": 1}  # cadence
    assert _step(s) == {"dec": 1, "p": 16}


def test_cadence_blocks_priced_fit(model_config, monkeypatch, clock):
    """Under the priced TPOT hold a cadence step is decode-only (no fit)."""
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=1)
    s.tpot_target_s = 0.02
    s.priced = True
    monkeypatch.setattr(s, "_hold_for_tpot", lambda decoders, queue: True)
    monkeypatch.setattr(s, "_fit_tokens", lambda: 32)
    s._cadence_left = 1
    s.add_request(_req("short", 16, h, 2, clock.t))
    assert _step(s) == {"dec": 1}
    assert _step(s) == {"dec": 1, "short": 16}  # priced fit once cadence paid


def test_ttft_bound_ignores_running_chunks(model_config, monkeypatch, clock):
    """An old prompt already chunking keeps the cadence between its chunks;
    the bound acts only for prompts not started yet."""
    s, h = _with_decoder(model_config, monkeypatch, clock, cadence=2, max_wait=10.0)
    s.add_request(_req("long", 400, h, 2, clock.t - 20.0))
    s.requests["long"].arrival_time = clock.t - 20.0
    pattern = [set(_step(s)) for _ in range(6)]
    assert pattern == [{"dec", "long"}, {"dec"}, {"dec"}] * 2
    s._cadence_left = 2
    s.add_request(_req("old", 16, h, 3, clock.t - 11.0))
    assert "old" in _step(s)
