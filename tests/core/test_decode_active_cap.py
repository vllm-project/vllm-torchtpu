# SPDX-License-Identifier: Apache-2.0
"""PrefillSpacingScheduler decode active cap (VLLM_TPU_DECODE_ACTIVE_CAP /
_AHEAD) on a real async vLLM scheduler over a CPU KV cache.

Budget 64 tokens per step; a decoding request takes 1 of them (no spec
decode, prefill lookahead 1, so a held request parks 1 token short of its
prompt end). Steps are scheduled back to back without outputs (async
scheduling runs ahead); a decoding request leaves by being aborted.
"""

import pytest
from vllm import SamplingParams
from vllm.v1.request import Request, RequestStatus

from tests.core.test_prefill_spacing_srpt import (  # noqa: F401
    T0,
    _build,
    clock,
    model_config,
)
from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler

_KEYS = ("VLLM_TPU_DECODE_ACTIVE_CAP", "VLLM_TPU_DECODE_ACTIVE_CAP_AHEAD")


def _sched(model_config, monkeypatch, *, cap=0, ahead=None, cadence=0, **kw):
    monkeypatch.setenv("VLLM_TPU_DECODE_ACTIVE_CAP", str(cap))
    if ahead is None:
        monkeypatch.delenv("VLLM_TPU_DECODE_ACTIVE_CAP_AHEAD", raising=False)
    else:
        monkeypatch.setenv("VLLM_TPU_DECODE_ACTIVE_CAP_AHEAD", str(ahead))
    monkeypatch.setenv("VLLM_TPU_PREFILL_DECODE_CADENCE", str(cadence))
    monkeypatch.setenv("VLLM_TPU_PREFILL_CADENCE_MAX_WAIT_S", "10")
    kw.setdefault("srpt", False)
    return _build(model_config, monkeypatch, **kw)


def _req(rid, num_tokens, hasher, seed, arrival, max_tokens=200):
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [seed * 100_000 + i for i in range(num_tokens)]
    r = Request(rid, prompt, params, None, block_hasher=hasher)
    r.arrival_time = arrival
    return r


def _step(s):
    return dict(s.schedule().num_scheduled_tokens)


def _leave(s, *rids):
    s.finish_requests(list(rids), RequestStatus.FINISHED_ABORTED)


def test_envs_default_off_and_skip_compile_cache(monkeypatch):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
        assert k in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        assert k not in tpu_compiler._tpu_compile_env_factors()
    assert envs.VLLM_TPU_DECODE_ACTIVE_CAP == 0
    assert envs.VLLM_TPU_DECODE_ACTIVE_CAP_AHEAD == -1


def _replay(s, h, clock):
    steps = []
    s.add_request(_req("a", 64, h, 1, T0 - 3))
    s.add_request(_req("b", 16, h, 2, T0 - 2))
    for i in range(14):
        if i in (2, 5, 9):
            s.add_request(_req(f"c{i}", 16, h, 3 + i, clock.t))
            s.add_request(_req(f"d{i}", 40, h, 4 + i, clock.t))
        if i == 7:
            _leave(s, "b")
        steps.append(_step(s))
        clock.t += 0.1
    return steps


@pytest.mark.parametrize("srpt", [False, True])
def test_key_off_identity(model_config, monkeypatch, clock, srpt):
    """Unset, and a cap no larger than max_num_seqs never reaches, give the
    default schedule step for step."""
    kw = dict(
        srpt=srpt,
        spacing_frac=0.7,
        max_num_batched_tokens=24,
        long_prefill_token_threshold=16,
        cadence=2,
    )
    runs = []
    for cap in (0, 8):
        clock.t = T0
        s, h = _sched(model_config, monkeypatch, cap=cap, **kw)
        runs.append(_replay(s, h, clock))
    assert runs[0] == runs[1]
    assert s._dcap_gated_steps == 0


def test_request_past_cap_parks_one_token_short(model_config, monkeypatch, clock):
    """Cap 1, one request decoding: a new 48-token prompt prefills 47 tokens,
    then waits; it runs its last token (first decode) the step after the
    decoder leaves."""
    s, h = _sched(model_config, monkeypatch, cap=1)
    s.add_request(_req("dec", 16, h, 1, clock.t))
    assert _step(s) == {"dec": 16}
    s.add_request(_req("a", 48, h, 2, clock.t))
    assert _step(s) == {"dec": 1, "a": 47}
    for _ in range(3):
        assert _step(s) == {"dec": 1}
    assert s._dcap_gated_steps == 4
    _leave(s, "dec")
    assert _step(s) == {"a": 1}
    assert _step(s) == {"a": 1}  # decoding now
    s.add_request(_req("b", 16, h, 3, clock.t))
    assert _step(s) == {"a": 1, "b": 15}


def test_cap_counts_requests_let_in_this_step(model_config, monkeypatch, clock):
    """Cap 2, nothing decoding: of three short prompts in one step, two run
    to their end and the third parks."""
    s, h = _sched(model_config, monkeypatch, cap=2)
    for i, rid in enumerate(("x", "y", "z")):
        s.add_request(_req(rid, 16, h, i + 1, clock.t))
    assert _step(s) == {"x": 16, "y": 16, "z": 15}
    assert _step(s) == {"x": 1, "y": 1}
    _leave(s, "x")
    assert _step(s) == {"y": 1, "z": 1}


def test_long_prompt_chunks_then_parks(model_config, monkeypatch, clock):
    """A multi-chunk prompt past the cap still prefills chunk by chunk and
    only its end is held."""
    s, h = _sched(model_config, monkeypatch, cap=1)
    s.add_request(_req("dec", 16, h, 1, clock.t))
    _step(s)
    s.add_request(_req("long", 160, h, 2, clock.t))
    got = [_step(s).get("long", 0) for _ in range(4)]
    assert got == [63, 63, 33, 0]
    assert s.requests["long"].num_computed_tokens == 159


@pytest.mark.parametrize("srpt", [False, True])
def test_parked_finish_oldest_first(model_config, monkeypatch, clock, srpt):
    """Two parked requests, one slot frees: the older one decodes first."""
    s, h = _sched(model_config, monkeypatch, cap=1, srpt=srpt)
    s.add_request(_req("dec", 16, h, 1, clock.t))
    _step(s)
    s.add_request(_req("new", 16, h, 2, T0 - 1))
    s.add_request(_req("old", 32, h, 3, T0 - 5))
    out = _step(s)
    assert out["new"] == 15 and out["old"] == 31
    _leave(s, "dec")
    assert _step(s) == {"old": 1}
    _leave(s, "old")
    assert _step(s) == {"new": 1}


def test_parked_gets_no_reserve_cut(model_config, monkeypatch, clock):
    """SRPT reserve: a parked request is never cut for a waiting request with
    a smaller aged key; it finishes as soon as a slot frees."""
    s, h = _sched(model_config, monkeypatch, cap=1, srpt=True, aging=100.0)
    s.add_request(_req("dec", 16, h, 1, clock.t))
    _step(s)
    s.add_request(_req("p", 16, h, 2, clock.t))
    assert _step(s)["p"] == 15
    s.add_request(_req("w", 8, h, 3, T0 - 100))  # key 8 - 1e4: ranks first
    _leave(s, "dec")
    out = _step(s)
    assert out["p"] == 1
    assert "p" not in s._reserve


def test_tail_only_step_inside_cadence(model_config, monkeypatch, clock):
    """Cadence owes decode-only steps; a decode slot frees: the step runs
    only the parked request's last token, a waiting prompt stays deferred,
    and the step still counts as decode-only for the cadence."""
    s, h = _sched(model_config, monkeypatch, cap=2, cadence=4)
    s.add_request(_req("d1", 16, h, 1, clock.t))
    s.add_request(_req("d2", 16, h, 2, clock.t))
    assert _step(s) == {"d1": 16, "d2": 16}
    s._cadence_left = 0
    s.add_request(_req("p", 32, h, 3, clock.t))
    assert _step(s) == {"d1": 1, "d2": 1, "p": 31}
    assert s._cadence_left == 2  # ceil(4 * 31 / 64)
    s.add_request(_req("w", 16, h, 4, clock.t))
    _leave(s, "d2")
    assert _step(s) == {"d1": 1, "p": 1}
    assert s._cadence_left == 1
    assert s._dcap_tail_steps == 1
    assert _step(s) == {"d1": 1, "p": 1}  # w waits out the cadence
    assert s._cadence_left == 0
    assert _step(s) == {"d1": 1, "p": 1, "w": 15}  # cap 2 full: w parks


def test_ahead_zero_caps_running(model_config, monkeypatch, clock):
    """AHEAD=0: RUNNING admission stops at the cap (no prefill ahead)."""
    s, h = _sched(model_config, monkeypatch, cap=1, ahead=0)
    assert s.max_num_active_reqs == 1
    s.add_request(_req("dec", 16, h, 1, clock.t))
    _step(s)
    s.add_request(_req("a", 16, h, 2, clock.t))
    assert _step(s) == {"dec": 1}
    _leave(s, "dec")
    assert _step(s) == {"a": 16}


def test_ahead_limits_prefill_ahead(model_config, monkeypatch, clock):
    s, h = _sched(model_config, monkeypatch, cap=1, ahead=1)
    assert s.max_num_active_reqs == 2
    s.add_request(_req("dec", 16, h, 1, clock.t))
    _step(s)
    s.add_request(_req("a", 16, h, 2, clock.t))
    s.add_request(_req("b", 16, h, 3, clock.t))
    out = _step(s)
    assert out == {"dec": 1, "a": 15}


def test_fully_cached_request_does_not_block_window(model_config, monkeypatch, clock):
    """Cap full: a waiting request with nothing to prefill (its prompt is
    cached) would end the waiting pass; it goes behind a prompt that can
    still prefill ahead."""
    s, h = _sched(model_config, monkeypatch, cap=1)
    s.add_request(_req("x", 33, h, 5, clock.t))
    _step(s)
    _step(s)
    _leave(s, "x")
    s.add_request(_req("dec", 16, h, 1, clock.t))
    assert _step(s) == {"dec": 16}
    s.add_request(_req("x2", 33, h, 5, clock.t))  # 32 tokens cached
    s.add_request(_req("y", 48, h, 6, clock.t))
    out = _step(s)
    assert out == {"dec": 1, "y": 47}


def test_resumed_request_not_held(model_config, monkeypatch, clock):
    """A request that already has output tokens is never held."""
    s, h = _sched(model_config, monkeypatch, cap=1)
    s.add_request(_req("dec", 16, h, 1, clock.t))
    s.schedule()
    r = _req("r", 32, h, 2, clock.t)
    r.append_output_token_ids([7])
    s._in_schedule = True
    try:
        assert s._dcap_gate(r, 0, 33) == 33
        fresh = _req("f", 32, h, 3, clock.t)
        assert s._dcap_gate(fresh, 0, 32) == 31
    finally:
        s._in_schedule = False


def test_parked_not_pending_for_ttft_bound(model_config, monkeypatch, clock):
    s, h = _sched(model_config, monkeypatch, cap=1)
    s.add_request(_req("dec", 16, h, 1, clock.t))
    _step(s)
    s.add_request(_req("p", 16, h, 2, clock.t - 100))
    _step(s)
    assert s._dcap_parked(s.requests["p"])
    assert s._oldest_pending_arrival(s.waiting) is None
