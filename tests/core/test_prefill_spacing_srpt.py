# SPDX-License-Identifier: Apache-2.0
"""PrefillSpacingScheduler with VLLM_TPU_PREFILL_SRPT: aged shortest remaining
prefill first over the waiting window, the running prefill chunks and the
reserve, on a real (async) vLLM scheduler over a CPU KV cache.

The scheduler's wall clock is a fake (``pss.time``) and every request's
``arrival_time`` is set explicitly, so aging is deterministic.
"""

import time
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import prefill_spacing_scheduler as pss
from vllm_torchtpu.core.prefill_spacing_scheduler import PrefillSpacingScheduler

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"
T0 = 1_000_000.0


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


@pytest.fixture
def clock(monkeypatch):
    """Fake wall clock for the scheduler (monotonic stays real)."""
    now = SimpleNamespace(t=T0)
    fake = SimpleNamespace(time=lambda: now.t, monotonic=time.monotonic)
    monkeypatch.setattr(pss, "time", fake)
    return now


def _build(
    model_config,
    monkeypatch,
    *,
    srpt: bool,
    aging: float = 10.0,
    spacing_frac: float = 0.0,
    num_blocks: int = 64,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    long_prefill_token_threshold: int = 0,
):
    monkeypatch.setenv("VLLM_TPU_PREFILL_SRPT", "1" if srpt else "0")
    monkeypatch.setenv("VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S", str(aging))
    monkeypatch.setenv("VLLM_TPU_PREFILL_SPACING_FRAC", str(spacing_frac))
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = PrefillSpacingScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(BLOCK_SIZE, sha256)


def _req(rid: str, num_tokens: int, hasher, seed: int, arrival: float) -> Request:
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [seed * 100_000 + i for i in range(num_tokens)]
    r = Request(rid, prompt, params, None, block_hasher=hasher)
    r.arrival_time = arrival
    return r


def test_envs_default_off_and_skip_compile_cache(monkeypatch):
    for k in ("VLLM_TPU_PREFILL_SRPT", "VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S"):
        monkeypatch.delenv(k, raising=False)
        assert k in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        assert k not in tpu_compiler._tpu_compile_env_factors()
    assert envs.VLLM_TPU_PREFILL_SRPT is False
    assert envs.VLLM_TPU_PREFILL_SRPT_AGING_TOKENS_PER_S == 256.0


def test_env_vars_are_read(model_config, monkeypatch):
    s, _ = _build(model_config, monkeypatch, srpt=True, aging=2048.0)
    assert s.srpt is True
    assert s.srpt_aging == 2048.0


def _replay(model_config, monkeypatch, clock, **kw):
    """A fixed mixed scenario; returns every step's scheduled tokens."""
    s, h = _build(
        model_config,
        monkeypatch,
        max_num_batched_tokens=24,
        long_prefill_token_threshold=16,
        **kw,
    )
    steps = []
    s.add_request(_req("a", 64, h, 1, T0 - 3))
    s.add_request(_req("b", 48, h, 2, T0 - 2))
    for i in range(6):
        if i == 2:
            s.add_request(_req("c", 16, h, 3, clock.t))
            s.add_request(_req("d", 40, h, 4, clock.t))
        steps.append(dict(s.schedule().num_scheduled_tokens))
        clock.t += 0.1
    return steps


def test_aging_zero_matches_off(model_config, monkeypatch, clock):
    """With no aging the SRPT key is the tokens left: same schedule as off."""
    off = _replay(model_config, monkeypatch, clock, srpt=False)
    clock.t = T0
    on = _replay(model_config, monkeypatch, clock, srpt=True, aging=0.0)
    assert on == off


def test_waiting_window_shortest_uncached_first(model_config, monkeypatch, clock):
    """Fresh requests: the shortest uncached prompt is admitted first, even
    behind an older longer one (aging too small to matter)."""
    s, h = _build(
        model_config, monkeypatch, srpt=True, aging=1.0, max_num_batched_tokens=16
    )
    s.add_request(_req("long", 48, h, 1, T0 - 1))
    s.add_request(_req("short", 16, h, 2, T0))
    assert s.schedule().num_scheduled_tokens == {"short": 16}


@pytest.mark.parametrize(("srpt", "first"), [(False, "short"), (True, "long")])
def test_aged_long_prompt_goes_first(model_config, monkeypatch, clock, srpt, first):
    """48 tokens waiting 10 s at 10 tok/s keys at -52, ahead of a fresh 16."""
    s, h = _build(
        model_config, monkeypatch, srpt=srpt, aging=10.0, max_num_batched_tokens=16
    )
    s.add_request(_req("long", 48, h, 1, T0 - 10))
    s.add_request(_req("short", 16, h, 2, T0))
    assert set(s.schedule().num_scheduled_tokens) == {first}


@pytest.mark.parametrize(("srpt", "expected"), [(False, False), (True, True)])
def test_no_starvation_under_stream_of_short_prompts(
    model_config, monkeypatch, clock, srpt, expected
):
    """Four fresh 16-token prompts each step fill the 64-token budget; they
    finish at once. Off, the 48-token prompt never runs; with aging at
    10 tok/s it runs once its key, 48 - 10 * wait, drops below a fresh
    short's 16: after (48 - 16) / 10 = 3.2 s, i.e. step 33 at 0.1 s/step."""
    s, h = _build(
        model_config, monkeypatch, srpt=srpt, aging=10.0, max_num_batched_tokens=64
    )
    s.add_request(_req("long", 48, h, 1, T0))
    ran_at = None
    for step in range(60):
        for j in range(4):
            s.add_request(_req(f"s{step}_{j}", 16, h, 1000 + 4 * step + j, clock.t))
        out = s.schedule().num_scheduled_tokens
        if "long" in out:
            ran_at = step
            break
        assert len(out) == 4
        s.finish_requests(list(out), RequestStatus.FINISHED_ABORTED)
        clock.t += 0.1
    assert (ran_at is not None) == expected
    if expected:
        assert 31 <= ran_at <= 34


def test_running_chunks_ordered_by_aged_key(model_config, monkeypatch, clock):
    """Two running chunks: off, fewest tokens left first; with aging the older
    chunk with more left goes first once its credit covers the difference."""

    def run(srpt):
        s, h = _build(
            model_config,
            monkeypatch,
            srpt=srpt,
            aging=10.0,
            max_num_batched_tokens=24,
            long_prefill_token_threshold=16,
        )
        s.add_request(_req("old", 64, h, 1, T0 - 5))  # 48 left, key -2
        s.add_request(_req("new", 48, h, 2, T0))  # 40 left, key 40
        first = s.schedule().num_scheduled_tokens
        return first, s.schedule().num_scheduled_tokens

    first_off, off = run(False)
    assert first_off == {"new": 16, "old": 8}  # waiting window: shortest first
    assert off == {"new": 16, "old": 8}  # running: new has 32 left vs old 56
    first_on, on = run(True)
    assert first_on == {"old": 16, "new": 8}
    assert on == {"old": 16, "new": 8}


def test_async_ranks_by_tokens_left_after_inflight_chunk(
    model_config, monkeypatch, clock
):
    """Async scheduling advances num_computed_tokens at schedule time, so the
    next step ranks a chunk by what is left after the chunk still in flight
    (no update_from_output between the steps)."""
    s, h = _build(
        model_config,
        monkeypatch,
        srpt=True,
        aging=0.0,
        max_num_batched_tokens=48,
    )
    assert isinstance(s, AsyncScheduler)
    s.add_request(_req("big", 64, h, 1, T0))
    assert s.schedule().num_scheduled_tokens == {"big": 48}  # 16 left in flight
    s.add_request(_req("mid", 32, h, 2, T0))
    # big (16 left) goes before mid (32), not by prompt length.
    out = s.schedule().num_scheduled_tokens
    assert list(out) == ["big", "mid"]
    assert out == {"big": 16, "mid": 32}


@pytest.mark.parametrize(("srpt", "big_tokens"), [(False, 64), (True, 8)])
def test_reserve_uses_aged_key(model_config, monkeypatch, clock, srpt, big_tokens):
    """A running chunk leaves budget only to waiting requests that rank ahead
    of it: off, by tokens left (56 > 48: no reserve); with aging the 56-token
    request that waited 10 s (key -44) outranks the fresh chunk (key 48)."""
    s, h = _build(
        model_config,
        monkeypatch,
        srpt=srpt,
        aging=10.0,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=0,
    )
    s.add_request(_req("big", 112, h, 1, T0))
    assert s.schedule().num_scheduled_tokens == {"big": 64}  # 48 left
    s.add_request(_req("old", 56, h, 2, T0 - 10))
    out = s.schedule().num_scheduled_tokens
    assert out.get("big", 0) == min(big_tokens, 48)
    if srpt:
        assert out["old"] == 56


def _decoding(model_config, monkeypatch, clock, **kw):
    """A scheduler with one decoding request and two waiting prompts."""
    s, h = _build(model_config, monkeypatch, max_num_batched_tokens=64, **kw)
    s.add_request(_req("dec", 16, h, 1, T0))
    s.schedule()
    s.add_request(_req("long", 48, h, 2, T0 - 10))
    s.add_request(_req("short", 16, h, 3, T0))
    return s


@pytest.mark.parametrize("srpt", [False, True])
def test_spacing_debt_still_defers_all_prefill(model_config, monkeypatch, clock, srpt):
    """SRPT picks which prefill runs, never whether: with decode-only debt
    owed, neither the aged long prompt nor the short one is admitted."""
    s = _decoding(model_config, monkeypatch, clock, srpt=srpt, spacing_frac=0.7)
    s._debt_s = 1.0
    out = s.schedule().num_scheduled_tokens
    assert set(out) == {"dec"}
    s._debt_s = 0.0
    out = s.schedule().num_scheduled_tokens
    assert "dec" in out and {"long", "short"} <= set(out)


@pytest.mark.parametrize("srpt", [False, True])
def test_tpot_hold_still_defers_all_prefill(model_config, monkeypatch, clock, srpt):
    s = _decoding(model_config, monkeypatch, clock, srpt=srpt)
    s.tpot_target_s = 0.02
    monkeypatch.setattr(s, "_hold_for_tpot", lambda decoders, queue: True)
    assert set(s.schedule().num_scheduled_tokens) == {"dec"}


@pytest.mark.parametrize(
    ("srpt", "expected"), [(False, {"dec", "short"}), (True, {"dec"})]
)
def test_priced_hold_runs_fits_in_srpt_order(
    model_config, monkeypatch, clock, srpt, expected
):
    """Priced TPOT hold runs whole prefills that fit the bin, in scheduler
    order. With SRPT the aged long prompt (48 > 32-token bin) is first and the
    base waiting loop stops at it, so the short one waits too: aging keeps a
    starved prompt ahead of fits until a full prefill step runs."""
    s = _decoding(model_config, monkeypatch, clock, srpt=srpt)
    s.tpot_target_s = 0.02
    s.priced = True
    monkeypatch.setattr(s, "_hold_for_tpot", lambda decoders, queue: True)
    monkeypatch.setattr(s, "_fit_tokens", lambda: 32)
    assert set(s.schedule().num_scheduled_tokens) == expected
