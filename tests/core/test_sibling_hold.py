# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SIBLING_HOLD: in-batch shared-prefix dedupe on a real (async) vLLM
scheduler over a CPU KV cache.

A leader prompt is prefilled in 16-token chunks; a sibling sharing its first
80 tokens arrives while the leader is still prefilling. Off, the sibling is
admitted at once and prefills the rest of the shared prefix itself; on, it is
held until the leader has scheduled the shared blocks and then starts from an
80-token hit. Both wall clocks (``pas.time``, ``pss.time``) are fakes.
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
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core import prefill_spacing_scheduler as pss
from vllm_torchtpu.core import prefix_aware_scheduler as pas
from vllm_torchtpu.core.prefill_spacing_scheduler import PrefillSpacingScheduler
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"
T0 = 1_000_000.0
SHARED = 80
_ENVS = ("VLLM_TPU_SIBLING_HOLD", "VLLM_TPU_SIBLING_HOLD_MAX_S")


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


@pytest.fixture
def clock(monkeypatch):
    now = SimpleNamespace(t=T0)
    fake = SimpleNamespace(time=lambda: now.t, monotonic=time.monotonic)
    monkeypatch.setattr(pss, "time", fake)
    monkeypatch.setattr(pas, "time", fake)
    return now


def _build(
    model_config,
    monkeypatch,
    *,
    hold,
    max_s=4.0,
    cls=PrefillSpacingScheduler,
    max_num_batched_tokens=32,
    long_prefill_token_threshold=16,
    srpt=False,
):
    for k in _ENVS:
        monkeypatch.delenv(k, raising=False)
    if hold is not None:
        monkeypatch.setenv("VLLM_TPU_SIBLING_HOLD", "1" if hold else "0")
        monkeypatch.setenv("VLLM_TPU_SIBLING_HOLD_MAX_S", str(max_s))
    monkeypatch.setenv("VLLM_TPU_PREFILL_SRPT", "1" if srpt else "0")
    monkeypatch.setenv("VLLM_TPU_PREFILL_SPACING_FRAC", "0")
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
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
    num_blocks = 128
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
    scheduler = cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(BLOCK_SIZE, sha256)


def _req(rid, hasher, tail_seed, *, shared=SHARED, tail=16, prefix_seed=7):
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prompt = [prefix_seed * 100_000 + i for i in range(shared)]
    prompt += [tail_seed * 100_000 + i for i in range(tail)]
    r = Request(rid, prompt, params, None, block_hasher=hasher)
    r.arrival_time = T0
    return r


def _run(s, h, steps, add):
    """Schedule ``steps`` steps; ``add[step]`` requests arrive before it.
    Returns per-step scheduled tokens and each request's computed tokens at
    its first scheduled step (its prefix hit)."""
    out, first_hit = [], {}
    for step in range(steps):
        for r in add.get(step, ()):
            s.add_request(r)
        so = s.schedule()
        for nr in so.scheduled_new_reqs:
            first_hit[nr.req_id] = nr.num_computed_tokens
        out.append(dict(so.num_scheduled_tokens))
    return out, first_hit


def test_envs_default_off_and_skip_compile_cache(monkeypatch):
    for k in _ENVS:
        monkeypatch.delenv(k, raising=False)
        assert k in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        assert k not in tpu_compiler._tpu_compile_env_factors()
    assert envs.VLLM_TPU_SIBLING_HOLD is False
    assert envs.VLLM_TPU_SIBLING_HOLD_MAX_S == 4.0


@pytest.mark.parametrize("cls", [PrefillSpacingScheduler, PrefixAwareScheduler])
def test_key_off_identical(model_config, monkeypatch, clock, cls):
    """Unset and =0 give the same schedule, step for step."""
    runs = []
    for hold in (None, False):
        s, h = _build(model_config, monkeypatch, hold=hold, cls=cls)
        assert s._sib_hold is False
        add = {0: [_req("lead", h, 1)], 1: [_req("sib", h, 2)]}
        runs.append(_run(s, h, 8, add))
    assert runs[0] == runs[1]


@pytest.mark.parametrize("cls", [PrefillSpacingScheduler, PrefixAwareScheduler])
def test_sibling_waits_for_leader_prefix(model_config, monkeypatch, clock, cls):
    """Leader 96 tokens: 32 at step 0 (alone), then 16 per step, so it
    schedules the 80 shared tokens by step 3. Blocks are hits from the step
    after they are scheduled. Off, the sibling starts at step 1 from a
    32-token hit and prefills 64 tokens; on, it starts at step 4 from the
    80-token hit and prefills only its own 16."""
    s, h = _build(model_config, monkeypatch, hold=False, cls=cls)
    out, hit = _run(s, h, 6, {0: [_req("lead", h, 1)], 1: [_req("sib", h, 2)]})
    assert "sib" in out[1]
    assert hit["sib"] == 32  # recomputes 64 of the leader's prompt

    s, h = _build(model_config, monkeypatch, hold=True, cls=cls)
    out, hit = _run(s, h, 6, {0: [_req("lead", h, 1)], 1: [_req("sib", h, 2)]})
    assert all("sib" not in o for o in out[:4])
    assert out[4] == {"lead": 16, "sib": 16}
    assert hit["sib"] == SHARED
    # The leader is never slowed down by the held sibling.
    assert [o["lead"] for o in out[:5]] == [32, 16, 16, 16, 16]
    assert s._sib_holds == 3


def test_same_step_arrival_admits_only_leader(model_config, monkeypatch, clock):
    """Both arrive together: the first admitted becomes the leader, the other
    is held although the budget has room for it (same-step blocks are not
    hits, so admitting it would prefill the leader's chunk twice)."""
    s, h = _build(model_config, monkeypatch, hold=True)
    out, hit = _run(s, h, 6, {0: [_req("lead", h, 1), _req("sib", h, 2)]})
    assert out[0] == {"lead": 16}
    assert all("sib" not in o for o in out[:5])
    assert hit["sib"] == SHARED


def test_unrelated_request_not_held(model_config, monkeypatch, clock):
    s, h = _build(model_config, monkeypatch, hold=True)
    other = _req("other", h, 3, prefix_seed=9)
    out, hit = _run(s, h, 3, {0: [_req("lead", h, 1)], 1: [other]})
    assert out[1] == {"lead": 16, "other": 16}
    assert hit["other"] == 0
    assert s._sib_holds == 0


def test_hold_is_bounded(model_config, monkeypatch, clock):
    """Past VLLM_TPU_SIBLING_HOLD_MAX_S the sibling is admitted while the
    leader still has shared blocks to prefill (counted once as expired)."""
    s, h = _build(model_config, monkeypatch, hold=True, max_s=1.0)
    add = {0: [_req("lead", h, 1)], 1: [_req("sib", h, 2)]}
    out = []
    first_hit = {}
    for step in range(5):
        for r in add.get(step, ()):
            s.add_request(r)
        so = s.schedule()
        for nr in so.scheduled_new_reqs:
            first_hit[nr.req_id] = nr.num_computed_tokens
        out.append(dict(so.num_scheduled_tokens))
        clock.t += 0.6
    # Held at step 1 (t=0.6) and 2 (t=1.2, 0.6 s held); admitted at step 3
    # (1.2 s held) from the 64 tokens the leader scheduled by step 2.
    assert "sib" not in out[1] and "sib" not in out[2]
    assert "sib" in out[3]
    assert first_hit["sib"] == 64
    assert s._sib_expired == 1


def test_connector_load_not_held(model_config, monkeypatch, clock):
    """A block the host tier can load costs a DMA, not a prefill: no hold."""
    s, h = _build(model_config, monkeypatch, hold=True)

    class Probe:
        def peek_external_prefix_tokens(self, request, local):
            return request.num_tokens - local

    s._paw_probe = Probe()
    out, _ = _run(s, h, 2, {0: [_req("lead", h, 1)], 1: [_req("sib", h, 2)]})
    assert "sib" in out[1]
    assert s._sib_holds == 0


def test_held_sibling_reserves_no_budget(model_config, monkeypatch, clock):
    """PSS reserve: off, the shorter waiting sibling (aged SRPT key) cuts the
    leader's chunk; on, the held sibling is out of the window's admissible
    list, and the leader keeps its whole chunk."""
    kw = dict(max_num_batched_tokens=32, long_prefill_token_threshold=0, srpt=True)
    for hold, lead_tokens in ((False, 0), (True, 32)):
        s, h = _build(model_config, monkeypatch, hold=hold, **kw)
        s.add_request(_req("lead", h, 1, shared=128, tail=16))
        assert s.schedule().num_scheduled_tokens == {"lead": 32}
        sib = _req("sib", h, 2, shared=128, tail=0)
        sib.arrival_time = T0 - 100  # aged: ranks ahead of the leader's chunk
        s.add_request(sib)
        out = s.schedule().num_scheduled_tokens
        assert out.get("lead", 0) == lead_tokens
        assert ("sib" in out) is not hold
        assert (s._window_cold == []) is hold


def test_preempted_or_resumed_not_held(model_config, monkeypatch, clock):
    s, h = _build(model_config, monkeypatch, hold=True)
    s.add_request(_req("lead", h, 1))
    s.schedule()
    sib = _req("sib", h, 2)
    sib.num_preemptions = 1
    assert not s._sibling_leader_pending(sib)
    sib.num_preemptions = 0
    assert s._sibling_leader_pending(sib)
