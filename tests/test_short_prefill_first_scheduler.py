import json

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.core.short_prefill_first_scheduler import (
    ShortPrefillFirstScheduler,
)
from vllm_torchtpu.core.short_prompts_first import fit_short_prompts

BUDGET = 4096
BLOCK = 16
SHORT = 2048


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-llama")
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32000,
                "max_position_embeddings": 131072,
                "torch_dtype": "bfloat16",
            }
        )
    )
    return str(path)


def _scheduler(
    cls, model_dir, monkeypatch, short=SHORT, max_seqs=16, num_blocks=40_000
):
    monkeypatch.setenv("TPU_SHORT_PREFILL_FIRST_TOKENS", str(short))
    model_config = ModelConfig(
        model=model_dir, skip_tokenizer_init=True, max_model_len=131072
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=BUDGET,
            max_num_seqs=max_seqs,
            max_model_len=131072,
            enable_chunked_prefill=True,
            async_scheduling=True,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK, cache_dtype="auto", enable_prefix_caching=True
        ),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["l0"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    config.cache_config.num_gpu_blocks = num_blocks
    return cls(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=BLOCK,
    )


init_none_hash(sha256)
_HASHER = get_request_block_hasher(BLOCK, sha256)


def _request(req_id, prompt, first_token=0, max_tokens=4):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(first_token, first_token + prompt)),
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        block_hasher=_HASHER,
    )


def _step(scheduler):
    output = scheduler.schedule()
    scheduled = dict(output.num_scheduled_tokens)
    req_ids = list(scheduled)
    sampled = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        sampled.append([] if request.is_prefill_chunk else [7])
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
        ),
    )
    return scheduled


def test_a_short_prompt_shares_the_step_with_a_running_long_prefill(
    model_dir, monkeypatch
):
    base = _scheduler(AsyncScheduler, model_dir, monkeypatch)
    short_first = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    for scheduler in (base, short_first):
        scheduler.add_request(_request("long", 20_000))
        assert _step(scheduler) == {"long": BUDGET}
        scheduler.add_request(_request("short", 500, first_token=50_000))
    assert _step(base) == {"long": BUDGET}
    assert _step(short_first) == {"short": 500, "long": BUDGET - 500}


def test_a_long_prefill_alone_keeps_the_whole_budget(model_dir, monkeypatch):
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    scheduler.add_request(_request("long", 20_000))
    assert _step(scheduler) == {"long": BUDGET}
    assert _step(scheduler) == {"long": BUDGET}


def test_a_waiting_long_prompt_gets_no_reservation(model_dir, monkeypatch):
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    scheduler.add_request(_request("long", 20_000))
    _step(scheduler)
    scheduler.add_request(_request("other", 3000, first_token=50_000))
    assert _step(scheduler) == {"long": BUDGET}


def test_a_short_prompt_moves_ahead_of_a_waiting_long_one(model_dir, monkeypatch):
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    scheduler.add_request(_request("long", 20_000))
    _step(scheduler)
    scheduler.add_request(_request("other", 9000, first_token=50_000))
    scheduler.add_request(_request("short", 300, first_token=70_000))
    assert _step(scheduler) == {"short": 300, "long": BUDGET - 300}


def test_cached_tokens_do_not_count_toward_the_short_limit(model_dir, monkeypatch):
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    scheduler.add_request(_request("warm", 6000, max_tokens=1))
    while "warm" in scheduler.requests:
        _step(scheduler)
    scheduler.add_request(_request("long", 20_000, first_token=50_000))
    _step(scheduler)
    scheduler.add_request(_request("turn", 6400))
    assert _step(scheduler) == {"turn": 400, "long": BUDGET - 400}


def test_short_prompts_that_do_not_fit_wait_their_turn(model_dir, monkeypatch):
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    scheduler.add_request(_request("long", 20_000))
    _step(scheduler)
    for i in range(3):
        scheduler.add_request(_request(f"s{i}", 1500, first_token=50_000 + 2000 * i))
    assert _step(scheduler) == {"s0": 1500, "s1": 1500, "long": BUDGET - 3000}


def test_no_free_sequence_slot_means_no_reservation(model_dir, monkeypatch):
    scheduler = _scheduler(
        ShortPrefillFirstScheduler, model_dir, monkeypatch, max_seqs=1
    )
    scheduler.add_request(_request("long", 20_000))
    _step(scheduler)
    scheduler.add_request(_request("short", 500, first_token=50_000))
    assert _step(scheduler) == {"long": BUDGET}


def test_every_prompt_token_is_computed_once(model_dir, monkeypatch):
    prompts = {"a": 30_000, "b": 700, "c": 12_000, "d": 1900, "e": 64, "f": 2049}
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    prefilled = dict.fromkeys(prompts, 0)
    pending = list(prompts.items())
    steps = 0
    while scheduler.requests or pending:
        if pending and steps % 2 == 0:
            req_id, prompt = pending.pop(0)
            scheduler.add_request(
                _request(req_id, prompt, first_token=100_000 * (ord(req_id) - 96))
            )
        scheduled = _step(scheduler)
        assert sum(scheduled.values()) <= BUDGET
        for req_id, tokens in scheduled.items():
            prefilled[req_id] += min(tokens, prompts[req_id] - prefilled[req_id])
        steps += 1
        assert steps < 200
    assert prefilled == prompts


def test_waiting_order_is_untouched_without_a_running_prefill(model_dir, monkeypatch):
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    scheduler.add_request(_request("decode", 100, max_tokens=50))
    _step(scheduler)
    scheduler.add_request(_request("other", 9000, first_token=50_000))
    scheduler.add_request(_request("short", 300, first_token=70_000))
    assert scheduler._promote_short_waiting() == []
    assert [r.request_id for r in scheduler.waiting] == ["other", "short"]


def test_each_waiting_prompt_is_looked_up_once(model_dir, monkeypatch):
    scheduler = _scheduler(ShortPrefillFirstScheduler, model_dir, monkeypatch)
    scheduler.add_request(_request("long", 20_000))
    _step(scheduler)
    lookups = []
    look_up = scheduler._look_up_uncached
    scheduler._look_up_uncached = lambda request: lookups.append(
        request.request_id
    ) or look_up(request)
    scheduler.add_request(_request("other", 9000, first_token=50_000))
    for _ in range(3):
        _step(scheduler)
    assert lookups == ["other"]


def test_a_reservation_the_kv_cache_cannot_hold_does_not_stall_the_prefill(
    model_dir, monkeypatch
):
    # 300 blocks hold 4,800 tokens. The long prefill's first chunk takes 256 of
    # them; the two short prompts reserve the whole budget but need 128 blocks
    # each, more than the 44 left. Capping the prefill at 0 for them would
    # schedule nothing on every step: the prefill never reaches allocate_slots,
    # so nothing is preempted, and the shorts never fit.
    scheduler = _scheduler(
        ShortPrefillFirstScheduler, model_dir, monkeypatch, num_blocks=300
    )
    scheduler.add_request(_request("long", 4_500))
    assert _step(scheduler) == {"long": BUDGET}
    scheduler.add_request(_request("a", SHORT, first_token=50_000))
    scheduler.add_request(_request("b", SHORT, first_token=60_000))
    for _ in range(3):
        _step(scheduler)
    assert scheduler.requests["long"].num_computed_tokens >= 4_500


def test_fit_short_prompts_takes_what_fits_in_order():
    candidates = [("a", 30), ("b", 80), ("c", 20), ("d", 10)]
    taken = fit_short_prompts(candidates, reserved=10, free_tokens=70, free_slots=3)
    assert taken == (70, ["a", "c", "d"])
    taken = fit_short_prompts(candidates, reserved=0, free_tokens=200, free_slots=2)
    assert taken == (110, ["a", "b"])


def test_fit_short_prompts_draws_no_candidate_once_full():
    drawn = []

    def candidates():
        for prompt, tokens in [("a", 50), ("b", 50), ("c", 50)]:
            drawn.append(prompt)
            yield prompt, tokens

    assert fit_short_prompts(
        candidates(), reserved=0, free_tokens=100, free_slots=5
    ) == (100, ["a", "b"])
    assert drawn == ["a", "b"]
