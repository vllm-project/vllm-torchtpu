# SPDX-License-Identifier: Apache-2.0
"""TPU_DETOK_PROMPT_SUFFIX: the streaming detokenizer is primed with only the
prompt's last N tokens, the request is left intact, and the decoded deltas match
a full-prompt prime on a byte-level BPE (multi-byte UTF-8 split over tokens)."""

import random

import pytest
import tokenizers
import tokenizers.decoders
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine import detokenizer as dt

from vllm_torchtpu import detok_prompt_suffix as dps

_TEXT = ("def foo(x):\n    return x + 1  # comment\n"
         "深度求索的模型在长上下文中表现良好。日本語のテキスト、한국어 텍스트 "
         "emoji 😀🚀🔥👨‍👩‍👧 math ∑∫√π≈≠ rare 𠀀𪚥 The quick brown fox. ") * 3


@pytest.fixture(scope="module")
def bpe():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        [_TEXT] * 4,
        trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>", "</s>"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tok


def _steps(tok, prime, gen, skip):
    s = tokenizers.decoders.DecodeStream(ids=prime, skip_special_tokens=skip)
    return [s.step(tok, g) for g in gen]


@pytest.mark.parametrize("skip", [True, False])
def test_suffix_prime_matches_full_prime(bpe, skip):
    ids = bpe.encode(_TEXT * 4).ids
    tail = bpe.encode(" assistant:\n").ids  # served prompts end in the template
    rng = random.Random(0)
    compared = 0
    for case in range(300):
        p = rng.randrange(8, len(ids) - 40)
        prompt = ids[:p] + (tail if case % 3 else [])
        gen = (ids[p:p + 30] if case % 2 else
               [rng.randrange(0, bpe.get_vocab_size()) for _ in range(30)])
        ref = _steps(bpe, prompt, gen, skip)
        if any(r is not None and len(r) > 60 for r in ref):
            # The stock full-prompt stream emits the whole prompt text when the
            # prompt ends inside a multi-byte character; nothing to match.
            continue
        compared += 1
        assert _steps(bpe, prompt[-32:], gen, skip) == ref, case
    assert compared > 200


def test_install_primes_with_suffix_and_restores_request(monkeypatch):
    seen = []

    class RecordingStream:
        def __init__(self, ids=None, skip_special_tokens=False):
            seen.append(list(ids))

    monkeypatch.setenv("TPU_DETOK_PROMPT_SUFFIX", "8")
    monkeypatch.setattr(dt.FastIncrementalDetokenizer, "__init__",
                        dt.FastIncrementalDetokenizer.__init__)
    monkeypatch.setattr(dt.FastIncrementalDetokenizer, "_tpu_prompt_suffix", 0,
                        raising=False)
    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", RecordingStream)
    dps.install()

    class FakeBackend:
        _tokenizer = object()

    prompt = list(range(100, 200))
    req = EngineCoreRequest(
        request_id="r", prompt_token_ids=prompt, mm_features=None,
        sampling_params=SamplingParams(max_tokens=4), pooling_params=None,
        arrival_time=0.0, lora_request=None, cache_salt=None,
        data_parallel_rank=None)
    dt.FastIncrementalDetokenizer(FakeBackend(), req)
    assert seen == [prompt[-8:]]
    assert req.prompt_token_ids is prompt
    short = EngineCoreRequest(
        request_id="s", prompt_token_ids=[1, 2, 3], mm_features=None,
        sampling_params=SamplingParams(max_tokens=4), pooling_params=None,
        arrival_time=0.0, lora_request=None, cache_salt=None,
        data_parallel_rank=None)
    dt.FastIncrementalDetokenizer(FakeBackend(), short)
    assert seen[-1] == [1, 2, 3]


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TPU_DETOK_PROMPT_SUFFIX", raising=False)
    orig = dt.FastIncrementalDetokenizer.__init__
    dps.install()
    assert dt.FastIncrementalDetokenizer.__init__ is orig
