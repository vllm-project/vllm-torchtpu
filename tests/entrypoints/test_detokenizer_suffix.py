# SPDX-License-Identifier: Apache-2.0
"""Exactness of VLLM_TPU_DETOK_SUFFIX_PRIME (vllm_torchtpu/detokenizer_suffix.py).

A DecodeStream primed with the chosen prompt suffix must return the same text,
token by token, as one primed with the whole prompt.
"""

import random

import pytest
from tokenizers.decoders import DecodeStream

from vllm_torchtpu.detokenizer_suffix import (
    LONGEST_SUFFIX,
    REPLACEMENT_CHARACTER,
    SHORTEST_SUFFIX,
    prompt_suffix,
)

from .test_segment_tokenize import SPECIALS, _toy_tokenizer

# Not in the toy corpus, so byte-level BPE keeps them as several byte tokens.
SPLIT_CHARS = ["🙂", "ß", "Ω", "中"]
ATOMS = ["hello", " world", " naïve", " café", " 東京", " 😀", "\n", "  ", "123"]


@pytest.fixture(scope="module")
def tok():
    return _toy_tokenizer()


def _ids(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def _random_ids(tok, rng, parts):
    pieces = rng.choices(ATOMS + SPLIT_CHARS + SPECIALS, k=parts)
    return _ids(tok, "".join(pieces))


def _steps(tok, prime, continuation, skip_special_tokens):
    stream = DecodeStream(ids=prime, skip_special_tokens=skip_special_tokens)
    return [stream.step(tok, t) for t in continuation]


def test_short_and_missing_prompts_come_back_unchanged(tok):
    """A prompt of at most 4 tokens has no shorter suffix to prime with."""
    assert prompt_suffix(tok, None) is None
    prompt = _ids(tok, "hello")[:SHORTEST_SUFFIX]
    assert prompt_suffix(tok, prompt) is prompt


def test_the_suffix_is_the_shortest_clean_tail(tok):
    """The chosen suffix decodes to whole characters and no shorter one does."""
    rng = random.Random(0)
    for _ in range(200):
        prompt = _random_ids(tok, rng, rng.randint(4, 30))
        suffix = prompt_suffix(tok, prompt)
        if suffix is prompt:
            continue
        assert SHORTEST_SUFFIX <= len(suffix) <= LONGEST_SUFFIX
        assert suffix == prompt[-len(suffix) :]
        text = tok.decode(suffix, skip_special_tokens=False)
        assert text and REPLACEMENT_CHARACTER not in text
        for shorter in range(SHORTEST_SUFFIX, len(suffix)):
            text = tok.decode(prompt[-shorter:], skip_special_tokens=False)
            assert not text or REPLACEMENT_CHARACTER in text


def test_a_prompt_ending_inside_a_character_keeps_the_full_prompt(tok):
    """Every suffix of a prompt cut inside a character decodes to U+FFFD."""
    whole = _ids(tok, "hello world naïve café 🙂")
    cuts = [
        k
        for k in range(1, len(whole))
        if tok.decode(whole[:k]).endswith(REPLACEMENT_CHARACTER)
    ]
    assert cuts, "the test character must span several tokens"
    prompt = whole[: cuts[-1]]
    assert prompt_suffix(tok, prompt) is prompt


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_the_suffix_prime_gives_the_same_text_as_the_full_prime(
    tok, skip_special_tokens
):
    """Served output text must not change when the stream is primed short."""
    rng = random.Random(1 if skip_special_tokens else 2)
    for _ in range(300):
        prompt = _random_ids(tok, rng, rng.randint(1, 30))
        continuation = _random_ids(tok, rng, rng.randint(1, 12))
        suffix = prompt_suffix(tok, prompt, skip_special_tokens)
        assert _steps(tok, suffix, continuation, skip_special_tokens) == _steps(
            tok, prompt, continuation, skip_special_tokens
        )


def test_a_suffix_of_skipped_special_tokens_is_not_chosen(tok):
    """A suffix that decodes to nothing would restart the stream at a sequence
    start, where some decoders drop the next token's leading space."""
    prompt = _ids(tok, "hello world" + "".join(SPECIALS[:6]))
    suffix = prompt_suffix(tok, prompt, skip_special_tokens=True)
    assert tok.decode(suffix, skip_special_tokens=True)


def test_the_patch_primes_short_keeps_output_and_installs_once(tok, monkeypatch):
    """The patched detokenizer returns the unpatched text, and installing it
    twice or with the switch off leaves the class as it was."""
    pytest.importorskip("vllm")
    from transformers import PreTrainedTokenizerFast
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

    from vllm_torchtpu import detokenizer_suffix

    cls = FastIncrementalDetokenizer
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_tpu_detok_suffix_prime", False, raising=False)
    hf = PreTrainedTokenizerFast(tokenizer_object=tok)
    rng = random.Random(3)
    prompt = _random_ids(tok, rng, 40)
    continuation = _random_ids(tok, rng, 20)
    request = EngineCoreRequest(
        request_id="r",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=len(continuation)),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    def served(detok):
        for t in continuation:
            detok.update([t], False)
        return detok.get_next_output_text(finished=True, delta=False)

    full = served(cls(hf, request))
    monkeypatch.setenv("VLLM_TPU_DETOK_SUFFIX_PRIME", "0")
    detokenizer_suffix.patch_detokenizer_suffix_prime()
    assert not cls._tpu_detok_suffix_prime
    monkeypatch.setenv("VLLM_TPU_DETOK_SUFFIX_PRIME", "1")
    detokenizer_suffix.patch_detokenizer_suffix_prime()
    patched_init = cls.__init__
    detokenizer_suffix.patch_detokenizer_suffix_prime()
    assert cls.__init__ is patched_init
    assert served(cls(hf, request)) == full
    assert request.prompt_token_ids == prompt
