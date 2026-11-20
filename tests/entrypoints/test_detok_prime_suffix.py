# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A decode stream primed with the prompt suffix writes the same text for
every output token as one primed with the whole prompt."""

import random
from types import SimpleNamespace

import pytest
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)

from vllm_torchtpu.detok_prime_suffix import (
    LONGEST_SUFFIX,
    REPLACEMENT_CHARACTER,
    SHORTEST_SUFFIX,
    prompt_suffix,
)

from .test_segment_tokenize import (
    SPECIALS,
    _random_text,
    _toy_tokenizer,
)


def _streamed(tokenizer, prime, outputs, skip_special_tokens):
    stream = decoders.DecodeStream(
        ids=list(prime), skip_special_tokens=skip_special_tokens
    )
    texts = []
    for token_id in outputs:
        try:
            texts.append(stream.step(tokenizer, token_id))
        except Exception as error:
            texts.append(f"error: {error}")
            stream = decoders.DecodeStream(skip_special_tokens=skip_special_tokens)
    return texts


def _cases(tokenizer, seed, count):
    rng = random.Random(seed)
    vocab = tokenizer.get_vocab_size()
    for _ in range(count):
        prompt = tokenizer.encode(_random_text(rng, rng.randrange(1, 40))).ids
        if rng.random() < 0.2:
            prompt += [rng.randrange(vocab) for _ in range(rng.randrange(1, 4))]
        if rng.random() < 0.5:
            outputs = tokenizer.encode(_random_text(rng, rng.randrange(1, 10))).ids
        else:
            outputs = [rng.randrange(vocab) for _ in range(rng.randrange(1, 30))]
        yield prompt, outputs


def _emoji_pieces(tokenizer):
    pieces = tokenizer.encode("😀").ids
    if len(pieces) < 2:
        pytest.skip("the toy tokenizer writes the emoji as 1 token")
    return pieces


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_suffix_primed_stream_matches_whole_prompt(skip_special_tokens):
    tokenizer = _toy_tokenizer()
    for prompt, outputs in _cases(tokenizer, int(skip_special_tokens), 3000):
        suffix = prompt_suffix(tokenizer, prompt, skip_special_tokens)
        assert list(prompt[len(prompt) - len(suffix) :]) == list(suffix)
        assert _streamed(tokenizer, suffix, outputs, skip_special_tokens) == (
            _streamed(tokenizer, prompt, outputs, skip_special_tokens)
        )


def _metaspace_tokenizer(decoder):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="first"
    )
    bytes_ = [f"<0x{i:02X}>" for i in range(256)]
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", *bytes_])
    tokenizer.train_from_iterator(["hello world the quick brown fox"] * 50, trainer)
    tokenizer.add_special_tokens([AddedToken(s, normalized=False) for s in SPECIALS])
    tokenizer.decoder = decoder
    return tokenizer


@pytest.mark.parametrize(
    "decoder",
    [
        decoders.Metaspace(replacement="▁", prepend_scheme="first"),
        decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
                decoders.Strip(" ", 1, 0),
            ]
        ),
    ],
    ids=["metaspace", "strip-first-space"],
)
def test_prompt_ending_in_special_tokens_keeps_the_first_space(decoder):
    tokenizer = _metaspace_tokenizer(decoder)
    prompt = tokenizer.encode("hello world").ids + [
        tokenizer.token_to_id(s) for s in SPECIALS[:SHORTEST_SUFFIX]
    ]
    outputs = tokenizer.encode("hello world").ids
    suffix = prompt_suffix(tokenizer, prompt, True)
    assert _streamed(tokenizer, suffix, outputs, True) == (
        _streamed(tokenizer, prompt, outputs, True)
    )


def test_suffix_is_short_and_whole():
    tokenizer = _toy_tokenizer()
    prompt = tokenizer.encode("naïve café — 東京 😀 " * 40).ids
    suffix = prompt_suffix(tokenizer, prompt, True)
    assert SHORTEST_SUFFIX <= len(suffix) <= LONGEST_SUFFIX
    assert REPLACEMENT_CHARACTER not in tokenizer.decode(suffix)


def test_prompt_ending_inside_a_character_is_kept_whole():
    tokenizer = _toy_tokenizer()
    pieces = _emoji_pieces(tokenizer)
    prompt = tokenizer.encode("plain text " * 20).ids + pieces[:-1]
    assert prompt_suffix(tokenizer, prompt, True) == prompt


def test_short_prompt_is_kept_whole():
    tokenizer = _toy_tokenizer()
    prompt = tokenizer.encode(SPECIALS[0] + "ab cd ef").ids[:SHORTEST_SUFFIX]
    assert prompt_suffix(tokenizer, prompt, True) == prompt


def test_patch_primes_the_detokenizer_with_the_suffix(monkeypatch):
    detokenizer = pytest.importorskip("vllm.v1.engine.detokenizer")
    from vllm import SamplingParams

    from vllm_torchtpu import detok_prime_suffix

    monkeypatch.setenv("VLLM_TPU_DETOK_PRIME_SUFFIX", "1")
    cls = detokenizer.FastIncrementalDetokenizer
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(cls, "_tpu_prime_suffix", False, raising=False)
    primed = []
    stream = decoders.DecodeStream

    def recording_stream(ids=None, **kwargs):
        primed.append(list(ids or []))
        return stream(ids=ids, **kwargs)

    monkeypatch.setattr(decoders, "DecodeStream", recording_stream)
    detok_prime_suffix.patch_detokenizer_prime_suffix()

    tokenizer = _toy_tokenizer()
    prompt = tokenizer.encode("plain text " * 50).ids
    request = SimpleNamespace(
        request_id="r",
        sampling_params=SamplingParams(skip_special_tokens=True),
        prompt_token_ids=prompt,
        prompt_embeds=None,
    )
    cls(SimpleNamespace(_tokenizer=tokenizer), request)
    assert primed and SHORTEST_SUFFIX <= len(primed[-1]) < len(prompt)
    assert primed[-1] == prompt[-len(primed[-1]) :]


def test_patch_is_off_by_default(monkeypatch):
    detokenizer = pytest.importorskip("vllm.v1.engine.detokenizer")
    from vllm_torchtpu import detok_prime_suffix

    monkeypatch.delenv("VLLM_TPU_DETOK_PRIME_SUFFIX", raising=False)
    cls = detokenizer.FastIncrementalDetokenizer
    before = cls.__init__
    detok_prime_suffix.patch_detokenizer_prime_suffix()
    assert cls.__init__ is before
