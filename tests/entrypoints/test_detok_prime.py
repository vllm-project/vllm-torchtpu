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
"""Exactness of VLLM_TPU_DETOK_PRIME_TOKENS (vllm_torchtpu/detok_prime.py).

A detokenizer primed with the prompt tail must emit the same deltas, stop
strings and final text as one primed with the whole prompt, for every prompt
and output. Real tokenizers are loaded from the HF hub (or local paths via
DETOK_PRIME_TEST_MODELS) and skipped when unavailable; DETOK_PRIME_TEST_GLM points at GLM-5.2's.
"""

import os
import random
import zlib
from types import SimpleNamespace

import pytest
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from vllm_torchtpu.detok_prime import (
    _PrimedRequest,
    patch_detokenizer_prime_tail,
    prime_ids,
)

transformers = pytest.importorskip("transformers")
vllm_detok = pytest.importorskip("vllm.v1.engine.detokenizer")
from vllm import SamplingParams  # noqa: E402

GLM = os.environ.get("DETOK_PRIME_TEST_GLM", "zai-org/GLM-5.2-FP8")
DEFAULT_MODELS = (
    GLM,
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "openai-community/gpt2",
    "google-bert/bert-base-uncased",
)
MODELS = [
    m for m in os.environ.get("DETOK_PRIME_TEST_MODELS", "").split(",") if m
] or list(DEFAULT_MODELS)
CASES = int(os.environ.get("DETOK_PRIME_TEST_CASES", "400"))
SEED = int(os.environ.get("DETOK_PRIME_TEST_SEED", "0"))

CORPUS = [
    "To be, or not to be, that is the question:",
    "  def f(x):\n\treturn x  # 123 4567 89\r\n",
    "naïve café — résumé coöperate",
    "東京の天気は晴れです。我们今天去公园散步。한국어 문장입니다.",
    "emoji 😀🎉 family 👨‍👩‍👧‍👦 flags 🇯🇵🇺🇸 skin 👍🏽",
    "math 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ∑∫√∞ ≠ ≤ ꙮ",
    "مرحبا بالعالم שלום עולם",
    "combining é ä ñ zalgo z̶̷a̵",
    "\n\n\n   leading spaces\t\ttabs   trailing   ",
    '{"key": [1, 2, 3], "nested": {"a": null}}',
    "literal replacement char � here",
    "<html><body>Hello</body></html>",
    "ünïcödé ÀÉÎÕÜ ßæøå ĳŀł",
    " ",
    "\n",
]


def _toy_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|user|>", "<|assistant|>", "<think>", "</think>"],
    )
    tok.train_from_iterator(CORPUS * 50, trainer=trainer)
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok)


_TOKENIZERS: dict = {}


def _load(name):
    if name not in _TOKENIZERS:
        if name == "toy":
            _TOKENIZERS[name] = _toy_tokenizer()
        else:
            try:
                _TOKENIZERS[name] = transformers.AutoTokenizer.from_pretrained(name)
            except Exception as e:  # noqa: BLE001
                _TOKENIZERS[name] = e
    tok = _TOKENIZERS[name]
    if isinstance(tok, Exception):
        pytest.skip(f"tokenizer {name} unavailable: {tok}")
    if not isinstance(tok, transformers.TokenizersBackend):
        pytest.skip(f"{name} is not a fast tokenizer")
    return tok


def _make(tok, req):
    cls = vllm_detok.FastIncrementalDetokenizer
    init = getattr(cls, "_tpu_detok_orig_init", cls.__init__)
    d = object.__new__(cls)
    init(d, tok, req)
    return d


def _drive(d, out_ids, chunks, stop_terminated):
    """Feed output ids in chunks; record every delta, stop and the final text."""
    log, i = [], 0
    for c in chunks:
        new = out_ids[i : i + c]
        i += c
        last = i >= len(out_ids)
        try:
            stop = d.update(new, stop_terminated=last and stop_terminated)
        except Exception as e:  # noqa: BLE001
            log.append(("raise", type(e).__name__))
            return log
        finished = last or stop is not None
        log.append((stop, d.get_next_output_text(finished, True)))
        if finished:
            break
    log.append(("final", d.output_text, d.get_next_output_text(True, False)))
    return log


def _random_text(rng, n_pieces):
    return "".join(rng.choice(CORPUS) for _ in range(n_pieces))


def _specials(tok):
    return sorted(tok.get_added_vocab().values()) or [tok.eos_token_id or 0]


def _gen_case(rng, tok):
    """One (prompt_ids, out_ids, kind) case."""
    specials = _specials(tok)
    vocab = len(tok)
    prompt = tok.encode(_random_text(rng, rng.randint(3, 25)), add_special_tokens=False)
    prompt = prompt or [rng.randrange(vocab)]
    kind = rng.choice(
        ["text", "random", "split", "specials", "trailing_specials", "cut"]
    )
    if kind == "text":
        out = tok.encode(_random_text(rng, rng.randint(1, 6)), add_special_tokens=False)
    elif kind == "random":
        out = [rng.randrange(vocab) for _ in range(rng.randint(1, 40))]
    elif kind == "split":
        # One encoded text split inside a character, across the prompt boundary.
        ids = tok.encode(_random_text(rng, rng.randint(2, 8)), add_special_tokens=False)
        cuts = [
            k for k in range(1, len(ids)) if tok.decode(ids[:k]).endswith("�")
        ]
        k = rng.choice(cuts) if cuts else rng.randrange(1, max(2, len(ids)))
        prompt, out = prompt + ids[:k], ids[k:] or ids[-1:]
    elif kind == "specials":
        out = tok.encode(_random_text(rng, 3), add_special_tokens=False)
        for _ in range(rng.randint(1, 6)):
            out.insert(rng.randrange(len(out) + 1), rng.choice(specials))
    elif kind == "trailing_specials":
        # Longer than small N, so an all-special tail must widen.
        prompt = prompt + [rng.choice(specials) for _ in range(rng.randint(1, 48))]
        out = tok.encode(_random_text(rng, 2), add_special_tokens=False)
    else:  # cut: prompt ends mid-word or mid-character
        k = rng.randrange(1, len(prompt) + 1)
        tail = tok.encode(_random_text(rng, 2), add_special_tokens=False)
        prompt, out = prompt[:k], tail
    return prompt, out or [rng.randrange(vocab)], kind


def _params(rng, tok, out_ids):
    skip = rng.random() < 0.5
    spaces = rng.random() < 0.5
    stop = None
    text = tok.decode(out_ids, skip_special_tokens=skip)
    if rng.random() < 0.4 and len(text) >= 2:
        a = rng.randrange(len(text) - 1)
        stop = [text[a : a + rng.randint(1, 4)]]
        if rng.random() < 0.5:
            stop.append(rng.choice(["\n", " the", "。", "😀"]))
    return SamplingParams(
        max_tokens=len(out_ids) + 1,
        skip_special_tokens=skip,
        spaces_between_special_tokens=spaces,
        stop=stop,
        include_stop_str_in_output=rng.random() < 0.5,
        min_tokens=rng.choice([0, 0, 0, 2]),
    )


def _check(tok, prompt, out, params, n, rng):
    req = SimpleNamespace(request_id="r", prompt_token_ids=prompt, sampling_params=params)
    tail = prime_ids(tok._tokenizer, prompt, params.skip_special_tokens, n)
    primed = req if tail is prompt else _PrimedRequest(req, tail)
    chunks = []
    while sum(chunks) < len(out):
        chunks.append(rng.choice([1, 1, 1, 2, 3, 5]))
    stop_term = rng.random() < 0.3
    full = _drive(_make(tok, req), out, chunks, stop_term)
    short = _drive(_make(tok, primed), out, chunks, stop_term)
    assert short == full, (prompt[-8:], out, params, n, len(tail or []))
    return tail is not prompt


@pytest.mark.parametrize("name", ["toy", *MODELS])
def test_tail_priming_matches_full_priming(name):
    tok = _load(name)
    rng = random.Random(zlib.crc32(name.encode()) + SEED)
    shortened = 0
    for case in range(CASES):
        prompt, out, _ = _gen_case(rng, tok)
        params = _params(rng, tok, out)
        n = rng.choice([1, 2, 4, 32])
        shortened += _check(tok, prompt, out, params, n, rng)
    # Most cases must actually exercise a shortened prime.
    assert shortened > CASES // 2, shortened


@pytest.mark.parametrize("name", ["toy", *MODELS])
def test_prompt_ending_mid_character_falls_back(name):
    tok = _load(name)
    rng = random.Random(7)
    hits = 0
    for _ in range(200):
        ids = tok.encode(_random_text(rng, 30), add_special_tokens=False)
        cuts = [k for k in range(40, len(ids)) if tok.decode(ids[:k]).endswith("�")]
        if not cuts:
            continue
        k = rng.choice(cuts)
        prompt, out = ids[:k], ids[k : k + 20]
        params = SamplingParams(max_tokens=32, skip_special_tokens=rng.random() < 0.5)
        tail = prime_ids(tok._tokenizer, prompt, params.skip_special_tokens, 4)
        assert tail is prompt
        _check(tok, prompt, out, params, 4, rng)
        hits += 1
    if not hits:
        pytest.skip(f"{name}: no prompt cut ends mid-character")


def test_glm_long_chat_prompts():
    tok = _load(GLM)
    rng = random.Random(3)
    for _ in range(20):
        msgs = [
            {"role": rng.choice(["user", "assistant"]), "content": _random_text(rng, 40)}
            for _ in range(rng.randint(2, 12))
        ]
        msgs[-1]["role"] = "user"
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if not isinstance(prompt, list):
            prompt = prompt["input_ids"]
        out = tok.encode("<think>" + _random_text(rng, 6) + "</think>", add_special_tokens=False)
        _check(tok, prompt, out, _params(rng, tok, out), 32, rng)


def test_patch_installs_and_primes_tail(monkeypatch):
    tok = _load("toy")
    cls = vllm_detok.FastIncrementalDetokenizer
    monkeypatch.setenv("VLLM_TPU_DETOK_PRIME_TOKENS", "8")
    if getattr(cls, "_tpu_detok_prime_tokens", None) is not None:
        monkeypatch.setattr(cls, "__init__", cls._tpu_detok_orig_init)
    monkeypatch.setattr(cls, "_tpu_detok_prime_tokens", None, raising=False)
    patch_detokenizer_prime_tail()
    assert cls._tpu_detok_prime_tokens == 8

    seen = []
    real = tokenizers.decoders.DecodeStream

    def recording(ids=None, skip_special_tokens=False):
        seen.append(len(ids or []))
        return real(ids=ids, skip_special_tokens=skip_special_tokens)

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", recording)
    prompt = tok.encode(_random_text(random.Random(0), 30), add_special_tokens=False)
    out = tok.encode(" 東京 😀 done", add_special_tokens=False)
    params = SamplingParams(max_tokens=64)
    req = SimpleNamespace(request_id="r", prompt_token_ids=prompt, sampling_params=params)
    d = vllm_detok.FastIncrementalDetokenizer(tok, req)
    ref = _make(tok, req)
    assert seen[0] <= 16 < len(prompt)
    chunks = [1] * len(out)
    assert _drive(d, out, chunks, False) == _drive(ref, out, chunks, False)


def test_prime_ids_edges():
    tok = _load("toy")
    t = tok._tokenizer
    assert prime_ids(t, None, True, 32) is None
    assert prime_ids(t, [], True, 32) == []
    ids = tok.encode("hello world " * 10, add_special_tokens=False)
    assert prime_ids(t, ids, True, 0) is ids
    assert prime_ids(t, ids[:5], True, 32) == ids[:5]
    assert prime_ids(t, ids, True, 4) == ids[-4:]
    special = tok.convert_tokens_to_ids("<think>")
    # All-special tail decodes empty when skipped: widen past it.
    tail = prime_ids(t, ids + [special] * 10, True, 4)
    assert len(tail) == 16
    assert prime_ids(t, ids + [special] * 10, False, 4) == [special] * 4
