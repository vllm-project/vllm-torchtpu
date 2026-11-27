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
"""tpu_frontend.detok_prompt_tail: a DecodeStream primed with a safe prompt
tail must emit byte-identical text to one primed with the whole prompt."""

import json
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tokenizers
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import bytes_to_unicode
from vllm.logger import _print_warning_once
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine import detokenizer as vllm_detok

from vllm_torchtpu import frontend_fix as ff

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return {k: v for k, v in x.items()}\n",
    "中文测试：我们今天讨论大型语言模型的推理延迟。日本語のテキスト。한국어 문장.",
    "Emoji 🚀🔥👍🏽 family 👨‍👩‍👧‍👦 flags 🇺🇸🇯🇵 keycap 1️⃣ ✅",
    "Math ∑ α·β ≤ ∞ ∂f/∂x ≈ 0 Ünïcödé ñ ç ø ß 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𓀀",
    "مرحبا بالعالم. नमस्ते दुनिया। Привет, мир!",
    "I don't know , isn't it ? Yes ! 'quoted' do not .",
]


def _bpe_trainer(**kwargs):
    return trainers.BpeTrainer(
        vocab_size=700, special_tokens=SPECIALS, show_progress=False, **kwargs
    )


def _byte_level():
    """GPT-2/Qwen style: byte-level BPE; returns the tokenizer and the ids of
    its single-byte tokens 0x80-0xff."""
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(
        CORPUS * 20,
        _bpe_trainer(initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    vocab = t.get_vocab(with_added_tokens=False)
    bu = bytes_to_unicode()
    return t, [vocab[bu[b]] for b in range(0x80, 0x100) if bu[b] in vocab]


def _metaspace():
    """SentencePiece style: Metaspace pre-tokenizer and decoder."""
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.Metaspace()
    t.decoder = decoders.Metaspace()
    t.train_from_iterator(CORPUS * 20, _bpe_trainer())
    return t, []


def _byte_fallback():
    """Llama style: BPE with <0xNN> byte tokens, decoded by Replace,
    ByteFallback, Fuse and Strip."""
    t = Tokenizer(models.BPE())
    t.normalizer = normalizers.Sequence(
        [normalizers.Prepend("\u2581"), normalizers.Replace(" ", "\u2581")]
    )
    t.train_from_iterator(CORPUS * 20, _bpe_trainer())
    spec = json.loads(t.to_str())
    vocab = spec["model"]["vocab"]
    first = max(vocab.values()) + 1
    vocab.update({f"<0x{b:02X}>": first + b for b in range(256)})
    spec["model"]["byte_fallback"] = True
    t = Tokenizer.from_str(json.dumps(spec))
    t.decoder = decoders.Sequence(
        [
            decoders.Replace("\u2581", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )
    return t, [first + b for b in range(0x80, 0x100)]


def _wordpiece():
    """BERT style: WordPiece with its cleanup of spaces before punctuation."""
    t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.decoder = decoders.WordPiece()
    t.train_from_iterator(
        CORPUS * 20,
        trainers.WordPieceTrainer(
            vocab_size=700, special_tokens=["[UNK]", *SPECIALS], show_progress=False
        ),
    )
    return t, []


@pytest.fixture(scope="module")
def tok():
    return _byte_level()[0]


def _stream(tok, prime, cont, skip):
    s = tokenizers.decoders.DecodeStream(ids=list(prime), skip_special_tokens=skip)
    out = []
    for t in cont:
        try:
            out.append(s.step(tok, t))
        except Exception as e:  # mirror vLLM's reset-on-invalid-prefix
            out.append(("EXC", str(e)[:30]))
            s = tokenizers.decoders.DecodeStream(skip_special_tokens=skip)
    return out


def _rand_seq(rng, tok, n, byte_ids, special_ids):
    v = tok.get_vocab_size(with_added_tokens=False)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.05:
            out.append(rng.choice(special_ids))
        elif r < 0.35 and byte_ids:
            out.append(rng.choice(byte_ids))
        else:
            out.append(rng.randrange(v))
    return out


@pytest.mark.parametrize(
    "build",
    [_byte_level, _metaspace, _byte_fallback, _wordpiece],
    ids=lambda f: f.__name__[1:],
)
def test_tail_priming_is_byte_identical(build):
    """Every decoder family vLLM's fast tokenizers use: the emitted text
    depends on the prompt only through the decoded text at its end, which a
    safe tail reproduces."""
    tok, byte_ids = build()
    rng = random.Random(0)
    special_ids = [
        tok.token_to_id(s) for s in SPECIALS if tok.token_to_id(s) is not None
    ]
    assert special_ids
    n_tail = n_full = 0
    for i in range(300):
        if i % 3 == 0:
            prompt = tok.encode("".join(rng.choice(CORPUS) for _ in range(3))).ids
            prompt = [special_ids[1]] + prompt + [special_ids[2], special_ids[1]]
            prompt += tok.encode("assistant\n").ids
        elif i % 3 == 1:
            prompt = _rand_seq(rng, tok, rng.randint(1, 200), byte_ids, special_ids)
        else:
            t = tok.encode(rng.choice(CORPUS[2:])).ids
            prompt = tok.encode(rng.choice(CORPUS)).ids + t[: rng.randint(1, len(t))]
        if rng.random() < 0.5:
            cont = tok.encode(rng.choice(CORPUS)).ids[: rng.randint(1, 60)]
        else:
            cont = _rand_seq(rng, tok, rng.randint(1, 60), byte_ids, special_ids)
        for skip in (True, False):
            tail = ff.safe_prompt_tail(tok, prompt, skip, 8)
            if tail is prompt:
                n_full += 1
            else:
                n_tail += 1
                assert len(tail) == 8
            full_out = _stream(tok, prompt, cont, skip)
            tail_out = _stream(tok, tail, cont, skip)
            assert full_out == tail_out, (i, skip, prompt[-10:], cont)
            assert "".join(x for x in full_out if isinstance(x, str)).encode(
                "utf-8", "surrogatepass"
            ) == "".join(x for x in tail_out if isinstance(x, str)).encode(
                "utf-8", "surrogatepass"
            )
    assert n_tail > 100
    if byte_ids:  # a tail can end inside a multi-byte character
        assert n_full > 0


def test_safe_tail_falls_back_on_partial_utf8_and_empty(tok):
    emoji = tok.encode("🚀").ids
    # A prompt ending inside the character needs it split over several ids.
    assert len(emoji) > 1, emoji
    prompt = tok.encode("hello world, this is a test ").ids
    p = prompt + emoji[:-1]
    assert ff.safe_prompt_tail(tok, p, True, 8) is p
    eos = tok.token_to_id("<|im_end|>")
    p = prompt + [eos] * 10
    assert ff.safe_prompt_tail(tok, p, True, 8) is p  # tail decodes to ""
    assert ff.safe_prompt_tail(tok, p, False, 8) is not p
    short = prompt[:5]
    assert ff.safe_prompt_tail(tok, short, True, 8) is short
    assert ff.safe_prompt_tail(tok, prompt, True, 0) is prompt


def test_prompt_tail_view_delegates():
    class R:
        request_id = "x"
        sampling_params = object()
        prompt_token_ids = [1, 2, 3, 4]

    r = R()
    v = ff._PromptTailView(r, [3, 4])
    assert v.prompt_token_ids == [3, 4]
    assert v.request_id == "x" and v.sampling_params is r.sampling_params
    assert r.prompt_token_ids == [1, 2, 3, 4]


@pytest.fixture
def detok_cls(monkeypatch):
    """vLLM's FastIncrementalDetokenizer, restored after the test."""
    cls = vllm_detok.FastIncrementalDetokenizer
    monkeypatch.setattr(cls, "__init__", cls.__init__)
    monkeypatch.setattr(ff, "_DETOK_INSTALLED", False)
    return cls


def test_install_patches_vllm_detokenizer(detok_cls):
    orig = detok_cls.__init__
    assert ff.maybe_install_detok_prompt_tail()
    assert detok_cls.__init__ is not orig
    assert detok_cls.__init__.__wrapped__ is orig
    # A second install leaves the first patch in place.
    patched = detok_cls.__init__
    assert ff.maybe_install_detok_prompt_tail() is False
    assert detok_cls.__init__ is patched


def _in_process(monkeypatch, name):
    monkeypatch.setattr(
        ff.multiprocessing, "current_process", lambda: SimpleNamespace(name=name)
    )


@pytest.mark.parametrize("name", ["MainProcess", "APIServer_DPRank_1", "ApiServer_0"])
def test_install_in_every_api_server_process(monkeypatch, detok_cls, name):
    """vLLM's DP supervisor runs its API servers as APIServer_DPRank_<n>."""
    _in_process(monkeypatch, name)
    orig = detok_cls.__init__
    assert ff.maybe_install_detok_prompt_tail()
    assert detok_cls.__init__.__wrapped__ is orig


def test_install_elsewhere_warns_and_leaves_vllm_alone(monkeypatch, detok_cls):
    _in_process(monkeypatch, "EngineCore_DP0")
    logger = Mock()
    monkeypatch.setattr(ff, "logger", logger)
    orig = detok_cls.__init__
    assert ff.maybe_install_detok_prompt_tail() is False
    assert detok_cls.__init__ is orig
    (call,) = logger.warning.call_args_list
    assert "EngineCore_DP0" in call.args


def _request(prompt_token_ids):
    return EngineCoreRequest(
        request_id="r",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=64),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.fixture
def primed(monkeypatch):
    """The ids each DecodeStream vLLM builds is primed with."""
    seen = []
    real = tokenizers.decoders.DecodeStream

    def spy(ids=None, **kwargs):
        seen.append(None if ids is None else list(ids))
        return real(ids=ids, **kwargs)

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", spy)
    return seen


def _detokenize(tok, prompt, cont):
    """vLLM's own detokenizer for a request: its class and its whole output."""
    detok = vllm_detok.IncrementalDetokenizer.from_new_request(tok, _request(prompt))
    for token_id in cont:
        detok.update([token_id], False)
    return type(detok), detok.get_next_output_text(finished=True, delta=False)


def test_installed_wrapper_primes_vllms_detokenizer_with_the_tail(
    monkeypatch, detok_cls, primed, tok
):
    """The patch on vLLM's real FastIncrementalDetokenizer: the stream gets the
    8-token tail and the output equals the unpatched class's."""
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    prompt = tok.encode(" ".join(CORPUS)).ids
    cont = tok.encode(CORPUS[2] + CORPUS[3]).ids
    assert len(prompt) > ff.DETOK_PROMPT_TAIL_TOKENS
    expected = _detokenize(fast, prompt, cont)
    assert expected[0] is detok_cls and primed == [prompt]
    primed.clear()
    monkeypatch.setattr(ff._DetokStats, "tail", 0)
    assert ff.maybe_install_detok_prompt_tail()
    assert _detokenize(fast, prompt, cont) == expected
    assert primed == [prompt[-ff.DETOK_PROMPT_TAIL_TOKENS :]]
    assert ff._DetokStats.tail == 1


def test_an_error_in_the_tail_logs_and_primes_with_the_full_prompt(
    monkeypatch, detok_cls, primed, tok
):
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    prompt = tok.encode(" ".join(CORPUS)).ids
    cont = tok.encode(CORPUS[0]).ids
    expected = _detokenize(fast, prompt, cont)
    primed.clear()

    def broken(*args):
        raise RuntimeError("decode moved")

    monkeypatch.setattr(ff, "safe_prompt_tail", broken)
    logger = Mock()
    monkeypatch.setattr(ff, "logger", logger)
    monkeypatch.setattr(ff._DetokStats, "error", 0)
    assert ff.maybe_install_detok_prompt_tail()
    assert _detokenize(fast, prompt, cont) == expected
    assert primed == [prompt]
    (call,) = logger.warning_once.call_args_list
    assert "RuntimeError" in call.args and ff._DetokStats.error == 1
    (detail,) = logger.debug.call_args_list[-1:]
    assert "decode moved" in str(detail.args)


def test_repeated_tail_errors_of_one_type_warn_once(monkeypatch, detok_cls, tok):
    """Through vLLM's own warning_once: two failing requests with different
    messages of one error type log one warning, and the warning's arguments
    hold no exception (warning_once caches them)."""
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    prompt = tok.encode(" ".join(CORPUS)).ids
    cont = tok.encode(CORPUS[0]).ids
    messages = iter(["first failure", "second failure"])

    def broken(*args):
        raise RuntimeError(next(messages))

    monkeypatch.setattr(ff, "safe_prompt_tail", broken)
    vllm_logger = ff.logger._resolve()
    warnings = []
    monkeypatch.setattr(
        vllm_logger, "warning", lambda msg, *args, **kw: warnings.append(args)
    )
    _print_warning_once.cache_clear()
    assert ff.maybe_install_detok_prompt_tail()
    _detokenize(fast, prompt, cont)
    _detokenize(fast, prompt, cont)
    (args,) = warnings
    assert all(isinstance(a, str) for a in args) and "RuntimeError" in args
