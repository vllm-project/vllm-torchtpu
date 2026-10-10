# SPDX-License-Identifier: Apache-2.0
"""QX_DETOK_PROMPT_TAIL: a DecodeStream primed with a safe prompt tail must
emit byte-identical text to one primed with the whole prompt."""

import os
import random
import sys
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")

try:
    from vllm_torchtpu import frontend_fix as ff
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "vllm_torchtpu"))
    import frontend_fix as ff

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return {k: v for k, v in x.items()}\n",
    "中文测试：我们今天讨论大型语言模型的推理延迟。日本語のテキスト。한국어 문장.",
    "Emoji 🚀🔥👍🏽 family 👨‍👩‍👧‍👦 flags 🇺🇸🇯🇵 keycap 1️⃣ ✅",
    "Math ∑ α·β ≤ ∞ ∂f/∂x ≈ 0 Ünïcödé ñ ç ø ß 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𓀀",
    "مرحبا بالعالم. नमस्ते दुनिया। Привет, мир!",
]


@pytest.fixture(scope="module")
def tok():
    qwen_dir = os.environ.get("QWEN35_TOKENIZER_DIR")
    if qwen_dir and os.path.isfile(os.path.join(qwen_dir, "tokenizer.json")):
        return Tokenizer.from_file(os.path.join(qwen_dir, "tokenizer.json"))
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIALS,
        show_progress=False,
    )
    t.train_from_iterator(CORPUS * 20, trainer)
    return t


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
        elif r < 0.35:
            out.append(rng.choice(byte_ids))
        else:
            out.append(rng.randrange(v))
    return out


def test_tail_priming_is_byte_identical(tok):
    rng = random.Random(0)
    vocab = tok.get_vocab(with_added_tokens=False)
    from transformers.convert_slow_tokenizer import bytes_to_unicode  # type: ignore

    bu = bytes_to_unicode()
    byte_ids = [vocab[bu[b]] for b in range(0x80, 0x100) if bu[b] in vocab]
    special_ids = [tok.token_to_id(s) for s in SPECIALS if tok.token_to_id(s) is not None]
    assert byte_ids and special_ids
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
    assert n_tail > 100 and n_full > 0


def test_safe_tail_falls_back_on_partial_utf8_and_empty(tok):
    emoji = tok.encode("🚀").ids
    prompt = tok.encode("hello world, this is a test ").ids
    if len(emoji) > 1:
        assert ff.safe_prompt_tail(tok, prompt + emoji[:-1], True, 8) is not None
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


def test_install_patches_vllm_detokenizer(monkeypatch):
    detok = pytest.importorskip("vllm.v1.engine.detokenizer")
    cls = detok.FastIncrementalDetokenizer
    orig = cls.__init__
    monkeypatch.setenv("QX_DETOK_PROMPT_TAIL", "8")
    monkeypatch.setattr(ff, "_DETOK_INSTALLED", False)
    try:
        assert ff.maybe_install_detok_prompt_tail()
        assert getattr(cls.__init__, "__qx_detok_tail__", False)
    finally:
        cls.__init__ = orig
        ff._DETOK_INSTALLED = False


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("QX_DETOK_PROMPT_TAIL", "0")
    monkeypatch.setattr(ff, "_DETOK_INSTALLED", False)
    assert ff.maybe_install_detok_prompt_tail() is False
