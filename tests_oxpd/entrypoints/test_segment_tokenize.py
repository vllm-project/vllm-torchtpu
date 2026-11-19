# SPDX-License-Identifier: Apache-2.0
"""Exactness of VLLM_TPU_SEGMENT_TOKENIZE (vllm_torchtpu/segment_tokenize.py).

The piecewise path must return the same ids as the backend's own
``encode(text, add_special_tokens=False)`` for every input, cold or cached.
"""

import os
import random

import pytest
import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from vllm_torchtpu.segment_tokenize import SegmentTokenizer

SPECIALS = [
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "[gMASK]",
    "<sop>",
    "<think>",
    "</think>",
    "/nothink",
]
# Overlapping added tokens exercise leftmost-longest matching.
OVERLAP = ["<a>", "<a>b", "b<a"]

CORPUS = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "  def f(x):\n\treturn x  # 123 4567 89\r\n"
    "naïve café — 東京 😀  \n\n\n   tabs\t\tand  spaces   \n"
)


def _toy_tokenizer(extra_added=()):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                tokenizers.Regex(
                    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
                    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator([CORPUS] * 50, trainer=trainer)
    tok.add_special_tokens([AddedToken(s, normalized=False) for s in SPECIALS])
    tok.add_tokens([AddedToken(s, normalized=False) for s in OVERLAP])
    for t in extra_added:
        tok.add_tokens([t])
    return tok


def _random_text(rng, n_parts):
    atoms = (
        SPECIALS
        + OVERLAP
        + [
            "<",
            "|",
            "<|user",
            "user|>",
            "<a",
            "a>",
            "b",
            "\n",
            "\r\n",
            " ",
            "  ",
            "\t",
            "'s",
            "123456",
            "😀",
            "東京",
        ]
    )
    out = []
    for _ in range(n_parts):
        if rng.random() < 0.5:
            out.append(rng.choice(atoms))
        else:
            s = rng.randrange(len(CORPUS))
            out.append(CORPUS[s : s + rng.randrange(1, 120)])
    return "".join(out)


def test_toy_tokenizer_matches_full_encode():
    tok = _toy_tokenizer()
    assert SegmentTokenizer.eligible(tok) is None
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    rng = random.Random(0)
    for i in range(3000):
        text = _random_text(rng, rng.randrange(1, 40))
        want = tok.encode(text, add_special_tokens=False).ids
        assert seg.encode(text) == want, repr(text)
        # Second pass is served (partly) from the cache and must not change.
        assert seg.encode(text) == want, repr(text)
    assert seg.hits > 0 and seg.misses > 0


def test_edge_inputs():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    for text in [
        "",
        "<|user|>",
        "<|user|><|user|>",
        "<a>b<a>",
        "b<a>b",
        "<a>bb<a",
        " <|user|> ",
        "\n<|assistant|>\n",
        "x" * 5000 + "<sop>" + "y" * 5000,
    ]:
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids, repr(
            text
        )


def test_cache_is_bounded():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=500)
    rng = random.Random(1)
    for _ in range(200):
        text = "<|user|>" + _random_text(rng, 20)
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids
        assert seg._cached_tokens <= 500


def test_shared_backend_truncation_does_not_leak():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    tok.enable_truncation(max_length=3)  # what transformers does per call
    text = CORPUS * 3
    tok.no_truncation()
    want = tok.encode(text, add_special_tokens=False).ids
    tok.enable_truncation(max_length=3)
    assert seg.encode(text) == want


@pytest.mark.parametrize("flag", ["lstrip", "rstrip", "single_word", "normalized"])
def test_ineligible_added_tokens(flag):
    tok = _toy_tokenizer(extra_added=[AddedToken("<x>", **{flag: True})])
    assert SegmentTokenizer.eligible(tok) is not None


def test_encode_special_tokens_ineligible():
    tok = _toy_tokenizer()
    tok.encode_special_tokens = True
    assert SegmentTokenizer.eligible(tok) is not None


GLM = os.environ.get("SEGMENT_TOKENIZE_TEST_MODEL", "zai-org/GLM-5.2-FP8")


def _glm_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(GLM)
    except Exception as e:  # no network / no cache
        pytest.skip(f"{GLM} tokenizer unavailable: {e}")


def _conversation(rng, n_msgs, words):
    vocab = CORPUS.split() + ["<|user|>", "</tool_call>", "{", "}", "\n\n", "```", "😀"]
    msgs = [
        {"role": "system", "content": " ".join(rng.choice(vocab) for _ in range(words))}
    ]
    for i in range(n_msgs):
        body = " ".join(rng.choice(vocab) for _ in range(rng.randrange(1, words)))
        role = ["user", "assistant", "tool"][i % 3]
        m = {"role": role, "content": body}
        if role == "assistant":
            m["tool_calls"] = [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": {"command": body[:40]}},
                }
            ]
        if role == "tool":
            m["tool_call_id"] = f"c{i - 1}"
        msgs.append(m)
    msgs.append({"role": "user", "content": "go on"})
    return msgs


def test_glm_chat_prompts_match_full_encode():
    hf = _glm_tokenizer()
    backend = hf.backend_tokenizer
    assert SegmentTokenizer.eligible(backend) is None
    seg = SegmentTokenizer(backend, max_cached_tokens=1 << 22)
    rng = random.Random(2)
    for _ in range(40):
        msgs = _conversation(rng, rng.randrange(1, 30), 300)
        # Continue the conversation turn by turn, as an agent does.
        for turn in range(3):
            prompt = hf.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            want = hf(prompt, add_special_tokens=False)["input_ids"]
            assert seg.encode(prompt) == want
            msgs = msgs[:-1] + _conversation(rng, 3, 200)[1:]
    assert seg.hits > 0


def test_glm_same_block_hashes():
    """Same ids give the same vLLM prefix-cache block hashes (and so hits)."""
    pytest.importorskip("vllm")
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

    hf = _glm_tokenizer()
    seg = SegmentTokenizer(hf.backend_tokenizer, max_cached_tokens=1 << 22)
    rng = random.Random(3)
    msgs = _conversation(rng, 60, 400)
    prompt = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    a = hf(prompt, add_special_tokens=False)["input_ids"]
    b = seg.encode(prompt)
    fn = get_hash_fn_by_name("sha256")
    init_none_hash(fn)

    def hashes(ids, bs=256):
        out, prev = [], None
        for s in range(0, len(ids) - bs + 1, bs):
            prev = hash_block_tokens(fn, prev, ids[s : s + bs])
            out.append(prev)
        return out

    assert len(a) >= 256 and hashes(a) == hashes(b)


def test_renderer_patch_routes_and_falls_back(monkeypatch):
    pytest.importorskip("vllm")
    from vllm.renderers import base as renderer_base

    from vllm_torchtpu import envs
    from vllm_torchtpu import segment_tokenize as st

    hf = _glm_tokenizer()
    cls = renderer_base.BaseRenderer
    saved = cls.__dict__["_tokenize_prompt"]
    monkeypatch.setenv("VLLM_TPU_SEGMENT_TOKENIZE", "1")
    try:
        st.patch_renderer_segment_tokenize()
        assert cls._tpu_segment_tokenize

        class P:  # minimal TokenizeParams stand-in
            def __init__(self, add_special_tokens=False, max_length=None):
                self.a, self.m = add_special_tokens, max_length

            def get_encode_kwargs(self):
                return dict(
                    truncation=self.m is not None,
                    max_length=self.m,
                    add_special_tokens=self.a,
                )

        class R:
            _tokenize_prompt = cls._tokenize_prompt

            def get_tokenizer(self):
                return hf

            def _wants_offsets(self, prompt, params):
                return False

            def _build_tokens_prompt(self, ids, prompt, offset_mapping=None):
                return {"prompt_token_ids": ids, "fast": True}

        calls = []
        monkeypatch.setattr(st, "_INSTANCES", {})
        r = R()
        text = "[gMASK]<sop><|user|>hello world<|assistant|>"
        out = r._tokenize_prompt({"prompt": text}, P())
        assert (
            out["fast"]
            and out["prompt_token_ids"]
            == hf(text, add_special_tokens=False)["input_ids"]
        )

        # Over-long prompts and add_special_tokens go to the original path.
        def orig(self, prompt, params):
            return calls.append(params) or {"orig": True}

        monkeypatch.setattr(st, "_INSTANCES", {})
        cls._tpu_segment_tokenize = False
        cls._tokenize_prompt = orig
        st.patch_renderer_segment_tokenize()
        R._tokenize_prompt = cls._tokenize_prompt
        assert r._tokenize_prompt({"prompt": text}, P(add_special_tokens=True)) == {
            "orig": True
        }
        assert r._tokenize_prompt({"prompt": text}, P(max_length=2)) == {"orig": True}
        assert len(calls) == 2
    finally:
        cls._tokenize_prompt = saved
        cls._tpu_segment_tokenize = False
        assert envs.VLLM_TPU_SEGMENT_TOKENIZE is True
