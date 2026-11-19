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
"""Exactness of VLLM_TPU_SEGMENT_TOKENIZE (vllm_torchtpu/segment_tokenize.py).

The piecewise path must return the ids of the backend's own
``encode(text, add_special_tokens=False)`` for every input, cold or cached,
and the renderer must keep vLLM's tokenization where the output is not a
concatenation of piece encodings.
"""

import random

import pytest
import tokenizers
from tokenizers import (
    AddedToken,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

from vllm_torchtpu.segment_tokenize import SegmentTokenizer

# Message delimiters in the style of chat templates.
SPECIALS = ["]~b]", "[e~[", "]~!b[", "<mm:think>", "<tool_call>", "</tool_call>"]
# Overlapping added tokens exercise leftmost-longest matching.
OVERLAP = ["<a>", "<a>b", "b<a"]

CORPUS = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "  def f(x):\n\treturn x  # 123 4567 89\r\n"
    "naïve café — 東京 😀  \n\n\n   tabs\t\tand  spaces   \n"
    "e\u0301 composed by NFC\n"
)


def _toy_tokenizer(extra_added=(), normalizer=None):
    tok = Tokenizer(models.BPE())
    if normalizer is not None:
        tok.normalizer = normalizer
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
            "]~",
            "b]",
            "[e~",
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
            "\u0301",
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


def _full(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("normalizer", [None, normalizers.NFC()])
def test_random_text_matches_full_encode(normalizer):
    tok = _toy_tokenizer(normalizer=normalizer)
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    rng = random.Random(0)
    for _ in range(3000):
        text = _random_text(rng, rng.randrange(1, 40))
        want = _full(tok, text)
        assert seg.encode(text) == want, repr(text)
        # The second encode is served (partly) from the cache.
        assert seg.encode(text) == want, repr(text)
    assert seg.hits > 0 and seg.misses > 0


def test_growing_conversation_reuses_earlier_turns():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=100_000)
    rng = random.Random(1)
    prompt = "]~!b[]~b]system\n" + CORPUS * 3 + "[e~[\n"
    for turn in range(20):
        role = "user" if turn % 2 == 0 else "ai"
        prompt += f"]~b]{role}\n" + _random_text(rng, 10).replace("]~b]", "") + "[e~[\n"
        hits = seg.hits
        assert seg.encode(prompt + "]~b]ai\n") == _full(tok, prompt + "]~b]ai\n")
        if turn:
            assert seg.hits > hits


def test_edge_inputs():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    for text in [
        "",
        "]~b]",
        "]~b]]~b]",
        "<a>b<a>",
        "b<a>b",
        "<a>bb<a",
        " ]~b] ",
        "\n[e~[\n",
        "x" * 5000 + "<tool_call>" + "y" * 5000,
    ]:
        assert seg.encode(text) == _full(tok, text), repr(text)


def test_cache_is_bounded():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=500)
    rng = random.Random(2)
    for _ in range(200):
        text = "]~b]" + _random_text(rng, 20)
        assert seg.encode(text) == _full(tok, text)
        assert seg.cached_tokens <= 500


def test_truncation_on_the_original_backend_does_not_reach_the_copy():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    text = CORPUS * 3
    want = _full(tok, text)
    tok.enable_truncation(max_length=3)  # what transformers does per call
    assert seg.encode(text) == want


@pytest.mark.parametrize("flag", ["lstrip", "rstrip", "single_word"])
def test_added_token_that_is_not_a_hard_split_is_refused(flag):
    tok = _toy_tokenizer(extra_added=[AddedToken("<x>", **{flag: True})])
    assert SegmentTokenizer.ineligible_reason(tok) is not None
    with pytest.raises(ValueError, match="would not be exact"):
        SegmentTokenizer(tok, max_cached_tokens=10)


def test_normalized_added_token_needs_an_identity_normalizer():
    added = [AddedToken("<x>", normalized=True)]
    assert SegmentTokenizer.ineligible_reason(_toy_tokenizer(added)) is None
    tok = _toy_tokenizer(added, normalizer=normalizers.Lowercase())
    assert SegmentTokenizer.ineligible_reason(tok) is not None


def _metaspace_tokenizer(prepend_scheme):
    vocab = {"[unk]": 0, "\u2581hello": 1, "world": 2, "\u2581world": 3, "<s>": 4}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme=prepend_scheme)
    tok.add_special_tokens([AddedToken("<s>", normalized=False)])
    return tok


def test_metaspace_that_prefixes_only_the_first_piece_is_refused():
    tok = _metaspace_tokenizer("first")
    assert SegmentTokenizer.ineligible_reason(tok) is not None
    with pytest.raises(ValueError, match="would not be exact"):
        SegmentTokenizer(tok, max_cached_tokens=10)


def test_metaspace_that_prefixes_every_piece_is_exact():
    tok = _metaspace_tokenizer("always")
    seg = SegmentTokenizer(tok, max_cached_tokens=10)
    for text in ["hello<s>world", "<s>hello world<s>", "world"]:
        assert seg.encode(text) == _full(tok, text)


def test_probe_refuses_what_the_static_checks_miss(monkeypatch):
    monkeypatch.setattr(
        SegmentTokenizer, "ineligible_reason", staticmethod(lambda backend: None)
    )
    with pytest.raises(ValueError, match="probe"):
        SegmentTokenizer(_metaspace_tokenizer("first"), max_cached_tokens=10)


def test_vocab_with_gaps():
    vocab = {"[unk]": 0, "a": 1, "b": 50}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken("<s>", normalized=False)])
    seg = SegmentTokenizer(tok, max_cached_tokens=10)
    assert seg.encode("a b<s>b") == _full(tok, "a b<s>b")


def test_encode_special_tokens_is_refused():
    tok = _toy_tokenizer()
    tok.encode_special_tokens = True
    assert SegmentTokenizer.ineligible_reason(tok) is not None


# ---------------------------------------------------------------------------
# The renderer.
# ---------------------------------------------------------------------------


@pytest.fixture
def renderer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("vllm")
    from vllm_torchtpu.segment_tokenize import SegmentTokenizeHfRenderer

    toy = _toy_tokenizer()
    hf = transformers.PreTrainedTokenizerFast(tokenizer_object=toy)
    # The constructor needs a VllmConfig; the tokenization path needs only
    # the tokenizer and the segment tokenizer.
    r = object.__new__(SegmentTokenizeHfRenderer)
    r.tokenizer = hf
    r.segment_tokenizer = SegmentTokenizer(hf.backend_tokenizer, 10_000)
    return r


def _params(**kw):
    from vllm.renderers.params import TokenizeParams

    return TokenizeParams(
        **{"max_total_tokens": None, "add_special_tokens": False, **kw}
    )


TEXT = "]~!b[]~b]user\n" + CORPUS + "[e~[\n]~b]ai\n"


def test_renderer_tokenizes_through_the_segment_cache(renderer):
    want = renderer.tokenizer(TEXT, add_special_tokens=False)["input_ids"]
    out = renderer._tokenize_prompt({"prompt": TEXT}, _params())
    assert out["prompt_token_ids"] == want
    assert renderer.segment_tokenizer.misses > 0


@pytest.mark.parametrize(
    "kw",
    [
        {"add_special_tokens": True},
        {"return_token_offsets": True},
        {"max_total_tokens": 4, "truncate_prompt_tokens": 3},
    ],
    ids=["special_tokens", "offsets", "truncated"],
)
def test_renderer_keeps_vllm_tokenization_where_pieces_do_not_compose(renderer, kw):
    from vllm.renderers.hf import HfRenderer

    params = _params(**kw)
    want = HfRenderer._tokenize_prompt(renderer, {"prompt": TEXT}, params)
    renderer.segment_tokenizer.misses = 0
    out = renderer._tokenize_prompt({"prompt": TEXT}, params)
    assert out == want
    if kw != {"max_total_tokens": 4, "truncate_prompt_tokens": 3}:
        assert renderer.segment_tokenizer.misses == 0


def test_renderer_refuses_a_tokenizer_without_a_rust_backend():
    pytest.importorskip("vllm")
    from vllm_torchtpu.segment_tokenize import SegmentTokenizeHfRenderer

    with pytest.raises(ValueError, match="fast tokenizer"):
        SegmentTokenizeHfRenderer(None, object())


@pytest.mark.parametrize("flag", ["0", "1"])
def test_flag_selects_the_hf_renderer(monkeypatch, flag):
    pytest.importorskip("vllm")
    from vllm.renderers import registry

    from vllm_torchtpu.tokenizers import register_tokenizers

    monkeypatch.setenv("VLLM_TPU_SEGMENT_TOKENIZE", flag)
    monkeypatch.setattr(
        registry,
        "RENDERER_REGISTRY",
        registry.RendererRegistry(dict(registry.RENDERER_REGISTRY.renderers)),
    )
    register_tokenizers()
    cls = registry.RENDERER_REGISTRY.load_renderer_cls("hf")
    assert cls.__name__ == (
        "SegmentTokenizeHfRenderer" if flag == "1" else "HfRenderer"
    )
