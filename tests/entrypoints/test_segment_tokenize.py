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

The piecewise path must return the same ids as the backend's own
``encode(text, add_special_tokens=False)`` for every input, cold or cached.
"""

import gc
import os
import random
import threading
import weakref

import pytest
import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers
from vllm.exceptions import VLLMValidationError
from vllm.renderers import base as renderer_base

from vllm_torchtpu.segment_tokenize import SegmentTokenizer, TokenizerMadeIds

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


def test_packed_ids_equal_the_ids_through_prefix_reuse():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000, max_prompts=4, packed=True)
    rng = random.Random(3)
    reused_total = 0
    for _ in range(200):
        # A conversation grows turn by turn, so later turns reuse earlier ids.
        text = _random_text(rng, rng.randrange(1, 20))
        for _ in range(4):
            prompt, reused = seg.encode_prompt(text)
            assert prompt.ids == tok.encode(text, add_special_tokens=False).ids
            assert prompt.packed is not None and prompt.packed.tolist() == prompt.ids
            reused_total += reused
            text += _random_text(rng, rng.randrange(1, 10))
    assert reused_total > 0


def test_cut_long_pieces_match_full_encode():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000, part_chars=16)
    assert seg._part_chars == 16
    rng = random.Random(1)
    cuts = 0
    for _ in range(2000):
        text = _random_text(rng, rng.randrange(1, 60))
        pieces = [p for p in seg._split.split(text) if p]
        cuts += len(seg._cut(pieces)[0]) - len(pieces)
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids
    assert cuts > 1000


def test_cut_needs_parts_that_pre_tokenize_as_the_whole(monkeypatch):
    import vllm_torchtpu.segment_tokenize as st

    def ids(tok, text):
        return tok.encode(text, add_special_tokens=False).ids

    warnings = []
    monkeypatch.setattr(st.logger, "warning", lambda *args: warnings.append(args))
    # GPT-2's ByteLevel regex splits "w \nx" into " ", "\n", "x" but the part
    # "w \n" alone into " \n"; add_prefix_space also puts a space before "x".
    for add_prefix_space in (False, True):
        tok = _toy_tokenizer()
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        assert ids(tok, "abcd \n") + ids(tok, "efgh") != ids(tok, "abcd \nefgh")
        assert SegmentTokenizer(tok, 10_000, part_chars=4)._part_chars == 0
        assert "ByteLevel" in warnings[-1][1] and "\\n" in warnings[-1][2]
    tok = _toy_tokenizer()  # its split regex keeps "\s*[\r\n]+" as 1 pre-token
    seg = SegmentTokenizer(tok, 10_000, part_chars=4)
    assert seg._part_chars == 4
    for text in ["abcd \nefgh", "abcd\r\nefgh", "abcd\t\nefgh", "abcdefgh\nijkl"]:
        assert len(seg._cut([text])[0]) == 2
        assert seg.encode(text) == ids(tok, text)
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    assert SegmentTokenizer(tok, 10_000, part_chars=16)._part_chars == 0


def test_cold_pieces_encode_in_parallel():
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    both_inside = threading.Barrier(2, timeout=5)

    class Backend:
        def encode_batch(self, pieces, add_special_tokens):
            both_inside.wait()
            return tok.encode_batch(pieces, add_special_tokens=add_special_tokens)

    seg._backend = Backend()
    texts = ["<|user|>" + "a" * 100, "<|user|>" + "b" * 100]
    results = {}

    def encode(text):
        try:
            results[text] = seg.encode(text)
        except threading.BrokenBarrierError:
            results[text] = None

    threads = [threading.Thread(target=encode, args=(t,)) for t in texts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for text in texts:
        assert results[text] == tok.encode(text, add_special_tokens=False).ids


def test_cached_instance_keeps_its_tokenizer(monkeypatch):
    import vllm_torchtpu.segment_tokenize as st

    monkeypatch.setattr(st, "_INSTANCES", {})

    class Wrapper:
        def __init__(self, backend):
            self._tokenizer = backend

    tokenizer = Wrapper(_toy_tokenizer())
    alive = weakref.ref(tokenizer)
    st.get_segment_tokenizer(tokenizer, 10_000)
    del tokenizer
    gc.collect()
    assert alive() is not None


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


@pytest.mark.parametrize("flag", ["lstrip", "rstrip", "single_word"])
def test_ineligible_added_tokens(flag):
    tok = _toy_tokenizer(extra_added=[AddedToken("<x>", **{flag: True})])
    assert SegmentTokenizer.eligible(tok) is not None


NORMALIZED = ["<｜User｜>", "<｜Assistant｜>", "<dsml:", "</dsml:", "｜DSML｜"]


def _normalized_atoms_text(rng, n_parts):
    atoms = NORMALIZED + ["<｜", "｜>", "<dsml", "</", "｜DS", "ML｜", ":"]
    return "".join(
        rng.choice(atoms) if rng.random() < 0.4 else _random_text(rng, 2)
        for _ in range(n_parts)
    )


def test_normalized_added_tokens_match_full_encode():
    """Normalized added tokens under a normalizer that keeps the text, as
    DeepSeek-V4.1 has."""
    tok = _toy_tokenizer(
        extra_added=[AddedToken(t, normalized=True) for t in NORMALIZED]
    )
    assert SegmentTokenizer.eligible(tok) is None
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000, max_prompts=8)
    rng = random.Random(4)
    for _ in range(3000):
        text = _normalized_atoms_text(rng, rng.randrange(1, 30))
        want = tok.encode(text, add_special_tokens=False).ids
        assert seg.encode(text) == want, repr(text)
        longer = text + _normalized_atoms_text(rng, rng.randrange(1, 10))
        assert seg.encode(longer) == tok.encode(longer, add_special_tokens=False).ids, (
            repr(longer)
        )


@pytest.mark.parametrize(
    "token", ["<|user", "|>x", "sop", "q<|user", "<|assistant|><sop>"]
)
def test_normalized_token_overlapping_a_raw_one_is_ineligible(token):
    tok = _toy_tokenizer(extra_added=[AddedToken(token, normalized=True)])
    assert SegmentTokenizer.eligible(tok) is not None


def test_normalized_token_under_a_changing_normalizer_is_ineligible():
    from tokenizers import normalizers

    tok = _toy_tokenizer(extra_added=[AddedToken("<x>", normalized=True)])
    tok.normalizer = normalizers.Lowercase()
    assert SegmentTokenizer.eligible(tok) is not None
    tok.normalizer = normalizers.Sequence([])
    assert SegmentTokenizer.eligible(tok) is None


def test_tokenizer_without_added_tokens_is_ineligible(monkeypatch):
    """With nothing to split at, the split pattern would be empty and match
    at every position; the original path runs instead."""
    import vllm_torchtpu.segment_tokenize as st

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(
        [CORPUS] * 5,
        trainer=trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    assert tok.get_added_tokens_decoder() == {}
    assert SegmentTokenizer.eligible(tok) is not None

    class Wrapper:
        def __init__(self, backend):
            self._tokenizer = backend

    monkeypatch.setattr(st, "_INSTANCES", {})
    assert st.get_segment_tokenizer(Wrapper(tok), 10_000) is None


def _metaspace_tokenizer(prepend_scheme):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Metaspace(prepend_scheme=prepend_scheme)]
    )
    tok.decoder = decoders.Metaspace(prepend_scheme=prepend_scheme)
    tok.train_from_iterator([CORPUS] * 50, trainer=trainers.BpeTrainer(vocab_size=400))
    tok.add_special_tokens([AddedToken(s, normalized=False) for s in SPECIALS])
    return tok


def test_metaspace_marking_only_the_first_split_is_ineligible():
    """A piece encoded alone is the first split of its own text, so it would
    get the marker the full encode gives only to the text's first piece."""
    tok = _metaspace_tokenizer("first")
    user, assistant = tok.token_to_id("<|user|>"), tok.token_to_id("<|assistant|>")
    full = tok.encode("To be<|user|>or not<|assistant|> to be").ids
    middle = full[full.index(user) + 1 : full.index(assistant)]
    assert tok.encode("or not", add_special_tokens=False).ids != middle
    assert SegmentTokenizer.eligible(tok) is not None


def test_metaspace_marking_every_split_matches_full_encode():
    tok = _metaspace_tokenizer("always")
    assert SegmentTokenizer.eligible(tok) is None
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000)
    rng = random.Random(5)
    for _ in range(500):
        text = _random_text(rng, rng.randrange(1, 20))
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids


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
    """The patched BaseRenderer._tokenize_prompt, on vLLM's own renderer and
    TokenizeParams: the fast path returns what the original returns, and
    special tokens, truncation and offsets go to the original."""
    transformers = pytest.importorskip("transformers")
    from vllm.renderers import base as renderer_base
    from vllm.renderers.params import TokenizeParams

    from vllm_torchtpu import segment_tokenize as st

    cls = renderer_base.BaseRenderer
    original = cls._tokenize_prompt
    fallbacks = []

    def counted_original(self, prompt, params):
        fallbacks.append(params)
        return original(self, prompt, params)

    monkeypatch.setattr(cls, "_tokenize_prompt", counted_original)
    monkeypatch.setattr(cls, "validate_token_ids", cls.validate_token_ids)
    monkeypatch.setattr(cls, "_tpu_segment_tokenize", False, raising=False)
    monkeypatch.setattr(st, "_INSTANCES", {})
    monkeypatch.setattr(st, "TOKENIZER_MADE_IDS", TokenizerMadeIds())
    monkeypatch.setenv("VLLM_TPU_SEGMENT_TOKENIZE", "1")
    st.patch_renderer_segment_tokenize()
    assert cls._tokenize_prompt is not counted_original

    class OffsetsRenderer(cls):
        def render_messages(self, messages, params):
            raise NotImplementedError

        def _can_produce_offsets(self):
            return True

    # __init__ needs a whole VllmConfig; the tokenize path reads only the
    # tokenizer.
    renderer = object.__new__(OffsetsRenderer)
    renderer.tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=_toy_tokenizer()
    )
    prompt = {"prompt": "[gMASK]<sop><|user|>To be, or not<|assistant|>to be"}

    plain = TokenizeParams(max_total_tokens=None, add_special_tokens=False)
    fast = renderer._tokenize_prompt(dict(prompt), plain)
    assert fallbacks == []
    assert fast == original(renderer, dict(prompt), plain)
    assert st._INSTANCES

    for params in (
        TokenizeParams(max_total_tokens=None, add_special_tokens=True),
        TokenizeParams(
            max_total_tokens=64, truncate_prompt_tokens=2, add_special_tokens=False
        ),
        TokenizeParams(
            max_total_tokens=None, add_special_tokens=False, return_token_offsets=True
        ),
    ):
        got = renderer._tokenize_prompt(dict(prompt), params)
        assert fallbacks[-1] is params
        assert got == original(renderer, dict(prompt), params)
    assert len(fallbacks) == 3
    assert len(got["prompt_token_offsets"]) == len(got["prompt_token_ids"])


def test_tokenizer_made_ids_are_taken_once_by_identity():
    made = TokenizerMadeIds(max_tokens=4)
    ids = [1, 2, 3]
    made.add(ids)
    assert not made.take([1, 2, 3])
    assert made.take(ids)
    assert not made.take(ids)

    made.add(ids)
    ids.append(4)
    assert not made.take(ids)

    first, second, third = [1, 1], [2], [3]
    for lst in (first, second, third):
        made.add(lst)
    assert len(made) == 3
    fourth = [4, 4]
    made.add(fourth)
    assert len(made) == 3
    assert not made.take(first)
    assert made.take(second) and made.take(third) and made.take(fourth)

    whole = list(range(10))
    made.add([0])
    made.add(whole)
    assert len(made) == 1 and made.take(whole)


def test_vocabulary_check_skips_only_segment_tokenizer_ids(monkeypatch):
    from vllm_torchtpu import segment_tokenize as st

    toy = _toy_tokenizer()

    class HF:
        backend_tokenizer = toy
        max_token_id = toy.get_vocab_size(with_added_tokens=True) - 1

    class ModelConfig:
        @staticmethod
        def get_vocab_size():
            return HF.max_token_id + 1

    class Params:
        def get_encode_kwargs(self):
            return dict(truncation=False, max_length=None, add_special_tokens=False)

    cls = renderer_base.BaseRenderer
    monkeypatch.setattr(cls, "_tokenize_prompt", cls._tokenize_prompt)
    monkeypatch.setattr(cls, "validate_token_ids", cls.validate_token_ids)
    monkeypatch.setattr(cls, "_tpu_segment_tokenize", False, raising=False)
    monkeypatch.setenv("VLLM_TPU_SEGMENT_TOKENIZE", "1")
    monkeypatch.setattr(st, "_INSTANCES", {})
    monkeypatch.setattr(st, "TOKENIZER_MADE_IDS", TokenizerMadeIds())
    st.patch_renderer_segment_tokenize()

    class R:
        _tokenize_prompt = cls._tokenize_prompt
        validate_token_ids = cls.validate_token_ids
        _build_tokens_prompt = staticmethod(cls._build_tokens_prompt)
        tokenizer = HF()
        model_config = ModelConfig()

        def get_tokenizer(self):
            return self.tokenizer

        def _wants_offsets(self, prompt, params):
            return False

    r = R()
    text = "[gMASK]<sop><|user|>" + CORPUS + "<|assistant|>"
    ids = r._tokenize_prompt({"prompt": text}, Params())["prompt_token_ids"]
    assert ids == toy.encode(text, add_special_tokens=False).ids
    assert len(st.TOKENIZER_MADE_IDS) == 1

    # A vocabulary one id smaller makes vLLM's check fail on these ids,
    # so passing proves the check was skipped.
    HF.max_token_id = max(ids) - 1
    r.validate_token_ids(ids)
    # Taken once: a second check of the same list runs in full.
    with pytest.raises(VLLMValidationError):
        r.validate_token_ids(ids)

    HF.max_token_id = toy.get_vocab_size(with_added_tokens=True) - 1
    out_of_vocab = HF.max_token_id + 1
    for sent in ([5, out_of_vocab], [5, -1]):
        with pytest.raises(VLLMValidationError):
            r.validate_token_ids(sent)

    grown = r._tokenize_prompt({"prompt": text}, Params())["prompt_token_ids"]
    grown.append(out_of_vocab)
    with pytest.raises(VLLMValidationError):
        r.validate_token_ids(grown)
