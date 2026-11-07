# SPDX-License-Identifier: Apache-2.0
"""Exactness of VLLM_TPU_SEGMENT_TOKENIZE (vllm_torchtpu/segment_tokenize.py).

The piecewise path must return the same ids as the backend's own
``encode(text, add_special_tokens=False)`` for every input, cold or cached.
"""

import multiprocessing
import os
import random
import struct
import sys

import pytest
import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from vllm_torchtpu.segment_tokenize import (
    SegmentTokenizer,
    SharedPieceStore,
    tokenizer_digest,
)

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


def _needs_fastokens(fast):
    if fast:
        pytest.importorskip("fastokens")


@pytest.mark.parametrize("fast", [False, True])
def test_toy_tokenizer_matches_full_encode(fast):
    _needs_fastokens(fast)
    tok = _toy_tokenizer()
    assert SegmentTokenizer.eligible(tok) is None
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000, use_fastokens=fast)
    assert (seg._miss_encoder is not seg._backend) == fast
    rng = random.Random(0)
    for i in range(3000):
        text = _random_text(rng, rng.randrange(1, 40))
        want = tok.encode(text, add_special_tokens=False).ids
        assert seg.encode(text) == want, repr(text)
        # Second pass is served (partly) from the cache and must not change.
        assert seg.encode(text) == want, repr(text)
    assert seg.hits > 0 and seg.misses > 0


@pytest.mark.parametrize("fast", [False, True])
def test_edge_inputs(fast):
    _needs_fastokens(fast)
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000, use_fastokens=fast)
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


def test_fastokens_load_failure_keeps_hf_encoder(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastokens", None)
    tok = _toy_tokenizer()
    seg = SegmentTokenizer(tok, max_cached_tokens=10_000, use_fastokens=True)
    assert seg._miss_encoder is seg._backend
    text = "<|user|>" + CORPUS
    assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids


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


@pytest.mark.parametrize("fast", [False, True])
def test_glm_chat_prompts_match_full_encode(fast):
    _needs_fastokens(fast)
    hf = _glm_tokenizer()
    backend = hf.backend_tokenizer
    assert SegmentTokenizer.eligible(backend) is None
    seg = SegmentTokenizer(backend, max_cached_tokens=1 << 22, use_fastokens=fast)
    assert (seg._miss_encoder is not seg._backend) == fast
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


def _long_text(rng, n_parts):
    """Texts whose pieces between added tokens are mostly long enough to cache."""
    out = []
    for _ in range(n_parts):
        out.append(rng.choice(SPECIALS))
        s = rng.randrange(len(CORPUS))
        out.append((CORPUS * 4)[s : s + rng.randrange(60, 300)])
    return "".join(out)


def _shared_seg(tok, root, max_written=1 << 20, max_prompts=0):
    store = SharedPieceStore(root, tokenizer_digest(tok), max_written)
    return SegmentTokenizer(
        tok, max_cached_tokens=1 << 20, max_prompts=max_prompts, shared=store
    )


def test_shared_pieces_across_tokenizers_match_full_encode(tmp_path):
    tok = _toy_tokenizer()
    segs = [_shared_seg(tok, str(tmp_path)) for _ in range(4)]
    rng = random.Random(4)
    history = [_long_text(rng, 4) for _ in range(30)]
    for _ in range(600):
        i = rng.randrange(len(history))
        history[i] += _long_text(rng, 1)
        text = history[i]
        seg = rng.choice(segs)
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids
    assert sum(s.shared_hits for s in segs) > 0
    assert all(s.hits > 0 and s.misses > 0 for s in segs)


@pytest.mark.parametrize("max_prompts", [0, 8])
def test_shared_pieces_with_prefix_reuse_match_full_encode(tmp_path, max_prompts):
    tok = _toy_tokenizer()
    segs = [_shared_seg(tok, str(tmp_path), max_prompts=max_prompts) for _ in range(8)]
    rng = random.Random(9)
    system = _long_text(rng, 3)
    history = [system + _long_text(rng, 3) for _ in range(20)]
    for _ in range(500):
        i = rng.randrange(len(history))
        history[i] += _long_text(rng, 1) + rng.choice(["", "<|user|>", "<a", "b"])
        text = history[i]
        assert (
            rng.choice(segs).encode(text)
            == tok.encode(text, add_special_tokens=False).ids
        )
    assert sum(s.shared_hits for s in segs) > 0


def _encode_in_child(root, texts, queue):
    tok = _toy_tokenizer()
    seg = _shared_seg(tok, root)
    queue.put(([seg.encode(t) for t in texts], seg.shared_hits))


def test_shared_pieces_reach_another_process(tmp_path):
    tok = _toy_tokenizer()
    rng = random.Random(5)
    texts = [_long_text(rng, 6) for _ in range(20)]
    writer = _shared_seg(tok, str(tmp_path))
    for t in texts:
        writer.encode(t)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_encode_in_child, args=(str(tmp_path), texts, queue))
    child.start()
    ids, shared_hits = queue.get(timeout=120)
    child.join(timeout=60)
    assert ids == [tok.encode(t, add_special_tokens=False).ids for t in texts]
    assert shared_hits > 0


def test_shared_file_with_other_text_is_not_used(tmp_path):
    tok = _toy_tokenizer()
    seg = _shared_seg(tok, str(tmp_path))
    store = seg._shared
    piece = CORPUS[:100]
    other = CORPUS[1:101].encode()
    path = store._path(piece.encode())
    with open(path, "wb") as f:
        f.write(SharedPieceStore._HEADER.pack(SharedPieceStore._MAGIC, len(other)))
        f.write(other)
        f.write(struct.pack("=3I", 1, 2, 3))
    assert seg.encode(piece) == tok.encode(piece, add_special_tokens=False).ids
    assert seg.shared_hits == 0


@pytest.mark.parametrize("cut", [0, 3, 11, 50])
def test_truncated_shared_file_is_not_used(tmp_path, cut):
    tok = _toy_tokenizer()
    writer = _shared_seg(tok, str(tmp_path))
    piece = CORPUS[:150]
    writer.encode(piece)
    path = writer._shared._path(piece.encode())
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(data[: len(data) - 1] if cut == 0 else data[:cut])
    reader = _shared_seg(tok, str(tmp_path))
    assert reader.encode(piece) == tok.encode(piece, add_special_tokens=False).ids
    assert reader.shared_hits == 0


def test_shared_writes_are_bounded(tmp_path):
    tok = _toy_tokenizer()
    seg = _shared_seg(tok, str(tmp_path), max_written=2000)
    rng = random.Random(6)
    for _ in range(300):
        seg.encode(_long_text(rng, 3))
    store = seg._shared
    assert 0 < store._written_tokens <= 2000
    assert sorted(os.listdir(store.dir)) == sorted(
        os.path.basename(path) for path, _ in store._written
    )


def test_other_tokenizer_reads_none_of_the_pieces(tmp_path):
    tok = _toy_tokenizer()
    other = _toy_tokenizer(extra_added=[AddedToken("zzqq", normalized=False)])
    assert tokenizer_digest(tok) != tokenizer_digest(other)
    rng = random.Random(7)
    texts = [_long_text(rng, 5) for _ in range(10)]
    first = _shared_seg(tok, str(tmp_path))
    for t in texts:
        first.encode(t)
    second = _shared_seg(other, str(tmp_path))
    for t in texts:
        assert second.encode(t) == other.encode(t, add_special_tokens=False).ids
    assert second.shared_hits == 0


def test_unwritable_shared_dir_keeps_encoding(tmp_path):
    tok = _toy_tokenizer()
    seg = _shared_seg(tok, str(tmp_path))
    not_a_dir = tmp_path / "file"
    not_a_dir.write_bytes(b"")
    seg._shared.dir = str(not_a_dir)
    rng = random.Random(8)
    for _ in range(20):
        text = _long_text(rng, 3)
        assert seg.encode(text) == tok.encode(text, add_special_tokens=False).ids
    assert seg._shared._write_failed
    assert seg.shared_hits == 0
    assert not seg._shared._written
