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
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from vllm.renderers import base as rbase
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import TokenizeParams
from vllm.tokenizers.hf import get_cached_tokenizer

from vllm_torchtpu import frontend_tokcache as tokcache_mod


class TestFrontendTokCache(unittest.TestCase):
    def setUp(self):
        self.bid = 151644
        self.boundary = tokcache_mod.BOUNDARY

        def mock_encode(s: str) -> list[int]:
            parts = s.split(self.boundary)
            out: list[int] = []
            for idx, part in enumerate(parts):
                if idx > 0:
                    out.append(self.bid)
                out.extend(ord(ch) for ch in part)
            return out

        self.encode_fn = mock_encode

    def test_incremental_multiturn_and_branching(self):
        cache = tokcache_mod.PrefixTokenCache(
            self.encode_fn,
            self.bid,
            min_tokens=4,
            verify_first=64,
        )
        base = f"{self.boundary}system\nYou are an assistant 🚀 中文.<|im_end|>\n"
        cur = base
        branch_base = ""
        for t in range(1, 20):
            prompt = (
                cur
                + f"{self.boundary}user\nTurn {t} αβγ 🎉<|im_end|>\n"
                + f"{self.boundary}assistant\n"
            )
            ids = cache.encode(prompt)
            self.assertEqual(ids, self.encode_fn(prompt))
            if t == 8:
                branch_base = cur
            cur = prompt + f"Reply {t}.<|im_end|>\n"

        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 18)
        self.assertFalse(cache.disabled)

        # Branched conversation from turn 8 (falls back to _index prefix lookup)
        cur = branch_base
        for t in range(8, 12):
            prompt = (
                cur
                + f"{self.boundary}user\nBranched {t}<|im_end|>\n"
                + f"{self.boundary}assistant\n"
            )
            ids = cache.encode(prompt)
            self.assertEqual(ids, self.encode_fn(prompt))
            cur = prompt + f"Branched reply {t}.<|im_end|>\n"

        # Same-turn suffix edit (no new boundary in suffix)
        edited = prompt + "extra text without boundary"
        self.assertEqual(cache.encode(edited), self.encode_fn(edited))
        self.assertFalse(cache.disabled)


B = tokcache_mod.BOUNDARY
BID = 151644


def _encode(s: str, *, context_after_bang: bool = False, fake_boundary: str = ""):
    """A character-level encoder with BID for each boundary.

    ``context_after_bang``: a character right after a boundary that follows
    "!" gets a different id, so encoding the text after that boundary on its
    own differs from encoding the whole prompt (a merge across the boundary).
    ``fake_boundary``: this substring also encodes to BID (more boundary
    tokens than boundary substrings)."""
    if fake_boundary:
        s = s.replace(fake_boundary, "\x00")
    out: list[int] = []
    for i, part in enumerate(s.split(B)):
        ids = [BID if ch == "\x00" else ord(ch) for ch in part]
        if i > 0:
            out.append(BID)
            if context_after_bang and ids and out[-2:-1] == [ord("!")]:
                ids[0] += 100_000
        out.extend(ids)
    return out


def _chat(turns: int, final: str = "") -> str:
    """A chat prompt whose turns each end with "!" before the next boundary."""
    body = "".join(f"{B}user\nturn {i} text!" for i in range(turns))
    return body + f"{B}user\n{final}"


def _counting(encode):
    calls = []

    def fn(s):
        calls.append(s)
        return encode(s)

    return fn, calls


def test_a_verify_mismatch_returns_the_full_encode_and_turns_the_cache_off():
    def encode(s):
        return _encode(s, context_after_bang=True)

    fn, calls = _counting(encode)
    cache = tokcache_mod.PrefixTokenCache(fn, BID, min_tokens=1)
    first = _chat(2)
    assert cache.encode(first) == encode(first)  # a miss, cached
    second = _chat(3)
    # The cached prefix plus the suffix encoded alone differ from the full
    # encode; the check of the first hits catches it.
    assert cache.encode(second) == encode(second)
    assert cache.disabled and cache.verified == 1 and cache.hits == 1
    calls.clear()
    third = _chat(4)
    assert cache.encode(third) == encode(third)
    assert calls == [third]  # later prompts take the full encode only
    assert cache.calls == 2


def test_a_suffix_with_an_extra_boundary_token_takes_the_full_encode(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(tokcache_mod, "logger", logger)

    def encode(s):
        return _encode(s, fake_boundary="@@")

    cache = tokcache_mod.PrefixTokenCache(encode, BID, min_tokens=1)
    first = _chat(2)
    assert cache.encode(first) == encode(first)
    second = _chat(2, final="look @@ here")
    assert cache.encode(second) == encode(second)
    (call,) = logger.warning_once.call_args_list
    assert "boundary/token mismatch" in call.args[0]
    assert cache.hits == 0 and not cache.disabled


def _conversation(name: str, turns: int = 2) -> str:
    """A chat prompt whose first turn is its own, so its prefix keys are too."""
    return f"{B}system\nconversation {name}!" + _chat(turns)


def test_the_cache_evicts_the_least_recently_used_prompts_to_its_limits():
    cache = tokcache_mod.PrefixTokenCache(_encode, BID, min_tokens=1, max_entries=2)
    prompts = [_conversation(str(i)) for i in range(3)]
    sizes = [len(_encode(p)) for p in prompts]
    for p in prompts:
        cache.encode(p)
    assert len(cache._entries) == 2 and cache._total_tokens == sum(sizes[1:])
    # Each conversation owns its keys: the evicted one's left the index with
    # it, and the cached ones' stay.
    evicted = prompts[0].encode()
    evicted_keys, _ = cache._prefix_keys_with_state(evicted, cache._boundaries(evicted))
    assert evicted_keys and not set(evicted_keys) & set(cache._index)
    for entry in cache._entries.values():
        assert set(entry.keys) <= set(cache._index)

    by_tokens = tokcache_mod.PrefixTokenCache(
        _encode, BID, min_tokens=1, max_tokens=sizes[0] + sizes[1]
    )
    for p in prompts:
        by_tokens.encode(p)
    assert len(by_tokens._entries) == 2
    assert by_tokens._total_tokens <= sizes[0] + sizes[1]


def test_eviction_keeps_a_shared_prefix_key_the_newer_entry_owns():
    """Two conversations that share their first turns share those prefix
    keys; the index points them at the newer entry, and evicting the older
    one leaves them in place."""
    cache = tokcache_mod.PrefixTokenCache(_encode, BID, min_tokens=1, max_entries=1)
    cache.encode(_chat(2, final="older"))
    cache.encode(_chat(2, final="newer"))
    ((newer_eid, entry),) = cache._entries.items()
    assert entry.keys and set(entry.keys) <= set(cache._index)
    assert all(cache._index[k][0] == newer_eid for k in entry.keys)


def _toy_hf_tokenizer():
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[tokcache_mod.BOUNDARY, "<|im_end|>"],
        show_progress=False,
    )
    t.train_from_iterator(["user hi assistant hello world"] * 20, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=t)


def test_installed_cache_uses_the_deployed_sizes():
    """The installed cache keeps at most 2048 prompts and, by default, 256M
    token ids; the token limit is the caller's when given."""
    cache = tokcache_mod._build_cache_for(_toy_hf_tokenizer())
    assert cache is not None
    assert (cache._max_entries, cache._max_tokens) == (2048, 256_000_000)
    smaller = tokcache_mod._build_cache_for(_toy_hf_tokenizer(), 64_000_000)
    assert smaller is not None and smaller._max_tokens == 64_000_000
    assert (cache._min_tokens, cache._verify_first, cache._verify_every) == (
        2048,
        8,
        0,
    )


def test_only_a_hugging_face_fast_tokenizer_gets_a_cache():
    """The cache copies the Rust tokenizer behind a PreTrainedTokenizerFast; any
    other tokenizer keeps vLLM's tokenization, whatever attributes it has."""
    fast = _toy_hf_tokenizer()
    assert tokcache_mod._build_cache_for(fast) is not None
    lookalike = SimpleNamespace(backend_tokenizer=fast.backend_tokenizer)
    assert tokcache_mod._build_cache_for(lookalike) is None


@pytest.fixture
def patched_renderer(monkeypatch):
    """A real HfRenderer around the toy tokenizer, with the tokenization cache
    installed on BaseRenderer; returns it and vLLM's unpatched method."""
    unpatched = rbase.BaseRenderer._tokenize_prompt
    # Restored at teardown, after the install below replaces it.
    monkeypatch.setattr(rbase.BaseRenderer, "_tokenize_prompt", unpatched)
    monkeypatch.setattr(tokcache_mod, "_INSTALLED", False)
    assert tokcache_mod.maybe_install_tokenize_prefix_cache()
    renderer = object.__new__(HfRenderer)
    renderer.tokenizer = _toy_hf_tokenizer()
    return renderer, unpatched


@pytest.fixture
def cache_encodes(monkeypatch):
    """The PrefixTokenCache instances that encoded a prompt."""
    caches = []
    encode = tokcache_mod.PrefixTokenCache.encode

    def spy(self, text):
        caches.append(self)
        return encode(self, text)

    monkeypatch.setattr(tokcache_mod.PrefixTokenCache, "encode", spy)
    return caches


def _turns(n):
    """A chat prompt of ``n`` turns, each long enough that the first already
    passes the cache's 2048-token minimum."""
    b = tokcache_mod.BOUNDARY
    body = "hello world user hi " * 600
    turns = [f"{b}user\n{body}{i}<|im_end|>\n{b}assistant\n" for i in range(n)]
    return "".join(turns)


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_patched_renderer_matches_vllms_tokens_on_a_multi_turn_chat(
    patched_renderer, cache_encodes, add_special_tokens
):
    renderer, unpatched = patched_renderer
    params = TokenizeParams(
        max_total_tokens=None, add_special_tokens=add_special_tokens
    )
    for n in (1, 2, 3):
        prompt = {"prompt": _turns(n)}
        expected = unpatched(renderer, dict(prompt), params)
        assert renderer._tokenize_prompt(dict(prompt), params) == expected
    # The later turns reused the first turns' tokens through the cache.
    (cache,) = set(cache_encodes)
    assert cache.hits >= 2 and not cache.disabled


def test_first_unexpected_error_turns_the_cache_off(patched_renderer, monkeypatch):
    """The failing prompt and every later one get vLLM's own tokens, and the
    cache is not tried again after the first error."""
    renderer, unpatched = patched_renderer
    calls = []

    def broken(self, text):
        calls.append(text)
        raise RuntimeError("tokenizer state lost")

    monkeypatch.setattr(tokcache_mod.PrefixTokenCache, "encode", broken)
    params = TokenizeParams(max_total_tokens=None)
    for n in (1, 2, 3):
        prompt = {"prompt": _turns(n)}
        expected = unpatched(renderer, dict(prompt), params)
        assert renderer._tokenize_prompt(dict(prompt), params) == expected
    assert len(calls) == 1


class _SilentLogger:
    """Drops every record: a captured record would keep the exception, its
    frames and the cache alive."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.mark.parametrize("cause", ["error", "verify_mismatch"])
def test_a_cache_that_turns_off_is_dropped(patched_renderer, monkeypatch, cause):
    """Once the cache turns off, by an unexpected error or a verify mismatch,
    the wrapper drops it, so its entries are freed."""
    renderer, unpatched = patched_renderer
    monkeypatch.setattr(tokcache_mod, "logger", _SilentLogger())
    refs = []
    build = tokcache_mod._build_cache_for

    def tracked(*args, **kwargs):
        cache = build(*args, **kwargs)
        refs.append(weakref.ref(cache))
        return cache

    monkeypatch.setattr(tokcache_mod, "_build_cache_for", tracked)
    params = TokenizeParams(max_total_tokens=None)
    renderer._tokenize_prompt({"prompt": _turns(1)}, params)
    (ref,) = refs
    assert ref()._entries
    if cause == "error":

        def broken(self, text):
            raise RuntimeError("tokenizer state lost")

        monkeypatch.setattr(tokcache_mod.PrefixTokenCache, "encode", broken)
    else:
        # The cached prompt's token ids no longer match a full encode, so the
        # next hit's verify turns the cache off and returns the full encode.
        (entry,) = ref()._entries.values()
        entry.ids[0] += 1
    prompt = {"prompt": _turns(2)}
    expected = unpatched(renderer, dict(prompt), params)
    assert renderer._tokenize_prompt(dict(prompt), params) == expected
    if cause == "verify_mismatch":
        assert ref().disabled
    # The wrapper drops a cache the step before turned off.
    renderer._tokenize_prompt({"prompt": _turns(3)}, params)
    gc.collect()
    assert ref() is None


def test_the_cache_serves_vllms_cached_tokenizer(patched_renderer, cache_encodes):
    """vLLM's renderer holds its tokenizer wrapped by get_cached_tokenizer, a
    subclass of the Hugging Face tokenizer's own class; the cache takes it as
    a fast tokenizer and serves the prompts it reuses."""
    renderer, unpatched = patched_renderer
    renderer.tokenizer = get_cached_tokenizer(_toy_hf_tokenizer())
    assert type(renderer.tokenizer) is not PreTrainedTokenizerFast
    params = TokenizeParams(max_total_tokens=None)
    for n in (1, 2):
        prompt = {"prompt": _turns(n)}
        expected = unpatched(renderer, dict(prompt), params)
        assert renderer._tokenize_prompt(dict(prompt), params) == expected
    (cache,) = set(cache_encodes)
    assert cache.hits == 1 and not cache.disabled


def test_installed_wrapper_builds_its_cache_with_the_token_limit(
    monkeypatch, cache_encodes
):
    monkeypatch.setattr(
        rbase.BaseRenderer, "_tokenize_prompt", rbase.BaseRenderer._tokenize_prompt
    )
    monkeypatch.setattr(tokcache_mod, "_INSTALLED", False)
    assert tokcache_mod.maybe_install_tokenize_prefix_cache(max_tokens=64_000_000)
    renderer = object.__new__(HfRenderer)
    renderer.tokenizer = _toy_hf_tokenizer()
    renderer._tokenize_prompt(
        {"prompt": _turns(1)}, TokenizeParams(max_total_tokens=None)
    )
    (cache,) = set(cache_encodes)
    assert cache._max_tokens == 64_000_000


def test_vllm_gets_a_copy_of_the_cached_token_ids(patched_renderer, cache_encodes):
    """vLLM's _build_tokens_prompt copies the ids into the prompt it returns,
    so changing that list in place (RequestState.apply_streaming_update
    extends a streaming-input request's prompt_token_ids) leaves the cache's
    entries, and later hits, unchanged."""
    renderer, unpatched = patched_renderer
    params = TokenizeParams(max_total_tokens=None)
    first = renderer._tokenize_prompt({"prompt": _turns(1)}, params)
    (cache,) = set(cache_encodes)
    stored = [e.ids for e in cache._entries.values()]
    assert all(first["prompt_token_ids"] is not ids for ids in stored)
    first["prompt_token_ids"].extend([7, 7, 7])
    for n in (1, 2):
        prompt = {"prompt": _turns(n)}
        expected = unpatched(renderer, dict(prompt), params)
        assert renderer._tokenize_prompt(dict(prompt), params) == expected
    assert cache.hits >= 2


def test_patched_renderer_leaves_truncation_to_vllm(patched_renderer):
    renderer, unpatched = patched_renderer
    params = TokenizeParams(max_total_tokens=None, truncate_prompt_tokens=10)
    prompt = {"prompt": _turns(1)}
    expected = unpatched(renderer, dict(prompt), params)
    assert len(expected["prompt_token_ids"]) == 10
    assert renderer._tokenize_prompt(dict(prompt), params) == expected


if __name__ == "__main__":
    unittest.main()
