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
import unittest

import pytest

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


def _toy_hf_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

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
    """2048 entries and 64M tokens: what every IX prefill leg ran."""
    cache = tokcache_mod._build_cache_for(_toy_hf_tokenizer())
    assert cache is not None
    assert (cache._max_entries, cache._max_tokens) == (2048, 64_000_000)
    assert (cache._min_tokens, cache._verify_first, cache._verify_every) == (
        2048,
        8,
        0,
    )


@pytest.fixture
def patched_renderer(monkeypatch):
    """A real HfRenderer around the toy tokenizer, with the tokenization cache
    installed on BaseRenderer; returns it and vLLM's unpatched method."""
    from vllm.renderers import base as rbase
    from vllm.renderers.hf import HfRenderer

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
    from vllm.renderers.params import TokenizeParams

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


def test_patched_renderer_leaves_truncation_to_vllm(patched_renderer, cache_encodes):
    from vllm.renderers.params import TokenizeParams

    renderer, unpatched = patched_renderer
    params = TokenizeParams(max_total_tokens=None, truncate_prompt_tokens=10)
    prompt = {"prompt": _turns(1)}
    expected = unpatched(renderer, dict(prompt), params)
    assert len(expected["prompt_token_ids"]) == 10
    assert renderer._tokenize_prompt(dict(prompt), params) == expected


if __name__ == "__main__":
    unittest.main()
