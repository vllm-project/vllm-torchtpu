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


if __name__ == "__main__":
    unittest.main()
