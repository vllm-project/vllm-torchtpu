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
import sys
import unittest
from pathlib import Path

try:
    from vllm_torchtpu import frontend_tokcache as tokcache_mod
except Exception:
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "src" / "vllm_torchtpu")
    )
    import frontend_tokcache as tokcache_mod


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


if __name__ == "__main__":
    unittest.main()
