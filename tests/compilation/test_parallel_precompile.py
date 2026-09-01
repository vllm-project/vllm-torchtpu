# Copyright 2026 The vLLM Authors.
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

import os
import unittest
from unittest.mock import MagicMock, patch

from vllm_torchtpu.compilation.parallel_precompile import parallel_precompile


class TestParallelPrecompile(unittest.TestCase):

    def test_parallel_precompile_disabled(self):
        with patch.dict(os.environ, {
                "TPU_PARALLEL_PRECOMPILE": "0",
                "WORLD_SIZE": "4",
                "LOCAL_RANK": "1"
        }):
            executed = []
            items = [10, 20, 30, 40]
            parallel_precompile(items, lambda x: executed.append(x))
            self.assertEqual(executed, [10, 20, 30, 40])

    def test_parallel_precompile_single_worker(self):
        with patch.dict(os.environ, {
                "TPU_PARALLEL_PRECOMPILE": "1",
                "WORLD_SIZE": "1",
                "LOCAL_RANK": "0"
        }):
            executed = []
            items = ["a", "b", "c"]
            parallel_precompile(items, lambda x: executed.append(x))
            self.assertEqual(executed, ["a", "b", "c"])

    def test_parallel_precompile_multi_worker_partition(self):
        # 4 items across 4 workers: worker 1 should do item 1 first (Phase 1), then 2, 3, 0 (Phase 2)
        with patch.dict(os.environ, {
                "TPU_PARALLEL_PRECOMPILE": "1",
                "WORLD_SIZE": "4",
                "LOCAL_RANK": "1"
        }):
            executed = []
            items = [0, 1, 2, 3]
            parallel_precompile(items, lambda x: executed.append(x))
            # Phase 1: idx 1 (value 1)
            # Phase 2 remaining: [0, 2, 3], rotated by -1 -> [2, 3, 0]
            self.assertEqual(executed, [1, 2, 3, 0])

    def test_parallel_precompile_with_polling(self):
        with patch.dict(os.environ, {
                "TPU_PARALLEL_PRECOMPILE": "1",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0"
        }):
            executed = []
            items = [100, 200]
            poll_fn = MagicMock(
                return_value="/tmp/nonexistent_artifact_test.bin")
            with patch("os.path.exists", return_value=True):
                parallel_precompile(items,
                                    lambda x: executed.append(x),
                                    poll_cache_fn=poll_fn)
            # Worker 0: item 0 (100) in Phase 1, item 1 (200) in Phase 2
            self.assertEqual(executed, [100, 200])
            poll_fn.assert_called_with(200)


if __name__ == "__main__":
    unittest.main()
