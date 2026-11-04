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
"""CPU contracts for static kernel body identity and configuration isolation."""

import functools
import importlib
import unittest

from jax._src import xla_bridge

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc
from vllm_torchtpu.kernels.qwen4_exp import resident_topk as resident

score = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)


class BodyFactoryReuseTest(unittest.TestCase):
    def setUp(self):
        self.initial_backends_initialized = xla_bridge.backends_are_initialized()

    def check_key(self, factory, args):
        factory.cache_clear()
        body = factory(*args)
        self.assertIs(body, factory(*args))
        for index, value in enumerate(args):
            changed = list(args)
            changed[index] = not value if isinstance(value, bool) else value + 1
            self.assertIsNot(body, factory(*changed), (factory.__name__, index))
        info = factory.cache_info()
        self.assertEqual(info.hits, 1)
        self.assertEqual(info.misses, len(args) + 1)

    def test_direct_key_and_body(self):
        self.check_key(
            sc._topk_body_for_shape, (32, 2048, 16, 1, 2048, 1, True, False, False)
        )
        body = sc._topk_body_for_shape(32, 2048, 16, 1, 2048, 1, True, False, False)
        self.assertIsInstance(body, functools.partial)
        self.assertIs(body.func, sc._topk_body)
        self.assertEqual(
            body.keywords,
            dict(
                b=32,
                n=2048,
                k=16,
                p=1,
                slice_len=2048,
                num_waves=1,
                write_empty=True,
                return_scores=False,
            ),
        )

    def test_resident_key(self):
        self.check_key(
            resident._resident_body_for_shape,
            (32, 66560, 16, 4, 16640, 1, True, False, 32768, 4, 1),
        )

    def test_score_key_and_body(self):
        # Field markers check static key isolation without executing a kernel.
        args = (256, 8, 8, 2, 11, 0, 29)
        self.check_key(score._score_body_for_config, args)
        body = score._score_body_for_config(*args)
        self.assertIsInstance(body, functools.partial)
        self.assertIs(body.func, score.kernel)
        self.assertEqual(
            body.keywords,
            dict(
                zip(
                    (
                        "bq_sz",
                        "bkv_p",
                        "seq_batch_size",
                        "buffer_count",
                        "kv_layout",
                        "chunk_idx",
                        "components",
                    ),
                    args,
                )
            ),
        )

    def tearDown(self):
        self.assertEqual(
            self.initial_backends_initialized, xla_bridge.backends_are_initialized()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
