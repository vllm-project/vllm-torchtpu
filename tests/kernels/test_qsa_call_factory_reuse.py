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
"""CPU contracts for static Pallas call identity and resource key isolation."""

import importlib
import unittest

import numpy as np
from jax._src import xla_bridge

score = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)


class CallFactoryReuseTest(unittest.TestCase):
    def test_constructor_identity_and_resource_keys(self):
        initial_backend_state = xla_bridge.backends_are_initialized()
        factory = score._score_call_for_config
        factory.cache_clear()
        body = (1, 8, 1, 2, score.KVLayout.HEAD_ALONG_SUBLANE, 0, None)
        args = (
            body,
            (512, 520, 128),
            np.dtype("int32"),
            5,
            5,
            True,
            100 * 1024 * 1024,
            "static-call-test",
        )
        call = factory(*args)
        self.assertIs(call, factory(*args))
        # Constructor-only checks; changed markers are never executed as kernels.
        changes = (
            (128, *body[1:]),
            (256, 520, 128),
            np.dtype("float32"),
            6,
            6,
            False,
            96 * 1024 * 1024,
            "other-call-test",
        )
        for field, value in enumerate(changes):
            changed = list(args)
            changed[field] = value
            self.assertIsNot(call, factory(*changed), field)
        self.assertEqual(factory.cache_info().misses, 9)
        self.assertEqual(initial_backend_state, xla_bridge.backends_are_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
