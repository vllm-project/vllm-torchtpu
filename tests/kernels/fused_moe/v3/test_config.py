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
"""Config: its defaults, its constraints and the variables that set it.
No device needed."""

import jax.numpy as jnp
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.fused_moe.v3 import Config
from vllm_torchtpu.kernels.fused_moe.v3.config import (
    SORTED_ROWS_EXPERTS_PER_SHARD,
    SORTED_ROWS_TOKENS_PER_SHARD,
    sorted_rows_by_shape,
)

pytestmark = pytest.mark.cpu_test


class ConfigTest(parameterized.TestCase):
    @parameterized.parameters(
        dict(result_slots=1),
        dict(result_slots=3, row_fetch_ahead=3),
        dict(row_fetch_ahead=0),
        dict(weight_slots=1),
        dict(weight_slots=3, weight_prefetch=3),
        dict(row_scale_dtype=jnp.float16),
        dict(activation_block=100),
        dict(token_block=100),
        dict(token_block=-128),
        dict(tables_blocks_per_step=-1),
        dict(stream_block=100),
        dict(stream_rows=0),
        dict(stream_rows=12),
        dict(vmem_fraction=0.0),
        dict(vmem_fraction=1.5),
        dict(result_rows="fp16"),
        dict(elementwise_dtype="float16"),
        dict(combine_dtype="int8"),
        dict(gather_block_rows=12),
        dict(gather_ring=0),
        dict(scatter_chunk_pairs=100),
        dict(scatter_ring=0),
        dict(combine_tokens_per_tile=100),
        dict(combine_unroll=48),
        dict(activation_columns_per_pass=100),
        dict(tables_zero_words=100),
        dict(weight_dma_priority=2),
        dict(sorted_rows="yes"),
        dict(sorted_rows=1),
    )
    def test_refused(self, **fields):
        with self.assertRaises(ValueError):
            Config(**fields)

    def test_sorted_rows_by_shape(self):
        # Sorted from 32 experts per shard up at any token count, and at 256
        # tokens per shard or fewer for any expert count; per-row otherwise.
        self.assertEqual(
            (SORTED_ROWS_EXPERTS_PER_SHARD, SORTED_ROWS_TOKENS_PER_SHARD), (32, 256)
        )
        for experts, tokens, want in (
            (64, 1024, True),
            (32, 1024, True),
            (48, 512, True),
            (16, 256, True),
            (4, 8, True),
            (16, 512, False),
            (16, 1024, False),
            (8, 512, False),
            (31, 257, False),
        ):
            self.assertEqual(
                sorted_rows_by_shape(experts, tokens), want, (experts, tokens)
            )

    def test_for_shape(self):
        # Without an entry the record is the defaults; with one, the entry's
        # fields sit on top of them and nothing else moves.
        shape = dict(hidden=4096, inter=1024, experts_per_shard=64, top_k=10)
        self.assertEqual(Config.for_shape(**shape), Config())
        from vllm_torchtpu.kernels.fused_moe.v3 import config as config_module

        saved = dict(config_module.TUNED_BY_SHAPE)
        try:
            config_module.TUNED_BY_SHAPE[(4096, 1024, 64, 10)] = {
                "gather_ring": 4,
                "combine_unroll": 16,
            }
            self.assertEqual(
                Config.for_shape(**shape), Config(gather_ring=4, combine_unroll=16)
            )
            self.assertEqual(Config.for_shape(**dict(shape, top_k=8)), Config())
        finally:
            config_module.TUNED_BY_SHAPE.clear()
            config_module.TUNED_BY_SHAPE.update(saved)

    def test_hashable(self):
        self.assertEqual(hash(Config()), hash(Config()))
        self.assertNotEqual(Config(), Config(result_slots=4))


if __name__ == "__main__":
    absltest.main()
