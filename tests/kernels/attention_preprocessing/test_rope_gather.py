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
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.attention_preprocessing import rope_gather


@pytest.mark.parametrize("rows", [5, 240, 256, 257, 288, 520, 1024, 2048])
def test_blocked_rope_gather_matches_one_gather(rows):
    table = jax.random.normal(jax.random.PRNGKey(0), (4096, 64), jnp.float32)
    positions = jax.random.randint(jax.random.PRNGKey(1), (rows,), -8, 4104)
    ref = rope_gather.gather_rows(table, positions)
    got = rope_gather.gather_rows_blocked(table, positions)
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(got))
    inside = jnp.clip(positions, 0, table.shape[0] - 1)
    ref = table[inside]
    got = rope_gather.by_row_blocks(lambda r: table[r], inside, True)
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(got))
