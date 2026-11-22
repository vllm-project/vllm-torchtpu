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
"""`rows_join.join_pieces` against dequantizing each piece and concatenating."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.models.minimax_m3.kernels import rows_join


@pytest.mark.parametrize(
    "rows,hidden", [((512, 512), 256), ((384, 384, 384), 384), ((256, 128), 128)]
)
def test_matches_dequantize_and_concatenate(rows, hidden):
    rng = np.random.default_rng(len(rows) + hidden)
    pieces = tuple(
        (
            jnp.asarray(rng.normal(0, 100, (n, hidden)), jnp.float8_e4m3fn),
            jnp.asarray(rng.uniform(1e-3, 1, (n, 1)), jnp.float32),
        )
        for n in rows
    )
    tpu7x = AbstractDevice("TPU7x", 1, "tpu")  # the pipeline reads the chip's tiling
    with use_abstract_mesh(AbstractMesh((), (), (), abstract_device=tpu7x)):
        got = rows_join.join_pieces(pieces, interpret=True)
    want = jnp.concatenate(
        [(q.astype(jnp.float32) * s).astype(jnp.bfloat16) for q, s in pieces]
    )
    np.testing.assert_array_equal(
        np.asarray(got, np.float32), np.asarray(want, np.float32)
    )
