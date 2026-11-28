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
"""Helpers shared by the Mamba state-copy tests: the row copy's program for one
geometry, and a NumPy copy that stands in for the Pallas copy kernel so the
copy programs run on the host.
"""

import contextlib
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops import mamba_state_copy_op


def row_copy_program(num_pools, split, step_rows):
    """The row copy's JAX program for one geometry, without registering an op."""
    captured = {}

    def fake_jax_op(_name, fn, **_kwargs):
        captured["fn"] = fn
        return fn

    with patch.object(mamba_state_copy_op.pallas, "jax_op", fake_jax_op):
        mamba_state_copy_op._row_copy_fn.__wrapped__(num_pools, split, step_rows)
    return captured["fn"]


def numpy_copy_blocks(pool, src, dst, in_pool=None):
    """`pool[dst[i]] = pool[src[i]]` in NumPy, for indices inside the pool.

    One pair at a time, in order, each reading the pool as the pairs before
    it left it. A later padding pair therefore cannot restore a row that a
    pair the guard let through overwrote, as copying all pairs from the
    original pool at once could. `in_pool` (the kernel's debug-check input)
    is ignored."""
    out = np.array(pool)
    for s, d in zip(np.asarray(src).tolist(), np.asarray(dst).tolist()):
        out[d] = out[s]
    return jnp.asarray(out)


@contextlib.contextmanager
def numpy_copy_kernel():
    """Runs `pool_adapters.copy_blocks` on the host: its bounds guard runs as
    written, and only the Pallas kernel behind it is `numpy_copy_blocks`."""
    with patch.object(pool_adapters, "_copy_blocks_unchecked", numpy_copy_blocks):
        yield
