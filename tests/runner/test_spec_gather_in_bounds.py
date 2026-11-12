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
"""TPU_SPEC_GATHER_IN_BOUNDS: the JAX op's ``logits[idx]`` gathers the same
rows as torch index_select (bit for bit, incl. -inf / NaN), on CPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("rows,reqs", [(8, 1), (24, 3), (48, 6), (128, 16)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_embedding_matches_index_select(rows, reqs, dtype):
    g = torch.Generator().manual_seed(rows)
    logits = torch.randn(rows, 1000, generator=g).to(dtype)
    logits[0, 3] = float("-inf")
    logits[1, 7] = float("nan")
    k = rows // reqs - 1
    target = torch.zeros(rows, dtype=torch.int32)
    n = 0
    for r in range(reqs):
        for j in range(k):
            target[n] = r * (k + 1) + j
            n += 1
    bonus = torch.zeros(max(reqs, 8), dtype=torch.int32)
    bonus[:reqs] = torch.arange(reqs, dtype=torch.int32) * (k + 1) + k
    for idx in (target, bonus):
        a = torch.index_select(logits, 0, idx)
        x = jnp.asarray(logits.float().numpy()).astype(
            jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
        )
        with jax.default_device(jax.devices("cpu")[0]):
            g = np.asarray(x[jnp.asarray(idx.numpy())].astype(jnp.float32))
        b = torch.from_numpy(g).to(dtype)
        assert a.dtype == b.dtype and a.shape == b.shape
        # Value-exact (NaN payloads may change in this CPU float32 detour).
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
