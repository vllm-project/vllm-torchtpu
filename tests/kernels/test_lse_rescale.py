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
"""Pallas `lse_rescale_head_major` vs the XLA expression it replaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.lse_rescale import lse_rescale_head_major


def _reference(output, lse, merged_lse, out_dtype):
    weight = jnp.exp(lse - merged_lse)[..., None]
    weighted = jnp.where(weight > 0, output.astype(jnp.float32) * weight, 0.0)
    return jnp.swapaxes(weighted.astype(out_dtype), 0, 1)


def _partial(tokens, heads, dim, ranks=4):
    """Rank 0's partial, lse and the merged lse. Rank 0 owns none of every
    third token: its lse is -inf there and its partial NaN."""
    rng = np.random.default_rng(tokens * heads + dim)
    output = rng.standard_normal((tokens, heads, dim)).astype(np.float32)
    lses = (rng.standard_normal((ranks, tokens, heads)) * 4).astype(np.float32)
    empty = np.arange(tokens) % 3 == 0
    lses[0, empty] = -np.inf
    output[empty] = np.nan
    merged = jax.nn.logsumexp(jnp.asarray(lses), axis=0)
    return jnp.asarray(output, jnp.bfloat16), jnp.asarray(lses[0]), merged


@pytest.mark.tpu_test
@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("heads,dim", [(64, 512), (16, 128), (8, 128), (4, 64)])
@pytest.mark.parametrize("tokens", [1, 4, 16, 200, 4096])
def test_matches_reference(tokens, heads, dim, out_dtype):
    output, lse, merged = _partial(tokens, heads, dim)
    got = np.asarray(lse_rescale_head_major(output, lse, merged, out_dtype), np.float32)
    want = np.asarray(_reference(output, lse, merged, out_dtype), np.float32)
    assert got.shape == (heads, tokens, dim)
    assert not np.isnan(got).any()
    # Mosaic's exp and XLA's differ by an ulp; bf16 can round either way.
    rtol = 1e-2 if out_dtype == "bfloat16" else 1e-5
    np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-6)
