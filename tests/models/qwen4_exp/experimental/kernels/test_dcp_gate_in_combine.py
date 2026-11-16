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
"""VLLM_TPU_QSA_DCP_GATE_IN_COMBINE: the pair combine slicing the gate half of
the whole q/gate row gives the same bits as a pre-sliced gate."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen4_exp.experimental.kernels.qsa_rpa import (
    dcp_combine_step,
)


@pytest.mark.parametrize("rank", [0, 1])
def test_whole_row_gate_matches_sliced(rank):
    rng = np.random.default_rng(rank)
    tokens, heads, dim = 64, 3, 256
    out_pair = jnp.asarray(rng.normal(0, 1, (2, tokens, 2 * heads, dim)), jnp.float32)
    lse_pair = jnp.asarray(rng.normal(0, 1, (2, tokens, 2 * heads)), jnp.float32)
    lse_pair = lse_pair.at[1, :5].set(-jnp.inf)
    qgate = jnp.asarray(rng.normal(0, 1, (tokens, heads, 2 * dim)), jnp.bfloat16)
    r = jnp.asarray([rank], jnp.int32)
    want = jax.jit(dcp_combine_step)(out_pair, lse_pair, qgate[..., dim:], r)
    got = jax.jit(dcp_combine_step)(out_pair, lse_pair, qgate, r)
    assert got.shape == want.shape
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint16)),
        np.asarray(jax.lax.bitcast_convert_type(want, jnp.uint16)),
    )
