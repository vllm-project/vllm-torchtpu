# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The prefill's attention over the step's gathered keys and values."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.qwen38.attention import (
    local_attention,
    reference_attention,
)

CORES = 8


@pytest.mark.parametrize("interleave", [128, 256])
def test_keys_gathered_from_rows_dealt_in_interleave_runs(interleave):
    # One request of 4,096 tokens dealt to the cores in runs of the KV
    # interleave (core c's local row l is flat row
    # ((l // run) * cores + c) * run + l % run); core 0's queries.
    hq, hkv, dim, tokens = 2, 1, 128, 4096
    rng = np.random.default_rng(0)
    q = rng.standard_normal((tokens, hq, dim)).astype(np.float32)
    k = rng.standard_normal((tokens, hkv * dim)).astype(np.float32)
    v = rng.standard_normal((tokens, hkv * dim)).astype(np.float32)
    local = np.arange(tokens // CORES)
    rows = [
        ((local // interleave) * CORES + c) * interleave + local % interleave
        for c in range(CORES)
    ]
    gathered = np.concatenate(rows)  # core-major
    reference = reference_attention(
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(k, jnp.float8_e4m3fn),
        jnp.asarray(v, jnp.float8_e4m3fn),
        jnp.arange(tokens),
        jnp.zeros(tokens, jnp.int32),
        sm_scale=dim**-0.5,
        k_scale=1.0,
        v_scale=1.0,
    )
    out = local_attention(
        jnp.asarray(q[rows[0]], jnp.bfloat16),
        jnp.asarray(k[gathered], jnp.float8_e4m3fn),
        jnp.asarray(v[gathered], jnp.float8_e4m3fn),
        jnp.asarray(rows[0], jnp.int32),
        jnp.zeros(len(rows[0]), jnp.int32),
        sm_scale=dim**-0.5,
        k_scale=1.0,
        v_scale=1.0,
        cores=CORES,
        interleave=interleave,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(out, np.float32),
        np.asarray(reference, np.float32)[rows[0]],
        atol=0.05,
    )
