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
"""Both arms of the VMEM-resident MoE experiment match the oracle, and the
oracle rejects a corrupted expert. Interpret mode on CPU, small shapes."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.vmem_resident_moe import (
    ARMS,
    build,
    make_inputs,
    reference,
)

DIMS = dict(experts=3, hidden=256, inter=128, rows=32)
TOKENS, STEPS = 16, 3


@pytest.fixture(scope="module")
def case():
    if jax.default_backend() != "cpu":
        pytest.skip("CPU Pallas interpreter; run with JAX_PLATFORMS=cpu")
    sched, x0, gate, ws = make_inputs(
        jax.random.PRNGKey(0), tokens=TOKENS, steps=STEPS, **DIMS
    )
    return sched, x0, gate, ws, reference(sched, x0, gate, ws)


def _run(arm, sched, x0, gate, ws, hoist=True):
    f = build(
        arm,
        items=sched.items,
        vmem_limit_bytes=32 * 2**20,
        interpret=True,
        hoist=hoist,
        **DIMS,
    )
    return f(sched, x0, gate, ws)[:TOKENS]


@pytest.mark.parametrize("hoist", [True, False])
@pytest.mark.parametrize("arm", ARMS)
def test_matches_reference(case, arm, hoist):
    sched, x0, gate, ws, ref = case
    out = _run(arm, sched, x0, gate, ws, hoist=hoist)
    assert jnp.max(jnp.abs(out - ref[:TOKENS])) <= 1e-5


def test_oracle_rejects_corrupted_expert(case):
    sched, x0, gate, ws, ref = case
    e0 = int(sched.expert[0])
    bad = list(ws)
    bad[2] = ws[2].at[e0].set(ws[2][(e0 + 1) % DIMS["experts"]])
    assert (
        jnp.max(jnp.abs(_run("resident", sched, x0, gate, tuple(bad)) - ref[:TOKENS]))
        > 0.1
    )
