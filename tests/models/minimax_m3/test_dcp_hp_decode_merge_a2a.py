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
"""TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A: (host devices, interpret) the
decode rows' DCP partials of a head-parallel step merged by all-to-all + the
f32 merge kernel match an f64 reference of the LSE merge (closer than the bf16
reduce-scatter), at DCP 4 with two groups and dead (rank, row) partials; the
flag is read from `envs` at trace time."""

import subprocess
import sys
import textwrap

import pytest

_E2E = textwrap.dedent(
    """
    import os, sys
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np, jax, jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads as H, msa_fp4
    dcp, Hl, D, rows = 4, 8, 128, 64
    nh = dcp * Hl
    rng = np.random.default_rng(0)
    out = rng.standard_normal((8, nh, rows, D)).astype(np.float32)
    lse = (rng.standard_normal((8, nh, rows)) * 3 + 5).astype(np.float32)
    dead = rng.uniform(size=(8, 1, rows)) < 0.25
    dead[0] = False  # every row keeps a live rank
    dead[4] = False
    out = np.where(dead[..., None], 0.0, out)
    lse = np.where(dead, -np.inf, lse).astype(np.float32)
    out_bf = jnp.asarray(out, jnp.bfloat16)
    mesh = Mesh(np.array(jax.devices()).reshape(2, dcp), ("g", "d"))
    spec = P(("g", "d"))
    sh = NamedSharding(mesh, spec)
    o_d, l_d = jax.device_put(out_bf, sh), jax.device_put(jnp.asarray(lse), sh)

    def run(a2a):
        f = jax.shard_map(
            lambda o, l: H.merge_partials(o[0], l[0], "d", rs_out_precision=True,
                                          a2a=a2a, interpret=True)[None],
            mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False)
        return np.asarray(jax.jit(f)(o_d, l_d).astype(jnp.float32), np.float64).reshape(8, rows, Hl, D)

    ob = np.asarray(out_bf.astype(jnp.float32), np.float64)
    hd = np.asarray(msa_fp4.hadamard(D), np.float64)
    ref = np.zeros((8, rows, Hl, D))
    for g in range(2):
        for r in range(dcp):
            hs = slice(r * Hl, (r + 1) * Hl)
            ls = lse[g * dcp:(g + 1) * dcp, hs].astype(np.float64)
            mx = ls.max(0)
            w = np.exp(ls - mx)
            w = w / w.sum(0)
            m = (w[..., None] * ob[g * dcp:(g + 1) * dcp, hs]).sum(0) @ hd / D
            ref[g * dcp + r] = m.transpose(1, 0, 2)
    rel = lambda x: float(np.sqrt(np.mean((x - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))
    got_a2a, got_rs = run(True), run(False)
    assert np.all(np.isfinite(got_a2a))
    ea, er = rel(got_a2a), rel(got_rs)
    # f32 merge rounded once to bf16: ~2^-9 relative; the RS path rounds twice.
    assert ea < 3e-3, ea
    assert ea <= er * 1.05, (ea, er)
    # the a2a output is exactly the rounded f32 merge of the bf16 partials
    assert np.max(np.abs(got_a2a - ref)) <= 2 ** -7 * np.max(np.abs(ref)), np.max(np.abs(got_a2a - ref))

    envs.TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A = False
    assert not H.hp_decode_merge_a2a()
    envs.TPU_MINIMAX_M3_DCP_HP_DECODE_MERGE_A2A = True
    assert H.hp_decode_merge_a2a()
    print("OK", ea, er)
    """
)


@pytest.mark.timeout(900)
def test_hp_decode_merge_a2a_matches_reference():
    res = subprocess.run(
        [sys.executable, "-c", _E2E], capture_output=True, text=True, timeout=880
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "OK" in res.stdout
