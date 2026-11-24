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
"""TPU_MINIMAX_M3_DCP_LSE_PLANES: `msa_dcp._exchange_partials` with the LSE
halves in two lane planes returns the same bits as the interleaved pairs, on
a 2-device CPU platform (a subprocess: the device count is fixed at JAX
start-up), for ordinary, -inf, NaN and subnormal LSE bit patterns."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _worker(num_tokens: int, num_heads: int) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, PartitionSpec as P

    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp

    dcp, head_dim = 2, 128
    rng = np.random.default_rng(num_tokens * 31 + num_heads)
    parts = rng.standard_normal((dcp, dcp, num_heads, num_tokens, head_dim))
    parts = jnp.asarray(parts, jnp.bfloat16)
    lse = rng.standard_normal((dcp, dcp, num_tokens, num_heads)).astype(np.float32) * 30
    bits = lse.view(np.uint32)
    bits[0, 0, 0, 0] = 0xFF800000  # -inf
    bits[0, 1, 1 % num_tokens, 0] = 0x7FC01234  # NaN with payload
    bits[1, 0, 0, num_heads - 1] = 0x00000001  # smallest subnormal
    bits[1, 1, num_tokens - 1, 0] = 0x80007FFF  # negative subnormal
    lse = jnp.asarray(bits.view(np.float32))
    mesh = Mesh(np.array(jax.devices()[:dcp]), ("x",))

    def body(p, l):
        rank = jax.lax.axis_index("x")
        return msa_dcp._exchange_partials(p[0], l[0], rank, "x", dcp)

    def run():
        fn = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("x"), P("x")),
                out_specs=(P("x"), P("x")),
                check_vma=False,
            )
        )
        o, l = fn(parts, lse)
        return np.asarray(o.view(jnp.uint16)), np.asarray(l).view(np.uint32)

    envs.TPU_MINIMAX_M3_DCP_LSE_PLANES = False
    ref_o, ref_l = run()
    envs.TPU_MINIMAX_M3_DCP_LSE_PLANES = True
    new_o, new_l = run()
    assert np.array_equal(ref_o, new_o)
    assert np.array_equal(ref_l, new_l), np.argwhere(ref_l != new_l)[:5]
    # Positive control: rank 0 receives rank 1's block 0, so its LSEs are
    # the input's [1, 0] block bit for bit.
    np.testing.assert_array_equal(
        new_l.reshape(dcp, dcp, num_tokens, num_heads)[0, 1], bits[1, 0]
    )
    print("OK")


@pytest.mark.parametrize("num_tokens,num_heads", [(8, 8), (32, 8), (96, 8), (40, 4)])
def test_lse_planes_bitwise(num_tokens, num_heads):
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
        PYTHONPATH=f"{ROOT}:{ROOT}/src",
    )
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(num_tokens), str(num_heads)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert res.returncode == 0 and "OK" in res.stdout, res.stderr[-4000:]


if __name__ == "__main__":
    _worker(int(sys.argv[1]), int(sys.argv[2]))
