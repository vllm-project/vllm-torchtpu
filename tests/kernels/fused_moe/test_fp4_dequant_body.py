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
"""Host-side exactness of the fused EP kernel's bf16-dequant four-bit body.

`dequant_contract` and `expert_ffn_dequant` are plain jnp once the block
reader is (the kernel's reader is a Mosaic bitcast; here it slices an f32
array of e2m1 values). With e2m1 weights, power-of-two block scales and
small-integer activations every product and partial sum is exact, so the
chunked contraction must equal the dense formula bit for bit -- which pins
the scale fold, the block order and the chunk boundaries, including a
ragged last chunk. The device suite covers the Mosaic reader.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import host
from vllm_torchtpu.kernels.fused_moe.v2 import kernel as K

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def _weights(rng, k, n, qb):
    w = rng.choice(E2M1, size=(k, n)).astype(np.float32)
    s = (2.0 ** rng.integers(-3, 2, size=(k // qb, n))).astype(np.float32)
    return w, s


def _reader(w, qb):
    def read(b, n0, n1):
        return jnp.asarray(w[b * qb : (b + 1) * qb, n0:n1])

    return read


@pytest.mark.parametrize("k,n,qb,chunk", [(256, 512, 64, 128), (384, 640, 128, 256), (128, 96, 64, 512)])
def test_dequant_contract_is_the_dense_formula(k, n, qb, chunk):
    rng = np.random.default_rng(0)
    w, s = _weights(rng, k, n, qb)
    lhs = rng.integers(-4, 5, size=(16, k)).astype(np.float32)
    got = K.dequant_contract(
        jnp.asarray(lhs, jnp.bfloat16), _reader(w, qb), jnp.asarray(s), qb=qb,
        n_out=n, chunk=chunk, stage=None,
    )
    deq = w * np.repeat(s, qb, axis=0)
    want = lhs @ deq
    assert got.dtype == jnp.float32 and got.shape == (16, n)
    np.testing.assert_array_equal(np.asarray(got), want)


def test_expert_ffn_dequant_matches_reference_formula():
    rng = np.random.default_rng(1)
    hidden, inter, qb, rows = 256, 128, 64, 8
    w1, s1 = _weights(rng, hidden, 2 * inter, qb)
    w2, s2 = _weights(rng, inter, hidden, qb)
    act = jnp.asarray(rng.integers(-4, 5, size=(rows, hidden)), jnp.float8_e4m3fn)
    act_scale = jnp.asarray(2.0 ** rng.integers(-6, -2, size=(rows, 1)), jnp.float32)
    acc2, mid_scale = K.expert_ffn_dequant(
        act, act_scale, _reader(w1, qb), _reader(w2, qb), jnp.asarray(s1),
        jnp.asarray(s2), qb=qb, act_fn="silu", chunk=128, stage=None,
    )
    assert mid_scale is None
    a = np.asarray(act.astype(jnp.float32))
    acc1 = (a @ (w1 * np.repeat(s1, qb, axis=0))) * np.asarray(act_scale)
    gate, up = jnp.asarray(acc1[:, :inter]), jnp.asarray(acc1[:, inter:])
    mid = np.asarray((K._apply_act(gate, up, "silu")).astype(jnp.bfloat16).astype(jnp.float32))
    want = mid @ (w2 * np.repeat(s2, qb, axis=0)).astype(jnp.bfloat16).astype(np.float32)
    # The intermediate is no longer integer-valued, so the down matmul's f32
    # sum order (XLA vs numpy) shows at ~1e-5; the contraction itself is
    # pinned bit-exact by the test above.
    np.testing.assert_allclose(np.asarray(acc2), want, rtol=1e-4, atol=1e-3)


class _Info:
    vmem_capacity_bytes = 64 * 2**20
    generation = 7
    num_lanes = 128

    def get_sublane_tiling(self, dtype):
        return {1: 32, 2: 16, 4: 8}[jnp.dtype(dtype).itemsize]


def test_minimax_shape_fits_one_weight_slot_with_the_dequant_stage():
    kw = dict(weight_format=host.WeightFormat.FP4, rhs_qb=128, info=_Info())
    base = host.fit_weight_slots(16, 128, 6144, 3072, **kw)
    dq = host.fit_weight_slots(16, 128, 6144, 3072, fp4_body="dequant", **kw)
    assert base.fits and dq.fits and dq.nbuf == base.nbuf == 1
    stage = 6144 * host.FP4_DEQUANT_CHUNK * 2
    assert dq.vmem_bytes - base.vmem_bytes >= stage
    names = [n for n, _, _ in host.vmem_scratch_arrays(
        16, 128, 6144, 3072, nbuf=1, weight_format=host.WeightFormat.FP4,
        rhs_qb=128, fp4_body="dequant")]
    assert names[-1] == "dq_vm"
    # Other formats never grow the stage.
    names8 = [n for n, _, _ in host.vmem_scratch_arrays(
        16, 128, 6144, 3072, nbuf=1, weight_format=host.WeightFormat.FP8,
        fp4_body="dequant")]
    assert "dq_vm" not in names8
