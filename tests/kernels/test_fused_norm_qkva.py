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
"""TPU_FUSED_NORM_QKVA: the fused MLA decode prologue vs today's three ops.

Today: XLA add + RMSNorm, ``xla_quantized_matmul`` on the fp8 ``[6144, 2624]``
weight, the split and the q_a / kv_a ``rms_norm``s. y and the new residual
must be bit for bit; q_c, kv_c and k_pe within 2 bf16 ulps (the kernel's bf16 x
fp8 dot accumulates in f32 per 1024-row slab, XLA's fp8 dot in its own order);
the q_c fp8 output bit for bit ``quantize_tensor`` of the kernel's own q_c.
A planted fault (kv_a eps 1e-2 in the reference) must be caught.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import fused_norm_qkva as K
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor

H, Q, KV, ROPE = 6144, 2048, 512, 64
EPS, Q_EPS, KV_EPS = 1e-5, 1e-6, 1e-6


def _inputs(rows, seed=0):
    rng = np.random.default_rng(seed + rows)
    x = (rng.standard_normal((rows, H), dtype=np.float32) * 0.5).astype(jnp.bfloat16)
    r = (rng.standard_normal((rows, H), dtype=np.float32) * 2.0).astype(jnp.bfloat16)
    x[-1] = 0  # a padded row: amax 0 -> scale 1
    r[-1] = 0
    ln_w = (1 + 0.1 * rng.standard_normal(H, dtype=np.float32)).astype(jnp.bfloat16)
    w_q = (rng.standard_normal((H, Q + KV + ROPE), dtype=np.float32) * 0.3).astype(
        jnp.float8_e4m3fn
    )
    w_s = rng.uniform(0.002, 0.02, Q + KV + ROPE).astype(np.float32)
    qa_w = (1 + 0.1 * rng.standard_normal(Q, dtype=np.float32)).astype(jnp.bfloat16)
    kva_w = (1 + 0.1 * rng.standard_normal(KV, dtype=np.float32)).astype(jnp.bfloat16)
    return tuple(jnp.asarray(a) for a in (x, r, ln_w, w_q, w_s, qa_w, kva_w))


def _bits(a):
    return np.asarray(a).tobytes()


def _max_ulps(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    mag = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
    return float(np.max(np.abs(a - b) / np.exp2(np.floor(np.log2(mag)) - 7)))


@pytest.mark.parametrize("rows", [4, 8, 16, 32, 48, 64])
def test_matches_today(rows):
    on_tpu = jax.devices()[0].platform == "tpu"
    args = _inputs(rows)
    emit_q8 = rows > 16
    got = K.fused_norm_qkva(
        *args,
        eps=EPS,
        q_eps=Q_EPS,
        kv_eps=KV_EPS,
        q_rank=Q,
        kv_rank=KV,
        emit_q8=emit_q8,
        hbm_pin=on_tpu,
        interpret=not on_tpu,
    )
    want = K.reference(*args, eps=EPS, q_eps=Q_EPS, kv_eps=KV_EPS, q_rank=Q, kv_rank=KV)
    assert _bits(got[0]) == _bits(want[0])  # y
    assert _bits(got[1]) == _bits(want[1])  # new residual
    for g, w in zip(got[2:5], want[2:5]):
        assert _max_ulps(g, w) <= 2.0
    if emit_q8:
        q8, qs = quantize_tensor(got[2], jnp.float8_e4m3fn)
        assert _bits(got[5]) == _bits(q8)
        assert _bits(got[6]) == _bits(qs)
    bad = K.reference(
        *args, eps=EPS, q_eps=Q_EPS, kv_eps=1e-2, q_rank=Q, kv_rank=KV
    )
    assert _bits(got[3]) != _bits(bad[3])
