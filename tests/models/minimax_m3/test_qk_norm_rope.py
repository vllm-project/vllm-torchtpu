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
"""`qk_norm_rope` against `attention.norm_rope_heads`' numerics in jnp."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4, qk_norm_rope

D, ROT, EPS = 128, 64, 1e-6


def _reference(qkv, cos_sin, weight, nq, nkv):
    """The TPU rms_norm provider (f32, rounded once) on the stacked q, k,
    index_q, index_k heads, then the FP32 NeoX rotary embedding."""
    t = qkv.shape[0]
    heads = qkv.reshape(t, -1, D)
    v = heads[:, nq + nkv : nq + 2 * nkv]
    x = jnp.concatenate([heads[:, : nq + nkv], heads[:, nq + 2 * nkv :]], axis=1)
    x = x.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    x = (x * jax.lax.rsqrt(var + EPS) * weight[None]).astype(qkv.dtype)
    cos, sin = jnp.split(cos_sin.astype(jnp.float32)[:, None, :], 2, axis=-1)
    rot = x[..., :ROT].astype(jnp.float32)
    x1, x2 = jnp.split(rot, 2, axis=-1)
    rot = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    x = jnp.concatenate([rot.astype(qkv.dtype), x[..., ROT:]], axis=-1)
    q, k, iq, ik = jnp.split(x, [nq, nq + nkv, nq + 2 * nkv], axis=1)
    return q, k, v, iq, ik


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("q_head_major", [False, True])
@pytest.mark.parametrize(
    "num_tokens,nq,nkv,block_tokens",
    [
        (8, 8, 1, 256),
        (24, 8, 1, 256),
        (512, 8, 1, 256),
        (300, 4, 2, 128),
        (1100, 8, 1, 1024),  # sub-blocks, a partial last block
    ],
)
def test_matches_reference(num_tokens, nq, nkv, block_tokens, q_head_major, rotate):
    rng = np.random.default_rng(num_tokens + nq)
    qkv = jnp.asarray(
        rng.normal(0, 2, (num_tokens, (nq + 3 * nkv + 1) * D)), jnp.bfloat16
    )
    ang = rng.uniform(0, 100, (num_tokens, ROT // 2))
    cos_sin = jnp.asarray(np.concatenate([np.cos(ang), np.sin(ang)], 1), jnp.bfloat16)
    weight = jnp.asarray(1 + rng.normal(0, 0.3, (nq + 2 * nkv + 1, D)), jnp.float32)
    got = qk_norm_rope.qk_norm_rope(
        qkv,
        cos_sin,
        weight,
        num_heads=nq,
        num_kv_heads=nkv,
        head_dim=D,
        eps=EPS,
        q_head_major=q_head_major,
        rotate=rotate,
        block_tokens=block_tokens,
        interpret=True,
    )
    want = _reference(qkv, cos_sin, weight, nq, nkv)
    if rotate:  # the 4-bit DCP op's rotation of q and index_q, as XLA did it
        want = tuple(
            msa_fp4.rotate_queries(x) if i in (0, 3) else x for i, x in enumerate(want)
        )
    want_q = (
        want[0].transpose(1, 0, 2) if q_head_major else want[0].reshape(num_tokens, -1)
    )
    for g, w in zip(got, (want_q, *(x.reshape(num_tokens, -1) for x in want[1:]))):
        assert g.shape == w.shape and g.dtype == w.dtype
        np.testing.assert_array_equal(
            np.asarray(g, np.float32), np.asarray(w, np.float32)
        )


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("q_head_major", [False, True])
@pytest.mark.parametrize("num_tokens", [8, 300, 8192])
def test_lowers_for_tpu(num_tokens, q_head_major, rotate):
    nq, nkv = 8, 1
    args = (
        jax.ShapeDtypeStruct((num_tokens, (nq + 3 * nkv + 1) * D), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_tokens, ROT), jnp.bfloat16),
        jax.ShapeDtypeStruct((nq + 2 * nkv + 1, D), jnp.float32),
    )
    fn = functools.partial(
        qk_norm_rope.qk_norm_rope,
        num_heads=nq,
        num_kv_heads=nkv,
        head_dim=D,
        eps=EPS,
        q_head_major=q_head_major,
        rotate=rotate,
    )
    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
