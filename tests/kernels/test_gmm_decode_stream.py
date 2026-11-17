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
"""CPU (Pallas interpret) tests for the streamed decode MoE MLP kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import gmm_decode_stream as S
from vllm_torchtpu.kernels.megablox.gmm_v2 import gate_up_act


def test_parse_spec():
    assert S.parse_spec("") is None
    assert S.parse_spec("0") is None
    assert S.parse_spec("on") == (768, 4, 512)
    assert S.parse_spec("1536:3") == (1536, 3, 512)
    assert S.parse_spec("768:4@256") == (768, 4, 256)
    with pytest.raises(ValueError):
        S.parse_spec("768:4@4096")
    with pytest.raises(ValueError):
        S.parse_spec("700:4")
    with pytest.raises(ValueError):
        S.parse_spec("768:1")


def test_plan():
    assert S.plan(32, 8) == ((16, 32), None)
    assert S.plan(192, 48) == ((16, 32, 64), None)
    assert S.plan(512, 128) == ((16, 32, 64, 128), None)
    assert S.plan(1024, 256) == ((16, 32, 48, 128), 128)


def test_eligible_rejects():
    fp4 = jnp.float4_e2m1fn
    lhs = jax.ShapeDtypeStruct((32, 512), jnp.bfloat16)
    w1 = jax.ShapeDtypeStruct((4, 512, 1024), fp4)
    w2 = jax.ShapeDtypeStruct((4, 512, 1024), fp4)
    s1 = jax.ShapeDtypeStruct((4, 4, 1, 1024), jnp.float32)
    s2 = jax.ShapeDtypeStruct((4, 4, 1, 1024), jnp.float32)
    assert S.eligible(lhs, w1, w2, s1, s2, None, 256, 512) == "no fused activation"
    assert S.eligible(lhs, w1, w2, None, s2, "silu", 256, 512) == "no scales"
    assert S.eligible(lhs, w1, w2, s1, s2, "silu", 256, 16) == "size_m"
    assert S.eligible(lhs, w1, w2, s1, s2, "silu", 384, 512) == "chunk rows"
    bad = jax.ShapeDtypeStruct((4, 512, 1024), jnp.bfloat16)
    assert S.eligible(lhs, bad, w2, s1, s2, "silu", 256, 512) == "rhs not fp4"


def _reference(lhs, w1, s1, w2, s2, gs, act):
    lhs = np.asarray(lhs.astype(jnp.float32))
    w1f = np.asarray(w1.astype(jnp.float32)) * np.repeat(np.asarray(s1)[:, :, 0], 128, 1)
    w2f = np.asarray(w2.astype(jnp.float32)) * np.repeat(np.asarray(s2)[:, :, 0], 128, 1)
    f = w2.shape[1]
    out = np.zeros((lhs.shape[0], w2.shape[2]), np.float32)
    off = 0
    for g, n in enumerate(np.asarray(gs)):
        if n:
            a = lhs[off : off + n] @ w1f[g]
            h = gate_up_act(jnp.asarray(a[:, :f]), jnp.asarray(a[:, f:]), act)
            h = np.asarray(h.astype(jnp.bfloat16).astype(jnp.float32))
            out[off : off + n] = h @ w2f[g]
        off += n
    return out, off


@pytest.mark.parametrize(
    "group_sizes,chunk,slots,m,max_group",
    [
        ([3, 0, 5, 9, 0, 1], 256, 3, 64, 32),  # unaligned group starts, 3 slots
        ([0, 0, 20, 0, 4, 0], 128, 2, 64, 32),  # leading empty groups, 2 slots
        ([0, 0, 0, 0, 0, 0], 256, 3, 64, 32),  # no work
        ([17, 0, 0, 0, 0, 30], 256, 4, 64, 32),  # groups past one 16-row slab
        ([1, 1, 1, 1, 1, 1], 256, 3, 64, 32),  # many entries in one slab
        ([5, 140, 0, 2, 60, 40], 256, 3, 640, 160),  # > 512 rows: split, bf16 tiles
        ([30, 0, 100, 0, 0, 3], 128, 2, 512, 128),  # <= 512 rows, group split at 128
    ],
)
def test_stream_matches_dequantized_reference(group_sizes, chunk, slots, m, max_group):
    k, f, g = 512, 512, 6
    keys = jax.random.split(jax.random.key(0), 5)
    w1 = (jax.random.normal(keys[0], (g, k, 2 * f)) * 2).astype(jnp.float4_e2m1fn)
    w2 = (jax.random.normal(keys[1], (g, f, 2 * f)) * 2).astype(jnp.float4_e2m1fn)
    s1 = jax.random.uniform(keys[2], (g, k // 128, 1, 2 * f), jnp.float32, 0.01, 0.03)
    s2 = jax.random.uniform(keys[3], (g, f // 128, 1, 2 * f), jnp.float32, 0.01, 0.03)
    lhs = jax.random.uniform(keys[4], (m, k), jnp.float32, -1, 1).astype(jnp.bfloat16)
    gs = jnp.asarray(group_sizes, jnp.int32)
    out = S.gmm_decode_stream_mlp(
        lhs, w1, w2, gs, s1, s2, jnp.zeros((1,), jnp.int32),
        fuse_act="swigluoai", out_dtype=jnp.bfloat16, chunk_rows=chunk,
        slots=slots, max_group_size=max_group, interpret=True,
    )
    ref, valid = _reference(lhs, w1, s1, w2, s2, gs, "swigluoai")
    out = np.asarray(out.astype(jnp.float32))
    if valid:
        scale = np.abs(ref[:valid]).max()
        assert np.abs(out[:valid] - ref[:valid]).max() <= 1e-2 * scale
    # Rows of the last 16-row slab past the groups are zero, as gmm_v2 masks them.
    assert np.all(out[valid : -(-valid // 16) * 16] == 0)


def test_set_decode_stream_key(monkeypatch):
    """Key gmm_v2_decode_stream: parsed once; refused with moe_skinny_gmm
    bf16_after (the replaced calls would no longer run the fp8 feed)."""
    from vllm_torchtpu.kernels.megablox import gmm_v2 as G

    try:
        G.set_decode_stream("1536:3@512")
        assert G._decode_stream == (1536, 3, 512) and G.decode_stream_on()
        G.set_decode_stream("")
        assert not G.decode_stream_on()
        monkeypatch.setattr(G, "_skinny_gmm_path", "bf16_after")
        with pytest.raises(ValueError, match="gmm_v2_decode_stream"):
            G.set_decode_stream("on")
        G.set_decode_stream("")  # off is always accepted
    finally:
        G.set_decode_stream("")


def test_gmm_v2_mlp_dispatch():
    """Key off: gmm_v2_mlp lowers without the streamed kernel; on, an
    eligible decode call lowers to it (tpu7x)."""
    from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

    from vllm_torchtpu.kernels.megablox import gmm_v2 as G

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    sd = jax.ShapeDtypeStruct
    fp4 = jnp.float4_e2m1fn
    k, f, g, m = 1024, 1024, 4, 64
    args = (
        sd((m, k), jnp.bfloat16),
        sd((g, k, 2 * f), fp4),
        sd((g, f, 2 * f), fp4),
        sd((g,), jnp.int32),
        sd((g, k // 128, 1, 2 * f), jnp.float32),
        sd((g, f // 128, 1, 2 * f), jnp.float32),
    )

    def text():
        jax.clear_caches()

        def fn(lhs, w1, w2, gs, s1, s2):
            return G.gmm_v2_mlp(
                lhs, w1, w2, gs, rhs1_scale=s1, rhs2_scale=s2,
                fuse_act="swigluoai", preferred_element_type=jnp.bfloat16,
                maybe_quantize_lhs=False, max_group_size=m,
            )

        with use_abstract_mesh(tpu7x):
            return jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",)).as_text()

    try:
        assert "gmm_decode_stream" not in text()
        G.set_decode_stream("512:3@512")
        assert "gmm_decode_stream" in text()
    finally:
        G.set_decode_stream("")
        jax.clear_caches()
