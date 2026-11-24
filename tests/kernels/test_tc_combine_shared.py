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
"""tc_combine_shared (shared-expert down_proj in the combine's output pass)
against tc_combine + the unfused shared down_proj and add (interpret mode)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import tc_combine as K

TOPK = 4
E = 128


def _inputs(t, local, h, ks, seed, empty_block=False):
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((t, E)), axis=1)[:, :TOPK]
    if empty_block:
        # The first token block routes nothing locally.
        ids[: K.TOKEN_BLOCK] = np.arange(local, local + TOPK)
    valid = ids < local
    keys = np.where(valid.reshape(-1), ids.reshape(-1), local)
    order = np.argsort(keys, kind="stable")
    pos = np.empty_like(order)
    pos[order] = np.arange(order.size)
    group_sizes = np.bincount(ids[valid], minlength=local)[:local].astype(np.int32)
    count = int(valid.sum())
    y = (rng.standard_normal((t * TOPK, h)) * 3).astype(np.float32)
    y[count:] = np.nan  # rows past the computed prefix are never written
    w = rng.uniform(0.05, 1, (t, TOPK)).astype(np.float32)
    act = (rng.standard_normal((t, ks)) * 2).astype(np.float32)
    ws = rng.standard_normal((ks, h)).astype(np.float32) * 0.05
    scale = 2.0 ** rng.integers(-3, 3, size=(h,)).astype(np.float32)
    return (
        jnp.asarray(y).astype(jnp.bfloat16),
        jnp.asarray(pos.astype(np.int32)),
        jnp.asarray(w.reshape(-1)).astype(jnp.bfloat16),
        jnp.asarray(valid.reshape(-1)),
        jnp.asarray(group_sizes),
        jnp.asarray(act).astype(jnp.bfloat16),
        jnp.asarray(ws).astype(jnp.float8_e4m3fn),
        jnp.asarray(scale).astype(jnp.bfloat16),
    )


def _unfused(y, pos, w, valid, gs, act, ws, scale, routed_scale):
    """The served graph: routed combine * routed_scale, shared down_proj
    (bf16 x bf16(W), bf16 out) * per-channel scale, bf16 add."""
    routed = K.tc_combine(y, pos, w, valid, gs, topk=TOPK, interpret=True)
    routed = routed * jnp.asarray(routed_scale, jnp.bfloat16)
    shared = jnp.dot(act, ws.astype(jnp.bfloat16), preferred_element_type=jnp.bfloat16)
    shared = shared * scale
    return routed + shared


@pytest.mark.parametrize(
    "t,local,routed_scale,empty_block",
    [
        (512, 16, 2.0, False),
        (768, 16, 1.0, False),
        (512, 16, 2.0, True),
        (1024, 8, 2.0, False),
    ],
)
def test_shared_combine_matches_unfused(t, local, routed_scale, empty_block):
    h, ks = 1024, 384
    args = _inputs(t, local, h, ks, seed=t + local, empty_block=empty_block)
    got = K.tc_combine_shared(
        *args, topk=TOPK, routed_scale=routed_scale, interpret=True
    )
    want = _unfused(*args, routed_scale)
    g = np.asarray(got.astype(jnp.float32))
    r = np.asarray(want.astype(jnp.float32))
    assert np.isfinite(g).all()
    # Same roundings; only the shared dot's f32 accumulation order may differ.
    np.testing.assert_allclose(g, r, rtol=1e-2, atol=1e-2)
    bitwise = np.mean(
        np.asarray(got).view(np.uint16) == np.asarray(want).view(np.uint16)
    )
    assert bitwise > 0.99, bitwise


def test_shared_combine_fallback_matches_kernel():
    """The plain-gather fallback (rows out of token order) adds the shared
    expert the same way."""
    t, local, h, ks = 512, 16, 1024, 384
    y, pos, w, valid, gs, act, ws, scale = _inputs(t, local, h, ks, seed=7)
    out = K._shared_epilogue(
        K._gather_combine(y, pos, w, valid, TOPK),
        (act, ws.astype(jnp.bfloat16), scale.reshape(1, -1), 2.0),
    )
    got = K.tc_combine_shared(
        y, pos, w, valid, gs, act, ws, scale, topk=TOPK, routed_scale=2.0,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(got.astype(jnp.float32)),
        np.asarray(out.astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )


def test_supported_shared_vmem():
    assert K.supported_shared(8192, 6144, 4, 16, jnp.bfloat16, 384)
    assert not K.supported_shared(8192, 6144, 4, 16, jnp.bfloat16, 3072)
