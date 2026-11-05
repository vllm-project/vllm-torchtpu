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
"""`absorb_q` vs the head-major batched matmul it replaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.absorb_q import (
    HEAD_TAIL_LANES,
    absorb_q,
    absorb_q_head,
    head_block_tokens,
)


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [4, 200, 256, 300, 4096])
def test_matches_head_major_bmm(tokens):
    """Decode sizes take XLA's einsum and prefill sizes the kernel (with a
    partial last block at 300); both equal the bmm the call site ran."""
    heads, nope, rank = 64, 192, 512
    k1, k2 = jax.random.split(jax.random.key(tokens))
    q_nope = jax.random.normal(k1, (tokens, heads, nope)).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (heads, nope, rank)) * 0.05).astype(jnp.bfloat16)
    bmm = jnp.matmul(jnp.swapaxes(q_nope, 0, 1), w, preferred_element_type=jnp.float32)
    want = jnp.swapaxes(bmm, 0, 1).astype(q_nope.dtype)
    np.testing.assert_array_equal(
        np.asarray(absorb_q(q_nope, w), np.float32), np.asarray(want, np.float32)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [256, 300, 4096])
def test_head_input_matches_slice(tokens):
    """`absorb_q_head` of the q projection's 2-D output equals `absorb_q` of
    its q_nope slice bit for bit, and returns each head's last 128 lanes."""
    heads, nope, rope, rank = 32, 192, 64, 512
    k1, k2 = jax.random.split(jax.random.key(tokens))
    q = jax.random.normal(k1, (tokens, heads, nope + rope)).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (heads, nope, rank)) * 0.05).astype(jnp.bfloat16)
    out, tail = absorb_q_head(q.reshape(tokens, -1), w)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32),
        np.asarray(absorb_q(q[..., :nope], w), np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(tail, np.float32), np.asarray(q[..., 128:], np.float32)
    )


@pytest.mark.parametrize(("heads", "block"), [(32, 256), (64, 128)])
def test_head_kernel_block_follows_the_head_count(monkeypatch, heads, block):
    """In interpret mode on the host: 256 tokens per grid step at 32 heads and
    128 at 64 (256 x 64 head rows overflow VMEM beside W_UK), with a partial
    last block at 300 tokens; the outputs equal the per-head matmul and the
    raw q tail lanes. f32, since the host's dot takes no bf16 x bf16 = f32."""
    grids = []
    pallas_call = pl.pallas_call

    def interpreted(*args, **kwargs):
        grids.append(kwargs["grid"])
        return pallas_call(*args, **kwargs, interpret=pltpu.InterpretParams())

    monkeypatch.setattr(pl, "pallas_call", interpreted)
    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    tokens, nope, rope, rank = 300, 192, 64, 128
    k1, k2 = jax.random.split(jax.random.key(heads))
    q = jax.random.normal(k1, (tokens, heads, nope + rope))
    w = jax.random.normal(k2, (heads, nope, rank)) * 0.05
    out, pe = absorb_q_head(q.reshape(tokens, -1), w)
    assert grids == [(-(-tokens // block),)]
    want = jnp.einsum(
        "thd,hdr->thr", q[:, :, :nope], w, precision=jax.lax.Precision.HIGHEST
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(want), rtol=1e-5, atol=1e-5)
    tail = q[:, :, -HEAD_TAIL_LANES:]
    np.testing.assert_array_equal(np.asarray(pe), np.asarray(tail))


@pytest.mark.parametrize(
    ("heads", "tokens", "want"),
    [(32, 4096, 256), (64, 4096, 128), (16, 4096, 512), (64, 100, 100)],
)
def test_head_block_is_the_largest_that_fits_vmem(heads, tokens, want):
    """bf16, nope 192 + rope 64, rank 512, in 56 MiB: 256 tokens at 64 heads
    need more (measured 75.8 MiB), 128 fit."""
    vmem = 56 * 1024 * 1024
    assert head_block_tokens(tokens, heads, 256, 192, 512, 2, vmem, 8) == want


def test_head_block_refuses_a_head_count_with_no_room():
    with pytest.raises(ValueError, match="under 8"):
        head_block_tokens(4096, 1024, 256, 192, 512, 2, 56 * 1024 * 1024, 8)
