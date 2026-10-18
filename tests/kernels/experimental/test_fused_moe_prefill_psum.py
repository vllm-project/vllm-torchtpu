# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE prefill kernel's cross-rank sum (no TPU needed): each combine
chunk of the default build (_combine) and of the stage-A build (combine) is
reduced as a reduce-scatter plus an all-gather, never as one all-reduce, and
the result equals psum's, including chunks whose rows do not divide by the
axis size (padded and sliced back).

The Pallas combine body needs a TPU, so every chunk here takes the XLA
gather-sum branch (combine_step_tokens patched to False); the reduction that
follows is the same. Runs on up to 16 devices; CPU virtual devices are
sufficient, e.g. XLA_FLAGS=--xla_force_host_platform_device_count=16."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel as v1,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as v2,
)

pytestmark = pytest.mark.multichip

H = 256
SLOTS = 3
BUILDS = {
    "v1": (v1, lambda *a: v1._combine(*a)),
    "v2": (v2, lambda *a: v2.combine(*a)),
}


@pytest.fixture
def mesh():
    # A fixture, not a module-level skipif: collection must not claim a backend.
    n = min(jax.device_count(), 16)
    if n < 2:
        pytest.skip("requires two or more devices; CPU virtual devices are sufficient")
    return Mesh(np.asarray(jax.devices()[:n]), ("d",))


@pytest.fixture(params=sorted(BUILDS))
def build(request, monkeypatch):
    module, fn = BUILDS[request.param]
    monkeypatch.setattr(module, "combine_step_tokens", lambda tokens: False)
    return module, fn


def _psum_combine(slab, slab_row, coeff, axis_name, chunk_tokens):
    """The all-reduce form it replaces: the same chunks, one psum per chunk."""
    tokens = slab_row.shape[0]
    h = slab.shape[1] * slab.shape[2]
    chunks = next(
        n
        for n in range(1, tokens + 1)
        if not tokens % n and tokens // n <= chunk_tokens
    )
    chunk = tokens // chunks
    outs = []
    for c in range(chunks):
        rows = slice(c * chunk, (c + 1) * chunk)
        gathered = slab.reshape(-1, h)[slab_row[rows]].astype(jnp.float32)
        y = jnp.sum(gathered * coeff[rows][..., None], axis=1).astype(jnp.bfloat16)
        outs.append(lax.psum(y, axis_name))
    return outs[0] if chunks == 1 else jnp.concatenate(outs, 0)


def _reduce(mesh, fn):
    return jax.jit(
        jax.shard_map(
            lambda s, r, c: fn(s, r, c, "d"),
            mesh=mesh,
            in_specs=(P("d"), P("d"), P("d")),
            out_specs=P(),
            check_vma=False,
        )
    )


def _inputs(ranks, tokens, seed):
    """Per rank: a [2 * tokens, H / 128, 128] slab, rows into it and weights.
    Integers in [-4, 4] times weights in {0, 1} over 3 slots and 16 ranks stay
    below 256 in magnitude: every partial sum is exact in bf16, so the result
    must match psum's whatever order either adds in, on any backend."""
    rng = np.random.default_rng(seed)
    rows = 2 * tokens
    slab = jnp.asarray(rng.integers(-4, 5, (ranks * rows, H // 128, 128)), jnp.bfloat16)
    slab_row = jnp.asarray(rng.integers(0, rows, (ranks * tokens, SLOTS)), jnp.int32)
    coeff = jnp.asarray(rng.integers(0, 2, (ranks * tokens, SLOTS)), jnp.float32)
    return slab, slab_row, coeff


def _count(hlo, op):
    return sum(
        f" {op}(" in line or f" {op}-start(" in line for line in hlo.splitlines()
    )


# 512-4096: the serving prefill token buckets (one or two 2048-row chunks).
# 1000 and 24: one chunk whose rows do not divide by 16 (padded). 6000: three
# 2000-row chunks.
@pytest.mark.parametrize("tokens", [24, 512, 1000, 1024, 2048, 4096, 6000])
def test_equals_psum(mesh, build, tokens):
    module, fn = build
    ranks = mesh.devices.size
    slab, slab_row, coeff = _inputs(ranks, tokens, tokens)

    def want_fn(s, r, c, a):
        return _psum_combine(s, r, c, a, module._COMBINE_CHUNK_TOKENS)

    got = np.asarray(_reduce(mesh, fn)(slab, slab_row, coeff))
    want = np.asarray(_reduce(mesh, want_fn)(slab, slab_row, coeff))
    assert got.shape == (tokens, H)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))
    # Positive control: dropping one rank's weights must change the sum.
    coeff0 = coeff.at[:tokens].set(0)
    other = np.asarray(_reduce(mesh, want_fn)(slab, slab_row, coeff0))
    assert not np.array_equal(got, other)


@pytest.mark.parametrize("tokens", [512, 1000, 1024, 2048, 4096])
def test_no_all_reduce(mesh, build, tokens):
    module, fn = build
    slab, slab_row, coeff = _inputs(mesh.devices.size, tokens, 0)
    hlo = _reduce(mesh, fn).lower(slab, slab_row, coeff).as_text(dialect="hlo")
    chunks = -(-tokens // module._COMBINE_CHUNK_TOKENS)
    assert _count(hlo, "all-reduce") == 0
    assert _count(hlo, "reduce-scatter") == chunks
    assert _count(hlo, "all-gather") == chunks
