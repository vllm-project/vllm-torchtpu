# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE prefill kernel's cross-rank sum (no TPU needed): each combine
chunk is reduced as a reduce-scatter plus an all-gather, never as one
all-reduce, and the result equals psum's, including chunks whose rows do not
divide by the axis size (padded and sliced back).

Runs on up to 16 devices; CPU virtual devices are sufficient, e.g.
XLA_FLAGS=--xla_force_host_platform_device_count=16."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel as kernel,
)

pytestmark = pytest.mark.multichip

H = 256


@pytest.fixture
def mesh():
    # A fixture, not a module-level skipif: collection must not claim a backend.
    n = min(jax.device_count(), 16)
    if n < 2:
        pytest.skip("requires two or more devices; CPU virtual devices are sufficient")
    return Mesh(np.asarray(jax.devices()[:n]), ("d",))


def _reduce(mesh, tokens, fn):
    """fn(tokens, combine_chunk, "d") under shard_map, the combine chunk being
    this rank's rows of a [ranks * tokens, H] input."""

    def body(x):
        return fn(tokens, lambda rows: x[rows], "d")

    return jax.jit(
        jax.shard_map(body, mesh=mesh, in_specs=P("d"), out_specs=P(), check_vma=False)
    )


def _psum_chunks(tokens, combine_chunk, axis_name):
    """The all-reduce form it replaces: _chunked_psum's chunking with one psum
    per chunk."""
    chunks = next(
        n
        for n in range(1, tokens + 1)
        if not tokens % n and tokens // n <= kernel._COMBINE_CHUNK_TOKENS
    )
    chunk = tokens // chunks
    outs = [
        lax.psum(combine_chunk(slice(c * chunk, (c + 1) * chunk)), axis_name)
        for c in range(chunks)
    ]
    return outs[0] if chunks == 1 else jnp.concatenate(outs, 0)


def _count(hlo, op):
    return sum(
        f" {op}(" in line or f" {op}-start(" in line for line in hlo.splitlines()
    )


# 512-4096: the serving prefill token buckets (one or two 2048-row chunks).
# 1000 and 24: one chunk whose rows do not divide by 16 (padded). 6000: three
# 2000-row chunks.
@pytest.mark.parametrize("tokens", [24, 512, 1000, 1024, 2048, 4096, 6000])
def test_equals_psum(mesh, tokens):
    ranks = mesh.devices.size
    rng = np.random.default_rng(tokens)
    # Integers in [-16, 16]: every partial sum is exact in bf16, so the result
    # must match psum's whatever order either adds in, on any backend, and a
    # row reduced into the wrong place or dropped by the padding shows.
    x = jnp.asarray(rng.integers(-16, 17, (ranks * tokens, H)), jnp.bfloat16)
    got = np.asarray(_reduce(mesh, tokens, kernel._chunked_psum)(x))
    want = np.asarray(_reduce(mesh, tokens, _psum_chunks)(x))
    assert got.shape == (tokens, H)
    np.testing.assert_array_equal(got.view(np.uint16), want.view(np.uint16))
    # Positive control: dropping one rank's rows must change the sum.
    x0 = x.at[:tokens].set(0)
    assert not np.array_equal(got, np.asarray(_reduce(mesh, tokens, _psum_chunks)(x0)))


@pytest.mark.parametrize("tokens", [512, 1000, 1024, 2048, 4096])
def test_no_all_reduce(mesh, tokens):
    x = jax.ShapeDtypeStruct((mesh.devices.size * tokens, H), jnp.bfloat16)
    hlo = _reduce(mesh, tokens, kernel._chunked_psum).lower(x).as_text(dialect="hlo")
    chunks = -(-tokens // kernel._COMBINE_CHUNK_TOKENS)
    assert _count(hlo, "all-reduce") == 0
    assert _count(hlo, "reduce-scatter") == chunks
    assert _count(hlo, "all-gather") == chunks
