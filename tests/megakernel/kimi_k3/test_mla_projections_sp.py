# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel MLA projections match the replicated ones.

Runs on an eight-device CPU mesh with
XLA_FLAGS=--xla_force_host_platform_device_count=8.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as pm

pytestmark = pytest.mark.cpu_test

_RANKS = 8


def _weights(hidden, query_rank):
    keys = jax.random.split(jax.random.key(0), 7)

    def normal(key, shape, scale):
        return (scale * jax.random.normal(key, shape)).astype(jnp.bfloat16)

    return pm.ProjectionWeights(
        query_down_projection=normal(keys[0], (hidden, query_rank), hidden**-0.5),
        query_up_projection=normal(keys[1], (query_rank, 3 * 256), query_rank**-0.5),
        key_value_down_projection=normal(keys[2], (hidden, 576), hidden**-0.5),
        key_value_up_projection=normal(keys[3], (512, 3 * 256), 512**-0.5),
        output_gate_projection=normal(keys[4], (hidden, 384), hidden**-0.5),
        query_normalization=1 + normal(keys[5], (query_rank,), 0.1),
        key_value_normalization=1 + normal(keys[6], (512,), 0.1),
    )


@pytest.mark.parametrize("tokens", [512, 4096])
def test_sp_projections_match_replicated(tokens):
    if jax.device_count() < _RANKS:
        pytest.skip(f"needs {_RANKS} devices")
    weights = _weights(256, 128)
    hidden = jax.random.normal(jax.random.key(1), (tokens, 256)).astype(jnp.bfloat16)
    # Rows past a short request's length are padding; they must not move.
    hidden = hidden.at[tokens - 100 :].set(0)
    expected = pm.mla_projections(hidden, weights, return_latent=True)

    mesh = Mesh(np.array(jax.devices()[:_RANKS]), ("k3_sp",))
    sharded = jax.jit(
        jax.shard_map(
            lambda h, w: pm.mla_projections(h, w, return_latent=True, sp_axis="k3_sp"),
            mesh=mesh,
            in_specs=(P(), P()),
            out_specs=P(),
            check_vma=False,
        )
    )(hidden, weights)

    # The CPU dot may block a 1/8 slice differently: a few values move by one
    # BF16 ulp.
    for got, want in zip(sharded, expected, strict=True):
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)
        assert np.mean(got != want) < 1e-4
