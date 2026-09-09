# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Owner GMM contracts: partial expert tiles and ignored capacity tails."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.megablox import gmm_v2_owner as owner
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.kernels.megablox.gmm_v2_owner import gmm_v2_owner
from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import \
    blockwise_onehot_unpermute


@pytest.mark.nightly
@pytest.mark.parametrize("sizes", [(129, 0, 31, 63), (0, 0, 0, 0)])
@pytest.mark.parametrize("shape", [(320, 32, 256, 512), (320, 48, 768, 1536),
                                   (320, 32, 512, 6144)])
@pytest.mark.parametrize("scale_dtype", [jnp.float32, jnp.bfloat16])
def test_owner_gmm_preserves_valid_routes_and_ignores_nan_tail(
        sizes, shape, scale_dtype):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU")
    rng = np.random.default_rng(17)
    routes, tokens, k, n = shape
    valid = sum(sizes)
    lhs = rng.normal(0, 0.2, (routes, k)).astype(np.float32)
    lhs[valid:] = np.nan
    ids = rng.integers(0, tokens, routes, dtype=np.int32)
    ids[valid:] = -1
    weights = rng.uniform(0, 0.2, routes).astype(np.float32)
    weights[valid:] = np.nan
    lhs = jnp.asarray(lhs, jnp.bfloat16)
    ids = jnp.asarray(ids)
    weights = jnp.asarray(weights, jnp.bfloat16)
    rhs = jnp.asarray(rng.normal(0, 10, (4, k, n)), jnp.float8_e4m3fn)
    scales = jnp.full((4, 1, 1, n), 0.01, scale_dtype)
    groups = jnp.asarray(sizes, jnp.int32)

    routed = gmm_v2(lhs, rhs, groups, scales, zero_initialize=False)
    expected = blockwise_onehot_unpermute(jnp.pad(routed, ((0, 64), (0, 0))),
                                          jnp.pad(ids, (0, 64),
                                                  constant_values=-1),
                                          jnp.pad(weights, (0, 64)),
                                          groups.sum()[None],
                                          num_tokens=tokens)
    actual = gmm_v2_owner(lhs,
                          rhs,
                          groups,
                          scales,
                          ids,
                          weights,
                          num_tokens=tokens)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.isfinite(np.asarray(actual, dtype=np.float32)).all()
    if valid == 0:
        np.testing.assert_array_equal(np.asarray(actual), 0)


def _specs(rows=320, tokens=32, experts=4, k=256, n=512):
    return (jax.ShapeDtypeStruct((rows, k), jnp.bfloat16),
            jax.ShapeDtypeStruct((experts, k, n), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((experts, ), jnp.int32),
            jax.ShapeDtypeStruct((experts, 1, 1, n), jnp.float32),
            jax.ShapeDtypeStruct((rows, ), jnp.int32),
            jax.ShapeDtypeStruct((rows, ), jnp.bfloat16))


@pytest.fixture
def tpu_info(monkeypatch):
    info = types.SimpleNamespace(vmem_capacity_bytes=64 * 1024 * 1024,
                                 smem_capacity_bytes=1024 * 1024,
                                 mxu_column_size=256,
                                 num_lanes=128,
                                 fp8_ops_per_second=1,
                                 get_sublane_tiling=lambda dtype: 16)
    monkeypatch.setattr(owner.pltpu, "get_tpu_info", lambda: info)
    return info


def test_owner_admission_uses_capacity_and_device_budget(tpu_info):
    small = _specs()
    assert owner.can_use_gmm_v2_owner(*small, num_tokens=32)
    # No tensors are allocated/read: even an all-empty runtime routing must
    # reserve its full static capacity, so large shapes are rejected.
    assert not owner.can_use_gmm_v2_owner(*_specs(rows=65536, n=4096),
                                          num_tokens=32)
    assert not owner.can_use_gmm_v2_owner(*_specs(experts=100000),
                                          num_tokens=32)
    tpu_info.vmem_capacity_bytes = 1024 * 1024
    assert not owner.can_use_gmm_v2_owner(*small, num_tokens=32)


@pytest.mark.parametrize("index,replacement", [
    (0, jax.ShapeDtypeStruct((320, 256), jnp.float32)),
    (1, jax.ShapeDtypeStruct((4, 256, 512), jnp.bfloat16)),
    (2, jax.ShapeDtypeStruct((4, ), jnp.int16)),
    (3, None),
    (3, jax.ShapeDtypeStruct((4, 2, 1, 512), jnp.float32)),
    (4, jax.ShapeDtypeStruct((319, ), jnp.int32)),
    (5, jax.ShapeDtypeStruct((320, ), jnp.float32)),
])
def test_owner_admission_rejects_unsupported_formats(tpu_info, index,
                                                     replacement):
    inputs = list(_specs())
    inputs[index] = replacement
    assert not owner.can_use_gmm_v2_owner(*inputs, num_tokens=32)


@pytest.mark.parametrize("tokens,tile_n,tile_m", [
    (0, None, 128),
    (33, None, 128),
    (32, 0, 128),
    (32, 384, 128),
    (32, 1024, 128),
    (32, None, 0),
    (32, None, 192),
])
def test_owner_admission_rejects_invalid_alignment(tpu_info, tokens, tile_n,
                                                   tile_m):
    assert not owner.can_use_gmm_v2_owner(
        *_specs(), num_tokens=tokens, tile_n=tile_n, tile_m=tile_m)


def test_owner_admission_has_no_model_shape_whitelist(tpu_info):
    for tokens, topk, experts, k, n in ((16, 2, 4, 256, 512), (48, 6, 16, 768,
                                                               1536),
                                        (64, 8, 32, 2048, 6144), (256, 10, 64,
                                                                  1024, 4096)):
        assert owner.can_use_gmm_v2_owner(*_specs(rows=tokens * topk,
                                                  experts=experts,
                                                  k=k,
                                                  n=n),
                                          num_tokens=tokens)
