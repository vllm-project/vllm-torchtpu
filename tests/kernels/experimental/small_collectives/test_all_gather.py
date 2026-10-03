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
"""Tests for the small all-gather Pallas kernel (COLLECTIVE_ID = 8)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_gather as ag

from .oracle import AXIS, N, blocks, run_sharded

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < N:
        pytest.skip(f"needs at least {N} JAX devices")
    devices = np.array(jax.devices()[:N])
    if jax.default_backend() == "tpu":
        yield Mesh(devices, (AXIS,))
    else:
        with pltpu.force_tpu_interpret_mode():
            yield Mesh(devices, (AXIS,))


def _cossim_and_sqnr_db(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    g = got.astype(np.float64).ravel()
    w = want.astype(np.float64).ravel()
    denom = np.linalg.norm(g) * np.linalg.norm(w)
    cossim = 1.0 if denom == 0.0 else float(np.dot(g, w) / denom)
    noise = np.sum((g - w) ** 2)
    sig = np.sum(w**2)
    sqnr_db = float("inf") if noise == 0.0 else float(10.0 * np.log10(sig / noise))
    return cossim, sqnr_db


def _all_gather(
    mesh,
    per_core,
    ranks,
    *,
    rank_operand: bool = True,
    impl: str = "oneshot",
    gather_dimension: int = 0,
    tiled: bool = True,
):
    n = per_core.shape[0]

    def body(x, r):
        return ag.small_all_gather(
            x,
            num_cores=n,
            axis=AXIS,
            rank=r if rank_operand else None,
            impl=impl,
            gather_dimension=gather_dimension,
            tiled=tiled,
        )

    if per_core.ndim == 3 and gather_dimension == 0 and tiled:
        return run_sharded(mesh, body, per_core, ranks)

    spec = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(jnp.asarray(per_core), spec)
    rank = jax.device_put(jnp.asarray(ranks, dtype=jnp.int32).reshape(n, 1), spec)
    fn = jax.jit(
        jax.shard_map(
             lambda blk, rnk: body(blk[0], rnk[0]),
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    out = np.asarray(fn(x, rank[:, None]))
    return out


@pytest.mark.parametrize("rows", [1, 3, 4, 8, 12, 16, 32])
@pytest.mark.parametrize("impl", ["oneshot", "hierarchical"])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_exact_all_gather_and_gate7(mesh, rows, impl, rank_operand):
    per_core = blocks((N, rows, 512), seed=200 + rows)
    got = _all_gather(
        mesh, per_core, range(N), rank_operand=rank_operand, impl=impl
    )
    want = np.asarray(ag.reference(per_core, gather_dimension=0, tiled=True))
    for core in range(N):
        assert np.array_equal(got[core], want), f"core {core} mismatch"
        cossim, sqnr_db = _cossim_and_sqnr_db(got[core], want)
        assert cossim >= 0.99999
        assert sqnr_db >= 45.0


@pytest.mark.parametrize("hidden", [64, 128, 200, 512, 4000, 6144])
@pytest.mark.parametrize("impl", ["oneshot", "hierarchical"])
def test_arbitrary_widths_and_unaligned_col_padding(mesh, hidden, impl):
    per_core = blocks((N, 5, hidden), seed=300 + hidden)
    got = _all_gather(mesh, per_core, range(N), impl=impl)
    want = np.asarray(ag.reference(per_core))
    for core in range(N):
        assert np.array_equal(got[core], want)


def test_1d_and_3d_shapes_and_gather_dimension_1(mesh):
    # 1D input per core: [512] -> [N * 512]
    per_core_1d = blocks((N, 512), seed=401)
    spec = NamedSharding(mesh, P(AXIS))
    x_1d = jax.device_put(jnp.asarray(per_core_1d), spec)
    r_1d = jax.device_put(jnp.arange(N, dtype=jnp.int32), spec)
    fn_1d = jax.jit(
        jax.shard_map(
            lambda blk, rnk: ag.small_all_gather(
                blk[0], num_cores=N, axis_name=AXIS, rank=rnk
            )[None],
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    got_1d = np.asarray(fn_1d(x_1d, r_1d))
    want_1d = np.asarray(per_core_1d.reshape(N * 512))
    for core in range(N):
        assert np.array_equal(got_1d[core], want_1d)

    # 2D gather_dimension=1: [4, 256] -> [4, N * 256]
    per_core_g1 = blocks((N, 4, 256), seed=402)
    x_g1 = jax.device_put(jnp.asarray(per_core_g1), spec)
    fn_g1 = jax.jit(
        jax.shard_map(
            lambda blk, rnk: ag.small_all_gather(
                blk[0], num_cores=N, axis=AXIS, rank=rnk, gather_dimension=1
            )[None],
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    got_g1 = np.asarray(fn_g1(x_g1, r_1d))
    want_g1 = np.asarray(ag.reference(per_core_g1, gather_dimension=1, tiled=True))
    for core in range(N):
        assert np.array_equal(got_g1[core], want_g1)

    # 3D input per core: [3, 4, 128] -> [N * 3, 4, 128]
    per_core_3d = blocks((N, 3, 4, 128), seed=403)
    x_3d = jax.device_put(jnp.asarray(per_core_3d), spec)
    fn_3d = jax.jit(
        jax.shard_map(
            lambda blk, rnk: ag.small_all_gather(
                blk[0], num_cores=N, axis=AXIS, rank=rnk, impl="hierarchical"
            )[None],
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    got_3d = np.asarray(fn_3d(x_3d, r_1d))
    want_3d = np.asarray(ag.reference(per_core_3d, gather_dimension=0, tiled=True))
    for core in range(N):
        assert np.array_equal(got_3d[core], want_3d)


def test_agrees_with_lax_all_gather(mesh):
    per_core = blocks((N, 6, 512), seed=404)
    want = run_sharded(
        mesh,
        lambda x, r: jax.lax.all_gather(x, AXIS, axis=0, tiled=True),
        per_core,
        range(N),
    )
    got = _all_gather(mesh, per_core, range(N), impl="oneshot")
    for core in range(N):
        assert np.array_equal(got[core], want[core])


def test_wrong_rank_order_fails_oracle(mesh):
    """Positive control: swapping core 0 and core 1 ranks places their slices
    into the wrong output slots, which the oracle catches."""
    per_core = blocks((N, 8, 512), seed=405)
    swapped = [1, 0, *range(2, N)]
    got = _all_gather(mesh, per_core, swapped)
    want = np.asarray(ag.reference(per_core))
    for core in range(N):
        assert not np.array_equal(got[core], want)


@pytest.mark.parametrize(
    "rows,hidden",
    [
        (1, 512),        # 1 KiB per core
        (8, 2048),       # 32 KiB per core
        (32, 6144),      # 384 KiB per core (GLM-5.2 H=6144)
        (1024, 8192),    # 16 MiB per core -> 128 MiB gathered on 8 cores
    ],
)
def test_payload_range_1kib_to_128mib_and_32_cores_jaxpr(rows, hidden):
    """Verify trace and shape lowering across 1 KiB..128 MiB and 32 cores (2x2x4)."""
    x = jnp.zeros((rows, hidden), jnp.bfloat16)
    r = jnp.array([0], jnp.int32)
    for n_cores in (8, 32):
        for impl in ("oneshot", "hierarchical"):
            jaxpr = jax.make_jaxpr(
                lambda a, rk: ag.small_all_gather(
                    a, num_cores=n_cores, axis=AXIS, rank=rk, impl=impl
                )
            )(x, r)
            out_aval = jaxpr.out_avals[0]
            assert out_aval.shape == (n_cores * rows, hidden)
            assert out_aval.dtype == jnp.bfloat16


def test_negative_controls_validation():
    x = jnp.zeros((8, 4000), jnp.bfloat16)
    with pytest.raises(ValueError, match="lanes"):
        ag.small_all_gather(
            x, num_cores=N, axis=AXIS, allow_col_padding=False
        )
    with pytest.raises(ValueError, match="Unknown all_gather impl"):
        ag.small_all_gather(
            jnp.zeros((8, 512), jnp.bfloat16), num_cores=N, axis=AXIS, impl="bogus"
        )
    with pytest.raises(ValueError, match="1D"):
        ag.small_all_gather(jnp.zeros((), jnp.bfloat16), num_cores=N, axis=AXIS)
    with pytest.raises(ValueError, match="axis"):
        ag.small_all_gather(jnp.zeros((8, 512), jnp.bfloat16), num_cores=N)
