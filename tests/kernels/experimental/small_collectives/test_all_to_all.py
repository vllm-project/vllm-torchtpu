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
"""Tests for the small all-to-all Pallas kernel (COLLECTIVE_ID = 6)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_to_all as a2a

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


def _all_to_all(
    mesh,
    per_core,
    ranks,
    *,
    rank_operand: bool = True,
    impl: str = "oneshot",
    split_axis: int = 0,
    concat_axis: int = 0,
):
    n = per_core.shape[0]

    def body(x, r):
        return a2a.small_all_to_all(
            x,
            axis_name=AXIS,
            num_cores=n,
            split_axis=split_axis,
            concat_axis=concat_axis,
            rank=r if rank_operand else None,
            impl=impl,
        )

    if per_core.ndim == 3 and split_axis == 0 and concat_axis == 0:
        return run_sharded(mesh, body, per_core, ranks)

    spec = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(jnp.asarray(per_core), spec)
    rank = jax.device_put(jnp.asarray(ranks, dtype=jnp.int32), spec)
    fn = jax.jit(
        jax.shard_map(
            lambda blk, rnk: body(blk[0], rnk)[None],
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )
    return np.asarray(fn(x, rank))


@pytest.mark.parametrize("rows_per_peer", [1, 3, 4, 8, 12, 16, 24])
@pytest.mark.parametrize("impl", ["oneshot", "hierarchical"])
@pytest.mark.parametrize("rank_operand", [False, True])
def test_matches_exact_all_to_all_and_gate7(mesh, rows_per_peer, impl, rank_operand):
    per_core = blocks((N, N * rows_per_peer, 512), seed=500 + rows_per_peer)
    got = _all_to_all(
        mesh, per_core, range(N), rank_operand=rank_operand, impl=impl
    )
    want = np.asarray(a2a.reference(per_core, split_axis=0, concat_axis=0))
    for core in range(N):
        assert np.array_equal(got[core], want[core]), f"core {core} mismatch"
        cossim, sqnr_db = _cossim_and_sqnr_db(got[core], want[core])
        assert cossim >= 0.99999
        assert sqnr_db >= 45.0


@pytest.mark.parametrize("hidden", [64, 128, 200, 512, 4000, 6144])
@pytest.mark.parametrize("impl", ["oneshot", "hierarchical"])
def test_arbitrary_widths_and_unaligned_col_padding(mesh, hidden, impl):
    per_core = blocks((N, N * 3, hidden), seed=600 + hidden)
    got = _all_to_all(mesh, per_core, range(N), impl=impl)
    want = np.asarray(a2a.reference(per_core))
    for core in range(N):
        assert np.array_equal(got[core], want[core])


@pytest.mark.parametrize(
    "split_axis,concat_axis,shape",
    [
        (0, 1, (N, N * 4, 256)),
        (1, 0, (N, 6, N * 128)),
        (1, 1, (N, 4, N * 160)),
    ],
)
def test_split_and_concat_axes_match_lax_all_to_all(
    mesh, split_axis, concat_axis, shape
):
    per_core = blocks(shape, seed=700 + split_axis * 10 + concat_axis)
    got = _all_to_all(
        mesh,
        per_core,
        range(N),
        split_axis=split_axis,
        concat_axis=concat_axis,
    )
    spec = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(jnp.asarray(per_core), spec)
    want = np.asarray(
        jax.jit(
            jax.shard_map(
                lambda blk: jax.lax.all_to_all(
                    blk[0],
                    AXIS,
                    split_axis=split_axis,
                    concat_axis=concat_axis,
                    tiled=True,
                )[None],
                mesh=mesh,
                in_specs=P(AXIS),
                out_specs=P(AXIS),
                check_vma=False,
            )
        )(x)
    )
    ref = np.asarray(
        a2a.reference(per_core, split_axis=split_axis, concat_axis=concat_axis)
    )
    for core in range(N):
        assert np.array_equal(got[core], want[core])
        assert np.array_equal(got[core], ref[core])


def test_1d_and_3d_all_to_all(mesh):
    # 1D input per core: [N * 128] -> [N * 128]
    per_core_1d = blocks((N, N * 128), seed=710)
    got_1d = _all_to_all(mesh, per_core_1d, range(N))
    want_1d = np.asarray(a2a.reference(per_core_1d, split_axis=0, concat_axis=0))
    for core in range(N):
        assert np.array_equal(got_1d[core], want_1d[core])

    # 3D input per core: [N * 2, 4, 128] -> [N * 2, 4, 128]
    per_core_3d = blocks((N, N * 2, 4, 128), seed=711)
    got_3d = _all_to_all(mesh, per_core_3d, range(N), impl="hierarchical")
    want_3d = np.asarray(a2a.reference(per_core_3d, split_axis=0, concat_axis=0))
    for core in range(N):
        assert np.array_equal(got_3d[core], want_3d[core])


def test_wrong_rank_order_fails_oracle(mesh):
    """Positive control: swapping core 0 and core 1 ranks routes their chunks
    to the wrong slots, which the oracle catches."""
    per_core = blocks((N, N * 8, 512), seed=720)
    swapped = [1, 0, *range(2, N)]
    got = _all_to_all(mesh, per_core, swapped)
    want = np.asarray(a2a.reference(per_core))
    for core in (0, 1):
        assert not np.array_equal(got[core], want[core])


@pytest.mark.parametrize(
    "rows_per_peer,hidden",
    [
        (1, 64),         # 1 KiB total across 8 peers
        (4, 2048),       # 128 KiB per core
        (8, 6144),       # GLM-5.2 H=6144
        (256, 8192),     # 32 MiB..128 MiB per core
    ],
)
def test_payload_range_1kib_to_128mib_and_32_cores_jaxpr(rows_per_peer, hidden):
    r = jnp.array([0], jnp.int32)
    for n_cores in (8, 32):
        x = jnp.zeros((n_cores * rows_per_peer, hidden), jnp.bfloat16)
        for impl in ("oneshot", "hierarchical"):
            jaxpr = jax.make_jaxpr(
                lambda a, rk: a2a.small_all_to_all(
                    a, axis_name=AXIS, num_cores=n_cores, rank=rk, impl=impl
                )
            )(x, r)
            out_aval = jaxpr.out_avals[0]
            assert out_aval.shape == (n_cores * rows_per_peer, hidden)
            assert out_aval.dtype == jnp.bfloat16


def test_negative_controls_validation():
    with pytest.raises(ValueError, match="split"):
        a2a.small_all_to_all(
            jnp.zeros((13, 512), jnp.bfloat16), axis_name=AXIS, num_cores=N
        )
    with pytest.raises(ValueError, match="lanes"):
        a2a.small_all_to_all(
            jnp.zeros((N * 8, 4000), jnp.bfloat16),
            axis_name=AXIS,
            num_cores=N,
            allow_col_padding=False,
        )
    with pytest.raises(ValueError, match="Unknown all_to_all impl"):
        a2a.small_all_to_all(
            jnp.zeros((N * 8, 512), jnp.bfloat16),
            axis_name=AXIS,
            num_cores=N,
            impl="invalid",
        )
