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
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.mla import dcp_token_parallel as dtp


def _separate_all_gather_ref(
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    page_ids: jax.Array,
    axis_name: str,
    read_pages: tuple,
) -> tuple[jax.Array, jax.Array]:
    ln = read_pages[0](nope_cache, page_ids)
    lr = read_pages[1](rope_cache, page_ids)
    g_n = lax.all_gather(ln.reshape(-1), axis_name, tiled=True).reshape(
        -1, *nope_cache.shape[1:]
    )
    g_r = lax.all_gather(lr.reshape(-1), axis_name, tiled=True).reshape(
        -1, *rope_cache.shape[1:]
    )
    return g_n, g_r


def _count_all_gather_eqns(jaxpr) -> int:
    count = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "all_gather":
            count += 1
        for sub in jax.core.jaxprs_in_params(eqn.params):
            count += _count_all_gather_eqns(sub)
    return count


def _make_mesh(world: int) -> Mesh:
    devs = jax.devices()
    if len(devs) < world:
        pytest.skip(f"needs {world} JAX devices, found {len(devs)}")
    return Mesh(np.array(devs[:world]), ("dcp",))


@pytest.mark.parametrize(
    "case_name,rope_tail,read_width_rope",
    [
        ("packed_rope_2d_linear", (32, 128), 16),
        ("packed_rope_direct_width16", (256, 16), 16),
        ("unpacked_rope_quarter_major", (64, 128), 128),
    ],
)
def test_should_fuse_and_match_separate_all_gather_when_page_gather_returns_2d(
    case_name: str,
    rope_tail: tuple[int, int],
    read_width_rope: int,
) -> None:
    del case_name
    world = 2
    local_pages = 5
    mesh = _make_mesh(world)
    rng = np.random.default_rng(17)
    nope_np = rng.integers(
        1, 0x7FFFFFFF, size=(world * local_pages, 256, 128), dtype=np.uint32
    )
    rope_np = rng.integers(
        1, 0x7FFFFFFF, size=(world * local_pages, *rope_tail), dtype=np.uint32
    )
    # Out-of-order with a repeated page id to catch cross-page or lane mixing.
    page_ids = jnp.asarray([3, 0, 4, 3], dtype=jnp.int32)

    def read_nope_linear(cache: jax.Array, pids: jax.Array) -> jax.Array:
        return cache[pids].reshape(pids.shape[0] * 256, 128)

    def read_rope_linear(cache: jax.Array, pids: jax.Array) -> jax.Array:
        return cache[pids].reshape(-1, read_width_rope)

    def run_fused(n: jax.Array, r: jax.Array, pids: jax.Array):
        return dtp.gather_latent(
            n, r, pids, "dcp", read_pages=(read_nope_linear, read_rope_linear)
        )

    def run_ref(n: jax.Array, r: jax.Array, pids: jax.Array):
        return _separate_all_gather_ref(
            n, r, pids, "dcp", read_pages=(read_nope_linear, read_rope_linear)
        )

    sm_fused = jax.shard_map(
        run_fused,
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    sm_ref = jax.shard_map(
        run_ref,
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    n_arr = jnp.asarray(nope_np)
    r_arr = jnp.asarray(rope_np)

    closed = jax.make_jaxpr(sm_fused)(n_arr, r_arr, page_ids)
    assert _count_all_gather_eqns(closed.jaxpr) == 1

    gn, gr = sm_fused(n_arr, r_arr, page_ids)
    rn, rr = sm_ref(n_arr, r_arr, page_ids)
    assert gn.shape == rn.shape
    assert gr.shape == rr.shape
    np.testing.assert_array_equal(np.asarray(gn), np.asarray(rn))
    np.testing.assert_array_equal(np.asarray(gr), np.asarray(rr))

    # Positive control: swapping page order must fail exact equality.
    flipped_ids = jnp.asarray([0, 3, 4, 3], dtype=jnp.int32)
    _, rr_bad = sm_ref(n_arr, r_arr, flipped_ids)
    assert not np.array_equal(np.asarray(gr), np.asarray(rr_bad))


def test_should_fuse_and_match_when_using_default_take_pages() -> None:
    world = 2
    local_pages = 3
    mesh = _make_mesh(world)
    rng = np.random.default_rng(29)
    nope_np = rng.integers(
        1, 0x7FFFFFFF, size=(world * local_pages, 16, 128), dtype=np.uint32
    )
    rope_np = rng.integers(
        1, 0x7FFFFFFF, size=(world * local_pages, 8, 128), dtype=np.uint32
    )
    page_ids = jnp.asarray([2, 0], dtype=jnp.int32)

    sm_fused = jax.shard_map(
        lambda n, r, p: dtp.gather_latent(n, r, p, "dcp"),
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    sm_ref = jax.shard_map(
        lambda n, r, p: _separate_all_gather_ref(
            n, r, p, "dcp", (dtp.take_pages, dtp.take_pages)
        ),
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    n_arr, r_arr = jnp.asarray(nope_np), jnp.asarray(rope_np)

    closed = jax.make_jaxpr(sm_fused)(n_arr, r_arr, page_ids)
    assert _count_all_gather_eqns(closed.jaxpr) == 1

    gn, gr = sm_fused(n_arr, r_arr, page_ids)
    rn, rr = sm_ref(n_arr, r_arr, page_ids)
    np.testing.assert_array_equal(np.asarray(gn), np.asarray(rn))
    np.testing.assert_array_equal(np.asarray(gr), np.asarray(rr))


@pytest.mark.parametrize(
    "reason,nope_dtype,rope_dtype,nope_rows,rope_rows",
    [
        ("mismatched_dtype", np.uint32, np.int32, 16, 8),
        ("unaligned_rows", np.uint32, np.uint32, 12, 8),
    ],
)
def test_should_fall_back_to_separate_all_gathers_when_ineligible(
    reason: str,
    nope_dtype: type,
    rope_dtype: type,
    nope_rows: int,
    rope_rows: int,
) -> None:
    del reason
    world = 2
    local_pages = 2
    mesh = _make_mesh(world)
    rng = np.random.default_rng(43)
    nope_np = rng.integers(1, 1000, size=(world * local_pages, nope_rows, 16)).astype(
        nope_dtype
    )
    rope_np = rng.integers(1, 1000, size=(world * local_pages, rope_rows, 16)).astype(
        rope_dtype
    )
    page_ids = jnp.asarray([1, 0], dtype=jnp.int32)

    sm_fn = jax.shard_map(
        lambda n, r, p: dtp.gather_latent(n, r, p, "dcp"),
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    sm_ref = jax.shard_map(
        lambda n, r, p: _separate_all_gather_ref(
            n, r, p, "dcp", (dtp.take_pages, dtp.take_pages)
        ),
        mesh=mesh,
        in_specs=(P("dcp"), P("dcp"), P()),
        out_specs=(P(), P()),
        check_vma=False,
    )
    n_arr, r_arr = jnp.asarray(nope_np), jnp.asarray(rope_np)

    closed = jax.make_jaxpr(sm_fn)(n_arr, r_arr, page_ids)
    assert _count_all_gather_eqns(closed.jaxpr) == 2

    gn, gr = sm_fn(n_arr, r_arr, page_ids)
    rn, rr = sm_ref(n_arr, r_arr, page_ids)
    np.testing.assert_array_equal(np.asarray(gn), np.asarray(rn))
    np.testing.assert_array_equal(np.asarray(gr), np.asarray(rr))
