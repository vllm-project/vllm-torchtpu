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
"""Tests of the FP8 matmul fused with the all-reduce of its output.

The kernel must give, bit for bit, fp8_skinny's matmul on each core followed by
the float32 sum of the bf16 partials in rank order, rounded once to bf16. The
bitwise tests run on the eight cores of one host, at the tuned shape and at two
shapes that take the default ring setting. The geometry tests run anywhere.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.quantized_matmul import fp8_matmul_all_reduce as k
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

N = 8
AXIS = "d"


def test_tuned_settings_match_fp8_skinny():
    for (kk, n), (tile, nbuf) in k.TUNED.items():
        assert fp8_skinny.TUNED[(kk, n)] == ("n", tile, nbuf)


@pytest.mark.parametrize(
    "kk,n,config",
    [
        (1024, 6144, (2048, 3)),  # the table
        (2048, 4096, (1024, 3)),  # 2048 columns give only 2 tiles
        (512, 7168, (1024, 3)),  # 2048 does not divide 7168
        (4096, 8192, (512, 3)),  # 3 x 4096 x 1024 bytes exceed RING_BYTES
        (1024, 256, None),  # fewer than 3 tiles of any width
    ],
)
def test_pick_config(kk, n, config):
    assert k.pick_config(kk, n) == config


@pytest.mark.parametrize(
    "m,kk,n,cores,ok",
    [
        (4, 1024, 6144, 16, True),
        (16, 1024, 6144, 8, True),
        (16, 2048, 4096, 8, True),  # not in the table: the default setting
        (17, 1024, 6144, 16, False),  # above MAX_M
        (4, 1000, 6144, 8, False),  # k not a lane multiple
        (4, 1024, 6144, 1, False),
        (4, 1024, 6144, 32, False),  # 192-column owner chunks
        (4, 1024, 256, 2, False),  # no ring setting
    ],
)
def test_supports(m, kk, n, cores, ok):
    assert k.supports(m, kk, n, cores) is ok


def test_owner_waves_cover_every_owner_once_after_its_last_column():
    assert k.owner_waves(16, 6144, 2048) == [
        list(range(0, 5)),
        list(range(5, 10)),
        list(range(10, 16)),
    ]
    # 896-column chunks on 1024-column tiles: owners 6 and 7 both end in tile 6.
    assert k.owner_waves(8, 7168, 1024) == [[0], [1], [2], [3], [4], [5], [6, 7]]


@pytest.fixture(scope="module")
def mesh():
    """The eight cores of one host, asked for inside the test so that pytest
    collection does not claim the chip."""
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    return Mesh(np.array(jax.devices()[:N]), (AXIS,))


def _operands(rows, kk, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N * rows, kk), dtype=np.float32)
    w = rng.standard_normal((N * kk, n), dtype=np.float32) * 64
    w_s = rng.uniform(1e-4, 1e-3, N * n).astype(np.float32)
    return (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(w, jnp.float8_e4m3fn),
        jnp.asarray(w_s),
        jnp.arange(N, dtype=jnp.int32),
    )


def _run(mesh, body, operands):
    spec = P(AXIS)
    args = [jax.device_put(a, NamedSharding(mesh, spec)) for a in operands]
    fn = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(spec,) * len(operands),
        out_specs=spec,
        check_vma=False,
    )
    return np.asarray(jax.jit(fn)(*args))


@pytest.mark.multichip
@pytest.mark.parametrize("waves", [False, True])
@pytest.mark.parametrize(
    "rows,kk,n,rank_as_data",
    [
        (1, 1024, 6144, True),
        (4, 1024, 6144, True),
        (16, 1024, 6144, True),
        (4, 1024, 6144, False),
        (3, 2048, 4096, True),
        (16, 512, 7168, True),
    ],
)
def test_bitwise_fp8_skinny_then_rank_order_sum(mesh, rows, kk, n, rank_as_data, waves):
    tile, nbuf = k.pick_config(kk, n)

    def fused(x, w, w_s, rank):
        return k.fp8_matmul_all_reduce(
            x,
            w,
            w_s,
            rank if rank_as_data else None,
            num_cores=N,
            axis=AXIS,
            waves=waves,
        )

    def reference(x, w, w_s, rank):
        del rank
        y = fp8_skinny.fp8_skinny_matmul(x, w, w_s, config=("n", tile, nbuf))
        parts = jax.lax.all_gather(y, AXIS)
        acc = parts[0].astype(jnp.float32)
        for r in range(1, N):
            acc = acc + parts[r].astype(jnp.float32)
        return acc.astype(y.dtype)

    operands = _operands(rows, kk, n, seed=rows + kk + 17 * int(waves))
    got = _run(mesh, fused, operands)
    want = _run(mesh, reference, operands)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.view(np.int16), want.view(np.int16))


@pytest.mark.multichip
def test_stacked_weight_index(mesh):
    """idx = 1 selects the second weight of each core's two-weight stack."""
    rows, kk, n = 4, 1024, 6144
    x, w1, w_s, rank = _operands(rows, kk, n, seed=1)
    _, w0, _, _ = _operands(rows, kk, n, seed=2)
    # Core c's shard of w_stack is [w0 slice c, w1 slice c], shape [2, kk, n].
    w_stack = jnp.stack([w0.reshape(N, kk, n), w1.reshape(N, kk, n)], axis=1).reshape(
        N * 2, kk, n
    )
    idx = jnp.ones((N,), jnp.int32)

    def stacked(x, w_stack, idx, w_s, rank):
        return k.fp8_matmul_all_reduce_stacked(
            x, w_stack, idx, w_s, rank, num_cores=N, axis=AXIS
        )

    def single(x, w, w_s, rank):
        return k.fp8_matmul_all_reduce(x, w, w_s, rank, num_cores=N, axis=AXIS)

    got = _run(mesh, stacked, (x, w_stack, idx, w_s, rank))
    want = _run(mesh, single, (x, w1, w_s, rank))
    np.testing.assert_array_equal(got.view(np.int16), want.view(np.int16))


@pytest.mark.multichip
def test_num_cores_not_the_axis_size_raises(mesh):
    """num_cores=4 on the 8-core axis raises instead of leaving 4 cores waiting
    at the barrier. (4, 1024, 6144) on 4 cores passes `supports`."""

    def fused(x, w, w_s, rank):
        return k.fp8_matmul_all_reduce(x, w, w_s, rank, num_cores=4, axis=AXIS)

    with pytest.raises(ValueError, match="has 8 cores"):
        _run(mesh, fused, _operands(4, 1024, 6144, seed=0))


def _call(
    x_shape=(4, 1024),
    x_dtype=jnp.bfloat16,
    w_shape=(1024, 6144),
    w_dtype=jnp.float8_e4m3fn,
    w_s_shape=(6144,),
):
    """Calls the kernel wrapper outside any shard_map. Every case below fails
    an input check before the kernel lowers, so it runs on CPU."""
    k.fp8_matmul_all_reduce(
        jnp.zeros(x_shape, x_dtype),
        jnp.zeros(w_shape, w_dtype),
        jnp.ones(w_s_shape, jnp.float32),
        num_cores=N,
        axis=AXIS,
    )


def test_unsupported_shape_raises():
    with pytest.raises(ValueError, match="unsupported"):
        _call(x_shape=(17, 1024))


def test_x_not_bf16_raises():
    with pytest.raises(ValueError, match="x must be bfloat16, got float32"):
        _call(x_dtype=jnp.float32)


def test_weight_not_e4m3_raises():
    with pytest.raises(ValueError, match="float8_e4m3fn, got float8_e5m2"):
        _call(w_dtype=jnp.float8_e5m2)


def test_weight_k_not_x_k_raises():
    with pytest.raises(ValueError, match="the weight has k=2048"):
        _call(w_shape=(2048, 6144))


def test_w_s_shape_raises():
    with pytest.raises(ValueError, match=r"w_s must have shape \(6144,\)"):
        _call(w_s_shape=(1, 6144))
