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
"""One-shot DMA all-gather kernel against XLA's all-gather: bitwise, back to
back, and with a late member."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.dma_all_gather import dma_all_gather

_SPEC = P("dp", "tp")
# Words a packed route can take at the edges: zero, -1, 1, the int32 extremes,
# the empty route (expert id -1, weight +0.0) and expert 32767 with weight +0.0.
_EDGES = np.array([0, -1, 1, -(2**31), 2**31 - 1, -1 << 16, 0x7FFF0000], np.int32)
# Odd int32 multiplier that makes each call's input depend on the call index.
_MIX = -1640531535


def _mesh(dp, tp):
    devices = sorted(jax.devices(), key=lambda d: d.id)[:8]
    return Mesh(np.asarray(devices).reshape(dp, tp), ("dp", "tp"))


def _blocks(seed, dp, tp, shape):
    rng = np.random.default_rng(seed)
    x = rng.integers(-(2**31), 2**31 - 1, size=(dp, tp) + shape, dtype=np.int32)
    x.reshape(-1)[: len(_EDGES)] = _EDGES
    return x


def _expected(x, axis):
    """Per rank (d, t): the blocks of its group along ``axis``, in order."""
    dp, tp = x.shape[:2]
    out = np.empty((dp, tp, x.shape[0 if axis == "dp" else 1]) + x.shape[2:], x.dtype)
    for d in range(dp):
        for t in range(tp):
            out[d, t] = x[:, t] if axis == "dp" else x[d, :]
    return out


def _gather_both(mesh, axis, x, interpret=False):
    """dma_all_gather and lax.all_gather of every rank's block in one program."""
    n = mesh.shape[axis]

    def body(a):
        a = a[0, 0]
        got = dma_all_gather(a, axis=axis, num_members=n, interpret=interpret)
        return got[None, None], lax.all_gather(a, axis)[None, None]

    fn = jax.shard_map(
        body, mesh=mesh, in_specs=(_SPEC,), out_specs=(_SPEC, _SPEC), check_vma=False
    )
    got, ref = jax.jit(fn)(jnp.asarray(x))
    return np.asarray(got), np.asarray(ref)


def _hold(delay_ns, x):
    """``x`` unchanged, after this rank waits ``delay_ns[0]`` ns: a late member."""

    def kernel(delay_ref, x_hbm, o_hbm, sem):
        @pl.when(delay_ref[0] > 0)
        def _():
            pl.delay(delay_ref[0])

        copy = pltpu.make_async_copy(x_hbm, o_hbm, sem.at[0])
        copy.start()
        copy.wait()

    space = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[space],
            out_specs=space,
            scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        ),
    )(delay_ns, x)


def _chain(mesh, axis, x, *, calls, late_ns, interpret=False):
    """``calls`` back-to-back gathers: call ``i`` gathers every member's block,
    and member ``m`` takes block ``(m + i + 1) % n`` mixed with ``i`` and ``m``
    as its next block. With ``late_ns``, member ``i % n`` enters call ``i``
    that much later than the others."""
    n = mesh.shape[axis]

    def body(a):
        me = lax.axis_index(axis)

        def step(i, block):
            if late_ns:
                hold = jnp.where(lax.rem(i, n) == me, late_ns, 0)
                block = _hold(hold.astype(jnp.int32)[None], block)
            got = dma_all_gather(block, axis=axis, num_members=n, interpret=interpret)
            nxt = lax.dynamic_index_in_dim(got, lax.rem(me + i + 1, n), keepdims=False)
            return nxt ^ (i * jnp.int32(_MIX) + me)

        return lax.fori_loop(0, calls, step, a[0, 0])[None, None]

    fn = jax.shard_map(
        body, mesh=mesh, in_specs=(_SPEC,), out_specs=_SPEC, check_vma=False
    )
    return np.asarray(jax.jit(fn)(jnp.asarray(x)))


def _chain_reference(x, axis, calls):
    """`_chain` on the host, per gather group."""
    blocks = x.view(np.uint32).copy()
    if axis == "tp":
        blocks = blocks.transpose(1, 0, *range(2, x.ndim))
    n = blocks.shape[0]
    for i in range(calls):
        prev = blocks.copy()
        for m in range(n):
            mix = np.uint32((i * _MIX + m) & 0xFFFFFFFF)
            blocks[m] = prev[(m + i + 1) % n] ^ mix
    if axis == "tp":
        blocks = blocks.transpose(1, 0, *range(2, x.ndim))
    return blocks.view(np.int32)


def _require_eight_tpu_devices():
    devices = jax.devices()
    if devices[0].platform != "tpu" or len(devices) < 8:
        pytest.skip("needs 8 TPU devices")


@pytest.mark.parametrize(
    "dp,tp,axis", [(8, 1, "dp"), (4, 2, "dp"), (2, 4, "dp"), (2, 4, "tp")]
)
@pytest.mark.parametrize("shape", [(10, 112), (10, 28), (10, 136), (3, 5)])
def test_matches_xla_all_gather(dp, tp, axis, shape):
    _require_eight_tpu_devices()
    x = _blocks(dp * 100 + shape[1], dp, tp, shape)
    got, ref = _gather_both(_mesh(dp, tp), axis, x)
    np.testing.assert_array_equal(ref, _expected(x, axis))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("dp,tp,axis", [(8, 1, "dp"), (4, 2, "dp"), (2, 4, "tp")])
@pytest.mark.parametrize("late_ns", [0, 10_000])
def test_back_to_back_calls_with_a_late_member(dp, tp, axis, late_ns):
    """64 chained calls, each member's next block taken from its own call's
    output: a block landing in the wrong call or slot changes the result."""
    _require_eight_tpu_devices()
    x = _blocks(7 + dp, dp, tp, (10, 112))
    got = _chain(_mesh(dp, tp), axis, x, calls=64, late_ns=late_ns)
    np.testing.assert_array_equal(got, _chain_reference(x, axis, 64))


@pytest.mark.cpu_test
@pytest.mark.parametrize("dp,tp,axis", [(8, 1, "dp"), (2, 4, "dp"), (2, 4, "tp")])
def test_interpret_matches_xla_all_gather(dp, tp, axis):
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices")
    x = _blocks(dp, dp, tp, (10, 28))
    mesh = _mesh(dp, tp)
    got, ref = _gather_both(mesh, axis, x, interpret=True)
    np.testing.assert_array_equal(ref, _expected(x, axis))
    np.testing.assert_array_equal(got, ref)
    chained = _chain(mesh, axis, x, calls=3, late_ns=0, interpret=True)
    np.testing.assert_array_equal(chained, _chain_reference(x, axis, 3))


@pytest.mark.cpu_test
def test_one_member_needs_no_kernel():
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices")
    x = _blocks(1, 1, 8, (10, 28))
    got, ref = _gather_both(_mesh(1, 8), "dp", x)
    np.testing.assert_array_equal(got, ref)
    lowered = jax.jit(
        functools.partial(dma_all_gather, axis="dp", num_members=1)
    ).lower(jnp.asarray(x[0, 0]))
    # The module is named after the function; a kernel lowers to tpu_custom_call.
    assert "tpu_custom_call" not in lowered.as_text()
