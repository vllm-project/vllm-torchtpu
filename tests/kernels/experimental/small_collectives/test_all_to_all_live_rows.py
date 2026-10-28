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
"""The small all-to-all with ``live_rows`` (TPU_DCP_MLA_SEND_SMALL_A2A_LIVE_ROWS)
on 8 cores against the whole-block call, on the DCP merge's send blocks (8-,
16- and 32-token buckets, 4 heads per rank) with a real token count from 0 to
the bucket.

Every received row up to the tile row that holds the last real partial row
equals the whole-block call's, bit for bit, and the decode path's unpack
(``send_small_unpack_local``) returns the same partials and lse with the
switch on and off. In interpret mode the rows past that tile row keep the
output's uninitialized fill (int16 max, which the random send words never
take), so they were not sent. Positive control: a call one tile short leaves
real partial rows unsent, and the unpack comparison sees it.

On TPU it runs compiled (``multichip``). On a host with 8 CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu``) the
same checks run with the kernel in Pallas interpret mode.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_to_all as a2a

from .oracle import AXIS, N

LANE = "lane"
V = 512
LOCAL = 4
# kernels.mla.sparse.kernel.SEND_PAD_ROWS: the lse word row is row PAD - 1 of
# a send block, the partial rows start at PAD.
PAD = 16
ROW0 = (PAD - 1) // 8 * 8
FILL = np.iinfo(np.int16).max
FLAG = "TPU_DCP_MLA_SEND_SMALL_A2A_LIVE_ROWS"

# (bucket tokens, real tokens): decode batch 1-8 with MTP 3 and the edges.
CASES = [
    (8, 1),
    (8, 5),
    (8, 8),
    (16, 0),
    (16, 9),
    (16, 12),
    (16, 16),
    (32, 17),
    (32, 20),
    (32, 24),
    (32, 28),
    (32, 32),
]


@pytest.fixture(scope="module", params=["tpu", "interpret"])
def live_mesh(request):
    """The 8 TPU cores of one host, compiled; or 8 CPU devices with the
    kernel in interpret mode. A (lane, axis) mesh of one lane, as DCP's."""
    if request.param == "tpu":
        if jax.default_backend() != "tpu" or jax.device_count() < N:
            pytest.skip("needs the 8 TPU cores of one host")
    elif jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip("needs 8 CPU devices (--xla_force_host_platform_device_count)")
    return request.param, Mesh(np.array(jax.devices()[:N]).reshape(1, N), (LANE, AXIS))


def _blocks(tokens, seed):
    """[sender, destination, PAD + tokens * LOCAL, V] int16 words, never FILL."""
    rng = np.random.default_rng(seed)
    shape = (N, N, PAD + tokens * LOCAL, V)
    return rng.integers(-32768, FILL, size=shape).astype(np.int16)


def _live_rows(real):
    """`merge_send_layout`'s row count from the lse word row: the word row
    and the real tokens' partial rows."""
    return 1 + real * LOCAL


def _end(block, real):
    """The first block row the live call does not move."""
    return min(-(-(PAD + real * LOCAL) // 8) * 8, block)


def _interpret(mode, mp):
    if mode == "interpret":
        interpret = functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams())
        mp.setattr(a2a.pl, "pallas_call", interpret)


def _call(mode, mesh, body, blocks, *scalars):
    """``body(x, rank, lane, *scalars)`` per core over one program;
    ``scalars`` int32 [1] each, the same on every core."""
    n, _, block, _ = blocks.shape
    spec = P(AXIS)
    sharding = NamedSharding(mesh, spec)
    x = jax.device_put(jnp.asarray(blocks.reshape(n * n * block, V)), sharding)
    rank = jax.device_put(jnp.arange(n, dtype=jnp.int32), sharding)
    lane = jax.device_put(jnp.zeros(n, jnp.int32), sharding)
    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec, spec, spec) + (P(),) * len(scalars),
            out_specs=spec,
            check_vma=False,
        )
    )
    with pytest.MonkeyPatch.context() as mp:
        _interpret(mode, mp)
        out = fn(x, rank, lane, *(jnp.asarray([s], jnp.int32) for s in scalars))
    return jax.tree.map(np.asarray, out)


def _exchange(x, rank, lane, *live_rows):
    """The whole-block call, then one live call per count."""
    common = dict(
        num_cores=N, axis=AXIS, row0=ROW0, rank=rank, lane_axis=LANE, lane=lane
    )
    calls = [a2a.small_all_to_all(x, **common)]
    for rows in live_rows:
        calls.append(
            a2a.small_all_to_all(x, **common, live_rows=rows, live_row0=PAD - 1)
        )
    return tuple(calls)


@pytest.mark.parametrize("rows", [40, 72, 136])
def test_live_sizes_take_one_copy_of_the_count(rows):
    sizes = a2a.live_sizes(jnp.int32(8), rows)
    assert [size for size, _ in sizes] == list(range(8, rows + 1, 8))
    for count in range(8, rows + 1, 8):
        taken = [size for size, live in a2a.live_sizes(jnp.int32(count), rows) if live]
        assert taken == [count], (rows, count, taken)


@pytest.mark.multichip
@pytest.mark.parametrize("tokens", [16, 32])
def test_every_copy_size_matches_the_whole_block(live_mesh, tokens):
    """Every copy size from one tile to the whole block (64 and 128 partial
    rows): live partial rows 0, 8, ..., the bucket, one live call each in
    one program. Each call writes exactly the rows up to its tile row."""
    mode, mesh = live_mesh
    blocks = _blocks(tokens, seed=17 * tokens)
    block = blocks.shape[2]
    parts = range(0, tokens * LOCAL + 1, 8)
    whole, *lives = _call(mode, mesh, _exchange, blocks, *(1 + p for p in parts))
    want = a2a.reference(blocks, ROW0).reshape(N, N, block, V)
    np.testing.assert_array_equal(
        whole.reshape(N, N, block, V)[:, :, ROW0:], want[:, :, ROW0:]
    )
    assert len(lives) == block // 8 - 1
    for p, live in zip(parts, lives):
        end = min(PAD + p, block)
        live = live.reshape(N, N, block, V)
        np.testing.assert_array_equal(live[:, :, ROW0:end], want[:, :, ROW0:end])
        if mode == "interpret":
            assert (live[:, :, end:] == FILL).all(), p


@pytest.mark.multichip
@pytest.mark.parametrize("tokens,real", CASES)
def test_live_rows_match_the_whole_block(live_mesh, tokens, real):
    mode, mesh = live_mesh
    blocks = _blocks(tokens, seed=37 * tokens + real)
    block = blocks.shape[2]
    end = _end(block, real)
    whole, live = _call(mode, mesh, _exchange, blocks, _live_rows(real))
    whole = whole.reshape(N, N, block, V)
    live = live.reshape(N, N, block, V)
    want = a2a.reference(blocks, ROW0).reshape(N, N, block, V)
    np.testing.assert_array_equal(whole[:, :, ROW0:], want[:, :, ROW0:])
    np.testing.assert_array_equal(live[:, :, ROW0:end], want[:, :, ROW0:end])
    if mode == "interpret":
        assert (live[:, :, end:] == FILL).all()


@pytest.mark.multichip
def test_repeated_live_calls_keep_the_semaphores_balanced(live_mesh):
    """Three live counts and the whole block back to back in one program
    (one call per layer); a wait that counts other bytes than its copies
    move would let a later call read rows that have not landed."""
    mode, mesh = live_mesh
    tokens, reals = 32, (20, 32, 1)
    blocks = _blocks(tokens, seed=91)
    block = blocks.shape[2]
    want = a2a.reference(blocks, ROW0).reshape(N, N, block, V)
    whole, *lives = _call(
        mode, mesh, _exchange, blocks, *(_live_rows(r) for r in reals)
    )
    np.testing.assert_array_equal(
        whole.reshape(N, N, block, V)[:, :, ROW0:], want[:, :, ROW0:]
    )
    for real, live in zip(reals, lives):
        end = _end(block, real)
        np.testing.assert_array_equal(
            live.reshape(N, N, block, V)[:, :, ROW0:end], want[:, :, ROW0:end]
        )


def _cp():
    try:
        from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS
        from vllm_torchtpu.layers.adapter import cp_mla_attention
    except ImportError:
        pytest.skip("cp_mla_attention does not import here")
    assert SEND_PAD_ROWS == PAD
    return cp_mla_attention


def _unpack(mode, mesh, blocks, rows, live):
    """`send_small_unpack_local` with the switch at `live`: per core the
    partials as int16 [senders, tl, V] and the lse bits as int32 [senders, tl]."""
    cp = _cp()
    tl = blocks.shape[2] - PAD

    def body(x, rank, lane, rows):
        o, lse = cp.send_small_unpack_local(
            x, rows, rank, lane, world=N, axis_name=AXIS, lane_axis=LANE
        )
        return (
            jax.lax.bitcast_convert_type(o, jnp.int16),
            jax.lax.bitcast_convert_type(lse, jnp.int32),
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(FLAG, "1" if live else "0")
        o, lse = _call(mode, mesh, body, blocks, rows)
    return o.reshape(N, N, tl, V), lse.reshape(N, N, tl)


@pytest.mark.multichip
@pytest.mark.parametrize("tokens,real", CASES)
def test_unpack_is_bit_identical_with_the_switch(live_mesh, tokens, real):
    mode, mesh = live_mesh
    blocks = _blocks(tokens, seed=53 * tokens + real)
    rows = _live_rows(real)
    o_off, lse_off = _unpack(mode, mesh, blocks, rows, live=False)
    o_on, lse_on = _unpack(mode, mesh, blocks, rows, live=True)
    np.testing.assert_array_equal(o_on, o_off)
    np.testing.assert_array_equal(lse_on, lse_off)
    assert (o_off[:, :, real * LOCAL :] == 0).all()


@pytest.mark.multichip
def test_a_copy_one_tile_short_fails(live_mesh):
    """Positive control: the live call told one tile fewer leaves real
    partial rows unsent, and the comparison with the switch off has to see
    it."""
    mode, mesh = live_mesh
    cp = _cp()
    tokens, real = 16, 12
    blocks = _blocks(tokens, seed=7)
    block = blocks.shape[2]
    rows = _live_rows(real)
    o_off, _ = _unpack(mode, mesh, blocks, rows, live=False)

    def short(x, rank, lane, rows):
        recv = a2a.small_all_to_all(
            x,
            num_cores=N,
            axis=AXIS,
            row0=ROW0,
            rank=rank,
            lane_axis=LANE,
            lane=lane,
            live_rows=rows - 8,
            live_row0=PAD - 1,
        ).reshape(N, block, V)
        o, _ = cp._mask_unpack(recv, rows, block - PAD)
        return jax.lax.bitcast_convert_type(o, jnp.int16)

    o_short = _call(mode, mesh, short, blocks, rows).reshape(o_off.shape)
    sent, unsent = slice(0, real * LOCAL - 8), slice(real * LOCAL - 8, real * LOCAL)
    np.testing.assert_array_equal(o_short[:, :, sent], o_off[:, :, sent])
    assert (o_short[:, :, unsent] != o_off[:, :, unsent]).any()
