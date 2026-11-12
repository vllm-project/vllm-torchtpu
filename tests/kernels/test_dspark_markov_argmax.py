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
"""The fused DSpark Markov kernel against the step-by-step reference.

Runs in TPU interpret mode over 8 simulated CPU devices. Integer inputs keep
every bias and score exact in bf16, so the reference is plain numpy and the
test checks the kernel's control: per-step exchange, rank order, first-index
ties, pad columns past the shard width, and the chain of ``prev`` ids.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import markov_argmax as mk  # noqa: E402

N, K, RD = 8, 3, 256


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32))


def _reference(lg, w1, w2, p0, vs):
    """[rows, K] ids: the model's per-step arithmetic on exact integer data."""
    prev = p0.copy()
    out = []
    for i in range(K):
        e = w1[prev]
        # Exact f32 sums (small multiples of 1/32), rounded to bf16 as the
        # model rounds the bias and the sum.
        bias = _bf16(np.einsum("rk,nvk->nrv", e, w2.reshape(N, vs, RD)))
        scores = _bf16(lg[:, :, i, :] + bias)
        local_max = scores.max(-1)  # [N, rows]
        local_idx = scores.argmax(-1) + np.arange(N)[:, None] * vs
        best = local_max.argmax(0)  # first rank of the largest max
        prev = local_idx[best, np.arange(lg.shape[1])]
        out.append(prev)
    return np.stack(out, 1)


def _inputs(rows, vs, seed, tie_heavy):
    rng = np.random.default_rng(seed)
    vocab = N * vs
    hi = 2 if tie_heavy else 6
    lg = rng.integers(-hi, hi + 1, size=(N, rows, K, vs)).astype(np.float32)
    w1 = (rng.integers(-2, 3, size=(vocab, RD)) / 4).astype(np.float32)
    w2 = (rng.integers(-2, 3, size=(vocab, RD)) / 8).astype(np.float32)
    p0 = rng.integers(0, vocab, size=(rows,)).astype(np.int64)
    return lg, w1, w2, p0


def _run(lg, w1, w2, p0, vs):
    devs = jax.devices()
    if len(devs) < N:
        pytest.skip("needs 8 (simulated) devices")
    mesh = Mesh(np.array(devs[:N]), ("x",))
    vp = mk.padded_vocab(vs)
    w2r = jnp.asarray(w2, jnp.bfloat16).reshape(N, vs, RD)
    # Pad rows hold large values: only the kernel's column mask may hide them.
    w2p = jnp.pad(w2r, ((0, 0), (0, vp - vs), (0, 0)), constant_values=8.0)

    def body(lg, p0, w1, w2p, rank):
        return mk.markov_argmax(
            lg[0], p0[0], w1, w2p[0], rank[0], N, vs,
            interpret=pltpu.InterpretParams(),
        )[None]

    f = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("x"), P("x"), P(), P("x"), P("x")),
            out_specs=P("x"),
            check_vma=False,
        )
    )

    def put(a, s):
        return jax.device_put(a, NamedSharding(mesh, s))

    rows = lg.shape[1]
    out = f(
        put(jnp.asarray(lg, jnp.bfloat16), P("x")),
        put(jnp.tile(jnp.asarray(p0, jnp.int32)[None], (N, 1)), P("x")),
        put(jnp.asarray(w1, jnp.bfloat16), P()),
        put(w2p, P("x")),
        put(jnp.arange(N, dtype=jnp.int32).reshape(N, 1), P("x")),
    )
    out = np.asarray(out)
    assert out.shape == (N, rows, K)
    return out


@pytest.mark.parametrize("rows,tie_heavy", [(3, False), (2, True)])
def test_markov_argmax_matches_reference(rows, tie_heavy):
    vs = 4000  # padded to 7168: the last tile is partly past the shard
    lg, w1, w2, p0 = _inputs(rows, vs, seed=rows, tie_heavy=tie_heavy)
    out = _run(lg, w1, w2, p0, vs)
    ref = _reference(lg, w1, w2, p0, vs)
    assert (out == out[0:1]).all(), "ranks disagree"
    np.testing.assert_array_equal(out[0], ref)


def test_markov_argmax_padded_vocab():
    assert mk.padded_vocab(25008) % mk.TILE_COLS == 0
    assert mk.padded_vocab(25008) - 25008 < mk.TILE_COLS
    assert mk.padded_rows(1) == 16 and mk.padded_rows(17) == 32


def test_markov_argmax_ties_go_to_the_lowest_id():
    """Equal maxima on two ranks and twice within a rank, including one past
    the first bias tile: the lowest global id wins, as torch.argmax picks."""
    vs, rows = 4000, 2
    lg, w1, w2, p0 = _inputs(rows, vs, seed=7, tie_heavy=True)
    lg = np.minimum(lg, 0.0)
    w2[:] = 0.0  # no bias: the planted maxima stay equal
    lg[5, :, :, 2] = 5.0
    lg[3, :, :, 3500] = 5.0
    lg[3, :, :, 3700] = 5.0
    out = _run(lg, w1, w2, p0, vs)
    np.testing.assert_array_equal(out[0], np.full((rows, K), 3 * vs + 3500))
    np.testing.assert_array_equal(out[0], _reference(lg, w1, w2, p0, vs))
