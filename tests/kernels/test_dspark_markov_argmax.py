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
"""The DSpark Markov chain kernel (additional_config dspark_markov_kernel)
against the step-by-step chain.

Runs in TPU interpret mode over 8 simulated CPU devices. Integer logits, +-1
markov_w1 entries and quarter markov_w2 entries keep every bias and score
exact in bf16 and f32, so the numpy chain is the exact answer and the kernel
must match it bitwise: per-step exchange, rank order, lowest-id ties, pad
columns past the shard, and the chain of ``prev`` ids.
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

N, STEPS, DIM, SHARD = 8, 3, 256, 1000


def _inputs(rows, seed, hi=8):
    rng = np.random.default_rng(seed)
    vocab = N * SHARD
    logits = rng.integers(-hi, hi + 1, (STEPS, rows, vocab)).astype(np.float32)
    w1 = np.zeros((vocab, DIM), np.float32)
    cols = rng.integers(0, DIM, (vocab, 4))
    np.put_along_axis(w1, cols, rng.choice([-1.0, 1.0], (vocab, 4)), axis=1)
    w2 = rng.integers(-1, 2, (vocab, DIM)).astype(np.float32) / 4
    anchors = rng.integers(0, vocab, rows)
    return logits, w1, w2, anchors


def _reference(logits, w1, w2, anchors):
    prev, out = anchors, []
    for i in range(STEPS):
        prev = (logits[i] + w1[prev] @ w2.T).argmax(-1)  # first index on ties
        out.append(prev)
    return np.stack(out, 1)


def _run(logits, w1, w2, anchors, perm=tuple(range(N))):
    """Mesh position d holds vocab shard perm[d]."""
    devs = jax.devices()
    if len(devs) < N:
        pytest.skip("needs 8 (simulated) devices")
    mesh = Mesh(np.array(devs[:N]), ("x",))
    rows = anchors.shape[0]
    width = mk.padded_width(SHARD)
    bf = lambda x: jnp.asarray(x, jnp.float32).astype(jnp.bfloat16)  # noqa: E731
    words = jax.lax.bitcast_convert_type(
        bf(w1).reshape(w1.shape[0], DIM // 2, 2), jnp.int32
    )
    scal, lg, even, odd = [], [], [], []
    for d in range(N):
        lo = perm[d] * SHARD
        scal.append(np.concatenate([[d, lo, SHARD], anchors]).astype(np.int32))
        # Pad columns hold large logits: only the kernel's column mask hides them.
        lg.append(
            np.pad(
                logits[:, :, lo : lo + SHARD],
                ((0, 0), (0, 0), (0, width - SHARD)),
                constant_values=100.0,
            )
        )
        w2r = np.pad(w2[lo : lo + SHARD], ((0, width - SHARD), (0, 0)))
        e, o = mk.split_halves(bf(w2r))
        even.append(e)
        odd.append(o)

    def body(scal, lg, words, even, odd):
        return mk.markov_argmax(
            scal[0], lg[0], words[0], even[0], odd[0], N,
            interpret=pltpu.InterpretParams(),
        )[None]

    f = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(P("x"),) * 5, out_specs=P("x"), check_vma=False
        )
    )

    def put(a):
        return jax.device_put(a, NamedSharding(mesh, P("x")))

    out = f(
        put(jnp.stack([jnp.asarray(s) for s in scal])),
        put(jnp.stack([bf(x) for x in lg])),
        put(jnp.broadcast_to(words[None], (N, *words.shape))),
        put(jnp.stack(even)),
        put(jnp.stack(odd)),
    )
    out = np.asarray(out)
    assert out.shape == (N, rows, mk.LANES)
    assert (out == out[0:1]).all(), "ranks disagree"
    return out[0, :, :STEPS]


@pytest.mark.parametrize("rows,hi", [(8, 8), (3, 1)])
def test_markov_chain_matches_reference(rows, hi):
    logits, w1, w2, anchors = _inputs(rows, seed=rows, hi=hi)
    np.testing.assert_array_equal(
        _run(logits, w1, w2, anchors), _reference(logits, w1, w2, anchors)
    )


def test_markov_chain_ties_go_to_the_lowest_id():
    """Equal maxima on two ranks and twice within a rank: the lowest global id
    wins, with shards in a mesh order other than vocab order."""
    rows = 2
    logits, w1, w2, anchors = _inputs(rows, seed=7)
    logits = np.minimum(logits, 0.0)
    w2[:] = 0.0  # no bias: the planted maxima stay equal
    logits[:, :, 5 * SHARD + 2] = 5.0
    logits[:, :, 3 * SHARD + 500] = 5.0
    logits[:, :, 3 * SHARD + 700] = 5.0
    got = _run(logits, w1, w2, anchors, perm=(5, 2, 7, 3, 0, 6, 1, 4))
    np.testing.assert_array_equal(got, np.full((rows, STEPS), 3 * SHARD + 500))
    np.testing.assert_array_equal(got, _reference(logits, w1, w2, anchors))


def test_fits_bounds_vmem():
    width = mk.padded_width(200_064 // 8)
    assert width % mk.CHUNK == 0 and width - 200_064 // 8 < mk.CHUNK
    assert mk.fits(8, 7, 8, width, 64)
    assert not mk.fits(8, 7, 4096, width, 64)
    assert not mk.fits(8, mk.LANES + 1, 8, width, 64)
