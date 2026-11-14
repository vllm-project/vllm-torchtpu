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
"""DSpark Markov chain kernel vs the per-step chain it replaces, in TPU
interpret mode on 8 host devices (remote DMAs and semaphores included); run
with XLA_FLAGS=--xla_force_host_platform_device_count=8. It does not gate
Mosaic lowering."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives import markov_argmax as mk

N, STEPS, DIM, SHARD = 8, 7, 256, 1000
BF16 = ml_dtypes.bfloat16

pytestmark = pytest.mark.skipif(
    len(jax.devices("cpu")) < N, reason="needs 8 host devices (XLA_FLAGS)"
)


def _reference(logits, w1, w2, anchors):
    """The chain in f32 (bf16 inputs), first index of the max."""
    prev, out = anchors, []
    for step_logits in logits:
        bias = w1[prev].astype(np.float32) @ w2.astype(np.float32).T
        prev = (step_logits.astype(np.float32) + bias).argmax(-1)
        out.append(prev)
    return np.stack(out, 1)


@pytest.mark.parametrize("rows", [8, 24])
def test_matches_per_step_chain(rows: int) -> None:
    rng = np.random.default_rng(rows)
    vocab, width = N * SHARD, mk.padded_width(SHARD)
    logits = rng.standard_normal((STEPS, rows, vocab)).astype(BF16)
    logits[0, 0, 5] = logits[0, 0, 5 + SHARD] = 30.0  # a tie across shards
    w1 = (rng.standard_normal((vocab, DIM)) * 0.5).astype(BF16)
    w2 = (rng.standard_normal((vocab, DIM)) * 0.05).astype(BF16)
    anchors = rng.integers(0, vocab, rows)

    def shard(r):
        lo = r * SHARD
        lg = np.zeros((STEPS, rows, width), BF16)
        lg[:, :, :SHARD] = logits[:, :, lo : lo + SHARD]
        w2s = np.zeros((width, DIM), BF16)
        w2s[:SHARD] = w2[lo : lo + SHARD]
        scal = np.concatenate([[r, lo, SHARD], anchors]).astype(np.int32)
        return scal, lg, w2s

    scal, lg, w2s = (np.concatenate(p) for p in zip(*(shard(r) for r in range(N))))
    words = np.concatenate([np.ascontiguousarray(w1).view(np.int32)] * N)
    mesh = Mesh(np.array(jax.devices("cpu")[:N]), ("x",))

    def body(scal, lg, words, w2s):
        even, odd = mk.split_halves(w2s)
        return mk.markov_argmax(
            scal, lg, words, even, odd, N, interpret=pltpu.InterpretParams()
        )

    run = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(P("x"),) * 4, out_specs=P("x"), check_vma=False
        )
    )
    out = np.asarray(run(*map(jnp.asarray, (scal, lg, words, w2s))))
    want = _reference(logits, w1, w2, anchors)
    for r in range(N):
        np.testing.assert_array_equal(out.reshape(N, rows, -1)[r, :, :STEPS], want)
