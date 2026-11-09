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
"""tile_scatter equals XLA's drop-mode scatter, duplicates included."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.tile_scatter import MAX_UPDATES, tile_scatter


@pytest.fixture(autouse=True)
def _tpu_only(device):
    if device.type != "tpu":
        pytest.skip("Pallas TPU kernel")


def _reference(table, slots, values):
    size = table.shape[0]
    return table.at[jnp.where(slots >= 0, slots, size)].set(values, mode="drop")


def _decode_slots(rng, size, requests=48, tokens=6):
    starts = rng.integers(0, size - tokens, requests)
    slots = np.concatenate([np.arange(s, s + tokens) for s in starts])
    slots[-tokens:] = -1  # padded request
    return slots


def _cases(rng, size):
    decode = _decode_slots(rng, size)
    decode[6:12] = np.arange(size - 6, size)  # the partial last tile, if any
    decode[12:18] = decode[18:24]  # duplicate slots, different values
    decode[30] = size + 5  # out of range: dropped
    dups = rng.integers(0, size, 200)
    dups[::3] = dups[1::3][: len(dups[::3])]
    narrow = rng.integers(0, min(size, 300), 64)  # many updates per tile
    yield "decode", decode
    yield "random_dups", dups
    yield "narrow", narrow
    yield "all_dropped", np.full(17, -1)
    yield "single", np.array([size - 1])
    yield "over_limit", rng.integers(-1, size, MAX_UPDATES + 5)


@pytest.mark.parametrize("size", [30198976, 4096 + 64, 4096, 128])
def test_matches_xla_scatter(size):
    rng = np.random.default_rng(size)
    table = rng.integers(0, 1 << 30, size).astype(np.int32)
    fn = jax.jit(tile_scatter)
    for name, slots in _cases(rng, size):
        slots = jnp.asarray(slots, jnp.int32)
        values = jnp.asarray(rng.integers(0, 1 << 30, slots.shape[0]), jnp.int32)
        ref = np.asarray(_reference(jnp.asarray(table), slots, values))
        out = np.asarray(fn(jnp.asarray(table), slots, values))
        assert np.array_equal(out, ref), name


def test_last_duplicate_wins():
    table = jnp.zeros(4096 + 64, jnp.int32)
    slots = jnp.asarray([5, 9, 5, 4100, 4100, 5], jnp.int32)
    values = jnp.asarray([1, 2, 3, 4, 5, 6], jnp.int32)
    out = np.asarray(jax.jit(tile_scatter)(table, slots, values))
    assert (out[5], out[9], out[4100]) == (6, 2, 5)
    assert np.array_equal(out, np.asarray(_reference(table, slots, values)))
