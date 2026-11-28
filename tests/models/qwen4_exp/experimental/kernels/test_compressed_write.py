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
"""Run-based compressed-key writer (Pallas interpret mode) against a NumPy
read-modify-write reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.models.qwen4_exp.experimental.kernels import compressed_write

WIDTH = 128
ROWS = 64


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )


def _halves(view):
    """The view's rows as ``(rows, 2, width)`` BF16, with the kernel's bitcast."""
    if view.dtype == jnp.bfloat16:
        return np.asarray(view.astype(jnp.float32))

    def kernel(src, dst):
        dst[...] = src.bitcast(jnp.bfloat16)[...]

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((view.shape[0], 2, WIDTH), jnp.bfloat16),
        interpret=True,
    )(view)
    return np.asarray(out.astype(jnp.float32))


def _pool(dtype, rng):
    if dtype == jnp.bfloat16:
        data = rng.standard_normal((ROWS, 2, WIDTH)).astype(np.float32)
        return jnp.asarray(data, jnp.bfloat16)
    # Random bytes, kept finite in BF16 (exponent byte below 0x7f).
    raw = rng.integers(0, 256, (ROWS, 4, WIDTH), dtype=np.uint8)
    raw[:, 1::2] &= 0x3F
    return jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)


# (destination pair, halves) of each writer, in token order; None = no writer.
CASES = {
    "no_writers": [],
    "one_full_run": [(p, (1, 1)) for p in range(8, 24)],
    "fragmented_runs": [(3, (1, 1)), (4, (1, 1)), (9, (1, 1)), (30, (1, 1))]
    + [(p, (1, 1)) for p in range(40, 47)],
    "partial_edges": [(5, (0, 1)), (6, (1, 1)), (7, (1, 1)), (8, (1, 0))]
    + [(20, (1, 0)), (33, (0, 1))],
    "unordered": [(50, (1, 1)), (2, (0, 1)), (51, (1, 1)), (1, (1, 0))],
}


def _step(writers, rng, tokens=48):
    """Scatter the writers over ``tokens`` positions; the rest are non-writers
    with random pairs and masks the kernel must ignore."""
    positions = np.sort(rng.choice(tokens, size=len(writers), replace=False))
    pair = rng.integers(0, ROWS, tokens).astype(np.int32)
    write = np.zeros(tokens, bool)
    mask = rng.integers(0, 2, (tokens, 2)).astype(bool)
    for position, (dest, halves) in zip(positions, writers):
        pair[position] = dest
        write[position] = True
        mask[position] = halves
    values = rng.standard_normal((tokens, 2, WIDTH)).astype(np.float32)
    return pair, write, mask, values


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn], ids=str)
@pytest.mark.parametrize("case", sorted(CASES))
def test_write_runs_matches_reference(interpret, case, dtype):
    rng = np.random.default_rng(len(case))
    view = _pool(dtype, rng)
    pair, write, mask, values = _step(CASES[case], rng)
    values_bf = jnp.asarray(values, jnp.bfloat16)

    expected = _halves(view).copy()
    new = np.asarray(values_bf.astype(jnp.float32))
    for t in np.flatnonzero(write):
        for h in range(2):
            if mask[t, h]:
                expected[pair[t], h] = new[t, h]

    out = compressed_write.write_runs(
        view,
        jnp.asarray(pair),
        jnp.asarray(write),
        jnp.asarray(mask),
        values_bf,
        requests=4,
        ratio=4,
    )
    assert out.shape == view.shape and out.dtype == view.dtype
    np.testing.assert_array_equal(_halves(out), expected)


def test_writer_capacity_bounds_pairs():
    # A request of n tokens touches at most n // (2 * ratio) + 2 pairs.
    assert compressed_write.writer_capacity(8192, 1, 4) >= 8192 // 8 + 2
    assert compressed_write.writer_capacity(1, 32, 4) >= 32 * 2
    assert compressed_write.writer_capacity(100, 3, 4) % 8 == 0


@pytest.mark.parametrize(
    "change, message",
    [
        (dict(view=((4, 4, WIDTH), "bfloat16")), "one BF16 pair"),
        (dict(view=((4, 2 * WIDTH), "bfloat16")), "packing, width"),
        (dict(values=((8, 2, WIDTH), "float32")), "values must be"),
        (dict(values=((8, 2, 64), "bfloat16")), "values must be"),
        (dict(mask=((8,), "bool")), "mask"),
        (dict(ratio=0), "positive"),
        (dict(requests=0), "positive"),
    ],
)
def test_write_runs_rejects_bad_inputs(change, message):
    args = dict(
        view=((4, 2, WIDTH), "bfloat16"),
        pair=((8,), "int32"),
        write=((8,), "bool"),
        mask=((8, 2), "bool"),
        values=((8, 2, WIDTH), "bfloat16"),
        requests=1,
        ratio=4,
    )
    args.update(change)
    args = {
        k: jnp.zeros(v[0], v[1]) if isinstance(v, tuple) else v for k, v in args.items()
    }
    with pytest.raises(ValueError, match=message):
        compressed_write.write_runs(**args)


def test_write_runs_rejects_empty_step():
    with pytest.raises(ValueError, match="positive"):
        compressed_write.write_runs(
            jnp.zeros((4, 2, WIDTH), jnp.bfloat16),
            jnp.zeros((0,), jnp.int32),
            jnp.zeros((0,), bool),
            jnp.zeros((0, 2), bool),
            jnp.zeros((0, 2, WIDTH), jnp.bfloat16),
            1,
            4,
        )
