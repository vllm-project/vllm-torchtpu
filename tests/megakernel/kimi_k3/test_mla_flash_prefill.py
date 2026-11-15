# SPDX-License-Identifier: Apache-2.0
"""Packed MLA flash prefill against a float32 reference, in interpret mode."""

import functools

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3 import config
from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels

pytestmark = pytest.mark.cpu_test

# (offset, length) per sequence; rows 1212..1279 are padding.
WINDOW = ((0, 512), (512, 700))
# (base, cached) per sequence in arena layer 1 of 2.
PREFIXES = ((0, 300), (320, 520))
# Key block 5 at tile 128: streamed after sequence 1's first block, window and prefix.
STALE_ROWS = slice(640, 768)


def reference(q, k, v, arena_k, arena_v, sequences, prefixes):
    rows = np.arange(q.shape[1])
    owner = np.full(rows.shape, -1)
    for s, (offset, length) in enumerate(sequences):
        owner[offset : offset + length] = s
    real = owner[:, None] >= 0
    mask = real & (owner[:, None] == owner) & (rows <= rows[:, None])
    if prefixes is not None:
        # Padding rows (owner -1) read the last entry; `real` masks them.
        base, cached = np.asarray(prefixes)[owner].T[..., None]
        cols = np.arange(arena_k.shape[2])
        mask = np.concatenate([real & (cols >= base) & (cols < base + cached), mask], 1)
        k = np.concatenate([arena_k[1], k], axis=1)
        v = np.concatenate([arena_v[1].swapaxes(1, 2), v], axis=1)
    scores = np.where(mask, q @ k.swapaxes(1, 2), -np.inf)
    top = scores.max(axis=-1, keepdims=True)
    p = np.exp2(scores - np.where(np.isfinite(top), top, 0))
    total = p.sum(axis=-1, keepdims=True)
    return (p @ v) / np.where(total > 0, total, 1)


@pytest.mark.parametrize(
    "tokens,sequences,prefixes,scaled",
    [
        pytest.param(1280, WINDOW, None, None, id="streamed"),
        pytest.param(1280, WINDOW, None, "window", id="stale-rerun"),
        pytest.param(1280, WINDOW, PREFIXES, None, id="prefixed"),
        pytest.param(1280, WINDOW, PREFIXES, "arena", id="prefixed-rerun"),
        # T % 1024 == 0: 1024-row query tiles (two fragments), 512-row key blocks.
        pytest.param(3072, ((0, 256), (256, 2816)), None, None, id="q-tile-1024"),
    ],
)
def test_flash_prefill_matches_float32_reference(
    monkeypatch, tokens, sequences, prefixes, scaled
):
    monkeypatch.setattr(
        pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    # Pinned: config.active() would install the default config process-wide.
    tiles = config.KernelConfig(mla_flash_qtile=1024, mla_flash_ktile=2048)
    monkeypatch.setattr(kernels, "_config", lambda: tiles)
    rng = np.random.default_rng(2825)
    normal = functools.partial(rng.standard_normal, dtype=np.float32)
    query = normal((3, tokens, 256)) * kernels.SCALE_LOG2
    key, value = normal((3, tokens, 256)), normal((3, tokens, 128))
    arena_k, arena_v = normal((2, 3, 1024, 256)), normal((2, 3, 128, 1024))
    # About 100x the scores: a stale gap far above STALE_LIMIT forces a rerun.
    if scaled == "window":
        key[:, STALE_ROWS] *= 100
    elif scaled == "arena":
        arena_k[1, :, STALE_ROWS] *= 100
    inputs = [
        jnp.asarray(x, jnp.bfloat16) for x in (query, key, value, arena_k, arena_v)
    ]
    offsets, lengths = np.asarray(sequences).T
    segments = [offsets, lengths, np.zeros_like(offsets)]
    if prefixes is not None:
        segments += list(np.asarray(prefixes).T)
    actual = kernels.mla_flash_prefill_packed(
        *inputs[:3],
        jnp.array([offsets[-1] + lengths[-1]], jnp.int32),
        jnp.asarray(np.concatenate(segments), jnp.int32),
        count=len(sequences),
        tile=128,
        arena=None if prefixes is None else tuple(inputs[3:]),
        layer=1,
    )
    expected = reference(
        *(np.asarray(x, np.float32) for x in inputs), sequences, prefixes
    )
    np.testing.assert_allclose(np.asarray(actual, np.float32), expected, atol=0.03)
