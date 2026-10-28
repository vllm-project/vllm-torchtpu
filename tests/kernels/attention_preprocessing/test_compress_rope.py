# SPDX-License-Identifier: Apache-2.0
"""In-kernel ring reads/writes and chunked weight copy reproduce the pipelined
compressor plus epic's ring update.

The kernel copies its weight and ring records by DMA, which interpret mode
does not discharge, so the value checks run on TPU only.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.attention_preprocessing import (
    compress_rope,
    compress_serial,
    rope_gather,
)
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    CompressConfig,
)

_CAPACITY = 8  # ring rows per request (K5 speculative decoding)
_RECORD_WORDS = 8 * 128  # one FP32 [1024] record = 8 pool rows


def _inputs(tokens, ratio, cache_rows=4096, hidden=5120, seed=7, banks=1):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(tokens, hidden)) * 0.5, jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(hidden, ratio * 512)) * 0.02, jnp.bfloat16)
    gamma = jnp.asarray(rng.normal(size=(512,)), jnp.bfloat16)
    # Two requests: the first starts mid-group, the second inside a later tile;
    # the last 5 rows are padding.
    split = tokens // 2 + 3
    starts, ends, blocks = (0, split), (split, tokens - 5), (3, 6)
    positions = np.concatenate((np.arange(11, 11 + split), np.arange(tokens - split)))
    has_prev = np.ones((tokens, 1), np.int32)
    valid = np.ones((tokens, 1), np.int32)
    if ratio == 2:
        valid[:, 0] = positions % 2 == 1
    valid[-5:] = 0
    angles = rng.uniform(-3, 3, size=(cache_rows, 32))
    cache = jnp.asarray(
        np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32
    )
    # Per-step RoPE rows at each token's group start, as the caller gathers them.
    group_starts = jnp.asarray(positions // ratio * ratio, jnp.int32)
    rows = rope_gather.gather_rows(cache, group_starts)
    ring_write = np.full(tokens, -1, np.int32)
    ring_read = np.full(tokens, -1, np.int32)
    for start, end, block in zip(starts, ends, blocks, strict=True):
        has_prev[start] = 0
        # epic: the first row pools with the stored previous position; only
        # the request's last `capacity` rows are published.
        ring_read[start] = block * _CAPACITY + (positions[start] - 1) % _CAPACITY
        for r in range(max(start, end - _CAPACITY), end):
            ring_write[r] = block * _CAPACITY + positions[r] % _CAPACITY
    records = 10 * _CAPACITY
    stored = rng.normal(size=(records, ratio * 512)).astype(np.float32)
    pool = stored.view(np.int32).reshape(-1, 128) if ratio == 2 else None
    state = None
    if ratio == 2:
        state = jnp.asarray(pool)
        if banks == 2:  # bank boundary inside request 1's ring block
            cut = (3 * _CAPACITY + 5) * 8
            state = (jnp.asarray(pool[:cut]), jnp.asarray(pool[cut:]))
    return dict(
        x=x,
        weight=weight,
        gamma=gamma,
        rows=rows,
        has_prev=jnp.asarray(has_prev),
        valid=jnp.asarray(valid),
        state=state,
        ring_write=jnp.asarray(ring_write) if ratio == 2 else None,
        ring_read=jnp.asarray(ring_read) if ratio == 2 else None,
        stored=stored,
        pool=pool,
    )


def _run(inputs, ratio, config=None):
    run = functools.partial(
        compress_rope.fused_compress_rope, ratio=ratio, config=config
    )
    i = inputs
    return jax.jit(run)(
        i["x"],
        i["weight"],
        i["gamma"],
        i["rows"],
        i["has_prev"],
        i["valid"],
        i["state"],
        i["ring_write"],
        i["ring_read"],
    )


def _pool(state):
    if isinstance(state, tuple):
        state = jnp.concatenate(state)
    return np.asarray(state)


def _tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")


@pytest.mark.parametrize(
    ("tokens", "block"),
    [(2048, 256), (6144, 256), (288, 144), (240, 240), (96, 96), (128, 128)],
)
def test_default_config_tiles_every_token(tokens, block):
    assert compress_rope.default_config(tokens).token_block == block


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="Invalid CompressConfig"):
        compress_rope.check_config(
            CompressConfig(token_block=200, weight_chunk=100), 2048, 1024
        )


def test_ring_arguments_are_checked():
    tokens = 32
    x = jnp.zeros((tokens, 256), jnp.bfloat16)
    args = (
        x,
        jnp.zeros((256, 1024), jnp.bfloat16),
        jnp.zeros((512,), jnp.bfloat16),
        jnp.zeros((tokens, 64), jnp.float32),
        jnp.ones((tokens, 1), jnp.int32),
        jnp.ones((tokens, 1), jnp.int32),
    )
    with pytest.raises(ValueError, match="state pool"):
        compress_rope.fused_compress_rope(*args, ratio=2)
    records = jnp.full((tokens,), -1, jnp.int32)
    with pytest.raises(ValueError, match="pool banks"):
        compress_rope.fused_compress_rope(
            *args, jnp.zeros((12, 128), jnp.int32), records, records, ratio=2
        )
    with pytest.raises(ValueError, match="no compressor state"):
        compress_rope.fused_compress_rope(
            x,
            jnp.zeros((256, 512), jnp.bfloat16),
            *args[2:],
            jnp.zeros((8, 128), jnp.int32),
            records,
            records,
            ratio=1,
        )


@pytest.mark.parametrize(
    ("ratio", "banks"), [(1, 1), (2, 1), (2, 2)], ids=["r1", "r2", "r2-2bank"]
)
# The pipelined reference needs 256-row multiples; 288 etc. are covered by
# test_block_sizes_do_not_change_values against the kernel itself.
@pytest.mark.parametrize("tokens", [256, 512, 2048])
def test_matches_pipelined_compressor_and_epic_ring(tokens, ratio, banks):
    _tpu()
    i = _inputs(tokens, ratio, banks=banks)
    # Reference: the caller-gathered ring row of every has_prev == 0 row.
    ring_prev = np.zeros((tokens, ratio * 512), np.float32)
    if ratio == 2:
        reads = np.asarray(i["ring_read"])
        ring_prev[reads >= 0] = i["stored"][reads[reads >= 0]]
    cos, sin = i["rows"][:, :32], i["rows"][:, 32:]
    score, latent, words, scales = jax.jit(
        functools.partial(compress_serial.fused_compress_pipelined, ratio=ratio)
    )(
        i["x"],
        i["weight"],
        i["gamma"],
        i["has_prev"],
        jnp.asarray(ring_prev),
        i["valid"],
        cos,
        sin,
    )
    actual = _run(i, ratio)
    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(latent))
    # Invalid rows are never published; the reference may encode their zero
    # latents with -0.0 RoPE lanes, the kernel with +0.0.
    rows = np.asarray(i["valid"][:, 0]) != 0
    np.testing.assert_array_equal(np.asarray(actual[1])[rows], np.asarray(words)[rows])
    assert not np.asarray(actual[1])[~rows].any()
    np.testing.assert_array_equal(np.asarray(actual[2]), np.asarray(scales))
    if ratio == 1:
        assert actual[3] is None
        return
    # epic writes the FP32 projection of each published row; nothing else.
    expected = i["stored"].copy()
    writes = np.asarray(i["ring_write"])
    expected[writes[writes >= 0]] = np.asarray(score)[writes >= 0]
    assert isinstance(actual[3], tuple) == (banks == 2)
    np.testing.assert_array_equal(
        _pool(actual[3]), expected.view(np.int32).reshape(-1, 128)
    )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize(
    ("tokens", "config"),
    [
        (288, CompressConfig(token_block=96)),
        (288, CompressConfig(token_block=288)),
        (2048, CompressConfig(token_block=128)),
        (100, None),
    ],
)
def test_block_sizes_do_not_change_values(tokens, config, ratio):
    """Token tiling is bitwise; weight_chunk is not (it sets matmul N)."""
    _tpu()
    base = _run(_inputs(tokens, ratio), ratio)
    if config is None:  # padded token count against an explicit 16-aligned run
        i = _inputs(tokens, ratio)
        pad = 112 - tokens
        for key, value in (("has_prev", 1), ("valid", 0)):
            i[key] = jnp.pad(i[key], ((0, pad), (0, 0)), constant_values=value)
        for key in ("x", "rows"):
            i[key] = jnp.pad(i[key], ((0, pad), (0, 0)))
        if ratio == 2:
            for key in ("ring_write", "ring_read"):
                i[key] = jnp.pad(i[key], (0, pad), constant_values=-1)
        out = _run(i, ratio)
        other = (*(o[:tokens] for o in out[:3]), out[3])
    else:
        other = _run(_inputs(tokens, ratio), ratio, config)
    for a, b in zip(base[:3], other[:3], strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    if ratio == 2:
        np.testing.assert_array_equal(_pool(base[3]), _pool(other[3]))
