# SPDX-License-Identifier: Apache-2.0
"""Per-step RoPE rows reproduce the precomputed-coefficient kernel.

The kernels view their HBM output as u32 head pairs, which interpret mode does
not discharge, so value checks run on TPU only.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.attention_preprocessing import (
    qb_overlap3,
    qb_rope,
    reference,
    rope_gather,
)


def _inputs(tokens, heads, positions, cache_rows=4096, rank=1280, head_dim=512):
    rng = np.random.default_rng(7)
    qr = jnp.asarray(rng.normal(size=(tokens, rank)), jnp.bfloat16)
    qr = qr.at[0].set(0).at[1].multiply(1e-6)
    weight = jnp.asarray(rng.normal(size=(rank, heads * head_dim)), jnp.float8_e4m3fn)
    scale = jnp.asarray(rng.uniform(0.01, 0.1, heads * head_dim), jnp.float32)
    angles = rng.uniform(-3, 3, size=(cache_rows, 32))
    cache = jnp.asarray(
        np.concatenate((np.cos(angles), np.sin(angles)), 1), jnp.float32
    )
    rows = rope_gather.gather_rows(cache, jnp.asarray(positions, jnp.int32))
    return qr, weight, scale, rows


def _split(rows):
    half = rows.shape[1] // 2
    return rows[:, :half], rows[:, half:]


@pytest.mark.parametrize(
    ("tokens", "heads", "kind"),
    [(512, 32, "contiguous"), (1024, 16, "jumps"), (2048, 64, "clamped")],
)
def test_matches_precomputed_coefficients(tokens, heads, kind):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")
    if kind == "contiguous":
        positions = np.arange(tokens) + 17
    elif kind == "jumps":
        positions = np.concatenate(
            [np.arange(tokens // 2) + 3000, np.arange(tokens // 2)]
        )
    else:  # padding rows past the table end are clamped, like negative rows
        positions = np.arange(tokens) + 4096 - tokens // 2
        positions[:4] = -1
    qr, weight, scale, rows = _inputs(tokens, heads, positions)
    cos, sin = _split(rows)
    expected = jax.jit(qb_overlap3.fused_qb_rope_overlap3)(qr, weight, scale, cos, sin)
    actual = jax.jit(qb_rope.fused_qb_rope)(qr, weight, scale, rows)
    assert actual.shape == (tokens, heads, 512) and actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    )


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (2048, (512, 16)),
        (6144, (512, 16)),
        (288, (288, 16)),
        (256, (256, 16)),  # T=240 pads to 256: FP8 qr tiles are 32 rows
        (96, (96, 16)),
    ],
)
def test_default_config_tiles_every_token(tokens, expected):
    config = qb_rope.default_config(tokens, 1280, 64)
    assert (config.token_block, config.head_block) == expected
    assert config.quant_fragment % 32 == 0 and config.token_block % 32 == 0
    qb_rope.check_config(config, tokens, 64)


def test_fp8_row_alignment_is_enforced():
    # 48-row quantization fragments straddle the FP8 (32, 128) tile (E2003 on TPU).
    bad = qb_rope.QBRopeConfig(token_block=288, quant_fragment=48)
    with pytest.raises(ValueError, match="quant_fragment"):
        qb_rope.check_config(bad, 288, 64)


def test_invalid_config_is_rejected():
    bad = qb_rope.QBRopeConfig(token_block=200, head_block=6, post_fragment=12)
    with pytest.raises(ValueError, match="Invalid QBRopeConfig"):
        qb_rope.check_config(bad, 2048, 64)


def test_long_inputs_split_into_vmem_sized_segments():
    config = qb_rope.QBRopeConfig(token_block=512, vmem_limit_bytes=40 * 1024 * 1024)
    segment = qb_rope._segment_tokens(config, 8192, 1280, 64, 512)
    assert segment < 8192 and 8192 % segment == 0 and segment % 512 == 0
    assert qb_rope.vmem_bytes(config, segment, 1280, 64) <= config.vmem_limit_bytes


@pytest.mark.parametrize("tokens", [96, 100, 240, 288, 384, 2048])
def test_block_sizes_do_not_change_values(tokens):
    """Every element's arithmetic is independent of tiling, so results are bitwise."""
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a TPU backend")
    positions = (np.arange(tokens) * 37) % 4096
    qr, weight, scale, rows = _inputs(tokens, 64, positions)
    default = jax.jit(qb_rope.fused_qb_rope)(qr, weight, scale, rows)
    padded = -(-tokens // 32) * 32
    small = qb_rope.QBRopeConfig(
        token_block=32,
        head_block=8,
        output_slots=2,
        quant_fragment=32,
        post_fragment=16,
        segment_tokens=padded
        if padded <= 512
        else 512
        if padded % 512 == 0
        else padded,
    )
    other = jax.jit(functools.partial(qb_rope.fused_qb_rope, config=small))(
        qr, weight, scale, rows
    )
    assert default.shape == (tokens, 64, 512)
    np.testing.assert_array_equal(
        np.asarray(default, np.float32), np.asarray(other, np.float32)
    )
    expected = reference.qb_rope(qr, weight, scale, *_split(rows))
    np.testing.assert_allclose(
        np.asarray(default, np.float32),
        np.asarray(expected, np.float32),
        atol=1e-2,
        rtol=1e-2,
    )
