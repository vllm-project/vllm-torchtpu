# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU_DSV41_ENGRAM_HBM device lookup against the host exchange's layout.

Runs on CPU devices: XLA_FLAGS=--xla_force_host_platform_device_count=8.
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram_hbm

WORLD, NODE, TOKENS, PART_COLS, HEAD_ROWS, DIM, BLOCK = 8, 4, 5, 3, 40, 8, 4
COLS = PART_COLS * NODE


def _reference(raw, scale):
    values = raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    power = (scale.astype(np.int32) << 23).view(np.float32)
    power = np.repeat(power, BLOCK, -1)
    invalid = np.repeat(scale == 255, BLOCK, -1)
    return np.where(invalid, 0.0, values * power).astype(ml_dtypes.bfloat16)


def test_lookup_matches_host_exchange_layout():
    if len(jax.devices()) < WORLD:
        pytest.skip("needs 8 devices")
    rng = np.random.default_rng(0)
    offsets = np.arange(COLS) * HEAD_ROWS  # global row id = head offset + id
    # Rank r holds the heads of node-local shard r % NODE, as the host does.
    meta = tuple(
        (
            PART_COLS * (r % NODE),
            int(offsets[PART_COLS * (r % NODE)]),
            int(offsets[PART_COLS * (r % NODE)] + PART_COLS * HEAD_ROWS),
            PART_COLS * HEAD_ROWS,
        )
        for r in range(WORLD)
    )
    rows = PART_COLS * HEAD_ROWS
    full_raw = rng.integers(0, 255, (COLS * HEAD_ROWS, DIM), dtype=np.uint8)
    full_scale = rng.integers(100, 150, (COLS * HEAD_ROWS, DIM // BLOCK), np.uint8)
    table = np.concatenate([full_raw[m[1] : m[2]] for m in meta])
    scales = np.concatenate([full_scale[m[1] : m[2]] for m in meta])
    hashes = np.stack(
        [offsets + rng.integers(0, HEAD_ROWS, (TOKENS, COLS)) for _ in range(WORLD)]
    ).astype(np.int32)
    hashes[3, 1, 5] = -1  # a dead n-gram reads as zeros

    mesh = Mesh(np.array(jax.devices()[:WORLD]), (engram_hbm._AXIS,))
    lookup = engram_hbm.make_lookup(
        mesh, TOKENS, COLS, PART_COLS, rows, DIM, BLOCK, NODE, meta
    )
    out = lookup(
        jnp.asarray(hashes.reshape(WORLD * TOKENS, COLS)),
        jnp.asarray(table),
        jnp.asarray(scales),
        jnp.arange(WORLD, dtype=jnp.int32).reshape(WORLD, 1),
    )
    out = np.asarray(out).reshape(WORLD, TOKENS, COLS, DIM)

    expected = np.zeros_like(out)
    for r in range(WORLD):
        for t in range(TOKENS):
            for c in range(COLS):
                g = hashes[r, t, c]
                if g >= 0:
                    expected[r, t, c] = _reference(full_raw[g], full_scale[g])
    np.testing.assert_array_equal(out.view(np.uint16), expected.view(np.uint16))


def test_dequant_matches_reference():
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 255, (6, DIM), dtype=np.uint8)
    scale = rng.integers(100, 150, (6, DIM // BLOCK), dtype=np.uint8)
    scale[2] = 255
    got = np.asarray(engram_hbm.dequant(jnp.asarray(raw), jnp.asarray(scale), BLOCK))
    want = _reference(raw, scale)
    same = got.view(np.uint16) == want.view(np.uint16)
    both_nan = np.isnan(got.astype(np.float32)) & np.isnan(want.astype(np.float32))
    assert (same | both_nan).all()


def test_default_env_and_uses_hbm_lookup_gating(monkeypatch):
    from types import SimpleNamespace

    from vllm_torchtpu import envs
    from vllm_torchtpu.distributed import ep_mesh

    monkeypatch.delenv("TPU_DSV41_ENGRAM_HBM", raising=False)
    assert envs.environment_variables["TPU_DSV41_ENGRAM_HBM"]() == "first"

    def _check(layer_idx: int, *, dev_type: str = "tpu", dp_size: int = 8) -> bool:
        staged: object = SimpleNamespace(device=SimpleNamespace(type=dev_type))
        return engram_hbm.should_use_hbm_lookup(
            staged_raw=staged,  # type: ignore[arg-type]
            dp_size=dp_size,
            layer_hash_index=layer_idx,
        )

    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: object())
    monkeypatch.setattr(envs, "TPU_DSV41_ENGRAM_HBM", "first")
    assert _check(0) is True
    assert _check(1) is False
    assert _check(0, dev_type="cpu") is False
    assert _check(0, dp_size=1) is False

    monkeypatch.setattr(envs, "TPU_DSV41_ENGRAM_HBM", "all")
    assert _check(0) is True
    assert _check(1) is True

    monkeypatch.setattr(envs, "TPU_DSV41_ENGRAM_HBM", "off")
    assert _check(0) is False

    monkeypatch.setattr(envs, "TPU_DSV41_ENGRAM_HBM", "first")
    monkeypatch.setattr(ep_mesh, "get_ep_group", lambda: None)
    assert _check(0) is False
