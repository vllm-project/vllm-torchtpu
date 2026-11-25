# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-KERN K6 (VLLM_TPU_MK_KERN_STOCK_KV): the integer-domain value mask equals the f32 path's mask bit for bit (DEV sanity).

K6-M  `union_attn._kern_value_f8` on every e4m3fn byte (0..255) in every row position, with and without the NaN mask: f32 of its output
      equals the flag-off f32 path (`where(v == v, v, 0)` when dcp > 1, then `where(row < length, v * 1.0, 0)`) bit for bit, including
      -0.0 (byte 0x80 kept as -0.0) and the NaN codes 0x7F / 0xFF (-> +0.0).
K6-S  the served scales are 1.0, so `(dot * sm_scale) * scale_k` == `dot * sm_scale` and `pv * scale_v` == `pv` exactly (checked on random
      f32 values); the only device question (f8-push vs bf16-push MXU accumulation) is out of reach on CPU and is the LLO leg's / the DEV
      car's device bitwise row.
"""

from __future__ import annotations

import numpy as np

from tests.megakernel import test_hix700_sidecar as _S  # noqa: F401  (vllm.envs stub + XLA host devices, as the other q38 tests)

import jax.numpy as jnp  # noqa: E402

from vllm_torchtpu.megakernel.wholestep import union_attn as UA  # noqa: E402

F8 = jnp.float8_e4m3fn


def _bits32(x) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32)


def test_k6_value_mask_matches_f32_path():
    TILE = 256
    codes = np.arange(256, dtype=np.uint8)
    v8 = jnp.asarray(np.tile(codes[None, :], (TILE, 1))).view(F8)  # every byte in every row
    for length in (0, 1, 100, 255, 256):
        rows = np.arange(TILE)[:, None] < length
        for nan_mask in (False, True):
            got = np.asarray(UA._kern_value_f8(v8, jnp.asarray(rows), nan_mask).astype(jnp.float32))
            val = np.asarray(v8.astype(jnp.float32))
            if nan_mask:
                val = np.where(val == val, val, 0.0)
            ref = np.where(rows, val * np.float32(1.0), 0.0).astype(np.float32)
            if not nan_mask:  # without the mask a NaN code stays NaN in both paths: compare the masked rows' bits only where finite
                fin = np.isfinite(ref)
                assert (_bits32(got)[fin] == _bits32(ref)[fin]).all() and (np.isnan(got) == np.isnan(ref)).all()
            else:
                assert (_bits32(got) == _bits32(ref)).all(), (length, nan_mask)
    print("Q38KERN K6-M every e4m3fn byte x 5 lengths x NaN mask on / off: integer mask == f32 path bit for bit", flush=True)


def test_k6_unit_scale_is_exact():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(100000).astype(np.float32) * np.float32(37.0)
    sm = np.float32(256 ** -0.5)
    assert (_bits32((x * sm) * np.float32(1.0)) == _bits32(x * sm)).all()
    assert (_bits32(x * np.float32(1.0)) == _bits32(x)).all()
    print("Q38KERN K6-S x 1.0 after the dot is exact on 100000 random f32 values", flush=True)
