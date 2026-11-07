"""The staged PD repack moves staging bytes into the pool byte for byte, NaN encodings included.

The carrier is FP8-typed but holds BF16 compressed keys and FP32 GDN state. On TPU an XLA gather of
the FP8 carrier canonicalised 0xFF to 0x7F (served rowdump riders, lane_xfer section 11.29), which
halves a BF16 value whose low byte is 0xFF. The repack now reads staging rows with a DMA row gather.
These tests run it in Pallas TPU interpret mode on CPU and check every byte value survives.
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging as st  # noqa: E402

F8 = jnp.float8_e4m3fn


def _carrier(pages: int, rows: int, seed: int = 0):
    """A [pages, rows, 1, 4, 256] FP8-typed carrier whose bytes cover all 256 values, 0xFF and 0x7F
    on every row."""
    rng = np.random.default_rng(seed)
    u8 = rng.integers(0, 256, size=(pages, rows, 1, 4, 256), dtype=np.uint8)
    u8[..., 0, 0] = 0xFF
    u8[..., 0, 1] = 0x7F
    u8[..., 1, 0] = 0x80
    u8[..., 1, 1] = 0x00
    return u8, jax.lax.bitcast_convert_type(jnp.asarray(u8), F8)


def _canonicalising_gather(stage, flat):
    """What the TPU XLA gather did: the FP8 values pass through a wider float and every NaN comes back as
    the canonical positive NaN (0x7F). A plain CPU f8 -> f32 -> f8 round trip keeps the sign, so the
    canonicalisation is explicit here."""
    x = stage[flat].astype(jnp.float32)
    return jnp.where(jnp.isnan(x), jnp.abs(x), x).astype(stage.dtype)


def _dma(stage, flat):
    return st.dma_row_gather(stage, flat, interpret=True)


@pytest.mark.parametrize("n", [1, 63, 64, 130])
def test_dma_row_gather_is_byte_exact(n):
    u8, carrier = _carrier(3, 64)
    stage = carrier.reshape(-1, 4, 256)
    flat = jnp.asarray(np.random.default_rng(n).integers(0, stage.shape[0], size=n), jnp.int32)
    got = np.asarray(jax.lax.bitcast_convert_type(_dma(stage, flat), jnp.uint8))
    want = u8.reshape(-1, 4, 256)[np.asarray(flat)]
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def _repack_rows(gather):
    fn = st.make_repack(
        src_block=64, dst_block=64, src_gdn=None, dst_gdn=None, kv_head=0, v_head_start=0,
        channels=[0], write=lambda pool, rows, page, logical, valid: pool, gather=gather,
    )
    u8, carrier = _carrier(4, 64, seed=1)
    flat_u8 = u8.reshape(-1, 4, 256)
    copy_src = jnp.asarray(np.arange(5, 5 + 100, dtype=np.int32))
    main_src = jnp.asarray(np.arange(130, 130 + 20, dtype=np.int32))
    main, copies, gdn = fn.rows(carrier, main_src, copy_src, jnp.zeros((0,), jnp.int32))
    return flat_u8, np.asarray(main), np.asarray(copies), np.asarray(copy_src), np.asarray(main_src)


def test_repack_copies_every_byte_with_the_dma_gather():
    flat_u8, main, copies, copy_src, main_src = _repack_rows(_dma)
    assert np.array_equal(copies, flat_u8[copy_src])  # compressed keys, rings, PLE: bytes unchanged
    assert np.array_equal(main[:, :2], flat_u8[main_src][:, 0:2])  # main KV head pair (kv_head 0)


def test_the_byte_check_catches_nan_canonicalisation():
    """Power: the same check fails when the gather canonicalises NaNs (the TPU XLA gather's behaviour)."""
    flat_u8, main, copies, copy_src, main_src = _repack_rows(_canonicalising_gather)
    diff = copies != flat_u8[copy_src]
    assert diff.any()
    assert set(zip(flat_u8[copy_src][diff].tolist(), copies[diff].tolist())) == {(0xFF, 0x7F)}


def test_default_gather_is_the_dma_kernel_on_tpu(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    seen = {}
    real = st.dma_row_gather

    def spy(stage, flat, **kw):
        seen["called"] = True
        return real(stage, flat, interpret=True)

    monkeypatch.setattr(st, "dma_row_gather", spy)
    fn = st.make_repack(
        src_block=64, dst_block=64, src_gdn=None, dst_gdn=None, kv_head=0, v_head_start=0,
        channels=[0], write=lambda pool, rows, page, logical, valid: pool,
    )
    _, carrier = _carrier(2, 64)
    fn.rows(carrier, jnp.arange(3, dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32), jnp.zeros((0,), jnp.int32))
    assert seen.get("called")
