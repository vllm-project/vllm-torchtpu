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
"""CPU tests for `megakernel_runtime.qwen38_gdn_state` (Pallas interpret mode).

The fake pool has the real Qwen3.8 TP8 per-block geometry — fp8
``(blocks, 128, 1, 4, 256)``, 512-token manager blocks of 4 kernel blocks —
with few blocks. The numpy reference decodes the pool bytes the way the stock
fused GDN V3 kernel does (``kernels/gdn/v3/memory_ref.py`` region DMA at
``block * stride + kb0``, then ``vmem_ldst.load_state_region``: Mosaic
``ref.bitcast`` packing ``pack`` second-minor rows into one u32 word per lane,
token-major element stream, first ``rows_used`` rows reshaped to the logical
state), independently of `pool_adapters`.

Run without the repo conftest (it imports vllm):
  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest --noconftest \
      tests/megakernel_runtime/test_qwen38_gdn_state.py
"""

from __future__ import annotations

import sys
import types

try:  # pool_adapters reads vllm.envs.VLLM_KV_CACHE_LAYOUT (HND check) only.
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vllm_torchtpu.megakernel_runtime import qwen38_gdn_state as gs  # noqa: E402

NMB = 20  # manager blocks (block 0 = null block)
SPLIT, KB, H2P, PACK, LANES = 4, 128, 1, 4, 256
POOL_SHAPE = (NMB * SPLIT, KB, H2P, PACK, LANES)
NL, NS, T = 2, 3, 4
VH, KD, VD, QKV = 6, 128, 128, 1280

LAYOUT = gs.Qwen38GdnPoolLayout(pool_shape=POOL_SHAPE, pool_dtype="float8_e4m3fn")


# ---------------------------------------------------------------- reference
def _f32_rows(raw_block: np.ndarray) -> np.ndarray:
    """One kernel block's uint8 bytes [KB, H2P, PACK, LANES] -> u32 typed rows.

    Mosaic bitcast to 32 bit: ``PACK`` consecutive second-minor rows pack into
    one word per lane (little-endian, row 0 in the low byte); the typed ref is
    then token-major ``(KB * H2P * PACK/4, LANES)``.
    """
    kb, h2p, pack, lanes = raw_block.shape
    ratio = 4  # fp8 bytes per u32
    g = raw_block.reshape(kb, h2p, pack // ratio, ratio, lanes)
    words = np.ascontiguousarray(np.moveaxis(g, 3, -1)).view(np.uint32)[..., 0]
    return words.reshape(-1, lanes)


def _write_f32_rows(raw_block: np.ndarray, rows: np.ndarray, row0: int) -> None:
    """Inverse of `_f32_rows` for typed rows [row0, row0 + len(rows))."""
    kb, h2p, pack, lanes = raw_block.shape
    allrows = _f32_rows(raw_block).copy()
    allrows[row0 : row0 + rows.shape[0]] = rows.view(np.uint32)
    b = allrows.reshape(kb, h2p, pack // 4, lanes, 1).view(np.uint8)  # (...,lanes,4)
    raw_block[...] = np.moveaxis(b, -1, 3).reshape(kb, h2p, pack, lanes)


def ref_read(raw: np.ndarray, mgr: int) -> tuple[np.ndarray, np.ndarray]:
    """(ssm [VH, KD, VD] f32, conv [3, QKV] f32) of manager block `mgr`."""
    base = mgr * SPLIT
    ssm_rows = np.concatenate([_f32_rows(raw[base + j]) for j in range(3)], axis=0)
    ssm = ssm_rows.reshape(-1)[: VH * KD * VD].view(np.float32).reshape(VH, KD, VD)
    conv_rows = _f32_rows(raw[base + 3])[:16]  # conv region: kernel block 3, rows 0..15
    conv = conv_rows.reshape(-1)[: 3 * QKV].view(np.float32).reshape(3, QKV)
    return ssm, conv


def ref_conv_pad_row(raw: np.ndarray, mgr: int) -> np.ndarray:
    return _f32_rows(raw[mgr * SPLIT + 3])[15]


def ref_write(raw: np.ndarray, mgr: int, ssm: np.ndarray, conv: np.ndarray) -> None:
    base = mgr * SPLIT
    flat = ssm.astype(np.float32).reshape(3, -1, LANES)
    for j in range(3):
        _write_f32_rows(raw[base + j], flat[j], 0)
    _write_f32_rows(raw[base + 3], conv.astype(np.float32).reshape(15, LANES), 0)


def _bf16_round(x: np.ndarray) -> np.ndarray:
    return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))


# ---------------------------------------------------------------- fixtures
def _make_pool(seed: int = 0):
    """Random bytes everywhere, then real f32 states in every manager block."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=POOL_SHAPE, dtype=np.uint8)
    for m in range(NMB):
        ssm = rng.standard_normal((VH, KD, VD)).astype(np.float32)
        conv = rng.standard_normal((3, QKV)).astype(np.float32)
        ref_write(raw, m, ssm, conv)
    return raw


def _to_pool(raw: np.ndarray):
    return jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)


def _from_pool(pool) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8))


def _metadata():
    """Per-layer vLLM metadata -> read blocks and write blocks [NL, NS(, T)]."""
    # request 2 is a padded bucket slot -> everything on null block 0.
    ckpt = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]],
            [[9, 14, 12, 16], [13, 11, 18, 15], [0, 0, 0, 0]],
        ],
        np.int32,
    )
    state_indices = np.array([[1, 5, 0], [9, 13, 0]], np.int32)  # base block
    slot_read_offsets = np.zeros((NL, NMB), np.int32)
    slot_read_offsets[0, 1], slot_read_offsets[0, 5] = 2, 0
    slot_read_offsets[1, 9], slot_read_offsets[1, 13] = 3, 1
    read = jnp.stack(
        [
            gs.read_blocks_from_metadata(
                jnp.asarray(state_indices[l]),
                jnp.asarray(slot_read_offsets[l]),
                jnp.asarray(ckpt[l]),
            )
            for l in range(NL)
        ]
    )
    return read, jnp.asarray(ckpt)


def _gather(pool, read, **kw):
    return gs.gather_gdn_states(pool, read, layout=LAYOUT, T=T, interpret=True, **kw)


def _scatter(pool, conv_ckpt, rec_ckpt, write):
    return gs.scatter_gdn_ckpts(
        pool, conv_ckpt, rec_ckpt, write, layout=LAYOUT, interpret=True
    )


def _random_ckpts(seed: int):
    rng = np.random.default_rng(seed)
    conv_ckpt = rng.standard_normal((NL, NS, 8, QKV)).astype(np.float32)
    rec_ckpt = np.asarray(
        jnp.asarray(rng.standard_normal((NL, NS * T * VH, KD, VD)), jnp.bfloat16)
    )
    return conv_ckpt, rec_ckpt


# ---------------------------------------------------------------- checkers
def check_gather(raw, read, conv, rec):
    conv, rec = np.asarray(conv), np.asarray(rec.astype(jnp.float32))
    assert conv.shape == (NL, NS, 8, QKV) and conv.dtype == np.float32
    assert rec.shape == (NL, NS * VH, KD, VD)
    read = np.asarray(read)
    for l in range(NL):
        for s in range(NS):
            ssm_ref, conv_ref = ref_read(raw, int(read[l, s]))
            np.testing.assert_array_equal(conv[l, s, :3], conv_ref)
            np.testing.assert_array_equal(conv[l, s, 3:], 0.0)
            np.testing.assert_array_equal(
                rec[l, s * VH : (s + 1) * VH], _bf16_round(ssm_ref)
            )


def check_scatter(before, after, conv_ckpt, rec_ckpt, write):
    write = np.asarray(write)
    rec_f32 = np.asarray(jnp.asarray(rec_ckpt).astype(jnp.float32))
    written = {int(b) for b in write.reshape(-1)}
    for l in range(NL):
        for s in range(NS):
            for t in range(T):
                b = int(write[l, s, t])
                if b == 0:
                    continue  # null block: shared by padded slots
                ssm, conv = ref_read(after, b)
                row = (s * T + t) * VH
                np.testing.assert_array_equal(ssm, rec_f32[l, row : row + VH])
                # conv window ending at token t = last 3 inputs: rows t+1..t+3
                np.testing.assert_array_equal(conv, conv_ckpt[l, s, t + 1 : t + 4])
                np.testing.assert_array_equal(ref_conv_pad_row(after, b), 0)
    for m in range(NMB):
        rows = slice(m * SPLIT, (m + 1) * SPLIT)
        if m not in written:
            np.testing.assert_array_equal(after[rows], before[rows])
        else:
            # Tokens 400..511 (kernel block 3, rows 16..127) are preserved.
            np.testing.assert_array_equal(
                after[m * SPLIT + 3, 16:], before[m * SPLIT + 3, 16:]
            )


# ---------------------------------------------------------------- tests
def test_layout_matches_qwen4_exp_geometry():
    assert LAYOUT.token_bytes == 1024
    assert LAYOUT.split == 4 and LAYOUT.conv_dim == QKV
    assert LAYOUT.ssm_tokens == 384
    assert (LAYOUT.conv_tok0, LAYOUT.conv_tokens) == (384, 16)  # power of two
    assert LAYOUT.required_tokens == 400
    assert LAYOUT.ssm_rows == 768 and LAYOUT.conv_rows == 16
    # Cross-check against the planner's own derivation.
    from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

    st = derive_pooled_gdn_state_layout(
        ssm_bytes=VH * KD * VD * 4, conv_bytes=3 * QKV * 4, token_bytes=1024
    )
    assert (st.ssm_tokens, st.conv_tokens) == (LAYOUT.ssm_tokens, LAYOUT.conv_tokens)
    # A bf16 KV pool has the same 1024-byte rows.
    bf = gs.Qwen38GdnPoolLayout(
        pool_shape=(NMB * SPLIT, KB, 1, 2, LANES), pool_dtype="bfloat16"
    )
    assert bf.token_bytes == 1024 and bf.conv_tokens == 16


def test_reference_matches_pool_adapters_typed_view():
    """The numpy byte decoder agrees with the Pallas typed view (sanity of ref)."""
    from vllm_torchtpu.kernels import pool_adapters

    raw = _make_pool(1)
    idx = jnp.asarray([3, 7], jnp.int32)
    got = pool_adapters.gather_region(
        _to_pool(raw), idx, tok0=384, ntok=16, out_dtype=jnp.float32, split=4,
        interpret=True,
    )
    for i, m in enumerate((3, 7)):
        np.testing.assert_array_equal(
            np.asarray(got[i]).view(np.uint32), _f32_rows(raw[m * SPLIT + 3])[:16]
        )


def test_gather_matches_v3_reference():
    raw = _make_pool(0)
    read, _ = _metadata()
    np.testing.assert_array_equal(np.asarray(read), [[3, 5, 0], [16, 11, 0]])
    conv, rec = _gather(_to_pool(raw), read)
    assert rec.dtype == jnp.bfloat16
    check_gather(raw, read, conv, rec)


def test_gather_has_initial_state_zeroes_fresh_slots():
    raw = _make_pool(0)
    read, _ = _metadata()
    mask = jnp.asarray([[True, False, True], [False, True, True]])
    conv, rec = _gather(_to_pool(raw), read, has_initial_state=mask)
    conv, rec = np.asarray(conv), np.asarray(rec.astype(jnp.float32))
    assert not conv[0, 1].any() and not rec[0, VH : 2 * VH].any()
    assert not conv[1, 0].any() and not rec[1, :VH].any()
    ssm_ref, conv_ref = ref_read(raw, 3)
    np.testing.assert_array_equal(conv[0, 0, :3], conv_ref)
    np.testing.assert_array_equal(rec[0, :VH], _bf16_round(ssm_ref))


def test_scatter_places_each_token_checkpoint():
    raw = _make_pool(0)
    _, write = _metadata()
    conv_ckpt, rec_ckpt = _random_ckpts(3)
    pool = _scatter(_to_pool(raw), jnp.asarray(conv_ckpt), jnp.asarray(rec_ckpt), write)
    check_scatter(raw, _from_pool(pool), conv_ckpt, rec_ckpt, write)


def test_scatter_under_jit_with_donation():
    raw = _make_pool(0)
    _, write = _metadata()
    conv_ckpt, rec_ckpt = _random_ckpts(4)
    fn = jax.jit(_scatter, donate_argnums=0)
    pool = fn(_to_pool(raw), jnp.asarray(conv_ckpt), jnp.asarray(rec_ckpt), write)
    check_scatter(raw, _from_pool(pool), conv_ckpt, rec_ckpt, write)


def test_round_trip_resumes_from_accepted_checkpoint():
    """Scatter T checkpoints, then gather at read offset = accepted - 1."""
    raw = _make_pool(0)
    _, write = _metadata()
    conv_ckpt, rec_ckpt = _random_ckpts(5)
    pool = _scatter(_to_pool(raw), jnp.asarray(conv_ckpt), jnp.asarray(rec_ckpt), write)
    accepted = np.array([[2, 4, 1], [1, 3, 1]])  # tokens accepted per request
    offs = accepted - 1
    picked = np.take_along_axis(np.asarray(write), offs[..., None], 2)[..., 0]
    read = jnp.asarray(picked)
    conv, rec = _gather(pool, read)
    conv, rec = np.asarray(conv), np.asarray(rec)
    for l in range(NL):
        for s in range(NS - 1):  # skip the padded slot (null block)
            t = int(offs[l, s])
            np.testing.assert_array_equal(
                conv[l, s, :3], conv_ckpt[l, s, t + 1 : t + 4]
            )
            row = (s * T + t) * VH
            np.testing.assert_array_equal(
                rec[l, s * VH : (s + 1) * VH].view(np.uint16),
                rec_ckpt[l, row : row + VH].view(np.uint16),
            )


# ---------------------------------------------------------------- fault controls
def test_fault_conv_window_off_by_one_is_caught(monkeypatch):
    """A window ending one token early (rows t..t+2) must fail placement."""
    monkeypatch.setattr(
        gs,
        "_conv_windows",
        lambda c, T: jnp.stack([c[:, :, t : t + 3] for t in range(T)], axis=2),
    )
    raw = _make_pool(0)
    _, write = _metadata()
    conv_ckpt, rec_ckpt = _random_ckpts(6)
    pool = _scatter(_to_pool(raw), jnp.asarray(conv_ckpt), jnp.asarray(rec_ckpt), write)
    with pytest.raises(AssertionError):
        check_scatter(raw, _from_pool(pool), conv_ckpt, rec_ckpt, write)


def test_fault_token_order_is_caught():
    """Checkpoints written to reversed blocks (t -> T-1-t) must fail placement."""
    raw = _make_pool(0)
    _, write = _metadata()
    conv_ckpt, rec_ckpt = _random_ckpts(7)
    pool = _scatter(
        _to_pool(raw), jnp.asarray(conv_ckpt), jnp.asarray(rec_ckpt), write[..., ::-1]
    )
    with pytest.raises(AssertionError):
        check_scatter(raw, _from_pool(pool), conv_ckpt, rec_ckpt, write)


def test_fault_state_transpose_is_caught(monkeypatch):
    """A k/v-transposed rec operand must fail the gather check."""
    orig = gs._ssm_rows_to_rec
    monkeypatch.setattr(
        gs,
        "_ssm_rows_to_rec",
        lambda *a: jnp.swapaxes(orig(*a), -1, -2),
    )
    raw = _make_pool(0)
    read, _ = _metadata()
    conv, rec = _gather(_to_pool(raw), read)
    with pytest.raises(AssertionError):
        check_gather(raw, read, conv, rec)


def test_rejects_mismatched_pool():
    raw = _make_pool(0)
    read, _ = _metadata()
    pool = _to_pool(raw)[: SPLIT * (NMB - 1)]
    with pytest.raises(ValueError):
        _gather(pool, read)
