# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""glm52_decode_mk rope layouts: MKConfig.rope_neox (glm-agentx TPU_MLA_ROPE_NEOX_PERMUTE /
TPU_INDEXER_ROPE_NEOX_PERMUTE) against the GPT-J interleaved default, and the adapter's layout detection.

The permutes put each rope head's columns in half-split order P = [0, 2, ..., 62, 1, 3, ..., 63]. The megakernel then
takes rope values in that order and must return P(rope_gptj(x)) bit for bit: the same bf16(bf16(x*c) + bf16(y*s))
per element, on the same operand pair, only at another lane. Checked in TPU interpret mode on CPU for both table
halves (lanes 0..63: k_pe / indexer q and k; lanes 64..127: q_pe in the head's last 128 columns).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import kernel as K
from vllm_torchtpu.kernels.glm52_decode_mk import vllm_adapter as A
from vllm_torchtpu.kernels.glm52_decode_mk import wrapper as W

DR = 64
P = np.concatenate(
    [np.arange(0, DR, 2), np.arange(1, DR, 2)]
)  # half-split position i <- interleaved P[i]


def _rope_both_halves(x, tab, neox):
    M = x.shape[0]

    def body(x_ref, t_ref, o_ref):
        xv = x_ref[...]
        o_ref[0] = K.rope(xv, t_ref[0], t_ref[1], neox)
        o_ref[1] = K.rope(xv, t_ref[2], t_ref[3], neox)

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((2, M, 128), jnp.float32),
        interpret=True,
    )(x, tab)


def _bf16_values(rng, shape, scale=1.0):
    return np.asarray(
        jnp.asarray(rng.standard_normal(shape) * scale, jnp.bfloat16).astype(
            jnp.float32
        )
    )


@pytest.mark.parametrize("tmax", [16, 32])
def test_rope_neox_is_the_permuted_gptj_rope_bitwise(tmax):
    rng = np.random.default_rng(tmax)
    cfg = dataclasses.replace(C.FULL, TMAX=tmax)
    M = cfg.M
    pos = rng.integers(0, 1 << 20, size=tmax)
    inv = 1.0 / (10000.0 ** (np.arange(0, DR, 2) / DR))
    ang = pos[:, None] * inv[None, :]
    cs = jnp.asarray(np.concatenate([np.cos(ang), np.sin(ang)], 1), jnp.bfloat16)
    tab_g = W.rope_table(cfg, cs)
    tab_n = W.rope_table(dataclasses.replace(cfg, rope_neox=True), cs)
    x = _bf16_values(rng, (M, 128), 3.0)
    # the neox operand: each 64-lane rope segment of x in half-split order
    xn = x.copy()
    xn[:, 0:DR] = x[:, 0:DR][:, P]
    xn[:, 128 - DR :] = x[:, 128 - DR :][:, P]
    with jax.default_device(jax.devices("cpu")[0]):
        g = np.asarray(_rope_both_halves(jnp.asarray(x), tab_g, False))
        n = np.asarray(_rope_both_halves(jnp.asarray(xn), tab_n, True))
    # lanes 0..63 (table half 0) and 64..127 (half 1): neox == P(gptj), bit for bit
    assert np.array_equal(
        n[0, :, 0:DR].view(np.uint32), g[0, :, 0:DR][:, P].view(np.uint32)
    )
    assert np.array_equal(
        n[1, :, 128 - DR :].view(np.uint32), g[1, :, 128 - DR :][:, P].view(np.uint32)
    )
    # outside its rope segment each half is the identity in both layouts
    assert np.array_equal(n[0, :, DR:], xn[:, DR:]) and np.array_equal(
        g[0, :, DR:], x[:, DR:]
    )
    assert np.array_equal(n[1, :, : 128 - DR], xn[:, : 128 - DR])
    # and the rope is not the identity (the check would pass vacuously otherwise)
    assert not np.array_equal(g[0, :tmax, 0:DR], x[:tmax, 0:DR])


def test_rope_neox_wrong_partner_is_caught():
    """Positive control: the GPT-J partner on half-split operands (a layout mismatch) differs."""
    rng = np.random.default_rng(1)
    cfg = dataclasses.replace(C.FULL, TMAX=16)
    cs = jnp.asarray(rng.uniform(-1, 1, size=(16, DR)), jnp.bfloat16)
    x = _bf16_values(rng, (cfg.M, 128))
    xn = x.copy()
    xn[:, 0:DR] = x[:, 0:DR][:, P]
    with jax.default_device(jax.devices("cpu")[0]):
        g = np.asarray(_rope_both_halves(jnp.asarray(x), W.rope_table(cfg, cs), False))
        bad = np.asarray(
            _rope_both_halves(
                jnp.asarray(xn),
                W.rope_table(dataclasses.replace(cfg, rope_neox=True), cs),
                False,
            )
        )
    assert not np.array_equal(bad[0, :16, 0:DR], g[0, :16, 0:DR][:, P])


# ------------------------------------------------------------------ adapter: the served layout


def _layer(mla_flag, idx_flag):
    """A served layer: self_attn holding the MLA wrapper (glm-agentx marks `_rope_neox_permuted`; None = an older
    tree without the marker) and an indexer (idx_flag "none" = an IndexShare layer without one)."""
    ly = torch.nn.Module()
    ly.self_attn = torch.nn.Module()
    ly.self_attn.mla_attn = torch.nn.Module()
    if mla_flag is not None:
        ly.self_attn.mla_attn._rope_neox_permuted = mla_flag
    if idx_flag != "none":
        ly.self_attn.indexer = torch.nn.Module()
        if idx_flag is not None:
            ly.self_attn.indexer._rope_neox_permuted = idx_flag
    return ly


@pytest.mark.parametrize("flag", [True, False])
def test_rope_layout_uniform(flag):
    layers = [_layer(flag, flag if i % 4 == 0 else "none") for i in range(8)]
    assert A.rope_layout(layers) is flag


def test_rope_layout_older_tree_is_interleaved():
    assert A.rope_layout([_layer(None, None), _layer(None, "none")]) is False


@pytest.mark.parametrize(
    "layers",
    [
        [_layer(True, False)],  # MLA permuted, indexer not: one table cannot serve both
        [_layer(False, True)],
        [_layer(True, True), _layer(False, "none")],  # mixed across layers
    ],
)
def test_rope_layout_mixed_declines(layers):
    with pytest.raises(A.Unsupported, match="rope layouts differ"):
        A.rope_layout(layers)


def test_rope_cache_operand_keeps_an_owning_buffer():
    t = torch.arange(64 * 8, dtype=torch.float32).reshape(8, 64)
    assert A._rope_cache_operand(t) is t
    v = torch.arange(64 * 16, dtype=torch.float32).reshape(16, 64)[:8]
    out = A._rope_cache_operand(v)
    assert out is not v and torch.equal(out, v) and A._owns_storage(out)
