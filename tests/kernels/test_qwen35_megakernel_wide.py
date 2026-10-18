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
"""Wide step (mb_rows > 0): one call over several microbatches matches the per-microbatch calls.

CPU interpret, one rank, a 4-layer hybrid stack with EP copy slots (no copies used), real routing:
the 32-row wide step (two 16-row microbatches, the MoE phase once over all rows) against the
two 16-row calls run back to back on the same pools: hidden rows, the KV pages the step writes
and the GatedDeltaNet checkpoints. The routed-expert sum differs only by the bf16 high/low split
of the routing weight in the one-hot scatter (2^-16 relative), so the comparison has a tolerance.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from vllm_torchtpu.kernels.qwen35_megakernel import megakernel as K
from vllm_torchtpu.kernels.qwen35_megakernel import pool_io

from qwen35_megakernel_test_utils import unpack_conv_checkpoint, unpack_rec_checkpoint


def _tiles(c):
    return dataclasses.replace(
        K.default_tiles(c), nbd=4, nbe=4, nkv=2, hier=0, rsag=False, kv_blk=256, kv_sub=128
    )


import pytest
from vllm_torchtpu.kernels.qwen35_megakernel import context as CTX


@pytest.mark.parametrize("dyn_row,scatter_tn,copies", [(True, True, False), (False, False, False),
                                                       (True, True, True)])
def test_wide_step_matches_microbatch_calls(monkeypatch, dyn_row, scatter_tn, copies):
    """copies: experts 0..3 of every layer carry a redundant copy code (slots 128..131 hold their
    weights and scale rows), so they go through the copy-candidate list and the Po2C passes instead
    of the primary scan; on one rank the primary slot is always chosen."""
    monkeypatch.setattr(CTX, "WIDE_DYN_ROW", dyn_row)
    monkeypatch.setattr(CTX, "WIDE_SCATTER_TN", scatter_tn)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("tp",))
    c_mb = K.MKConfig(
        R=1, B=16, L=4, att_every=4, H=512, NH=4, NKV=1, HD=256, KH=2, VH=4, KD=128, VD=128,
        E=128, TOPK=2, I=128, IS=128, ctx=256, T=4, moe_tp=False, stack_e=False, e_slots=132,
        paged_kv=True, fused_pool_io=True, num_pages=384, page_size=128, max_pages_per_seq=2,
        ckpt_k=4,
    )
    c_wide = dataclasses.replace(c_mb, B=32, mb_rows=16)
    shapes = K.weight_shapes(c_mb, _tiles(c_mb))
    assert K.weight_shapes(c_wide, _tiles(c_wide)) == shapes
    rng = np.random.default_rng(42)
    lay, _ = K.smalls_layout(c_mb)
    w_leaves = []
    for k in K.W_NAMES:
        spec = shapes[k]
        if k in ("e13", "e2"):
            for s_ in spec:
                v = rng.standard_normal(s_.shape).astype(np.float32) * 0.05
                if copies:
                    v[128:132] = v[0:4]
                w_leaves.append(jnp.asarray(v, s_.dtype))
            continue
        s_, dt = spec
        v = rng.standard_normal(s_).astype(np.float32) * 0.02
        if k == "smalls":
            v[...] = np.abs(v)
            o, n = lay["ecopy"]
            v[:, :, o:o + n] = 0.0
            if copies:  # expert e < 4 duplicated at slot 128 + e of rank 0: code 1 + 0 * EXTRA + e
                v[:, :, o:o + 4] = 1.0 + np.arange(4, dtype=np.float32)
            o, n = lay["alog"]
            v[:, :, o:o + 128] = -1.0
        if k == "es":
            v = rng.uniform(0.5, 1.5, s_).astype(np.float32)
            if copies:
                v[:, 128:132] = v[:, 0:4]
        w_leaves.append(jnp.asarray(v, dt))

    n_seq = 8
    raw_init = jnp.asarray(rng.integers(0, 40, size=(c_mb.num_pages, 2, 64, 4, c_mb.page_size),
                                        dtype=np.uint8)).view(K.FP8)
    tokens = rng.standard_normal((n_seq * 4, c_mb.H)).astype(np.float32) * 0.1
    pos = np.tile(np.array([16, 17, 18, 19], np.int32), n_seq)
    pt = np.array([[1 + 2 * s, 2 + 2 * s] for s in range(n_seq)], np.int32)
    ckpt = np.zeros((3, n_seq, 4), np.int32)
    slot = 24
    for sub in range(3):
        for s in range(n_seq):
            ckpt[sub, s] = slot + np.arange(4)
            slot += 4
    roff = np.zeros((3, n_seq), np.int32)

    def run(c, calls, leaves=None):
        leaves = w_leaves if leaves is None else leaves
        step, _, _, _ = pool_io.make_flat_pooled_step(
            mesh, c, _tiles(c), interpret=True, fused_pool_io=True
        )
        raw = raw_init
        hids = []
        for r0, s0 in calls:
            b, ns = c.B, c.NS
            outs = step(
                jnp.asarray(pos[r0:r0 + b]),
                jnp.asarray(tokens[r0:r0 + b], dtype=K.BF16),
                *leaves,
                raw,
                jnp.asarray(pt[s0:s0 + ns]),
                jnp.asarray(ckpt[:, s0:s0 + ns]),
                jnp.asarray(roff[:, s0:s0 + ns]),
            )
            hids.append(np.asarray(outs[0][:b], dtype=np.float32))
            raw = outs[1]
        return np.concatenate(hids, axis=0), np.asarray(raw.view(jnp.uint8))

    h_ref, raw_ref = run(c_mb, [(0, 0), (16, 4)])
    h_wide, raw_wide = run(c_wide, [(0, 0)])
    if copies:
        # with TOPK 2 the routed sum is order-free, so the copy-coded run is bit-identical to the
        # same weights without copy codes in both plans (the copy candidates are not dropped)
        i_sm = K.W_NAMES.index("smalls") + len(shapes["e13"]) + len(shapes["e2"]) - 2
        sm = np.asarray(w_leaves[i_sm]).copy()
        o, n = lay["ecopy"]
        sm[:, :, o:o + n] = 0.0
        plain = list(w_leaves)
        plain[i_sm] = jnp.asarray(sm, w_leaves[i_sm].dtype)
        h_wide0, _ = run(c_wide, [(0, 0)], plain)
        h_ref0, _ = run(c_mb, [(0, 0), (16, 4)], plain)
        print("copy codes vs none: wide identical", float(np.mean(h_wide == h_wide0)),
              "ref identical", float(np.mean(h_ref == h_ref0)))
        assert np.array_equal(h_wide, h_wide0) and np.array_equal(h_ref, h_ref0)
    assert np.isfinite(h_wide).all() and np.isfinite(h_ref).all()
    close = np.isclose(h_wide, h_ref, atol=2e-2, rtol=2e-2)
    print("hidden identical frac", float(np.mean(h_wide == h_ref)), "close frac", float(close.mean()),
          "max abs diff", float(np.abs(h_wide - h_ref).max()), "ref max", float(np.abs(h_ref).max()))
    # a bf16 rounding flip of a near-zero element is not a divergence; the gate is the share of
    # rows outside the tolerance and the largest difference
    assert close.mean() > 0.999 and np.abs(h_wide - h_ref).max() < 0.1
    kv_pages = pt[:, 0]
    kvf = lambda a: a.view(jnp.dtype(K.FP8)).astype(np.float32)
    np.testing.assert_allclose(kvf(raw_wide[kv_pages]), kvf(raw_ref[kv_pages]), atol=0.25, rtol=0.13)
    for sub in range(3):
        for s in range(n_seq):
            for t in range(4):
                p0 = int(ckpt[sub, s, t]) * c_mb.gdn_pages
                np.testing.assert_allclose(
                    unpack_rec_checkpoint(raw_wide[p0:p0 + c_mb.rec_pages], c_mb.vh),
                    unpack_rec_checkpoint(raw_ref[p0:p0 + c_mb.rec_pages], c_mb.vh),
                    atol=2e-2, rtol=2e-2)
                np.testing.assert_allclose(
                    unpack_conv_checkpoint(raw_wide[p0 + c_mb.rec_pages], c_mb.QKV),
                    unpack_conv_checkpoint(raw_ref[p0 + c_mb.rec_pages], c_mb.QKV),
                    atol=2e-2, rtol=2e-2)
