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
"""Segment-MK GDN state read / write vs the stock pooled GDN contract (CPU).

One MTP K=3 verify step over a served-geometry unified pool (TP8 per rank:
FP8 `[blocks * 4, 128, 1, 4, 256]`, f32 ssm `[6, 128, 128]` in rows 0..383,
f32 conv `[3, 1280]` in rows 384..399). The stock side is the pool region
adapters (`pool_adapters.gather_region` / `scatter_region`, the bytes the v3
state plan moves) addressed the way `run_jax_gdn_attention_pooled` and the
v3 `memory_ref` address them: read `ckpt[s, off[state[s]]]` when
`seq - q_len > 0`, write checkpoint `t` to `ckpt[s, t]`. The MK side is what
serving runs: `gdn_blocks` on the raw runner metadata, `Qwen38PoolGDN._gather`
(the in-kernel pool read) and `scatter_gdn_ckpts_fast`.

Requests cover the served cases: continuing with each read offset, the first
verify after a prefill (offset 0), a brand-new request (no prior state) and
padded rows. Unrelated blocks hold garbage bytes so a misaddressed read
shows up as non-finite / wrong values.
"""

from __future__ import annotations

import sys
import types

try:
    import vllm.envs  # noqa: F401
except ImportError:  # CPU venv without vllm: pool_adapters only reads the KV layout.
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm"].envs = _envs
    sys.modules["vllm.envs"] = _envs

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import (
    HAS_STATE,
    POOL,
    READ_BLOCKS,
    Qwen38PoolGDN,
)
from vllm_torchtpu.megakernel_runtime.qwen38_gdn_state import Qwen38GdnPoolLayout
from vllm_torchtpu.megakernel_runtime.qwen38_scatter import scatter_gdn_ckpts_fast
from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import T_VERIFY, gdn_blocks

T = T_VERIFY
NB = 24  # manager blocks
VH, KD, VD, QKV = 6, 128, 128, 1280
LAY = Qwen38GdnPoolLayout(pool_shape=(NB * 4, 128, 1, 4, 256), pool_dtype="float8_e4m3fn")

# Runner metadata for one verify step of NS_LIVE requests (+ padded rows).
# Request: (state block, ckpt blocks [K+1], read offset, seq_len); q_len = T.
REQS = [
    (3, [3, 4, 5, 6], 2, 40),  # continuing, accepted 3 last step
    (7, [7, 8, 9, 10], 0, 41),  # continuing, accepted 1
    (11, [11, 12, 13, 14], 3, 50),  # continuing, accepted all 4
    (15, [15, 16, 17, 18], 0, 9),  # first verify after a 5-token prefill
    (19, [19, 20, 21, 22], 1, 4),  # brand new: seq == q_len, no prior state
]
NS = 8  # B / T rows the kernel sees (bucket 32)


def _metadata():
    n_req_rows = 16  # runner's max_reqs padding
    st = np.zeros(n_req_rows, np.int32)
    ck = np.zeros((n_req_rows, T), np.int32)
    seq = np.zeros(n_req_rows, np.int32)
    off = np.zeros(NB, np.int32)
    for s, (b, c, o, sl) in enumerate(REQS):
        st[s], ck[s], seq[s], off[b] = b, c, sl, o
    qsl = np.zeros(n_req_rows + 1, np.int32)
    qsl[1 : len(REQS) + 1] = np.arange(1, len(REQS) + 1) * T
    qsl[len(REQS) + 1 :] = len(REQS) * T
    return [jnp.asarray(x) for x in (st, off, ck, qsl, seq)]


def _ssm_rows(ssm):
    """[n, 6, 128, 128] f32 -> [n, 384, 256] (row-major 256-lane typed rows)."""
    return ssm.reshape(ssm.shape[0], -1, 256)


def _conv_rows(conv):
    """[n, 3, 1280] f32 -> [n, 16, 256] (15 rows + one zero pad row)."""
    r = conv.reshape(conv.shape[0], 15, 256)
    return jnp.pad(r, ((0, 0), (0, 1), (0, 0)))


def stock_write(pool, ids, ssm, conv):
    pool = pool_adapters.scatter_region(
        pool, _ssm_rows(ssm), ids, tok0=0, ntok=LAY.ssm_tokens, split=LAY.split, interpret=True
    )
    return pool_adapters.scatter_region(
        pool, _conv_rows(conv), ids, tok0=LAY.conv_tok0, ntok=LAY.conv_tokens,
        split=LAY.split, interpret=True,
    )


def stock_read(pool, ids):
    """The stock kernel's typed view: ssm in d_v-wide f32 rows, conv in 256-lane rows."""
    ssm = pool_adapters.gather_region(
        pool, ids, tok0=0, ntok=LAY.ssm_tokens, out_dtype=jnp.float32, out_lanes=VD,
        split=LAY.split, interpret=True,
    )
    conv = pool_adapters.gather_region(
        pool, ids, tok0=LAY.conv_tok0, ntok=LAY.conv_tokens, out_dtype=jnp.float32,
        split=LAY.split, interpret=True,
    )
    n = ids.shape[0]
    return ssm[:, : VH * KD].reshape(n, VH, KD, VD), conv[:, :15].reshape(n, 3, QKV)


def stock_addressing(meta):
    """(read block, has_state, write blocks [NS, T]) per the stock pooled op."""
    st, off, ck, qsl, seq = (np.asarray(x) for x in meta)
    rb, hi, wb = [], [], []
    for s in range(NS):
        q = qsl[s + 1] - qsl[s] if s + 1 < len(qsl) else 0
        live = q > 0
        o = int(np.clip(off[st[s]], 0, T - 1))  # ple_speculative_state clamps
        rb.append(ck[s, o] if live else 0)
        hi.append(bool(live and seq[s] - q > 0))
        wb.append(ck[s, :T] if live else np.zeros(T, np.int32))
    return np.array(rb), np.array(hi), np.stack(wb)


def mk_gather(pool, rb, hi):
    """`Qwen38PoolGDN._gather` for one layer, in a grid-less interpret pallas_call."""
    blk = Qwen38PoolGDN.__new__(Qwen38PoolGDN)
    blk.lin_in_layout = lambda g, ex: types.SimpleNamespace(vh=VH, KD=KD, VD=VD, QKV=QKV)
    f8 = jnp.float8_e4m3fn
    hr = 8

    def kernel(rb_ref, hi_ref, pool_ref, rec_out, conv_out, *scr_refs):
        names = [f"pg_raw{b}{j}" for b in range(2) for j in range(3)] + [
            "pg_craw0", "pg_craw1", "pg_sem", "pg_rec", "pg_conv",
        ]
        scr = dict(zip(names, scr_refs))
        ctx = types.SimpleNamespace(
            extra={POOL: pool_ref, READ_BLOCKS: rb_ref, HAS_STATE: hi_ref},
            scr=scr,
            ex=types.SimpleNamespace(NS=NS, T=T, interpret=True),
            g=None,
        )
        blk._gather(ctx, 0)
        rec_out[...] = scr["pg_rec"][...]
        conv_out[...] = scr["pg_conv"][...]

    scratch = [pltpu.VMEM((128, 1, 4, 256), f8) for _ in range(6)] + [
        pltpu.VMEM((16, 1, 4, 256), f8),
        pltpu.VMEM((16, 1, 4, 256), f8),
        pltpu.SemaphoreType.DMA((2, 4)),
        pltpu.VMEM((NS * VH, KD, VD), jnp.bfloat16),
        pltpu.VMEM((NS, hr, QKV), jnp.float32),
    ]
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        kernel,
        in_specs=[smem, smem, pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        out_shape=[
            jax.ShapeDtypeStruct((NS * VH, KD, VD), jnp.bfloat16),
            jax.ShapeDtypeStruct((NS, hr, QKV), jnp.float32),
        ],
        scratch_shapes=scratch,
        interpret=True,
    )(rb.reshape(1, NS).astype(jnp.int32), hi.reshape(1, NS).astype(jnp.int32), pool)


@pytest.fixture(scope="module")
def pool_and_states():
    rng = np.random.default_rng(0)
    # Garbage everywhere (KV bytes of other layers / requests), incl. NaN patterns.
    raw = rng.integers(0, 256, size=LAY.pool_shape, dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    ids = jnp.arange(1, NB, dtype=jnp.int32)
    ssm = jnp.asarray(rng.standard_normal((NB - 1, VH, KD, VD)), jnp.float32)
    conv = jnp.asarray(rng.standard_normal((NB - 1, 3, QKV)), jnp.float32)
    pool = stock_write(pool, ids, ssm, conv)
    full_ssm = jnp.concatenate([jnp.zeros((1, VH, KD, VD)), ssm])
    full_conv = jnp.concatenate([jnp.zeros((1, 3, QKV)), conv])
    return pool, full_ssm, full_conv


def test_stock_adapter_round_trip(pool_and_states):
    pool, ssm, conv = pool_and_states
    ids = jnp.arange(1, NB, dtype=jnp.int32)
    s, c = stock_read(pool, ids)
    np.testing.assert_array_equal(np.asarray(s), np.asarray(ssm[1:]))
    np.testing.assert_array_equal(np.asarray(c), np.asarray(conv[1:]))


def test_mk_read_blocks_match_stock_addressing():
    meta = _metadata()
    rb, wb, hi = gdn_blocks(meta, NS, True)
    srb, shi, swb = stock_addressing(meta)
    np.testing.assert_array_equal(np.asarray(rb[0]), srb)
    np.testing.assert_array_equal(np.asarray(hi[0]), shi)
    np.testing.assert_array_equal(np.asarray(wb[0]), swb)


def test_mk_state_read_matches_stock(pool_and_states):
    pool, ssm, conv = pool_and_states
    meta = _metadata()
    rb, _, hi = gdn_blocks(meta, NS, True)
    rec, hist = mk_gather(pool, rb[0], hi[0])
    srb, shi, _ = stock_addressing(meta)
    s_ssm, s_conv = stock_read(pool, jnp.asarray(srb, jnp.int32))
    m = jnp.asarray(shi)
    want_rec = jnp.where(m[:, None, None, None], s_ssm, 0.0).astype(jnp.bfloat16)
    want_conv = jnp.where(m[:, None, None], s_conv, 0.0)
    got_rec = rec.reshape(NS, VH, KD, VD)
    assert np.isfinite(np.asarray(got_rec, np.float32)).all()
    np.testing.assert_array_equal(np.asarray(got_rec), np.asarray(want_rec))
    np.testing.assert_array_equal(np.asarray(hist[:, :3]), np.asarray(want_conv))
    np.testing.assert_array_equal(np.asarray(hist[:, 3:]), 0.0)


def test_mk_state_write_matches_stock(pool_and_states):
    pool, _, _ = pool_and_states
    meta = _metadata()
    _, wb, _ = gdn_blocks(meta, NS, True)
    rng = np.random.default_rng(1)
    rec_ck = jnp.asarray(rng.standard_normal((1, NS * T * VH, KD, VD)), jnp.bfloat16)
    conv_ck = jnp.asarray(rng.standard_normal((1, NS, 8, QKV)), jnp.float32)
    got = scatter_gdn_ckpts_fast(pool, conv_ck, rec_ck, wb, layout=LAY, interpret=True)

    # Stock: checkpoint t of request s = state after token t, conv window
    # ending at token t (library history rows t+1..t+3), into ckpt[s, t].
    _, _, swb = stock_addressing(meta)
    live = [s for s in range(NS) if swb[s].any()]
    ids = jnp.asarray([swb[s, t] for s in live for t in range(T)], jnp.int32)
    ssm = rec_ck.reshape(NS, T, VH, KD, VD)[jnp.asarray(live)].reshape(-1, VH, KD, VD)
    win = jnp.stack([conv_ck[0, :, t + 1 : t + 4] for t in range(T)], 1)
    win = win[jnp.asarray(live)].reshape(-1, 3, QKV)
    want = stock_write(pool, ids, ssm.astype(jnp.float32), win)

    # Every manager block must be byte-identical, the null block included: the
    # stock kernel moves no bytes for padded rows.
    g = np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint8)).reshape(NB, -1)
    w = np.asarray(jax.lax.bitcast_convert_type(want, jnp.uint8)).reshape(NB, -1)
    bad = [m for m in range(NB) if not np.array_equal(g[m], w[m])]
    assert not bad, f"manager blocks differing from the stock write: {bad}"


def test_shadow_call_leaves_the_pool_untouched(pool_and_states):
    """`write=False` (shadow check) sends every checkpoint to block 0: nothing is written."""
    pool, _, _ = pool_and_states
    _, wb, _ = gdn_blocks(_metadata(), NS, False)
    assert not np.asarray(wb).any()
    rng = np.random.default_rng(2)
    rec_ck = jnp.asarray(rng.standard_normal((1, NS * T * VH, KD, VD)), jnp.bfloat16)
    conv_ck = jnp.asarray(rng.standard_normal((1, NS, 8, QKV)), jnp.float32)
    got = scatter_gdn_ckpts_fast(pool, conv_ck, rec_ck, wb, layout=LAY, interpret=True)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(got, jnp.uint8)),
        np.asarray(jax.lax.bitcast_convert_type(pool, jnp.uint8)),
    )


def test_read_check_catches_a_misaddressed_block(pool_and_states):
    """Positive control: reading each request's neighbouring block must fail the check."""
    pool, _, _ = pool_and_states
    meta = _metadata()
    rb, _, hi = gdn_blocks(meta, NS, True)
    rec, _ = mk_gather(pool, jnp.where(hi[0], rb[0] + 1, 0), hi[0])
    srb, shi, _ = stock_addressing(meta)
    s_ssm, _ = stock_read(pool, jnp.asarray(srb, jnp.int32))
    want = jnp.where(jnp.asarray(shi)[:, None, None, None], s_ssm, 0.0).astype(jnp.bfloat16)
    assert not np.array_equal(np.asarray(rec.reshape(NS, VH, KD, VD)), np.asarray(want))
