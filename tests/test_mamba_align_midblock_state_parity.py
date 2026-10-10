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
"""Stored GDN checkpoints are unchanged when the mid-block align stop merges.

CPU-only. A tiny GDN layer (the reference math of
tests/kernels/test_gdn_attention_v3.py::gdn_attention_ref) runs a warm
prefill on an emulated Mamba "align" state pool exactly as the TPU runner
drives it: per chunk the state block at column ``(end - 1) // block_size`` is
seeded from the previous column on a crossing
(``TPUModelRunner._collect_mamba_state_seed_copies``), the kernel resumes
from it and writes the chunk's final state back, and the state is stored in
the pool dtype. The prompt's partial tail is copied into a CoW block before
the next chunk overwrites it (``MambaManager.allocate_new_blocks``).

The geometry is scaled down 32x (block 20 / hash unit 4 for 640 / 128). The
schedules are the ones the split produces for a hit at floor128 of the
previous prompt: stock ``[ceil(h), f640(P), f128(P), P]`` and, with
TPU_SCHED_SKIP_MIDBLOCK_ALIGN_STOP=1, ``[f640(P), f128(P), P]``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platforms", "cpu")

BS, HB = 20, 4
N_KQ, N_V, D_K, D_V, K = 1, 2, 8, 8, 4
DIM = 2 * N_KQ * D_K + N_V * D_V


def _l2n(x, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 / jnp.sqrt(jnp.sum(x32 * x32, -1, keepdims=True) + eps)).astype(x.dtype)


def gdn_chunk(qkv, b, a, conv_state, rec_state, w, a_log, dt_bias):
    """One sequence chunk resumed from (conv_state, rec_state); the math of
    gdn_attention_ref for a single request with an initial state."""
    n = qkv.shape[0]
    x_full = jnp.concatenate([conv_state, qkv], axis=0)
    acc = jnp.zeros((n, DIM), jnp.float32)
    for k in range(K):
        acc += x_full[k : k + n].astype(jnp.float32) * w[:, 0, k].astype(jnp.float32)[None]
    conv_out = jax.nn.silu(acc.astype(qkv.dtype))
    new_conv = x_full[-(K - 1) :]
    kd = N_KQ * D_K
    q = _l2n(jnp.repeat(conv_out[:, :kd].reshape(n, N_KQ, D_K), N_V // N_KQ, 1))
    k_ = _l2n(jnp.repeat(conv_out[:, kd : 2 * kd].reshape(n, N_KQ, D_K), N_V // N_KQ, 1))
    v = conv_out[:, 2 * kd :].reshape(n, N_V, D_V).astype(jnp.float32)
    beta = jax.nn.sigmoid(b.astype(jnp.float32))
    g = -jnp.exp(a_log)[None] * jax.nn.softplus(a.astype(jnp.float32) + dt_bias[None])

    def step(s, xs):
        q_t, k_t, v_t, beta_t, g_t = xs
        q_t = q_t * D_K**-0.5
        eg = jnp.exp(g_t)
        v_new = beta_t[:, None] * (v_t - eg[:, None] * jnp.einsum("hd,hdm->hm", k_t, s))
        out = eg[:, None] * jnp.einsum("hd,hdm->hm", q_t, s) + jnp.sum(
            q_t * k_t, -1, keepdims=True
        ) * v_new
        s = s * eg[:, None, None] + jnp.einsum("hd,hm->hdm", k_t, v_new)
        return s.astype(jnp.float32), out.astype(qkv.dtype)

    rec, out = jax.lax.scan(step, rec_state, (q, k_, v, beta, g))
    return new_conv, rec, out


def _inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)  # noqa: E731
    return dict(
        qkv=f(n, DIM).astype(jnp.bfloat16),
        b=f(n, N_V).astype(jnp.bfloat16),
        a=f(n, N_V).astype(jnp.bfloat16),
        w=f(DIM, 1, K) * 0.5,
        a_log=f(N_V) * 0.1,
        dt_bias=f(N_V) * 0.1,
    )


def run_align_prefill(x, hit, prompt, ends, store_dtype, no_round_at=()):
    """Emulate the TPU align-mode pool for one warm prefill.

    Returns (states stored at cacheable checkpoints, final state, outputs).
    """
    cols = {}  # block-table column -> block id
    pool_conv, pool_rec = {}, {}
    next_id = [1]

    def block(col):
        if col not in cols:
            cols[col] = next_id[0]
            next_id[0] += 1
        return cols[col]

    def store(bid, conv, rec, rnd=True):
        dt = store_dtype if rnd else jnp.float32
        pool_conv[bid] = conv.astype(dt).astype(jnp.float32)
        pool_rec[bid] = rec.astype(dt).astype(jnp.float32)

    # The hit state (whole prefix in one cold pass) sits in the CoW block at
    # column (hit - 1) // BS.
    zc = jnp.zeros((K - 1, DIM), jnp.bfloat16)
    zr = jnp.zeros((N_V, D_K, D_V), jnp.float32)
    c0, r0, _ = gdn_chunk(x["qkv"][:hit], x["b"][:hit], x["a"][:hit], zc, zr, x["w"], x["a_log"], x["dt_bias"])
    store(block((hit - 1) // BS), c0, r0)

    fine = prompt // HB * HB
    full = prompt // BS * BS
    ckpts, outs = {}, []
    start = hit
    for end in ends:
        prev, curr = (start - 1) // BS, (end - 1) // BS
        if prev != curr:  # seed copy on a block crossing
            pool_conv[block(curr)] = pool_conv[block(prev)]
            pool_rec[block(curr)] = pool_rec[block(prev)]
        bid = block(curr)
        conv_in = pool_conv[bid].astype(jnp.bfloat16)
        conv, rec, out = gdn_chunk(
            x["qkv"][start:end], x["b"][start:end], x["a"][start:end],
            conv_in, pool_rec[bid], x["w"], x["a_log"], x["dt_bias"],
        )
        store(bid, conv, rec, rnd=end not in no_round_at)
        outs.append(out)
        if end in (full, fine) and end < prompt:
            # f640: the block itself is hashed; f128: CoW copy before the
            # next chunk overwrites the running block.
            ckpts[end] = (np.asarray(pool_conv[bid]), np.asarray(pool_rec[bid]))
        start = end
    final = (np.asarray(pool_conv[block((prompt - 1) // BS)]), np.asarray(pool_rec[block((prompt - 1) // BS)]))
    return ckpts, final, np.asarray(jnp.concatenate(outs))


HIT = 148  # mid-block hit (148 % 20 = 8), 4-aligned
STOCK = [160, 180, 184, 187]  # ceil(h), f640, f128, P  (prompt 187)
MERGED = [180, 184, 187]


def _bits(a):
    return np.asarray(a).view(np.uint32)


@pytest.mark.parametrize("seed", range(4))
def test_checkpoints_bit_identical_with_lossless_state_store(seed):
    x = _inputs(200, seed)
    hit, prompt = HIT, 187
    a = run_align_prefill(x, hit, prompt, STOCK, jnp.float32)
    b = run_align_prefill(x, hit, prompt, MERGED, jnp.float32)
    assert sorted(a[0]) == sorted(b[0]) == [180, 184]
    for pos in (180, 184):
        for i in (0, 1):
            np.testing.assert_array_equal(_bits(a[0][pos][i]), _bits(b[0][pos][i]))
    for i in (0, 1):
        np.testing.assert_array_equal(_bits(a[1][i]), _bits(b[1][i]))
    np.testing.assert_array_equal(a[2].view(np.uint16), b[2].view(np.uint16))


@pytest.mark.parametrize("seed", range(4))
def test_bf16_pool_differs_only_by_the_dropped_round_trip(seed):
    """With the production bf16 pool the stock path also rounds the recurrent
    state at ceil(h). Removing exactly that round trip from the stock path
    reproduces the merged path bit for bit; the conv state (bf16 inputs) is
    exact either way."""
    x = _inputs(200, seed)
    hit, prompt = HIT, 187
    stock = run_align_prefill(x, hit, prompt, STOCK, jnp.bfloat16)
    stock_nr = run_align_prefill(x, hit, prompt, STOCK, jnp.bfloat16, no_round_at=(160,))
    merged = run_align_prefill(x, hit, prompt, MERGED, jnp.bfloat16)
    for pos in (180, 184):
        for i in (0, 1):
            np.testing.assert_array_equal(_bits(stock_nr[0][pos][i]), _bits(merged[0][pos][i]))
        np.testing.assert_array_equal(_bits(stock[0][pos][0]), _bits(merged[0][pos][0]))
        # Same magnitude as one extra bf16 rounding (2^-8 relative).
        rec_s, rec_m = stock[0][pos][1], merged[0][pos][1]
        assert np.max(np.abs(rec_s - rec_m)) <= 2.0**-6 * max(1.0, np.max(np.abs(rec_m)))
    for i in (0, 1):
        np.testing.assert_array_equal(_bits(stock_nr[1][i]), _bits(merged[1][i]))
