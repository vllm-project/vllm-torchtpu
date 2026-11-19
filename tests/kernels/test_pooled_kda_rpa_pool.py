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
"""Pooled KDA on a dense-attention (RPA v3) KV pool.

The pooled KDA op keeps its recurrent and conv state in the unified KV pool
that the model's attention layers share. Kimi-Linear and Kimi-K3 share an MLA
pool `(blocks, block_size, packing, lanes)` that packs several tokens per row,
and the op infers that packing from `pool.shape[2]`. A model whose attention
layers are dense (GLM-5.3 is the motivating one) shares the RPA v3 cache
`(blocks, block_size, kv_heads_x2 // packing, packing, head_dim)` instead,
which holds one token per row; its layers pass `pool_tokens_per_row=1`.

The geometry is one tensor-parallel shard of a dense-attention hybrid with
64 KDA heads of 128 channels, a width-4 short convolution, a bounded gate
(`lower_bound = -5`) and 64 attention KV heads of 256 lanes, i.e. 8 KDA heads
and 8 KV heads per shard at TP=8 (GLM-5.3-Flash). A token row is 8 KiB at
TP=8 and 64 KiB at TP=1. Either way the fp32 recurrent state fills 64 rows
and the fp32 conv state (4.5 rows) pads to 8, so a manager block must hold at
least 72 tokens.

The CPU tests cover the region arithmetic and argument validation, with the
Pallas pieces swapped for pure-JAX stand-ins where the op has to run. The TPU
tests check the dense path against an fp64 recurrence written from the model
definition, the pooled path against the dense one, and prefix-cache hits
against misses.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    _build_fused_core,
    _pooled_kda_core,
)

HEAD_DIM, KERNEL_SIZE = 128, 4
LOWER_BOUND = -5.0  # The bounded gate's `linear_lower_bound`.
EPS = 1e-5  # The layer's output-norm eps.
PACK, LANES = 2, 256  # RPA v3 bf16 packing; the attention head size.
NUM_BLOCKS, BLOCK_SIZE = 5, 128  # A manager block of 128 tokens >= 72.


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One tensor-parallel shard of the KDA layer and the attention pool."""

    num_heads: int  # KDA heads per shard.
    kv_rows: int  # RPA `kv_heads_x2 // packing` per shard.

    @property
    def token_bytes(self) -> int:
        return self.kv_rows * PACK * LANES * 2

    @property
    def layout(self):
        # Both state regions are fp32 whatever the pool dtype.
        return derive_pooled_gdn_state_layout(
            ssm_bytes=self.num_heads * HEAD_DIM * HEAD_DIM * 4,
            conv_bytes=self.conv_elems * 4,
            token_bytes=self.token_bytes,
        )

    @property
    def conv_elems(self) -> int:
        return (KERNEL_SIZE - 1) * 3 * self.num_heads * HEAD_DIM

    @property
    def ssm_region_rows(self) -> int:
        return self.layout.ssm_tokens * self.token_bytes // (4 * HEAD_DIM)

    @property
    def conv_region_rows(self) -> int:
        return self.layout.conv_tokens * self.token_bytes // (4 * LANES)

    def pool_shape(self, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE):
        return (num_blocks, block_size, self.kv_rows, PACK, LANES)


# Dense-attention RPA pool, 8 KDA heads and 8 KV heads (16 rows of K and V,
# packed in pairs) per shard with head_dim 128, as in GLM-5.3-Flash at TP=8.
TP8 = Geometry(num_heads=8, kv_rows=8)
# The same model unsharded. Only its layout arithmetic is checked: 64 KDA
# heads per shard do not fit the fused kernel's VMEM budget.
TP1 = Geometry(num_heads=64, kv_rows=64)


def _bf16(rng, shape, scale):
    return jnp.asarray(
        rng.standard_normal(shape).astype(np.float32) * scale, dtype=jnp.bfloat16
    )


def _weights(geom, rng):
    """The layer's parameters, in the dtypes the layer hands the ops."""
    h, d = geom.num_heads, HEAD_DIM
    return dict(
        conv_weight=jnp.asarray(
            rng.standard_normal((KERNEL_SIZE, 3, h, d)).astype(np.float32) * 0.5
        ),
        a_log=jnp.asarray(rng.uniform(0.0, 2.5, h).astype(np.float32)),
        dt_bias=jnp.asarray(rng.standard_normal(h * d).astype(np.float32) * 0.5),
        norm_weight=jnp.asarray(1.0 + 0.1 * rng.standard_normal(d).astype(np.float32)),
    )


def _activations(geom, rng, query_lens, seq_lens):
    h, d = geom.num_heads, HEAD_DIM
    num_tokens = int(sum(query_lens))
    return dict(
        mixed_qkv=_bf16(rng, (num_tokens, 3 * h * d), 0.5),
        raw_gate=_bf16(rng, (num_tokens, h * d), 1.0),
        beta=_bf16(rng, (num_tokens, h), 1.0),
        output_gate=_bf16(rng, (num_tokens, h * d), 1.0),
        query_start_loc=jnp.asarray([0] + list(np.cumsum(query_lens)), jnp.int32),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
    )


def _initial_states(geom, rng, num_seqs):
    """Per-request states.

    The conv state holds past pre-convolution inputs: bf16 activations
    widened to fp32, so every word's low half is zero. Arbitrary fp32 bits
    would not survive the bf16 pool: TPU canonicalizes bf16 NaN payloads and
    flushes bf16 denormals when it moves the pool.
    """
    h, d = geom.num_heads, HEAD_DIM
    conv = _bf16(rng, (num_seqs, KERNEL_SIZE - 1, 3, h, d), 0.5).astype(jnp.float32)
    ssm = jnp.asarray(rng.standard_normal((num_seqs, h, d, d)).astype(np.float32) * 0.1)
    return conv, ssm


def _distribution(decode_end, num_seqs):
    return jnp.asarray([decode_end, num_seqs, num_seqs], jnp.int32)


@functools.cache
def _dense_core():
    return jax.jit(_build_fused_core(LOWER_BOUND, EPS))


def _dense(geom, weights, acts, conv_init, ssm_init, distribution):
    """The dense path: the fused core over null-slot-prepended buffers, in
    the flat conv layout the pooled op hands it after gathering."""
    h, d = geom.num_heads, HEAD_DIM
    num_seqs = acts["seq_lens"].shape[0]
    conv_buf = jnp.concatenate([jnp.zeros_like(conv_init[:1]), conv_init], axis=0)
    ssm_buf = jnp.concatenate([jnp.zeros_like(ssm_init[:1]), ssm_init], axis=0)
    output, new_conv, new_ssm = _dense_core()(
        acts["mixed_qkv"],
        acts["raw_gate"],
        acts["beta"],
        acts["output_gate"],
        conv_buf.reshape(num_seqs + 1, KERNEL_SIZE - 1, 3 * h * d),
        ssm_buf,
        weights["conv_weight"],
        weights["a_log"],
        weights["dt_bias"],
        weights["norm_weight"],
        acts["query_start_loc"],
        jnp.arange(1, num_seqs + 1, dtype=jnp.int32),
        acts["seq_lens"],
        distribution,
    )
    return (
        output,
        new_conv.reshape(num_seqs + 1, KERNEL_SIZE - 1, 3, h, d)[1:],
        new_ssm[1:],
    )


@functools.cache
def _pooled_core(pool_block_tokens, pool_tokens_per_row, use_jit=True):
    core = functools.partial(
        _pooled_kda_core,
        lower_bound=LOWER_BOUND,
        eps=EPS,
        pool_block_tokens=pool_block_tokens,
        pool_tokens_per_row=pool_tokens_per_row,
    )
    return jax.jit(core) if use_jit else core


def _pooled(
    weights,
    acts,
    pool,
    block_ids,
    distribution,
    pool_tokens_per_row=1,
    pool_block_tokens=BLOCK_SIZE,
    use_jit=True,
):
    return _pooled_core(pool_block_tokens, pool_tokens_per_row, use_jit)(
        acts["mixed_qkv"],
        acts["raw_gate"],
        acts["beta"],
        acts["output_gate"],
        pool,
        weights["conv_weight"],
        weights["a_log"],
        weights["dt_bias"],
        weights["norm_weight"],
        acts["query_start_loc"],
        jnp.asarray(block_ids, jnp.int32),
        acts["seq_lens"],
        distribution,
    )


def _garbage_pool(geom, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, seed=99):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal(geom.pool_shape(num_blocks, block_size)).astype(np.float32),
        dtype=jnp.bfloat16,
    )


def _pack_pool(geom, pool, conv, ssm, block_ids):
    """Write states into the given blocks' state regions, the way the op
    lays them out; every other byte of the pool keeps its value."""
    layout = geom.layout
    idx = jnp.asarray(block_ids, jnp.int32)
    n = len(block_ids)
    conv_vals = jnp.pad(
        conv.reshape(n, -1),
        ((0, 0), (0, geom.conv_region_rows * LANES - geom.conv_elems)),
    ).reshape(n, geom.conv_region_rows, LANES)
    pool = pool_adapters.scatter_region(
        pool, conv_vals, idx, tok0=layout.ssm_tokens, ntok=layout.conv_tokens
    )
    ssm_rows = geom.num_heads * HEAD_DIM
    ssm_vals = jnp.pad(
        ssm.reshape(n, ssm_rows, HEAD_DIM),
        ((0, 0), (0, geom.ssm_region_rows - ssm_rows), (0, 0)),
    )
    return pool_adapters.scatter_region(
        pool, ssm_vals, idx, tok0=0, ntok=layout.ssm_tokens
    )


def _unpack_pool(geom, pool, block_ids, split=1):
    """Read the states back through the op's own gather machinery; also
    returns the conv region's padding elements."""
    layout = geom.layout
    idx = jnp.asarray(block_ids, jnp.int32)
    n = len(block_ids)
    conv_region = pool_adapters.gather_region(
        pool,
        idx,
        tok0=layout.ssm_tokens,
        ntok=layout.conv_tokens,
        out_dtype=jnp.float32,
        split=split,
    ).reshape(n, -1)
    conv = conv_region[:, : geom.conv_elems].reshape(
        n, KERNEL_SIZE - 1, 3, geom.num_heads, HEAD_DIM
    )
    ssm = pool_adapters.gather_region(
        pool,
        idx,
        tok0=0,
        ntok=layout.ssm_tokens,
        out_dtype=jnp.float32,
        out_lanes=HEAD_DIM,
        split=split,
    )
    ssm = ssm[:, : geom.num_heads * HEAD_DIM].reshape(
        n, geom.num_heads, HEAD_DIM, HEAD_DIM
    )
    return conv, ssm, conv_region[:, geom.conv_elems :]


def _f64(x):
    # np.asarray keeps bf16 as ml_dtypes.bfloat16 and fp64 inputs (threaded
    # reference state) exact.
    return np.asarray(x).astype(np.float64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _reference(
    geom,
    weights,
    acts,
    conv_init,
    ssm_init,
    *,
    transpose_state=False,
    decay_values=False,
):
    """fp64 KDA with a bounded gate, one token at a time.

    Causal depthwise conv over [carried window; new tokens] and SiLU; q and k
    L2-normalized (q also scaled by 1/sqrt(D)); the bounded per-channel
    log-decay `lb * sigmoid(exp(A_log) * (gate + dt_bias))`; the delta rule on
    a [H, D_k, D_v] state whose decay scales the D_k rows; and the per-head
    RMS norm gated by sigmoid(output_gate). A request whose seq_len equals its
    query length is fresh and starts from zero state, whatever its slot holds.

    `transpose_state` (reading stored state as [D_v, D_k]) and `decay_values`
    (decaying D_v) are the two layout slips the tolerances must tell apart.
    """
    h, d, k_size = geom.num_heads, HEAD_DIM, KERNEL_SIZE
    x = _f64(acts["mixed_qkv"])
    w = _f64(weights["conv_weight"]).reshape(k_size, 3 * h * d)
    gate = _f64(acts["raw_gate"]).reshape(-1, h, d)
    beta = _sigmoid(_f64(acts["beta"]))
    out_gate = _sigmoid(_f64(acts["output_gate"]).reshape(-1, h, d))
    rate = np.exp(_f64(weights["a_log"]))[:, None]
    dt_bias = _f64(weights["dt_bias"]).reshape(h, d)
    norm_w = _f64(weights["norm_weight"])
    qsl = np.asarray(acts["query_start_loc"])
    seq_lens = np.asarray(acts["seq_lens"])
    conv0 = _f64(conv_init).reshape(-1, k_size - 1, 3 * h * d)
    ssm0 = _f64(ssm_init)

    out = np.zeros((x.shape[0], h, d))
    new_conv, new_ssm = [], []
    for s in range(seq_lens.shape[0]):
        lo, hi = int(qsl[s]), int(qsl[s + 1])
        if int(seq_lens[s]) == hi - lo:
            window, state = np.zeros((k_size - 1, 3 * h * d)), np.zeros((h, d, d))
        else:
            window, state = conv0[s], ssm0[s]
            if transpose_state:
                state = state.transpose(0, 2, 1)
        ext = np.concatenate([window, x[lo:hi]], axis=0)
        for i, t in enumerate(range(lo, hi)):
            pre = (w * ext[i : i + k_size]).sum(axis=0)
            q, k, v = (pre * _sigmoid(pre)).reshape(3, h, d)
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(d)
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
            decay = np.exp(LOWER_BOUND * _sigmoid(rate * (gate[t] + dt_bias)))
            state = state * (decay[:, None, :] if decay_values else decay[:, :, None])
            v_new = beta[t][:, None] * (v - np.einsum("hk,hkv->hv", k, state))
            state = state + np.einsum("hk,hv->hkv", k, v_new)
            o = np.einsum("hk,hkv->hv", q, state)
            o = o / np.sqrt((o * o).mean(-1, keepdims=True) + EPS) * norm_w
            out[t] = o * out_gate[t]
        if transpose_state:
            state = state.transpose(0, 2, 1)
        new_conv.append(ext[-(k_size - 1) :].reshape(k_size - 1, 3, h, d))
        new_ssm.append(state)
    return out, np.stack(new_conv), np.stack(new_ssm)


def _rel_err(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _bits(x):
    uint = jnp.uint16 if x.dtype.itemsize == 2 else jnp.uint32
    return np.asarray(jnp.asarray(x).view(uint))


# Relative (Frobenius) error bounds against the fp64 recurrence. The repo's
# KDA oracle (test_gdn_attention_v3_kda.kda_attention_ref) sits at 4-6e-3
# (output) and 1-5e-3 (state) from this reference, mostly from its bf16
# rounding of the conv output; the layout slips below sit at 0.5-1.4.
OUT_TOL = 2e-2
STATE_TOL = 1e-2

# One decode with history, one fresh prefill, one continuation. The query
# lengths sum to 32 (the serving padding), so the kernel's 32-token chunk
# divides evenly into its triangular blocks.
MIXED = ([1, 15, 16], [9, 15, 24], 1)


# --- CPU: region arithmetic and argument validation -------------------------


def test_rpa_pool_state_layout():
    for geom in (TP8, TP1):
        assert geom.layout.ssm_tokens == 64
        # 4.5 rows of fp32 conv state round up to a power of two.
        assert geom.layout.conv_tokens == 8
        assert geom.layout.required_tokens == 72
        # No padding rows in the recurrent region at either TP.
        assert geom.ssm_region_rows == geom.num_heads * HEAD_DIM
        assert geom.conv_region_rows * LANES >= geom.conv_elems


def test_pooled_rpa_needs_explicit_row_packing():
    """Left to infer, the op reads the RPA pool's head axis as packed tokens
    (8 tokens/row here) and cannot place a 128-token manager block on it."""
    rng = np.random.default_rng(2)
    weights = _weights(TP8, rng)
    acts = _activations(TP8, rng, [2], [2])
    rows = BLOCK_SIZE // TP8.kv_rows
    with pytest.raises(
        ValueError,
        match=f"does not split into whole pool blocks: {rows} rows vs pool "
        f"block of {BLOCK_SIZE}",
    ):
        _pooled(
            weights,
            acts,
            _garbage_pool(TP8),
            [1],
            _distribution(0, 1),
            pool_tokens_per_row=None,
        )


def test_pooled_rpa_rejects_block_smaller_than_state():
    """With one token per row the manager block is measured in tokens: a
    71-token block holds the recurrent state but not the conv state."""
    rng = np.random.default_rng(2)
    weights = _weights(TP8, rng)
    acts = _activations(TP8, rng, [2], [2])
    required = TP8.layout.required_tokens
    small_block = required - 1
    pool = jnp.zeros(TP8.pool_shape(block_size=small_block), jnp.bfloat16)
    with pytest.raises(
        ValueError, match=f"exceed the pool block: {required} > {small_block}"
    ):
        _pooled(
            weights,
            acts,
            pool,
            [1],
            _distribution(0, 1),
            pool_block_tokens=small_block,
        )


def _region_view(pool, split):
    """The pool's bytes as (manager blocks, manager rows, row bytes)."""
    raw = jax.lax.bitcast_convert_type(pool, jnp.uint8)
    return raw.reshape(pool.shape[0] // split, split * pool.shape[1], -1)


def _fake_gather_region(
    calls,
    pool,
    state_indices,
    *,
    tok0,
    ntok,
    out_dtype,
    out_lanes=None,
    split=1,
    num_valid=None,
    row_offset=0,
):
    """A pure-JAX stand-in for the Pallas gather.

    Reads manager-block rows `[tok0, tok0 + ntok)` as `out_dtype`. Its byte
    order inside a row differs from the Pallas kernel's in-VMEM bitcast, but
    it is the exact inverse of `_fake_scatter_region`, which is all the
    region arithmetic needs. Rows the real kernel leaves uninitialized (the
    leading `row_offset` rows and the requests past `num_valid`) come back
    as NaN poison so a caller that reads them is caught.
    """
    calls.append(("gather", tok0, ntok, split))
    n = state_indices.shape[0]
    rows = _region_view(pool, split)[state_indices, tok0 : tok0 + ntok]
    itemsize = jnp.dtype(out_dtype).itemsize
    vals = jax.lax.bitcast_convert_type(rows.reshape(n, -1, itemsize), out_dtype)
    vals = vals.reshape(n, -1, out_lanes or pool.shape[-1])
    if num_valid is not None:
        live = jnp.arange(n) < num_valid
        vals = jnp.where(live[:, None, None], vals, jnp.nan)
    poison = jnp.full((row_offset, *vals.shape[1:]), jnp.nan, vals.dtype)
    return jnp.concatenate([poison, vals], axis=0)


def _fake_scatter_region(
    calls,
    pool,
    vals,
    state_indices,
    *,
    tok0,
    ntok,
    split=1,
    num_valid=None,
    row_offset=0,
):
    calls.append(("scatter", tok0, ntok, split))
    n = state_indices.shape[0]
    raw = jax.lax.bitcast_convert_type(vals[row_offset:], jnp.uint8).reshape(
        n, ntok, -1
    )
    view = _region_view(pool, split)
    if num_valid is not None:
        live = jnp.arange(n) < num_valid
        raw = jnp.where(
            live[:, None, None], raw, view[state_indices, tok0 : tok0 + ntok]
        )
    view = view.at[state_indices, tok0 : tok0 + ntok].set(raw)
    itemsize = jnp.dtype(pool.dtype).itemsize
    return jax.lax.bitcast_convert_type(view.reshape(*pool.shape, itemsize), pool.dtype)


def _fake_build_fused_core(lower_bound, eps, num_spec_tokens=0):
    """A stand-in for the fused kernel: a recognizable state update."""
    del lower_bound, eps, num_spec_tokens

    def fused_core(mixed_qkv, raw_gate, beta, output_gate, conv, ssm, *args):
        del mixed_qkv, beta, output_gate, args
        return jnp.zeros_like(raw_gate), conv + 1.0, ssm * 2.0

    return fused_core


@pytest.mark.parametrize("split", [1, 4], ids=["one-kernel-block", "split4"])
def test_pooled_rpa_regions_use_one_token_per_row(monkeypatch, split):
    """With `pool_tokens_per_row=1` the state regions are token rows
    [0, 64) and [64, 72) of each manager block, measured on the RPA pool's
    token axis; they round-trip, and nothing else in the pool moves.

    The Pallas gather/scatter and the fused kernel are swapped for pure-JAX
    stand-ins, so this runs anywhere. With split 4 (32-token kernel blocks)
    the recurrent region spans two whole kernel blocks and the conv region
    sits inside the third.
    """
    calls = []
    monkeypatch.setattr(
        pool_adapters, "gather_region", functools.partial(_fake_gather_region, calls)
    )
    monkeypatch.setattr(
        pool_adapters, "scatter_region", functools.partial(_fake_scatter_region, calls)
    )
    monkeypatch.setattr(kda_attention_op, "_build_fused_core", _fake_build_fused_core)

    geom = TP8
    layout = geom.layout
    rng = np.random.default_rng(6)
    weights = _weights(geom, rng)
    query_lens, seq_lens, decode_end = MIXED
    acts = _activations(geom, rng, query_lens, seq_lens)
    block_ids = [1, 3, 4]
    conv_init, ssm_init = _initial_states(geom, rng, len(block_ids))

    pool = _pack_pool(geom, _garbage_pool(geom), conv_init, ssm_init, block_ids)
    pool = pool.reshape(NUM_BLOCKS * split, BLOCK_SIZE // split, *pool.shape[2:])
    before = _bits(pool).reshape(geom.pool_shape())
    calls.clear()
    _, pool_after = _pooled(
        weights,
        acts,
        pool,
        block_ids,
        _distribution(decode_end, len(block_ids)),
        use_jit=False,
    )

    ssm_region = (0, layout.ssm_tokens, split)
    conv_region = (layout.ssm_tokens, layout.conv_tokens, split)
    assert calls == [
        ("gather", *ssm_region),
        ("gather", *conv_region),
        ("scatter", *ssm_region),
        ("scatter", *conv_region),
    ]

    assert pool_after.shape == pool.shape
    conv_back, ssm_back, conv_padding = _unpack_pool(
        geom, pool_after, block_ids, split=split
    )
    np.testing.assert_array_equal(np.asarray(conv_back), np.asarray(conv_init + 1.0))
    np.testing.assert_array_equal(np.asarray(ssm_back), np.asarray(ssm_init * 2.0))
    assert not np.asarray(conv_padding.view(jnp.uint32)).any()

    after = _bits(pool_after).reshape(geom.pool_shape())
    for block in range(NUM_BLOCKS):
        if block not in block_ids:
            np.testing.assert_array_equal(after[block], before[block])
    for block in block_ids:
        np.testing.assert_array_equal(
            after[block, layout.required_tokens :],
            before[block, layout.required_tokens :],
        )


@pytest.mark.parametrize("num_spec_tokens", [0, 3], ids=["plain", "spec3"])
def test_fused_core_zeros_uninitialized_padded_tail_rows(monkeypatch, num_spec_tokens):
    """Padded bucket rows `[query_start_loc[-1], num_tokens)` left uninitialized
    by `fused_conv1d_gdn` (e.g. NaN / Inf recycled HBM bytes) must be zeroed
    before `_gated_output_norm` so they never poison SP MoE unpermute."""
    geom = TP8
    h, d = geom.num_heads, HEAD_DIM
    rng = np.random.default_rng(17)
    weights = _weights(geom, rng)
    real_tokens, padded_tokens = 5, 8

    def _fake_fused_conv1d_gdn(*, qkv, states, **kwargs):
        del qkv, kwargs
        valid = jnp.ones((real_tokens, h * d), dtype=jnp.bfloat16) * 2.0
        poison = jnp.full((padded_tokens - real_tokens, h * d), jnp.nan, jnp.bfloat16)
        return states, jnp.concatenate([valid, poison], axis=0)

    monkeypatch.setattr(
        kda_attention_op.gdn_wrapper, "fused_conv1d_gdn", _fake_fused_conv1d_gdn
    )
    core = _build_fused_core(LOWER_BOUND, EPS, num_spec_tokens=num_spec_tokens)

    mixed_qkv = _bf16(rng, (padded_tokens, 3 * h * d), 0.5)
    raw_gate = _bf16(rng, (padded_tokens, h * d), 1.0)
    beta = _bf16(rng, (padded_tokens, h), 1.0)
    gate_real = _bf16(rng, (real_tokens, h * d), 1.0)
    gate_tail = jnp.full((padded_tokens - real_tokens, h * d), jnp.nan, jnp.bfloat16)
    output_gate = jnp.concatenate([gate_real, gate_tail], axis=0)

    conv_init, ssm_init = _initial_states(geom, rng, 1)
    conv_buf = jnp.concatenate([jnp.zeros_like(conv_init[:1]), conv_init], axis=0)
    ssm_buf = jnp.concatenate([jnp.zeros_like(ssm_init[:1]), ssm_init], axis=0)
    query_start_loc = jnp.asarray([0, real_tokens], dtype=jnp.int32)
    state_indices = jnp.asarray([1], dtype=jnp.int32)
    seq_lens = jnp.asarray([real_tokens], dtype=jnp.int32)
    dist = _distribution(0, 1)
    offsets = jnp.zeros((2,), dtype=jnp.int32) if num_spec_tokens else None

    out, _, _ = core(
        mixed_qkv,
        raw_gate,
        beta,
        output_gate,
        conv_buf.reshape(2, KERNEL_SIZE - 1, 3 * h * d),
        ssm_buf,
        weights["conv_weight"],
        weights["a_log"],
        weights["dt_bias"],
        weights["norm_weight"],
        query_start_loc,
        state_indices,
        seq_lens,
        dist,
        dist,
        offsets,
    )
    assert out.shape == (padded_tokens, h, d)
    assert bool(jnp.all(jnp.isfinite(out)))
    np.testing.assert_array_equal(
        np.asarray(out[real_tokens:].astype(jnp.float32)),
        np.zeros((padded_tokens - real_tokens, h, d), dtype=np.float32),
    )


@pytest.mark.parametrize("num_spec_tokens", [0, 3], ids=["plain", "spec3"])
def test_pooled_trims_request_bucket_padding_to_token_bucket(
    monkeypatch, num_spec_tokens
):
    """A step holds at most `num_tokens` live requests, so the pooled op
    gathers/scatters only `min(num_reqs, num_tokens)` state regions (times
    the verify window) instead of the whole `max_num_seqs` request bucket,
    and the live requests' states still round-trip."""
    gathered = []

    def _gather(pool, state_indices, **kwargs):
        gathered.append(int(state_indices.shape[0]))
        return _fake_gather_region([], pool, state_indices, **kwargs)

    monkeypatch.setattr(pool_adapters, "gather_region", _gather)
    monkeypatch.setattr(
        pool_adapters, "scatter_region", functools.partial(_fake_scatter_region, [])
    )
    seen = {}

    def _fake_core(lower_bound, eps, num_spec_tokens=0):
        del lower_bound, eps, num_spec_tokens

        def fused_core(mixed_qkv, raw_gate, beta, output_gate, conv, ssm, *args):
            del mixed_qkv, beta, output_gate
            seen["qsl"], seen["state"], seen["seq_lens"], seen["dist"] = (
                args[4],
                args[5],
                args[6],
                args[7],
            )
            return jnp.zeros_like(raw_gate), conv + 1.0, ssm * 2.0

        return fused_core

    monkeypatch.setattr(kda_attention_op, "_build_fused_core", _fake_core)

    geom = TP8
    rng = np.random.default_rng(31)
    weights = _weights(geom, rng)
    num_tokens, bucket, live = 4, 16, 2
    window = num_spec_tokens + 1
    query_lens = [2, 2] + [0] * (bucket - live)
    seq_lens = [9, 5] + [0] * (bucket - live)
    acts = _activations(geom, rng, query_lens, seq_lens)
    block_ids = [1, 3] + [0] * (bucket - live)
    conv_init, ssm_init = _initial_states(geom, rng, live)
    pool = _pack_pool(geom, _garbage_pool(geom), conv_init, ssm_init, [1, 3])
    distribution = jnp.asarray([live, live, bucket], jnp.int32)
    extra = ()
    if num_spec_tokens:
        ckpt = np.zeros((bucket, window), np.int32)
        ckpt[0, 0], ckpt[1, 0] = 1, 3
        extra = (distribution, jnp.zeros((NUM_BLOCKS,), jnp.int32), jnp.asarray(ckpt))
    _, pool_after = kda_attention_op._pooled_kda_core(
        acts["mixed_qkv"],
        acts["raw_gate"],
        acts["beta"],
        acts["output_gate"],
        pool,
        weights["conv_weight"],
        weights["a_log"],
        weights["dt_bias"],
        weights["norm_weight"],
        acts["query_start_loc"],
        jnp.asarray(block_ids, jnp.int32),
        acts["seq_lens"],
        distribution,
        *extra,
        lower_bound=LOWER_BOUND,
        eps=EPS,
        pool_block_tokens=BLOCK_SIZE,
        num_spec_tokens=num_spec_tokens,
        pool_tokens_per_row=1,
    )
    assert gathered == [num_tokens * window, num_tokens * window]
    assert seen["seq_lens"].shape == (num_tokens,)
    assert seen["qsl"].shape == (num_tokens + 1,)
    assert int(seen["qsl"][-1]) == num_tokens
    assert np.asarray(seen["dist"]).max() <= num_tokens
    conv_back, ssm_back, _ = _unpack_pool(geom, pool_after, [1, 3])
    np.testing.assert_array_equal(np.asarray(conv_back), np.asarray(conv_init + 1.0))
    np.testing.assert_array_equal(np.asarray(ssm_back), np.asarray(ssm_init * 2.0))


@pytest.mark.parametrize("num_spec_tokens", [0, 3], ids=["plain", "spec3"])
def test_pooled_moves_only_live_request_rows(monkeypatch, num_spec_tokens):
    """The region gather/scatter visit only the `distribution[2]` live
    requests (times the verify window) and land them at dense row r + 1, so
    the scratch slot needs no concatenate and the scatter no slice. Padded
    request rows are never read back into the pool: their blocks keep their
    bytes, and the (poisoned) unvisited dense rows never reach the pool."""
    region_calls = []

    def _gather(pool, state_indices, **kwargs):
        region_calls.append(
            ("gather", kwargs.get("num_valid"), kwargs.get("row_offset"))
        )
        return _fake_gather_region([], pool, state_indices, **kwargs)

    def _scatter(pool, vals, state_indices, **kwargs):
        region_calls.append(
            ("scatter", kwargs.get("num_valid"), kwargs.get("row_offset"))
        )
        return _fake_scatter_region([], pool, vals, state_indices, **kwargs)

    monkeypatch.setattr(pool_adapters, "gather_region", _gather)
    monkeypatch.setattr(pool_adapters, "scatter_region", _scatter)
    seen = {}

    def _fake_core(lower_bound, eps, num_spec_tokens=0):
        del lower_bound, eps, num_spec_tokens

        def fused_core(mixed_qkv, raw_gate, beta, output_gate, conv, ssm, *args):
            del mixed_qkv, beta, output_gate, args
            seen["conv0"], seen["ssm0"] = conv[0], ssm[0]
            return jnp.zeros_like(raw_gate), conv + 1.0, ssm * 2.0

        return fused_core

    monkeypatch.setattr(kda_attention_op, "_build_fused_core", _fake_core)

    geom = TP8
    rng = np.random.default_rng(37)
    weights = _weights(geom, rng)
    bucket, live = 4, 2
    window = num_spec_tokens + 1
    query_lens = [1, 1] + [0] * (bucket - live)
    seq_lens = [9, 5] + [0] * (bucket - live)
    acts = _activations(geom, rng, query_lens, seq_lens)
    # Padded requests point at block 2, whose bytes must survive untouched.
    block_ids = [1, 3] + [2] * (bucket - live)
    conv_init, ssm_init = _initial_states(geom, rng, live)
    pool = _pack_pool(geom, _garbage_pool(geom), conv_init, ssm_init, [1, 3])
    before = _bits(pool).reshape(geom.pool_shape())
    region_calls.clear()  # _pack_pool scatters through the stand-in too
    distribution = jnp.asarray([live, live, live], jnp.int32)
    extra = ()
    if num_spec_tokens:
        ckpt = np.full((bucket, window), 2, np.int32)
        # Slot 0 of each live window is the request's state block; the other
        # checkpoint slots share a spare block (they round-trip identically).
        ckpt[0], ckpt[1] = 4, 0
        ckpt[0, 0], ckpt[1, 0] = 1, 3
        extra = (distribution, jnp.zeros((NUM_BLOCKS,), jnp.int32), jnp.asarray(ckpt))
    _, pool_after = kda_attention_op._pooled_kda_core(
        acts["mixed_qkv"],
        acts["raw_gate"],
        acts["beta"],
        acts["output_gate"],
        pool,
        weights["conv_weight"],
        weights["a_log"],
        weights["dt_bias"],
        weights["norm_weight"],
        acts["query_start_loc"],
        jnp.asarray(block_ids, jnp.int32),
        acts["seq_lens"],
        distribution,
        *extra,
        lower_bound=LOWER_BOUND,
        eps=EPS,
        pool_block_tokens=BLOCK_SIZE,
        num_spec_tokens=num_spec_tokens,
        pool_tokens_per_row=1,
    )
    assert [c[0] for c in region_calls] == ["gather", "gather", "scatter", "scatter"]
    for _, num_valid, row_offset in region_calls:
        assert int(num_valid) == live * window
        assert row_offset == 1
    # The scratch slot reaches the kernel zeroed, not as gather poison.
    assert not np.asarray(seen["conv0"]).any()
    assert not np.asarray(seen["ssm0"]).any()
    conv_back, ssm_back, _ = _unpack_pool(geom, pool_after, [1, 3])
    np.testing.assert_array_equal(np.asarray(conv_back), np.asarray(conv_init + 1.0))
    np.testing.assert_array_equal(np.asarray(ssm_back), np.asarray(ssm_init * 2.0))
    after = _bits(pool_after).reshape(geom.pool_shape())
    np.testing.assert_array_equal(after[2], before[2])


# --- TPU: the fused kernel and the Pallas pool adapters ---------------------


@pytest.mark.tpu_test
def test_dense_matches_fp64_reference():
    geom = TP8
    query_lens, seq_lens, decode_end = MIXED
    num_seqs = len(query_lens)
    rng = np.random.default_rng(0)
    weights = _weights(geom, rng)
    acts = _activations(geom, rng, query_lens, seq_lens)
    conv_init, ssm_init = _initial_states(geom, rng, num_seqs)
    distribution = _distribution(decode_end, num_seqs)

    out, conv, ssm = _dense(geom, weights, acts, conv_init, ssm_init, distribution)
    ref_out, ref_conv, ref_ssm = _reference(geom, weights, acts, conv_init, ssm_init)

    out_err = _rel_err(_f64(out), ref_out)
    assert out_err < OUT_TOL, out_err
    np.testing.assert_allclose(_f64(out), ref_out, rtol=5e-2, atol=1e-2)
    # The conv state holds raw bf16 inputs in fp32: exact.
    np.testing.assert_array_equal(_f64(conv), ref_conv)
    ssm_err = _rel_err(_f64(ssm), ref_ssm)
    assert ssm_err < STATE_TOL, ssm_err

    # The bounds discriminate: both classic layout slips land far outside.
    for slip in (dict(transpose_state=True), dict(decay_values=True)):
        slip_out, _, slip_ssm = _reference(
            geom, weights, acts, conv_init, ssm_init, **slip
        )
        assert _rel_err(_f64(out), slip_out) > 10 * OUT_TOL, slip
        assert _rel_err(_f64(ssm), slip_ssm) > 10 * STATE_TOL, slip


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("query_lens", "seq_lens", "decode_end"),
    [
        MIXED,
        # Decode-only: the prefill segment is skipped.
        ([1, 1], [4, 7], 2),
        # Prefill-only, all fresh: stale state in the blocks is never read.
        ([6, 2], [6, 2], 0),
    ],
    ids=["mixed", "decode-only", "prefill-fresh"],
)
def test_pooled_rpa_matches_dense(query_lens, seq_lens, decode_end):
    geom = TP8
    num_seqs = len(query_lens)
    rng = np.random.default_rng(1)
    weights = _weights(geom, rng)
    acts = _activations(geom, rng, query_lens, seq_lens)
    conv_init, ssm_init = _initial_states(geom, rng, num_seqs)
    distribution = _distribution(decode_end, num_seqs)
    block_ids = [1, 2, 3][:num_seqs]

    out_dense, conv_dense, ssm_dense = _dense(
        geom, weights, acts, conv_init, ssm_init, distribution
    )

    pool = _pack_pool(geom, _garbage_pool(geom), conv_init, ssm_init, block_ids)
    before = _bits(pool)
    out_pool, pool_after = _pooled(weights, acts, pool, block_ids, distribution)

    # Same kernel on the same values, but the pooled graph feeds it gathered
    # buffers, so XLA may reassociate the f32 sums around it.
    np.testing.assert_allclose(_f64(out_pool), _f64(out_dense), rtol=5e-2, atol=4e-3)
    conv_back, ssm_back, conv_padding = _unpack_pool(geom, pool_after, block_ids)
    # The conv state update is pure data movement: bitwise equal.
    np.testing.assert_array_equal(_bits(conv_back), _bits(conv_dense))
    np.testing.assert_allclose(_f64(ssm_back), _f64(ssm_dense), rtol=1e-2, atol=5e-4)
    # The conv region's padding is zero-filled, so a block's bytes depend
    # only on its state (what a prefix-cache copy relies on).
    assert not np.asarray(conv_padding.view(jnp.uint32)).any()

    # Nothing outside the state regions moves: block 0, unlisted blocks, and
    # the attention rows of the listed blocks.
    after = _bits(pool_after)
    for block in range(NUM_BLOCKS):
        if block not in block_ids:
            np.testing.assert_array_equal(after[block], before[block])
    region_end = geom.layout.required_tokens
    for block in block_ids:
        np.testing.assert_array_equal(
            after[block, region_end:], before[block, region_end:]
        )


@pytest.mark.tpu_test
def test_pooled_rpa_manager_block_split():
    """A pool born at kernel-block granularity: 128-token manager blocks
    over 32-token kernel blocks (split 4). The recurrent region spans two
    whole kernel blocks (window path), the conv region sits inside the third
    (offset path). Byte-identical reshape of the split=1 pool, so both agree.
    """
    geom = TP8
    query_lens, seq_lens, decode_end = MIXED
    num_seqs = len(query_lens)
    rng = np.random.default_rng(3)
    weights = _weights(geom, rng)
    acts = _activations(geom, rng, query_lens, seq_lens)
    conv_init, ssm_init = _initial_states(geom, rng, num_seqs)
    distribution = _distribution(decode_end, num_seqs)
    block_ids = [1, 2, 3]

    pool = _pack_pool(geom, _garbage_pool(geom), conv_init, ssm_init, block_ids)
    out_ref, pool_ref = _pooled(weights, acts, pool, block_ids, distribution)

    split, kernel_block = 4, BLOCK_SIZE // 4
    pool_split = pool.reshape(NUM_BLOCKS * split, kernel_block, *pool.shape[2:])
    out_split, pool_split_after = _pooled(
        weights, acts, pool_split, block_ids, distribution
    )

    np.testing.assert_allclose(_f64(out_split), _f64(out_ref), rtol=5e-2, atol=4e-3)
    conv_split, ssm_split, _ = _unpack_pool(
        geom, pool_split_after, block_ids, split=split
    )
    conv_ref, ssm_ref, _ = _unpack_pool(geom, pool_ref, block_ids)
    np.testing.assert_array_equal(_bits(conv_split), _bits(conv_ref))
    np.testing.assert_allclose(_f64(ssm_split), _f64(ssm_ref), rtol=1e-2, atol=5e-4)

    after = _bits(pool_split_after).reshape(_bits(pool).shape)
    seed = _bits(pool)
    for block in range(NUM_BLOCKS):
        if block not in block_ids:
            np.testing.assert_array_equal(after[block], seed[block])


@pytest.mark.tpu_test
def test_pooled_rpa_multistep_state_evolution():
    """Two prefill chunks and three decodes of one request. The pooled path
    threads only the pool, so every step's state crosses the gather/scatter
    seam; the dense path and the fp64 recurrence thread their own state."""
    geom = TP8
    rng = np.random.default_rng(4)
    weights = _weights(geom, rng)
    steps = [(128, 128, 0), (96, 224, 0), (1, 225, 1), (1, 226, 1), (1, 227, 1)]
    conv_init, ssm_init = _initial_states(geom, rng, 1)

    conv_dense, ssm_dense = conv_init, ssm_init
    conv_ref, ssm_ref = _f64(conv_init), _f64(ssm_init)
    pool = _pack_pool(geom, _garbage_pool(geom), conv_init, ssm_init, [1])
    for step, (query_len, seq_len, decode_end) in enumerate(steps):
        acts = _activations(geom, rng, [query_len], [seq_len])
        distribution = _distribution(decode_end, 1)

        out_dense, conv_dense, ssm_dense = _dense(
            geom, weights, acts, conv_dense, ssm_dense, distribution
        )
        out_pool, pool = _pooled(weights, acts, pool, [1], distribution)
        ref_out, conv_ref, ssm_ref = _reference(geom, weights, acts, conv_ref, ssm_ref)

        np.testing.assert_allclose(
            _f64(out_pool),
            _f64(out_dense),
            rtol=5e-2,
            atol=4e-3,
            err_msg=f"pooled vs dense diverged at step {step} {steps[step]}",
        )
        err = _rel_err(_f64(out_pool), ref_out)
        assert err < OUT_TOL, (step, err)

    conv_back, ssm_back, _ = _unpack_pool(geom, pool, [1])
    np.testing.assert_array_equal(_bits(conv_back), _bits(conv_dense))
    np.testing.assert_array_equal(_f64(conv_back), conv_ref)
    np.testing.assert_allclose(_f64(ssm_back), _f64(ssm_dense), rtol=1e-2, atol=5e-4)
    assert _rel_err(_f64(ssm_back), ssm_ref) < STATE_TOL


@pytest.mark.tpu_test
def test_prefix_cache_hit_matches_miss():
    """`--mamba-cache-mode align`: the cached state sits on a block boundary.

    Request A prefills a block-aligned prefix in block 1. A hit copies block
    1 into its own block 2 (the unified pool's state copy is `copy_blocks`)
    and computes only the suffix; a miss recomputes the prefix in block 3 and
    then the suffix. Same kernel, same shapes, same state bytes: the miss's
    prefix reproduces A bit for bit, the hit and miss leave identical state
    bytes, and the cached block is left as it was.
    """
    geom = TP8
    rng = np.random.default_rng(5)
    weights = _weights(geom, rng)
    prefix_len, suffix_len = BLOCK_SIZE, 32
    prefix = _activations(geom, rng, [prefix_len], [prefix_len])
    suffix = _activations(geom, rng, [suffix_len], [prefix_len + suffix_len])
    prefill = _distribution(0, 1)

    pool = _garbage_pool(geom)
    out_a, pool = _pooled(weights, prefix, pool, [1], prefill)
    cached = _bits(pool)[1]

    pool = pool_adapters.copy_blocks(
        pool, jnp.asarray([1], jnp.int32), jnp.asarray([2], jnp.int32)
    )
    out_hit, pool = _pooled(weights, suffix, pool, [2], prefill)

    out_miss_prefix, pool = _pooled(weights, prefix, pool, [3], prefill)
    out_miss, pool = _pooled(weights, suffix, pool, [3], prefill)

    np.testing.assert_array_equal(_bits(out_miss_prefix), _bits(out_a))
    # The suffix outputs agree up to floating-point rounding.
    np.testing.assert_allclose(_f64(out_hit), _f64(out_miss), rtol=5e-3, atol=2e-3)
    region = slice(0, geom.layout.required_tokens)
    final = _bits(pool)
    np.testing.assert_array_equal(final[2, region], final[3, region])
    np.testing.assert_array_equal(final[1], cached)

    # Control: the suffix from a block holding some other state differs, so
    # the equality above is not vacuous.
    out_stale, _ = _pooled(weights, suffix, pool, [4], prefill)
    assert not np.array_equal(_bits(out_stale), _bits(out_hit))

    # And the hit is right, not just consistent: the fp64 recurrence over the
    # whole sequence.
    zeros_conv, zeros_ssm = _initial_states(geom, rng, 1)  # Ignored: fresh.
    _, conv_ref, ssm_ref = _reference(geom, weights, prefix, zeros_conv, zeros_ssm)
    ref_out, _, _ = _reference(geom, weights, suffix, conv_ref, ssm_ref)
    assert _rel_err(_f64(out_hit), ref_out) < OUT_TOL
