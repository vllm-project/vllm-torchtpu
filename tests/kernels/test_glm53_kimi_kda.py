# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 KDA on the Kimi ops, at GLM-5.3-Flash geometry.

GLM-5.3's KDA layers run on the same fused conv1d + GDN v3 ops as Kimi-Linear
and Kimi-K3 (`GLM53_KDA_BACKEND=kimi`). Two things are GLM-specific and are
what this file covers:

* The geometry: 64 KDA heads of 128 channels, a width-4 short convolution and
  a bounded gate (`linear_lower_bound = -5`). The dense path is checked
  against an fp64 recurrence written from the model definition.
* The pool: GLM-5.3's attention layers are dense, so the unified pool its KDA
  layers share is the RPA v3 cache `(blocks, block_size, kv_heads_x2 //
  packing, packing, head_dim)`, which holds one token per row. The Kimi models
  share an MLA pool that packs two tokens per row, and the pooled op infers
  that packing from `pool.shape[2]`; on the RPA pool the layer passes
  `pool_tokens_per_row=1` instead. The pooled path is checked against the
  dense one, and prefix-cache hits against misses.

With 64 attention KV heads (x2 for K and V, /2 for bf16 packing) of 256, a
token row is 64 KiB at TP1 and 8 KiB at TP8. Either way the recurrent state
fills 64 rows and the conv state pads to 4, so a manager block must hold at
least 68 tokens.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    _build_fused_core,
    _pooled_kda_core,
)

HEAD_DIM, KERNEL_SIZE = 128, 4


@pytest.fixture(autouse=True)
def _require_tpu_unless_cpu_test(request):
    """Skip the Pallas-kernel tests off TPU; `cpu_test` tests run anywhere.

    A fixture rather than a `skipif`: it runs at setup, after `-m`
    deselection, so collecting this module never builds a PJRT client (see
    `tests/kernels/fused_moe/test_fused_ep_moe_v2.py`). CI already runs the
    unmarked tests on TPU agents only; this keeps local CPU runs green too.
    """
    is_cpu_test = request.node.get_closest_marker("cpu_test") is not None
    if not is_cpu_test and jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel test; needs a TPU backend")


LOWER_BOUND = -5.0  # GLM-5.3 `linear_lower_bound`.
EPS = 1e-5  # The layer's `o_norm` eps.
PACK, LANES = 2, 256  # RPA v3 bf16 packing; GLM's attention head size.
NUM_BLOCKS, BLOCK_SIZE = 5, 128  # Manager block of 128 tokens >= 68.


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One tensor-parallel shard of GLM-5.3-Flash's KDA and attention."""

    num_heads: int  # KDA heads per shard.
    kv_rows: int  # RPA `kv_heads_x2 // packing` per shard.

    @property
    def token_bytes(self) -> int:
        return self.kv_rows * PACK * LANES * 2

    @property
    def layout(self):
        return derive_pooled_gdn_state_layout(
            ssm_bytes=self.num_heads * HEAD_DIM * HEAD_DIM * 4,
            conv_bytes=(KERNEL_SIZE - 1) * 3 * self.num_heads * HEAD_DIM * 2,
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
        return self.layout.conv_tokens * self.token_bytes // (2 * LANES)

    def pool_shape(self, num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE):
        return (num_blocks, block_size, self.kv_rows, PACK, LANES)


TP8 = Geometry(num_heads=8, kv_rows=8)
TP1 = Geometry(num_heads=64, kv_rows=64)


def _bf16(rng, shape, scale):
    return jnp.asarray(
        rng.standard_normal(shape).astype(np.float32) * scale, dtype=jnp.bfloat16
    )


def _weights(geom, rng):
    """The layer's parameters, in the dtypes the GLM layer hands the ops."""
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
    """Per-request states. The conv state is bf16 values, as in the pool."""
    h, d = geom.num_heads, HEAD_DIM
    conv = _bf16(rng, (num_seqs, KERNEL_SIZE - 1, 3, h, d), 0.5)
    ssm = jnp.asarray(rng.standard_normal((num_seqs, h, d, d)).astype(np.float32) * 0.1)
    return conv, ssm


def _distribution(decode_end, num_seqs):
    return jnp.asarray([decode_end, num_seqs, num_seqs], jnp.int32)


@functools.cache
def _dense_core():
    return jax.jit(_build_fused_core(LOWER_BOUND, EPS))


def _dense(geom, weights, acts, conv_init, ssm_init, distribution, conv_dtype):
    """The dense (per-layer cache) path over null-slot-prepended buffers.

    GLM declares its dense conv cache as f32 `(K - 1, 1, 3 * H * D)` slots
    (the fused kernel's compact layout, so it is read in place); `conv_dtype`
    bf16 gives the pooled path's gathered buffer instead.
    """
    h, d = geom.num_heads, HEAD_DIM
    num_seqs = acts["seq_lens"].shape[0]
    conv = conv_init.astype(conv_dtype)
    conv_buf = jnp.concatenate([jnp.zeros_like(conv[:1]), conv], axis=0)
    ssm_buf = jnp.concatenate([jnp.zeros_like(ssm_init[:1]), ssm_init], axis=0)
    output, new_conv, new_ssm = _dense_core()(
        acts["mixed_qkv"],
        acts["raw_gate"],
        acts["beta"],
        acts["output_gate"],
        conv_buf.reshape(num_seqs + 1, KERNEL_SIZE - 1, 1, 3 * h * d),
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
def _pooled_core(pool_block_tokens, pool_tokens_per_row):
    return jax.jit(
        functools.partial(
            _pooled_kda_core,
            lower_bound=LOWER_BOUND,
            eps=EPS,
            pool_block_tokens=pool_block_tokens,
            pool_tokens_per_row=pool_tokens_per_row,
        )
    )


def _pooled(weights, acts, pool, block_ids, distribution, pool_tokens_per_row=1):
    return _pooled_core(BLOCK_SIZE, pool_tokens_per_row)(
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


def _garbage_pool(geom, num_blocks=NUM_BLOCKS, seed=99):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal(geom.pool_shape(num_blocks)).astype(np.float32),
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
        out_dtype=jnp.bfloat16,
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
    """fp64 GLM-5.3 KDA, one token at a time, from the model definition.

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
# TPU-validated KDA oracle (test_gdn_attention_v3_kda.kda_attention_ref) sits
# at 4-6e-3 (output) and 1-5e-3 (state) from this reference, mostly from its
# bf16 rounding of the conv output; the layout slips below sit at 0.5-1.4.
OUT_TOL = 2e-2
STATE_TOL = 1e-2

# One decode with history, one fresh prefill, one continuation.
# Sum of query lengths is 32 (matching serving padding), ensuring chunk_size (32)
# divides evenly into triangular_block_size across both TP1 (2) and TP8 (16).
MIXED = ([1, 15, 16], [9, 15, 24], 1)


@pytest.mark.parametrize(
    "geom",
    [
        TP8,
        pytest.param(
            TP1,
            marks=pytest.mark.xfail(
                reason=(
                    "TP=1 is unsupported: 64 KDA heads per shard exceed TPU v7x "
                    "VMEM in the fused kernel (docs/models/glm53_flash.md, Known "
                    "limitations). Serving uses TP=8."
                ),
                raises=jax.errors.JaxRuntimeError,
                strict=True,
            ),
        ),
    ],
    ids=["tp8", "tp1"],
)
def test_dense_matches_fp64_reference(geom):
    query_lens, seq_lens, decode_end = MIXED
    num_seqs = len(query_lens)
    rng = np.random.default_rng(0)
    weights = _weights(geom, rng)
    acts = _activations(geom, rng, query_lens, seq_lens)
    conv_init, ssm_init = _initial_states(geom, rng, num_seqs)
    distribution = _distribution(decode_end, num_seqs)

    out, conv, ssm = _dense(
        geom, weights, acts, conv_init, ssm_init, distribution, jnp.float32
    )
    ref_out, ref_conv, ref_ssm = _reference(geom, weights, acts, conv_init, ssm_init)

    out_err = _rel_err(_f64(out), ref_out)
    assert out_err < OUT_TOL, out_err
    np.testing.assert_allclose(_f64(out), ref_out, rtol=5e-2, atol=1e-2)
    # The conv state holds raw inputs: exact.
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

    # The pooled path stores the conv state as bf16. It holds bf16 inputs,
    # so the dense f32 cache and a bf16 one agree exactly.
    out_bf16, conv_bf16, ssm_bf16 = _dense(
        geom, weights, acts, conv_init, ssm_init, distribution, jnp.bfloat16
    )
    np.testing.assert_array_equal(_f64(conv_bf16), _f64(conv))
    np.testing.assert_allclose(_f64(out_bf16), _f64(out), rtol=1e-2, atol=2e-3)
    np.testing.assert_allclose(_f64(ssm_bf16), _f64(ssm), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    ("geom", "query_lens", "seq_lens", "decode_end"),
    [
        (TP8, *MIXED),
        # Decode-only: the prefill segment is skipped.
        (TP8, [1, 1], [4, 7], 2),
        # Prefill-only, all fresh: stale state in the blocks is never read.
        (TP8, [6, 2], [6, 2], 0),
        pytest.param(
            TP1,
            *MIXED,
            marks=pytest.mark.xfail(
                reason=(
                    "TP=1 is unsupported: 64 KDA heads per shard exceed TPU v7x "
                    "VMEM in the fused kernel (docs/models/glm53_flash.md, Known "
                    "limitations). Serving uses TP=8."
                ),
                raises=jax.errors.JaxRuntimeError,
                strict=True,
            ),
        ),
    ],
    ids=["tp8-mixed", "tp8-decode-only", "tp8-prefill-fresh", "tp1-mixed"],
)
def test_pooled_rpa_matches_dense(geom, query_lens, seq_lens, decode_end):
    num_seqs = len(query_lens)
    rng = np.random.default_rng(1)
    weights = _weights(geom, rng)
    acts = _activations(geom, rng, query_lens, seq_lens)
    conv_init, ssm_init = _initial_states(geom, rng, num_seqs)
    distribution = _distribution(decode_end, num_seqs)
    block_ids = [1, 2, 3][:num_seqs]

    out_dense, conv_dense, ssm_dense = _dense(
        geom, weights, acts, conv_init, ssm_init, distribution, jnp.float32
    )

    pool = _pack_pool(geom, _garbage_pool(geom), conv_init, ssm_init, block_ids)
    before = _bits(pool)
    out_pool, pool_after = _pooled(weights, acts, pool, block_ids, distribution)

    # Same kernel on the same values, but the pooled graph feeds it gathered
    # buffers, so XLA may reassociate the f32 sums around it.
    np.testing.assert_allclose(_f64(out_pool), _f64(out_dense), rtol=5e-2, atol=4e-3)
    conv_back, ssm_back, conv_padding = _unpack_pool(geom, pool_after, block_ids)
    # Conv state is data movement; the dense f32 cache holds the same values.
    np.testing.assert_array_equal(_f64(conv_back), _f64(conv_dense))
    np.testing.assert_allclose(_f64(ssm_back), _f64(ssm_dense), rtol=1e-2, atol=5e-4)
    # The conv region's padding is zero-filled, so a block's bytes depend
    # only on its state (what a prefix-cache copy relies on).
    assert not np.asarray(conv_padding.view(jnp.uint16)).any()

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


@pytest.mark.cpu_test
def test_rpa_pool_state_layout():
    for geom in (TP8, TP1):
        assert geom.layout.ssm_tokens == 64
        assert geom.layout.conv_tokens == 4
        # No padding rows in the recurrent region at either TP.
        assert geom.ssm_region_rows == geom.num_heads * HEAD_DIM


@pytest.mark.cpu_test
def test_pooled_rpa_needs_explicit_row_packing():
    """Left to infer, the op reads the RPA pool's head axis as packed tokens
    and cannot place a 128-token manager block on it."""
    rng = np.random.default_rng(2)
    weights = _weights(TP8, rng)
    acts = _activations(TP8, rng, [2], [2])
    with pytest.raises(ValueError, match="does not split into whole pool"):
        _pooled(
            weights,
            acts,
            _garbage_pool(TP8),
            [1],
            _distribution(0, 1),
            pool_tokens_per_row=None,
        )


@pytest.mark.cpu_test
def test_pooled_rpa_rejects_block_smaller_than_state():
    rng = np.random.default_rng(2)
    weights = _weights(TP8, rng)
    acts = _activations(TP8, rng, [2], [2])
    small_block = TP8.layout.required_tokens - 4  # 64 tokens: SSM fits, conv not.
    pool = jnp.zeros(TP8.pool_shape(block_size=small_block), jnp.bfloat16)
    with pytest.raises(ValueError, match="exceed the pool block"):
        _pooled_core(small_block, 1)(
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
            jnp.asarray([1], jnp.int32),
            acts["seq_lens"],
            _distribution(0, 1),
        )


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
            geom, weights, acts, conv_dense, ssm_dense, distribution, jnp.float32
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
    np.testing.assert_array_equal(_f64(conv_back), _f64(conv_dense))
    np.testing.assert_array_equal(_f64(conv_back), conv_ref)
    np.testing.assert_allclose(_f64(ssm_back), _f64(ssm_dense), rtol=1e-2, atol=5e-4)
    assert _rel_err(_f64(ssm_back), ssm_ref) < STATE_TOL


def test_prefix_cache_hit_matches_miss_bitwise():
    """`--mamba-cache-mode align`: the cached state sits on a block boundary.

    Request A prefills a block-aligned prefix in block 1. A hit copies block
    1 into its own block 2 (the unified pool's state copy is `copy_blocks`)
    and computes only the suffix; a miss recomputes the prefix in block 3 and
    then the suffix. Same kernel, same shapes, same state bytes: the hit's
    suffix output and state must equal the miss's bit for bit, and the cached
    block must be left as it was.
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
    # Suffix output is numerically equal across hit and miss up to floating-point
    # rounding.
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
