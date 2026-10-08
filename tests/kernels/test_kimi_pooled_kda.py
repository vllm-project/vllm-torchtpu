# SPDX-License-Identifier: Apache-2.0
"""Pooled (unified-KV-pool) KDA path against the dense two-cache path.

Phase 1.1 of KIMI_LINEAR_POOLED_APC_PLAN.md: the pooled op gathers each
request's conv/SSM regions out of its pool block into dense batch-local
buffers, runs the fused conv1d + GDN v3 kernel, and scatters the updated
states back. The dense reference is the same fused core with identity
indices over null-slot-prepended buffers: identical values into the same
kernel, so the paths agree up to any reassociation XLA applies around the
gather-fed buffers (the conv state update is pure data movement and stays
bitwise exact).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    _build_fused_core,
    _pooled_kda_core,
)

# Kimi-Linear TP8 shard geometry on an MLA pool.
NUM_HEADS, HEAD_DIM, KERNEL_SIZE = 4, 128, 4
NUM_BLOCKS, BLOCK_SIZE, PACK, LANES = 5, 107, 2, 640
TOK_BYTES = PACK * LANES * 2  # bf16; an fp8 pool packs 4 tokens in the same row

SSM_ELEMS = NUM_HEADS * HEAD_DIM * HEAD_DIM
CONV_ELEMS = (KERNEL_SIZE - 1) * 3 * NUM_HEADS * HEAD_DIM
LAYOUT = derive_pooled_gdn_state_layout(
    ssm_bytes=SSM_ELEMS * 4, conv_bytes=CONV_ELEMS * 2, token_bytes=TOK_BYTES
)
SSM_REGION_ROWS = LAYOUT.ssm_tokens * TOK_BYTES // (4 * HEAD_DIM)
CONV_REGION_ROWS = LAYOUT.conv_tokens * TOK_BYTES // (2 * LANES)

LOWER_BOUND = None
EPS = 1e-5


def _inputs(rng, query_lens, seq_lens):
    """Random ragged batch; request 0 decodes, the rest prefill."""
    num_tokens = int(sum(query_lens))

    def bf16(shape):
        return jnp.asarray(
            rng.standard_normal(shape).astype(np.float32) * 0.5, dtype=jnp.bfloat16
        )

    return dict(
        mixed_qkv=bf16((num_tokens, 3 * NUM_HEADS * HEAD_DIM)),
        raw_gate=bf16((num_tokens, NUM_HEADS * HEAD_DIM)),
        beta=bf16((num_tokens, NUM_HEADS)),
        output_gate=bf16((num_tokens, NUM_HEADS * HEAD_DIM)),
        conv_weight=bf16((KERNEL_SIZE, 3, NUM_HEADS, HEAD_DIM)),
        a_log=jnp.asarray(
            rng.standard_normal(NUM_HEADS).astype(np.float32), dtype=jnp.float32
        ),
        dt_bias=jnp.asarray(
            rng.standard_normal(NUM_HEADS * HEAD_DIM).astype(np.float32),
            dtype=jnp.float32,
        ),
        norm_weight=bf16((HEAD_DIM,)),
        query_start_loc=jnp.asarray([0] + list(np.cumsum(query_lens)), dtype=jnp.int32),
        seq_lens=jnp.asarray(seq_lens, dtype=jnp.int32),
    )


def _initial_states(rng, num_seqs):
    conv = jnp.asarray(
        rng.standard_normal((num_seqs, KERNEL_SIZE - 1, 3, NUM_HEADS, HEAD_DIM)).astype(
            np.float32
        )
        * 0.5,
        dtype=jnp.bfloat16,
    )
    ssm = jnp.asarray(
        rng.standard_normal((num_seqs, NUM_HEADS, HEAD_DIM, HEAD_DIM)).astype(
            np.float32
        )
        * 0.1,
        dtype=jnp.float32,
    )
    return conv, ssm


def _pool_pack(pool_dtype):
    """Tokens one packed row holds: the packing axis fills a 32-bit word."""
    return 4 // jnp.dtype(pool_dtype).itemsize


def _raw(pool):
    """The pool's bytes, for exact before/after comparisons."""
    raw_dtype = jnp.uint16 if jnp.dtype(pool.dtype).itemsize == 2 else jnp.uint8
    return np.asarray(pool.view(raw_dtype))


def _pack_pool(
    conv_init,
    ssm_init,
    block_ids,
    conv_shard_heads=None,
    pool_dtype=jnp.bfloat16,
    num_blocks=NUM_BLOCKS,
):
    """Write the given states into the pool's state regions; the rest of the
    pool (including block 0 and the attention rows of every block) keeps its
    sentinel bytes."""
    rng = np.random.default_rng(99)
    # Keep the byte capacity fixed; typed scatter packs BF16/FP32 state.
    pool = jnp.asarray(
        rng.standard_normal(
            (num_blocks, BLOCK_SIZE, _pool_pack(pool_dtype), LANES)
        ).astype(np.float32),
        dtype=pool_dtype,
    )
    idx = jnp.asarray(block_ids, dtype=jnp.int32)

    shard_heads = NUM_HEADS if conv_shard_heads is None else conv_shard_heads
    capacity = CONV_REGION_ROWS * LANES // (NUM_HEADS // shard_heads)
    chunks = []
    for start in range(0, NUM_HEADS, shard_heads):
        chunk = conv_init[:, :, :, start : start + shard_heads].reshape(
            len(block_ids), -1
        )
        chunks.append(jnp.pad(chunk, ((0, 0), (0, capacity - chunk.shape[1]))))
    conv_vals = jnp.concatenate(chunks, axis=1).reshape(
        len(block_ids), CONV_REGION_ROWS, LANES
    )
    pool = pool_adapters.scatter_region(
        pool, conv_vals, idx, tok0=LAYOUT.ssm_tokens, ntok=LAYOUT.conv_tokens
    )

    ssm_vals = ssm_init.reshape(len(block_ids), NUM_HEADS * HEAD_DIM, HEAD_DIM)
    ssm_vals = jnp.pad(
        ssm_vals, ((0, 0), (0, SSM_REGION_ROWS - NUM_HEADS * HEAD_DIM), (0, 0))
    )
    pool = pool_adapters.scatter_region(
        pool, ssm_vals, idx, tok0=0, ntok=LAYOUT.ssm_tokens
    )
    return pool


def _unpack_pool(pool, block_ids, split=1, conv_shard_heads=None):
    idx = jnp.asarray(block_ids, dtype=jnp.int32)
    conv = pool_adapters.gather_region(
        pool,
        idx,
        tok0=LAYOUT.ssm_tokens,
        ntok=LAYOUT.conv_tokens,
        out_dtype=jnp.bfloat16,
        split=split,
    )
    shard_heads = NUM_HEADS if conv_shard_heads is None else conv_shard_heads
    fragments = NUM_HEADS // shard_heads
    conv = conv.reshape(len(block_ids), fragments, -1)
    live = (KERNEL_SIZE - 1) * 3 * shard_heads * HEAD_DIM
    conv = jnp.concatenate(
        [
            conv[:, i, :live].reshape(
                len(block_ids), KERNEL_SIZE - 1, 3, shard_heads, HEAD_DIM
            )
            for i in range(fragments)
        ],
        axis=3,
    )
    ssm = pool_adapters.gather_region(
        pool,
        idx,
        tok0=0,
        ntok=LAYOUT.ssm_tokens,
        out_dtype=jnp.float32,
        out_lanes=HEAD_DIM,
        split=split,
    )
    ssm = ssm[:, : NUM_HEADS * HEAD_DIM, :].reshape(
        len(block_ids), NUM_HEADS, HEAD_DIM, HEAD_DIM
    )
    return conv, ssm


def _fused_kda_core_dense(inputs, conv_init, ssm_init, distribution):
    """The dense reference: the fused conv1d + GDN v3 core over
    null-slot-prepended dense buffers, exactly what the pooled op runs after
    gathering its regions (including the flat conv-state layout the kernel
    expects)."""
    num_seqs = inputs["seq_lens"].shape[0]
    conv_buf = jnp.concatenate([jnp.zeros_like(conv_init[:1]), conv_init], axis=0)
    ssm_buf = jnp.concatenate([jnp.zeros_like(ssm_init[:1]), ssm_init], axis=0)
    identity = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    output, new_conv, new_ssm = _build_fused_core(LOWER_BOUND, EPS)(
        inputs["mixed_qkv"],
        inputs["raw_gate"],
        inputs["beta"],
        inputs["output_gate"],
        conv_buf.reshape(num_seqs + 1, KERNEL_SIZE - 1, -1),
        ssm_buf,
        inputs["conv_weight"],
        inputs["a_log"],
        inputs["dt_bias"],
        inputs["norm_weight"],
        inputs["query_start_loc"],
        identity,
        inputs["seq_lens"],
        distribution,
    )
    return (
        output,
        new_conv.reshape(num_seqs + 1, KERNEL_SIZE - 1, 3, NUM_HEADS, HEAD_DIM),
        new_ssm,
    )


def _run_pooled(
    inputs, pool, block_ids, distribution, pool_block_tokens, conv_shard_heads=None
):
    return _pooled_kda_core(
        inputs["mixed_qkv"],
        inputs["raw_gate"],
        inputs["beta"],
        inputs["output_gate"],
        pool,
        inputs["conv_weight"],
        inputs["a_log"],
        inputs["dt_bias"],
        inputs["norm_weight"],
        inputs["query_start_loc"],
        jnp.asarray(block_ids, dtype=jnp.int32),
        inputs["seq_lens"],
        distribution,
        lower_bound=LOWER_BOUND,
        eps=EPS,
        pool_block_tokens=pool_block_tokens,
        conv_shard_heads=conv_shard_heads,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("query_lens", "seq_lens", "decode_end", "conv_shard_heads"),
    [
        # One decode with history, one fresh prefill, one continuation.
        ([1, 5, 3], [9, 5, 11], 1, None),
        # Decode-only batch: the prefill segment is skipped entirely.
        ([1, 1], [4, 7], 2, None),
        # Prefill-only, all fresh: no initial state is ever read.
        ([6, 2], [6, 2], 0, None),
        # Stage-3 rank-blocked convolution fragments.
        ([1, 5, 3], [9, 5, 11], 1, 1),
    ],
    ids=["mixed", "decode-only", "prefill-fresh", "rank-blocked"],
)
def test_pooled_kda_matches_dense_bitwise(
    query_lens, seq_lens, decode_end, conv_shard_heads
):
    _check_pooled_matches_dense(
        query_lens, seq_lens, decode_end, conv_shard_heads, jnp.bfloat16
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "conv_shard_heads", [None, 1], ids=["one-shard", "rank-blocked"]
)
def test_pooled_kda_fp8_pool_keeps_bf16_conv(conv_shard_heads):
    """An fp8 KV pool changes how many tokens an attention row packs, not the
    KDA state. Conv stays bf16 in the rows that block sizing and the Raiden
    manifest reserve for it, so it round-trips the pool bit for bit."""
    _check_pooled_matches_dense(
        [1, 5, 3], [9, 5, 11], 1, conv_shard_heads, jnp.float8_e4m3fn
    )


def _check_pooled_matches_dense(
    query_lens, seq_lens, decode_end, conv_shard_heads, pool_dtype
):
    num_seqs = len(query_lens)
    rng = np.random.default_rng(0)
    inputs = _inputs(rng, query_lens, seq_lens)
    conv_init, ssm_init = _initial_states(rng, num_seqs)
    distribution = jnp.asarray([decode_end, num_seqs, num_seqs], dtype=jnp.int32)
    block_ids = [1, 2, 3][:num_seqs]

    out_dense, conv_dense, ssm_dense = _fused_kda_core_dense(
        inputs, conv_init, ssm_init, distribution
    )

    pool = _pack_pool(conv_init, ssm_init, block_ids, conv_shard_heads, pool_dtype)
    before = _raw(pool)
    out_pool, pool_after = _run_pooled(
        inputs,
        pool,
        block_ids,
        distribution,
        pool_block_tokens=BLOCK_SIZE * _pool_pack(pool_dtype),
        conv_shard_heads=conv_shard_heads,
    )

    # Same arithmetic on identical values, but compiled in a different graph
    # (gather-fed buffers instead of plain ones), so fusion can reassociate
    # the non-Pallas f32 sums: compare numerically, not bitwise.
    np.testing.assert_allclose(
        np.asarray(out_pool, np.float32),
        np.asarray(out_dense, np.float32),
        rtol=5e-2,
        atol=4e-3,
    )

    conv_back, ssm_back = _unpack_pool(
        pool_after, block_ids, conv_shard_heads=conv_shard_heads
    )
    # The conv state update is pure data movement: bitwise equal.
    np.testing.assert_array_equal(
        np.asarray(conv_back.view(jnp.uint16)),
        np.asarray(conv_dense[1:].view(jnp.uint16)),
    )
    np.testing.assert_allclose(
        np.asarray(ssm_back, np.float32),
        np.asarray(ssm_dense[1:], np.float32),
        rtol=1e-2,
        atol=5e-4,
    )

    # Everything outside the state regions is untouched: block 0, unlisted
    # blocks, and the attention token rows of the state blocks.
    assert pool_after.dtype == pool.dtype
    after = _raw(pool_after)
    for block in range(NUM_BLOCKS):
        if block not in block_ids:
            np.testing.assert_array_equal(after[block], before[block])
    region_end = LAYOUT.ssm_tokens + LAYOUT.conv_tokens
    for block in block_ids:
        np.testing.assert_array_equal(
            after[block, region_end:], before[block, region_end:]
        )


def test_pooled_kda_rejects_oversized_state():
    inputs = _inputs(np.random.default_rng(0), [2], [2])
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=SSM_ELEMS * 4, conv_bytes=CONV_ELEMS * 2, token_bytes=TOK_BYTES
    )
    tiny_pool = jnp.zeros(
        (2, layout.required_tokens - 1, PACK, LANES), dtype=jnp.bfloat16
    )
    with pytest.raises(ValueError, match="exceed the pool block"):
        _run_pooled(
            inputs,
            tiny_pool,
            [1],
            jnp.asarray([0, 1, 1], dtype=jnp.int32),
            pool_block_tokens=(layout.required_tokens - 1) * PACK,
        )


@pytest.mark.tpu_test
def test_pooled_kda_manager_block_split():
    """Manager ids on a pool born at kernel-block granularity (split > 1).

    A backend with a fixed kernel block size splits each manager block into
    `split` consecutive pool blocks. Here the kernel block is a single pool
    row, so manager block m is kernel blocks [m*split, (m+1)*split) and the
    op must remap the manager ids through gather_region's split branch
    (state_indices * split + kb). The split pool is a byte-identical
    reshape of the split=1 pool, so both paths must agree.
    """
    query_lens, seq_lens, decode_end = [1, 5, 3], [9, 5, 11], 1
    num_seqs = len(query_lens)
    rng = np.random.default_rng(0)
    inputs = _inputs(rng, query_lens, seq_lens)
    conv_init, ssm_init = _initial_states(rng, num_seqs)
    distribution = jnp.asarray([decode_end, num_seqs, num_seqs], dtype=jnp.int32)
    block_ids = [1, 2, 3][:num_seqs]
    pool_block_tokens = BLOCK_SIZE * PACK

    pool = _pack_pool(conv_init, ssm_init, block_ids)
    out_ref, pool_ref = _run_pooled(
        inputs, pool, block_ids, distribution, pool_block_tokens=pool_block_tokens
    )

    split = BLOCK_SIZE  # one pool row per kernel block
    pool_split = pool.reshape(NUM_BLOCKS * split, 1, PACK, LANES)
    out_split, pool_split_after = _run_pooled(
        inputs, pool_split, block_ids, distribution, pool_block_tokens=pool_block_tokens
    )

    np.testing.assert_allclose(
        np.asarray(out_split, np.float32),
        np.asarray(out_ref, np.float32),
        rtol=5e-2,
        atol=4e-3,
    )

    # Unpack both pools through the same gather_region machinery the op
    # uses — raw byte views of the f32 region hit NaN bit patterns that
    # poison allclose. Conv is pure data movement (bitwise); the ssm state
    # went through the same arithmetic in a different XLA graph, so compare
    # numerically.
    conv_split, ssm_split = _unpack_pool(pool_split_after, block_ids, split=split)
    conv_ref, ssm_ref = _unpack_pool(pool_ref, block_ids)
    np.testing.assert_array_equal(
        np.asarray(conv_split.view(jnp.uint16)), np.asarray(conv_ref.view(jnp.uint16))
    )
    np.testing.assert_allclose(
        np.asarray(ssm_split, np.float32),
        np.asarray(ssm_ref, np.float32),
        rtol=1e-2,
        atol=5e-4,
    )

    # Kernel blocks outside the batch's manager blocks keep their seed
    # bytes exactly.
    after = np.asarray(pool_split_after.view(jnp.uint16)).reshape(
        NUM_BLOCKS, BLOCK_SIZE, PACK, LANES
    )
    seed = np.asarray(pool.view(jnp.uint16))
    for block in range(NUM_BLOCKS):
        if block not in block_ids:
            np.testing.assert_array_equal(after[block], seed[block])


@pytest.mark.tpu_test
def test_pooled_kda_multistep_state_evolution():
    """State must round-trip the pool across a chunked prefill and decodes.

    One request: two prefill chunks followed by three decode steps. The dense
    path threads its buffers; the pooled path threads only the pool, so every
    step's state goes through the gather/scatter seam. Outputs are compared
    per step and the final states at the end.
    """
    rng = np.random.default_rng(0)
    steps = [
        (128, 128, 0),
        (96, 224, 0),
        (1, 225, 1),
        (1, 226, 1),
        (1, 227, 1),
    ]  # (query_len, seq_len_after, decode_end)

    conv_init, ssm_init = _initial_states(rng, 1)
    # Dense: null slot 0 + the request's slot 1, threaded across steps.
    conv_buf = jnp.concatenate([jnp.zeros_like(conv_init[:1]), conv_init], axis=0)
    ssm_buf = jnp.concatenate([jnp.zeros_like(ssm_init[:1]), ssm_init], axis=0)
    # Pooled: the request's state lives in pool block 1.
    pool = _pack_pool(conv_init, ssm_init, [1])

    for step_idx, (query_len, seq_len, decode_end) in enumerate(steps):
        inputs = _inputs(rng, [query_len], [seq_len])
        distribution = jnp.asarray([decode_end, 1, 1], dtype=jnp.int32)

        out_dense, conv_buf, ssm_buf = _fused_kda_core_dense(
            inputs, conv_buf[1:], ssm_buf[1:], distribution
        )
        out_pool, pool = _run_pooled(
            inputs, pool, [1], distribution, pool_block_tokens=BLOCK_SIZE * PACK
        )

        np.testing.assert_allclose(
            np.asarray(out_pool, np.float32),
            np.asarray(out_dense, np.float32),
            rtol=5e-2,
            atol=4e-3,
            err_msg=f"output diverged at step {step_idx} {steps[step_idx]}",
        )

    conv_back, ssm_back = _unpack_pool(pool, [1])
    # Conv state is pure data movement through the seam: bitwise equal.
    np.testing.assert_array_equal(
        np.asarray(conv_back.view(jnp.uint16)),
        np.asarray(conv_buf[1:].view(jnp.uint16)),
    )
    np.testing.assert_allclose(
        np.asarray(ssm_back, np.float32),
        np.asarray(ssm_buf[1:], np.float32),
        rtol=1e-2,
        atol=5e-4,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("split", [1, BLOCK_SIZE])
@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize(
    "pool_dtype", [jnp.bfloat16, jnp.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_pooled_kda_speculative_checkpoints(split, seed, pool_dtype):
    """Noncontiguous blocks, ragged windows, rollback and padded requests.

    The dense window is the oracle; compare every checkpoint, not just the
    final output, while repeatedly selecting a different accepted prefix.
    """
    rng = np.random.default_rng(seed)
    window, num_reqs = 6, 3
    live_states = 2 * window
    num_blocks = live_states + 2
    physical = rng.permutation(np.arange(1, live_states + 1)).reshape(2, window)
    ckpts = jnp.asarray(np.vstack([physical, np.zeros(window, np.int32)]), jnp.int32)
    state_ids = ckpts[:, 0]
    conv, ssm = _initial_states(rng, live_states + 1)
    conv = conv.at[-1].set(0)
    ssm = ssm.at[-1].set(0)
    pool = _pack_pool(
        conv,
        ssm,
        [*physical.reshape(-1), 0],
        num_blocks=num_blocks,
        pool_dtype=pool_dtype,
    )
    pack = pool.shape[2]
    sentinel = np.asarray(pool[-1].view(jnp.uint8)).copy()
    pool = pool.reshape(num_blocks * split, BLOCK_SIZE // split, pack, LANES)
    conv_buf = jnp.concatenate(
        [conv[-1:], conv[:-1], jnp.repeat(conv[-1:], window, axis=0)]
    )
    ssm_buf = jnp.concatenate(
        [ssm[-1:], ssm[:-1], jnp.repeat(ssm[-1:], window, axis=0)]
    )
    identity = 1 + jnp.arange(num_reqs, dtype=jnp.int32) * window
    steps = [
        # Pure prefill must only seed checkpoint zero.
        ([8, 7, 0], [8, 7, 0], 0, [0, 0, 0]),
        ([6, 3, 0], [14, 10, 0], 2, [0, 0, 0]),
        # Resume independently from last and intermediate checkpoints.
        ([6, 1, 0], [20, 9, 0], 2, [5, 1, 0]),
        ([1, 6, 0], [16, 14, 0], 2, [0, 0, 0]),
        # Mixed verify + continuation prefill.
        ([6, 9, 0], [22, 23, 0], 1, [0, 4, 0]),
    ]
    for lengths, seq_lens, window_end, offsets in steps:
        inputs = _inputs(rng, lengths, seq_lens)
        distribution = jnp.asarray([window_end, 2, 2], jnp.int32)
        pool_offsets = (
            jnp.zeros(num_blocks, jnp.int32).at[state_ids].set(jnp.asarray(offsets))
        )
        dense_offsets = (
            jnp.zeros(num_reqs * window + 1, jnp.int32)
            .at[identity]
            .set(jnp.asarray(offsets))
        )
        out_ref, conv_buf, ssm_buf = _build_fused_core(-5.0, EPS, window - 1)(
            inputs["mixed_qkv"],
            inputs["raw_gate"],
            inputs["beta"],
            inputs["output_gate"],
            conv_buf.reshape(num_reqs * window + 1, KERNEL_SIZE - 1, -1),
            ssm_buf,
            inputs["conv_weight"],
            inputs["a_log"],
            inputs["dt_bias"],
            inputs["norm_weight"],
            inputs["query_start_loc"],
            identity,
            inputs["seq_lens"],
            distribution,
            distribution,
            dense_offsets,
        )
        out, pool = _pooled_kda_core(
            inputs["mixed_qkv"],
            inputs["raw_gate"],
            inputs["beta"],
            inputs["output_gate"],
            pool,
            inputs["conv_weight"],
            inputs["a_log"],
            inputs["dt_bias"],
            inputs["norm_weight"],
            inputs["query_start_loc"],
            state_ids,
            inputs["seq_lens"],
            distribution,
            distribution,
            pool_offsets,
            ckpts,
            lower_bound=-5.0,
            eps=EPS,
            pool_block_tokens=BLOCK_SIZE * pack,
            num_spec_tokens=window - 1,
        )
        np.testing.assert_allclose(
            np.asarray(out, np.float32),
            np.asarray(out_ref, np.float32),
            rtol=5e-2,
            atol=4e-3,
        )
        actual_conv, actual_ssm = _unpack_pool(
            pool, list(physical.reshape(-1)), split=split
        )
        np.testing.assert_array_equal(
            np.asarray(actual_conv.view(jnp.uint16)),
            np.asarray(
                conv_buf[1 : live_states + 1]
                .reshape(actual_conv.shape)
                .view(jnp.uint16)
            ),
        )
        np.testing.assert_allclose(
            np.asarray(actual_ssm),
            np.asarray(ssm_buf[1 : live_states + 1]),
            rtol=1e-2,
            atol=5e-4,
        )
        restored = pool.reshape(num_blocks, BLOCK_SIZE, pack, LANES)
        np.testing.assert_array_equal(
            np.asarray(restored[-1].view(jnp.uint8)), sentinel
        )
