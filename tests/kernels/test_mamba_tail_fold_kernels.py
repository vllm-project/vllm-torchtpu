# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MAMBA_TAIL_FOLD: a folded prefill writes the same boundary state
the tail split would have cached.

The split runs a prompt as `[start, boundary)` then `[boundary, end)`, and
the first step leaves the GDN, conv and PLE state at `boundary` in a block
the prefix cache serves. The fold runs `[start, end)` in one step and must
write that same state into the checkpoint block. GDN: the tile ending at the
boundary is marked as a last tile with the checkpoint block as destination
(CPU metadata test; kernel equality on TPU). PLE: the history before the
boundary (CPU against `short_conv`; pool write on TPU).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import metadata
from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
from vllm_torchtpu.kernels.qwen4_exp.pool import (
    prefill_checkpoint_history,
    read_region,
    write_prefill_checkpoint,
)


def _per_seq(lens, start_seq, chunk, prefill_ckpt=None):
    cfg = SimpleNamespace(
        batch_size=sum(lens), chunk_size=chunk, tile_size=chunk, seq_tile_size=1
    )
    qsl = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)
    n = len(lens)
    return metadata.compute_per_seq_metadata(
        cfg=cfg,
        seq_lens=jnp.asarray([256 + t for t in lens], jnp.int32),
        query_start_loc=qsl,
        state_indices=jnp.arange(1, n + 1, dtype=jnp.int32),
        read_offsets=jnp.zeros((n,), jnp.int32),
        start_seq=jnp.int32(start_seq),
        end_seq=jnp.int32(n),
        prefill_ckpt=prefill_ckpt,
    )


def test_per_seq_metadata_marks_the_checkpoint_tile():
    # A decode (BATCHED segment), a folded 1000-token prefill with its
    # boundary at 512, and a plain 300-token prefill; 128-row tiles.
    lens, chunk = (1, 1000, 300), 128
    offsets = jnp.asarray([0, 512, 0], jnp.int32)
    blocks = jnp.asarray([0, 9, 0], jnp.int32)
    base = _per_seq(lens, 1, chunk)
    folded = _per_seq(lens, 1, chunk, (offsets, blocks))
    num_tiles = int(base.num_tiles)
    assert int(folded.num_tiles) == num_tiles == 8 + 3

    last = np.asarray(base.p_id_is_last_tile.data)[:num_tiles]
    folded_last = np.asarray(folded.p_id_is_last_tile.data)[:num_tiles]
    override = np.asarray(folded.p_id_to_state_override.data)[:num_tiles]
    # Tile 3 of the folded sequence ends at 512: it writes block 9.
    expected_last = last.copy()
    expected_last[3] = True
    np.testing.assert_array_equal(folded_last, expected_last)
    assert override.tolist() == [0, 0, 0, 9] + [0] * (num_tiles - 4)
    # Everything else is unchanged; the next tile still resumes from carry.
    for name in (
        "p_id_to_s_idx",
        "p_id_to_r_base",
        "p_id_to_r_size",
        "p_id_is_first_tile",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(folded, name).data)[:num_tiles],
            np.asarray(getattr(base, name).data)[:num_tiles],
        )
    assert not bool(np.asarray(folded.p_id_is_first_tile.data)[4])
    assert base.p_id_to_state_override is None


def test_per_seq_metadata_skips_null_or_misaligned_checkpoints():
    lens, chunk = (1000,), 128
    for offsets, blocks in (([512], [0]), ([500], [9]), ([0], [9])):
        folded = _per_seq(lens, 0, chunk, (jnp.asarray(offsets), jnp.asarray(blocks)))
        n = int(folded.num_tiles)
        assert not np.asarray(folded.p_id_to_state_override.data)[:n].any()


def test_ple_checkpoint_history_matches_the_split_chunk():
    rng = np.random.default_rng(5)
    channels, state_len, dilation = 64, 9, 3
    lens = (1000, 700, 40)
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)
    offsets = jnp.asarray([512, 512, 0], jnp.int32)
    x = jnp.asarray(rng.normal(size=(sum(lens), channels)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(channels, 4)), jnp.bfloat16)
    history = jnp.asarray(rng.normal(size=(3, state_len, channels)), jnp.bfloat16)
    got = np.asarray(prefill_checkpoint_history(x, starts, offsets, state_len))
    for r in range(2):
        # The split's first step: the chunk's first `offset` tokens.
        start, offset = int(starts[r]), int(offsets[r])
        for initial in (False, True):
            _, updated = short_conv(
                x[start : start + offset],
                weight,
                history[r : r + 1],
                jnp.asarray([0, offset], jnp.int32),
                jnp.asarray([0], jnp.int32),
                jnp.asarray([initial]),
                dilation=dilation,
                request_owned_history=True,
            )
            np.testing.assert_array_equal(got[r], np.asarray(updated)[0])


@pytest.mark.tpu_test
def test_ple_prefill_checkpoint_writes_only_checkpoint_blocks():
    rng = np.random.default_rng(6)
    channels, state_len = 128, 9
    pool = jnp.asarray(rng.integers(-100, 100, size=(7, 1664, 1, 2, 256)), jnp.bfloat16)
    before = np.asarray(pool).view(np.uint8).copy()
    starts = jnp.asarray([0, 1000, 1700, 1740], jnp.int32)
    offsets = jnp.asarray([512, 0, 512], jnp.int32)
    ids = jnp.asarray([3, 4, 0], jnp.int32)  # row 1: no offset; row 2: null
    x = jnp.asarray(rng.normal(size=(1740, channels)), jnp.bfloat16)
    pool = jax.jit(
        write_prefill_checkpoint, donate_argnums=(0,), static_argnames="state_len"
    )(pool, x, starts, offsets, ids, state_len=state_len)
    restored = read_region(
        pool, jnp.asarray([3]), elements=state_len * channels, dtype=jnp.bfloat16
    ).reshape(state_len, channels)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(x[503:512]))
    after = np.asarray(pool).view(np.uint8)
    np.testing.assert_array_equal(after[[0, 1, 2, 4, 5, 6]], before[[0, 1, 2, 4, 5, 6]])


# --- GDN kernel on the unified pool (TPU) ---------------------------------

N_KQ, N_V, DIM = 2, 6, 128
WIDTH = N_KQ * DIM * 2 + N_V * DIM
KBS, LANES, PAYLOAD = 512, 256, (1, 4)
BLOCK = 512  # manager block: the checkpoint alignment
SPLIT = BLOCK // KBS
NUM_BLOCKS = 8


def _gdn_inputs(tokens, seed):
    rng = np.random.default_rng(seed)

    def normal(shape, dtype=jnp.bfloat16, scale=0.5):
        return jnp.asarray(rng.normal(size=shape) * scale, dtype)

    return dict(
        qkv=normal((tokens, WIDTH)),
        b=normal((tokens, N_V)),
        a=normal((tokens, N_V)),
        weight=normal((WIDTH, 1, 4), scale=0.2),
        bias=normal((WIDTH,), scale=0.1),
        a_log=normal((N_V,), jnp.float32),
        dt_bias=normal((N_V,), jnp.float32) - 1,
    )


def _gdn(pool, x, rows, lens, seq_lens, state, distribution, prefill_ckpt=None):
    from vllm_torchtpu.layers.core import gdn_attention

    jax.clear_caches()
    qsl = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)
    return gdn_attention.run_jax_gdn_attention_pooled_local(
        mixed_qkv=x["qkv"][rows],
        b=x["b"][rows],
        a=x["a"][rows],
        recurrent_state=jnp.array(pool, copy=True),
        conv_weight=x["weight"],
        conv_bias=x["bias"],
        A_log=x["a_log"],
        dt_bias=x["dt_bias"],
        query_start_loc=qsl,
        state_indices=jnp.asarray(state, jnp.int32),
        distribution=jnp.asarray(distribution, jnp.int32),
        seq_lens=jnp.asarray(seq_lens, jnp.int32),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=DIM,
        d_v=DIM,
        kernel_size=4,
        pool_block_tokens=BLOCK,
        prefill_ckpt=prefill_ckpt,
    )


def _block(pool, block):
    return np.asarray(pool[block * SPLIT : (block + 1) * SPLIT]).view(np.uint8)


@pytest.mark.tpu_test
@pytest.mark.parametrize("chunk_pair", [False, True])
def test_gdn_fold_writes_the_split_boundary_state(chunk_pair, monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_GDN_CHUNK_PAIR", chunk_pair)
    # Batch: one decode, the folded 1000-token prompt, a plain 300 prefill.
    tokens = 1 + 1000 + 300
    x = _gdn_inputs(tokens, seed=7)
    pool = jnp.zeros((NUM_BLOCKS * SPLIT, KBS) + PAYLOAD + (LANES,), jnp.int8)
    folded_rows = slice(1, 1001)

    # Split: [0, 512) into block 1, then [512, 1000) resuming from block 1.
    first, _ = _gdn(pool, x, slice(1, 513), (512,), (512,), (1,), (0, 1, 1))
    boundary = _block(first, 1)
    second, tail_out = _gdn(
        first, x, slice(513, 1001), (488,), (1000,), (1,), (0, 1, 1)
    )

    # Fold: the whole batch in one call; the prompt's running state in block
    # 2, its boundary checkpoint into block 1. Blocks 3 / 4: the others.
    batch = dict(
        rows=slice(0, tokens),
        lens=(1, 1000, 300),
        seq_lens=(301, 1000, 300),
        state=(3, 2, 4),
        distribution=(1, 3, 3),
    )
    ckpt = (jnp.asarray([0, 512, 0], jnp.int32), jnp.asarray([0, 1, 0], jnp.int32))
    folded, out = _gdn(pool, x, prefill_ckpt=ckpt, **batch)
    plain, plain_out = _gdn(pool, x, **batch)

    np.testing.assert_array_equal(_block(folded, 1), boundary)
    np.testing.assert_array_equal(_block(folded, 2), _block(second, 1))
    np.testing.assert_array_equal(
        np.asarray(out[folded_rows], np.float32)[512:],
        np.asarray(tail_out, np.float32),
    )
    # The checkpoint changes nothing else: outputs, other requests' states,
    # and every block but the checkpoint match the plain call.
    np.testing.assert_array_equal(np.asarray(out), np.asarray(plain_out))
    for block in range(NUM_BLOCKS):
        if block != 1:
            np.testing.assert_array_equal(_block(folded, block), _block(plain, block))
    assert not _block(plain, 1).any()


@pytest.mark.tpu_test
@pytest.mark.parametrize("chunk_pair", [False, True])
def test_gdn_nobound_fold_writes_the_split_boundary_state(chunk_pair, monkeypatch):
    """VLLM_TPU_MAMBA_TAIL_FOLD_NOBOUNDARY: a prefix hit at 512 and a 473-token
    rest with no block boundary inside. The split runs [512, 768) then
    [768, 985); the fold runs [512, 985) with the checkpoint at offset 256 in
    another block (the first speculative scratch block). Both resume from the
    hit state seeded into the running block, as serving's seed copy does.
    The checkpoint sits at offset 256, so the fold unit must be 256 (as
    served): with the default unit (the 512 block) the kernel drops every
    checkpoint of a step padded to at most 512 tokens."""
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_GDN_CHUNK_PAIR", chunk_pair)
    monkeypatch.setattr(envs, "VLLM_TPU_MAMBA_TAIL_FOLD_UNIT", 256)
    tokens = 985
    x = _gdn_inputs(tokens, seed=11)
    empty = jnp.zeros((NUM_BLOCKS * SPLIT, KBS) + PAYLOAD + (LANES,), jnp.int8)
    # The hit: [0, 512) leaves its state in block 1.
    hit, _ = _gdn(empty, x, slice(0, 512), (512,), (512,), (1,), (0, 1, 1))

    def seeded(dst):
        return hit.at[dst * SPLIT : (dst + 1) * SPLIT].set(hit[1 * SPLIT : 2 * SPLIT])

    # Split: running block 2 seeded with the hit state.
    first, head_out = _gdn(
        seeded(2), x, slice(512, 768), (256,), (768,), (2,), (0, 1, 1)
    )
    boundary = _block(first, 2)
    second, tail_out = _gdn(first, x, slice(768, 985), (217,), (985,), (2,), (0, 1, 1))

    # Fold: running block 5 seeded likewise; checkpoint at 256 into block 6.
    ckpt = (jnp.asarray([256], jnp.int32), jnp.asarray([6], jnp.int32))
    args = (x, slice(512, 985), (473,), (985,), (5,), (0, 1, 1))
    folded, out = _gdn(seeded(5), *args, prefill_ckpt=ckpt)
    plain, plain_out = _gdn(seeded(5), *args)

    np.testing.assert_array_equal(_block(folded, 6), boundary)
    np.testing.assert_array_equal(_block(folded, 5), _block(second, 2))
    np.testing.assert_array_equal(
        np.asarray(out, np.float32)[:256], np.asarray(head_out, np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(out, np.float32)[256:], np.asarray(tail_out, np.float32)
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(plain_out))
    for block in range(NUM_BLOCKS):
        if block != 6:
            np.testing.assert_array_equal(_block(folded, block), _block(plain, block))
