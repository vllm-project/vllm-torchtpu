"""GDN V3 pooled state seam: plan math, byte compatibility, equivalence.

`TestV3StateSourcePlan` and the source-hygiene test are pure Python/CPU.
The remaining classes execute Pallas kernels and run on TPU (same gating
as test_pool_adapters.py).
"""

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kernels.gdn.v3.gdn_pool_state_utils import (
    GRANULE,
    WORDS_PER_GRANULE,
    on_cpu,
    pack_bf16_rows,
    put_words,
    read_states,
    state_words,
    unpack_bf16_rows,
    words_to_pool,
    write_states,
)

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import wrapper

# 35B-like slice: manager block = 3 kernel blocks; ssm = 2 whole kernel
# blocks (one contiguous 512 KB DMA per slot); conv = 16 token rows at
# the start of the third kernel block.
N_KQ, N_V, D_K, D_V = 2, 8, 128, 128
KERNEL_SIZE = 4
DIM = N_KQ * D_K * 2 + N_V * D_V  # 1536
SPLIT, KBS, LANES = 3, 256, 256
PAYLOAD = (1, 2)
TOK_BYTES = 2 * LANES * 2  # bf16 pool
SSM_NTOK = N_V * D_K * D_V * 4 // TOK_BYTES  # 512
CONV_TOK0 = SSM_NTOK
CONV_NTOK = 16
CONV_ROWS = (KERNEL_SIZE - 1) * DIM // LANES  # 18
CONV_SLOT_ROWS = CONV_NTOK * TOK_BYTES // (2 * LANES)  # 32
NUM_MGR = 5  # null block + 4 slots
POOL_SHAPE = (NUM_MGR * SPLIT, KBS) + PAYLOAD + (LANES,)

# Granule coordinates of the same two regions. The kernel is told neither of
# them -- it rederives the conv region's start from the pool's own geometry --
# so these exist for the tests to place and read bytes with. Both regions are
# plain byte ranges, so their extents follow from the state dtypes alone.
CONV_GRANULES = (KERNEL_SIZE - 1) * DIM * 4 // GRANULE  # 36
# Pool tokens the conv region covers, from the start of its kernel block.
CONV_END_TOK = CONV_GRANULES * GRANULE // TOK_BYTES  # 18


def _plan(pool=None, **overrides):
    if pool is None:
        pool = jax.ShapeDtypeStruct(POOL_SHAPE, jnp.bfloat16)
    kwargs = dict(
        split=SPLIT,
        ssm_ntok=SSM_NTOK,
        conv_tok0=CONV_TOK0,
        conv_ntok=CONV_NTOK,
        conv_dim=DIM,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )
    kwargs.update(overrides)
    return pool_adapters.v3_state_source(pool, **kwargs)


class TestV3StateSourcePlan:
    """Plan-construction math only; runs on CPU."""

    def test_ssm_region_one_contiguous_whole_block_window(self):
        r = _plan().recurrent
        assert (r.kb0, r.nblocks, r.row0, r.nrows) == (0, 2, 0, KBS)
        assert r.view_dtype == jnp.dtype(jnp.float32)
        assert r.lane_split == LANES // D_V
        assert r.rows_used == N_V * D_K

    def test_conv_region_rows_of_last_kernel_block(self):
        plan = _plan()
        c = plan.conv
        assert (c.kb0, c.nblocks, c.row0, c.nrows) == (2, 1, 0, CONV_NTOK)
        assert c.view_dtype == jnp.dtype(jnp.bfloat16)
        assert c.lane_split == 1
        assert c.rows_used == CONV_ROWS
        assert plan.stride == SPLIT
        assert plan.whole_block_dma is False

    def test_plan_is_static_hashable_and_stable(self):
        assert _plan() == _plan()
        assert hash(_plan()) == hash(_plan())

    def test_subblock_regions_when_pool_is_manager_grained(self):
        pool = jax.ShapeDtypeStruct(
            (NUM_MGR, SPLIT * KBS) + PAYLOAD + (LANES,), jnp.bfloat16
        )
        plan = _plan(pool, split=1)
        r = plan.recurrent
        assert (r.kb0, r.nblocks, r.row0, r.nrows) == (0, 1, 0, SSM_NTOK)
        c = plan.conv
        assert (c.kb0, c.nblocks, c.row0, c.nrows) == (0, 1, CONV_TOK0, CONV_NTOK)
        assert plan.stride == 1

    def test_one_byte_pool_keeps_regions_and_view_dtypes(self):
        pool = jax.ShapeDtypeStruct((NUM_MGR * SPLIT, KBS, 1, 4, LANES), jnp.int8)
        plan = _plan(pool)
        assert plan.recurrent.nblocks == 2
        assert plan.recurrent.view_dtype == jnp.dtype(jnp.float32)
        assert plan.conv.view_dtype == jnp.dtype(jnp.bfloat16)

    def test_conv_region_straddling_kernel_blocks_rejected(self):
        with pytest.raises(NotImplementedError):
            _plan(conv_tok0=KBS - 8)


def test_v3_kernel_package_stays_pool_agnostic():
    """The dense path must be byte-for-byte untouched by the seam: the V3
    kernel package sees only the generic state-source plan, never the
    pool (pool knowledge lives in pool_adapters / the pooled caller)."""
    import vllm_torchtpu.kernels.gdn.v3 as v3_pkg

    for path in pathlib.Path(list(v3_pkg.__path__)[0]).glob("*.py"):
        assert "pool_adapters" not in path.read_text(), path


# --------------------------------------------------------------------
# TPU tests below: byte-compat and end-to-end equivalence of the seam
# against the pre-seam round trip (gather -> dense kernel on identity
# indices -> scatter), rebuilt here from the surviving adapters.
# --------------------------------------------------------------------


def _garbage_pool(seed, num_mgr: int = NUM_MGR):
    rng = np.random.default_rng(seed)
    shape = (num_mgr * SPLIT, KBS) + PAYLOAD + (LANES,)
    return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)


_STATE_SHAPES = dict(recurrent_shape=(N_V, D_K, D_V), conv_shape=(KERNEL_SIZE - 1, DIM))


def _write_states(pool, ssm, conv, idx, split=SPLIT, ssm_dtype=jnp.float32):
    return write_states(pool, ssm, conv, idx, split=split, recurrent_dtype=ssm_dtype)


def _read_states(pool, idx, split=SPLIT, ssm_dtype=jnp.float32):
    return read_states(
        pool, idx, split=split, recurrent_dtype=ssm_dtype, **_STATE_SHAPES
    )


def _rand_inputs(seed, num_tokens):
    rngs = iter(jax.random.split(jax.random.key(seed), 8))
    return dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, DIM)),
        b=jax.random.normal(next(rngs), (num_tokens, N_V)),
        a=jax.random.normal(next(rngs), (num_tokens, N_V)),
        conv_weight=jax.random.normal(next(rngs), (DIM, 1, KERNEL_SIZE)),
        conv_bias=jax.random.normal(next(rngs), (DIM,)),
        a_log=jax.random.normal(next(rngs), (N_V,)),
        dt_bias=jax.random.normal(next(rngs), (N_V,)),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )


def _run_roundtrip(pool, idx, kwargs):
    """The pre-seam pooled V3 path."""
    ssm, conv = _read_states(pool, idx)
    identity = jnp.arange(idx.shape[0], dtype=jnp.int32)
    (new_conv, new_ssm), out = wrapper.fused_conv1d_gdn(
        states=(conv, ssm), state_indices=identity, **kwargs
    )
    return _write_states(pool, new_ssm, new_conv, idx), out


def _merge(pool, split=SPLIT):
    """Kernel blocks -> manager blocks, as the pooled caller does."""
    return pool.reshape(pool.shape[0] // split, split * pool.shape[1], *pool.shape[2:])


def _split(pool, shape):
    return pool.reshape(shape)


def _run_seam(pool, idx, kwargs, split=SPLIT):
    out_pool, out = wrapper.fused_conv1d_gdn(
        states=_merge(pool, split),
        state_indices=idx,
        conv_dtype=jnp.float32,
        recurrent_dtype=jnp.float32,
        **kwargs,
    )
    return _split(out_pool, pool.shape), out


@pytest.mark.tpu_test
class TestSeamMatchesRoundTrip:
    def _compare(self, pool, idx, kwargs):
        pool_rt, out_rt = _run_roundtrip(jnp.copy(pool), idx, kwargs)
        pool_seam, out_seam = _run_seam(jnp.copy(pool), idx, kwargs)

        # The two paths lay the state out differently in VMEM, so the
        # f32 accumulations reassociate: outputs and states agree to a
        # couple of bf16 ulps (operand precision), not bitwise.
        tol = dict(rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(np.asarray(out_seam), np.asarray(out_rt), **tol)
        ssm_rt, conv_rt = _read_states(pool_rt, idx)
        ssm_seam, conv_seam = _read_states(pool_seam, idx)
        np.testing.assert_allclose(np.asarray(ssm_seam), np.asarray(ssm_rt), **tol)
        np.testing.assert_allclose(
            np.asarray(conv_seam, dtype=np.float32),
            np.asarray(conv_rt, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        # Blocks neither path addresses must stay byte-identical; the
        # addressed windows are covered by the logical compares above and
        # the write-layout tests (seam bytes == granule-offset bytes for
        # identical values).
        touched = set()
        for mgr in np.asarray(idx):
            touched.update(range(int(mgr) * SPLIT, int(mgr) * SPLIT + SPLIT))
        for kb in range(pool_seam.shape[0]):
            if kb not in touched:
                assert jnp.array_equal(
                    pool_seam[kb].view(jnp.int16), pool_rt[kb].view(jnp.int16)
                ), kb

    def test_decode_continuation(self):
        n = 4
        idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(10), 2))
        pool = _write_states(
            _garbage_pool(0),
            jax.random.normal(next(keys), (n, N_V, D_K, D_V)),
            jax.random.normal(next(keys), (n, KERNEL_SIZE - 1, DIM)).astype(
                jnp.bfloat16
            ),
            idx,
        )
        kwargs = _rand_inputs(1, n) | dict(
            query_start_loc=jnp.arange(n + 1),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
        )
        self._compare(pool, idx, kwargs)

    def test_per_ckpt_read_selects_the_named_block(self):
        """Rollback must resume from `ckpt_indices[s, read_offsets[s]]`.

        With per-checkpoint indices the group is scattered across ordinary
        pool blocks, so selecting checkpoint `r` means indexing the block
        named at column `r` — not adding an affine pitch to a base block.
        Checked as an equivalence through the *same* kernel path: reading
        column `r` of a pool whose columns hold unrelated states must equal
        reading column 0 of a pool that has state `r` there. Comparing one
        path against itself keeps float reassociation out of the result.
        """
        n, num_spec = 2, 3
        window = num_spec + 1
        # Deliberately non-contiguous columns, so no affine base + pitch
        # map could reproduce this addressing.
        # Distinct, non-contiguous block per (seq, checkpoint): reusing a
        # block across slots would let one sequence's write clobber
        # another's, which is a property of the addressing, not the kernel.
        ckpt_blocks = jnp.asarray(
            np.array([[1, 3, 5, 7], [8, 6, 4, 2]], dtype=np.int32)
        )
        num_mgr = 9
        keys = iter(jax.random.split(jax.random.key(77), 2 * window))
        states = []
        pool = _garbage_pool(11, num_mgr)
        for t in range(window):
            ssm_t = jax.random.normal(next(keys), (n, N_V, D_K, D_V))
            conv_t = jax.random.normal(next(keys), (n, KERNEL_SIZE - 1, DIM)).astype(
                jnp.bfloat16
            )
            states.append((ssm_t, conv_t))
            pool = _write_states(pool, ssm_t, conv_t, ckpt_blocks[:, t])

        kwargs = _rand_inputs(12, n) | dict(
            query_start_loc=jnp.arange(n + 1),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
        )

        def _run(src, read_ckpt):
            return wrapper.fused_conv1d_gdn(
                states=_merge(jnp.copy(src)),
                state_indices=jnp.zeros((n,), dtype=jnp.int32),
                conv_dtype=jnp.float32,
                recurrent_dtype=jnp.float32,
                read_offsets=jnp.full((n,), read_ckpt, dtype=jnp.int32),
                ckpt_indices=ckpt_blocks,
                num_spec_tokens=num_spec,
                **kwargs,
            )[1]

        for r in range(window):
            ssm_r, conv_r = states[r]
            # Same states, but the one the offset names now sits at column 0.
            at_col0 = _write_states(
                _garbage_pool(11, num_mgr), ssm_r, conv_r, ckpt_blocks[:, 0]
            )
            np.testing.assert_allclose(
                np.asarray(_run(pool, r)),
                np.asarray(_run(at_col0, 0)),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"checkpoint {r}",
            )

    def test_per_ckpt_writes_only_touch_named_blocks(self):
        """Checkpoint `t` is written to `ckpt_indices[:, t]`; nothing else
        in the pool may change."""
        n, num_spec = 2, 3
        window = num_spec + 1
        ckpt_blocks = np.array([[1, 3, 5, 7], [8, 6, 4, 2]], dtype=np.int32)
        num_mgr = 9
        pool = _garbage_pool(13, num_mgr)
        kwargs = _rand_inputs(14, n * window) | dict(
            query_start_loc=jnp.arange(0, n * window + 1, window),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
        )
        before = np.asarray(pool.view(jnp.int16))
        out_pool, _ = wrapper.fused_conv1d_gdn(
            states=_merge(jnp.copy(pool)),
            state_indices=jnp.zeros((n,), dtype=jnp.int32),
            conv_dtype=jnp.float32,
            recurrent_dtype=jnp.float32,
            read_offsets=jnp.zeros((n,), dtype=jnp.int32),
            ckpt_indices=jnp.asarray(ckpt_blocks),
            num_spec_tokens=num_spec,
            **kwargs,
        )
        out_pool = _split(out_pool, pool.shape)

        named = set(int(m) for m in ckpt_blocks.reshape(-1))
        touched_kb = {kb for m in named for kb in range(m * SPLIT, m * SPLIT + SPLIT)}
        after = np.asarray(out_pool.view(jnp.int16))
        changed = {
            kb
            for kb in range(after.shape[0])
            if not np.array_equal(after[kb], before[kb])
        }
        assert changed, "the run wrote no state at all"
        assert changed <= touched_kb, sorted(changed - touched_kb)

    def test_fresh_prefill_on_garbage_pool(self):
        idx = jnp.array([2, 4], dtype=jnp.int32)
        kwargs = _rand_inputs(2, 128) | dict(
            query_start_loc=jnp.array([0, 64, 128]),
            distribution=jnp.array([0, 2, 2], dtype=jnp.int32),
            seq_lens=jnp.array([64, 64], dtype=jnp.int32),
        )
        self._compare(_garbage_pool(3), idx, kwargs)

    def test_chunked_prefill_continuation(self):
        idx = jnp.array([1, 3], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(11), 2))
        pool = _write_states(
            _garbage_pool(4),
            jax.random.normal(next(keys), (2, N_V, D_K, D_V)),
            jax.random.normal(next(keys), (2, KERNEL_SIZE - 1, DIM)).astype(
                jnp.bfloat16
            ),
            idx,
        )
        kwargs = _rand_inputs(5, 112) | dict(
            query_start_loc=jnp.array([0, 64, 112]),
            distribution=jnp.array([0, 2, 2], dtype=jnp.int32),
            seq_lens=jnp.array([96, 80], dtype=jnp.int32),
        )
        self._compare(pool, idx, kwargs)

    def test_mixed_decode_and_prefill(self):
        idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(12), 2))
        decode_idx = idx[:2]
        pool = _write_states(
            _garbage_pool(6),
            jax.random.normal(next(keys), (2, N_V, D_K, D_V)),
            jax.random.normal(next(keys), (2, KERNEL_SIZE - 1, DIM)).astype(
                jnp.bfloat16
            ),
            decode_idx,
        )
        kwargs = _rand_inputs(7, 130) | dict(
            query_start_loc=jnp.array([0, 1, 2, 66, 130]),
            distribution=jnp.array([2, 4, 4], dtype=jnp.int32),
            seq_lens=jnp.array([7, 9, 64, 64], dtype=jnp.int32),
        )
        self._compare(pool, idx, kwargs)


@pytest.mark.tpu_test
class TestSeamSkipsInvalidSlots:
    def test_padded_slots_and_null_block_untouched(self):
        # 2 active decodes on manager blocks 2 and 4; rows 2..3 padded to
        # the null block. Only the active windows' state regions may move.
        idx = jnp.array([2, 4, 0, 0], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(13), 2))
        pool = _write_states(
            _garbage_pool(8),
            jax.random.normal(next(keys), (2, N_V, D_K, D_V)),
            jax.random.normal(next(keys), (2, KERNEL_SIZE - 1, DIM)).astype(
                jnp.bfloat16
            ),
            idx[:2],
        )
        before = jnp.copy(pool)
        kwargs = _rand_inputs(9, 2) | dict(
            query_start_loc=jnp.array([0, 1, 2, 2, 2]),
            distribution=jnp.array([2, 2, 2], dtype=jnp.int32),
            seq_lens=jnp.array([5, 5, 0, 0], dtype=jnp.int32),
        )
        after, _ = _run_seam(pool, idx, kwargs)

        def b16(x):
            return x.view(jnp.int16)

        # Null manager block and non-addressed manager blocks: untouched.
        for kb in (0, 1, 2, 3, 4, 5, 9, 10, 11):
            assert jnp.array_equal(b16(after[kb]), b16(before[kb])), kb
        for mgr in (2, 4):
            # ssm blocks written...
            assert not jnp.array_equal(
                b16(after[mgr * SPLIT]), b16(before[mgr * SPLIT])
            ), mgr
            # ...and the conv block's rows outside the conv region kept.
            conv_kb = mgr * SPLIT + 2
            assert jnp.array_equal(
                b16(after[conv_kb, CONV_END_TOK:]), b16(before[conv_kb, CONV_END_TOK:])
            ), mgr


@pytest.mark.tpu_test
class TestPooledSpecWindows:
    """Speculative verify windows on the pooled path.

    Each checkpoint is its own pool block, named by `ckpt_indices`, and the
    read offset picks which one a sequence resumes from. Reference is the
    dense SPEC kernel (validated token-by-token in test_gdn_attention_v3)
    on compact slot groups, fed byte-equal initial states: matching outputs
    across multi-step schedules prove the pooled DMA wrote each window
    position's checkpoint to the block the next step's read offset names.
    """

    NUM_SPEC = 1
    WINDOW = NUM_SPEC + 1
    N = 3

    def _pool(self, split):
        shape = (
            (NUM_MGR * self.WINDOW * split, KBS) + PAYLOAD + (LANES,)
            if split > 1
            else (NUM_MGR * self.WINDOW, 1280) + PAYLOAD + (LANES,)
        )
        rng = np.random.default_rng(20)
        return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)

    def _steps(self):
        """Three-step schedule; every step's per-seq read offsets pick a
        checkpoint the previous step wrote."""
        n = self.N
        return [
            # Verify windows (2, 1, 2 tokens), resuming from checkpoint 0.
            dict(
                num_tokens=5,
                query_start_loc=jnp.array([0, 2, 3, 5]),
                distribution=jnp.array([n, n, n], dtype=jnp.int32),
                seq_lens=jnp.array([10, 7, 9], dtype=jnp.int32),
                read_offsets=jnp.zeros((n,), dtype=jnp.int32),
            ),
            # Mixed acceptance: seqs 0/2 roll back to checkpoint 1, seq 1
            # to checkpoint 0.
            dict(
                num_tokens=5,
                query_start_loc=jnp.array([0, 2, 4, 5]),
                distribution=jnp.array([n, n, n], dtype=jnp.int32),
                seq_lens=jnp.array([12, 9, 10], dtype=jnp.int32),
                read_offsets=jnp.array([1, 0, 1], dtype=jnp.int32),
            ),
            # PER_SEQ resume with a non-zero read offset: seq 0 stays a
            # 1-token window, seq 1 becomes a 5-token prefill continuation
            # resuming from checkpoint 1 (the prefix-cache-resume shape);
            # seq 2 idles.
            dict(
                num_tokens=6,
                query_start_loc=jnp.array([0, 1, 6, 6]),
                distribution=jnp.array([1, 1, n], dtype=jnp.int32),
                seq_lens=jnp.array([14, 14, 0], dtype=jnp.int32),
                read_offsets=jnp.array([0, 1, 0], dtype=jnp.int32),
            ),
        ]

    @pytest.mark.parametrize("split", [6, 1], ids=["kernel_grained", "manager"])
    def test_pooled_spec_matches_dense(self, split):
        n, window = self.N, self.WINDOW
        pool = self._pool(split)
        # Checkpoint 0 is the request's state block; the rest are ordinary
        # pool blocks elsewhere, deliberately *not* adjacent, so a stray
        # affine base + offset would land on the wrong state.
        pool_idx = jnp.array([4, 1, 3], dtype=jnp.int32)
        ckpt_indices = jnp.array([[4, 7], [1, 9], [3, 5]], dtype=jnp.int32)

        keys = iter(jax.random.split(jax.random.key(21), 2))
        ssm0 = jax.random.normal(next(keys), (n, N_V, D_K, D_V))
        conv0 = jax.random.normal(next(keys), (n, KERNEL_SIZE - 1, DIM))
        pool = _write_states(pool, ssm0, conv0, pool_idx, split=split)

        # Dense reference: compact slot groups of `window` consecutive
        # slots, byte-equal initial states at each group's base. Garbage in
        # the other slots must never be read.
        rng = np.random.default_rng(22)
        num_slots = 1 + n * window
        dense_base = jnp.array([1 + i * window for i in range(n)], dtype=jnp.int32)
        dense_conv = jnp.asarray(
            rng.standard_normal((num_slots, KERNEL_SIZE - 1, DIM)), dtype=jnp.float32
        )
        dense_rec = jnp.asarray(
            rng.standard_normal((num_slots, N_V, D_K, D_V)), dtype=jnp.float32
        )
        dense_conv = dense_conv.at[dense_base].set(conv0)
        dense_rec = dense_rec.at[dense_base].set(ssm0.astype(jnp.float32))

        for step_num, step in enumerate(self._steps()):
            kwargs = _rand_inputs(30 + step_num, step["num_tokens"]) | dict(
                query_start_loc=step["query_start_loc"],
                distribution=step["distribution"],
                seq_lens=step["seq_lens"],
                num_spec_tokens=self.NUM_SPEC,
                read_offsets=step["read_offsets"],
            )
            (dense_conv, dense_rec), out_dense = wrapper.fused_conv1d_gdn(
                states=(dense_conv, dense_rec), state_indices=dense_base, **kwargs
            )
            pool_shape = pool.shape
            pool, out_pooled = wrapper.fused_conv1d_gdn(
                states=_merge(pool, split),
                state_indices=pool_idx,
                ckpt_indices=ckpt_indices,
                conv_dtype=jnp.float32,
                recurrent_dtype=jnp.float32,
                **kwargs,
            )
            pool = _split(pool, pool_shape)
            np.testing.assert_allclose(
                np.asarray(out_pooled),
                np.asarray(out_dense),
                rtol=2e-2,
                atol=2e-2,
                err_msg=f"step {step_num}",
            )


_write_states_bf16 = functools.partial(_write_states, ssm_dtype=jnp.bfloat16)
_read_states_bf16 = functools.partial(_read_states, ssm_dtype=jnp.bfloat16)


def _bits(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.int16))


@pytest.mark.tpu_test
class TestPooledBf16Recurrent:
    """bf16 recurrent state with an fp32 conv state.

    The kernel upcasts the recurrent state once on load and rounds it once on
    store. Against the fp32 kernel fed the same bf16-representable values, the
    recurrent state is bit-exact; against the tuple path it stays within 32
    bf16 ULPs, and conv state and output stay within standard float tolerance.
    """

    def _states(self, seed, n):
        keys = iter(jax.random.split(jax.random.key(seed), 2))
        ssm = jax.random.normal(next(keys), (n, N_V, D_K, D_V)).astype(jnp.bfloat16)
        conv = jax.random.normal(next(keys), (n, KERNEL_SIZE - 1, DIM))
        return ssm, conv

    def _tuple(self, ssm, conv, kwargs):
        ident = jnp.arange(ssm.shape[0], dtype=jnp.int32)
        (new_conv, new_ssm), out = wrapper.fused_conv1d_gdn(
            states=(conv, ssm), state_indices=ident, **kwargs
        )
        return new_ssm, new_conv, out

    def _pooled(self, ssm, conv, idx, kwargs, write=_write_states_bf16):
        pool = write(_garbage_pool(0), ssm, conv, idx)
        out_pool, out = wrapper.fused_conv1d_gdn(
            states=_merge(pool),
            state_indices=idx,
            conv_dtype=jnp.float32,
            recurrent_dtype=jnp.bfloat16,
            **kwargs,
        )
        new_ssm, new_conv = _read_states_bf16(_split(out_pool, pool.shape), idx)
        return new_ssm, new_conv, out

    def _assert_same(self, got, want, ssm_ulps=0):
        # Recurrent state: bf16 bit patterns may differ by at most `ssm_ulps`
        # units in the last place; 0 means bit-exact.
        diff = np.abs(_bits(got[0]).astype(np.int32) - _bits(want[0]).astype(np.int32))
        assert int(diff.max()) <= ssm_ulps
        np.testing.assert_allclose(
            np.asarray(got[1]), np.asarray(want[1]), rtol=2e-2, atol=2e-2
        )
        np.testing.assert_allclose(
            np.asarray(got[2]), np.asarray(want[2]), rtol=2e-2, atol=2e-2
        )

    def _mixed_kwargs(self, seed):
        # Two decodes (BATCHED) plus two 64-token continuations (PER_SEQ).
        return _rand_inputs(seed, 130) | dict(
            query_start_loc=jnp.array([0, 1, 2, 66, 130]),
            distribution=jnp.array([2, 4, 4], dtype=jnp.int32),
            seq_lens=jnp.array([7, 9, 96, 80], dtype=jnp.int32),
        )

    def test_pooled_bf16_recurrent_matches_tuple_path(self):
        idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        ssm, conv = self._states(40, 4)
        kwargs = self._mixed_kwargs(41)
        # The pooled and tuple paths compile to different programs, so their
        # fp32 results can differ in the last bits and round to adjacent bf16
        # values. Conv state and output stay within standard tolerance.
        self._assert_same(
            self._pooled(ssm, conv, idx, kwargs),
            self._tuple(ssm, conv, kwargs),
            ssm_ulps=32,
        )

    def test_bf16_recurrent_matches_fp32_on_bf16_values(self):
        ssm, conv = self._states(42, 4)
        kwargs = self._mixed_kwargs(43)
        # The kernel donates `states`, so each call gets its own buffers.
        ssm32, conv32 = ssm.astype(jnp.float32), conv.copy()
        got = self._tuple(ssm, conv, kwargs)
        new_ssm, new_conv, out = self._tuple(ssm32, conv32, kwargs)
        self._assert_same(got, (new_ssm.astype(jnp.bfloat16), new_conv, out))

    def test_multi_tile_prefill_with_bf16_recurrent_keeps_fp32_carry(self):
        # 200 tokens span 4 PER_SEQ tiles of 64: the carry between tiles must
        # stay fp32, so only the final state is rounded.
        ssm, conv = self._states(44, 1)
        kwargs = _rand_inputs(45, 200) | dict(
            query_start_loc=jnp.array([0, 200]),
            distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([230], dtype=jnp.int32),
        )
        ssm32, conv32 = ssm.astype(jnp.float32), conv.copy()
        got = self._tuple(ssm, conv, kwargs)
        new_ssm, new_conv, out = self._tuple(ssm32, conv32, kwargs)
        self._assert_same(got, (new_ssm.astype(jnp.bfloat16), new_conv, out))

    def test_spec_windows_with_bf16_recurrent_match_fp32_on_bf16_values(self):
        # Reuses the pooled spec schedule on the tuple path. Rounding the
        # fp32 arm between steps mirrors the bf16 store/load round trip.
        spec = TestPooledSpecWindows
        n, window = spec.N, spec.WINDOW
        base = jnp.array([1 + i * window for i in range(n)], dtype=jnp.int32)
        rng = np.random.default_rng(46)
        num_slots = 1 + n * window
        conv = jnp.asarray(
            rng.standard_normal((num_slots, KERNEL_SIZE - 1, DIM)), jnp.float32
        )
        rec = jnp.asarray(rng.standard_normal((num_slots, N_V, D_K, D_V)), jnp.bfloat16)
        conv16, rec16 = conv, rec
        conv32, rec32 = conv.copy(), rec.astype(jnp.float32)
        for step_num, step in enumerate(spec._steps(spec)):
            kwargs = _rand_inputs(47 + step_num, step["num_tokens"]) | dict(
                query_start_loc=step["query_start_loc"],
                distribution=step["distribution"],
                seq_lens=step["seq_lens"],
                num_spec_tokens=spec.NUM_SPEC,
                read_offsets=step["read_offsets"],
            )
            (conv16, rec16), out16 = wrapper.fused_conv1d_gdn(
                states=(conv16, rec16), state_indices=base, **kwargs
            )
            (conv32, rec32), out32 = wrapper.fused_conv1d_gdn(
                states=(conv32, rec32), state_indices=base, **kwargs
            )
            rec32 = rec32.astype(jnp.bfloat16)
            self._assert_same((rec16, conv16, out16), (rec32, conv32, out32))
            rec32 = rec32.astype(jnp.float32)

    def test_conv_first_pool_layout_is_detected(self):
        """Negative control: regions written in the other order must not match."""

        def write_conv_first(pool, ssm, conv, idx, split=SPLIT):
            def build(pool, ssm, conv, idx):
                words = state_words(pool, split)
                words = put_words(words, idx, 0, jnp.asarray(conv, jnp.float32))
                body = pack_bf16_rows(ssm).reshape(idx.shape[0], -1)
                start = CONV_GRANULES * WORDS_PER_GRANULE
                words = words.at[idx, start : start + body.shape[1]].set(body)
                return words_to_pool(words, pool)

            return on_cpu(build, pool, ssm, conv, idx)

        idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        ssm, conv = self._states(48, 4)
        kwargs = _rand_inputs(49, 4) | dict(
            query_start_loc=jnp.arange(5),
            distribution=jnp.array([4, 4, 4], dtype=jnp.int32),
            seq_lens=jnp.full((4,), 9, dtype=jnp.int32),
        )
        got = self._pooled(ssm, conv, idx, kwargs, write=write_conv_first)
        want = self._tuple(ssm, conv, kwargs)
        assert not np.array_equal(np.asarray(got[2]), np.asarray(want[2]))

    def test_bf16_conv_state_is_rejected(self):
        ssm, conv = self._states(50, 1)
        kwargs = _rand_inputs(51, 1) | dict(
            query_start_loc=jnp.arange(2),
            distribution=jnp.array([1, 1, 1], dtype=jnp.int32),
            seq_lens=jnp.array([3], dtype=jnp.int32),
        )
        with pytest.raises(NotImplementedError, match="fp32 conv state"):
            self._tuple(ssm, conv.astype(jnp.bfloat16), kwargs)


@pytest.mark.tpu_test
class TestPooledCallerV3:
    def test_fp32_ssm_round_trips_through_fp8_pool(self):
        """The pool's own element type is irrelevant: the DMA moves bytes."""
        gdn_attention = pytest.importorskip("vllm_torchtpu.layers.core.gdn_attention")
        n = 4
        pool_idx = jnp.arange(1, n + 1, dtype=jnp.int32)
        dense_idx = jnp.arange(n, dtype=jnp.int32)
        recurrent = jax.random.normal(jax.random.key(41), (n, N_V, D_K, D_V))
        conv = jnp.zeros((n, KERNEL_SIZE - 1, DIM), dtype=jnp.float32)
        pool = jnp.zeros((NUM_MGR * SPLIT, KBS, 1, 4, LANES), dtype=jnp.float8_e4m3fn)
        nwords = recurrent.size // n

        def _seed(pool, recurrent, pool_idx):
            body = recurrent.reshape(n, -1).view(jnp.uint32)
            words = state_words(pool, SPLIT).at[pool_idx, 0:nwords].set(body)
            return words_to_pool(words, pool)

        pool = on_cpu(_seed, pool, recurrent, pool_idx)
        kwargs = _rand_inputs(42, n) | dict(
            query_start_loc=jnp.arange(n + 1),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
        )

        conv_init, recurrent_init = conv.copy(), recurrent.copy()
        (_, dense_ssm), dense_out = wrapper.fused_conv1d_gdn(
            states=(conv, recurrent), state_indices=dense_idx, **kwargs
        )
        pooled = functools.partial(
            gdn_attention.run_jax_gdn_attention_pooled_local,
            mixed_qkv=kwargs["qkv"],
            b=kwargs["b"],
            a=kwargs["a"],
            conv_weight=kwargs["conv_weight"],
            conv_bias=kwargs["conv_bias"],
            A_log=kwargs["a_log"],
            dt_bias=kwargs["dt_bias"],
            query_start_loc=kwargs["query_start_loc"],
            state_indices=pool_idx,
            distribution=kwargs["distribution"],
            seq_lens=kwargs["seq_lens"],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=SPLIT * KBS,
        )
        new_pool, pooled_out = pooled(recurrent_state=pool)

        def _read(new_pool, pool_idx):
            return (
                state_words(new_pool, SPLIT)[pool_idx, 0:nwords]
                .view(jnp.float32)
                .reshape(n, N_V, D_K, D_V)
            )

        pooled_ssm = on_cpu(_read, new_pool, pool_idx)

        np.testing.assert_allclose(
            np.asarray(pooled_out), np.asarray(dense_out), rtol=5e-2, atol=5e-2
        )
        np.testing.assert_allclose(
            np.asarray(pooled_ssm), np.asarray(dense_ssm), rtol=5e-2, atol=5e-2
        )

        # A bf16 SSM cache takes the same caller path, two rows per word.
        recurrent16 = recurrent_init.astype(jnp.bfloat16)

        def _seed16(pool, recurrent16, pool_idx):
            body = pack_bf16_rows(recurrent16).reshape(n, -1)
            words = state_words(pool, SPLIT).at[pool_idx, 0 : body.shape[1]].set(body)
            return words_to_pool(words, pool)

        def _read16(new_pool, pool_idx):
            words = state_words(new_pool, SPLIT)[pool_idx, 0 : nwords // 2]
            return unpack_bf16_rows(words.reshape(n, N_V, D_K // 2, D_V))

        pool16 = on_cpu(_seed16, jnp.zeros_like(pool), recurrent16, pool_idx)
        (_, dense_ssm16), dense_out16 = wrapper.fused_conv1d_gdn(
            states=(conv_init, recurrent16), state_indices=dense_idx, **kwargs
        )
        new_pool16, pooled_out16 = pooled(
            recurrent_state=pool16, recurrent_state_dtype=jnp.bfloat16
        )
        pooled_ssm16 = on_cpu(_read16, new_pool16, pool_idx)

        np.testing.assert_allclose(
            np.asarray(pooled_out16, np.float32),
            np.asarray(dense_out16, np.float32),
            rtol=5e-2,
            atol=5e-2,
        )
        np.testing.assert_allclose(
            np.asarray(pooled_ssm16, np.float32),
            np.asarray(dense_ssm16, np.float32),
            rtol=5e-2,
            atol=5e-2,
        )

    def test_caller_matches_roundtrip(self):
        gdn_attention = pytest.importorskip("vllm_torchtpu.layers.core.gdn_attention")
        n = 4
        idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(14), 2))
        pool = _write_states(
            _garbage_pool(15),
            jax.random.normal(next(keys), (n, N_V, D_K, D_V)),
            jax.random.normal(next(keys), (n, KERNEL_SIZE - 1, DIM)).astype(
                jnp.bfloat16
            ),
            idx,
        )
        kwargs = _rand_inputs(16, n) | dict(
            query_start_loc=jnp.arange(n + 1),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
        )
        pool_rt, out_rt = _run_roundtrip(jnp.copy(pool), idx, kwargs)

        new_pool, out = gdn_attention.run_jax_gdn_attention_pooled_local(
            mixed_qkv=kwargs["qkv"],
            b=kwargs["b"],
            a=kwargs["a"],
            recurrent_state=jnp.copy(pool),
            conv_weight=kwargs["conv_weight"],
            conv_bias=kwargs["conv_bias"],
            A_log=kwargs["a_log"],
            dt_bias=kwargs["dt_bias"],
            query_start_loc=kwargs["query_start_loc"],
            state_indices=idx,
            distribution=kwargs["distribution"],
            seq_lens=kwargs["seq_lens"],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=SPLIT * KBS,
        )

        np.testing.assert_allclose(
            np.asarray(out), np.asarray(out_rt), rtol=1e-5, atol=1e-5
        )
        assert jnp.array_equal(new_pool.view(jnp.int16), pool_rt.view(jnp.int16))

    @pytest.mark.parametrize("page_size", [128, 256, 512])
    @pytest.mark.parametrize(
        ("num_kv_heads", "kv_packing", "head_dim", "pool_dtype"),
        [
            (1, 4, 256, jnp.float8_e4m3fn),
            (1, 2, 256, jnp.bfloat16),
            (2, 2, 256, jnp.bfloat16),
            (4, 4, 256, jnp.float8_e4m3fn),
        ],
    )
    def test_caller_matches_dense_with_native_seq_along_lane_pool(
        self, monkeypatch, page_size, num_kv_heads, kv_packing, head_dim, pool_dtype
    ):
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
        gdn_attention = pytest.importorskip("vllm_torchtpu.layers.core.gdn_attention")
        n = 4
        pool_idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        dense_idx = jnp.arange(n, dtype=jnp.int32)

        # A 1536-token manager block holds the state for every geometry here:
        # the smallest token row (one fp8 KV head pair, 512 B) needs
        # SSM=1024 + Conv=64 tokens.
        split = 1536 // page_size
        pool = jnp.zeros(
            (
                NUM_MGR * split,
                num_kv_heads * 2,
                head_dim // kv_packing,
                kv_packing,
                page_size,
            ),
            dtype=pool_dtype,
        )
        conv = jnp.zeros((n, KERNEL_SIZE - 1, DIM), dtype=jnp.float32)
        recurrent_f32 = jnp.zeros((n, N_V, D_K, D_V), dtype=jnp.float32)

        kwargs = _rand_inputs(17, n) | dict(
            query_start_loc=jnp.arange(n + 1),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n,), 9, dtype=jnp.int32),
        )
        (dense_conv, dense_ssm), dense_out = wrapper.fused_conv1d_gdn(
            states=(conv, recurrent_f32),
            state_indices=dense_idx,
            **kwargs,
        )

        new_pool, pooled_out = gdn_attention.run_jax_gdn_attention_pooled_local(
            mixed_qkv=kwargs["qkv"],
            b=kwargs["b"],
            a=kwargs["a"],
            recurrent_state=pool,
            conv_weight=kwargs["conv_weight"],
            conv_bias=kwargs["conv_bias"],
            A_log=kwargs["a_log"],
            dt_bias=kwargs["dt_bias"],
            query_start_loc=kwargs["query_start_loc"],
            state_indices=pool_idx,
            distribution=kwargs["distribution"],
            seq_lens=kwargs["seq_lens"],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=split * page_size,
            recurrent_state_dtype=jnp.float32,
        )
        assert new_pool.shape == pool.shape
        np.testing.assert_allclose(
            np.asarray(pooled_out), np.asarray(dense_out), rtol=1e-5, atol=1e-5
        )

        # state update by feeding next token into both dense
        kwargs2 = kwargs | dict(
            qkv=jax.random.normal(jax.random.key(18), (n, DIM)),
            b=jax.random.normal(jax.random.key(19), (n, N_V)),
            a=jax.random.normal(jax.random.key(20), (n, N_V)),
            seq_lens=jnp.full((n,), 10, dtype=jnp.int32),
        )
        (_, _), dense_out2 = wrapper.fused_conv1d_gdn(
            states=(dense_conv, dense_ssm),
            state_indices=dense_idx,
            **kwargs2,
        )

        new_pool2, pooled_out2 = gdn_attention.run_jax_gdn_attention_pooled_local(
            mixed_qkv=kwargs2["qkv"],
            b=kwargs2["b"],
            a=kwargs2["a"],
            recurrent_state=new_pool,
            conv_weight=kwargs2["conv_weight"],
            conv_bias=kwargs2["conv_bias"],
            A_log=kwargs2["a_log"],
            dt_bias=kwargs2["dt_bias"],
            query_start_loc=kwargs2["query_start_loc"],
            state_indices=pool_idx,
            distribution=kwargs2["distribution"],
            seq_lens=kwargs2["seq_lens"],
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D_K,
            d_v=D_V,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=split * page_size,
            recurrent_state_dtype=jnp.float32,
        )
        assert new_pool2.shape == pool.shape
        np.testing.assert_allclose(
            np.asarray(pooled_out2), np.asarray(dense_out2), rtol=1e-3, atol=1e-3
        )
