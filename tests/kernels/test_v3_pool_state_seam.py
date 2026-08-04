"""GDN V3 pooled state seam: plan math, byte compatibility, equivalence.

`TestV3StateSourcePlan` and the source-hygiene test are pure Python/CPU.
The remaining classes execute Pallas kernels and run on TPU (same gating
as test_pool_adapters.py).
"""
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
POOL_SHAPE = (NUM_MGR * SPLIT, KBS) + PAYLOAD + (LANES, )


def _plan(pool=None, **overrides):
    if pool is None:
        pool = jax.ShapeDtypeStruct(POOL_SHAPE, jnp.bfloat16)
    kwargs = dict(split=SPLIT,
                  ssm_ntok=SSM_NTOK,
                  conv_tok0=CONV_TOK0,
                  conv_ntok=CONV_NTOK,
                  conv_dim=DIM,
                  n_v=N_V,
                  d_k=D_K,
                  d_v=D_V,
                  kernel_size=KERNEL_SIZE)
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

    def test_plan_is_static_hashable_and_stable(self):
        assert _plan() == _plan()
        assert hash(_plan()) == hash(_plan())

    def test_subblock_regions_when_pool_is_manager_grained(self):
        pool = jax.ShapeDtypeStruct(
            (NUM_MGR, SPLIT * KBS) + PAYLOAD + (LANES, ), jnp.bfloat16)
        plan = _plan(pool, split=1)
        r = plan.recurrent
        assert (r.kb0, r.nblocks, r.row0, r.nrows) == (0, 1, 0, SSM_NTOK)
        c = plan.conv
        assert (c.kb0, c.nblocks, c.row0, c.nrows) == (0, 1, CONV_TOK0,
                                                       CONV_NTOK)
        assert plan.stride == 1

    def test_one_byte_pool_keeps_regions_and_view_dtypes(self):
        pool = jax.ShapeDtypeStruct((NUM_MGR * SPLIT, KBS, 1, 4, LANES),
                                    jnp.int8)
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


def _garbage_pool(seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(POOL_SHAPE), dtype=jnp.bfloat16)


def _write_states(pool, ssm, conv, idx):
    """Adapter-convention write of both regions (the byte layout the seam
    must reproduce exactly)."""
    n = idx.shape[0]
    pool = pool_adapters.scatter_region(pool,
                                        ssm.astype(jnp.float32).reshape(
                                            n, N_V * D_K, D_V),
                                        idx,
                                        tok0=0,
                                        ntok=SSM_NTOK,
                                        split=SPLIT)
    rows = conv.astype(jnp.bfloat16).reshape(n, CONV_ROWS, LANES)
    rows = jnp.pad(rows, ((0, 0), (0, CONV_SLOT_ROWS - CONV_ROWS), (0, 0)))
    return pool_adapters.scatter_region(pool,
                                        rows,
                                        idx,
                                        tok0=CONV_TOK0,
                                        ntok=CONV_NTOK,
                                        split=SPLIT)


def _read_states(pool, idx):
    n = idx.shape[0]
    ssm = pool_adapters.gather_region(pool,
                                      idx,
                                      tok0=0,
                                      ntok=SSM_NTOK,
                                      split=SPLIT,
                                      out_dtype=jnp.float32,
                                      out_lanes=D_V).reshape(n, N_V, D_K, D_V)
    conv = pool_adapters.gather_region(pool,
                                       idx,
                                       tok0=CONV_TOK0,
                                       ntok=CONV_NTOK,
                                       split=SPLIT,
                                       out_dtype=jnp.bfloat16)
    conv = conv[:, :CONV_ROWS, :].reshape(n, KERNEL_SIZE - 1, DIM)
    return ssm, conv


def _rand_inputs(seed, num_tokens):
    rngs = iter(jax.random.split(jax.random.key(seed), 8))
    return dict(
        qkv=jax.random.normal(next(rngs), (num_tokens, DIM)),
        b=jax.random.normal(next(rngs), (num_tokens, N_V)),
        a=jax.random.normal(next(rngs), (num_tokens, N_V)),
        conv_weight=jax.random.normal(next(rngs), (DIM, 1, KERNEL_SIZE)),
        conv_bias=jax.random.normal(next(rngs), (DIM, )),
        a_log=jax.random.normal(next(rngs), (N_V, )),
        dt_bias=jax.random.normal(next(rngs), (N_V, )),
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
    (new_conv, new_ssm), out = wrapper.fused_conv1d_gdn(conv_state=conv,
                                                        recurrent_state=ssm,
                                                        state_indices=identity,
                                                        **kwargs)
    return _write_states(pool, new_ssm, new_conv, idx), out


def _run_seam(pool, idx, kwargs):
    return wrapper.fused_conv1d_gdn(conv_state=None,
                                    recurrent_state=None,
                                    state_indices=idx,
                                    state_source=pool,
                                    state_plan=_plan(pool),
                                    **kwargs)


class TestSeamMatchesRoundTrip:

    def _compare(self, pool, idx, kwargs):
        pool_rt, out_rt = _run_roundtrip(jnp.copy(pool), idx, kwargs)
        pool_seam, out_seam = _run_seam(jnp.copy(pool), idx, kwargs)

        # The two paths lay the state out differently in VMEM, so the
        # f32 accumulations reassociate: outputs and states agree to a
        # couple of bf16 ulps (operand precision), not bitwise.
        tol = dict(rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(np.asarray(out_seam), np.asarray(out_rt),
                                   **tol)
        ssm_rt, conv_rt = _read_states(pool_rt, idx)
        ssm_seam, conv_seam = _read_states(pool_seam, idx)
        np.testing.assert_allclose(np.asarray(ssm_seam), np.asarray(ssm_rt),
                                   **tol)
        np.testing.assert_allclose(np.asarray(conv_seam, dtype=np.float32),
                                   np.asarray(conv_rt, dtype=np.float32),
                                   rtol=2e-2,
                                   atol=2e-2)
        # Blocks neither path addresses must stay byte-identical; the
        # addressed windows are covered by the logical compares above and
        # the write-layout tests (seam bytes == scatter_region bytes for
        # identical values).
        touched = set()
        for mgr in np.asarray(idx):
            touched.update(range(int(mgr) * SPLIT, int(mgr) * SPLIT + SPLIT))
        for kb in range(pool_seam.shape[0]):
            if kb not in touched:
                assert jnp.array_equal(pool_seam[kb].view(jnp.int16),
                                       pool_rt[kb].view(jnp.int16)), kb

    def test_decode_continuation(self):
        n = 4
        idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(10), 2))
        pool = _write_states(
            _garbage_pool(0), jax.random.normal(next(keys),
                                                (n, N_V, D_K, D_V)),
            jax.random.normal(next(keys),
                              (n, KERNEL_SIZE - 1, DIM)).astype(jnp.bfloat16),
            idx)
        kwargs = _rand_inputs(1, n) | dict(
            query_start_loc=jnp.arange(n + 1),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n, ), 9, dtype=jnp.int32),
        )
        self._compare(pool, idx, kwargs)

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
            _garbage_pool(4), jax.random.normal(next(keys),
                                                (2, N_V, D_K, D_V)),
            jax.random.normal(next(keys),
                              (2, KERNEL_SIZE - 1, DIM)).astype(jnp.bfloat16),
            idx)
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
            _garbage_pool(6), jax.random.normal(next(keys),
                                                (2, N_V, D_K, D_V)),
            jax.random.normal(next(keys),
                              (2, KERNEL_SIZE - 1, DIM)).astype(jnp.bfloat16),
            decode_idx)
        kwargs = _rand_inputs(7, 130) | dict(
            query_start_loc=jnp.array([0, 1, 2, 66, 130]),
            distribution=jnp.array([2, 4, 4], dtype=jnp.int32),
            seq_lens=jnp.array([7, 9, 64, 64], dtype=jnp.int32),
        )
        self._compare(pool, idx, kwargs)


class TestSeamSkipsInvalidSlots:

    def test_padded_slots_and_null_block_untouched(self):
        # 2 active decodes on manager blocks 2 and 4; rows 2..3 padded to
        # the null block. Only the active windows' state regions may move.
        idx = jnp.array([2, 4, 0, 0], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(13), 2))
        pool = _write_states(
            _garbage_pool(8), jax.random.normal(next(keys),
                                                (2, N_V, D_K, D_V)),
            jax.random.normal(next(keys),
                              (2, KERNEL_SIZE - 1, DIM)).astype(jnp.bfloat16),
            idx[:2])
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
            assert not jnp.array_equal(b16(after[mgr * SPLIT]),
                                       b16(before[mgr * SPLIT])), mgr
            # ...and the conv block's rows outside the conv region kept.
            conv_kb = mgr * SPLIT + 2
            assert jnp.array_equal(b16(after[conv_kb, CONV_NTOK:]),
                                   b16(before[conv_kb, CONV_NTOK:])), mgr


class TestPooledCallerV3:

    def test_caller_matches_roundtrip(self):
        gdn_attention = pytest.importorskip(
            "vllm_torchtpu.layers.common.gdn_attention")
        n = 4
        idx = jnp.array([1, 2, 3, 4], dtype=jnp.int32)
        keys = iter(jax.random.split(jax.random.key(14), 2))
        pool = _write_states(
            _garbage_pool(15), jax.random.normal(next(keys),
                                                 (n, N_V, D_K, D_V)),
            jax.random.normal(next(keys),
                              (n, KERNEL_SIZE - 1, DIM)).astype(jnp.bfloat16),
            idx)
        kwargs = _rand_inputs(16, n) | dict(
            query_start_loc=jnp.arange(n + 1),
            distribution=jnp.array([n, n, n], dtype=jnp.int32),
            seq_lens=jnp.full((n, ), 9, dtype=jnp.int32),
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

        np.testing.assert_allclose(np.asarray(out),
                                   np.asarray(out_rt),
                                   rtol=1e-5,
                                   atol=1e-5)
        assert jnp.array_equal(new_pool.view(jnp.int16),
                               pool_rt.view(jnp.int16))
