"""TPU roundtrip tests for the unified-pool region adapters."""
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import pool_adapters

# The real RPA v3 bf16 layout: (blocks, block_size, heads_x2 // packing,
# kv_packing, padded_head) with packing 2.
NB, BS, KH, PACK, LANES = 4, 64, 2, 2, 128
TOK_BYTES = KH * PACK * LANES * 2  # bf16 pool


def _pool():
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal((NB, BS, KH, PACK, LANES)),
                       dtype=jnp.bfloat16)


def test_f32_region_roundtrip_preserves_complement():
    pool = _pool()
    before = np.asarray(pool)
    idx = jnp.asarray([2, 1], dtype=jnp.int32)
    ntok = 32  # f32 region of 32 tokens at block start
    rows = ntok * TOK_BYTES // (4 * LANES)
    rng = np.random.default_rng(1)
    vals = jnp.asarray(rng.standard_normal((2, rows, LANES)),
                       dtype=jnp.float32)

    pool = pool_adapters.scatter_region(pool, vals, idx, tok0=0, ntok=ntok)
    got = pool_adapters.gather_region(pool,
                                      idx,
                                      tok0=0,
                                      ntok=ntok,
                                      out_dtype=jnp.float32)

    np.testing.assert_array_equal(np.asarray(got), np.asarray(vals))
    # Tokens outside the region are untouched.
    after = np.asarray(pool)
    np.testing.assert_array_equal(after[:, ntok:], before[:, ntok:])
    untouched_blocks = [b for b in range(NB) if b not in (1, 2)]
    np.testing.assert_array_equal(after[untouched_blocks],
                                  before[untouched_blocks])


def test_same_dtype_region_roundtrip():
    pool = _pool()
    idx = jnp.asarray([3, 0], dtype=jnp.int32)
    tok0, ntok = 32, 8  # conv-style slot after the ssm region
    rows = ntok * TOK_BYTES // (2 * LANES)
    rng = np.random.default_rng(2)
    vals = jnp.asarray(rng.standard_normal((2, rows, LANES)),
                       dtype=jnp.bfloat16)

    pool = pool_adapters.scatter_region(pool, vals, idx, tok0=tok0, ntok=ntok)
    got = pool_adapters.gather_region(pool,
                                      idx,
                                      tok0=tok0,
                                      ntok=ntok,
                                      out_dtype=jnp.bfloat16)

    np.testing.assert_array_equal(np.asarray(got), np.asarray(vals))


def test_copy_blocks():
    pool = _pool()
    before = np.asarray(pool)
    src = jnp.asarray([0, 2], dtype=jnp.int32)
    dst = jnp.asarray([3, 1], dtype=jnp.int32)

    pool = pool_adapters.copy_blocks(pool, src, dst)

    after = np.asarray(pool)
    np.testing.assert_array_equal(after[3], before[0])
    np.testing.assert_array_equal(after[1], before[2])
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[2], before[2])


def test_misaligned_region_rejected():
    pool = _pool()
    idx = jnp.asarray([0], dtype=jnp.int32)
    with pytest.raises(AssertionError):
        pool_adapters.gather_region(pool,
                                    idx,
                                    tok0=8,
                                    ntok=32,
                                    out_dtype=jnp.float32)


class TestSidePoolGeometry:
    """The envelope side pool: bf16 (nb, 1, rows, 128), same-dtype payload.

    The side pool is typed bf16 (the conv payload dtype): the adapters'
    cross-dtype bitcast does NOT roundtrip for an int8 pool with bf16
    values at this geometry, so same-dtype access is required here.
    """

    NB, ROWS, LANES = 4, 48, 128

    def _side_pool(self):
        import jax.numpy as jnp
        vals = jnp.arange(self.NB * self.ROWS * self.LANES,
                          dtype=jnp.int32) % 251
        return vals.astype(jnp.bfloat16).reshape(self.NB, 1, self.ROWS,
                                                 self.LANES)

    def test_bf16_roundtrip_on_side_pool(self):
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._side_pool()
        idx = jnp.array([2, 1], dtype=jnp.int32)
        got = pool_adapters.gather_region(pool,
                                          idx,
                                          tok0=0,
                                          ntok=1,
                                          out_dtype=jnp.bfloat16)
        assert got.shape == (2, self.ROWS, self.LANES)
        new_vals = got[::-1]
        pool2 = pool_adapters.scatter_region(pool,
                                             new_vals,
                                             idx,
                                             tok0=0,
                                             ntok=1)
        got2 = pool_adapters.gather_region(pool2,
                                           idx,
                                           tok0=0,
                                           ntok=1,
                                           out_dtype=jnp.bfloat16)
        assert (got2 == new_vals).all()
        # untouched block intact
        assert (pool2[0] == pool[0]).all()

    def test_copy_blocks_on_side_pool(self):
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._side_pool()
        src = jnp.array([3, 0], dtype=jnp.int32)
        dst = jnp.array([1, 2], dtype=jnp.int32)
        pool2 = pool_adapters.copy_blocks(pool, src, dst)
        assert (pool2[1] == pool[3]).all()
        assert (pool2[2] == pool[0]).all()
        assert (pool2[0] == pool[0]).all()
        assert (pool2[3] == pool[3]).all()


class TestInt8PoolPairs:
    """Adapter dtype pairs an int8-born pool needs: f32 ssm (ratio 4) and
    bf16 conv (ratio 2), at a real in-block geometry (block 512-token
    envelope: payload rows are int8 bytes, lanes 128)."""

    NB, BS, ROWS, LANES = 4, 1, 1024, 128  # 1024 int8 rows = 512 bf16 / 256 f32

    def _pool(self):
        import jax.numpy as jnp
        vals = jnp.arange(self.NB * self.BS * self.ROWS * self.LANES,
                          dtype=jnp.int32) % 251
        return vals.astype(jnp.int8).reshape(self.NB, self.BS, self.ROWS,
                                             self.LANES)

    def _roundtrip(self, out_dtype, rows_out):
        # Byte-level check (arbitrary bytes make NaNs; float == is unusable):
        # gather blocks [2,1], scatter them back swapped, and assert the
        # int8 bytes of the blocks swapped exactly.
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._pool()
        idx = jnp.array([2, 1], dtype=jnp.int32)
        got = pool_adapters.gather_region(pool,
                                          idx,
                                          tok0=0,
                                          ntok=self.BS,
                                          out_dtype=out_dtype)
        assert got.shape == (2, rows_out, self.LANES), got.shape
        pool2 = pool_adapters.scatter_region(pool,
                                             got[::-1],
                                             idx,
                                             tok0=0,
                                             ntok=self.BS)
        assert (pool2[2] == pool[1]).all()
        assert (pool2[1] == pool[2]).all()
        assert (pool2[0] == pool[0]).all()

    def test_f32_roundtrip_on_int8_pool(self):
        import jax.numpy as jnp
        self._roundtrip(jnp.float32, self.ROWS // 4)

    def test_bf16_roundtrip_on_int8_pool(self):
        import jax.numpy as jnp
        self._roundtrip(jnp.bfloat16, self.ROWS // 2)


class TestInt8Pool3D:
    """1-byte pools as 3-D (nb, rows, lanes): the block ref is 2-D, so
    the bitcast follows the old validated sublane convention."""

    NB, ROWS, LANES = 4, 1024, 128

    def _pool(self):
        import jax.numpy as jnp
        vals = jnp.arange(self.NB * self.ROWS * self.LANES,
                          dtype=jnp.int32) % 251
        return vals.astype(jnp.int8).reshape(self.NB, self.ROWS, self.LANES)

    def _roundtrip(self, out_dtype, ratio):
        # Byte-level swap check; float == is unusable on arbitrary bytes.
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._pool()
        idx = jnp.array([2, 1], dtype=jnp.int32)
        got = pool_adapters.gather_region(pool,
                                          idx,
                                          tok0=0,
                                          ntok=self.ROWS,
                                          out_dtype=out_dtype)
        assert got.shape == (2, self.ROWS // ratio, self.LANES), got.shape
        pool2 = pool_adapters.scatter_region(pool,
                                             got[::-1],
                                             idx,
                                             tok0=0,
                                             ntok=self.ROWS)
        assert (pool2[2] == pool[1]).all()
        assert (pool2[1] == pool[2]).all()
        assert (pool2[0] == pool[0]).all()

    def test_f32_roundtrip_3d(self):
        import jax.numpy as jnp
        self._roundtrip(jnp.float32, 4)

    def test_bf16_roundtrip_3d(self):
        import jax.numpy as jnp
        self._roundtrip(jnp.bfloat16, 2)

    def test_partial_range_f32_3d(self):
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._pool()
        idx = jnp.array([3], dtype=jnp.int32)
        # second half of the block only; complement must be intact
        half = self.ROWS // 2
        got = pool_adapters.gather_region(pool,
                                          idx,
                                          tok0=half,
                                          ntok=half,
                                          out_dtype=jnp.float32)
        # 1.5f has no NaN issues: safe for float comparison.
        new_vals = jnp.full_like(got, 1.5)
        pool2 = pool_adapters.scatter_region(pool,
                                             new_vals,
                                             idx,
                                             tok0=half,
                                             ntok=half)
        assert (pool2[3, :half] == pool[3, :half]).all()
        got2 = pool_adapters.gather_region(pool2,
                                           idx,
                                           tok0=half,
                                           ntok=half,
                                           out_dtype=jnp.float32)
        assert (got2 == new_vals).all()


class TestFp8PoolSsmPattern:
    """The fp8-config ssm access: f32 region at tok0=0 (partial range from
    row 0) of a 1-byte 4-D pool — the exact production corner."""

    NB, BS, LANES = 4, 1024, 128

    def test_partial_from_zero_f32_on_fp8_pool(self):
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        vals = jnp.arange(self.NB * self.BS * self.LANES,
                          dtype=jnp.int32) % 251
        pool = vals.astype(jnp.int8).reshape(
            self.NB, self.BS, self.LANES).view(jnp.float8_e4m3fn)
        idx = jnp.array([2], dtype=jnp.int32)
        half = self.BS // 2
        got = pool_adapters.gather_region(pool,
                                          idx,
                                          tok0=0,
                                          ntok=half,
                                          out_dtype=jnp.float32)
        new_vals = jnp.full_like(got, 1.5)
        pool2 = pool_adapters.scatter_region(pool,
                                             new_vals,
                                             idx,
                                             tok0=0,
                                             ntok=half)
        got2 = pool_adapters.gather_region(pool2,
                                           idx,
                                           tok0=0,
                                           ntok=half,
                                           out_dtype=jnp.float32)
        assert (got2 == new_vals).all()
        # complement (rows half..BS) byte-intact
        assert (pool2[2, half:].view(jnp.int8) == pool[2, half:].view(
            jnp.int8)).all()
        assert (pool2[0].view(jnp.int8) == pool[0].view(jnp.int8)).all()


class TestLaneSplitGather:
    """out_lanes narrower than pool lanes: each 128-wide lane slice of the
    wide view carries a contiguous range of the narrow view's rows, so the
    narrow view is the wide view's lane slices concatenated along rows."""

    NB, BS, H2P, PACK, LANES = 3, 64, 2, 2, 256
    OUT_LANES = 128

    def _pool(self):
        import jax.numpy as jnp
        n = self.NB * self.BS * self.H2P * self.PACK * self.LANES
        vals = (jnp.arange(n, dtype=jnp.int32) % 253).astype(jnp.bfloat16)
        return vals.reshape(self.NB, self.BS, self.H2P, self.PACK, self.LANES)

    def test_f32_lane_split_matches_reference(self):
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._pool()
        idx = jnp.array([2, 0], dtype=jnp.int32)
        wide = pool_adapters.gather_region(pool,
                                           idx,
                                           tok0=0,
                                           ntok=self.BS,
                                           out_dtype=jnp.float32)
        narrow = pool_adapters.gather_region(pool,
                                             idx,
                                             tok0=0,
                                             ntok=self.BS,
                                             out_dtype=jnp.float32,
                                             out_lanes=self.OUT_LANES)
        ref = jnp.concatenate(
            [wide[:, :, :self.OUT_LANES], wide[:, :, self.OUT_LANES:]], axis=1)
        assert narrow.shape == ref.shape, (narrow.shape, ref.shape)
        assert (narrow.view(jnp.int32) == ref.view(jnp.int32)).all()

    def test_f32_lane_merge_scatter_roundtrip(self):
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._pool()
        idx = jnp.array([1], dtype=jnp.int32)
        narrow = pool_adapters.gather_region(pool,
                                             idx,
                                             tok0=0,
                                             ntok=self.BS,
                                             out_dtype=jnp.float32,
                                             out_lanes=self.OUT_LANES)
        pool2 = pool_adapters.scatter_region(pool,
                                             narrow,
                                             idx,
                                             tok0=0,
                                             ntok=self.BS)
        # writing back exactly what was read must leave bytes unchanged
        assert (pool2.view(jnp.int16) == pool.view(jnp.int16)).all()


class TestPartialSelfConsistency:
    """Production needs gather/scatter to be mutual inverses per window
    (linear byte order is NOT the contract — Mosaic bitcast interleaves
    everywhere, full-range included). Map which windows on 1-byte pools
    preserve that: sizes 128/256/512, offsets 0/512, and the two
    compositions."""

    NB, ROWS, LANES = 4, 1024, 128

    def _pool(self):
        import jax.numpy as jnp
        n = self.NB * self.ROWS * self.LANES
        vals = jnp.arange(n, dtype=jnp.int32) % 251
        return vals.astype(jnp.int8).reshape(self.NB, self.ROWS, self.LANES)

    def _scatter_gather_id(self, tok0, ntok):
        # gather∘scatter = id: write known f32 vals, read back
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._pool()
        idx = jnp.array([2], dtype=jnp.int32)
        rows = (ntok * self.LANES) // 4 // self.LANES
        vals = (jnp.arange(rows * self.LANES, dtype=jnp.float32) +
                1.5).reshape(1, rows, self.LANES)
        pool2 = pool_adapters.scatter_region(pool,
                                             vals,
                                             idx,
                                             tok0=tok0,
                                             ntok=ntok)
        got = pool_adapters.gather_region(pool2,
                                          idx,
                                          tok0=tok0,
                                          ntok=ntok,
                                          out_dtype=jnp.float32)
        ok = bool((got == vals).all())
        comp = bool(
            (pool2[2, :tok0] == pool[2, :tok0]).all()
            and (pool2[2, tok0 + ntok:] == pool[2, tok0 + ntok:]).all())
        assert ok and comp, f"tok0={tok0} ntok={ntok} roundtrip={ok} complement={comp}"

    def _gather_scatter_id(self, tok0, ntok):
        # scatter∘gather = id: read, write back, pool bytes unchanged
        import jax.numpy as jnp

        from vllm_torchtpu.kernels import pool_adapters
        pool = self._pool()
        idx = jnp.array([2], dtype=jnp.int32)
        got = pool_adapters.gather_region(pool,
                                          idx,
                                          tok0=tok0,
                                          ntok=ntok,
                                          out_dtype=jnp.float32)
        pool2 = pool_adapters.scatter_region(pool,
                                             got,
                                             idx,
                                             tok0=tok0,
                                             ntok=ntok)
        assert bool(
            (pool2 == pool).all()), f"tok0={tok0} ntok={ntok} bytes changed"

    def test_sg_id_w128_t0(self):
        self._scatter_gather_id(0, 128)

    def test_sg_id_w256_t0(self):
        self._scatter_gather_id(0, 256)

    def test_sg_id_w512_t0(self):
        self._scatter_gather_id(0, 512)

    def test_sg_id_w512_t512(self):
        self._scatter_gather_id(512, 512)

    def test_sg_id_w128_t512(self):
        # tok0 % ntok == 0 required: 512 % 128 == 0 ok
        self._scatter_gather_id(512, 128)

    def test_gs_id_w512_t0(self):
        self._gather_scatter_id(0, 512)

    def test_gs_id_w512_t512(self):
        self._gather_scatter_id(512, 512)


def _arange_pool(shape, dtype):
    import math
    n = math.prod(shape)
    return (jnp.arange(n, dtype=jnp.float32) %
            251).astype(dtype).reshape(shape)


class TestGatherScatterBlocks:
    """split>1: a manager-block token range routes through one DMA window
    per request and equals the per-kernel-block gathers concatenated."""

    def test_split_gather_matches_per_block_gathers(self):
        pool = _arange_pool((12, 16, 2, 128), jnp.bfloat16)
        mgr = jnp.array([0, 1, 2], dtype=jnp.int32)
        multi = pool_adapters.gather_region(pool,
                                            mgr,
                                            tok0=0,
                                            ntok=32,
                                            out_dtype=jnp.float32,
                                            out_lanes=128,
                                            split=3)
        singles = jnp.concatenate([
            pool_adapters.gather_region(pool,
                                        mgr * 3 + j,
                                        tok0=0,
                                        ntok=16,
                                        out_dtype=jnp.float32,
                                        out_lanes=128) for j in range(2)
        ],
                                  axis=1)
        assert multi.shape == singles.shape
        assert jnp.array_equal(multi.view(jnp.uint32),
                               singles.view(jnp.uint32))

    def test_split_subblock_region_matches_offset_call(self):
        # a range inside ONE kernel block (the conv shape) routes to the
        # plain path at the kernel-block offset
        pool = _arange_pool((12, 16, 2, 128), jnp.bfloat16)
        mgr = jnp.array([0, 3], dtype=jnp.int32)
        got = pool_adapters.gather_region(pool,
                                          mgr,
                                          tok0=32,
                                          ntok=8,
                                          out_dtype=jnp.bfloat16,
                                          split=3)
        want = pool_adapters.gather_region(pool,
                                           mgr * 3 + 2,
                                           tok0=0,
                                           ntok=8,
                                           out_dtype=jnp.bfloat16)
        assert jnp.array_equal(got.view(jnp.uint16), want.view(jnp.uint16))


class TestGatherScatterBlocksFp8:
    """fp8 (1-byte) pool with f32 region access — the ratio-4 bitcast on a
    manager-block window (the 397B fp8 serving shape)."""

    def _pool(self):
        # (nb, block, k2p, p, hd): fp8 packing p=4, 5-D like the real cache
        shape = (12, 16, 2, 4, 128)
        import math
        n = math.prod(shape)
        return (jnp.arange(n, dtype=jnp.float32) % 240).astype(
            jnp.float8_e4m3fn).reshape(shape)

    def test_split_gather_fp8_to_f32_matches_per_block(self):
        pool = self._pool()
        mgr = jnp.array([0, 2], dtype=jnp.int32)
        multi = pool_adapters.gather_region(pool,
                                            mgr,
                                            tok0=0,
                                            ntok=32,
                                            out_dtype=jnp.float32,
                                            out_lanes=128,
                                            split=3)
        singles = jnp.concatenate([
            pool_adapters.gather_region(pool,
                                        mgr * 3 + j,
                                        tok0=0,
                                        ntok=16,
                                        out_dtype=jnp.float32,
                                        out_lanes=128) for j in range(2)
        ],
                                  axis=1)
        assert jnp.array_equal(multi.view(jnp.uint32),
                               singles.view(jnp.uint32))

    def test_split_scatter_fp8_roundtrip(self):
        pool = self._pool()
        mgr = jnp.array([1, 3], dtype=jnp.int32)
        vals = pool_adapters.gather_region(pool,
                                           mgr,
                                           tok0=0,
                                           ntok=32,
                                           out_dtype=jnp.float32,
                                           split=3)
        new_pool = pool_adapters.scatter_region(pool,
                                                vals,
                                                mgr,
                                                tok0=0,
                                                ntok=32,
                                                split=3)
        # pure roundtrip: bytes unchanged everywhere (incl. the
        # copied-through third kernel block of each manager block)
        assert jnp.array_equal(new_pool.view(jnp.uint8), pool.view(jnp.uint8))
