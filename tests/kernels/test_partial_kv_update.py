"""TPU tests for the in-place partial slot update kernel (b/546309078).

The cases here are the ones that actually bit during development, not a
generic sweep. Each names the failure it guards against, because several of
them look redundant until you know what got through without them:

* indices colliding inside one sublane block. The rank-2 path reads a block,
  edits it and writes it back, and Pallas does not refetch a block whose index
  has not changed -- so a step applying only its own row reads the original
  pool and reverts its predecessors. A benchmark whose indices are evenly
  spread never collides and never notices.
* duplicate indices. Two writes to one slot must resolve the way PyTorch and
  numpy resolve them, last-write-wins, which is not automatic once updates are
  applied by masked select rather than in sequence.
* the complement. A kernel that writes correct values into the target slots
  and quietly disturbs their neighbours passes a naive value check.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.partial_kv_update import pallas_scatter_slots

SUBLANES = 8
LANES = 128


def _pool(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)


def _vals(n, trailing, seed=1):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((n,) + trailing), dtype=jnp.bfloat16)


def _reference(pool, indices, vals):
    """numpy's fancy-index assignment, which is also PyTorch's semantics."""
    want = np.asarray(pool).copy()
    want[np.asarray(indices)] = np.asarray(vals)
    return want


def _scatter(pool, indices, vals):
    idx = jnp.asarray(indices, dtype=jnp.int32)
    return np.asarray(pallas_scatter_slots(pool, idx, vals))


@pytest.mark.parametrize(
    "shape",
    [
        (4096, LANES * 8),  # rank 2, the shape the bug was filed against
        (4096, SUBLANES, LANES),  # rank 3, the canonical pool shape
        (512, 16, 16, LANES),  # rank 4, paged attention shaped
    ],
)
def test_scatter_matches_reference(shape):
    pool = _pool(shape)
    indices = [5, 1, 6, 11, 0]
    vals = _vals(len(indices), shape[1:])

    got = _scatter(pool, indices, vals)

    np.testing.assert_array_equal(got, _reference(pool, indices, vals))


@pytest.mark.parametrize("shape", [(4096, LANES * 8), (4096, SUBLANES, LANES)])
def test_unselected_slots_are_untouched(shape):
    """The whole point: everything outside the written slots must survive."""
    pool = _pool(shape)
    before = np.asarray(pool).copy()
    indices = [5, 1, 6]
    vals = _vals(len(indices), shape[1:])

    got = _scatter(pool, indices, vals)

    mask = np.ones(shape[0], dtype=bool)
    mask[indices] = False
    np.testing.assert_array_equal(got[mask], before[mask])


@pytest.mark.parametrize("shape", [(4096, LANES * 8), (4096, SUBLANES, LANES)])
def test_indices_colliding_in_one_sublane_block(shape):
    """Every index inside a single 8-row block.

    The rank-2 path applies every update targeting a block rather than only
    the current step's, which is what makes each step idempotent. Without that
    the last step to touch the block wins and the earlier updates are lost:
    the first version of this kernel reverted two of three rows here, 25% of
    the tensor, while passing a spread-index benchmark.
    """
    pool = _pool(shape)
    indices = [0, 1, 2, 3, 4, 5, 6, 7]  # one block, fully covered
    assert len({i // SUBLANES for i in indices}) == 1
    vals = _vals(len(indices), shape[1:])

    got = _scatter(pool, indices, vals)

    np.testing.assert_array_equal(got, _reference(pool, indices, vals))


@pytest.mark.parametrize("shape", [(4096, LANES * 8), (4096, SUBLANES, LANES)])
def test_partial_collision_leaves_block_neighbours_alone(shape):
    """Some slots of a block written, the rest of that block untouched."""
    pool = _pool(shape)
    before = np.asarray(pool).copy()
    indices = [1, 3, 6]  # block 0, but not slots 0/2/4/5/7
    vals = _vals(len(indices), shape[1:])

    got = _scatter(pool, indices, vals)

    np.testing.assert_array_equal(got, _reference(pool, indices, vals))
    for untouched in (0, 2, 4, 5, 7):
        np.testing.assert_array_equal(got[untouched], before[untouched])


@pytest.mark.parametrize("shape", [(4096, LANES * 8), (4096, SUBLANES, LANES)])
def test_duplicate_indices_resolve_last_write_wins(shape):
    """Matches numpy and PyTorch: the later update to a slot is the one kept.

    Not automatic. The kernel applies updates by masked select rather than in
    sequence, so this only holds because the masks are applied in ascending
    order of update index.
    """
    pool = _pool(shape)
    indices = [3, 9, 3]  # slot 3 written twice, from different blocks
    vals = _vals(len(indices), shape[1:])

    got = _scatter(pool, indices, vals)

    np.testing.assert_array_equal(got, _reference(pool, indices, vals))
    np.testing.assert_array_equal(got[3], np.asarray(vals)[2])


def test_single_update():
    """na == 1 -- the degenerate grid, and the decode-of-one case."""
    shape = (4096, SUBLANES, LANES)
    pool = _pool(shape)
    indices = [17]
    vals = _vals(1, shape[1:])

    got = _scatter(pool, indices, vals)

    np.testing.assert_array_equal(got, _reference(pool, indices, vals))


def test_rank2_num_slots_must_tile_evenly():
    """A partial trailing block would write past the end of the aliased
    output, so this raises rather than corrupting."""
    pool = _pool((4100, LANES * 8))
    vals = _vals(2, (LANES * 8,))

    with pytest.raises(ValueError, match="multiple of"):
        _scatter(pool, [0, 9], vals)


def test_dtype_mismatch_raises():
    """A cross-dtype view has to happen on a ref inside the kernel; taking it
    in XLA costs a pool-sized layout copy, so the caller is told instead of
    silently paying for it."""
    pool = _pool((4096, SUBLANES, LANES))
    vals = jnp.zeros((2, SUBLANES, LANES), dtype=jnp.float32)

    with pytest.raises(ValueError, match="dtype"):
        _scatter(pool, [0, 9], vals)


def test_rank1_is_rejected():
    with pytest.raises(ValueError, match="ndim"):
        _scatter(_pool((4096,)), [0], jnp.zeros((1,), dtype=jnp.bfloat16))
