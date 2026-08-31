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
import torch

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.partial_kv_update import (
    pallas_fill_hybrid_slots, pallas_fill_hybrid_slots_, pallas_fill_slots,
    pallas_fill_slots_, pallas_normal_fill_hybrid_slots,
    pallas_normal_fill_slots, pallas_normal_fill_slots_, pallas_scatter_slots)

SUBLANES = 8
LANES = 128


def _pool(shape, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape), dtype=jnp.bfloat16)


def _vals(n, trailing, seed=1):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((n, ) + trailing),
                       dtype=jnp.bfloat16)


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
    vals = _vals(2, (LANES * 8, ))

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
        _scatter(_pool((4096, )), [0], jnp.zeros((1, ), dtype=jnp.bfloat16))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 16, 1, 128),
        (16, 16, 1, 640),  # Kimi MLA attention page
        (9, 3, 12, 128),
        (9, 12, 128, 128),
    ],
)
def test_constant_fill_updates_only_selected_slots(shape, dtype):
    pool = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    pool = pool.astype(dtype)
    before = np.asarray(pool).copy()
    indices = jnp.asarray([1, 5, 7], dtype=jnp.int32)
    fill_value = jnp.asarray([0.015], dtype=jnp.float32)

    got = np.asarray(pallas_fill_slots(pool, indices, fill_value))

    expected_value = np.asarray(0.015, dtype=np.dtype(dtype)).item()
    np.testing.assert_array_equal(
        got[np.asarray(indices)],
        np.full(got[np.asarray(indices)].shape,
                expected_value,
                dtype=np.dtype(dtype)),
    )
    untouched = np.ones(shape[0], dtype=bool)
    untouched[np.asarray(indices)] = False
    np.testing.assert_array_equal(got[untouched], before[untouched])


def test_constant_fill_duplicate_slots_are_idempotent():
    pool = _pool((16, 8, 128))
    indices = jnp.asarray([3, 3, 9], dtype=jnp.int32)

    got = np.asarray(
        pallas_fill_slots(pool, indices, jnp.asarray([2.0],
                                                     dtype=jnp.float32)))

    expected = np.full(got[3].shape, 2.0, dtype=np.dtype("bfloat16"))
    np.testing.assert_array_equal(got[3], expected)
    np.testing.assert_array_equal(got[9], expected)


def test_constant_fill_supports_packed_fp8_attention_pages():
    shape = (9, 4, 1, 4, LANES)
    pool = jnp.full(shape, 0.5, dtype=jnp.float8_e4m3fn)
    selected = np.asarray([1, 5, 7], dtype=np.int32)

    got = np.asarray(
        pallas_fill_slots(pool, jnp.asarray(selected),
                          jnp.asarray([0.25],
                                      dtype=jnp.float32))).astype(np.float32)

    np.testing.assert_array_equal(
        got[selected], np.full(got[selected].shape, 0.25, dtype=np.float32))
    untouched = np.ones(shape[0], dtype=bool)
    untouched[selected] = False
    np.testing.assert_array_equal(
        got[untouched], np.full(got[untouched].shape, 0.5, dtype=np.float32))


def test_constant_fill_rejects_empty_indices():
    with pytest.raises(ValueError, match="non-empty"):
        pallas_fill_slots(_pool((16, 8, 128)), jnp.asarray([],
                                                           dtype=jnp.int32),
                          jnp.asarray([1.0], dtype=jnp.float32))


def test_constant_fill_rejects_rank2_pool():
    with pytest.raises(ValueError, match="ndim >= 3"):
        pallas_fill_slots(_pool((16, 128)), jnp.asarray([1], dtype=jnp.int32),
                          jnp.asarray([1.0], dtype=jnp.float32))


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((16, 16, 10, 1, 128), torch.bfloat16),
        ((16, 16, 1, 640), torch.bfloat16),
        ((9, 4, 1, 4, 128), torch.float8_e4m3fn),
        ((9, 3, 12, 128), torch.float32),
        ((9, 12, 128, 128), torch.bfloat16),
    ],
)
def test_constant_fill_torch_bridge_donates_and_writes_back(shape, dtype):
    source = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    cache = source.to(dtype).to("tpu")
    before = cache.cpu().float().numpy()
    selected = [2, shape[0] - 2]
    indices = torch.tensor(selected, dtype=torch.int32, device="tpu")
    fill_value = torch.tensor([0.25], dtype=torch.float32, device="tpu")

    returned = pallas_fill_slots_(cache, indices, fill_value)
    got = cache.cpu().float().numpy()

    assert returned is cache
    for slot in selected:
        np.testing.assert_array_equal(
            got[slot], np.full(shape[1:], 0.25, dtype=np.float32))
    untouched = np.ones(shape[0], dtype=bool)
    untouched[selected] = False
    np.testing.assert_array_equal(got[untouched], before[untouched])


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((16, 16, 1, 640), jnp.bfloat16),  # Kimi MLA attention page
        ((9, 3, 12, 128), jnp.float32),  # compact KDA conv state
        ((9, 12, 128, 128), jnp.float32),  # compact KDA recurrent state
    ],
)
def test_normal_fill_updates_only_selected_slots(shape, dtype):
    pool = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    pool = pool.astype(dtype)
    before = np.asarray(pool).copy()
    selected = np.asarray([1, 5, 7], dtype=np.int32)
    mean = 0.4
    std = 0.2

    got = np.asarray(
        pallas_normal_fill_slots(
            pool,
            jnp.asarray(selected),
            jnp.asarray([mean, std], dtype=jnp.float32),
            jnp.asarray([1234], dtype=jnp.int32),
        ))

    samples = got[selected].astype(np.float32).reshape(-1)
    assert samples.mean() == pytest.approx(mean, abs=0.015)
    assert samples.std() == pytest.approx(std, abs=0.015)
    untouched = np.ones(shape[0], dtype=bool)
    untouched[selected] = False
    np.testing.assert_array_equal(got[untouched], before[untouched])


def test_normal_fill_is_seeded_and_slot_stable():
    pool = jnp.zeros((16, 8, 128), dtype=jnp.bfloat16)
    indices = jnp.asarray([9, 3], dtype=jnp.int32)
    params = jnp.asarray([0.0, 1.0], dtype=jnp.float32)

    first = np.asarray(
        pallas_normal_fill_slots(pool, indices, params,
                                 jnp.asarray([7], dtype=jnp.int32)))
    reordered = np.asarray(
        pallas_normal_fill_slots(pool, indices[::-1], params,
                                 jnp.asarray([7], dtype=jnp.int32)))
    different_seed = np.asarray(
        pallas_normal_fill_slots(pool, indices, params,
                                 jnp.asarray([8], dtype=jnp.int32)))

    np.testing.assert_array_equal(first[3], reordered[3])
    np.testing.assert_array_equal(first[9], reordered[9])
    assert not np.array_equal(first[3], different_seed[3])


def test_normal_fill_validates_parameters():
    pool = _pool((16, 8, 128))
    indices = jnp.asarray([1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="mean_std"):
        pallas_normal_fill_slots(pool, indices,
                                 jnp.asarray([0.0], dtype=jnp.float32),
                                 jnp.asarray([1], dtype=jnp.int32))
    with pytest.raises(ValueError, match="seed"):
        pallas_normal_fill_slots(pool, indices,
                                 jnp.asarray([0.0, 1.0], dtype=jnp.float32),
                                 jnp.asarray([1.0], dtype=jnp.float32))


def test_unified_hybrid_fill_preserves_null_padding_and_unrelated_bytes():
    shape = (8, 4, 2, LANES)
    pool = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    pool = pool.astype(jnp.bfloat16)
    before = np.asarray(pool).copy()

    manager_indices = jnp.asarray([1, 2], dtype=jnp.int32)
    updated = pallas_fill_hybrid_slots(
        pool,
        manager_indices,
        # manager 1: attention + state; manager 2: state only
        jnp.asarray([3, 2], dtype=jnp.int32),
        jnp.asarray([0.25], dtype=jnp.float32),
        split=2,
        ssm_bytes=512,
        ssm_itemsize=4,
        conv_bytes=256,
    )
    got = np.asarray(updated)
    ssm = np.asarray(
        pool_adapters.gather_region(updated,
                                    manager_indices,
                                    tok0=0,
                                    ntok=1,
                                    out_dtype=jnp.float32,
                                    split=2))

    # Manager block zero is the null page and was not selected.
    np.testing.assert_array_equal(got[:2], before[:2])
    # State-only manager 2 preserves everything outside its two live regions.
    np.testing.assert_array_equal(ssm[1],
                                  np.full((1, LANES), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(
        got[4, 1, 0], np.full((LANES, ), 0.25, dtype=np.dtype("bfloat16")))
    np.testing.assert_array_equal(got[4, 1, 1], before[4, 1, 1])
    np.testing.assert_array_equal(got[4, 2:], before[4, 2:])
    np.testing.assert_array_equal(got[5], before[5])
    # The overlapping manager has typed state over a raw attention fill.
    np.testing.assert_array_equal(ssm[0],
                                  np.full((1, LANES), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(
        got[2, 1, 0], np.full((LANES, ), 0.25, dtype=np.dtype("bfloat16")))
    np.testing.assert_array_equal(
        got[2, 1, 1], np.full((LANES, ), 0.25, dtype=np.dtype("bfloat16")))
    np.testing.assert_array_equal(
        got[3], np.full(shape[1:], 0.25, dtype=np.dtype("bfloat16")))


def test_unified_hybrid_fill_supports_attention_only_pages():
    pool = jnp.zeros((8, 4, 2, LANES), dtype=jnp.bfloat16)
    updated = pallas_fill_hybrid_slots(
        pool,
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([1], dtype=jnp.int32),
        jnp.asarray([0.25], dtype=jnp.float32),
        split=2,
        ssm_bytes=0,
        ssm_itemsize=0,
        conv_bytes=0,
    )
    got = np.asarray(updated)

    np.testing.assert_array_equal(got[:2], np.zeros_like(got[:2]))
    np.testing.assert_array_equal(got[2:4], np.full_like(got[2:4], 0.25))
    np.testing.assert_array_equal(got[4:], np.zeros_like(got[4:]))


def test_unified_hybrid_fill_rejects_unknown_ssm_itemsize():
    with pytest.raises(ValueError, match="BF16 or FP32"):
        pallas_fill_hybrid_slots(
            jnp.zeros((8, 4, 2, LANES), dtype=jnp.bfloat16),
            jnp.asarray([1], dtype=jnp.int32),
            jnp.asarray([3], dtype=jnp.int32),
            jnp.asarray([0.25], dtype=jnp.float32),
            split=2,
            ssm_bytes=512,
            ssm_itemsize=1,
            conv_bytes=256,
        )


@pytest.mark.parametrize(("ssm_dtype", "ssm_itemsize"), [
    (jnp.bfloat16, 2),
    (jnp.float32, 4),
])
def test_unified_hybrid_normal_fill_uses_typed_state_distributions(
        ssm_dtype, ssm_itemsize):
    pool = jnp.zeros((8, 4, 2, LANES), dtype=jnp.bfloat16)
    manager_indices = jnp.asarray([1], dtype=jnp.int32)
    updated = pallas_normal_fill_hybrid_slots(
        pool,
        manager_indices,
        jnp.asarray([3], dtype=jnp.int32),
        jnp.asarray([0.4, 0.2], dtype=jnp.float32),
        jnp.asarray([1234], dtype=jnp.int32),
        split=2,
        ssm_bytes=LANES * ssm_itemsize,
        ssm_itemsize=ssm_itemsize,
        conv_bytes=256,
    )
    got = np.asarray(updated)

    ssm = np.asarray(
        pool_adapters.gather_region(updated,
                                    manager_indices,
                                    tok0=0,
                                    ntok=1,
                                    out_dtype=ssm_dtype,
                                    split=2))[:,
                                              0].astype(np.float32).reshape(-1)
    conv = got[2, 1, 0].astype(np.float32).reshape(-1)
    assert ssm.mean() == pytest.approx(0.4, abs=0.05)
    assert ssm.std() == pytest.approx(0.2, abs=0.05)
    assert conv.mean() == pytest.approx(0.4, abs=0.05)
    assert conv.std() == pytest.approx(0.2, abs=0.05)


@pytest.mark.parametrize(("ssm_dtype", "ssm_itemsize"), [
    (jnp.bfloat16, 2),
    (jnp.float32, 4),
])
def test_unified_hybrid_fp8_pool_uses_typed_state_abi(ssm_dtype, ssm_itemsize):
    # Real FP8 attention caches keep their four-byte pack as its own payload
    # dimension: (blocks, tokens, packed KV heads, pack, lanes).
    shape = (8, 4, 1, 4, LANES)
    pool = jnp.full(shape, 0.5, dtype=jnp.float8_e4m3fn)
    manager_indices = jnp.asarray([1], dtype=jnp.int32)
    updated = pallas_fill_hybrid_slots(
        pool,
        manager_indices,
        jnp.asarray([3], dtype=jnp.int32),
        jnp.asarray([0.25], dtype=jnp.float32),
        split=2,
        ssm_bytes=LANES * ssm_itemsize,
        ssm_itemsize=ssm_itemsize,
        conv_bytes=256,
    )

    got = np.asarray(updated).astype(np.float32)
    ssm = np.asarray(
        pool_adapters.gather_region(updated,
                                    manager_indices,
                                    tok0=0,
                                    ntok=1,
                                    out_dtype=ssm_dtype,
                                    split=2))
    conv = np.asarray(
        pool_adapters.gather_region(updated,
                                    manager_indices,
                                    tok0=1,
                                    ntok=1,
                                    out_dtype=jnp.bfloat16,
                                    split=2)).astype(np.float32)

    np.testing.assert_array_equal(got[:2], np.full_like(got[:2], 0.5))
    np.testing.assert_array_equal(ssm[0, 0].astype(np.float32),
                                  np.full((LANES, ), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(conv[0, 0],
                                  np.full((LANES, ), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(got[3],
                                  np.full(shape[1:], 0.25, dtype=np.float32))


def test_unified_hybrid_torch_bridge_donates_and_writes_back():
    shape = (8, 4, 2, LANES)
    cache = torch.zeros(shape, dtype=torch.bfloat16, device="tpu")
    indices = torch.tensor([1], dtype=torch.int32, device="tpu")
    fill_kinds = torch.tensor([3], dtype=torch.int32, device="tpu")
    fill_value = torch.tensor([0.25], dtype=torch.float32, device="tpu")

    returned = pallas_fill_hybrid_slots_(cache,
                                         indices,
                                         fill_kinds,
                                         fill_value,
                                         split=2,
                                         ssm_bytes=512,
                                         ssm_itemsize=4,
                                         conv_bytes=256)
    got = cache.cpu().float().numpy()

    assert returned is cache
    np.testing.assert_array_equal(got[:2], np.zeros_like(got[:2]))
    np.testing.assert_array_equal(got[2, 0, 0],
                                  np.zeros((LANES, ), dtype=np.float32))
    np.testing.assert_array_equal(got[2, 0, 1],
                                  np.full((LANES, ), 0.25, dtype=np.float32))
    np.testing.assert_array_equal(
        got[2, 1:], np.full(got[2, 1:].shape, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(got[3],
                                  np.full(shape[1:], 0.25, dtype=np.float32))


@pytest.mark.parametrize(("ssm_bytes", "ssm_itemsize"), [
    (256, 2),
    (512, 4),
])
def test_unified_hybrid_fp8_torch_bridge_donates_and_writes_back(
        ssm_bytes, ssm_itemsize):
    shape = (8, 4, 1, 4, LANES)
    cache = torch.full(shape, 0.5, dtype=torch.float8_e4m3fn, device="tpu")
    indices = torch.tensor([1], dtype=torch.int32, device="tpu")
    fill_kinds = torch.tensor([3], dtype=torch.int32, device="tpu")
    fill_value = torch.tensor([0.25], dtype=torch.float32, device="tpu")

    returned = pallas_fill_hybrid_slots_(cache,
                                         indices,
                                         fill_kinds,
                                         fill_value,
                                         split=2,
                                         ssm_bytes=ssm_bytes,
                                         ssm_itemsize=ssm_itemsize,
                                         conv_bytes=256)
    got = cache.cpu().float().numpy()

    assert returned is cache
    np.testing.assert_array_equal(
        got[:2], np.full(got[:2].shape, 0.5, dtype=np.float32))
    np.testing.assert_array_equal(got[3],
                                  np.full(shape[1:], 0.25, dtype=np.float32))


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((16, 16, 1, 640), torch.bfloat16),
        ((9, 3, 12, 128), torch.float32),
        ((9, 12, 128, 128), torch.float32),
    ],
)
def test_normal_fill_torch_bridge_donates_and_writes_back(shape, dtype):
    cache = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    cache = cache.to(dtype).to("tpu")
    before = cache.cpu().float().numpy()
    selected = [2, shape[0] - 2]
    indices = torch.tensor(selected, dtype=torch.int32, device="tpu")
    params = torch.tensor([0.4, 0.2], dtype=torch.float32, device="tpu")
    seed = torch.tensor([1234], dtype=torch.int32, device="tpu")

    returned = pallas_normal_fill_slots_(cache, indices, params, seed)
    got = cache.cpu().float().numpy()

    assert returned is cache
    samples = got[selected].reshape(-1)
    assert samples.mean() == pytest.approx(0.4, abs=0.015)
    assert samples.std() == pytest.approx(0.2, abs=0.015)
    untouched = np.ones(shape[0], dtype=bool)
    untouched[selected] = False
    np.testing.assert_array_equal(got[untouched], before[untouched])
