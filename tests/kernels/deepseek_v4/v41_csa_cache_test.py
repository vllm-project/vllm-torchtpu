"""V4.1 FP8 CSA cache encoding and physical-address contracts."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41 import attention_kernels, cache, csa

pytestmark = pytest.mark.cpu_test


def values(count=17):
    rng = np.random.default_rng(121)
    x = rng.normal(size=(count, 512)).astype(np.float32)
    x[0] = 0
    if count > 1:
        x[1] *= 0.000001
    if count > 2:
        x[2] *= 1000
    return torch.from_numpy(x).to(torch.bfloat16)


def reference_pack(v):
    # Independent NumPy bit packing with exact power-of-two scaling. The native
    # JAX reference uses an approximate CPU exp2; actual Pallas parity is a TPU check.
    import ml_dtypes

    x = v.float().numpy()
    groups = x[:, :448].reshape(-1, 7, 64)
    scales = np.exp2(np.ceil(np.log2(np.maximum(np.abs(groups).max(1), 1e-4) / 448.0)))
    q = (
        (groups / scales[:, None])
        .astype(ml_dtypes.float8_e4m3fn)
        .view(np.uint8)
        .reshape(-1, 448)
    )
    scale_bytes = (scales.view(np.uint32) >> 23).astype(np.uint8)
    slabs = (
        np.concatenate((q, scale_bytes), axis=1).reshape(-1, 4, 128).astype(np.uint32)
    )
    nope = (
        slabs[:, 0] | (slabs[:, 1] << 8) | (slabs[:, 2] << 16) | (slabs[:, 3] << 24)
    ).view(np.int32)
    bits = x[:, 448:].astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.uint32)
    rope = (bits[:, :32] | (bits[:, 32:] << 16)).view(np.int32)
    return nope, rope


@pytest.mark.parametrize("page_tokens", [4, 8, 16, 32])
def test_packed_cache_bit_exact_and_isolated(page_tokens):
    pages = 5
    stride = page_tokens * 640 + 1024
    offset = 1536
    raw = torch.full(
        ((offset + pages * stride + 1024) // 512, 128), 0x5A5A5A5A, dtype=torch.int32
    )
    owner = csa.PackedCSACache(
        raw, ((pages, page_tokens, 640), torch.uint8, stride, offset)
    )
    slots = torch.tensor(
        [0, 1, 2, 3, page_tokens - 1, page_tokens + 2, 4 * page_tokens + 1, -1, -1],
        dtype=torch.int32,
    )
    # Avoid duplicate valid writes for the smallest page.
    slots[4] = 2 * page_tokens
    v = values(len(slots))
    old = raw.clone()
    owner.write(slots, v)
    expected_nope, expected_rope = reference_pack(v[:-2])
    np.testing.assert_array_equal(
        raw[owner.row_indices(slots[:-2]).long()].numpy(), expected_nope
    )
    np.testing.assert_array_equal(
        raw.reshape(-1, 32)[owner.rope_indices(slots[:-2]).long()].numpy(),
        expected_rope,
    )
    changed = torch.zeros_like(raw, dtype=torch.bool)
    changed[owner.row_indices(slots[:-2]).long()] = True
    changed.reshape(-1, 32)[owner.rope_indices(slots[:-2]).long()] = True
    torch.testing.assert_close(raw[~changed], old[~changed], rtol=0, atol=0)
    gpu_words = jax.jit(csa.write_csa_values)(
        jnp.asarray(old.numpy()),
        jnp.asarray(
            owner.row_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1).numpy()
        ),
        jnp.asarray(
            owner.rope_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1).numpy()
        ),
        jnp.asarray(v.float().numpy(), jnp.bfloat16),
    )
    np.testing.assert_array_equal(gpu_words, raw.numpy())
    decoded = owner[slots[:-2] // page_tokens, slots[:-2] % page_tokens]
    torch.testing.assert_close(decoded[:, 448:], v[:-2, 448:], rtol=0, atol=0)
    assert decoded.isfinite().all()


def test_holey_reindex_compacts_without_changing_selected_pairs():
    nope = jnp.array(
        [[40, -1, 91, -1, 3], [8, 8, -1, 7, -1], [-1, -1, -1, -1, -1]], jnp.int32
    )
    rope = jnp.array(
        [[201, -1, 311, -1, 19], [50, 50, -1, 11, -1], [-1, -1, -1, -1, -1]], jnp.int32
    )
    n, r = jax.jit(csa.compact_csa_records)(nope, rope)
    assert n.shape == r.shape == (3, 128)
    np.testing.assert_array_equal(n[0, :3], [40, 91, 3])
    np.testing.assert_array_equal(r[0, :3], [201, 311, 19])
    np.testing.assert_array_equal(n[1, :3], [8, 8, 7])
    np.testing.assert_array_equal(r[1, :3], [50, 50, 11])
    assert np.all(np.asarray(n[:2, 3:]) == -1) and np.all(np.asarray(n[2]) == -1)


def test_large_pool_uses_record_indices_without_byte_overflow():
    raw = torch.empty((20 * 2**30 // 512, 128), device="meta", dtype=torch.int32)
    stride = 16 * 640 + 1024
    offset = 9 * 2**30
    owner = csa.PackedCSACache(raw, ((1024, 16, 640), torch.uint8, stride, offset))
    slots = torch.tensor([0, 1, 16, 16383], dtype=torch.int32)
    for slot, n, r in zip(
        slots.tolist(),
        owner.row_indices(slots).tolist(),
        owner.rope_indices(slots).tolist(),
    ):
        assert n == (offset + slot // 16 * stride) // 512 + slot % 16
        assert r == (offset + slot // 16 * stride + 16 * 512) // 128 + slot % 16
    assert (
        owner.row_indices(slots).dtype == owner.rope_indices(slots).dtype == torch.int32
    )


@pytest.mark.parametrize(
    "offset,stride,page_tokens", [(128, 5120, 8), (0, 5136, 8), (0, 4096, 3)]
)
def test_reject_incompatible_page_geometry(offset, stride, page_tokens):
    with pytest.raises(ValueError, match="alignment"):
        csa.PackedCSACache(
            torch.empty((64, 128), dtype=torch.int32),
            ((2, page_tokens, 640), torch.uint8, stride, offset),
        )


def test_csa_wiring_keeps_swa_state_and_shared_record_pairs():
    rng = np.random.default_rng(672)
    rows = jnp.asarray(rng.normal(size=(128, 512)), jnp.bfloat16)
    pool = cache.scatter_cache_values(
        jnp.zeros((256, 128), jnp.int32), jnp.arange(128, dtype=jnp.int32), rows
    )
    q = jnp.asarray(rng.normal(size=(3, 8, 512)), jnp.bfloat16)
    swa = jnp.tile(jnp.arange(64, dtype=jnp.int32), (3, 1))
    nope = jnp.array([[1, -1, 8], [7, 8, -1], [-1, -1, -1]], jnp.int32)
    rope = jnp.array([[200, -1, 206], [311, 312, -1], [-1, -1, -1]], jnp.int32)
    sink = jnp.zeros((8,), jnp.float32)
    maximum, denominator, acc = attention_kernels.paged_attention_state_jax(
        q, pool, swa, 0.125
    )
    captured = []

    def native(*args, **kwargs):
        captured.append((args, kwargs))
        return args[0]

    with patch.object(csa.sparse_mla, "sparse_ragged_paged_attention", native):
        result = csa.csa_attention_jax(q, pool, pool, swa, nope, rope, sink, 0.125)
    args, kw = captured[0]
    assert kw["sm_scale"] == 0.125
    np.testing.assert_array_equal(args[8], acc.astype(q.dtype))
    np.testing.assert_array_equal(args[9], denominator)
    np.testing.assert_array_equal(args[10], maximum)
    n, r = kw["physical_indices"]
    assert n.shape == r.shape == (3, 128)
    np.testing.assert_array_equal(n[0, :2], [1, 8])
    np.testing.assert_array_equal(r[0, :2], [200, 206])
    np.testing.assert_array_equal(result, q)
