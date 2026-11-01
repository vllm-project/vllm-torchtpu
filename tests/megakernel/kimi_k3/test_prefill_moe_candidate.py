# SPDX-License-Identifier: Apache-2.0
"""Production dispatch and rank-uniform metadata capacity regression checks."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3 import config
from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused, layer
from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe.moe_down_valu import (
    down_valu_compact,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("window", [512, 1024, 2048, 4096, 6144, 8192])
@pytest.mark.parametrize("ep", [8, 16, 32])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("reduce_dtype", ["float32", "bfloat16"])
def test_factory_preserves_unsupported_shapes(
    monkeypatch, window, ep, enabled, reduce_dtype
):
    cfg = config.KernelConfig(
        prefill_expert_parallel=ep,
        prefill_fused_moe=enabled,
        moe_reduce_dtype=reduce_dtype,
    )
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(
        kernels,
        "_make_indexed_moe_sp_ring_reference",
        lambda mesh, *, seq_len: ("reference", seq_len),
    )
    monkeypatch.setattr(
        fused,
        "make_indexed_moe_sp_ring",
        lambda mesh, *, seq_len: ("fused", seq_len),
    )
    monkeypatch.setattr(
        layer,
        "make_indexed_moe_sp_ring",
        lambda mesh, *, seq_len: ("bf16", seq_len),
    )
    result = kernels.make_indexed_moe_sp_ring(None, seq_len=window)
    expected = "reference"
    if enabled and ep == 16:
        if reduce_dtype == "float32" and window in (512, 1024, 2048, 4096, 8192):
            expected = "fused"
        elif reduce_dtype == "bfloat16" and window in (4096, 8192):
            expected = "bf16"
    assert result == (expected, window)


def test_rollback_has_distinct_compile_cache_key():
    cfg = config.KernelConfig()
    assert cfg.prefill_fused_moe
    assert cfg.key() != cfg.replace(prefill_fused_moe=False).key()


@pytest.mark.parametrize("peak", [0, 32768, 32769, 131072])
def test_metadata_overflow_is_uniform_and_uncapped(monkeypatch, peak):
    from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import moe_down_valu

    # No TPU work: exercise the real branch wrapper with a stand-in consumer.
    # The stand-in reports capacity, activation rows and streaming mode.
    def consume(x, rhs, scales, counts, destinations, probabilities, tables, **kw):
        return jnp.array(
            [
                x.shape[0],
                destinations.shape[0],
                probabilities.shape[0],
                kw["stream_metadata"],
                kw.get("metadata_row_capacity", 0),
            ]
        )

    monkeypatch.setattr(moe_down_valu, "down_valu", consume)
    x = jnp.zeros((131072, 1), dtype=jnp.bfloat16)
    routes = jnp.arange(131072, dtype=jnp.int32)
    # Local counts differ across ranks, but the identical global peak determines
    # which collective-containing branch all ranks execute.
    results = []
    for local_count in [0, 17, peak]:
        results.append(
            down_valu_compact(
                x,
                None,
                None,
                jnp.array([local_count]),
                routes,
                routes,
                None,
                metadata_prefix_rows=32768,
                slice_input=False,
                global_route_max=jnp.int32(peak),
            )
        )
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    compact = peak <= 32768
    np.testing.assert_array_equal(
        results[0],
        [
            131072,
            32768 if compact else 131072,
            32768 if compact else 131072,
            not compact,
            32768 if compact else 0,
        ],
    )


def test_direct_fused_factory_rejects_bf16(monkeypatch):
    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    monkeypatch.setattr(kernels, "_config", lambda: cfg)
    with pytest.raises(ValueError, match="FP32"):
        fused.make_indexed_moe_sp_ring(None, seq_len=4096)


@pytest.mark.parametrize("tokens", [512, 1024, 2048, 4096, 8192])
def test_fused_validity_matches_sequence_parallel_ownership(tokens):
    rows = tokens // 32
    stripe = min(rows, 64)
    for live in (0, 1, 17, tokens - 17, tokens):
        for rank in (0, 1, 15, 31):
            owned = np.arange(rank * stripe, tokens, 32 * stripe)[:, None]
            owned = (owned + np.arange(stripe)[None, :]).reshape(rows, 1)
            actual = fused._stripe_validity(rows, jnp.int32(live), jnp.int32(rank))
            np.testing.assert_array_equal(actual, owned < live)


@pytest.mark.parametrize("window", [512, 1024, 8192])
@pytest.mark.parametrize("ep", [8, 16])
@pytest.mark.parametrize("fused_ep8", [False, True])
def test_factory_fuses_ep8_only_when_enabled(monkeypatch, window, ep, fused_ep8):
    cfg = config.KernelConfig(
        prefill_expert_parallel=ep, prefill_fused_moe_ep8=fused_ep8
    )
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(
        kernels,
        "_make_indexed_moe_sp_ring_reference",
        lambda mesh, *, seq_len: ("reference", seq_len),
    )
    monkeypatch.setattr(
        fused, "make_indexed_moe_sp_ring", lambda mesh, *, seq_len: ("fused", seq_len)
    )
    expected = "fused" if ep == 16 or fused_ep8 else "reference"
    assert kernels.make_indexed_moe_sp_ring(None, seq_len=window) == (expected, window)
    bf16 = cfg.replace(moe_reduce_dtype="bfloat16")
    monkeypatch.setattr(config, "active", lambda: bf16)
    monkeypatch.setattr(
        layer, "make_indexed_moe_sp_ring", lambda mesh, *, seq_len: ("bf16", seq_len)
    )
    result = kernels.make_indexed_moe_sp_ring(None, seq_len=window)
    if ep == 8:
        assert result == ("reference", window)


def test_expert_layouts():
    assert fused.experts_for(16) == fused.EP16
    ep16, ep8 = fused.EP16, fused.experts_for(8)
    assert (ep16.local, ep16.groups, ep16.ranks, ep16.middle) == (56, 16, 2, 1536)
    assert (ep8.local, ep8.groups, ep8.ranks, ep8.middle) == (112, 8, 4, 768)
    with pytest.raises(ValueError):
        fused.experts_for(32)


def _wire(ids, probs, tokens):
    """The all-gathered route wire as _metadata_steps reads it."""
    if tokens < 4096:
        records = np.concatenate((ids, probs.view(np.int32)), axis=1)  # [T, 32]
        return records.reshape(tokens // 4, 128)
    per_rank = tokens // 32
    planes = []
    for r in range(32):
        rows = slice(r * per_rank, (r + 1) * per_rank)
        planes.append(ids[rows].reshape(-1, 128))
        planes.append(probs[rows].view(np.int32).reshape(-1, 128))
    return np.concatenate(planes, axis=0)


@pytest.mark.parametrize("tokens", [512])
@pytest.mark.parametrize("ep, rank", [(16, 0), (16, 7), (8, 0), (8, 13), (8, 31)])
def test_metadata_matches_reference(tokens, ep, rank):
    import jax
    from jax.experimental import pallas as pl

    experts = fused.experts_for(ep)
    rng = np.random.default_rng(tokens + 31 * ep + rank)
    ids = np.stack([rng.choice(896, 16, replace=False) for _ in range(tokens)])
    ids = ids.astype(np.int32)
    probs = rng.random((tokens, 16)).astype(np.float32) + 0.01
    probs[rng.random((tokens, 16)) < 0.05] = 0.0  # padding rows' routes
    wire = _wire(ids, probs, tokens)

    def kernel(wire_ref, keys, probabilities, token_ids, counts):
        steps = fused._metadata_steps(
            wire_ref,
            keys,
            probabilities,
            token_ids,
            counts,
            None,
            tokens=tokens,
            rank=rank,
            counts_ready=lambda: None,
            experts=experts,
        )
        for _ in steps:
            pass

    shape = (tokens // 8, 128)
    out = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(shape, jnp.int32),
            jax.ShapeDtypeStruct(shape, jnp.float32),
            jax.ShapeDtypeStruct(shape, jnp.int32),
            jax.ShapeDtypeStruct((experts.local,), jnp.int32),
        ),
        interpret=True,
    )(jnp.asarray(wire))
    _, probabilities, token_ids, counts = (np.asarray(o) for o in out)
    local = ids - rank // experts.ranks * experts.local
    valid = (probs != 0) & (local >= 0) & (local < experts.local)
    np.testing.assert_array_equal(
        counts, np.bincount(local[valid], minlength=experts.local)
    )
    # Expert-major, then token-major order of the live routes.
    order = sorted(
        (int(local[t, j]), t * 16 + j)
        for t in range(tokens)
        for j in range(16)
        if valid[t, j]
    )
    live = len(order)
    np.testing.assert_array_equal(
        token_ids.reshape(-1)[:live], [position // 16 for _, position in order]
    )
    np.testing.assert_array_equal(
        probabilities.reshape(-1)[:live],
        [probs.reshape(-1)[position] for _, position in order],
    )
