# SPDX-License-Identifier: Apache-2.0
"""Production dispatch and rank-uniform metadata capacity regression checks."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.megakernel.kimi_k3 import config
from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused
from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe.moe_down_valu import (
    down_valu_compact,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("window", [512, 1024, 2048, 4096, 6144, 8192, 16384])
@pytest.mark.parametrize("ep", [8, 16, 32])
@pytest.mark.parametrize("enabled", [False, True])
def test_factory_preserves_unsupported_shapes(monkeypatch, window, ep, enabled):
    cfg = config.KernelConfig(
        prefill_expert_parallel=ep,
        prefill_fused_moe=enabled,
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
    result = kernels.make_indexed_moe_sp_ring(None, seq_len=window)
    expected = "reference"
    if enabled and ep in (8, 16) and window in (512, 1024, 2048, 4096, 8192, 16384):
        expected = "fused"
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


@pytest.mark.parametrize("tokens", [512, 1024, 2048, 4096, 8192, 16384])
def test_fused_validity_matches_sequence_parallel_ownership(tokens):
    rows = tokens // 32
    stripe = min(rows, 64)
    for live in (0, 1, 17, tokens - 17, tokens):
        for rank in (0, 1, 15, 31):
            owned = np.arange(rank * stripe, tokens, 32 * stripe)[:, None]
            owned = (owned + np.arange(stripe)[None, :]).reshape(rows, 1)
            actual = fused._stripe_validity(rows, jnp.int32(live), jnp.int32(rank))
            np.testing.assert_array_equal(actual, owned < live)


@pytest.mark.parametrize("local_experts", [56, 112])
def test_metadata_steps_ep16_and_ep8(local_experts):
    import jax
    from jax.experimental import pallas as pl

    tokens = 512
    rows = tokens // 32
    rng = np.random.default_rng(1030 + local_experts)
    ids_all = rng.integers(0, 896, size=(tokens, 16), dtype=np.int32)
    probs_all = rng.uniform(0.01, 1.0, size=(tokens, 16)).astype(np.float32)
    probs_all[-5:, :] = 0.0
    wire_np = np.concatenate([ids_all, probs_all.view(np.int32)], axis=1).reshape(
        rows * 8, 128
    )
    wire = jnp.asarray(wire_np)
    expert_tp = 32 // (896 // local_experts)
    live_mask = probs_all != 0
    hist_896 = np.bincount(ids_all[live_mask], minlength=896)

    def kern(wire_ref, rank_ref, keys_ref, probs_ref, tids_ref, counts_ref):
        for _ in fused._metadata_steps(
            wire_ref,
            keys_ref,
            probs_ref,
            tids_ref,
            counts_ref,
            None,
            tokens=tokens,
            rank=rank_ref[0],
            counts_ready=lambda: None,
            local_experts=local_experts,
        ):
            pass

    out_shapes = (
        jax.ShapeDtypeStruct((rows * 4, 128), jnp.int32),
        jax.ShapeDtypeStruct((tokens // 8, 128), jnp.float32),
        jax.ShapeDtypeStruct((tokens // 8, 128), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
    )
    fn = jax.jit(pl.pallas_call(kern, out_shape=out_shapes, interpret=True))
    for rank in (0, 3, 4, 17, 31):
        _, probs_out, tids_out, counts_out = fn(
            wire, jnp.array([rank], dtype=jnp.int32)
        )
        first_exp = (rank // expert_tp) * local_experts
        expected_counts = hist_896[first_exp : first_exp + local_experts]
        np.testing.assert_array_equal(np.asarray(counts_out), expected_counts)
        total_local = int(expected_counts.sum())
        expected_routes = []
        for e in range(first_exp, first_exp + local_experts):
            tok_idx, slot_idx = np.where((ids_all == e) & live_mask)
            for idx in np.lexsort((slot_idx, tok_idx)):
                expected_routes.append(
                    (tok_idx[idx], probs_all[tok_idx[idx], slot_idx[idx]])
                )
        np.testing.assert_array_equal(
            np.asarray(tids_out).reshape(-1)[:total_local],
            [t for t, _ in expected_routes],
        )
        np.testing.assert_array_equal(
            np.asarray(probs_out).reshape(-1)[:total_local],
            np.array([p for _, p in expected_routes], dtype=np.float32),
        )


def test_metadata_steps_16k_ep8():
    import jax
    from jax.experimental import pallas as pl

    tokens = 16384
    local_experts = 112
    rows = tokens // 32
    rng = np.random.default_rng(16384)
    ids_all = rng.integers(0, 896, size=(tokens, 16), dtype=np.int32)
    probs_all = rng.uniform(0.01, 1.0, size=(tokens, 16)).astype(np.float32)
    probs_all[-37:, :] = 0.0
    wire_np = np.concatenate([ids_all, probs_all.view(np.int32)], axis=1).reshape(
        rows * 8, 128
    )
    wire = jnp.asarray(wire_np)
    expert_tp = 4
    live_mask = probs_all != 0
    hist_896 = np.bincount(ids_all[live_mask], minlength=896)

    def kern(wire_ref, rank_ref, keys_ref, probs_ref, tids_ref, counts_ref, max_ref):
        for _ in fused._metadata_steps(
            wire_ref,
            keys_ref,
            probs_ref,
            tids_ref,
            counts_ref,
            max_ref,
            tokens=tokens,
            rank=rank_ref[0],
            counts_ready=lambda: None,
            local_experts=local_experts,
        ):
            pass

    out_shapes = (
        jax.ShapeDtypeStruct((rows * 4, 128), jnp.int32),
        jax.ShapeDtypeStruct((tokens // 8, 128), jnp.float32),
        jax.ShapeDtypeStruct((tokens // 8, 128), jnp.int32),
        jax.ShapeDtypeStruct((local_experts,), jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    fn = jax.jit(pl.pallas_call(kern, out_shape=out_shapes, interpret=True))
    expected_max = max(int(hist_896[g * 112 : (g + 1) * 112].sum()) for g in range(8))
    for rank in (0, 5, 31):
        _, probs_out, tids_out, counts_out, max_out = fn(
            wire, jnp.array([rank], dtype=jnp.int32)
        )
        first_exp = (rank // expert_tp) * local_experts
        expected_counts = hist_896[first_exp : first_exp + local_experts]
        np.testing.assert_array_equal(np.asarray(counts_out), expected_counts)
        assert int(max_out[0]) == expected_max
        total_local = int(expected_counts.sum())
        expected_routes = []
        for e in range(first_exp, first_exp + local_experts):
            tok_idx, slot_idx = np.where((ids_all == e) & live_mask)
            for idx in np.lexsort((slot_idx, tok_idx)):
                expected_routes.append(
                    (tok_idx[idx], probs_all[tok_idx[idx], slot_idx[idx]])
                )
        np.testing.assert_array_equal(
            np.asarray(tids_out).reshape(-1)[:total_local],
            [t for t, _ in expected_routes],
        )
        np.testing.assert_array_equal(
            np.asarray(probs_out).reshape(-1)[:total_local],
            np.array([p for _, p in expected_routes], dtype=np.float32),
        )
