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
