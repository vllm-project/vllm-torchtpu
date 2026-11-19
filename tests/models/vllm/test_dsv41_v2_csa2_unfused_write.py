# SPDX-License-Identifier: Apache-2.0
"""Unfused (TP > 1) writers publish the same CSA2 records as the fused compressor."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import encode_csa2_kv
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    SPARSECORE_MAX_POOL_BANKS,
    SPARSECORE_MAX_POOL_BYTES,
    PagedCacheRows,
    reserve_scale_companion,
)

pytestmark = pytest.mark.cpu_test


def _cache(banks):
    rows = 64
    raw = tuple(
        torch.zeros((rows // banks, 128), dtype=torch.int32) for _ in range(banks)
    )
    scale = tuple(torch.zeros((rows // banks,), dtype=torch.float32) for _ in raw)
    # 8 pages of 8 records, 512 bytes each.
    layout = ((8, 8, 512), torch.uint8, 8 * 512, 0)
    if banks == 1:
        return PagedCacheRows(raw[0], layout, scale=scale[0])
    return PagedCacheRows(raw, layout, scale=scale)


@pytest.mark.parametrize("banks", [1, 2])
def test_unfused_write_matches_csa2_encoding(banks):
    cache = _cache(banks)
    gen = torch.Generator().manual_seed(0)
    values = torch.randn((6, 512), generator=gen).to(torch.bfloat16)
    # Slots straddle both banks; -1 drops its write.
    slots = torch.tensor([0, 9, -1, 33, 40, 63])
    cache.write(slots, values)
    words, scales = (
        torch.from_numpy(np.array(x))
        for x in encode_csa2_kv(jnp.asarray(values.float().numpy(), jnp.bfloat16))
    )
    raw = torch.cat((cache.raw,) if cache.raw_hi is None else (cache.raw, cache.raw_hi))
    scale = torch.cat(
        (cache.scale,) if cache.scale_hi is None else (cache.scale, cache.scale_hi)
    )
    valid = slots >= 0
    rows = slots[valid]
    assert torch.equal(raw[rows], words[valid])
    assert torch.equal(scale[rows], scales[valid])
    untouched = torch.ones(64, dtype=torch.bool)
    untouched[rows] = False
    assert not raw[untouched].any() and not scale[untouched].any()


def test_fp8_main_records_require_scale_companion():
    layout = ((8, 8, 512), torch.uint8, 8 * 512, 0)
    with pytest.raises(ValueError, match="CSA2 scale"):
        PagedCacheRows(torch.zeros((64, 128), dtype=torch.int32), layout)


def test_scale_companion_budget():
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    # One FP32 scale per 512-byte row: the pool keeps 128/129 of the budget.
    assert reserve_scale_companion(cfg, 129 << 20) == 128 << 20
    # 36 GiB (the DP8 measurement) now reaches the 2-bank cap; with the old
    # 1/5 RoPE reservation it stopped at 28.8 GiB.
    assert (
        reserve_scale_companion(cfg, 36 << 30, auto=True)
        == SPARSECORE_MAX_POOL_BANKS * SPARSECORE_MAX_POOL_BYTES
    )
    cfg.cache_config.cache_dtype = "auto"
    assert reserve_scale_companion(cfg, 1000) == 1000
