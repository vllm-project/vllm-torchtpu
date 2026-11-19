# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_Q128_FALLBACK: the 3200-token bucket on paired Q128/KV512 tiles.

Off (default): unchanged tile choices, including the 3200-token bucket's
unpaired Q256/KV128 fallback. On: only token counts that are a multiple of 128
but not of 256, on a TP1 stage's 3200-token pages, move to (128, 512); every
other bucket and page shape is unchanged.
"""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles


@pytest.fixture
def v7x(monkeypatch):
    # The default-tile branch logs a warning through vLLM's logger; keep the test
    # independent of vLLM.
    monkeypatch.setattr(
        prefill_tiles, "logger", type("L", (), {"warning_once": staticmethod(lambda *a, **k: None)})()
    )
    if jax.default_backend() != "tpu":
        from jax._src import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
    yield


def _tiles(tokens, page, heads=24):
    query = jax.ShapeDtypeStruct((tokens, heads, 256), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((4, page, 1, 4, 256), jnp.float8_e4m3fn)
    return prefill_tiles.get_sparse_prefill_tiles(
        query, cache, kv_heads=2 if heads == 24 else 1, ratio=4, with_query_tile=True
    )


@pytest.mark.parametrize("flag", ["0", "1"])
def test_fallback_scope(v7x, monkeypatch, flag):
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_Q128_FALLBACK", flag)
    on = flag == "1"
    # The changed case: 3200 tokens on 3200-token pages.
    assert _tiles(3200, 3200) == ((128, 512, None) if on else (256, 128, None))
    # Unchanged: buckets that are multiples of 256 keep the T8192 entry.
    for tokens in (256, 1024, 2048, 4096, 6400, 8192):
        assert _tiles(tokens, 3200) == (256, 512, None)
    # Unchanged: other page shapes (the 1664-token unified pool) and tiny steps.
    assert _tiles(3200, 1664) == (256, 128, None)
    assert _tiles(128, 3200) == _tiles(128, 3200)  # default, flag-independent below
    assert _tiles(128, 3200)[1] == 128
    # Unchanged: TP shards (Q12 per rank) take their own table.
    assert _tiles(3200, 3200, heads=12) == (512, 512, 128)
