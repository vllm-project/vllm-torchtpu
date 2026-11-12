# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the TurboQuant attention backend, config validation, and kernel entry."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.model_executor.layers.quantization.turboquant.config import TQ_PRESETS
from vllm.v1.kv_cache_interface import FullAttentionSpec, get_kv_quant_mode

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.layers.adapter import turboquant_attention
from vllm_torchtpu.layers.adapter.turboquant_attention import (
    TURBOQUANT_KV_CACHE_DTYPES,
    PallasTurboQuantAttentionBackend,
    _tq_rpa_kernel_entry,
    check_turboquant_config,
    get_tq_layout,
    get_tq_tables,
    set_active_turboquant_preset,
    validate_turboquant_head_dim,
)
from vllm_torchtpu.layers.core import turboquant as tq
from vllm_torchtpu.platforms.tpu_block_size_utils import (
    TURBOQUANT_KV_CACHE_DTYPES as PLATFORM_TQ_DTYPES,
)
from vllm_torchtpu.platforms.tpu_block_size_utils import unified_kv_layout_enabled

PAGE_SIZE = 16


@pytest.fixture(autouse=True)
def _reset_active_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(turboquant_attention, "_active_preset", None)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("preset", "tp", "head_dim"),
    [("turboquant_k8v4", 1, 128), ("turboquant_4bit_nc", 2, 256)],
)
def test_kernel_entry_matches_reference(preset: str, tp: int, head_dim: int):
    if jax.devices()[0].platform != "tpu" or len(jax.devices()) < tp:
        pytest.skip(f"needs {tp} TPU devices")
    rng = np.random.default_rng(0)
    num_kv_heads, num_q_heads, num_pages = 2 * tp, 4 * tp, 16
    layout, tables = get_tq_layout(preset, head_dim), get_tq_tables(preset, head_dim)
    sm_scale = head_dim**-0.5

    kv_lens = jnp.asarray([20, 9], jnp.int32)
    cu = jnp.asarray([0, 20, 29], jnp.int32)
    distribution = jnp.asarray([0, 0, 2], jnp.int32)
    page_indices = jnp.asarray(rng.permutation(num_pages)[:8].astype(np.int32))
    q = jnp.asarray(rng.standard_normal((32, num_q_heads, head_dim)), jnp.bfloat16)
    k = jnp.asarray(rng.standard_normal((32, num_kv_heads, head_dim)), jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal((32, num_kv_heads, head_dim)), jnp.bfloat16)
    cache = jnp.zeros(
        (num_pages, num_kv_heads, layout.page_words_128(PAGE_SIZE), 128), jnp.uint8
    )
    mesh = Mesh(np.asarray(jax.devices()[:tp]), ("model",))

    entry = jax.jit(
        lambda *a: _tq_rpa_kernel_entry(
            *a,
            None,
            None,
            None,
            None,
            mesh=mesh,
            sliding_window=None,
            sm_scale=sm_scale,
            skip_kv_update=False,
            tq_preset=preset,
        )
    )
    new_cache, out = entry(cache, q, k, v, kv_lens, page_indices, cu, distribution)
    ref_cache, ref_out = tq.ref_tq_ragged_paged_attention(
        q,
        k,
        v,
        cache,
        kv_lens,
        page_indices,
        cu,
        distribution,
        layout=layout,
        tables=tables,
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=sm_scale,
    )
    np.testing.assert_array_equal(np.asarray(new_cache), np.asarray(ref_cache))
    np.testing.assert_allclose(
        np.asarray(out[:29], np.float32),
        np.asarray(ref_out[:29], np.float32),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("preset", TURBOQUANT_KV_CACHE_DTYPES)
def test_backend_page_geometry(preset: str, head_dim: int):
    assert PallasTurboQuantAttentionBackend.supports_kv_cache_dtype(preset)
    page_words = get_tq_layout(preset, head_dim).page_words_128(PAGE_SIZE)
    assert PallasTurboQuantAttentionBackend.get_kv_cache_shape(
        10, PAGE_SIZE, 4, head_dim, preset
    ) == (10, 4, page_words, 128)
    page_bytes = PallasTurboQuantAttentionBackend.get_kv_cache_page_size_bytes(
        PAGE_SIZE, 4, head_dim, preset
    )
    assert page_bytes == page_words * 4 * 128 < PAGE_SIZE * 4 * 2 * head_dim * 2


def test_set_and_resolve_active_preset(monkeypatch: pytest.MonkeyPatch):
    assert not PallasTurboQuantAttentionBackend.supports_kv_cache_dtype("fp8")
    monkeypatch.setattr(turboquant_attention, "_active_preset", None)
    assert PallasTurboQuantAttentionBackend.get_kv_cache_shape(
        8, PAGE_SIZE, 4, 128, torch.uint8
    ) == (8, 4, PAGE_SIZE, 128)
    with pytest.raises(ValueError, match="Unknown TurboQuant KV cache dtype"):
        set_active_turboquant_preset("turboquant_invalid")
    set_active_turboquant_preset("  TURBOQUANT_4BIT_NC  ")
    page_words = get_tq_layout("turboquant_4bit_nc", 128).page_words_128(PAGE_SIZE)
    assert PallasTurboQuantAttentionBackend.get_kv_cache_shape(
        8, PAGE_SIZE, 4, 128, torch.uint8
    ) == (8, 4, page_words, 128)


def test_validate_turboquant_head_dim():
    for d in (128, 256, 512):
        validate_turboquant_head_dim(d)
    for bad in (-128, 0, 64, 96, 192, 384):
        with pytest.raises(NotImplementedError, match="head_dim"):
            validate_turboquant_head_dim(bad)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"sinks": object()}, "sinks"),
        ({"use_causal_mask": False}, "causal mask"),
        ({"k_scale": 1.0}, "scales"),
    ],
)
def test_kernel_entry_rejects_unsupported_features(kw: dict[str, Any], match: str):
    defaults: dict[str, Any] = dict(
        sinks=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        tq_preset="turboquant_4bit_nc",
    )
    with pytest.raises(NotImplementedError, match=match):
        _tq_rpa_kernel_entry(*([None] * 8), **(defaults | kw))  # type: ignore[arg-type]


def _mock_vllm_config(**kw: Any) -> Any:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            cache_dtype=kw.get("cache_dtype", "turboquant_4bit_nc"),
            user_specified_block_size=kw.get("user_specified_block_size", False),
            block_size=kw.get("block_size", 16),
            enable_prefix_caching=kw.get("enable_prefix_caching", True),
            mamba_cache_mode=kw.get("mamba_cache_mode", "align"),
            mamba_block_size=kw.get("mamba_block_size", 16),
            kv_cache_dtype_skip_layers=list(kw.get("kv_cache_dtype_skip_layers", ())),
        ),
        model_config=SimpleNamespace(
            use_mla=kw.get("use_mla", False),
            get_head_size=lambda: kw.get("head_size", 128),
            is_hybrid=kw.get("is_hybrid", False),
            max_model_len=kw.get("max_model_len", 4096),
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=kw.get("pcp", 1),
            decode_context_parallel_size=kw.get("dcp", 1),
        ),
    )


@pytest.mark.parametrize(
    ("cfg_kwargs", "exc", "match"),
    [
        ({"use_mla": True}, NotImplementedError, "MLA"),
        ({"head_size": 64}, NotImplementedError, "head_dim"),
        ({"pcp": 2}, NotImplementedError, "PCP"),
        ({"dcp": 2}, NotImplementedError, "DCP"),
        (
            {"user_specified_block_size": True, "block_size": 32},
            ValueError,
            "block_size",
        ),
        (
            {
                "cache_dtype": "turboquant_k8v4",
                "user_specified_block_size": True,
                "block_size": 128,
            },
            ValueError,
            "block_size",
        ),
    ],
)
def test_check_turboquant_config_rejects_invalid(
    cfg_kwargs: dict[str, Any], exc: type[Exception], match: str
):
    with pytest.raises(exc, match=match):
        check_turboquant_config(_mock_vllm_config(**cfg_kwargs))


def test_check_turboquant_config_updates_hybrid_and_page_sizes(
    monkeypatch: pytest.MonkeyPatch,
):
    assert (
        frozenset(PLATFORM_TQ_DTYPES)
        == frozenset(TURBOQUANT_KV_CACHE_DTYPES)
        == frozenset(TQ_PRESETS)
    )
    monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
    cfg = _mock_vllm_config(
        is_hybrid=True,
        max_model_len=8192,
        kv_cache_dtype_skip_layers=["model.layers.0.self_attn"],
    )
    cfg.model_config.architecture = "Qwen3_5ForConditionalGeneration"
    assert not unified_kv_layout_enabled(cfg)
    assert not unified_kv_layout_enabled(
        SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16),
            model_config=SimpleNamespace(architecture="DeepseekV4ForCausalLM"),
        )
    )
    check_turboquant_config(cfg)
    assert not cfg.cache_config.enable_prefix_caching
    assert cfg.cache_config.mamba_cache_mode == "none"
    assert cfg.cache_config.mamba_block_size == 8192
    assert cfg.cache_config.block_size == 64
    assert cfg.cache_config.kv_cache_dtype_skip_layers == []

    monkeypatch.setattr(turboquant_attention, "_active_preset", None)
    for preset, max_len, page_sz, supported in [
        ("turboquant_k8v4", 512, 32, [16, 32, 64]),
        ("turboquant_4bit_nc", 512, 16, [16, 64]),
        ("turboquant_4bit_nc", 4096, 64, [16, 64]),
    ]:
        c = _mock_vllm_config(cache_dtype=preset, max_model_len=max_len)
        check_turboquant_config(c)
        assert c.cache_config.block_size == page_sz
        monkeypatch.setattr(turboquant_attention, "_active_preset", None)
        spec = FullAttentionSpec(
            block_size=page_sz,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.uint8,
            kv_quant_mode=get_kv_quant_mode(preset),
        )
        assert (
            PallasTurboQuantAttentionBackend.get_supported_kernel_block_sizes(spec)
            == supported
        )
        customized = PallasTurboQuantAttentionBackend.customize_spec(spec)
        expected_bytes = PallasTurboQuantAttentionBackend.get_kv_cache_page_size_bytes(
            page_sz, 4, 128, preset
        )
        assert customized.state_content_bytes == expected_bytes // page_sz
