# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation and selection tests for the optional Kimi K3 mega KDA path."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.kimi_k3.mega_kda import (_layout_supported,
                                                    kda_forward_inference)
from vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op import \
    _get_kda_prefill_kernel


def test_kda_prefill_kernel_defaults_to_chunked(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL", raising=False)
    assert _get_kda_prefill_kernel() == "chunked"


@pytest.mark.parametrize("value", ["chunked", " CHUNKED ", "mega", " MEGA "])
def test_kda_prefill_kernel_accepts_supported_values(monkeypatch,
                                                     value: str) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL", value)
    assert _get_kda_prefill_kernel() == value.strip().lower()


def test_kda_prefill_kernel_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TORCHTPU_K3_KDA_PREFILL_KERNEL", "automatic")
    with pytest.raises(ValueError, match="must be 'chunked' or 'mega'"):
        _get_kda_prefill_kernel()


@pytest.mark.parametrize(
    ("query_start_loc", "num_tokens", "supported"),
    [
        ([0, 64], 64, True),
        ([0, 32, 64], 64, True),
        ([0, 20, 40, 64], 64, False),
        ([0, 37, 92, 116], 128, True),
        ([0, 64, 128, 192, 256], 256, True),
        ([0, 20, 20, 40, 40], 64, True),
    ],
)
def test_mega_kda_layout_guard(query_start_loc: list[int], num_tokens: int,
                               supported: bool) -> None:
    actual = _layout_supported(jnp.asarray(query_start_loc, dtype=jnp.int32),
                               num_tokens)
    assert bool(actual) is supported


def test_mega_kda_layout_guard_requires_full_tiles() -> None:
    with pytest.raises(ValueError, match="requires full token tiles"):
        _layout_supported(jnp.asarray([0, 63], dtype=jnp.int32), 63)


def _valid_mega_inputs(tokens: int = 64) -> dict:
    shape = (1, tokens, 1, 128)
    return {
        "q": jnp.zeros(shape, dtype=jnp.bfloat16),
        "k": jnp.zeros(shape, dtype=jnp.bfloat16),
        "v": jnp.zeros(shape, dtype=jnp.bfloat16),
        "g": jnp.zeros(shape, dtype=jnp.bfloat16),
        "beta": jnp.zeros(shape[:3], dtype=jnp.bfloat16),
        "segment_ids": jnp.ones(shape[:2], dtype=jnp.int32),
        "A_log": jnp.zeros((1, ), dtype=jnp.float32),
        "dt_bias": jnp.zeros((128, ), dtype=jnp.float32),
        "initial_state": jnp.zeros((1, 1, 1, 128, 128), dtype=jnp.float32),
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "safe_gate": True,
        "lower_bound": -5.0,
        "N_max": 1,
    }


def test_mega_kda_requires_bfloat16_inputs() -> None:
    inputs = _valid_mega_inputs()
    inputs["q"] = inputs["q"].astype(jnp.float32)
    with pytest.raises(ValueError, match="requires BF16"):
        kda_forward_inference(**inputs)


def test_mega_kda_requires_64_token_padding() -> None:
    inputs = _valid_mega_inputs(tokens=63)
    with pytest.raises(ValueError, match="multiple of chunk_size"):
        kda_forward_inference(**inputs)


@pytest.mark.parametrize("override", [{
    "safe_gate": False
}, {
    "lower_bound": None
}])
def test_mega_kda_requires_safe_fused_gate(override: dict) -> None:
    inputs = _valid_mega_inputs()
    inputs.update(override)
    with pytest.raises(ValueError, match="safe fused gate"):
        kda_forward_inference(**inputs)
