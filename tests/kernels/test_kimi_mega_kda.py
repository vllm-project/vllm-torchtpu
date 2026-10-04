# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation tests for the Kimi K3 mega KDA kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.kimi_k3.mega_kda import (
    _layout_supported,
    kda_forward_inference,
)
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op


@pytest.mark.parametrize("spec_tokens", [0, 5])
def test_fused_kda_passes_window_distribution(monkeypatch, spec_tokens):
    """The adapter forwards one distribution; kernel owns segment dispatch."""
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return (
            kwargs["conv_state"] + 1,
            kwargs["recurrent_state"] + 2,
        ), jnp.arange(16, dtype=jnp.float32).reshape(8, 2)

    monkeypatch.setattr(kda_attention_op.gdn_wrapper, "fused_conv1d_gdn", fake_kernel)
    monkeypatch.setattr(kda_attention_op, "_check_kda_abi", lambda *args: (8, 2, 1, 2))
    monkeypatch.setattr(
        kda_attention_op, "_gated_output_norm", lambda output, *args: output
    )
    core = kda_attention_op._build_fused_core(-5.0, 1e-6, spec_tokens)
    with jax.disable_jit():
        out, conv, recurrent = core(
            jnp.zeros((8, 6)),
            jnp.zeros((8, 2)),
            jnp.zeros((8, 1)),
            jnp.zeros((8, 2)),
            jnp.zeros((4, 3, 6)),
            jnp.zeros((4, 1, 2, 2)),
            jnp.zeros((4, 3, 1, 2)),
            jnp.zeros((1,)),
            jnp.zeros((2,)),
            jnp.ones((2,)),
            jnp.asarray([0, 6, 8]),
            jnp.asarray([1, 2]),
            jnp.asarray([6, 2]),
            jnp.asarray([0, 2, 2]),
            jnp.asarray([1, 2, 2]),
            jnp.zeros((4,), jnp.int32),
        )
    assert len(calls) == 1
    assert calls[0]["distribution"].tolist() == [1, 2, 2]
    assert calls[0]["num_spec_tokens"] == spec_tokens
    np.testing.assert_array_equal(conv, jnp.ones_like(conv))
    np.testing.assert_array_equal(recurrent, jnp.full_like(recurrent, 2))
    np.testing.assert_array_equal(
        out, jnp.arange(16, dtype=jnp.float32).reshape(8, 1, 2)
    )


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
def test_mega_kda_layout_guard(
    query_start_loc: list[int], num_tokens: int, supported: bool
) -> None:
    actual = _layout_supported(
        jnp.asarray(query_start_loc, dtype=jnp.int32), num_tokens
    )
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
        "A_log": jnp.zeros((1,), dtype=jnp.float32),
        "dt_bias": jnp.zeros((128,), dtype=jnp.float32),
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


@pytest.mark.parametrize("override", [{"safe_gate": False}, {"lower_bound": None}])
def test_mega_kda_requires_safe_fused_gate(override: dict) -> None:
    inputs = _valid_mega_inputs()
    inputs.update(override)
    with pytest.raises(ValueError, match="safe fused gate"):
        kda_forward_inference(**inputs)
