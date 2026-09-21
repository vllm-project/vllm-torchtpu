# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation tests for the Kimi K3 mega KDA kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.kimi_k3.mega_kda import (_layout_supported,
                                                    kda_forward_inference)
from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import \
    _guard_kda_verify_window


@pytest.mark.parametrize("spec_tokens,window_end,total_tokens,expected_calls",
                         [(5, 0, 8, ["prefill"]), (5, 1, 6, ["verify"]),
                          (5, 1, 8, ["verify", "prefill"]),
                          (0, 1, 8, ["combined"])])
def test_fused_kda_segment_dispatch(monkeypatch, spec_tokens, window_end,
                                    total_tokens, expected_calls):
    calls = []

    def fake_kernel(**kwargs):
        mode = ("verify" if kwargs["batched_only"] else
                "prefill" if kwargs["prefill_only"] else "combined")
        calls.append(mode)
        increment = 1 if mode == "verify" else 10
        states = (kwargs["conv_state"] + increment,
                  kwargs["recurrent_state"] + increment)
        return states, jnp.full((8, 2), increment, jnp.float32)

    monkeypatch.setattr(kda_attention_op.gdn_wrapper, "fused_conv1d_gdn",
                        fake_kernel)
    monkeypatch.setattr(kda_attention_op, "_check_kda_abi", lambda *args:
                        (8, 2, 1, 2))
    monkeypatch.setattr(kda_attention_op, "_gated_output_norm",
                        lambda output, *args: output)
    core = kda_attention_op._build_fused_core(-5.0, 1e-6, spec_tokens)
    with jax.disable_jit():
        out, conv, recurrent = core(jnp.zeros((8, 6)), jnp.zeros((8, 2)),
                                    jnp.zeros((8, 1)), jnp.zeros((8, 2)),
                                    jnp.zeros((4, 3, 6)),
                                    jnp.zeros((4, 1, 2, 2)),
                                    jnp.zeros((4, 3, 1, 2)), jnp.zeros((1, )),
                                    jnp.zeros((2, )), jnp.ones((2, )),
                                    jnp.asarray([0, 6, total_tokens]),
                                    jnp.asarray([1, 2]),
                                    jnp.asarray([6, total_tokens - 6]),
                                    jnp.asarray([0, 2, 2]),
                                    jnp.asarray([window_end, 2, 2]),
                                    jnp.zeros((4, ), jnp.int32))
    assert calls == expected_calls
    expected_state = sum(1 if mode == "verify" else 10 for mode in calls)
    assert bool(jnp.all(conv == expected_state))
    assert bool(jnp.all(recurrent == expected_state))
    assert bool(jnp.all(out[:6] == (1 if "verify" in calls else 10)))
    if total_tokens == 8:
        assert bool(jnp.all(out[6:] == 10))


@pytest.mark.parametrize(("num_window_reqs", "expected"), [(0, 2), (1, 1)])
def test_kda_verify_window_guard(num_window_reqs: int, expected: int) -> None:
    guarded = jax.jit(lambda count: _guard_kda_verify_window(
        count, lambda: jnp.asarray(1), lambda: jnp.asarray(2)))

    assert int(guarded(jnp.asarray(num_window_reqs))) == expected


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
